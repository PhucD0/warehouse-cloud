import argparse
import json
import os
import subprocess
import time
import uuid
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import qrcode
import threading
import urllib.request
import urllib.error


WINDOW_NAME = "Smart Warehouse AI"

ITEM_DB_PATH = "items_db.json"
PENDING_ITEM_PATH = "pending_item.json"
QR_DIR = "qr_codes"
MEASURE_BACKGROUND_PATH = "measure_background.npy"
MEASURE_BACKGROUND_COLOR_PATH = "measure_background_color.npy"
SHELF_BACKGROUND_PATH = "shelf_background.npy"
WAREHOUSE_EVENTS_PATH = "warehouse_events.jsonl"

DISPLAY_SCALE = 0.85
INFO_BAR_HEIGHT = 180


# ============================================================
# Camera (Updated to USB Camera with V4L2 & YUYV Fix)
# ============================================================

class USBCameraV4L2:
    """
    USB camera reader for Jetson using V4L2.

    Default external camera profile:
    - device: /dev/video2
    - format: MJPG
    - capture: 1280x720
    - output: 1280x720

    This avoids the old 640x480 YUYV capture + resize pipeline.
    """
    def __init__(
        self,
        camera_index=2,
        capture_width=1280,
        capture_height=720,
        output_width=1280,
        output_height=720,
        fourcc="MJPG",
        opencv_rotate_180=False,
    ):
        self.camera_index = camera_index
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.output_width = output_width
        self.output_height = output_height
        self.fourcc = fourcc.upper()
        self.opencv_rotate_180 = opencv_rotate_180
        self.cap = None

    def open(self):
        print(f"[CAMERA] Opening USB camera index={self.camera_index} via V4L2...")
        print(
            f"[CAMERA] Request capture={self.capture_width}x{self.capture_height}, "
            f"output={self.output_width}x{self.output_height}, fourcc={self.fourcc}"
        )

        # OpenCV compatibility:
        # Newer OpenCV supports cv2.VideoCapture(index, backend).
        # OpenCV 3.2 on Jetson Python 3.6 only supports cv2.VideoCapture(index).
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        except TypeError:
            print("[CAMERA] OpenCV backend argument not supported. Falling back to cv2.VideoCapture(index).")
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.is_opened():
            return False

        try:
            fourcc_value = cv2.VideoWriter_fourcc(*self.fourcc)
        except AttributeError:
            fourcc_value = cv2.VideoWriter.fourcc(*self.fourcc)

        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc_value)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)

        time.sleep(1.0)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc_str = "".join(
            [chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)]
        )

        print(f"[CAMERA] Actual capture={actual_w}x{actual_h}, fourcc={actual_fourcc_str}")

        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("[CAMERA] ERROR: Cannot read first frame.")
            return False

        print(f"[CAMERA] Connected successfully to USB camera index {self.camera_index}.")
        return True

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        if not self.is_opened():
            return False, None

        ret, frame = self.cap.read()

        if not ret or frame is None:
            return False, None

        # MJPG is normally decoded by OpenCV to BGR.
        # If YUYV is selected manually, convert if needed.
        if self.fourcc == "YUYV":
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
            except cv2.error:
                pass

        if frame.shape[1] != self.output_width or frame.shape[0] != self.output_height:
            frame = cv2.resize(
                frame,
                (self.output_width, self.output_height),
                interpolation=cv2.INTER_LINEAR,
            )

        if self.opencv_rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        return True, frame.copy()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ============================================================
# JSON helpers
# ============================================================

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}")

    with open(path, "r") as f:
        return json.load(f)


def load_json_default(path, default_data):
    if not os.path.exists(path):
        return default_data

    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# ============================================================
# Geometry
# ============================================================

def get_quad_points(config):
    c = config["corners_px"]

    tl = np.array(c["top_left"], dtype=np.float32)
    tr = np.array(c["top_right"], dtype=np.float32)
    br = np.array(c["bottom_right"], dtype=np.float32)
    bl = np.array(c["bottom_left"], dtype=np.float32)

    return tl, tr, br, bl


def get_homography_from_config(config):
    tl, tr, br, bl = get_quad_points(config)
    src = np.array([tl, tr, br, bl], dtype=np.float32)

    w = int(round(config["pixel_scale"]["avg_width_px"]))
    h = int(round(config["pixel_scale"]["avg_height_px"]))

    w = max(w, 50)
    h = max(h, 50)

    dst = np.array(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ],
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(src, dst)
    H_inv = cv2.getPerspectiveTransform(dst, src)

    return H, H_inv, w, h


def warp_area(frame, H, w, h):
    return cv2.warpPerspective(frame, H, (w, h))


def preprocess(img):
    """
    Lighting-stable preprocessing for background subtraction.

    Instead of using raw grayscale directly, this function normalizes
    illumination so that detection is less sensitive to the scene being
    slightly too bright or too dark.

    Pipeline:
    BGR -> LAB -> L channel -> CLAHE -> mean brightness normalization -> blur
    """
    if img is None:
        return None

    if len(img.shape) == 2:
        gray = img.copy()
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab)

        # Local contrast normalization. Helps when shelf/object has shadows.
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        gray = clahe.apply(l_channel)

    # Normalize global brightness toward a stable target.
    # Clamp alpha to avoid over-amplifying dark noise or killing contrast.
    target_mean = 128.0
    mean_val = float(np.mean(gray))

    if mean_val > 1.0:
        alpha = target_mean / mean_val
        alpha = max(0.65, min(1.65, alpha))
        gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=0)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def capture_background_average(camera, H, out_w, out_h, sample_count=15, delay_sec=0.03):
    """
    Capture a more stable background by averaging multiple preprocessed frames.
    Used when pressing:
      b = measurement background
      k = shelf background
    """
    frames = []

    for _ in range(max(1, sample_count)):
        ret, frame = camera.read()

        if not ret or frame is None:
            continue

        warped = warp_area(frame, H, out_w, out_h)
        processed = preprocess(warped)

        if processed is not None:
            frames.append(processed.astype(np.float32))

        time.sleep(delay_sec)

    if not frames:
        return None

    avg = np.mean(frames, axis=0)
    return np.clip(avg, 0, 255).astype(np.uint8)




def capture_background_color_average(camera, H, out_w, out_h, sample_count=15, delay_sec=0.03):
    """
    Capture averaged BGR background for color-aware measurement detection.
    This is used only for the measurement area.
    """
    frames = []

    for _ in range(max(1, sample_count)):
        ret, frame = camera.read()

        if not ret or frame is None:
            continue

        warped = warp_area(frame, H, out_w, out_h)
        frames.append(warped.astype(np.float32))

        time.sleep(delay_sec)

    if not frames:
        return None

    avg = np.mean(frames, axis=0)
    return np.clip(avg, 0, 255).astype(np.uint8)


def build_measure_foreground_mask(
    current_warped,
    background_gray,
    background_color=None,
    gray_threshold=35,
    color_threshold=14,
):
    """
    Build a foreground mask for the measurement area.

    Old logic used only grayscale/L-channel difference. That fails when the
    object has similar brightness to the background. This combines:
      - illumination-normalized grayscale difference
      - LAB chroma difference from the averaged color background
    """
    current_gray = preprocess(current_warped)
    gray_diff = cv2.absdiff(background_gray, current_gray)
    _, gray_mask = cv2.threshold(gray_diff, gray_threshold, 255, cv2.THRESH_BINARY)

    if background_color is None:
        mask = gray_mask
    else:
        bg_color = background_color

        if bg_color.shape[:2] != current_warped.shape[:2]:
            bg_color = cv2.resize(
                bg_color,
                (current_warped.shape[1], current_warped.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        cur_lab = cv2.cvtColor(current_warped, cv2.COLOR_BGR2LAB)
        bg_lab = cv2.cvtColor(bg_color, cv2.COLOR_BGR2LAB)

        # A/B channels capture color change and are less sensitive to brightness.
        cur_ab = cur_lab[:, :, 1:3].astype(np.int16)
        bg_ab = bg_lab[:, :, 1:3].astype(np.int16)
        chroma_diff = np.sqrt(np.sum((cur_ab - bg_ab) ** 2, axis=2)).astype(np.uint8)

        _, chroma_mask = cv2.threshold(chroma_diff, color_threshold, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_or(gray_mask, chroma_mask)

    # Remove border flicker from the calibrated measurement ROI.
    h, w = mask.shape[:2]
    mx = max(2, int(round(w * 0.015)))
    my = max(2, int(round(h * 0.015)))

    mask[:my, :] = 0
    mask[:, :mx] = 0
    mask[:, w - mx:] = 0

    # Keep bottom almost intact because the object usually sits on the surface.
    mask[h - max(1, my // 2):, :] = 0

    kernel_small = np.ones((3, 3), np.uint8)
    kernel_mid = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_mid, iterations=2)
    mask = cv2.dilate(mask, kernel_small, iterations=1)

    return mask

def rect_poly_to_original(poly_rect, H_inv):
    pts = np.array(poly_rect, dtype=np.float32).reshape(-1, 1, 2)
    pts_orig = cv2.perspectiveTransform(pts, H_inv)
    return pts_orig.reshape(-1, 2).astype(np.int32)


def interval_to_rect_poly(config, shelf_w_px, shelf_h_px, level_index, start_cm, end_cm):
    shelf_width_cm = float(config["physical_size_cm"]["width"])
    num_levels = int(config.get("num_levels", 4))

    x1 = (start_cm / shelf_width_cm) * shelf_w_px
    x2 = (end_cm / shelf_width_cm) * shelf_w_px

    y1 = ((level_index - 1) / num_levels) * shelf_h_px
    y2 = (level_index / num_levels) * shelf_h_px

    return np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    )


# ============================================================
# Shelf state
# ============================================================

def create_initial_shelf_state(shelf_config):
    shelf_id = shelf_config.get("shelf_id", "SHELF_A")
    physical = shelf_config["physical_size_cm"]

    shelf_width_cm = float(physical["width"])
    shelf_height_cm = float(physical["height"])
    shelf_depth_cm = float(physical["depth"])

    num_levels = int(shelf_config.get("num_levels", 4))
    level_height_cm = shelf_height_cm / num_levels

    levels = []

    for i in range(num_levels):
        levels.append(
            {
                "level_id": f"T{i + 1}",
                "level_index": i + 1,
                "physical_size_cm": {
                    "width": shelf_width_cm,
                    "height": level_height_cm,
                    "depth": shelf_depth_cm,
                },
                "free_intervals": [
                    {
                        "start_cm": 0.0,
                        "end_cm": shelf_width_cm,
                        "width_cm": shelf_width_cm,
                    }
                ],
                "occupied_intervals": [],
            }
        )

    return {
        "shelf_id": shelf_id,
        "model": "dynamic_level_intervals",
        "description": "Generated by smart_warehouse_ai.py",
        "physical_size_cm": {
            "width": shelf_width_cm,
            "height": shelf_height_cm,
            "depth": shelf_depth_cm,
        },
        "num_levels": num_levels,
        "priority": {
            "level_order": "top_to_bottom",
            "placement_order": "left_to_right",
        },
        "levels": levels,
    }


def recompute_free_intervals(level):
    width = float(level["physical_size_cm"]["width"])

    occupied = sorted(
        level["occupied_intervals"],
        key=lambda x: float(x["start_cm"]),
    )

    free = []
    current = 0.0

    for occ in occupied:
        occ_start = float(occ["start_cm"])
        occ_end = float(occ["end_cm"])

        if occ_start > current:
            free.append(
                {
                    "start_cm": round(current, 3),
                    "end_cm": round(occ_start, 3),
                    "width_cm": round(occ_start - current, 3),
                }
            )

        current = max(current, occ_end)

    if current < width:
        free.append(
            {
                "start_cm": round(current, 3),
                "end_cm": round(width, 3),
                "width_cm": round(width - current, 3),
            }
        )

    level["free_intervals"] = free


def find_placement(state, item_width_cm, item_height_cm, item_depth_cm=0.0):
    for level in state["levels"]:
        level_size = level["physical_size_cm"]
        level_height = float(level_size["height"])
        level_depth = float(level_size["depth"])

        if level["level_id"] != "T1":
            if item_height_cm > level_height:
                continue

        if item_depth_cm > 0 and item_depth_cm > level_depth:
            continue

        for free in level["free_intervals"]:
            free_width = float(free["width_cm"])

            if item_width_cm <= free_width:
                start_cm = float(free["start_cm"])
                end_cm = start_cm + item_width_cm

                return {
                    "found": True,
                    "shelf_id": state["shelf_id"],
                    "level_id": level["level_id"],
                    "level_index": int(level["level_index"]),
                    "start_cm": round(start_cm, 3),
                    "end_cm": round(end_cm, 3),
                    "width_cm": round(item_width_cm, 3),
                    "height_cm": round(item_height_cm, 3),
                    "depth_cm": round(item_depth_cm, 3),
                    "free_start_cm": float(free["start_cm"]),
                    "free_end_cm": float(free["end_cm"]),
                }

    return {
        "found": False,
        "reason": "No suitable free interval found",
    }


def make_accept_zone(placement, tolerance_cm):
    start = max(placement["free_start_cm"], placement["start_cm"] - tolerance_cm)
    end = min(placement["free_end_cm"], placement["end_cm"] + tolerance_cm)

    return {
        "level_id": placement["level_id"],
        "level_index": placement["level_index"],
        "start_cm": round(start, 3),
        "end_cm": round(end, 3),
        "width_cm": round(end - start, 3),
    }


def build_suggested_position(placement, accept_zone=None):
    suggested = {
        "shelf_id": placement["shelf_id"],
        "level_id": placement["level_id"],
        "level_index": placement["level_index"],
        "start_cm": placement["start_cm"],
        "end_cm": placement["end_cm"],
        "position_label": (
            f"{placement['shelf_id']}-{placement['level_id']}-"
            f"X{placement['start_cm']}-{placement['end_cm']}"
        ),
    }

    if accept_zone is not None:
        suggested["accept_zone"] = accept_zone

    return suggested


def compute_reserved_interval(state, item, actual, occupied_padding_cm, item_gap_cm):
    """
    Compute a RESERVED interval for shelf planning.

    actual_start/actual_end are kept as vision-detected values for traceability.

    Important fix:
    The reserved interval used for shelf_state must NOT depend directly on
    actual_width from the live contour. That contour can become too large when
    the mask includes shadows, shelf edges, box edges, or lighting changes.

    Therefore:
    - actual_* keeps the raw vision-detected interval.
    - start_cm/end_cm used for next placement are based on measured item width
      + safety padding + item gap.
    """
    shelf_width_cm = float(state["physical_size_cm"]["width"])

    actual_start = float(actual["start_cm"])
    actual_end = float(actual["end_cm"])
    actual_width = max(0.001, actual_end - actual_start)
    actual_center = (actual_start + actual_end) / 2.0

    measured_width = max(0.001, float(item["size_cm"]["width"]))

    # Core fix:
    # Do NOT use max(actual_width, measured_width).
    # actual_width can be wrong when bbox is oversized.
    reserved_width = measured_width + (2.0 * occupied_padding_cm) + item_gap_cm

    # Small lower bound to avoid unrealistically tiny reserved zones.
    reserved_width = max(reserved_width, 0.8)

    reserved_start = actual_center - reserved_width / 2.0
    reserved_end = actual_center + reserved_width / 2.0

    if reserved_start < 0:
        reserved_end = reserved_end - reserved_start
        reserved_start = 0.0

    if reserved_end > shelf_width_cm:
        overflow = reserved_end - shelf_width_cm
        reserved_start = max(0.0, reserved_start - overflow)
        reserved_end = shelf_width_cm

    return {
        "actual_start_cm": round(actual_start, 3),
        "actual_end_cm": round(actual_end, 3),
        "actual_width_cm": round(actual_width, 3),
        "reserved_start_cm": round(reserved_start, 3),
        "reserved_end_cm": round(reserved_end, 3),
        "reserved_width_cm": round(reserved_end - reserved_start, 3),
        "occupied_padding_cm": occupied_padding_cm,
        "item_gap_cm": item_gap_cm,
        "reserved_source": "measured_width_plus_padding",
    }


def update_state_with_actual_position(
    state,
    item,
    placement,
    actual,
    occupied_padding_cm=0.4,
    item_gap_cm=0.3,
):
    item_id = item["item_id"]

    reserved = compute_reserved_interval(
        state=state,
        item=item,
        actual=actual,
        occupied_padding_cm=occupied_padding_cm,
        item_gap_cm=item_gap_cm,
    )

    for level in state["levels"]:
        if int(level["level_index"]) == int(actual["level_index"]):
            occupied = {
                "item_id": item_id,
                "start_cm": reserved["reserved_start_cm"],
                "end_cm": reserved["reserved_end_cm"],
                "width_cm": reserved["reserved_width_cm"],
                "actual_start_cm": reserved["actual_start_cm"],
                "actual_end_cm": reserved["actual_end_cm"],
                "actual_width_cm": reserved["actual_width_cm"],
                "height_cm": item["size_cm"]["height"],
                "depth_cm": item["size_cm"].get("depth", 0.0),
                "suggested_start_cm": placement["start_cm"],
                "suggested_end_cm": placement["end_cm"],
                "reserved_position_note": "start_cm/end_cm include safety padding and item gap for next placement recommendation",
                "occupied_padding_cm": reserved["occupied_padding_cm"],
                "item_gap_cm": reserved["item_gap_cm"],
                "placed_at": datetime.now().isoformat(timespec="seconds"),
                "bbox_rect": actual.get("bbox_rect"),
                "area_px": actual.get("area_px"),
                "tracking_status": "registered",
                "source": "smart_warehouse_ai_vision",
            }

            level["occupied_intervals"].append(occupied)
            recompute_free_intervals(level)
            return True

    return False


# ============================================================
# 1. Khu v?c ÃO Ã?C v?t th? (Measure area) - GI? NGUYÃŠN LOGIC CU
# ============================================================

def detect_object_from_background(
    current_warped,
    background_gray,
    threshold_value=35,
    min_area=80,
    background_color=None,
):
    """
    Detect object in measurement ROI using color-aware background subtraction.

    Important fixes:
    - use LAB chroma difference in addition to grayscale difference
    - union valid contours instead of taking only the largest contour
    - avoid thin/border artifacts so the measured bbox is more stable
    """
    mask = build_measure_foreground_mask(
        current_warped=current_warped,
        background_gray=background_gray,
        background_color=background_color,
        gray_threshold=threshold_value,
        color_threshold=max(10, int(round(threshold_value * 0.40))),
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = mask.shape[:2]
    valid_boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Reject tiny stripes and border artifacts.
        if w < 4 or h < 4:
            continue

        fill_ratio = float(area) / max(1.0, float(w * h))
        aspect = w / max(1.0, float(h))

        # Very thin horizontal strips are usually table/edge flicker.
        if aspect > 8.0 and h < max(8, int(h_img * 0.08)):
            continue

        # Very sparse regions are usually noise/reflection.
        if fill_ratio < 0.05 and area < min_area * 3:
            continue

        valid_boxes.append((x, y, w, h, area))

    if not valid_boxes:
        return None, mask

    # Union valid components because boxes often split into logo/body/shadow parts.
    x1 = min(b[0] for b in valid_boxes)
    y1 = min(b[1] for b in valid_boxes)
    x2 = max(b[0] + b[2] for b in valid_boxes)
    y2 = max(b[1] + b[3] for b in valid_boxes)

    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    x2 = max(x1 + 1, min(x2, w_img))
    y2 = max(y1 + 1, min(y2, h_img))

    total_area = sum(b[4] for b in valid_boxes)

    return {
        "x": int(x1),
        "y": int(y1),
        "w": int(x2 - x1),
        "h": int(y2 - y1),
        "area_px": float(total_area),
    }, mask


def estimate_size_cm_from_measure_bbox(bbox, measure_config):
    cm_per_px_x = float(measure_config["pixel_scale"]["cm_per_px_x"])
    cm_per_px_y = float(measure_config["pixel_scale"]["cm_per_px_y"])

    width_cm = bbox["w"] * cm_per_px_x
    height_cm = bbox["h"] * cm_per_px_y

    return round(width_cm, 2), round(height_cm, 2)



# ============================================================
# YOLO measurement detector
# ============================================================

def _get_yolo_class_name(model, cls_id):
    names = getattr(model, "names", {})

    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))

    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return names[cls_id]

    return str(cls_id)


def load_measure_yolo_detector(args):
    """
    Load YOLO/YOLO-Worldv2 for the measurement area.
    This keeps YOLO as the detector and OpenCV calibration as the measuring layer.
    """
    if args.measure_detector not in ("yolo", "yolo_fallback"):
        print("[MEASURE-YOLO] Disabled. Using background subtraction for measurement.")
        return None

    if not os.path.exists(args.measure_yolo_model):
        print(f"[MEASURE-YOLO] Model not found: {args.measure_yolo_model}")

        if args.measure_detector == "yolo":
            print("[MEASURE-YOLO] Strict YOLO mode requested, but model is missing.")

        return None

    print("")
    print("======================================")
    print("[MEASURE-YOLO] Loading measurement detector")
    print(f"[MEASURE-YOLO] Model: {args.measure_yolo_model}")
    print(f"[MEASURE-YOLO] Classes: {args.measure_yolo_classes}")
    print(f"[MEASURE-YOLO] imgsz={args.measure_yolo_imgsz}, conf={args.measure_yolo_conf}")
    print("======================================")

    try:
        if "world" in args.measure_yolo_model.lower():
            from ultralytics import YOLOWorld
            model = YOLOWorld(args.measure_yolo_model)
            classes = [x.strip() for x in args.measure_yolo_classes.split(",") if x.strip()]
            model.set_classes(classes)
            print(f"[MEASURE-YOLO] YOLO-World classes set: {classes}")
        else:
            from ultralytics import YOLO
            model = YOLO(args.measure_yolo_model, task="detect")

        return model

    except Exception as e:
        print(f"[MEASURE-YOLO] Failed to load model: {e}")

        if args.measure_detector == "yolo":
            print("[MEASURE-YOLO] Strict YOLO mode cannot continue without model.")

        return None


def detect_measure_object_yolo(
    current_warped,
    model,
    conf_thres=0.05,
    imgsz=320,
    min_area_px=80,
    allowed_classes="box,bottle,cup",
):
    """
    Detect object in the measurement ROI with YOLO and return bbox compatible
    with estimate_size_cm_from_measure_bbox().

    Level-1 measurement:
      YOLO bbox pixel width/height -> OpenCV calibration cm_per_px conversion.
    """
    if model is None or current_warped is None:
        return None, None

    h_img, w_img = current_warped.shape[:2]
    mask = np.zeros((h_img, w_img), dtype=np.uint8)

    try:
        results = model.predict(
            current_warped,
            imgsz=imgsz,
            conf=conf_thres,
            verbose=False,
        )
    except Exception as e:
        print(f"[MEASURE-YOLO] predict failed: {e}")
        return None, mask

    if not results or results[0].boxes is None:
        return None, mask

    allowed = set([x.strip().lower() for x in str(allowed_classes).split(",") if x.strip()])
    candidates = []

    for box in results[0].boxes:
        conf = float(box.conf[0].item())

        if conf < conf_thres:
            continue

        cls_id = int(box.cls[0].item()) if box.cls is not None else -1
        cls_name = _get_yolo_class_name(model, cls_id)
        cls_name_l = str(cls_name).lower()

        if allowed and cls_name_l not in allowed:
            continue

        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(float)

        x1 = max(0, min(x1, w_img - 1))
        x2 = max(0, min(x2, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(0, min(y2, h_img - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh

        if area < float(min_area_px):
            continue

        # Reject near-full ROI detections. These are usually background/ROI mistakes.
        if bw > 0.95 * w_img or bh > 0.95 * h_img:
            continue

        if bw < 4 or bh < 4:
            continue

        candidates.append(
            {
                "x": int(round(x1)),
                "y": int(round(y1)),
                "w": int(round(bw)),
                "h": int(round(bh)),
                "area_px": float(area),
                "conf": round(conf, 3),
                "cls_id": cls_id,
                "cls_name": cls_name,
                "source": "measure_yolo",
            }
        )

    if not candidates:
        return None, mask

    # In the measurement area there should be one item. Prefer high confidence,
    # then larger bbox area.
    candidates.sort(
        key=lambda b: (
            float(b.get("conf", 0.0)),
            float(b.get("area_px", 0.0)),
        ),
        reverse=True,
    )

    best = candidates[0]

    cv2.rectangle(
        mask,
        (best["x"], best["y"]),
        (best["x"] + best["w"], best["y"] + best["h"]),
        255,
        -1,
    )

    return best, mask


# ============================================================
# 2. Khu v?c K? HÃ€NG (Shelf area) - THÃŠM B? L?C CH?N V?T L? < 1x1 CM
# ============================================================

def _tight_bbox_from_contour(contour, x, y, w, h):
    """
    Build a tighter bbox from the filled contour footprint to reduce sudden
    expansion caused by sparse foreground noise around the object.
    """
    if w <= 1 or h <= 1:
        return x, y, w, h

    roi = np.zeros((h, w), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.drawContours(roi, [shifted], -1, 255, -1)

    kernel = np.ones((3, 3), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)

    col_counts = np.count_nonzero(roi, axis=0)
    row_counts = np.count_nonzero(roi, axis=1)

    min_col_support = max(2, int(round(h * 0.08)))
    min_row_support = max(2, int(round(w * 0.08)))

    valid_cols = np.where(col_counts >= min_col_support)[0]
    valid_rows = np.where(row_counts >= min_row_support)[0]

    if len(valid_cols) == 0 or len(valid_rows) == 0:
        return x, y, w, h

    x1 = x + int(valid_cols[0])
    x2 = x + int(valid_cols[-1]) + 1
    y1 = y + int(valid_rows[0])
    y2 = y + int(valid_rows[-1]) + 1

    tw = max(1, x2 - x1)
    th = max(1, y2 - y1)

    return x1, y1, tw, th


def clone_candidate_for_tracking(cand):
    out = dict(cand)
    out["bbox_rect"] = dict(cand.get("bbox_rect", {}))
    return out


def stabilize_live_candidates(
    prev_candidates,
    curr_candidates,
    smooth_alpha=0.32,
    max_grow_ratio=1.12,
    max_shrink_ratio=0.88,
    max_center_shift_cm=0.22,
):
    """
    Stabilize LIVE monitor boxes over time.

    - smooth center/width changes
    - clamp sudden growth
    - reduce one-frame bbox explosion caused by noisy foreground masks
    """
    if not curr_candidates:
        return []

    prev_by_level = {}
    for p in prev_candidates or []:
        prev_by_level.setdefault(int(p.get("level_index", 1)), []).append(clone_candidate_for_tracking(p))

    used = set()
    stable = []

    curr_sorted = sorted(
        curr_candidates,
        key=lambda c: (int(c.get("level_index", 1)), float(c.get("center_cm", 0.0)))
    )

    for cand in curr_sorted:
        cur = clone_candidate_for_tracking(cand)
        level_index = int(cur.get("level_index", 1))

        cur_center = float(cur.get("center_cm", 0.0))
        cur_width = max(0.05, float(cur.get("width_cm", 0.05)))

        best_idx = None
        best_score = 1e18

        for idx, prev in enumerate(prev_by_level.get(level_index, [])):
            if (level_index, idx) in used:
                continue

            prev_center = float(prev.get("center_cm", 0.0))
            prev_width = max(0.05, float(prev.get("width_cm", 0.05)))

            max_dist = max(max_center_shift_cm * 4.0, prev_width * 0.9)
            dist = abs(cur_center - prev_center)
            if dist > max_dist:
                continue

            score = dist + 0.35 * abs(cur_width - prev_width)
            if score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            stable.append(cur)
            continue

        used.add((level_index, best_idx))
        prev = prev_by_level[level_index][best_idx]

        prev_center = float(prev.get("center_cm", 0.0))
        prev_width = max(0.05, float(prev.get("width_cm", 0.05)))

        raw_center = min(max(cur_center, prev_center - max_center_shift_cm), prev_center + max_center_shift_cm)
        raw_width = min(cur_width, prev_width * max_grow_ratio)
        raw_width = max(raw_width, prev_width * max_shrink_ratio)

        sm_center = (1.0 - smooth_alpha) * prev_center + smooth_alpha * raw_center
        sm_width = (1.0 - smooth_alpha) * prev_width + smooth_alpha * raw_width

        shelf_width_cm = float(cur.get("shelf_width_cm", 1.0))
        shelf_w_px = int(cur.get("shelf_w_px", 1))

        sm_start = max(0.0, sm_center - sm_width / 2.0)
        sm_end = min(shelf_width_cm, sm_center + sm_width / 2.0)

        cur["start_cm"] = round(sm_start, 3)
        cur["end_cm"] = round(sm_end, 3)
        cur["width_cm"] = round(max(0.001, sm_end - sm_start), 3)
        cur["center_cm"] = round((sm_start + sm_end) / 2.0, 3)

        bcur = dict(cur.get("bbox_rect", {}))
        bprev = dict(prev.get("bbox_rect", {}))

        prev_y = float(bprev.get("y", bcur.get("y", 0)))
        prev_h = max(1.0, float(bprev.get("h", bcur.get("h", 1))))
        cur_y = float(bcur.get("y", prev_y))
        cur_h = max(1.0, float(bcur.get("h", prev_h)))

        y_shift = max(2.0, prev_h * 0.18)
        raw_y = min(max(cur_y, prev_y - y_shift), prev_y + y_shift)
        raw_h = min(cur_h, prev_h * max_grow_ratio)
        raw_h = max(raw_h, prev_h * max_shrink_ratio)

        sm_y = (1.0 - smooth_alpha) * prev_y + smooth_alpha * raw_y
        sm_h = (1.0 - smooth_alpha) * prev_h + smooth_alpha * raw_h

        x_px = int(round((sm_start / max(1e-6, shelf_width_cm)) * shelf_w_px))
        w_px = int(round((max(0.001, sm_end - sm_start) / max(1e-6, shelf_width_cm)) * shelf_w_px))
        x_px = max(0, min(x_px, max(0, shelf_w_px - 1)))
        w_px = max(1, min(w_px, shelf_w_px - x_px))

        cur["bbox_rect"] = {
            "x": int(x_px),
            "y": int(round(sm_y)),
            "w": int(w_px),
            "h": int(round(sm_h)),
        }

        stable.append(cur)

    return stable


def detect_new_objects_on_shelf(current_warped, before_gray, shelf_config, threshold_value=35, min_area=120):
    current_gray = preprocess(current_warped)
    diff = cv2.absdiff(before_gray, current_gray)

    _, mask = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    shelf_h_px, shelf_w_px = mask.shape[:2]
    num_levels = int(shelf_config.get("num_levels", 4))

    shelf_width_cm = float(shelf_config["physical_size_cm"]["width"])
    shelf_height_cm = float(shelf_config["physical_size_cm"]["height"])

    cm_per_px_x = shelf_width_cm / shelf_w_px
    cm_per_px_y = shelf_height_cm / shelf_h_px

    level_h = shelf_h_px / num_levels
    candidates = []

    for level_idx in range(num_levels):
        y1_level = int(round(level_idx * level_h))
        y2_level = int(round((level_idx + 1) * level_h))

        if level_idx == num_levels - 1:
            y2_level = shelf_h_px

        level_mask = mask[y1_level:y2_level, :]

        contours, _ = cv2.findContours(
            level_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            x, y, w, h = _tight_bbox_from_contour(contour, x, y, w, h)

            obj_width_cm = w * cm_per_px_x
            obj_height_cm = h * cm_per_px_y

            if obj_width_cm <= 1.0 or obj_height_cm <= 1.0:
                continue

            y_global = y + y1_level

            # Extra metadata used by monitor unknown-object detection.
            # Without these fields, is_strong_unknown_candidate() may ignore
            # real unregistered objects even when LIVE bbox is visible.
            fill_ratio = float(area) / max(1.0, float(w * h))
            aspect_ratio = float(w) / max(1.0, float(h))

            x_start_cm = (x / shelf_w_px) * shelf_width_cm
            x_end_cm = ((x + w) / shelf_w_px) * shelf_width_cm

            candidates.append(
                {
                    "level_id": f"T{level_idx + 1}",
                    "level_index": level_idx + 1,
                    "bbox_rect": {
                        "x": int(x),
                        "y": int(y_global),
                        "w": int(w),
                        "h": int(h),
                    },
                    "area_px": float(area),
                    "start_cm": round(x_start_cm, 3),
                    "end_cm": round(x_end_cm, 3),
                    "width_cm": round(obj_width_cm, 3),
                    "height_cm": round(obj_height_cm, 3),
                    "center_cm": round((x_start_cm + x_end_cm) / 2.0, 3),
                    "fill_ratio": round(fill_ratio, 4),
                    "aspect_ratio": round(aspect_ratio, 3),
                    "shelf_w_px": int(shelf_w_px),
                    "shelf_width_cm": float(shelf_width_cm),
                }
            )

    return candidates, mask



# ============================================================
# YOLO async shelf detector
# ============================================================

def _shelf_yolo_get_class_name(model, cls_id):
    names = getattr(model, "names", {})

    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))

    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return names[cls_id]

    return str(cls_id)


def _shelf_yolo_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def _shelf_yolo_containment_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    smaller = max(1.0, min(area_a, area_b))
    return inter / smaller


def shelf_yolo_class_agnostic_nms(candidates, iou_thres=0.35, containment_thres=0.65):
    """
    YOLO-World can detect the same object as box/carton/bottle/cup depending on prompt.
    This suppresses duplicate boxes across classes.
    """
    if not candidates:
        return [], []

    sorted_candidates = sorted(
        candidates,
        key=lambda x: float(x.get("conf", 0.0)),
        reverse=True,
    )

    kept = []
    rejected = []

    for cand in sorted_candidates:
        b = cand["bbox_rect"]

        box_a = [
            float(b["x"]),
            float(b["y"]),
            float(b["x"] + b["w"]),
            float(b["y"] + b["h"]),
        ]

        duplicate = False
        duplicate_of = None

        for kept_cand in kept:
            kb = kept_cand["bbox_rect"]

            box_b = [
                float(kb["x"]),
                float(kb["y"]),
                float(kb["x"] + kb["w"]),
                float(kb["y"] + kb["h"]),
            ]

            iou = _shelf_yolo_iou_xyxy(box_a, box_b)
            containment = _shelf_yolo_containment_xyxy(box_a, box_b)

            if iou >= iou_thres or containment >= containment_thres:
                duplicate = True
                duplicate_of = kept_cand
                break

        if duplicate:
            rejected.append(
                {
                    "cls_name": cand.get("cls_name", "unknown"),
                    "conf": cand.get("conf", 0.0),
                    "level_id": cand.get("level_id", "?"),
                    "reason": "duplicate_nms",
                    "duplicate_of": duplicate_of.get("cls_name", "unknown") if duplicate_of else "unknown",
                }
            )
        else:
            kept.append(cand)

    kept.sort(key=lambda x: (int(x.get("level_index", 1)), float(x.get("start_cm", 0.0))))
    return kept, rejected


def shelf_yolo_results_to_candidates(
    result,
    model,
    shelf_config,
    shelf_w_px,
    shelf_h_px,
    conf_thres=0.05,
    min_area_px=120,
    allowed_classes="box,bottle,cup",
):
    shelf_width_cm = float(shelf_config["physical_size_cm"]["width"])
    shelf_height_cm = float(shelf_config["physical_size_cm"]["height"])
    num_levels = int(shelf_config.get("num_levels", 4))

    allowed = set([x.strip().lower() for x in str(allowed_classes).split(",") if x.strip()])
    raw_candidates = []
    rejected = []

    if result is None or result.boxes is None:
        shelf_yolo_results_to_candidates.last_rejected = []
        return []

    for box in result.boxes:
        conf = float(box.conf[0].item())

        if conf < conf_thres:
            continue

        cls_id = int(box.cls[0].item()) if box.cls is not None else -1
        cls_name = _shelf_yolo_get_class_name(model, cls_id)
        cls_name_l = str(cls_name).lower()

        if allowed and cls_name_l not in allowed:
            rejected.append(
                {
                    "cls_name": cls_name,
                    "conf": round(conf, 3),
                    "reason": "class_not_allowed",
                }
            )
            continue

        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(float)

        x1 = max(0, min(x1, shelf_w_px - 1))
        x2 = max(0, min(x2, shelf_w_px - 1))
        y1 = max(0, min(y1, shelf_h_px - 1))
        y2 = max(0, min(y2, shelf_h_px - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        w = x2 - x1
        h = y2 - y1
        area_px = w * h

        if area_px < float(min_area_px):
            continue

        width_cm = (w / shelf_w_px) * shelf_width_cm
        height_cm = (h / shelf_h_px) * shelf_height_cm

        level_h_cm = shelf_height_cm / max(1, num_levels)
        width_ratio = width_cm / max(0.001, shelf_width_cm)
        height_ratio_level = height_cm / max(0.001, level_h_cm)
        aspect = width_cm / max(0.001, height_cm)

        # Reject shelf boards or ROI-size mistakes.
        if width_ratio > 0.88:
            rejected.append(
                {
                    "cls_name": cls_name,
                    "conf": round(conf, 3),
                    "reason": "too_wide_shelf_like",
                }
            )
            continue

        if width_ratio > 0.72 and height_ratio_level < 0.45:
            rejected.append(
                {
                    "cls_name": cls_name,
                    "conf": round(conf, 3),
                    "reason": "horizontal_shelf_edge_like",
                }
            )
            continue

        if height_ratio_level > 1.35:
            rejected.append(
                {
                    "cls_name": cls_name,
                    "conf": round(conf, 3),
                    "reason": "too_tall_for_level",
                }
            )
            continue

        if aspect > 6.0 and height_cm < 1.0:
            rejected.append(
                {
                    "cls_name": cls_name,
                    "conf": round(conf, 3),
                    "reason": "thin_horizontal_artifact",
                }
            )
            continue

        cy = (y1 + y2) / 2.0
        level_index = int(cy / (shelf_h_px / num_levels)) + 1
        level_index = max(1, min(level_index, num_levels))

        start_cm = (x1 / shelf_w_px) * shelf_width_cm
        end_cm = (x2 / shelf_w_px) * shelf_width_cm
        center_cm = (start_cm + end_cm) / 2.0

        raw_candidates.append(
            {
                "level_id": f"T{level_index}",
                "level_index": level_index,
                "bbox_rect": {
                    "x": int(round(x1)),
                    "y": int(round(y1)),
                    "w": int(round(w)),
                    "h": int(round(h)),
                },
                "area_px": float(area_px),
                "start_cm": round(start_cm, 3),
                "end_cm": round(end_cm, 3),
                "width_cm": round(width_cm, 3),
                "height_cm": round(height_cm, 3),
                "center_cm": round(center_cm, 3),
                "fill_ratio": 1.0,
                "aspect_ratio": round(aspect, 3),
                "shelf_w_px": int(shelf_w_px),
                "shelf_width_cm": float(shelf_width_cm),
                "conf": round(conf, 3),
                "cls_id": cls_id,
                "cls_name": cls_name,
                "source": "shelf_yolo_async",
            }
        )

    kept, nms_rejected = shelf_yolo_class_agnostic_nms(
        raw_candidates,
        iou_thres=0.35,
        containment_thres=0.65,
    )

    rejected.extend(nms_rejected)
    shelf_yolo_results_to_candidates.last_rejected = rejected

    return kept


def build_shelf_mask_from_candidates(candidates, shelf_w_px, shelf_h_px):
    """
    Synthetic mask from YOLO boxes.
    This lets the existing monitor/outbound logic still use monitor_mask-style
    presence checks even when shelf detection comes from YOLO.
    """
    mask = np.zeros((int(shelf_h_px), int(shelf_w_px)), dtype=np.uint8)

    for cand in candidates or []:
        b = cand.get("bbox_rect", {}) or {}

        x = int(b.get("x", 0))
        y = int(b.get("y", 0))
        w = int(b.get("w", 0))
        h = int(b.get("h", 0))

        x1 = max(0, min(x, int(shelf_w_px) - 1))
        y1 = max(0, min(y, int(shelf_h_px) - 1))
        x2 = max(0, min(x + w, int(shelf_w_px)))
        y2 = max(0, min(y + h, int(shelf_h_px)))

        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return mask


class ShelfYOLOAsyncWorker:
    """
    Runs YOLO-Worldv2 shelf detection in a background thread.
    Main loop stays smooth and uses the latest YOLO candidates.
    """
    def __init__(
        self,
        model_path,
        classes,
        conf,
        imgsz,
        interval_sec,
        min_area_px,
        shelf_config,
        shelf_w_px,
        shelf_h_px,
    ):
        self.model_path = model_path
        self.classes = classes
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.interval_sec = float(interval_sec)
        self.min_area_px = int(min_area_px)
        self.shelf_config = shelf_config
        self.shelf_w_px = int(shelf_w_px)
        self.shelf_h_px = int(shelf_h_px)

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.pending_frame = None
        self.latest_candidates = None
        self.latest_mask = None
        self.latest_timestamp = 0.0
        self.latest_infer_ms = 0.0
        self.latest_rejected = []
        self.latest_error = None
        self.version = 0

        self.model = None
        self.thread = None

    def start(self):
        if not os.path.exists(self.model_path):
            print(f"[SHELF-YOLO] Model not found: {self.model_path}")
            return False

        print("")
        print("======================================")
        print("[SHELF-YOLO] Loading async shelf detector")
        print(f"[SHELF-YOLO] Model: {self.model_path}")
        print(f"[SHELF-YOLO] Classes: {self.classes}")
        print(f"[SHELF-YOLO] imgsz={self.imgsz}, conf={self.conf}, interval={self.interval_sec}s")
        print("======================================")

        try:
            if "world" in self.model_path.lower():
                from ultralytics import YOLOWorld
                self.model = YOLOWorld(self.model_path)
                class_list = [x.strip() for x in str(self.classes).split(",") if x.strip()]
                self.model.set_classes(class_list)
                print(f"[SHELF-YOLO] YOLO-World classes set: {class_list}")
            else:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path, task="detect")

        except Exception as e:
            print(f"[SHELF-YOLO] Failed to load model: {e}")
            self.latest_error = str(e)
            return False

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return True

    def submit(self, warped_shelf):
        if warped_shelf is None:
            return

        with self.lock:
            self.pending_frame = warped_shelf.copy()

    def get_latest(self, max_age_sec=5.0):
        with self.lock:
            if self.latest_candidates is None:
                return None, None, {
                    "ready": False,
                    "age_sec": None,
                    "infer_ms": self.latest_infer_ms,
                    "version": self.version,
                    "error": self.latest_error,
                    "rejected": list(self.latest_rejected or []),
                }

            age = time.time() - float(self.latest_timestamp)

            info = {
                "ready": age <= float(max_age_sec),
                "age_sec": age,
                "infer_ms": self.latest_infer_ms,
                "version": self.version,
                "error": self.latest_error,
                "rejected": list(self.latest_rejected or []),
            }

            if age > float(max_age_sec):
                return None, None, info

            return (
                [clone_candidate_for_tracking(c) for c in self.latest_candidates],
                self.latest_mask.copy() if self.latest_mask is not None else None,
                info,
            )

    def stop(self):
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def _loop(self):
        while not self.stop_event.is_set():
            frame = None

            with self.lock:
                if self.pending_frame is not None:
                    frame = self.pending_frame.copy()
                    self.pending_frame = None

            if frame is None:
                time.sleep(0.03)
                continue

            try:
                t0 = time.time()

                results = self.model.predict(
                    frame,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    verbose=False,
                )

                infer_ms = (time.time() - t0) * 1000.0

                candidates = shelf_yolo_results_to_candidates(
                    result=results[0] if results else None,
                    model=self.model,
                    shelf_config=self.shelf_config,
                    shelf_w_px=self.shelf_w_px,
                    shelf_h_px=self.shelf_h_px,
                    conf_thres=self.conf,
                    min_area_px=self.min_area_px,
                    allowed_classes=self.classes,
                )

                mask = build_shelf_mask_from_candidates(
                    candidates=candidates,
                    shelf_w_px=self.shelf_w_px,
                    shelf_h_px=self.shelf_h_px,
                )

                rejected = getattr(shelf_yolo_results_to_candidates, "last_rejected", [])

                with self.lock:
                    self.latest_candidates = candidates
                    self.latest_mask = mask
                    self.latest_timestamp = time.time()
                    self.latest_infer_ms = infer_ms
                    self.latest_rejected = rejected
                    self.latest_error = None
                    self.version += 1

            except Exception as e:
                with self.lock:
                    self.latest_error = str(e)

                print(f"[SHELF-YOLO] inference error: {e}")

            time.sleep(max(0.05, self.interval_sec))


def interval_overlap(a_start, a_end, b_start, b_end):
    left = max(a_start, b_start)
    right = min(a_end, b_end)

    if right <= left:
        return 0.0

    return right - left


def evaluate_candidate(candidate, placement, accept_zone, item_width_cm, min_overlap_ratio, max_width_factor):
    if candidate["level_index"] != placement["level_index"]:
        return {
            "accepted": False,
            "reason": "wrong_level",
            "score": 0.0,
        }

    c_start = float(candidate["start_cm"])
    c_end = float(candidate["end_cm"])
    c_width = max(0.001, c_end - c_start)

    z_start = float(accept_zone["start_cm"])
    z_end = float(accept_zone["end_cm"])

    max_allowed_width = max(item_width_cm * max_width_factor, item_width_cm + 1.5)

    if c_width > max_allowed_width:
        return {
            "accepted": False,
            "reason": "too_wide_maybe_hand",
            "score": 0.0,
        }

    overlap = interval_overlap(c_start, c_end, z_start, z_end)
    overlap_ratio = overlap / c_width

    center_inside = z_start <= candidate["center_cm"] <= z_end

    accepted = center_inside or overlap_ratio >= min_overlap_ratio

    return {
        "accepted": accepted,
        "reason": "accepted" if accepted else "low_overlap",
        "score": round(overlap_ratio, 3),
        "overlap_cm": round(overlap, 3),
        "center_inside": center_inside,
    }


def choose_best_candidate(candidates, placement, accept_zone, item_width_cm, min_overlap_ratio, max_width_factor):
    evaluated = []

    for cand in candidates:
        ev = evaluate_candidate(
            candidate=cand,
            placement=placement,
            accept_zone=accept_zone,
            item_width_cm=item_width_cm,
            min_overlap_ratio=min_overlap_ratio,
            max_width_factor=max_width_factor,
        )

        item = dict(cand)
        item["evaluation"] = ev
        evaluated.append(item)

    accepted = [x for x in evaluated if x["evaluation"]["accepted"]]

    if accepted:
        accepted.sort(
            key=lambda x: (
                x["evaluation"]["score"],
                x["area_px"],
            ),
            reverse=True,
        )
        return accepted[0], evaluated

    if evaluated:
        evaluated.sort(key=lambda x: x["area_px"], reverse=True)
        return evaluated[0], evaluated

    return None, evaluated


# ============================================================
# Item / QR / DB
# ============================================================

def generate_item_id():
    return "ITEM_" + uuid.uuid4().hex[:6].upper()


def generate_qr_code(item_id):
    os.makedirs(QR_DIR, exist_ok=True)

    qr_path = os.path.join(QR_DIR, f"{item_id}.png")

    qr = qrcode.QRCode(
        version=2,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )

    qr.add_data(item_id)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(qr_path)

    return qr_path


def create_item(width_cm, height_cm, note="smart warehouse auto item"):
    item_id = generate_item_id()
    qr_path = generate_qr_code(item_id)
    now = datetime.now().isoformat(timespec="seconds")

    item = {
        "item_id": item_id,
        "qr_data": item_id,
        "size_cm": {
            "width": width_cm,
            "height": height_cm,
            "depth": 0.0,
        },
        "status": "waiting_for_placement",
        "suggested_position": None,
        "placed_position": None,
        "qr_path": qr_path,
        "note": note,
        "created_at": now,
        "updated_at": now,
    }

    db = load_json_default(ITEM_DB_PATH, {"items": []})
    db["items"].append(item)
    save_json(db, ITEM_DB_PATH)

    save_json(
        {
            "item_id": item_id,
            "status": "waiting_for_placement",
            "size_cm": item["size_cm"],
            "qr_path": qr_path,
            "created_at": now,
        },
        PENDING_ITEM_PATH,
    )

    return item


def update_items_db_after_placement(
    item,
    placement,
    accept_zone,
    actual,
    state=None,
    occupied_padding_cm=0.4,
    item_gap_cm=0.3,
):
    db = load_json_default(ITEM_DB_PATH, {"items": []})
    now = datetime.now().isoformat(timespec="seconds")

    if state is not None:
        reserved = compute_reserved_interval(
            state=state,
            item=item,
            actual=actual,
            occupied_padding_cm=occupied_padding_cm,
            item_gap_cm=item_gap_cm,
        )
    else:
        reserved = {
            "actual_start_cm": actual["start_cm"],
            "actual_end_cm": actual["end_cm"],
            "actual_width_cm": actual["width_cm"],
            "reserved_start_cm": actual["start_cm"],
            "reserved_end_cm": actual["end_cm"],
            "reserved_width_cm": actual["width_cm"],
            "occupied_padding_cm": occupied_padding_cm,
            "item_gap_cm": item_gap_cm,
        }

    suggested_position = {
        "shelf_id": placement["shelf_id"],
        "level_id": placement["level_id"],
        "level_index": placement["level_index"],
        "start_cm": placement["start_cm"],
        "end_cm": placement["end_cm"],
        "position_label": (
            f"{placement['shelf_id']}-{placement['level_id']}-"
            f"X{placement['start_cm']}-{placement['end_cm']}"
        ),
        "accept_zone": accept_zone,
    }

    actual_position = {
        "shelf_id": placement["shelf_id"],
        "level_id": actual["level_id"],
        "level_index": actual["level_index"],
        "actual_start_cm": reserved["actual_start_cm"],
        "actual_end_cm": reserved["actual_end_cm"],
        "actual_width_cm": reserved["actual_width_cm"],
        "reserved_start_cm": reserved["reserved_start_cm"],
        "reserved_end_cm": reserved["reserved_end_cm"],
        "reserved_width_cm": reserved["reserved_width_cm"],
        "position_label": (
            f"{placement['shelf_id']}-{actual['level_id']}-"
            f"X{reserved['reserved_start_cm']}-{reserved['reserved_end_cm']}"
        ),
        "bbox_rect": actual.get("bbox_rect"),
        "area_px": actual.get("area_px"),
        "occupied_padding_cm": reserved["occupied_padding_cm"],
        "item_gap_cm": reserved["item_gap_cm"],
        "source": "smart_warehouse_ai_vision",
    }

    found = False

    for db_item in db["items"]:
        if db_item.get("item_id") == item["item_id"]:
            db_item["status"] = "placed"
            db_item["suggested_position"] = suggested_position
            db_item["placed_position"] = actual_position
            db_item["updated_at"] = now
            found = True
            break

    if not found:
        item_copy = dict(item)
        item_copy["status"] = "placed"
        item_copy["suggested_position"] = suggested_position
        item_copy["placed_position"] = actual_position
        item_copy["updated_at"] = now
        db["items"].append(item_copy)

    save_json(db, ITEM_DB_PATH)

    return suggested_position, actual_position


def clear_pending_item():
    if os.path.exists(PENDING_ITEM_PATH):
        os.remove(PENDING_ITEM_PATH)


# ============================================================
# Drawing
# ============================================================

def draw_quad(frame, config, color, label):
    output = frame

    tl, tr, br, bl = get_quad_points(config)
    poly = np.array([tl, tr, br, bl], dtype=np.int32)

    cv2.polylines(output, [poly], True, color, 2)

    cv2.putText(
        output,
        label,
        tuple(tl.astype(int) + np.array([5, -8])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
    )


def draw_shelf_levels(frame, shelf_config):
    tl, tr, br, bl = get_quad_points(shelf_config)
    shelf_poly = np.array([tl, tr, br, bl], dtype=np.int32)

    cv2.polylines(frame, [shelf_poly], True, (0, 255, 0), 3)

    num_levels = int(shelf_config.get("num_levels", 4))

    for i in range(num_levels + 1):
        alpha = i / num_levels

        left = (1 - alpha) * tl + alpha * bl
        right = (1 - alpha) * tr + alpha * br

        left_i = tuple(left.astype(int))
        right_i = tuple(right.astype(int))

        cv2.line(frame, left_i, right_i, (0, 255, 255), 1)

        if i < num_levels:
            cv2.putText(
                frame,
                f"T{i + 1}",
                (max(5, left_i[0] - 35), left_i[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )


def draw_measure_bbox_on_original(frame, measure_config, bbox, H_measure_inv):
    if bbox is None:
        return

    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

    poly_rect = np.array(
        [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ],
        dtype=np.float32,
    )

    poly_orig = rect_poly_to_original(poly_rect, H_measure_inv)
    cv2.polylines(frame, [poly_orig], True, (255, 0, 255), 2)


def draw_interval_on_original(
    frame,
    shelf_config,
    H_shelf_inv,
    shelf_w_px,
    shelf_h_px,
    level_index,
    start_cm,
    end_cm,
    color,
    label,
    thickness=2,
    fill=False,
    alpha=0.25,
):
    poly_rect = interval_to_rect_poly(
        config=shelf_config,
        shelf_w_px=shelf_w_px,
        shelf_h_px=shelf_h_px,
        level_index=level_index,
        start_cm=start_cm,
        end_cm=end_cm,
    )

    poly_orig = rect_poly_to_original(poly_rect, H_shelf_inv)

    if fill:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly_orig], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.polylines(frame, [poly_orig], True, color, thickness)

    cx = int(np.mean(poly_orig[:, 0]))
    cy = int(np.mean(poly_orig[:, 1]))

    cv2.putText(
        frame,
        label,
        (cx - 45, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
    )


def draw_shelf_candidates(frame, candidates, H_shelf_inv):
    for cand in candidates:
        b = cand["bbox_rect"]

        poly_rect = np.array(
            [
                [b["x"], b["y"]],
                [b["x"] + b["w"], b["y"]],
                [b["x"] + b["w"], b["y"] + b["h"]],
                [b["x"], b["y"] + b["h"]],
            ],
            dtype=np.float32,
        )

        poly_orig = rect_poly_to_original(poly_rect, H_shelf_inv)

        ev = cand.get("evaluation", {})
        accepted = ev.get("accepted", False)

        color = (0, 255, 0) if accepted else (0, 0, 255)

        cv2.polylines(frame, [poly_orig], True, color, 2)

        cx = int(np.mean(poly_orig[:, 0]))
        cy = int(np.mean(poly_orig[:, 1]))

        label = f"{cand['level_id']} {cand['start_cm']}-{cand['end_cm']} {ev.get('reason', '')}"

        cv2.putText(
            frame,
            label,
            (cx - 45, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
        )



def _ui_put_text(img, text, org, font_scale=0.55, color=(230, 230, 230), thickness=1):
    cv2.putText(
        img,
        str(text),
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _ui_draw_card(img, x, y, w, h, title, value, accent=(0, 200, 255), value_color=(255, 255, 255)):
    cv2.rectangle(img, (x, y), (x + w, y + h), (20, 24, 30), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 80, 95), 1)
    cv2.rectangle(img, (x, y), (x + w, y + 4), accent, -1)
    _ui_put_text(img, title, (x + 12, y + 25), 0.47, (150, 165, 185), 1)
    _ui_put_text(img, value, (x + 12, y + 55), 0.58, value_color, 1)


def _ui_wrap_text(text, max_chars=48):
    words = str(text).split()
    lines = []
    cur = ""
    for word in words:
        if len(cur) + len(word) + 1 <= max_chars:
            cur = word if not cur else cur + " " + word
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def _ui_draw_lines(img, lines, x, y, line_h=22, font_scale=0.48, color=(220, 230, 240), max_lines=None):
    drawn = 0
    for line in lines:
        if max_lines is not None and drawn >= max_lines:
            _ui_put_text(img, "...", (x, y), font_scale, color, 1)
            return y + line_h
        _ui_put_text(img, line, (x, y), font_scale, color, 1)
        y += line_h
        drawn += 1
    return y


def _ui_make_mask_panel(mask, panel_w, panel_h, title, border_color):
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (55, 65, 80), 1)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, 24), (20, 24, 30), -1)
    _ui_put_text(panel, title, (8, 17), 0.45, border_color, 1)

    if mask is None:
        _ui_put_text(panel, "NOT READY", (panel_w // 2 - 55, panel_h // 2), 0.55, (120, 120, 120), 1)
        return panel

    if len(mask.shape) == 2:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        mask_bgr = mask.copy()

    content_h = panel_h - 30
    resized = cv2.resize(mask_bgr, (panel_w - 12, content_h - 8), interpolation=cv2.INTER_NEAREST)
    panel[28:28 + resized.shape[0], 6:6 + resized.shape[1]] = resized
    cv2.rectangle(panel, (5, 27), (panel_w - 6, panel_h - 6), border_color, 1)
    return panel


def make_display(frame, measure_mask, shelf_mask, messages):
    """
    Demo-friendly UI layout.

    Camera view is kept clean on the left.
    Measure/Shelf masks are moved to a right sidebar.
    Current stage, item, placement, status, monitor and controls are separated.
    """
    h, w = frame.shape[:2]

    header_h = 135
    side_w = 360
    ui_scale = float(os.environ.get("WAREHOUSE_UI_SCALE", str(DISPLAY_SCALE)))

    canvas_h = h + header_h
    canvas_w = w + side_w
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:, :] = (8, 11, 16)

    stage_line = messages[0] if len(messages) > 0 else "Stage: unknown"
    item_line = messages[1] if len(messages) > 1 else "Current item: none"
    place_line = messages[2] if len(messages) > 2 else "Placement: none"
    status_line = messages[3] if len(messages) > 3 else "Status: unknown"
    monitor_line = messages[4] if len(messages) > 4 else "Monitor: unknown"

    stage_value = stage_line
    bg_value = ""
    if "|" in stage_line:
        parts = stage_line.split("|", 1)
        stage_value = parts[0].replace("Stage:", "").strip()
        bg_value = parts[1].strip()
    else:
        stage_value = stage_line.replace("Stage:", "").strip()

    status_value = status_line.replace("Status:", "").strip()
    item_value = item_line.strip()
    place_value = place_line.strip()

    stage_color = (0, 200, 255)
    if "OUTBOUND" in stage_value:
        stage_color = (0, 80, 255)
    elif "WAIT_PLACE" in stage_value:
        stage_color = (255, 0, 255)
    elif "REVIEW" in stage_value:
        stage_color = (0, 220, 255)
    elif "WAIT_ITEM" in stage_value:
        stage_color = (0, 210, 120)

    monitor_ok = "WARNING" not in monitor_line and "MISSING" not in monitor_line and "UNKNOWN" not in monitor_line
    monitor_color = (0, 220, 80) if monitor_ok else (0, 0, 255)

    # Header
    cv2.rectangle(canvas, (0, 0), (canvas_w, header_h), (10, 14, 20), -1)
    _ui_put_text(canvas, "SMART WAREHOUSE AI - EDGE DASHBOARD", (18, 28), 0.62, (0, 230, 255), 2)
    _ui_put_text(canvas, bg_value, (18, 55), 0.48, (155, 170, 190), 1)

    card_y = 70
    gap = 12
    card_w = (w - 36 - gap * 2) // 3

    _ui_draw_card(canvas, 18, card_y, card_w, 52, "MODE", stage_value, stage_color)
    _ui_draw_card(canvas, 18 + card_w + gap, card_y, card_w, 52, "ITEM", item_value[:42], (0, 200, 255))
    _ui_draw_card(canvas, 18 + (card_w + gap) * 2, card_y, card_w, 52, "PLACEMENT", place_value[:42], (255, 170, 0))

    # Main camera area
    canvas[header_h:header_h + h, 0:w] = frame
    cv2.rectangle(canvas, (0, header_h), (w - 1, header_h + h - 1), (55, 65, 80), 1)

    # Sidebar
    sx = w
    cv2.rectangle(canvas, (sx, 0), (canvas_w - 1, canvas_h - 1), (12, 16, 23), -1)
    cv2.line(canvas, (sx, 0), (sx, canvas_h), (65, 75, 90), 1)

    _ui_put_text(canvas, "VISION PANELS", (sx + 16, 32), 0.58, (0, 230, 255), 2)

    panel_w = side_w - 28
    panel_h = 170

    measure_panel = _ui_make_mask_panel(measure_mask, panel_w, panel_h, "MEASURE MASK", (255, 0, 255))
    shelf_panel = _ui_make_mask_panel(shelf_mask, panel_w, panel_h, "SHELF MASK", (0, 255, 255))

    y = 48
    canvas[y:y + panel_h, sx + 14:sx + 14 + panel_w] = measure_panel

    y += panel_h + 16
    canvas[y:y + panel_h, sx + 14:sx + 14 + panel_w] = shelf_panel

    y += panel_h + 22

    # Status
    cv2.rectangle(canvas, (sx + 14, y), (canvas_w - 14, y + 118), (20, 24, 30), -1)
    cv2.rectangle(canvas, (sx + 14, y), (canvas_w - 14, y + 118), (70, 80, 95), 1)
    cv2.rectangle(canvas, (sx + 14, y), (canvas_w - 14, y + 4), stage_color, -1)

    _ui_put_text(canvas, "STATUS", (sx + 26, y + 25), 0.50, (150, 165, 185), 1)
    wrapped_status = _ui_wrap_text(status_value, max_chars=38)
    _ui_draw_lines(
        canvas,
        wrapped_status,
        sx + 26,
        y + 52,
        line_h=20,
        font_scale=0.45,
        color=(245, 245, 245),
        max_lines=3,
    )

    y += 132

    # Monitor
    monitor_box_h = 115
    cv2.rectangle(canvas, (sx + 14, y), (canvas_w - 14, min(y + monitor_box_h, canvas_h - 12)), (20, 24, 30), -1)
    cv2.rectangle(canvas, (sx + 14, y), (canvas_w - 14, min(y + monitor_box_h, canvas_h - 12)), (70, 80, 95), 1)
    cv2.rectangle(canvas, (sx + 14, y), (canvas_w - 14, y + 4), monitor_color, -1)

    _ui_put_text(canvas, "MONITOR", (sx + 26, y + 25), 0.50, (150, 165, 185), 1)

    monitor_text = monitor_line.replace("MONITOR OK |", "OK |").replace("MONITOR WARNING |", "WARNING |")
    wrapped_monitor = _ui_wrap_text(monitor_text, max_chars=38)
    _ui_draw_lines(
        canvas,
        wrapped_monitor,
        sx + 26,
        y + 52,
        line_h=19,
        font_scale=0.42,
        color=(230, 245, 230) if monitor_ok else (220, 220, 255),
        max_lines=3,
    )

    y += 128

    # Controls
    _ui_put_text(canvas, "CONTROLS", (sx + 16, y), 0.50, (150, 165, 185), 1)

    controls = [
        "b: measure background",
        "k: shelf background",
        "c: confirm / create QR",
        "r: re-measure",
        "o: outbound mode",
        "e: export / remove",
        "x: cancel    q: quit",
    ]

    _ui_draw_lines(
        canvas,
        controls,
        sx + 18,
        y + 24,
        line_h=19,
        font_scale=0.42,
        color=(210, 225, 240),
        max_lines=8,
    )

    display = cv2.resize(
        canvas,
        None,
        fx=ui_scale,
        fy=ui_scale,
        interpolation=cv2.INTER_LINEAR,
    )

    return display


def candidate_is_position_stable(previous_candidate, current_candidate, tolerance_cm):
    if previous_candidate is None:
        return False

    if previous_candidate.get("level_index") != current_candidate.get("level_index"):
        return False

    prev_center = float(previous_candidate.get("center_cm", 0.0))
    curr_center = float(current_candidate.get("center_cm", 0.0))

    prev_width = float(previous_candidate.get("width_cm", 0.0))
    curr_width = float(current_candidate.get("width_cm", 0.0))

    center_diff = abs(curr_center - prev_center)
    width_diff = abs(curr_width - prev_width)

    return center_diff <= tolerance_cm and width_diff <= tolerance_cm


# ============================================================
# Shelf tracking / monitoring level 1
# ============================================================

def get_expected_counts_by_level(state):
    counts = {}

    for level in state.get("levels", []):
        level_id = level.get("level_id", f"T{level.get('level_index')}")
        counts[level_id] = len(level.get("occupied_intervals", []))

    return counts


def get_detected_counts_by_level(live_candidates, num_levels):
    counts = {f"T{i + 1}": 0 for i in range(num_levels)}

    for cand in live_candidates:
        level_id = cand.get("level_id")
        if level_id in counts:
            counts[level_id] += 1

    return counts


def build_monitor_summary(state, live_candidates, shelf_config):
    num_levels = int(shelf_config.get("num_levels", 4))

    expected = get_expected_counts_by_level(state)
    detected = get_detected_counts_by_level(live_candidates, num_levels)

    parts = []
    has_warning = False

    for i in range(num_levels):
        level_id = f"T{i + 1}"
        exp = expected.get(level_id, 0)
        det = detected.get(level_id, 0)

        if exp == det:
            status = "OK"
        elif det < exp:
            status = "MISSING?"
            has_warning = True
        else:
            status = "UNKNOWN?"
            has_warning = True

        parts.append(f"{level_id}: exp={exp}, det={det}, {status}")

    if has_warning:
        prefix = "MONITOR WARNING"
    else:
        prefix = "MONITOR OK"

    return prefix + " | " + " | ".join(parts)


def get_visual_registered_interval_cm(occ, shelf_width_cm, visual_padding_cm=0.12):
    """
    Compute a compact interval only for drawing the yellow REG box.

    Logic:
    - start_cm/end_cm in shelf_state may be a reserved interval with padding/gap.
    - actual_start_cm/actual_end_cm is closer to the detected object.
    - suggested_end_cm - suggested_start_cm is close to the measured item width.
    - For display, use measured/suggested width centered around actual center.
    """
    reg_start = float(occ.get("start_cm", occ.get("actual_start_cm", 0.0)))
    reg_end = float(occ.get("end_cm", occ.get("actual_end_cm", reg_start)))

    if reg_end < reg_start:
        reg_start, reg_end = reg_end, reg_start

    actual_start = float(occ.get("actual_start_cm", reg_start))
    actual_end = float(occ.get("actual_end_cm", reg_end))

    if actual_end < actual_start:
        actual_start, actual_end = actual_end, actual_start

    actual_center = (actual_start + actual_end) / 2.0

    # Use suggested width as measured item width if available.
    suggested_start = occ.get("suggested_start_cm", None)
    suggested_end = occ.get("suggested_end_cm", None)

    if suggested_start is not None and suggested_end is not None:
        measured_width = abs(float(suggested_end) - float(suggested_start))
    else:
        measured_width = abs(actual_end - actual_start)

    measured_width = max(0.35, measured_width)
    visual_width = measured_width + 2.0 * float(visual_padding_cm)

    # Do not draw wider than reserved interval.
    reserved_width = max(0.001, reg_end - reg_start)
    visual_width = min(visual_width, reserved_width)

    visual_start = actual_center - visual_width / 2.0
    visual_end = actual_center + visual_width / 2.0

    # Keep the visual interval inside the reserved interval when possible.
    if visual_start < reg_start:
        visual_end += (reg_start - visual_start)
        visual_start = reg_start

    if visual_end > reg_end:
        visual_start -= (visual_end - reg_end)
        visual_end = reg_end

    visual_start = max(0.0, min(visual_start, shelf_width_cm))
    visual_end = max(0.0, min(visual_end, shelf_width_cm))

    if visual_end <= visual_start:
        visual_start = reg_start
        visual_end = reg_end

    return round(visual_start, 3), round(visual_end, 3)



def draw_registered_items_on_original(
    frame,
    state,
    shelf_config,
    H_shelf_inv,
    shelf_w_px,
    shelf_h_px,
):
    """
    Draw yellow REG box.

    REG is the registered item location. For UI clarity, this function draws
    a compact visual interval instead of the full reserved interval.
    The reserved interval is still kept in shelf_state for placement planning.
    """
    shelf_width_cm = float(shelf_config["physical_size_cm"]["width"])

    for level in state.get("levels", []):
        level_index = int(level.get("level_index", 1))

        for occ in level.get("occupied_intervals", []):
            item_id = occ.get("item_id", "ITEM")

            visual_start, visual_end = get_visual_registered_interval_cm(
                occ=occ,
                shelf_width_cm=shelf_width_cm,
                visual_padding_cm=0.12,
            )

            draw_interval_on_original(
                frame=frame,
                shelf_config=shelf_config,
                H_shelf_inv=H_shelf_inv,
                shelf_w_px=shelf_w_px,
                shelf_h_px=shelf_h_px,
                level_index=level_index,
                start_cm=visual_start,
                end_cm=visual_end,
                color=(0, 255, 255),
                label=f"REG {item_id}",
                thickness=2,
                fill=False,
            )


def draw_live_occupancy_on_original(frame, live_candidates, H_shelf_inv):
    for cand in live_candidates:
        b = cand["bbox_rect"]

        poly_rect = np.array(
            [
                [b["x"], b["y"]],
                [b["x"] + b["w"], b["y"]],
                [b["x"] + b["w"], b["y"] + b["h"]],
                [b["x"], b["y"] + b["h"]],
            ],
            dtype=np.float32,
        )

        poly_orig = rect_poly_to_original(poly_rect, H_shelf_inv)

        cv2.polylines(frame, [poly_orig], True, (0, 255, 0), 2)

        cx = int(np.mean(poly_orig[:, 0]))
        cy = int(np.mean(poly_orig[:, 1]))

        label = f"LIVE {cand['level_id']}"

        cv2.putText(
            frame,
            label,
            (cx - 35, cy + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
        )


# ============================================================
# Event log + monitor warning debounce
# ============================================================


# ---------- Cloud backend event forwarding ----------
WAREHOUSE_CLOUD_URL = os.environ.get("WAREHOUSE_CLOUD_URL", "").rstrip("/")
WAREHOUSE_CLOUD_TIMEOUT = float(os.environ.get("WAREHOUSE_CLOUD_TIMEOUT", "6"))
WAREHOUSE_REMOVE_POLL_INTERVAL = float(os.environ.get("WAREHOUSE_REMOVE_POLL_INTERVAL", "3"))


# ---------- MQTT transport from Jetson ----------
# v4 architecture:
#   - Jetson -> MQTT Broker -> Cloud Backend for warehouse events/metadata.
#   - Cloud Backend -> MQTT Broker -> Jetson for dashboard remove/export commands.
#   - Jetson -> MQTT Broker -> ESP32 for LED commands.
# HTTP is kept only as an optional fallback/debug transport.
WAREHOUSE_EVENT_TRANSPORT = os.environ.get("WAREHOUSE_EVENT_TRANSPORT", "mqtt").strip().lower()
WAREHOUSE_REMOVE_REQUEST_TRANSPORT = os.environ.get("WAREHOUSE_REMOVE_REQUEST_TRANSPORT", "mqtt").strip().lower()
WAREHOUSE_MQTT_EVENT_ENABLED = os.environ.get("WAREHOUSE_MQTT_EVENT_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
WAREHOUSE_MQTT_COMMAND_ENABLED = os.environ.get("WAREHOUSE_MQTT_COMMAND_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
WAREHOUSE_MQTT_LED_ENABLED = os.environ.get("WAREHOUSE_MQTT_LED_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
WAREHOUSE_MQTT_HOST = os.environ.get("WAREHOUSE_MQTT_HOST", "").strip()
WAREHOUSE_MQTT_PORT = int(os.environ.get("WAREHOUSE_MQTT_PORT", "1883"))
WAREHOUSE_MQTT_USERNAME = os.environ.get("WAREHOUSE_MQTT_USERNAME", "")
WAREHOUSE_MQTT_PASSWORD = os.environ.get("WAREHOUSE_MQTT_PASSWORD", "")
WAREHOUSE_MQTT_TLS = os.environ.get("WAREHOUSE_MQTT_TLS", "false").strip().lower() in ("1", "true", "yes", "on")
WAREHOUSE_MQTT_TOPIC_PREFIX = os.environ.get("WAREHOUSE_MQTT_TOPIC_PREFIX", "warehouse").strip().strip("/") or "warehouse"
WAREHOUSE_MQTT_EVENT_TOPIC = os.environ.get("WAREHOUSE_MQTT_EVENT_TOPIC", f"{WAREHOUSE_MQTT_TOPIC_PREFIX}/jetson/events").strip()
WAREHOUSE_MQTT_COMMAND_TOPIC = os.environ.get("WAREHOUSE_MQTT_COMMAND_TOPIC", f"{WAREHOUSE_MQTT_TOPIC_PREFIX}/jetson/commands").strip()
WAREHOUSE_MQTT_QOS = int(os.environ.get("WAREHOUSE_MQTT_QOS", "1"))
WAREHOUSE_MQTT_COMMAND_QOS = int(os.environ.get("WAREHOUSE_MQTT_COMMAND_QOS", str(WAREHOUSE_MQTT_QOS)))
WAREHOUSE_MQTT_RETAIN = os.environ.get("WAREHOUSE_MQTT_RETAIN", "false").strip().lower() in ("1", "true", "yes", "on")
WAREHOUSE_MQTT_EVENT_RETAIN = os.environ.get("WAREHOUSE_MQTT_EVENT_RETAIN", "false").strip().lower() in ("1", "true", "yes", "on")
WAREHOUSE_MQTT_LED_BLINK_MS = int(os.environ.get("WAREHOUSE_MQTT_LED_BLINK_MS", "500"))
WAREHOUSE_MQTT_LED_TIMEOUT_MS = int(os.environ.get("WAREHOUSE_MQTT_LED_TIMEOUT_MS", "120000"))

_mqtt_led_client = None
_mqtt_led_ready = False
_mqtt_command_queue = deque(maxlen=50)
_mqtt_seen_command_ids = set()


def mqtt_led_topic(shelf_id):
    shelf = str(shelf_id or "SHELF_A").strip() or "SHELF_A"
    return f"{WAREHOUSE_MQTT_TOPIC_PREFIX}/{shelf}/led/command"


def init_mqtt_led_client():
    """
    Start one MQTT client for both:
      1) warehouse event publishing to cloud backend subscriber, and
      2) LED command publishing to ESP32.

    The function name is kept for compatibility with the existing main() flow.
    Failure must not stop camera/AI execution.
    """
    global _mqtt_led_client, _mqtt_led_ready

    mqtt_needed = (
        (WAREHOUSE_EVENT_TRANSPORT in ("mqtt", "both") and WAREHOUSE_MQTT_EVENT_ENABLED)
        or (WAREHOUSE_REMOVE_REQUEST_TRANSPORT in ("mqtt", "both") and WAREHOUSE_MQTT_COMMAND_ENABLED)
        or WAREHOUSE_MQTT_LED_ENABLED
    )

    if not mqtt_needed:
        print("[MQTT] Disabled. Event transport and LED command publishing are not using MQTT.")
        return

    if not WAREHOUSE_MQTT_HOST:
        print("[MQTT] Disabled: WAREHOUSE_MQTT_HOST is empty")
        return

    try:
        import paho.mqtt.client as mqtt

        client_id = os.environ.get("WAREHOUSE_MQTT_CLIENT_ID", f"jetson-warehouse-v4-{uuid.uuid4().hex[:8]}")
        clean_session = os.environ.get("WAREHOUSE_MQTT_CLEAN_SESSION", "true").strip().lower() in ("1", "true", "yes", "on")
        client = mqtt.Client(client_id=client_id, clean_session=clean_session)

        if WAREHOUSE_MQTT_USERNAME:
            client.username_pw_set(WAREHOUSE_MQTT_USERNAME, WAREHOUSE_MQTT_PASSWORD)

        if WAREHOUSE_MQTT_TLS:
            client.tls_set()

        def on_connect(client, userdata, flags, rc):
            global _mqtt_led_ready
            _mqtt_led_ready = (rc == 0)
            print(
                f"[MQTT] Connected rc={rc} "
                f"host={WAREHOUSE_MQTT_HOST}:{WAREHOUSE_MQTT_PORT} tls={WAREHOUSE_MQTT_TLS}"
            )
            print(f"[MQTT-EVENT] topic={WAREHOUSE_MQTT_EVENT_TOPIC} transport={WAREHOUSE_EVENT_TRANSPORT}")
            print(f"[MQTT-COMMAND] topic={WAREHOUSE_MQTT_COMMAND_TOPIC} transport={WAREHOUSE_REMOVE_REQUEST_TRANSPORT}")
            print(f"[MQTT-LED] prefix={WAREHOUSE_MQTT_TOPIC_PREFIX} enabled={WAREHOUSE_MQTT_LED_ENABLED}")

            if (
                rc == 0
                and WAREHOUSE_REMOVE_REQUEST_TRANSPORT in ("mqtt", "both")
                and WAREHOUSE_MQTT_COMMAND_ENABLED
            ):
                try:
                    client.subscribe(WAREHOUSE_MQTT_COMMAND_TOPIC, qos=WAREHOUSE_MQTT_COMMAND_QOS)
                    print(f"[MQTT-COMMAND] Subscribed {WAREHOUSE_MQTT_COMMAND_TOPIC} qos={WAREHOUSE_MQTT_COMMAND_QOS}")
                except Exception as e:
                    print(f"[MQTT-COMMAND] Subscribe failed: {e}")

        def on_disconnect(client, userdata, rc):
            global _mqtt_led_ready
            _mqtt_led_ready = False
            print(f"[MQTT] Disconnected rc={rc}")

        def on_message(client, userdata, msg):
            try:
                topic = getattr(msg, "topic", "")
                if topic != WAREHOUSE_MQTT_COMMAND_TOPIC:
                    return

                raw = msg.payload.decode("utf-8", errors="ignore")
                if not raw.strip():
                    return

                cmd = json.loads(raw)
                if not isinstance(cmd, dict):
                    print("[MQTT-COMMAND] Ignored non-dict payload")
                    return

                command_name = str(cmd.get("command") or cmd.get("type") or "").strip().lower()
                if command_name not in ("remove_item", "remove_request", "outbound_remove"):
                    print(f"[MQTT-COMMAND] Ignored unsupported command={command_name}")
                    return

                command_id = str(cmd.get("command_id") or cmd.get("request_id") or "").strip()
                if command_id:
                    if command_id in _mqtt_seen_command_ids:
                        print(f"[MQTT-COMMAND] Duplicate skipped command_id={command_id}")
                        return
                    _mqtt_seen_command_ids.add(command_id)
                    if len(_mqtt_seen_command_ids) > 500:
                        _mqtt_seen_command_ids.clear()

                _mqtt_command_queue.append(cmd)
                print(
                    f"[MQTT-COMMAND] Queued {command_name} / "
                    f"{cmd.get('item_id', '')} request={cmd.get('request_id', '')}"
                )

            except Exception as e:
                print(f"[MQTT-COMMAND] Message error: {e}")

        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.on_message = on_message
        client.connect(WAREHOUSE_MQTT_HOST, WAREHOUSE_MQTT_PORT, keepalive=30)
        client.loop_start()

        _mqtt_led_client = client
        print(f"[MQTT] Started client_id={client_id}")

    except Exception as e:
        _mqtt_led_client = None
        _mqtt_led_ready = False
        print(f"[MQTT] Init failed: {e}")


def stop_mqtt_led_client():
    global _mqtt_led_client, _mqtt_led_ready
    try:
        if _mqtt_led_client is not None:
            _mqtt_led_client.loop_stop()
            _mqtt_led_client.disconnect()
    except Exception as e:
        print(f"[MQTT] Stop error: {e}")
    finally:
        _mqtt_led_ready = False
        _mqtt_led_client = None


def publish_led_command(command, shelf_id=None):
    if not WAREHOUSE_MQTT_LED_ENABLED:
        return

    if _mqtt_led_client is None:
        print("[MQTT-LED] Skip publish: client not initialized")
        return

    try:
        shelf = shelf_id or command.get("shelf_id") or "SHELF_A"
        topic = mqtt_led_topic(shelf)
        payload = json.dumps(command, ensure_ascii=False, default=str)
        info = _mqtt_led_client.publish(topic, payload, qos=WAREHOUSE_MQTT_QOS, retain=WAREHOUSE_MQTT_RETAIN)
        print(f"[MQTT-LED] Published {command.get('command')} -> {topic}: {payload}")
        return info
    except Exception as e:
        print(f"[MQTT-LED] Publish error: {e}")
        return None


def publish_event_to_mqtt(event):
    """
    Publish one warehouse event to the MQTT broker.
    Cloud Backend v4 subscribes to WAREHOUSE_MQTT_EVENT_TOPIC and processes it.
    """
    if not WAREHOUSE_MQTT_EVENT_ENABLED:
        return

    if _mqtt_led_client is None:
        print("[MQTT-EVENT] Skip publish: client not initialized")
        return

    if not isinstance(event, dict):
        print("[MQTT-EVENT] Skip publish: event is not a dict")
        return

    try:
        event_for_mqtt = dict(event)
        event_for_mqtt.setdefault("source", "jetson-nano")
        event_for_mqtt.setdefault("transport", "mqtt")
        event_for_mqtt.setdefault("event_id", f"jetson-{uuid.uuid4().hex}")

        payload = json.dumps(event_for_mqtt, ensure_ascii=False, default=str)
        info = _mqtt_led_client.publish(
            WAREHOUSE_MQTT_EVENT_TOPIC,
            payload,
            qos=WAREHOUSE_MQTT_QOS,
            retain=WAREHOUSE_MQTT_EVENT_RETAIN,
        )
        print(
            f"[MQTT-EVENT] Published {event_for_mqtt.get('event_type')} / "
            f"{event_for_mqtt.get('item_id', '')} -> {WAREHOUSE_MQTT_EVENT_TOPIC}"
        )
        return info
    except Exception as e:
        print(f"[MQTT-EVENT] Publish error: {e}")
        return None


def publish_event_to_mqtt_async(event):
    try:
        threading.Thread(
            target=publish_event_to_mqtt,
            args=(event.copy() if isinstance(event, dict) else event,),
            daemon=True,
        ).start()
    except Exception as e:
        print(f"[MQTT-EVENT] Thread error: {e}")


def send_event_transport_async(event):
    """
    v4 event transport selector.
      - mqtt: Jetson -> MQTT Broker -> Cloud Backend subscriber
      - http: Jetson -> Cloud Backend /api/events
      - both: publish both, useful for temporary debugging only
      - none: do not forward events

    Remove/export request polling still uses HTTP through WAREHOUSE_CLOUD_URL.
    """
    transport = WAREHOUSE_EVENT_TRANSPORT

    if transport in ("mqtt", "both"):
        publish_event_to_mqtt_async(event)

    if transport in ("http", "both"):
        send_event_to_cloud_async(event)


def _position_from_payload(payload):
    if not isinstance(payload, dict):
        return {}

    for key in ("actual_position", "placed_position", "removed_position", "suggested_position", "position"):
        val = payload.get(key)
        if isinstance(val, dict):
            return val

    return {}


def led_blink_for_position(shelf_id, level_id, item_id=None, reason="placement_suggestion", request_id=None):
    if not level_id:
        print(f"[MQTT-LED] Skip blink: missing level_id for item={item_id}")
        return

    publish_led_command(
        {
            "command": "blink",
            "shelf_id": shelf_id or "SHELF_A",
            "level_id": level_id,
            "item_id": item_id,
            "request_id": request_id,
            "reason": reason,
            "blink_ms": WAREHOUSE_MQTT_LED_BLINK_MS,
            "timeout_ms": WAREHOUSE_MQTT_LED_TIMEOUT_MS,
            "source": "jetson-nano",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        shelf_id=shelf_id or "SHELF_A",
    )


def led_clear(shelf_id="SHELF_A", level_id=None, item_id=None, reason="clear", request_id=None):
    publish_led_command(
        {
            "command": "clear",
            "shelf_id": shelf_id or "SHELF_A",
            "level_id": level_id,
            "item_id": item_id,
            "request_id": request_id,
            "reason": reason,
            "source": "jetson-nano",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        shelf_id=shelf_id or "SHELF_A",
    )


def handle_led_side_effect_for_event(event_type, payload, event=None):
    """
    Keep AI/cloud flow unchanged; only add MQTT LED side-effects for key physical events.
    """
    try:
        if not isinstance(payload, dict):
            payload = {}

        event = event or {}
        item_id = event.get("item_id") or payload.get("item_id")

        if event_type == "item_created":
            pos = payload.get("suggested_position") or payload.get("position") or {}
            shelf_id = pos.get("shelf_id") or payload.get("shelf_id") or "SHELF_A"
            level_id = pos.get("level_id") or payload.get("level_id")
            led_blink_for_position(
                shelf_id=shelf_id,
                level_id=level_id,
                item_id=item_id,
                reason="placement_suggestion",
            )

        elif event_type == "item_placed":
            pos = _position_from_payload(payload)
            shelf_id = pos.get("shelf_id") or payload.get("shelf_id") or "SHELF_A"
            level_id = pos.get("level_id") or payload.get("level_id")
            led_clear(
                shelf_id=shelf_id,
                level_id=level_id,
                item_id=item_id,
                reason="item_placed",
            )

        elif event_type == "item_removed":
            pos = _position_from_payload(payload)
            shelf_id = pos.get("shelf_id") or payload.get("shelf_id") or "SHELF_A"
            level_id = pos.get("level_id") or payload.get("level_id")
            led_clear(
                shelf_id=shelf_id,
                level_id=level_id,
                item_id=item_id,
                reason="item_removed",
            )

    except Exception as e:
        print(f"[MQTT-LED] Event side-effect error: {e}")
# ---------- End MQTT transport from Jetson ----------


def send_event_to_cloud(event):
    """
    Send one warehouse event to the public cloud backend.
    Cloud failure must not stop the local Jetson/OpenCV workflow.
    """
    if not WAREHOUSE_CLOUD_URL:
        return

    if not isinstance(event, dict):
        print("[Cloud] Skip: event is not a dict")
        return

    try:
        url = WAREHOUSE_CLOUD_URL + "/api/events"
        data = json.dumps(event, ensure_ascii=False, default=str).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=WAREHOUSE_CLOUD_TIMEOUT) as resp:
            status = resp.getcode()
            body = resp.read(300).decode("utf-8", errors="ignore")

        if status >= 400:
            print(f"[Cloud] Send failed: HTTP {status} - {body}")
        else:
            print(f"[Cloud] Sent {event.get('event_type')} / {event.get('item_id', '')}")

    except urllib.error.HTTPError as e:
        try:
            body = e.read(300).decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        print(f"[Cloud] HTTP error: {e.code} - {body}")

    except Exception as e:
        print(f"[Cloud] Send error: {e}")


def send_event_to_cloud_async(event):
    """
    Send event in background thread to avoid blocking camera/OpenCV loop.
    """
    try:
        threading.Thread(
            target=send_event_to_cloud,
            args=(event.copy() if isinstance(event, dict) else event,),
            daemon=True
        ).start()
    except Exception as e:
        print(f"[Cloud] Thread error: {e}")


def fetch_pending_remove_requests_from_mqtt_queue():
    """
    Return pending remove/export requests received from Cloud Backend via MQTT.

    Cloud publishes commands to WAREHOUSE_MQTT_COMMAND_TOPIC.
    This function adapts those command payloads into the same shape that the
    old HTTP polling flow used, so the outbound AI flow stays unchanged.
    """
    requests = []

    while _mqtt_command_queue:
        cmd = _mqtt_command_queue.popleft()

        if not isinstance(cmd, dict):
            continue

        command_name = str(cmd.get("command") or cmd.get("type") or "").strip().lower()
        if command_name not in ("remove_item", "remove_request", "outbound_remove"):
            continue

        item_id = str(cmd.get("item_id") or "").strip()
        if not item_id:
            print("[MQTT-COMMAND] Skip remove command without item_id")
            continue

        position = cmd.get("position") if isinstance(cmd.get("position"), dict) else {}

        requests.append(
            {
                "request_id": cmd.get("request_id") or cmd.get("command_id") or f"mqtt-{uuid.uuid4().hex}",
                "type": "remove_item",
                "status": "pending",
                "item_id": item_id,
                "shelf_id": cmd.get("shelf_id") or position.get("shelf_id"),
                "level_id": cmd.get("level_id") or position.get("level_id"),
                "position": position or None,
                "requested_at": cmd.get("requested_at") or cmd.get("timestamp"),
                "updated_at": cmd.get("updated_at") or cmd.get("timestamp"),
                "requested_by": cmd.get("requested_by") or "cloud-dashboard",
                "source": cmd.get("source") or "mqtt_command",
                "note": cmd.get("note") or "",
                "raw_command": cmd,
            }
        )

    return requests


def fetch_pending_remove_requests_from_cloud():
    """
    Poll cloud backend for pending dashboard remove requests.
    Returns a list of pending requests. Failure must not stop local vision loop.
    """
    if not WAREHOUSE_CLOUD_URL:
        return []

    try:
        url = WAREHOUSE_CLOUD_URL + "/api/remove-requests?status=pending"
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json"},
            method="GET"
        )

        with urllib.request.urlopen(req, timeout=WAREHOUSE_CLOUD_TIMEOUT) as resp:
            raw = resp.read(20000).decode("utf-8", errors="ignore")

        data = json.loads(raw)
        requests = data.get("requests", [])
        return [
            r for r in requests
            if isinstance(r, dict)
            and r.get("status") == "pending"
            and r.get("item_id")
        ]

    except Exception as e:
        print(f"[Cloud] Remove request poll error: {e}")
        return []


def find_outbound_item_by_id(outbound_items, item_id):
    """
    Find local placed item from shelf_state by item_id.
    Returns (index, item) or (None, None).
    """
    target = str(item_id or "")
    for idx, item in enumerate(outbound_items or []):
        if str(item.get("item_id")) == target:
            return idx, item
    return None, None

# ---------- End cloud backend event forwarding ----------


def is_false_inventory_count_warning_event(event_type, payload):
    """
    Safety filter for cloud/local warning events.

    Prevent fake alerts such as:
      inventory_count_warning with expected=0 detected=0 status=ok
      inventory_count_warning with expected=1 detected=1 status=ok

    Real warnings are still allowed:
      detected < expected
      detected > expected
      unknown_count > 0
      missing_items not empty
      status is suspected_missing / unknown_object_detected / count_mismatch
    """
    if event_type != "inventory_count_warning":
        return False

    if not isinstance(payload, dict):
        return False

    status = str(payload.get("status", "")).strip().lower()
    display_status = str(payload.get("display_status", "")).strip().lower()

    if bool(payload.get("clear_mismatch", False)):
        return True

    try:
        expected = int(payload.get("expected_count", 0))
    except Exception:
        expected = 0

    try:
        detected = int(payload.get("detected_count", 0))
    except Exception:
        detected = 0

    try:
        unknown = int(payload.get("unknown_count", 0))
    except Exception:
        unknown = 0

    missing_items = payload.get("missing_items", []) or []

    logically_ok = (
        expected == detected
        and unknown == 0
        and len(missing_items) == 0
    )

    ok_statuses = {"", "ok", "normal", "clear", "cleared", "none"}

    if logically_ok and (status in ok_statuses or display_status in ok_statuses):
        return True

    return False


def append_event(event_type, payload=None):
    if payload is None:
        payload = {}

    if is_false_inventory_count_warning_event(event_type, payload):
        shelf_id = payload.get("shelf_id", "?") if isinstance(payload, dict) else "?"
        level_id = payload.get("level_id", "?") if isinstance(payload, dict) else "?"
        expected = payload.get("expected_count", "?") if isinstance(payload, dict) else "?"
        detected = payload.get("detected_count", "?") if isinstance(payload, dict) else "?"
        print(
            f"[EVENT] skip false inventory_count_warning: "
            f"{shelf_id}/{level_id} expected={expected} detected={detected}"
        )
        return


    event = {
        "event_id": (payload.get("event_id") if isinstance(payload, dict) else None) or f"jetson-{uuid.uuid4().hex}",
        "event_type": event_type,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **payload,
    }

    with open(WAREHOUSE_EVENTS_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")

    print(f"[EVENT] {event_type}: {payload}")

    handle_led_side_effect_for_event(event_type, payload, event)

    # Send to cloud with both top-level fields and nested payload.
    # Top-level fields keep compatibility with local event log.
    # Nested payload helps the backend parse size_cm, suggested_position,
    # actual_position, removed_position, expected_count, detected_count, etc.
    _cloud_event = dict(event)
    _cloud_event["payload"] = dict(payload) if isinstance(payload, dict) else {}
    send_event_transport_async(_cloud_event)

def candidate_matches_registered_item(candidate, occupied, min_overlap_cm=0.15, min_overlap_ratio=0.12):
    cand_start = float(candidate.get("start_cm", 0.0))
    cand_end = float(candidate.get("end_cm", 0.0))
    cand_width = max(0.001, cand_end - cand_start)

    occ_start = float(occupied.get("start_cm", occupied.get("actual_start_cm", 0.0)))
    occ_end = float(occupied.get("end_cm", occupied.get("actual_end_cm", occ_start)))
    occ_width = max(0.001, occ_end - occ_start)

    overlap = interval_overlap(cand_start, cand_end, occ_start, occ_end)
    overlap_ratio = overlap / min(cand_width, occ_width)

    cand_center = float(candidate.get("center_cm", -999.0))
    occ_center = (occ_start + occ_end) / 2.0

    center_condition = (occ_start <= cand_center <= occ_end) or (cand_start <= occ_center <= cand_end)

    return overlap >= min_overlap_cm and (overlap_ratio >= min_overlap_ratio or center_condition)


def is_strong_unknown_candidate(candidate):
    """
    Decide whether an unmatched LIVE candidate should be counted as UNKNOWN.

    This fixes the case where an unregistered object is visibly detected on
    the shelf but monitor still reports OK because the old candidate did not
    carry enough geometry information.
    """
    width_cm = float(candidate.get("width_cm", 0.0))
    height_cm = float(candidate.get("height_cm", 0.0))
    fill_ratio = float(candidate.get("fill_ratio", 1.0))
    area_px = float(candidate.get("area_px", 0.0))

    bbox = candidate.get("bbox_rect", {}) or {}
    bbox_w = float(bbox.get("w", 0.0))
    bbox_h = float(bbox.get("h", 0.0))

    # If older candidates do not have height_cm, still allow a pixel-based check.
    if height_cm <= 0.0 and bbox_h > 0:
        # Unknown fallback: enough bbox pixels means this is likely a real object.
        height_cm = 999.0

    # Reject tiny sparkles/noise.
    if width_cm < 0.45:
        return False

    if height_cm < 0.45:
        return False

    if area_px < 45.0 and bbox_w < 8 and bbox_h < 8:
        return False

    # Reject long horizontal shelf-edge artifacts.
    aspect = width_cm / max(0.001, height_cm)

    if aspect > 6.0 and height_cm < 1.0:
        return False

    # Reject very sparse noise, but keep threshold low because real objects may
    # have broken foreground masks under lighting changes.
    if fill_ratio < 0.04 and area_px < 180.0:
        return False

    return True


def get_presence_interval_cm(occupied, shelf_width_cm, zone_scale=0.65):
    """
    Return a tighter item-presence interval.

    start_cm/end_cm in shelf_state are reserved intervals and can include padding.
    For missing detection, we should not use the full reserved zone because it may
    overlap with neighboring objects.

    This function uses the actual detected center when available, then checks only
    the central portion of the registered item zone.
    """
    reg_start = float(occupied.get("start_cm", occupied.get("actual_start_cm", 0.0)))
    reg_end = float(occupied.get("end_cm", occupied.get("actual_end_cm", reg_start)))

    if reg_end < reg_start:
        reg_start, reg_end = reg_end, reg_start

    reserved_width = max(0.001, reg_end - reg_start)

    actual_start = float(occupied.get("actual_start_cm", reg_start))
    actual_end = float(occupied.get("actual_end_cm", reg_end))

    if actual_end < actual_start:
        actual_start, actual_end = actual_end, actual_start

    actual_width = max(0.001, actual_end - actual_start)

    # Use actual center if it looks reasonable. Otherwise use reserved center.
    if 0.2 <= actual_width <= max(reserved_width * 1.3, 0.8):
        center = (actual_start + actual_end) / 2.0
        base_width = min(max(actual_width, 0.6), reserved_width)
    else:
        center = (reg_start + reg_end) / 2.0
        base_width = reserved_width

    # Tighter zone to avoid neighbor overlap.
    zone_scale = max(0.35, min(1.0, float(zone_scale)))
    target_width = base_width * zone_scale
    target_width = max(0.45, min(target_width, reserved_width))

    start = center - target_width / 2.0
    end = center + target_width / 2.0

    start = max(reg_start, start)
    end = min(reg_end, end)

    start = max(0.0, min(start, shelf_width_cm))
    end = max(0.0, min(end, shelf_width_cm))

    if end <= start:
        start = max(0.0, center - 0.25)
        end = min(shelf_width_cm, center + 0.25)

    return start, end


def registered_item_present_from_mask(
    monitor_mask,
    occupied,
    level_index,
    shelf_config,
    min_area_px=45.0,
    min_ratio=0.018,
    zone_scale=0.65,
):
    """
    Decide whether a registered item is still present by inspecting the mask
    inside that item's own zone.

    This is stricter than checking overlap with any live candidate and fixes:
    - removed item still counted because neighbor blob overlaps old reserved zone
    - long shelf-edge artifacts counted as object
    """
    if monitor_mask is None:
        return False, {
            "white_pixels": 0,
            "ratio": 0.0,
            "max_area": 0.0,
            "zone": None,
        }

    shelf_h_px, shelf_w_px = monitor_mask.shape[:2]
    num_levels = int(shelf_config.get("num_levels", 4))
    shelf_width_cm = float(shelf_config["physical_size_cm"]["width"])

    start_cm, end_cm = get_presence_interval_cm(
        occupied=occupied,
        shelf_width_cm=shelf_width_cm,
        zone_scale=zone_scale,
    )

    x1 = int(round((start_cm / shelf_width_cm) * shelf_w_px))
    x2 = int(round((end_cm / shelf_width_cm) * shelf_w_px))

    x1 = max(0, min(x1, shelf_w_px - 1))
    x2 = max(0, min(x2, shelf_w_px))

    level_h = shelf_h_px / num_levels

    y1 = int(round((level_index - 1) * level_h))
    y2 = int(round(level_index * level_h))

    if level_index == num_levels:
        y2 = shelf_h_px

    # Ignore horizontal shelf-board bands at level boundaries.
    margin_y = max(2, int(round((y2 - y1) * 0.10)))
    y1_inner = min(y2, y1 + margin_y)
    y2_inner = max(y1_inner + 1, y2 - margin_y)

    if x2 <= x1 + 1 or y2_inner <= y1_inner + 1:
        return False, {
            "white_pixels": 0,
            "ratio": 0.0,
            "max_area": 0.0,
            "zone": [start_cm, end_cm],
        }

    roi = monitor_mask[y1_inner:y2_inner, x1:x2]

    if roi.size == 0:
        return False, {
            "white_pixels": 0,
            "ratio": 0.0,
            "max_area": 0.0,
            "zone": [start_cm, end_cm],
        }

    # Clean tiny sparkles inside the ROI.
    kernel = np.ones((3, 3), np.uint8)
    roi_clean = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)

    white_pixels = int(cv2.countNonZero(roi_clean))
    ratio = float(white_pixels) / float(roi_clean.size)

    contours, _ = cv2.findContours(roi_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max([cv2.contourArea(c) for c in contours], default=0.0)

    present = (max_area >= float(min_area_px)) or (ratio >= float(min_ratio))

    debug = {
        "white_pixels": white_pixels,
        "ratio": round(ratio, 4),
        "max_area": round(float(max_area), 2),
        "zone": [round(start_cm, 3), round(end_cm, 3)],
    }

    return present, debug


def get_detected_counts_by_level_from_state(
    state,
    live_candidates,
    num_levels,
    monitor_mask=None,
    shelf_config=None,
    min_area_px=45.0,
    min_ratio=0.018,
    zone_scale=0.65,
):
    """
    Robust level-count monitor.

    Core logic:
    - expected_count = number of registered items in shelf_state
    - detected_count = number of registered items whose own zone is occupied
                       + strong unknown objects

    This prevents the system from still counting a removed item just because
    a neighboring item or shelf artifact overlaps the old reserved interval.
    """
    counts = {f"T{i + 1}": 0 for i in range(num_levels)}
    unknown_counts = {f"T{i + 1}": 0 for i in range(num_levels)}
    missing_items = {f"T{i + 1}": [] for i in range(num_levels)}

    all_occupied = []

    for level in state.get("levels", []):
        level_index = int(level.get("level_index", 1))
        level_id = level.get("level_id", f"T{level_index}")

        for occ in level.get("occupied_intervals", []):
            occ_with_level = dict(occ)
            occ_with_level["_level_index"] = level_index
            occ_with_level["_level_id"] = level_id
            all_occupied.append(occ_with_level)

    matched_candidate_ids = set()

    for occ in all_occupied:
        level_id = occ["_level_id"]
        level_index = int(occ["_level_index"])
        item_id = occ.get("item_id", "ITEM")

        if monitor_mask is not None and shelf_config is not None:
            present, dbg = registered_item_present_from_mask(
                monitor_mask=monitor_mask,
                occupied=occ,
                level_index=level_index,
                shelf_config=shelf_config,
                min_area_px=min_area_px,
                min_ratio=min_ratio,
                zone_scale=zone_scale,
            )

            if present:
                counts[level_id] = counts.get(level_id, 0) + 1
            else:
                missing_items[level_id].append(
                    {
                        "item_id": item_id,
                        "presence_debug": dbg,
                    }
                )

            # Also mark candidates that overlap this item so they are not counted as unknown.
            for idx, cand in enumerate(live_candidates):
                if int(cand.get("level_index", -1)) != level_index:
                    continue
                if candidate_matches_registered_item(cand, occ):
                    matched_candidate_ids.add(idx)

        else:
            # Fallback if monitor mask is unavailable.
            present = False

            for idx, cand in enumerate(live_candidates):
                if int(cand.get("level_index", -1)) != level_index:
                    continue

                if candidate_matches_registered_item(cand, occ):
                    present = True
                    matched_candidate_ids.add(idx)

            if present:
                counts[level_id] = counts.get(level_id, 0) + 1
            else:
                missing_items[level_id].append({"item_id": item_id})

    for idx, cand in enumerate(live_candidates):
        if idx in matched_candidate_ids:
            continue

        if not is_strong_unknown_candidate(cand):
            continue

        level_id = cand.get("level_id")

        if level_id in unknown_counts:
            unknown_counts[level_id] += 1
            counts[level_id] = counts.get(level_id, 0) + 1

    return counts, unknown_counts, missing_items


def build_monitor_report(
    state,
    live_candidates,
    shelf_config,
    monitor_mask=None,
    presence_min_area=45.0,
    presence_min_ratio=0.018,
    presence_zone_scale=0.65,
):
    num_levels = int(shelf_config.get("num_levels", 4))
    shelf_id = state.get("shelf_id", shelf_config.get("shelf_id", "SHELF_A"))

    expected_counts = get_expected_counts_by_level(state)

    detected_counts, unknown_counts, missing_items = get_detected_counts_by_level_from_state(
        state=state,
        live_candidates=live_candidates,
        num_levels=num_levels,
        monitor_mask=monitor_mask,
        shelf_config=shelf_config,
        min_area_px=presence_min_area,
        min_ratio=presence_min_ratio,
        zone_scale=presence_zone_scale,
    )

    levels = []
    has_warning = False

    for i in range(num_levels):
        level_id = f"T{i + 1}"

        expected = int(expected_counts.get(level_id, 0))
        detected = int(detected_counts.get(level_id, 0))
        unknown = int(unknown_counts.get(level_id, 0))
        missing = missing_items.get(level_id, [])

        if expected == detected and unknown == 0 and not missing:
            status = "ok"
            event_status = "ok"
        elif missing:
            status = "MISSING?"
            event_status = "suspected_missing"
            has_warning = True
        elif unknown > 0:
            status = "UNKNOWN?"
            event_status = "unknown_object_detected"
            has_warning = True
        else:
            status = "COUNT_MISMATCH"
            event_status = "count_mismatch"
            has_warning = True

        levels.append(
            {
                "shelf_id": shelf_id,
                "level_id": level_id,
                "expected_count": expected,
                "detected_count": detected,
                "unknown_count": unknown,
                "missing_items": missing,
                "status": status,
                "event_status": event_status,
            }
        )

    return {
        "shelf_id": shelf_id,
        "has_warning": has_warning,
        "levels": levels,
    }


def monitor_report_to_summary(report):
    if report is None:
        return "Shelf monitor: not ready"

    prefix = "MONITOR WARNING" if report.get("has_warning") else "MONITOR OK"

    parts = []

    for level in report.get("levels", []):
        parts.append(
            f"{level['level_id']}: "
            f"exp={level['expected_count']}, "
            f"det={level['detected_count']}, "
            f"{level['status']}"
        )

    return prefix + " | " + " | ".join(parts)




def _send_cloud_event_only(event_type, payload=None):
    """
    Send event to cloud without writing many periodic status rows to local jsonl.
    append_event() is still used for major item_created/item_placed/item_removed events.
    """
    if payload is None:
        payload = {}

    # In v4, periodic status normally goes through MQTT.
    # WAREHOUSE_CLOUD_URL is still used for remove-request polling, but it is
    # not required for event publishing when WAREHOUSE_EVENT_TRANSPORT=mqtt.
    if WAREHOUSE_EVENT_TRANSPORT in ("http", "both") and not WAREHOUSE_CLOUD_URL:
        return

    event = {
        "event_id": (payload.get("event_id") if isinstance(payload, dict) else None) or f"jetson-{uuid.uuid4().hex}",
        "event_type": event_type,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **payload,
    }

    # Keep same format as append_event(): top-level fields + nested payload.
    event["payload"] = dict(payload) if isinstance(payload, dict) else {}

    send_event_transport_async(event)


def _build_inventory_status_payload_for_cloud(report, state, shelf_config, live_candidates=None):
    if report is None:
        return None

    shelf_id = report.get(
        "shelf_id",
        state.get("shelf_id", shelf_config.get("shelf_id", "SHELF_A")),
    )

    levels = report.get("levels", [])
    has_warning = bool(report.get("has_warning", False))

    total_expected = get_total_expected_count(state)
    total_detected = get_total_detected_count(report)

    return {
        "shelf_id": shelf_id,
        "status": "warning" if has_warning else "ok",
        "has_warning": has_warning,
        "total_expected": int(total_expected),
        "total_detected": int(total_detected),
        "levels": levels,
        "live_count": len(live_candidates or []),
        "source": "jetson_periodic_inventory_status",
        "note": "Periodic full inventory status from Jetson edge monitor.",
    }


def _push_legacy_level_status_events_to_cloud(report):
    """
    Send per-level status to cloud without creating fake warning alerts.

    - inventory_level_status is sent for both OK and WARNING levels.
    - inventory_count_warning is sent ONLY for real warning levels.
    - OK levels must NOT be sent as inventory_count_warning, otherwise the
      web dashboard may create false active alerts such as expected=0/detected=0.
    """
    if report is None or not WAREHOUSE_CLOUD_URL:
        return

    shelf_id = report.get("shelf_id", "SHELF_A")

    for level in report.get("levels", []):
        level_id = level.get("level_id")

        expected_count = int(level.get("expected_count", 0))
        detected_count = int(level.get("detected_count", 0))
        unknown_count = int(level.get("unknown_count", 0))
        missing_items = level.get("missing_items", []) or []

        event_status = str(level.get("event_status", level.get("status", "ok"))).lower()
        display_status = str(level.get("status", event_status)).lower()

        logically_ok = (
            expected_count == detected_count
            and unknown_count == 0
            and len(missing_items) == 0
        )

        is_warning = not logically_ok

        if logically_ok:
            event_status = "ok"
            display_status = "ok"

        payload = {
            "shelf_id": shelf_id,
            "level_id": level_id,
            "expected_count": expected_count,
            "detected_count": detected_count,
            "unknown_count": unknown_count,
            "missing_items": missing_items,
            "status": event_status,
            "display_status": display_status,
            "has_warning": bool(is_warning),
            "clear_mismatch": bool(not is_warning),
            "source": "jetson_periodic_level_status",
            "note": (
                "Periodic per-level status from Jetson. "
                "Only real warnings are emitted as inventory_count_warning."
            ),
        }

        # Clean per-level status event. Backend/frontend can use this to clear stale mismatch.
        _send_cloud_event_only("inventory_level_status", payload)

        # Legacy warning event. Send ONLY if this level is truly abnormal.
        if is_warning:
            warning_payload = dict(payload)
            warning_payload["note"] = "Real inventory count warning from Jetson edge monitor."
            _send_cloud_event_only("inventory_count_warning", warning_payload)


def maybe_send_periodic_inventory_status(
    report,
    state,
    shelf_config,
    live_candidates,
    last_sent_ts,
    interval_sec,
):
    """
    Send inventory status every N seconds.

    Important:
    - Sends full shelf snapshot: inventory_status_update
    - Sends per-level OK/WARNING status: inventory_level_status
    - Sends inventory_count_warning only for real warning levels
    """
    if not WAREHOUSE_CLOUD_URL:
        return last_sent_ts

    if report is None:
        return last_sent_ts

    now_ts = time.time()
    interval_sec = max(0.5, float(interval_sec))

    if now_ts - float(last_sent_ts or 0.0) < interval_sec:
        return last_sent_ts

    payload = _build_inventory_status_payload_for_cloud(
        report=report,
        state=state,
        shelf_config=shelf_config,
        live_candidates=live_candidates,
    )

    if payload is not None:
        _send_cloud_event_only("inventory_status_update", payload)
        _push_legacy_level_status_events_to_cloud(report)

    return now_ts



def update_monitor_warning_events(
    report,
    warning_counters,
    last_event_times,
    required_frames,
    cooldown_sec,
):
    """
    Send warning events when monitor detects mismatch.

    Also send inventory_status when the monitor is OK, so the cloud dashboard
    can clear stale mismatch alerts after lighting/camera conditions recover.
    """
    if report is None:
        return

    now = time.time()

    # Send OK status periodically and immediately after a warning state recovers.
    # This prevents stale warning/mismatch states on the cloud dashboard.
    ok_status_interval_sec = float(os.environ.get("WAREHOUSE_MONITOR_OK_EVENT_INTERVAL", "8"))

    has_warning = bool(report.get("has_warning"))
    last_had_warning = bool(last_event_times.get("__monitor_had_warning__", False))
    last_ok_status_time = float(last_event_times.get("__monitor_ok_status_time__", 0.0))

    if not has_warning:
        for level in report.get("levels", []):
            warning_counters[level["level_id"]] = 0

        should_send_ok = last_had_warning or ((now - last_ok_status_time) >= ok_status_interval_sec)

        if should_send_ok:
            append_event(
                "inventory_status",
                {
                    "shelf_id": report["shelf_id"],
                    "status": "ok",
                    "source": "jetson_monitor_recovery",
                    "levels": [
                        {
                            "shelf_id": level.get("shelf_id", report["shelf_id"]),
                            "level_id": level["level_id"],
                            "expected_count": level["expected_count"],
                            "detected_count": level["detected_count"],
                            "status": "ok",
                        }
                        for level in report.get("levels", [])
                    ],
                    "note": "Shelf monitor currently matches expected item count.",
                },
            )

            last_event_times["__monitor_ok_status_time__"] = now

        last_event_times["__monitor_had_warning__"] = False
        return

    last_event_times["__monitor_had_warning__"] = True

    for level in report.get("levels", []):
        level_id = level["level_id"]
        status = level["event_status"]

        if status == "ok":
            warning_counters[level_id] = 0
            continue

        warning_counters[level_id] = warning_counters.get(level_id, 0) + 1

        if warning_counters[level_id] < required_frames:
            continue

        event_key = (
            f"{report['shelf_id']}|{level_id}|{status}|"
            f"{level['expected_count']}|{level['detected_count']}"
        )

        last_time = last_event_times.get(event_key, 0.0)

        if now - last_time < cooldown_sec:
            continue

        append_event(
            "inventory_count_warning",
            {
                "shelf_id": report["shelf_id"],
                "level_id": level_id,
                "expected_count": level["expected_count"],
                "detected_count": level["detected_count"],
                "status": status,
                "note": (
                    "Level-1 monitor warning based on count mismatch. "
                    "This may indicate missing item, merged blobs, or unknown object."
                ),
            },
        )

        last_event_times[event_key] = now


# ============================================================
# On-screen warehouse monitor UI
# ============================================================

def get_total_expected_count(state):
    total = 0

    for level in state.get("levels", []):
        total += len(level.get("occupied_intervals", []))

    return total


def get_total_detected_count(report):
    if report is None:
        return 0

    total = 0

    for level in report.get("levels", []):
        total += int(level.get("detected_count", 0))

    return total


def draw_transparent_box(frame, x, y, w, h, color=(0, 0, 0), alpha=0.55):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_warehouse_monitor_panel(frame, monitor_report, state, shelf_config):
    num_levels = int(shelf_config.get("num_levels", 4))

    panel_x = 10
    panel_y = 115
    panel_w = 520
    row_h = 26
    panel_h = 95 + (num_levels * row_h)

    draw_transparent_box(
        frame,
        panel_x,
        panel_y,
        panel_w,
        panel_h,
        color=(0, 0, 0),
        alpha=0.58,
    )

    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (0, 255, 255),
        1,
    )

    title_y = panel_y + 25

    if monitor_report is None:
        cv2.putText(
            frame,
            "WAREHOUSE MONITOR: SHELF BG NOT READY",
            (panel_x + 12, title_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "Empty shelf, then press k to capture shelf background.",
            (panel_x + 12, title_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
        )

        return

    has_warning = bool(monitor_report.get("has_warning", False))

    if has_warning:
        title_color = (0, 0, 255)
        title = "WAREHOUSE MONITOR: WARNING"
    else:
        title_color = (0, 255, 0)
        title = "WAREHOUSE MONITOR: OK"

    cv2.putText(
        frame,
        title,
        (panel_x + 12, title_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        title_color,
        2,
    )

    total_expected = get_total_expected_count(state)
    total_detected = get_total_detected_count(monitor_report)

    total_text = f"Total items: expected={total_expected} | detected={total_detected}"

    cv2.putText(
        frame,
        total_text,
        (panel_x + 12, title_y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
    )

    start_y = title_y + 62

    for idx, level in enumerate(monitor_report.get("levels", [])):
        level_id = level["level_id"]
        expected = level["expected_count"]
        detected = level["detected_count"]
        status = level["status"]

        y = start_y + idx * row_h

        if status == "OK" or status == "ok":
            color = (0, 255, 0)
            status_text = "OK"
        elif "MISSING" in status:
            color = (0, 0, 255)
            status_text = "MISSING?/MERGED"
        elif "UNKNOWN" in status:
            color = (0, 165, 255)
            status_text = "UNKNOWN OBJECT?"
        else:
            color = (0, 0, 255)
            status_text = str(status)

        line = f"{level_id}: expected={expected} | detected={detected} | {status_text}"

        cv2.putText(
            frame,
            line,
            (panel_x + 18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            1,
        )

    if has_warning:
        warning_levels = []

        for level in monitor_report.get("levels", []):
            if level.get("event_status") != "ok":
                warning_levels.append(level["level_id"])

        alert_text = "ALERT at: " + ", ".join(warning_levels)

        cv2.putText(
            frame,
            alert_text,
            (panel_x + 12, panel_y + panel_h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
    else:
        cv2.putText(
            frame,
            "All shelf levels match expected item count.",
            (panel_x + 12, panel_y + panel_h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            1,
        )



# ============================================================
# Outbound / Item removal workflow
# ============================================================

def load_item_status_map():
    db = load_json_default(ITEM_DB_PATH, {"items": []})
    status_map = {}

    for item in db.get("items", []):
        item_id = item.get("item_id")
        if item_id:
            status_map[item_id] = item.get("status", "unknown")

    return status_map


def get_placed_items_from_state(state):
    """
    Return placed items using shelf_state.json as the physical source of truth.

    Important fix:
    For outbound selection/removal, use the same compact visual interval as
    the yellow REG box. The old logic used the reserved interval, which can be
    wider than the object and can overlap neighboring objects.
    """
    status_map = load_item_status_map()
    placed_items = []

    shelf_width_cm = float(
        state.get("physical_size_cm", {}).get("width", 12.5)
    )

    for level in state.get("levels", []):
        level_id = level.get("level_id", f"T{level.get('level_index')}")
        level_index = int(level.get("level_index", 1))

        for occ in level.get("occupied_intervals", []):
            item_id = occ.get("item_id")

            if not item_id:
                continue

            status = status_map.get(item_id, "placed")

            if status != "placed":
                continue

            reserved_start = float(occ.get("start_cm", 0.0))
            reserved_end = float(occ.get("end_cm", reserved_start))

            visual_start, visual_end = get_visual_registered_interval_cm(
                occ=occ,
                shelf_width_cm=shelf_width_cm,
                visual_padding_cm=0.12,
            )

            item = {
                "item_id": item_id,
                "level_id": level_id,
                "level_index": level_index,

                # Use compact interval for outbound drawing/removal detection.
                "start_cm": float(visual_start),
                "end_cm": float(visual_end),

                # Keep original reserved interval for traceability/debug.
                "reserved_start_cm": reserved_start,
                "reserved_end_cm": reserved_end,

                "actual_start_cm": float(occ.get("actual_start_cm", reserved_start)),
                "actual_end_cm": float(occ.get("actual_end_cm", reserved_end)),
                "bbox_rect": occ.get("bbox_rect"),
                "placed_at": occ.get("placed_at"),
            }

            placed_items.append(item)

    placed_items.sort(key=lambda x: (x["level_index"], x["start_cm"], x["item_id"]))
    return placed_items


def remove_item_from_shelf_state(state, item_id):
    """
    Remove an item from shelf_state.json and recompute free intervals.
    """
    removed = None

    for level in state.get("levels", []):
        occupied = level.get("occupied_intervals", [])
        new_occupied = []

        for occ in occupied:
            if occ.get("item_id") == item_id:
                removed = {
                    "level_id": level.get("level_id"),
                    "level_index": level.get("level_index"),
                    "occupied_record": occ,
                }
            else:
                new_occupied.append(occ)

        if len(new_occupied) != len(occupied):
            level["occupied_intervals"] = new_occupied
            recompute_free_intervals(level)

    return removed


def update_items_db_after_removal(item_id, removed_info):
    db = load_json_default(ITEM_DB_PATH, {"items": []})
    now = datetime.now().isoformat(timespec="seconds")

    found = False

    for item in db.get("items", []):
        if item.get("item_id") == item_id:
            item["status"] = "removed"
            item["removed_at"] = now
            item["removal_source"] = "smart_warehouse_ai_vision"
            item["removed_from"] = removed_info
            item["updated_at"] = now
            found = True
            break

    if not found:
        db.setdefault("items", []).append(
            {
                "item_id": item_id,
                "status": "removed",
                "removed_at": now,
                "removal_source": "smart_warehouse_ai_vision",
                "removed_from": removed_info,
                "updated_at": now,
            }
        )

    save_json(db, ITEM_DB_PATH)


def live_object_overlaps_target(live_candidates, target_item, overlap_threshold_cm=0.25):
    """
    Check whether any live detected object still overlaps the outbound target zone.

    Important fix:
    The outbound target now uses the compact visual interval, not the full
    reserved planning interval. This prevents neighboring objects from keeping
    the removed item falsely "present".
    """
    if target_item is None:
        return False

    target_level = int(target_item["level_index"])
    target_start = float(target_item.get("remove_start_cm", target_item.get("start_cm", 0.0)))
    target_end = float(target_item.get("remove_end_cm", target_item.get("end_cm", target_start)))

    if target_end < target_start:
        target_start, target_end = target_end, target_start

    target_width = max(0.001, target_end - target_start)

    for cand in live_candidates or []:
        if int(cand.get("level_index", -1)) != target_level:
            continue

        cand_start = float(cand.get("start_cm", 0.0))
        cand_end = float(cand.get("end_cm", cand_start))

        if cand_end < cand_start:
            cand_start, cand_end = cand_end, cand_start

        overlap = interval_overlap(target_start, target_end, cand_start, cand_end)

        if overlap <= 0:
            continue

        cand_width = max(0.001, cand_end - cand_start)
        overlap_ratio_target = overlap / target_width
        overlap_ratio_candidate = overlap / cand_width

        center = float(cand.get("center_cm", -999.0))
        center_inside = target_start <= center <= target_end

        if (
            overlap >= float(overlap_threshold_cm)
            or overlap_ratio_target >= 0.35
            or overlap_ratio_candidate >= 0.35
            or center_inside
        ):
            return True

    return False


def draw_outbound_target_on_original(
    frame,
    target_item,
    shelf_config,
    H_shelf_inv,
    shelf_w_px,
    shelf_h_px,
    color=(0, 0, 255),
    label_prefix="REMOVE",
):
    if target_item is None:
        return

    level_index = int(target_item["level_index"])
    start_cm = float(target_item["start_cm"])
    end_cm = float(target_item["end_cm"])
    item_id = target_item["item_id"]

    draw_interval_on_original(
        frame=frame,
        shelf_config=shelf_config,
        H_shelf_inv=H_shelf_inv,
        shelf_w_px=shelf_w_px,
        shelf_h_px=shelf_h_px,
        level_index=level_index,
        start_cm=start_cm,
        end_cm=end_cm,
        color=color,
        label=f"{label_prefix} {item_id}",
        thickness=3,
        fill=True,
        alpha=0.28,
    )


def format_outbound_selection(items, selected_idx):
    if not items:
        return "No placed item available for outbound."

    selected_idx = max(0, min(selected_idx, len(items) - 1))
    item = items[selected_idx]

    return (
        f"Outbound select {selected_idx + 1}/{len(items)}: "
        f"{item['item_id']} at {item['level_id']} "
        f"{item['start_cm']:.2f}->{item['end_cm']:.2f}cm"
    )


# ============================================================
# Main workflow
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Smart Warehouse AI integrated workflow")

    # M?c d?nh k?t n?i c?ng c?m Camera USB Jetson r?i t?i index 2
    parser.add_argument(
        "--camera-index",
        type=int,
        default=2,
        help="USB camera device index. Default is 2 for /dev/video2."
    )

    # Default USB camera profile: MJPG 1280x720.
    parser.add_argument("--capture-width", type=int, default=1280)
    parser.add_argument("--capture-height", type=int, default=720)
    parser.add_argument("--output-width", type=int, default=1280)
    parser.add_argument("--output-height", type=int, default=720)
    parser.add_argument("--fourcc", type=str, default="MJPG", choices=["MJPG", "YUYV"])

    parser.add_argument("--shelf-config", type=str, default="shelf_config.json")
    parser.add_argument("--measure-config", type=str, default="measure_config.json")
    parser.add_argument("--state", type=str, default="shelf_state.json")

    parser.add_argument("--measure-threshold", type=int, default=35)
    parser.add_argument("--measure-min-area", type=int, default=80)
    # Measurement detector:
    # - yolo: use YOLO-Worldv2 bbox only
    # - yolo_fallback: use YOLO first, fallback to background subtraction if YOLO misses
    # - background: old background-subtraction measurement
    parser.add_argument("--measure-detector", type=str, default="yolo_fallback",
                        choices=["yolo", "yolo_fallback", "background"])
    parser.add_argument("--measure-yolo-model", type=str, default="yolov8s-worldv2.pt")
    parser.add_argument("--measure-yolo-conf", type=float, default=0.05)
    parser.add_argument("--measure-yolo-imgsz", type=int, default=320)
    parser.add_argument("--measure-yolo-classes", type=str, default="box,bottle,cup")
    parser.add_argument("--measure-yolo-min-area", type=int, default=80)

    parser.add_argument("--shelf-threshold", type=int, default=35)
    parser.add_argument("--shelf-min-area", type=int, default=120)
    # Shelf detector:
    # - yolo: use YOLO-Worldv2 async detector for shelf monitor/placement/outbound
    # - yolo_fallback: use YOLO first, fallback to OpenCV shelf background if YOLO is not ready
    # - background: old OpenCV background subtraction shelf logic
    parser.add_argument("--shelf-detector", type=str, default="yolo",
                        choices=["yolo", "yolo_fallback", "background"])
    parser.add_argument("--shelf-yolo-model", type=str, default="yolov8s-worldv2.pt")
    parser.add_argument("--shelf-yolo-conf", type=float, default=0.05)
    parser.add_argument("--shelf-yolo-imgsz", type=int, default=320)
    parser.add_argument("--shelf-yolo-classes", type=str, default="box,bottle,cup")
    parser.add_argument("--shelf-yolo-interval-sec", type=float, default=1.5)
    parser.add_argument("--shelf-yolo-max-age-sec", type=float, default=5.0)
    parser.add_argument("--shelf-yolo-min-area", type=int, default=120)
    # Live monitor box stabilization
    parser.add_argument("--monitor-smooth-alpha", type=float, default=0.32)
    parser.add_argument("--monitor-max-grow-ratio", type=float, default=1.12)
    parser.add_argument("--monitor-max-shrink-ratio", type=float, default=0.88)
    parser.add_argument("--monitor-max-center-shift-cm", type=float, default=0.22)


    parser.add_argument("--measure-stable-frames", type=int, default=12)
    parser.add_argument("--place-stable-frames", type=int, default=12)

    parser.add_argument("--tolerance-cm", type=float, default=0.6)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.5)
    parser.add_argument("--max-width-factor", type=float, default=2.5)

    parser.add_argument("--place-arm-delay-sec", type=float, default=10.0)
    parser.add_argument("--position-stable-tolerance-cm", type=float, default=0.25)
    parser.add_argument("--occupied-padding-cm", type=float, default=0.4)
    parser.add_argument("--item-gap-cm", type=float, default=0.3)

    parser.add_argument("--monitor-warning-frames", type=int, default=30)
    parser.add_argument("--monitor-event-cooldown-sec", type=float, default=10.0)
    # Periodically push full inventory status to cloud/dashboard.
    # This fixes stale web mismatch after Jetson monitor returns to OK.
    parser.add_argument("--cloud-status-interval-sec", type=float, default=3.0)

    # More stable background capture under changing lighting.
    parser.add_argument("--background-samples", type=int, default=15)

    # Outbound / item removal
    parser.add_argument("--outbound-empty-frames", type=int, default=15)
    parser.add_argument("--outbound-overlap-threshold-cm", type=float, default=0.25)

    # Item-level presence monitor.
    # Used to decide whether a registered item is still visible in its zone.
    parser.add_argument("--presence-min-area", type=float, default=45.0)
    parser.add_argument("--presence-min-ratio", type=float, default=0.018)
    parser.add_argument("--presence-zone-scale", type=float, default=0.65)

    parser.add_argument("--note", type=str, default="smart warehouse auto item")

    return parser.parse_args()


def main():
    args = parse_args()

    init_mqtt_led_client()

    shelf_config = load_json(args.shelf_config)
    measure_config = load_json(args.measure_config)

    if os.path.exists(args.state):
        shelf_state = load_json(args.state)
    else:
        shelf_state = create_initial_shelf_state(shelf_config)
        save_json(shelf_state, args.state)

    H_shelf, H_shelf_inv, shelf_w_px, shelf_h_px = get_homography_from_config(shelf_config)
    H_measure, H_measure_inv, measure_w_px, measure_h_px = get_homography_from_config(measure_config)

    measure_background = None
    if os.path.exists(MEASURE_BACKGROUND_PATH):
        try:
            measure_background = np.load(MEASURE_BACKGROUND_PATH)
            print(f"Loaded measurement background: {MEASURE_BACKGROUND_PATH}")
        except Exception:
            measure_background = None

    measure_background_color = None
    if os.path.exists(MEASURE_BACKGROUND_COLOR_PATH):
        try:
            measure_background_color = np.load(MEASURE_BACKGROUND_COLOR_PATH)
            print(f"Loaded measurement color background: {MEASURE_BACKGROUND_COLOR_PATH}")
        except Exception:
            measure_background_color = None

    shelf_background = None
    if os.path.exists(SHELF_BACKGROUND_PATH):
        try:
            shelf_background = np.load(SHELF_BACKGROUND_PATH)
            print(f"Loaded shelf background: {SHELF_BACKGROUND_PATH}")
        except Exception:
            shelf_background = None

    camera_cfg = shelf_config.get("camera", {})

    measure_yolo_model = load_measure_yolo_detector(args)

    # USB camera profile comes from shelf_config.json if available.
    # Otherwise, use the default MJPG 1280x720 profile.
    camera = USBCameraV4L2(
        camera_index=int(camera_cfg.get("camera_index", args.camera_index)),
        capture_width=int(camera_cfg.get("capture_width", args.capture_width)),
        capture_height=int(camera_cfg.get("capture_height", args.capture_height)),
        output_width=int(camera_cfg.get("output_width", args.output_width)),
        output_height=int(camera_cfg.get("output_height", args.output_height)),
        fourcc=str(camera_cfg.get("fourcc", args.fourcc)),
        opencv_rotate_180=bool(camera_cfg.get("opencv_rotate_180", False)),
    )

    shelf_yolo_worker = None
    if args.shelf_detector in ("yolo", "yolo_fallback"):
        shelf_yolo_worker = ShelfYOLOAsyncWorker(
            model_path=args.shelf_yolo_model,
            classes=args.shelf_yolo_classes,
            conf=args.shelf_yolo_conf,
            imgsz=args.shelf_yolo_imgsz,
            interval_sec=args.shelf_yolo_interval_sec,
            min_area_px=args.shelf_yolo_min_area,
            shelf_config=shelf_config,
            shelf_w_px=shelf_w_px,
            shelf_h_px=shelf_h_px,
        )

        if not shelf_yolo_worker.start():
            print("[SHELF-YOLO] Async worker failed to start.")

            if args.shelf_detector == "yolo":
                print("[SHELF-YOLO] Strict YOLO shelf mode selected. Shelf monitor will wait for YOLO.")

            shelf_yolo_worker = None

    print("======================================")
    print(" Smart Warehouse AI (USB Camera Mode - MJPG 1280x720)")
    print("======================================")
    print("Workflow:")
    print("  1. Put item into measurement area")
    print("  2. AI measures item and creates ITEM_ID + QR automatically")
    print("  3. AI suggests shelf placement")
    print("  4. Put item into highlighted accept zone")
    print("  5. AI confirms placement and updates JSON metadata")
    print("--------------------------------------")
    print("Controls:")
    print("  b = capture EMPTY measurement background")
    print("  k = capture EMPTY shelf background")
    print("  x = cancel current item/cycle")
    print("  q = quit")
    print("======================================")

    if not camera.open():
        print(f"ERROR: KhÃ´ng th? k?t n?i t?i camera USB t?i index {args.camera_index}.")
        print("M?o: Ã?m b?o camera dÃ£ c?m ch?c ch?n vÃ  khÃ´ng b? ph?n m?m test cu chi?m gi?.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    stage = "WAIT_ITEM"
    current_item = None
    placement = None
    accept_zone = None
    before_shelf_gray = None

    measured_size_review = None
    measured_bbox_review = None

    measure_history = deque(maxlen=args.measure_stable_frames)
    place_stable_count = 0
    place_monitor_start_time = 0.0
    last_place_candidate = None

    last_measure_bbox = None
    last_measure_mask = None
    last_shelf_mask = None
    last_monitor_mask = None
    last_candidates = []
    last_live_candidates = []
    prev_live_candidates = []
    monitor_summary = "Shelf monitor: not ready"
    current_monitor_report = None
    monitor_warning_counters = {}
    monitor_last_event_times = {}
    status = "Waiting for item in measurement area."

    cooldown_until = 0.0

    outbound_items = []
    outbound_selected_idx = 0
    outbound_target = None
    outbound_empty_count = 0
    cloud_remove_last_poll = 0.0
    cloud_remove_last_missing_item_id = None
    cloud_status_last_push = 0.0

    try:
        while True:
            ret, frame = camera.read()

            if not ret or frame is None:
                print("ERROR: Failed to read frame from USB Camera.")
                break

            now = time.time()

            warped_measure = warp_area(frame, H_measure, measure_w_px, measure_h_px)
            warped_shelf = warp_area(frame, H_shelf, shelf_w_px, shelf_h_px)

            if shelf_yolo_worker is not None:
                shelf_yolo_worker.submit(warped_shelf)

            view = frame.copy()

            draw_shelf_levels(view, shelf_config)
            draw_quad(view, measure_config, (0, 200, 255), "MEASURE AREA")

            last_measure_bbox = None
            last_candidates = []
            last_live_candidates = []
            # Keep prev_live_candidates across frames for monitor smoothing.
            last_monitor_mask = None

            shelf_detection_source = "none"
            shelf_detection_ready = False
            shelf_yolo_info = None
            raw_live_candidates = []
            monitor_mask = None

            if args.shelf_detector in ("yolo", "yolo_fallback") and shelf_yolo_worker is not None:
                yolo_candidates, yolo_mask, shelf_yolo_info = shelf_yolo_worker.get_latest(
                    max_age_sec=args.shelf_yolo_max_age_sec,
                )

                if yolo_candidates is not None:
                    raw_live_candidates = yolo_candidates
                    monitor_mask = yolo_mask
                    shelf_detection_ready = True
                    shelf_detection_source = (
                        f"SHELF-YOLO age={shelf_yolo_info.get('age_sec', 0.0):.1f}s "
                        f"infer={shelf_yolo_info.get('infer_ms', 0.0):.0f}ms "
                        f"rej={len(shelf_yolo_info.get('rejected', []))}"
                    )

            if not shelf_detection_ready:
                if args.shelf_detector in ("background", "yolo_fallback") and shelf_background is not None:
                    raw_live_candidates, monitor_mask = detect_new_objects_on_shelf(
                        current_warped=warped_shelf,
                        before_gray=shelf_background,
                        shelf_config=shelf_config,
                        threshold_value=args.shelf_threshold,
                        min_area=args.shelf_min_area,
                    )

                    shelf_detection_ready = True
                    shelf_detection_source = (
                        "SHELF-BG-FALLBACK"
                        if args.shelf_detector == "yolo_fallback"
                        else "SHELF-BG"
                    )

            if shelf_detection_ready:
                live_candidates = stabilize_live_candidates(
                    prev_candidates=prev_live_candidates,
                    curr_candidates=raw_live_candidates,
                    smooth_alpha=args.monitor_smooth_alpha,
                    max_grow_ratio=args.monitor_max_grow_ratio,
                    max_shrink_ratio=args.monitor_max_shrink_ratio,
                    max_center_shift_cm=args.monitor_max_center_shift_cm,
                )

                last_live_candidates = live_candidates
                prev_live_candidates = [clone_candidate_for_tracking(c) for c in live_candidates]
                last_monitor_mask = monitor_mask

                monitor_report = build_monitor_report(
                    state=shelf_state,
                    live_candidates=last_live_candidates,
                    shelf_config=shelf_config,
                    monitor_mask=monitor_mask,
                    presence_min_area=args.presence_min_area,
                    presence_min_ratio=args.presence_min_ratio,
                    presence_zone_scale=args.presence_zone_scale,
                )

                current_monitor_report = monitor_report
                monitor_summary = monitor_report_to_summary(monitor_report) + f" | {shelf_detection_source}"

                cloud_status_last_push = maybe_send_periodic_inventory_status(
                    report=monitor_report,
                    state=shelf_state,
                    shelf_config=shelf_config,
                    live_candidates=last_live_candidates,
                    last_sent_ts=cloud_status_last_push,
                    interval_sec=args.cloud_status_interval_sec,
                )

                update_monitor_warning_events(
                    report=monitor_report,
                    warning_counters=monitor_warning_counters,
                    last_event_times=monitor_last_event_times,
                    required_frames=args.monitor_warning_frames,
                    cooldown_sec=args.monitor_event_cooldown_sec,
                )

                draw_registered_items_on_original(
                    frame=view,
                    state=shelf_state,
                    shelf_config=shelf_config,
                    H_shelf_inv=H_shelf_inv,
                    shelf_w_px=shelf_w_px,
                    shelf_h_px=shelf_h_px,
                )

                draw_live_occupancy_on_original(
                    frame=view,
                    live_candidates=last_live_candidates,
                    H_shelf_inv=H_shelf_inv,
                )

            else:
                current_monitor_report = None

                if args.shelf_detector == "yolo":
                    if shelf_yolo_info is not None and shelf_yolo_info.get("error"):
                        monitor_summary = f"Shelf monitor: YOLO error: {shelf_yolo_info.get('error')}"
                    else:
                        monitor_summary = "Shelf monitor: waiting for YOLO shelf detection..."
                elif args.shelf_detector == "yolo_fallback":
                    monitor_summary = "Shelf monitor: YOLO not ready and Shelf BG fallback not ready. Press k if fallback needed."
                else:
                    monitor_summary = "Shelf monitor: Shelf BG not ready. Empty shelf, then press k."

            # ---------- Remove/export request intake ----------
            # v4 main path: Cloud Backend -> MQTT Broker -> Jetson command topic.
            # Optional fallback: Jetson can still poll Cloud Backend by HTTP if
            # WAREHOUSE_REMOVE_REQUEST_TRANSPORT=http or both.
            if stage == "WAIT_ITEM" and (now - cloud_remove_last_poll) >= WAREHOUSE_REMOVE_POLL_INTERVAL:
                cloud_remove_last_poll = now

                pending_remove_requests = []
                if WAREHOUSE_REMOVE_REQUEST_TRANSPORT in ("mqtt", "both"):
                    pending_remove_requests.extend(fetch_pending_remove_requests_from_mqtt_queue())

                if WAREHOUSE_REMOVE_REQUEST_TRANSPORT in ("http", "both") and WAREHOUSE_CLOUD_URL:
                    pending_remove_requests.extend(fetch_pending_remove_requests_from_cloud())

                if pending_remove_requests:
                    cloud_req = pending_remove_requests[0]
                    cloud_item_id = str(cloud_req.get("item_id") or "")
                    cloud_request_id = cloud_req.get("request_id")

                    outbound_items = get_placed_items_from_state(shelf_state)
                    matched_idx, matched_item = find_outbound_item_by_id(outbound_items, cloud_item_id)

                    if matched_item is not None:
                        outbound_selected_idx = matched_idx
                        outbound_target = matched_item

                        # Some local shelf_state entries do not store shelf_id at top-level.
                        # Enrich the target using cloud request/state defaults to avoid KeyError.
                        if "shelf_id" not in outbound_target or not outbound_target.get("shelf_id"):
                            outbound_target["shelf_id"] = (
                                cloud_req.get("shelf_id")
                                or shelf_state.get("shelf_id")
                                or shelf_config.get("shelf_id")
                                or "SHELF_A"
                            )

                        if "level_id" not in outbound_target or not outbound_target.get("level_id"):
                            outbound_target["level_id"] = cloud_req.get("level_id") or "T?"

                        target_shelf_id = outbound_target.get("shelf_id", "SHELF_A")
                        target_level_id = outbound_target.get("level_id", cloud_req.get("level_id", "T?"))

                        target_start_cm = outbound_target.get(
                            "start_cm",
                            outbound_target.get("reserved_start_cm", outbound_target.get("actual_start_cm", 0.0))
                        )
                        target_end_cm = outbound_target.get(
                            "end_cm",
                            outbound_target.get("reserved_end_cm", outbound_target.get("actual_end_cm", 0.0))
                        )

                        try:
                            target_start_cm = float(target_start_cm)
                        except Exception:
                            target_start_cm = 0.0

                        try:
                            target_end_cm = float(target_end_cm)
                        except Exception:
                            target_end_cm = 0.0

                        outbound_empty_count = 0
                        cloud_remove_last_missing_item_id = None
                        stage = "OUTBOUND_REMOVE"

                        status = (
                            f"Cloud remove request: {cloud_item_id}. "
                            f"Remove it from {target_level_id}."
                        )

                        led_blink_for_position(
                            shelf_id=target_shelf_id,
                            level_id=target_level_id,
                            item_id=cloud_item_id,
                            reason="outbound_request",
                            request_id=cloud_request_id,
                        )

                        print("")
                        print("======================================")
                        print(f"REMOVE REQUEST: {cloud_item_id}")
                        print(f"Request ID: {cloud_request_id}")
                        print(
                            f"Target: {target_shelf_id} "
                            f"{target_level_id} "
                            f"{target_start_cm:.2f}->{target_end_cm:.2f}cm"
                        )
                        print("Jetson switched to OUTBOUND_REMOVE automatically.")
                        print("======================================")
                        print("")

                    elif cloud_remove_last_missing_item_id != cloud_item_id:
                        cloud_remove_last_missing_item_id = cloud_item_id
                        status = (
                            f"Cloud remove request for {cloud_item_id}, "
                            "but item is not placed in local shelf_state."
                        )
                        print(f"[RemoveRequest] Pending remove request for {cloud_item_id}, but not found locally.")
                        append_event(
                            "remove_request_failed",
                            {
                                "request_id": cloud_req.get("request_id"),
                                "item_id": cloud_item_id,
                                "shelf_id": cloud_req.get("shelf_id") or shelf_state.get("shelf_id", shelf_config.get("shelf_id", "SHELF_A")),
                                "level_id": cloud_req.get("level_id"),
                                "status": "not_found_locally",
                                "source": "jetson_mqtt_command_handler",
                                "note": "Jetson received remove command but item is not placed in local shelf_state.",
                            },
                        )
            # ---------- End remove/export request intake ----------

            if stage == "WAIT_ITEM":
                if args.measure_detector == "background" and measure_background is None:
                    status = "Measurement background not ready. Empty measure area, then press b."
                    measure_history.clear()
                elif now < cooldown_until:
                    status = "Cooldown after previous item. Please clear measurement area."
                else:
                    bbox = None
                    mask = None
                    measure_source = None

                    if args.measure_detector in ("yolo", "yolo_fallback") and measure_yolo_model is not None:
                        bbox, mask = detect_measure_object_yolo(
                            current_warped=warped_measure,
                            model=measure_yolo_model,
                            conf_thres=args.measure_yolo_conf,
                            imgsz=args.measure_yolo_imgsz,
                            min_area_px=args.measure_yolo_min_area,
                            allowed_classes=args.measure_yolo_classes,
                        )

                        if bbox is not None:
                            measure_source = f"YOLO:{bbox.get('cls_name', 'item')} {bbox.get('conf', 0.0)}"

                    if bbox is None and args.measure_detector in ("background", "yolo_fallback"):
                        if measure_background is not None:
                            bbox, mask = detect_object_from_background(
                                current_warped=warped_measure,
                                background_gray=measure_background,
                                threshold_value=args.measure_threshold,
                                min_area=args.measure_min_area,
                                background_color=measure_background_color,
                            )

                            if bbox is not None:
                                measure_source = "BG-FALLBACK" if args.measure_detector == "yolo_fallback" else "BG"
                        elif args.measure_detector == "yolo_fallback":
                            measure_source = "YOLO"

                    last_measure_mask = mask
                    last_measure_bbox = bbox

                    if bbox is None:
                        measure_history.clear()

                        if args.measure_detector == "yolo":
                            status = "YOLO measurement: waiting for item in measurement area."
                        elif args.measure_detector == "yolo_fallback":
                            if measure_background is None:
                                status = "YOLO measurement: waiting for item. BG fallback not ready; press b if needed."
                            else:
                                status = "YOLO/BG measurement: waiting for item in measurement area."
                        else:
                            status = "Waiting for item in measurement area."

                    else:
                        width_cm, height_cm = estimate_size_cm_from_measure_bbox(bbox, measure_config)
                        measure_history.append((width_cm, height_cm))

                        draw_measure_bbox_on_original(view, measure_config, bbox, H_measure_inv)

                        status = (
                            f"Measuring[{measure_source}]... {len(measure_history)}/{args.measure_stable_frames} "
                            f"current={width_cm}x{height_cm}cm"
                        )

                        if len(measure_history) >= args.measure_stable_frames:
                            widths = [x[0] for x in measure_history]
                            heights = [x[1] for x in measure_history]

                            width_med = round(float(np.median(widths)), 2)
                            height_med = round(float(np.median(heights)), 2)

                            measured_size_review = {
                                "width": width_med,
                                "height": height_med,
                                "depth": 0.0,
                            }

                            measured_bbox_review = dict(bbox)

                            status = (
                                f"Measured item [{measure_source}]: {width_med}cm x {height_med}cm. "
                                f"Press c to create ITEM_ID + QR, or r to re-measure."
                            )

                            print("")
                            print("======================================")
                            print("Measurement ready for review")
                            print(f"Measured size: {width_med}cm x {height_med}cm")
                            print(f"Measure source: {measure_source}")
                            print("Press c in the camera window to confirm and create ITEM_ID + QR.")
                            print("Press r to reject this measurement and measure again.")
                            print("======================================")
                            print("")

                            stage = "REVIEW_MEASURE"
                            measure_history.clear()

            elif stage == "REVIEW_MEASURE":
                if measured_bbox_review is not None:
                    draw_measure_bbox_on_original(
                        view,
                        measure_config,
                        measured_bbox_review,
                        H_measure_inv,
                    )

                if measured_size_review is not None:
                    status = (
                        f"Review measurement: "
                        f"{measured_size_review['width']}cm x {measured_size_review['height']}cm. "
                        f"Press c=confirm/create QR, r=re-measure, x=cancel."
                    )
                else:
                    status = "No measurement to review. Press r or x."

            elif stage == "WAIT_PLACE":
                if placement and accept_zone:
                    draw_interval_on_original(
                        frame=view,
                        shelf_config=shelf_config,
                        H_shelf_inv=H_shelf_inv,
                        shelf_w_px=shelf_w_px,
                        shelf_h_px=shelf_h_px,
                        level_index=placement["level_index"],
                        start_cm=placement["start_cm"],
                        end_cm=placement["end_cm"],
                        color=(255, 0, 255),
                        label="SUGGEST",
                        thickness=2,
                        fill=True,
                        alpha=0.25,
                    )

                    draw_interval_on_original(
                        frame=view,
                        shelf_config=shelf_config,
                        H_shelf_inv=H_shelf_inv,
                        shelf_w_px=shelf_w_px,
                        shelf_h_px=shelf_h_px,
                        level_index=accept_zone["level_index"],
                        start_cm=accept_zone["start_cm"],
                        end_cm=accept_zone["end_cm"],
                        color=(255, 120, 0),
                        label="ACCEPT ZONE",
                        thickness=2,
                        fill=False,
                    )

                if now < place_monitor_start_time:
                    remaining = place_monitor_start_time - now
                    place_stable_count = 0
                    last_place_candidate = None
                    last_shelf_mask = None
                    last_candidates = []
                    status = (
                        f"Place item into accept zone. "
                        f"Monitoring starts in {remaining:.1f}s..."
                    )

                else:
                    candidates = []
                    shelf_mask = last_monitor_mask

                    if args.shelf_detector in ("yolo", "yolo_fallback") and shelf_yolo_worker is not None:
                        # Use latest async YOLO candidates from the monitor block.
                        candidates = [clone_candidate_for_tracking(c) for c in last_live_candidates]
                        shelf_mask = last_monitor_mask

                    if (
                        not candidates
                        and args.shelf_detector in ("background", "yolo_fallback")
                        and before_shelf_gray is not None
                    ):
                        candidates, shelf_mask = detect_new_objects_on_shelf(
                            current_warped=warped_shelf,
                            before_gray=before_shelf_gray,
                            shelf_config=shelf_config,
                            threshold_value=args.shelf_threshold,
                            min_area=args.shelf_min_area,
                        )

                    best, evaluated = choose_best_candidate(
                        candidates=candidates,
                        placement=placement,
                        accept_zone=accept_zone,
                        item_width_cm=current_item["size_cm"]["width"],
                        min_overlap_ratio=args.min_overlap_ratio,
                        max_width_factor=args.max_width_factor,
                    )

                    last_shelf_mask = shelf_mask
                    last_candidates = evaluated

                    if evaluated:
                        draw_shelf_candidates(view, evaluated, H_shelf_inv)

                    if best is not None and best.get("evaluation", {}).get("accepted", False):
                        if candidate_is_position_stable(
                            previous_candidate=last_place_candidate,
                            current_candidate=best,
                            tolerance_cm=args.position_stable_tolerance_cm,
                        ):
                            place_stable_count += 1
                        else:
                            place_stable_count = 1

                        last_place_candidate = dict(best)

                        status = (
                            f"Placement accepted and stable "
                            f"{place_stable_count}/{args.place_stable_frames}: "
                            f"{best['level_id']} {best['start_cm']}->{best['end_cm']}cm"
                        )

                        if place_stable_count >= args.place_stable_frames:
                            ok = update_state_with_actual_position(
                                state=shelf_state,
                                item=current_item,
                                placement=placement,
                                actual=best,
                                occupied_padding_cm=args.occupied_padding_cm,
                                item_gap_cm=args.item_gap_cm,
                            )

                            if ok:
                                save_json(shelf_state, args.state)

                                suggested_position, actual_position = update_items_db_after_placement(
                                    item=current_item,
                                    placement=placement,
                                    accept_zone=accept_zone,
                                    actual=best,
                                    state=shelf_state,
                                    occupied_padding_cm=args.occupied_padding_cm,
                                    item_gap_cm=args.item_gap_cm,
                                )

                                clear_pending_item()

                                print("")
                                print("======================================")
                                print(f"PLACEMENT CONFIRMED: {current_item['item_id']}")
                                print(f"Suggested: {suggested_position['position_label']}")
                                print(f"Actual/reserved: {actual_position['position_label']}")
                                print(f"Updated: {args.state}")
                                print(f"Updated: {ITEM_DB_PATH}")

                                append_event(
                                    "item_placed",
                                    {
                                        "item_id": current_item["item_id"],
                                        "shelf_id": placement["shelf_id"],
                                        "level_id": actual_position["level_id"],
                                        "item_size_cm": current_item["size_cm"],
                                        "suggested_position": suggested_position,
                                        "actual_position": actual_position,
                                    },
                                )

                                print("======================================")
                                print("")

                                status = f"Confirmed {current_item['item_id']}. Waiting for next item."

                                current_item = None
                                placement = None
                                accept_zone = None
                                before_shelf_gray = None
                                place_stable_count = 0
                                place_monitor_start_time = 0.0
                                last_place_candidate = None
                                last_candidates = []
                                last_shelf_mask = None

                                stage = "WAIT_ITEM"
                                cooldown_until = time.time() + 2.0
                            else:
                                status = "Failed to update shelf state."

                    else:
                        place_stable_count = 0
                        last_place_candidate = None

                        if best is None:
                            status = "Waiting for item to be placed in accept zone."
                        else:
                            ev = best.get("evaluation", {})
                            status = f"Object detected but not accepted: {ev.get('reason')}"

            elif stage == "OUTBOUND_SELECT":
                outbound_items = get_placed_items_from_state(shelf_state)

                if not outbound_items:
                    status = "No placed item available for outbound. Press x to return."
                    outbound_target = None
                else:
                    outbound_selected_idx = max(0, min(outbound_selected_idx, len(outbound_items) - 1))
                    outbound_target = outbound_items[outbound_selected_idx]

                    draw_outbound_target_on_original(
                        frame=view,
                        target_item=outbound_target,
                        shelf_config=shelf_config,
                        H_shelf_inv=H_shelf_inv,
                        shelf_w_px=shelf_w_px,
                        shelf_h_px=shelf_h_px,
                        color=(0, 165, 255),
                        label_prefix="SELECT",
                    )

                    status = (
                        format_outbound_selection(outbound_items, outbound_selected_idx)
                        + " | n/p=change item | e=start removal | x=cancel"
                    )

            elif stage == "OUTBOUND_REMOVE":
                if outbound_target is None:
                    status = "No outbound target. Press x to return."
                else:
                    draw_outbound_target_on_original(
                        frame=view,
                        target_item=outbound_target,
                        shelf_config=shelf_config,
                        H_shelf_inv=H_shelf_inv,
                        shelf_w_px=shelf_w_px,
                        shelf_h_px=shelf_h_px,
                        color=(0, 0, 255),
                        label_prefix="REMOVE",
                    )

                    if args.shelf_detector == "background" and shelf_background is None:
                        status = "Shelf background not ready. Press x, then capture k."
                        outbound_empty_count = 0
                    elif args.shelf_detector in ("yolo", "yolo_fallback") and current_monitor_report is None and not last_live_candidates:
                        status = "Waiting for shelf YOLO detection before confirming outbound removal..."
                        outbound_empty_count = 0
                    else:
                        still_present = live_object_overlaps_target(
                            live_candidates=last_live_candidates,
                            target_item=outbound_target,
                            overlap_threshold_cm=args.outbound_overlap_threshold_cm,
                        )

                        if still_present:
                            outbound_empty_count = 0
                            status = (
                                f"Remove {outbound_target['item_id']} from "
                                f"{outbound_target['level_id']}. Waiting until target zone is empty..."
                            )
                        else:
                            outbound_empty_count += 1
                            status = (
                                f"Target zone empty {outbound_empty_count}/{args.outbound_empty_frames}. "
                                f"Confirming removal of {outbound_target['item_id']}..."
                            )

                            if outbound_empty_count >= args.outbound_empty_frames:
                                removed_info = remove_item_from_shelf_state(
                                    state=shelf_state,
                                    item_id=outbound_target["item_id"],
                                )

                                if removed_info is not None:
                                    save_json(shelf_state, args.state)

                                    update_items_db_after_removal(
                                        item_id=outbound_target["item_id"],
                                        removed_info=removed_info,
                                    )

                                    append_event(
                                        "item_removed",
                                        {
                                            "item_id": outbound_target["item_id"],
                                            "shelf_id": shelf_state.get("shelf_id", shelf_config.get("shelf_id", "SHELF_A")),
                                            "level_id": outbound_target["level_id"],
                                            "level_index": outbound_target["level_index"],
                                            "removed_position": {
                                                "start_cm": outbound_target["start_cm"],
                                                "end_cm": outbound_target["end_cm"],
                                                "actual_start_cm": outbound_target["actual_start_cm"],
                                                "actual_end_cm": outbound_target["actual_end_cm"],
                                            },
                                            "status": "removed",
                                        },
                                    )

                                    print("")
                                    print("======================================")
                                    print(f"ITEM REMOVED: {outbound_target['item_id']}")
                                    print(f"Updated: {args.state}")
                                    print(f"Updated: {ITEM_DB_PATH}")
                                    print("======================================")
                                    print("")

                                    status = f"Removed {outbound_target['item_id']}. Returning to inbound mode."

                                    outbound_items = []
                                    outbound_selected_idx = 0
                                    outbound_target = None
                                    outbound_empty_count = 0
                                    stage = "WAIT_ITEM"
                                    cooldown_until = time.time() + 1.0
                                else:
                                    status = "Failed to remove item from shelf_state. Press x to cancel."

            elif stage == "NO_PLACEMENT":
                status = "No placement available. Press x to cancel current item."

            if stage.startswith("OUTBOUND"):
                if outbound_target is not None:
                    item_text = (
                        f"Outbound target: {outbound_target['item_id']} | "
                        f"{outbound_target['level_id']} "
                        f"{outbound_target['start_cm']:.2f}->{outbound_target['end_cm']:.2f}cm"
                    )
                else:
                    item_text = "Outbound target: none"
            elif current_item is None:
                if measured_size_review is not None:
                    item_text = (
                        f"Measured item pending review: "
                        f"{measured_size_review['width']}x{measured_size_review['height']}cm"
                    )
                else:
                    item_text = "Current item: none"
            else:
                item_text = (
                    f"Current item: {current_item['item_id']} | "
                    f"{current_item['size_cm']['width']}x{current_item['size_cm']['height']}cm"
                )

            if placement is None:
                place_text = "Placement: none"
            else:
                place_text = (
                    f"Placement: {placement['level_id']} "
                    f"{placement['start_cm']}->{placement['end_cm']}cm"
                )

            if args.measure_detector == "yolo":
                measure_bg_text = "Measure: YOLO"
            elif args.measure_detector == "yolo_fallback":
                measure_bg_state = "BG READY" if measure_background is not None else "BG NOT READY"
                measure_bg_text = f"Measure: YOLO+Fallback ({measure_bg_state})"
            else:
                measure_bg_text = "Measure BG: READY" if measure_background is not None else "Measure BG: NOT READY"

            if args.shelf_detector == "yolo":
                shelf_bg_text = "Shelf: YOLO"
            elif args.shelf_detector == "yolo_fallback":
                shelf_bg_state = "BG READY" if shelf_background is not None else "BG NOT READY"
                shelf_bg_text = f"Shelf: YOLO+Fallback ({shelf_bg_state})"
            else:
                shelf_bg_text = "Shelf BG: READY" if shelf_background is not None else "Shelf BG: NOT READY"

            bg_text = measure_bg_text + " | " + shelf_bg_text

            draw_warehouse_monitor_panel(
                frame=view,
                monitor_report=current_monitor_report,
                state=shelf_state,
                shelf_config=shelf_config,
            )

            messages = [
                f"Stage: {stage} | {bg_text}",
                item_text,
                place_text,
                f"Status: {status}",
                monitor_summary,
                "Controls: b=measure BG | k=shelf BG | c=confirm | o=outbound | n/p=select | e=export | x=cancel | q=quit",
            ]

            display_shelf_mask = last_shelf_mask if last_shelf_mask is not None else last_monitor_mask

            display = make_display(
                frame=view,
                measure_mask=last_measure_mask,
                shelf_mask=display_shelf_mask,
                messages=messages,
            )

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("b"):
                print("Capturing stable EMPTY measurement background...")
                measure_background = capture_background_average(
                    camera=camera,
                    H=H_measure,
                    out_w=measure_w_px,
                    out_h=measure_h_px,
                    sample_count=args.background_samples,
                )

                if measure_background is not None:
                    measure_background_color = capture_background_color_average(
                        camera=camera,
                        H=H_measure,
                        out_w=measure_w_px,
                        out_h=measure_h_px,
                        sample_count=args.background_samples,
                    )

                    np.save(MEASURE_BACKGROUND_PATH, measure_background)

                    if measure_background_color is not None:
                        np.save(MEASURE_BACKGROUND_COLOR_PATH, measure_background_color)

                    measure_history.clear()
                    status = f"Captured stable empty measurement background: {MEASURE_BACKGROUND_PATH}"
                else:
                    status = "ERROR: Failed to capture measurement background."

                print(status)

            elif key == ord("k"):
                print("Capturing stable EMPTY shelf background...")
                shelf_background = capture_background_average(
                    camera=camera,
                    H=H_shelf,
                    out_w=shelf_w_px,
                    out_h=shelf_h_px,
                    sample_count=args.background_samples,
                )

                if shelf_background is not None:
                    np.save(SHELF_BACKGROUND_PATH, shelf_background)
                    status = f"Captured stable empty shelf background: {SHELF_BACKGROUND_PATH}"
                else:
                    status = "ERROR: Failed to capture shelf background."

                print(status)

            elif key == ord("c"):
                if stage != "REVIEW_MEASURE":
                    print("No measurement waiting for confirmation.")
                    continue

                if measured_size_review is None:
                    print("No measured size to confirm.")
                    continue

                width_med = float(measured_size_review["width"])
                height_med = float(measured_size_review["height"])

                current_item = create_item(
                    width_cm=width_med,
                    height_cm=height_med,
                    note=args.note,
                )

                placement = find_placement(
                    state=shelf_state,
                    item_width_cm=width_med,
                    item_height_cm=height_med,
                    item_depth_cm=0.0,
                )

                if not placement["found"]:
                    status = f"No suitable placement for {current_item['item_id']}: {placement.get('reason')}"
                    print(status)
                    stage = "NO_PLACEMENT"
                else:
                    accept_zone = make_accept_zone(placement, args.tolerance_cm)
                    suggested_position = build_suggested_position(placement, accept_zone)
                    before_shelf_gray = preprocess(warped_shelf)
                    place_monitor_start_time = time.time() + args.place_arm_delay_sec
                    place_stable_count = 0
                    last_place_candidate = None

                    status = (
                        f"Created {current_item['item_id']} | "
                        f"{width_med}x{height_med}cm | "
                        f"Suggest {placement['level_id']} "
                        f"{placement['start_cm']}->{placement['end_cm']}cm"
                    )

                    print("")
                    print("======================================")
                    print("New item created")
                    print(f"Item ID: {current_item['item_id']}")
                    print(f"Size: {width_med}cm x {height_med}cm")
                    print(f"Suggested: {placement['level_id']} {placement['start_cm']} -> {placement['end_cm']}cm")
                    print("======================================")
                    print("")

                    append_event(
                        "item_created",
                        {
                            "item_id": current_item["item_id"],
                            "shelf_id": placement["shelf_id"],
                            "level_id": placement["level_id"],
                            "item_size_cm": current_item["size_cm"],
                            "size_cm": current_item["size_cm"],
                            "suggested_position": suggested_position,
                        },
                    )

                    measured_size_review = None
                    measured_bbox_review = None
                    stage = "WAIT_PLACE"
                    place_stable_count = 0

            elif key == ord("r"):
                if stage == "REVIEW_MEASURE":
                    measured_size_review = None
                    measured_bbox_review = None
                    measure_history.clear()
                    stage = "WAIT_ITEM"
                    status = "Measurement rejected. Put item in measure area again."
                    print(status)

            elif key == ord("o"):
                if stage in ["WAIT_PLACE", "REVIEW_MEASURE"]:
                    print("Cannot enter outbound while an inbound item is being processed. Press x to cancel first if needed.")
                    continue

                outbound_items = get_placed_items_from_state(shelf_state)
                outbound_selected_idx = 0
                outbound_target = outbound_items[0] if outbound_items else None
                outbound_empty_count = 0
                stage = "OUTBOUND_SELECT"

                if outbound_target is None:
                    status = "No placed item available for outbound."
                else:
                    status = format_outbound_selection(outbound_items, outbound_selected_idx)

                print(status)

            elif key == ord("n"):
                if stage == "OUTBOUND_SELECT" and outbound_items:
                    outbound_selected_idx = (outbound_selected_idx + 1) % len(outbound_items)
                    outbound_target = outbound_items[outbound_selected_idx]
                    status = format_outbound_selection(outbound_items, outbound_selected_idx)
                    print(status)
                else:
                    print("n is only used in OUTBOUND_SELECT mode.")

            elif key == ord("p"):
                if stage == "OUTBOUND_SELECT" and outbound_items:
                    outbound_selected_idx = (outbound_selected_idx - 1) % len(outbound_items)
                    outbound_target = outbound_items[outbound_selected_idx]
                    status = format_outbound_selection(outbound_items, outbound_selected_idx)
                    print(status)
                else:
                    print("p is only used in OUTBOUND_SELECT mode.")

            elif key == ord("e"):
                if stage == "OUTBOUND_SELECT" and outbound_target is not None:
                    outbound_empty_count = 0
                    stage = "OUTBOUND_REMOVE"
                    status = f"Start removal for {outbound_target['item_id']}. Remove it from shelf."
                    print(status)
                else:
                    print("e is only used after selecting an outbound item.")

            elif key == ord("x"):
                if current_item is not None:
                    print(f"Cancelled current item/cycle: {current_item['item_id']}")

                clear_pending_item()
                current_item = None
                placement = None
                accept_zone = None
                before_shelf_gray = None
                measured_size_review = None
                measured_bbox_review = None
                measure_history.clear()
                place_stable_count = 0
                place_monitor_start_time = 0.0
                last_place_candidate = None
                last_candidates = []
                last_shelf_mask = None
                last_live_candidates = []
                prev_live_candidates = []
                last_monitor_mask = None
                outbound_items = []
                outbound_selected_idx = 0
                outbound_target = None
                outbound_empty_count = 0
                stage = "WAIT_ITEM"
                cooldown_until = time.time() + 1.0
                status = "Cycle cancelled. Waiting for next item."

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        try:
            if shelf_yolo_worker is not None:
                shelf_yolo_worker.stop()
        except Exception:
            pass

        stop_mqtt_led_client()
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":
    main()
