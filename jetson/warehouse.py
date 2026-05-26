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

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)

        if not self.is_opened():
            return False

        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter.fourcc(*self.fourcc),
        )
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
    shelf_width_cm = float(state["physical_size_cm"]["width"])

    actual_start = float(actual["start_cm"])
    actual_end = float(actual["end_cm"])
    actual_width = max(0.001, actual_end - actual_start)
    actual_center = (actual_start + actual_end) / 2.0

    measured_width = float(item["size_cm"]["width"])

    reserved_width = max(actual_width, measured_width)
    reserved_width = reserved_width + (2.0 * occupied_padding_cm) + item_gap_cm

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

def detect_object_from_background(current_warped, background_gray, threshold_value=35, min_area=80):
    current_gray = preprocess(current_warped)
    diff = cv2.absdiff(background_gray, current_gray)

    _, mask = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

    # Gi? nguyÃªn kernel l?n 5x5 d? n?i li?n thÃ¢n chai trong su?t khÃ´ng b? d?t gÃ£y
    kernel_large = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_large, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=5)
    mask = cv2.dilate(mask, kernel_large, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            valid.append(contour)

    if not valid:
        return None, mask

    largest = max(valid, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Ã?y d? w vÃ  h d? tÃ­nh toÃ¡n do d?c kÃ­ch thu?c chÃ­nh xÃ¡c ? hÃ m main
    return {
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h),
        "area_px": float(cv2.contourArea(largest)),
    }, mask


def estimate_size_cm_from_measure_bbox(bbox, measure_config):
    cm_per_px_x = float(measure_config["pixel_scale"]["cm_per_px_x"])
    cm_per_px_y = float(measure_config["pixel_scale"]["cm_per_px_y"])

    width_cm = bbox["w"] * cm_per_px_x
    height_cm = bbox["h"] * cm_per_px_y

    return round(width_cm, 2), round(height_cm, 2)


# ============================================================
# 2. Khu v?c K? HÃ€NG (Shelf area) - THÃŠM B? L?C CH?N V?T L? < 1x1 CM
# ============================================================

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

    # L?y kÃ­ch thu?c th?c t? c?a k? d? tÃ­nh t? l? quy d?i cm/pixel
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

            # --- B? L?C KÃCH THU?C V?T L? TRÃŠN K? (> 1x1 cm) ---
            obj_width_cm = w * cm_per_px_x
            obj_height_cm = h * cm_per_px_y

            # N?u kÃ­ch thu?c v?t th? l? nh? hon ho?c b?ng 1cm x 1cm th?c t?, b? qua trÃ¹ng nhi?u h?t
            if obj_width_cm <= 1.0 or obj_height_cm <= 1.0:
                continue
            # --------------------------------------------------

            y_global = y + y1_level
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
                    "center_cm": round((x_start_cm + x_end_cm) / 2.0, 3),
                }
            )

    return candidates, mask


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


def make_display(frame, measure_mask, shelf_mask, messages):
    h, w = frame.shape[:2]

    small_w = w // 3
    small_h = h // 3

    if measure_mask is None:
        measure_panel = np.zeros((small_h, small_w, 3), dtype=np.uint8)
    else:
        measure_panel = cv2.cvtColor(measure_mask, cv2.COLOR_GRAY2BGR)
        measure_panel = cv2.resize(measure_panel, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

    if shelf_mask is None:
        shelf_panel = np.zeros((small_h, small_w, 3), dtype=np.uint8)
    else:
        shelf_panel = cv2.cvtColor(shelf_mask, cv2.COLOR_GRAY2BGR)
        shelf_panel = cv2.resize(shelf_panel, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

    frame_show = frame.copy()

    x0 = w - small_w - 10
    y0 = 10
    frame_show[y0:y0 + small_h, x0:x0 + small_w] = measure_panel
    cv2.putText(frame_show, "Measure mask", (x0, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    y1 = y0 + small_h + 10
    frame_show[y1:y1 + small_h, x0:x0 + small_w] = shelf_panel
    cv2.putText(frame_show, "Shelf mask", (x0, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    canvas = np.zeros((h + INFO_BAR_HEIGHT, w, 3), dtype=np.uint8)
    canvas[INFO_BAR_HEIGHT:INFO_BAR_HEIGHT + h, :] = frame_show

    y = 25
    for msg in messages:
        cv2.putText(
            canvas,
            msg,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            1,
        )
        y += 25

    display = cv2.resize(
        canvas,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
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


def draw_registered_items_on_original(
    frame,
    state,
    shelf_config,
    H_shelf_inv,
    shelf_w_px,
    shelf_h_px,
):
    for level in state.get("levels", []):
        level_index = int(level.get("level_index", 1))

        for occ in level.get("occupied_intervals", []):
            item_id = occ.get("item_id", "ITEM")

            bbox = occ.get("bbox_rect")

            if bbox is not None:
                x = bbox.get("x", 0)
                y = bbox.get("y", 0)
                w = bbox.get("w", 0)
                h = bbox.get("h", 0)

                poly_rect = np.array(
                    [
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h],
                    ],
                    dtype=np.float32,
                )

                poly_orig = rect_poly_to_original(poly_rect, H_shelf_inv)

                cv2.polylines(frame, [poly_orig], True, (0, 255, 255), 2)

                cx = int(np.mean(poly_orig[:, 0]))
                cy = int(np.mean(poly_orig[:, 1]))

                cv2.putText(
                    frame,
                    f"REG {item_id}",
                    (cx - 45, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 255, 255),
                    1,
                )

            else:
                start_cm = float(occ.get("start_cm", 0.0))
                end_cm = float(occ.get("end_cm", start_cm))

                draw_interval_on_original(
                    frame=frame,
                    shelf_config=shelf_config,
                    H_shelf_inv=H_shelf_inv,
                    shelf_w_px=shelf_w_px,
                    shelf_h_px=shelf_h_px,
                    level_index=level_index,
                    start_cm=start_cm,
                    end_cm=end_cm,
                    color=(0, 255, 255),
                    label=f"REG {item_id}",
                    thickness=1,
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

def append_event(event_type, payload=None):
    if payload is None:
        payload = {}

    event = {
        "event_type": event_type,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **payload,
    }

    with open(WAREHOUSE_EVENTS_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")

    print(f"[EVENT] {event_type}: {payload}")



    # Send to cloud with both top-level fields and nested payload.
    # Top-level fields keep compatibility with local event log.
    # Nested payload helps the backend parse size_cm, suggested_position,
    # actual_position, removed_position, expected_count, detected_count, etc.
    _cloud_event = dict(event)
    _cloud_event["payload"] = dict(payload) if isinstance(payload, dict) else {}
    send_event_to_cloud_async(_cloud_event)
def build_monitor_report(state, live_candidates, shelf_config):
    num_levels = int(shelf_config.get("num_levels", 4))
    shelf_id = state.get("shelf_id", shelf_config.get("shelf_id", "SHELF_A"))

    expected_counts = get_expected_counts_by_level(state)
    detected_counts = get_detected_counts_by_level(live_candidates, num_levels)

    levels = []
    has_warning = False

    for i in range(num_levels):
        level_id = f"T{i + 1}"

        expected = int(expected_counts.get(level_id, 0))
        detected = int(detected_counts.get(level_id, 0))

        if expected == detected:
            status = "ok"
            event_status = "ok"
        elif detected < expected:
            status = "MISSING?"
            event_status = "suspected_missing_or_merged"
            has_warning = True
        else:
            status = "UNKNOWN?"
            event_status = "unknown_object_detected"
            has_warning = True

        levels.append(
            {
                "shelf_id": shelf_id,
                "level_id": level_id,
                "expected_count": expected,
                "detected_count": detected,
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


def update_monitor_warning_events(
    report,
    warning_counters,
    last_event_times,
    required_frames,
    cooldown_sec,
):
    if report is None:
        return

    now = time.time()

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
    Only items with status='placed' in items_db.json are selectable.
    """
    status_map = load_item_status_map()
    placed_items = []

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

            item = {
                "item_id": item_id,
                "level_id": level_id,
                "level_index": level_index,
                "start_cm": float(occ.get("start_cm", 0.0)),
                "end_cm": float(occ.get("end_cm", 0.0)),
                "actual_start_cm": float(occ.get("actual_start_cm", occ.get("start_cm", 0.0))),
                "actual_end_cm": float(occ.get("actual_end_cm", occ.get("end_cm", 0.0))),
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
    Check whether any live detected object still overlaps the registered item zone.
    If there is no overlap for enough frames, the item is considered removed.
    """
    if target_item is None:
        return False

    target_level = int(target_item["level_index"])
    target_start = float(target_item["start_cm"])
    target_end = float(target_item["end_cm"])

    for cand in live_candidates:
        if int(cand.get("level_index", -1)) != target_level:
            continue

        cand_start = float(cand.get("start_cm", 0.0))
        cand_end = float(cand.get("end_cm", 0.0))

        overlap = interval_overlap(target_start, target_end, cand_start, cand_end)

        if overlap >= overlap_threshold_cm:
            return True

        center = float(cand.get("center_cm", -999.0))
        if target_start <= center <= target_end:
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

    parser.add_argument("--shelf-threshold", type=int, default=35)
    parser.add_argument("--shelf-min-area", type=int, default=120)

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

    # More stable background capture under changing lighting.
    parser.add_argument("--background-samples", type=int, default=15)

    # Outbound / item removal
    parser.add_argument("--outbound-empty-frames", type=int, default=15)
    parser.add_argument("--outbound-overlap-threshold-cm", type=float, default=0.25)

    parser.add_argument("--note", type=str, default="smart warehouse auto item")

    return parser.parse_args()


def main():
    args = parse_args()

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

    shelf_background = None
    if os.path.exists(SHELF_BACKGROUND_PATH):
        try:
            shelf_background = np.load(SHELF_BACKGROUND_PATH)
            print(f"Loaded shelf background: {SHELF_BACKGROUND_PATH}")
        except Exception:
            shelf_background = None

    camera_cfg = shelf_config.get("camera", {})

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

    try:
        while True:
            ret, frame = camera.read()

            if not ret or frame is None:
                print("ERROR: Failed to read frame from USB Camera.")
                break

            now = time.time()

            warped_measure = warp_area(frame, H_measure, measure_w_px, measure_h_px)
            warped_shelf = warp_area(frame, H_shelf, shelf_w_px, shelf_h_px)

            view = frame.copy()

            draw_shelf_levels(view, shelf_config)
            draw_quad(view, measure_config, (0, 200, 255), "MEASURE AREA")

            last_measure_bbox = None
            last_candidates = []
            last_live_candidates = []
            last_monitor_mask = None

            if shelf_background is not None:
                live_candidates, monitor_mask = detect_new_objects_on_shelf(
                    current_warped=warped_shelf,
                    before_gray=shelf_background,
                    shelf_config=shelf_config,
                    threshold_value=args.shelf_threshold,
                    min_area=args.shelf_min_area,
                )

                last_live_candidates = live_candidates
                last_monitor_mask = monitor_mask

                monitor_report = build_monitor_report(
                    state=shelf_state,
                    live_candidates=last_live_candidates,
                    shelf_config=shelf_config,
                )

                current_monitor_report = monitor_report
                monitor_summary = monitor_report_to_summary(monitor_report)

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
                monitor_summary = "Shelf monitor: Shelf BG not ready. Empty shelf, then press k."

            # ---------- Cloud remove request polling ----------
            # Only poll when Jetson is idle. Inbound / manual outbound flows are not interrupted.
            if (
                WAREHOUSE_CLOUD_URL
                and stage == "WAIT_ITEM"
                and (now - cloud_remove_last_poll) >= WAREHOUSE_REMOVE_POLL_INTERVAL
            ):
                cloud_remove_last_poll = now
                pending_remove_requests = fetch_pending_remove_requests_from_cloud()

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

                        print("")
                        print("======================================")
                        print(f"CLOUD REMOVE REQUEST: {cloud_item_id}")
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
                        print(f"[Cloud] Pending remove request for {cloud_item_id}, but not found locally.")
            # ---------- End cloud remove request polling ----------

            if stage == "WAIT_ITEM":
                if measure_background is None:
                    status = "Measurement background not ready. Empty measure area, then press b."
                    measure_history.clear()
                elif now < cooldown_until:
                    status = "Cooldown after previous item. Please clear measurement area."
                else:
                    bbox, mask = detect_object_from_background(
                        current_warped=warped_measure,
                        background_gray=measure_background,
                        threshold_value=args.measure_threshold,
                        min_area=args.measure_min_area,
                    )

                    last_measure_mask = mask
                    last_measure_bbox = bbox

                    if bbox is None:
                        measure_history.clear()
                        status = "Waiting for item in measurement area."
                    else:
                        width_cm, height_cm = estimate_size_cm_from_measure_bbox(bbox, measure_config)
                        measure_history.append((width_cm, height_cm))

                        draw_measure_bbox_on_original(view, measure_config, bbox, H_measure_inv)

                        status = (
                            f"Measuring... {len(measure_history)}/{args.measure_stable_frames} "
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
                                f"Measured item: {width_med}cm x {height_med}cm. "
                                f"Press c to create ITEM_ID + QR, or r to re-measure."
                            )

                            print("")
                            print("======================================")
                            print("Measurement ready for review")
                            print(f"Measured size: {width_med}cm x {height_med}cm")
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

                    if shelf_background is None:
                        status = "Shelf background not ready. Press x, then capture k."
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

            measure_bg_text = "Measure BG: READY" if measure_background is not None else "Measure BG: NOT READY"
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
                    np.save(MEASURE_BACKGROUND_PATH, measure_background)
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
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":
    main()
