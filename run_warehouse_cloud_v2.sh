#!/bin/bash
cd ~
export WAREHOUSE_CLOUD_URL=https://warehouse-cloud.onrender.com
export WAREHOUSE_CLOUD_TIMEOUT=8

echo "[RUN] WAREHOUSE_CLOUD_URL=$WAREHOUSE_CLOUD_URL"
curl -4 --tlsv1.2 -s https://warehouse-cloud.onrender.com/api/health
echo

export WAREHOUSE_REMOVE_POLL_INTERVAL=0.7
export WAREHOUSE_CLOUD_TIMEOUT=2
python3 warehouse_v2.py \
  --measure-detector yolo \
  --measure-yolo-model yolov8s-worldv2.pt \
  --measure-yolo-conf 0.05 \
  --measure-yolo-imgsz 320 \
  --measure-yolo-classes "box,bottle,cup" \
  --shelf-detector yolo \
  --shelf-yolo-model yolov8s-worldv2.pt \
  --shelf-yolo-conf 0.05 \
  --shelf-yolo-imgsz 416 \
  --shelf-yolo-interval-sec 2.0 \
  --shelf-yolo-max-age-sec 6.0 \
  --shelf-yolo-classes "box,bottle,cup"
  --cloud-status-interval-sec 3.0
  --cloud-snapshot-interval-sec 10.0
