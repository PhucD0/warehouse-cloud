#!/bin/bash
cd ~

export WAREHOUSE_CLOUD_URL=https://warehouse-cloud.onrender.com
export WAREHOUSE_CLOUD_TIMEOUT=8
export WAREHOUSE_REMOVE_POLL_INTERVAL=0.7
export WAREHOUSE_CLOUD_TIMEOUT=2

# Jetson -> MQTT Broker -> ESP32 LED command
# HiveMQ Cloud broker configured from current project environment.
export WAREHOUSE_MQTT_LED_ENABLED=true
export WAREHOUSE_MQTT_HOST="ff9ad900e3824825abc729dc69e1d37c.s1.eu.hivemq.cloud"
export WAREHOUSE_MQTT_PORT=8883
export WAREHOUSE_MQTT_USERNAME="lqh-coding-frenzy"
export WAREHOUSE_MQTT_PASSWORD="Jach.khung1"
export WAREHOUSE_MQTT_TLS=true
export WAREHOUSE_MQTT_TOPIC_PREFIX="warehouse"
export WAREHOUSE_MQTT_QOS=1
export WAREHOUSE_MQTT_RETAIN=false
export WAREHOUSE_MQTT_LED_BLINK_MS=500
export WAREHOUSE_MQTT_LED_TIMEOUT_MS=120000

# UI scale, optional
export WAREHOUSE_UI_SCALE=0.72

echo "[RUN] WAREHOUSE_CLOUD_URL=$WAREHOUSE_CLOUD_URL"
echo "[RUN] MQTT_LED=$WAREHOUSE_MQTT_LED_ENABLED host=$WAREHOUSE_MQTT_HOST port=$WAREHOUSE_MQTT_PORT tls=$WAREHOUSE_MQTT_TLS prefix=$WAREHOUSE_MQTT_TOPIC_PREFIX"
curl -4 --tlsv1.2 -s https://warehouse-cloud.onrender.com/api/health
echo

python3 warehouse_v3.py   --measure-detector yolo   --measure-yolo-model yolov8s-worldv2.pt   --measure-yolo-conf 0.05   --measure-yolo-imgsz 320   --measure-yolo-classes "box,bottle,cup"   --shelf-detector yolo   --shelf-yolo-model yolov8s-worldv2.pt   --shelf-yolo-conf 0.05   --shelf-yolo-imgsz 416   --shelf-yolo-interval-sec 2.0   --shelf-yolo-max-age-sec 6.0   --shelf-yolo-classes "box,bottle,cup"   --cloud-status-interval-sec 3.0
