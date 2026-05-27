#!/bin/bash
cd ~

# Cloud backend is still used for dashboard/API health check only.
# In v4, Jetson <-> Cloud communication goes THROUGH MQTT broker:
#   Jetson -> MQTT Broker -> Cloud Backend for events/metadata.
#   Cloud Backend -> MQTT Broker -> Jetson for remove/export commands.
export WAREHOUSE_CLOUD_URL=https://warehouse-cloud.onrender.com
export WAREHOUSE_CLOUD_TIMEOUT=2
export WAREHOUSE_REMOVE_POLL_INTERVAL=0.7
export WAREHOUSE_REMOVE_REQUEST_TRANSPORT=mqtt

# v4 event transport: Jetson -> MQTT Broker -> Cloud Backend subscriber
export WAREHOUSE_EVENT_TRANSPORT=mqtt
export WAREHOUSE_MQTT_EVENT_ENABLED=true
export WAREHOUSE_MQTT_EVENT_TOPIC="warehouse/jetson/events"

# v4 command transport: Cloud Backend -> MQTT Broker -> Jetson
export WAREHOUSE_MQTT_COMMAND_ENABLED=true
export WAREHOUSE_MQTT_COMMAND_TOPIC="warehouse/jetson/commands"
export WAREHOUSE_MQTT_COMMAND_QOS=1

# Jetson -> MQTT Broker -> ESP32 LED commands
export WAREHOUSE_MQTT_LED_ENABLED=true
export WAREHOUSE_MQTT_TOPIC_PREFIX="warehouse"
export WAREHOUSE_MQTT_QOS=1
export WAREHOUSE_MQTT_RETAIN=false
export WAREHOUSE_MQTT_EVENT_RETAIN=false
export WAREHOUSE_MQTT_LED_BLINK_MS=500
export WAREHOUSE_MQTT_LED_TIMEOUT_MS=120000

# HiveMQ Cloud broker
export WAREHOUSE_MQTT_HOST="ff9ad900e3824825abc729dc69e1d37c.s1.eu.hivemq.cloud"
export WAREHOUSE_MQTT_PORT=8883
export WAREHOUSE_MQTT_USERNAME="lqh-coding-frenzy"
export WAREHOUSE_MQTT_PASSWORD="Jach.khung1"
export WAREHOUSE_MQTT_TLS=true
export WAREHOUSE_MQTT_CLIENT_ID="jetson-warehouse-v4"

# UI scale, optional
export WAREHOUSE_UI_SCALE=0.72

# Periodic monitor status push interval
export WAREHOUSE_MONITOR_OK_EVENT_INTERVAL=8

echo "[RUN:v4] WAREHOUSE_CLOUD_URL=$WAREHOUSE_CLOUD_URL"
echo "[RUN:v4] EVENT_TRANSPORT=$WAREHOUSE_EVENT_TRANSPORT topic=$WAREHOUSE_MQTT_EVENT_TOPIC"
echo "[RUN:v4] REMOVE_REQUEST_TRANSPORT=$WAREHOUSE_REMOVE_REQUEST_TRANSPORT command_topic=$WAREHOUSE_MQTT_COMMAND_TOPIC"
echo "[RUN:v4] MQTT host=$WAREHOUSE_MQTT_HOST port=$WAREHOUSE_MQTT_PORT tls=$WAREHOUSE_MQTT_TLS prefix=$WAREHOUSE_MQTT_TOPIC_PREFIX"
curl -4 --tlsv1.2 -s "$WAREHOUSE_CLOUD_URL/api/health"
echo

python3 warehouse_v4.py \
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
  --shelf-yolo-classes "box,bottle,cup" \
  --cloud-status-interval-sec 3.0
