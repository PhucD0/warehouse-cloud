# ESP32 Warehouse LED Controller

ESP-IDF firmware for the shelf-level LED indicator. It connects to WiFi, subscribes to HiveMQ, and blinks one 5mm LED for the suggested shelf level.

## LED wiring

Use one LED per shelf level:

```text
T1 -> GPIO16 -> resistor -> LED -> GND
T2 -> GPIO17 -> resistor -> LED -> GND
T3 -> GPIO18 -> resistor -> LED -> GND
T4 -> GPIO19 -> resistor -> LED -> GND
```

The firmware uses active-high output: GPIO 1 turns the LED on.

## Configure

Open an ESP-IDF terminal, then run:

```powershell
cd C:\Users\jach9\Downloads\IOT-DoAn\warehouse-cloud\esp32-led-controller
idf.py set-target esp32
idf.py menuconfig
```

In `Warehouse LED Controller`, set:

- `WiFi SSID`
- `WiFi Password`
- `HiveMQ MQTT broker URI`, for example `mqtts://xxxxx.s1.eu.hivemq.cloud:8883`
- `HiveMQ username`
- `HiveMQ password`
- `LED command topic`, default `warehouse/SHELF_A/led/command`

`sdkconfig` is intentionally ignored because it contains WiFi and HiveMQ secrets. Safe defaults live in `sdkconfig.defaults`.

## Build and flash

```powershell
idf.py build
idf.py -p COM_PORT flash monitor
```

Replace `COM_PORT` with the ESP32 serial port, for example `COM5`.

## MQTT command format

Blink one shelf level:

```json
{
  "command": "blink",
  "shelf_id": "SHELF_A",
  "level_id": "T2",
  "item_id": "ITEM_ABC123",
  "blink_ms": 500,
  "timeout_ms": 120000,
  "source": "warehouse-cloud"
}
```

Clear all LEDs:

```json
{
  "command": "clear",
  "shelf_id": "SHELF_A",
  "item_id": "ITEM_ABC123",
  "reason": "item_placed"
}
```
