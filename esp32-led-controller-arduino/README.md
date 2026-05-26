# ESP32 Warehouse LED Controller for Arduino IDE

Arduino IDE sketch for the warehouse shelf LED indicator. This is the same MQTT
interface as the ESP-IDF firmware:

```text
Render/Warehouse Cloud -> HiveMQ -> warehouse/SHELF_A/led/command -> ESP32
```

## Hardware

Use one active-high LED per shelf level:

```text
T1 -> GPIO16 -> resistor -> LED -> GND
T2 -> GPIO17 -> resistor -> LED -> GND
T3 -> GPIO18 -> resistor -> LED -> GND
T4 -> GPIO19 -> resistor -> LED -> GND
```

## Arduino IDE setup

1. Install the ESP32 board package from Boards Manager.
2. Select board `NodeMCU-32S` or `ESP32 Dev Module`.
3. Install libraries from Library Manager:
   - `PubSubClient` by Nick O'Leary
   - `ArduinoJson` by Benoit Blanchon
4. Open:

```text
esp32-led-controller-arduino/warehouse_led_controller_arduino/warehouse_led_controller_arduino.ino
```

5. Copy `arduino_secrets.h.example` to `arduino_secrets.h`.
6. Fill WiFi and HiveMQ values in `arduino_secrets.h`.
7. Upload to the ESP32.

`arduino_secrets.h` is ignored by git so WiFi and HiveMQ credentials stay local.

## Notes about sdkconfig

Arduino IDE sketches do not use ESP-IDF `sdkconfig`. The ESP-IDF project still
uses `esp32-led-controller/sdkconfig.defaults`; this Arduino sketch keeps the
same GPIOs, MQTT topic, blink interval, and timeout in normal C/C++ constants.

## MQTT commands

Blink a shelf level:

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
