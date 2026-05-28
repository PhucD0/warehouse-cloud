#include <Arduino.h>
#include <ArduinoJson.h>
#include <PubSubClient.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>

#include "arduino_secrets.h"

#ifndef WIFI_SSID
#error "Create arduino_secrets.h from arduino_secrets.h.example and set WIFI_SSID."
#endif

#ifndef WIFI_PASSWORD
#error "Create arduino_secrets.h from arduino_secrets.h.example and set WIFI_PASSWORD."
#endif

#ifndef MQTT_BROKER
#error "Create arduino_secrets.h from arduino_secrets.h.example and set MQTT_BROKER."
#endif

#ifndef MQTT_USERNAME
#error "Create arduino_secrets.h from arduino_secrets.h.example and set MQTT_USERNAME."
#endif

#ifndef MQTT_PASSWORD
#error "Create arduino_secrets.h from arduino_secrets.h.example and set MQTT_PASSWORD."
#endif

#ifndef MQTT_PORT
#define MQTT_PORT 8883
#endif

#ifndef MQTT_USE_INSECURE_TLS
#define MQTT_USE_INSECURE_TLS 1
#endif

static const char SHELF_ID[] = "SHELF_A";
static const char MQTT_TOPIC[] = "warehouse/SHELF_A/led/command";
static const char MQTT_CLIENT_ID_PREFIX[] = "warehouse-led-esp32-";

static const uint8_t LED_GPIO_T1 = 16;
static const uint8_t LED_GPIO_T2 = 17;
static const uint8_t LED_GPIO_T3 = 18;
static const uint8_t LED_GPIO_T4 = 19;
static const uint8_t LED_GPIOS[] = {LED_GPIO_T1, LED_GPIO_T2, LED_GPIO_T3, LED_GPIO_T4};
static const uint8_t LED_COUNT = sizeof(LED_GPIOS) / sizeof(LED_GPIOS[0]);
static const uint8_t ALL_LED_MASK = (uint8_t)((1U << LED_COUNT) - 1U);

static const unsigned long DEFAULT_BLINK_MS = 500;
static const unsigned long DEFAULT_TIMEOUT_MS = 120000;
static const unsigned long ALERT_BLINK_MS = 75;
static const unsigned long ALERT_TIMEOUT_MS = 300000;
static const unsigned long WIFI_RETRY_MS = 500;
static const unsigned long MQTT_RETRY_MS = 5000;
static const size_t MQTT_PAYLOAD_BUFFER_SIZE = 1024;

WiFiClientSecure secureClient;
PubSubClient mqttClient(secureClient);

char mqttHost[128] = "";
uint16_t mqttPort = MQTT_PORT;
unsigned long lastMqttAttemptMs = 0;

uint8_t activeLedMask = 0;
unsigned long blinkIntervalMs = DEFAULT_BLINK_MS;
unsigned long blinkDeadlineMs = 0;
unsigned long lastBlinkToggleMs = 0;
bool blinkPhaseOn = false;

static unsigned long clampUnsignedLong(unsigned long value, unsigned long minValue, unsigned long maxValue) {
  if (value < minValue) {
    return minValue;
  }
  if (value > maxValue) {
    return maxValue;
  }
  return value;
}

static bool timeReached(unsigned long now, unsigned long target) {
  return (long)(now - target) >= 0;
}

static void writeLedMask(uint8_t ledMask) {
  ledMask &= ALL_LED_MASK;
  for (uint8_t index = 0; index < LED_COUNT; index++) {
    const bool on = (ledMask & (1U << index)) != 0;
    digitalWrite(LED_GPIOS[index], on ? HIGH : LOW);
  }
}

static void setAllLeds(bool on) {
  writeLedMask(on ? ALL_LED_MASK : 0);
}

static int levelIndexFromId(const char *levelId) {
  if (levelId == nullptr || strlen(levelId) != 2) {
    return -1;
  }
  if (levelId[0] != 'T' && levelId[0] != 't') {
    return -1;
  }
  if (levelId[1] < '1' || levelId[1] > '4') {
    return -1;
  }
  return levelId[1] - '1';
}

static uint8_t ledMaskFromLevelIndex(int levelIndex) {
  if (levelIndex < 0 || levelIndex >= LED_COUNT) {
    return 0;
  }
  return (uint8_t)(1U << levelIndex);
}

static void printLedMask(uint8_t ledMask) {
  bool printed = false;
  for (uint8_t index = 0; index < LED_COUNT; index++) {
    if ((ledMask & (1U << index)) == 0) {
      continue;
    }
    if (printed) {
      Serial.print(",");
    }
    Serial.print("T");
    Serial.print(index + 1);
    printed = true;
  }
  if (!printed) {
    Serial.print("none");
  }
}

static void ledClear(const char *reason) {
  activeLedMask = 0;
  blinkPhaseOn = false;
  blinkDeadlineMs = 0;
  setAllLeds(false);

  Serial.print("LEDs cleared");
  if (reason != nullptr && reason[0] != '\0') {
    Serial.print(": ");
    Serial.print(reason);
  }
  Serial.println();
}

static void ledStartSignal(
  uint8_t ledMask,
  unsigned long blinkMs,
  unsigned long timeoutMs,
  unsigned long minBlinkMs,
  const char *label
) {
  ledMask &= ALL_LED_MASK;
  if (ledMask == 0) {
    Serial.println("Ignoring LED command with empty LED mask");
    return;
  }

  blinkIntervalMs = clampUnsignedLong(blinkMs, minBlinkMs, 5000);
  timeoutMs = clampUnsignedLong(timeoutMs, 1000, 3600000);

  activeLedMask = ledMask;
  blinkDeadlineMs = millis() + timeoutMs;
  lastBlinkToggleMs = millis();
  blinkPhaseOn = true;

  writeLedMask(activeLedMask);

  Serial.print(label);
  Serial.print(" ");
  printLedMask(activeLedMask);
  Serial.print(" every ");
  Serial.print(blinkIntervalMs);
  Serial.print(" ms for ");
  Serial.print(timeoutMs);
  Serial.println(" ms");
}

static void ledStartBlink(int levelIndex, unsigned long blinkMs, unsigned long timeoutMs) {
  if (levelIndex < 0 || levelIndex >= LED_COUNT) {
    Serial.print("Ignoring invalid level index: ");
    Serial.println(levelIndex);
    return;
  }

  ledStartSignal(ledMaskFromLevelIndex(levelIndex), blinkMs, timeoutMs, 100, "Blinking");
}

static void ledStartMissingAlert(const char *levelId, unsigned long blinkMs, unsigned long timeoutMs) {
  uint8_t ledMask = ALL_LED_MASK;

  if (levelId != nullptr && levelId[0] != '\0') {
    const int levelIndex = levelIndexFromId(levelId);
    if (levelIndex < 0) {
      Serial.print("Invalid alert level_id: ");
      Serial.println(levelId);
      return;
    }
    ledMask = ledMaskFromLevelIndex(levelIndex);
  }

  ledStartSignal(ledMask, blinkMs, timeoutMs, 40, "Missing item alert");
}

static void serviceLed() {
  if (activeLedMask == 0) {
    return;
  }

  const unsigned long now = millis();
  if (timeReached(now, blinkDeadlineMs)) {
    ledClear("timeout");
    return;
  }

  if (!timeReached(now, lastBlinkToggleMs + blinkIntervalMs)) {
    return;
  }

  lastBlinkToggleMs = now;
  blinkPhaseOn = !blinkPhaseOn;
  writeLedMask(blinkPhaseOn ? activeLedMask : 0);
}

static bool parseMqttBroker() {
  String broker = MQTT_BROKER;
  broker.trim();

  if (broker.startsWith("mqtts://")) {
    broker.remove(0, 8);
  } else if (broker.startsWith("mqtt://")) {
    broker.remove(0, 7);
  }

  const int pathIndex = broker.indexOf('/');
  if (pathIndex >= 0) {
    broker = broker.substring(0, pathIndex);
  }

  const int colonIndex = broker.lastIndexOf(':');
  if (colonIndex > 0) {
    const String portPart = broker.substring(colonIndex + 1);
    const int parsedPort = portPart.toInt();
    if (parsedPort > 0 && parsedPort <= 65535) {
      mqttPort = (uint16_t)parsedPort;
    }
    broker = broker.substring(0, colonIndex);
  }

  if (broker.length() == 0 || broker.length() >= sizeof(mqttHost)) {
    return false;
  }

  broker.toCharArray(mqttHost, sizeof(mqttHost));
  return true;
}

static bool isMissingWarningStatus(const char *status) {
  if (status == nullptr) {
    return false;
  }
  return strcmp(status, "suspected_missing_or_merged") == 0 ||
         strcmp(status, "missing_suspected") == 0 ||
         strcmp(status, "MISSING?") == 0;
}

static bool isExplicitAlertCommand(const char *command) {
  if (command == nullptr) {
    return false;
  }
  return strcmp(command, "alert") == 0 ||
         strcmp(command, "warning") == 0 ||
         strcmp(command, "missing_alert") == 0;
}

static bool isMissingAlertMetadata(const char *alertType, const char *reason, const char *status) {
  if (alertType != nullptr && strcmp(alertType, "missing_item") == 0) {
    return true;
  }
  if (reason != nullptr &&
      (strcmp(reason, "missing_item") == 0 || strcmp(reason, "inventory_count_warning") == 0)) {
    return true;
  }
  return isMissingWarningStatus(status);
}

static void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  Serial.print("Connecting to WiFi");
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    delay(WIFI_RETRY_MS);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("WiFi connected, IP: ");
  Serial.println(WiFi.localIP());
}

static void subscribeLedTopic() {
  if (mqttClient.subscribe(MQTT_TOPIC, 1)) {
    Serial.print("Subscribed to ");
    Serial.println(MQTT_TOPIC);
  } else {
    Serial.print("Failed to subscribe to ");
    Serial.println(MQTT_TOPIC);
  }
}

static void connectMqtt() {
  if (mqttClient.connected()) {
    return;
  }

  const unsigned long now = millis();
  if (!timeReached(now, lastMqttAttemptMs + MQTT_RETRY_MS)) {
    return;
  }
  lastMqttAttemptMs = now;

  String clientId = MQTT_CLIENT_ID_PREFIX;
  clientId += String((uint32_t)ESP.getEfuseMac(), HEX);

  Serial.print("Connecting to MQTT ");
  Serial.print(mqttHost);
  Serial.print(":");
  Serial.print(mqttPort);
  Serial.print(" as ");
  Serial.println(clientId);

  if (mqttClient.connect(clientId.c_str(), MQTT_USERNAME, MQTT_PASSWORD)) {
    Serial.println("MQTT connected");
    subscribeLedTopic();
  } else {
    Serial.print("MQTT connect failed, state=");
    Serial.println(mqttClient.state());
  }
}

static void handleLedPayload(const char *payload, unsigned int length) {
  StaticJsonDocument<MQTT_PAYLOAD_BUFFER_SIZE> doc;
  const DeserializationError error = deserializeJson(doc, payload, length);
  if (error) {
    Serial.print("Invalid JSON command: ");
    Serial.println(error.c_str());
    return;
  }

  const char *command = doc["command"] | "";
  const char *eventType = doc["event_type"] | "";

  const char *shelfId = doc["shelf_id"] | "";
  if (shelfId[0] == '\0') {
    shelfId = doc["payload"]["shelf_id"] | "";
  }
  if (shelfId[0] != '\0' && strcmp(shelfId, SHELF_ID) != 0) {
    Serial.print("Ignoring MQTT message for shelf ");
    Serial.println(shelfId);
    return;
  }

  if (strcmp(command, "clear") == 0) {
    const char *reason = doc["reason"] | "mqtt_clear";
    ledClear(reason);
    return;
  }

  const char *levelId = doc["level_id"] | "";
  if (levelId[0] == '\0') {
    levelId = doc["payload"]["level_id"] | "";
  }

  const char *status = doc["status"] | "";
  if (status[0] == '\0') {
    status = doc["payload"]["status"] | "";
  }

  const char *alertType = doc["alert_type"] | "";
  if (alertType[0] == '\0') {
    alertType = doc["payload"]["alert_type"] | "";
  }

  const char *reason = doc["reason"] | "";
  if (reason[0] == '\0') {
    reason = doc["payload"]["reason"] | "";
  }

  long expectedCount = doc["expected_count"] | -1;
  if (expectedCount < 0) {
    expectedCount = doc["payload"]["expected_count"] | -1;
  }

  long detectedCount = doc["detected_count"] | -1;
  if (detectedCount < 0) {
    detectedCount = doc["payload"]["detected_count"] | -1;
  }

  const bool countLooksMissing = expectedCount >= 0 && detectedCount >= 0 && detectedCount < expectedCount;
  const bool isInventoryWarning = strcmp(eventType, "inventory_count_warning") == 0;
  const bool isInventoryStatus = strcmp(eventType, "inventory_status") == 0;
  const bool isMissingEvent =
    (isInventoryWarning && (countLooksMissing || isMissingWarningStatus(status))) ||
    (isInventoryStatus && countLooksMissing);
  const bool isMissingAlert =
    isExplicitAlertCommand(command) ||
    isMissingEvent ||
    (strcmp(command, "blink") == 0 && isMissingAlertMetadata(alertType, reason, status));

  if (isMissingAlert) {
    const unsigned long blinkMs = doc["blink_ms"] | ALERT_BLINK_MS;
    const unsigned long timeoutMs = doc["timeout_ms"] | ALERT_TIMEOUT_MS;
    ledStartMissingAlert(levelId, blinkMs, timeoutMs);
    return;
  }

  if (command[0] == '\0') {
    Serial.print("Ignoring MQTT event without LED action: ");
    Serial.println(eventType[0] != '\0' ? eventType : "(no event_type)");
    return;
  }

  if (strcmp(command, "blink") != 0) {
    Serial.print("Ignoring unknown command: ");
    Serial.println(command);
    return;
  }

  const int levelIndex = levelIndexFromId(levelId);
  if (levelIndex < 0) {
    Serial.print("Invalid level_id: ");
    Serial.println(levelId);
    return;
  }

  const unsigned long blinkMs = doc["blink_ms"] | DEFAULT_BLINK_MS;
  const unsigned long timeoutMs = doc["timeout_ms"] | DEFAULT_TIMEOUT_MS;
  ledStartBlink(levelIndex, blinkMs, timeoutMs);
}

static void mqttCallback(char *topic, byte *payload, unsigned int length) {
  if (length >= MQTT_PAYLOAD_BUFFER_SIZE) {
    Serial.print("Ignoring oversized MQTT payload, bytes=");
    Serial.println(length);
    return;
  }

  char message[MQTT_PAYLOAD_BUFFER_SIZE];
  memcpy(message, payload, length);
  message[length] = '\0';

  Serial.print("MQTT command on ");
  Serial.print(topic);
  Serial.print(": ");
  Serial.println(message);

  handleLedPayload(message, length);
}

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println();
  Serial.println("Warehouse LED Controller starting");

  for (uint8_t index = 0; index < LED_COUNT; index++) {
    pinMode(LED_GPIOS[index], OUTPUT);
  }
  setAllLeds(false);

  if (!parseMqttBroker()) {
    Serial.println("Invalid MQTT_BROKER. Expected mqtts://host:8883");
    while (true) {
      delay(1000);
    }
  }

#if MQTT_USE_INSECURE_TLS
  secureClient.setInsecure();
#else
  secureClient.setCACert(MQTT_CA_CERT);
#endif

  mqttClient.setServer(mqttHost, mqttPort);
  mqttClient.setCallback(mqttCallback);
  mqttClient.setBufferSize(MQTT_PAYLOAD_BUFFER_SIZE);
  mqttClient.setKeepAlive(30);

  connectWiFi();
  lastMqttAttemptMs = millis() - MQTT_RETRY_MS;
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    ledClear("wifi_disconnected");
    connectWiFi();
  }

  connectMqtt();
  mqttClient.loop();
  serviceLed();
}
