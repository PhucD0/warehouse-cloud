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

static const unsigned long DEFAULT_BLINK_MS = 500;
static const unsigned long DEFAULT_TIMEOUT_MS = 120000;
static const unsigned long WIFI_RETRY_MS = 500;
static const unsigned long MQTT_RETRY_MS = 5000;
static const size_t MQTT_PAYLOAD_BUFFER_SIZE = 1024;

WiFiClientSecure secureClient;
PubSubClient mqttClient(secureClient);

char mqttHost[128] = "";
uint16_t mqttPort = MQTT_PORT;
unsigned long lastMqttAttemptMs = 0;

int activeLevel = -1;
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

static void setAllLeds(bool on) {
  for (uint8_t index = 0; index < 4; index++) {
    digitalWrite(LED_GPIOS[index], on ? HIGH : LOW);
  }
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

static void ledClear(const char *reason) {
  activeLevel = -1;
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

static void ledStartBlink(int levelIndex, unsigned long blinkMs, unsigned long timeoutMs) {
  if (levelIndex < 0 || levelIndex >= 4) {
    Serial.print("Ignoring invalid level index: ");
    Serial.println(levelIndex);
    return;
  }

  blinkIntervalMs = clampUnsignedLong(blinkMs, 100, 5000);
  timeoutMs = clampUnsignedLong(timeoutMs, 1000, 3600000);

  activeLevel = levelIndex;
  blinkDeadlineMs = millis() + timeoutMs;
  lastBlinkToggleMs = millis();
  blinkPhaseOn = true;

  setAllLeds(false);
  digitalWrite(LED_GPIOS[activeLevel], HIGH);

  Serial.print("Blinking T");
  Serial.print(levelIndex + 1);
  Serial.print(" on GPIO");
  Serial.print(LED_GPIOS[levelIndex]);
  Serial.print(" every ");
  Serial.print(blinkIntervalMs);
  Serial.print(" ms for ");
  Serial.print(timeoutMs);
  Serial.println(" ms");
}

static void serviceLed() {
  if (activeLevel < 0) {
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
  setAllLeds(false);
  if (blinkPhaseOn) {
    digitalWrite(LED_GPIOS[activeLevel], HIGH);
  }
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
  if (command[0] == '\0') {
    Serial.println("Missing command field");
    return;
  }

  const char *shelfId = doc["shelf_id"] | "";
  if (shelfId[0] != '\0' && strcmp(shelfId, SHELF_ID) != 0) {
    Serial.print("Ignoring command for shelf ");
    Serial.println(shelfId);
    return;
  }

  if (strcmp(command, "clear") == 0) {
    const char *reason = doc["reason"] | "mqtt_clear";
    ledClear(reason);
    return;
  }

  if (strcmp(command, "blink") != 0) {
    Serial.print("Ignoring unknown command: ");
    Serial.println(command);
    return;
  }

  const char *levelId = doc["level_id"] | "";
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

  for (uint8_t index = 0; index < 4; index++) {
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
