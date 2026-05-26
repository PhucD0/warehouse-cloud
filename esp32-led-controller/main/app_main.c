#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON.h"
#include "driver/gpio.h"
#include "esp_crt_bundle.h"
#include "esp_err.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "mqtt_client.h"
#include "nvs_flash.h"

#define WIFI_CONNECTED_BIT BIT0

static const char *TAG = "warehouse_led";

static EventGroupHandle_t s_wifi_event_group;
static SemaphoreHandle_t s_led_mutex;
static esp_mqtt_client_handle_t s_mqtt_client;

static const gpio_num_t s_led_gpios[4] = {
    (gpio_num_t)CONFIG_LED_GPIO_T1,
    (gpio_num_t)CONFIG_LED_GPIO_T2,
    (gpio_num_t)CONFIG_LED_GPIO_T3,
    (gpio_num_t)CONFIG_LED_GPIO_T4,
};

static int s_active_level = -1;
static int s_blink_ms = CONFIG_LED_DEFAULT_BLINK_MS;
static int64_t s_deadline_us = 0;
static bool s_led_phase_on = false;

static int clamp_int(int value, int min_value, int max_value)
{
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static void set_all_leds(bool on)
{
    for (size_t i = 0; i < 4; i++) {
        gpio_set_level(s_led_gpios[i], on ? 1 : 0);
    }
}

static int level_index_from_id(const char *level_id)
{
    if (level_id == NULL || strlen(level_id) != 2) {
        return -1;
    }
    if (level_id[0] != 'T' && level_id[0] != 't') {
        return -1;
    }
    if (level_id[1] < '1' || level_id[1] > '4') {
        return -1;
    }
    return level_id[1] - '1';
}

static void led_start_blink(int level_index, int blink_ms, int timeout_ms)
{
    if (level_index < 0 || level_index >= 4) {
        ESP_LOGW(TAG, "Ignoring invalid level index: %d", level_index);
        return;
    }

    blink_ms = clamp_int(blink_ms, 100, 5000);
    timeout_ms = clamp_int(timeout_ms, 1000, 3600000);

    xSemaphoreTake(s_led_mutex, portMAX_DELAY);
    s_active_level = level_index;
    s_blink_ms = blink_ms;
    s_deadline_us = esp_timer_get_time() + ((int64_t)timeout_ms * 1000);
    s_led_phase_on = false;
    xSemaphoreGive(s_led_mutex);

    set_all_leds(false);
    ESP_LOGI(TAG, "Blinking T%d on GPIO%d every %d ms for %d ms",
             level_index + 1, s_led_gpios[level_index], blink_ms, timeout_ms);
}

static void led_clear(const char *reason)
{
    xSemaphoreTake(s_led_mutex, portMAX_DELAY);
    s_active_level = -1;
    s_led_phase_on = false;
    s_deadline_us = 0;
    xSemaphoreGive(s_led_mutex);

    set_all_leds(false);
    ESP_LOGI(TAG, "LEDs cleared%s%s", reason ? ": " : "", reason ? reason : "");
}

static void led_task(void *arg)
{
    (void)arg;

    while (true) {
        int active_level;
        int blink_ms;
        bool turn_selected_on = false;
        bool expired = false;

        xSemaphoreTake(s_led_mutex, portMAX_DELAY);
        active_level = s_active_level;
        blink_ms = s_blink_ms;

        if (active_level >= 0 && esp_timer_get_time() >= s_deadline_us) {
            s_active_level = -1;
            s_led_phase_on = false;
            active_level = -1;
            expired = true;
        } else if (active_level >= 0) {
            s_led_phase_on = !s_led_phase_on;
            turn_selected_on = s_led_phase_on;
        }
        xSemaphoreGive(s_led_mutex);

        if (expired) {
            ESP_LOGI(TAG, "LED command timed out");
        }

        if (active_level < 0) {
            set_all_leds(false);
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        set_all_leds(false);
        if (turn_selected_on) {
            gpio_set_level(s_led_gpios[active_level], 1);
        }

        vTaskDelay(pdMS_TO_TICKS(blink_ms));
    }
}

static void handle_led_payload(const char *payload)
{
    cJSON *root = cJSON_Parse(payload);
    if (root == NULL) {
        ESP_LOGW(TAG, "Invalid JSON command: %s", payload);
        return;
    }

    const cJSON *command = cJSON_GetObjectItemCaseSensitive(root, "command");
    if (!cJSON_IsString(command) || command->valuestring == NULL) {
        ESP_LOGW(TAG, "Missing command field");
        cJSON_Delete(root);
        return;
    }

    if (strcmp(command->valuestring, "clear") == 0) {
        const cJSON *reason = cJSON_GetObjectItemCaseSensitive(root, "reason");
        led_clear(cJSON_IsString(reason) ? reason->valuestring : "mqtt_clear");
        cJSON_Delete(root);
        return;
    }

    if (strcmp(command->valuestring, "blink") != 0) {
        ESP_LOGW(TAG, "Ignoring unknown command: %s", command->valuestring);
        cJSON_Delete(root);
        return;
    }

    const cJSON *level_id = cJSON_GetObjectItemCaseSensitive(root, "level_id");
    if (!cJSON_IsString(level_id) || level_id->valuestring == NULL) {
        ESP_LOGW(TAG, "Blink command missing level_id");
        cJSON_Delete(root);
        return;
    }

    const int level_index = level_index_from_id(level_id->valuestring);
    if (level_index < 0) {
        ESP_LOGW(TAG, "Invalid level_id: %s", level_id->valuestring);
        cJSON_Delete(root);
        return;
    }

    const cJSON *blink_ms_json = cJSON_GetObjectItemCaseSensitive(root, "blink_ms");
    const cJSON *timeout_ms_json = cJSON_GetObjectItemCaseSensitive(root, "timeout_ms");
    const int blink_ms = cJSON_IsNumber(blink_ms_json) ? blink_ms_json->valueint : CONFIG_LED_DEFAULT_BLINK_MS;
    const int timeout_ms = cJSON_IsNumber(timeout_ms_json) ? timeout_ms_json->valueint : CONFIG_LED_DEFAULT_TIMEOUT_MS;

    led_start_blink(level_index, blink_ms, timeout_ms);
    cJSON_Delete(root);
}

static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data)
{
    (void)handler_args;
    (void)base;

    esp_mqtt_event_handle_t event = event_data;

    switch ((esp_mqtt_event_id_t)event_id) {
    case MQTT_EVENT_CONNECTED:
        ESP_LOGI(TAG, "MQTT connected");
        esp_mqtt_client_subscribe(event->client, CONFIG_LED_TOPIC, 1);
        break;

    case MQTT_EVENT_DISCONNECTED:
        ESP_LOGW(TAG, "MQTT disconnected");
        break;

    case MQTT_EVENT_DATA:
        if (event->current_data_offset != 0 || event->data_len != event->total_data_len) {
            ESP_LOGW(TAG, "Ignoring fragmented MQTT payload");
            break;
        }

        char *payload = calloc(1, event->data_len + 1);
        if (payload == NULL) {
            ESP_LOGE(TAG, "Out of memory while copying MQTT payload");
            break;
        }
        memcpy(payload, event->data, event->data_len);
        ESP_LOGI(TAG, "MQTT command on %.*s: %s", event->topic_len, event->topic, payload);
        handle_led_payload(payload);
        free(payload);
        break;

    case MQTT_EVENT_ERROR:
        ESP_LOGE(TAG, "MQTT transport error");
        break;

    default:
        break;
    }
}

static void mqtt_start(void)
{
    const esp_mqtt_client_config_t mqtt_cfg = {
        .broker = {
            .address.uri = CONFIG_LED_MQTT_BROKER_URI,
            .verification.crt_bundle_attach = esp_crt_bundle_attach,
        },
        .credentials = {
            .username = CONFIG_LED_MQTT_USERNAME,
            .authentication.password = CONFIG_LED_MQTT_PASSWORD,
        },
        .session.keepalive = 30,
    };

    s_mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    ESP_ERROR_CHECK(esp_mqtt_client_register_event(s_mqtt_client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL));
    ESP_ERROR_CHECK(esp_mqtt_client_start(s_mqtt_client));
}

static void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    (void)arg;
    (void)event_data;

    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
        ESP_LOGW(TAG, "WiFi disconnected, reconnecting");
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "WiFi got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

static void wifi_start(void)
{
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event_handler, NULL, NULL));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = CONFIG_LED_WIFI_SSID,
            .password = CONFIG_LED_WIFI_PASSWORD,
            .threshold.authmode = strlen(CONFIG_LED_WIFI_PASSWORD) == 0 ? WIFI_AUTH_OPEN : WIFI_AUTH_WPA2_PSK,
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Connecting to WiFi SSID: %s", CONFIG_LED_WIFI_SSID);
    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdFALSE, portMAX_DELAY);
}

static void configure_led_gpio(void)
{
    uint64_t pin_mask = 0;
    for (size_t i = 0; i < 4; i++) {
        pin_mask |= (1ULL << s_led_gpios[i]);
    }

    gpio_config_t io_conf = {
        .pin_bit_mask = pin_mask,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&io_conf));
    set_all_leds(false);
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    s_led_mutex = xSemaphoreCreateMutex();
    if (s_led_mutex == NULL) {
        ESP_LOGE(TAG, "Failed to create LED mutex");
        return;
    }

    configure_led_gpio();
    xTaskCreate(led_task, "led_task", 3072, NULL, 5, NULL);

    wifi_start();
    mqtt_start();
}
