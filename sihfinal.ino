#include <WiFi.h>
#include <WebServer.h>

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "irrigation_model.h"

const int TENSOR_ARENA_SIZE = 8 * 1024;  
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// History buffer for trend calculation
const int HISTORY_SIZE = 10;
float moisture1_history[HISTORY_SIZE];
float moisture2_history[HISTORY_SIZE];
int history_index = 0;
bool history_full = false;

bool mlReady = false;

bool AUTO_MODE_DEFAULT = true;
int THRESHOLD_PERCENT = 40;
unsigned long MAX_WATER_MILLIS = 30UL * 1000UL;

const int MOIST_PIN1 = 34;
const int MOIST_PIN2 = 35;
const int VALVE1_PIN = 25;
const int VALVE2_PIN = 27;
const int PUMP_PIN   = 26;

int DRY_RAW_1 = 4095, WET_RAW_1 = 1298;
int DRY_RAW_2 = 3388, WET_RAW_2 = 1281;

WebServer server(80);
unsigned long lastSensorMillis = 0;
const unsigned long SENSOR_INTERVAL = 3000UL;

int raw1 = 0, raw2 = 0;
int pct1 = 0, pct2 = 0;
bool valve1State = false, valve2State = false, pumpState = false;
bool autoMode = AUTO_MODE_DEFAULT;
bool useML = true;  // NEW: toggle between ML and simple threshold
unsigned long valveOpenSince[2] = {0, 0};

float mlConfidence = 0.0;  // how confident the model is

int rawToPercent(int raw, int dryVal, int wetVal) {
    if (dryVal == wetVal) return 0;
    long v = map(raw, dryVal, wetVal, 0, 100);
    if (v < 0) v = 0;
    if (v > 100) v = 100;
    return (int)v;
}

void setValve(int idx, bool on) {
    if (idx == 1) {
        digitalWrite(VALVE1_PIN, on ? HIGH : LOW);
        valve1State = on;
        valveOpenSince[0] = on ? millis() : 0;
    } else {
        digitalWrite(VALVE2_PIN, on ? HIGH : LOW);
        valve2State = on;
        valveOpenSince[1] = on ? millis() : 0;
    }
    bool shouldPump = valve1State || valve2State;
    if (pumpState != shouldPump) {
        digitalWrite(PUMP_PIN, shouldPump ? HIGH : LOW);
        pumpState = shouldPump;
    }
}

void setPump(bool on) {
    digitalWrite(PUMP_PIN, on ? HIGH : LOW);
    pumpState = on;
}


void setupML() {
    // Load the model
    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ML: Model version mismatch!");
        mlReady = false;
        return;
    }

    // Create interpreter
    interpreter = new tflite::MicroInterpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE
    );

    // Allocate memory for model's tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ML: Failed to allocate tensors!");
        mlReady = false;
        return;
    }

    // Get pointers to input/output
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Initialize history
    for (int i = 0; i < HISTORY_SIZE; i++) {
        moisture1_history[i] = 50.0;
        moisture2_history[i] = 50.0;
    }

    mlReady = true;
    Serial.println("ML: Model loaded successfully!");
    Serial.print("ML: Input size: ");
    Serial.println(input->bytes);
    Serial.print("ML: Arena used: ");
    Serial.println(interpreter->arena_used_bytes());
}

void updateHistory(float m1, float m2) {
    moisture1_history[history_index] = m1;
    moisture2_history[history_index] = m2;
    history_index = (history_index + 1) % HISTORY_SIZE;
    if (history_index == 0) history_full = true;
}

float calculateAverage(float* arr, int size) {
    float sum = 0;
    int count = history_full ? size : history_index;
    if (count == 0) return 0;
    for (int i = 0; i < count; i++) sum += arr[i];
    return sum / count;
}

float calculateTrend(float* arr, int size) {
    // Simple: difference between recent and older readings
    int count = history_full ? size : history_index;
    if (count < 2) return 0;
    float recent = arr[(history_index - 1 + size) % size];
    float older = arr[(history_index - count / 2 + size) % size];
    return recent - older;  // positive = getting wetter, negative = drying
}

// Returns true if model says "water this zone"
bool mlShouldWater(int zone) {
    if (!mlReady) return false;

    // Get current hour estimate (rough, since ESP32 has no RTC)
    int hour = (millis() / 3600000) % 24;

    // Prepare input features (must match training!)
    // [moisture1, moisture2, hour, avg1, trend1, avg2, trend2]
    input->data.f[0] = (float)pct1 / 100.0;
    input->data.f[1] = (float)pct2 / 100.0;
    input->data.f[2] = (float)hour / 24.0;
    input->data.f[3] = calculateAverage(moisture1_history, HISTORY_SIZE) / 100.0;
    input->data.f[4] = calculateTrend(moisture1_history, HISTORY_SIZE) / 100.0;
    input->data.f[5] = calculateAverage(moisture2_history, HISTORY_SIZE) / 100.0;
    input->data.f[6] = calculateTrend(moisture2_history, HISTORY_SIZE) / 100.0;

    // Run the model
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("ML: Inference failed!");
        return false;
    }

    float prob_no_water = output->data.f[0];
    float prob_water = output->data.f[1];

    mlConfidence = max(prob_no_water, prob_water);

    Serial.print("ML prediction (zone ");
    Serial.print(zone);
    Serial.print("): water=");
    Serial.print(prob_water);
    Serial.print(" confidence=");
    Serial.println(mlConfidence);

    return prob_water > 0.5;  
}

void setup() {
    Serial.begin(115200);
    delay(100);

    pinMode(VALVE1_PIN, OUTPUT);
    pinMode(VALVE2_PIN, OUTPUT);
    pinMode(PUMP_PIN, OUTPUT);
    digitalWrite(VALVE1_PIN, LOW);
    digitalWrite(VALVE2_PIN, LOW);
    digitalWrite(PUMP_PIN, LOW);

    // Initialize ML model
    setupML();

    WiFi.mode(WIFI_AP);
    WiFi.softAP("ESP32_Irrigation", "12345678");
    Serial.print("AP IP: ");
    Serial.println(WiFi.softAPIP());

    server.on("/", HTTP_GET, handleRoot);
    server.on("/status", HTTP_GET, handleStatus);
    server.on("/control", HTTP_GET, handleControl);
    server.begin();
}

void loop() {
    server.handleClient();
    unsigned long now = millis();

    if (now - lastSensorMillis > SENSOR_INTERVAL) {
        lastSensorMillis = now;

        // Read sensors
        raw1 = analogRead(MOIST_PIN1);
        raw2 = analogRead(MOIST_PIN2);
        pct1 = rawToPercent(raw1, DRY_RAW_1, WET_RAW_1);
        pct2 = rawToPercent(raw2, DRY_RAW_2, WET_RAW_2);

        // Update history buffer for ML
        updateHistory((float)pct1, (float)pct2);

        Serial.print("Moisture: ");
        Serial.print(pct1);
        Serial.print("% / ");
        Serial.print(pct2);
        Serial.println("%");

        if (autoMode) {
            if (useML && mlReady) {
                //  ML-BASED DECISION 
                bool shouldWater1 = mlShouldWater(1);
                bool shouldWater2 = mlShouldWater(2);

                if (!valve1State && shouldWater1)
                    setValve(1, true);
                if (valve1State && (!shouldWater1 || 
                    millis() - valveOpenSince[0] > MAX_WATER_MILLIS))
                    setValve(1, false);

                if (!valve2State && shouldWater2)
                    setValve(2, true);
                if (valve2State && (!shouldWater2 || 
                    millis() - valveOpenSince[1] > MAX_WATER_MILLIS))
                    setValve(2, false);

            } else {
                //  SIMPLE THRESHOLD (fallback) 
                if (!valve1State && pct1 < THRESHOLD_PERCENT)
                    setValve(1, true);
                if (valve1State && (pct1 >= THRESHOLD_PERCENT || 
                    millis() - valveOpenSince[0] > MAX_WATER_MILLIS))
                    setValve(1, false);

                if (!valve2State && pct2 < THRESHOLD_PERCENT)
                    setValve(2, true);
                if (valve2State && (pct2 >= THRESHOLD_PERCENT || 
                    millis() - valveOpenSince[1] > MAX_WATER_MILLIS))
                    setValve(2, false);
            }
        }
    }
}