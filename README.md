# 💧SOLFARM - An AI-Powered Smart Irrigation System (SIH 2025)

## An ESP32-based automated irrigation controller leveraging TensorFlow Lite for Microcontrollers to optimize water usage through predictive machine learning.
## Developed for the Smart India Hackathon (SIH) 2025, this project transitions traditional threshold-based watering systems into an intelligent, data-driven architecture. By analyzing real-time soil moisture, historical trends, and time of day, the embedded TinyML model makes localized watering decisions directly on the edge.
## 🚀 Core Features
### Edge AI (TinyML) Engine: Runs a TensorFlow Lite model directly on the ESP32. It analyzes a 7-feature input array (current moisture, rolling averages, moisture trends, and time of day) to predict optimal watering times rather than relying on static thresholds.

Dual-Mode Operation: Features seamless toggling between the ML-driven mode and a traditional threshold-based fallback mode.

Multi-Zone Control: Independently monitors and controls two separate irrigation zones (Valves 1 & 2) while sharing a single centralized water pump.

Fail-Safe Mechanisms: Implements a hardware timeout (MAX_WATER_MILLIS) to prevent overwatering or flooding in the event of a sensor failure or logical error.

Standalone Web Server: Operates as a Wi-Fi Access Point (ESP32_Irrigation), hosting a local web dashboard to monitor zone statuses, view ML confidence scores, and manually override valves.

## 🛠 Hardware Configuration
### Component,ESP32 Pin,Note
### Moisture Sensor 1,GPIO 34,Analog input (Zone 1)
### Moisture Sensor 2,GPIO 35,Analog input (Zone 2)
### Valve 1 Relay,GPIO 25,Digital output
### Valve 2 Relay,GPIO 27,Digital output
### Water Pump Relay,GPIO 26,Digital output (Activates if any valve is open)
Note: Analog values are calibrated using raw dry (4095, 3388) and wet (1298, 1281) readings to normalize moisture into a 0-100% scale.

## 🧠 Machine Learning Architecture

The system uses a pre-trained neural network converted to a C-byte array (irrigation_model.h). The TFLite Micro interpreter allocates an 8KB tensor arena to process the following feature vector every 3 seconds:

Current Moisture (Zone 1): Normalized 0.0 - 1.0

Current Moisture (Zone 2): Normalized 0.0 - 1.0

Estimated Hour: Normalized time of day (0.0 - 1.0)

Moisture Average (Zone 1): 10-reading rolling average

Moisture Trend (Zone 1): Delta between recent and older readings

Moisture Average (Zone 2): 10-reading rolling average

Moisture Trend (Zone 2): Delta between recent and older readings

Output: The model returns two probabilities (prob_no_water, prob_water). If prob_water > 0.5, the system triggers the respective zone's irrigation cycle.

## 💻 Software Dependencies
### To compile and upload this code via the Arduino IDE or PlatformIO, ensure you have the following libraries installed:

WiFi.h & WebServer.h (Built-in with ESP32 core)

TensorFlowLite_ESP32 (Provides the tflite::MicroInterpreter and ops resolvers)

## ⚙️ Setup & Installation

Clone the Repository: Download the source code, ensuring irrigation_model.h is in the same directory as the main .ino or .cpp file.

Calibrate Sensors: Submerge your specific capacitive soil sensors in water and expose them to dry air. Update the DRY_RAW and WET_RAW variables in the code with your specific ADC readings.

Flash the ESP32: Compile and upload the code using a baud rate of 115200.

Connect: Connect a phone or laptop to the Wi-Fi network ESP32_Irrigation (Password: 12345678).

Access Dashboard: Open a web browser and navigate to the IP address printed in the serial monitor (typically 192.168.4.1) to view the control panel.
