# YOLOv5 on ESP32 with TensorFlow Lite Micro

## Focus of Work

- Deploy a YOLOv5 model on an ESP32 microcontroller
- Use TensorFlow Lite Micro for on-device inference
- Manage storage using LittleFS or SPIFFS for model files
- Integrate camera input (OV2640 or compatible)
- Optimize model using INT8 post-training quantization
- Provide a modular and extensible C++ codebase for embedded object detection

---

## Folder Structure
tflite_esp32/
├── include/
│ ├── tflite_inference.h # Header for TFLite inference
│ └── camera_interface.h # Header for camera interface
├── lib/
│ └── tfmicro/ # TensorFlow Lite Micro library
├── src/
│ ├── main.cpp # Main application
│ ├── tflite_inference.cpp # TFLite inference logic
│ └── camera_interface.cpp # Camera setup and capture
├── models/
│ ├── yolov5s_int8.tflite # Quantized model
│ └── labelmap.txt
├── platformio.ini
└── README.md
## Setup Instructions

#### 1. Clone Repository
```bash
git clone https://github.com/neffatihiba/tflite_esp32.git
cd tflite_esp32
```
#### 2. Install Dependencies
```bash
pip install tensorflow ultralytics numpy
```
#### 3. Convert YOLOv5 Model to TensorFlow Lite
# Clone YOLOv5 repository
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
```
# Export YOLOv5 model to TFLite
```bash
python export.py --weights yolov5s.pt --include tflite
# Copy the TFLite model to the project folder
cp yolov5s.tflite ../tflite_esp32/models/yolov5s_int8.tflite
```
#### 4. Quantize the Model for ESP32 
# Python snippet for INT8 post-training quantization
```bash
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model('path_to_saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('../tflite_esp32/models/yolov5s_int8.tflite', 'wb') as f:
 f.write(tflite_model)
```

#### 5. Flash the Code to ESP32
```bash
pio run --target upload
```
#### 6. Open Serial Monitor
```bash
pio device monitor --baud 115200
```
