#include <TensorFlowLite_ESP32.h>
#include <FS.h>
#include <SPIFFS.h>
#include <SPI.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "yolov5s.h"

const char* class_names[] = {
    "person", "bicycle", "car", 
};

const int kInputImageSize = 640;
const int kArenaSize = 20000;

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model;
static tflite::MicroMutableOpResolver<10> resolver;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;
uint8_t tensor_arena[kArenaSize];
uint8_t input_image[kInputImageSize * kInputImageSize * 3];

bool loadImage(const char* filename, uint8_t* input_image) {
    File file = SPIFFS.open(filename, FILE_READ);
    if (!file) {
        Serial.println("Failed to open image file");
        return false;
    }
    size_t bytesRead = file.read(input_image, kInputImageSize * kInputImageSize * 3);
    file.close();
    
    if (bytesRead != kInputImageSize * kInputImageSize * 3) {
        Serial.println("Failed to read the complete image file");
        return false;
    }
    
    return true;
}

void printResults(const std::vector<float>& scores, const std::vector<int>& classes) {
    for (size_t i = 0; i < scores.size(); ++i) {
        if (classes[i] < sizeof(class_names) / sizeof(class_names[0])) {
            Serial.printf("Detected object: %s with confidence: %.6f\n", class_names[classes[i]], scores[i]);
        } else {
            Serial.printf("Detected object with unknown class ID %d and confidence: %.6f\n", classes[i], scores[i]);
        }
    }
}

void setup() {
    Serial.begin(115200);
    if (!SPIFFS.begin(true)) {
        Serial.println("An Error has occurred while mounting SPIFFS");
        return;
    }

    model = tflite::GetModel(yolov5s_fp16_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report("Model schema version %d not supported", model->version());
        return;
    }

    // Register the operators needed for the model
    resolver.AddFullyConnected();
    resolver.AddMul();
    resolver.AddAdd();
    resolver.AddLogistic();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();

    interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kArenaSize, error_reporter);
    interpreter->AllocateTensors();

    input = interpreter->input(0);
    output = interpreter->output(0);

    if (!loadImage("/img2.png", input_image)) {
        error_reporter->Report("Failed to load image.");
        return;
    }
}

void loop() {
    memcpy(input->data.uint8, input_image, kInputImageSize * kInputImageSize * 3);

    if (interpreter->Invoke() != kTfLiteOk) {
        error_reporter->Report("Invoke failed.");
        return;
    }

    std::vector<float> scores;
    std::vector<int> classes;

    for (int i = 0; i < output->dims->data[1]; ++i) {
        float score = output->data.f[i * 6 + 2];
        if (score > 0.5) {
            int class_id = static_cast<int>(output->data.f[i * 6 + 1]);
            scores.push_back(score);
            classes.push_back(class_id);
        }
    }

    printResults(scores, classes);

    delay(10000); 
}
