
#include <TensorFlowLite_ESP32.h>
#include <FS.h>
#include <SPIFFS.h>
#include <SD.h>
#include <SPI.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "yolov5s.h" 

const char* class_names[] = {
    "person", "bicycle", "car", 
};

const int kInputImageSize = 640;
const int chipSelect = 5; 

bool loadImage(const char* filename, uint8_t* input_image) {
    File file = SPIFFS.open(filename, FILE_READ);
    if (!file) {
        Serial.println("Failed to open image file");
        return false;
    }
    file.read(input_image, kInputImageSize * kInputImageSize * 3);
    file.close();
    return true;
}

bool writeResultsToSD(const char* filename, const std::vector<float>& scores, const std::vector<int>& classes, const std::vector<std::array<float, 4>>& bbs) {
    File file = SD.open(filename, FILE_WRITE);
    if (!file) {
        Serial.println("Failed to open results file on SD card");
        return false;
    }

    for (size_t i = 0; i < scores.size(); ++i) {
        file.printf("Class: %s, Score: %.2f, BBox: [%.2f, %.2f, %.2f, %.2f]\n",
                    class_names[classes[i]], scores[i], bbs[i][0], bbs[i][1], bbs[i][2], bbs[i][3]);
    }
    file.close();
    return true;
}

// TensorFlow Lite objects and variables
static tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model;
static tflite::MicroMutableOpResolver<10> micro_op_resolver;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;
const int tensor_arena_size = 60 * 1024;
uint8_t tensor_arena[tensor_arena_size];
uint8_t input_image[kInputImageSize * kInputImageSize * 3];

void setup() {
    Serial.begin(115200);
    if (!SPIFFS.begin(true)) {
        Serial.println("An Error has occurred while mounting SPIFFS");
        return;
    }

    if (!SD.begin(chipSelect)) {
        Serial.println("Initialization failed!");
        return;
    }

    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report("Model schema version %d not supported", model->version());
        return;
    }

    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D, tflite::ops::micro::Register_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_DEPTHWISE_CONV_2D, tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED, tflite::ops::micro::Register_FULLY_CONNECTED());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX, tflite::ops::micro::Register_SOFTMAX());

    interpreter = new tflite::MicroInterpreter(model, micro_op_resolver, tensor_arena, tensor_arena_size, error_reporter);
    interpreter->AllocateTensors();

    input = interpreter->input(0);
    output = interpreter->output(0);

    if (!loadImage("/image.jpg", input_image)) {
        error_reporter->Report("Failed to load image.");
        return;
    }
}

void loop() {
    for (int i = 0; i < kInputImageSize * kInputImageSize * 3; ++i) {
        input->data.uint8[i] = input_image[i];
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        error_reporter->Report("Invoke failed.");
        return;
    }

    std::vector<float> scores;
    std::vector<int> classes;
    std::vector<std::array<float, 4>> bbs;

    for (int i = 0; i < output->dims->data[1]; ++i) {
        float score = output->data.f[i * 6 + 2];
        if (score > 0.5) {
            int class_id = static_cast<int>(output->data.f[i * 6 + 1]);
            float x_min = output->data.f[i * 6 + 3];
            float y_min = output->data.f[i * 6 + 4];
            float x_max = output->data.f[i * 6 + 5];
            float y_max = output->data.f[i * 6 + 6];
            scores.push_back(score);
            classes.push_back(class_id);
            bbs.push_back({x_min, y_min, x_max, y_max});
        }
    }

    if (!writeResultsToSD("/results.txt", scores, classes, bbs)) {
        error_reporter->Report("Failed to write results to SD card.");
        return;
    }

    delay(10000); 
}
