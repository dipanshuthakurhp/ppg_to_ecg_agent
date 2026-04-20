#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model + input
#include "model_data.h"
#include "test_input.h"

const int kArenaSize = 200 * 1024;
uint8_t tensor_arena[kArenaSize];

extern "C" void app_main(void) {

    tflite::InitializeTarget();

    printf("\n=== ESP32 TFLite Micro (INT8) ===\n");

    // Load model
    const tflite::Model* model = tflite::GetModel(model_tflite);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema mismatch!\n");
        return;
    }

    // ✅ Use MicroMutableOpResolver (optimized)
    tflite::MicroMutableOpResolver<12> resolver;

    resolver.AddConv2D();
    resolver.AddReshape();
    resolver.AddAdd();
    resolver.AddExpandDims();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddTanh();

    resolver.AddMaxPool2D();
    resolver.AddTransposeConv();
    resolver.AddShape();
    resolver.AddStridedSlice();
    resolver.AddConcatenation();

    // Interpreter
    tflite::MicroInterpreter interpreter(
        model,
        resolver,
        tensor_arena,
        kArenaSize
    );

    // Allocate tensors
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("Tensor allocation failed!\n");
        return;
    }

    // Input tensor
    TfLiteTensor* input = interpreter.input(0);

    printf("Input bytes: %d\n", input->bytes);

    // Copy INT8 input
    int input_size = input->bytes / sizeof(int8_t);

    for (int i = 0; i < input_size; i++) {
        input->data.int8[i] = test_input[i];
    }

    // Run inference
    if (interpreter.Invoke() != kTfLiteOk) {
        printf("Inference failed!\n");
        return;
    }

    // Output tensor
    TfLiteTensor* output = interpreter.output(0);

    printf("\n=== Output (INT8 raw) ===\n");

    int output_size = output->bytes / sizeof(int8_t);

    for (int i = 0; i < output_size; i++) {
        printf("%d ", output->data.int8[i]);
    }

    printf("\n");

    printf("\n=== Done ===\n");
}