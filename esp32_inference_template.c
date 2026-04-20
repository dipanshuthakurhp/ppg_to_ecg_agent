#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

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

resolver.AddMaxPool2D();        // ✅ required
resolver.AddTransposeConv();    // ✅ required (Conv1DTranspose → this)
resolver.AddShape();            // ✅ required
resolver.AddStridedSlice();     // ✅ required
resolver.AddConcatenation();    // ✅ required

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

    // ✅ Copy INT8 input
    int input_size = input->bytes / sizeof(int8_t);

    for (int i = 0; i < input_size; i++) {
        input->data.int8[i] = test_input[i];
    }

    // Timing
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Run inference
    if (interpreter.Invoke() != kTfLiteOk) {
        printf("Inference failed!\n");
        return;
    }

    gettimeofday(&end, NULL);

    long time_us = (end.tv_sec - start.tv_sec) * 1000000 +
                   (end.tv_usec - start.tv_usec);

    printf("Inference time: %ld us\n", time_us);

    // Output tensor
    TfLiteTensor* output = interpreter.output(0);

    printf("\n=== Output (INT8 raw) ===\n");

    int output_size = output->bytes / sizeof(int8_t);

    for (int i = 0; i < output_size; i++) {
        printf("%d ", output->data.int8[i]);
    }

    printf("\n");

    // ✅ Dequantized output (IMPORTANT)
    // printf("\n=== Output (Dequantized) ===\n");

    // for (int i = 0; i < output_size; i++) {
    //     float value = (output->data.int8[i] - output->params.zero_point) *
    //                   output->params.scale;
    //     printf("%f ", value);
    // }

    printf("\n=== Done ===\n");
}