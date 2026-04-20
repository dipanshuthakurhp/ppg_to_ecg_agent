from __future__ import annotations
import subprocess
from pathlib import Path
import re

def get_main_dir(esp_project: str) -> str:
    esp_project = Path(esp_project)
    main_dir = esp_project / "main"
    if not main_dir.exists():
        raise FileNotFoundError(f"ESP-IDF project main/ not found at: {main_dir}")
    return str(main_dir)

def generate_model_c(tflite_path: str, out_c_path: str) -> str:
    tflite_path = Path(tflite_path)
    out = Path(out_c_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        xxd_txt = subprocess.check_output(["xxd", "-i", str(tflite_path)], text=True)
        lines = xxd_txt.splitlines()
        if lines:
            m = re.search(r"unsigned char\s+([a-zA-Z0-9_]+)\s*\[\]", lines[0])
            if m:
                base = m.group(1)
                xxd_txt = xxd_txt.replace(f"unsigned char {base}[]", "const unsigned char model_tflite[]")
                xxd_txt = xxd_txt.replace(f"unsigned int {base}_len", "const unsigned int model_tflite_len")
        xxd_txt = xxd_txt.replace("unsigned char", "const unsigned char", 1) if "const unsigned char" not in xxd_txt else xxd_txt
        out.write_text(xxd_txt, encoding="utf-8")
        return str(out)
    except Exception:
        data = tflite_path.read_bytes()
        with open(out, "w", encoding="utf-8") as f:
            f.write("#include <stdint.h>\n\n")
            f.write("const unsigned char model_tflite[] = {")
            for i, b in enumerate(data):
                if i % 12 == 0:
                    f.write("\n  ")
                f.write(str(b) + ", ")
            f.write("\n};\n")
            f.write(f"const unsigned int model_tflite_len = {len(data)};\n")
        return str(out)

def ensure_main_includes_generated_files(main_c_path: str) -> None:
    p = Path(main_c_path)
    if not p.exists():
        return
    txt = p.read_text(encoding="utf-8", errors="ignore")
    marker = "// --- PhysioFusion Agent includes ---"
    if marker in txt:
        return
    patch = "\n\n" + marker + "\n" +             "// For quick compile-time inclusion (first test), you can add:\n" +             "//   #include \"model_data.c\"\n" +             "//   #include \"test_input.c\"\n"
    p.write_text(txt + patch, encoding="utf-8")

def idf_build(esp_project: str) -> None:
    esp_project = str(Path(esp_project).resolve())
    subprocess.run(["idf.py", "build"], cwd=esp_project, check=True)

def idf_flash_monitor(esp_project: str, port: str) -> None:
    esp_project = str(Path(esp_project).resolve())
    subprocess.run(["idf.py", "-p", port, "flash"], cwd=esp_project, check=True)
    subprocess.run(["idf.py", "-p", port, "monitor"], cwd=esp_project, check=True)

def generate_inference_main_c(main_dir: str) -> str:
    """Generate a template main.c with TFLite Micro inference code"""
    main_c_code = r'''
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

    // Op resolver
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

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("Tensor allocation failed!\n");
        return;
    }

    TfLiteTensor* input = interpreter.input(0);

    printf("Input bytes: %d\n", input->bytes);

    int input_size = input->bytes / sizeof(int8_t);

    for (int i = 0; i < input_size; i++) {
        input->data.int8[i] = test_input[i];
    }

    if (interpreter.Invoke() != kTfLiteOk) {
        printf("Inference failed!\n");
        return;
    }

    TfLiteTensor* output = interpreter.output(0);

    printf("\n=== Output (INT8 raw) ===\n");

    int output_size = output->bytes / sizeof(int8_t);

    for (int i = 0; i < output_size; i++) {
        printf("%d ", output->data.int8[i]);
    }

    printf("\n=== Done ===\n");
}
'''
    
    main_path = Path(main_dir) / "main.cpp"
    main_path.write_text(main_c_code, encoding="utf-8")
    return str(main_path)

