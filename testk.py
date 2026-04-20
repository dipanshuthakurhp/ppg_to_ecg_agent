import typer
from pathlib import Path
from datetime import datetime
import json
import tensorflow as tf
from src import train_agent, quantize, export_esp32, make_test_input

model_path = "D:\PPG to ECG\\4. Project file and model\Model.h5"
tflite_path = quantize.to_int8(model_path)
print("Quantized model:", tflite_path)

main_dir = export_esp32.get_main_dir("D:\esp32_firmware")

model_c = export_esp32.generate_model_c(tflite_path, str(Path(main_dir) / "model_data.c"))
print("model_data.c written:", model_c)

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

for op in interpreter._get_ops_details():
    print(op['op_name'])