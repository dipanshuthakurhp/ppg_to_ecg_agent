import typer
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from src import train_agent, quantize, export_esp32, make_test_input
 
model_path = train_agent.run_training("unet")
tflite_path = quantize.to_int8(model_path)


interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

for op in interpreter._get_ops_details():
    print(op['op_name'])