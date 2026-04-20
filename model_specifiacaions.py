import tensorflow as tf
from src import train_agent, quantize, export_esp32, make_test_input
from pathlib import Path
model = tf.keras.models.load_model("D:\physiofusion_agent_v2\models\CNN_baseline.h5", compile=False)
model.summary()

import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="D:\physiofusion_agent_v2\models\\tflite\\best_int8.tflite")
interpreter.allocate_tensors()

ops = interpreter._get_ops_details()

for i, op in enumerate(ops):
    print(i, op['op_name'])

