import tensorflow as tf


model = tf.keras.models.load_model("models\saved_keras\model.h5", compile=False)
model.summary()

import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="D:\physiofusion_agent_v2\models\\tflite\\best_int8.tflite")
interpreter.allocate_tensors()

ops = interpreter._get_ops_details()

for i, op in enumerate(ops):
    print(i, op['op_name'])

