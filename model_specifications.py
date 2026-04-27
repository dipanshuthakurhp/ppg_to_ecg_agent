import tensorflow as tf
import os

print("===  H5 Model   ===")
model_path = "models\saved_keras\model.h5"
model = tf.keras.models.load_model(model_path, compile=False)
model.summary()
file_size_bytes = os.path.getsize(model_path)
file_size_mb = file_size_bytes / (1024 * 1024)
print("The size of model is ", file_size_mb, "MB")

print("\n\n===  TFLite Model   ===")

tflite_path = "models\\tflite\model_int8.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

file_size_bytes = os.path.getsize(tflite_path)
file_size_mb = file_size_bytes / (1024 * 1024)
print("The size of tflite model is ", file_size_mb, "MB\n")
ops = interpreter._get_ops_details()
for i, op in enumerate(ops):
    print(op['op_name'], end="   ")

