import tensorflow as tf
import numpy as np
import os

clear_segments_path = "D:\physiofusion_agent_v2\clear_segments.npy"
clear_segments = np.load(clear_segments_path, allow_pickle=True)
clear_segments = np.array(clear_segments.tolist())

# Separate PPG (X) and ECG (y)
X = clear_segments[:, 0, :]  # PPG
X = (X - np.min(X)) / (np.max(X) - np.min(X)) * 2 - 1
X = np.expand_dims(X, axis=-1)

def to_int8(model_path: str) -> str:
    os.makedirs("models/tflite", exist_ok=True)
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except (ValueError, KeyError):
        # Fallback for Keras 3.x compatibility issues with legacy H5 models
        print("Warning: Keras 3.x compatibility issue detected. Loading model without compilation...")
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

    calib_data = X[:50].astype(np.float32)

    def representative_dataset():
        for i in range(calib_data.shape[0]):
            yield [calib_data[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()

    tflite_path = "models/tflite/model_int8.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_quant_model)

    return tflite_path
