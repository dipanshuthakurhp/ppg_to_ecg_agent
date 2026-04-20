import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

# ------------------ Load CSV ------------------
ecg = pd.read_csv(r"D:\physiofusion_agent_v2\data\ecg\ecg_2.csv")["A1"].values
ppg = pd.read_csv(r"D:\physiofusion_agent_v2\data\ppg\ppg_2.csv")["A1"].values

# ------------------ Load Model ------------------
model = tf.keras.models.load_model(r"models\saved_keras\model.h5", compile=False)

print("Model Input Shape:", model.input_shape)

# ------------------ Preprocess Input ------------------
EXPECTED_LEN = model.input_shape[1]

# Trim or pad PPG
ppg = ppg[:EXPECTED_LEN]

# Normalize PPG (same as training)
ppg = (ppg - np.mean(ppg)) / np.std(ppg)

# Reshape based on model
if len(model.input_shape) == 3:
    ppg = ppg.reshape(1, EXPECTED_LEN, 1)
elif len(model.input_shape) == 2:
    ppg = ppg.reshape(1, EXPECTED_LEN)
else:
    raise ValueError("Unexpected model input shape")

# ------------------ Prediction ------------------
predicted_ecg = model.predict(ppg)

# Flatten output
predicted_ecg = predicted_ecg.flatten()

# ------------------ Your TFLite Output ------------------
tflite_data = [
-62, -52, -34, -25, -33, -48, -59, -69, -81, -87, -92, -88, -86, -78, -72, -68,
-70, -71, -74, -79, -81, -86, -90, -94, -99, -101, -104, -106, -108, -108, -110,
-109, -110, -109, -109, -109, -109, -109, -109, -107, -107, -106, -106, -106,
-106, -106, -104, -99, -90, -79, -75, -66, -52, -29, -22, -40, -58, -73, -86,
-92, -94, -91, -86, -80, -75, -69, -66, -68, -72, -79, -83, -89, -94, -98, -102,
-105, -107, -108, -110, -110, -111, -110, -111, -110, -110, -110, -110, -109,
-107, -107, -106, -106, -107, -107, -106, -102, -94, -81, -70, -49, -27, -18,
-34, -60, -77, -86, -94, -96, -96, -90, -83, -74, -66, -61, -62, -67, -76, -84,
-92, -98, -103, -106, -107, -109, -110, -111, -112, -111, -111, -111, -111,
-111, -110, -109, -107, -107, -107, -108, -109, -107, -102, -90, -74, -50,
-26, -17, -34, -60, -77, -87, -94, -96, -96, -91, -83, -75, -66, -61, -62,
-67, -76, -84, -92, -98, -103, -106, -108, -109, -110, -111, -112, -111,
-112, -111, -111, -110, -110, -108, -107, -107, -108, -109, -109, -106,
-99, -86, -71, -49, -26, -19, -32, -54, -71, -83, -92, -94, -93, -88,
-81, -73, -68, -64, -64, -69, -75, -84, -88, -95, -99, -102, -105, -107,
-109, -109, -111, -110, -111, -110, -110, -110, -110, -109, -108, -107,
-107, -106, -107, -104, -100, -92, -79, -62, -45, -37, -45, -61, -74,
-84, -92, -94, -93, -88, -80, -74, -70, -71, -75, -81, -90, -94
]

tflite_data = np.array(tflite_data)

# ------------------ Match Lengths ------------------
min_len = min(len(ecg), len(predicted_ecg), len(tflite_data))

ecg = ecg[:min_len]
predicted_ecg = predicted_ecg[:min_len]
tflite_data = tflite_data[:min_len]

# ------------------ Normalize for Comparison ------------------
def normalize_01(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

ecg = normalize_01(ecg)
predicted_ecg = normalize_01(predicted_ecg)
tflite_data = normalize_01(tflite_data)

# ------------------ Plot ------------------
plt.figure(figsize=(10, 4))
plt.plot(ecg, label="Actual ECG")
plt.plot(tflite_data, label="TFLite Output")
plt.plot(predicted_ecg, label="H5 Model Output")

plt.title("PPG → ECG Comparison")
plt.xlabel("Samples")
plt.ylabel("Normalized Amplitude")
plt.grid()
plt.legend()

# ------------------ RMSE ------------------
def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))

h5_rmse = rmse(ecg, predicted_ecg)
tflite_rmse = rmse(ecg, tflite_data)

print("RMSE (H5 Model):", h5_rmse)
print("RMSE (TFLite Model):", tflite_rmse)

plt.show()