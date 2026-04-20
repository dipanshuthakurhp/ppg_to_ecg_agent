import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 9, activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.Conv1D(64, 9, activation='relu', padding='same'),

        tf.keras.layers.MaxPooling1D(2),  # ↓ downsample

        tf.keras.layers.Conv1D(128, 9, activation='relu', padding='same'),
        tf.keras.layers.Conv1D(128, 9, activation='relu', padding='same'),

        tf.keras.layers.Conv1DTranspose(128, 9, strides=2, padding='same', activation='relu'),  # ↑ upsample

        tf.keras.layers.Conv1D(1, 9, activation='tanh', padding='same')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='mse'
    )
    return model


def run_training():
    os.makedirs("models/saved_keras", exist_ok=True)
    clear_segments_path = "clear_segments.npy"
    clear_segments = np.load(clear_segments_path, allow_pickle=True)
    clear_segments = np.array(clear_segments.tolist())
    X = clear_segments[:, 0, :]  # PPG
    y = clear_segments[:, 1, :]  # ECG

    X = (X - np.min(X)) / (np.max(X) - np.min(X)) * 2 - 1
    y = (y - np.min(y)) / (np.max(y) - np.min(y)) * 2 - 1

    X = np.expand_dims(X, axis=-1)
    y = np.expand_dims(y, axis=-1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model(X_train.shape[1:])

    model_path = "D:\PPG to ECG\\4. Project file and model\Model.h5"
    history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32
    )
    model.save("models\saved_keras\\model.h5")
    model.summary()

    return model_path
