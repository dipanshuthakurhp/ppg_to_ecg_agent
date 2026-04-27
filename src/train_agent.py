import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt


def build_model(input_shape=(250, 1)):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        # Encoder
        tf.keras.layers.Conv1D(64, 9, activation='relu', padding='same'),
        tf.keras.layers.Conv1D(64, 9, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(2),  # 250 → 125

        tf.keras.layers.Conv1D(128, 9, activation='relu', padding='same'),
        tf.keras.layers.Conv1D(128, 9, activation='relu', padding='same'),

        # Decoder (learnable upsampling)
        tf.keras.layers.Conv1DTranspose(
            128, kernel_size=9, strides=2, padding='same', activation='relu'
        ),  # 125 → 250

        tf.keras.layers.Conv1D(64, 9, activation='relu', padding='same'),

        # Output
        tf.keras.layers.Conv1D(1, 9, activation='tanh', padding='same')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='mse'
    )

    return model


def run_training(type="default"):
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

    model_path = "models\saved_keras\model.h5"
    history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32
    )
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig("reports\model_training.pdf", format = "pdf")

    model.save("models\saved_keras\model.h5")
    model.summary()

    return model_path
