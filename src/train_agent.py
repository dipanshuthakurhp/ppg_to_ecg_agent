import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model


def build_unet(input_len: int = 250, base_filters: int = 32) -> Model:
    inp = layers.Input(shape=(input_len, 1), name="ppg_input")

    def enc_block(x, f):
        x = layers.Conv1D(f, 9, padding="same", activation="relu")(x)
        x = layers.Conv1D(f, 9, padding="same", activation="relu")(x)
        return x

    def down(x, f):
        skip = enc_block(x, f)
        x = layers.Conv1D(f, 2, strides=2, padding="same")(skip)
        return x, skip

    def crop_to_match(x, skip):
        # crop x to match skip length
        diff = x.shape[1] - skip.shape[1]
        if diff is not None and diff > 0:
            x = layers.Cropping1D((0, diff))(x)
        return x

    def up(x, skip, f):
        x = layers.Conv1DTranspose(f, 2, strides=2, padding="same")(x)

        # Fix shape mismatch safely
        x = crop_to_match(x, skip)

        x = layers.Concatenate()([x, skip])

        x = layers.Conv1D(f, 9, padding="same", activation="relu")(x)
        x = layers.Conv1D(f, 9, padding="same", activation="relu")(x)
        return x

    # Encoder
    x, s1 = down(inp, base_filters)        # 250 → 125
    x, s2 = down(x, base_filters * 2)      # 125 → 63
    x, s3 = down(x, base_filters * 4)      # 63 → 32

    # Bottleneck
    x = enc_block(x, base_filters * 8)

    # Decoder
    x = up(x, s3, base_filters * 4)        # 32 → 63
    x = up(x, s2, base_filters * 2)        # 63 → 125
    x = up(x, s1, base_filters)            # 125 → 250

    # Ensure exact output length
    x = layers.Cropping1D((0, x.shape[1] - input_len))(x)

    out = layers.Conv1D(1, 1, activation="tanh", name="ecg_output")(x)

    model = Model(inp, out, name="UNet")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="mse"
    )

    return model


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
    if type =="default":
        model = build_model(X_train.shape[1:])
    elif type == "unet":
        model = build_unet()

    model_path = "models\saved_keras\model.h5"
    history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32
    )
    model.save("models\saved_keras\model.h5")
    model.summary()

    return model_path
