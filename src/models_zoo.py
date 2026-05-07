import tensorflow as tf
from enum import Enum

def build_lstm_model(input_shape=(250, 1)):
    inputs = tf.keras.Input(shape=input_shape)

    # 3 layers total: LSTM -> LSTM -> TimeDistributed(Dense)
    x = tf.keras.layers.LSTM(234, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(96, return_sequences=True)(x)
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1)
    )(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae"]
    )
    return model


def build_current_model(input_shape=(250, 1)):
    inputs = tf.keras.Input(shape=input_shape)

    # 7 layers total:
    # Conv -> Conv -> Pool -> Conv -> ConvTranspose -> Conv -> Output
    x = tf.keras.layers.Conv1D(40, 9, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.Conv1D(96, 9, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Conv1D(224, 7, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv1DTranspose(
        224, kernel_size=7, strides=2, activation="relu", padding="same"
    )(x)

    x = tf.keras.layers.Conv1D(216, 5, activation="relu", padding="same")(x)
    outputs = tf.keras.layers.Conv1D(1, 5, activation="tanh", padding="same")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="mse",
        metrics=["mae"]
    )
    return model


def build_cnn_lstm_model(input_shape=(250, 1)):
    inputs = tf.keras.Input(shape=input_shape)

    # 7 layers total:
    # Conv -> Conv -> Pool -> LSTM -> ConvTranspose -> Conv -> Output
    x = tf.keras.layers.Conv1D(40, 7, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.Conv1D(128, 7, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.LSTM(192, return_sequences=True)(x)

    x = tf.keras.layers.Conv1DTranspose(
        192, kernel_size=7, strides=2, activation="relu", padding="same"
    )(x)

    x = tf.keras.layers.Conv1D(168, 7, activation="relu", padding="same")(x)
    outputs = tf.keras.layers.Conv1D(1, 7, activation="tanh", padding="same")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="mse",
        metrics=["mae"]
    )
    return model


def build_dense_cnn(input_shape=(250, 1)):
    inputs = tf.keras.Input(shape=input_shape)

    # 11 layers total:
    # Conv -> Conv -> Pool -> Conv -> Conv -> Conv -> ConvTranspose -> Conv -> Conv -> Conv -> Output
    x = tf.keras.layers.Conv1D(391, 9, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.Conv1D(391, 9, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x = tf.keras.layers.Conv1D(238, 7, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv1D(238, 7, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv1D(282, 5, activation="relu", padding="same")(x)

    x = tf.keras.layers.Conv1DTranspose(
        282, kernel_size=7, strides=2, activation="relu", padding="same"
    )(x)

    x = tf.keras.layers.Conv1D(238, 7, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv1D(238, 5, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv1D(391, 5, activation="relu", padding="same")(x)

    outputs = tf.keras.layers.Conv1D(1, 1, activation="tanh", padding="same")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="mse",
        metrics=["mae"]
    )
    return model


class Models(Enum):
    CNN = build_current_model()
    LSTM = build_lstm_model()
    CNN_LSTM =build_cnn_lstm_model()
    DENSE_CNN = build_dense_cnn()