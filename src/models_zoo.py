"""
models_zoo.py
=============
PhysioFusion – comparison model architectures for PPG-to-ECG reconstruction.

All builders follow the same signature:
    build_<name>(input_len=250) -> tf.keras.Model

Input shape : (batch, input_len, 1)   – single-channel PPG segment
Output shape: (batch, input_len, 1)   – reconstructed ECG segment

Existing baselines (loaded separately from models/):
    CNN_baseline        – Conv64×2 → MaxPool → Conv128×2 → UpSampling → Conv1
    Model               – Conv64×2 → MaxPool → Conv128×2 → Conv1DTranspose → Conv1
    ResNet_CNN          – Residual blocks (64→128) → Conv1 output

New models added here:
    TCN                 – Dilated causal Conv1D stack (receptive field >> 250)
    UNet                – Encoder-decoder with symmetric skip connections
    DepthwiseSepCNN     – MobileNet-style depthwise-separable Conv1D (ESP32-friendly)
    BiLSTMConv          – Bidirectional LSTM encoder + Conv1D decoder (accuracy ceiling)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# ─────────────────────────────────────────────────────────────────────────────
# Helper: residual TCN block
# ─────────────────────────────────────────────────────────────────────────────
def _tcn_block(x, filters: int, kernel_size: int, dilation: int, dropout: float = 0.1):
    """One dilated causal residual block used inside build_tcn."""
    skip = x
    # Branch
    for _ in range(2):
        x = layers.Conv1D(
            filters, kernel_size,
            padding="causal",
            dilation_rate=dilation,
            kernel_initializer="he_normal",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if dropout > 0:
            x = layers.SpatialDropout1D(dropout)(x)
    # 1×1 projection if channel count changed
    if skip.shape[-1] != filters:
        skip = layers.Conv1D(filters, 1, padding="same")(skip)
    return layers.Add()([x, skip])


# ─────────────────────────────────────────────────────────────────────────────
# 1. TCN  –  Temporal Convolutional Network
# ─────────────────────────────────────────────────────────────────────────────
def build_tcn(input_len: int = 250, filters: int = 64, kernel_size: int = 8,
              num_blocks: int = 5, dropout: float = 0.1) -> Model:
    """
    Dilated causal TCN with exponentially growing dilation (1, 2, 4, 8, 16).

    For kernel=8, dilations [1,2,4,8,16]:
        Receptive field = 1 + 2 × (8-1) × (1+2+4+8+16) = 1 + 14×31 = 435 samples
    Easily covers the full 250-sample window.

    Architecture: Input → stem Conv → 5 TCN blocks → Conv1 output
    """
    inp = layers.Input(shape=(input_len, 1), name="ppg_input")
    x = layers.Conv1D(filters, 1, padding="same", name="stem")(inp)

    for i in range(num_blocks):
        dilation = 2 ** i
        x = _tcn_block(x, filters, kernel_size, dilation, dropout)

    out = layers.Conv1D(1, 1, activation="tanh", name="ecg_output")(x)
    return Model(inp, out, name="TCN")


# ─────────────────────────────────────────────────────────────────────────────
# 2. U-Net  –  encoder-decoder with symmetric skip connections
# ─────────────────────────────────────────────────────────────────────────────
def build_unet(input_len: int = 250, base_filters: int = 32) -> Model:
    """
    1-D U-Net: 3 encoder levels + bottleneck + 3 decoder levels with skip connections.

    The skip connections concatenate encoder feature maps at each resolution
    into the corresponding decoder stage, preserving fine temporal structure
    (QRS width, T-wave morphology) that plain MaxPool→UpSample loses.

    Encoder strides: 250 → 125 → 63 → 32  (using stride-2 Conv)
    Decoder: ConvTranspose back to 250 at each level.
    """
    inp = layers.Input(shape=(input_len, 1), name="ppg_input")

    def enc_block(x, f):
        x = layers.Conv1D(f, 9, padding="same", activation="relu")(x)
        x = layers.Conv1D(f, 9, padding="same", activation="relu")(x)
        return x

    def down(x, f):
        skip = enc_block(x, f)
        x = layers.Conv1D(f, 2, strides=2, padding="same")(skip)
        return x, skip

    def up(x, skip, f):
        x = layers.Conv1DTranspose(f, 2, strides=2, padding="same", activation="relu")(x)
        # trim/pad to match skip length before concat
        diff = tf.shape(skip)[1] - tf.shape(x)[1]
        x = layers.ZeroPadding1D((0, 0))(x)   # placeholder; handled by Lambda below
        x = layers.Lambda(
            lambda t: tf.concat([
                t[0][:, :tf.shape(t[1])[1], :],
                t[1]
            ], axis=-1)
        )([x, skip])
        x = layers.Conv1D(f, 9, padding="same", activation="relu")(x)
        x = layers.Conv1D(f, 9, padding="same", activation="relu")(x)
        return x

    # Encoder
    x, s1 = down(inp,   base_filters)       # 250 → 125
    x, s2 = down(x,     base_filters * 2)   # 125 → 63
    x, s3 = down(x,     base_filters * 4)   # 63  → 32

    # Bottleneck
    x = enc_block(x, base_filters * 8)

    # Decoder
    x = up(x, s3, base_filters * 4)         # 32  → 63
    x = up(x, s2, base_filters * 2)         # 63  → 125
    x = up(x, s1, base_filters)             # 125 → 250

    # Crop/pad to exact input_len then output
    x = layers.Lambda(lambda t: t[:, :input_len, :])(x)
    out = layers.Conv1D(1, 1, activation="tanh", name="ecg_output")(x)
    return Model(inp, out, name="UNet")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Depthwise-Separable CNN  –  MobileNet-style, ESP32-friendly
# ─────────────────────────────────────────────────────────────────────────────
def _ds_block(x, filters: int, kernel_size: int = 9):
    """
    Depthwise-separable Conv1D block.

    Implementation: expand input to 4-D (B, T, 1, C), apply DepthwiseConv2D
    along the time axis only, squeeze back to 3-D, then apply a pointwise
    Conv1D.  This is mathematically identical to a depthwise-separable Conv1D
    and works on TF 2.8+ without any custom ops.

    Why not use groups= in Conv1D?
        groups= was added in TF 2.6 but TFLite INT8 conversion does not
        support grouped convolutions reliably until TF 2.14.
        The DepthwiseConv2D route converts cleanly to INT8 on TF 2.10+.
    """
    # Depthwise: treat (B, T, C) as (B, T, 1, C) and apply 2-D depthwise
    # with kernel (kernel_size, 1) → spatial-only filtering along T
    x = layers.Lambda(lambda t: tf.expand_dims(t, axis=2))(x)        # (B,T,1,C)
    x = layers.DepthwiseConv2D(
        kernel_size=(kernel_size, 1),
        padding="same",
        depth_multiplier=1,
        use_bias=False,
    )(x)
    x = layers.Lambda(lambda t: tf.squeeze(t, axis=2))(x)             # (B,T,C)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # Pointwise: 1×1 Conv1D changes channel depth
    x = layers.Conv1D(filters, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def build_depthwise_sep_cnn(input_len: int = 250) -> Model:
    """
    Encoder-decoder using depthwise-separable Conv1D blocks.

    Compared to CNN_baseline (standard Conv1D):
        - ~8× fewer multiply-add operations per block
        - ~5× smaller parameter count
        - Quantises to INT8 TFLite target < 150 KB
        - Fits comfortably inside ESP32's 200 KB tensor arena

    Architecture mirrors CNN_baseline depth but with DS blocks throughout.
    """
    inp = layers.Input(shape=(input_len, 1), name="ppg_input")

    x = _ds_block(inp,  64)
    x = _ds_block(x,    64)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = _ds_block(x,   128)
    x = _ds_block(x,   128)
    x = layers.Conv1DTranspose(128, 9, strides=2, padding="same", activation="relu")(x)
    out = layers.Conv1D(1, 9, padding="same", activation="tanh", name="ecg_output")(x)
    # Crop to exact input_len (Conv1DTranspose may produce T+1 on odd lengths)
    out = layers.Lambda(lambda t: t[:, :input_len, :], name="ecg_output_crop")(out)
    return Model(inp, out, name="DepthwiseSepCNN")


# ─────────────────────────────────────────────────────────────────────────────
# 4. BiLSTM + Conv decoder  –  accuracy ceiling (not ESP32-deployable)
# ─────────────────────────────────────────────────────────────────────────────
def build_bilstm_conv(input_len: int = 250, lstm_units: int = 128) -> Model:
    """
    Bidirectional LSTM encoder captures long-range temporal dependencies in
    both directions across the 250-sample PPG window, then a Conv1D decoder
    reconstructs the ECG sequence at full resolution.

    This is included as an *accuracy ceiling* in the comparison table.
    It will NOT deploy to ESP32 (INT8 LSTM ops exceed the 200 KB arena),
    but establishes the upper bound on achievable MSE/R² for reviewers.

    Architecture:
        Input → BiLSTM(128) → BiLSTM(64, return_seq=True) →
        Conv1D(128) → Conv1D(64) → Conv1D(1, tanh)
    """
    inp = layers.Input(shape=(input_len, 1), name="ppg_input")

    # Encoder: two stacked BiLSTMs
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True), name="bilstm_1"
    )(inp)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units // 2, return_sequences=True), name="bilstm_2"
    )(x)

    # Decoder: conv layers to reconstruct waveform
    x = layers.Conv1D(128, 9, padding="same", activation="relu", name="dec_conv1")(x)
    x = layers.Conv1D(64,  9, padding="same", activation="relu", name="dec_conv2")(x)
    out = layers.Conv1D(1,  1, activation="tanh", name="ecg_output")(x)
    return Model(inp, out, name="BiLSTMConv")


# ─────────────────────────────────────────────────────────────────────────────
# Registry: all new models
# ─────────────────────────────────────────────────────────────────────────────
NEW_MODELS = {
    "TCN":             build_tcn,
    "UNet":            build_unet,
    "DepthwiseSepCNN": build_depthwise_sep_cnn,
    "BiLSTMConv":      build_bilstm_conv,
}


if __name__ == "__main__":
    # Quick smoke-test: build + summarise each model
    import numpy as np
    dummy = np.random.randn(4, 250, 1).astype("float32")
    for name, builder in NEW_MODELS.items():
        m = builder(input_len=250)
        out = m(dummy)
        print(f"{name:20s}  params={m.count_params():>10,}  output={out.shape}")
