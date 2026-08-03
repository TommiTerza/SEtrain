"""
Exact GTCRN TensorFlow port with optional QAT annotation.

Design goals:
1) Mirror the PyTorch GTCRN architecture block-by-block.
2) Keep parameterization aligned with PyTorch, including fixed ERB matrices.
3) Support QAT annotation for Conv2D and Conv2DTranspose (including deconvs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

try:
    import tensorflow_model_optimization as tfmot
except Exception:  # pragma: no cover - dependency availability is checked by caller
    tfmot = None


BASE_NFFT = 512
BASE_LOW_BINS = 65
BASE_HIGH_BANDS = 64


class ScalarPReLU(tf.keras.layers.Layer):
    """PReLU with a single shared alpha parameter (matches nn.PReLU default)."""

    def __init__(self, alpha_init=0.25, **kwargs):
        super().__init__(**kwargs)
        self.alpha_init = float(alpha_init)
        self.alpha = None

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.alpha_init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        positive = tf.nn.relu(inputs)
        negative = inputs - positive
        return positive + self.alpha * negative

    def get_config(self):
        config = super().get_config()
        config.update({"alpha_init": self.alpha_init})
        return config


class SFE(tf.keras.layers.Layer):
    """Subband Feature Extraction equivalent to PyTorch nn.Unfold(1x3) path."""

    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        if int(kernel_size) != 3:
            raise ValueError("Only kernel_size=3 is supported for exact GTCRN parity.")
        self.kernel_size = int(kernel_size)

    def call(self, x):
        # x: (B, T, F, C)
        freq = tf.shape(x)[2]
        ch = tf.shape(x)[3]
        x_pad = tf.pad(x, [[0, 0], [0, 0], [1, 1], [0, 0]])
        slices = [x_pad[:, :, i : i + freq, :] for i in range(3)]

        # Initial concat is kernel-major channel order.
        x_cat = tf.concat(slices, axis=-1)  # (B,T,F,3*C)

        # Reorder to channel-major kernel order to match nn.Unfold flattening.
        shape = tf.shape(x_cat)
        x_cat = tf.reshape(x_cat, [shape[0], shape[1], shape[2], 3, ch])  # (B,T,F,K,C)
        x_cat = tf.transpose(x_cat, [0, 1, 2, 4, 3])  # (B,T,F,C,K)
        x_cat = tf.reshape(x_cat, [shape[0], shape[1], shape[2], ch * 3])  # (B,T,F,C*K)
        return x_cat

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


class GroupedConv2DTranspose(tf.keras.layers.Layer):
    """
    Grouped Conv2DTranspose implemented with tf.nn.conv2d_transpose.

    This supports anisotropic dilation_rate (e.g. (5, 1)), which Keras
    Conv2DTranspose rejects in some TF versions.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Tuple[int, int],
        strides: Tuple[int, int] = (1, 1),
        padding: str = "same",
        dilation_rate: Tuple[int, int] = (1, 1),
        groups: int = 1,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        self.strides = (int(strides[0]), int(strides[1]))
        self.padding = str(padding).lower()
        self.dilation_rate = (int(dilation_rate[0]), int(dilation_rate[1]))
        self.groups = int(groups)
        self.use_bias = bool(use_bias)

        if self.padding not in {"same", "valid"}:
            raise ValueError(f"Unsupported padding: {padding}")
        if self.groups <= 0:
            raise ValueError("groups must be > 0")

        self.kernel = None
        self.bias = None
        self._in_channels = None
        self._in_channels_per_group = None
        self._out_channels_per_group = None

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        if in_channels <= 0:
            raise ValueError(f"Invalid input channel count: {in_channels}")
        if in_channels % self.groups != 0:
            raise ValueError(
                f"{self.name}: input channels {in_channels} must be divisible by groups {self.groups}"
            )
        if self.filters % self.groups != 0:
            raise ValueError(
                f"{self.name}: filters {self.filters} must be divisible by groups {self.groups}"
            )

        self._in_channels = in_channels
        self._in_channels_per_group = in_channels // self.groups
        self._out_channels_per_group = self.filters // self.groups

        kernel_shape = (
            self.kernel_size[0],
            self.kernel_size[1],
            self.filters,
            self._in_channels_per_group,
        )
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer="glorot_uniform",
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer="zeros",
                trainable=True,
            )
        super().build(input_shape)

    @staticmethod
    def _output_dim(in_dim, stride, kernel, dilation, padding):
        if padding == "same":
            return in_dim * stride
        effective_kernel = (kernel - 1) * dilation + 1
        return (in_dim - 1) * stride + effective_kernel

    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        in_h = tf.shape(inputs)[1]
        in_w = tf.shape(inputs)[2]
        out_h = self._output_dim(in_h, self.strides[0], self.kernel_size[0], self.dilation_rate[0], self.padding)
        out_w = self._output_dim(in_w, self.strides[1], self.kernel_size[1], self.dilation_rate[1], self.padding)

        strides = [1, self.strides[0], self.strides[1], 1]
        dilations = [1, self.dilation_rate[0], self.dilation_rate[1], 1]
        padding = self.padding.upper()

        if self.groups == 1:
            output_shape = tf.stack([batch, out_h, out_w, self.filters])
            y = tf.nn.conv2d_transpose(
                inputs,
                self.kernel,
                output_shape=output_shape,
                strides=strides,
                padding=padding,
                dilations=dilations,
            )
        else:
            x_splits = tf.split(inputs, num_or_size_splits=self.groups, axis=-1)
            y_splits = []
            for i, x_group in enumerate(x_splits):
                k_group = self.kernel[
                    :,
                    :,
                    i * self._out_channels_per_group : (i + 1) * self._out_channels_per_group,
                    :,
                ]
                output_shape = tf.stack([batch, out_h, out_w, self._out_channels_per_group])
                y_group = tf.nn.conv2d_transpose(
                    x_group,
                    k_group,
                    output_shape=output_shape,
                    strides=strides,
                    padding=padding,
                    dilations=dilations,
                )
                y_splits.append(y_group)
            y = tf.concat(y_splits, axis=-1)

        if self.use_bias and (self.bias is not None):
            y = tf.nn.bias_add(y, self.bias)
        return y

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.dilation_rate,
                "groups": self.groups,
                "use_bias": self.use_bias,
            }
        )
        return config


@dataclass
class QATSummary:
    total_wrappers: int
    conv2d_wrappers: int
    deconv_wrappers: int
    wrapper_lines: List[str]


def _require_tfmot():
    if tfmot is None:
        raise RuntimeError(
            "tensorflow-model-optimization is required for QAT annotation/apply but is not installed."
        )


if tfmot is not None:
    _QuantizeConfigBase = tfmot.quantization.keras.QuantizeConfig
else:  # pragma: no cover - fallback for environments without tfmot
    class _QuantizeConfigBase:  # pylint: disable=too-few-public-methods
        pass


class GroupedConv2DTransposeQuantizeConfig(_QuantizeConfigBase):
    """QAT config for GroupedConv2DTranspose custom layer."""

    def __init__(self):
        _require_tfmot()
        self.weight_quantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer(
            num_bits=8,
            per_axis=False,
            symmetric=False,
            narrow_range=False,
        )
        self.output_quantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=8,
            per_axis=False,
            symmetric=False,
            narrow_range=False,
        )

    def get_weights_and_quantizers(self, layer):
        return [(layer.kernel, self.weight_quantizer)]

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        return

    def get_output_quantizers(self, layer):
        return [self.output_quantizer]

    def get_config(self):
        return {}


def _maybe_annotate(layer: tf.keras.layers.Layer, annotate: bool) -> tf.keras.layers.Layer:
    if not annotate:
        return layer
    _require_tfmot()
    return tfmot.quantization.keras.quantize_annotate_layer(layer)


def compute_erb_subbands(n_fft: int) -> Tuple[int, int]:
    base_nfreqs = BASE_NFFT // 2 + 1
    low_ratio = BASE_LOW_BINS / base_nfreqs
    high_ratio = BASE_HIGH_BANDS / max(base_nfreqs - BASE_LOW_BINS, 1)

    nfreqs = n_fft // 2 + 1
    if nfreqs < 3:
        raise ValueError(f"n_fft={n_fft} is too small to build ERB filters")

    erb_low = int(round(nfreqs * low_ratio))
    erb_low = max(1, min(erb_low, nfreqs - 2))

    remaining_bins = nfreqs - erb_low
    erb_high = int(round(remaining_bins * high_ratio))
    erb_high = max(2, min(erb_high, remaining_bins))
    return erb_low, erb_high


def compute_encoder_width(freq_bins: int) -> int:
    def conv_out(width, kernel_size, stride, padding, dilation=1):
        return max(1, (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1)

    width = int(freq_bins)
    width = conv_out(width, kernel_size=5, stride=2, padding=2)
    width = conv_out(width, kernel_size=5, stride=2, padding=2)
    return int(width)


def erb_filter_banks(
    erb_subband_1: int,
    erb_subband_2: int,
    nfft: int = 512,
    high_lim: int = 8000,
    fs: int = 16000,
) -> np.ndarray:
    def hz2erb(freq_hz):
        return 21.4 * np.log10(0.00437 * freq_hz + 1)

    def erb2hz(erb_f):
        return (10 ** (erb_f / 21.4) - 1) / 0.00437

    low_lim = erb_subband_1 / nfft * fs
    erb_low = hz2erb(low_lim)
    erb_high = hz2erb(high_lim)
    erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
    bins = np.round(erb2hz(erb_points) / fs * nfft).astype(np.int32)
    erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

    erb_filters[0, bins[0] : bins[1]] = (
        (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) / (bins[1] - bins[0] + 1e-12)
    )
    for i in range(erb_subband_2 - 2):
        erb_filters[i + 1, bins[i] : bins[i + 1]] = (
            (np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12) / (bins[i + 1] - bins[i] + 1e-12)
        )
        erb_filters[i + 1, bins[i + 1] : bins[i + 2]] = (
            (bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12) / (bins[i + 2] - bins[i + 1] + 1e-12)
        )

    erb_filters[-1, bins[-2] : bins[-1] + 1] = 1 - erb_filters[-2, bins[-2] : bins[-1] + 1]
    erb_filters = np.abs(erb_filters[:, erb_subband_1:]).astype(np.float32)
    return erb_filters


def _match_spatial(x, ref):
    ref_t = tf.shape(ref)[1]
    ref_f = tf.shape(ref)[2]
    x = x[:, :ref_t, :ref_f, :]
    pad_t = tf.maximum(0, ref_t - tf.shape(x)[1])
    pad_f = tf.maximum(0, ref_f - tf.shape(x)[2])
    return tf.pad(x, [[0, 0], [0, pad_t], [0, pad_f], [0, 0]])


def _match_frequency(x, target_f: int):
    x = x[:, :, : int(target_f), :]
    pad_f = tf.maximum(0, int(target_f) - tf.shape(x)[2])
    return tf.pad(x, [[0, 0], [0, 0], [0, pad_f], [0, 0]])


def _apply_grouped_conv2d_transpose(
    x,
    filters: int,
    kernel_size: Tuple[int, int],
    strides: Tuple[int, int],
    dilation_rate: Tuple[int, int],
    groups: int,
    padding: str,
    annotate_qat: bool,
    name: str,
):
    layer = GroupedConv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation_rate,
        groups=groups,
        use_bias=True,
        name=name,
    )
    if annotate_qat:
        _require_tfmot()
        layer = tfmot.quantization.keras.quantize_annotate_layer(
            layer,
            quantize_config=GroupedConv2DTransposeQuantizeConfig(),
        )
    return layer(x)


def _apply_conv_op(
    x,
    filters: int,
    kernel_size: Tuple[int, int],
    strides: Tuple[int, int],
    dilation_rate: Tuple[int, int],
    groups: int,
    use_deconv: bool,
    annotate_qat: bool,
    name: str,
    padding: str = "same",
):
    if use_deconv:
        return _apply_grouped_conv2d_transpose(
            x=x,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            groups=groups,
            padding=padding,
            annotate_qat=annotate_qat,
            name=name,
        )

    layer = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation_rate,
        groups=groups,
        use_bias=True,
        name=name,
    )
    layer = _maybe_annotate(layer, annotate_qat)
    return layer(x)


def _apply_tra(x, channels: int, prefix: str):
    zt = tf.reduce_mean(tf.square(x), axis=2)
    at = tf.keras.layers.GRU(
        units=channels * 2,
        return_sequences=True,
        reset_after=True,
        name=f"{prefix}_att_gru",
    )(zt)
    at = tf.keras.layers.Dense(channels, use_bias=True, name=f"{prefix}_att_fc")(at)
    at = tf.keras.layers.Activation("sigmoid", name=f"{prefix}_att_sigmoid")(at)
    at = tf.expand_dims(at, axis=2)
    return tf.keras.layers.Multiply(name=f"{prefix}_att_mul")([x, at])


def _apply_grnn(x, hidden_size: int, bidirectional: bool, prefix: str):
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
    units = hidden_size // 2
    if units <= 0:
        raise ValueError(f"{prefix}: hidden_size={hidden_size} leads to invalid GRU units.")

    if bidirectional:
        y1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units, return_sequences=True, reset_after=True, name=f"{prefix}_rnn1_gru"),
            merge_mode="concat",
            name=f"{prefix}_rnn1",
        )(x1)
        y2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units, return_sequences=True, reset_after=True, name=f"{prefix}_rnn2_gru"),
            merge_mode="concat",
            name=f"{prefix}_rnn2",
        )(x2)
    else:
        y1 = tf.keras.layers.GRU(units, return_sequences=True, reset_after=True, name=f"{prefix}_rnn1")(x1)
        y2 = tf.keras.layers.GRU(units, return_sequences=True, reset_after=True, name=f"{prefix}_rnn2")(x2)

    return tf.keras.layers.Concatenate(axis=-1, name=f"{prefix}_concat")([y1, y2])


def _apply_dpgrnn(x, width: int, hidden_size: int, prefix: str):
    # x: (B, T, F, C)
    b = tf.shape(x)[0]
    t = tf.shape(x)[1]
    f = tf.shape(x)[2]
    c = tf.shape(x)[3]

    intra_x = tf.reshape(x, [b * t, f, c])
    intra_x = _apply_grnn(intra_x, hidden_size=hidden_size // 2, bidirectional=True, prefix=f"{prefix}_intra_rnn")
    intra_x = tf.keras.layers.Dense(hidden_size, use_bias=True, name=f"{prefix}_intra_fc")(intra_x)
    intra_x = tf.reshape(intra_x, [b, t, width, hidden_size])
    intra_x = tf.keras.layers.LayerNormalization(
        axis=[-2, -1], epsilon=1e-8, name=f"{prefix}_intra_ln"
    )(intra_x)
    intra_out = tf.keras.layers.Add(name=f"{prefix}_intra_add")([x, intra_x])

    inter_x = tf.transpose(intra_out, [0, 2, 1, 3])
    inter_x = tf.reshape(inter_x, [b * f, t, c])
    inter_x = _apply_grnn(inter_x, hidden_size=hidden_size, bidirectional=False, prefix=f"{prefix}_inter_rnn")
    inter_x = tf.keras.layers.Dense(hidden_size, use_bias=True, name=f"{prefix}_inter_fc")(inter_x)
    inter_x = tf.reshape(inter_x, [b, width, t, hidden_size])
    inter_x = tf.transpose(inter_x, [0, 2, 1, 3])
    inter_x = tf.keras.layers.LayerNormalization(
        axis=[-2, -1], epsilon=1e-8, name=f"{prefix}_inter_ln"
    )(inter_x)
    inter_out = tf.keras.layers.Add(name=f"{prefix}_inter_add")([intra_out, inter_x])
    return inter_out


def _apply_conv_block(
    x,
    out_channels: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    groups: int,
    use_deconv: bool,
    is_last: bool,
    prefix: str,
    qat_annotate_conv: bool,
    qat_annotate_deconv: bool,
):
    annotate = qat_annotate_deconv if use_deconv else qat_annotate_conv
    x = _apply_conv_op(
        x=x,
        filters=out_channels,
        kernel_size=kernel_size,
        strides=stride,
        dilation_rate=(1, 1),
        groups=groups,
        use_deconv=use_deconv,
        annotate_qat=annotate,
        name=f"{prefix}_conv",
        padding="same",
    )
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=f"{prefix}_bn")(x)
    if is_last:
        return tf.keras.layers.Activation("tanh", name=f"{prefix}_act_tanh")(x)
    return ScalarPReLU(name=f"{prefix}_act_prelu")(x)


def _apply_gt_conv_block(
    x,
    hidden_channels: int,
    dilation_t: int,
    use_deconv: bool,
    prefix: str,
    qat_annotate_conv: bool,
    qat_annotate_deconv: bool,
):
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)

    x1 = SFE(kernel_size=3, name=f"{prefix}_sfe")(x1)
    channels_half = x.shape[-1] // 2

    annotate_point = qat_annotate_deconv if use_deconv else qat_annotate_conv

    x1 = _apply_conv_op(
        x=x1,
        filters=hidden_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        dilation_rate=(1, 1),
        groups=1,
        use_deconv=use_deconv,
        annotate_qat=annotate_point,
        name=f"{prefix}_point_conv1",
        padding="valid",
    )
    x1 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=f"{prefix}_point_bn1")(x1)
    x1 = ScalarPReLU(name=f"{prefix}_point_act")(x1)

    depth_ref = x1
    pad_size = int((3 - 1) * dilation_t)
    x1 = tf.pad(x1, [[0, 0], [pad_size, 0], [1, 1], [0, 0]])

    annotate_depth = qat_annotate_deconv if use_deconv else qat_annotate_conv
    x1 = _apply_conv_op(
        x=x1,
        filters=hidden_channels,
        kernel_size=(3, 3),
        strides=(1, 1),
        dilation_rate=(dilation_t, 1),
        groups=hidden_channels,
        use_deconv=use_deconv,
        annotate_qat=annotate_depth,
        name=f"{prefix}_depth_conv",
        padding="valid",
    )
    x1 = _match_spatial(x1, depth_ref)

    x1 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=f"{prefix}_depth_bn")(x1)
    x1 = ScalarPReLU(name=f"{prefix}_depth_act")(x1)

    x1 = _apply_conv_op(
        x=x1,
        filters=int(channels_half),
        kernel_size=(1, 1),
        strides=(1, 1),
        dilation_rate=(1, 1),
        groups=1,
        use_deconv=use_deconv,
        annotate_qat=annotate_point,
        name=f"{prefix}_point_conv2",
        padding="valid",
    )
    x1 = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=f"{prefix}_point_bn2")(x1)

    x1 = _apply_tra(x1, channels=int(channels_half), prefix=f"{prefix}_tra")

    stacked = tf.stack([x1, x2], axis=-1)
    shp = tf.shape(stacked)
    out = tf.reshape(stacked, [shp[0], shp[1], shp[2], shp[3] * 2])
    return out


def build_gtcrn_tf_model(
    n_fft: int = 512,
    hop_len: int = 256,
    win_len: int = 512,
    qat_annotate: bool = False,
    quantize_deconv: bool = True,
) -> tf.keras.Model:
    """
    Build the GTCRN TensorFlow model.

    Args:
        n_fft, hop_len, win_len: STFT settings.
        qat_annotate: if True, annotate Conv2D/Conv2DTranspose for QAT.
        quantize_deconv: if True, annotate deconv layers too.
    """
    if qat_annotate:
        _require_tfmot()

    erb_low, erb_high = compute_erb_subbands(int(n_fft))
    output_bins = int(erb_low + erb_high)
    encoder_width = compute_encoder_width(output_bins)
    nfreqs = int(n_fft // 2 + 1)

    erb_filters = erb_filter_banks(erb_low, erb_high, nfft=int(n_fft))
    erb_fc_kernel = erb_filters.T  # Dense kernel shape: (in_dim, units)
    ierb_fc_kernel = erb_filters   # Dense kernel shape: (in_dim, units)

    inp = tf.keras.Input(shape=(None,), name="wave_input")
    n_samples = tf.shape(inp)[1]

    center_pad = int(n_fft // 2)
    wave = tf.pad(inp, [[0, 0], [center_pad, center_pad]], mode="REFLECT")
    spec_complex = tf.signal.stft(
        wave,
        frame_length=int(win_len),
        frame_step=int(hop_len),
        fft_length=int(n_fft),
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )  # (B, T, F)

    spec_real = tf.math.real(spec_complex)
    spec_imag = tf.math.imag(spec_complex)
    spec_mag = tf.sqrt(spec_real**2 + spec_imag**2 + 1e-12)

    feat = tf.stack([spec_mag, spec_real, spec_imag], axis=-1)  # (B,T,F,3)
    spec = tf.stack([spec_real, spec_imag], axis=-1)            # (B,T,F,2)

    # ERB bm
    erb_fc = tf.keras.layers.Dense(
        units=int(erb_high),
        use_bias=False,
        trainable=False,
        kernel_initializer=tf.keras.initializers.Constant(erb_fc_kernel),
        name="erb_fc",
    )

    feat_low = feat[:, :, :erb_low, :]
    feat_high = feat[:, :, erb_low:, :]
    feat_high = tf.transpose(feat_high, [0, 1, 3, 2])  # (B,T,C,Fh)
    feat_high = erb_fc(feat_high)                       # (B,T,C,H)
    feat_high = tf.transpose(feat_high, [0, 1, 3, 2])  # (B,T,H,C)
    feat = tf.concat([feat_low, feat_high], axis=2)    # (B,T,Ferb,C)

    feat = SFE(kernel_size=3, name="sfe")(feat)

    # Encoder
    en_outs = []
    feat = _apply_conv_block(
        feat, out_channels=16, kernel_size=(1, 5), stride=(1, 2), groups=1,
        use_deconv=False, is_last=False, prefix="encoder_en_convs_0",
        qat_annotate_conv=qat_annotate, qat_annotate_deconv=False,
    )
    en_outs.append(feat)
    feat = _apply_conv_block(
        feat, out_channels=16, kernel_size=(1, 5), stride=(1, 2), groups=2,
        use_deconv=False, is_last=False, prefix="encoder_en_convs_1",
        qat_annotate_conv=qat_annotate, qat_annotate_deconv=False,
    )
    en_outs.append(feat)
    feat = _apply_gt_conv_block(
        feat, hidden_channels=16, dilation_t=1, use_deconv=False, prefix="encoder_en_convs_2",
        qat_annotate_conv=qat_annotate, qat_annotate_deconv=False,
    )
    en_outs.append(feat)
    feat = _apply_gt_conv_block(
        feat, hidden_channels=16, dilation_t=2, use_deconv=False, prefix="encoder_en_convs_3",
        qat_annotate_conv=qat_annotate, qat_annotate_deconv=False,
    )
    en_outs.append(feat)
    feat = _apply_gt_conv_block(
        feat, hidden_channels=16, dilation_t=5, use_deconv=False, prefix="encoder_en_convs_4",
        qat_annotate_conv=qat_annotate, qat_annotate_deconv=False,
    )
    en_outs.append(feat)

    feat = _apply_dpgrnn(feat, width=encoder_width, hidden_size=16, prefix="dpgrnn1")
    feat = _apply_dpgrnn(feat, width=encoder_width, hidden_size=16, prefix="dpgrnn2")

    # Decoder
    n_layers = 5
    for i in range(n_layers):
        skip = en_outs[n_layers - 1 - i]
        feat = _match_spatial(feat, skip)
        feat = tf.keras.layers.Add(name=f"decoder_skip_add_{i}")([feat, skip])

        if i == 0:
            feat = _apply_gt_conv_block(
                feat, hidden_channels=16, dilation_t=5, use_deconv=True, prefix="decoder_de_convs_0",
                qat_annotate_conv=qat_annotate, qat_annotate_deconv=qat_annotate and bool(quantize_deconv),
            )
        elif i == 1:
            feat = _apply_gt_conv_block(
                feat, hidden_channels=16, dilation_t=2, use_deconv=True, prefix="decoder_de_convs_1",
                qat_annotate_conv=qat_annotate, qat_annotate_deconv=qat_annotate and bool(quantize_deconv),
            )
        elif i == 2:
            feat = _apply_gt_conv_block(
                feat, hidden_channels=16, dilation_t=1, use_deconv=True, prefix="decoder_de_convs_2",
                qat_annotate_conv=qat_annotate, qat_annotate_deconv=qat_annotate and bool(quantize_deconv),
            )
        elif i == 3:
            feat = _apply_conv_block(
                feat, out_channels=16, kernel_size=(1, 5), stride=(1, 2), groups=2,
                use_deconv=True, is_last=False, prefix="decoder_de_convs_3",
                qat_annotate_conv=qat_annotate, qat_annotate_deconv=qat_annotate and bool(quantize_deconv),
            )
            target = en_outs[n_layers - 2 - i]
            feat = _match_spatial(feat, target)
        else:
            feat = _apply_conv_block(
                feat, out_channels=2, kernel_size=(1, 5), stride=(1, 2), groups=1,
                use_deconv=True, is_last=True, prefix="decoder_de_convs_4",
                qat_annotate_conv=qat_annotate, qat_annotate_deconv=qat_annotate and bool(quantize_deconv),
            )
            feat = _match_frequency(feat, output_bins)

    # ERB bs
    ierb_fc = tf.keras.layers.Dense(
        units=int(nfreqs - erb_low),
        use_bias=False,
        trainable=False,
        kernel_initializer=tf.keras.initializers.Constant(ierb_fc_kernel),
        name="ierb_fc",
    )

    m_low = feat[:, :, :erb_low, :]
    m_high = feat[:, :, erb_low:, :]
    m_high = tf.transpose(m_high, [0, 1, 3, 2])  # (B,T,2,H)
    m_high = ierb_fc(m_high)                      # (B,T,2,Fh)
    m_high = tf.transpose(m_high, [0, 1, 3, 2])  # (B,T,Fh,2)
    m = tf.concat([m_low, m_high], axis=2)       # (B,T,F,2)
    m = _match_spatial(m, spec)

    # Complex ratio mask
    s_real = spec[:, :, :, 0] * m[:, :, :, 0] - spec[:, :, :, 1] * m[:, :, :, 1]
    s_imag = spec[:, :, :, 1] * m[:, :, :, 0] + spec[:, :, :, 0] * m[:, :, :, 1]
    spec_enh = tf.complex(s_real, s_imag)

    inverse_window_fn = tf.signal.inverse_stft_window_fn(
        int(hop_len), forward_window_fn=tf.signal.hann_window
    )
    output = tf.signal.inverse_stft(
        spec_enh,
        frame_length=int(win_len),
        frame_step=int(hop_len),
        fft_length=int(n_fft),
        window_fn=inverse_window_fn,
    )

    output = output[:, center_pad:]
    output = output[:, :n_samples]
    pad_right = tf.maximum(0, n_samples - tf.shape(output)[1])
    output = tf.pad(output, [[0, 0], [0, pad_right]])

    return tf.keras.Model(inputs=inp, outputs=output, name="GTCRN_TF_EXACT")


def apply_qat_to_annotated_model(annotated_model: tf.keras.Model) -> tf.keras.Model:
    _require_tfmot()
    with tfmot.quantization.keras.quantize_scope(
        {
            "ScalarPReLU": ScalarPReLU,
            "SFE": SFE,
            "GroupedConv2DTranspose": GroupedConv2DTranspose,
            "GroupedConv2DTransposeQuantizeConfig": GroupedConv2DTransposeQuantizeConfig,
        }
    ):
        return tfmot.quantization.keras.quantize_apply(annotated_model)


def count_tf_params_like_torch(model: tf.keras.Model) -> int:
    """
    Count TF variables similarly to PyTorch parameter counting:
    - include trainable and fixed non-trainable weights (e.g., ERB matrices),
    - exclude BatchNorm running stats.
    """
    total = 0
    for var in model.weights:
        name = var.name
        if ("moving_mean" in name) or ("moving_variance" in name):
            continue
        total += int(np.prod(var.shape))
    return int(total)


def count_torch_gtcrn_params(network_config: Dict[str, int]) -> int:
    from models.gtcrn_end2end import GTCRN as TorchGTCRN

    model = TorchGTCRN(**network_config)
    return int(sum(p.numel() for p in model.parameters()))


def compare_tf_torch_param_counts(tf_model: tf.keras.Model, network_config: Dict[str, int]) -> Tuple[int, int, int]:
    torch_total = count_torch_gtcrn_params(network_config)
    tf_total = count_tf_params_like_torch(tf_model)
    return torch_total, tf_total, tf_total - torch_total


def summarize_qat_wrappers(model: tf.keras.Model) -> QATSummary:
    wrappers = []

    def _walk(layer):
        yield layer
        for sub_layer in getattr(layer, "layers", []):
            yield from _walk(sub_layer)

    for layer in _walk(model):
        cls_name = layer.__class__.__name__
        if "QuantizeWrapper" not in cls_name:
            continue
        wrapped = getattr(layer, "layer", None)
        wrapped_cls = wrapped.__class__.__name__ if wrapped is not None else "Unknown"
        wrappers.append((layer.name, wrapped_cls))

    conv2d = sum(1 for _, wrapped_cls in wrappers if wrapped_cls == "Conv2D")
    deconv2d = sum(
        1
        for _, wrapped_cls in wrappers
        if wrapped_cls in {"Conv2DTranspose", "GroupedConv2DTranspose"}
    )
    lines = [f"{name} -> {wrapped_cls}" for name, wrapped_cls in wrappers]
    return QATSummary(
        total_wrappers=len(wrappers),
        conv2d_wrappers=conv2d,
        deconv_wrappers=deconv2d,
        wrapper_lines=lines,
    )
