"""
Time-domain GTCRN (TD-GTCRN)
================================

This module replaces the original STFT-based front/back-end with a learnable
analysis/synthesis audio codec built from causal 1-D convolutions. The core
GTCRN encoder, grouped dual-path RNN stack, temporal attention, and decoder are
retained, but now operate directly on latent channels produced by the codec.

Key changes:
- Analysis encoder: causal Conv1d with softplus to produce non-negative latent
  representations at a hop of `stride` samples.
- Synthesis decoder: ConvTranspose1d that overlap-adds latent frames back into
  waveform samples.
- Ratio-mask head: predicts bounded masks over latent channels instead of
  complex STFT masks.
- Causal layer normalisation replaces batch norm in all temporal blocks.

The interface remains a simple waveform-to-waveform mapping usable for both
training and streaming inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CumulativeLayerNorm(nn.Module):
    """Causal layer normalisation along the temporal dimension."""

    def __init__(self, channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        cumulative_sum = torch.cumsum(x, dim=-1)
        entry_count = torch.arange(1, x.size(-1) + 1, device=x.device, dtype=x.dtype)
        entry_count = entry_count.view(1, 1, -1)

        mean = cumulative_sum / entry_count
        cumulative_square = torch.cumsum(x.pow(2), dim=-1)
        var = cumulative_square / entry_count - mean.pow(2)
        std = torch.sqrt(torch.clamp(var, min=self.eps))

        x_hat = (x - mean) / std
        return x_hat * self.gamma + self.beta


class CausalLayerNorm2d(nn.Module):
    """Causal layer norm wrapper for 4-D tensors (B, C, T, F)."""

    def __init__(self, channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.norm = CumulativeLayerNorm(channels, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Treat the latent axis as batch to reuse the 1-D implementation.
        b, c, t, f = x.shape
        x_r = x.permute(0, 3, 1, 2).contiguous().view(b * f, c, t)
        x_r = self.norm(x_r)
        x_r = x_r.view(b, f, c, t).permute(0, 2, 3, 1).contiguous()
        return x_r


class CausalLayerNorm1d(nn.Module):
    """Causal layer norm for (B, C, T) tensors."""

    def __init__(self, channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.norm = CumulativeLayerNorm(channels, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class AnalysisEncoder(nn.Module):
    """Learnable analysis filterbank using a causal Conv1d."""

    def __init__(
        self,
        latent_channels: int = 128,
        kernel_size: int = 64,
        stride: int = 32,
        activation: str = "softplus",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, latent_channels, kernel_size, stride=stride, bias=False)
        if activation == "softplus":
            self.act = nn.Softplus()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:  # pragma: no cover - validated via config
            raise ValueError(f"Unsupported activation: {activation}")
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_left = kernel_size - stride

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        # x: (B, T)
        b, length = x.shape
        x = x.unsqueeze(1)
        pad_right = (self.stride - (length % self.stride)) % self.stride
        x = F.pad(x, (self.pad_left, pad_right))
        z = self.act(self.conv(x))
        return z, pad_right


class SynthesisDecoder(nn.Module):
    """Inverse filterbank implemented with ConvTranspose1d."""

    def __init__(self, latent_channels: int, kernel_size: int = 64, stride: int = 32) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose1d(latent_channels, 1, kernel_size, stride=stride, bias=False)
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_left = kernel_size - stride

    def forward(self, z: torch.Tensor, target_length: int) -> torch.Tensor:
        y = self.deconv(z)
        start = self.pad_left
        end = start + target_length
        return y[..., start:end]


class SFE(nn.Module):
    """Sub-band feature extraction operating across latent channels."""

    def __init__(self, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(
            kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, (kernel_size - 1) // 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1] * self.kernel_size, x.shape[2], x.shape[3])
        return xs


class TRA(nn.Module):
    """Temporal recurrent attention module."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.att_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels * 2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zt = torch.mean(x.pow(2), dim=-1)
        at = self.att_gru(zt.transpose(1, 2))[0]
        at = self.att_fc(at).transpose(1, 2)
        at = self.att_act(at)
        return x * at[..., None]


def _activation(name: Optional[str]) -> nn.Module:
    if name is None:
        return nn.Identity()
    if name == "prelu":
        return nn.PReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation: {name}")


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        groups: int = 1,
        use_deconv: bool = False,
        activation: Optional[str] = "prelu",
    ) -> None:
        super().__init__()
        if use_deconv:
            conv_module = nn.ConvTranspose2d
            output_padding = tuple(max(s - 1, 0) for s in stride)
            self.conv = conv_module(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                output_padding=output_padding,
                groups=groups,
            )
        else:
            conv_module = nn.Conv2d
            self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.norm = CausalLayerNorm2d(out_channels)
        self.act = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class GTConvBlock(nn.Module):
    """Group Temporal Convolution block adapted with causal normalisation."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        use_deconv: bool = False,
    ) -> None:
        super().__init__()
        self.pad_size = (kernel_size[0] - 1) * dilation[0]

        def build_conv(in_channels: int, out_channels: int, kernel: Tuple[int, int], stride: Tuple[int, int] = (1, 1),
                       padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1) -> nn.Module:
            if use_deconv:
                output_padding = tuple(max(s - 1, 0) for s in stride)
                return nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    output_padding=output_padding,
                )
            return nn.Conv2d(
                in_channels,
                out_channels,
                kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )

        self.sfe = SFE(kernel_size=3, stride=1)

        self.point_conv1 = build_conv(in_channels // 2 * 3, hidden_channels, (1, 1))
        self.point_norm1 = CausalLayerNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = build_conv(
            hidden_channels,
            hidden_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=hidden_channels,
        )
        self.depth_norm = CausalLayerNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = build_conv(hidden_channels, in_channels // 2, (1, 1))
        self.point_norm2 = CausalLayerNorm2d(in_channels // 2)

        self.tra = TRA(in_channels // 2)

    def shuffle(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()
        x = rearrange(x, "b c g t f -> b (c g) t f")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_norm1(self.point_conv1(x1)))
        h1 = F.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_act(self.depth_norm(self.depth_conv(h1)))
        h1 = self.point_norm2(self.point_conv2(h1))

        h1 = self.tra(h1)
        return self.shuffle(h1, x2)


class GRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(
            input_size // 2,
            hidden_size // 2,
            num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
        self.rnn2 = nn.GRU(
            input_size // 2,
            hidden_size // 2,
            num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if h is None:
            if self.bidirectional:
                h = torch.zeros(self.num_layers * 2, x.shape[0], self.hidden_size, device=x.device)
            else:
                h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return y, h


class DPGRNN(nn.Module):
    def __init__(self, input_size: int, width: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        intra_x = self.intra_rnn(intra_x)[0]
        intra_x = self.intra_fc(intra_x)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size)
        intra_x = self.intra_ln(intra_x)
        intra_out = x + intra_x

        x = intra_out.permute(0, 2, 1, 3)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        inter_x = self.inter_rnn(inter_x)[0]
        inter_x = self.inter_fc(inter_x)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size)
        inter_x = inter_x.permute(0, 2, 1, 3)
        inter_x = self.inter_ln(inter_x)
        inter_out = intra_out + inter_x

        dual_out = inter_out.permute(0, 3, 1, 2)
        return dual_out


class Encoder(nn.Module):
    def __init__(self, input_channels: int = 3, width: int = 16) -> None:
        super().__init__()
        self.en_convs = nn.ModuleList(
            [
                ConvBlock(input_channels, width, (1, 5), stride=(1, 2), padding=(0, 2)),
                ConvBlock(width, width, (1, 5), stride=(1, 2), padding=(0, 2), groups=2),
                GTConvBlock(width, width, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1)),
                GTConvBlock(width, width, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1)),
                GTConvBlock(width, width, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1)),
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        en_outs = []
        for layer in self.en_convs:
            x = layer(x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    def __init__(self, width: int = 16, output_channels: int = 1) -> None:
        super().__init__()
        self.de_convs = nn.ModuleList(
            [
                GTConvBlock(width, width, (3, 3), stride=(1, 1), padding=(2 * 5, 1), dilation=(5, 1), use_deconv=True),
                GTConvBlock(width, width, (3, 3), stride=(1, 1), padding=(2 * 2, 1), dilation=(2, 1), use_deconv=True),
                GTConvBlock(width, width, (3, 3), stride=(1, 1), padding=(2 * 1, 1), dilation=(1, 1), use_deconv=True),
                ConvBlock(width, width, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True),
                ConvBlock(width, output_channels, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True, activation=None),
            ]
        )

    def forward(self, x: torch.Tensor, en_outs: list[torch.Tensor]) -> torch.Tensor:
        n_layers = len(self.de_convs)
        for i in range(n_layers):
            x = self.de_convs[i](x + en_outs[n_layers - 1 - i])
        return x


class MaskHead(nn.Module):
    """Applies a bounded ratio mask over latent channels."""

    def __init__(self, activation: str = "sigmoid") -> None:
        super().__init__()
        if activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported mask activation: {activation}")

    def forward(self, mask_logits: torch.Tensor) -> torch.Tensor:
        return self.act(mask_logits)


@dataclass
class CodecConfig:
    latent_channels: int = 128
    kernel_size: int = 64
    stride: int = 32
    activation: str = "softplus"
    mask_activation: str = "sigmoid"


class GTCRN(nn.Module):
    def __init__(self, codec: CodecConfig | None = None) -> None:
        super().__init__()
        if codec is None:
            codec = CodecConfig()
        elif isinstance(codec, dict):
            codec = CodecConfig(**codec)

        encoded_width = self._compute_encoded_width(codec.latent_channels)

        self.analysis = AnalysisEncoder(
            latent_channels=codec.latent_channels,
            kernel_size=codec.kernel_size,
            stride=codec.stride,
            activation=codec.activation,
        )
        self.synthesis = SynthesisDecoder(
            latent_channels=codec.latent_channels,
            kernel_size=codec.kernel_size,
            stride=codec.stride,
        )

        self.sfe = SFE(3, 1)
        self.encoder = Encoder(input_channels=3, width=16)
        self.dpgrnn1 = DPGRNN(16, encoded_width, 16)
        self.dpgrnn2 = DPGRNN(16, encoded_width, 16)
        self.decoder = Decoder(width=16, output_channels=1)
        self.mask = MaskHead(codec.mask_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        original_length = x.shape[-1]
        latent, _pad_right = self.analysis(x)

        latent_feat = latent.transpose(1, 2)  # (B, frames, latent)
        feat = latent_feat.unsqueeze(1)  # (B, 1, frames, latent)
        feat = self.sfe(feat)

        feat, en_outs = self.encoder(feat)
        feat = self.dpgrnn1(feat)
        feat = self.dpgrnn2(feat)

        mask_logits = self.decoder(feat, en_outs)
        mask = self.mask(mask_logits).squeeze(1)  # (B, frames, latent)

        masked_latent = mask * latent_feat
        masked_latent = masked_latent.transpose(1, 2)  # (B, latent, frames)

        enhanced = self.synthesis(masked_latent, original_length)
        return enhanced.squeeze(1)

    @staticmethod
    def _compute_encoded_width(latent_channels: int) -> int:
        width = (latent_channels + 1) // 2
        width = (width + 1) // 2
        return max(width, 1)


__all__ = ["GTCRN", "CodecConfig"]
