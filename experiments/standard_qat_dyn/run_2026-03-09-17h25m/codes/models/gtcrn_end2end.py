"""
GTCRN: ShuffleNetV2 + SFE + TRA + 2 DPGRNN
Ultra tiny, 33.0 MMACs, 23.67 K params
"""
import torch
import torch.ao.quantization as quant
import numpy as np
import torch.nn as nn
from einops import rearrange
from copy import deepcopy

from .quant_constraints import (
    ConstrainedMovingAverageMinMaxObserver,
    ConstrainedMovingAveragePerChannelMinMaxObserver,
    _constrain_scale_fixed,
    _constrain_scale_pow2,
)


class StaticQuantizedGRU(nn.Module):
    """
    Static Q/DQ emulation for GRU.

    This module stores quantized int weights (per-channel symmetric) and runs a
    float GRU kernel on dequantized weights. Input activations are quantized
    with fixed qparams captured during calibration/QAT.
    """

    def __init__(
        self,
        source_gru: nn.GRU,
        *,
        input_scale: float,
        input_zero_point: int,
        act_quant_min: int,
        act_quant_max: int,
        weight_bit_width: int = 8,
        weight_scale_constraint_mode: str = "none",
        weight_scale_constraint_frac_bits: int | None = None,
        weight_scale_constraint_pow2_rounding: str = "nearest",
    ):
        super().__init__()
        if not isinstance(source_gru, nn.GRU):
            raise TypeError(f"StaticQuantizedGRU expects nn.GRU, got {type(source_gru).__name__}")
        if weight_bit_width < 2 or weight_bit_width > 16:
            raise ValueError(f"weight_bit_width must be in [2, 16], got {weight_bit_width}")

        self.weight_bit_width = int(weight_bit_width)
        self.weight_scale_constraint_mode = str(weight_scale_constraint_mode).strip().lower()
        self.weight_scale_constraint_frac_bits = weight_scale_constraint_frac_bits
        self.weight_scale_constraint_pow2_rounding = str(weight_scale_constraint_pow2_rounding).strip().lower()
        self.weight_quant_min = -(1 << (self.weight_bit_width - 1))
        self.weight_quant_max = (1 << (self.weight_bit_width - 1)) - 1
        self.act_quant_min = int(act_quant_min)
        self.act_quant_max = int(act_quant_max)

        input_scale = float(input_scale)
        if not np.isfinite(input_scale) or input_scale <= 0.0:
            input_scale = 1.0
        self.register_buffer("input_scale", torch.tensor([input_scale], dtype=torch.float32))
        self.register_buffer("input_zero_point", torch.tensor([int(input_zero_point)], dtype=torch.int32))

        self.float_gru = nn.GRU(
            input_size=source_gru.input_size,
            hidden_size=source_gru.hidden_size,
            num_layers=source_gru.num_layers,
            bias=source_gru.bias,
            batch_first=source_gru.batch_first,
            dropout=source_gru.dropout,
            bidirectional=source_gru.bidirectional,
        )
        self.float_gru.load_state_dict(source_gru.state_dict(), strict=True)
        self.float_gru.eval()

        self.quantized_weight_names = []
        with torch.no_grad():
            for name, param in self.float_gru.named_parameters():
                if not name.startswith("weight_"):
                    continue
                int_repr, scales, dequant = self._quantize_weight_per_channel_symmetric(param.detach())
                param.copy_(dequant)

                key = name.replace(".", "__")
                self.register_buffer(f"{key}_int_repr", int_repr)
                self.register_buffer(f"{key}_scale", scales)
                self.register_buffer(
                    f"{key}_zero_point",
                    torch.zeros_like(scales, dtype=torch.int32),
                )
                self.quantized_weight_names.append(name)

        for parameter in self.float_gru.parameters():
            parameter.requires_grad_(False)

    def _constrain_scales(self, scales: torch.Tensor) -> torch.Tensor:
        mode = self.weight_scale_constraint_mode
        if mode in {"", "none", "off", "false", "0"}:
            return scales
        if mode in {"pow2", "power2", "power_of_two", "power-of-two"}:
            return _constrain_scale_pow2(scales, self.weight_scale_constraint_pow2_rounding)
        if mode in {"fixed", "qmn", "fixed_qmn", "qformat"}:
            if self.weight_scale_constraint_frac_bits is None:
                raise ValueError("weight_scale_constraint_frac_bits is required for mode='fixed'")
            return _constrain_scale_fixed(scales, int(self.weight_scale_constraint_frac_bits))
        raise ValueError(f"Unsupported weight scale constraint mode: {mode}")

    def _quantize_weight_per_channel_symmetric(self, weight: torch.Tensor):
        # GRU parameters are 2D: (out_features, in_features).
        max_abs = weight.abs().amax(dim=1)
        scales = max_abs / float(self.weight_quant_max)
        scales = torch.clamp(scales, min=1e-8)
        scales = self._constrain_scales(scales)
        scales = torch.clamp(scales, min=1e-8)

        q = torch.round(weight / scales[:, None])
        q = torch.clamp(q, self.weight_quant_min, self.weight_quant_max)

        if self.weight_bit_width <= 8:
            q_storage = q.to(torch.int8)
        else:
            q_storage = q.to(torch.int16)
        dequant = q_storage.to(torch.float32) * scales[:, None]
        return q_storage, scales.to(torch.float32), dequant

    def _qdq_activation(self, tensor: torch.Tensor) -> torch.Tensor:
        scale = self.input_scale.to(device=tensor.device, dtype=tensor.dtype)
        zero_point = self.input_zero_point.to(device=tensor.device, dtype=tensor.dtype)
        q = torch.round(tensor / scale + zero_point)
        q = torch.clamp(q, self.act_quant_min, self.act_quant_max)
        return (q - zero_point) * scale

    def forward(self, x, h=None):
        x_qdq = self._qdq_activation(x)
        h_qdq = self._qdq_activation(h) if h is not None else None
        return self.float_gru(x_qdq, h_qdq)


class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft//2 + 1
        self.erb_subband_1 = erb_subband_1
        self.output_bins = erb_subband_1 + erb_subband_2
        self.erb_fc = nn.Linear(nfreqs-erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs-erb_subband_1, bias=False)
        self.erb_quant = quant.QuantStub()
        self.erb_dequant = quant.DeQuantStub()
        self.ierb_quant = quant.QuantStub()
        self.ierb_dequant = quant.DeQuantStub()
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        erb_f = 21.4*np.log10(0.00437*freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        freq_hz = (10**(erb_f/21.4) - 1)/0.00437
        return freq_hz

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1/nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points)/fs*nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                                / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2-2):
            erb_filters[i + 1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12)\
                                                    / (bins[i+1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i + 2])  + 1e-12) \
                                                    / (bins[i + 2] - bins[i+1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1]+1] = 1- erb_filters[-2, bins[-2]:bins[-1]+1]
        
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))
    
    def bm(self, x):
        """x: (B,C,T,F)"""
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_quant(x[..., self.erb_subband_1:])
        x_high = self.erb_fc(x_high)
        x_high = self.erb_dequant(x_high)
        return torch.cat([x_low, x_high], dim=-1)
    
    def bs(self, x_erb):
        """x: (B,C,T,F_erb)"""
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_quant(x_erb[..., self.erb_subband_1:])
        x_erb_high = self.ierb_fc(x_erb_high)
        x_erb_high = self.ierb_dequant(x_erb_high)
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction"""
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1,kernel_size), stride=(1, stride), padding=(0, (kernel_size-1)//2))
        
    def forward(self, x):
        """x: (B,C,T,F)"""
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1]*self.kernel_size, x.shape[2], x.shape[3])
        return xs


class TRA(nn.Module):
    """Temporal Recurrent Attention"""
    def __init__(self, channels, grouped: bool = False):
        super().__init__()
        self.grouped = bool(grouped)
        if self.grouped:
            self.att_gru = GRNN(
                input_size=channels,
                hidden_size=channels * 2,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
                grouped=True,
            )
        else:
            self.att_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_quant = quant.QuantStub()
        self.att_dequant = quant.DeQuantStub()
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x: (B,C,T,F)"""
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.att_gru(zt.transpose(1,2))[0]
        at = self.att_quant(at)
        at = self.att_fc(at)
        at = self.att_dequant(at).transpose(1,2)
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1)

        return x * At


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.use_deconv = use_deconv
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

    def fuse_model(self):
        if self.use_deconv:
            return
        if not isinstance(self.bn, nn.Identity):
            quant.fuse_modules_qat(self, [['conv', 'bn']], inplace=True)

    def forward(self, x, output_size=None):
        x = self.quant(x)
        if self.use_deconv and output_size is not None:
            # Some PyTorch builds don't accept output_size in ConvTranspose2d.forward.
            # Fall back to manual padding/cropping when it's unsupported.
            try:
                x = self.conv(x, output_size)
                used_output_size = True
            except TypeError:
                x = self.conv(x)
                used_output_size = False
            if not used_output_size:
                _, _, target_t, target_f = output_size
                t_diff = target_t - x.shape[2]
                f_diff = target_f - x.shape[3]
                if t_diff < 0:
                    x = x[:, :, :target_t, :]
                elif t_diff > 0:
                    x = nn.functional.pad(x, (0, 0, 0, t_diff))
                if f_diff < 0:
                    x = x[:, :, :, :target_f]
                elif f_diff > 0:
                    x = nn.functional.pad(x, (0, f_diff, 0, 0))
        else:
            x = self.conv(x)
        x = self.dequant(x)
        x = self.bn(x)
        return self.act(x)


class GTConvBlock(nn.Module):
    """Group Temporal Convolution"""
    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        use_deconv=False,
        tra_grouped: bool = False,
    ):
        super().__init__()
        self.use_deconv = use_deconv
        self.pad_size = (kernel_size[0]-1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
    
        self.sfe = SFE(kernel_size=3, stride=1)
        
        self.point_conv1 = conv_module(in_channels//2*3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()
        self.point_quant1 = quant.QuantStub()
        self.point_dequant1 = quant.DeQuantStub()

        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                            stride=stride, padding=padding,
                                            dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()
        self.point_quant2 = quant.QuantStub()
        self.point_dequant2 = quant.DeQuantStub()

        self.point_conv2 = conv_module(hidden_channels, in_channels//2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels//2)
        self.point_quant3 = quant.QuantStub()
        self.point_dequant3 = quant.DeQuantStub()
        
        self.tra = TRA(in_channels // 2, grouped=tra_grouped)

    def fuse_model(self):
        if self.use_deconv:
            return
        fuse_groups = []
        if not isinstance(self.point_bn1, nn.Identity):
            fuse_groups.append(['point_conv1', 'point_bn1'])
        if not isinstance(self.depth_bn, nn.Identity):
            fuse_groups.append(['depth_conv', 'depth_bn'])
        if not isinstance(self.point_bn2, nn.Identity):
            fuse_groups.append(['point_conv2', 'point_bn2'])
        if fuse_groups:
            quant.fuse_modules_qat(self, fuse_groups, inplace=True)

    def shuffle(self, x1, x2):
        """x1, x2: (B,C,T,F)"""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        x = rearrange(x, 'b c g t f -> b (c g) t f')  # (B,2C,T,F)
        return x

    def forward(self, x):
        """x: (B, C, T, F)"""
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        x1 = self.sfe(x1)
        h1 = self.point_quant1(x1)
        h1 = self.point_conv1(h1)
        h1 = self.point_dequant1(h1)
        h1 = self.point_bn1(h1)
        h1 = self.point_act(h1)
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.point_quant2(h1)
        h1 = self.depth_conv(h1)
        h1 = self.point_dequant2(h1)
        h1 = self.depth_bn(h1)
        h1 = self.depth_act(h1)
        h1 = self.point_quant3(h1)
        h1 = self.point_conv2(h1)
        h1 = self.point_dequant3(h1)
        h1 = self.point_bn2(h1)

        h1 = self.tra(h1)

        x =  self.shuffle(h1, x2)
        
        return x


class GRNN(nn.Module):
    """Grouped or standard RNN."""
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        batch_first=True,
        bidirectional=False,
        grouped=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.grouped = grouped

        if self.grouped:
            if input_size % 2 != 0 or hidden_size % 2 != 0:
                raise ValueError(
                    "Grouped GRNN requires even input_size and hidden_size. "
                    f"Got input_size={input_size}, hidden_size={hidden_size}."
                )
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
        else:
            self.rnn = nn.GRU(
                input_size,
                hidden_size,
                num_layers,
                batch_first=batch_first,
                bidirectional=bidirectional,
            )

    def forward(self, x, h=None):
        """
        x: (B, seq_length, input_size)
        h: (num_layers * num_directions, B, hidden_size)
        """
        if h is None:
            num_directions = 2 if self.bidirectional else 1
            h = torch.zeros(
                self.num_layers * num_directions,
                x.shape[0],
                self.hidden_size,
                device=x.device,
            )

        if self.grouped:
            x1, x2 = torch.chunk(x, chunks=2, dim=-1)
            h1, h2 = torch.chunk(h, chunks=2, dim=-1)
            h1, h2 = h1.contiguous(), h2.contiguous()
            y1, h1 = self.rnn1(x1, h1)
            y2, h2 = self.rnn2(x2, h2)
            y = torch.cat([y1, y2], dim=-1)
            h = torch.cat([h1, h2], dim=-1)
            return y, h

        return self.rnn(x, h)
    
    
class DPGRNN(nn.Module):
    """Grouped Dual-path RNN"""
    def __init__(self, input_size, width, hidden_size, grouped_intra_rnn=True, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(
            input_size=input_size,
            hidden_size=hidden_size // 2,
            bidirectional=True,
            grouped=grouped_intra_rnn,
        )
        self.intra_quant = quant.QuantStub()
        self.intra_dequant = quant.DeQuantStub()
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_quant = quant.QuantStub()
        self.inter_dequant = quant.DeQuantStub()
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm(((width, hidden_size)), eps=1e-8)
    
    def forward(self, x):
        """x: (B, C, T, F)"""
        ## Intra RNN
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)[0]  # (B*T,F,C)
        intra_x = self.intra_quant(intra_x)
        intra_x = self.intra_fc(intra_x)      # (B*T,F,C)
        intra_x = self.intra_dequant(intra_x)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size) # (B,T,F,C)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        x = intra_out.permute(0,2,1,3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) 
        inter_x = self.inter_rnn(inter_x)[0]  # (B*F,T,C)
        inter_x = self.inter_quant(inter_x)
        inter_x = self.inter_fc(inter_x)      # (B*F,T,C)
        inter_x = self.inter_dequant(inter_x)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size) # (B,F,T,C)
        inter_x = inter_x.permute(0,2,1,3)   # (B,T,F,C)
        inter_x = self.inter_ln(inter_x) 
        inter_out = torch.add(intra_out, inter_x)
        
        dual_out = inter_out.permute(0,3,1,2)  # (B,C,T,F)
        
        return dual_out


class Encoder(nn.Module):
    def __init__(self, tra_grouped: bool = False):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(3*3, 16, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            GTConvBlock(
                16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False, tra_grouped=tra_grouped
            ),
            GTConvBlock(
                16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False, tra_grouped=tra_grouped
            ),
            GTConvBlock(
                16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False, tra_grouped=tra_grouped
            )
        ])

    def forward(self, x):
        en_outs = []
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)
        return x, en_outs

    def fuse_model(self):
        for layer in self.en_convs:
            if hasattr(layer, 'fuse_model'):
                layer.fuse_model()


class Decoder(nn.Module):
    def __init__(self, tra_grouped: bool = False):
        super().__init__()
        self.de_convs = nn.ModuleList([
            GTConvBlock(
                16, 16, (3,3), stride=(1,1), padding=(2*5,1), dilation=(5,1), use_deconv=True, tra_grouped=tra_grouped
            ),
            GTConvBlock(
                16, 16, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True, tra_grouped=tra_grouped
            ),
            GTConvBlock(
                16, 16, (3,3), stride=(1,1), padding=(2*1,1), dilation=(1,1), use_deconv=True, tra_grouped=tra_grouped
            ),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            ConvBlock(16, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True)
        ])

    def forward(self, x, en_outs, final_freq):
        """
        Decode with skip connections while keeping spatial sizes aligned.

        This is required when n_fft changes because encoder/decoder frequency
        widths are no longer fixed constants.
        """
        n_layers = len(self.de_convs)
        for i in range(n_layers):
            skip = en_outs[n_layers - 1 - i]
            x = self._match_spatial_dims(x, skip)
            x = x + skip

            module = self.de_convs[i]
            output_size = None
            if isinstance(module, ConvBlock) and module.use_deconv:
                if i < n_layers - 1:
                    target_freq = en_outs[n_layers - 2 - i].shape[-1]
                else:
                    target_freq = final_freq
                target_time = x.shape[2]
                output_size = (x.shape[0], module.conv.out_channels, target_time, target_freq)

            if isinstance(module, ConvBlock):
                x = module(x, output_size=output_size)
            else:
                x = module(x)

        return x

    @staticmethod
    def _match_spatial_dims(x, ref):
        t_diff = ref.shape[-2] - x.shape[-2]
        f_diff = ref.shape[-1] - x.shape[-1]

        if t_diff < 0:
            x = x[..., :ref.shape[-2], :]
        elif t_diff > 0:
            x = nn.functional.pad(x, (0, 0, 0, t_diff))

        if f_diff < 0:
            x = x[..., :, :ref.shape[-1]]
        elif f_diff > 0:
            x = nn.functional.pad(x, (0, f_diff, 0, 0))

        return x

    def fuse_model(self):
        for layer in self.de_convs:
            if hasattr(layer, 'fuse_model'):
                layer.fuse_model()
    

class Mask(nn.Module):
    """Complex Ratio Mask"""
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:,0] * mask[:,0] - spec[:,1] * mask[:,1]
        s_imag = spec[:,1] * mask[:,0] + spec[:,0] * mask[:,1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s


class GTCRN(nn.Module):
    BASE_NFFT = 512
    BASE_LOW_BINS = 65
    BASE_HIGH_BANDS = 64

    def __init__(
        self,
        n_fft=512,
        hop_len=256,
        win_len=512,
        grouped_intra_rnn=True,
        tra_grouped: bool = False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len

        erb_low, erb_high = self._compute_erb_subbands(self.n_fft)
        self.erb = ERB(erb_low, erb_high, nfft=self.n_fft)
        self.sfe = SFE(3, 1)

        self.encoder = Encoder(tra_grouped=tra_grouped)
        encoder_width = self._compute_encoder_width(self.erb.output_bins)

        self.dpgrnn1 = DPGRNN(16, encoder_width, 16, grouped_intra_rnn=grouped_intra_rnn)
        self.dpgrnn2 = DPGRNN(16, encoder_width, 16, grouped_intra_rnn=grouped_intra_rnn)
        
        self.decoder = Decoder(tra_grouped=tra_grouped)

        self.mask = Mask()
        self.is_qat_prepared = False
        self.qat_backend = None

    def fuse_model(self):
        self.encoder.fuse_model()
        self.decoder.fuse_model()

    def prepare_qat(
        self,
        backend='fbgemm',
        quantize_conv: bool = True,
        quantize_deconv=False,
        per_channel_weights: bool = False,
        quantize_linear: bool = False,
        quantize_gru: bool = False,
        scale_constraint_mode: str = "none",
        scale_constraint_frac_bits: int | None = None,
        scale_constraint_pow2_rounding: str = "nearest",
    ):
        if self.is_qat_prepared:
            return self

        backend = backend.lower()
        if backend not in ('fbgemm', 'qnnpack'):
            raise ValueError(f"Unsupported quantization backend: {backend}")
        torch.backends.quantized.engine = backend

        self.train()
        self.fuse_model()

        act_qconfig = quant.get_default_qat_qconfig(backend)
        weight_qconfig = quant.get_default_qat_qconfig("fbgemm") if per_channel_weights else act_qconfig

        scale_constraint_mode = str(scale_constraint_mode).strip().lower() if scale_constraint_mode is not None else "none"
        scale_constraint_pow2_rounding = str(scale_constraint_pow2_rounding).strip().lower()
        enable_scale_constraint = scale_constraint_mode not in {"", "none", "off", "false", "0"}

        def _build_constrained_fake_quant(base_fakequant, observer_cls, default_quant_min, default_quant_max):
            kwargs = dict(getattr(getattr(base_fakequant, "p", None), "keywords", {}) or {})
            quant_min = int(kwargs.pop("quant_min", default_quant_min))
            quant_max = int(kwargs.pop("quant_max", default_quant_max))
            kwargs.pop("observer", None)
            return quant.FakeQuantize.with_args(
                observer=observer_cls,
                quant_min=quant_min,
                quant_max=quant_max,
                **kwargs,
                scale_constraint_mode=scale_constraint_mode,
                scale_constraint_frac_bits=scale_constraint_frac_bits,
                scale_constraint_pow2_rounding=scale_constraint_pow2_rounding,
            )

        if enable_scale_constraint:
            # Note: default QAT qconfigs use FusedMovingAvgObsFakeQuantize, which bypasses
            # Python-level observer.calculate_qparams(). To actually constrain scales we
            # must use the non-fused FakeQuantize module.
            act_fakequant = _build_constrained_fake_quant(
                act_qconfig.activation,
                ConstrainedMovingAverageMinMaxObserver,
                default_quant_min=0,
                default_quant_max=255,
            )

            weight_kwargs = dict(getattr(getattr(weight_qconfig.weight, "p", None), "keywords", {}) or {})
            weight_observer = weight_kwargs.get("observer")
            weight_observer_cls = (
                ConstrainedMovingAveragePerChannelMinMaxObserver
                if (
                    isinstance(weight_observer, type)
                    and issubclass(weight_observer, quant.MovingAveragePerChannelMinMaxObserver)
                )
                else ConstrainedMovingAverageMinMaxObserver
            )
            weight_fakequant = _build_constrained_fake_quant(
                weight_qconfig.weight,
                weight_observer_cls,
                default_quant_min=-128,
                default_quant_max=127,
            )
            qat_qconfig = quant.QConfig(activation=act_fakequant, weight=weight_fakequant)
        else:
            qat_qconfig = quant.QConfig(activation=act_qconfig.activation, weight=weight_qconfig.weight)

        if quantize_gru:
            if enable_scale_constraint:
                default_gru_qat_qconfig = quant.default_dynamic_qat_qconfig
                gru_qat_qconfig = quant.QConfig(
                    activation=_build_constrained_fake_quant(
                        default_gru_qat_qconfig.activation,
                        ConstrainedMovingAverageMinMaxObserver,
                        default_quant_min=0,
                        default_quant_max=255,
                    ),
                    weight=_build_constrained_fake_quant(
                        default_gru_qat_qconfig.weight,
                        ConstrainedMovingAverageMinMaxObserver,
                        default_quant_min=-128,
                        default_quant_max=127,
                    ),
                )
            else:
                gru_qat_qconfig = quant.default_dynamic_qat_qconfig
        else:
            gru_qat_qconfig = None
        deconv_qconfig = None
        deconv_qconfig_axis1 = None
        if quantize_deconv:
            if per_channel_weights and backend == "fbgemm":
                raise ValueError(
                    "per_channel_weights=True with quantize_deconv=True requires backend='qnnpack' "
                    "(FBGEMM conversion disables per-channel weights for ConvTranspose2d)."
                )

            if per_channel_weights:
                deconv_qconfig = qat_qconfig
                # ConvTranspose2d weight layout is (in_channels, out_channels/groups, kH, kW).
                # QNNPACK expects per-channel quantization over dim=1 when groups==1 (out_channels),
                # but over dim=0 when groups>1 (in_channels). We select between these below.
                deconv_weight_axis1_kwargs = dict(
                    observer=ConstrainedMovingAveragePerChannelMinMaxObserver
                    if enable_scale_constraint
                    else quant.MovingAveragePerChannelMinMaxObserver,
                    quant_min=-128,
                    quant_max=127,
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric,
                    reduce_range=False,
                    ch_axis=1,
                )
                if enable_scale_constraint:
                    deconv_weight_axis1_kwargs.update(
                        scale_constraint_mode=scale_constraint_mode,
                        scale_constraint_frac_bits=scale_constraint_frac_bits,
                        scale_constraint_pow2_rounding=scale_constraint_pow2_rounding,
                    )
                deconv_weight_axis1 = quant.FakeQuantize.with_args(**deconv_weight_axis1_kwargs)
                deconv_qconfig_axis1 = quant.QConfig(
                    activation=qat_qconfig.activation if enable_scale_constraint else act_qconfig.activation,
                    weight=deconv_weight_axis1,
                )
            else:
                # Quantized ConvTranspose2d conversion under FBGEMM disables per-channel weight quantization.
                # Keep per-tensor weight fake-quant for deconvs (via QNNPACK defaults) so conversion succeeds.
                per_tensor_weight_qconfig = quant.get_default_qat_qconfig("qnnpack")
                if enable_scale_constraint:
                    per_tensor_weight = _build_constrained_fake_quant(
                        per_tensor_weight_qconfig.weight,
                        ConstrainedMovingAverageMinMaxObserver,
                        default_quant_min=-128,
                        default_quant_max=127,
                    )
                else:
                    per_tensor_weight = per_tensor_weight_qconfig.weight
                deconv_qconfig = quant.QConfig(
                    activation=qat_qconfig.activation if enable_scale_constraint else act_qconfig.activation,
                    weight=per_tensor_weight,
                )

        for module in self.modules():
            module.qconfig = None

        for module in self.modules():
            if quantize_conv and isinstance(module, ConvBlock):
                if module.use_deconv:
                    if not quantize_deconv:
                        continue
                    if per_channel_weights and isinstance(module.conv, nn.ConvTranspose2d) and module.conv.groups == 1:
                        module.conv.qconfig = deconv_qconfig_axis1
                    else:
                        module.conv.qconfig = deconv_qconfig
                    module.quant.qconfig = qat_qconfig
                    module.dequant.qconfig = qat_qconfig
                    continue

                module.conv.qconfig = qat_qconfig
                module.quant.qconfig = qat_qconfig
                module.dequant.qconfig = qat_qconfig
                continue

            if quantize_conv and isinstance(module, GTConvBlock):
                if module.use_deconv:
                    if not quantize_deconv:
                        continue
                    point_deconv_qconfig = (
                        deconv_qconfig_axis1
                        if per_channel_weights and isinstance(module.point_conv1, nn.ConvTranspose2d) and module.point_conv1.groups == 1
                        else deconv_qconfig
                    )
                    depth_deconv_qconfig = deconv_qconfig
                else:
                    point_deconv_qconfig = qat_qconfig
                    depth_deconv_qconfig = qat_qconfig

                module.point_quant1.qconfig = qat_qconfig
                module.point_quant2.qconfig = qat_qconfig
                module.point_quant3.qconfig = qat_qconfig
                module.point_dequant1.qconfig = qat_qconfig
                module.point_dequant2.qconfig = qat_qconfig
                module.point_dequant3.qconfig = qat_qconfig

                module.point_conv1.qconfig = point_deconv_qconfig
                module.depth_conv.qconfig = depth_deconv_qconfig
                module.point_conv2.qconfig = point_deconv_qconfig
                continue

            if quantize_linear and isinstance(module, TRA):
                module.att_quant.qconfig = qat_qconfig
                module.att_fc.qconfig = qat_qconfig
                module.att_dequant.qconfig = qat_qconfig
                continue

            if quantize_linear and isinstance(module, DPGRNN):
                module.intra_quant.qconfig = qat_qconfig
                module.intra_fc.qconfig = qat_qconfig
                module.intra_dequant.qconfig = qat_qconfig
                module.inter_quant.qconfig = qat_qconfig
                module.inter_fc.qconfig = qat_qconfig
                module.inter_dequant.qconfig = qat_qconfig
                continue

            if quantize_gru and isinstance(module, nn.GRU):
                module.qconfig = gru_qat_qconfig

        quant.prepare_qat(self, inplace=True)
        self.is_qat_prepared = True
        self.qat_backend = backend
        return self

    @staticmethod
    def _normalize_scale_constraint_mode(mode):
        if mode is None:
            return "none"
        return str(mode).strip().lower()

    @staticmethod
    def _apply_scale_constraint_to_scale(scale, *, mode, frac_bits, pow2_rounding):
        mode = GTCRN._normalize_scale_constraint_mode(mode)
        if mode in {"", "none", "off", "false", "0"}:
            return scale
        if mode in {"pow2", "power2", "power_of_two", "power-of-two"}:
            return _constrain_scale_pow2(scale, pow2_rounding)
        if mode in {"fixed", "qmn", "fixed_qmn", "qformat"}:
            if frac_bits is None:
                raise ValueError("frac_bits is required for mode='fixed'")
            return _constrain_scale_fixed(scale, int(frac_bits))
        raise ValueError(f"Unsupported scale constraint mode: {mode}")

    @staticmethod
    def _extract_gru_input_qparams(gru_module):
        default = {
            "scale": 1.0,
            "zero_point": 0,
            "quant_min": 0,
            "quant_max": 255,
        }
        fake_quant = getattr(gru_module, "activation_post_process", None)
        if fake_quant is None:
            return default

        scale = getattr(fake_quant, "scale", None)
        zero_point = getattr(fake_quant, "zero_point", None)
        if scale is None or scale.numel() == 0:
            observer = getattr(fake_quant, "activation_post_process", None)
            if observer is not None and hasattr(observer, "calculate_qparams"):
                scale, zero_point = observer.calculate_qparams()

        if scale is None or zero_point is None:
            return default

        scale_value = float(scale.detach().reshape(-1)[0].item())
        if (not np.isfinite(scale_value)) or (scale_value <= 0.0):
            scale_value = default["scale"]

        zero_point_value = int(zero_point.detach().reshape(-1)[0].item())
        quant_min = int(getattr(fake_quant, "quant_min", default["quant_min"]))
        quant_max = int(getattr(fake_quant, "quant_max", default["quant_max"]))
        return {
            "scale": scale_value,
            "zero_point": zero_point_value,
            "quant_min": quant_min,
            "quant_max": quant_max,
        }

    @staticmethod
    def _set_named_module(root_module, module_name, new_module):
        if not module_name:
            raise ValueError("module_name cannot be empty")

        parent = root_module
        parts = module_name.split(".")
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        leaf = parts[-1]
        if isinstance(parent, (nn.ModuleList, nn.Sequential)) and leaf.isdigit():
            parent[int(leaf)] = new_module
        elif isinstance(parent, nn.ModuleDict):
            parent[leaf] = new_module
        elif leaf.isdigit() and hasattr(parent, "__setitem__"):
            parent[int(leaf)] = new_module
        else:
            setattr(parent, leaf, new_module)

    @staticmethod
    def _build_constrained_fake_quant_from_existing(
        fake_quant_module,
        *,
        mode,
        frac_bits,
        pow2_rounding,
    ):
        observer = getattr(fake_quant_module, "activation_post_process", None)
        if observer is None:
            return None

        qscheme = getattr(observer, "qscheme", None)
        is_per_channel = qscheme in {
            torch.per_channel_affine,
            torch.per_channel_symmetric,
            torch.per_channel_affine_float_qparams,
        }
        observer_cls = (
            ConstrainedMovingAveragePerChannelMinMaxObserver
            if is_per_channel
            else ConstrainedMovingAverageMinMaxObserver
        )

        fq_kwargs = {"dtype": getattr(observer, "dtype", torch.quint8)}
        if qscheme is not None:
            fq_kwargs["qscheme"] = qscheme
        if hasattr(observer, "reduce_range"):
            fq_kwargs["reduce_range"] = bool(getattr(observer, "reduce_range"))
        if is_per_channel and hasattr(observer, "ch_axis"):
            fq_kwargs["ch_axis"] = int(getattr(observer, "ch_axis"))
        if hasattr(observer, "is_dynamic"):
            fq_kwargs["is_dynamic"] = bool(getattr(observer, "is_dynamic"))
        if hasattr(observer, "averaging_constant"):
            fq_kwargs["averaging_constant"] = float(getattr(observer, "averaging_constant"))

        constrained = quant.FakeQuantize(
            observer=observer_cls,
            quant_min=int(getattr(fake_quant_module, "quant_min", 0)),
            quant_max=int(getattr(fake_quant_module, "quant_max", 255)),
            **fq_kwargs,
            scale_constraint_mode=mode,
            scale_constraint_frac_bits=frac_bits,
            scale_constraint_pow2_rounding=pow2_rounding,
        )
        scale = getattr(fake_quant_module, "scale", None)
        if torch.is_tensor(scale):
            constrained.to(scale.device)
        constrained.train(fake_quant_module.training)
        constrained.load_state_dict(fake_quant_module.state_dict(), strict=False)
        return constrained

    def enable_scale_constraints(self, mode="pow2", frac_bits=None, pow2_rounding="nearest"):
        """
        Enable/update scale constraints on already-prepared QAT fake-quant modules.

        This supports phased training schedules where QAT starts with unconstrained
        fake-quant modules and constraints are enabled later without rebuilding the model.
        Returns (replaced_count, updated_count).
        """
        if not self.is_qat_prepared:
            raise RuntimeError("Scale constraints can be enabled only after prepare_qat(...).")

        mode = self._normalize_scale_constraint_mode(mode)
        if mode in {"", "none", "off", "false", "0"}:
            return 0, 0
        pow2_rounding = str(pow2_rounding).strip().lower() if pow2_rounding is not None else "nearest"

        replaced_count = 0
        updated_count = 0

        def _visit(parent):
            nonlocal replaced_count, updated_count
            for child_name, child in list(parent.named_children()):
                replacement = None
                if isinstance(child, quant.FusedMovingAvgObsFakeQuantize):
                    replacement = self._build_constrained_fake_quant_from_existing(
                        child,
                        mode=mode,
                        frac_bits=frac_bits,
                        pow2_rounding=pow2_rounding,
                    )
                elif isinstance(child, quant.FakeQuantize):
                    observer = getattr(child, "activation_post_process", None)
                    if isinstance(
                        observer,
                        (ConstrainedMovingAverageMinMaxObserver, ConstrainedMovingAveragePerChannelMinMaxObserver),
                    ):
                        observer.scale_constraint_mode = mode
                        observer.scale_constraint_frac_bits = frac_bits
                        observer.scale_constraint_pow2_rounding = pow2_rounding
                        updated_count += 1
                    elif isinstance(
                        observer,
                        (quant.MovingAverageMinMaxObserver, quant.MovingAveragePerChannelMinMaxObserver),
                    ):
                        replacement = self._build_constrained_fake_quant_from_existing(
                            child,
                            mode=mode,
                            frac_bits=frac_bits,
                            pow2_rounding=pow2_rounding,
                        )

                if replacement is not None:
                    setattr(parent, child_name, replacement)
                    child = replacement
                    replaced_count += 1

                _visit(child)

        _visit(self)
        return replaced_count, updated_count

    def convert_qat(
        self,
        inplace=False,
        dynamic_quantize_gru=False,
        static_quantize_gru=False,
        static_gru_weight_bit_width: int = 8,
        dynamic_gru_dtype=torch.qint8,
        dynamic_gru_scale_constraint_mode: str = "none",
        dynamic_gru_scale_constraint_frac_bits: int | None = None,
        dynamic_gru_scale_constraint_pow2_rounding: str = "nearest",
    ):
        model = self if inplace else deepcopy(self)
        if not model.is_qat_prepared:
            raise RuntimeError("Call prepare_qat(...) and load a QAT checkpoint before conversion.")
        if next(model.parameters()).device.type != 'cpu':
            raise RuntimeError("Quantized conversion only supports CPU execution. Move model to CPU first.")
        if dynamic_quantize_gru and static_quantize_gru:
            raise ValueError("dynamic_quantize_gru and static_quantize_gru are mutually exclusive.")
        if static_quantize_gru and dynamic_gru_dtype != torch.qint8:
            raise ValueError("static_quantize_gru currently supports only dynamic_gru_dtype=torch.qint8.")

        if model.qat_backend is not None:
            torch.backends.quantized.engine = model.qat_backend

        static_gru_qparams = {}
        if static_quantize_gru:
            constraint_mode = (
                str(dynamic_gru_scale_constraint_mode).strip().lower()
                if dynamic_gru_scale_constraint_mode is not None
                else "none"
            )
            for module_name, module in model.named_modules():
                if not isinstance(module, nn.GRU):
                    continue
                qparams = model._extract_gru_input_qparams(module)
                constrained_scale = model._apply_scale_constraint_to_scale(
                    torch.tensor([qparams["scale"]], dtype=torch.float32),
                    mode=constraint_mode,
                    frac_bits=dynamic_gru_scale_constraint_frac_bits,
                    pow2_rounding=dynamic_gru_scale_constraint_pow2_rounding,
                )
                scale_value = float(torch.clamp(constrained_scale, min=1e-8).item())
                qparams["scale"] = scale_value
                static_gru_qparams[module_name] = qparams

        model.eval()
        quant.convert(model, inplace=True)
        if static_quantize_gru:
            constraint_mode = (
                str(dynamic_gru_scale_constraint_mode).strip().lower()
                if dynamic_gru_scale_constraint_mode is not None
                else "none"
            )
            replaced = 0
            for module_name, module in list(model.named_modules()):
                if not isinstance(module, nn.GRU):
                    continue
                qparams = static_gru_qparams.get(
                    module_name,
                    {"scale": 1.0, "zero_point": 0, "quant_min": 0, "quant_max": 255},
                )
                static_gru = StaticQuantizedGRU(
                    module,
                    input_scale=float(qparams["scale"]),
                    input_zero_point=int(qparams["zero_point"]),
                    act_quant_min=int(qparams["quant_min"]),
                    act_quant_max=int(qparams["quant_max"]),
                    weight_bit_width=int(static_gru_weight_bit_width),
                    weight_scale_constraint_mode=constraint_mode,
                    weight_scale_constraint_frac_bits=dynamic_gru_scale_constraint_frac_bits,
                    weight_scale_constraint_pow2_rounding=dynamic_gru_scale_constraint_pow2_rounding,
                )
                GTCRN._set_named_module(model, module_name, static_gru)
                replaced += 1
            if replaced == 0:
                raise RuntimeError("static_quantize_gru=True but no nn.GRU modules were found after conversion.")
        elif dynamic_quantize_gru:
            constraint_mode = (
                str(dynamic_gru_scale_constraint_mode).strip().lower()
                if dynamic_gru_scale_constraint_mode is not None
                else "none"
            )
            enable_dynamic_gru_scale_constraint = constraint_mode not in {"", "none", "off", "false", "0"}
            if enable_dynamic_gru_scale_constraint and dynamic_gru_dtype != torch.qint8:
                raise ValueError(
                    "Dynamic GRU scale constraint currently supports only dynamic_gru_dtype=torch.qint8."
                )
            if enable_dynamic_gru_scale_constraint:
                constrained_dynamic_gru_qconfig = quant.QConfig(
                    activation=quant.default_dynamic_qconfig.activation,
                    weight=ConstrainedMovingAverageMinMaxObserver.with_args(
                        dtype=torch.qint8,
                        qscheme=torch.per_tensor_symmetric,
                        scale_constraint_mode=constraint_mode,
                        scale_constraint_frac_bits=dynamic_gru_scale_constraint_frac_bits,
                        scale_constraint_pow2_rounding=dynamic_gru_scale_constraint_pow2_rounding,
                    ),
                )
                # Dynamic GRU kernels compute activation qparams internally at runtime.
                # This constrained qconfig affects GRU weight packing scales.
                qconfig_spec = {nn.GRU: constrained_dynamic_gru_qconfig}
                quant.quantize_dynamic(
                    model,
                    qconfig_spec=qconfig_spec,
                    inplace=True,
                )
            else:
                quant.quantize_dynamic(
                    model,
                    {nn.GRU},
                    dtype=dynamic_gru_dtype,
                    inplace=True,
                )
        model.is_qat_prepared = False
        return model

    def forward(self, x):
        """
        x: (B, L)
        """
        device = x.device
        n_samples = x.shape[1]
        
        stft_kwargs = {'n_fft': self.n_fft, 'hop_length': self.hop_len, 'win_length': self.win_len,
                       'window': torch.hann_window(self.win_len).to(device), 'onesided': True}
        
        spec = torch.stft(x,  **stft_kwargs, return_complex=True)
        spec = torch.view_as_real(spec)

        spec_real = spec[..., 0].permute(0,2,1)
        spec_imag = spec[..., 1].permute(0,2,1)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,F)
        
        spec = spec.permute(0,3,2,1)  # (B,2,T,F)

        feat = self.erb.bm(feat)
        feat = self.sfe(feat)

        feat, en_outs = self.encoder(feat)
        
        feat = self.dpgrnn1(feat)
        feat = self.dpgrnn2(feat)

        m_feat = self.decoder(feat, en_outs, self.erb.output_bins)
        
        m = self.erb.bs(m_feat)

        spec_enh = self.mask(m, spec) # (B,2,T,F)
        spec_enh = spec_enh.permute(0,3,2,1)  # (B,F,T,2)
        
        spec_enh = torch.complex(spec_enh[...,0], spec_enh[...,1])
        output = torch.istft(spec_enh, **stft_kwargs)
        output = torch.nn.functional.pad(output, (0, n_samples-output.shape[1]))
        
        return output

    @classmethod
    def _compute_erb_subbands(cls, n_fft):
        base_nfreqs = cls.BASE_NFFT // 2 + 1
        low_ratio = cls.BASE_LOW_BINS / base_nfreqs
        high_ratio = cls.BASE_HIGH_BANDS / max(base_nfreqs - cls.BASE_LOW_BINS, 1)

        nfreqs = n_fft // 2 + 1
        if nfreqs < 3:
            raise ValueError(f"n_fft={n_fft} is too small to build ERB filters")

        erb_low = int(round(nfreqs * low_ratio))
        erb_low = max(1, min(erb_low, nfreqs - 2))

        remaining_bins = nfreqs - erb_low
        erb_high = int(round(remaining_bins * high_ratio))
        erb_high = max(2, min(erb_high, remaining_bins))

        return erb_low, erb_high

    @staticmethod
    def _compute_encoder_width(freq_bins):
        def conv_out(width, kernel_size, stride, padding, dilation=1):
            return max(1, (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1)

        width = freq_bins
        width = conv_out(width, kernel_size=5, stride=2, padding=2)
        width = conv_out(width, kernel_size=5, stride=2, padding=2)
        return width


if __name__ == "__main__":
    model = GTCRN().eval()

    """complexity count"""
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (16000,), as_strings=True,
                                            print_per_layer_stat=False, verbose=True)
    params = 0
    for p in model.parameters():
        params += p.numel()
    print(flops, params/1e3)

    """causality check"""
    a = torch.randn(1, 16000)
    b = torch.randn(1, 16000)
    c = torch.randn(1, 16000)
    x1 = torch.cat([a, b], dim=1)
    x2 = torch.cat([a, c], dim=1)

    y1 = model(x1)[0]
    y2 = model(x2)[0]

    print((y1[:16000-256*2] - y2[:16000-256*2]).abs().max())
    print((y1[16000:] - y2[16000:]).abs().max())
