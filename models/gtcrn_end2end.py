"""
GTCRN: ShuffleNetV2 + SFE + TRA + 2 DPGRNN
Ultra tiny, 33.0 MMACs, 23.67 K params
"""
import os
import torch
import numpy as np
import torch.nn as nn
from collections.abc import Mapping, Sequence
from typing import Optional, Callable, Any
from einops import rearrange

from .spectral_preprocess import SpectralPreprocessor


def _default_gru_factory(input_size: int, hidden_size: int, **kwargs) -> nn.Module:
    kwargs.pop("threshold_x", None)
    kwargs.pop("threshold_h", None)
    return nn.GRU(input_size, hidden_size, **kwargs)


def _to_plain(obj):
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_plain(v) for v in obj)
    return obj


def _normalize_threshold_entry(entry: Any, default_x: Optional[float], default_h: Optional[float]) -> dict[str, Optional[float]]:
    if isinstance(entry, Mapping):
        x_val = entry.get("x")
        h_val = entry.get("h")
        x = default_x if x_val is None else x_val
        h = default_h if h_val is None else h_val
    elif entry is None:
        x, h = default_x, default_h
    else:
        x = entry
        h = entry
    return {"x": x, "h": h}


def _normalize_threshold_list(values: Any, count: int, default_x: Optional[float], default_h: Optional[float]) -> list[dict[str, Optional[float]]]:
    seq: Sequence[Any]
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        seq = values  # type: ignore[assignment]
    else:
        seq = []
    normalized: list[dict[str, Optional[float]]] = []
    for idx in range(count):
        entry = seq[idx] if idx < len(seq) else None
        normalized.append(_normalize_threshold_entry(entry, default_x, default_h))
    return normalized


def _normalize_dp_thresholds(values: Any, default_x: Optional[float], default_h: Optional[float]) -> dict[str, dict[str, Optional[float]]]:
    mapping = values if isinstance(values, Mapping) else {}
    return {
        "intra_rnn1": _normalize_threshold_entry(mapping.get("intra_rnn1"), default_x, default_h),
        "intra_rnn2": _normalize_threshold_entry(mapping.get("intra_rnn2"), default_x, default_h),
        "inter_rnn1": _normalize_threshold_entry(mapping.get("inter_rnn1"), default_x, default_h),
        "inter_rnn2": _normalize_threshold_entry(mapping.get("inter_rnn2"), default_x, default_h),
    }


class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft//2 + 1
        self.erb_subband_1 = erb_subband_1
        self.output_bins = erb_subband_1 + erb_subband_2
        self.erb_fc = nn.Linear(nfreqs-erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs-erb_subband_1, bias=False)
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
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)
    
    def bs(self, x_erb):
        """x: (B,C,T,F_erb)"""
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
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
    def __init__(
        self,
        channels,
        gru_factory: Optional[Callable[..., nn.Module]] = None,
        delta_threshold: Optional[dict[str, Optional[float]]] = None,
        log_gru_inputs: bool = False,
        log_file_base: Optional[str] = None,
    ):
        super().__init__()
        if gru_factory is None:
            gru_factory = _default_gru_factory
        threshold_x = (delta_threshold or {}).get("x")
        threshold_h = (delta_threshold or {}).get("h")
        self.log_gru_inputs = log_gru_inputs
        self.log_file_base = log_file_base
        self._log_paths: dict[str, str] | None = None
        self._log_buffers: dict[str, list[np.ndarray]] | None = None
        if self.log_gru_inputs and self.log_file_base is not None:
            base, ext = os.path.splitext(self.log_file_base)
            ext = ext if ext else ".pkl"
            self._log_paths = {
                "x": f"{base}_x{ext}",
                "h": f"{base}_h{ext}",
            }
            for path in self._log_paths.values():
                directory = os.path.dirname(path)
                if directory:
                    os.makedirs(directory, exist_ok=True)
            self._log_buffers = {key: [] for key in self._log_paths}
        self.att_gru = gru_factory(
            channels,
            channels * 2,
            num_layers=1,
            batch_first=True,
            threshold_x=threshold_x,
            threshold_h=threshold_h,
        )
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x: (B,C,T,F)"""
        logging_active = (
            self.log_gru_inputs
            and self._log_paths is not None
            and self._log_buffers is not None
            and not self.training
        )
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        gru_input = zt.transpose(1,2)  # (B,T,C)
        if logging_active:
            x_vectors = gru_input.detach().cpu().reshape(-1, gru_input.shape[-1]).numpy()
            self._log_buffers["x"].append(x_vectors)
        at = self.att_gru(gru_input)[0]
        if logging_active:
            h_vectors = at.detach().cpu().reshape(-1, at.shape[-1]).numpy()
            self._log_buffers["h"].append(h_vectors)
            import pickle
            for key, path in self._log_paths.items():
                buffers = self._log_buffers.get(key, [])
                if not buffers:
                    continue
                payload = np.concatenate(buffers, axis=0)
                with open(path, "wb") as f:
                    pickle.dump(payload, f)
            self._log_buffers = {key: [] for key in self._log_paths}
        at = self.att_fc(at).transpose(1,2)
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
    def forward(self, x, output_size=None):
        if self.use_deconv and output_size is not None:
            x = self.conv(x, output_size=output_size)
        else:
            x = self.conv(x)
        return self.act(self.bn(x))


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
        gru_factory: Optional[Callable[..., nn.Module]] = None,
        tra_threshold: Optional[dict[str, Optional[float]]] = None,
        log_gru_inputs: bool = False,
        log_file_base: Optional[str] = None,
    ):
        super().__init__()
        self.use_deconv = use_deconv
        self.pad_size = (kernel_size[0]-1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
    
        self.sfe = SFE(kernel_size=3, stride=1)
        
        self.point_conv1 = conv_module(in_channels//2*3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                            stride=stride, padding=padding,
                                            dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = conv_module(hidden_channels, in_channels//2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels//2)
        
        self.tra = TRA(
            in_channels//2,
            gru_factory=gru_factory,
            delta_threshold=tra_threshold,
            log_gru_inputs=log_gru_inputs,
            log_file_base=log_file_base,
        )

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
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        h1 = self.point_bn2(self.point_conv2(h1))

        h1 = self.tra(h1)

        x =  self.shuffle(h1, x2)
        
        return x


class GRNN(nn.Module):
    """Grouped RNN"""
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        batch_first=True,
        bidirectional=False,
        log_gru_inputs=False,
        log_file=None,
        gru_factory: Optional[Callable[..., nn.Module]] = None,
        rnn_thresholds: Optional[Sequence[dict[str, Optional[float]]]] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        if gru_factory is None:
            gru_factory = _default_gru_factory
        thresholds = list(rnn_thresholds) if rnn_thresholds is not None else [{"x": None, "h": None}] * 2
        while len(thresholds) < 2:
            thresholds.append({"x": None, "h": None})
        self.rnn1 = gru_factory(
            input_size//2,
            hidden_size//2,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            threshold_x=thresholds[0].get("x"),
            threshold_h=thresholds[0].get("h"),
        )
        self.rnn2 = gru_factory(
            input_size//2,
            hidden_size//2,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            threshold_x=thresholds[1].get("x"),
            threshold_h=thresholds[1].get("h"),
        )
        self.log_gru_inputs = log_gru_inputs
        self.log_file = log_file
        self._log_keys = ("x1", "h1", "x2", "h2")
        self._log_paths: dict[str, str] | None = None
        self._log_buffers: dict[str, list] | None = None
        if self.log_gru_inputs and self.log_file is not None:
            base, ext = os.path.splitext(self.log_file)
            ext = ext if ext else ".pkl"
            self._log_paths = {
                "x1": f"{base}_x1{ext}",
                "h1": f"{base}_h1{ext}",
                "x2": f"{base}_x2{ext}",
                "h2": f"{base}_h2{ext}",
            }
            for path in self._log_paths.values():
                directory = os.path.dirname(path)
                if directory:
                    os.makedirs(directory, exist_ok=True)
            self._log_buffers = {key: [] for key in self._log_keys}

    def forward(self, x, h=None):
        """
        x: (B, seq_length, input_size)
        h: (num_layers, B, hidden_size)
        """
        if h== None:
            if self.bidirectional:
                h = torch.zeros(self.num_layers*2, x.shape[0], self.hidden_size, device=x.device)
            else:
                h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        logging_active = (
            self.log_gru_inputs
            and self.log_file is not None
            and not self.training
            and self._log_buffers is not None
        )
        if logging_active:
            x1_vectors = x1.detach().cpu().reshape(-1, x1.shape[-1]).numpy()
            x2_vectors = x2.detach().cpu().reshape(-1, x2.shape[-1]).numpy()
            self._log_buffers["x1"].append(x1_vectors)
            self._log_buffers["x2"].append(x2_vectors)
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        if logging_active:
            # y1/y2 tensors hold the hidden activations at every timestep.
            if y1.dim() == 3:
                y1_vectors = y1.detach().cpu().permute(1, 0, 2).reshape(-1, y1.shape[-1]).numpy()
            else:
                y1_vectors = y1.detach().cpu().reshape(-1, y1.shape[-1]).numpy()
            if y2.dim() == 3:
                y2_vectors = y2.detach().cpu().permute(1, 0, 2).reshape(-1, y2.shape[-1]).numpy()
            else:
                y2_vectors = y2.detach().cpu().reshape(-1, y2.shape[-1]).numpy()
            self._log_buffers["h1"].append(y1_vectors)
            self._log_buffers["h2"].append(y2_vectors)
        if logging_active and self._log_paths is not None:
            import pickle
            for key, path in self._log_paths.items():
                buffers = self._log_buffers.get(key, [])
                if not buffers:
                    continue
                payload = np.concatenate(buffers, axis=0)
                with open(path, 'wb') as f:
                    pickle.dump(payload, f)
            self._log_buffers = {key: [] for key in self._log_keys}
        return y, h


class DeltaGRU(nn.Module):
    """
    Wraps nn.GRU but conditionally reuses previous inputs/hidden states based on per-element deltas.

    For each feature, if |x(t) - x(t-1)| < threshold_x we reuse x(t-1); otherwise we keep x(t).
    The same per-element rule applies to h(t-1) vs h(t-2) with threshold_h.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        threshold_x: Optional[float] = None,
        threshold_h: Optional[float] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=False,
            dropout=dropout,
            bidirectional=bidirectional,
            **factory_kwargs,
        )
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.threshold_x = threshold_x
        self.threshold_h = threshold_h

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        if x.dim() != 3:
            raise ValueError(f"Expected 3-D input (B,T,C) or (T,B,C), got {tuple(x.shape)}.")

        self.gru.flatten_parameters()
        time_major = x.transpose(0, 1) if self.batch_first else x  # (T,B,C)
        seq_len, batch_size, _ = time_major.shape

        if h0 is None:
            h0 = self._init_hidden(batch_size, x)

        if self.num_directions == 1:
            outputs, h_last = self._run_direction(time_major, h0, reverse=False)
        else:
            h0_view = h0.reshape(self.num_layers, self.num_directions, batch_size, self.hidden_size)
            fwd_outputs, fwd_hidden = self._run_direction(time_major, h0_view[:, 0], reverse=False)
            bwd_outputs, bwd_hidden = self._run_direction(time_major.flip(0), h0_view[:, 1], reverse=True)
            outputs = torch.cat([fwd_outputs, bwd_outputs], dim=-1)
            h_last = torch.stack([fwd_hidden, bwd_hidden], dim=1).reshape(
                self.num_layers * self.num_directions, batch_size, self.hidden_size
            )

        output = outputs.transpose(0, 1) if self.batch_first else outputs
        return output, h_last

    def _init_hidden(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        device = reference.device
        dtype = reference.dtype
        return torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
            dtype=dtype,
        )

    def _threshold_active(self, value: Optional[float]) -> bool:
        return value is not None and value > 0

    def _run_direction(
        self,
        seq: torch.Tensor,
        h0: torch.Tensor,
        reverse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len, batch_size, _ = seq.shape
        prev_x_actual: Optional[torch.Tensor] = None
        prev_hidden = [h0[layer] for layer in range(self.num_layers)]
        prev_prev_hidden: list[Optional[torch.Tensor]] = [None] * self.num_layers
        outputs: list[torch.Tensor] = []

        for t in range(seq_len):
            x_curr_actual = seq[t]
            x_t = self._gate_input(x_curr_actual, prev_x_actual)
            prev_x_actual = x_curr_actual

            layer_input = x_t
            for layer in range(self.num_layers):
                h_prev = prev_hidden[layer]
                h_prev_prev = prev_prev_hidden[layer]
                h_in = self._gate_hidden(h_prev, h_prev_prev)
                weight_ih, weight_hh, bias_ih, bias_hh = self._get_gru_params(layer, reverse)
                layer_input = self._gru_cell(layer_input, h_in, weight_ih, weight_hh, bias_ih, bias_hh)
                prev_prev_hidden[layer] = h_prev
                prev_hidden[layer] = layer_input
            outputs.append(layer_input)

        outputs_tensor = torch.stack(outputs, dim=0)
        if reverse:
            outputs_tensor = outputs_tensor.flip(0)
        hidden_tensor = torch.stack(prev_hidden, dim=0)
        return outputs_tensor, hidden_tensor

    def _get_gru_params(self, layer: int, reverse: bool):
        suffix = "" if not reverse else "_reverse"
        weight_ih = getattr(self.gru, f"weight_ih_l{layer}{suffix}")
        weight_hh = getattr(self.gru, f"weight_hh_l{layer}{suffix}")
        bias_ih = getattr(self.gru, f"bias_ih_l{layer}{suffix}", None)
        bias_hh = getattr(self.gru, f"bias_hh_l{layer}{suffix}", None)
        return weight_ih, weight_hh, bias_ih, bias_hh

    def _gru_cell(
        self,
        input_t: torch.Tensor,
        hidden: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: Optional[torch.Tensor],
        bias_hh: Optional[torch.Tensor],
    ) -> torch.Tensor:
        gi = torch.matmul(input_t, weight_ih.t())
        gh = torch.matmul(hidden, weight_hh.t())
        if bias_ih is not None:
            gi = gi + bias_ih
        if bias_hh is not None:
            gh = gh + bias_hh
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)
        resetgate = torch.sigmoid(i_r + h_r)
        updategate = torch.sigmoid(i_z + h_z)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + updategate * (hidden - newgate)
        return hy

    def _gate_input(self, current: torch.Tensor, prev_actual: Optional[torch.Tensor]) -> torch.Tensor:
        if prev_actual is None or not self._threshold_active(self.threshold_x):
            return current
        delta = (current - prev_actual).abs()
        reuse_prev = delta < self.threshold_x
        return torch.where(reuse_prev, prev_actual, current)

    def _gate_hidden(
        self,
        prev_actual: torch.Tensor,
        prev_prev_actual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if prev_prev_actual is None or not self._threshold_active(self.threshold_h):
            return prev_actual
        delta = (prev_actual - prev_prev_actual).abs()
        reuse_prev = delta < self.threshold_h
        return torch.where(reuse_prev, prev_prev_actual, prev_actual)
    
    
class DPGRNN(nn.Module):
    """Grouped Dual-path RNN"""
    def __init__(
        self,
        input_size,
        width,
        hidden_size,
        log_gru_inputs=False,
        log_file_base=None,
        gru_factory: Optional[Callable[..., nn.Module]] = None,
        intra_rnn_thresholds: Optional[Sequence[dict[str, Optional[float]]]] = None,
        inter_rnn_thresholds: Optional[Sequence[dict[str, Optional[float]]]] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size
        # Prepare distinct log file names for intra/inter if a base is provided
        if log_file_base is not None:
            import os
            root, ext = os.path.splitext(log_file_base)
            intra_log = f"{root}_intra{ext}"
            inter_log = f"{root}_inter{ext}"
        else:
            intra_log = None
            inter_log = None

        self.intra_rnn = GRNN(
            input_size=input_size,
            hidden_size=hidden_size//2,
            bidirectional=True,
            log_gru_inputs=log_gru_inputs,
            log_file=intra_log,
            gru_factory=gru_factory,
            rnn_thresholds=intra_rnn_thresholds,
        )
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=False,
            log_gru_inputs=log_gru_inputs,
            log_file=inter_log,
            gru_factory=gru_factory,
            rnn_thresholds=inter_rnn_thresholds,
        )
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm(((width, hidden_size)), eps=1e-8)
    
    def forward(self, x):
        """x: (B, C, T, F)"""
        ## Intra RNN
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)[0]  # (B*T,F,C)
        intra_x = self.intra_fc(intra_x)      # (B*T,F,C)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size) # (B,T,F,C)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        x = intra_out.permute(0,2,1,3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) 
        inter_x = self.inter_rnn(inter_x)[0]  # (B*F,T,C)
        inter_x = self.inter_fc(inter_x)      # (B*F,T,C)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size) # (B,F,T,C)
        inter_x = inter_x.permute(0,2,1,3)   # (B,T,F,C)
        inter_x = self.inter_ln(inter_x) 
        inter_out = torch.add(intra_out, inter_x)
        
        dual_out = inter_out.permute(0,3,1,2)  # (B,C,T,F)
        
        return dual_out


class Encoder(nn.Module):
    def __init__(
        self,
        tra_gru_factory: Optional[Callable[..., nn.Module]] = None,
        tra_thresholds: Optional[Sequence[dict[str, Optional[float]]]] = None,
        log_gru_inputs: bool = False,
        tra_log_bases: Optional[Sequence[Optional[str]]] = None,
    ):
        super().__init__()
        thresholds = list(tra_thresholds) if tra_thresholds is not None else [{"x": None, "h": None}] * 3
        while len(thresholds) < 3:
            thresholds.append({"x": None, "h": None})
        log_bases = list(tra_log_bases) if tra_log_bases is not None else [None] * 3
        while len(log_bases) < 3:
            log_bases.append(None)
        self.en_convs = nn.ModuleList([
            ConvBlock(3*3, 16, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False,
                        gru_factory=tra_gru_factory, tra_threshold=thresholds[0],
                        log_gru_inputs=log_gru_inputs, log_file_base=log_bases[0]),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False,
                        gru_factory=tra_gru_factory, tra_threshold=thresholds[1],
                        log_gru_inputs=log_gru_inputs, log_file_base=log_bases[1]),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False,
                        gru_factory=tra_gru_factory, tra_threshold=thresholds[2],
                        log_gru_inputs=log_gru_inputs, log_file_base=log_bases[2])
        ])

    def forward(self, x):
        en_outs = []
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    def __init__(
        self,
        tra_gru_factory: Optional[Callable[..., nn.Module]] = None,
        tra_thresholds: Optional[Sequence[dict[str, Optional[float]]]] = None,
        log_gru_inputs: bool = False,
        tra_log_bases: Optional[Sequence[Optional[str]]] = None,
    ):
        super().__init__()
        thresholds = list(tra_thresholds) if tra_thresholds is not None else [{"x": None, "h": None}] * 3
        while len(thresholds) < 3:
            thresholds.append({"x": None, "h": None})
        log_bases = list(tra_log_bases) if tra_log_bases is not None else [None] * 3
        while len(log_bases) < 3:
            log_bases.append(None)
        self.de_convs = nn.ModuleList([
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*5,1), dilation=(5,1), use_deconv=True,
                        gru_factory=tra_gru_factory, tra_threshold=thresholds[0],
                        log_gru_inputs=log_gru_inputs, log_file_base=log_bases[0]),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True,
                        gru_factory=tra_gru_factory, tra_threshold=thresholds[1],
                        log_gru_inputs=log_gru_inputs, log_file_base=log_bases[1]),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*1,1), dilation=(1,1), use_deconv=True,
                        gru_factory=tra_gru_factory, tra_threshold=thresholds[2],
                        log_gru_inputs=log_gru_inputs, log_file_base=log_bases[2]),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            ConvBlock(16, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True)
        ])

    def forward(self, x, en_outs, final_freq):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            skip = en_outs[N_layers-1-i]
            x = self._match_spatial_dims(x, skip)
            x = x + skip

            module = self.de_convs[i]
            output_size = None
            if isinstance(module, ConvBlock) and module.use_deconv:
                if i < N_layers - 1:
                    target_freq = en_outs[N_layers-2-i].shape[-1]
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
        preprocess=None,
        log_gru_inputs=False,
        log_file=None,
        use_delta_gru: bool = False,
        delta_gru_threshold_x: Optional[float] = None,
        delta_gru_threshold_h: Optional[float] = None,
        delta_gru_thresholds: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        self._use_delta_gru = use_delta_gru
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.preprocessor = None
        if preprocess is not None:
            if not isinstance(preprocess, Mapping):
                raise TypeError("preprocess configuration must be a mapping or None")
            self.preprocessor = SpectralPreprocessor(**_to_plain(preprocess))

        erb_low, erb_high = self._compute_erb_subbands(self.n_fft)
        self.erb = ERB(erb_low, erb_high, nfft=self.n_fft)
        self.sfe = SFE(3, 1)

        gru_factory = self._build_gru_factory(use_delta_gru)

        base_thresh_x = delta_gru_threshold_x
        base_thresh_h = delta_gru_threshold_h
        thresholds_cfg = delta_gru_thresholds if isinstance(delta_gru_thresholds, Mapping) else {}

        encoder_tra_thresholds = _normalize_threshold_list(
            thresholds_cfg.get("encoder_tra_blocks"),
            3,
            base_thresh_x,
            base_thresh_h,
        )
        decoder_tra_thresholds = _normalize_threshold_list(
            thresholds_cfg.get("decoder_tra_blocks"),
            3,
            base_thresh_x,
            base_thresh_h,
        )
        dp1_thresholds = _normalize_dp_thresholds(thresholds_cfg.get("dpgrnn1"), base_thresh_x, base_thresh_h)
        dp2_thresholds = _normalize_dp_thresholds(thresholds_cfg.get("dpgrnn2"), base_thresh_x, base_thresh_h)

        encoder_log_bases = None
        decoder_log_bases = None
        # If a log file base was provided, create distinct bases for all logging targets
        if log_file is not None:
            import os
            root, ext = os.path.splitext(log_file)
            dp1_base = f"{root}_dp1{ext}"
            dp2_base = f"{root}_dp2{ext}"
            encoder_log_bases = [f"{root}_enc_tra{i}{ext}" for i in range(3)]
            decoder_log_bases = [f"{root}_dec_tra{i}{ext}" for i in range(3)]
        else:
            dp1_base = None
            dp2_base = None
        self.encoder = Encoder(
            tra_gru_factory=gru_factory,
            tra_thresholds=encoder_tra_thresholds,
            log_gru_inputs=log_gru_inputs,
            tra_log_bases=encoder_log_bases,
        )
        encoder_width = self._compute_encoder_width(self.erb.output_bins)

        self.dpgrnn1 = DPGRNN(
            16,
            encoder_width,
            16,
            log_gru_inputs=log_gru_inputs,
            log_file_base=dp1_base,
            gru_factory=gru_factory,
            intra_rnn_thresholds=[
                dp1_thresholds["intra_rnn1"],
                dp1_thresholds["intra_rnn2"],
            ],
            inter_rnn_thresholds=[
                dp1_thresholds["inter_rnn1"],
                dp1_thresholds["inter_rnn2"],
            ],
        )
        self.dpgrnn2 = DPGRNN(
            16,
            encoder_width,
            16,
            log_gru_inputs=log_gru_inputs,
            log_file_base=dp2_base,
            gru_factory=gru_factory,
            intra_rnn_thresholds=[
                dp2_thresholds["intra_rnn1"],
                dp2_thresholds["intra_rnn2"],
            ],
            inter_rnn_thresholds=[
                dp2_thresholds["inter_rnn1"],
                dp2_thresholds["inter_rnn2"],
            ],
        )
        self.decoder = Decoder(
            tra_gru_factory=gru_factory,
            tra_thresholds=decoder_tra_thresholds,
            log_gru_inputs=log_gru_inputs,
            tra_log_bases=decoder_log_bases,
        )

        self.mask = Mask()

    def forward(self, x):
        """
        x: (B, L)
        """
        device = x.device
        n_samples = x.shape[1]
        
        stft_kwargs = {'n_fft': self.n_fft, 'hop_length': self.hop_len, 'win_length': self.win_len,
                       'window': torch.hann_window(self.win_len).to(device), 'onesided': True}
        
        spec_complex = torch.stft(x,  **stft_kwargs, return_complex=True)
        if self.preprocessor is not None:
            spec_complex = self.preprocessor(spec_complex)

        spec_ri = torch.view_as_real(spec_complex)

        spec_real = spec_ri[..., 0].permute(0,2,1)
        spec_imag = spec_ri[..., 1].permute(0,2,1)
        spec_mag = torch.abs(spec_complex).permute(0,2,1)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,F)
        
        spec = spec_ri.permute(0,3,2,1)  # (B,2,T,F)

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

    def load_state_dict(self, state_dict, strict: bool = True):
        if self._use_delta_gru:
            state_dict = self._ensure_delta_prefixes(state_dict)
        return super().load_state_dict(state_dict, strict=strict)

    @staticmethod
    def _ensure_delta_prefixes(state_dict):
        if any(".gru." in key for key in state_dict.keys()):
            return state_dict
        new_state = state_dict.__class__()
        target_names = {"att_gru", "rnn1", "rnn2"}
        for key, value in state_dict.items():
            parts = key.split(".")
            for idx, part in enumerate(parts):
                if part in target_names:
                    if idx + 1 < len(parts) and parts[idx + 1] == "gru":
                        break
                    parts = parts[:idx + 1] + ["gru"] + parts[idx + 1:]
                    break
            new_key = ".".join(parts)
            new_state[new_key] = value
        return new_state

    @staticmethod
    def _build_gru_factory(use_delta_gru: bool) -> Callable[..., nn.Module]:
        if not use_delta_gru:
            print("Using standard GRU")
            return _default_gru_factory

        def factory(input_size: int, hidden_size: int, **kwargs) -> nn.Module:
            threshold_x = kwargs.pop("threshold_x", None)
            threshold_h = kwargs.pop("threshold_h", None)
            print(f"Using DeltaGRU with threshold_x={threshold_x}, threshold_h={threshold_h}")
            return DeltaGRU(
                input_size=input_size,
                hidden_size=hidden_size,
                threshold_x=threshold_x,
                threshold_h=threshold_h,
                **kwargs,
            )

        return factory

    @classmethod
    def _compute_erb_subbands(cls, n_fft):
        base_nfreqs = cls.BASE_NFFT // 2 + 1
        low_ratio = cls.BASE_LOW_BINS / base_nfreqs
        high_ratio = cls.BASE_HIGH_BANDS / max(base_nfreqs - cls.BASE_LOW_BINS, 1)

        nfreqs = n_fft // 2 + 1
        if nfreqs < 3:
            raise ValueError(f"n_fft={n_fft} is too small to build ERB filters")

        erb_low = int(round(nfreqs * low_ratio))
        # keep at least two bins for the ERB mapping and one bin below the split
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


class GTCRNCore(nn.Module):
    """Feature-only wrapper around :class:`GTCRN` without STFT/ISTFT.

    This module consumes the three-channel STFT features that the full GTCRN
    builds internally (magnitude, real, imaginary) and produces the complex mask
    predicted by the network. It is helpful for profiling the arithmetic of the
    learned model without including FFT overhead.
    """

    def __init__(self, erb: ERB, sfe: SFE, encoder: Encoder,
                 dpgrnn1: DPGRNN, dpgrnn2: DPGRNN, decoder: Decoder, mask: Mask):
        super().__init__()
        self.erb = erb
        self.sfe = sfe
        self.encoder = encoder
        self.dpgrnn1 = dpgrnn1
        self.dpgrnn2 = dpgrnn2
        self.decoder = decoder
        self.mask = mask

    @classmethod
    def from_full_model(cls, model: "GTCRN") -> "GTCRNCore":
        """Build a core wrapper that reuses the weights of a trained GTCRN."""
        return cls(model.erb, model.sfe, model.encoder,
                   model.dpgrnn1, model.dpgrnn2, model.decoder, model.mask)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Run the GTCRN network on pre-computed STFT features.

        Parameters
        ----------
        feat:
            Tensor with shape ``(B, 3, T, F)`` where the three channels are the
            magnitude, real and imaginary STFT components. ``T`` can be any
            number of frames ≥ 1. For per-frame profiling, set ``T=1``.

        Returns
        -------
        torch.Tensor
            Complex mask with shape ``(B, 2, T, F_full)`` that corresponds to
            the output of :meth:`GTCRN.forward` before inverse STFT.
        """
        if feat.dim() != 4 or feat.shape[1] != 3:
            raise ValueError("feat must have shape (B, 3, T, F)")

        feat_erb = self.erb.bm(feat)
        feat_erb = self.sfe(feat_erb)

        encoded, skips = self.encoder(feat_erb)
        encoded = self.dpgrnn1(encoded)
        encoded = self.dpgrnn2(encoded)

        mask_erb = self.decoder(encoded, skips, self.erb.output_bins)
        mask_full = self.erb.bs(mask_erb)
        return mask_full


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
