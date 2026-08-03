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
    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x: (B,C,T,F)"""
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.att_gru(zt.transpose(1,2))[0]
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
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False):
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
        
        self.tra = TRA(in_channels//2)

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
    """Grouped RNN"""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)

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
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return y, h
    
    
class DPGRNN(nn.Module):
    """Grouped Dual-path RNN"""
    def __init__(self, input_size, width, hidden_size, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size//2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
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
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(3*3, 16, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False)
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
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*5,1), dilation=(5,1), use_deconv=True),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*1,1), dilation=(1,1), use_deconv=True),
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
        win_len=512
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len

        erb_low, erb_high = self._compute_erb_subbands(self.n_fft)
        self.erb = ERB(erb_low, erb_high, nfft=self.n_fft)
        self.sfe = SFE(3, 1)

        self.encoder = Encoder()
        encoder_width = self._compute_encoder_width(self.erb.output_bins)

        self.dpgrnn1 = DPGRNN(16, encoder_width, 16)
        self.dpgrnn2 = DPGRNN(16, encoder_width, 16)
        
        self.decoder = Decoder()

        self.mask = Mask()
        self.is_qat_prepared = False
        self.qat_backend = None

    def fuse_model(self):
        self.encoder.fuse_model()
        self.decoder.fuse_model()

    def prepare_qat(self, backend='fbgemm', quantize_deconv=False, per_channel_weights: bool = False):
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
        qat_qconfig = quant.QConfig(activation=act_qconfig.activation, weight=weight_qconfig.weight)
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
                deconv_weight_axis1 = quant.FakeQuantize.with_args(
                    observer=quant.MovingAveragePerChannelMinMaxObserver,
                    quant_min=-128,
                    quant_max=127,
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric,
                    reduce_range=False,
                    ch_axis=1,
                )
                deconv_qconfig_axis1 = quant.QConfig(
                    activation=act_qconfig.activation,
                    weight=deconv_weight_axis1,
                )
            else:
                # Quantized ConvTranspose2d conversion under FBGEMM disables per-channel weight quantization.
                # Keep per-tensor weight fake-quant for deconvs (via QNNPACK defaults) so conversion succeeds.
                per_tensor_weight_qconfig = quant.get_default_qat_qconfig("qnnpack")
                deconv_qconfig = quant.QConfig(
                    activation=act_qconfig.activation,
                    weight=per_tensor_weight_qconfig.weight,
                )

        for module in self.modules():
            module.qconfig = None

        for module in self.modules():
            if isinstance(module, ConvBlock):
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

            if isinstance(module, GTConvBlock):
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

        quant.prepare_qat(self, inplace=True)
        self.is_qat_prepared = True
        self.qat_backend = backend
        return self

    def convert_qat(self, inplace=False, dynamic_quantize_gru=False, dynamic_gru_dtype=torch.qint8):
        model = self if inplace else deepcopy(self)
        if not model.is_qat_prepared:
            raise RuntimeError("Call prepare_qat(...) and load a QAT checkpoint before conversion.")
        if next(model.parameters()).device.type != 'cpu':
            raise RuntimeError("Quantized conversion only supports CPU execution. Move model to CPU first.")

        if model.qat_backend is not None:
            torch.backends.quantized.engine = model.qat_backend
        model.eval()
        quant.convert(model, inplace=True)
        if dynamic_quantize_gru:
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
