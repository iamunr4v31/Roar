import torch
import math
from roar.collections.tts.modules.submodules import (
    ConditionalInput,
    ConditionalLayerNorm,
    ConvNorm,
    ConvReLUNorm,
    fused_add_tanh_sigmoid_multiply,
)
from roar.collections.tts.parts.utils.helpers import (
    binarize_attention_parallel,
    regulate_len,
)
from roar.core.classes import NeuralModule, adapter_mixins, typecheck
from roar.core.neural_types.elements import (
    EncodedRepresentation,
    Index,
    LengthsType,
    LogprobsType,
    MelSpectrogramType,
    ProbsType,
    RegressionValuesType,
    TokenDurationType,
    TokenIndex,
    TokenLogDurationType,
)
from roar.collections.tts.modules.monotonic_align import maximum_path
from roar.core.neural_types.neural_type import NeuralType


class GlowWN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        condition_dim=192,
        p_dropout=0.1,
    ):
        super(GlowWN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.condition_dim = condition_dim
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = torch.nn.Dropout(p_dropout)

        if condition_dim != 0:
            self.cond_layer = ConvNorm(
                condition_dim, 2 * hidden_channels * n_layers, 1, use_weight_norm=True
            )

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = ConvNorm(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
                use_weight_norm=True,
            )
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            res_skip_layer = ConvNorm(
                hidden_channels, res_skip_channels, 1, use_weight_norm=True
            )
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, conditioning=None):
        output = torch.zeros_like(x)

        conditioning_ls = None
        if conditioning is not None:
            conditioning = self.cond_layer(conditioning)
            conditioning_ls = torch.chunk(conditioning, chunks=self.n_layers, dim=1)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if conditioning_ls is not None:
                conditioning_l = conditioning_ls[i]
            else:
                conditioning_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, conditioning_l)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts, res_skip = torch.chunk(res_skip_acts, chunks=2, dim=1)
                x = (x + res_acts) * x_mask
                output = output + res_skip
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(layer)


class WN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        condition_dim=192,
        p_dropout=0.1,
        condition_types=[],
    ):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.condition_dim = condition_dim
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = torch.nn.Dropout(p_dropout)

        self.cond_layer = ConditionalInput(
            hidden_channels, condition_dim, condition_types
        )
        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = ConvNorm(
                hidden_channels,
                hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
                use_weight_norm=True,
            )
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            res_skip_layer = ConvNorm(
                hidden_channels, res_skip_channels, 1, use_weight_norm=True
            )
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, conditioning=None):
        output = torch.zeros_like(x)

        x = self.cond_layer(x, conditioning)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            # acts = fused_add_tanh_sigmoid_multiply(x_in, conditioning_l)
            acts = torch.nn.functional.softplus(x_in)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts, res_skip = torch.chunk(res_skip_acts, chunks=2, dim=1)
                x = (x + res_acts) * x_mask
                output = output + res_skip
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        # if self.condition_dim != 0:
        #     self.cond_layer.remove_weight_norm()
        for layer in self.in_layers:
            layer.remove_weight_norm()
        for layer in self.res_skip_layers:
            layer.remove_weight_norm()


class ActNorm(torch.nn.Module):
    """Activation Normalization bijector as an alternative to Batch Norm. It computes
    mean and std from a sample data in advance and it uses these values
    for normalization at training.

    Args:
        channels (int): input channels.
        ddi (False): data depended initialization flag.

    Shapes:
        - inputs: (B, C, T)
        - outputs: (B, C, T)
    """

    def __init__(self, channels, ddi=False):  # pylint: disable=unused-argument
        super().__init__()
        self.channels = channels
        self.initialized = not ddi

        self.logs = torch.nn.Parameter(torch.zeros(1, channels, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x, x_mask=None, reverse=False):  # pylint: disable=unused-argument
        if x_mask is None:
            x_mask = torch.ones(x.size(0), 1, x.size(2)).to(
                device=x.device, dtype=x.dtype
            )
        x_len = torch.sum(x_mask, [1, 2])
        if not self.initialized:
            self.initialize(x, x_mask)
            self.initialized = True

        if reverse:
            z = (x - self.bias) * torch.exp(-self.logs) * x_mask
            logdet = None
        else:
            z = (self.bias + torch.exp(self.logs) * x) * x_mask
            logdet = torch.sum(self.logs) * x_len  # [b]

        return z, logdet

    def store_inverse(self):
        pass

    def set_ddi(self, ddi):
        self.initialized = not ddi

    def initialize(self, x, x_mask):
        with torch.no_grad():
            denom = torch.sum(x_mask, [0, 2])
            m = torch.sum(x * x_mask, [0, 2]) / denom
            m_sq = torch.sum(x * x * x_mask, [0, 2]) / denom
            v = m_sq - (m**2)
            logs = 0.5 * torch.log(torch.clamp_min(v, 1e-6))

            bias_init = (
                (-m * torch.exp(-logs)).view(*self.bias.shape).to(dtype=self.bias.dtype)
            )
            logs_init = (-logs).view(*self.logs.shape).to(dtype=self.logs.dtype)

            self.bias.data.copy_(bias_init)
            self.logs.data.copy_(logs_init)


class InvConvNear(torch.nn.Module):
    def __init__(self, channels, n_split=4, no_jacobian=False, **kwargs):
        super().__init__()
        assert n_split % 2 == 0
        self.channels = channels
        self.n_split = n_split
        self.no_jacobian = no_jacobian
        self.weight_inv = None

        w_init = torch.linalg.qr(
            torch.FloatTensor(self.n_split, self.n_split).normal_()
        )[0]
        if torch.det(w_init) < 0:
            w_init[:, 0] = -1 * w_init[:, 0]
        self.weight = torch.nn.Parameter(w_init)

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        b, c, t = x.size()
        assert c % self.n_split == 0
        if x_mask is None:
            x_mask = 1
            x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
        else:
            x_len = torch.sum(x_mask, [1, 2])

        x = x.view(b, 2, c // self.n_split, self.n_split // 2, t)
        x = (
            x.permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(b, self.n_split, c // self.n_split, t)
        )

        if reverse:
            if self.weight_inv is not None:
                weight = self.weight_inv
            else:
                weight = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)
            logdet = None
        else:
            weight = self.weight
            if self.no_jacobian:
                logdet = 0
            else:
                logdet = torch.logdet(self.weight) * (c / self.n_split) * x_len  # [b]

        weight = weight.view(self.n_split, self.n_split, 1, 1)
        z = torch.nn.functional.conv2d(x, weight)

        z = z.view(b, 2, self.n_split // 2, c // self.n_split, t)
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
        return z, logdet

    def store_inverse(self):
        weight_inv = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)
        self.weight_inv = torch.nn.Parameter(weight_inv, requires_grad=False)


class CouplingBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0,
        sigmoid_scale=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.sigmoid_scale = sigmoid_scale

        start = torch.nn.Conv1d(in_channels // 2, hidden_channels, 1)
        start = torch.nn.utils.weight_norm(start)
        self.start = start
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  It helps to stabilze training.
        end = torch.nn.Conv1d(hidden_channels, in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        self.wn = GlowWN(
            in_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels,
            p_dropout,
        )

    def forward(self, x, x_mask=None, reverse=False, g=None, **kwargs):
        b, c, t = x.size()
        if x_mask is None:
            x_mask = 1
        x_0, x_1 = x[:, : self.in_channels // 2], x[:, self.in_channels // 2 :]

        x = self.start(x_0) * x_mask
        x = self.wn(x, x_mask, g)
        out = self.end(x)

        z_0 = x_0
        m = out[:, : self.in_channels // 2, :]
        logs = out[:, self.in_channels // 2 :, :]
        if self.sigmoid_scale:
            logs = torch.log(1e-6 + torch.sigmoid(logs + 2))

        if reverse:
            z_1 = (x_1 - m) * torch.exp(-logs) * x_mask
            logdet = None
        else:
            z_1 = (m + torch.exp(logs) * x_1) * x_mask
            logdet = torch.sum(logs * x_mask, [1, 2])

        z = torch.cat([z_0, z_1], 1)
        return z, logdet

    def store_inverse(self):
        self.wn.remove_weight_norm()


def squeeze(x, x_mask=None, num_sqz=2):
    """GlowTTS squeeze operation
    Increase number of channels and reduce number of time steps
    by the same factor.

    Note:
        each 's' is a n-dimensional vector.
        ``[s1,s2,s3,s4,s5,s6] --> [[s1, s3, s5], [s2, s4, s6]]``
    """
    b, c, t = x.size()

    t = (t // num_sqz) * num_sqz
    x = x[:, :, :t]
    x_sqz = x.view(b, c, t // num_sqz, num_sqz)
    x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c * num_sqz, t // num_sqz)

    if x_mask is not None:
        x_mask = x_mask[:, :, num_sqz - 1 :: num_sqz]
    else:
        x_mask = torch.ones(b, 1, t // num_sqz).to(device=x.device, dtype=x.dtype)
    return x_sqz * x_mask, x_mask


def unsqueeze(x, x_mask=None, num_sqz=2):
    """GlowTTS unsqueeze operation (revert the squeeze)

    Note:
        each 's' is a n-dimensional vector.
        ``[[s1, s3, s5], [s2, s4, s6]] --> [[s1, s3, s5, s2, s4, s6]]``
    """
    b, c, t = x.size()

    x_unsqz = x.view(b, num_sqz, c // num_sqz, t)
    x_unsqz = (
        x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c // num_sqz, t * num_sqz)
    )

    if x_mask is not None:
        x_mask = x_mask.unsqueeze(-1).repeat(1, 1, 1, num_sqz).view(b, 1, t * num_sqz)
    else:
        x_mask = torch.ones(b, 1, t * num_sqz).to(device=x.device, dtype=x.dtype)
    return x_unsqz * x_mask, x_mask


class PreNet(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size,
        n_layers,
        dropout,
        condition_dim=384,
        condition_types=[],
    ) -> None:
        super().__init__()
        self.core_net = torch.nn.ModuleList([])
        for i in range(n_layers):
            self.core_net.append(
                ConvReLUNorm(
                    in_channels=in_channels if i == 0 else hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    condition_dim=condition_dim,
                    condition_types=condition_types,
                )
            )
        self.core_net.append(torch.nn.Conv1d(hidden_channels, out_channels, 1))

        self.core_net = torch.nn.Sequential(*self.core_net)

    def forward(self, input, mask):
        output = input + self.core_net(input)
        return output * mask if mask is not None else output


class TemporalPredictor(NeuralModule):
    """Predicts a single float per each temporal location"""

    def __init__(
        self,
        input_size,
        filter_size,
        kernel_size,
        dropout,
        n_layers=2,
        condition_types=[],
    ):
        super(TemporalPredictor, self).__init__()
        self.cond_input = ConditionalInput(input_size, input_size, condition_types)
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                ConvReLUNorm(
                    input_size if i == 0 else filter_size,
                    filter_size,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    condition_dim=input_size,
                    condition_types=condition_types,
                )
            )
        self.fc = torch.nn.Linear(filter_size, 1, bias=True)

        # Use for adapter input dimension
        self.filter_size = filter_size

    @property
    def input_types(self):
        return {
            "enc": NeuralType(("B", "T", "D"), EncodedRepresentation()),
            "enc_mask": NeuralType(("B", "T", 1), TokenDurationType()),
            "conditioning": NeuralType(
                ("B", "T", "D"), EncodedRepresentation(), optional=True
            ),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(("B", "T"), EncodedRepresentation()),
        }

    def forward(self, enc, enc_mask, conditioning=None):
        enc = self.cond_input(enc, conditioning)
        out = enc * enc_mask
        out = out.transpose(1, 2)

        for layer in self.layers:
            out = layer(out, conditioning=conditioning)

        out = out.transpose(1, 2)
        out = self.fc(out) * enc_mask
        return out.squeeze(-1)


class FlowSpecDecoder(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_blocks,
        n_layers,
        p_dropout=0.0,
        n_split=4,
        n_sqz=2,
        sigmoid_scale=False,
        gin_channels=0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.gin_channels = gin_channels

        self.flows = torch.nn.ModuleList()
        for b in range(n_blocks):
            self.flows.append(ActNorm(channels=in_channels * n_sqz))
            self.flows.append(
                InvConvNear(channels=in_channels * n_sqz, n_split=n_split)
            )
            self.flows.append(
                CouplingBlock(
                    in_channels * n_sqz,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    gin_channels=gin_channels,
                    p_dropout=p_dropout,
                    sigmoid_scale=sigmoid_scale,
                )
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            flows = self.flows
            logdet_tot = 0
        else:
            flows = reversed(self.flows)
            logdet_tot = None

        if self.n_sqz > 1:
            x, x_mask = squeeze(x, x_mask, self.n_sqz)
        for f in flows:
            if not reverse:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
                logdet_tot += logdet
            else:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
        if self.n_sqz > 1:
            x, x_mask = unsqueeze(x, x_mask, self.n_sqz)
        return x, logdet_tot

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()


# GlowTTS -> rewrite in NeMo code.
class FlowGenerator(torch.nn.Module):
    def __init__(
        self,
        n_vocab,
        hidden_channels,
        filter_channels,
        filter_channels_dp,
        out_channels,
        kernel_size=3,
        n_heads=2,
        n_layers_enc=6,
        p_dropout=0.0,
        n_blocks_dec=12,
        kernel_size_dec=5,
        dilation_rate=5,
        n_block_layers=4,
        p_dropout_dec=0.0,
        n_speakers=0,
        gin_channels=0,
        n_split=4,
        n_sqz=1,
        sigmoid_scale=False,
        window_size=None,
        block_length=None,
        mean_only=False,
        hidden_channels_enc=None,
        hidden_channels_dec=None,
        prenet=False,
        **kwargs,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_heads = n_heads
        self.n_layers_enc = n_layers_enc
        self.p_dropout = p_dropout
        self.n_blocks_dec = n_blocks_dec
        self.kernel_size_dec = kernel_size_dec
        self.dilation_rate = dilation_rate
        self.n_block_layers = n_block_layers
        self.p_dropout_dec = p_dropout_dec
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.window_size = window_size
        self.block_length = block_length
        self.mean_only = mean_only
        self.hidden_channels_enc = hidden_channels_enc
        self.hidden_channels_dec = hidden_channels_dec
        self.prenet = prenet

        self.encoder = TextEncoder(
            n_vocab,
            out_channels,
            hidden_channels_enc or hidden_channels,
            filter_channels,
            filter_channels_dp,
            n_heads,
            n_layers_enc,
            kernel_size,
            p_dropout,
            window_size=window_size,
            block_length=block_length,
            mean_only=mean_only,
            prenet=prenet,
            gin_channels=gin_channels,
        )

        self.decoder = FlowSpecDecoder(
            out_channels,
            hidden_channels_dec or hidden_channels,
            kernel_size_dec,
            dilation_rate,
            n_blocks_dec,
            n_block_layers,
            p_dropout=p_dropout_dec,
            n_split=n_split,
            n_sqz=n_sqz,
            sigmoid_scale=sigmoid_scale,
            gin_channels=gin_channels,
        )

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
            nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

    def forward(
        self,
        x,
        x_lengths,
        y=None,
        y_lengths=None,
        g=None,
        gen=False,
        noise_scale=1.0,
        length_scale=1.0,
    ):
        if g is not None:
            g = F.normalize(self.emb_g(g)).unsqueeze(-1)  # [b, h]
        x_m, x_logs, logw, x_mask = self.encoder(x, x_lengths, g=g)

        if gen:
            w = torch.exp(logw) * x_mask * length_scale
            w_ceil = torch.ceil(w)
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_max_length = None
        else:
            y_max_length = y.size(2)
        y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
        z_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

        if gen:
            attn = commons.generate_path(
                w_ceil.squeeze(1), attn_mask.squeeze(1)
            ).unsqueeze(1)
            z_m = torch.matmul(
                attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)
            ).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
            z_logs = torch.matmul(
                attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)
            ).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
            logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

            z = (z_m + torch.exp(z_logs) * torch.randn_like(z_m) * noise_scale) * z_mask
            y, logdet = self.decoder(z, z_mask, g=g, reverse=True)
            return (
                (y, z_m, z_logs, logdet, z_mask),
                (x_m, x_logs, x_mask),
                (attn, logw, logw_),
            )
        else:
            z, logdet = self.decoder(y, z_mask, g=g, reverse=False)
            with torch.no_grad():
                x_s_sq_r = torch.exp(-2 * x_logs)
                logp1 = torch.sum(
                    -0.5 * torch.log(2 * torch.pi) - x_logs, [1]
                ).unsqueeze(-1)  # [b, t, 1]
                logp2 = torch.matmul(
                    x_s_sq_r.transpose(1, 2), -0.5 * (z**2)
                )  # [b, t, d] x [b, d, t'] = [b, t, t']
                logp3 = torch.matmul(
                    (x_m * x_s_sq_r).transpose(1, 2), z
                )  # [b, t, d] x [b, d, t'] = [b, t, t']
                logp4 = torch.sum(-0.5 * (x_m**2) * x_s_sq_r, [1]).unsqueeze(
                    -1
                )  # [b, t, 1]
                logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']

                attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()
            z_m = torch.matmul(
                attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)
            ).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
            z_logs = torch.matmul(
                attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)
            ).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
            logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask
            return (
                (z, z_m, z_logs, logdet, z_mask),
                (x_m, x_logs, x_mask),
                (attn, logw, logw_),
            )

    def preprocess(self, y, y_lengths, y_max_length):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
            y = y[:, :, :y_max_length]
        y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
        return y, y_lengths, y_max_length

    def store_inverse(self):
        self.decoder.store_inverse()
