import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange

from roar.utils import logging
from roar.collections.tts.modules.submodules import (
    ConditionalInput,
    ConditionalLayerNorm,
    ConditionalRMSNorm,
)
from roar.collections.tts.modules import activations
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
from roar.core.neural_types.neural_type import NeuralType

from roar.utils.gpu_utils import is_gpu_ampere_or_newer
from roar.collections.tts.modules.attention import MultiHeadAttn

HAVE_FLASH = True
try:
    from roar.collections.tts.modules.attention import MultiHeadAttnFlash
except ImportError:
    HAVE_FLASH = False


def average_features(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = F.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = F.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = F.pad(torch.cumsum(pitch, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    pitch_sums = (
        torch.gather(pitch_cums, 2, dce) - torch.gather(pitch_cums, 2, dcs)
    ).float()
    pitch_nelems = (
        torch.gather(pitch_nonzero_cums, 2, dce)
        - torch.gather(pitch_nonzero_cums, 2, dcs)
    ).float()

    pitch_avg = torch.where(
        pitch_nelems == 0.0, pitch_nelems, pitch_sums / pitch_nelems
    )
    return pitch_avg


def log_to_duration(log_dur, min_dur, max_dur, mask):
    dur = torch.clamp(torch.exp(log_dur) - 1.0, min_dur, max_dur)
    dur *= mask.squeeze(2)
    return dur


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_size=2048,
        filter_size=512,
        output_size=2048,
        kernel_size=3,
        dropout=0.5,
        activation_fn="ReLU",
        norm_type="layernorm",  # "layernorm", "groupnorm", "rmsnorm",
        pre_lnorm=False,
    ):
        super(ConvBlock, self).__init__()
        assert hasattr(nn, activation_fn) or hasattr(
            activations, activation_fn
        ), f"Activation function {activation_fn} not found in torch.nn"

        if hasattr(nn, activation_fn):
            self.activation = getattr(nn, activation_fn)()
        else:  # alias for SiLU in Swish
            self.activation = getattr(activations, activation_fn)()

        if norm_type == "layernorm":
            self.norm = ConditionalLayerNorm(input_size)
        elif norm_type == "groupnorm":
            self.norm = nn.GroupNorm(8, input_size)
        elif norm_type == "rmsnorm":
            self.norm = ConditionalRMSNorm(input_size)

        self.pre_lnorm = pre_lnorm

        self.net = nn.Sequential(
            nn.Conv1d(
                input_size,
                filter_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            self.activation,
            nn.Dropout(dropout),
            nn.Conv1d(
                filter_size,
                filter_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            self.activation,
            nn.Dropout(dropout),
            nn.Conv1d(
                filter_size,
                output_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            self.activation,
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # B x T x C -> B x C x T
        x = rearrange(x, "b t c -> b c t")
        if self.pre_lnorm:
            x = x + self.net(self.norm(x))
        else:
            x = self.norm(x + self.net(x))
        # B x C x T -> B x T x C
        x = rearrange(x, "b c t -> b t c")
        return x


class TemporalPredictor(NeuralModule):
    # Pitch Energy and Duration Predictor. DurationPredictor can be replaced with styletts2 DifferentiableDurationPredictor with stochastic upsampling.
    """Predicts a single float per each temporal location"""

    def __init__(
        self,
        input_size=512,
        filter_size=256,
        kernel_size=3,
        dropout_conv=0.5,
        n_head=8,
        d_head=64,
        dropout=0.1,
        dropatt=0.1,
        n_layers=10,
        pre_lnorm=False,
        condition_types=[],
        **kwargs,
    ):
        super(TemporalPredictor, self).__init__()
        self.cond_input = ConditionalInput(input_size, input_size, condition_types)
        self.layers = nn.ModuleList()
        AttentionBlock = MultiHeadAttn
        if kwargs.get("use_flash", False):
            if not HAVE_FLASH:
                logging.warning(
                    "Flash attention is not available. Falling back to regular attn."
                )
            elif not is_gpu_ampere_or_newer():
                logging.warning(
                    "Flash attention is only available on Ampere or newer GPUs. Falling back to regular attn."
                )
            else:
                AttentionBlock = MultiHeadAttnFlash

        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.layers.append(
                nn.ModuleList(
                    [
                        ConvBlock(
                            input_size,
                            filter_size,
                            filter_size,
                            kernel_size,
                            dropout_conv,
                            pre_lnorm=pre_lnorm,
                        ),
                        AttentionBlock(
                            n_head,
                            filter_size,  # d_model needs to be the same as the channel size of the conv block
                            d_head,
                            dropout,
                            dropatt,
                            pre_lnorm=pre_lnorm,
                            condition_types=condition_types,
                        ),
                    ]
                )
            )

        self.layers.append(
            nn.ModuleList[
                ConvBlock(
                    filter_size,
                    filter_size,
                    input_size,
                    kernel_size,
                    dropout_conv,
                    pre_lnorm=pre_lnorm,
                ),
                AttentionBlock(
                    n_head,
                    input_size,
                    d_head,
                    dropout,
                    dropatt,
                    pre_lnorm=pre_lnorm,
                    condition_types=condition_types,
                ),
            ]
        )

        self.out_proj = nn.Linear(input_size, 1)

    def forward(self, enc, enc_mask, conditioning=None):
        enc = self.cond_input(enc, conditioning)
        for conv_norm, attention in self.layers:
            enc = conv_norm(enc)
            enc = attention(enc, enc_mask)
        enc = self.out_proj(enc)
        enc = rearrange(enc, "... 1 -> ...")  # squeeze out the last dimension
        return enc


class DifferentiableDurationPredictor(TemporalPredictor):
    """Predicts a single float per each temporal location"""

    def __init__(
        self,
        input_size=512,
        filter_size=256,
        kernel_size=3,
        dropout_conv=0.5,
        n_head=8,
        d_head=64,
        dropout=0.1,
        dropatt=0.1,
        frame_size=80,  # number of frames that each phoneme can occupy. This is the maximum duration of a phoneme.
        n_layers=10,
        pre_lnorm=False,
        condition_types=[],
        **kwargs,
    ):
        super(TemporalPredictor, self).__init__()
        self.cond_input = ConditionalInput(input_size, input_size, condition_types)
        self.layers = nn.ModuleList()
        AttentionBlock = MultiHeadAttn
        if kwargs.get("use_flash", False):
            if not HAVE_FLASH:
                logging.warning(
                    "Flash attention is not available. Falling back to regular attn."
                )
            elif not is_gpu_ampere_or_newer():
                logging.warning(
                    "Flash attention is only available on Ampere or newer GPUs. Falling back to regular attn."
                )
            else:
                AttentionBlock = MultiHeadAttnFlash

        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.layers.append(
                nn.ModuleList(
                    [
                        ConvBlock(
                            input_size,
                            filter_size,
                            filter_size,
                            kernel_size,
                            dropout_conv,
                            pre_lnorm=pre_lnorm,
                        ),
                        AttentionBlock(
                            n_head,
                            filter_size,  # d_model needs to be the same as the channel size of the conv block
                            d_head,
                            dropout,
                            dropatt,
                            pre_lnorm=pre_lnorm,
                            condition_types=condition_types,
                        ),
                    ]
                )
            )

        self.layers.append(
            nn.ModuleList[
                ConvBlock(
                    filter_size,
                    filter_size,
                    input_size,
                    kernel_size,
                    dropout_conv,
                    pre_lnorm=pre_lnorm,
                ),
                AttentionBlock(
                    n_head,
                    input_size,
                    d_head,
                    dropout,
                    dropatt,
                    pre_lnorm=pre_lnorm,
                    condition_types=condition_types,
                ),
            ]
        )

        self.out_proj = nn.Linear(input_size, frame_size)

    def forward(self, enc, enc_mask, conditioning=None):
        enc = self.cond_input(enc, conditioning)
        for conv_norm, attention in self.layers:
            enc = conv_norm(enc)
            enc = attention(enc, enc_mask)
        enc = self.out_proj(enc)  # B x T x F
        enc = F.sigmoid(enc, dim=-1)  # d_hat > k = 1.0, d_hat <= k = 0.0
        return enc
