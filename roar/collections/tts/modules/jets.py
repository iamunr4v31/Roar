import torch

from roar.collections.tts.modules.submodules import (
    ConditionalInput,
    ConditionalLayerNorm,
)
from roar.collections.tts.modules.attention import MultiHeadCrossAttn
from roar.collections.tts.parts.utils.helpers import (
    binarize_attention_parallel,
    regulate_len,
    rand_slice_segments,
)
from roar.core.classes import NeuralModule, adapter_mixins, typecheck
from roar.core.neural_types.elements import (
    EncodedRepresentation,
    Index,
    LengthsType,
    LogprobsType,
    MelSpectrogramType,
    AudioSignal,
    ProbsType,
    RegressionValuesType,
    TokenDurationType,
    TokenIndex,
    TokenLogDurationType,
)
from roar.core.neural_types.neural_type import NeuralType


def average_features(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = torch.nn.functional.pad(
        torch.cumsum(pitch != 0.0, dim=2), (1, 0)
    )
    pitch_cums = torch.nn.functional.pad(torch.cumsum(pitch, dim=2), (1, 0))

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


class ConvReLUNorm(torch.nn.Module, adapter_mixins.AdapterModuleMixin):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        dropout=0.0,
        condition_dim=384,
        condition_types=[],
    ):
        super(ConvReLUNorm, self).__init__()
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size // 2),
        )
        self.norm = ConditionalLayerNorm(
            out_channels, condition_dim=condition_dim, condition_types=condition_types
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, signal, conditioning=None):
        out = torch.nn.functional.relu(self.conv(signal))
        out = self.norm(out.transpose(1, 2), conditioning).transpose(1, 2)
        out = self.dropout(out)

        if self.is_adapter_available():
            out = self.forward_enabled_adapters(out.transpose(1, 2)).transpose(1, 2)

        return out


class TemporalPredictor(NeuralModule):
    """Predicts a single float per each temporal location"""

    def __init__(
        self,
        input_size,
        filter_size=512,
        n_attn_heads=8,
        kernel_size=3,
        dropout=0.5,
        n_layers=30,
        cross_attn_interval=3,
        pre_ln=False,
        condition_types=[],
    ):
        super(TemporalPredictor, self).__init__()
        d_head = filter_size // n_attn_heads
        assert n_attn_heads * d_head == filter_size
        self.cross_attn_interval = cross_attn_interval
        self.cond_input = ConditionalInput(input_size, input_size, condition_types)
        # self.layers = torch.nn.ModuleList()
        self.conv_layers = torch.nn.ModuleList()
        self.attn_layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.conv_layers.append(
                ConvReLUNorm(
                    input_size if i == 0 else filter_size,
                    filter_size,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    condition_dim=input_size,
                    condition_types=condition_types,
                )
            )
            if i % cross_attn_interval == cross_attn_interval - 1:
                self.attn_layers = MultiHeadCrossAttn(n_attn_heads, filter_size, d_head, 0.2, pre_lnorm=pre_ln)
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
        idx = 0
        for i, layer in enumerate(self.conv_layers):
            out = layer(out, conditioning=conditioning)
            if i % self.cross_attn_interval == self.cross_attn_interval - 1:
                out = out.transpose(1, 2)
                out = self.attn_layers[idx](out, conditioning=conditioning)
                out = out.transpose(1, 2)
                idx+=1
                
        out = out.transpose(1, 2)
        out = self.fc(out) * enc_mask
        return out.squeeze(-1)


class SpeechPromptEncoder(NeuralModule):
    def __init__(self, prompt_encoder: NeuralModule, n_mel_channels: int = 80) -> None:
        super().__init__()
        self.prompt_encoder = prompt_encoder
        self.inp_proj = torch.nn.Linear(n_mel_channels, prompt_encoder.d_model)
    
    def forward(self, inp):
        out = self.inp_proj(inp)
        out = self.prompt_encoder(out)
        return out

class JETSModule(NeuralModule, adapter_mixins.AdapterModuleMixin):
    def __init__(
        self,
        encoder_module: NeuralModule,
        decoder_module: NeuralModule,
        duration_predictor: NeuralModule,
        pitch_predictor: NeuralModule,
        energy_predictor: NeuralModule,
        aligner: NeuralModule,
        speaker_encoder: NeuralModule,
        waveform_generator: NeuralModule,
        n_speakers: int,
        symbols_embedding_dim: int,
        pitch_embedding_kernel_size: int,
        energy_embedding_kernel_size: int,
        segment_size: int = 4,
        n_mel_channels: int = 80,  # deprecated
        min_token_duration: int = 0,
        max_token_duration: int = 75,
        use_log_energy: bool = True,
    ):
        super().__init__()

        self.encoder = encoder_module
        self.decoder = decoder_module
        self.duration_predictor = duration_predictor
        self.pitch_predictor = pitch_predictor
        self.energy_predictor = energy_predictor
        self.aligner = aligner
        self.speaker_encoder = speaker_encoder
        self.waveform_generator = waveform_generator
        self.learn_alignment = aligner is not None
        self.use_duration_predictor = True
        self.binarize = False
        self.use_log_energy = use_log_energy
        self.segment_size = segment_size

        # TODO: combine self.speaker_emb with self.speaker_encoder
        # cfg: remove `n_speakers`, create `speaker_encoder.lookup_module`
        # state_dict: move `speaker_emb.weight` to `speaker_encoder.lookup_module.table.weight`
        if n_speakers > 1 and speaker_encoder is None:
            self.speaker_emb = torch.nn.Embedding(n_speakers, symbols_embedding_dim)
        else:
            self.speaker_emb = None

        self.min_token_duration = min_token_duration
        self.max_token_duration = max_token_duration

        self.pitch_emb = torch.nn.Conv1d(
            1,
            symbols_embedding_dim,
            kernel_size=pitch_embedding_kernel_size,
            padding=int((pitch_embedding_kernel_size - 1) / 2),
        )

        if self.energy_predictor is not None:
            self.energy_emb = torch.nn.Conv1d(
                1,
                symbols_embedding_dim,
                kernel_size=energy_embedding_kernel_size,
                padding=int((energy_embedding_kernel_size - 1) / 2),
            )

        # Store values precomputed from training data for convenience
        self.register_buffer("pitch_mean", torch.zeros(1))
        self.register_buffer("pitch_std", torch.zeros(1))

        # self.proj = torch.nn.Linear(self.decoder.d_model, n_mel_channels, bias=True)

    @property
    def input_types(self):
        return {
            "text": NeuralType(("B", "T_text"), TokenIndex()),
            "durs": NeuralType(("B", "T_text"), TokenDurationType()),
            "pitch": NeuralType(("B", "T_audio"), RegressionValuesType()),
            "energy": NeuralType(
                ("B", "T_audio"), RegressionValuesType(), optional=True
            ),
            "speaker": NeuralType(("B"), Index(), optional=True),
            "pace": NeuralType(optional=True),
            "spec": NeuralType(
                ("B", "D", "T_spec"), MelSpectrogramType(), optional=True
            ),
            "attn_prior": NeuralType(
                ("B", "T_spec", "T_text"), ProbsType(), optional=True
            ),
            "mel_lens": NeuralType(("B"), LengthsType(), optional=True),
            "input_lens": NeuralType(("B"), LengthsType(), optional=True),
            "reference_spec": NeuralType(
                ("B", "D", "T_spec"), MelSpectrogramType(), optional=True
            ),
            "reference_spec_lens": NeuralType(("B"), LengthsType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "wav": NeuralType(("B", "S", "T"), AudioSignal()),
            "num_frames": NeuralType(("B"), TokenDurationType()),
            "durs_predicted": NeuralType(("B", "T_text"), TokenDurationType()),
            "log_durs_predicted": NeuralType(("B", "T_text"), TokenLogDurationType()),
            "pitch_predicted": NeuralType(("B", "T_text"), RegressionValuesType()),
            "attn_soft": NeuralType(("B", "S", "T_spec", "T_text"), ProbsType()),
            "attn_logprob": NeuralType(("B", "S", "T_spec", "T_text"), LogprobsType()),
            "attn_hard": NeuralType(("B", "S", "T_spec", "T_text"), ProbsType()),
            "attn_hard_dur": NeuralType(("B", "T_text"), TokenDurationType()),
            "pitch": NeuralType(("B", "T_audio"), RegressionValuesType()),
            "energy_pred": NeuralType(("B", "T_text"), RegressionValuesType()),
            "energy_tgt": NeuralType(("B", "T_audio"), RegressionValuesType()),
            "z_start_idxs": NeuralType(("B"), Index()),
        }

    def get_speaker_embedding(
        self, batch_size, speaker, reference_spec, reference_spec_lens
    ):
        """spk_emb: Bx1xD"""
        if self.speaker_encoder is not None:
            spk_emb = self.speaker_encoder(
                batch_size, speaker, reference_spec, reference_spec_lens
            ).unsqueeze(1)
        elif self.speaker_emb is not None:
            if speaker is None:
                raise ValueError(
                    "Please give speaker id to get lookup speaker embedding."
                )
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
        else:
            spk_emb = None

        return spk_emb

    @typecheck()
    def forward(
        self,
        *,
        text,
        durs=None,
        pitch=None,
        energy=None,
        speaker=None,
        pace=1.0,
        spec=None,
        attn_prior=None,
        mel_lens=None,
        input_lens=None,
        reference_spec=None,
        reference_spec_lens=None,
    ):
        if not self.learn_alignment and self.training:
            assert durs is not None
            assert pitch is not None

        # Calculate speaker embedding
        spk_emb = self.get_speaker_embedding(
            batch_size=text.shape[0],
            speaker=speaker,
            reference_spec=reference_spec,
            reference_spec_lens=reference_spec_lens,
        )

        # Input FFT
        enc_out, enc_mask = self.encoder(input=text, conditioning=spk_emb)

        # Predict duration
        log_durs_predicted = self.duration_predictor(
            enc_out, enc_mask, conditioning=spk_emb
        )
        durs_predicted = log_to_duration(
            log_dur=log_durs_predicted,
            min_dur=self.min_token_duration,
            max_dur=self.max_token_duration,
            mask=enc_mask,
        )

        attn_soft, attn_hard, attn_hard_dur, attn_logprob = None, None, None, None
        if self.learn_alignment and spec is not None:
            attn_soft, attn_logprob = self.aligner(
                spec,
                enc_out.permute(0, 2, 1),
                enc_mask == 0,
                attn_prior,
                conditioning=spk_emb,
            )
            attn_hard = binarize_attention_parallel(attn_soft, input_lens, mel_lens)
            attn_hard_dur = attn_hard.sum(2)[:, 0, :]

        # Predict pitch
        pitch_predicted = self.pitch_predictor(enc_out, enc_mask, conditioning=spk_emb)
        if pitch is not None:
            if self.learn_alignment and pitch.shape[-1] != pitch_predicted.shape[-1]:
                # Pitch during training is per spectrogram frame, but during inference, it should be per character
                pitch = average_features(pitch.unsqueeze(1), attn_hard_dur).squeeze(1)
            elif not self.learn_alignment:
                # If alignment is not learnt attn_hard_dur is None, hence durs_predicted
                pitch = average_features(pitch.unsqueeze(1), durs_predicted).squeeze(1)
            pitch_emb = self.pitch_emb(pitch.unsqueeze(1))
        else:
            pitch_emb = self.pitch_emb(pitch_predicted.unsqueeze(1))

        enc_out = enc_out + pitch_emb.transpose(1, 2)

        # Predict energy
        if self.energy_predictor is not None:
            energy_pred = self.energy_predictor(
                enc_out, enc_mask, conditioning=spk_emb
            ).squeeze(-1)

            if energy is not None:
                # Average energy over characters
                if self.learn_alignment:
                    energy_tgt = average_features(energy.unsqueeze(1), attn_hard_dur)
                else:
                    energy_tgt = average_features(energy.unsqueeze(1), durs_predicted)
                if self.use_log_energy:
                    energy_tgt = torch.log(1.0 + energy_tgt)
                energy_emb = self.energy_emb(energy_tgt)
                energy_tgt = energy_tgt.squeeze(1)
            else:
                energy_emb = self.energy_emb(energy_pred.unsqueeze(1))
                energy_tgt = None

            enc_out = enc_out + energy_emb.transpose(1, 2)
        else:
            energy_pred = None
            energy_tgt = None

        if self.learn_alignment and spec is not None:
            len_regulated, dec_lens = regulate_len(attn_hard_dur, enc_out, pace)
        elif spec is None and durs is not None:
            len_regulated, dec_lens = regulate_len(durs, enc_out, pace)
        # Use predictions during inference
        elif spec is None:
            len_regulated, dec_lens = regulate_len(durs_predicted, enc_out, pace)
        else:
            raise ValueError(
                f"Something unexpected happened when 'spec' is not None and 'self.learn_alignment' is False."
            )

        # Output FFT
        dec_out, _ = self.decoder(
            input=len_regulated, seq_lens=dec_lens, conditioning=spk_emb
        )

        z_segments, z_start_idxs = (
            rand_slice_segments(  # TODO: Change to rand_slice_segments from utils
                dec_out.transpose(1, 2), mel_lens, self.segment_size
            )
        )
        wav = self.waveform_generator(x=z_segments)
        # spect = self.proj(dec_out).transpose(1, 2)
        return (
            wav,
            dec_lens,
            durs_predicted,
            log_durs_predicted,
            pitch_predicted,
            attn_soft,
            attn_logprob,
            attn_hard,
            attn_hard_dur,
            pitch,
            energy_pred,
            energy_tgt,
            z_start_idxs,
        )

    def infer(
        self,
        *,
        text,
        pitch=None,
        speaker=None,
        energy=None,
        pace=1.0,
        volume=None,
        reference_spec=None,
        reference_spec_lens=None,
    ):
        # Calculate speaker embedding
        spk_emb = self.get_speaker_embedding(
            batch_size=text.shape[0],
            speaker=speaker,
            reference_spec=reference_spec,
            reference_spec_lens=reference_spec_lens,
        )

        # Input FFT
        enc_out, enc_mask = self.encoder(input=text, conditioning=spk_emb)

        # Predict duration and pitch
        log_durs_predicted = self.duration_predictor(
            enc_out, enc_mask, conditioning=spk_emb
        )
        durs_predicted = log_to_duration(
            log_dur=log_durs_predicted,
            min_dur=self.min_token_duration,
            max_dur=self.max_token_duration,
            mask=enc_mask,
        )

        if pitch is not None:
            assert (
                pitch.shape[-1] == text.shape[-1]
            ), f"pitch.shape[-1]: {pitch.shape[-1]} != len(text)"
            pitch_emb = self.pitch_emb(pitch)
        else:
            pitch_pred = self.pitch_predictor(enc_out, enc_mask, conditioning=spk_emb)
            pitch_emb = self.pitch_emb(pitch_pred.unsqueeze(1))

        # pitch_predicted = (
        #     self.pitch_predictor(enc_out, enc_mask, conditioning=spk_emb) + pitch
        # )
        # pitch_emb = self.pitch_emb(pitch_predicted.unsqueeze(1))
        enc_out = enc_out + pitch_emb.transpose(1, 2)

        if self.energy_predictor is not None:
            if energy is not None:
                assert (
                    energy.shape[-1] == text.shape[-1]
                ), f"energy.shape[-1]: {energy.shape[-1]} != len(text)"
                energy_emb = self.energy_emb(energy)
            else:
                energy_pred = self.energy_predictor(
                    enc_out, enc_mask, conditioning=spk_emb
                ).squeeze(-1)
                energy_emb = self.energy_emb(energy_pred.unsqueeze(1))
            enc_out = enc_out + energy_emb.transpose(1, 2)

        # Expand to decoder time dimension
        len_regulated, dec_lens = regulate_len(durs_predicted, enc_out, pace)
        volume_extended = None
        if volume is not None:
            volume_extended, _ = regulate_len(
                durs_predicted, volume.unsqueeze(-1), pace
            )
            volume_extended = volume_extended.squeeze(-1).float()

        # Output FFT
        dec_out, _ = self.decoder(
            input=len_regulated, seq_lens=dec_lens, conditioning=spk_emb
        )
        # spect = self.proj(dec_out).transpose(1, 2)
        wav = self.waveform_generator(x=dec_out.transpose(1, 2))
        return (
            wav,
            dec_lens,
            durs_predicted,
            log_durs_predicted,
            pitch_pred,
            volume_extended,
        )
