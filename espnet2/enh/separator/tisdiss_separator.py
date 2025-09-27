# Copyright (C) 2025 WingSing Fung:
#
# SPDX-License-Identifier: Apache-2.0


import math
from collections import OrderedDict
from re import A
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from espnet2.enh.layers.complex_utils import new_complex_like
from espnet2.enh.layers.tflocoformer import TFLocoformerBlock
from espnet2.enh.layers.spk_spliter import SpkSplitSwiGLUConvDeconv2d, SpkSplitConv2d
from espnet2.enh.layers.time_segmentation import TimeSegmentation
from espnet2.enh.layers.band_split import BandSplitEncoder, BandSplitDecoder, BandSplitEncoderConv1D, BandSplitDecoderConv1D, TIGER_BandSplitEncoder, TIGER_BandSplitDecoder
from packaging.version import parse as V
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange, reduce, repeat
from librosa import filters
import numpy as np

from espnet2.enh.separator.abs_separator import AbsSeparator

is_torch_2_0_plus = V(torch.__version__) >= V("2.0.0")

# BandSplit version selection
BAND_SPLIT_ENCODERS = {
    'linear': BandSplitEncoder,
    'conv1d': BandSplitEncoderConv1D,
}

BAND_SPLIT_DECODERS = {
    'linear': BandSplitDecoder,
    'conv1d': BandSplitDecoderConv1D,
}


class TISDiSS_Separator(AbsSeparator):
    """
    Args:
        input_dim: int
            placeholder, not used
        num_spk: int
            number of output sources/speakers.
        n_layers: int
            number of Locoformer blocks.
        emb_dim: int
            Size of hidden dimension in the encoding Conv2D.
        norm_type: str
            Normalization layer. Must be either "layernorm" or "rmsgroupnorm".
        num_groups: int
            Number of groups in RMSGroupNorm layer.
        tf_order: str
            Order of frequency and temporal modeling. Must be either "ft" or "tf".
        n_heads: int
            Number of heads in multi-head self-attention.
        flash_attention: bool
            Whether to use flash attention. Only compatible with half precision.
        ffn_type: str or list
            Feed-forward network (FFN)-type chosen from "conv1d" or "swiglu_conv1d".
            Giving the list (e.g., ["conv1d", "conv1d"]) makes the model Macaron-style.
        ffn_hidden_dim: int or list
            Number of hidden dimensions in FFN.
            Giving the list (e.g., [256, 256]) makes the model Macaron-style.
        conv1d_kernel: int
            Kernel size in Conv1d.
        conv1d_shift: int
            Shift size of Conv1d kernel.
        dropout: float
            Dropout probability.
        eps: float
            Small constant for normalization layer.
        spk_spliter_type: str
            Type of speaker spliter for ablation study. 
            Must be either "swiglu_conv_deconv2d" (default) or "conv2d".
            "swiglu_conv_deconv2d": uses SwiGLU activation with conv-deconv structure.
            "conv2d": simple Conv2d that directly transforms dim to dim*num_spks.
    """

    def __init__(
        self,
        input_dim,
        num_spk: int = 2,
        # general setup
        emb_dim: int = 128,
        norm_type: str = "rmsgrouporm",
        num_groups: int = 4,  # used only in RMSGroupNorm
        tf_order: str = "ft",
        # self-attention related
        n_heads: int = 4,
        flash_attention: bool = False,  # available when using mixed precision
        attention_dim: int = 128,
        # ffn related
        ffn_type: Union[str, list] = "swiglu_conv1d",
        ffn_hidden_dim: Union[int, list] = 384,
        conv1d_kernel: int = 4,
        conv1d_shift: int = 1,
        dropout: float = 0.0,
        # others
        eps: float = 1.0e-5,
        # efficient training related
        encoder_repeat_times: int = 2,
        encoder_n_layers: int = 1,
        reconstructor_repeat_times: int = 3,
        reconstructor_n_layers: int = 1,
        spk_path_enable: bool = True,
        repeat_residual_module: bool = False,
        reconstructor_repeat_residual_module: bool = True,
        # transformer_blocks
        # = encoder_transformer_blocks * encoder_n_layers * encoder_repeat_times + reconstructor_transformer_blocks * reconstructor_n_layers * reconstructor_repeat_times
        # = 2*1*2+3*1*3=13 Transformer blocks
        # early split and multi decoder related
        encoder_decoder: bool = True,  # False: don't calculate for the encoder decoder loss
        encoder_multi_decoder: bool = True,  # False: don't calculate for the encoder multi decoder loss
        encoder_n_layers_multi_decoder: bool = False,  # False: don't calculate for the encoder multi decoder loss for each layer
        reconstructor_multi_decoder: bool = True,  # False: don't calculate for the reconstructor multi decoder loss
        reconstructor_n_layers_multi_decoder: bool = False,  # False: don't calculate for the reconstructor multi decoder loss for each layer
        spliter_loss: bool = False,  # False: don't calculate for the spliter loss
        split_compress: bool = False,  # False: don't compress the split output dimension to 1/spk_num
        # when reconstructor_n_layers larger than 1,reconstructor_multi_decoder is False and reconstructor_n_layers_multi_decoder is True, will calculate the multi decoder loss for each layer for last repeat time
        # when reconstructor_n_layers larger than 1,reconstructor_multi_decoder is True and reconstructor_n_layers_multi_decoder is True, will calculate the multi decoder loss for each layer for each repeat time
        mask: bool = False,  # False: don't use mask, True: use separated output as mask and multiply with original input
        # spliter ablation study
        spk_spliter_type: str = "swiglu_conv_deconv2d",  # "swiglu_conv_deconv2d" or "conv2d"
        # input type:
        input_type: str = "stft" , # "stft" or "conv"
        # time domain processing parameters:
        segment_size: int = 96,  # Time segment size for time-domain processing, only used when input_type="conv"
        # band split
        band_split: bool = False,
        split_type: str = "mel",  # "mel", "tiger", or "bandsplit"
        band_split_encoder_type: str = "linear",
        band_split_decoder_type: str = "linear",
        stft_n_fft=128,
        stft_hop_length=64,
        stft_win_length=128,
        stft_normalized=False,
        sample_rate=8000,
        num_bands=25,
        stereo=False,
        band_split_decoder_depth=2,
        mlp_expansion_factor=4,
        # bandsplit related
        freqs_per_bands=None,  # for bandsplit type, if None, use default
    ):
        super().__init__()
        assert is_torch_2_0_plus, "Support only pytorch >= 2.0.0"

        self._num_spk = num_spk
        self.encoder_n_layers = encoder_n_layers
        self.encoder_repeat_times = encoder_repeat_times
        self.reconstructor_n_layers = reconstructor_n_layers
        self.reconstructor_repeat_times = reconstructor_repeat_times
        self.spk_path_enable = spk_path_enable
        self.encoder_decoder = encoder_decoder
        self.encoder_multi_decoder = encoder_multi_decoder
        self.encoder_n_layers_multi_decoder = encoder_n_layers_multi_decoder
        self.reconstructor_multi_decoder = reconstructor_multi_decoder
        self.reconstructor_n_layers_multi_decoder = reconstructor_n_layers_multi_decoder
        self.spliter_loss = spliter_loss
        self.split_compress = num_spk if split_compress else 1
        self.mask = mask
        self.spk_spliter_type = spk_spliter_type
        self.input_type = input_type
        self.segment_size = segment_size
        self.input_dim = input_dim  # Save input_dim for time-domain processing mode
        self.band_split = band_split
        self.split_type = split_type
        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length
        self.stft_win_length = stft_win_length
        self.stft_normalized = stft_normalized
        self.sample_rate = sample_rate
        self.num_bands = num_bands
        self.stereo = stereo
        self.band_split_decoder_depth = band_split_decoder_depth
        self.mlp_expansion_factor = mlp_expansion_factor
        self.freqs_per_bands = freqs_per_bands
        
        

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)

        """ encoder blocks """
        if self.input_type == "stft":
            if self.band_split:
                if self.split_type == "mel":
                    self.mel_band_init(
                        stft_n_fft=self.stft_n_fft,
                        stft_hop_length=self.stft_hop_length,
                        stft_win_length=self.stft_win_length,
                        stft_normalized=self.stft_normalized,
                        sample_rate=self.sample_rate,
                        dim=emb_dim,
                        num_bands=self.num_bands,
                        stereo=self.stereo,
                        band_split_decoder_depth=self.band_split_decoder_depth,
                        mlp_expansion_factor=self.mlp_expansion_factor,
                        band_split_encoder_type=band_split_encoder_type,
                        band_split_decoder_type=band_split_decoder_type,
                    )
                elif self.split_type == "tiger":
                    self.tiger_band_init(
                        stft_n_fft=self.stft_n_fft,
                        sample_rate=self.sample_rate,
                        dim=emb_dim,
                        num_spk=num_spk,
                    )
                elif self.split_type == "bandsplit":
                    self.bandsplit_band_init(
                        stft_n_fft=self.stft_n_fft,
                        stft_hop_length=self.stft_hop_length,
                        stft_win_length=self.stft_win_length,
                        stft_normalized=self.stft_normalized,
                        sample_rate=self.sample_rate,
                        dim=emb_dim,
                        stereo=self.stereo,
                        freqs_per_bands=self.freqs_per_bands,
                        band_split_decoder_depth=self.band_split_decoder_depth,
                        mlp_expansion_factor=self.mlp_expansion_factor,
                        band_split_encoder_type=band_split_encoder_type,
                        band_split_decoder_type=band_split_decoder_type,
                    )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(2, emb_dim, ks, padding=padding),
                    nn.GroupNorm(1, emb_dim, eps=eps),  # gLN
                )
        elif self.input_type == "conv":
            self.time_segmentation = TimeSegmentation(segment_size=segment_size)
            self.time_linear = nn.Linear(input_dim, emb_dim)

        assert attention_dim % n_heads == 0, (attention_dim, n_heads)
        rope_freq = RotaryEmbedding(attention_dim // n_heads)
        rope_time = RotaryEmbedding(attention_dim // n_heads)
        
        """ separator blocks """
        if encoder_n_layers > 0:
            self.encoder_blocks = nn.ModuleList([])
            for _ in range(encoder_n_layers):
                self.encoder_blocks.append(
                    TFLocoformerBlock(
                        rope_freq,
                        rope_time,
                        # general setup
                        emb_dim=emb_dim,
                        norm_type=norm_type,
                        num_groups=num_groups,
                        tf_order=tf_order,
                        # self-attention related
                        n_heads=n_heads,
                        flash_attention=flash_attention,
                        attention_dim=attention_dim,
                        # ffn related
                        ffn_type=ffn_type,
                        ffn_hidden_dim=ffn_hidden_dim,
                        conv1d_kernel=conv1d_kernel,
                        conv1d_shift=conv1d_shift,
                        dropout=dropout,
                        eps=eps,
                    )
                )

        """ spk split """
        if self.spk_spliter_type == "swiglu_conv_deconv2d":
            self.spk_spliter = SpkSplitSwiGLUConvDeconv2d(
                emb_dim, emb_dim * 2, num_spk, ks, padding=padding, dropout=dropout, split_compress=self.split_compress
            )
        elif self.spk_spliter_type == "conv2d":
            self.spk_spliter = SpkSplitConv2d(
                emb_dim, num_spk, ks, padding=padding, dropout=dropout, split_compress=self.split_compress
            )
        else:
            raise ValueError(f"Unsupported spk_spliter_type: {self.spk_spliter_type}. "
                           f"Supported types are: 'swiglu_conv_deconv2d', 'conv2d'")
        if self.split_compress > 1:
            rope_freq = RotaryEmbedding(attention_dim // n_heads // self.split_compress)
            rope_time = RotaryEmbedding(attention_dim // n_heads // self.split_compress)
        if spk_path_enable:
            rope_spk = RotaryEmbedding(attention_dim // n_heads // self.split_compress)
        else:
            rope_spk = None

        """ reconstructor blocks """
        self.reconstructor_blocks = nn.ModuleList([])
        for _ in range(reconstructor_n_layers):
            self.reconstructor_blocks.append(
                TFLocoformerBlock(
                    rope_freq,
                    rope_time,
                    rope_spk,
                    spk_path_enable=spk_path_enable,
                    tf_order=tf_order,
                    # general setup
                    emb_dim=emb_dim//self.split_compress,
                    norm_type=norm_type,
                    num_groups=num_groups,
                    # self-attention related
                    n_heads=n_heads,
                    flash_attention=flash_attention,
                    attention_dim=attention_dim//self.split_compress,
                    # ffn related
                    ffn_type=ffn_type,
                    ffn_hidden_dim=[f//self.split_compress for f in ffn_hidden_dim] if isinstance(ffn_hidden_dim, list) else ffn_hidden_dim//self.split_compress,
                    conv1d_kernel=conv1d_kernel,
                    conv1d_shift=conv1d_shift,
                    dropout=dropout,
                    eps=eps,
                )
            )
        
        """ decoder blocks"""
        if self.input_type == "stft":
            if not self.band_split:
                self.reconstructor_deconv = nn.ConvTranspose2d(emb_dim//self.split_compress, 2, ks, padding=padding)
        elif self.input_type == "conv":
            self.reconstructor_deconv = nn.ConvTranspose2d(emb_dim//self.split_compress, input_dim, ks, padding=padding)
        
        self.repeat_residual_module = repeat_residual_module
        self.reconstructor_repeat_residual_module = reconstructor_repeat_residual_module
        if self.repeat_residual_module:
            self.concat_block = nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1, 1, groups=emb_dim), nn.PReLU()
            )
            if self.split_compress > 1 and self.reconstructor_repeat_residual_module:
                self.compress_conv =  nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim//self.split_compress, ks, padding=padding),
                    nn.GroupNorm(1, emb_dim//self.split_compress, eps=eps),  # gLN
                )
                self.reconstructor_concat_block = nn.Sequential(
                    nn.Conv2d(emb_dim//self.split_compress, emb_dim//self.split_compress, 1, 1, groups=emb_dim//self.split_compress), nn.PReLU()
                )

    def mel_band_init(
        self,
        stft_n_fft=128,
        stft_hop_length=64,
        stft_win_length=128,
        stft_normalized=False,
        sample_rate=8000,
        num_bands=40,
        dim=None,
        stereo=False,
        multi_decode=False,
        band_split_decoder_depth=2,
        mlp_expansion_factor=4,
        band_split_encoder_type: str = "linear",
        band_split_decoder_type: str = "linear",
    ):
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )

        freqs = torch.stft(
            torch.randn(1, stft_n_fft * 2 + 1),
            **self.stft_kwargs,
            window=torch.ones(stft_n_fft),
            return_complex=True,
        ).shape[1]

        # create mel filter bank
        # with librosa.filters.mel as in section 2 of paper

        mel_filter_bank_numpy = filters.mel(
            sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands
        )

        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)

        # for some reason, it doesn't include the first freq? just force a value for now

        mel_filter_bank[0][0] = 1.0

        # In some systems/envs we get 0.0 instead of ~1.9e-18 in the last position,
        # so let's force a positive value

        mel_filter_bank[-1, -1] = 1.0

        # binary as in paper (then estimated masks are averaged for overlapping regions)

        freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(
            dim=0
        ).all(), "all frequencies need to be covered by all bands for now"

        repeated_freq_indices = repeat(torch.arange(freqs), "f -> b f", b=num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = repeat(freq_indices, "f -> f s", s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, "f s -> (f s)")

        self.register_buffer("freq_indices", freq_indices, persistent=False)
        self.register_buffer("freqs_per_band", freqs_per_band, persistent=False)

        num_freqs_per_band = reduce(freqs_per_band, "b f -> b", "sum")
        num_bands_per_freq = reduce(freqs_per_band, "b f -> f", "sum")

        self.register_buffer("num_freqs_per_band", num_freqs_per_band, persistent=False)
        print(f"num_freqs_per_band: {num_freqs_per_band}")
        self.register_buffer("num_bands_per_freq", num_bands_per_freq, persistent=False)

        # band split and mask estimator

        freqs_per_bands_with_complex = tuple(
            2 * f * self.audio_channels for f in num_freqs_per_band.tolist()
        )
        
        # Use dictionary selection to pass all parameters at once
        encoder_cls = BAND_SPLIT_ENCODERS[band_split_encoder_type]
        decoder_cls = BAND_SPLIT_DECODERS[band_split_decoder_type]
        
        self.band_split_encoder = encoder_cls(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex,
            eps=1e-8,
            depth=band_split_decoder_depth,
            mlp_expansion_factor=mlp_expansion_factor
        )
        self.band_split_decoder = decoder_cls(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex,
            eps=1e-8,
            depth=band_split_decoder_depth,
            mlp_expansion_factor=mlp_expansion_factor
        )

    def tiger_band_init(
        self,
        stft_n_fft=128,
        sample_rate=8000,
        dim=None,
        num_spk=2,
        eps=1e-8,
    ):
        """Initialize TIGER band split components.
        
        Args:
            stft_n_fft: STFT window size
            sample_rate: Sample rate
            dim: Feature dimension
            num_spk: Number of speakers/sources
            eps: Small value for numerical stability
        """
        self.enc_dim = stft_n_fft // 2 + 1
        self.feature_dim = dim
        self.num_sources = num_spk
        
        # Calculate band widths similar to TIGER
        # 8k hz, half is 4k hz, stft = 128, 128/2+1 = 65
        # a bin is 4000/64 = 62.5 hz
        bandwidth_1 = int(np.floor(62.5 / (sample_rate / 2.) * self.enc_dim))
        bandwidth_2 = int(np.floor(125 / (sample_rate / 2.) * self.enc_dim))
        bandwidth_4 = int(np.floor(250 / (sample_rate / 2.) * self.enc_dim))
        
        self.band_width = [1] # the first bin is 0 hz
        self.band_width += [bandwidth_1] * 16 # 62.5 * 16= 1000 hz
        self.band_width += [bandwidth_2] * 8 # 125 * 8 = 1000 hz
        self.band_width += [bandwidth_4] * 8 # 250 * 8 = 2000 hz

        if self.enc_dim - np.sum(self.band_width) > 0:
            self.band_width.append(self.enc_dim - np.sum(self.band_width))
        else:
            print(f"rest band width: {self.enc_dim - np.sum(self.band_width)}")
        self.nband = len(self.band_width)
        
        print(f"TIGER band split initialized with {self.nband} bands")
        print(f"Band widths: {self.band_width}")
        print(f"Total frequency bins: {np.sum(self.band_width)} (should be {self.enc_dim})")
        
        # Initialize TIGER encoder and decoder
        self.tiger_encoder = TIGER_BandSplitEncoder(
            band_width=self.band_width,
            feature_dim=self.feature_dim,
            eps=eps
        )
        
        self.tiger_decoder = TIGER_BandSplitDecoder(
            band_width=self.band_width,
            feature_dim=self.feature_dim,
            num_sources=1
        )
        
        # Store necessary parameters for forward pass
        self.tiger_eps = eps

    def bandsplit_band_init(
        self,
        stft_n_fft=128,
        stft_hop_length=64,
        stft_win_length=128,
        stft_normalized=False,
        sample_rate=8000,
        dim=None,
        stereo=False,
        freqs_per_bands=None,
        band_split_decoder_depth=2,
        mlp_expansion_factor=4,
        band_split_encoder_type: str = "linear",
        band_split_decoder_type: str = "linear",
    ):
        """Initialize bandsplit components using BS-RoFormer style band splitting.
        
        Args:
            stft_n_fft: STFT window size
            stft_hop_length: STFT hop length
            stft_win_length: STFT window length
            stft_normalized: Whether STFT is normalized
            sample_rate: Sample rate for calculating band widths
            dim: Feature dimension
            stereo: Whether stereo audio
            freqs_per_bands: Tuple of frequencies per band, if None use default
            band_split_decoder_depth: Decoder depth
            mlp_expansion_factor: MLP expansion factor
            band_split_encoder_type: Encoder type
            band_split_decoder_type: Decoder type
        """
        # Calculate frequency bins directly from STFT formula
        freqs = stft_n_fft // 2 + 1
        
        if freqs_per_bands is None:
            # Use TIGER-style dynamic band width calculation
            # Calculate band widths similar to TIGER
            bandwidth_1 = int(np.floor(62.5 / (sample_rate / 2.) * freqs))
            bandwidth_2 = int(np.floor(125 / (sample_rate / 2.) * freqs))
            bandwidth_4 = int(np.floor(250 / (sample_rate / 2.) * freqs))
            
            band_width = [1]  # the first bin is 0 hz
            band_width += [bandwidth_1] * 16  # 62.5 * 16= 1000 hz
            band_width += [bandwidth_2] * 8   # 125 * 8 = 1000 hz
            band_width += [bandwidth_4] * 8   # 250 * 8 = 2000 hz

            if freqs - np.sum(band_width) > 0:
                band_width.append(freqs - np.sum(band_width))
            else:
                print(f"rest band width: {freqs - np.sum(band_width)}")
            
            freqs_per_bands = tuple(band_width)
            
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )

        # Validate freqs_per_bands
        assert len(freqs_per_bands) > 1, "Need at least 2 bands"
        assert sum(freqs_per_bands) == freqs, f'the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}'

        # Convert to complex representation (real + imag) and account for stereo
        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)
        
        print(f"Bandsplit initialized with {len(freqs_per_bands)} bands")
        print(f"Frequencies per band: {freqs_per_bands}")
        print(f"Complex frequencies per band: {freqs_per_bands_with_complex}")
        print(f"Total frequency bins: {freqs}")
        print(f"Sample rate: {sample_rate}, STFT n_fft: {stft_n_fft}")
        
        # Store freqs_per_bands for forward pass
        self.freqs_per_bands = freqs_per_bands
        self.freqs_per_bands_with_complex = freqs_per_bands_with_complex
        
        # Use dictionary selection to pass all parameters at once
        encoder_cls = BAND_SPLIT_ENCODERS[band_split_encoder_type]
        decoder_cls = BAND_SPLIT_DECODERS[band_split_decoder_type]
        
        self.band_split_encoder = encoder_cls(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex,
            eps=1e-8,
            depth=band_split_decoder_depth,
            mlp_expansion_factor=mlp_expansion_factor
        )
        self.band_split_decoder = decoder_cls(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex,
            eps=1e-8,
            depth=band_split_decoder_depth,
            mlp_expansion_factor=mlp_expansion_factor
        )

    def _reconstruct_output(self, batch, batch0, n_batch, n_frames, n_freqs, add_to_aux=True, aux_batch=None):
        """General function to reconstruct complex outputs
        
        Args:
            batch: Input batch tensor [B, J, H, T, F]
            batch0: Original complex batch used to create new complex tensors
            n_batch: Batch size
            n_frames: Number of frames
            n_freqs: Number of frequencies  
            add_to_aux: Whether to add to aux_batch
            aux_batch: List of aux_batches
            
        Returns:
            List of reconstructed complex tensors
        """
        if self.input_type == "stft":
            if self.band_split:
                if self.split_type == "mel":
                    batch_tmp = rearrange(batch, "b j h t f -> (b j) t f h")  # [B*J, T, F, H]
                    with torch.amp.autocast("cuda", enabled=False):
                        # print(f"batch_tmp b t f_band h : {batch_tmp.shape}")
                        batch_tmp = self.band_split_decoder(
                            batch_tmp
                        )  # b t f_band h -> b t (f c)
                        # print(f"batch_tmp b t (f c): {batch_tmp.shape}")
                    batch_tmp = rearrange(
                        batch_tmp,
                        "(b j) t (f c) -> b j f t c",
                        b=n_batch,
                        j=self.num_spk,
                        c=2,
                        t=n_frames,
                    ).contiguous()
                    # print(f"batch_tmp b j f t c: {batch_tmp.shape}")
                    stft_repr_zero = torch.zeros(
                        n_batch,
                        self.num_spk,
                        n_freqs,
                        n_frames,
                        2,
                        dtype=batch.dtype,
                        device=batch.device,
                    )  # b j f_org t c
                    stft_repr_zero = torch.view_as_complex(stft_repr_zero)  # b j f_org t
                    batch_tmp = torch.view_as_complex(batch_tmp)  # b j f t c -> b j f t
                    scatter_indices = repeat(
                        self.freq_indices,
                        "f -> b j f t",
                        b=n_batch,
                        j=self.num_spk,
                        t=stft_repr_zero.shape[-1],
                    )
                    batch_tmp = stft_repr_zero.scatter_add_(
                        2, scatter_indices, batch_tmp
                    )  # b j f t -> b j f_org t
                    batch_tmp = rearrange(
                        batch_tmp, "b j f t -> b j t f"
                    )  # b j f_org t -> b j t f_org
                elif self.split_type == "tiger":
                    # For TIGER decoder
                    # batch shape: [B, J, H, T, nband] where J is num_spk, H is feature_dim
                    nch = self.tiger_nch  # Use the nch value from encoder
                    
                    # Reshape to combine all speakers into batch dimension
                    # [B, J, H, T, nband] -> [B*J, nband, H, T]
                    sep_output = rearrange(batch, "b j h t nband -> (b j) nband h t")
                    
                    # Apply TIGER decoder
                    # The decoder will automatically handle dimension mismatch between
                    # sep_output [B*J, nband, H, T] and tiger_subband_spec [B*nch, BW, T]
                    sep_spec = self.tiger_decoder(
                        sep_output, 
                        self.tiger_subband_spec,
                        n_batch * self.num_spk,  # Adjusted batch size
                        1  # Set nch=1 since we're treating each separated path as single channel
                    )
                    # sep_spec shape: [B*J, F, T]
                    
                    # Reshape to match expected output format: [B, J, T, F]
                    batch_tmp = rearrange(
                        sep_spec, 
                        "(b j) f t -> b j t f", 
                        b=n_batch, 
                        j=self.num_spk
                    )
                elif self.split_type == "bandsplit":
                    # For bandsplit decoder - similar to mel but without frequency indexing
                    batch_tmp = rearrange(batch, "b j h t f -> (b j) t f h")  # [B*J, T, F, H]
                    with torch.amp.autocast("cuda", enabled=False):
                        # print(f"batch_tmp b t f_band h : {batch_tmp.shape}")
                        batch_tmp = self.band_split_decoder(
                            batch_tmp
                        )  # b t f_band h -> b t (f c)
                        # print(f"batch_tmp b t (f c): {batch_tmp.shape}")
                    batch_tmp = rearrange(
                        batch_tmp,
                        "(b j) t (f c) -> b j f t c",
                        b=n_batch,
                        j=self.num_spk,
                        c=2,
                        t=n_frames,
                    ).contiguous()
                    # print(f"batch_tmp b j f t c: {batch_tmp.shape}")
                    # For bandsplit, we directly convert to complex without scatter operations
                    batch_tmp = torch.view_as_complex(batch_tmp)  # b j f t c -> b j f t
                    batch_tmp = rearrange(
                        batch_tmp, "b j f t -> b j t f"
                    )  # b j f t -> b j t f
                    
                if self.mask:
                    # When mask=True, use separated output as mask and multiply with original input
                    # Using broadcasting: [B, J, T, F] * [B, 1, T, F] -> [B, J, T, F]
                    batch_tmp = batch_tmp * batch0
                    batch_tmp = [batch_tmp[:, src] for src in range(self.num_spk)]
                else:
                    batch_tmp = [batch_tmp[:, src] for src in range(self.num_spk)]
            else:
                batch_tmp = rearrange(batch, "b j h t f -> (b j) h t f")
                with torch.amp.autocast('cuda', enabled=False):
                    batch_tmp = self.reconstructor_deconv(batch_tmp)  # [B*J, 2, T, F]
                batch_tmp = batch_tmp.view(
                    [n_batch, self.num_spk, 2, n_frames, n_freqs]
                )  # [B, J, 2, T, F]
                batch_tmp = new_complex_like(
                    batch0, (batch_tmp[:, :, 0], batch_tmp[:, :, 1])
                ) # [B, J, T, F]
                
                if self.mask:
                    # When mask=True, use separated output as mask and multiply with original input
                    # Using broadcasting: [B, J, T, F] * [B, 1, T, F] -> [B, J, T, F]
                    batch_tmp = batch_tmp * batch0
                    batch_tmp = [batch_tmp[:, src] for src in range(self.num_spk)]
                else:
                    batch_tmp = [batch_tmp[:, src] for src in range(self.num_spk)]
        elif self.input_type == "conv":
            # Time-domain processing: merge segmented features back into time-domain signals
            B, J, H, T, F = batch.shape
            batch_tmp = rearrange(batch, "b j h t f -> (b j) h t f")
            batch_tmp = self.reconstructor_deconv(batch_tmp) # (B*J, D_enc, T, F)
            
            batch_tmp = self.time_segmentation.merge_feature(batch_tmp, length=n_frames)  # B*num_spk, D_enc, T overlap-add
            batch_tmp = rearrange(batch_tmp, "(b j) d t -> j b t d", b=B) # B*num_spk, D_enc, T -> num_spk, B, T, D_enc

        if add_to_aux and aux_batch is not None:
            aux_batch.append(batch_tmp)
        return batch_tmp

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor): batched single-channel audio tensor with
                in TF-domain [B, T, F]
            ilens (torch.Tensor): input lengths [B]
            additional (Dict or None): other data, currently unused in this model.

        Returns:
            enhanced (List[Union(torch.Tensor)]):
                    [(B, T), ...] list of len num_spk
                    of mono audio tensors with T samples.
            ilens (torch.Tensor): (B,)
            additional (Dict or None): other data, currently unused in this model,
                    we return it also in the output.
        """

        """ input encoder """
        if self.input_type == "stft":
            if input.ndim == 3:
                # in case the input does not have channel dimension
                batch0 = input.unsqueeze(1)
            elif input.ndim == 4:
                assert input.shape[1] == 1, "Only monaural input is supported."
                batch0 = input.transpose(1, 2)  # [B, M, T, F]

            batch = torch.cat((batch0.real, batch0.imag), dim=1)  # [B, 2*M, T, F]
            n_batch, _, n_frames, n_freqs = batch.shape
            # print(f"original batch: {batch.shape}")
            if self.band_split:
                if self.split_type == "mel":
                    stft_repr = rearrange(batch, "b m t f -> b f t m")  # [B, F, T, 2*M]
                    batch_arange = torch.arange(n_batch, device=batch.device)[..., None]
                    # account for stereo
                    batch = stft_repr[batch_arange, self.freq_indices]
                    # fold the complex (real and imag) into the frequencies dimension
                    batch = rearrange(batch, "b f t m -> b t (f m)")
                    batch = self.band_split_encoder(batch)  # b t (f m) -> b t f_band d
                    batch = rearrange(batch, "b t f_band d -> b d t f_band")
                    # print(
                    #     f"n_frames: {n_frames}, n_freqs: {n_freqs}, f_band: {batch.shape[-1]}, new_batch: {batch.shape}"
                    # )
                elif self.split_type == "tiger":
                    # For TIGER, we need to handle the spectrogram differently
                    # batch shape: [B, 2*M, T, F] where M is number of channels (usually 1)
                    nch = batch.shape[1] // 2  # Number of audio channels
                    self.tiger_nch = nch  # Store nch for decoder use
                    
                    # Reshape to match TIGER encoder input format
                    # Stack real and imag: [B*nch, 2, F, T]
                    spec_RI = rearrange(batch, "b (two m) t f -> (b m) two f t", two=2, m=nch)
                    
                    # Store original complex spectrogram for decoder
                    self.tiger_subband_spec = []
                    spec_complex = batch0.transpose(2, 3)  # [B, M, F, T]
                    # Reshape to match TIGER decoder expected format: [B*M, F, T]
                    spec_complex = spec_complex.view(-1, spec_complex.shape[-2], spec_complex.shape[-1])
                    # Split into subbands for decoder
                    band_idx = 0
                    for i in range(len(self.band_width)):
                        self.tiger_subband_spec.append(
                            spec_complex[:, band_idx:band_idx + self.band_width[i]] # [B*nch, BW, T]
                        )
                        band_idx += self.band_width[i]
                    
                    # Apply TIGER encoder
                    subband_features = self.tiger_encoder(spec_RI, n_batch, nch)
                    # subband_features shape: [B*nch, nband, feature_dim, T]
                    
                    # Reshape to match expected format: [B, feature_dim, T, nband]
                    batch = rearrange(subband_features, "(b m) nband d t -> b (d m) t nband", b=n_batch, m=nch)
                elif self.split_type == "bandsplit":
                    # For bandsplit, we split the frequency directly according to freqs_per_bands
                    # batch shape: [B, 2*M, T, F] where M is number of channels (usually 1)
                    # Rearrange to [B, T, F*2*M] format for band split encoder
                    batch = rearrange(batch, "b m t f -> b t (f m)")
                    
                    # Split according to freqs_per_bands_with_complex and encode
                    batch = self.band_split_encoder(batch)  # b t (f m) -> b t f_band d
                    batch = rearrange(batch, "b t f_band d -> b d t f_band")
                    # print(
                    #     f"n_frames: {n_frames}, n_freqs: {n_freqs}, f_band: {batch.shape[-1]}, new_batch: {batch.shape}"
                    # )
            else:
                with torch.amp.autocast('cuda', enabled=False):
                    batch = self.conv(batch)  # [B, -1, T, F]
        elif self.input_type == "conv":
            batch0 = input  # [B, T, D_enc]
            
            batch0 = self.time_linear(batch0)  # [B, T, D_enc] -> [B, T, emb_dim]
            
            n_batch, n_frames, _ = batch0.shape
            # Transpose for time segmentation
            batch0 = batch0.transpose(1, 2)  # [B, emb_dim, T]
            
            # 使用时间分割进行chunking
            segmented = self.time_segmentation.split_feature(batch0)  # [B, emb_dim, segment_size, n_chunks]
            n_freqs = segmented.shape[-1] # use n_chunks as frequency dimension
            batch = segmented # like [B, D, T, F] in tf mode

        """ separation """
        aux_batch = []
        batch_tmp = None
        if self.repeat_residual_module:
            mixture = batch.clone()
        """ encoder """
        for ii in range(self.encoder_n_layers * self.encoder_repeat_times):
            layer_idx = ii % self.encoder_n_layers
            if ii != 0 and self.repeat_residual_module:
                batch = self.concat_block(mixture + batch) # [B, -1, T, F]
            batch = self.encoder_blocks[layer_idx](batch)  # [B, -1, T, F]

            if self.encoder_decoder:
                if (
                    (
                        self.encoder_multi_decoder
                        and layer_idx == self.encoder_n_layers - 1
                    )  # multi decoder for each last layer for each repeat time
                    or (
                        ii == self.encoder_repeat_times * self.encoder_n_layers - 1
                    )  # last layer must has decoder loss
                    or (
                        self.encoder_n_layers_multi_decoder
                        and self.encoder_multi_decoder
                        and layer_idx != self.encoder_n_layers - 1
                    )  # multi decoder for each n layers for each repeat time
                    or (
                        self.encoder_n_layers_multi_decoder
                        and not self.encoder_multi_decoder
                        and ii
                        >= (self.encoder_repeat_times - 1) * self.encoder_n_layers
                    )  # multi decoder for each n layers for only last repeat time
                ):
                    batch_tmp = self.spk_spliter(batch)  # [B, num_spk, H, T, F]
                    # when reconstructor_n_layers = 0, the last layer will not add to aux_batch and will be the final output
                    if self.reconstructor_n_layers > 0:
                        self._reconstruct_output(batch_tmp, batch0, n_batch, n_frames, n_freqs, add_to_aux=True, aux_batch=aux_batch)
                    else:
                        is_last_layer = ii == self.encoder_repeat_times * self.encoder_n_layers - 1
                        batch_tmp = self._reconstruct_output(batch_tmp, batch0, n_batch, n_frames, n_freqs, add_to_aux=not is_last_layer, aux_batch=aux_batch)
        """ spk split """
        if batch_tmp is None:
            batch = self.spk_spliter(batch)  # [B, num_spk, H, T, F]
            if self.spliter_loss:
                self._reconstruct_output(batch, batch0, n_batch, n_frames, n_freqs, add_to_aux=True, aux_batch=aux_batch)
        else: # 
            batch = batch_tmp

        if self.repeat_residual_module and self.reconstructor_repeat_residual_module:
            if self.split_compress > 1:
                mixture = self.compress_conv(mixture) # [B, emb_dim//self.split_compress, T, F]
            # 将mixture扩展到与batch相同的维度 [B, emb_dim, T, F] -> [B, num_spk, emb_dim, T, F] or [B, emb_dim//self.split_compress, T, F] -> [B, num_spk, emb_dim//self.split_compress, T, F]
            mixture_expanded = mixture.unsqueeze(1).expand(-1, self.num_spk, -1, -1, -1)
        """ reconstructor """
        for ii in range(self.reconstructor_n_layers * self.reconstructor_repeat_times):
            layer_idx = ii % self.reconstructor_n_layers
            if self.repeat_residual_module and self.reconstructor_repeat_residual_module:
                batch_reshaped = rearrange(mixture_expanded + batch, "b j h t f -> (b j) h t f")
                if self.split_compress > 1:
                    batch_reshaped = self.reconstructor_concat_block(batch_reshaped)  # [B*num_spk, emb_dim//self.split_compress, T, F]
                else:
                    batch_reshaped = self.concat_block(batch_reshaped)  # [B*num_spk, emb_dim, T, F]
                batch = rearrange(batch_reshaped, "(b j) h t f -> b j h t f", j=self.num_spk)
            batch = self.reconstructor_blocks[layer_idx](batch)  # [B, num_spk, H, T, F]
            if (
                (
                    self.reconstructor_multi_decoder
                    and layer_idx == self.reconstructor_n_layers - 1
                )  # multi decoder for each last layer for each repeat time
                or (
                    ii
                    == self.reconstructor_repeat_times * self.reconstructor_n_layers - 1
                )  # last layer must has decoder loss
                or (
                    self.reconstructor_n_layers_multi_decoder
                    and self.reconstructor_multi_decoder
                    and layer_idx != self.reconstructor_n_layers - 1
                )  # multi decoder for each n layers for each repeat time
                or (
                    self.reconstructor_n_layers_multi_decoder
                    and not self.reconstructor_multi_decoder
                    and ii
                    >= (self.reconstructor_repeat_times - 1)
                    * self.reconstructor_n_layers
                )  # multi decoder for each n layers for only last repeat time
            ):
                # when using reconstructor_multi_decoder, all layers will be used for decoder
                is_last_layer = ii == self.reconstructor_repeat_times * self.reconstructor_n_layers - 1
                batch_tmp = self._reconstruct_output(
                    batch, batch0, n_batch, n_frames, n_freqs, 
                    add_to_aux=not is_last_layer, aux_batch=aux_batch
                )

        others = OrderedDict()
        others["aux_speech_pre"] = aux_batch
        return batch_tmp, ilens, others

    @property
    def num_spk(self):
        return self._num_spk


