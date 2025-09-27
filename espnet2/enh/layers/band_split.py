import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
import numpy as np

from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype

# helper functions

def exists(val):
    return val is not None
def default(v, d):
    return v if exists(v) else d

# norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma
    
# bandsplit module

class BandSplitEncoder(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...],
            **kwargs
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            self.to_features.append(net)

    def forward(self, x): 
        # b t (f*2*m)
        x = x.split(self.dim_inputs, dim=-1) # f_band * (b t band_width[i])

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input) # b t d
            outs.append(split_output)

        return torch.stack(outs, dim=-2) # b t f_band d 

def MLP(
        dim_in,
        dim_out,
        dim_hidden=None,
        depth=1,
        activation=nn.Tanh
):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class BandSplitDecoder(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...],
            depth,
            mlp_expansion_factor=4,
            **kwargs
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            net = []
            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1)
            )

            self.to_freqs.append(mlp)

    def forward(self, x):
        # b t f_band d
        x = x.unbind(dim=-2) # f_band * (b t d)
        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features) # b t band_width[i]
            outs.append(freq_out)
        # f_band * (b t band_width[i])
        return torch.cat(outs, dim=-1) # b t (f*2*m)

class BandSplitEncoderConv1D(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...],
            eps: float = 1e-8,
            **kwargs
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                nn.GroupNorm(1, dim_in, eps=eps),
                nn.Conv1d(dim_in, dim, 3, padding=1)
            )

            self.to_features.append(net)

    def forward(self, x): 
        # b t (f*2*m)
        x = x.transpose(1, 2)  # b (f*2*m) t
        x = x.split(self.dim_inputs, dim=1) # f_band * (b band_width[i] t)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)  # b dim t
            outs.append(split_output)

        # Combine into b f_band dim t then convert to b t f_band dim
        result = torch.stack(outs, dim=1)  # b f_band dim t
        return result.permute(0, 3, 1, 2).contiguous()  # b t f_band dim 

class BandSplitDecoderConv1D(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...],
            **kwargs
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                nn.ConvTranspose1d(dim, dim_in, 3, padding=1)
            )

            self.to_freqs.append(net)

    def forward(self, x):
        # b t f_band d
        x = x.permute(0, 3, 1, 2)  # b d t f_band
        x = x.unbind(dim=-1)       # f_band * (b d t)
        
        outs = []
        for band_features, conv_transpose in zip(x, self.to_freqs):
            freq_out = conv_transpose(band_features)  # b dim_in t
            outs.append(freq_out)
        
        # First concatenate then transpose together
        result = torch.cat(outs, dim=1)  # b (f*2*m) t
        return result.transpose(-1, -2).contiguous()  # b t (f*2*m)

class TIGER_BandSplitEncoder(Module):
    """TIGER BandSplit Encoder for frequency-domain processing.
    
    This encoder splits the input spectrogram into multiple frequency bands
    and applies normalization and bottleneck to each band.
    
    Args:
        band_width: List of bandwidth for each frequency band
        feature_dim: Output feature dimension for each band
        eps: Small value for numerical stability
    """
    @beartype
    def __init__(
        self,
        band_width: List[int],
        feature_dim: int = 128,
        eps: float = 1e-8
    ):
        super().__init__()
        self.band_width = band_width
        self.feature_dim = feature_dim
        self.nband = len(band_width)
        self.eps = eps
        
        # Normalization and bottleneck for each band
        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.band_width[i] * 2, self.eps),
                    nn.Conv1d(self.band_width[i] * 2, self.feature_dim, 1)
                )
            )
    
    def forward(self, spec_RI: torch.Tensor, batch_size: int, nch: int) -> torch.Tensor:
        """
        Args:
            spec_RI: Stacked real and imaginary parts of spectrogram 
                    shape: (B*nch, 2, F, T)
            batch_size: Batch size
            nch: Number of channels
            
        Returns:
            subband_feature: Normalized and bottlenecked features 
                           shape: (B*nch, nband, feature_dim, T)
        """
        # Split to subbands
        subband_spec_RI = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec_RI.append(
                spec_RI[:, :, band_idx:band_idx + self.band_width[i]].contiguous()
            )
            band_idx += self.band_width[i]
        
        # Normalization and bottleneck
        subband_feature = []
        for i in range(len(self.band_width)):
            # Reshape: (B*nch, 2, BW, T) -> (B*nch, BW*2, T)
            reshaped = subband_spec_RI[i].view(batch_size * nch, self.band_width[i] * 2, -1)
            # Apply normalization and bottleneck
            subband_feature.append(self.BN[i](reshaped))
        
        # Stack features: (B*nch, nband, feature_dim, T)
        subband_feature = torch.stack(subband_feature, 1)
        return subband_feature


class TIGER_BandSplitDecoder(Module):
    """TIGER BandSplit Decoder for frequency-domain processing.
    
    This decoder applies masks to separated features and reconstructs
    the spectrogram for each source.
    
    Args:
        band_width: List of bandwidth for each frequency band
        feature_dim: Input feature dimension for each band
        num_sources: Number of sources to separate
    """
    @beartype
    def __init__(
        self,
        band_width: List[int],
        feature_dim: int = 128,
        num_sources: int = 2
    ):
        super().__init__()
        self.band_width = band_width
        self.feature_dim = feature_dim
        self.num_sources = num_sources
        self.nband = len(band_width)
        
        # Mask estimation for each band
        self.mask = nn.ModuleList([])
        for i in range(self.nband):
            self.mask.append(
                nn.Sequential(
                    nn.PReLU(),
                    nn.Conv1d(
                        self.feature_dim, 
                        self.band_width[i] * 4 * num_sources, 
                        1, 
                        groups=num_sources
                    )
                )
            )
    
    def forward(
        self, 
        sep_output: torch.Tensor, 
        subband_spec: List[torch.Tensor],
        batch_size: int,
        nch: int
    ) -> torch.Tensor:
        """
        Args:
            sep_output: Separated features from separator network
                       shape: (B*nch, nband, feature_dim, T) or (B*J, nband, feature_dim, T)
            subband_spec: List of complex spectrograms for each band
                         each element shape: (B*nch, BW, T)
            batch_size: Batch size (could be B*nch or B*J)
            nch: Number of channels
            
        Returns:
            sep_subband_spec: Separated complex spectrogram
                            shape: (B*nch*K, F, T) or (B*J*K, F, T) where K is num_sources
        """
        sep_subband_spec = []
        
        # Handle dimension mismatch between sep_output and subband_spec
        actual_batch_size = sep_output.shape[0]  # This could be B*nch or B*J
        expected_batch_size = subband_spec[0].shape[0]  # This is B*nch from encoder
        
        # If batch sizes don't match, we need to adjust subband_spec
        adjusted_subband_spec = subband_spec
        if actual_batch_size != expected_batch_size:
            # This happens when we have B*J speakers but subband_spec has B*nch
            # We replicate subband_spec for each separated path
            adjusted_subband_spec = []
            expansion_factor = actual_batch_size // expected_batch_size
            for spec_band in subband_spec:
                # spec_band shape: [B*nch, BW, T]
                # Replicate to match: [B*J, BW, T]
                spec_expanded = spec_band.repeat(expansion_factor, 1, 1)
                adjusted_subband_spec.append(spec_expanded)
        
        for i in range(self.nband):
            # Apply mask network
            this_output = self.mask[i](sep_output[:, i])
            # Reshape: (B*nch, 4*BW*K) -> (B*nch, 2, 2, K, BW, T)
            this_output = this_output.view(
                actual_batch_size, 2, 2, self.num_sources, 
                self.band_width[i], -1
            )
            
            # Apply gating: mask = linear * sigmoid(gate)
            this_mask = this_output[:, 0] * torch.sigmoid(this_output[:, 1])
            this_mask_real = this_mask[:, 0]  # (B*nch, K, BW, T)
            this_mask_imag = this_mask[:, 1]  # (B*nch, K, BW, T)
            
            # Force mask sum to 1 (for each frequency bin)
            this_mask_real_sum = this_mask_real.sum(1).unsqueeze(1)  # (B*nch, 1, BW, T)
            this_mask_imag_sum = this_mask_imag.sum(1).unsqueeze(1)  # (B*nch, 1, BW, T)
            this_mask_real = this_mask_real - (this_mask_real_sum - 1) / self.num_sources
            this_mask_imag = this_mask_imag - this_mask_imag_sum / self.num_sources
            
            # Apply complex mask using adjusted subband_spec
            est_spec_real = (
                adjusted_subband_spec[i].real.unsqueeze(1) * this_mask_real - 
                adjusted_subband_spec[i].imag.unsqueeze(1) * this_mask_imag
            )  # (B*nch, K, BW, T)
            est_spec_imag = (
                adjusted_subband_spec[i].real.unsqueeze(1) * this_mask_imag + 
                adjusted_subband_spec[i].imag.unsqueeze(1) * this_mask_real
            )  # (B*nch, K, BW, T)
            
            sep_subband_spec.append(torch.complex(est_spec_real, est_spec_imag))
        
        # Concatenate all bands: (B*nch, K, F, T)
        sep_subband_spec = torch.cat(sep_subband_spec, 2)
        
        # Reshape for output: (B*nch*K, F, T)
        return sep_subband_spec.view(actual_batch_size * self.num_sources, -1, sep_subband_spec.size(-1))

