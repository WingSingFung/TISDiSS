import torch
import torch.nn as nn
from packaging.version import parse as V
is_torch_2_0_plus = V(torch.__version__) >= V("2.0.0")

class SpkSplitConv2d(torch.nn.Module):
    def __init__(
        self,
        dim,
        num_spks: int,
        conv2d_kernel,
        padding,
        dropout=0.0,
        split_compress: int = 1,
        **kwargs
    ):
        super().__init__()

        self.conv2d = nn.Conv2d(
            dim, dim * num_spks // split_compress, conv2d_kernel, padding=padding
        )
        self.dropout = nn.Dropout(dropout)
        self.num_spks = num_spks
        self.split_compress = split_compress

    def forward(self, x):
        """SpkSplitConv2d forward

        Args:
            x: torch.Tensor
                Input tensor, input: (b, h, t, f)
            output: (b, j, h, t, f)
        """

        b, h, t, f = x.shape

        x = self.conv2d(x)  # (b, h*num_spk/split_compress, t, f)
        x = x.view(b, self.num_spks, h//self.split_compress, t, f)  # (b, j, h/split_compress, t, f)
        return self.dropout(x)


class SpkSplitSwiGLUConvDeconv2d(torch.nn.Module):
    def __init__(
        self,
        dim,
        dim_inner,
        num_spks: int,
        conv2d_kernel,
        padding,
        dropout=0.0,
        split_compress: int = 1,
        **kwargs
    ):
        super().__init__()

        self.middle_dim = dim_inner * num_spks // split_compress
        self.conv2d = nn.Conv2d(
            dim, self.middle_dim * 2, conv2d_kernel, padding=padding
        )
        self.swish = nn.SiLU()
        self.deconv2d = nn.ConvTranspose2d(
            self.middle_dim, dim * num_spks // split_compress, conv2d_kernel, padding=padding
        )
        self.dropout = nn.Dropout(dropout)
        self.dim_inner = dim_inner
        self.num_spks = num_spks
        self.split_compress = split_compress

    def forward(self, x):
        """SwiGLUConvDeconv2d forward

        Args:
            x: torch.Tensor
                Input tensor, input: (b, h, t, f)
            output: (b, j, h, t, f)
        """

        b, h, t, f = x.shape

        x = self.conv2d(x)  # (b, h_inner*2*num_spk, t, f)
        gate = self.swish(x[:, self.middle_dim :, :, :])  # (b, h_inner*num_spk, t, f)
        x = x[:, : self.middle_dim, :, :] * gate  # (b, h_inner*num_spk, t, f)
        x = self.dropout(x)  # (b, h_inner*num_spk, t, f)
        x = self.deconv2d(x)  # (b, h*num_spk/split_compress, t, f)
        x = x.view(b, self.num_spks, h//self.split_compress, t, f)  # (b, j, h/split_compress, t, f)
        return self.dropout(x)
