import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import parse as V
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange
import math

is_torch_2_0_plus = V(torch.__version__) >= V("2.0.0")

class TFLocoformerBlock(nn.Module):
    def __init__(
        self,
        rope_freq,
        rope_time,
        rope_spk=None,
        spk_path_enable=False,
        # general setup
        emb_dim=128,
        norm_type="rmsgrouporm",
        num_groups=4,
        tf_order="ft",
        # self-attention related
        n_heads=4,
        flash_attention=False,
        attention_dim=128,
        # ffn related
        ffn_type="swiglu_conv1d",
        ffn_hidden_dim=384,
        conv1d_kernel=4,
        conv1d_shift=1,
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__()

        assert tf_order in ["tf", "ft"], tf_order
        self.tf_order = tf_order
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

        self.freq_path = LocoformerBlock(
            rope_freq,
            # general setup
            emb_dim=emb_dim,
            norm_type=norm_type,
            num_groups=num_groups,
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
        self.frame_path = LocoformerBlock(
            rope_time,
            # general setup
            emb_dim=emb_dim,
            norm_type=norm_type,
            num_groups=num_groups,
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
        self.spk_path_enable = spk_path_enable
        if spk_path_enable:
            self.spk_path = LocoformerBlock(
                rope_spk,
                # general setup
                emb_dim=emb_dim,
                norm_type=norm_type,
                num_groups=num_groups,
                # self-attention related
                n_heads=n_heads,
                flash_attention=flash_attention,
                attention_dim=attention_dim,
                # ffn related
                ffn_hidden_dim=ffn_hidden_dim,
                conv1d_kernel=conv1d_kernel,
                conv1d_shift=conv1d_shift,
                dropout=dropout,
                eps=eps,
            )

    def forward(self, input):
        """TF-Locoformer forward.

        input: torch.Tensor
            Input tensor, (n_batch, channel, n_frame, n_freq)
        """

        if self.tf_order == "ft":
            output = self.freq_frame_process(input)
        else:
            output = self.frame_freq_process(input)

        return output

    def freq_frame_process(self, input):
        # input: (b, h, t, f) or (b, j, h, t, f)
        if not self.spk_path_enable:
            output = rearrange(input, "b h t f -> b t f h")
            output = self.freq_path(output)

            output = rearrange(output, "b t f h -> b f t h")
            output = self.frame_path(output)

            output = rearrange(output, "b f t h -> b h t f")
        else:
            b, j, h, t, f = input.shape
            output = rearrange(input, "b j h t f -> b (j t) f h")
            output = self.freq_path(output)

            output = rearrange(output, "b (j t) f h -> b (j f) t h", j=j, t=t, f=f)
            output = self.frame_path(output)

            output = rearrange(output, "b (j f) t h -> b (t f) j h", j=j, t=t, f=f)
            output = self.spk_path(output, flatten=True)

            output = rearrange(output, "b (t f) j h -> b j h t f", j=j, t=t, f=f)
        return output

    def frame_freq_process(self, input):
        # input: (b, h, t, f) or (b, j, h, t, f)
        if not self.spk_path_enable:
            output = rearrange(input, "b h t f -> b f t h")
            output = self.frame_path(output)

            output = rearrange(output, "b f t h -> b t f h")
            output = self.freq_path(output)

            output = rearrange(output, "b t f h -> b h t f")
        else:
            b, j, h, t, f = input.shape
            output = rearrange(input, "b j h t f -> b (j f) t h")
            output = self.frame_path(output)

            output = rearrange(output, "b (j f) t h -> b (j t) f h", j=j, t=t, f=f)
            output = self.freq_path(output)

            output = rearrange(output, "b (j t) f h -> b (t f) j h", j=j, t=t, f=f)
            output = self.spk_path(output, flatten=True)

            output = rearrange(output, "b (t f) j h -> b j h t f", j=j, t=t, f=f)
        return output


class LocoformerBlock(nn.Module):
    def __init__(
        self,
        rope=None,
        # general setup
        emb_dim=128,
        norm_type="rmsgrouporm",
        num_groups=4,
        # self-attention related
        n_heads=4,
        flash_attention=False,
        attention_dim=128,
        # ffn related
        ffn_type=["swiglu_conv1d"],
        ffn_hidden_dim=[384],
        conv1d_kernel=4,
        conv1d_shift=1,
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__()

        FFN = {
            "conv1d": ConvDeconv1d,
            "swiglu_conv1d": SwiGLUConvDeconv1d,
        }
        Norm = {
            "layernorm": nn.LayerNorm,
            "rmsgroupnorm": RMSGroupNorm,
        }
        assert norm_type in Norm, norm_type

        self.macaron_style = isinstance(ffn_type, list) and len(ffn_type) == 2
        if self.macaron_style:
            assert (
                isinstance(ffn_hidden_dim, list) and len(ffn_hidden_dim) == 2
            ), "Two FFNs required when using Macaron-style model"

        # initialize FFN
        self.ffn_norm = nn.ModuleList([])
        self.ffn = nn.ModuleList([])
        for f_type, f_dim in zip(ffn_type[::-1], ffn_hidden_dim[::-1]):
            assert f_type in FFN, f_type
            if norm_type == "rmsgroupnorm":
                self.ffn_norm.append(Norm[norm_type](num_groups, emb_dim, eps=eps))
            else:
                self.ffn_norm.append(Norm[norm_type](emb_dim, eps=eps))
            self.ffn.append(
                FFN[f_type](
                    emb_dim,
                    f_dim,
                    conv1d_kernel,
                    conv1d_shift,
                    dropout=dropout,
                )
            )

        # initialize self-attention
        if norm_type == "rmsgroupnorm":
            self.attn_norm = Norm[norm_type](num_groups, emb_dim, eps=eps)
        else:
            self.attn_norm = Norm[norm_type](emb_dim, eps=eps)
        self.attn = MultiHeadSelfAttention(
            emb_dim,
            attention_dim=attention_dim,
            n_heads=n_heads,
            rope=rope,
            dropout=dropout,
            flash_attention=flash_attention,
        )

    def forward(self, x, flatten=False):
        """Locoformer block Forward.

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either the number of frames or freqs
        """
        B, T, F, C = x.shape

        if self.macaron_style:
            # FFN before self-attention
            input_ = x
            output = self.ffn_norm[-1](x)  # [B, T, F, C]
            output = self.ffn[-1](output)  # [B, T, F, C]
            output = output + input_
        else:
            output = x

        # Self-attention
        input_ = output
        output = self.attn_norm(output)
        output = output.view([B * T, F, C])
        output = self.attn(output, flatten=flatten)  # [B*T, F, C]
        output = output.view([B, T, F, C]) + input_

        # FFN after self-attention
        input_ = output
        output = self.ffn_norm[0](output)  # [B, T, F, C]
        output = self.ffn[0](output)  # [B, T, F, C]
        output = output + input_

        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        attention_dim,
        n_heads=8,
        dropout=0.0,
        rope=None,
        flash_attention=False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dropout = dropout

        self.rope = rope
        self.qkv = nn.Linear(emb_dim, attention_dim * 3, bias=False)
        self.aggregate_heads = nn.Sequential(
            nn.Linear(attention_dim, emb_dim, bias=False), nn.Dropout(dropout)
        )

        if flash_attention:
            self.flash_attention_config = dict(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            )
        else:
            self.flash_attention_config = dict(
                enable_flash=False, enable_math=True, enable_mem_efficient=True
            )

    def forward(self, input, flatten=False):
        # get query, key, and value
        query, key, value = self.get_qkv(input)

        # rotary positional encoding
        if self.rope is not None:
            query, key = self.apply_rope(query, key)

        if flatten:
            q_shape = query.shape  # (B, T, H, D)
            # flatten first two dimensions (B*T, H, D) to prevent errors
            query = query.flatten(0, 1)
            key = key.flatten(0, 1)
            value = value.flatten(0, 1)
        # pytorch 2.0 flash attention: q, k, v, mask, dropout, softmax_scale
        try:
            with torch.backends.cuda.sdp_kernel(**self.flash_attention_config):
                output = F.scaled_dot_product_attention(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                )
        except Exception as e:
            print("scaled_dot_product_attention failed with exception:")
            print(e)
            print("--- Debug info ---")
            print(
                "query shape:",
                query.shape,
                "dtype:",
                query.dtype,
                "device:",
                query.device,
            )
            print("key shape:", key.shape, "dtype:", key.dtype, "device:", key.device)
            print(
                "value shape:",
                value.shape,
                "dtype:",
                value.dtype,
                "device:",
                value.device,
            )
            print("Expected dtype: float16 or bfloat16, head_dim % 8 == 0")
            print("flash_attention_config:", self.flash_attention_config)
            raise
        if flatten:
            output = output.view(q_shape)  # (B, T, H, D)

        output = output.transpose(1, 2)  # (batch, seq_len, head, -1)
        output = output.reshape(output.shape[:2] + (-1,))
        return self.aggregate_heads(output)

    def get_qkv(self, input):
        n_batch, seq_len = input.shape[:2]
        x = self.qkv(input).reshape(n_batch, seq_len, 3, self.n_heads, -1)
        x = x.movedim(-2, 1)  # (batch, head, seq_len, 3, -1)
        query, key, value = x[..., 0, :], x[..., 1, :], x[..., 2, :]
        return query, key, value

    @torch.cuda.amp.autocast(enabled=False)
    def apply_rope(self, query, key):
        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        return query, key


class ConvDeconv1d(nn.Module):
    def __init__(
        self, dim, dim_inner, conv1d_kernel, conv1d_shift, dropout=0.0, **kwargs
    ):
        super().__init__()

        self.diff_ks = conv1d_kernel - conv1d_shift

        self.net = nn.Sequential(
            nn.Conv1d(dim, dim_inner, conv1d_kernel, stride=conv1d_shift),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """ConvDeconv1d forward

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either the number of frames or freqs
        """
        b, s1, s2, h = x.shape
        x = x.view(b * s1, s2, h)
        x = x.transpose(-1, -2)
        x = self.net(x).transpose(-1, -2)
        x = x[..., self.diff_ks // 2 : self.diff_ks // 2 + s2, :]
        return x.view(b, s1, s2, h)


class SwiGLUConvDeconv1d(nn.Module):
    def __init__(
        self, dim, dim_inner, conv1d_kernel, conv1d_shift, dropout=0.0, **kwargs
    ):
        super().__init__()

        self.conv1d = nn.Conv1d(dim, dim_inner * 2, conv1d_kernel, stride=conv1d_shift)

        self.swish = nn.SiLU()
        self.deconv1d = nn.ConvTranspose1d(
            dim_inner, dim, conv1d_kernel, stride=conv1d_shift
        )
        self.dropout = nn.Dropout(dropout)
        self.dim_inner = dim_inner
        self.diff_ks = conv1d_kernel - conv1d_shift
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

    def forward(self, x):
        """SwiGLUConvDeconv1d forward

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either the number of frames or freqs
        """
        b, s1, s2, h = x.shape
        x = x.contiguous().view(b * s1, s2, h)
        x = x.transpose(-1, -2)

        # padding
        seq_len = (
            math.ceil((s2 + 2 * self.diff_ks - self.conv1d_kernel) / self.conv1d_shift)
            * self.conv1d_shift
            + self.conv1d_kernel
        )
        x = F.pad(x, (self.diff_ks, seq_len - s2 - self.diff_ks))

        # conv-deconv1d
        x = self.conv1d(x)
        gate = self.swish(x[..., self.dim_inner :, :])
        x = x[..., : self.dim_inner, :] * gate
        x = self.dropout(x)
        x = self.deconv1d(x).transpose(-1, -2)

        # cut necessary part
        x = x[..., self.diff_ks : self.diff_ks + s2, :]
        return self.dropout(x).view(b, s1, s2, h)


class RMSGroupNorm(nn.Module):
    def __init__(self, num_groups, dim, eps=1e-8, bias=False):
        """
        Root Mean Square Group Normalization (RMSGroupNorm).
        Unlike Group Normalization in vision, RMSGroupNorm
        is applied to each TF bin.

        Args:
            num_groups: int
                Number of groups
            dim: int
                Number of dimensions
            eps: float
                Small constant to avoid division by zero.
            bias: bool
                Whether to add a bias term. RMSNorm does not use bias.

        """
        super().__init__()

        assert dim % num_groups == 0, (dim, num_groups)
        self.num_groups = num_groups
        self.dim_per_group = dim // self.num_groups

        self.gamma = nn.Parameter(torch.Tensor(dim).to(torch.float32))
        nn.init.ones_(self.gamma)

        self.bias = bias
        if self.bias:
            self.beta = nn.Parameter(torch.Tensor(dim).to(torch.float32))
            nn.init.zeros_(self.beta)
        self.eps = eps
        self.num_groups = num_groups

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input):
        others = input.shape[:-1]
        input = input.view(others + (self.num_groups, self.dim_per_group))

        # normalization
        norm_ = input.norm(2, dim=-1, keepdim=True)
        rms = norm_ * self.dim_per_group ** (-1.0 / 2)
        output = input / (rms + self.eps)

        # reshape and affine transformation
        output = output.view(others + (-1,))
        output = output * self.gamma
        if self.bias:
            output = output + self.beta

        return output


