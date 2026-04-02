"""
Flash DiT's codes are built from original Lightning DiT.
(https://github.com/hustvl/LightningDiT)

by Simin Huo
"""

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import PatchEmbed, Mlp
from models.swiglu_ffn import SwiGLUFFN 
from models.rmsnorm import RMSNorm


ATTN = os.environ.get('ATTN_IMPL', 'flash_attn')
assert ATTN in ['flash_attn', 'sdpa'], "ATTN_IMPL must be either 'flash_attn' or 'sdpa'"
if ATTN == 'flash_attn':
    try:
        from flash_attn import (
            flash_attn_qkvpacked_func,
            flash_attn_varlen_qkvpacked_func,
            flash_attn_varlen_kvpacked_func
        )
        from flash_attn.bert_padding import pad_input, unpad_input

    except ImportError:
        print("flash_attn is not installed. Please install it from https://github.com/Dao-AILab/flash-attention.")
        print("Falling back to sdpa.")
        ATTN = 'sdpa'

# support SDPA (PyTorch >= 2.0)
HAS_SDPA = hasattr(F, 'scaled_dot_product_attention')
if ATTN_IMPL == 'sdpa' and not HAS_SDPA:
    print("PyTorch does't support SDPA (Pytorch >= 2.0).")
    print("Falling back to math.")
    ATTN_IMPL = 'math'

# define attention backend
if ATTN_IMPL == 'flash_attn' and HAS_FLASH_ATTN:
    ACTIVE_ATTN = 'flash_attn'
elif HAS_SDPA and ATTN_IMPL in ['flash_attn', 'sdpa']:
    ACTIVE_ATTN = 'sdpa'
else:
    ACTIVE_ATTN = 'math'

print(f"Current Attention backend: {ACTIVE_ATTN}")

@torch.compile
def modulate(x, shift, scale):
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def window_partition(x, window_size: tuple):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size: tuple, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def rearrange(x, Hgroup, Wgroup):
    B, H, W, C = x.shape

    x = x.reshape(B, -1, Hgroup, W, C).transpose(1, 2)
    x = x.reshape(B, -1, W, C)
    x = x.reshape(B, H, -1, Wgroup, C).transpose(2, 3)
    x = x.reshape(B, H, -1, C)
    return x


def restore(x, Hgroup, Wgroup):
    B, H, W, C = x.shape

    x = x.reshape(B, H, Wgroup, -1, C).transpose(2, 3)
    x = x.reshape(B, H, -1, C)

    x = x.reshape(B, Hgroup, -1, W, C).transpose(1, 2)
    x = x.reshape(B, -1, W, C)
    return x


class Attention(nn.Module):
    """
    Attention module of FlashDiT.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        
        if use_rmsnorm:
            norm_layer = RMSNorm
            
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Same as DiT.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        Args:
            t: A 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim: The dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.
        Returns:
            An (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding
    
    @torch.compile
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    Same as DiT.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    @torch.compile
    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels)

        self.kernel_size = kernel_size
        self.in_channels = in_channels

    def forward(self, x):
        x = self.depthwise(x)
        return x
    

class FlashDiTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including: 
    - ROPE
    - QKNorm 
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False, 
        use_rmsnorm=False,
        wo_shift=False,
        window_size=(8,8),
        **block_kwargs
    ):
        super().__init__()
        
        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            
        # Initialize attention layer
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )

        self.dwconv = DepthwiseConv2d(hidden_size, kernel_size=3, stride=1, padding=1)

        self.window_size = window_size
        
        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )
            
        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 5 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 7 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    @torch.compile
    def forward(self, x, c):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp, gate_conv = self.adaLN_modulation(c).chunk(5, dim=1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_conv = self.adaLN_modulation(c).chunk(7, dim=1)

        B, N, C = x.shape
        H = W = int(math.sqrt(N))

        xc = modulate(self.norm1(x), shift_msa, scale_msa)

        xc = xc.view(B, H, W, C)

        xconv = xc.permute(0, 3, 1, 2).contiguous()
        xconv = self.dwconv(xconv).permute(0, 2, 3, 1).view(B, N, C)

        xc = rearrange(xc, H // self.window_size[0], W // self.window_size[1])
        x_windows = window_partition(xc, self.window_size)  # nW*B, window_size[0], window_size[1], C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size[0]*window_size[1], C
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        xattn = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        xattn = restore(xattn, H // self.window_size[0], W // self.window_size[1]).view(B, N, C)  # B H W C

        x = x + (1 + gate_conv).unsqueeze(1) * xconv + (1 + gate_msa).unsqueeze(1) * xattn
        x = x + (1 + gate_mlp).unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of FlashDiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class FlashDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
        use_qknorm=False,
        use_swiglu=False,
        use_rope=False,
        use_rmsnorm=False,
        wo_shift=False,
        use_checkpoint=False,
        window_size=(8, 8)
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels if not learn_sigma else in_channels * 2
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.use_checkpoint = use_checkpoint
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        self.blocks = nn.ModuleList([
            FlashDiTBlock(hidden_size, 
                     num_heads, 
                     mlp_ratio=mlp_ratio, 
                     use_qknorm=use_qknorm, 
                     use_swiglu=use_swiglu, 
                     use_rmsnorm=use_rmsnorm,
                     wo_shift=wo_shift,
                     window_size=window_size
                     ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
        self.apply(_basic_init)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in FlashDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t=None, y=None):
        """
        Forward pass of FlashDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        use_checkpoint: boolean to toggle checkpointing
        """

        use_checkpoint = self.use_checkpoint

        x = self.x_embedder(x) # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)

        for block in self.blocks:
            if use_checkpoint:
                x = checkpoint(block, x, c, use_reentrant=True)
            else:
                x = block(x, c)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=None, cfg_interval_start=None):
        """
        Forward pass of FlashDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        
        if cfg_interval is True:
            timestep = t[0]
            if timestep < cfg_interval_start:
                half_eps = cond_eps

        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                             FlashDiT Configs                              #
#################################################################################

def FlashDiT_XL_1(**kwargs):
    return FlashDiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)

def FlashDiT_XL_2(**kwargs):
    return FlashDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def FlashDiT_L_2(**kwargs):
    return FlashDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def FlashDiT_B_1(**kwargs):
    return FlashDiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def FlashDiT_B_2(**kwargs):
    return FlashDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def FlashDiT_1p0B_1(**kwargs):
    return FlashDiT(depth=24, hidden_size=1536, patch_size=1, num_heads=24, **kwargs)

def FlashDiT_1p0B_2(**kwargs):
    return FlashDiT(depth=24, hidden_size=1536, patch_size=2, num_heads=24, **kwargs)

def FlashDiT_1p6B_1(**kwargs):
    return FlashDiT(depth=28, hidden_size=1792, patch_size=1, num_heads=28, **kwargs)

def FlashDiT_1p6B_2(**kwargs):
    return FlashDiT(depth=28, hidden_size=1792, patch_size=2, num_heads=28, **kwargs)

FlashDiT_models = {
    'FlashDiT-B/1': FlashDiT_B_1, 'FlashDiT-B/2': FlashDiT_B_2,
    'FlashDiT-L/2': FlashDiT_L_2,
    'FlashDiT-XL/1': FlashDiT_XL_1, 'FlashDiT-XL/2': FlashDiT_XL_2,
    'FlashDiT-1p0B/1': FlashDiT_1p0B_1, 'FlashDiT-1p0B/2': FlashDiT_1p0B_2,
    'FlashDiT-1p6B/1': FlashDiT_1p6B_1, 'FlashDiT-1p6B/2': FlashDiT_1p6B_2,
}