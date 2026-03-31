
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from m2se_vtts.models.dit_modules import (
    AdaLayerNorm_Final,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding,
)

class InputEmbedding(nn.Module):

    def __init__(self, mel_dim: int, cond_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(mel_dim + cond_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.proj(torch.cat((x, cond), dim=-1))
        x = self.conv_pos_embed(x, mask=mask) + x
        return x

class DiT(nn.Module):

    def __init__(
        self,
        *,
        mel_dim: int = 80,
        cond_dim: int = 256,
        dim: int,
        depth: int = 8,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.1,
        qk_norm: Optional[str] = None,
        pe_attn_head: Optional[int] = None,
        attn_backend: str = "torch",
        attn_mask_enabled: bool = False,
        long_skip_connection: bool = False,
        checkpoint_activations: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        self.input_embed = InputEmbedding(mel_dim, cond_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                    drop_path_rate=dpr[i],
                )
                for i in range(depth)
            ]
        )

        self.long_skip_connection = (
            nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        )

        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

    def initialize_weights(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    @staticmethod
    def _ckpt_wrapper(module):
        def ckpt_forward(*inputs):
            return module(*inputs)
        return ckpt_forward

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len = x.shape[0], x.shape[1]

        mask = attention_mask.bool() if attention_mask is not None else None

        t = self.time_embed(t.float())

        x = self.input_embed(x, cond, mask=mask)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(
                    self._ckpt_wrapper(block), x, t, mask, rope,
                    use_reentrant=False,
                )
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output

def DiT_XL(**kwargs):
    return DiT(dim=1152, depth=28, heads=16, dim_head=72, **kwargs)

def DiT_L(**kwargs):
    return DiT(dim=1024, depth=24, heads=16, dim_head=64, **kwargs)

def DiT_B(**kwargs):
    return DiT(dim=768, depth=12, heads=12, dim_head=64, **kwargs)

def DiT_S(**kwargs):
    return DiT(dim=384, depth=12, heads=6, dim_head=64, **kwargs)

def DiT_Base(**kwargs):
    return DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, **kwargs)

def DiT_Small(**kwargs):
    return DiT(dim=768, depth=16, heads=12, dim_head=64, ff_mult=2, **kwargs)
