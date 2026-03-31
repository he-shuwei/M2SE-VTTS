
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from m2se_vtts.utils.commons import (
    EncSALayer,
    Linear,
    SinusoidalPositionalEmbedding,
    TransformerFFNLayer,
    MultiheadAttention,
)
from m2se_vtts.utils.commons import LayerNorm
from m2se_vtts.utils.hparams import hparams

class VisualTextCrossAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, text_features, visual_features, visual_mask=None):
        B, T_text, H = text_features.shape
        _, T_vis, _ = visual_features.shape

        q = self.q_proj(text_features)
        k = self.k_proj(visual_features)
        v = self.v_proj(visual_features)

        q = q.view(B, T_text, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_vis, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_vis, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        if visual_mask is not None:
            mask_expanded = visual_mask.unsqueeze(1).unsqueeze(2)
            if visual_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(~mask_expanded, float('-inf'))
            else:
                attn_scores = attn_scores.masked_fill(mask_expanded == 0, float('-inf'))

        attn_max = attn_scores.max(dim=-1, keepdim=True).values
        all_masked = torch.isinf(attn_max) & (attn_max < 0)

        attn_scores_safe = torch.where(
            all_masked.expand_as(attn_scores) & torch.isinf(attn_scores),
            torch.zeros_like(attn_scores),
            attn_scores,
        )

        attn_probs = F.softmax(attn_scores_safe.float(), dim=-1).to(attn_scores_safe.dtype)

        attn_probs = attn_probs.masked_fill(all_masked.expand_as(attn_probs), 0.0)

        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_text, H)

        output = self.out_proj(attn_output)

        return output

class VisualTextEncoderLayer(nn.Module):

    def __init__(self, hidden_size, num_heads, ffn_dim,
                 dropout=0.1, attention_dropout=0.1,
                 ffn_kernel_size=9, ffn_padding='SAME', ffn_act='gelu'):
        super().__init__()
        self.hidden_size = hidden_size

        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.self_attn = MultiheadAttention(
            hidden_size, num_heads,
            self_attention=True,
            dropout=attention_dropout,
            bias=False
        )
        self.self_attn_dropout = nn.Dropout(dropout)

        self.cross_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = VisualTextCrossAttention(
            hidden_size, num_heads,
            dropout=attention_dropout
        )
        self.cross_attn_dropout = nn.Dropout(dropout)
        self.cross_attn_gate = nn.Parameter(torch.ones(1))

        self.ffn_layer_norm = nn.LayerNorm(hidden_size)
        self.ffn = TransformerFFNLayer(
            hidden_size, ffn_dim,
            kernel_size=ffn_kernel_size,
            dropout=dropout,
            padding=ffn_padding,
            act=ffn_act
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, text_features, visual_features,
                text_padding_mask=None, visual_mask=None):
        residual = text_features
        x = self.self_attn_layer_norm(text_features)
        x, _ = self.self_attn(
            query=x, key=x, value=x,
            key_padding_mask=text_padding_mask
        )
        x = self.self_attn_dropout(x)
        x = residual + x
        if text_padding_mask is not None:
            x = x * (1 - text_padding_mask.float()).transpose(0, 1)[..., None]

        residual = x
        x_bf = self.cross_attn_layer_norm(x).transpose(0, 1)
        cross_out = self.cross_attn(x_bf, visual_features, visual_mask)
        cross_out = cross_out.transpose(0, 1)
        cross_out = self.cross_attn_dropout(cross_out)
        x = residual + self.cross_attn_gate * cross_out
        if text_padding_mask is not None:
            x = x * (1 - text_padding_mask.float()).transpose(0, 1)[..., None]

        residual = x
        x = self.ffn_layer_norm(x)
        x = self.ffn(x)
        x = self.ffn_dropout(x)
        x = residual + x
        if text_padding_mask is not None:
            x = x * (1 - text_padding_mask.float()).transpose(0, 1)[..., None]

        return x

class VisualTextEncoder(nn.Module):

    def __init__(self, hidden_size, num_heads, num_layers,
                 ffn_dim=None, dropout=0.1, attention_dropout=0.1,
                 ffn_kernel_size=9, ffn_padding='SAME', ffn_act='gelu'):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 4 * hidden_size

        self.layers = nn.ModuleList([
            VisualTextEncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                ffn_kernel_size=ffn_kernel_size,
                ffn_padding=ffn_padding,
                ffn_act=ffn_act,
            )
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, text_features, visual_features,
                text_padding_mask=None, visual_mask=None):
        x = text_features.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, visual_features,
                      text_padding_mask=text_padding_mask,
                      visual_mask=visual_mask)

        x = self.layer_norm(x)

        x = x.transpose(0, 1)

        return x

class VisualTextEncoderNoVisual(nn.Module):

    def __init__(self, hidden_size, num_heads, num_layers,
                 dropout=0.1, attention_dropout=0.1,
                 ffn_kernel_size=9, ffn_padding='SAME', ffn_act='gelu'):
        super().__init__()
        self.layers = nn.ModuleList([
            EncSALayer(
                c=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                relu_dropout=dropout,
                kernel_size=ffn_kernel_size,
                padding=ffn_padding,
                act=ffn_act,
            )
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, text_features, text_padding_mask=None):
        x = text_features.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, encoder_padding_mask=text_padding_mask)

        x = self.layer_norm(x)

        x = x.transpose(0, 1)

        return x

def build_visual_text_encoder(use_visual=True):
    hidden_size = hparams['hidden_size']
    num_heads = hparams.get('num_heads', 2)
    num_layers = hparams.get('vt_enc_layers', hparams.get('enc_layers', 4))
    dropout = hparams.get('dropout', 0.1)
    attention_dropout = hparams.get('attention_dropout', 0.1)
    ffn_kernel_size = hparams.get('enc_ffn_kernel_size', 9)
    ffn_padding = hparams.get('ffn_padding', 'SAME')
    ffn_act = hparams.get('ffn_act', 'gelu')

    if use_visual:
        return VisualTextEncoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=4 * hidden_size,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_kernel_size=ffn_kernel_size,
            ffn_padding=ffn_padding,
            ffn_act=ffn_act,
        )
    else:
        return VisualTextEncoderNoVisual(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_kernel_size=ffn_kernel_size,
            ffn_padding=ffn_padding,
            ffn_act=ffn_act,
        )
