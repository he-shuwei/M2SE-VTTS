
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple

class FeedForward(nn.Module):

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CrossAttentionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)
        attn_out, _ = self.attn(
            query=q_norm, key=kv_norm, value=kv_norm,
            key_padding_mask=key_padding_mask,
        )
        x = query + attn_out

        x = x + self.ff(self.norm_ff(x))
        return x

class LocalSpatialUnderstanding(nn.Module):

    def __init__(
        self,
        vision_dim: int = 1024,
        text_dim: int = 768,
        num_heads: int = 16,
        top_k: int = 280,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.top_k = top_k
        self.num_heads = num_heads

        self.caption_proj = nn.Linear(text_dim, vision_dim)

        self.enrich_rgb = CrossAttentionBlock(vision_dim, num_heads, dropout=dropout)

        self.enrich_depth = CrossAttentionBlock(vision_dim, num_heads, dropout=dropout)

        self.norm_score_q = nn.LayerNorm(vision_dim)
        self.norm_score_kv = nn.LayerNorm(vision_dim)
        self.score_attention = nn.MultiheadAttention(
            embed_dim=vision_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

        self.norm_rgb = nn.LayerNorm(vision_dim)
        self.norm_depth = nn.LayerNorm(vision_dim)

    def forward(
        self,
        caption_local: torch.Tensor,
        patch_rgb: torch.Tensor,
        patch_depth: torch.Tensor,
        caption_pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, M, _ = patch_rgb.shape
        k = min(self.top_k, M)

        caption_v = self.caption_proj(caption_local)

        enriched_rgb = self.enrich_rgb(patch_rgb, caption_v, key_padding_mask=caption_pad_mask)

        enriched_depth = self.enrich_depth(patch_depth, caption_v, key_padding_mask=caption_pad_mask)

        score_q = self.norm_score_q(caption_v)
        score_kv = self.norm_score_kv(patch_rgb)

        _, reverse_attn = self.score_attention(
            query=score_q, key=score_kv, value=score_kv,
            need_weights=True, average_attn_weights=True,
        )

        if caption_pad_mask is not None:
            vote_mask = (~caption_pad_mask).float().unsqueeze(-1)
            reverse_attn = reverse_attn * vote_mask
        patch_scores = reverse_attn.sum(dim=1)

        _, top_k_indices = torch.topk(patch_scores, k=k, dim=-1)

        idx_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, self.vision_dim)
        top_k_rgb = torch.gather(enriched_rgb, dim=1, index=idx_expanded)
        top_k_depth = torch.gather(enriched_depth, dim=1, index=idx_expanded)

        top_k_rgb = self.norm_rgb(top_k_rgb)
        top_k_depth = self.norm_depth(top_k_depth)

        return top_k_rgb, top_k_depth, top_k_indices

class LocalAwareGlobalSpatialUnderstanding(nn.Module):

    def __init__(
        self,
        vision_dim: int = 1024,
        text_dim: int = 768,
        num_heads: int = 16,
        num_iterations: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.num_iterations = num_iterations

        self.caption_proj = nn.Linear(text_dim, vision_dim)
        self.caption_global_proj = nn.Linear(text_dim, vision_dim)

        self.local_aware_rgb = nn.ModuleList([
            CrossAttentionBlock(vision_dim, num_heads, dropout=dropout)
            for _ in range(num_iterations)
        ])
        self.local_aware_depth = nn.ModuleList([
            CrossAttentionBlock(vision_dim, num_heads, dropout=dropout)
            for _ in range(num_iterations)
        ])

        self.semantic_guided_rgb = CrossAttentionBlock(vision_dim, num_heads, dropout=dropout)
        self.semantic_guided_depth = CrossAttentionBlock(vision_dim, num_heads, dropout=dropout)

        self.gate_proj = nn.Sequential(
            nn.Linear(vision_dim * 2, vision_dim),
            nn.GELU(),
            nn.Linear(vision_dim, vision_dim),
            nn.Sigmoid(),
        )

        self.global_modulation = nn.Sequential(
            nn.Linear(vision_dim * 2, vision_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vision_dim, vision_dim * 2),
        )

        self.caption_global_scale = nn.Parameter(torch.ones(1) * 0.1)

        self.norm_output = nn.LayerNorm(vision_dim)

    def forward(
        self,
        top_k_rgb: torch.Tensor,
        top_k_depth: torch.Tensor,
        patch_rgb: torch.Tensor,
        patch_depth: torch.Tensor,
        caption_local: torch.Tensor,
        caption_global: Optional[torch.Tensor] = None,
        global_rgb: Optional[torch.Tensor] = None,
        global_depth: Optional[torch.Tensor] = None,
        caption_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = top_k_rgb.shape[0]
        device = top_k_rgb.device

        caption_local_v = self.caption_proj(caption_local)

        if global_rgb is not None:
            if global_rgb.dim() == 2:
                global_rgb = global_rgb.unsqueeze(1)
            context_rgb = torch.cat([global_rgb, patch_rgb], dim=1)
        else:
            context_rgb = patch_rgb

        if global_depth is not None:
            if global_depth.dim() == 2:
                global_depth = global_depth.unsqueeze(1)
            context_depth = torch.cat([global_depth, patch_depth], dim=1)
        else:
            context_depth = patch_depth

        h_l_rgb = top_k_rgb
        h_l_depth = top_k_depth
        for i in range(self.num_iterations):
            h_l_rgb = self.local_aware_rgb[i](h_l_rgb, context_rgb)
            h_l_depth = self.local_aware_depth[i](h_l_depth, context_depth)

        h_g_rgb = self.semantic_guided_rgb(h_l_rgb, caption_local_v, key_padding_mask=caption_pad_mask)
        h_g_depth = self.semantic_guided_depth(h_l_depth, caption_local_v, key_padding_mask=caption_pad_mask)

        gate = self.gate_proj(torch.cat([h_g_rgb, h_g_depth], dim=-1))

        h_v = gate * h_g_rgb + (1 - gate) * h_g_depth

        if global_rgb is not None and global_depth is not None:
            g_rgb = global_rgb.squeeze(1) if global_rgb.dim() == 3 else global_rgb
            g_depth = global_depth.squeeze(1) if global_depth.dim() == 3 else global_depth

            global_concat = torch.cat([g_rgb, g_depth], dim=-1)
            modulation = self.global_modulation(global_concat)

            scale, shift = modulation.chunk(2, dim=-1)
            scale = 0.5 + torch.sigmoid(scale)
            shift = torch.tanh(shift)

            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
            h_v = scale * h_v + shift

        if caption_global is not None:
            caption_global_v = self.caption_global_proj(caption_global)
            if caption_global_v.dim() == 2:
                caption_global_v = caption_global_v.unsqueeze(1)
            h_v = h_v + self.caption_global_scale * caption_global_v

        h_v = self.norm_output(h_v)

        return h_v

class SpatialEnvironmentEncoder(nn.Module):

    def __init__(
        self,
        vision_dim: int = 1024,
        text_dim: int = 768,
        output_dim: int = 512,
        num_heads: int = 16,
        top_k: int = 280,
        num_lgsu_iterations: int = 2,
        dropout: float = 0.1,
        clip_model_path: Optional[str] = None,
        load_clip: bool = False,
        default_num_caption_tokens: int = 16,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.default_num_caption_tokens = default_num_caption_tokens

        self.fallback_caption_tokens = nn.Parameter(
            torch.randn(1, default_num_caption_tokens, text_dim) * 0.02
        )

        if load_clip and clip_model_path:
            from m2se_vtts.models.visual.clip_encoder import HFCLIPFeatureExtractor
            self.clip_encoder = HFCLIPFeatureExtractor(
                model_path=clip_model_path,
                vision_feature_dim=vision_dim,
                text_feature_dim=text_dim,
            )
        else:
            self.clip_encoder = None

        from m2se_vtts.models.visual.clip_encoder import DepthAdapter
        self.depth_patch_adapter = DepthAdapter(
            feature_dim=vision_dim,
            bottleneck_dim=256,
            dropout=dropout,
            num_layers=2,
        )
        self.depth_global_adapter = DepthAdapter(
            feature_dim=vision_dim,
            bottleneck_dim=256,
            dropout=dropout,
            num_layers=2,
        )

        self.local_spatial = LocalSpatialUnderstanding(
            vision_dim=vision_dim,
            text_dim=text_dim,
            num_heads=num_heads,
            top_k=top_k,
            dropout=dropout,
        )

        self.global_spatial = LocalAwareGlobalSpatialUnderstanding(
            vision_dim=vision_dim,
            text_dim=text_dim,
            num_heads=num_heads,
            num_iterations=num_lgsu_iterations,
            dropout=dropout,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(vision_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        patch_rgb: Optional[torch.Tensor] = None,
        global_rgb: Optional[torch.Tensor] = None,
        patch_depth: Optional[torch.Tensor] = None,
        global_depth: Optional[torch.Tensor] = None,
        caption_global: Optional[torch.Tensor] = None,
        caption_local: Optional[torch.Tensor] = None,
        rgb_image: Optional[torch.Tensor] = None,
        depth_image: Optional[torch.Tensor] = None,
        caption_tokens: Optional[torch.Tensor] = None,
        caption_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        use_preextracted = patch_rgb is not None and patch_depth is not None

        if use_preextracted:
            B = patch_rgb.shape[0]
            device = patch_rgb.device
        elif rgb_image is not None and depth_image is not None:
            if self.clip_encoder is None:
                raise RuntimeError("CLIP encoder not loaded for real-time extraction.")
            B = rgb_image.shape[0]
            device = rgb_image.device
            patch_rgb, global_rgb, patch_depth, global_depth = self.clip_encoder(
                rgb_image, depth_image
            )
        else:
            raise ValueError("Provide pre-extracted features or raw images.")

        _sample_nrg = patch_rgb.reshape(B, -1).abs().sum(dim=-1)
        _valid = (_sample_nrg > 0)
        _n_valid = _valid.sum().item()

        if _n_valid == 0:
            k = min(self.local_spatial.top_k, patch_rgb.shape[1])
            return patch_rgb.new_zeros(B, k, self.output_dim)

        if _n_valid < B:
            _idx = _valid.nonzero(as_tuple=True)[0]
            _sel = lambda t: t[_idx] if t is not None else None
            h_v_valid = self.forward(
                patch_rgb=patch_rgb[_idx],
                global_rgb=_sel(global_rgb),
                patch_depth=patch_depth[_idx],
                global_depth=_sel(global_depth),
                caption_global=_sel(caption_global),
                caption_local=_sel(caption_local),
                caption_features=_sel(caption_features),
            )
            h_v = patch_rgb.new_zeros(B, h_v_valid.shape[1], self.output_dim)
            _scatter_idx = _idx.view(-1, 1, 1).expand_as(h_v_valid)
            h_v = h_v.scatter(0, _scatter_idx, h_v_valid)
            return h_v

        if caption_global is None and caption_features is not None:
            caption_global = caption_features

        if self.clip_encoder is not None and caption_tokens is not None:
            if caption_global is None:
                caption_global = self.clip_encoder.encode_text_global(caption_tokens)
            if caption_local is None:
                caption_local = self.clip_encoder.encode_text_local(caption_tokens)

        if caption_global is None:
            caption_global = torch.zeros(B, 1, self.text_dim, device=device)
        if caption_local is None:
            caption_local = self.fallback_caption_tokens.expand(B, -1, -1).to(
                device=device, dtype=patch_rgb.dtype
            )

        caption_pad_mask = (caption_local.abs().sum(dim=-1) == 0)
        if not caption_pad_mask.any():
            caption_pad_mask = None
        else:
            all_masked = caption_pad_mask.all(dim=-1)
            if all_masked.any():
                caption_pad_mask = caption_pad_mask.clone()
                caption_pad_mask[all_masked, 0] = False

        patch_depth = self.depth_patch_adapter(patch_depth)
        global_depth = self.depth_global_adapter(global_depth)

        top_k_rgb, top_k_depth, _ = self.local_spatial(
            caption_local,
            patch_rgb,
            patch_depth,
            caption_pad_mask=caption_pad_mask,
        )

        h_v = self.global_spatial(
            top_k_rgb,
            top_k_depth,
            patch_rgb,
            patch_depth,
            caption_local,
            caption_global,
            global_rgb=global_rgb,
            global_depth=global_depth,
            caption_pad_mask=caption_pad_mask,
        )

        h_v = self.output_proj(h_v)

        return h_v

    def get_topk_indices(
        self,
        patch_rgb: torch.Tensor,
        patch_depth: torch.Tensor,
        caption_local: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = patch_rgb.shape[0]
        device = patch_rgb.device

        patch_depth = self.depth_patch_adapter(patch_depth)

        if caption_local is None:
            caption_local = self.fallback_caption_tokens.expand(B, -1, -1).to(
                device=device, dtype=patch_rgb.dtype)

        _, _, top_k_indices = self.local_spatial(
            caption_local, patch_rgb, patch_depth)
        return top_k_indices

    def get_output_dim(self) -> int:
        return self.output_dim
