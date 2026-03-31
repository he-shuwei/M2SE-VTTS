
import math
import torch
import torch.nn as nn
from typing import Optional

from m2se_vtts.utils.hparams import hparams
from m2se_vtts.utils.commons import Embedding, SinusoidalPositionalEmbedding
from m2se_vtts.models.visual.spatial_env import SpatialEnvironmentEncoder
from m2se_vtts.models.visual_text_fusion import VisualTextEncoder, VisualTextEncoderNoVisual

class M2SEEncoder(nn.Module):

    def __init__(
        self,
        phone_encoder,
        hidden_size: int = None,
        use_visual: bool = None,
        clip_model_path: str = None,
        load_clip: bool = None,
    ):
        super().__init__()

        hidden_size = hidden_size or hparams.get('hidden_size', 512)
        self.hidden_size = hidden_size
        self.use_visual = use_visual if use_visual is not None else hparams.get('use_visual', True)

        if load_clip is None:
            load_clip = hparams.get('load_clip', False)
        self.load_clip = load_clip

        if clip_model_path is None:
            clip_model_path = hparams.get(
                'clip_model_path',
                'checkpoints/clip-vit-large-patch14-336'
            )

        self.embed_tokens = Embedding(
            len(phone_encoder), hidden_size, padding_idx=0
        )
        self.embed_scale = math.sqrt(hidden_size)
        self.embed_positions = SinusoidalPositionalEmbedding(
            hidden_size, 0, init_size=2000
        )
        self.dropout = nn.Dropout(hparams['dropout'])

        num_heads = hparams.get('num_heads', 8)
        num_layers = hparams.get('vt_enc_layers', 3)
        ffn_dim = hparams.get('ffn_hidden_size', 2048)
        dropout = hparams['dropout']
        attention_dropout = hparams.get('attention_dropout', 0.1)
        ffn_kernel_size = hparams.get('enc_ffn_kernel_size', 9)
        ffn_padding = hparams.get('ffn_padding', 'SAME')
        ffn_act = hparams.get('ffn_act', 'gelu')

        if self.use_visual:
            self.spatial_env = SpatialEnvironmentEncoder(
                vision_dim=hparams.get('vision_dim', 1024),
                text_dim=hparams.get('text_dim', 768),
                output_dim=hidden_size,
                num_heads=hparams.get('spatial_num_heads', 16),
                top_k=hparams.get('top_k_regions', 280),
                num_lgsu_iterations=hparams.get('lgsu_iterations', 2),
                dropout=dropout,
                clip_model_path=clip_model_path,
                load_clip=load_clip,
                default_num_caption_tokens=hparams.get('default_num_caption_tokens', 16),
            )

            self.vt_encoder = VisualTextEncoder(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=num_layers,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                ffn_kernel_size=ffn_kernel_size,
                ffn_padding=ffn_padding,
                ffn_act=ffn_act,
            )
        else:
            self.vt_encoder = VisualTextEncoderNoVisual(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                attention_dropout=attention_dropout,
                ffn_kernel_size=ffn_kernel_size,
                ffn_padding=ffn_padding,
                ffn_act=ffn_act,
            )

    def forward_embedding(self, txt_tokens):
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        positions = self.embed_positions(txt_tokens)
        x = x + positions
        x = self.dropout(x)
        return x

    def forward(
        self,
        txt_tokens,
        patch_rgb=None,
        global_rgb=None,
        patch_depth=None,
        global_depth=None,
        caption_global=None,
        caption_local=None,
        rgb_image=None,
        depth_image=None,
        caption_tokens=None,
        caption_features=None,
        visual_feat=None,
    ):
        phoneme_emb = self.forward_embedding(txt_tokens)

        text_padding_mask = txt_tokens.eq(0)

        if self.use_visual:
            if visual_feat is not None:
                vis_feat = visual_feat
            elif patch_rgb is not None and patch_depth is not None:
                h_v = self.spatial_env(
                    patch_rgb=patch_rgb,
                    global_rgb=global_rgb,
                    patch_depth=patch_depth,
                    global_depth=global_depth,
                    caption_global=caption_global,
                    caption_local=caption_local,
                    caption_features=caption_features,
                )
                vis_feat = h_v
            elif rgb_image is not None and depth_image is not None:
                h_v = self.spatial_env(
                    rgb_image=rgb_image,
                    depth_image=depth_image,
                    caption_global=caption_global,
                    caption_local=caption_local,
                    caption_features=caption_features,
                    caption_tokens=caption_tokens,
                )
                vis_feat = h_v
            else:
                B = txt_tokens.shape[0]
                vis_feat = torch.zeros(B, 1, self.hidden_size, device=txt_tokens.device)

            per_sample_max = vis_feat.abs().reshape(vis_feat.shape[0], -1).max(dim=-1).values
            has_visual = per_sample_max > 1e-6

            visual_mask = has_visual.unsqueeze(1).expand(-1, vis_feat.shape[1])

            encoder_out = self.vt_encoder(
                phoneme_emb, vis_feat,
                text_padding_mask=text_padding_mask,
                visual_mask=visual_mask,
            )
        else:
            encoder_out = self.vt_encoder(
                phoneme_emb,
                text_padding_mask=text_padding_mask,
            )

        return encoder_out
