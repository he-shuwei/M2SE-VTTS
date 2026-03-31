
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Optional

class DepthAdapter(nn.Module):

    def __init__(
        self,
        feature_dim: int = 1024,
        bottleneck_dim: int = 256,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'norm': nn.LayerNorm(feature_dim),
                'down': nn.Linear(feature_dim, bottleneck_dim),
                'act': nn.GELU(),
                'up': nn.Linear(bottleneck_dim, feature_dim),
                'dropout': nn.Dropout(dropout),
            }))

        self.scales = nn.Parameter(torch.ones(num_layers) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            h = layer['norm'](x)
            h = layer['down'](h)
            h = layer['act'](h)
            h = layer['up'](h)
            h = layer['dropout'](h)
            x = x + self.scales[i] * h
        return x

class HFCLIPFeatureExtractor(nn.Module):

    def __init__(
        self,
        model_path: str = "openai/clip-vit-large-patch14-336",
        vision_feature_dim: int = 1024,
        text_feature_dim: int = 768,
        projection_dim: int = 768,
        freeze: bool = True,
        depth_adapter_bottleneck: int = 256,
        depth_adapter_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vision_feature_dim = vision_feature_dim
        self.text_feature_dim = text_feature_dim
        self.projection_dim = projection_dim
        self.model_path = model_path

        try:
            from transformers import CLIPModel, CLIPProcessor
            self.clip_model = CLIPModel.from_pretrained(model_path)
            self.processor = CLIPProcessor.from_pretrained(model_path)
            self._model_loaded = True
            print(f"Loaded CLIP model from {model_path}")

            config = self.clip_model.config
            self.vision_feature_dim = config.vision_config.hidden_size
            self.text_feature_dim = config.text_config.hidden_size
            self.projection_dim = config.projection_dim
            self.num_patches = (config.vision_config.image_size // config.vision_config.patch_size) ** 2

            print(f"  Vision hidden_size: {self.vision_feature_dim}")
            print(f"  Text hidden_size: {self.text_feature_dim}")
            print(f"  Projection dim: {self.projection_dim}")
            print(f"  Num patches: {self.num_patches}")

        except Exception as e:
            print(f"Warning: Failed to load CLIP model: {e}")
            print("Using dummy encoder for development.")
            self.clip_model = None
            self._model_loaded = False
            self.num_patches = 576

        self._freeze_clip = freeze and self.clip_model is not None
        if self._freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.eval()

        self.depth_patch_adapter = DepthAdapter(
            feature_dim=vision_feature_dim,
            bottleneck_dim=depth_adapter_bottleneck,
            dropout=dropout,
            num_layers=depth_adapter_layers,
        )
        self.depth_global_adapter = DepthAdapter(
            feature_dim=vision_feature_dim,
            bottleneck_dim=depth_adapter_bottleneck,
            dropout=dropout,
            num_layers=depth_adapter_layers,
        )

    def train(self, mode=True):
        super().train(mode)
        if self._freeze_clip and self.clip_model is not None:
            self.clip_model.eval()
        return self

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._model_loaded:
            vision_model = self.clip_model.vision_model

            vision_outputs = vision_model(
                pixel_values=image,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = vision_outputs.last_hidden_state

            global_features = hidden_states[:, 0:1, :]
            patch_features = hidden_states[:, 1:, :]

            return patch_features, global_features
        else:
            B = image.shape[0]
            patch_features = torch.randn(B, self.num_patches, self.vision_feature_dim, device=image.device)
            global_features = torch.randn(B, 1, self.vision_feature_dim, device=image.device)
            return patch_features, global_features

    @torch.no_grad()
    def encode_text_global(self, text_tokens: torch.Tensor) -> torch.Tensor:
        if self._model_loaded:
            text_outputs = self.clip_model.text_model(
                input_ids=text_tokens,
                return_dict=True,
            )
            pooled = text_outputs.pooler_output
            return pooled.unsqueeze(1)
        else:
            B = text_tokens.shape[0]
            return torch.randn(B, 1, self.text_feature_dim, device=text_tokens.device)

    @torch.no_grad()
    def encode_text_local(self, text_tokens: torch.Tensor) -> torch.Tensor:
        if self._model_loaded:
            text_outputs = self.clip_model.text_model(
                input_ids=text_tokens,
                return_dict=True,
            )
            return text_outputs.last_hidden_state
        else:
            B, L = text_tokens.shape
            return torch.randn(B, L, self.text_feature_dim, device=text_tokens.device)

    def forward(
        self,
        rgb_image: torch.Tensor,
        depth_image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_rgb, global_rgb = self.encode_image(rgb_image)

        patch_depth, global_depth = self.encode_image(depth_image)

        return patch_rgb, global_rgb, patch_depth, global_depth

    def get_feature_dims(self) -> dict:
        return {
            'vision_hidden_size': self.vision_feature_dim,
            'text_hidden_size': self.text_feature_dim,
            'projection_dim': self.projection_dim,
            'num_patches': self.num_patches,
        }
