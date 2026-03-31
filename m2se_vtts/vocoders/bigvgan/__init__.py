
import json
import os
from typing import Optional

import numpy as np
import torch

from .env   import AttrDict
from .mel   import mel_spectrogram
from .model import BigVGAN

__all__ = ['BigVGANVocoder', 'mel_spectrogram', 'load_bigvgan_config']

_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), 'config.json')

def load_bigvgan_config(config_path: Optional[str] = None) -> AttrDict:
    path = config_path or _DEFAULT_CONFIG
    if not os.path.exists(path):
        raise FileNotFoundError(f"BigVGAN config not found: {path}")
    with open(path) as f:
        return AttrDict(json.load(f))

class BigVGANVocoder:

    def __init__(self, ckpt_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 device=None):
        self.device = torch.device(device or (
            'cuda' if torch.cuda.is_available() else 'cpu'))
        self.h     = None
        self.model = None

        if ckpt_path and os.path.exists(ckpt_path):
            self.load_model(ckpt_path, config_path)
        elif ckpt_path:
            print(f'[BigVGANVocoder] checkpoint not found: {ckpt_path}')

    def load_model(self, ckpt_path: str, config_path: Optional[str] = None):
        cfg_candidates = [
            os.path.join(os.path.dirname(ckpt_path), 'config.json'),
            config_path,
            _DEFAULT_CONFIG,
        ]
        h = None
        for p in cfg_candidates:
            if p and os.path.exists(p):
                with open(p) as f:
                    h = AttrDict(json.load(f))
                break
        if h is None:
            raise FileNotFoundError('BigVGAN config.json not found.')

        self.h = h
        model  = BigVGAN(h)
        ckpt   = torch.load(ckpt_path, map_location='cpu')
        state  = ckpt.get('generator', ckpt)
        model.load_state_dict(state)
        model.remove_weight_norm()
        model.eval()
        model.to(self.device)
        self.model = model
        print(f'[BigVGANVocoder] loaded {ckpt_path}  (device={self.device})')

    def to_device(self, device):
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device)

    @torch.no_grad()
    def spec2wav(self, mel: np.ndarray, f0=None) -> np.ndarray:
        if self.model is None:
            hop = self.h.hop_size if self.h else 256
            return np.zeros(mel.shape[0] * hop, dtype=np.float32)

        x   = torch.FloatTensor(mel).unsqueeze(0).transpose(1, 2).to(self.device)
        wav = self.model(x).squeeze(0).squeeze(0).cpu().numpy()
        return wav

    @torch.no_grad()
    def spec2wav_batch(self, mels: torch.Tensor):
        if self.model is None:
            hop = self.h.hop_size if self.h else 256
            return [np.zeros(m.shape[0] * hop, dtype=np.float32) for m in mels]

        x    = mels.transpose(1, 2).to(self.device)
        wavs = self.model(x)
        return [w.squeeze(0).cpu().numpy() for w in wavs]

def get_vocoder(hparams_dict=None) -> BigVGANVocoder:
    from m2se_vtts.utils.hparams import hparams as hp
    h = hparams_dict or hp
    return BigVGANVocoder(
        ckpt_path   = h.get('vocoder_ckpt', ''),
        config_path = h.get('vocoder_config', None),
    )
