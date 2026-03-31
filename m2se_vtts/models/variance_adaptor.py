
import logging
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from m2se_vtts.utils.commons import SinusoidalPositionalEmbedding
from m2se_vtts.utils.hparams import hparams

logger = logging.getLogger(__name__)

class VarAdaptorLayerNorm(torch.nn.LayerNorm):

    def __init__(self, nout, dim=-1):
        super(VarAdaptorLayerNorm, self).__init__(nout, eps=1e-5)
        self.dim = dim

    def forward(self, x):
        if self.dim == -1:
            return super(VarAdaptorLayerNorm, self).forward(x)
        return super(VarAdaptorLayerNorm, self).forward(
            x.transpose(1, -1)
        ).transpose(1, -1)

class DurationPredictor(torch.nn.Module):

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3,
                 dropout_rate=0.1, offset=1.0, padding='SAME'):
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.ConstantPad1d(
                    ((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                    if padding == 'SAME'
                    else (kernel_size - 1, 0), 0),
                torch.nn.Conv1d(in_chans, n_chans, kernel_size,
                                stride=1, padding=0),
                torch.nn.ReLU(),
                VarAdaptorLayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]

        if hparams['dur_loss'] in ['mse', 'huber']:
            odims = 1
        elif hparams['dur_loss'] == 'mog':
            odims = 15
        elif hparams['dur_loss'] == 'crf':
            odims = 32
            from torchcrf import CRF
            self.crf = CRF(odims, batch_first=True)

        self.linear = torch.nn.Linear(n_chans, odims)

    def _forward(self, xs, x_masks=None, is_inference=False):
        xs = xs.transpose(1, -1)
        for f in self.conv:
            xs = f(xs)
            if x_masks is not None:
                xs = xs * (1 - x_masks.float())[:, None, :]
        xs = self.linear(xs.transpose(1, -1))
        if x_masks is not None:
            xs = xs * (1 - x_masks.float())[:, :, None]

        if is_inference:
            return self.out2dur(xs), xs
        else:
            if hparams['dur_loss'] in ['mse']:
                xs = xs.squeeze(-1)
        return xs

    def out2dur(self, xs):
        if hparams['dur_loss'] in ['mse']:
            xs = xs.squeeze(-1)
            dur = torch.clamp(
                torch.round(xs.exp() - self.offset), min=0
            ).long()
        elif hparams['dur_loss'] == 'mog':
            raise NotImplementedError("MoG duration loss not implemented")
        elif hparams['dur_loss'] == 'crf':
            dur = torch.LongTensor(self.crf.decode(xs)).to(xs.device)
        return dur

    def forward(self, xs, x_masks=None):
        return self._forward(xs, x_masks, False)

    def inference(self, xs, x_masks=None):
        return self._forward(xs, x_masks, True)

class LengthRegulator(torch.nn.Module):

    def __init__(self, pad_value=0.0):
        super(LengthRegulator, self).__init__()
        self.pad_value = pad_value

    def forward(self, dur, dur_padding=None, alpha=1.0):
        assert alpha > 0
        dur = torch.round(dur.float() * alpha).long()
        if dur_padding is not None:
            dur = dur * (1 - dur_padding.long())
        token_idx = torch.arange(1, dur.shape[1] + 1)[None, :, None].to(dur.device)
        dur_cumsum = torch.cumsum(dur, 1)
        dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode='constant', value=0)
        pos_idx = torch.arange(dur.sum(-1).max())[None, None].to(dur.device)
        token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & \
                     (pos_idx < dur_cumsum[:, :, None])
        mel2ph = (token_idx * token_mask.long()).sum(1)
        return mel2ph

class PitchPredictor(torch.nn.Module):

    def __init__(self, idim, n_layers=5, n_chans=384, odim=2,
                 kernel_size=5, dropout_rate=0.1, padding='SAME'):
        super(PitchPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.ConstantPad1d(
                    ((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                    if padding == 'SAME'
                    else (kernel_size - 1, 0), 0),
                torch.nn.Conv1d(in_chans, n_chans, kernel_size,
                                stride=1, padding=0),
                torch.nn.ReLU(),
                VarAdaptorLayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, odim)
        self.embed_positions = SinusoidalPositionalEmbedding(
            idim, 0, init_size=4096)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))

    def forward(self, xs, x_masks=None):
        B, T, _ = xs.shape
        pos_ids = torch.arange(1, T + 1, device=xs.device).unsqueeze(0).expand(B, T)
        positions = self.pos_embed_alpha * self.embed_positions(pos_ids)
        xs = xs + positions
        xs = xs.transpose(1, -1)
        for f in self.conv:
            xs = f(xs)
            if x_masks is not None:
                xs = xs * (1 - x_masks.float())[:, None, :]
        xs = self.linear(xs.transpose(1, -1))
        if x_masks is not None:
            xs = xs * (1 - x_masks.float())[:, :, None]
        return xs

class EnergyPredictor(PitchPredictor):
    pass

def mel2ph_to_dur(mel2ph, T_txt, max_dur=None):
    B, _ = mel2ph.shape
    dur = mel2ph.new_zeros(B, T_txt + 1).scatter_add(
        1, mel2ph, torch.ones_like(mel2ph)
    )
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    return dur
