
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from m2se_vtts.utils.hparams import hparams
from m2se_vtts.utils.commons import Embedding
from m2se_vtts.utils.pitch import f0_to_coarse, denorm_f0
from m2se_vtts.models.encoder import M2SEEncoder
from m2se_vtts.models.variance_adaptor import (
    DurationPredictor, LengthRegulator, PitchPredictor, EnergyPredictor
)
from m2se_vtts.models.diffusion import GaussianDiffusion

class M2SEVTTS(nn.Module):

    def __init__(self, phone_encoder, out_dims=80):
        super().__init__()
        self.hidden_size = hparams['hidden_size']
        self.out_dims = out_dims

        self.encoder = M2SEEncoder(phone_encoder, self.hidden_size)

        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        self.dur_predictor = DurationPredictor(
            self.hidden_size,
            n_layers=hparams['dur_predictor_layers'],
            n_chans=predictor_hidden,
            kernel_size=hparams['dur_predictor_kernel'],
            dropout_rate=hparams.get('predictor_dropout', 0.5),
        )
        self.length_regulator = LengthRegulator()

        if hparams['use_pitch_embed']:
            self.pitch_predictor = PitchPredictor(
                self.hidden_size,
                n_layers=hparams.get('predictor_layers', 5),
                n_chans=predictor_hidden,
                odim=2,
                kernel_size=hparams.get('predictor_kernel', 5),
                dropout_rate=hparams.get('predictor_dropout', 0.1),
            )
            self.pitch_embed = nn.Sequential(
                nn.Linear(1, self.hidden_size),
                nn.Dropout(hparams['dropout']),
            )
            if hparams.get('use_uv', True):
                self.uv_embed = nn.Embedding(2, self.hidden_size)

        if hparams.get('use_energy_embed', False):
            self.energy_predictor = EnergyPredictor(
                self.hidden_size,
                n_layers=hparams.get('predictor_layers', 5),
                n_chans=predictor_hidden,
                odim=1,
                kernel_size=hparams.get('predictor_kernel', 5),
                dropout_rate=hparams.get('predictor_dropout', 0.1),
            )
            self.energy_embed = nn.Sequential(
                nn.Linear(1, self.hidden_size),
                nn.Dropout(hparams['dropout']),
            )

        if hparams.get('use_spk_embed', False):
            spk_embed_dim = hparams.get('spk_embed_dim', 192)
            self.spk_embed_proj = nn.Linear(spk_embed_dim, self.hidden_size)
        if hparams.get('use_spk_id', False):
            self.spk_embed = Embedding(hparams['num_spk'], self.hidden_size)

    def forward(
        self,
        txt_tokens,
        mel2ph=None,
        spk_embed=None,
        ref_mels=None,
        f0=None,
        uv=None,
        energy=None,
        infer=False,
        patch_rgb=None,
        global_rgb=None,
        patch_depth=None,
        global_depth=None,
        caption_global=None,
        caption_local=None,
        rgb_image=None,
        depth_image=None,
        caption_features=None,
        visual_feat=None,
        skip_decoder=True,
        **kwargs,
    ):
        ret = {}

        encoder_out = self.encoder(
            txt_tokens,
            patch_rgb=patch_rgb,
            global_rgb=global_rgb,
            patch_depth=patch_depth,
            global_depth=global_depth,
            caption_global=caption_global,
            caption_local=caption_local,
            rgb_image=rgb_image,
            depth_image=depth_image,
            caption_features=caption_features,
            visual_feat=visual_feat,
        )

        if hparams.get('use_spk_embed', False) and spk_embed is not None:
            spk_emb = self.spk_embed_proj(spk_embed)[:, None, :]
            encoder_out = encoder_out + spk_emb
        elif hparams.get('use_spk_id', False) and spk_embed is not None:
            spk_emb = self.spk_embed(spk_embed)[:, None, :]
            encoder_out = encoder_out + spk_emb

        src_padding = txt_tokens.eq(0)
        dur_input = encoder_out.detach() + hparams['predictor_grad'] * (encoder_out - encoder_out.detach())

        if infer:
            dur, dur_raw = self.dur_predictor.inference(dur_input, src_padding)
            ret['dur'] = dur_raw.squeeze(-1) if dur_raw.dim() == 3 else dur_raw
            if mel2ph is None:
                mel2ph = self.length_regulator(dur, src_padding).detach()
        else:
            ret['dur'] = self.dur_predictor(dur_input, src_padding)

        ret['mel2ph'] = mel2ph

        decoder_inp = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph.unsqueeze(2).expand(-1, -1, self.hidden_size)
        decoder_inp = torch.gather(decoder_inp, 1, mel2ph_)

        tgt_nonpadding = (mel2ph > 0).float()
        tgt_padding = mel2ph.eq(0)

        decoder_inp_pre_pitch = decoder_inp

        if hparams['use_pitch_embed']:
            pitch_inp = decoder_inp.detach() + hparams['predictor_grad'] * (decoder_inp - decoder_inp.detach())
            pitch_pred = self.pitch_predictor(pitch_inp, x_masks=tgt_padding)
            ret['pitch_pred'] = pitch_pred

            if f0 is not None:
                if pitch_pred.shape[-1] >= 2:
                    ret['pitch_pred_f0'] = pitch_pred[:, :, 0]
                else:
                    ret['pitch_pred_f0'] = pitch_pred.squeeze(-1)

                ss_max = hparams.get('pitch_ss_prob', 0.0)
                if ss_max > 0 and self.training and hasattr(self, '_global_step'):
                    ss_warmup = hparams.get('pitch_ss_warmup', 5000)
                    ss_prob = min(ss_max, ss_max * self._global_step / max(1, ss_warmup))
                    use_pred = random.random() < ss_prob
                else:
                    use_pred = False

                if use_pred:
                    pred_f0_val = pitch_pred[:, :, 0].detach()
                    pitch_emb = self.pitch_embed(pred_f0_val[:, :, None])
                    if hparams.get('use_uv', True) and pitch_pred.shape[-1] >= 2:
                        pred_uv_binary = (pitch_pred[:, :, 1].detach() > 0).long()
                        pitch_emb = pitch_emb + self.uv_embed(pred_uv_binary)
                    elif hparams.get('use_uv', True) and uv is not None:
                        pitch_emb = pitch_emb + self.uv_embed(uv.long())
                else:
                    pitch_emb = self.pitch_embed(f0[:, :, None])
                    if hparams.get('use_uv', True) and uv is not None:
                        pitch_emb = pitch_emb + self.uv_embed(uv.long())
            else:
                if pitch_pred.shape[-1] >= 2:
                    pred_f0 = pitch_pred[:, :, 0]
                    pred_uv = pitch_pred[:, :, 1]
                else:
                    pred_f0 = pitch_pred.squeeze(-1)
                    pred_uv = None
                ret['f0_denorm'] = denorm_f0(pred_f0, pred_uv, hparams)
                pitch_emb = self.pitch_embed(pred_f0[:, :, None])
                if hparams.get('use_uv', True) and pred_uv is not None:
                    pred_uv_binary = (pred_uv > 0).long()
                    pitch_emb = pitch_emb + self.uv_embed(pred_uv_binary)

            decoder_inp = decoder_inp + pitch_emb

        if hparams.get('use_energy_embed', False):
            energy_inp = decoder_inp_pre_pitch.detach() + hparams['predictor_grad'] * (decoder_inp_pre_pitch - decoder_inp_pre_pitch.detach())
            energy_pred = self.energy_predictor(energy_inp, x_masks=tgt_padding).squeeze(-1)
            ret['energy_pred'] = energy_pred

            if energy is not None:
                energy_emb = self.energy_embed(energy[:, :, None])
            else:
                energy_emb = self.energy_embed(energy_pred[:, :, None])
            decoder_inp = decoder_inp + energy_emb

        ret['encoder_out'] = decoder_inp
        return ret

class M2SEVTTSDiffusion(nn.Module):

    def __init__(
        self,
        phone_encoder,
        out_dims=80,
        denoise_fn=None,
        timesteps=100,
        loss_type='l1',
        spec_min=None,
        spec_max=None,
    ):
        super().__init__()
        self.out_dims = out_dims
        self.uncond_prob = hparams.get('uncond_prob', 0.1)
        self.fs2 = M2SEVTTS(phone_encoder, out_dims)
        self.diffusion = GaussianDiffusion(
            denoise_fn=denoise_fn,
            timesteps=timesteps,
            loss_type=loss_type,
            spec_min=spec_min if spec_min is not None else -6.0,
            spec_max=spec_max if spec_max is not None else 1.5,
        )

    def _make_zero_visual(self, visual_feat):
        if visual_feat is not None:
            return torch.zeros_like(visual_feat)
        return None

    @staticmethod
    def _mask_visual_samples(feat, drop_mask):
        if feat is None:
            return feat
        mask = (~drop_mask).float()
        for _ in range(feat.dim() - 1):
            mask = mask.unsqueeze(-1)
        return feat * mask

    def forward(
        self,
        txt_tokens,
        mel2ph=None,
        spk_embed=None,
        ref_mels=None,
        f0=None,
        uv=None,
        energy=None,
        infer=False,
        patch_rgb=None,
        global_rgb=None,
        patch_depth=None,
        global_depth=None,
        caption_global=None,
        caption_local=None,
        rgb_image=None,
        depth_image=None,
        caption_features=None,
        visual_feat=None,
        use_cfg=False,
        cfg_scale=2.0,
        **kwargs,
    ):
        if not infer and self.training and self.uncond_prob > 0:
            B = txt_tokens.shape[0]
            drop_mask = torch.rand(B, device=txt_tokens.device) < self.uncond_prob
            if drop_mask.any():
                visual_feat = self._mask_visual_samples(visual_feat, drop_mask)
                patch_rgb = self._mask_visual_samples(patch_rgb, drop_mask)
                global_rgb = self._mask_visual_samples(global_rgb, drop_mask)
                patch_depth = self._mask_visual_samples(patch_depth, drop_mask)
                global_depth = self._mask_visual_samples(global_depth, drop_mask)
                caption_features = self._mask_visual_samples(caption_features, drop_mask)
                caption_global = self._mask_visual_samples(caption_global, drop_mask)
                caption_local = self._mask_visual_samples(caption_local, drop_mask)

        ret = self.fs2(
            txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed,
            ref_mels=ref_mels, f0=f0, uv=uv, energy=energy,
            infer=infer,
            patch_rgb=patch_rgb,
            global_rgb=global_rgb,
            patch_depth=patch_depth,
            global_depth=global_depth,
            caption_global=caption_global,
            caption_local=caption_local,
            rgb_image=rgb_image,
            depth_image=depth_image,
            caption_features=caption_features,
            visual_feat=visual_feat,
            skip_decoder=True,
        )

        cond = ret['encoder_out']

        if not infer:
            nonpadding = (mel2ph > 0).float()
            diff_loss = self.diffusion(
                ref_mels, cond, attention_mask=nonpadding, infer=False
            )
            ret['diff_loss'] = diff_loss
        else:
            mel_bins = hparams.get('audio_num_mel_bins', self.out_dims)
            mel_shape = (cond.shape[0], cond.shape[1], mel_bins)

            nonpadding = (ret['mel2ph'] > 0).float()

            if use_cfg:
                ret_uncond = self.fs2(
                    txt_tokens, mel2ph=ret['mel2ph'], spk_embed=spk_embed,
                    ref_mels=ref_mels, f0=f0, uv=uv, energy=energy,
                    infer=True,
                    patch_rgb=self._make_zero_visual(patch_rgb),
                    global_rgb=self._make_zero_visual(global_rgb),
                    patch_depth=self._make_zero_visual(patch_depth),
                    global_depth=self._make_zero_visual(global_depth),
                    caption_global=self._make_zero_visual(caption_global),
                    caption_local=self._make_zero_visual(caption_local),
                    visual_feat=self._make_zero_visual(visual_feat),
                    caption_features=self._make_zero_visual(caption_features),
                    rgb_image=None,
                    depth_image=None,
                    skip_decoder=True,
                )
                cond_uncond = ret_uncond['encoder_out']
                mel_out = self.diffusion._infer_with_cfg(
                    mel_shape, cond,
                    attention_mask=nonpadding,
                    cfg_scale=cfg_scale,
                    uncond=cond_uncond,
                )
            else:
                mel_out = self.diffusion(None, cond, attention_mask=nonpadding, infer=True)
            ret['mel_out'] = mel_out

        return ret

    def out2mel(self, mel_out):
        return self.diffusion.denorm_spec(mel_out)
