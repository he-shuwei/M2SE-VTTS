
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2se_vtts.utils.hparams import hparams

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    if repeat:
        noise = torch.randn((1, *shape[1:]), device=device)
        return noise.repeat(shape[0], *((1,) * (len(shape) - 1)))
    else:
        return torch.randn(shape, device=device)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

class GaussianDiffusion(nn.Module):

    def __init__(
        self,
        denoise_fn,
        timesteps=1000,
        loss_type='l1',
        spec_min=-6.0,
        spec_max=1.5,
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type

        if isinstance(spec_min, list):
            spec_min = torch.FloatTensor(spec_min)[None, None, :]
        elif isinstance(spec_min, (int, float)):
            spec_min = torch.FloatTensor([spec_min])
        if isinstance(spec_max, list):
            spec_max = torch.FloatTensor(spec_max)[None, None, :]
        elif isinstance(spec_max, (int, float)):
            spec_max = torch.FloatTensor([spec_max])
        self.register_buffer('spec_min', spec_min)
        self.register_buffer('spec_max', spec_max)

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.num_timesteps = int(timesteps)

        to_torch = lambda x: torch.tensor(x, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer('posterior_variance', to_torch(posterior_variance))

        self.register_buffer('posterior_log_variance_clipped',
                             to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) /
                                      (1.0 - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1.0 - alphas_cumprod_prev) *
                                      np.sqrt(alphas) / (1.0 - alphas_cumprod)))

        gamma = hparams.get('min_snr_gamma', 0)
        if gamma > 0:
            snr = alphas_cumprod / (1.0 - alphas_cumprod)
            snr_weights = np.minimum(snr, gamma) / np.maximum(snr, 1e-8)
            snr_weights = np.maximum(snr_weights, 0.05)
        else:
            snr_weights = np.ones_like(alphas_cumprod)
        self.register_buffer('snr_weights', to_torch(snr_weights))

    def _denoise(self, x, t, cond, **kwargs):
        t_norm = t.float() / self.num_timesteps
        return self.denoise_fn(x, t_norm, cond, **kwargs)

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x_t, t, cond, attention_mask=None, clip_denoised=True):
        noise_pred = self._denoise(x_t, t, cond, attention_mask=attention_mask)

        x_start = self.predict_start_from_noise(x_t, t, noise_pred)

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x_t, t=t
        )

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x_t, t, cond, attention_mask=None, clip_denoised=True):
        b = x_t.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(
            x_t=x_t, t=t, cond=cond,
            attention_mask=attention_mask,
            clip_denoised=clip_denoised
        )

        noise = noise_like(x_t.shape, x_t.device)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, attention_mask=None):
        device = self.betas.device
        b = shape[0]

        x = torch.randn(shape, device=device)

        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(
                x, t, cond,
                attention_mask=attention_mask
            )

        return x

    @torch.no_grad()
    def ddim_sample(self, x_t, t, t_prev, cond, attention_mask=None,
                    clip_denoised=True, eta=0.0):
        noise_pred = self._denoise(x_t, t, cond, attention_mask=attention_mask)

        x_start = self.predict_start_from_noise(x_t, t, noise_pred)
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        alpha_t = extract(self.alphas_cumprod, t, x_t.shape)
        alpha_t_prev = torch.where(
            t_prev.view(-1, *([1] * (len(x_t.shape) - 1))) >= 0,
            extract(self.alphas_cumprod, t_prev.clamp(min=0), x_t.shape),
            torch.ones_like(alpha_t),
        )

        sigma = eta * torch.sqrt(
            (1 - alpha_t_prev) / (1 - alpha_t).clamp(min=1e-8)
            * (1 - alpha_t / alpha_t_prev.clamp(min=1e-8))
        ).clamp(min=0)

        pred_dir = torch.sqrt((1 - alpha_t_prev - sigma ** 2).clamp(min=0)) * noise_pred

        x_prev = torch.sqrt(alpha_t_prev) * x_start + pred_dir

        if eta > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma * noise

        return x_prev

    @torch.no_grad()
    def ddim_sample_loop(self, shape, cond, attention_mask=None,
                         ddim_steps=None, eta=0.0):
        device = self.betas.device
        b = shape[0]

        if ddim_steps is None:
            ddim_steps = hparams.get('ddim_steps', 20)

        timesteps = np.linspace(0, self.num_timesteps - 1, ddim_steps, dtype=int).tolist()
        timesteps = list(reversed(timesteps))

        x = torch.randn(shape, device=device)

        for i, t_val in enumerate(timesteps):
            t = torch.full((b,), t_val, device=device, dtype=torch.long)
            t_prev_val = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t_prev = torch.full((b,), t_prev_val, device=device, dtype=torch.long)
            x = self.ddim_sample(x, t, t_prev, cond,
                                 attention_mask=attention_mask, eta=eta)

        return x

    def p_losses(self, x_start, t, cond, attention_mask=None, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        noise_pred = self._denoise(
            x_noisy, t, cond, attention_mask=attention_mask
        )

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise_pred, noise, reduction='none')
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise_pred, noise, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if attention_mask is not None:
            loss = loss.float().mean(dim=-1)
            mask_frames = attention_mask.float()

            w = self.snr_weights[t]
            loss = loss * w[:, None]

            loss = (loss * mask_frames).sum() / mask_frames.sum().clamp(min=1.0)
        else:
            w = self.snr_weights[t]
            loss = loss.mean(dim=(1, 2))
            loss = (loss * w).mean()

        return loss

    def forward(self, mel, cond, attention_mask=None, infer=False):
        if not infer:
            b = mel.shape[0]
            device = mel.device

            x_start = self.norm_spec(mel)

            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            loss = self.p_losses(x_start, t, cond, attention_mask=attention_mask)
            return loss
        else:
            b = cond.shape[0]

            if cond.dim() == 3:
                T = cond.shape[1]
            else:
                raise ValueError(
                    "For inference, cond must be 3D [B, T, cond_dim] "
                    "to determine output length."
                )

            mel_bins = hparams.get('audio_num_mel_bins', 80)
            shape = (b, T, mel_bins)

            use_ddim = hparams.get('use_ddim', True)
            if use_ddim:
                ddim_steps = hparams.get('ddim_steps', 20)
                ddim_eta = hparams.get('ddim_eta', 0.0)
                x = self.ddim_sample_loop(
                    shape, cond,
                    attention_mask=attention_mask,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                )
            else:
                x = self.p_sample_loop(shape, cond, attention_mask=attention_mask)

            x = self.denorm_spec(x)

            return x

    @torch.no_grad()
    def _infer_with_cfg(self, mel_shape, cond, attention_mask=None,
                        cfg_scale=1.0, uncond=None):
        device = self.betas.device
        b = mel_shape[0]

        if uncond is None:
            uncond = torch.zeros_like(cond)

        cond_double = torch.cat([cond, uncond], dim=0)
        mask_double = (torch.cat([attention_mask, attention_mask], dim=0)
                       if attention_mask is not None else None)

        x = torch.randn(mel_shape, device=device)

        use_ddim = hparams.get('use_ddim', True)

        if use_ddim:
            ddim_steps = hparams.get('ddim_steps', 20)
            ddim_eta = hparams.get('ddim_eta', 0.0)
            timesteps = np.linspace(
                0, self.num_timesteps - 1, ddim_steps, dtype=int).tolist()
            timesteps = list(reversed(timesteps))

            for i, t_val in enumerate(timesteps):
                t = torch.full((b,), t_val, device=device, dtype=torch.long)
                t_prev_val = timesteps[i + 1] if i + 1 < len(timesteps) else -1

                x_double = torch.cat([x, x], dim=0)
                t_double = torch.cat([t, t], dim=0)
                noise_both = self._denoise(
                    x_double, t_double, cond_double,
                    attention_mask=mask_double,
                )
                noise_cond, noise_uncond = noise_both.chunk(2, dim=0)

                noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

                alpha_t = extract(self.alphas_cumprod, t, x.shape)
                if t_prev_val >= 0:
                    t_prev = torch.full((b,), t_prev_val, device=device, dtype=torch.long)
                    alpha_t_prev = extract(self.alphas_cumprod, t_prev, x.shape)
                else:
                    alpha_t_prev = torch.ones_like(alpha_t)

                x_start = self.predict_start_from_noise(x, t, noise_pred)
                x_start.clamp_(-1.0, 1.0)

                sigma = ddim_eta * torch.sqrt(
                    (1 - alpha_t_prev) / (1 - alpha_t).clamp(min=1e-8)
                    * (1 - alpha_t / alpha_t_prev.clamp(min=1e-8))
                ).clamp(min=0)
                pred_dir = torch.sqrt((1 - alpha_t_prev - sigma ** 2).clamp(min=0)) * noise_pred
                x = torch.sqrt(alpha_t_prev) * x_start + pred_dir
                if ddim_eta > 0:
                    x = x + sigma * torch.randn_like(x)
        else:
            for i in reversed(range(0, self.num_timesteps)):
                t = torch.full((b,), i, device=device, dtype=torch.long)

                x_double = torch.cat([x, x], dim=0)
                t_double = torch.cat([t, t], dim=0)
                noise_both = self._denoise(
                    x_double, t_double, cond_double,
                    attention_mask=mask_double,
                )
                noise_cond, noise_uncond = noise_both.chunk(2, dim=0)

                noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

                x_start = self.predict_start_from_noise(x, t, noise_pred)
                x_start.clamp_(-1.0, 1.0)

                model_mean, _, model_log_variance = self.q_posterior(
                    x_start=x_start, x_t=x, t=t
                )
                noise = noise_like(x.shape, x.device)
                nonzero_mask = (1 - (t == 0).float()).reshape(
                    b, *((1,) * (len(x.shape) - 1))
                )
                x = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        x = self.denorm_spec(x)

        return x
