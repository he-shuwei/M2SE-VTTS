
import json
import logging
import math
import os
from functools import partial
import random
import sys
import traceback
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from functools import wraps
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from m2se_vtts.utils import (
    AvgrageMeter,
    batch_by_size,
    move_to_cuda,
    num_params,
    tensors_to_scalars,
)
from m2se_vtts.utils.hparams import hparams
from m2se_vtts.utils.pitch import denorm_f0
from m2se_vtts.utils.text import TokenTextEncoder

from m2se_vtts.data.dataset import M2SEDataset, ConcatM2SEDataset
from m2se_vtts.data.collator import M2SECollator

from m2se_vtts.models.m2se_vtts import M2SEVTTSDiffusion
from m2se_vtts.models.dit import DiT_S, DiT_B, DiT_L, DiT_XL, DiT_Base, DiT_Small

from m2se_vtts.tasks.trainer import Trainer

torch.multiprocessing.set_sharing_strategy(os.getenv('TORCH_SHARE_STRATEGY', 'file_system'))

log = logging.getLogger(__name__)

def data_loader(fn):
    @wraps(fn)
    def _get_data_loader(self):
        attr_name = '_lazy_' + fn.__name__
        try:
            value = getattr(self, attr_name)
        except AttributeError:
            try:
                value = fn(self)
            except AttributeError as e:
                traceback.print_exc()
                error = f'{fn.__name__}: An AttributeError was encountered: ' + str(e)
                raise RuntimeError(error) from e
            setattr(self, attr_name, value)
        return value
    return _get_data_loader

class BaseTask(nn.Module):

    def __init__(self):
        super().__init__()
        self.current_epoch = 0
        self.global_step = 0
        self.trainer = None
        self.use_ddp = False
        self.gradient_clip_norm = hparams.get('clip_grad_norm', 0)
        self.gradient_clip_val = hparams.get('clip_grad_value', 0)
        self.model = None
        self.training_losses_meter = None
        self.logger = None
        self.scheduler = None

    def build_model(self):
        raise NotImplementedError

    @data_loader
    def train_dataloader(self):
        raise NotImplementedError

    @data_loader
    def test_dataloader(self):
        raise NotImplementedError

    @data_loader
    def val_dataloader(self):
        raise NotImplementedError

    def build_scheduler(self, optimizer):
        return None

    def build_optimizer(self, model):
        raise NotImplementedError

    def configure_optimizers(self):
        optm = self.build_optimizer(self.model)
        self.scheduler = self.build_scheduler(optm)
        if isinstance(optm, (list, tuple)):
            return optm
        return [optm]

    def build_tensorboard(self, save_dir, name, version, **kwargs):
        root_dir = os.path.join(save_dir, name)
        os.makedirs(root_dir, exist_ok=True)
        log_dir = os.path.join(root_dir, "version_" + str(version))
        self.logger = SummaryWriter(log_dir=log_dir, **kwargs)

    def on_train_start(self):
        pass

    def on_epoch_start(self):
        self.training_losses_meter = {'total_loss': AvgrageMeter()}

    def _training_step(self, sample, batch_idx, optimizer_idx):
        raise NotImplementedError

    def training_step(self, sample, batch_idx, optimizer_idx=-1):
        loss_ret = self._training_step(sample, batch_idx, optimizer_idx)
        if loss_ret is None:
            return {'loss': None}

        total_loss, log_outputs = loss_ret
        log_outputs = tensors_to_scalars(log_outputs)

        for k, v in log_outputs.items():
            if k not in self.training_losses_meter:
                self.training_losses_meter[k] = AvgrageMeter()
            if not np.isnan(v):
                self.training_losses_meter[k].update(v)
        total_loss_val = total_loss.item()
        if not np.isnan(total_loss_val):
            self.training_losses_meter['total_loss'].update(total_loss_val)

        if optimizer_idx >= 0:
            log_outputs[f'lr_{optimizer_idx}'] = (
                self.trainer.optimizers[optimizer_idx].param_groups[0]['lr']
            )

        progress_bar_log = log_outputs
        tb_log = {f'tr/{k}': v for k, v in log_outputs.items()}
        return {
            'loss': total_loss,
            'progress_bar': progress_bar_log,
            'tb_log': tb_log,
        }

    def on_before_optimization(self, opt_idx):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.grad_norm = total_norm

        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_norm)
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler.step()

    def on_epoch_end(self):
        loss_outputs = {
            k: round(v.avg, 4) for k, v in self.training_losses_meter.items()
        }
        print(f"Epoch {self.current_epoch} ended. Steps: {self.global_step}. {loss_outputs}")

    def on_train_end(self):
        pass

    def validation_step(self, sample, batch_idx):
        raise NotImplementedError

    def validation_end(self, outputs):
        all_losses_meter = {'total_loss': AvgrageMeter()}
        for output in outputs:
            if output is None or len(output) == 0:
                continue
            if isinstance(output, dict):
                assert 'losses' in output
                n = output.pop('nsamples', 1)
                losses = tensors_to_scalars(output['losses'])
                total_loss = output.get('total_loss', sum(losses.values()))
            else:
                n = 1
                total_loss, losses = output
                losses = tensors_to_scalars(losses)

            if isinstance(total_loss, torch.Tensor):
                total_loss = total_loss.item()

            for k, v in losses.items():
                if k not in all_losses_meter:
                    all_losses_meter[k] = AvgrageMeter()
                all_losses_meter[k].update(v, n)
            all_losses_meter['total_loss'].update(total_loss, n)

        loss_output = {k: v.avg for k, v in all_losses_meter.items()}

        if self.use_ddp and torch.distributed.is_initialized():
            for k in list(loss_output.keys()):
                meter = all_losses_meter[k]
                sum_t = torch.tensor(meter.sum, device='cuda')
                cnt_t = torch.tensor(float(meter.cnt), device='cuda')
                torch.distributed.all_reduce(sum_t, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(cnt_t, op=torch.distributed.ReduceOp.SUM)
                loss_output[k] = round((sum_t / cnt_t.clamp(min=1)).item(), 4)
        else:
            loss_output = {k: round(v, 4) for k, v in loss_output.items()}

        print(f"| Valid results: {loss_output}")

        monitor_key = hparams.get('valid_monitor_key', 'val_loss')
        if monitor_key.startswith('val/') and monitor_key[4:] in loss_output:
            val_loss = loss_output[monitor_key[4:]]
        else:
            val_loss = loss_output['total_loss']

        return {
            'tb_log': {f'val/{k}': v for k, v in loss_output.items()},
            'val_loss': val_loss,
        }

    def test_start(self):
        pass

    def test_step(self, sample, batch_idx):
        return self.validation_step(sample, batch_idx)

    def test_end(self, outputs):
        return self.validation_end(outputs)

    @classmethod
    def start(cls):
        if 'MASTER_PORT' not in os.environ or os.environ.get('_M2SE_SET_PORT') == '1':
            import socket
            for _ in range(20):
                port = random.randint(15000, 30000)
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    if s.connect_ex(('127.0.0.1', port)) != 0:
                        break
            os.environ['MASTER_PORT'] = str(port)
            os.environ['_M2SE_SET_PORT'] = '1'
        random.seed(hparams['seed'])
        np.random.seed(hparams['seed'])

        work_dir = hparams['work_dir']
        trainer = Trainer(
            work_dir=work_dir,
            val_check_interval=hparams['val_check_interval'],
            tb_log_interval=hparams['tb_log_interval'],
            max_updates=hparams['max_updates'],
            num_sanity_val_steps=(
                hparams['num_sanity_val_steps']
                if not hparams.get('validate', False) else 10000
            ),
            accumulate_grad_batches=hparams.get('accumulate_grad_batches', 1),
            print_nan_grads=hparams.get('print_nan_grads', False),
            resume_from_checkpoint=hparams.get('resume_from_checkpoint', 0),
            amp=hparams.get('amp', False),
            monitor_key=hparams.get('valid_monitor_key', 'val_loss'),
            monitor_mode=hparams.get('valid_monitor_mode', 'min'),
            num_ckpt_keep=hparams.get('num_ckpt_keep', 3),
            save_best=hparams.get('save_best', True),
            seed=hparams['seed'],
            debug=hparams.get('debug', False),
        )

        if not hparams.get('infer', False):
            trainer.fit(cls)
        else:
            trainer.test(cls)

    def on_keyboard_interrupt(self):
        pass

class ModelEMA:

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.num_updates = 0
        self._init_shadow(model)

    @staticmethod
    def _strip_ddp(name):
        return name[len('module.'):] if name.startswith('module.') else name

    def _init_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[self._strip_ddp(name)] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for name, param in model.named_parameters():
            key = self._strip_ddp(name)
            if param.requires_grad and key in self.shadow:
                if self.shadow[key].device != param.data.device:
                    self.shadow[key] = self.shadow[key].to(param.data.device)
                self.shadow[key].mul_(decay).add_(
                    param.data, alpha=1 - decay
                )

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            key = self._strip_ddp(name)
            if param.requires_grad and key in self.shadow:
                self.backup[key] = param.data.clone()
                shadow = self.shadow[key]
                if shadow.device != param.data.device:
                    shadow = shadow.to(param.data.device)
                    self.shadow[key] = shadow
                param.data.copy_(shadow)

    def restore(self, model):
        for name, param in model.named_parameters():
            key = self._strip_ddp(name)
            if param.requires_grad and key in self.backup:
                param.data.copy_(self.backup[key])
        self.backup = {}

    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay,
                'num_updates': self.num_updates}

    def load_state_dict(self, state):
        self.shadow = state['shadow']
        self.decay = state.get('decay', self.decay)
        self.num_updates = state.get('num_updates', 0)

def _build_dit(factory, hp):
    return factory(
        mel_dim=hp['audio_num_mel_bins'],
        cond_dim=hp['hidden_size'],
        dropout=hp.get('dit_dropout', 0.1),
        qk_norm=hp.get('dit_qk_norm', None),
        pe_attn_head=hp.get('dit_pe_attn_head', None),
        attn_backend=hp.get('dit_attn_backend', 'torch'),
        attn_mask_enabled=hp.get('dit_attn_mask_enabled', True),
        long_skip_connection=hp.get('dit_long_skip_connection', False),
        checkpoint_activations=hp.get('dit_checkpoint_activations', False),
        drop_path_rate=hp.get('dit_drop_path_rate', 0.0),
    )

DIFF_DECODERS = {
    'transformer-S': lambda hp: _build_dit(DiT_S, hp),
    'transformer-B': lambda hp: _build_dit(DiT_B, hp),
    'transformer-L': lambda hp: _build_dit(DiT_L, hp),
    'transformer-XL': lambda hp: _build_dit(DiT_XL, hp),
    'transformer-Base': lambda hp: _build_dit(DiT_Base, hp),
    'transformer-Small': lambda hp: _build_dit(DiT_Small, hp),
}

class EpochReshufflingBatchSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, max_tokens, max_sentences,
                 rank=None, world_size=None):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        batches = batch_by_size(
            self.dataset.ordered_indices(),
            self.dataset.num_tokens,
            max_tokens=self.max_tokens,
            max_sentences=self.max_sentences,
        )
        np.random.shuffle(batches)

        if self.world_size is not None and self.world_size > 1:
            usable = (len(batches) // self.world_size) * self.world_size
            batches = batches[:usable]
            batches = batches[self.rank::self.world_size]

        yield from batches

    def __len__(self):
        n = len(self.dataset)
        est = max(1, n // max(1, self.max_sentences))
        if self.world_size is not None and self.world_size > 1:
            est = est // self.world_size
        return max(1, est)

def _dl_worker_init(worker_id, hp_snapshot):
    from m2se_vtts.utils.hparams import hparams as _hp
    if len(_hp) == 0:
        _hp.update(hp_snapshot)

class M2SETask(BaseTask):

    def __init__(self):
        super().__init__()
        self.dataset_cls = M2SEDataset
        self.collator = M2SECollator()
        self.phone_encoder = self._build_phone_encoder()
        self.stats = {}
        self.ema = None
        self._load_f0_stats()

    def _load_f0_stats(self):
        if hparams.get('pitch_norm') != 'standard':
            return
        if hparams.get('f0_mean') is not None and hparams.get('f0_std') is not None:
            return
        f0_stats_path = os.path.join(hparams['binary_data_dir'], 'f0_mean_std.npy')
        if os.path.exists(f0_stats_path):
            stats = np.load(f0_stats_path)
            hparams['f0_mean'] = float(stats[0])
            hparams['f0_std'] = float(stats[1])
            log.info(f"Loaded F0 stats: mean={hparams['f0_mean']:.2f}, std={hparams['f0_std']:.2f}")
        else:
            raise FileNotFoundError(
                f"F0 stats file not found: {f0_stats_path}. "
                f"Run binarization (05_binarize.py) first, or set f0_mean/f0_std in config."
            )

    def _build_phone_encoder(self):
        phone_list_path = os.path.join(hparams['binary_data_dir'], 'phone_set.json')
        if os.path.exists(phone_list_path):
            with open(phone_list_path, 'r') as f:
                phone_list = json.load(f)
            return TokenTextEncoder(None, vocab_list=phone_list, replace_oov='UNK')
        return TokenTextEncoder(
            None,
            vocab_list=['PAD', 'UNK'] + [f'ph{i}' for i in range(200)],
        )

    def build_model(self):
        diff_decoder_type = hparams.get('diff_decoder_type', 'transformer-B')

        self.model = M2SEVTTSDiffusion(
            phone_encoder=self.phone_encoder,
            out_dims=hparams['audio_num_mel_bins'],
            denoise_fn=DIFF_DECODERS[diff_decoder_type](hparams),
            timesteps=hparams.get('timesteps', 100),
            loss_type=hparams.get('diff_loss_type', 'l1'),
            spec_min=hparams.get('spec_min'),
            spec_max=hparams.get('spec_max'),
        )
        num_params(self.model)

        enc_ckpt = hparams.get('pretrained_encoder_path')
        if enc_ckpt and os.path.exists(enc_ckpt):
            self._load_pretrained_encoder(enc_ckpt)

        dec_ckpt = hparams.get('pretrained_decoder_path')
        if dec_ckpt and os.path.exists(dec_ckpt):
            self._load_pretrained_decoder(dec_ckpt)

        if hparams.get('use_controlnet_finetune', False):
            from m2se_vtts.models.controlled_dit import ControlledDiT
            original_dit = self.model.diffusion.denoise_fn
            controlled = ControlledDiT(original_dit)
            self.model.diffusion.denoise_fn = controlled
            log.info(
                f"ControlNet fine-tuning enabled: locked DiT frozen, "
                f"trainable copy + {len(controlled.zero_convs)} zero convolutions created."
            )
            num_params(self.model, model_name="model (ControlNet)")

        if hparams.get('use_ema', True):
            ema_decay = hparams.get('ema_decay', 0.9999)
            self.ema = ModelEMA(self.model, decay=ema_decay)
            log.info(f"EMA enabled with decay={ema_decay}")

        return self.model

    def _load_pretrained_encoder(self, ckpt_path):
        log.info(f"Loading pre-trained encoder from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location='cpu')
        pretrained = state.get('state_dict', state)

        pretrained = {
            (k[len('model.'):] if k.startswith('model.') else k): v
            for k, v in pretrained.items()
        }

        layer_rename = {
            'layer_norm1':  'self_attn_layer_norm',
            'layer_norm2':  'ffn_layer_norm',
        }

        model_state = self.model.state_dict()
        to_load = {}

        for src_key, val in pretrained.items():
            if src_key.startswith('embed_tokens.') or src_key.startswith('embed_positions.'):
                dst_key = f'fs2.encoder.{src_key}'

                if src_key == 'embed_tokens.weight' and dst_key in model_state:
                    target_size = model_state[dst_key].shape[0]
                    if val.shape[0] == target_size + 1:
                        val = val[:target_size].clone()

            elif src_key.startswith('text_encoder.'):
                rest = src_key[len('text_encoder.'):]
                parts = rest.split('.')
                if len(parts) >= 3 and parts[0] == 'layers' and parts[2] in layer_rename:
                    parts[2] = layer_rename[parts[2]]
                dst_key = 'fs2.encoder.vt_encoder.' + '.'.join(parts)

            else:
                continue

            if dst_key in model_state and model_state[dst_key].shape == val.shape:
                to_load[dst_key] = val

        missing, unexpected = self.model.load_state_dict(to_load, strict=False)
        log.info(
            f"  Encoder: loaded {len(to_load)} tensors. "
            f"Missing {len(missing)}, unexpected {len(unexpected)} (cross-attn layers expected)."
        )

        n_reset = 0
        for name, param in self.model.named_parameters():
            if 'cross_attn_gate' in name:
                param.data.zero_()
                n_reset += 1
        if n_reset > 0:
            log.info(f"  Reset {n_reset} cross_attn_gate(s) to zero for stable fine-tuning.")

    def _load_pretrained_decoder(self, ckpt_path):
        log.info(f"Loading pre-trained decoder from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location='cpu')

        ema_state = state.get('ema')
        if ema_state is not None and 'shadow' in ema_state:
            log.info("  Using EMA shadow weights (smoother generalization).")
            pretrained = ema_state['shadow']
        else:
            pretrained = state.get('state_dict', state)

        pretrained = {
            (k[len('model.'):] if k.startswith('model.') else k): v
            for k, v in pretrained.items()
        }

        model_state = self.model.state_dict()
        to_load = {}

        for src_key, val in pretrained.items():
            if src_key.startswith('diffusion.'):
                dst_key = src_key
                if dst_key in model_state and model_state[dst_key].shape == val.shape:
                    to_load[dst_key] = val

        missing, unexpected = self.model.load_state_dict(to_load, strict=False)
        log.info(
            f"  Decoder: loaded {len(to_load)} tensors. "
            f"Missing {len(missing)}, unexpected {len(unexpected)}."
        )

    def build_optimizer(self, model):
        predictor_module_names = ('dur_predictor', 'pitch_predictor', 'energy_predictor')
        no_decay_keywords = (
            'bias', 'layer_norm', 'LayerNorm', 'norm',
            'embed_tokens', 'embed_positions',
            'fallback_caption_tokens',
        )

        main_decay, main_no_decay = [], []
        pred_decay, pred_no_decay = [], []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_predictor = any(pn in name for pn in predictor_module_names)
            is_no_decay = any(nd in name for nd in no_decay_keywords)

            if is_predictor:
                (pred_no_decay if is_no_decay else pred_decay).append(param)
            else:
                (main_no_decay if is_no_decay else main_decay).append(param)

        weight_decay = hparams.get('weight_decay', 0.01)
        predictor_lr = hparams.get('predictor_lr', hparams['lr'])

        param_groups = [
            {'params': main_decay, 'lr': hparams['lr'], 'weight_decay': weight_decay},
            {'params': main_no_decay, 'lr': hparams['lr'], 'weight_decay': 0.0},
        ]
        if pred_decay or pred_no_decay:
            if pred_decay:
                param_groups.append(
                    {'params': pred_decay, 'lr': predictor_lr, 'weight_decay': weight_decay},
                )
            if pred_no_decay:
                param_groups.append(
                    {'params': pred_no_decay, 'lr': predictor_lr, 'weight_decay': 0.0},
                )
            log.info(
                f"Optimizer: predictor lr={predictor_lr:.1e} "
                f"({len(pred_decay)}+{len(pred_no_decay)} params), "
                f"main lr={hparams['lr']:.1e} "
                f"({len(main_decay)}+{len(main_no_decay)} params)"
            )

        return torch.optim.AdamW(
            param_groups,
            lr=hparams['lr'],
            betas=(
                hparams.get('optimizer_adam_beta1', 0.9),
                hparams.get('optimizer_adam_beta2', 0.98),
            ),
            eps=hparams.get('optimizer_eps', 1e-6),
        )

    def build_scheduler(self, optimizer):
        from torch.optim.lr_scheduler import LambdaLR

        warmup_updates = hparams.get('warmup_updates', 4000)
        scheduler_type = hparams.get('scheduler_type', 'cosine')

        if scheduler_type == 'noam':
            hidden_size = hparams['hidden_size']

            def lr_lambda(step):
                step = max(step, 1)
                return hidden_size ** (-0.5) * min(
                    step ** (-0.5), step * warmup_updates ** (-1.5)
                )
        elif scheduler_type == 'cosine':
            max_updates = hparams['max_updates']

            def lr_lambda(step):
                if step < warmup_updates:
                    return step / max(1, warmup_updates)
                progress = (step - warmup_updates) / max(1, max_updates - warmup_updates)
                return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
            )
            return LambdaLR(optimizer, lr_lambda)

    @data_loader
    def train_dataloader(self):
        dataset = self.dataset_cls('train', shuffle=True)
        return self._build_dataloader(dataset, shuffle=True)

    @data_loader
    def val_dataloader(self):
        val_prefixes = hparams.get('val_prefixes', None)
        if val_prefixes:
            dataset = ConcatM2SEDataset(val_prefixes, shuffle=False)
        else:
            dataset = self.dataset_cls('valid', shuffle=False)
        return self._build_dataloader(dataset, shuffle=False, is_eval=True)

    @data_loader
    def test_dataloader(self):
        test_prefix = hparams.get('test_set_name', 'test_seen')
        dataset = self.dataset_cls(test_prefix, shuffle=False)
        return self._build_dataloader(dataset, shuffle=False, is_eval=True)

    def _build_dataloader(self, dataset, shuffle=True, is_eval=False):
        if is_eval:
            max_tokens = hparams.get('max_valid_tokens', -1)
            max_sentences = hparams.get('max_valid_sentences', 1)
            if max_tokens <= 0:
                max_tokens = hparams['max_tokens'] // 4
            if max_sentences <= 0:
                max_sentences = max(1, hparams['max_sentences'] // 4)
        else:
            max_tokens = hparams['max_tokens']
            max_sentences = hparams['max_sentences']

        rank = None
        world_size = None
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        if shuffle:
            sampler = EpochReshufflingBatchSampler(
                dataset, max_tokens, max_sentences,
                rank=rank, world_size=world_size,
            )
        else:
            batches = batch_by_size(
                dataset.ordered_indices(),
                dataset.num_tokens,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
            )
            if world_size is not None and world_size > 1:
                usable = (len(batches) // world_size) * world_size
                batches = batches[:usable]
                batches = batches[rank::world_size]
            sampler = batches

        num_workers = hparams.get('ds_workers', 2)
        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=self.collator,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=partial(_dl_worker_init, hp_snapshot=dict(hparams)),
            prefetch_factor=hparams.get('prefetch_factor', 2) if num_workers > 0 else None,
            persistent_workers=hparams.get('persistent_workers', False) and num_workers > 0,
        )

    def _training_step(self, sample, batch_idx, _):
        model_raw = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(model_raw, 'fs2'):
            model_raw.fs2._global_step = self.global_step
        loss_output = self.run_model(self.model, sample)
        total_loss = sum(
            v for v in loss_output.values()
            if isinstance(v, torch.Tensor) and v.requires_grad
        )
        loss_output['batch_size'] = sample['txt_tokens'].size(0)
        loss_output['total_loss'] = total_loss.item()
        return total_loss, loss_output

    def run_model(self, model, sample, return_output=False, infer=False):
        txt_tokens = sample['txt_tokens']
        target = sample['mels']
        mel2ph = sample['mel2ph']
        f0 = sample.get('f0')
        uv = sample.get('uv')
        energy = sample.get('energy')
        spk_embed = (
            sample.get('spk_embed')
            if not hparams.get('use_spk_id', False)
            else sample.get('spk_ids')
        )
        patch_rgb = sample.get('patch_rgb')
        global_rgb = sample.get('global_rgb')
        patch_depth = sample.get('patch_depth')
        global_depth = sample.get('global_depth')
        caption_global = sample.get('caption_global')
        caption_local = sample.get('caption_local')
        visual_feat = sample.get('visual_feat')
        caption_features = sample.get('caption_features')

        output = model(
            txt_tokens,
            mel2ph=mel2ph,
            spk_embed=spk_embed,
            ref_mels=target,
            f0=f0,
            uv=uv,
            energy=energy,
            infer=infer,
            patch_rgb=patch_rgb,
            global_rgb=global_rgb,
            patch_depth=patch_depth,
            global_depth=global_depth,
            caption_global=caption_global,
            caption_local=caption_local,
            visual_feat=visual_feat,
            caption_features=caption_features,
        )

        losses = {}
        if 'diff_loss' in output:
            losses['mel'] = output['diff_loss']
        if 'dur' in output:
            self._add_dur_loss(output['dur'], mel2ph, txt_tokens, losses)
        if hparams.get('use_pitch_embed', True):
            self._add_pitch_loss(output, sample, losses)
        if hparams.get('use_energy_embed', False):
            self._add_energy_loss(output.get('energy_pred'), energy, mel2ph, losses)

        if not return_output:
            return losses
        return losses, output

    def _add_dur_loss(self, dur_pred, mel2ph, txt_tokens, losses):
        from m2se_vtts.models.variance_adaptor import mel2ph_to_dur

        T_txt = txt_tokens.shape[1]
        dur_gt = mel2ph_to_dur(mel2ph, T_txt).float()
        nonpadding = (txt_tokens != 0).float()
        dur_gt_log = torch.log(dur_gt + 1)
        loss = F.mse_loss(dur_pred * nonpadding, dur_gt_log * nonpadding, reduction='none')
        loss = loss.sum() / nonpadding.sum().clamp(min=1.0)
        losses['dur'] = loss

    def _add_pitch_loss(self, output, sample, losses):
        pitch_pred = output.get('pitch_pred')
        if pitch_pred is None:
            return
        f0 = sample['f0']
        uv = sample.get('uv')
        nonpadding = (sample['mel2ph'] > 0).float()

        if pitch_pred.shape[-1] >= 2:
            pitch_pred_f0 = pitch_pred[:, :, 0]
            pitch_pred_uv = pitch_pred[:, :, 1]
        else:
            pitch_pred_f0 = pitch_pred.squeeze(-1)
            pitch_pred_uv = None

        loss_f0 = F.l1_loss(pitch_pred_f0 * nonpadding, f0 * nonpadding, reduction='none')
        voiced_weight = hparams.get('f0_voiced_weight', 1.0)
        if voiced_weight > 1.0 and uv is not None:
            frame_weight = torch.where(uv < 0.5, torch.full_like(uv, voiced_weight), torch.ones_like(uv))
            loss_f0 = loss_f0 * frame_weight
        loss_f0 = loss_f0.sum() / nonpadding.sum().clamp(min=1.0)
        losses['f0'] = loss_f0 * hparams.get('lambda_f0', 1.0)

        if pitch_pred_uv is not None and uv is not None:
            safe_logits = torch.where(nonpadding.bool(), pitch_pred_uv, torch.zeros_like(pitch_pred_uv))
            safe_targets = torch.where(nonpadding.bool(), uv, torch.zeros_like(uv))
            uv_smooth = hparams.get('uv_label_smoothing', 0.0)
            if uv_smooth > 0:
                safe_targets = safe_targets * (1 - uv_smooth) + uv_smooth / 2
            loss_uv = F.binary_cross_entropy_with_logits(
                safe_logits, safe_targets, reduction='none'
            )
            loss_uv = (loss_uv * nonpadding).sum() / nonpadding.sum().clamp(min=1.0)
            losses['uv'] = loss_uv * hparams.get('lambda_uv', 1.0)

    def _add_energy_loss(self, energy_pred, energy, mel2ph, losses):
        if energy_pred is None or energy is None:
            return
        nonpadding = (mel2ph > 0).float()
        loss = F.mse_loss(energy_pred * nonpadding, energy * nonpadding, reduction='none')
        loss = loss.sum() / nonpadding.sum().clamp(min=1.0)
        losses['e'] = loss * hparams.get('lambda_energy', 0.1)

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        super().on_after_optimization(epoch, batch_idx, optimizer, optimizer_idx)
        if self.ema is not None:
            self.ema.update(self.model)

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'], model_out = self.run_model(
            self.model, sample, return_output=True, infer=False
        )
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        return outputs

    def test_start(self):
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
            log.info("Applied EMA weights for testing.")

    def test_end(self, outputs):
        if outputs and isinstance(outputs[0], dict) and 'losses' not in outputs[0]:
            log.info(f"Test completed: {len(outputs)} samples generated.")
            result = {}
        else:
            result = super().test_end(outputs)
        if self.ema is not None:
            self.ema.restore(self.model)
        return result

    def test_step(self, sample, batch_idx):
        spk_embed = (
            sample.get('spk_embed')
            if not hparams.get('use_spk_id', False)
            else sample.get('spk_ids')
        )
        txt_tokens = sample['txt_tokens']
        mel2ph = sample['mel2ph'] if hparams.get('use_gt_dur', False) else None
        f0 = sample.get('f0') if hparams.get('use_gt_f0', False) else None
        uv = sample.get('uv') if hparams.get('use_gt_f0', False) else None

        patch_rgb = sample.get('patch_rgb')
        global_rgb = sample.get('global_rgb')
        patch_depth = sample.get('patch_depth')
        global_depth = sample.get('global_depth')
        caption_global = sample.get('caption_global')
        caption_local = sample.get('caption_local')
        visual_feat = sample.get('visual_feat')
        caption_features = sample.get('caption_features')

        use_cfg = hparams.get('use_cfg_inference', False)
        cfg_scale = hparams.get('cfg_guidance_scale', 2.0)

        outputs = self.model(
            txt_tokens,
            spk_embed=spk_embed,
            mel2ph=mel2ph,
            f0=f0,
            uv=uv,
            ref_mels=sample.get('mels'),
            infer=True,
            patch_rgb=patch_rgb,
            global_rgb=global_rgb,
            patch_depth=patch_depth,
            global_depth=global_depth,
            caption_global=caption_global,
            caption_local=caption_local,
            visual_feat=visual_feat,
            caption_features=caption_features,
            use_cfg=use_cfg,
            cfg_scale=cfg_scale,
        )

        sample['outputs'] = outputs['mel_out']
        sample['mel2ph_pred'] = outputs.get('mel2ph')
        if hparams.get('use_pitch_embed', True):
            sample['f0'] = denorm_f0(sample['f0'], sample.get('uv'), hparams)
            sample['f0_pred'] = outputs.get('f0_denorm')
        return sample

    def log_audio_samples(self, num_samples=None):
        import matplotlib.pyplot as plt
        from m2se_vtts.utils.audio import save_wav
        from m2se_vtts.utils.plot import (
            spec_to_figure, f0_to_figure, dur_to_figure, energy_to_figure,
            patch_selection_figure, scene_images_figure,
        )
        from m2se_vtts.models.variance_adaptor import mel2ph_to_dur

        if self.logger is None:
            return

        num_samples = num_samples or hparams.get('eval_audio_num_samples', 10)
        if num_samples <= 0:
            return

        if not hasattr(self, '_vocoder'):
            self._vocoder = None
        if self._vocoder is None:
            try:
                from m2se_vtts.vocoders import get_vocoder
                self._vocoder = get_vocoder()
            except Exception as e:
                log.warning(f"Could not load vocoder: {e}. Audio logging disabled.")
                self._vocoder = False
                return
        if self._vocoder is False:
            return
        if self._vocoder.model is None:
            log.warning("Vocoder model not loaded (checkpoint missing?). Skipping audio logging.")
            self._vocoder = False
            return

        sr = hparams.get('audio_sample_rate', 16000)

        sample_dir = os.path.join(
            hparams['work_dir'], 'samples', f'step_{self.global_step}')
        gt_dir = os.path.join(sample_dir, 'gt')
        gen_dir = os.path.join(sample_dir, 'gen')
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(gen_dir, exist_ok=True)

        dataset = M2SEDataset('valid', shuffle=False)
        collator = M2SECollator()
        gpu_id = torch.cuda.current_device() if torch.cuda.is_available() else 0

        if self.ema is not None:
            self.ema.apply_shadow(self.model)

        was_training = self.model.training
        self.model.eval()

        count = 0
        with torch.no_grad():
            for idx in range(min(len(dataset), num_samples)):
                try:
                    item = dataset[idx]
                    batch = collator([item])
                    batch = move_to_cuda(batch, gpu_id)

                    spk_embed = (
                        batch.get('spk_embed')
                        if not hparams.get('use_spk_id', False)
                        else batch.get('spk_ids')
                    )
                    output = self.model(
                        batch['txt_tokens'],
                        mel2ph=None,
                        spk_embed=spk_embed,
                        f0=None,
                        uv=None,
                        ref_mels=batch.get('mels'),
                        infer=True,
                        patch_rgb=batch.get('patch_rgb'),
                        global_rgb=batch.get('global_rgb'),
                        patch_depth=batch.get('patch_depth'),
                        global_depth=batch.get('global_depth'),
                        caption_global=batch.get('caption_global'),
                        caption_local=batch.get('caption_local'),
                        visual_feat=batch.get('visual_feat'),
                        caption_features=batch.get('caption_features'),
                    )

                    mel_gt = batch['mels'][0]
                    mel_gen = output['mel_out'][0]
                    gt_mel_len = (batch['mel2ph'][0] > 0).sum().item()
                    mel_gt_np = mel_gt[:gt_mel_len].cpu().numpy()
                    mel_gen_np = mel_gen.cpu().numpy()

                    wav_gt = self._vocoder.spec2wav(mel_gt_np)
                    wav_gen = self._vocoder.spec2wav(mel_gen_np)

                    save_wav(wav_gt, os.path.join(gt_dir, f'{count:04d}.wav'), sr)
                    save_wav(wav_gen, os.path.join(gen_dir, f'{count:04d}.wav'), sr)

                    self.logger.add_audio(
                        f'audio_gt/{count}',
                        torch.from_numpy(wav_gt).float().clamp(-1, 1).unsqueeze(0),
                        self.global_step, sr)
                    self.logger.add_audio(
                        f'audio_gen/{count}',
                        torch.from_numpy(wav_gen).float().clamp(-1, 1).unsqueeze(0),
                        self.global_step, sr)

                    fig_gt = spec_to_figure(mel_gt_np)
                    fig_gen = spec_to_figure(mel_gen_np)
                    self.logger.add_figure(
                        f'mel_gt/{count}', fig_gt, self.global_step)
                    self.logger.add_figure(
                        f'mel_gen/{count}', fig_gen, self.global_step)
                    plt.close(fig_gt)
                    plt.close(fig_gen)

                    txt_tokens = batch['txt_tokens'][0]
                    T_txt = (txt_tokens > 0).sum().item()
                    dur_gt = mel2ph_to_dur(
                        batch['mel2ph'][0:1], T_txt)[0][:T_txt]
                    dur_pred_log = output['dur'][0][:T_txt]
                    dur_pred = (dur_pred_log.exp() - 1).clamp(min=0).round().long()

                    phone_labels = self.phone_encoder.decode_list(
                        txt_tokens[:T_txt].cpu().tolist())

                    fig_dur = dur_to_figure(dur_gt, dur_pred, phone_labels)
                    self.logger.add_figure(
                        f'dur/{count}', fig_dur, self.global_step)
                    plt.close(fig_dur)

                    if hparams.get('use_pitch_embed', True):
                        uv_gt = batch.get('uv')
                        f0_gt_denorm = denorm_f0(
                            batch['f0'][0:1],
                            uv_gt[0:1] if uv_gt is not None else None,
                            hparams,
                        )[0][:gt_mel_len]

                        f0_pred = output.get('f0_denorm')
                        if f0_pred is not None:
                            f0_pred_len = min(f0_pred.shape[1], gt_mel_len)
                            f0_pred_val = f0_pred[0][:f0_pred_len]
                        else:
                            f0_pred_val = None

                        fig_f0 = f0_to_figure(f0_gt_denorm, f0_pred=f0_pred_val)
                        self.logger.add_figure(
                            f'f0/{count}', fig_f0, self.global_step)
                        plt.close(fig_f0)

                    if hparams.get('use_energy_embed', False):
                        energy_gt = batch.get('energy')
                        energy_pred = output.get('energy_pred')
                        if energy_gt is not None:
                            fig_e = energy_to_figure(
                                energy_gt[0][:gt_mel_len],
                                energy_pred[0][:min(energy_pred.shape[1], gt_mel_len)]
                                if energy_pred is not None else None,
                            )
                            self.logger.add_figure(
                                f'energy/{count}', fig_e, self.global_step)
                            plt.close(fig_e)

                    try:
                        self._log_visual_info(
                            count, dataset.indexed_ds[idx], batch, plt)
                    except Exception as e:
                        log.warning(f"Visual info logging {idx} failed: {e}")

                    count += 1

                except Exception as e:
                    log.warning(f"Audio sample {idx} failed: {e}")
                    continue

        if self.ema is not None:
            self.ema.restore(self.model)
        if was_training:
            self.model.train()

        log.info(f"Logged {count} audio samples -> {sample_dir}")

    _SPLIT_DIR_MAP = {
        'valid': 'val-mini', 'train': 'train',
        'test_seen': 'test-seen', 'test_unseen': 'test-unseen',
    }

    def _log_visual_info(self, sample_idx, raw_item, batch, plt):
        from m2se_vtts.utils.plot import (
            patch_selection_figure, scene_images_figure,
        )
        step = self.global_step

        caption = raw_item.get('caption', '') or ''
        spoken = raw_item.get('txt', '') or ''
        item_name = raw_item.get('item_name', '')
        info_md = (
            f"**Item:** `{item_name}`  \n\n"
            f"**Caption:** {caption}  \n\n"
            f"**Spoken content:** {spoken}"
        )
        self.logger.add_text(f'info/{sample_idx}', info_md, step)

        rgb_img, depth_img = self._load_scene_images(raw_item)
        if rgb_img is not None and depth_img is not None:
            fig_scene = scene_images_figure(
                rgb_img, depth_img, caption=caption, item_name=item_name)
            self.logger.add_figure(
                f'scene/{sample_idx}', fig_scene, step)
            plt.close(fig_scene)
        elif rgb_img is not None:
            self.logger.add_image(
                f'rgb/{sample_idx}',
                torch.from_numpy(rgb_img).permute(2, 0, 1),
                step)

        if (batch.get('patch_rgb') is not None
                and batch.get('patch_depth') is not None):
            model_raw = (self.model.module
                         if hasattr(self.model, 'module') else self.model)
            if hasattr(model_raw, 'fs2') and hasattr(
                    model_raw.fs2.encoder, 'spatial_env'):
                spatial_env = model_raw.fs2.encoder.spatial_env
                top_k_idx = spatial_env.get_topk_indices(
                    batch['patch_rgb'],
                    batch['patch_depth'],
                    batch.get('caption_local'),
                )
                top_k_np = top_k_idx[0].cpu().numpy()

                vis_img = rgb_img if rgb_img is not None else np.full(
                    (336, 336, 3), 128, dtype=np.uint8)
                fig_patches = patch_selection_figure(vis_img, top_k_np)
                self.logger.add_figure(
                    f'patch_selection/{sample_idx}', fig_patches, step)
                plt.close(fig_patches)

    def _load_scene_images(self, raw_item):
        soundspace_id = raw_item.get('soundspace_id', '')
        if not soundspace_id or '/' not in soundspace_id:
            return None, None

        scene_id, speech_id = soundspace_id.split('/', 1)
        processed_dir = hparams.get('processed_data_dir', 'data/processed_data')

        for split_dir in self._SPLIT_DIR_MAP.values():
            img_dir = os.path.join(
                processed_dir, 'images', split_dir, scene_id, speech_id)
            rgb_path = os.path.join(img_dir, 'rgb.png')
            if not os.path.exists(rgb_path):
                continue
            try:
                from PIL import Image
                rgb_arr = np.array(Image.open(rgb_path).convert('RGB'))
                depth_path = os.path.join(img_dir, 'depth.png')
                depth_arr = None
                if os.path.exists(depth_path):
                    depth_arr = np.array(Image.open(depth_path))
                return rgb_arr, depth_arr
            except Exception:
                return None, None
        return None, None

    @staticmethod
    def _get_clip_input_image(rgb_panorama, size=336):
        if rgb_panorama is None:
            return np.full((size, size, 3), 128, dtype=np.uint8)
        try:
            from PIL import Image
            return np.array(
                Image.fromarray(rgb_panorama).resize(
                    (size, size), Image.LANCZOS))
        except Exception:
            return np.full((size, size, 3), 128, dtype=np.uint8)
