
import glob
import random
import subprocess
import copy
import logging
import os
import re
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim
import torch.utils.data
import tqdm
from torch.cuda.amp import GradScaler, autocast

_BF16_AVAILABLE = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 8
)
from torch.nn.parallel import DistributedDataParallel as DDP

from m2se_vtts.utils import move_to_cuda
from m2se_vtts.utils.hparams import hparams

log = logging.getLogger(__name__)

def get_all_ckpts(work_dir):
    ckpts = glob.glob(os.path.join(work_dir, 'model_ckpt_steps_*.ckpt'))
    ckpts.sort(key=lambda x: -int(re.findall(r'steps_(\d+)', x)[0]))
    return ckpts

def get_last_checkpoint(work_dir, resume_from=None):
    if resume_from is not None:
        ckpt_path = os.path.join(work_dir, f'model_ckpt_steps_{resume_from}.ckpt')
        if os.path.exists(ckpt_path):
            return torch.load(ckpt_path, map_location='cpu'), ckpt_path
    ckpts = get_all_ckpts(work_dir)
    if len(ckpts) > 0:
        return torch.load(ckpts[0], map_location='cpu'), ckpts[0]
    return None, None

class Trainer:

    def __init__(self, work_dir, val_check_interval, tb_log_interval,
                 max_updates, num_sanity_val_steps=5,
                 accumulate_grad_batches=1, print_nan_grads=False,
                 resume_from_checkpoint=0, amp=False,
                 monitor_key='val_loss', monitor_mode='min',
                 num_ckpt_keep=3, save_best=True,
                 seed=1234, debug=False):
        self.work_dir = work_dir
        self.val_check_interval = val_check_interval
        self.tb_log_interval = tb_log_interval
        self.max_updates = max_updates
        self.num_sanity_val_steps = num_sanity_val_steps
        self.accumulate_grad_batches = accumulate_grad_batches
        self.print_nan_grads = print_nan_grads
        self.resume_from_checkpoint = resume_from_checkpoint
        self.amp = amp
        self.monitor_key = monitor_key
        self.monitor_mode = monitor_mode
        self.num_ckpt_keep = num_ckpt_keep
        self.save_best = save_best
        self.seed = seed
        self.debug = debug

        self.task = None
        self.optimizers = []
        self.global_step = 0
        self.current_epoch = 0
        self.best_val = float('inf') if monitor_mode == 'min' else float('-inf')

        amp_dtype_str = hparams.get('amp_dtype', 'bf16' if _BF16_AVAILABLE else 'fp16')
        self.amp_dtype = (
            torch.bfloat16 if amp_dtype_str == 'bf16' else torch.float16
        )
        self.use_scaler = amp and (self.amp_dtype == torch.float16)
        self.scaler = GradScaler(init_scale=2**10) if self.use_scaler else None

        self._hparams_snapshot = dict(hparams)

    def fit(self, task_cls):
        self.task_cls = task_cls
        gpus = list(range(torch.cuda.device_count()))
        if len(gpus) > 1:
            self.ddp_run(gpus)
        else:
            self.run_single_process(gpus[0] if len(gpus) == 1 else None)

    def test(self, task_cls):
        self.task_cls = task_cls
        gpus = list(range(torch.cuda.device_count()))
        gpu = gpus[0] if len(gpus) > 0 else None
        self.run_single_process(gpu, test=True)

    def ddp_run(self, gpus):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        if 'MASTER_PORT' not in os.environ:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                port = s.getsockname()[1]
            os.environ['MASTER_PORT'] = str(port)
        mp.spawn(self._ddp_worker, nprocs=len(gpus), args=(gpus,))

    def _ddp_worker(self, rank, gpus):
        hparams.clear()
        hparams.update(self._hparams_snapshot)

        seed = hparams.get('seed', 1234)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        gpu = gpus[rank]
        self.ddp_init(rank, len(gpus))
        torch.cuda.set_device(gpu)

        self.run_single_process(gpu)

    @staticmethod
    def ddp_init(rank, world_size):
        import datetime
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=120),
        )

    def configure_ddp(self, model, gpu):
        model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
        return model

    def run_single_process(self, gpu=None, test=False):
        task = self.task_cls()
        self.task = task
        task.trainer = self
        task.use_ddp = dist.is_initialized()

        model = task.build_model()
        if gpu is not None:
            torch.cuda.set_device(gpu)
            model.cuda(gpu)

        if not test:
            self.optimizers = task.configure_optimizers()
            if not isinstance(self.optimizers, (list, tuple)):
                self.optimizers = [self.optimizers]

        if task.use_ddp:
            model = self.configure_ddp(model, gpu)
            task.model = model

        self.restore_weights(task)

        os.makedirs(self.work_dir, exist_ok=True)
        if not task.use_ddp or dist.get_rank() == 0:
            task.build_tensorboard(save_dir=self.work_dir, name='tb_logs', version='lastest')

        if test:
            self._run_test(task, gpu)
        else:
            self.train(task, gpu)

    def train(self, task, gpu=None):
        if not task.use_ddp or dist.get_rank() == 0:
            print(f"| AMP: {self.amp} (dtype={self.amp_dtype}, scaler={self.use_scaler})")
        task.on_train_start()
        dataloader = task.train_dataloader()

        if task.use_ddp:
            rank = dist.get_rank()
            seed = hparams.get('seed', 1234)
            rank_seed = seed + rank
            random.seed(rank_seed)
            np.random.seed(rank_seed)
            torch.manual_seed(rank_seed)
            torch.cuda.manual_seed(rank_seed)

        if self.num_sanity_val_steps > 0:
            self.run_evaluation(task, gpu, num_batches=self.num_sanity_val_steps)

        while self.global_step < self.max_updates:
            task.current_epoch = self.current_epoch
            task.global_step = self.global_step
            task.on_epoch_start()

            pbar = tqdm.tqdm(dataloader, desc=f'Epoch {self.current_epoch}',
                             total=len(dataloader), dynamic_ncols=True)
            for batch_idx, batch in enumerate(pbar):
                if self.global_step >= self.max_updates:
                    break

                if gpu is not None:
                    batch = move_to_cuda(batch, gpu)

                self.run_training_batch(task, batch, batch_idx)

                if self.global_step % self.tb_log_interval == 0:
                    for k, v in task.training_losses_meter.items():
                        if v.cnt > 0:
                            pbar.set_postfix(**{k: f'{v.avg:.4f}'})

                if (self.global_step > 0 and
                        self.global_step % self.val_check_interval == 0):
                    self.run_evaluation(task, gpu)

            task.on_epoch_end()
            self.current_epoch += 1

        task.on_train_end()

    def run_training_batch(self, task, batch, batch_idx):
        for opt_idx, optimizer in enumerate(self.optimizers):
            if self.amp:
                with autocast(dtype=self.amp_dtype):
                    output = task.training_step(batch, batch_idx, opt_idx)
            else:
                output = task.training_step(batch, batch_idx, opt_idx)

            loss = output.get('loss')
            if loss is None:
                continue

            loss = loss / self.accumulate_grad_batches
            if self.use_scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            self._accum_count = getattr(self, '_accum_count', 0) + 1
            if self._accum_count >= self.accumulate_grad_batches:
                self._accum_count = 0
                if self.use_scaler:
                    self.scaler.unscale_(optimizer)

                task.on_before_optimization(opt_idx)

                if self.use_scaler:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                self.global_step += 1
                task.global_step = self.global_step
                task.on_after_optimization(
                    self.current_epoch, batch_idx, optimizer, opt_idx)

            if self.global_step % self.tb_log_interval == 0:
                tb_log = output.get('tb_log', {})
                if task.logger is not None:
                    self.log_metrics_to_tb(task.logger, tb_log, self.global_step)

    def run_evaluation(self, task, gpu=None, num_batches=None):
        task.eval()

        if hasattr(task, 'ema') and task.ema is not None:
            task.ema.apply_shadow(task.model)

        outputs = []
        dataloader = task.val_dataloader()
        max_val_batches = hparams.get('max_val_batches', None)
        if num_batches is None and max_val_batches is not None:
            num_batches = max_val_batches
        total = num_batches or len(dataloader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if num_batches is not None and batch_idx >= num_batches:
                    break
                if gpu is not None:
                    batch = move_to_cuda(batch, gpu)
                output = self.evaluate(task, batch, batch_idx)
                if output is not None:
                    outputs.append(output)

        if hasattr(task, 'ema') and task.ema is not None:
            task.ema.restore(task.model)

        result = task.validation_end(outputs)
        task.train()

        if task.logger is not None:
            tb_log = result.get('tb_log', {})
            self.log_metrics_to_tb(task.logger, tb_log, self.global_step)

        val_loss = result.get('val_loss', None)
        if val_loss is not None:
            improved = False
            if self.monitor_mode == 'min' and val_loss < self.best_val:
                self.best_val = val_loss
                improved = True
            elif self.monitor_mode == 'max' and val_loss > self.best_val:
                self.best_val = val_loss
                improved = True

            if not task.use_ddp or dist.get_rank() == 0:
                self.save_checkpoint(task, improved)

        if (num_batches is None
                and hasattr(task, 'log_audio_samples')
                and hparams.get('eval_audio_num_samples', 0) > 0):
            try:
                task.log_audio_samples()
            except Exception as e:
                log.warning(f"Audio sample logging failed: {e}")

        return result

    def evaluate(self, task, batch, batch_idx):
        try:
            output = task.validation_step(batch, batch_idx)
            return output
        except Exception as e:
            log.warning(f"Validation step failed: {e}")
            return None

    def _run_test(self, task, gpu=None):
        task.eval()
        task.test_start()
        outputs = []
        dataloader = task.test_dataloader()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc='Testing')):
                if gpu is not None:
                    batch = move_to_cuda(batch, gpu)
                output = task.test_step(batch, batch_idx)
                if output is not None:
                    outputs.append(output)
        result = task.test_end(outputs)
        return result

    def save_checkpoint(self, task, is_best=False):
        state_dict = {}
        for k, v in task.state_dict().items():
            key = k.replace('model.module.', 'model.', 1) if 'model.module.' in k else k
            state_dict[key] = v.cpu()
        ckpt = {
            'state_dict': state_dict,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_val': self.best_val,
        }
        opt_states = []
        for opt in self.optimizers:
            opt_states.append(opt.state_dict())
        ckpt['optimizer_states'] = opt_states

        if hasattr(task, 'scheduler') and task.scheduler is not None:
            ckpt['scheduler_state'] = task.scheduler.state_dict()

        if self.scaler is not None:
            ckpt['scaler_state'] = self.scaler.state_dict()

        if hasattr(task, 'ema') and task.ema is not None:
            ckpt['ema'] = task.ema.state_dict()

        ckpt_path = os.path.join(
            self.work_dir, f'model_ckpt_steps_{self.global_step}.ckpt')
        torch.save(ckpt, ckpt_path)
        print(f"| Saved checkpoint: {ckpt_path}")

        if is_best and self.save_best:
            best_path = os.path.join(self.work_dir, 'model_ckpt_best.pt')
            torch.save(ckpt, best_path)
            print(f"| Saved best checkpoint: {best_path}")

        self._clean_old_ckpts()

    def _clean_old_ckpts(self):
        ckpts = get_all_ckpts(self.work_dir)
        if len(ckpts) > self.num_ckpt_keep:
            for ckpt_path in ckpts[self.num_ckpt_keep:]:
                os.remove(ckpt_path)
                print(f"| Removed old checkpoint: {ckpt_path}")

    def restore_weights(self, task):
        checkpoint, ckpt_path = get_last_checkpoint(
            self.work_dir,
            resume_from=self.resume_from_checkpoint if self.resume_from_checkpoint > 0 else None,
        )
        if checkpoint is not None:
            state_dict = checkpoint.get('state_dict', {})

            if task.use_ddp:
                state_dict = {
                    (k.replace('model.', 'model.module.', 1)
                     if k.startswith('model.') and not k.startswith('model.module.')
                     else k): v
                    for k, v in state_dict.items()
                }

            missing, unexpected = task.load_state_dict(state_dict, strict=False)
            if missing:
                log.warning(f"  Missing keys in checkpoint: {missing[:10]}"
                            f"{'...' if len(missing) > 10 else ''}")
            if unexpected:
                log.warning(f"  Unexpected keys in checkpoint: {unexpected[:10]}"
                            f"{'...' if len(unexpected) > 10 else ''}")
            self.global_step = checkpoint.get('global_step', 0)
            self.current_epoch = checkpoint.get('current_epoch', 0)
            self.best_val = checkpoint.get('best_val', self.best_val)
            task.global_step = self.global_step
            task.current_epoch = self.current_epoch

            self.restore_opt_state(checkpoint)

            if (hasattr(task, 'scheduler') and task.scheduler is not None
                    and 'scheduler_state' in checkpoint):
                try:
                    task.scheduler.load_state_dict(checkpoint['scheduler_state'])
                    log.info("  Restored scheduler state.")
                except Exception as e:
                    log.warning(f"Could not restore scheduler state: {e}")

            if hasattr(task, 'ema') and task.ema is not None and 'ema' in checkpoint:
                task.ema.load_state_dict(checkpoint['ema'])
                log.info("  Restored EMA state.")

            if self.scaler is not None and 'scaler_state' in checkpoint:
                try:
                    self.scaler.load_state_dict(checkpoint['scaler_state'])
                    log.info("  Restored GradScaler state.")
                except Exception as e:
                    log.warning(f"Could not restore GradScaler state: {e}")

            print(f"| Restored from {ckpt_path} (step {self.global_step})")

    def restore_opt_state(self, checkpoint):
        opt_states = checkpoint.get('optimizer_states', [])
        for opt, state in zip(self.optimizers, opt_states):
            try:
                opt.load_state_dict(state)
            except Exception as e:
                log.warning(f"Could not restore optimizer state: {e}")

    @staticmethod
    def log_metrics_to_tb(logger, metrics, step):
        for k, v in metrics.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    logger.add_scalar(f'{k}/{sub_k}', sub_v, step)
            else:
                v = Trainer.metrics_to_scalars(v)
                logger.add_scalar(k, v, step)

    @staticmethod
    def metrics_to_scalars(v):
        if isinstance(v, torch.Tensor):
            return v.item()
        return v
