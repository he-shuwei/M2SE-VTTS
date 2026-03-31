
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import warnings

import numpy as np
import soundfile as sf
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.WARNING)

def _build_model_and_load(hparams, ckpt_path):
    from m2se_vtts.tasks.m2se_vtts_task import M2SETask, ModelEMA

    task = M2SETask()
    task.build_model()
    model = task.model

    ckpt = torch.load(ckpt_path, map_location='cpu')

    if 'state_dict' in ckpt:
        sd = ckpt['state_dict']
        if 'model' in sd and isinstance(sd['model'], dict):
            state_dict = sd['model']
        else:
            state_dict = {k.replace('model.', '', 1): v
                          for k, v in sd.items() if k.startswith('model.')}
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'  Missing keys: {len(missing)} (first 3: {missing[:3]})')
    if unexpected:
        print(f'  Unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})')

    if 'ema' in ckpt and hparams.get('use_ema', True):
        ema = ModelEMA(model)
        ema.load_state_dict(ckpt['ema'])
        ema.apply_shadow(model)
        print('  Applied EMA weights.')

    model.eval()
    return model

def _move_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
    return batch

def _gpu_worker(gpu_id, indices, args, pred_dir, gt_dir, counter, lock):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')

    from m2se_vtts.utils.hparams import set_hparams, hparams
    sys.argv = ['', '--config', args.config]
    if args.exp_name:
        sys.argv += ['--exp_name', args.exp_name]
    set_hparams()
    hparams['infer'] = True

    model = _build_model_and_load(hparams, args.load_ckpt)
    model.to(device)

    from m2se_vtts.vocoders.bigvgan import get_vocoder
    vocoder = get_vocoder()
    vocoder.device = device
    if vocoder.model is not None:
        vocoder.model.to(device)

    from m2se_vtts.data.dataset import M2SEDataset
    from m2se_vtts.data.collator import M2SECollator

    test_prefix = args.test_set or hparams.get('test_set_name', 'test_seen')
    dataset = M2SEDataset(test_prefix, shuffle=False)
    collator = M2SECollator()

    use_cfg = (not args.no_cfg) and hparams.get('use_cfg_inference', False)
    cfg_scale = args.cfg_scale or hparams.get('cfg_guidance_scale', 2.0)
    sr = hparams.get('audio_sample_rate', 16000)
    batch_size = args.batch_size

    local_success = 0
    local_skip = 0
    local_error = 0

    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    desc = f'GPU{gpu_id}'
    for batch_indices in tqdm(batches, desc=desc, position=gpu_id, leave=True):
        try:
            samples_list = []
            item_names = []
            valid_mask = []

            for idx in batch_indices:
                item = dataset._get_item(idx)
                item_name = item.get('item_name', f'{idx:04d}')
                pred_path = os.path.join(pred_dir, f'{item_name}.wav')

                if os.path.exists(pred_path):
                    local_success += 1
                    valid_mask.append(False)
                    item_names.append(item_name)
                    samples_list.append(None)
                else:
                    valid_mask.append(True)
                    item_names.append(item_name)
                    samples_list.append(dataset[idx])

            active_samples = [s for s, v in zip(samples_list, valid_mask) if v]
            active_names = [n for n, v in zip(item_names, valid_mask) if v]

            if not active_samples:
                continue

            batch = collator(active_samples)
            batch = _move_to_device(batch, device)

            mel2ph = batch.get('mel2ph') if args.use_gt_dur else None
            f0 = batch.get('f0') if args.use_gt_f0 else None
            uv = batch.get('uv') if args.use_gt_f0 else None

            with torch.no_grad():
                output = model(
                    batch['txt_tokens'],
                    mel2ph=mel2ph,
                    spk_embed=batch.get('spk_embed'),
                    f0=f0,
                    uv=uv,
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
                    use_cfg=use_cfg,
                    cfg_scale=cfg_scale,
                )

            mel_out = output.get('mel_out')
            if mel_out is None or mel_out.numel() == 0:
                local_skip += len(active_names)
                continue

            mel2ph_out = output.get('mel2ph', batch.get('mel2ph'))
            mel_lengths = (mel2ph_out > 0).sum(dim=1)

            for i, name in enumerate(active_names):
                try:
                    mel_len = mel_lengths[i].item()
                    mel_i = mel_out[i, :mel_len]
                    wav_pred = vocoder.spec2wav(mel_i.cpu().numpy())
                    if wav_pred is not None and len(wav_pred) > 0:
                        sf.write(os.path.join(pred_dir, f'{name}.wav'), wav_pred, sr)

                    if not args.skip_gt and batch.get('mels') is not None:
                        gt_path = os.path.join(gt_dir, f'{name}.wav')
                        if not os.path.exists(gt_path):
                            mel_gt = batch['mels'][i, :mel_len].cpu().numpy()
                            wav_gt = vocoder.spec2wav(mel_gt)
                            if wav_gt is not None and len(wav_gt) > 0:
                                sf.write(gt_path, wav_gt, sr)

                    local_success += 1
                except Exception as e:
                    local_error += 1
                    if local_error <= 3:
                        tqdm.write(f'[GPU{gpu_id}] vocoder error {name}: {str(e)[:80]}')

        except Exception as e:
            local_error += len(batch_indices)
            if local_error <= 5:
                tqdm.write(f'[GPU{gpu_id} Error] batch: {str(e)[:120]}')

    with lock:
        counter[0] += local_success
        counter[1] += local_skip
        counter[2] += local_error

def main():
    parser = argparse.ArgumentParser(
        description='M2SE-VTTS batched multi-GPU inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--load_ckpt', required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--test_set', type=str, default=None)
    parser.add_argument('--num_gpus', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size per GPU')
    parser.add_argument('--use_gt_dur', action='store_true')
    parser.add_argument('--use_gt_f0', action='store_true')
    parser.add_argument('--no_cfg', action='store_true')
    parser.add_argument('--cfg_scale', type=float, default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--skip_gt', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.load_ckpt):
        print(f'[ERROR] Checkpoint not found: {args.load_ckpt}')
        sys.exit(1)

    from m2se_vtts.utils.hparams import set_hparams, hparams
    sys.argv = ['', '--config', args.config]
    if args.exp_name:
        sys.argv += ['--exp_name', args.exp_name]
    set_hparams()
    hparams['infer'] = True

    test_prefix = args.test_set or hparams.get('test_set_name', 'test_seen')

    pred_dir = os.path.join(args.output_dir, 'pred')
    gt_dir = os.path.join(args.output_dir, 'gt_recon')
    os.makedirs(pred_dir, exist_ok=True)
    if not args.skip_gt:
        os.makedirs(gt_dir, exist_ok=True)

    from m2se_vtts.data.dataset import M2SEDataset
    dataset = M2SEDataset(test_prefix, shuffle=False)
    n_samples = len(dataset)
    if args.max_samples:
        n_samples = min(n_samples, args.max_samples)
    del dataset

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.num_gpus is not None:
        num_gpus = min(args.num_gpus, num_gpus)

    if num_gpus == 0:
        print('[ERROR] No GPU available')
        sys.exit(1)

    print(f'=== M2SE-VTTS Inference ===')
    print(f'  Checkpoint : {args.load_ckpt}')
    print(f'  Config     : {args.config}')
    print(f'  Test set   : {test_prefix} ({n_samples} samples)')
    print(f'  GPUs       : {num_gpus}')
    print(f'  Batch size : {args.batch_size} per GPU')
    print(f'  Output     : {args.output_dir}')

    all_indices = list(range(n_samples))
    partitions = [[] for _ in range(num_gpus)]
    for i, idx in enumerate(all_indices):
        partitions[i % num_gpus].append(idx)

    for g in range(num_gpus):
        print(f'    GPU {g}: {len(partitions[g])} samples '
              f'({len(partitions[g]) // args.batch_size + 1} batches)')

    mp.set_start_method('spawn', force=True)
    counter = mp.Array('i', [0, 0, 0])
    lock = mp.Lock()

    t0 = time.time()
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=_gpu_worker,
            args=(gpu_id, partitions[gpu_id], args,
                  pred_dir, gt_dir, counter, lock),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=3600)
        if p.is_alive():
            print(f'[WARN] Worker {p.pid} timed out, killing...')
            p.kill()
            p.join(timeout=10)

    elapsed = time.time() - t0
    print(f'\n=== Done ({elapsed:.1f}s) ===')
    print(f'  Success : {counter[0]}')
    print(f'  Skipped : {counter[1]}')
    print(f'  Errors  : {counter[2]}')
    print(f'  Output  : {args.output_dir}')
    print(f'  Speed   : {n_samples / elapsed:.1f} samples/sec')

if __name__ == '__main__':
    main()
