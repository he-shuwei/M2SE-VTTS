
import argparse
import json
import os
import sys
from pathlib import Path
from multiprocessing import Process, Queue

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

def load_and_resample(wav_path, target_sr=16000):
    wav, sr = torchaudio.load(str(wav_path))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav[0:1]
    return wav

def worker_fn(gpu_id, items, args, progress_queue):
    device = f'cuda:{gpu_id}'
    processed = Path(args.processed_data_dir)

    from speechbrain.inference.speaker import EncoderClassifier
    classifier = EncoderClassifier.from_hparams(
        source=args.model_dir,
        savedir=args.model_dir,
        run_opts={"device": device},
    )

    n_success = 0
    n_skip = 0
    batch_size = args.batch_size

    todo = []
    for item in items:
        soundspace_id = item['soundspace_id']
        scene_id, speech_id = soundspace_id.split('/')
        split_name = item['_split']
        out_path = processed / 'spk_embeddings' / split_name / scene_id / f'{speech_id}.npy'
        if out_path.exists():
            n_success += 1
            progress_queue.put(1)
            continue

        wav_path = processed / item['src_wav_path']
        if not wav_path.exists():
            wav_path = processed / item['recv_wav_path']
        if not wav_path.exists():
            n_skip += 1
            progress_queue.put(1)
            continue

        todo.append((item, wav_path, out_path))

    for batch_start in range(0, len(todo), batch_size):
        batch_items = todo[batch_start:batch_start + batch_size]
        wavs = []
        valid_indices = []
        max_len = 0

        for i, (item, wav_path, out_path) in enumerate(batch_items):
            try:
                wav = load_and_resample(wav_path)
                wavs.append(wav.squeeze(0))
                max_len = max(max_len, wav.shape[-1])
                valid_indices.append(i)
            except Exception as e:
                n_skip += 1
                progress_queue.put(1)

        if not wavs:
            continue

        padded = torch.zeros(len(wavs), max_len)
        wav_lens = torch.zeros(len(wavs))
        for j, w in enumerate(wavs):
            padded[j, :w.shape[0]] = w
            wav_lens[j] = w.shape[0] / max_len

        try:
            with torch.no_grad():
                embeddings = classifier.encode_batch(
                    padded.to(device),
                    wav_lens.to(device),
                )
            embeddings = embeddings.squeeze(1).cpu().numpy().astype(np.float32)

            for j, idx in enumerate(valid_indices):
                item, wav_path, out_path = batch_items[idx]
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(out_path), embeddings[j])
                n_success += 1
                progress_queue.put(1)
        except Exception as e:
            for j, idx in enumerate(valid_indices):
                item, wav_path, out_path = batch_items[idx]
                try:
                    wav = load_and_resample(wav_path).to(device)
                    with torch.no_grad():
                        emb = classifier.encode_batch(wav)
                    emb = emb.squeeze().cpu().numpy().astype(np.float32)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(out_path), emb)
                    n_success += 1
                except Exception as e2:
                    n_skip += 1
                progress_queue.put(1)

    progress_queue.put(('done', gpu_id, n_success, n_skip))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', nargs='+',
                        default=['train', 'val-mini', 'test-seen', 'test-unseen'])
    parser.add_argument('--model_dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'checkpoints/spk_encoder'))
    parser.add_argument('--processed_data_dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'data/processed_data'))
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of GPU workers (0=auto-detect all GPUs)')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count() if args.num_workers <= 0 else args.num_workers
    if num_gpus == 0:
        print("[WARN] No GPUs found, falling back to CPU with 1 worker")
        num_gpus = 1

    processed = Path(args.processed_data_dir)

    all_items = []
    for split in args.splits:
        jsonl_path = processed / 'metadata' / f'{split}.jsonl'
        if not jsonl_path.exists():
            print(f"[WARN] {jsonl_path} not found, skipping")
            continue
        with open(jsonl_path) as f:
            items = [json.loads(l) for l in f if l.strip()]
        for item in items:
            item['_split'] = split
        all_items.extend(items)
        print(f"[{split}] {len(items)} items")

    total = len(all_items)
    print(f"\n[INFO] Total: {total} items, {num_gpus} GPU workers, batch_size={args.batch_size}")

    shards = [[] for _ in range(num_gpus)]
    for i, item in enumerate(all_items):
        shards[i % num_gpus].append(item)

    for i, shard in enumerate(shards):
        print(f"  GPU {i}: {len(shard)} items")

    progress_queue = Queue()
    workers = []
    for gpu_id in range(num_gpus):
        p = Process(target=worker_fn, args=(gpu_id, shards[gpu_id], args, progress_queue))
        p.start()
        workers.append(p)

    pbar = tqdm(total=total, desc="Extracting spk embeddings")
    done_count = 0
    total_success = 0
    total_skip = 0

    while done_count < num_gpus:
        msg = progress_queue.get()
        if isinstance(msg, tuple) and msg[0] == 'done':
            _, gpu_id, n_success, n_skip = msg
            total_success += n_success
            total_skip += n_skip
            done_count += 1
        else:
            pbar.update(1)

    pbar.close()

    for p in workers:
        p.join()

    print(f"\n[DONE] success={total_success}, skip={total_skip}")

if __name__ == '__main__':
    main()
