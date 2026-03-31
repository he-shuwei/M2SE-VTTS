
import os
import sys
import json
import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import librosa
import torch
from tqdm import tqdm
from textgrid import TextGrid
from multiprocessing import Process, Queue

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
sys.path.insert(0, _PROJECT_ROOT)

from m2se_vtts.utils.hparams import hparams, set_hparams
from m2se_vtts.utils.indexed_datasets import IndexedDatasetBuilder, IndexedDataset
from m2se_vtts.utils.text import TokenTextEncoder
from m2se_vtts.vocoders.bigvgan import mel_spectrogram as _bigvgan_mel_spectrogram

SPLITS: Dict[str, str] = {
    'train':        'train',
    'valid':        'val-mini',
    'test_seen':    'test-seen',
    'test_unseen':  'test-unseen',
}

def get_item_id(soundspace_id: str) -> str:
    return soundspace_id.replace('/', '_')

def get_speaker_id(soundspace_id: str) -> str:
    return soundspace_id.split('/')[-1].split('-')[0]

def load_wav(wav_path: str, sr: int = 16000) -> np.ndarray:
    wav, _ = librosa.load(wav_path, sr=sr)
    wav = librosa.util.normalize(wav) * 0.95
    return wav.astype(np.float32)

def extract_mel(wav: np.ndarray, hp: dict) -> np.ndarray:
    y = torch.from_numpy(wav).unsqueeze(0)
    mel = _bigvgan_mel_spectrogram(
        y,
        n_fft=hp['fft_size'],
        num_mels=hp['audio_num_mel_bins'],
        sampling_rate=hp['audio_sample_rate'],
        hop_size=hp['hop_size'],
        win_size=hp['win_size'],
        fmin=hp.get('fmin', 0),
        fmax=hp.get('fmax', None),
        center=False,
    )
    return mel.squeeze(0).T.numpy().astype(np.float32)

def extract_f0(wav: np.ndarray, mel_len: int, rmvpe_model, hp: dict) -> np.ndarray:
    sr = hp['audio_sample_rate']
    hop_size = hp['hop_size']

    f0 = rmvpe_model.infer_from_audio(wav, thred=0.03)

    rmvpe_hop = 160
    if sr != 16000:
        rmvpe_hop = int(rmvpe_hop * sr / 16000)

    if hop_size != rmvpe_hop:
        import torch.nn.functional as F_nn
        f0_t = torch.from_numpy(f0).float().unsqueeze(0).unsqueeze(0)
        f0_t = F_nn.interpolate(f0_t, size=mel_len, mode='linear', align_corners=True)
        f0 = f0_t.squeeze().numpy()
    else:
        if len(f0) < mel_len:
            f0 = np.pad(f0, (0, mel_len - len(f0)))
        else:
            f0 = f0[:mel_len]

    return f0.astype(np.float32)

def extract_energy(mel: np.ndarray) -> np.ndarray:
    return np.exp(mel).sum(axis=-1).astype(np.float32)

def parse_textgrid(tg_path: str) -> List[Tuple[str, float, float]]:
    tg = TextGrid.fromFile(tg_path)

    phone_tier = None
    for tier in tg.tiers:
        if 'phone' in tier.name.lower():
            phone_tier = tier
            break
    if phone_tier is None:
        phone_tier = tg.tiers[1] if len(tg.tiers) > 1 else tg.tiers[0]

    phones = []
    for interval in phone_tier:
        label = interval.mark.strip()
        if not label or label == 'sp':
            continue
        phones.append((label, float(interval.minTime), float(interval.maxTime)))
    return phones

def get_mel2ph(phones: List[Tuple[str, float, float]], mel_len: int, hp: dict) -> np.ndarray:
    hop_size = hp['hop_size']
    sr = hp['audio_sample_rate']
    sec_per_frame = hop_size / sr

    mel2ph = np.zeros(mel_len, dtype=np.int64)
    for ph_idx, (phone, start, end) in enumerate(phones):
        start_frame = int(start / sec_per_frame + 0.5)
        end_frame = int(end / sec_per_frame + 0.5)
        start_frame = max(0, min(start_frame, mel_len))
        end_frame = max(0, min(end_frame, mel_len))
        mel2ph[start_frame:end_frame] = ph_idx + 1

    return mel2ph

def process_item(item: dict, split_name: str, phone_encoder: TokenTextEncoder,
                 processed_data_dir: Path, mfa_output_dir: Path,
                 rmvpe_model, hp: dict) -> dict:
    soundspace_id = item['soundspace_id']
    item_id = get_item_id(soundspace_id)
    speaker_id = get_speaker_id(soundspace_id)

    recv_wav_path = processed_data_dir / item['recv_wav_path']
    if not recv_wav_path.exists():
        raise FileNotFoundError(f"recv_wav not found: {recv_wav_path}")

    tg_path = mfa_output_dir / speaker_id / f'{item_id}.TextGrid'
    if not tg_path.exists():
        raise FileNotFoundError(f"TextGrid not found: {tg_path}")

    sr = hp['audio_sample_rate']
    recv_wav = load_wav(str(recv_wav_path), sr=sr)
    mel = extract_mel(recv_wav, hp)
    mel_len = mel.shape[0]

    phones_with_time = parse_textgrid(str(tg_path))
    if not phones_with_time:
        raise ValueError("Empty phone sequence from TextGrid")

    phone_strs = [p[0] for p in phones_with_time]
    phone_ids = phone_encoder.encode(' '.join(phone_strs))
    mel2ph = get_mel2ph(phones_with_time, mel_len, hp)

    f0 = extract_f0(recv_wav, mel_len, rmvpe_model, hp)
    energy = extract_energy(mel)

    scene_id, speech_id = soundspace_id.split('/')
    clip_path = (processed_data_dir / 'clip_features' /
                 split_name / scene_id / speech_id / 'features.npz')
    clip_data = np.load(str(clip_path))

    result = {
        'item_name': item_id,
        'soundspace_id': soundspace_id,
        'txt': item.get('transcript', ''),
        'phone': np.array(phone_ids, dtype=np.int64),
        'mel': mel,
        'mel2ph': mel2ph,
        'f0': f0,
        'energy': energy,
        'patch_rgb': clip_data['patch_rgb'].astype(np.float32),
        'global_rgb': clip_data['global_rgb'].astype(np.float32),
        'patch_depth': clip_data['patch_depth'].astype(np.float32),
        'global_depth': clip_data['global_depth'].astype(np.float32),
        'caption_global': clip_data['caption_global'].astype(np.float32),
        'caption_local': clip_data['caption_local'].astype(np.float32),
        'caption': item.get('caption', ''),
        'geodesic_distance': float(item.get('geodesic_distance', 0.0)),
    }

    spk_embed_path = (processed_data_dir / 'spk_embeddings' /
                      split_name / scene_id / f'{speech_id}.npy')
    if spk_embed_path.exists():
        result['spk_embed'] = np.load(str(spk_embed_path)).astype(np.float32)

    return result

def _worker_fn(gpu_id, shard_items, split_name, phone_list,
               shard_prefix, processed_data_dir, mfa_output_dir,
               hp_snapshot, progress_queue):
    torch.cuda.set_device(gpu_id)

    phone_encoder = TokenTextEncoder(None, vocab_list=phone_list, replace_oov='UNK')

    from m2se_vtts.vocoders.rmvpe import RMVPE
    ckpt = hp_snapshot.get('rmvpe_ckpt', 'checkpoints/RMVPE/rmvpe.pt')
    rmvpe = RMVPE(ckpt, is_half=False, device=f'cuda:{gpu_id}')

    processed = Path(processed_data_dir)
    mfa_dir = Path(mfa_output_dir)

    builder = IndexedDatasetBuilder(shard_prefix)
    sizes = []
    f0_list = []
    n_ok = 0
    n_skip_tg = 0
    n_skip_other = 0

    for item in shard_items:
        try:
            binary_item = process_item(
                item, split_name, phone_encoder,
                processed, mfa_dir, rmvpe, hp_snapshot,
            )
            builder.add_item(binary_item)
            sizes.append(binary_item['mel'].shape[0])
            if binary_item['f0'] is not None:
                f0_list.append(binary_item['f0'])
            n_ok += 1
        except FileNotFoundError as e:
            if 'TextGrid' in str(e):
                n_skip_tg += 1
            else:
                n_skip_other += 1
        except Exception:
            n_skip_other += 1
        progress_queue.put(1)

    builder.finalize()

    np.save(f'{shard_prefix}.sizes.npy', np.array(sizes, dtype=np.int32))
    if f0_list:
        np.save(f'{shard_prefix}.f0s.npy', np.concatenate(f0_list))

    progress_queue.put(('done', gpu_id, n_ok, n_skip_tg, n_skip_other))

def merge_shards(shard_prefixes: List[str], output_prefix: str):
    merged_offsets = [0]
    all_sizes = []

    with open(f'{output_prefix}.data', 'wb') as fout:
        for sp in shard_prefixes:
            idx_data = np.load(f'{sp}.idx', allow_pickle=True).item()
            offsets = idx_data['offsets']

            with open(f'{sp}.data', 'rb') as fin:
                data = fin.read()
            fout.write(data)

            base = merged_offsets[-1]
            for off in offsets[1:]:
                merged_offsets.append(base + off)

            sizes_path = f'{sp}.sizes.npy'
            if os.path.exists(sizes_path):
                all_sizes.extend(np.load(sizes_path).tolist())

    np.save(open(f'{output_prefix}.idx', 'wb'), {'offsets': merged_offsets})
    np.save(f'{output_prefix}.sizes.npy', np.array(all_sizes, dtype=np.int32))

    return all_sizes

class M2SEBinarizer:

    def __init__(self, num_workers=0):
        self.processed_data_dir = Path(hparams.get('processed_data_dir', 'data/processed_data'))
        self.mfa_output_dir = Path(hparams.get('mfa_output_dir', 'data/processed_data/mfa/outputs'))
        self.binary_data_dir = Path(hparams['binary_data_dir'])
        self.binary_data_dir.mkdir(parents=True, exist_ok=True)

        self.num_workers = num_workers if num_workers > 0 else torch.cuda.device_count()
        if self.num_workers == 0:
            self.num_workers = 1

        print(f"[INFO] processed_data_dir: {self.processed_data_dir}")
        print(f"[INFO] mfa_output_dir:     {self.mfa_output_dir}")
        print(f"[INFO] binary_data_dir:    {self.binary_data_dir}")
        print(f"[INFO] num_workers:        {self.num_workers} GPUs")
        print(f"[INFO] Mel params: sr={hparams['audio_sample_rate']}, "
              f"n_fft={hparams['fft_size']}, hop={hparams['hop_size']}, "
              f"win={hparams['win_size']}, n_mels={hparams['audio_num_mel_bins']}, "
              f"fmin={hparams.get('fmin', 0)}, fmax={hparams.get('fmax', 'None')}")

    def binarize(self):
        phone_encoder, phone_list = self._build_phone_encoder()

        hp_snapshot = dict(hparams)

        all_f0s = []
        for prefix, split_name in SPLITS.items():
            jsonl_path = self.processed_data_dir / 'metadata' / f'{split_name}.jsonl'
            if not jsonl_path.exists():
                print(f"[WARN] Not found: {jsonl_path}, skipping {prefix}")
                continue

            with open(jsonl_path) as f:
                items = [json.loads(l) for l in f if l.strip()]
            print(f"\n[{prefix}] {len(items)} items from {split_name}")

            f0s = self._binarize_split_parallel(
                prefix, items, split_name, phone_list, hp_snapshot,
            )
            all_f0s.extend(f0s)

        if all_f0s:
            f0_arr = np.concatenate(all_f0s)
            voiced = f0_arr[f0_arr > 0]
            mean, std = float(np.mean(voiced)), float(np.std(voiced))
            np.save(self.binary_data_dir / 'f0_mean_std.npy', np.array([mean, std]))
            print(f"\n[INFO] Global f0: mean={mean:.2f} Hz, std={std:.2f} Hz")

        phone_set_out = self.binary_data_dir / 'phone_set.json'
        with open(phone_set_out, 'w') as f:
            json.dump(phone_list, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved phone_set.json ({len(phone_list)} phones) -> {phone_set_out}")

    def _build_phone_encoder(self) -> Tuple['TokenTextEncoder', List[str]]:
        candidates = [
            hparams.get('phone_set_path', ''),
        ]
        phone_list = None
        for p in candidates:
            if p and os.path.exists(p):
                with open(p) as f:
                    phone_list = json.load(f)
                print(f"[INFO] Loaded phone_set from {p}: {len(phone_list)} phones")
                break

        if phone_list is None:
            phone_list = [
                'sil', 'spn',
                'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2',
                'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
                'B', 'CH', 'D', 'DH',
                'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2',
                'F', 'G', 'HH',
                'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
                'JH', 'K', 'L', 'M', 'N', 'NG',
                'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2',
                'P', 'R', 'S', 'SH', 'T', 'TH',
                'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2',
                'V', 'W', 'Y', 'Z', 'ZH',
            ]
            print(f"[INFO] Using default MFA ARPAbet phone set: {len(phone_list)} phones")

        encoder = TokenTextEncoder(None, vocab_list=phone_list, replace_oov='UNK')
        return encoder, phone_list

    def _binarize_split_parallel(self, prefix: str, items: list, split_name: str,
                                  phone_list: list, hp_snapshot: dict) -> List[np.ndarray]:
        N = self.num_workers
        total = len(items)

        if total < 100 or N <= 1:
            return self._binarize_split_single(prefix, items, split_name, phone_list, hp_snapshot)

        shard_dir = self.binary_data_dir / f'_shards_{prefix}'
        shard_dir.mkdir(parents=True, exist_ok=True)

        shards = [[] for _ in range(N)]
        for i, item in enumerate(items):
            shards[i % N].append(item)

        shard_prefixes = []
        progress_queue = Queue()
        workers = []

        for gpu_id in range(N):
            sp = str(shard_dir / f'shard_{gpu_id}')
            shard_prefixes.append(sp)
            p = Process(
                target=_worker_fn,
                args=(gpu_id, shards[gpu_id], split_name, phone_list,
                      sp, str(self.processed_data_dir), str(self.mfa_output_dir),
                      hp_snapshot, progress_queue),
            )
            p.start()
            workers.append(p)
            print(f"  GPU {gpu_id}: {len(shards[gpu_id])} items")

        pbar = tqdm(total=total, desc=f'Binarizing {prefix}')
        done_count = 0
        total_ok = total_skip_tg = total_skip_other = 0

        while done_count < N:
            msg = progress_queue.get()
            if isinstance(msg, tuple) and msg[0] == 'done':
                _, gpu_id, n_ok, n_tg, n_other = msg
                total_ok += n_ok
                total_skip_tg += n_tg
                total_skip_other += n_other
                done_count += 1
            else:
                pbar.update(1)
        pbar.close()

        for p in workers:
            p.join()

        print(f"[{prefix}] Merging {N} shards ...")
        output_prefix = str(self.binary_data_dir / prefix)
        merge_shards(shard_prefixes, output_prefix)

        f0s = []
        for sp in shard_prefixes:
            f0_path = f'{sp}.f0s.npy'
            if os.path.exists(f0_path):
                f0s.append(np.load(f0_path))

        shutil.rmtree(shard_dir)

        print(f"[{prefix}] success={total_ok}, "
              f"skip_no_tg={total_skip_tg}, skip_other={total_skip_other}")
        return f0s

    def _binarize_split_single(self, prefix: str, items: list, split_name: str,
                                phone_list: list, hp_snapshot: dict) -> List[np.ndarray]:
        phone_encoder = TokenTextEncoder(None, vocab_list=phone_list, replace_oov='UNK')

        from m2se_vtts.vocoders.rmvpe import RMVPE
        ckpt = hp_snapshot.get('rmvpe_ckpt', 'checkpoints/RMVPE/rmvpe.pt')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rmvpe = RMVPE(ckpt, is_half=False, device=device)

        builder = IndexedDatasetBuilder(str(self.binary_data_dir / prefix))
        lengths = []
        f0s = []
        n_success = n_skip_tg = n_skip_other = 0

        for item in tqdm(items, desc=f'Binarizing {prefix}'):
            try:
                binary_item = process_item(
                    item, split_name, phone_encoder,
                    self.processed_data_dir, self.mfa_output_dir,
                    rmvpe, hp_snapshot,
                )
            except FileNotFoundError as e:
                if 'TextGrid' in str(e):
                    n_skip_tg += 1
                else:
                    print(f"\n[WARN] Skip {item['soundspace_id']}: {e}")
                    n_skip_other += 1
                continue
            except Exception as e:
                print(f"\n[WARN] Skip {item['soundspace_id']}: {e}")
                traceback.print_exc()
                n_skip_other += 1
                continue

            builder.add_item(binary_item)
            lengths.append(binary_item['mel'].shape[0])
            if binary_item['f0'] is not None:
                f0s.append(binary_item['f0'])
            n_success += 1

        builder.finalize()
        np.save(str(self.binary_data_dir / f'{prefix}.sizes.npy'),
                np.array(lengths, dtype=np.int32))

        print(f"[{prefix}] success={n_success}, "
              f"skip_no_tg={n_skip_tg}, skip_other={n_skip_other}")
        return f0s

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of GPU workers (0=auto-detect)')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--hparams', type=str, default='')
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--infer', action='store_true')
    args, _ = parser.parse_known_args()

    set_hparams(config=args.config)
    binarizer = M2SEBinarizer(num_workers=args.num_workers)
    binarizer.binarize()
