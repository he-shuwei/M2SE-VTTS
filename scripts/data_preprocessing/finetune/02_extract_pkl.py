
import os
import json
import pickle
import argparse
import numpy as np
from PIL import Image
import soundfile as sf
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def load_metadata(metadata_dir: str, split: str) -> dict:
    meta_split = 'val' if split == 'val-mini' else split
    meta_path = Path(metadata_dir) / meta_split

    if not meta_path.exists():
        meta_path = Path(metadata_dir) / split
        if not meta_path.exists():
            print(f"[WARN] Metadata dir not found: {meta_path}")
            return {}

    all_meta = {}
    for pkl_file in sorted(meta_path.glob('*.pkl')):
        scene_id = pkl_file.stem
        with open(pkl_file, 'rb') as f:
            meta_list = pickle.load(f)

        for item in meta_list:
            info = item['sound'][0].split(',')
            file_id = info[0]
            duration = float(info[1]) if len(info) > 1 else 0.0
            speaker_id = info[3].split('-')[0] if len(info) > 3 else 'unknown'
            transcript = ' '.join(item['sound'][1:])
            location = (item['location'].tolist()
                        if hasattr(item['location'], 'tolist')
                        else list(item['location']))

            all_meta[f"{scene_id}/{file_id}"] = {
                'transcript': transcript,
                'speaker_id': speaker_id,
                'duration': duration,
                'location': location,
            }

    return all_meta

def process_single_sample(args: tuple) -> tuple:
    pkl_path, output_dir, split, scene_id, speech_id, meta_info = args

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        return None, f"Error loading {pkl_path}: {e}"

    soundspace_id = f"{scene_id}/{speech_id}"

    audio_dir = Path(output_dir) / 'audio' / split / scene_id / speech_id
    audio_dir.mkdir(parents=True, exist_ok=True)

    src_wav_path = audio_dir / 'src_wav.wav'
    recv_wav_path = audio_dir / 'recv_wav.wav'

    sf.write(str(src_wav_path), data['source_audio'], 16000)
    sf.write(str(recv_wav_path), data['receiver_audio'], 16000)

    rir_dir = Path(output_dir) / 'rir' / split / scene_id
    rir_dir.mkdir(parents=True, exist_ok=True)
    rir_path = rir_dir / f'{speech_id}.npy'
    np.save(str(rir_path), np.asarray(data['rir'], dtype=np.float32))

    img_dir = Path(output_dir) / 'images' / split / scene_id / speech_id
    img_dir.mkdir(parents=True, exist_ok=True)

    rgb_views = [np.asarray(v, dtype=np.uint8) for v in data['rgb']]
    rgb_panorama = np.concatenate(rgb_views, axis=1)
    Image.fromarray(rgb_panorama).save(str(img_dir / 'rgb.png'))

    depth_views = [np.asarray(v, dtype=np.float32) for v in data['depth']]
    depth_panorama = np.concatenate(depth_views, axis=1)
    d_min, d_max = depth_panorama.min(), depth_panorama.max()
    depth_norm = ((depth_panorama - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
    if depth_norm.ndim == 3 and depth_norm.shape[2] == 1:
        depth_norm = depth_norm.squeeze(2)
    Image.fromarray(depth_norm).save(str(img_dir / 'depth.png'))

    src_duration = len(data['source_audio']) / 16000.0
    recv_duration = len(data['receiver_audio']) / 16000.0

    result = {
        'soundspace_id': soundspace_id,
        'transcript': meta_info.get('transcript', ''),
        'speaker_id': meta_info.get('speaker_id', 'unknown'),
        'src_wav_path': f'audio/{split}/{scene_id}/{speech_id}/src_wav.wav',
        'recv_wav_path': f'audio/{split}/{scene_id}/{speech_id}/recv_wav.wav',
        'rir_path': f'rir/{split}/{scene_id}/{speech_id}.npy',
        'rgb_path': f'images/{split}/{scene_id}/{speech_id}/rgb.png',
        'depth_path': f'images/{split}/{scene_id}/{speech_id}/depth.png',
        'geodesic_distance': float(data.get('geodesic_distance', 0.0)),
        'duration': meta_info.get('duration', src_duration),
        'src_duration': src_duration,
        'recv_duration': recv_duration,
        'location': meta_info.get('location', [0, 0]),
    }

    return result, None

def process_split(data_dir: str, filter_dir: str, output_dir: str,
                  split: str, num_workers: int = 8) -> list:

    print(f"\n{'=' * 60}")
    print(f"Processing split: {split}")
    print(f"{'=' * 60}")

    filter_path = Path(filter_dir) / f'{split}.json'
    if not filter_path.exists():
        print(f"[WARN] Filter list not found: {filter_path}")
        print(f"       Run 01_filter_data.py first!")
        return []

    with open(filter_path) as f:
        filter_data = json.load(f)

    filtered_files = filter_data['files']
    print(f"Filtered samples: {len(filtered_files)} (from {filter_data.get('total', '?')} total)")

    metadata_dir = Path(data_dir) / 'metadata'
    all_meta = load_metadata(str(metadata_dir), split)
    print(f"Metadata entries: {len(all_meta)}")

    tasks = []
    for item in filtered_files:
        pkl_path = item['pkl_path']
        scene_id = item['scene_id']
        speech_id = item['speech_id']
        sample_key = f"{scene_id}/{speech_id}"

        meta_info = all_meta.get(sample_key, {
            'transcript': '',
            'speaker_id': 'unknown',
            'duration': 0,
            'location': [0, 0],
        })

        tasks.append((pkl_path, output_dir, split, scene_id, speech_id, meta_info))

    print(f"Tasks to process: {len(tasks)}")

    results = []
    errors = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_sample, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"Extracting {split}"):
            result, error = future.result()
            if result:
                results.append(result)
            if error:
                errors.append(error)

    results.sort(key=lambda x: x['soundspace_id'])

    if errors:
        print(f"[WARN] {len(errors)} errors:")
        for e in errors[:5]:
            print(f"  - {e}")

    print(f"[{split}] Successfully processed: {len(results)}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Extract SoundSpaces-Speech PKL data')
    parser.add_argument('--data_dir', type=str,
                        default='data/raw_data/soundspaces_speech',
                        help='Raw data directory')
    parser.add_argument('--filter_dir', type=str,
                        default='data/processed_data/filtered_lists',
                        help='Directory with filtered file lists from 01_filter_data.py')
    parser.add_argument('--output_dir', type=str,
                        default='data/processed_data',
                        help='Output directory for processed data')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['train', 'val-mini', 'test-unseen'],
                        help='Splits to process')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metadata_out_dir = output_dir / 'metadata'
    metadata_out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SoundSpaces-Speech PKL Extraction")
    print("=" * 60)
    print(f"Data dir:    {args.data_dir}")
    print(f"Filter dir:  {args.filter_dir}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Splits:      {args.splits}")
    print(f"Workers:     {args.num_workers}")
    print("=" * 60)

    for split in args.splits:
        results = process_split(
            args.data_dir,
            args.filter_dir,
            args.output_dir,
            split,
            num_workers=args.num_workers,
        )

        if results:
            jsonl_path = metadata_out_dir / f'{split}.jsonl'
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Saved metadata: {jsonl_path} ({len(results)} entries)")

            durations = [r['src_duration'] for r in results]
            print(f"  Audio duration: min={min(durations):.2f}s, "
                  f"max={max(durations):.2f}s, mean={sum(durations)/len(durations):.2f}s")

    print("\nDone!")

if __name__ == '__main__':
    main()
