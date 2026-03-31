
import os
import json
import pickle
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

PERSON_CLASS_ID = 41
MIN_PERSON_PIXELS = 50

def check_sample(pkl_path: str) -> dict:
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        return {'pass': False, 'error': str(e), 'pkl_path': pkl_path}

    import numpy as np

    semantic_views = data.get('semantic', [])
    person_pixels = 0
    for view in semantic_views:
        arr = np.asarray(view)
        person_pixels += int(np.sum(arr == PERSON_CLASS_ID))

    speaker_visible = person_pixels >= MIN_PERSON_PIXELS

    rir = np.asarray(data.get('rir', []))
    if rir.size > 0:
        direct_sound = int(np.argmax(np.abs(rir))) == 0
    else:
        direct_sound = False

    passed = speaker_visible and direct_sound

    p = Path(pkl_path)
    speech_id = p.stem
    scene_id = p.parent.name

    return {
        'pass': passed,
        'pkl_path': pkl_path,
        'scene_id': scene_id,
        'speech_id': speech_id,
        'person_pixels': person_pixels,
        'speaker_visible': speaker_visible,
        'direct_sound': direct_sound,
    }

def collect_pkl_files(data_dir: str, split: str) -> list:
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        print(f"[WARN] Split directory not found: {split_dir}")
        return []

    pkl_files = []
    for scene_dir in sorted(split_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        for pkl_file in sorted(scene_dir.glob('*.pkl')):
            pkl_files.append(str(pkl_file))

    return pkl_files

def load_captions(caption_dir: str, split: str) -> dict:
    caption_path = Path(caption_dir) / f'{split}.jsonl'
    captions = {}
    if not caption_path.exists():
        print(f"[WARN] Caption file not found: {caption_path}")
        return captions
    with open(caption_path) as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                captions[item['soundspace_id']] = item.get('caption', '')
    return captions

def load_transcripts(data_dir: str, split: str) -> dict:
    meta_split = 'val' if split == 'val-mini' else split
    meta_path = Path(data_dir) / 'metadata' / meta_split
    if not meta_path.exists():
        meta_path = Path(data_dir) / 'metadata' / split
        if not meta_path.exists():
            print(f"[WARN] Metadata dir not found: {meta_path}")
            return {}

    transcripts = {}
    for pkl_file in sorted(meta_path.glob('*.pkl')):
        scene_id = pkl_file.stem
        with open(pkl_file, 'rb') as f:
            meta_list = pickle.load(f)
        for item in meta_list:
            info = item['sound'][0].split(',')
            file_id = info[0]
            transcript = ' '.join(item['sound'][1:]).strip()
            transcripts[f"{scene_id}/{file_id}"] = transcript
    return transcripts

def filter_split(data_dir: str, caption_dir: str, split: str,
                 num_workers: int = 8) -> dict:
    pkl_files = collect_pkl_files(data_dir, split)
    if not pkl_files:
        return {'split': split, 'total': 0, 'passed': 0, 'files': []}

    captions = load_captions(caption_dir, split)
    transcripts = load_transcripts(data_dir, split)
    print(f"\n[{split}] Processing {len(pkl_files)} PKL files with {num_workers} workers...")
    print(f"  Captions loaded: {len(captions)} (non-empty: {sum(1 for v in captions.values() if v.strip())})")
    print(f"  Transcripts loaded: {len(transcripts)} (non-empty: {sum(1 for v in transcripts.values() if v.strip())})")

    results = []
    passed_files = []
    n_speaker_fail = 0
    n_direct_fail = 0
    n_both_fail = 0
    n_no_caption = 0
    n_no_transcript = 0
    n_error = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(check_sample, p): p for p in pkl_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Filtering {split}"):
            result = future.result()
            if result is None or 'error' in result:
                n_error += 1
                continue

            if result['pass']:
                sid = f"{result['scene_id']}/{result['speech_id']}"

                caption = captions.get(sid, '').strip()
                if not caption:
                    n_no_caption += 1
                    continue

                transcript = transcripts.get(sid, '').strip()
                if not transcript:
                    n_no_transcript += 1
                    continue

                passed_files.append({
                    'pkl_path': result['pkl_path'],
                    'scene_id': result['scene_id'],
                    'speech_id': result['speech_id'],
                })
            else:
                if not result['speaker_visible'] and not result['direct_sound']:
                    n_both_fail += 1
                elif not result['speaker_visible']:
                    n_speaker_fail += 1
                else:
                    n_direct_fail += 1

    passed_files.sort(key=lambda x: (x['scene_id'], x['speech_id']))

    total = len(pkl_files)
    passed = len(passed_files)
    filtered = total - passed - n_error

    print(f"[{split}] Results:")
    print(f"  Total:            {total}")
    print(f"  Passed:           {passed} ({100*passed/max(total,1):.1f}%)")
    print(f"  Filtered out:     {filtered}")
    print(f"    - Speaker not visible: {n_speaker_fail}")
    print(f"    - No direct sound:     {n_direct_fail}")
    print(f"    - Both failed:         {n_both_fail}")
    print(f"    - No caption:          {n_no_caption}")
    print(f"    - No transcript:       {n_no_transcript}")
    print(f"    - Errors:              {n_error}")

    return {
        'split': split,
        'total': total,
        'passed': passed,
        'filtered': filtered,
        'stats': {
            'speaker_not_visible': n_speaker_fail,
            'no_direct_sound': n_direct_fail,
            'both_failed': n_both_fail,
            'no_caption': n_no_caption,
            'no_transcript': n_no_transcript,
            'errors': n_error,
        },
        'files': passed_files,
    }

def main():
    parser = argparse.ArgumentParser(description='Filter SoundSpaces-Speech PKL data')
    parser.add_argument('--data_dir', type=str,
                        default='data/raw_data/soundspaces_speech',
                        help='Raw data directory containing split folders')
    parser.add_argument('--caption_dir', type=str,
                        default='data/raw_data/captions',
                        help='Caption JSONL directory')
    parser.add_argument('--output_dir', type=str,
                        default='data/processed_data/filtered_lists',
                        help='Output directory for filtered file lists')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['train', 'val-mini', 'test-unseen'],
                        help='Splits to filter')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of parallel workers')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SoundSpaces-Speech Data Filtering")
    print("=" * 60)
    print(f"Data dir:    {args.data_dir}")
    print(f"Caption dir: {args.caption_dir}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Splits:      {args.splits}")
    print(f"Workers:     {args.num_workers}")
    print(f"Filters:")
    print(f"  - Speaker visible (semantic class {PERSON_CLASS_ID} >= {MIN_PERSON_PIXELS} px)")
    print(f"  - Direct sound (RIR argmax == 0)")
    print(f"  - Caption non-empty")
    print(f"  - Transcript non-empty")
    print("=" * 60)

    for split in args.splits:
        result = filter_split(args.data_dir, args.caption_dir, split,
                              num_workers=args.num_workers)

        output_path = output_dir / f'{split}.json'
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved: {output_path}")

    print("\nDone!")

if __name__ == '__main__':
    main()
