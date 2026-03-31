
import os
import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

SPLITS = ['train', 'val-mini', 'test-seen', 'test-unseen']

def get_item_id(soundspace_id: str) -> str:
    return soundspace_id.replace('/', '_')

def get_speaker_id(soundspace_id: str) -> str:
    speech_id = soundspace_id.split('/')[-1]
    return speech_id.split('-')[0]

def prepare_mfa_inputs(processed_dir: str, output_dir: str, splits: list):
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0
    missing = 0

    for split in splits:
        jsonl_path = processed_dir / 'metadata' / f'{split}.jsonl'
        if not jsonl_path.exists():
            print(f"[WARN] Not found: {jsonl_path}")
            continue

        with open(jsonl_path) as f:
            items = [json.loads(l) for l in f if l.strip()]

        print(f"\n[{split}] {len(items)} items")

        for item in tqdm(items, desc=split):
            soundspace_id = item['soundspace_id']
            item_id = get_item_id(soundspace_id)
            speaker_id = get_speaker_id(soundspace_id)

            spk_dir = output_dir / speaker_id
            spk_dir.mkdir(exist_ok=True)

            wav_dst = spk_dir / f'{item_id}.wav'
            lab_dst = spk_dir / f'{item_id}.lab'

            if wav_dst.exists() and lab_dst.exists():
                total += 1
                continue

            src_wav = processed_dir / item['src_wav_path']
            if not src_wav.exists():
                print(f"[WARN] wav not found: {src_wav}")
                missing += 1
                continue

            shutil.copy2(str(src_wav), str(wav_dst))

            transcript = item.get('transcript', '').lower().strip()
            if not transcript:
                print(f"[WARN] Empty transcript: {soundspace_id}")
                missing += 1
                if wav_dst.exists():
                    wav_dst.unlink()
                continue

            lab_dst.write_text(transcript, encoding='utf-8')
            total += 1

    print(f"\n{'=' * 60}")
    print(f"MFA Input Preparation Complete")
    print(f"{'=' * 60}")
    print(f"  Prepared:  {total} items")
    print(f"  Missing:   {missing} items")
    print(f"  Output:    {output_dir}")
    if output_dir.exists():
        spk_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        print(f"  Speakers:  {len(spk_dirs)}")

def main():
    parser = argparse.ArgumentParser(description='Prepare MFA input files')
    parser.add_argument('--processed_dir', default='data/processed_data',
                        help='Processed data directory')
    parser.add_argument('--output_dir', default='data/processed_data/mfa/inputs',
                        help='MFA input directory')
    parser.add_argument('--splits', type=str, nargs='+', default=SPLITS,
                        help='Splits to process')
    args = parser.parse_args()

    prepare_mfa_inputs(args.processed_dir, args.output_dir, args.splits)

if __name__ == '__main__':
    main()
