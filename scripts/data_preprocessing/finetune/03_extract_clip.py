
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor

class CLIPFeatureExtractor:

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14-336",
                 device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype

        print(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = self.model.to(device=device, dtype=dtype)
        self.model.eval()

        self.vision_dim = self.model.config.vision_config.hidden_size
        self.text_dim = self.model.config.text_config.hidden_size
        print(f"Vision dim: {self.vision_dim}, Text dim: {self.text_dim}")

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> tuple:
        images = images.to(device=self.device, dtype=self.dtype)
        vision_outputs = self.model.vision_model(
            pixel_values=images,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_state = vision_outputs.last_hidden_state
        global_features = last_hidden_state[:, :1, :]
        patch_features = last_hidden_state[:, 1:, :]
        return patch_features, global_features

    @torch.no_grad()
    def encode_text(self, texts: list) -> tuple:
        inputs = self.processor(
            text=texts, return_tensors="pt",
            padding=True, truncation=True, max_length=77,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        text_outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_state = text_outputs.last_hidden_state

        batch_size = input_ids.shape[0]
        global_features_list = []
        local_features_list = []

        for i in range(batch_size):
            seq_len = attention_mask[i].sum().item()
            eos_pos = seq_len - 1
            global_feat = last_hidden_state[i, eos_pos:eos_pos + 1, :]
            global_features_list.append(global_feat)
            local_feat = last_hidden_state[i, 1:eos_pos, :]
            local_features_list.append(local_feat)

        global_features = torch.stack(global_features_list, dim=0)
        return local_features_list, global_features

class ProcessedDataset(Dataset):

    def __init__(self, processed_dir: str, caption_dir: str, split: str,
                 processor: CLIPProcessor, skip_existing: bool = True):
        self.processed_dir = Path(processed_dir)
        self.processor = processor

        metadata_path = self.processed_dir / 'metadata' / f'{split}.jsonl'
        with open(metadata_path) as f:
            all_samples = [json.loads(line.strip()) for line in f if line.strip()]

        caption_path = Path(caption_dir) / f'{split}.jsonl'
        self.captions = {}
        if caption_path.exists():
            with open(caption_path) as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        sid = item.get('soundspace_id', '')
                        self.captions[sid] = item.get('caption', '')
            print(f"[{split}] Loaded {len(self.captions)} captions")
        else:
            print(f"[{split}] Caption file not found: {caption_path}")

        self.samples = []
        skipped = 0
        for sample in all_samples:
            sid = sample['soundspace_id']
            scene_id, speech_id = sid.split('/')
            feature_path = (self.processed_dir / 'clip_features' /
                            split / scene_id / speech_id / 'features.npz')
            if skip_existing and feature_path.exists():
                skipped += 1
            else:
                sample['_caption'] = self.captions.get(sid, '')
                sample['_split'] = split
                self.samples.append(sample)

        print(f"[{split}] Total: {len(all_samples)}, Skipped: {skipped}, "
              f"To process: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        rgb_path = self.processed_dir / sample['rgb_path']
        rgb_image = Image.open(rgb_path).convert("RGB")

        depth_path = self.processed_dir / sample['depth_path']
        depth_image = Image.open(depth_path).convert("RGB")

        rgb_tensor = self.processor(images=rgb_image, return_tensors="pt")["pixel_values"].squeeze(0)
        depth_tensor = self.processor(images=depth_image, return_tensors="pt")["pixel_values"].squeeze(0)

        return {
            "soundspace_id": sample['soundspace_id'],
            "split": sample['_split'],
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "caption": sample['_caption'],
        }

def collate_fn(batch):
    return {
        "soundspace_id": [item["soundspace_id"] for item in batch],
        "split": [item["split"] for item in batch],
        "rgb": torch.stack([item["rgb"] for item in batch]),
        "depth": torch.stack([item["depth"] for item in batch]),
        "caption": [item["caption"] for item in batch],
    }

def process_split(extractor: CLIPFeatureExtractor, processed_dir: str,
                  caption_dir: str, split: str, batch_size: int = 32,
                  num_workers: int = 4, skip_existing: bool = True):
    print(f"\n{'=' * 60}")
    print(f"Processing split: {split}")
    print(f"{'=' * 60}")

    processed_path = Path(processed_dir)

    dataset = ProcessedDataset(
        processed_dir, caption_dir, split,
        extractor.processor, skip_existing=skip_existing,
    )

    if len(dataset) == 0:
        print(f"[{split}] All samples already extracted, skipping...")
        return

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    for batch in tqdm(dataloader, desc=f"Extracting {split}"):
        soundspace_ids = batch["soundspace_id"]
        splits = batch["split"]
        rgb_images = batch["rgb"]
        depth_images = batch["depth"]
        captions = batch["caption"]

        patch_rgb, global_rgb = extractor.encode_image(rgb_images)
        patch_depth, global_depth = extractor.encode_image(depth_images)
        caption_local_list, caption_global = extractor.encode_text(captions)

        patch_rgb_np = patch_rgb.cpu().float().numpy()
        global_rgb_np = global_rgb.cpu().float().numpy()
        patch_depth_np = patch_depth.cpu().float().numpy()
        global_depth_np = global_depth.cpu().float().numpy()
        caption_global_np = caption_global.cpu().float().numpy()

        for i, sid in enumerate(soundspace_ids):
            scene_id, speech_id = sid.split('/')
            feature_dir = (processed_path / 'clip_features' /
                           splits[i] / scene_id / speech_id)
            feature_dir.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                str(feature_dir / 'features.npz'),
                patch_rgb=patch_rgb_np[i].astype(np.float16),
                global_rgb=global_rgb_np[i].astype(np.float16),
                patch_depth=patch_depth_np[i].astype(np.float16),
                global_depth=global_depth_np[i].astype(np.float16),
                caption_local=caption_local_list[i].cpu().float().numpy().astype(np.float16),
                caption_global=caption_global_np[i].astype(np.float16),
            )

    print(f"[{split}] Processed {len(dataset)} samples")

def main():
    parser = argparse.ArgumentParser(description="Extract CLIP features")
    parser.add_argument("--processed_dir", type=str, default="data/processed_data",
                        help="Processed data directory")
    parser.add_argument("--caption_dir", type=str, default="data/raw_data/captions",
                        help="Caption JSONL directory")
    parser.add_argument("--splits", type=str, nargs="+",
                        default=["train", "val-mini", "test-unseen"],
                        help="Splits to process")
    parser.add_argument("--model_name", type=str,
                        default="checkpoints/clip-vit-large-patch14-336",
                        help="CLIP model name or local path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("CLIP Feature Extraction (Restructured Pipeline)")
    print("=" * 60)
    print(f"Processed dir: {args.processed_dir}")
    print(f"Caption dir:   {args.caption_dir}")
    print(f"Splits:        {args.splits}")
    print(f"Model:         {args.model_name}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Device:        {args.device}")
    print("=" * 60)

    extractor = CLIPFeatureExtractor(
        model_name=args.model_name,
        device=args.device,
        dtype=torch.float16,
    )

    for split in args.splits:
        process_split(
            extractor, args.processed_dir, args.caption_dir, split,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )

    print("\n" + "=" * 60)
    print("Feature extraction completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
