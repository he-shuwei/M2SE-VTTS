import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from m2se_vtts.utils.hparams import hparams
from m2se_vtts.utils.indexed_datasets import IndexedDataset
from m2se_vtts.utils.pitch import f0_to_coarse, norm_interp_f0


def _load_spk2id(data_dir):
    path = os.path.join(data_dir, 'spk2id.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def _extract_spk_from_item_name(item_name):
    parts = item_name.rsplit('_', 1)
    if len(parts) == 2:
        return parts[1].split('-')[0]
    return None


class M2SEDataset(Dataset):

    def __init__(self, prefix, shuffle=False, data_dir=None):
        super().__init__()
        self.prefix = prefix
        self.shuffle = shuffle
        self.sort_by_len = hparams.get('sort_by_len', True)
        self.data_dir = data_dir or hparams['binary_data_dir']

        self.indexed_ds = IndexedDataset(os.path.join(self.data_dir, prefix))

        sizes_path = os.path.join(self.data_dir, f'{prefix}.sizes.npy')
        if os.path.exists(sizes_path):
            self.sizes = np.load(sizes_path).tolist()
        else:
            self.sizes = [self.indexed_ds[i]['mel'].shape[0] for i in range(len(self.indexed_ds))]

        self.use_visual = hparams.get('use_visual', True)
        self.spk2id = _load_spk2id(self.data_dir) if hparams.get('use_spk_id', False) else None

        self.use_spec_augment = hparams.get('use_spec_augment', False) and prefix == 'train'
        self.spec_aug_prob = hparams.get('spec_aug_prob', 0.5)

    def __len__(self):
        return len(self.sizes)

    def _get_item(self, index):
        return self.indexed_ds[index]

    def __getitem__(self, index):
        item = self._get_item(index)
        sample = {}

        sample['txt_tokens'] = torch.LongTensor(item['phone'])
        sample['item_name'] = item.get('item_name', str(index))

        mel = torch.FloatTensor(item['mel'])
        T = mel.shape[0]
        max_frames = hparams.get('max_frames', 8000)
        if T > max_frames:
            mel = mel[:max_frames]
            T = max_frames
        sample['mel'] = mel

        mel2ph = torch.LongTensor(item['mel2ph'])[:T]
        sample['mel2ph'] = mel2ph

        f0 = item.get('f0')
        if f0 is not None:
            f0 = f0[:T]
            f0, uv = norm_interp_f0(f0, hparams)
            sample['f0'] = torch.FloatTensor(f0)
            sample['uv'] = torch.FloatTensor(uv)

        energy = item.get('energy')
        if energy is not None:
            sample['energy'] = torch.FloatTensor(energy[:T])
        elif hparams.get('use_energy_embed', False):
            raise ValueError(
                f"Item '{sample['item_name']}' has no 'energy' field in binary data, "
                "but use_energy_embed=True. Re-run binarization with energy extraction."
            )

        if self.use_visual:
            if 'patch_rgb' in item:
                sample['patch_rgb'] = torch.FloatTensor(item['patch_rgb'])
            if 'global_rgb' in item:
                sample['global_rgb'] = torch.FloatTensor(item['global_rgb'])
            if 'patch_depth' in item:
                sample['patch_depth'] = torch.FloatTensor(item['patch_depth'])
            if 'global_depth' in item:
                sample['global_depth'] = torch.FloatTensor(item['global_depth'])
            if 'caption_global' in item:
                sample['caption_global'] = torch.FloatTensor(item['caption_global'])
            if 'caption_local' in item:
                sample['caption_local'] = torch.FloatTensor(item['caption_local'])

        if self.use_spec_augment and random.random() < self.spec_aug_prob:
            sample['mel'] = self._spec_augment(sample['mel'])

        if hparams.get('use_spk_embed', False) and 'spk_embed' in item:
            sample['spk_embed'] = torch.FloatTensor(item['spk_embed'])

        if self.spk2id is not None:
            spk_str = _extract_spk_from_item_name(sample['item_name'])
            if spk_str is not None and spk_str in self.spk2id:
                sample['spk_ids'] = self.spk2id[spk_str]
            else:
                sample['spk_ids'] = 0

        return sample

    def _spec_augment(self, mel, freq_mask_num=2, freq_mask_width=10,
                      time_mask_num=2, time_mask_width=50):
        mel = mel.clone()
        T, M = mel.shape
        for _ in range(freq_mask_num):
            f = random.randint(0, min(freq_mask_width, M - 1))
            f0 = random.randint(0, M - f)
            mel[:, f0:f0+f] = 0
        for _ in range(time_mask_num):
            t = random.randint(0, min(time_mask_width, T - 1))
            t0 = random.randint(0, T - t)
            mel[t0:t0+t, :] = 0
        return mel

    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
            if self.sort_by_len:
                indices = indices[np.argsort(np.array(self.sizes)[indices], kind='mergesort')]
        else:
            indices = np.arange(len(self))
        return indices

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return min(self.sizes[index], hparams.get('max_frames', 8000))


class ConcatM2SEDataset(Dataset):

    def __init__(self, prefixes, shuffle=False, data_dir=None):
        super().__init__()
        self.datasets = [M2SEDataset(p, shuffle=shuffle, data_dir=data_dir)
                         for p in prefixes]
        self.sizes = []
        self._offsets = []
        offset = 0
        for ds in self.datasets:
            self._offsets.append(offset)
            self.sizes.extend(ds.sizes)
            offset += len(ds)
        self.shuffle = shuffle
        self.sort_by_len = hparams.get('sort_by_len', True)

    def __len__(self):
        return len(self.sizes)

    def _locate(self, index):
        for i in range(len(self.datasets) - 1, -1, -1):
            if index >= self._offsets[i]:
                return self.datasets[i], index - self._offsets[i]
        raise IndexError(f'index {index} out of range')

    def __getitem__(self, index):
        ds, local_idx = self._locate(index)
        return ds[local_idx]

    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
            if self.sort_by_len:
                indices = indices[np.argsort(np.array(self.sizes)[indices],
                                             kind='mergesort')]
        else:
            indices = np.arange(len(self))
        return indices

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return min(self.sizes[index], hparams.get('max_frames', 8000))
