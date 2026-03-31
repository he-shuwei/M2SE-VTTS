import torch
from m2se_vtts.utils import collate_1d, collate_2d


def _all_have(samples, key):
    return all(key in s and s[key] is not None for s in samples)


class M2SECollator:

    def __call__(self, samples):
        if len(samples) == 0:
            return {}

        batch = {
            'nsamples': len(samples),
        }

        batch['txt_tokens'] = collate_1d(
            [s['txt_tokens'] for s in samples], pad_idx=0
        )

        batch['mels'] = collate_2d(
            [s['mel'] for s in samples], pad_idx=0
        )

        batch['mel2ph'] = collate_1d(
            [s['mel2ph'] for s in samples], pad_idx=0
        )

        if _all_have(samples, 'f0'):
            batch['f0'] = collate_1d(
                [s['f0'] for s in samples], pad_idx=0
            )

        if _all_have(samples, 'uv'):
            batch['uv'] = collate_1d(
                [s['uv'] for s in samples], pad_idx=0
            )

        if _all_have(samples, 'energy'):
            batch['energy'] = collate_1d(
                [s['energy'] for s in samples], pad_idx=0
            )

        if _all_have(samples, 'visual_feat'):
            visual_feats = [s['visual_feat'] for s in samples]
            if visual_feats[0].dim() == 1:
                batch['visual_feat'] = torch.stack(visual_feats, dim=0)
            else:
                batch['visual_feat'] = collate_2d(visual_feats, pad_idx=0)

        if _all_have(samples, 'patch_rgb'):
            batch['patch_rgb'] = collate_2d(
                [s['patch_rgb'] for s in samples], pad_idx=0
            )

        if _all_have(samples, 'global_rgb'):
            batch['global_rgb'] = torch.stack(
                [s['global_rgb'] for s in samples], dim=0
            )

        if _all_have(samples, 'patch_depth'):
            batch['patch_depth'] = collate_2d(
                [s['patch_depth'] for s in samples], pad_idx=0
            )

        if _all_have(samples, 'global_depth'):
            batch['global_depth'] = torch.stack(
                [s['global_depth'] for s in samples], dim=0
            )

        if _all_have(samples, 'caption_features'):
            batch['caption_features'] = torch.stack(
                [s['caption_features'] for s in samples], dim=0
            )

        if _all_have(samples, 'caption_global'):
            batch['caption_global'] = torch.stack(
                [s['caption_global'] for s in samples], dim=0
            )

        if _all_have(samples, 'caption_local'):
            caption_locals = [s['caption_local'] for s in samples]
            if caption_locals[0].dim() == 1:
                caption_locals = [c.unsqueeze(0) for c in caption_locals]
            batch['caption_local'] = collate_2d(caption_locals, pad_idx=0)

        if _all_have(samples, 'spk_embed'):
            batch['spk_embed'] = torch.stack(
                [s['spk_embed'] for s in samples], dim=0
            )
        if _all_have(samples, 'spk_ids'):
            batch['spk_ids'] = torch.LongTensor(
                [s['spk_ids'] for s in samples]
            )

        return batch
