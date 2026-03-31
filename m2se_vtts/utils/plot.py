import matplotlib.pyplot as plt
import numpy as np
import torch

LINE_COLORS = ['w', 'r', 'y', 'cyan', 'm', 'b', 'lime']

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def spec_to_figure(spec, vmin=None, vmax=None):
    spec = _to_numpy(spec)
    fig = plt.figure(figsize=(12, 6))
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    return fig

def spec_f0_to_figure(spec, f0s, figsize=None):
    spec = _to_numpy(spec)
    max_y = spec.shape[1]
    f0s = {k: _to_numpy(f0) / 10 for k, f0 in f0s.items()}
    fig = plt.figure(figsize=(12, 6) if figsize is None else figsize)
    plt.pcolor(spec.T)
    for i, (k, f0) in enumerate(f0s.items()):
        plt.plot(f0.clip(0, max_y), label=k, c=LINE_COLORS[i], linewidth=1, alpha=0.8)
    plt.legend()
    return fig

def dur_to_figure(dur_gt, dur_pred, txt):
    dur_gt = _to_numpy(dur_gt).astype(np.int64)
    dur_pred = _to_numpy(dur_pred).astype(np.int64)
    dur_gt_cum = np.cumsum(dur_gt)
    dur_pred_cum = np.cumsum(dur_pred)
    fig = plt.figure(figsize=(12, 6))
    for i in range(len(dur_gt_cum)):
        shift = (i % 8) + 1
        plt.text(dur_gt_cum[i], shift, txt[i])
        plt.text(dur_pred_cum[i], 10 + shift, txt[i])
        plt.vlines(dur_gt_cum[i], 0, 10, colors='b')
        plt.vlines(dur_pred_cum[i], 10, 20, colors='r')
    return fig

def f0_to_figure(f0_gt, f0_cwt=None, f0_pred=None):
    f0_gt = _to_numpy(f0_gt)
    fig = plt.figure(figsize=(12, 4))
    plt.plot(f0_gt, color='r', label='gt')
    if f0_cwt is not None:
        f0_cwt = _to_numpy(f0_cwt)
        plt.plot(f0_cwt, color='b', label='cwt')
    if f0_pred is not None:
        f0_pred = _to_numpy(f0_pred)
        plt.plot(f0_pred, color='green', label='pred')
    plt.ylabel('F0 (Hz)')
    plt.xlabel('Frame')
    plt.legend()
    plt.tight_layout()
    return fig

def energy_to_figure(energy_gt, energy_pred=None):
    energy_gt = _to_numpy(energy_gt)
    fig = plt.figure(figsize=(12, 3))
    plt.plot(energy_gt, color='r', label='gt', alpha=0.7)
    if energy_pred is not None:
        energy_pred = _to_numpy(energy_pred)
        plt.plot(energy_pred, color='green', label='pred', alpha=0.7)
    plt.ylabel('Energy')
    plt.xlabel('Frame')
    plt.legend()
    plt.tight_layout()
    return fig

def patch_selection_figure(rgb_arr, top_k_indices, total_patches=576,
                           grid_h=24, grid_w=24):
    H, W = rgb_arr.shape[:2]
    k = len(top_k_indices)
    aspect = W / max(H, 1)

    sel = np.zeros(total_patches, dtype=np.float32)
    valid = top_k_indices[(top_k_indices >= 0) & (top_k_indices < total_patches)]
    sel[valid] = 1.0
    sel_grid = sel.reshape(grid_h, grid_w)

    rep_h = int(np.ceil(H / grid_h))
    rep_w = int(np.ceil(W / grid_w))
    mask = np.repeat(np.repeat(sel_grid, rep_h, axis=0), rep_w, axis=1)[:H, :W]

    img = rgb_arr.astype(np.float32) / 255.0
    overlay = img.copy()
    overlay[mask < 0.5] *= 0.25
    sel_px = mask >= 0.5
    overlay[sel_px, 0] = np.clip(overlay[sel_px, 0] * 0.7 + 0.3, 0, 1)

    img_panel_w = max(10, min(22, 5 * aspect))
    grid_panel_w = 3
    fig_h = max(3.5, img_panel_w / aspect)
    fig, axes = plt.subplots(
        1, 2, figsize=(img_panel_w + grid_panel_w, fig_h),
        gridspec_kw={'width_ratios': [img_panel_w, grid_panel_w]})

    axes[0].imshow(np.clip(overlay, 0, 1), aspect='auto')
    for r in range(1, grid_h):
        axes[0].axhline(y=r * H / grid_h, color='w', lw=0.3, alpha=0.35)
    for c in range(1, grid_w):
        axes[0].axvline(x=c * W / grid_w, color='w', lw=0.3, alpha=0.35)
    axes[0].set_title(f'Top-{k} / {total_patches} patches (caption-guided)',
                      fontsize=12, pad=8)
    axes[0].axis('off')

    axes[1].imshow(sel_grid, cmap='OrRd', interpolation='nearest',
                   vmin=0, vmax=1)
    axes[1].set_title(f'{grid_h}\u00d7{grid_w} grid', fontsize=12, pad=8)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    for spine in axes[1].spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')

    plt.tight_layout()
    return fig

def scene_images_figure(rgb_arr, depth_arr, caption='', item_name=''):
    fig, axes = plt.subplots(2, 1, figsize=(14, 5))

    axes[0].imshow(rgb_arr)
    axes[0].set_title('RGB panorama', fontsize=11, loc='left')
    axes[0].axis('off')

    if depth_arr.ndim == 2:
        axes[1].imshow(depth_arr, cmap='plasma')
    else:
        axes[1].imshow(depth_arr)
    axes[1].set_title('Depth panorama', fontsize=11, loc='left')
    axes[1].axis('off')

    title = item_name
    if caption:
        title = f'{item_name}\n{caption}'
    fig.suptitle(title, fontsize=10, y=1.01, va='bottom',
                 fontstyle='italic', color='#444444')
    plt.tight_layout()
    return fig
