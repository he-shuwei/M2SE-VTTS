
import torch
import librosa
from librosa.filters import mel as librosa_mel_fn

_mel_basis_cache = {}
_hann_window_cache = {}

def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int = None,
    center: bool = False,
) -> torch.Tensor:
    if y.dim() == 1:
        y = y.unsqueeze(0)

    device = y.device
    key = f"{n_fft}_{num_mels}_{sampling_rate}_{hop_size}_{win_size}_{fmin}_{fmax}_{device}"

    if key not in _mel_basis_cache:
        mel_fb = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _mel_basis_cache[key]   = torch.from_numpy(mel_fb).float().to(device)
        _hann_window_cache[key] = torch.hann_window(win_size).to(device)

    mel_basis   = _mel_basis_cache[key]
    hann_window = _hann_window_cache[key]

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (padding, padding), mode='reflect').squeeze(1)

    spec = torch.stft(
        y, n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    mel  = torch.matmul(mel_basis, spec)
    mel  = torch.log(torch.clamp(mel, min=1e-5))
    return mel
