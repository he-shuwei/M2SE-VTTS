
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

if "sinc" in dir(torch):
    sinc = torch.sinc
else:
    def sinc(x: torch.Tensor):
        return torch.where(
            x == 0,
            torch.tensor(1.0, device=x.device, dtype=x.dtype),
            torch.sin(math.pi * x) / math.pi / x,
        )

def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even     = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f   = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time   = (torch.arange(-half_size, half_size) + 0.5) if even \
             else (torch.arange(kernel_size) - half_size)
    if cutoff == 0:
        return torch.zeros_like(time).view(1, 1, kernel_size)
    filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
    filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)

class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff=0.5, half_width=0.6, stride=1,
                 padding=True, padding_mode='replicate', kernel_size=12):
        super().__init__()
        self.kernel_size   = kernel_size
        self.even          = kernel_size % 2 == 0
        self.pad_left      = kernel_size // 2 - int(self.even)
        self.pad_right     = kernel_size // 2
        self.stride        = stride
        self.padding       = padding
        self.padding_mode  = padding_mode
        self.register_buffer('filter', kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x):
        _, C, _ = x.shape
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)

class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride    = ratio
        self.pad       = self.kernel_size // ratio - 1
        self.pad_left  = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        self.register_buffer('filter',
            kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, self.kernel_size))

    def forward(self, x):
        _, C, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode='replicate')
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        return x[..., self.pad_left:-self.pad_right]

class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        ks = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio,
            stride=ratio, kernel_size=ks)

    def forward(self, x):
        return self.lowpass(x)

class Activation1d(nn.Module):
    def __init__(self, activation, up_ratio=2, down_ratio=2,
                 up_kernel_size=12, down_kernel_size=12):
        super().__init__()
        self.act        = activation
        self.upsample   = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x
