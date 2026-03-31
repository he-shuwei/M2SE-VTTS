
import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

from .activations import Snake, SnakeBeta
from .anti_alias  import Activation1d
from .env         import AttrDict

def init_weights(m, mean=0.0, std=0.01):
    if m.__class__.__name__.find('Conv') != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class AMPBlock1(nn.Module):

    def __init__(self, h: AttrDict, channels: int, kernel_size: int = 3,
                 dilation: tuple = (1, 3, 5), activation: str = None):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, stride=1,
                               dilation=d, padding=get_padding(kernel_size, d)))
            for d in dilation
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, stride=1,
                               dilation=1, padding=get_padding(kernel_size, 1)))
            for _ in range(len(dilation))
        ])
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = nn.ModuleList([
            Activation1d(activation=_make_act(activation, channels, h.snake_logscale))
            for _ in range(self.num_layers)
        ])

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x  = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1: remove_weight_norm(l)
        for l in self.convs2: remove_weight_norm(l)

class AMPBlock2(nn.Module):

    def __init__(self, h: AttrDict, channels: int, kernel_size: int = 3,
                 dilation: tuple = (1, 3, 5), activation: str = None):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, stride=1,
                               dilation=d, padding=get_padding(kernel_size, d)))
            for d in dilation
        ])
        self.convs.apply(init_weights)
        self.num_layers = len(self.convs)
        self.activations = nn.ModuleList([
            Activation1d(activation=_make_act(activation, channels, h.snake_logscale))
            for _ in range(self.num_layers)
        ])

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x  = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs: remove_weight_norm(l)

def _make_act(activation: str, channels: int, alpha_logscale: bool):
    if activation == 'snake':
        return Snake(channels, alpha_logscale=alpha_logscale)
    if activation == 'snakebeta':
        return SnakeBeta(channels, alpha_logscale=alpha_logscale)
    raise ValueError(f"Unknown activation: {activation!r}")

class BigVGAN(nn.Module):

    def __init__(self, h: AttrDict):
        super().__init__()
        self.h = h
        self.num_kernels  = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        self.conv_pre = weight_norm(
            Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))

        resblock_cls = AMPBlock1 if h.resblock == '1' else AMPBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(nn.ModuleList([
                weight_norm(ConvTranspose1d(
                    h.upsample_initial_channel // (2 ** i),
                    h.upsample_initial_channel // (2 ** (i + 1)),
                    k, u, padding=(k - u) // 2))
            ]))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(resblock_cls(h, ch, k, d, activation=h.activation))

        ch = h.upsample_initial_channel // (2 ** len(self.ups))
        self.activation_post = Activation1d(activation=_make_act(
            h.activation, ch, h.snake_logscale))

        self.use_bias_at_final  = h.get('use_bias_at_final', True)
        self.use_tanh_at_final  = h.get('use_tanh_at_final', True)
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final))

        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            for up in self.ups[i]:
                x = up(x)
            xs = None
            for j in range(self.num_kernels):
                xs = self.resblocks[i * self.num_kernels + j](x) if xs is None \
                     else xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = self.activation_post(x)
        x = self.conv_post(x)
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            for l_i in l:
                remove_weight_norm(l_i)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
