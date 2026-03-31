
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as ckpt_fn

class ZeroLinear(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ControlledDiT(nn.Module):

    def __init__(self, pretrained_dit: nn.Module):
        super().__init__()
        dim = pretrained_dit.dim
        self.checkpoint_activations = getattr(pretrained_dit, 'checkpoint_activations', False)

        self.locked = pretrained_dit
        for param in self.locked.parameters():
            param.requires_grad = False

        self.trainable_time_embed = deepcopy(pretrained_dit.time_embed)
        self.trainable_input_embed = deepcopy(pretrained_dit.input_embed)
        self.trainable_blocks = deepcopy(pretrained_dit.transformer_blocks)

        for param in self.trainable_time_embed.parameters():
            param.requires_grad = True
        for param in self.trainable_input_embed.parameters():
            param.requires_grad = True
        for param in self.trainable_blocks.parameters():
            param.requires_grad = True

        depth = len(pretrained_dit.transformer_blocks)
        self.zero_convs = nn.ModuleList([ZeroLinear(dim) for _ in range(depth)])
        self.input_zero_conv = ZeroLinear(dim)
        self.output_t_zero_conv = ZeroLinear(dim)

    @staticmethod
    def _block_forward(block, x, t, mask, rope):
        return block(x, t, mask=mask, rope=rope)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        attention_mask=None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        mask = attention_mask.bool() if attention_mask is not None else None

        zero_cond = torch.zeros_like(cond)

        t_locked = self.locked.time_embed(t.float())
        with torch.no_grad():
            x_locked = self.locked.input_embed(x, zero_cond, mask=mask)

        t_train = self.trainable_time_embed(t.float())
        x_train = self.trainable_input_embed(x, cond, mask=mask)

        x_locked = x_locked + self.input_zero_conv(x_train)

        rope = self.locked.rotary_embed.forward_from_seq_len(seq_len)

        for locked_block, train_block, zero_conv in zip(
            self.locked.transformer_blocks, self.trainable_blocks, self.zero_convs
        ):
            if self.checkpoint_activations:
                x_locked = ckpt_fn(
                    self._block_forward, locked_block, x_locked, t_locked, mask, rope,
                    use_reentrant=False,
                )
                x_train = ckpt_fn(
                    self._block_forward, train_block, x_train, t_train, mask, rope,
                    use_reentrant=False,
                )
            else:
                x_locked = locked_block(x_locked, t_locked, mask=mask, rope=rope)
                x_train = train_block(x_train, t_train, mask=mask, rope=rope)

            x_locked = x_locked + zero_conv(x_train)

        t_final = t_locked + self.output_t_zero_conv(t_train)
        x_locked = self.locked.norm_out(x_locked, t_final)
        output = self.locked.proj_out(x_locked)

        return output
