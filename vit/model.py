import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Tuple

import math
import numpy as np

class MHSA(nn.Module):
    def __init__(
        self, 
         dim: int, 
         num_heads: int, 
         dim_head = int, 
         dropout:float=0.
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads 
        self.inner_dim = num_heads * dim_head
        self.scale = dim_head ** -0.5

        self.ln = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, 3 * self.inner_dim)
        self.attn = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.out_proj = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        x = self.ln(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)

        qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attn(qk)
        attn = self.dropout(attn)

        # out = einsum(qk, v, "b h a b, b h b d -> b a (h d)") 
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)
        
        return out


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.ffn(x)


class Transformer(nn.Module): 
    def __init__(
        self, 
        dim: int,
        hidden_dim: int,
        depth: int = 8,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0., 
    ):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MHSA(dim, num_heads, dim_head, dropout),
                MLP(dim, hidden_dim, dropout)
            ]))

    def forward(self, x: torch.Tensor):
        for att, ffn in self.layers:
            x = att(x) + x
            x = ffn(x) + x

        x = self.ln(x)
        return x

@dataclass
class ViTConfig():
    image_size: Tuple[int, int]
    patch_size: Tuple[int, int]
    num_classes: int
    dim: int
    dim_head: int
    depth: int
    num_heads: int
    hidden_dim: int
    channels: int
    pool: str = 'cls'
    dropout: float = 0.
    emb_dropout: float = 0.

class Rearrange(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b c h w -> b (h w) c')

class ViT(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        image_width, image_height = config.image_size
        patch_width, patch_height = config.patch_size

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.patchify = nn.Sequential(
            nn.Conv2d(
                in_channels=config.channels, 
                out_channels=config.dim, 
                kernel_size=config.patch_size, 
                stride=config.patch_size
            ),
            Rearrange(),
            nn.LayerNorm(config.dim)
        )

        self.pool = config.pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim))
        self.pe = nn.Parameter(torch.randn(1, num_patches + 1, config.dim))
        self.dropout = nn.Dropout(config.dropout)

        self.transformer = Transformer(
            config.dim, 
            config.hidden_dim, 
            config.depth, 
            config.num_heads, 
            config.dim_head, 
            config.dropout
        )
        self.mlp = nn.Linear(config.dim, config.num_classes)

    def forward(self, x):
        x = self.patchify(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pe[:, :(n+1)]
        x = self.dropout(x)
        x = self.transformer(x)
        
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.mlp(x)
        return x

class KellyAdamW(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, lr, betas, eps, weight_decay)
        self.running_loss = []
        self.epoch_losses = []
        
    def step(self, running_loss=None):
        if running_loss is not None:
            self.running_loss.append(running_loss)
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Weight decay first (AdamW style)
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Then Kelly-scaled Adam update
                state = self.state[p]
                kelly_fr = self._calculate_kelly_fraction(
                    p.grad.norm().item(),  # Convert tensor to float
                    self.running_loss[-1] if self.running_loss else 1.0,
                    self.epoch_losses
                )
                
                step_size = group['lr'] * kelly_fr
                p.data.addcdiv_(p.grad, torch.sqrt(state['v']) + self.eps, value=-step_size)

    def _calculate_kelly_fraction(
        self,
        gradient: float,
        current_loss: float,  
        historical_losses: list  
    ) -> float:
        if len(historical_losses) < 2:
            return 1.0  # Return 1.0 instead of learning_rate
            
        avg_historical_loss = np.mean(historical_losses[-5:])
        # Flip win_prob since lower loss is better
        win_prob = 1.0 / (1.0 + np.exp(-(avg_historical_loss - current_loss)))
        
        # Use parameter norm from current group instead of self.weights_
        param_norm = 0
        for group in self.param_groups:
            for p in group['params']:
                param_norm += p.data.norm().item()
        param_norm = param_norm / len(self.param_groups)
        
        edge = gradient / (param_norm + 1e-8)
        kelly_fraction = (win_prob * edge - (1 - win_prob)) / edge
        return np.clip(kelly_fraction, 0, 1.0)  # Clip between 0 and 1
