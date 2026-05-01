"""Lightweight FT-Transformer implementation for tabular regression.

Reference: Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data" (NeurIPS 2021).
Each numerical feature is tokenized to a d-dim embedding, a [CLS] token is prepended,
and a stack of transformer encoder blocks operates over the tokens. The CLS embedding
is used for the final regression head.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class NumericalTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight + self.bias


class TransformerBlock(nn.Module):
    def __init__(self, d_token: int, n_heads: int, ffn_mult: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        hidden = int(d_token * ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_token),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out)
        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))
        return x


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_token: int = 64,
        n_blocks: int = 3,
        n_heads: int = 8,
        ffn_mult: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = NumericalTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_token, n_heads, ffn_mult, dropout) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        h = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h[:, 0])
        return self.head(h).squeeze(-1)
