"""SASRec ringkas (Self-Attentive Sequential Recommendation, Kang & McAuley 2018).

Varian kompak: 2 layer transformer encoder kausal, panjang sekuens max 50.
Loss: BCE dengan negative sampling per step.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SASRec(nn.Module):
    def __init__(
        self,
        n_items: int,
        max_len: int = 50,
        emb_dim: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.n_items = n_items
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=4 * emb_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(emb_dim)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def _causal_mask(self, L: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(pos)
        pad_mask = seq == self.pad_idx
        out = self.encoder(
            x,
            mask=self._causal_mask(L, seq.device),
            src_key_padding_mask=pad_mask,
        )
        return self.ln(out)  # (B, L, d)

    @torch.no_grad()
    def score_last(self, seq: torch.Tensor) -> torch.Tensor:
        """Skor semua item dari state terakhir sekuens."""
        self.eval()
        h = self.forward(seq)[:, -1, :]          # (B, d)
        W = self.item_emb.weight                  # (n_items+1, d)
        return h @ W.T

    @torch.no_grad()
    def score_all_items_user(self, seq_1d: np.ndarray, device) -> np.ndarray:
        s = torch.from_numpy(seq_1d.astype(np.int64)).unsqueeze(0).to(device)
        scores = self.score_last(s).squeeze(0).cpu().numpy().astype(np.float32)
        # skip pad index
        scores[self.pad_idx] = -np.inf
        return scores[: self.n_items + 1][1:] if self.pad_idx == 0 else scores[: self.n_items]
