"""Neural Collaborative Filtering (He et al., 2017).

Implementasi ringkas dengan dua komponen:
- GMF: generalized matrix factorization (element-wise product embedding).
- MLP: tower concat user+item embedding -> MLP.
Keduanya digabung via NeuMF head.

Loss: BCE dengan negative sampling 1:4 (implicit positive rating >= 3.5).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        gmf_dim: int = 32,
        mlp_dim: int = 32,
        hidden: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.user_gmf = nn.Embedding(n_users, gmf_dim)
        self.item_gmf = nn.Embedding(n_items, gmf_dim)
        self.user_mlp = nn.Embedding(n_users, mlp_dim)
        self.item_mlp = nn.Embedding(n_items, mlp_dim)

        layers = []
        in_dim = 2 * mlp_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(gmf_dim + hidden[-1], 1)

        for emb in [self.user_gmf, self.item_gmf, self.user_mlp, self.item_mlp]:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        gmf = self.user_gmf(u) * self.item_gmf(i)
        mlp_in = torch.cat([self.user_mlp(u), self.item_mlp(i)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        x = torch.cat([gmf, mlp_out], dim=-1)
        return self.head(x).squeeze(-1)

    @torch.no_grad()
    def score_all_items(self, u_idx: int, device: str | torch.device) -> np.ndarray:
        self.eval()
        n = self.item_gmf.num_embeddings
        u = torch.full((n,), u_idx, dtype=torch.long, device=device)
        i = torch.arange(n, dtype=torch.long, device=device)
        out = self(u, i)
        return out.cpu().numpy().astype(np.float32)


class TwoTower(nn.Module):
    """Dua tower terpisah -> dot product. Cocok untuk retrieval ANN (FAISS)."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 64,
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        def tower(in_dim: int) -> nn.Sequential:
            layers = []
            prev = in_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, emb_dim))
            return nn.Sequential(*layers)

        self.user_tower = tower(emb_dim)
        self.item_tower = tower(emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def user_repr(self, u: torch.Tensor) -> torch.Tensor:
        return self.user_tower(self.user_emb(u))

    def item_repr(self, i: torch.Tensor) -> torch.Tensor:
        return self.item_tower(self.item_emb(i))

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return (self.user_repr(u) * self.item_repr(i)).sum(dim=-1)

    @torch.no_grad()
    def all_item_repr(self, device: str | torch.device, batch: int = 16384) -> np.ndarray:
        self.eval()
        n = self.item_emb.num_embeddings
        outs = []
        for s in range(0, n, batch):
            idx = torch.arange(s, min(s + batch, n), dtype=torch.long, device=device)
            outs.append(self.item_repr(idx).cpu().numpy().astype(np.float32))
        return np.concatenate(outs, axis=0)

    @torch.no_grad()
    def all_user_repr(self, device: str | torch.device, batch: int = 16384) -> np.ndarray:
        self.eval()
        n = self.user_emb.num_embeddings
        outs = []
        for s in range(0, n, batch):
            idx = torch.arange(s, min(s + batch, n), dtype=torch.long, device=device)
            outs.append(self.user_repr(idx).cpu().numpy().astype(np.float32))
        return np.concatenate(outs, axis=0)
