from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def cosine_scores(frame_embeddings: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
    frames = F.normalize(frame_embeddings, dim=-1)
    text = F.normalize(text_embedding, dim=-1)
    if text.ndim == 1:
        text = text.unsqueeze(0)
    return frames @ text.squeeze(0)


class ProjectionScorer(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 512,
        projection_dim: int = 256,
        hidden_dim: int = 256,
        extra_feature_dim: int = 0,
    ):
        super().__init__()
        self.extra_feature_dim = extra_feature_dim
        self.frame_proj = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(projection_dim * 4 + extra_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        frame_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        extra_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        frame = self.frame_proj(frame_embeddings)
        text = self.text_proj(text_embeddings)
        if text.ndim == 1:
            text = text.unsqueeze(0).expand_as(frame)
        elif text.shape[0] == 1 and frame.shape[0] != 1:
            text = text.expand(frame.shape[0], -1)
        features = torch.cat([frame, text, torch.abs(frame - text), frame * text], dim=-1)
        if self.extra_feature_dim:
            if extra_features is None:
                raise ValueError("extra_features are required for this scorer")
            if extra_features.ndim == 1:
                extra_features = extra_features.unsqueeze(-1)
            extra_features = extra_features.to(features.device, dtype=features.dtype)
            features = torch.cat([features, extra_features], dim=-1)
        return self.score_head(features).squeeze(-1)

    def predict_proba(
        self,
        frame_embeddings: torch.Tensor,
        text_embedding: torch.Tensor,
        extra_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self(frame_embeddings, text_embedding, extra_features))


def save_scorer(path: Path, model: ProjectionScorer, *, metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "embedding_dim": model.frame_proj[0].in_features,
        "projection_dim": model.frame_proj[0].out_features,
        "hidden_dim": model.score_head[0].out_features,
        "extra_feature_dim": model.extra_feature_dim,
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_scorer_metadata(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return dict(payload.get("metadata") or {})


def load_scorer(path: Path, *, map_location: str | torch.device = "cpu") -> ProjectionScorer:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = ProjectionScorer(
        embedding_dim=int(payload["embedding_dim"]),
        projection_dim=int(payload["projection_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        extra_feature_dim=int(payload.get("extra_feature_dim", 0)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
