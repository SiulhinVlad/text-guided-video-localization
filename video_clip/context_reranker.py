from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from data import VideoCaptionExample
from metrics import summarize_recall
from scoring import ProjectionScorer, load_scorer
from segments import TemporalSegment, non_max_suppression, temporal_iou
from video_clip.segment_reranker import (
    SegmentCandidate,
    SegmentEvaluationResult,
    _prediction_row,
    _video_duration,
    generate_segment_candidates,
    segment_features,
)
from video_clip.train import _load_or_encode_video_windows
from video_clip.windows import VideoWindow
from video_clip.xclip_backend import XCLIPEmbedder


@dataclass
class ContextTrainingSummary:
    examples: int
    groups: int
    pairs: int
    final_loss: float
    checkpoint: Path


@dataclass
class ContextualLocalizationResult:
    query: str
    video_path: Path
    candidates_scored: int
    final_segments: list[TemporalSegment]


class ContextualSpanReranker(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        model_dim: int = 256,
        hidden_dim: int = 256,
        extra_feature_dim: int = 9,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.extra_feature_dim = extra_feature_dim
        self.model_dim = model_dim
        self.video_proj = nn.Linear(embedding_dim, model_dim)
        self.text_proj = nn.Linear(embedding_dim, model_dim)
        self.mix_proj = nn.Sequential(
            nn.Linear(model_dim * 4 + 2, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.score_head = nn.Sequential(
            nn.Linear(model_dim + extra_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def contextualize(
        self,
        window_embeddings: torch.Tensor,
        text_embedding: torch.Tensor,
        *,
        video_duration: float,
        window_centers: torch.Tensor,
    ) -> torch.Tensor:
        video = self.video_proj(window_embeddings.float())
        text = self.text_proj(text_embedding.float())
        if text.ndim == 1:
            text = text.unsqueeze(0).expand_as(video)
        center = window_centers.to(video.device, dtype=video.dtype).unsqueeze(-1)
        duration = max(float(video_duration), 1e-6)
        pos = torch.cat([center / duration, torch.sin(center / duration * np.pi).to(video.dtype)], dim=-1)
        mixed = self.mix_proj(torch.cat([video, text, torch.abs(video - text), video * text, pos], dim=-1))
        return self.encoder(mixed.unsqueeze(0)).squeeze(0)

    def score_spans(
        self,
        context_embeddings: torch.Tensor,
        candidates: list[SegmentCandidate],
        extra_features: torch.Tensor,
    ) -> torch.Tensor:
        span_rows = contextual_span_embeddings(context_embeddings, candidates)
        features = torch.cat([span_rows, extra_features.to(span_rows.device, dtype=span_rows.dtype)], dim=-1)
        return self.score_head(features).squeeze(-1)


def train_contextual_reranker(
    examples: list[VideoCaptionExample],
    *,
    output: Path,
    base_checkpoint: Path,
    init_checkpoint: Path | None = None,
    embedder: XCLIPEmbedder,
    scales: tuple[float, ...],
    strides: tuple[float, ...],
    embedding_cache_dir: Path | None,
    frames_per_clip: int = 8,
    span_window_counts: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
    max_positive_segments: int = 32,
    max_hard_negatives: int = 96,
    target_mode: str = "iou",
    epochs: int = 3,
    learning_rate: float = 5e-5,
    weight_decay: float = 1e-4,
    device: str | None = None,
) -> ContextTrainingSummary:
    if len(scales) != len(strides):
        raise ValueError("scales and strides must have matching lengths")
    if target_mode not in {"binary", "iou"}:
        raise ValueError("target_mode must be 'binary' or 'iou'")

    model_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    base_scorer = load_scorer(base_checkpoint, map_location=model_device).to(model_device)
    base_scorer.eval()
    if init_checkpoint is not None:
        model = load_contextual_reranker(init_checkpoint, map_location=model_device).to(model_device)
    else:
        model = ContextualSpanReranker(extra_feature_dim=9).to(model_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    state = ContextState()
    final_loss = 0.0
    total_groups = 0
    total_pairs = 0
    model.train()
    for _epoch in range(epochs):
        running = 0.0
        seen = 0
        for example in tqdm(examples, desc="Training contextual reranker"):
            if example.video_path is None:
                continue
            text_embedding = state.text_embedding(example.caption, embedder).to(model_device)
            target = TemporalSegment(example.start, example.end, 1.0)
            for scale, stride in zip(scales, strides, strict=True):
                windows, embeddings, candidates = state.windows_and_candidates(
                    example.video_path,
                    embedder=embedder,
                    scale=scale,
                    stride=stride,
                    frames_per_clip=frames_per_clip,
                    span_window_counts=span_window_counts,
                    embedding_cache_dir=embedding_cache_dir,
                )
                if not candidates:
                    continue
                video_duration = _video_duration(windows)
                selected, labels, features = select_context_training_candidates(
                    candidates,
                    embeddings,
                    text_embedding.detach().cpu(),
                    target=target,
                    video_duration=video_duration,
                    base_scorer=base_scorer,
                    max_positive_segments=max_positive_segments,
                    max_hard_negatives=max_hard_negatives,
                    target_mode=target_mode,
                    device=model_device,
                )
                if not selected:
                    continue
                window_centers = torch.as_tensor(
                    [(window.start + window.end) / 2.0 for window in windows],
                    dtype=torch.float32,
                    device=model_device,
                )
                context = model.contextualize(
                    embeddings.to(model_device),
                    text_embedding,
                    video_duration=video_duration,
                    window_centers=window_centers,
                )
                logits = model.score_spans(context, selected, features.to(model_device))
                loss = loss_fn(logits, labels.to(model_device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running += float(loss.detach().cpu()) * labels.numel()
                seen += labels.numel()
                total_groups += 1
                total_pairs += labels.numel()
        final_loss = running / max(seen, 1)

    save_contextual_reranker(
        output,
        model.cpu(),
        metadata={
            "backend": "xclip-contextual-span-reranker",
            "base_checkpoint": str(base_checkpoint),
            "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
            "scales": list(scales),
            "strides": list(strides),
            "frames_per_clip": frames_per_clip,
            "span_window_counts": list(span_window_counts),
            "max_positive_segments": max_positive_segments,
            "max_hard_negatives": max_hard_negatives,
            "target_mode": target_mode,
            "examples": len(examples),
            "groups": total_groups,
            "pairs": total_pairs,
        },
    )
    return ContextTrainingSummary(
        examples=len(examples),
        groups=total_groups,
        pairs=total_pairs,
        final_loss=final_loss,
        checkpoint=output,
    )


def localize_contextual_reranker(
    video_path: Path,
    query: str,
    *,
    base_checkpoint: Path,
    checkpoint: Path,
    embedder: XCLIPEmbedder,
    scales: tuple[float, ...],
    strides: tuple[float, ...],
    embedding_cache_dir: Path | None,
    frames_per_clip: int = 8,
    span_window_counts: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
    top_k: int = 5,
    nms_iou_threshold: float = 0.5,
    device: str | None = None,
) -> ContextualLocalizationResult:
    if len(scales) != len(strides):
        raise ValueError("scales and strides must have matching lengths")
    if not 0.0 <= nms_iou_threshold <= 1.0:
        raise ValueError("nms_iou_threshold must be between 0 and 1")

    model_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    base_scorer = load_scorer(base_checkpoint, map_location=model_device).to(model_device)
    base_scorer.eval()
    model = load_contextual_reranker(checkpoint, map_location=model_device).to(model_device)
    model.eval()

    state = ContextState()
    text_embedding = state.text_embedding(query, embedder).to(model_device)
    scored: list[TemporalSegment] = []
    for scale, stride in zip(scales, strides, strict=True):
        windows, embeddings, candidates = state.windows_and_candidates(
            video_path,
            embedder=embedder,
            scale=scale,
            stride=stride,
            frames_per_clip=frames_per_clip,
            span_window_counts=span_window_counts,
            embedding_cache_dir=embedding_cache_dir,
        )
        if not candidates:
            continue
        video_duration = _video_duration(windows)
        features = context_candidate_features(
            candidates,
            embeddings,
            text_embedding.detach().cpu(),
            video_duration=video_duration,
            base_scorer=base_scorer,
            device=model_device,
        )
        window_centers = torch.as_tensor(
            [(window.start + window.end) / 2.0 for window in windows],
            dtype=torch.float32,
            device=model_device,
        )
        with torch.no_grad():
            context = model.contextualize(
                embeddings.to(model_device),
                text_embedding,
                video_duration=video_duration,
                window_centers=window_centers,
            )
            logits = model.score_spans(context, candidates, features.to(model_device))
            scores = torch.sigmoid(logits).cpu()
        scored.extend(
            TemporalSegment(candidate.start, candidate.end, float(score))
            for candidate, score in zip(candidates, scores, strict=True)
        )

    if not scored:
        raise RuntimeError(
            "No temporal candidates were generated. Verify that the video is decodable "
            "and that its embedding cache, if supplied, contains non-empty window embeddings."
        )

    return ContextualLocalizationResult(
        query=query,
        video_path=video_path,
        candidates_scored=len(scored),
        final_segments=non_max_suppression(
            scored,
            iou_threshold=nms_iou_threshold,
            top_k=top_k,
        ),
    )


def evaluate_contextual_reranker(
    examples: list[VideoCaptionExample],
    *,
    base_checkpoint: Path,
    checkpoint: Path,
    embedder: XCLIPEmbedder,
    scales: tuple[float, ...],
    strides: tuple[float, ...],
    embedding_cache_dir: Path | None,
    frames_per_clip: int,
    span_window_counts: tuple[int, ...],
    iou_thresholds: list[float],
    top_k: int = 5,
    limit: int | None = None,
    predictions_out: Path | None = None,
    device: str | None = None,
) -> SegmentEvaluationResult:
    selected = examples if limit is None else examples[:limit]
    model_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    base_scorer = load_scorer(base_checkpoint, map_location=model_device).to(model_device)
    base_scorer.eval()
    model = load_contextual_reranker(checkpoint, map_location=model_device).to(model_device)
    model.eval()

    state = ContextState()
    predictions: list[list[TemporalSegment]] = []
    targets: list[TemporalSegment] = []
    skipped = 0

    output_handle = None
    if predictions_out is not None:
        predictions_out.parent.mkdir(parents=True, exist_ok=True)
        output_handle = predictions_out.open("w", encoding="utf-8")

    try:
        for example in tqdm(selected, desc="Evaluating contextual reranker"):
            if example.video_path is None:
                skipped += 1
                continue
            text_embedding = state.text_embedding(example.caption, embedder).to(model_device)
            scored: list[TemporalSegment] = []
            for scale, stride in zip(scales, strides, strict=True):
                windows, embeddings, candidates = state.windows_and_candidates(
                    example.video_path,
                    embedder=embedder,
                    scale=scale,
                    stride=stride,
                    frames_per_clip=frames_per_clip,
                    span_window_counts=span_window_counts,
                    embedding_cache_dir=embedding_cache_dir,
                )
                if not candidates:
                    continue
                video_duration = _video_duration(windows)
                features = context_candidate_features(
                    candidates,
                    embeddings,
                    text_embedding.detach().cpu(),
                    video_duration=video_duration,
                    base_scorer=base_scorer,
                    device=model_device,
                )
                window_centers = torch.as_tensor(
                    [(window.start + window.end) / 2.0 for window in windows],
                    dtype=torch.float32,
                    device=model_device,
                )
                with torch.no_grad():
                    context = model.contextualize(
                        embeddings.to(model_device),
                        text_embedding,
                        video_duration=video_duration,
                        window_centers=window_centers,
                    )
                    logits = model.score_spans(context, candidates, features.to(model_device))
                    scores = torch.sigmoid(logits).cpu()
                scored.extend(
                    TemporalSegment(candidate.start, candidate.end, float(score))
                    for candidate, score in zip(candidates, scores, strict=True)
                )
            final_segments = non_max_suppression(scored, iou_threshold=0.5, top_k=top_k)
            target = TemporalSegment(example.start, example.end, 1.0)
            predictions.append(final_segments)
            targets.append(target)
            if output_handle is not None:
                output_handle.write(json.dumps(_prediction_row(example, final_segments, target)) + "\n")
    finally:
        if output_handle is not None:
            output_handle.close()

    return SegmentEvaluationResult(
        metrics=summarize_recall(predictions, targets, iou_thresholds),
        evaluated=len(targets),
        skipped=skipped,
    )


class ContextState:
    def __init__(self) -> None:
        self.video_cache: dict[tuple[Path, float, float], tuple[list[VideoWindow], torch.Tensor]] = {}
        self.candidate_cache: dict[tuple[Path, float, float], list[SegmentCandidate]] = {}
        self.text_cache: dict[str, torch.Tensor] = {}

    def text_embedding(self, caption: str, embedder: XCLIPEmbedder) -> torch.Tensor:
        if caption not in self.text_cache:
            self.text_cache[caption] = embedder.encode_text(caption)[0]
        return self.text_cache[caption]

    def windows_and_candidates(
        self,
        video_path: Path,
        *,
        embedder: XCLIPEmbedder,
        scale: float,
        stride: float,
        frames_per_clip: int,
        span_window_counts: tuple[int, ...],
        embedding_cache_dir: Path | None,
    ) -> tuple[list[VideoWindow], torch.Tensor, list[SegmentCandidate]]:
        cache_key = (video_path, float(scale), float(stride))
        if cache_key not in self.video_cache:
            self.video_cache[cache_key] = _load_or_encode_video_windows(
                video_path,
                embedder=embedder,
                clip_seconds=scale,
                stride_seconds=stride,
                frames_per_clip=frames_per_clip,
                embedding_cache_dir=embedding_cache_dir,
            )
        windows, embeddings = self.video_cache[cache_key]
        if cache_key not in self.candidate_cache:
            self.candidate_cache[cache_key] = generate_segment_candidates(
                windows,
                scale=scale,
                span_window_counts=span_window_counts,
            )
        return windows, embeddings, self.candidate_cache[cache_key]


def select_context_training_candidates(
    candidates: list[SegmentCandidate],
    embeddings: torch.Tensor,
    text_embedding: torch.Tensor,
    *,
    target: TemporalSegment,
    video_duration: float,
    base_scorer: ProjectionScorer,
    max_positive_segments: int,
    max_hard_negatives: int,
    target_mode: str,
    device: torch.device,
) -> tuple[list[SegmentCandidate], torch.Tensor, torch.Tensor]:
    features = context_candidate_features(
        candidates,
        embeddings,
        text_embedding,
        video_duration=video_duration,
        base_scorer=base_scorer,
        device=device,
    )
    ious = np.asarray(
        [
            temporal_iou(TemporalSegment(candidate.start, candidate.end, 1.0), target)
            for candidate in candidates
        ],
        dtype=np.float32,
    )
    base_scores = features[:, -1].numpy()
    positive_indices = np.argsort(ious)[::-1][:max_positive_segments]
    negative_pool = np.flatnonzero(ious <= 0.2)
    hard_negative_indices = negative_pool[np.argsort(base_scores[negative_pool])[::-1]][
        :max_hard_negatives
    ]
    selected_indices = list(dict.fromkeys([*positive_indices.tolist(), *hard_negative_indices.tolist()]))
    selected = [candidates[idx] for idx in selected_indices]
    if target_mode == "iou":
        labels = torch.as_tensor(ious[selected_indices], dtype=torch.float32)
        for idx, original_idx in enumerate(selected_indices):
            if original_idx in set(hard_negative_indices.tolist()):
                labels[idx] = 0.0
    else:
        labels = torch.as_tensor([1.0 if ious[idx] >= 0.5 else 0.0 for idx in selected_indices], dtype=torch.float32)
    return selected, labels, features[selected_indices]


def context_candidate_features(
    candidates: list[SegmentCandidate],
    embeddings: torch.Tensor,
    text_embedding: torch.Tensor,
    *,
    video_duration: float,
    base_scorer: ProjectionScorer,
    device: torch.device,
) -> torch.Tensor:
    features = segment_features(
        candidates,
        video_duration=video_duration,
        embeddings=embeddings,
        text_embedding=text_embedding,
        use_score_features=True,
    )
    from video_clip.segment_reranker import segment_embeddings

    span_embeddings = segment_embeddings(embeddings, candidates)
    with torch.no_grad():
        base_scores = base_scorer.predict_proba(
            span_embeddings.to(device),
            text_embedding.to(device),
            features.to(device),
        ).cpu()
    return torch.cat([features, base_scores.unsqueeze(1)], dim=1)


def contextual_span_embeddings(
    context_embeddings: torch.Tensor,
    candidates: list[SegmentCandidate],
) -> torch.Tensor:
    if not candidates:
        return torch.empty(0, context_embeddings.shape[-1], device=context_embeddings.device)
    prefix = torch.cat(
        [
            torch.zeros(1, context_embeddings.shape[-1], device=context_embeddings.device),
            context_embeddings.float().cumsum(dim=0),
        ]
    )
    rows = []
    for candidate in candidates:
        total = prefix[candidate.end_idx + 1] - prefix[candidate.start_idx]
        rows.append(total / (candidate.end_idx - candidate.start_idx + 1))
    return torch.stack(rows, dim=0)


def save_contextual_reranker(
    path: Path,
    model: ContextualSpanReranker,
    *,
    metadata: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "embedding_dim": model.video_proj.in_features,
        "model_dim": model.model_dim,
        "hidden_dim": model.score_head[0].out_features,
        "extra_feature_dim": model.extra_feature_dim,
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_contextual_reranker(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> ContextualSpanReranker:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = ContextualSpanReranker(
        embedding_dim=int(payload["embedding_dim"]),
        model_dim=int(payload["model_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        extra_feature_dim=int(payload.get("extra_feature_dim", 9)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
