from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data import VideoCaptionExample
from metrics import summarize_recall
from scoring import ProjectionScorer, save_scorer, load_scorer
from segments import TemporalSegment, non_max_suppression, temporal_iou
from video_clip.train import _load_or_encode_video_windows
from video_clip.windows import VideoWindow
from video_clip.xclip_backend import XCLIPEmbedder


@dataclass(frozen=True)
class SegmentCandidate:
    scale: float
    start_idx: int
    end_idx: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class SegmentTrainingSummary:
    examples: int
    pairs: int
    positives: int
    negatives: int
    final_loss: float
    checkpoint: Path


@dataclass
class SegmentEvaluationResult:
    metrics: dict[str, float]
    evaluated: int
    skipped: int


class BoundaryRefiner(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        projection_dim: int = 256,
        hidden_dim: int = 256,
        extra_feature_dim: int = 0,
    ):
        super().__init__()
        self.extra_feature_dim = extra_feature_dim
        self.video_proj = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.offset_head = nn.Sequential(
            nn.Linear(projection_dim * 4 + extra_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        video_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        extra_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        video = self.video_proj(video_embeddings)
        text = self.text_proj(text_embeddings)
        if text.ndim == 1:
            text = text.unsqueeze(0).expand_as(video)
        elif text.shape[0] == 1 and video.shape[0] != 1:
            text = text.expand(video.shape[0], -1)
        features = torch.cat([video, text, torch.abs(video - text), video * text], dim=-1)
        if self.extra_feature_dim:
            if extra_features is None:
                raise ValueError("extra_features are required for this boundary refiner")
            features = torch.cat([features, extra_features.to(features.device, dtype=features.dtype)], dim=-1)
        return self.offset_head(features)


@dataclass
class BoundaryTrainingSummary:
    examples: int
    pairs: int
    final_loss: float
    checkpoint: Path


def train_segment_reranker(
    examples: list[VideoCaptionExample],
    *,
    output: Path,
    embedder: XCLIPEmbedder,
    scales: tuple[float, ...],
    strides: tuple[float, ...],
    embedding_cache_dir: Path | None,
    frames_per_clip: int = 8,
    span_window_counts: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
    positive_iou_threshold: float = 0.5,
    negative_iou_threshold: float = 0.1,
    max_positive_segments: int = 16,
    negative_ratio: int = 4,
    target_mode: str = "binary",
    use_score_features: bool = False,
    epochs: int = 3,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str | None = None,
) -> SegmentTrainingSummary:
    if len(scales) != len(strides):
        raise ValueError("scales and strides must have matching lengths")

    model_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    video_rows, text_rows, feature_rows, labels = build_segment_training_pairs(
        examples,
        embedder=embedder,
        scales=scales,
        strides=strides,
        embedding_cache_dir=embedding_cache_dir,
        frames_per_clip=frames_per_clip,
        span_window_counts=span_window_counts,
        positive_iou_threshold=positive_iou_threshold,
        negative_iou_threshold=negative_iou_threshold,
        max_positive_segments=max_positive_segments,
        negative_ratio=negative_ratio,
        target_mode=target_mode,
        use_score_features=use_score_features,
    )
    if labels.numel() == 0:
        raise ValueError("No segment training pairs were produced.")

    model = ProjectionScorer(
        embedding_dim=video_rows.shape[-1],
        projection_dim=256,
        hidden_dim=256,
        extra_feature_dim=feature_rows.shape[-1],
    ).to(model_device)
    dataset = TensorDataset(video_rows.float(), text_rows.float(), feature_rows.float(), labels.float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    final_loss = 0.0
    model.train()
    for _epoch in range(epochs):
        running = 0.0
        seen = 0
        for batch_video, batch_text, batch_features, batch_labels in loader:
            batch_video = batch_video.to(model_device)
            batch_text = batch_text.to(model_device)
            batch_features = batch_features.to(model_device)
            batch_labels = batch_labels.to(model_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_video, batch_text, batch_features)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * batch_labels.numel()
            seen += batch_labels.numel()
        final_loss = running / max(seen, 1)

    positives = int((labels >= positive_iou_threshold).sum().item())
    negatives = int(labels.numel() - positives)
    save_scorer(
        output,
        model.cpu(),
        metadata={
            "backend": "xclip-segment-reranker",
            "scales": list(scales),
            "strides": list(strides),
            "frames_per_clip": frames_per_clip,
            "span_window_counts": list(span_window_counts),
            "positive_iou_threshold": positive_iou_threshold,
            "negative_iou_threshold": negative_iou_threshold,
            "target_mode": target_mode,
            "use_score_features": use_score_features,
            "examples": len(examples),
            "pairs": int(labels.numel()),
        },
    )
    return SegmentTrainingSummary(
        examples=len(examples),
        pairs=int(labels.numel()),
        positives=positives,
        negatives=negatives,
        final_loss=final_loss,
        checkpoint=output,
    )


def train_hard_negative_segment_reranker(
    examples: list[VideoCaptionExample],
    *,
    output: Path,
    base_checkpoint: Path,
    embedder: XCLIPEmbedder,
    scales: tuple[float, ...],
    strides: tuple[float, ...],
    embedding_cache_dir: Path | None,
    frames_per_clip: int = 8,
    span_window_counts: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
    max_positive_segments: int = 32,
    max_hard_negatives: int = 128,
    max_broad_negatives: int = 64,
    hard_negative_iou_threshold: float = 0.2,
    broad_negative_iou_threshold: float = 0.5,
    broad_negative_duration_ratio: float = 2.0,
    target_mode: str = "iou",
    use_score_features: bool = True,
    epochs: int = 3,
    batch_size: int = 256,
    learning_rate: float = 5e-5,
    weight_decay: float = 1e-4,
    device: str | None = None,
) -> SegmentTrainingSummary:
    if len(scales) != len(strides):
        raise ValueError("scales and strides must have matching lengths")
    if target_mode not in {"binary", "iou"}:
        raise ValueError("target_mode must be 'binary' or 'iou'")

    model_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    base_scorer = load_scorer(base_checkpoint, map_location=model_device).to(model_device)
    base_scorer.eval()
    use_score_features = use_score_features or base_scorer.extra_feature_dim > 4
    video_rows, text_rows, feature_rows, labels = build_hard_negative_training_pairs(
        examples,
        base_scorer=base_scorer,
        embedder=embedder,
        scales=scales,
        strides=strides,
        embedding_cache_dir=embedding_cache_dir,
        frames_per_clip=frames_per_clip,
        span_window_counts=span_window_counts,
        max_positive_segments=max_positive_segments,
        max_hard_negatives=max_hard_negatives,
        max_broad_negatives=max_broad_negatives,
        hard_negative_iou_threshold=hard_negative_iou_threshold,
        broad_negative_iou_threshold=broad_negative_iou_threshold,
        broad_negative_duration_ratio=broad_negative_duration_ratio,
        target_mode=target_mode,
        use_score_features=use_score_features,
        device=model_device,
    )
    if labels.numel() == 0:
        raise ValueError("No hard-negative training pairs were produced.")

    model = ProjectionScorer(
        embedding_dim=video_rows.shape[-1],
        projection_dim=256,
        hidden_dim=256,
        extra_feature_dim=feature_rows.shape[-1],
    ).to(model_device)
    _initialize_projection_scorer(model, base_scorer)
    dataset = TensorDataset(video_rows.float(), text_rows.float(), feature_rows.float(), labels.float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    final_loss = 0.0
    model.train()
    for _epoch in range(epochs):
        running = 0.0
        seen = 0
        for batch_video, batch_text, batch_features, batch_labels in loader:
            batch_video = batch_video.to(model_device)
            batch_text = batch_text.to(model_device)
            batch_features = batch_features.to(model_device)
            batch_labels = batch_labels.to(model_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_video, batch_text, batch_features)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * batch_labels.numel()
            seen += batch_labels.numel()
        final_loss = running / max(seen, 1)

    positives = int((labels >= broad_negative_iou_threshold).sum().item())
    negatives = int(labels.numel() - positives)
    save_scorer(
        output,
        model.cpu(),
        metadata={
            "backend": "xclip-hard-negative-segment-reranker",
            "base_checkpoint": str(base_checkpoint),
            "scales": list(scales),
            "strides": list(strides),
            "frames_per_clip": frames_per_clip,
            "span_window_counts": list(span_window_counts),
            "max_positive_segments": max_positive_segments,
            "max_hard_negatives": max_hard_negatives,
            "max_broad_negatives": max_broad_negatives,
            "hard_negative_iou_threshold": hard_negative_iou_threshold,
            "broad_negative_iou_threshold": broad_negative_iou_threshold,
            "broad_negative_duration_ratio": broad_negative_duration_ratio,
            "target_mode": target_mode,
            "use_score_features": use_score_features,
            "examples": len(examples),
            "pairs": int(labels.numel()),
        },
    )
    return SegmentTrainingSummary(
        examples=len(examples),
        pairs=int(labels.numel()),
        positives=positives,
        negatives=negatives,
        final_loss=final_loss,
        checkpoint=output,
    )


def train_boundary_refiner(
    examples: list[VideoCaptionExample],
    *,
    output: Path,
    embedder: XCLIPEmbedder,
    scorer_checkpoint: Path,
    scales: tuple[float, ...],
    strides: tuple[float, ...],
    embedding_cache_dir: Path | None,
    frames_per_clip: int = 8,
    span_window_counts: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
    min_train_iou: float = 0.3,
    max_segments_per_example: int = 64,
    use_score_features: bool = True,
    epochs: int = 3,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str | None = None,
) -> BoundaryTrainingSummary:
    if len(scales) != len(strides):
        raise ValueError("scales and strides must have matching lengths")

    model_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    scorer = load_scorer(scorer_checkpoint, map_location=model_device).to(model_device)
    scorer.eval()
    use_score_features = use_score_features or scorer.extra_feature_dim > 4
    video_rows, text_rows, feature_rows, offset_rows = build_boundary_training_pairs(
        examples,
        scorer=scorer,
        embedder=embedder,
        scales=scales,
        strides=strides,
        embedding_cache_dir=embedding_cache_dir,
        frames_per_clip=frames_per_clip,
        span_window_counts=span_window_counts,
        min_train_iou=min_train_iou,
        max_segments_per_example=max_segments_per_example,
        use_score_features=use_score_features,
        device=model_device,
    )
    if offset_rows.numel() == 0:
        raise ValueError("No boundary training pairs were produced.")

    model = BoundaryRefiner(
        embedding_dim=video_rows.shape[-1],
        projection_dim=256,
        hidden_dim=256,
        extra_feature_dim=feature_rows.shape[-1],
    ).to(model_device)
    dataset = TensorDataset(video_rows.float(), text_rows.float(), feature_rows.float(), offset_rows.float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()

    final_loss = 0.0
    model.train()
    for _epoch in range(epochs):
        running = 0.0
        seen = 0
        for batch_video, batch_text, batch_features, batch_offsets in loader:
            batch_video = batch_video.to(model_device)
            batch_text = batch_text.to(model_device)
            batch_features = batch_features.to(model_device)
            batch_offsets = batch_offsets.to(model_device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(batch_video, batch_text, batch_features)
            loss = loss_fn(predicted, batch_offsets)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * batch_offsets.shape[0]
            seen += batch_offsets.shape[0]
        final_loss = running / max(seen, 1)

    save_boundary_refiner(
        output,
        model.cpu(),
        metadata={
            "backend": "xclip-boundary-refiner",
            "scorer_checkpoint": str(scorer_checkpoint),
            "scales": list(scales),
            "strides": list(strides),
            "frames_per_clip": frames_per_clip,
            "span_window_counts": list(span_window_counts),
            "min_train_iou": min_train_iou,
            "max_segments_per_example": max_segments_per_example,
            "use_score_features": use_score_features,
            "examples": len(examples),
            "pairs": int(offset_rows.shape[0]),
        },
    )
    return BoundaryTrainingSummary(
        examples=len(examples),
        pairs=int(offset_rows.shape[0]),
        final_loss=final_loss,
        checkpoint=output,
    )


def build_boundary_training_pairs(
    examples: list[VideoCaptionExample],
    *,
    scorer: ProjectionScorer,
    embedder: XCLIPEmbedder,
    scales: tuple[float, ...],
    strides: tuple[float, ...],
    embedding_cache_dir: Path | None,
    frames_per_clip: int,
    span_window_counts: tuple[int, ...],
    min_train_iou: float,
    max_segments_per_example: int,
    use_score_features: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    video_cache: dict[tuple[Path, float, float], tuple[list[VideoWindow], torch.Tensor]] = {}
    candidate_cache: dict[tuple[Path, float, float], list[SegmentCandidate]] = {}
    text_cache: dict[str, torch.Tensor] = {}
    video_rows: list[torch.Tensor] = []
    text_rows: list[torch.Tensor] = []
    feature_rows: list[torch.Tensor] = []
    offset_rows: list[torch.Tensor] = []

    for example in tqdm(examples, desc="Building boundary refinement pairs"):
        if example.video_path is None:
            continue
        if example.caption not in text_cache:
            text_cache[example.caption] = embedder.encode_text(example.caption)[0]
        text_embedding = text_cache[example.caption]
        target = TemporalSegment(example.start, example.end, 1.0)
        selected: list[tuple[float, SegmentCandidate, torch.Tensor, torch.Tensor]] = []

        for scale, stride in zip(scales, strides, strict=True):
            cache_key = (example.video_path, float(scale), float(stride))
            if cache_key not in video_cache:
                video_cache[cache_key] = _load_or_encode_video_windows(
                    example.video_path,
                    embedder=embedder,
                    clip_seconds=scale,
                    stride_seconds=stride,
                    frames_per_clip=frames_per_clip,
                    embedding_cache_dir=embedding_cache_dir,
                )
            windows, embeddings = video_cache[cache_key]
            if cache_key not in candidate_cache:
                candidate_cache[cache_key] = generate_segment_candidates(
                    windows,
                    scale=scale,
                    span_window_counts=span_window_counts,
                )
            candidates = candidate_cache[cache_key]
            if not candidates:
                continue
            video_duration = _video_duration(windows)
            batch_embeddings = segment_embeddings(embeddings, candidates)
            batch_features = segment_features(
                candidates,
                video_duration=video_duration,
                embeddings=embeddings,
                text_embedding=text_embedding,
                use_score_features=use_score_features,
            )
            scores = scorer.predict_proba(
                batch_embeddings.to(device),
                text_embedding.to(device),
                batch_features.to(device),
            ).cpu()
            ious = [
                temporal_iou(TemporalSegment(candidate.start, candidate.end, 1.0), target)
                for candidate in candidates
            ]
            for idx, (candidate, score, iou) in enumerate(zip(candidates, scores, ious, strict=True)):
                if iou >= min_train_iou:
                    selected.append((float(score), candidate, batch_embeddings[idx], batch_features[idx]))

        selected = sorted(selected, key=lambda item: item[0], reverse=True)[:max_segments_per_example]
        for _score, candidate, embedding, features in selected:
            duration = max(candidate.duration, 1e-6)
            offsets = torch.tensor(
                [
                    np.clip((example.start - candidate.start) / duration, -1.0, 1.0),
                    np.clip((example.end - candidate.end) / duration, -1.0, 1.0),
                ],
                dtype=torch.float32,
            )
            video_rows.append(embedding.unsqueeze(0))
            text_rows.append(text_embedding.unsqueeze(0))
            feature_rows.append(features.unsqueeze(0))
            offset_rows.append(offsets.unsqueeze(0))

    if not video_rows:
        return torch.empty(0, 512), torch.empty(0, 512), torch.empty(0, 8 if use_score_features else 4), torch.empty(0, 2)
    return torch.cat(video_rows), torch.cat(text_rows), torch.cat(feature_rows), torch.cat(offset_rows)


def build_segment_training_pairs(
    examples: list[VideoCaptionExample],
    *,
    embedder: XCLIPEmbedder,
    scales: tuple[float, ...],
    strides: tuple[float, ...],
    embedding_cache_dir: Path | None,
    frames_per_clip: int,
    span_window_counts: tuple[int, ...],
    positive_iou_threshold: float,
    negative_iou_threshold: float,
    max_positive_segments: int,
    negative_ratio: int,
    target_mode: str,
    use_score_features: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if target_mode not in {"binary", "iou"}:
        raise ValueError("target_mode must be 'binary' or 'iou'")
    rng = np.random.default_rng(11)
    video_cache: dict[tuple[Path, float, float], tuple[list[VideoWindow], torch.Tensor]] = {}
    candidate_cache: dict[tuple[Path, float, float], list[SegmentCandidate]] = {}
    text_cache: dict[str, torch.Tensor] = {}
    video_rows: list[torch.Tensor] = []
    text_rows: list[torch.Tensor] = []
    feature_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []

    for example in tqdm(examples, desc="Building segment reranker pairs"):
        if example.video_path is None:
            continue
        if example.caption not in text_cache:
            text_cache[example.caption] = embedder.encode_text(example.caption)[0]
        text_embedding = text_cache[example.caption]

        selected_parts: list[tuple[np.ndarray, list[int], list[SegmentCandidate], torch.Tensor, float]] = []
        for scale, stride in zip(scales, strides, strict=True):
            cache_key = (example.video_path, float(scale), float(stride))
            if cache_key not in video_cache:
                video_cache[cache_key] = _load_or_encode_video_windows(
                    example.video_path,
                    embedder=embedder,
                    clip_seconds=scale,
                    stride_seconds=stride,
                    frames_per_clip=frames_per_clip,
                    embedding_cache_dir=embedding_cache_dir,
                )
            windows, embeddings = video_cache[cache_key]
            if cache_key not in candidate_cache:
                candidate_cache[cache_key] = generate_segment_candidates(
                    windows,
                    scale=scale,
                    span_window_counts=span_window_counts,
                )
            candidates = candidate_cache[cache_key]
            if not candidates:
                continue

            ious = np.asarray(
                [
                    temporal_iou(
                        TemporalSegment(candidate.start, candidate.end, 1.0),
                        TemporalSegment(example.start, example.end, 1.0),
                    )
                    for candidate in candidates
                ],
                dtype=np.float32,
            )
            if target_mode == "iou":
                ranked = np.argsort(ious)[::-1]
                selected_positive = ranked[:max_positive_segments]
                remaining = ranked[max_positive_segments:]
                negative_count = min(
                    remaining.size,
                    max(1, selected_positive.size * negative_ratio),
                )
                if negative_count:
                    hard_pool = remaining[ious[remaining] >= negative_iou_threshold]
                    easy_pool = remaining[ious[remaining] < negative_iou_threshold]
                    hard_count = min(hard_pool.size, negative_count // 2)
                    easy_count = negative_count - hard_count
                    chosen_parts = []
                    if hard_count:
                        chosen_parts.append(rng.choice(hard_pool, size=hard_count, replace=False))
                    if easy_count and easy_pool.size:
                        chosen_parts.append(
                            rng.choice(easy_pool, size=min(easy_count, easy_pool.size), replace=False)
                        )
                    selected_negative = (
                        np.concatenate(chosen_parts) if chosen_parts else np.asarray([], dtype=np.int64)
                    )
                else:
                    selected_negative = np.asarray([], dtype=np.int64)
                chosen = np.concatenate([selected_positive, selected_negative]).astype(np.int64)
                if chosen.size:
                    selected_parts.append((ious[chosen], chosen.tolist(), candidates, embeddings, _video_duration(windows)))
            else:
                positive_idx = np.flatnonzero(ious >= positive_iou_threshold)
                negative_idx = np.flatnonzero(ious <= negative_iou_threshold)
                if positive_idx.size:
                    ranked_positive = positive_idx[np.argsort(ious[positive_idx])[::-1]]
                    selected_positive = ranked_positive[:max_positive_segments]
                    selected_parts.append((np.ones(selected_positive.size, dtype=np.float32), selected_positive.tolist(), candidates, embeddings, _video_duration(windows)))
                if negative_idx.size and positive_idx.size:
                    negative_count = min(negative_idx.size, max(1, min(max_positive_segments, positive_idx.size) * negative_ratio))
                    selected_negative = rng.choice(negative_idx, size=negative_count, replace=False)
                    selected_parts.append((np.zeros(selected_negative.size, dtype=np.float32), selected_negative.tolist(), candidates, embeddings, _video_duration(windows)))

        for labels, indices, candidates, embeddings, video_duration in selected_parts:
            selected_candidates = [candidates[idx] for idx in indices]
            selected_embeddings = segment_embeddings(embeddings, selected_candidates)
            selected_text = text_embedding.unsqueeze(0).expand(selected_embeddings.shape[0], -1)
            selected_features = segment_features(
                selected_candidates,
                video_duration=video_duration,
                embeddings=embeddings,
                text_embedding=text_embedding,
                use_score_features=use_score_features,
            )
            selected_labels = torch.as_tensor(labels, dtype=torch.float32)
            video_rows.append(selected_embeddings)
            text_rows.append(selected_text)
            feature_rows.append(selected_features)
            label_rows.append(selected_labels)

    if not video_rows:
        empty = torch.empty(0, 512)
        return empty, empty, torch.empty(0, 4), torch.empty(0)
    return torch.cat(video_rows), torch.cat(text_rows), torch.cat(feature_rows), torch.cat(label_rows)


def build_hard_negative_training_pairs(
    examples: list[VideoCaptionExample],
    *,
    base_scorer: ProjectionScorer,
    embedder: XCLIPEmbedder,
    scales: tuple[float, ...],
    strides: tuple[float, ...],
    embedding_cache_dir: Path | None,
    frames_per_clip: int,
    span_window_counts: tuple[int, ...],
    max_positive_segments: int,
    max_hard_negatives: int,
    max_broad_negatives: int,
    hard_negative_iou_threshold: float,
    broad_negative_iou_threshold: float,
    broad_negative_duration_ratio: float,
    target_mode: str,
    use_score_features: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    video_cache: dict[tuple[Path, float, float], tuple[list[VideoWindow], torch.Tensor]] = {}
    candidate_cache: dict[tuple[Path, float, float], list[SegmentCandidate]] = {}
    text_cache: dict[str, torch.Tensor] = {}
    video_rows: list[torch.Tensor] = []
    text_rows: list[torch.Tensor] = []
    feature_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []

    for example in tqdm(examples, desc="Building duration hard-negative pairs"):
        if example.video_path is None:
            continue
        if example.caption not in text_cache:
            text_cache[example.caption] = embedder.encode_text(example.caption)[0]
        text_embedding = text_cache[example.caption]
        target = TemporalSegment(example.start, example.end, 1.0)
        target_duration = max(target.duration, 1e-6)

        rows: list[tuple[SegmentCandidate, torch.Tensor, torch.Tensor, float, float]] = []
        for scale, stride in zip(scales, strides, strict=True):
            cache_key = (example.video_path, float(scale), float(stride))
            if cache_key not in video_cache:
                video_cache[cache_key] = _load_or_encode_video_windows(
                    example.video_path,
                    embedder=embedder,
                    clip_seconds=scale,
                    stride_seconds=stride,
                    frames_per_clip=frames_per_clip,
                    embedding_cache_dir=embedding_cache_dir,
                )
            windows, embeddings = video_cache[cache_key]
            if cache_key not in candidate_cache:
                candidate_cache[cache_key] = generate_segment_candidates(
                    windows,
                    scale=scale,
                    span_window_counts=span_window_counts,
                )
            candidates = candidate_cache[cache_key]
            if not candidates:
                continue
            video_duration = _video_duration(windows)
            candidate_embeddings = segment_embeddings(embeddings, candidates)
            candidate_features = segment_features(
                candidates,
                video_duration=video_duration,
                embeddings=embeddings,
                text_embedding=text_embedding,
                use_score_features=use_score_features,
            )
            base_scores = base_scorer.predict_proba(
                candidate_embeddings.to(device),
                text_embedding.to(device),
                candidate_features.to(device),
            ).cpu()
            for candidate, embedding, features, score in zip(
                candidates,
                candidate_embeddings,
                candidate_features,
                base_scores,
                strict=True,
            ):
                iou = temporal_iou(TemporalSegment(candidate.start, candidate.end, 1.0), target)
                rows.append((candidate, embedding, features, float(score), float(iou)))

        if not rows:
            continue

        positive_order = np.argsort([row[4] for row in rows])[::-1][:max_positive_segments]
        hard_negative_pool = [
            idx
            for idx, row in enumerate(rows)
            if row[4] <= hard_negative_iou_threshold
        ]
        hard_negative_order = sorted(
            hard_negative_pool,
            key=lambda idx: rows[idx][3],
            reverse=True,
        )[:max_hard_negatives]
        broad_negative_pool = [
            idx
            for idx, row in enumerate(rows)
            if row[4] < broad_negative_iou_threshold
            and row[0].duration / target_duration >= broad_negative_duration_ratio
        ]
        broad_negative_order = sorted(
            broad_negative_pool,
            key=lambda idx: (rows[idx][3], rows[idx][0].duration / target_duration),
            reverse=True,
        )[:max_broad_negatives]

        selected_indices = list(dict.fromkeys([*positive_order, *hard_negative_order, *broad_negative_order]))
        selected_embeddings: list[torch.Tensor] = []
        selected_text: list[torch.Tensor] = []
        selected_features: list[torch.Tensor] = []
        selected_labels: list[float] = []
        for idx in selected_indices:
            candidate, embedding, features, _score, iou = rows[int(idx)]
            if idx in hard_negative_order or idx in broad_negative_order:
                label = 0.0
            elif target_mode == "iou":
                label = float(iou)
            else:
                label = 1.0
            selected_embeddings.append(embedding)
            selected_text.append(text_embedding)
            selected_features.append(features)
            selected_labels.append(label)

        if selected_embeddings:
            video_rows.append(torch.stack(selected_embeddings))
            text_rows.append(torch.stack(selected_text))
            feature_rows.append(torch.stack(selected_features))
            label_rows.append(torch.as_tensor(selected_labels, dtype=torch.float32))

    if not video_rows:
        empty = torch.empty(0, 512)
        return empty, empty, torch.empty(0, 8 if use_score_features else 4), torch.empty(0)
    return torch.cat(video_rows), torch.cat(text_rows), torch.cat(feature_rows), torch.cat(label_rows)


def evaluate_segment_reranker(
    examples: list[VideoCaptionExample],
    *,
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
    scorer = load_scorer(checkpoint, map_location=model_device).to(model_device)
    scorer.eval()

    video_cache: dict[tuple[Path, float, float], tuple[list[VideoWindow], torch.Tensor]] = {}
    candidate_cache: dict[tuple[Path, float, float], list[SegmentCandidate]] = {}
    text_cache: dict[str, torch.Tensor] = {}
    predictions: list[list[TemporalSegment]] = []
    targets: list[TemporalSegment] = []
    skipped = 0

    output_handle = None
    if predictions_out is not None:
        predictions_out.parent.mkdir(parents=True, exist_ok=True)
        output_handle = predictions_out.open("w", encoding="utf-8")

    try:
        for example in tqdm(selected, desc="Evaluating segment reranker"):
            if example.video_path is None:
                skipped += 1
                continue
            if example.caption not in text_cache:
                text_cache[example.caption] = embedder.encode_text(example.caption)[0]
            text_embedding = text_cache[example.caption].to(model_device)

            scored: list[TemporalSegment] = []
            for scale, stride in zip(scales, strides, strict=True):
                cache_key = (example.video_path, float(scale), float(stride))
                if cache_key not in video_cache:
                    video_cache[cache_key] = _load_or_encode_video_windows(
                        example.video_path,
                        embedder=embedder,
                        clip_seconds=scale,
                        stride_seconds=stride,
                        frames_per_clip=frames_per_clip,
                        embedding_cache_dir=embedding_cache_dir,
                    )
                windows, embeddings = video_cache[cache_key]
                if cache_key not in candidate_cache:
                    candidate_cache[cache_key] = generate_segment_candidates(
                        windows,
                        scale=scale,
                        span_window_counts=span_window_counts,
                    )
                candidates = candidate_cache[cache_key]
                video_duration = _video_duration(windows)
                use_score_features = scorer.extra_feature_dim > 4
                scored.extend(
                    score_candidates(
                        scorer,
                        embeddings,
                        text_embedding,
                        candidates,
                        video_duration=video_duration,
                        text_embedding_cpu=text_cache[example.caption],
                        use_score_features=use_score_features,
                        device=model_device,
                    )
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


def evaluate_boundary_refined_reranker(
    examples: list[VideoCaptionExample],
    *,
    checkpoint: Path,
    boundary_checkpoint: Path,
    embedder: XCLIPEmbedder,
    scales: tuple[float, ...],
    strides: tuple[float, ...],
    embedding_cache_dir: Path | None,
    frames_per_clip: int,
    span_window_counts: tuple[int, ...],
    iou_thresholds: list[float],
    top_k: int = 5,
    boundary_alpha: float = 1.0,
    max_boundary_offset: float = 1.0,
    limit: int | None = None,
    predictions_out: Path | None = None,
    device: str | None = None,
) -> SegmentEvaluationResult:
    selected = examples if limit is None else examples[:limit]
    model_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    scorer = load_scorer(checkpoint, map_location=model_device).to(model_device)
    scorer.eval()
    refiner = load_boundary_refiner(boundary_checkpoint, map_location=model_device).to(model_device)
    refiner.eval()

    video_cache: dict[tuple[Path, float, float], tuple[list[VideoWindow], torch.Tensor]] = {}
    candidate_cache: dict[tuple[Path, float, float], list[SegmentCandidate]] = {}
    text_cache: dict[str, torch.Tensor] = {}
    predictions: list[list[TemporalSegment]] = []
    targets: list[TemporalSegment] = []
    skipped = 0

    output_handle = None
    if predictions_out is not None:
        predictions_out.parent.mkdir(parents=True, exist_ok=True)
        output_handle = predictions_out.open("w", encoding="utf-8")

    try:
        for example in tqdm(selected, desc="Evaluating boundary-refined reranker"):
            if example.video_path is None:
                skipped += 1
                continue
            if example.caption not in text_cache:
                text_cache[example.caption] = embedder.encode_text(example.caption)[0]
            text_embedding = text_cache[example.caption].to(model_device)

            scored: list[TemporalSegment] = []
            for scale, stride in zip(scales, strides, strict=True):
                cache_key = (example.video_path, float(scale), float(stride))
                if cache_key not in video_cache:
                    video_cache[cache_key] = _load_or_encode_video_windows(
                        example.video_path,
                        embedder=embedder,
                        clip_seconds=scale,
                        stride_seconds=stride,
                        frames_per_clip=frames_per_clip,
                        embedding_cache_dir=embedding_cache_dir,
                    )
                windows, embeddings = video_cache[cache_key]
                if cache_key not in candidate_cache:
                    candidate_cache[cache_key] = generate_segment_candidates(
                        windows,
                        scale=scale,
                        span_window_counts=span_window_counts,
                    )
                candidates = candidate_cache[cache_key]
                video_duration = _video_duration(windows)
                use_score_features = scorer.extra_feature_dim > 4
                scored.extend(
                    score_refined_candidates(
                        scorer,
                        refiner,
                        embeddings,
                        text_embedding,
                        candidates,
                        video_duration=video_duration,
                        text_embedding_cpu=text_cache[example.caption],
                        use_score_features=use_score_features,
                        boundary_alpha=boundary_alpha,
                        max_boundary_offset=max_boundary_offset,
                        device=model_device,
                    )
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


def generate_segment_candidates(
    windows: list[VideoWindow],
    *,
    scale: float,
    span_window_counts: tuple[int, ...],
) -> list[SegmentCandidate]:
    candidates: list[SegmentCandidate] = []
    n = len(windows)
    for count in sorted(set(span_window_counts)):
        if count <= 0 or count > n:
            continue
        for start_idx in range(0, n - count + 1):
            end_idx = start_idx + count - 1
            candidates.append(
                SegmentCandidate(
                    scale=scale,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start=windows[start_idx].start,
                    end=windows[end_idx].end,
                )
            )
    return candidates


def segment_embeddings(
    embeddings: torch.Tensor, candidates: list[SegmentCandidate]
) -> torch.Tensor:
    if not candidates:
        return torch.empty(0, embeddings.shape[-1])
    prefix = torch.cat([torch.zeros(1, embeddings.shape[-1]), embeddings.float().cumsum(dim=0)])
    rows = []
    for candidate in candidates:
        total = prefix[candidate.end_idx + 1] - prefix[candidate.start_idx]
        rows.append(total / (candidate.end_idx - candidate.start_idx + 1))
    return torch.stack(rows, dim=0)


def segment_features(
    candidates: list[SegmentCandidate],
    *,
    video_duration: float,
    embeddings: torch.Tensor | None = None,
    text_embedding: torch.Tensor | None = None,
    use_score_features: bool = False,
) -> torch.Tensor:
    feature_dim = 8 if use_score_features else 4
    if not candidates:
        return torch.empty(0, feature_dim)
    safe_duration = max(video_duration, 1e-6)
    window_scores = None
    if use_score_features:
        if embeddings is None or text_embedding is None:
            raise ValueError("embeddings and text_embedding are required for score features")
        window_scores = F.normalize(embeddings.float(), dim=-1) @ F.normalize(
            text_embedding.float().cpu(), dim=-1
        )
    rows = []
    for candidate in candidates:
        duration = max(candidate.duration, 1e-6)
        center = (candidate.start + candidate.end) / 2.0
        row = [
            np.log2(duration) / 8.0,
            candidate.start / safe_duration,
            candidate.end / safe_duration,
            center / safe_duration,
        ]
        if window_scores is not None:
            values = window_scores[candidate.start_idx : candidate.end_idx + 1]
            row.extend(
                [
                    float(values.mean()),
                    float(values.max()),
                    float(values.min()),
                    float(values.std(unbiased=False)),
                ]
            )
        rows.append(row)
    return torch.as_tensor(rows, dtype=torch.float32)


def score_candidates(
    scorer: ProjectionScorer,
    embeddings: torch.Tensor,
    text_embedding: torch.Tensor,
    candidates: list[SegmentCandidate],
    *,
    video_duration: float,
    text_embedding_cpu: torch.Tensor,
    use_score_features: bool,
    device: torch.device,
    batch_size: int = 2048,
) -> list[TemporalSegment]:
    scored: list[TemporalSegment] = []
    for start in range(0, len(candidates), batch_size):
        batch_candidates = candidates[start : start + batch_size]
        batch_embeddings = segment_embeddings(embeddings, batch_candidates).to(device)
        batch_features = segment_features(
            batch_candidates,
            video_duration=video_duration,
            embeddings=embeddings,
            text_embedding=text_embedding_cpu,
            use_score_features=use_score_features,
        ).to(device)
        scores = scorer.predict_proba(batch_embeddings, text_embedding, batch_features).cpu()
        for candidate, score in zip(batch_candidates, scores, strict=True):
            scored.append(TemporalSegment(candidate.start, candidate.end, float(score)))
    return scored


def score_refined_candidates(
    scorer: ProjectionScorer,
    refiner: BoundaryRefiner,
    embeddings: torch.Tensor,
    text_embedding: torch.Tensor,
    candidates: list[SegmentCandidate],
    *,
    video_duration: float,
    text_embedding_cpu: torch.Tensor,
    use_score_features: bool,
    boundary_alpha: float,
    max_boundary_offset: float,
    device: torch.device,
    batch_size: int = 2048,
) -> list[TemporalSegment]:
    scored: list[TemporalSegment] = []
    for start in range(0, len(candidates), batch_size):
        batch_candidates = candidates[start : start + batch_size]
        batch_embeddings = segment_embeddings(embeddings, batch_candidates).to(device)
        batch_features = segment_features(
            batch_candidates,
            video_duration=video_duration,
            embeddings=embeddings,
            text_embedding=text_embedding_cpu,
            use_score_features=use_score_features,
        ).to(device)
        scores = scorer.predict_proba(batch_embeddings, text_embedding, batch_features).cpu()
        with torch.no_grad():
            offsets = refiner(batch_embeddings, text_embedding, batch_features).cpu()
        for candidate, score, offset in zip(batch_candidates, scores, offsets, strict=True):
            duration = max(candidate.duration, 1e-6)
            limit = max(0.0, float(max_boundary_offset))
            alpha = max(0.0, min(1.0, float(boundary_alpha)))
            start_offset = float(torch.clamp(offset[0], -limit, limit))
            end_offset = float(torch.clamp(offset[1], -limit, limit))
            predicted_start = candidate.start + start_offset * duration
            predicted_end = candidate.end + end_offset * duration
            refined_start = candidate.start + alpha * (predicted_start - candidate.start)
            refined_end = candidate.end + alpha * (predicted_end - candidate.end)
            refined_start = max(0.0, min(float(video_duration), refined_start))
            refined_end = max(0.0, min(float(video_duration), refined_end))
            if refined_end <= refined_start:
                refined_start, refined_end = candidate.start, candidate.end
            scored.append(TemporalSegment(refined_start, refined_end, float(score)))
    return scored


def save_boundary_refiner(
    path: Path, model: BoundaryRefiner, *, metadata: dict | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "embedding_dim": model.video_proj[0].in_features,
        "projection_dim": model.video_proj[0].out_features,
        "hidden_dim": model.offset_head[0].out_features,
        "extra_feature_dim": model.extra_feature_dim,
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_boundary_refiner(
    path: Path, *, map_location: str | torch.device = "cpu"
) -> BoundaryRefiner:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = BoundaryRefiner(
        embedding_dim=int(payload["embedding_dim"]),
        projection_dim=int(payload["projection_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        extra_feature_dim=int(payload.get("extra_feature_dim", 0)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _initialize_projection_scorer(model: ProjectionScorer, base_scorer: ProjectionScorer) -> None:
    model.frame_proj.load_state_dict(base_scorer.frame_proj.state_dict())
    model.text_proj.load_state_dict(base_scorer.text_proj.state_dict())
    base_state = base_scorer.state_dict()
    model_state = model.state_dict()
    copied = {}
    for key, value in model_state.items():
        if key in base_state and base_state[key].shape == value.shape:
            copied[key] = base_state[key]
        else:
            copied[key] = value
    model.load_state_dict(copied)


def _video_duration(windows: list[VideoWindow]) -> float:
    return max((window.end for window in windows), default=1.0)


def _prediction_row(
    example: VideoCaptionExample, predictions: list[TemporalSegment], target: TemporalSegment
) -> dict:
    prediction_rows = [
        {
            "start": segment.start,
            "end": segment.end,
            "score": segment.score,
            "iou_with_ground_truth": temporal_iou(segment, target),
        }
        for segment in predictions
    ]
    return {
        "video_id": example.video_id,
        "video_path": str(example.video_path) if example.video_path else None,
        "caption": example.caption,
        "ground_truth": {
            "start": target.start,
            "end": target.end,
            "duration": target.duration,
        },
        "predictions": prediction_rows,
        "best_iou": max((item["iou_with_ground_truth"] for item in prediction_rows), default=0.0),
    }
