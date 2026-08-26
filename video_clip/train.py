from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict
import hashlib

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data import VideoCaptionExample
from scoring import ProjectionScorer, save_scorer
from segments import TemporalSegment, temporal_iou
from video_clip.config import VideoClipTrainingConfig
from video_clip.windows import VideoWindow, sample_video_windows
from video_clip.xclip_backend import XCLIPEmbedder


@dataclass
class VideoClipTrainingSummary:
    examples: int
    pairs: int
    positives: int
    negatives: int
    final_loss: float
    checkpoint: Path


def train_projection_scorer(
    examples: list[VideoCaptionExample],
    *,
    output: Path,
    embedder: XCLIPEmbedder,
    config: VideoClipTrainingConfig | None = None,
    clip_seconds: float = 4.0,
    stride_seconds: float = 2.0,
    frames_per_clip: int = 8,
    multi_scales: tuple[float, ...] | None = None,
    multi_strides: tuple[float, ...] | None = None,
    embedding_cache_dir: Path | None = None,
) -> VideoClipTrainingSummary:
    config = config or VideoClipTrainingConfig()
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    video_embeddings, text_embeddings, extra_features, labels = build_training_pairs(
        examples,
        embedder=embedder,
        clip_seconds=clip_seconds,
        stride_seconds=stride_seconds,
        frames_per_clip=frames_per_clip,
        multi_scales=multi_scales,
        multi_strides=multi_strides,
        embedding_cache_dir=embedding_cache_dir,
        negative_ratio=config.negative_ratio,
        positive_iou_threshold=config.positive_iou_threshold,
        positive_window_coverage_threshold=config.positive_window_coverage_threshold,
        max_positive_windows=config.max_positive_windows,
        use_duration_feature=config.use_duration_feature,
    )
    if labels.numel() == 0:
        raise ValueError("No trainable pairs were produced. Check videos and timestamps.")

    model = ProjectionScorer(
        embedding_dim=video_embeddings.shape[-1],
        projection_dim=config.projection_dim,
        hidden_dim=config.hidden_dim,
        extra_feature_dim=extra_features.shape[-1] if extra_features.numel() else 0,
    ).to(device)
    dataset = TensorDataset(
        video_embeddings.float(),
        text_embeddings.float(),
        extra_features.float(),
        labels.float(),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()

    final_loss = 0.0
    model.train()
    for _epoch in range(config.epochs):
        running = 0.0
        seen = 0
        for batch_video, batch_text, batch_extra, batch_labels in loader:
            batch_video = batch_video.to(device)
            batch_text = batch_text.to(device)
            batch_extra = batch_extra.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_video, batch_text, batch_extra if model.extra_feature_dim else None)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * batch_labels.numel()
            seen += batch_labels.numel()
        final_loss = running / max(seen, 1)

    positives = int(labels.sum().item())
    negatives = int(labels.numel() - positives)
    save_scorer(
        output,
        model.cpu(),
        metadata={
            "backend": "xclip-window",
            "epochs": config.epochs,
            "clip_seconds": clip_seconds,
            "stride_seconds": stride_seconds,
            "frames_per_clip": frames_per_clip,
            "multi_scales": list(multi_scales) if multi_scales else None,
            "multi_strides": list(multi_strides) if multi_strides else None,
            "positive_iou_threshold": config.positive_iou_threshold,
            "positive_window_coverage_threshold": config.positive_window_coverage_threshold,
            "use_duration_feature": config.use_duration_feature,
            "examples": len(examples),
            "pairs": int(labels.numel()),
        },
    )
    return VideoClipTrainingSummary(
        examples=len(examples),
        pairs=int(labels.numel()),
        positives=positives,
        negatives=negatives,
        final_loss=final_loss,
        checkpoint=output,
    )


def build_training_pairs(
    examples: list[VideoCaptionExample],
    *,
    embedder: XCLIPEmbedder,
    clip_seconds: float,
    stride_seconds: float,
    frames_per_clip: int,
    multi_scales: tuple[float, ...] | None,
    multi_strides: tuple[float, ...] | None,
    embedding_cache_dir: Path | None,
    negative_ratio: int,
    positive_iou_threshold: float,
    positive_window_coverage_threshold: float,
    max_positive_windows: int | None = None,
    use_duration_feature: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(7)
    scales = multi_scales or (clip_seconds,)
    strides = multi_strides or (stride_seconds,)
    if len(scales) != len(strides):
        raise ValueError("multi_scales and multi_strides must have matching lengths")

    video_cache: OrderedDict[
        tuple[Path, float, float], tuple[list[VideoWindow], torch.Tensor]
    ] = OrderedDict()
    text_cache: dict[str, torch.Tensor] = {}
    video_rows: list[torch.Tensor] = []
    text_rows: list[torch.Tensor] = []
    feature_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []

    for example in tqdm(examples, desc="Building X-CLIP training pairs"):
        if example.video_path is None:
            continue
        if example.caption not in text_cache:
            text_cache[example.caption] = embedder.encode_text(example.caption)[0]
        text_embedding = text_cache[example.caption]

        for scale, scale_stride in zip(scales, strides, strict=True):
            cache_key = (example.video_path, float(scale), float(scale_stride))
            if cache_key not in video_cache:
                windows, embeddings = _load_or_encode_video_windows(
                    example.video_path,
                    embedder=embedder,
                    clip_seconds=scale,
                    stride_seconds=scale_stride,
                    frames_per_clip=frames_per_clip,
                    embedding_cache_dir=embedding_cache_dir,
                )
                video_cache[cache_key] = (windows, embeddings)
                if len(video_cache) > 16:
                    video_cache.popitem(last=False)
            else:
                video_cache.move_to_end(cache_key)

            windows, embeddings = video_cache[cache_key]
            if not windows:
                continue
            labels = _labels_for_windows(
                windows,
                start=example.start,
                end=example.end,
                iou_threshold=positive_iou_threshold,
                window_coverage_threshold=positive_window_coverage_threshold,
            )
            positive_idx = np.flatnonzero(labels)
            negative_idx = np.flatnonzero(~labels)
            if positive_idx.size == 0 or negative_idx.size == 0:
                continue

            if max_positive_windows is not None and positive_idx.size > max_positive_windows:
                positive_idx = rng.choice(positive_idx, size=max_positive_windows, replace=False)

            negative_count = min(negative_idx.size, max(1, positive_idx.size * negative_ratio))
            chosen_negative = rng.choice(negative_idx, size=negative_count, replace=False)
            chosen = np.concatenate([positive_idx, chosen_negative])
            rng.shuffle(chosen)

            selected = embeddings[torch.as_tensor(chosen, dtype=torch.long)]
            selected_text = text_embedding.unsqueeze(0).expand(selected.shape[0], -1)
            selected_features = _features_for_windows(
                [windows[int(idx)] for idx in chosen],
                use_duration_feature=use_duration_feature,
            )
            selected_labels = torch.as_tensor(labels[chosen], dtype=torch.float32)
            video_rows.append(selected)
            text_rows.append(selected_text)
            feature_rows.append(selected_features)
            label_rows.append(selected_labels)

    if not video_rows:
        empty = torch.empty(0, 512)
        return empty, empty, torch.empty(0, 0), torch.empty(0)
    return torch.cat(video_rows), torch.cat(text_rows), torch.cat(feature_rows), torch.cat(label_rows)


def _features_for_windows(
    windows: list[VideoWindow], *, use_duration_feature: bool
) -> torch.Tensor:
    if not use_duration_feature:
        return torch.empty(len(windows), 0)
    durations = np.asarray(
        [max(window.end - window.start, 1e-6) for window in windows], dtype=np.float32
    )
    values = np.log2(durations) / 5.0
    return torch.as_tensor(values[:, None], dtype=torch.float32)


def _load_or_encode_video_windows(
    video_path: Path,
    *,
    embedder: XCLIPEmbedder,
    clip_seconds: float,
    stride_seconds: float,
    frames_per_clip: int,
    embedding_cache_dir: Path | None,
) -> tuple[list[VideoWindow], torch.Tensor]:
    cache_path = (
        _embedding_cache_path(
            embedding_cache_dir,
            video_path=video_path,
            model_name=embedder.model_name,
            clip_seconds=clip_seconds,
            stride_seconds=stride_seconds,
            frames_per_clip=frames_per_clip,
        )
        if embedding_cache_dir is not None
        else None
    )
    if cache_path is not None and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        windows = [
            VideoWindow(frames=[], start=float(start), end=float(end))
            for start, end in payload["windows"]
        ]
        return windows, payload["embeddings"]

    windows_with_frames = sample_video_windows(
        video_path,
        clip_seconds=clip_seconds,
        stride_seconds=stride_seconds,
        frames_per_clip=frames_per_clip,
    )
    embeddings = embedder.encode_videos([window.frames for window in windows_with_frames])
    windows = [
        VideoWindow(frames=[], start=window.start, end=window.end)
        for window in windows_with_frames
    ]
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "windows": [(window.start, window.end) for window in windows],
                "embeddings": embeddings,
            },
            cache_path,
        )
    return windows, embeddings


def _embedding_cache_path(
    cache_dir: Path,
    *,
    video_path: Path,
    model_name: str,
    clip_seconds: float,
    stride_seconds: float,
    frames_per_clip: int,
) -> Path:
    key = "|".join(
        [
            str(video_path.resolve()),
            model_name,
            f"clip={clip_seconds:g}",
            f"stride={stride_seconds:g}",
            f"frames={frames_per_clip}",
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    scale_dir = f"clip{clip_seconds:g}_stride{stride_seconds:g}_frames{frames_per_clip}"
    return cache_dir / scale_dir / f"{video_path.stem}_{digest}.pt"


def _labels_for_windows(
    windows: list[VideoWindow],
    *,
    start: float,
    end: float,
    iou_threshold: float,
    window_coverage_threshold: float,
) -> np.ndarray:
    target = TemporalSegment(start=start, end=end, score=1.0)
    values = []
    for window in windows:
        candidate = TemporalSegment(start=window.start, end=window.end, score=1.0)
        overlap = max(0.0, min(candidate.end, target.end) - max(candidate.start, target.start))
        window_coverage = overlap / candidate.duration if candidate.duration > 0 else 0.0
        values.append(
            temporal_iou(candidate, target) >= iou_threshold
            or window_coverage >= window_coverage_threshold
        )
    return np.asarray(values, dtype=bool)
