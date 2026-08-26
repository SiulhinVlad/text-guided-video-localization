from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from scoring import ProjectionScorer, cosine_scores, load_scorer, load_scorer_metadata
from segments import (
    TemporalSegment,
    automatic_threshold,
    merge_close_segments,
    non_max_suppression,
    smooth_scores,
)
from video_clip.config import VideoClipConfig
from video_clip.train import _load_or_encode_video_windows
from video_clip.windows import VideoWindow
from video_clip.xclip_backend import XCLIPEmbedder


@dataclass
class VideoClipResult:
    query: str
    video_path: Path
    segments: list[TemporalSegment]
    timestamps: list[float]
    scores: list[float]


@dataclass
class WindowScore:
    start: float
    end: float
    score: float
    smoothed_score: float
    active: bool


@dataclass
class VideoClipExplanation:
    query: str
    video_path: Path
    cutoff: float
    windows: list[WindowScore]
    raw_segments: list[TemporalSegment]
    merged_segments: list[TemporalSegment]
    final_segments: list[TemporalSegment]


class VideoClipLocalizer:
    def __init__(
        self,
        config: VideoClipConfig | None = None,
        *,
        embedder: XCLIPEmbedder | None = None,
        scorer: ProjectionScorer | None = None,
    ):
        self.config = config or VideoClipConfig()
        self.embedder = embedder or XCLIPEmbedder(
            self.config.model_name,
            device=self.config.device,
            batch_size=self.config.batch_size,
        )
        self.scorer = scorer
        self.device = torch.device(self.config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.scorer is not None:
            self.scorer.to(self.device)
            self.scorer.eval()
        self._video_cache: dict[tuple[Path, float, float, int], tuple[list[VideoWindow], torch.Tensor]] = {}
        self._text_cache: dict[str, torch.Tensor] = {}

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Path | None, config: VideoClipConfig | None = None
    ) -> "VideoClipLocalizer":
        config = config or VideoClipConfig()
        scorer = None
        if checkpoint:
            metadata = load_scorer_metadata(checkpoint)
            backend = metadata.get("backend")
            if backend != "xclip-window" and "clip_seconds" not in metadata:
                raise ValueError(
                    f"{checkpoint} does not look like an X-CLIP window checkpoint. "
                    "Train one with `python video_clip_cli.py train ...` or omit --checkpoint "
                    "to evaluate raw X-CLIP cosine similarity."
                )
            scorer = load_scorer(checkpoint, map_location=config.device or "cpu")
        return cls(config=config, scorer=scorer)

    def localize(self, video_path: Path, query: str) -> VideoClipResult:
        if self.config.multi_scales:
            return self._localize_multi_scale(video_path, query)
        windows, score_values = self._score_windows(video_path, query)
        segments = _extract_window_segments(
            windows,
            score_values,
            threshold=self.config.score_threshold,
            min_duration=self.config.min_segment_seconds,
            merge_gap=self.config.merge_gap_seconds,
            smoothing_window=self.config.smoothing_window,
        )
        segments = non_max_suppression(segments, iou_threshold=0.5, top_k=self.config.top_k)
        return VideoClipResult(
            query=query,
            video_path=video_path,
            segments=segments,
            timestamps=[window.start for window in windows],
            scores=score_values,
        )

    def _localize_multi_scale(self, video_path: Path, query: str) -> VideoClipResult:
        scales = self.config.multi_scales or ()
        strides = self.config.multi_strides or tuple(max(1.0, scale / 4.0) for scale in scales)
        merge_gaps = self.config.multi_merge_gaps or tuple(max(1.0, scale / 2.0) for scale in scales)
        if len(strides) != len(scales) or len(merge_gaps) != len(scales):
            raise ValueError("multi_scales, multi_strides, and multi_merge_gaps must have matching lengths")

        all_segments: list[TemporalSegment] = []
        all_timestamps: list[float] = []
        all_scores: list[float] = []
        for clip_seconds, stride_seconds, merge_gap in zip(scales, strides, merge_gaps, strict=True):
            windows, score_values = self._score_windows(
                video_path,
                query,
                clip_seconds=clip_seconds,
                stride_seconds=stride_seconds,
            )
            all_timestamps.extend(window.start for window in windows)
            all_scores.extend(score_values)
            all_segments.extend(
                _extract_window_segments(
                    windows,
                    score_values,
                    threshold=self.config.score_threshold,
                    min_duration=self.config.min_segment_seconds,
                    merge_gap=merge_gap,
                    smoothing_window=self.config.smoothing_window,
                )
            )

        segments = non_max_suppression(all_segments, iou_threshold=0.5, top_k=self.config.top_k)
        return VideoClipResult(
            query=query,
            video_path=video_path,
            segments=segments,
            timestamps=all_timestamps,
            scores=all_scores,
        )

    def explain(self, video_path: Path, query: str) -> VideoClipExplanation:
        windows, score_values = self._score_windows(video_path, query)
        smoothed = smooth_scores(score_values, self.config.smoothing_window)
        cutoff = automatic_threshold(smoothed) if self.config.score_threshold == "auto" else float(
            self.config.score_threshold
        )
        active = smoothed >= cutoff
        raw_segments = _segments_from_active_windows(windows, smoothed, active)
        merged_segments = [
            segment
            for segment in merge_close_segments(raw_segments, merge_gap=self.config.merge_gap_seconds)
            if segment.duration >= self.config.min_segment_seconds
        ]
        final_segments = non_max_suppression(
            merged_segments, iou_threshold=0.5, top_k=self.config.top_k
        )
        return VideoClipExplanation(
            query=query,
            video_path=video_path,
            cutoff=float(cutoff),
            windows=[
                WindowScore(
                    start=window.start,
                    end=window.end,
                    score=float(score),
                    smoothed_score=float(smoothed_score),
                    active=bool(is_active),
                )
                for window, score, smoothed_score, is_active in zip(
                    windows, score_values, smoothed, active, strict=True
                )
            ],
            raw_segments=raw_segments,
            merged_segments=merged_segments,
            final_segments=final_segments,
        )

    def _score_windows(
        self,
        video_path: Path,
        query: str,
        *,
        clip_seconds: float | None = None,
        stride_seconds: float | None = None,
    ) -> tuple[list[VideoWindow], list[float]]:
        clip_seconds = self.config.clip_seconds if clip_seconds is None else clip_seconds
        stride_seconds = self.config.stride_seconds if stride_seconds is None else stride_seconds
        windows, video_embeddings = self._get_video_embeddings(
            video_path,
            clip_seconds=clip_seconds,
            stride_seconds=stride_seconds,
            frames_per_clip=self.config.frames_per_clip,
        )
        text_embedding = self._get_text_embedding(query)

        if self.scorer is None:
            video_embeddings = video_embeddings.to(self.device)
            text_embedding = text_embedding.to(self.device)
            scores = cosine_scores(video_embeddings, text_embedding).cpu()
        else:
            video_embeddings = video_embeddings.to(self.device)
            text_embedding = text_embedding.to(self.device)
            extra_features = None
            if self.scorer.extra_feature_dim:
                extra_features = _features_for_windows(windows).to(self.device)
            scores = self.scorer.predict_proba(
                video_embeddings, text_embedding, extra_features
            ).cpu()

        return windows, [float(item) for item in scores]

    def _get_video_embeddings(
        self,
        video_path: Path,
        *,
        clip_seconds: float,
        stride_seconds: float,
        frames_per_clip: int,
    ) -> tuple[list[VideoWindow], torch.Tensor]:
        cache_key = (Path(video_path), float(clip_seconds), float(stride_seconds), int(frames_per_clip))
        if cache_key not in self._video_cache:
            windows, embeddings = _load_or_encode_video_windows(
                Path(video_path),
                embedder=self.embedder,
                clip_seconds=clip_seconds,
                stride_seconds=stride_seconds,
                frames_per_clip=frames_per_clip,
                embedding_cache_dir=Path(self.config.embedding_cache_dir)
                if self.config.embedding_cache_dir
                else None,
            )
            self._video_cache[cache_key] = (windows, embeddings)
        return self._video_cache[cache_key]

    def _get_text_embedding(self, query: str) -> torch.Tensor:
        if query not in self._text_cache:
            self._text_cache[query] = self.embedder.encode_text(query)[0]
        return self._text_cache[query]


def _extract_window_segments(
    windows: list[VideoWindow],
    scores: list[float],
    *,
    threshold: float | str,
    min_duration: float,
    merge_gap: float,
    smoothing_window: int,
) -> list[TemporalSegment]:
    if len(windows) != len(scores):
        raise ValueError("windows and scores must have the same length")
    if not windows:
        return []

    smoothed = smooth_scores(scores, smoothing_window)
    cutoff = automatic_threshold(smoothed) if threshold == "auto" else float(threshold)
    active = smoothed >= cutoff

    raw = _segments_from_active_windows(windows, smoothed, active)

    merged = merge_close_segments(raw, merge_gap=merge_gap)
    return [segment for segment in merged if segment.duration >= min_duration]


def _segments_from_active_windows(
    windows: list[VideoWindow], smoothed: np.ndarray, active: np.ndarray
) -> list[TemporalSegment]:
    raw: list[TemporalSegment] = []
    start_idx: int | None = None
    for idx, is_active in enumerate(active):
        if is_active and start_idx is None:
            start_idx = idx
        elif not is_active and start_idx is not None:
            raw.append(_segment_from_windows(windows, smoothed, start_idx, idx - 1))
            start_idx = None
    if start_idx is not None:
        raw.append(_segment_from_windows(windows, smoothed, start_idx, len(active) - 1))
    return raw


def _features_for_windows(windows: list[VideoWindow]) -> torch.Tensor:
    durations = np.asarray(
        [max(window.end - window.start, 1e-6) for window in windows], dtype=np.float32
    )
    values = np.log2(durations) / 5.0
    return torch.as_tensor(values[:, None], dtype=torch.float32)


def _segment_from_windows(
    windows: list[VideoWindow], scores, start_idx: int, end_idx: int
) -> TemporalSegment:
    return TemporalSegment(
        start=windows[start_idx].start,
        end=windows[end_idx].end,
        score=float(scores[start_idx : end_idx + 1].mean()),
    )
