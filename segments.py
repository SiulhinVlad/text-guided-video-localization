from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class TemporalSegment:
    start: float
    end: float
    score: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def smooth_scores(scores: Sequence[float], window: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.size == 0 or window <= 1:
        return values
    window = min(window, values.size)
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def automatic_threshold(scores: Sequence[float]) -> float:
    values = np.asarray(scores, dtype=np.float32)
    if values.size == 0:
        return 0.0
    mean = float(values.mean())
    std = float(values.std())
    percentile = float(np.quantile(values, 0.75))
    return max(mean + 0.25 * std, percentile)


def extract_segments(
    timestamps: Sequence[float],
    scores: Sequence[float],
    *,
    threshold: float | str = "auto",
    min_duration: float = 1.0,
    merge_gap: float = 1.0,
    smoothing_window: int = 3,
) -> list[TemporalSegment]:
    if len(timestamps) != len(scores):
        raise ValueError("timestamps and scores must have the same length")
    if len(timestamps) == 0:
        return []

    times = np.asarray(timestamps, dtype=np.float32)
    smoothed = smooth_scores(scores, smoothing_window)
    cutoff = automatic_threshold(smoothed) if threshold == "auto" else float(threshold)
    active = smoothed >= cutoff

    raw: list[TemporalSegment] = []
    start_idx: int | None = None
    for idx, is_active in enumerate(active):
        if is_active and start_idx is None:
            start_idx = idx
        elif not is_active and start_idx is not None:
            raw.append(_segment_from_span(times, smoothed, start_idx, idx - 1))
            start_idx = None
    if start_idx is not None:
        raw.append(_segment_from_span(times, smoothed, start_idx, len(active) - 1))

    merged = merge_close_segments(raw, merge_gap=merge_gap)
    return [seg for seg in merged if seg.duration >= min_duration]


def _segment_from_span(
    times: np.ndarray, scores: np.ndarray, start_idx: int, end_idx: int
) -> TemporalSegment:
    step = _infer_step(times)
    start = float(times[start_idx])
    end = float(times[end_idx] + step)
    score = float(scores[start_idx : end_idx + 1].mean())
    return TemporalSegment(start=start, end=end, score=score)


def _infer_step(times: np.ndarray) -> float:
    if times.size < 2:
        return 1.0
    diffs = np.diff(times)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def merge_close_segments(
    segments: Iterable[TemporalSegment], *, merge_gap: float
) -> list[TemporalSegment]:
    ordered = sorted(segments, key=lambda item: item.start)
    if not ordered:
        return []
    merged = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.start - previous.end <= merge_gap:
            total = previous.duration + current.duration
            score = (
                (previous.score * previous.duration + current.score * current.duration) / total
                if total > 0
                else max(previous.score, current.score)
            )
            merged[-1] = TemporalSegment(
                start=previous.start,
                end=max(previous.end, current.end),
                score=float(score),
            )
        else:
            merged.append(current)
    return merged


def temporal_iou(left: TemporalSegment, right: TemporalSegment) -> float:
    inter_start = max(left.start, right.start)
    inter_end = min(left.end, right.end)
    intersection = max(0.0, inter_end - inter_start)
    union = left.duration + right.duration - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def non_max_suppression(
    segments: Sequence[TemporalSegment], *, iou_threshold: float = 0.5, top_k: int | None = None
) -> list[TemporalSegment]:
    ordered = sorted(segments, key=lambda item: item.score, reverse=True)
    kept: list[TemporalSegment] = []
    for candidate in ordered:
        if all(temporal_iou(candidate, selected) < iou_threshold for selected in kept):
            kept.append(candidate)
            if top_k is not None and len(kept) >= top_k:
                break
    return kept
