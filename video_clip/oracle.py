from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data import VideoCaptionExample
from metrics import summarize_recall
from segments import TemporalSegment, temporal_iou
from tqdm import tqdm
from video_clip.windows import sample_video_window_spans


@dataclass
class OracleEvaluationResult:
    metrics: dict[str, float]
    evaluated: int
    skipped: int


def evaluate_oracle_proposals(
    examples: list[VideoCaptionExample],
    *,
    scales: tuple[float, ...],
    strides: tuple[float, ...],
    iou_thresholds: list[float],
    limit: int | None = None,
    merged: bool = True,
) -> OracleEvaluationResult:
    if len(scales) != len(strides):
        raise ValueError("scales and strides must have matching lengths")

    selected = examples if limit is None else examples[:limit]
    span_cache: dict[tuple[Path, float, float], list[tuple[float, float]]] = {}
    predictions: list[list[TemporalSegment]] = []
    targets: list[TemporalSegment] = []
    skipped = 0

    for example in tqdm(selected, desc="Oracle proposals"):
        if example.video_path is None:
            skipped += 1
            continue
        target = TemporalSegment(example.start, example.end, 1.0)
        best_segment: TemporalSegment | None = None
        best_score = 0.0

        for scale, stride in zip(scales, strides, strict=True):
            cache_key = (example.video_path, float(scale), float(stride))
            if cache_key not in span_cache:
                span_cache[cache_key] = sample_video_window_spans(
                    example.video_path,
                    clip_seconds=scale,
                    stride_seconds=stride,
                )
            spans = span_cache[cache_key]
            candidates = _merged_candidates(spans) if merged else spans
            for start, end in candidates:
                candidate = TemporalSegment(start=start, end=end, score=1.0)
                score = temporal_iou(candidate, target)
                if score > best_score:
                    best_score = score
                    best_segment = candidate

        predictions.append([best_segment] if best_segment is not None else [])
        targets.append(target)

    return OracleEvaluationResult(
        metrics=summarize_recall(predictions, targets, iou_thresholds),
        evaluated=len(targets),
        skipped=skipped,
    )


def _merged_candidates(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    candidates: list[tuple[float, float]] = []
    for start_idx, (start, _end) in enumerate(spans):
        for _end_idx, (_candidate_start, end) in enumerate(spans[start_idx:], start=start_idx):
            candidates.append((start, end))
    return candidates
