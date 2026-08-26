from __future__ import annotations

from collections.abc import Iterable, Sequence

from segments import TemporalSegment, temporal_iou


def best_iou(predictions: Sequence[TemporalSegment], target: TemporalSegment) -> float:
    if not predictions:
        return 0.0
    return max(temporal_iou(prediction, target) for prediction in predictions)


def recall_at_iou(
    prediction_sets: Iterable[Sequence[TemporalSegment]],
    targets: Iterable[TemporalSegment],
    *,
    threshold: float,
) -> float:
    total = 0
    hits = 0
    for predictions, target in zip(prediction_sets, targets, strict=True):
        total += 1
        hits += best_iou(predictions, target) >= threshold
    return hits / total if total else 0.0


def mean_best_iou(
    prediction_sets: Iterable[Sequence[TemporalSegment]], targets: Iterable[TemporalSegment]
) -> float:
    values = [best_iou(predictions, target) for predictions, target in zip(prediction_sets, targets, strict=True)]
    return sum(values) / len(values) if values else 0.0


def summarize_recall(
    prediction_sets: Sequence[Sequence[TemporalSegment]],
    targets: Sequence[TemporalSegment],
    thresholds: Sequence[float],
) -> dict[str, float]:
    metrics = {"mean_best_iou": mean_best_iou(prediction_sets, targets)}
    for threshold in thresholds:
        metrics[f"recall@iou={threshold:g}"] = recall_at_iou(
            prediction_sets, targets, threshold=threshold
        )
    return metrics
