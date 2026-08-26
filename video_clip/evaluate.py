from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from data import VideoCaptionExample
from metrics import summarize_recall
from segments import TemporalSegment, temporal_iou
from video_clip.pipeline import VideoClipLocalizer
from tqdm import tqdm


@dataclass
class VideoClipEvaluationResult:
    metrics: dict[str, float]
    evaluated: int
    skipped: int


def evaluate_examples(
    examples: list[VideoCaptionExample],
    *,
    localizer: VideoClipLocalizer,
    iou_thresholds: list[float],
    limit: int | None = None,
    predictions_out: Path | None = None,
    resume: bool = False,
) -> VideoClipEvaluationResult:
    predictions: list[list[TemporalSegment]] = []
    targets: list[TemporalSegment] = []
    skipped = 0

    selected = examples if limit is None else examples[:limit]
    skipped_existing = 0
    output_handle = None
    if predictions_out is not None:
        predictions_out.parent.mkdir(parents=True, exist_ok=True)
        if resume and predictions_out.exists():
            existing_predictions, existing_targets = _read_prediction_rows(predictions_out)
            predictions.extend(existing_predictions)
            targets.extend(existing_targets)
            skipped_existing = len(existing_targets)
            selected = selected[skipped_existing:]
            output_handle = predictions_out.open("a", encoding="utf-8")
        else:
            output_handle = predictions_out.open("w", encoding="utf-8")

    try:
        for example in tqdm(selected, desc="Evaluating X-CLIP", initial=skipped_existing, total=len(selected) + skipped_existing):
            if example.video_path is None:
                skipped += 1
                continue
            result = localizer.localize(example.video_path, example.caption)
            target = TemporalSegment(example.start, example.end, 1.0)
            predictions.append(result.segments)
            targets.append(target)

            if output_handle is not None:
                output_handle.write(json.dumps(_prediction_row(example, result.segments, target)) + "\n")
    finally:
        if output_handle is not None:
            output_handle.close()

    return VideoClipEvaluationResult(
        metrics=summarize_recall(predictions, targets, iou_thresholds),
        evaluated=len(targets),
        skipped=skipped,
    )


def _count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _read_prediction_rows(path: Path) -> tuple[list[list[TemporalSegment]], list[TemporalSegment]]:
    predictions: list[list[TemporalSegment]] = []
    targets: list[TemporalSegment] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            predictions.append(
                [
                    TemporalSegment(
                        start=float(item["start"]),
                        end=float(item["end"]),
                        score=float(item["score"]),
                    )
                    for item in row.get("predictions", [])
                ]
            )
            target = row["ground_truth"]
            targets.append(
                TemporalSegment(
                    start=float(target["start"]),
                    end=float(target["end"]),
                    score=1.0,
                )
            )
    return predictions, targets


def _prediction_row(
    example: VideoCaptionExample, predictions: list[TemporalSegment], target: TemporalSegment
) -> dict:
    prediction_rows = []
    for segment in predictions:
        prediction_rows.append(
            {
                "start": segment.start,
                "end": segment.end,
                "score": segment.score,
                "iou_with_ground_truth": temporal_iou(segment, target),
            }
        )

    best_iou = max((item["iou_with_ground_truth"] for item in prediction_rows), default=0.0)
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
        "best_iou": best_iou,
    }
