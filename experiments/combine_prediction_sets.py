from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics import summarize_recall
from segments import TemporalSegment, temporal_iou


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--secondary-quota", type=int, default=2)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    args = parser.parse_args()

    primary_rows = read_rows(args.primary)
    secondary_rows = read_rows(args.secondary)
    combined_rows = []
    prediction_sets = []
    targets = []
    for primary, secondary in zip(primary_rows, secondary_rows, strict=True):
        if primary["video_id"] != secondary["video_id"] or primary["caption"] != secondary["caption"]:
            raise ValueError("Prediction files are not aligned")
        predictions = combine_predictions(
            primary.get("predictions") or [],
            secondary.get("predictions") or [],
            top_k=args.top_k,
            secondary_quota=args.secondary_quota,
            nms_iou=args.nms_iou,
        )
        row = dict(primary)
        target = TemporalSegment(
            float(row["ground_truth"]["start"]),
            float(row["ground_truth"]["end"]),
            1.0,
        )
        row["predictions"] = [prediction_to_row(prediction, target) for prediction in predictions]
        row["best_iou"] = max((item["iou_with_ground_truth"] for item in row["predictions"]), default=0.0)
        combined_rows.append(row)
        prediction_sets.append(predictions)
        targets.append(target)

    print(
        json.dumps(
            {
                "metrics": summarize_recall(prediction_sets, targets, args.thresholds),
                "evaluated": len(targets),
            },
            indent=2,
        )
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for row in combined_rows:
                handle.write(json.dumps(row) + "\n")


def combine_predictions(
    primary_rows: list[dict],
    secondary_rows: list[dict],
    *,
    top_k: int,
    secondary_quota: int,
    nms_iou: float,
) -> list[TemporalSegment]:
    primary = [segment_from_row(row, score=1.0 / (idx + 1)) for idx, row in enumerate(primary_rows)]
    secondary = [
        segment_from_row(row, score=1.0 / (idx + 1) - 0.01)
        for idx, row in enumerate(secondary_rows[:secondary_quota])
    ]
    selected: list[TemporalSegment] = []
    for candidate in [*primary[: max(0, top_k - secondary_quota)], *secondary, *primary, *secondary]:
        if all(temporal_iou(candidate, existing) < nms_iou for existing in selected):
            selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def segment_from_row(row: dict, *, score: float) -> TemporalSegment:
    return TemporalSegment(float(row["start"]), float(row["end"]), score)


def prediction_to_row(segment: TemporalSegment, target: TemporalSegment) -> dict:
    return {
        "start": segment.start,
        "end": segment.end,
        "score": segment.score,
        "iou_with_ground_truth": temporal_iou(segment, target),
    }


if __name__ == "__main__":
    main()
