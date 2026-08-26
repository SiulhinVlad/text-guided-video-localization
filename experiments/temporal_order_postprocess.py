from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from metrics import summarize_recall
from segments import TemporalSegment, temporal_iou


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--penalties", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--order-by", choices=["center", "start", "end"], default="center")
    args = parser.parse_args()

    rows = read_rows(args.predictions)
    baseline_top1 = evaluate_rows(rows, mode="top1", thresholds=args.thresholds)
    baseline_topk = evaluate_rows(rows, mode="topk", thresholds=args.thresholds)

    penalty_results = []
    best_rows = None
    best_score = -1.0
    for penalty in args.penalties:
        constrained_rows = apply_temporal_order(
            rows,
            penalty=penalty,
            order_by=args.order_by,
        )
        metrics = evaluate_rows(constrained_rows, mode="top1", thresholds=args.thresholds)
        penalty_results.append({"penalty": penalty, "metrics": metrics})
        if metrics["mean_best_iou"] > best_score:
            best_score = metrics["mean_best_iou"]
            best_rows = constrained_rows

    report = {
        "examples": len(rows),
        "videos": len({row["video_id"] for row in rows}),
        "baseline_top1": baseline_top1,
        "baseline_topk": baseline_topk,
        "constrained_top1_by_penalty": penalty_results,
    }
    print(json.dumps(report, indent=2))

    if args.output is not None and best_rows is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for row in best_rows:
                handle.write(json.dumps(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def evaluate_rows(rows: list[dict], *, mode: str, thresholds: list[float]) -> dict[str, float]:
    prediction_sets: list[list[TemporalSegment]] = []
    targets: list[TemporalSegment] = []
    for row in rows:
        predictions = row["predictions"] if mode == "topk" else row["predictions"][:1]
        prediction_sets.append([segment_from_row(prediction) for prediction in predictions])
        targets.append(
            TemporalSegment(
                float(row["ground_truth"]["start"]),
                float(row["ground_truth"]["end"]),
                1.0,
            )
        )
    return summarize_recall(prediction_sets, targets, thresholds)


def apply_temporal_order(rows: list[dict], *, penalty: float, order_by: str) -> list[dict]:
    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["video_id"]].append((index, row))

    output = [dict(row) for row in rows]
    for group in grouped.values():
        ordered_rows = [row for _, row in group]
        selected = select_ordered_predictions(ordered_rows, penalty=penalty, order_by=order_by)
        for (original_index, _), prediction in zip(group, selected, strict=True):
            updated = dict(rows[original_index])
            updated["predictions"] = [prediction]
            target = TemporalSegment(
                float(updated["ground_truth"]["start"]),
                float(updated["ground_truth"]["end"]),
                1.0,
            )
            updated["best_iou"] = temporal_iou(segment_from_row(prediction), target)
            output[original_index] = updated
    return output


def select_ordered_predictions(rows: list[dict], *, penalty: float, order_by: str) -> list[dict]:
    if not rows:
        return []
    candidates = [row["predictions"] or [{"start": 0.0, "end": 0.0, "score": 0.0}] for row in rows]
    dp: list[list[float]] = []
    back: list[list[int | None]] = []

    first_scores = [float(candidate["score"]) for candidate in candidates[0]]
    dp.append(first_scores)
    back.append([None] * len(first_scores))

    for row_index in range(1, len(candidates)):
        current_dp: list[float] = []
        current_back: list[int | None] = []
        for current in candidates[row_index]:
            current_position = position(current, order_by)
            best_value = float("-inf")
            best_previous = None
            for previous_index, previous in enumerate(candidates[row_index - 1]):
                previous_position = position(previous, order_by)
                violation = max(0.0, previous_position - current_position)
                video_duration = max(
                    1.0,
                    float(rows[row_index - 1]["ground_truth"]["end"]),
                    float(rows[row_index]["ground_truth"]["end"]),
                    previous_position,
                    current_position,
                )
                value = (
                    dp[row_index - 1][previous_index]
                    + float(current["score"])
                    - penalty * violation / video_duration
                )
                if value > best_value:
                    best_value = value
                    best_previous = previous_index
            current_dp.append(best_value)
            current_back.append(best_previous)
        dp.append(current_dp)
        back.append(current_back)

    selected_indices = [0] * len(candidates)
    selected_indices[-1] = max(range(len(dp[-1])), key=lambda idx: dp[-1][idx])
    for row_index in range(len(candidates) - 1, 0, -1):
        previous = back[row_index][selected_indices[row_index]]
        selected_indices[row_index - 1] = 0 if previous is None else previous

    return [
        dict(row_candidates[selected_index])
        for row_candidates, selected_index in zip(candidates, selected_indices, strict=True)
    ]


def position(prediction: dict, order_by: str) -> float:
    start = float(prediction["start"])
    end = float(prediction["end"])
    if order_by == "start":
        return start
    if order_by == "end":
        return end
    return 0.5 * (start + end)


def segment_from_row(row: dict) -> TemporalSegment:
    return TemporalSegment(float(row["start"]), float(row["end"]), float(row.get("score", 1.0)))


if __name__ == "__main__":
    main()
