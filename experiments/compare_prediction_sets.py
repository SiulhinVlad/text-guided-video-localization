from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    left_rows = read_rows(args.left)
    right_rows = read_rows(args.right)
    if len(left_rows) != len(right_rows):
        raise ValueError(f"Prediction files have different row counts: {len(left_rows)} vs {len(right_rows)}")

    deltas = []
    left_wins = []
    right_wins = []
    both_hit = both_miss = left_only = right_only = 0
    for left, right in zip(left_rows, right_rows, strict=True):
        if left["video_id"] != right["video_id"] or left["caption"] != right["caption"]:
            raise ValueError("Prediction files are not aligned")
        left_iou = best_iou(left)
        right_iou = best_iou(right)
        delta = right_iou - left_iou
        deltas.append(delta)
        if left_iou >= args.threshold and right_iou >= args.threshold:
            both_hit += 1
        elif left_iou < args.threshold <= right_iou:
            right_only += 1
            right_wins.append((delta, left, right))
        elif right_iou < args.threshold <= left_iou:
            left_only += 1
            left_wins.append((delta, left, right))
        else:
            both_miss += 1

    left_wins.sort(key=lambda item: item[0])
    right_wins.sort(key=lambda item: item[0], reverse=True)
    report = {
        "count": len(left_rows),
        "threshold": args.threshold,
        "left": args.left_name,
        "right": args.right_name,
        "mean_delta_right_minus_left": sum(deltas) / len(deltas) if deltas else 0.0,
        "right_better_count": sum(delta > 1e-9 for delta in deltas),
        "left_better_count": sum(delta < -1e-9 for delta in deltas),
        "same_count": sum(abs(delta) <= 1e-9 for delta in deltas),
        "both_hit": both_hit,
        "left_only": left_only,
        "right_only": right_only,
        "both_miss": both_miss,
        "right_only_examples": [summary(left, right) for _, left, right in right_wins[:10]],
        "left_only_examples": [summary(left, right) for _, left, right in left_wins[:10]],
    }
    print(json.dumps(report, indent=2))


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def best_iou(row: dict) -> float:
    predictions = row.get("predictions") or []
    return max((float(item.get("iou_with_ground_truth", 0.0)) for item in predictions), default=0.0)


def summary(left: dict, right: dict) -> dict:
    return {
        "video_id": left["video_id"],
        "caption": left["caption"],
        "ground_truth": left["ground_truth"],
        "left_best_iou": best_iou(left),
        "right_best_iou": best_iou(right),
        "left_top1": minimal_prediction(left),
        "right_top1": minimal_prediction(right),
    }


def minimal_prediction(row: dict) -> dict | None:
    predictions = row.get("predictions") or []
    if not predictions:
        return None
    first = predictions[0]
    return {
        "start": first["start"],
        "end": first["end"],
        "score": first.get("score"),
        "iou": first.get("iou_with_ground_truth"),
    }


if __name__ == "__main__":
    main()
