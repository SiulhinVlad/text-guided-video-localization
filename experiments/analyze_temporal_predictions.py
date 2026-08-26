from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.7])
    parser.add_argument("--iou-thresholds", type=float, nargs="+", dest="thresholds")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--examples-out", type=Path)
    parser.add_argument("--top-examples", type=int, default=20)
    args = parser.parse_args()

    rows = read_rows(args.predictions)
    if args.top_k is not None:
        rows = limit_predictions(rows, top_k=args.top_k)
    report = analyze(rows, thresholds=args.thresholds, top_examples=args.top_examples)
    print(json.dumps(report, indent=2))

    if args.examples_out is not None:
        args.examples_out.parent.mkdir(parents=True, exist_ok=True)
        with args.examples_out.open("w", encoding="utf-8") as handle:
            json.dump(report["examples"], handle, indent=2)


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def limit_predictions(rows: list[dict], *, top_k: int) -> list[dict]:
    limited = []
    for row in rows:
        copied = dict(row)
        copied["predictions"] = list(row.get("predictions", []))[:top_k]
        copied["best_iou"] = best_iou(copied)
        limited.append(copied)
    return limited


def analyze(rows: list[dict], *, thresholds: list[float], top_examples: int) -> dict:
    top1_ious = [top1_iou(row) for row in rows]
    best_ious = [best_iou(row) for row in rows]
    rank_counts = Counter(best_rank(row) for row in rows)

    threshold_reports = {
        str(threshold): threshold_breakdown(rows, threshold=threshold)
        for threshold in thresholds
    }
    duration_reports = {
        name: summarize_group(group)
        for name, group in group_by_duration(rows).items()
    }
    caption_count_reports = {
        str(count): summarize_group(group)
        for count, group in group_by_caption_count(rows).items()
    }

    examples = {
        "ranking_gap_at_0.5": ranking_gap_examples(rows, threshold=0.5, limit=top_examples),
        "ranking_gap_at_0.7": ranking_gap_examples(rows, threshold=0.7, limit=top_examples),
        "misses_at_0.5": miss_examples(rows, threshold=0.5, limit=top_examples),
        "boundary_near_misses": boundary_near_misses(rows, limit=top_examples),
        "worst_top1": worst_top1_examples(rows, limit=top_examples),
    }

    return {
        "count": len(rows),
        "videos": len({row["video_id"] for row in rows}),
        "overall": {
            "top1_mean_iou": safe_mean(top1_ious),
            "top1_median_iou": safe_median(top1_ious),
            "topk_mean_iou": safe_mean(best_ious),
            "topk_median_iou": safe_median(best_ious),
            "mean_topk_minus_top1": safe_mean([b - t for b, t in zip(best_ious, top1_ious, strict=True)]),
        },
        "best_candidate_rank": dict(sorted(rank_counts.items(), key=lambda item: str(item[0]))),
        "thresholds": threshold_reports,
        "ground_truth_duration_bins": duration_reports,
        "captions_per_video": caption_count_reports,
        "prediction_duration_ratio": duration_ratio_summary(rows),
        "examples": examples,
    }


def threshold_breakdown(rows: list[dict], *, threshold: float) -> dict:
    top1_hit = sum(top1_iou(row) >= threshold for row in rows)
    topk_hit = sum(best_iou(row) >= threshold for row in rows)
    ranking_gap = sum(top1_iou(row) < threshold <= best_iou(row) for row in rows)
    candidate_miss = sum(best_iou(row) < threshold for row in rows)
    return {
        "top1_recall": top1_hit / len(rows) if rows else 0.0,
        "topk_recall": topk_hit / len(rows) if rows else 0.0,
        "ranking_gap_count": ranking_gap,
        "ranking_gap_rate": ranking_gap / len(rows) if rows else 0.0,
        "candidate_miss_count": candidate_miss,
        "candidate_miss_rate": candidate_miss / len(rows) if rows else 0.0,
    }


def summarize_group(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    top1_ious = [top1_iou(row) for row in rows]
    best_ious = [best_iou(row) for row in rows]
    return {
        "count": len(rows),
        "top1_mean_iou": safe_mean(top1_ious),
        "topk_mean_iou": safe_mean(best_ious),
        "topk_recall@0.5": sum(value >= 0.5 for value in best_ious) / len(best_ious),
        "topk_recall@0.7": sum(value >= 0.7 for value in best_ious) / len(best_ious),
        "ranking_gap@0.5": sum(t < 0.5 <= b for t, b in zip(top1_ious, best_ious, strict=True)) / len(rows),
    }


def group_by_duration(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        duration = float(row["ground_truth"]["duration"])
        if duration < 5:
            key = "<5s"
        elif duration < 15:
            key = "5-15s"
        elif duration < 30:
            key = "15-30s"
        elif duration < 60:
            key = "30-60s"
        elif duration < 120:
            key = "60-120s"
        else:
            key = ">=120s"
        groups[key].append(row)
    return dict(groups)


def group_by_caption_count(rows: list[dict]) -> dict[int, list[dict]]:
    counts = Counter(row["video_id"] for row in rows)
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        count = counts[row["video_id"]]
        bucket = count if count < 8 else 8
        groups[bucket].append(row)
    return dict(groups)


def duration_ratio_summary(rows: list[dict]) -> dict:
    ratios = []
    top1_ratios = []
    for row in rows:
        gt_duration = max(float(row["ground_truth"]["duration"]), 1e-6)
        best = best_prediction(row)
        first = row["predictions"][0] if row["predictions"] else None
        if best is not None:
            ratios.append(pred_duration(best) / gt_duration)
        if first is not None:
            top1_ratios.append(pred_duration(first) / gt_duration)
    return {
        "best_candidate_median_pred_over_gt": safe_median(ratios),
        "best_candidate_mean_pred_over_gt": safe_mean(ratios),
        "top1_median_pred_over_gt": safe_median(top1_ratios),
        "top1_mean_pred_over_gt": safe_mean(top1_ratios),
        "top1_too_short_rate_lt_0.5x": sum(value < 0.5 for value in top1_ratios) / len(top1_ratios) if top1_ratios else 0.0,
        "top1_too_long_rate_gt_2x": sum(value > 2.0 for value in top1_ratios) / len(top1_ratios) if top1_ratios else 0.0,
    }


def ranking_gap_examples(rows: list[dict], *, threshold: float, limit: int) -> list[dict]:
    candidates = [
        row
        for row in rows
        if top1_iou(row) < threshold <= best_iou(row)
    ]
    candidates.sort(key=lambda row: best_iou(row) - top1_iou(row), reverse=True)
    return [example_summary(row) for row in candidates[:limit]]


def miss_examples(rows: list[dict], *, threshold: float, limit: int) -> list[dict]:
    candidates = [row for row in rows if best_iou(row) < threshold]
    candidates.sort(key=best_iou)
    return [example_summary(row) for row in candidates[:limit]]


def boundary_near_misses(rows: list[dict], *, limit: int) -> list[dict]:
    candidates = [
        row
        for row in rows
        if 0.5 <= best_iou(row) < 0.7
    ]
    candidates.sort(key=lambda row: abs(best_iou(row) - 0.7))
    return [example_summary(row) for row in candidates[:limit]]


def worst_top1_examples(rows: list[dict], *, limit: int) -> list[dict]:
    candidates = sorted(rows, key=top1_iou)
    return [example_summary(row) for row in candidates[:limit]]


def example_summary(row: dict) -> dict:
    best = best_prediction(row)
    first = row["predictions"][0] if row["predictions"] else None
    return {
        "video_id": row["video_id"],
        "caption": row["caption"],
        "ground_truth": row["ground_truth"],
        "top1": minimal_prediction(first),
        "best_topk": minimal_prediction(best),
        "top1_iou": top1_iou(row),
        "best_iou": best_iou(row),
        "best_rank": best_rank(row),
    }


def minimal_prediction(prediction: dict | None) -> dict | None:
    if prediction is None:
        return None
    return {
        "start": prediction["start"],
        "end": prediction["end"],
        "score": prediction.get("score"),
        "iou": prediction.get("iou_with_ground_truth"),
    }


def top1_iou(row: dict) -> float:
    predictions = row.get("predictions") or []
    if not predictions:
        return 0.0
    return float(predictions[0].get("iou_with_ground_truth", 0.0))


def best_iou(row: dict) -> float:
    predictions = row.get("predictions") or []
    if not predictions:
        return 0.0
    return max(float(prediction.get("iou_with_ground_truth", 0.0)) for prediction in predictions)


def best_rank(row: dict) -> int | str:
    predictions = row.get("predictions") or []
    if not predictions:
        return "none"
    values = [float(prediction.get("iou_with_ground_truth", 0.0)) for prediction in predictions]
    return values.index(max(values)) + 1


def best_prediction(row: dict) -> dict | None:
    predictions = row.get("predictions") or []
    if not predictions:
        return None
    return max(predictions, key=lambda prediction: float(prediction.get("iou_with_ground_truth", 0.0)))


def pred_duration(prediction: dict) -> float:
    return max(0.0, float(prediction["end"]) - float(prediction["start"]))


def safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def safe_median(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


if __name__ == "__main__":
    main()
