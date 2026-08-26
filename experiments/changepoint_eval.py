from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from data import read_jsonl_manifest
from metrics import summarize_recall
from scoring import load_scorer
from segments import TemporalSegment, non_max_suppression
from video_clip.context_reranker import (
    context_candidate_features,
    load_contextual_reranker,
)
from video_clip.segment_reranker import (
    SegmentCandidate,
    _prediction_row,
    _video_duration,
)
from video_clip.train import _load_or_encode_video_windows
from video_clip.xclip_backend import XCLIPEmbedder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", default="microsoft/xclip-base-patch32")
    parser.add_argument("--multi-scales", type=float, nargs="+", required=True)
    parser.add_argument("--multi-strides", type=float, nargs="+", required=True)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--embedding-cache-dir", type=Path)
    parser.add_argument("--embed-batch-size", type=int, default=4)
    parser.add_argument("--top-change-points", type=int, default=24)
    parser.add_argument("--max-boundary-gap", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()

    examples = read_jsonl_manifest(args.manifest, video_root=args.video_root)
    if args.limit is not None:
        examples = examples[: args.limit]

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    embedder = XCLIPEmbedder(args.model_name, device=args.device, batch_size=args.embed_batch_size)
    base_scorer = load_scorer(args.base_checkpoint, map_location=device).to(device)
    base_scorer.eval()
    model = load_contextual_reranker(args.checkpoint, map_location=device).to(device)
    model.eval()

    text_cache: dict[str, torch.Tensor] = {}
    video_cache: dict[tuple[Path, float, float], tuple[list, torch.Tensor]] = {}
    predictions: list[list[TemporalSegment]] = []
    targets: list[TemporalSegment] = []

    output_handle = None
    if args.predictions_out is not None:
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.predictions_out.open("w", encoding="utf-8")

    try:
        for example in tqdm(examples, desc="Evaluating change-point proposals"):
            if example.video_path is None:
                continue
            if example.caption not in text_cache:
                text_cache[example.caption] = embedder.encode_text(example.caption)[0]
            text_embedding_cpu = text_cache[example.caption]
            text_embedding = text_embedding_cpu.to(device)
            scored: list[TemporalSegment] = []

            for scale, stride in zip(args.multi_scales, args.multi_strides, strict=True):
                cache_key = (example.video_path, float(scale), float(stride))
                if cache_key not in video_cache:
                    video_cache[cache_key] = _load_or_encode_video_windows(
                        example.video_path,
                        embedder=embedder,
                        clip_seconds=scale,
                        stride_seconds=stride,
                        frames_per_clip=args.frames_per_clip,
                        embedding_cache_dir=args.embedding_cache_dir,
                    )
                windows, embeddings = video_cache[cache_key]
                candidates = changepoint_candidates(
                    windows,
                    embeddings,
                    scale=float(scale),
                    top_change_points=args.top_change_points,
                    max_boundary_gap=args.max_boundary_gap,
                )
                if not candidates:
                    continue
                video_duration = _video_duration(windows)
                features = context_candidate_features(
                    candidates,
                    embeddings,
                    text_embedding_cpu,
                    video_duration=video_duration,
                    base_scorer=base_scorer,
                    device=device,
                )
                window_centers = torch.as_tensor(
                    [(window.start + window.end) / 2.0 for window in windows],
                    dtype=torch.float32,
                    device=device,
                )
                with torch.no_grad():
                    context = model.contextualize(
                        embeddings.to(device),
                        text_embedding,
                        video_duration=video_duration,
                        window_centers=window_centers,
                    )
                    scores = torch.sigmoid(model.score_spans(context, candidates, features.to(device))).cpu()
                scored.extend(
                    TemporalSegment(candidate.start, candidate.end, float(score))
                    for candidate, score in zip(candidates, scores, strict=True)
                )

            final_segments = non_max_suppression(scored, iou_threshold=0.5, top_k=args.top_k)
            target = TemporalSegment(example.start, example.end, 1.0)
            predictions.append(final_segments)
            targets.append(target)
            if output_handle is not None:
                output_handle.write(json.dumps(_prediction_row(example, final_segments, target)) + "\n")
    finally:
        if output_handle is not None:
            output_handle.close()

    print(
        json.dumps(
            {
                "metrics": summarize_recall(predictions, targets, args.iou_thresholds),
                "evaluated": len(targets),
                "skipped": len(examples) - len(targets),
            },
            indent=2,
        )
    )


def changepoint_candidates(
    windows: list,
    embeddings: torch.Tensor,
    *,
    scale: float,
    top_change_points: int,
    max_boundary_gap: int,
) -> list[SegmentCandidate]:
    n = len(windows)
    if n == 0:
        return []
    if n == 1:
        return [SegmentCandidate(scale, 0, 0, windows[0].start, windows[0].end)]

    normalized = F.normalize(embeddings.float(), dim=-1)
    changes = 1.0 - (normalized[:-1] * normalized[1:]).sum(dim=-1).cpu().numpy()
    top = np.argsort(changes)[::-1][: max(0, min(top_change_points, n - 1))]
    boundaries = {0, n}
    boundaries.update(int(idx) + 1 for idx in top)
    ordered = sorted(boundaries)

    candidates: list[SegmentCandidate] = []
    seen: set[tuple[int, int]] = set()
    for left_pos, start_boundary in enumerate(ordered[:-1]):
        for right_boundary in ordered[left_pos + 1 : left_pos + 1 + max_boundary_gap]:
            start_idx = start_boundary
            end_idx = right_boundary - 1
            if start_idx < 0 or end_idx >= n or start_idx > end_idx:
                continue
            key = (start_idx, end_idx)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                SegmentCandidate(
                    scale=scale,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start=windows[start_idx].start,
                    end=windows[end_idx].end,
                )
            )
    return candidates


if __name__ == "__main__":
    main()
