from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from data import read_jsonl_manifest
from metrics import summarize_recall
from scoring import load_scorer
from segments import TemporalSegment, non_max_suppression
from video_clip.context_reranker import context_candidate_features, load_contextual_reranker
from video_clip.segment_reranker import (
    SegmentCandidate,
    _prediction_row,
    _video_duration,
    generate_segment_candidates,
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
    parser.add_argument("--span-window-counts", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--embedding-cache-dir", type=Path)
    parser.add_argument("--embed-batch-size", type=int, default=4)
    parser.add_argument("--actionness-alpha", type=float, default=0.05)
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
        for example in tqdm(examples, desc="Evaluating actionness reranker"):
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
                candidates = generate_segment_candidates(
                    windows,
                    scale=float(scale),
                    span_window_counts=tuple(args.span_window_counts),
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
                actionness = candidate_actionness_scores(candidates, embedding_actionness(embeddings))
                with torch.no_grad():
                    context = model.contextualize(
                        embeddings.to(device),
                        text_embedding,
                        video_duration=video_duration,
                        window_centers=window_centers,
                    )
                    base_scores = torch.sigmoid(model.score_spans(context, candidates, features.to(device))).cpu().numpy()

                adjusted = base_scores + args.actionness_alpha * actionness
                scored.extend(
                    TemporalSegment(candidate.start, candidate.end, float(score))
                    for candidate, score in zip(candidates, adjusted, strict=True)
                )

            final_segments = non_max_suppression(scored, iou_threshold=0.5, top_k=args.top_k)
            target = TemporalSegment(example.start, example.end, 1.0)
            predictions.append(final_segments)
            targets.append(target)
            if output_handle is not None:
                output_handle.write(json.dumps(_prediction_row(example, final_segments, target)) + "\n")
                output_handle.flush()
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


def embedding_actionness(embeddings: torch.Tensor) -> np.ndarray:
    n = int(embeddings.shape[0])
    if n == 0:
        return np.asarray([], dtype=np.float32)
    if n == 1:
        return np.asarray([0.5], dtype=np.float32)

    normalized = F.normalize(embeddings.float(), dim=-1)
    changes = (1.0 - (normalized[:-1] * normalized[1:]).sum(dim=-1)).cpu().numpy()
    action = np.zeros(n, dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)
    action[:-1] += changes
    action[1:] += changes
    counts[:-1] += 1.0
    counts[1:] += 1.0
    action = action / np.maximum(counts, 1.0)

    low = float(np.percentile(action, 10))
    high = float(np.percentile(action, 90))
    if high <= low + 1e-8:
        return np.full(n, 0.5, dtype=np.float32)
    return np.clip((action - low) / (high - low), 0.0, 1.0).astype(np.float32)


def candidate_actionness_scores(candidates: list[SegmentCandidate], window_actionness: np.ndarray) -> np.ndarray:
    if not candidates:
        return np.asarray([], dtype=np.float32)
    scores = []
    for candidate in candidates:
        values = window_actionness[candidate.start_idx : candidate.end_idx + 1]
        if values.size == 0:
            scores.append(0.0)
        else:
            scores.append(0.7 * float(values.mean()) + 0.3 * float(values.max()))
    return np.asarray(scores, dtype=np.float32)


if __name__ == "__main__":
    main()
