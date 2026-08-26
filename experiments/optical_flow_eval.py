from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from data import read_jsonl_manifest
from metrics import summarize_recall
from scoring import load_scorer
from segments import TemporalSegment, non_max_suppression
from video_clip.context_reranker import context_candidate_features, load_contextual_reranker
from video_clip.segment_reranker import SegmentCandidate, _prediction_row, _video_duration
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
    parser.add_argument("--flow-cache-dir", type=Path, default=Path("outputs/optical_flow_cache"))
    parser.add_argument("--flow-sample-seconds", type=float, default=1.0)
    parser.add_argument("--flow-width", type=int, default=160)
    parser.add_argument("--top-motion-points", type=int, default=32)
    parser.add_argument("--max-boundary-gap", type=int, default=8)
    parser.add_argument("--max-flow-samples", type=int, default=240)
    parser.add_argument("--embed-batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()

    examples = read_jsonl_manifest(args.manifest, video_root=args.video_root)
    if args.limit is not None:
        examples = examples[: args.limit]
    total_requested = len(examples)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    embedder = XCLIPEmbedder(args.model_name, device=args.device, batch_size=args.embed_batch_size)
    base_scorer = load_scorer(args.base_checkpoint, map_location=device).to(device)
    base_scorer.eval()
    model = load_contextual_reranker(args.checkpoint, map_location=device).to(device)
    model.eval()

    text_cache: dict[str, torch.Tensor] = {}
    video_cache: dict[tuple[Path, float, float], tuple[list, torch.Tensor]] = {}
    flow_cache: dict[Path, list[float]] = {}
    predictions: list[list[TemporalSegment]] = []
    targets: list[TemporalSegment] = []

    skipped_existing = 0
    output_handle = None
    if args.predictions_out is not None:
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        if args.resume and args.predictions_out.exists():
            existing_predictions, existing_targets = read_prediction_rows(args.predictions_out)
            predictions.extend(existing_predictions)
            targets.extend(existing_targets)
            skipped_existing = len(existing_targets)
            examples = examples[skipped_existing:]
            output_handle = args.predictions_out.open("a", encoding="utf-8")
        else:
            output_handle = args.predictions_out.open("w", encoding="utf-8")

    try:
        for example in tqdm(
            examples,
            desc="Evaluating optical-flow proposals",
            initial=skipped_existing,
            total=len(examples) + skipped_existing,
        ):
            if example.video_path is None:
                continue
            if example.caption not in text_cache:
                text_cache[example.caption] = embedder.encode_text(example.caption)[0]
            if example.video_path not in flow_cache:
                flow_cache[example.video_path] = optical_flow_boundaries(
                    example.video_path,
                    cache_dir=args.flow_cache_dir,
                    sample_seconds=args.flow_sample_seconds,
                    width=args.flow_width,
                    top_points=args.top_motion_points,
                    max_samples=args.max_flow_samples,
                )
            boundary_times = flow_cache[example.video_path]
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
                candidates = flow_candidates(
                    windows,
                    boundary_times=boundary_times,
                    scale=float(scale),
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
                output_handle.flush()
    finally:
        if output_handle is not None:
            output_handle.close()

    print(
        json.dumps(
            {
                "metrics": summarize_recall(predictions, targets, args.iou_thresholds),
                "evaluated": len(targets),
                "skipped": total_requested - len(targets),
            },
            indent=2,
        )
    )


def optical_flow_boundaries(
    video_path: Path,
    *,
    cache_dir: Path,
    sample_seconds: float,
    width: int,
    top_points: int,
    max_samples: int,
) -> list[float]:
    cache_path = flow_cache_path(
        cache_dir,
        video_path,
        sample_seconds=sample_seconds,
        width=width,
        max_samples=max_samples,
    )
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return [float(item) for item in payload["boundaries"]]

    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = float(frame_count / fps) if fps > 0 else 0.0
    step_frames = max(1, int(round(fps * sample_seconds)))

    samples: list[tuple[float, np.ndarray]] = []
    frame_idx = 0
    while True:
        if max_samples > 0 and len(samples) >= max_samples:
            break
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = capture.read()
        if not ok:
            break
        time = frame_idx / fps
        gray = preprocess_flow_frame(frame, width=width)
        samples.append((time, gray))
        frame_idx += step_frames
    capture.release()

    if len(samples) < 2:
        boundaries = [0.0, duration]
    else:
        magnitudes = []
        for (time, previous), (_next_time, current) in zip(samples[:-1], samples[1:], strict=False):
            flow = cv2.calcOpticalFlowFarneback(
                previous,
                current,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            mag, _angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append((time, float(np.mean(mag))))
        if magnitudes:
            values = np.asarray([item[1] for item in magnitudes], dtype=np.float32)
            if values.size >= 3:
                values = np.abs(values - np.median(values))
            top = np.argsort(values)[::-1][: min(top_points, len(magnitudes))]
            boundaries = [0.0, duration]
            boundaries.extend(float(magnitudes[int(idx)][0]) for idx in top)
        else:
            boundaries = [0.0, duration]

    clean = sorted({round(max(0.0, min(duration, item)), 3) for item in boundaries})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"boundaries": clean}), encoding="utf-8")
    return clean


def preprocess_flow_frame(frame_bgr: np.ndarray, *, width: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if w > width:
        height = max(1, int(round(h * width / w)))
        frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def flow_candidates(
    windows: list,
    *,
    boundary_times: list[float],
    scale: float,
    max_boundary_gap: int,
) -> list[SegmentCandidate]:
    if not windows:
        return []
    n = len(windows)
    boundaries = sorted({nearest_boundary_index(windows, time) for time in boundary_times})
    boundaries = [idx for idx in boundaries if 0 <= idx <= n]
    if 0 not in boundaries:
        boundaries.insert(0, 0)
    if n not in boundaries:
        boundaries.append(n)

    candidates: list[SegmentCandidate] = []
    seen: set[tuple[int, int]] = set()
    for left_pos, start_boundary in enumerate(boundaries[:-1]):
        for end_boundary in boundaries[left_pos + 1 : left_pos + 1 + max_boundary_gap]:
            start_idx = start_boundary
            end_idx = end_boundary - 1
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


def nearest_boundary_index(windows: list, time: float) -> int:
    starts = np.asarray([window.start for window in windows] + [windows[-1].end], dtype=np.float32)
    return int(np.argmin(np.abs(starts - float(time))))


def read_prediction_rows(path: Path) -> tuple[list[list[TemporalSegment]], list[TemporalSegment]]:
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


def flow_cache_path(
    cache_dir: Path,
    video_path: Path,
    *,
    sample_seconds: float,
    width: int,
    max_samples: int,
) -> Path:
    key = f"{video_path.resolve()}|sample={sample_seconds:g}|width={width}|max={max_samples}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{video_path.stem}_{digest}.json"


if __name__ == "__main__":
    main()
