from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_download_parser(subparsers)
    _add_eval_parser(subparsers)
    _add_explain_parser(subparsers)
    _add_oracle_parser(subparsers)
    _add_segment_train_parser(subparsers)
    _add_segment_eval_parser(subparsers)
    _add_hard_negative_train_parser(subparsers)
    _add_boundary_train_parser(subparsers)
    _add_boundary_eval_parser(subparsers)
    _add_listwise_train_parser(subparsers)
    _add_listwise_eval_parser(subparsers)
    _add_context_train_parser(subparsers)
    _add_context_eval_parser(subparsers)
    _add_context_explain_parser(subparsers)
    _add_train_parser(subparsers)
    args = parser.parse_args()

    if args.command == "download-model":
        _run_download_model(args)
    elif args.command == "eval":
        _run_eval(args)
    elif args.command == "explain":
        _run_explain(args)
    elif args.command == "oracle":
        _run_oracle(args)
    elif args.command == "segment-train":
        _run_segment_train(args)
    elif args.command == "segment-eval":
        _run_segment_eval(args)
    elif args.command == "hard-negative-train":
        _run_hard_negative_train(args)
    elif args.command == "boundary-train":
        _run_boundary_train(args)
    elif args.command == "boundary-eval":
        _run_boundary_eval(args)
    elif args.command == "listwise-train":
        _run_listwise_train(args)
    elif args.command == "listwise-eval":
        _run_listwise_eval(args)
    elif args.command == "context-train":
        _run_context_train(args)
    elif args.command == "context-eval":
        _run_context_eval(args)
    elif args.command == "context-explain":
        _run_context_explain(args)
    elif args.command == "train":
        _run_train(args)


def _add_download_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("download-model")
    parser.add_argument("--model-name", default="microsoft/xclip-base-patch32")


def _add_eval_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("eval")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-name", default="microsoft/xclip-base-patch32")
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", default="auto")
    parser.add_argument("--min-segment-seconds", type=float, default=1.0)
    parser.add_argument("--merge-gap-seconds", type=float, default=2.0)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--multi-scales", type=float, nargs="+")
    parser.add_argument("--multi-strides", type=float, nargs="+")
    parser.add_argument("--multi-merge-gaps", type=float, nargs="+")
    parser.add_argument("--embedding-cache-dir", type=Path)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device")


def _add_explain_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("explain")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-name", default="microsoft/xclip-base-patch32")
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", default="auto")
    parser.add_argument("--min-segment-seconds", type=float, default=1.0)
    parser.add_argument("--merge-gap-seconds", type=float, default=2.0)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--multi-scales", type=float, nargs="+")
    parser.add_argument("--multi-strides", type=float, nargs="+")
    parser.add_argument("--multi-merge-gaps", type=float, nargs="+")
    parser.add_argument("--top-windows", type=int, default=12)
    parser.add_argument("--all-windows", action="store_true")
    parser.add_argument("--ground-truth-start", type=float)
    parser.add_argument("--ground-truth-end", type=float)
    parser.add_argument("--device")


def _add_oracle_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("oracle")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--multi-scales", type=float, nargs="+", required=True)
    parser.add_argument("--multi-strides", type=float, nargs="+", required=True)
    parser.add_argument("--single-window-only", action="store_true")
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--limit", type=int)


def _add_segment_train_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("segment-train")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="microsoft/xclip-base-patch32")
    parser.add_argument("--multi-scales", type=float, nargs="+", required=True)
    parser.add_argument("--multi-strides", type=float, nargs="+", required=True)
    parser.add_argument("--span-window-counts", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--embedding-cache-dir", type=Path)
    parser.add_argument("--embed-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--positive-iou-threshold", type=float, default=0.5)
    parser.add_argument("--negative-iou-threshold", type=float, default=0.1)
    parser.add_argument("--max-positive-segments", type=int, default=16)
    parser.add_argument("--target-mode", choices=["binary", "iou"], default="binary")
    parser.add_argument("--use-score-features", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")


def _add_segment_eval_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("segment-eval")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", default="microsoft/xclip-base-patch32")
    parser.add_argument("--multi-scales", type=float, nargs="+", required=True)
    parser.add_argument("--multi-strides", type=float, nargs="+", required=True)
    parser.add_argument("--span-window-counts", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--embedding-cache-dir", type=Path)
    parser.add_argument("--embed-batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--device")


def _add_hard_negative_train_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("hard-negative-train")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="microsoft/xclip-base-patch32")
    parser.add_argument("--multi-scales", type=float, nargs="+", required=True)
    parser.add_argument("--multi-strides", type=float, nargs="+", required=True)
    parser.add_argument("--span-window-counts", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--embedding-cache-dir", type=Path)
    parser.add_argument("--embed-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-positive-segments", type=int, default=32)
    parser.add_argument("--max-hard-negatives", type=int, default=128)
    parser.add_argument("--max-broad-negatives", type=int, default=64)
    parser.add_argument("--hard-negative-iou-threshold", type=float, default=0.2)
    parser.add_argument("--broad-negative-iou-threshold", type=float, default=0.5)
    parser.add_argument("--broad-negative-duration-ratio", type=float, default=2.0)
    parser.add_argument("--target-mode", choices=["binary", "iou"], default="iou")
    parser.add_argument("--use-score-features", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")


def _add_boundary_train_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("boundary-train")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="microsoft/xclip-base-patch32")
    parser.add_argument("--multi-scales", type=float, nargs="+", required=True)
    parser.add_argument("--multi-strides", type=float, nargs="+", required=True)
    parser.add_argument("--span-window-counts", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--embedding-cache-dir", type=Path)
    parser.add_argument("--embed-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-train-iou", type=float, default=0.3)
    parser.add_argument("--max-segments-per-example", type=int, default=64)
    parser.add_argument("--use-score-features", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")


def _add_boundary_eval_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("boundary-eval")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--boundary-checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", default="microsoft/xclip-base-patch32")
    parser.add_argument("--multi-scales", type=float, nargs="+", required=True)
    parser.add_argument("--multi-strides", type=float, nargs="+", required=True)
    parser.add_argument("--span-window-counts", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--embedding-cache-dir", type=Path)
    parser.add_argument("--embed-batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--boundary-alpha", type=float, default=1.0)
    parser.add_argument("--max-boundary-offset", type=float, default=1.0)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--device")


def _add_listwise_train_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("listwise-train")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="microsoft/xclip-base-patch32")
    parser.add_argument("--multi-scales", type=float, nargs="+", required=True)
    parser.add_argument("--multi-strides", type=float, nargs="+", required=True)
    parser.add_argument("--span-window-counts", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--embedding-cache-dir", type=Path)
    parser.add_argument("--embed-batch-size", type=int, default=4)
    parser.add_argument("--candidate-pool-size", type=int, default=128)
    parser.add_argument("--oracle-pool-size", type=int, default=16)
    parser.add_argument("--target-temperature", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")


def _add_listwise_eval_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("listwise-eval")
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
    parser.add_argument("--candidate-pool-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--device")


def _add_context_train_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("context-train")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="microsoft/xclip-base-patch32")
    parser.add_argument("--multi-scales", type=float, nargs="+", required=True)
    parser.add_argument("--multi-strides", type=float, nargs="+", required=True)
    parser.add_argument("--span-window-counts", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--embedding-cache-dir", type=Path)
    parser.add_argument("--embed-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-positive-segments", type=int, default=32)
    parser.add_argument("--max-hard-negatives", type=int, default=96)
    parser.add_argument("--target-mode", choices=["binary", "iou"], default="iou")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")


def _add_context_eval_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("context-eval")
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
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--device")


def _add_context_explain_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("context-explain")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("checkpoints/xclip_segment_reranker_v2_2000videos.pt"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/xclip_context_reranker_v1.pt"),
    )
    parser.add_argument("--model-name", default="microsoft/xclip-base-patch32")
    parser.add_argument("--multi-scales", type=float, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--multi-strides", type=float, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--span-window-counts",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32],
    )
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--embedding-cache-dir", type=Path)
    parser.add_argument("--embed-batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--device")


def _add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="microsoft/xclip-base-patch32")
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--embed-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--positive-iou-threshold", type=float, default=0.5)
    parser.add_argument("--positive-window-coverage-threshold", type=float, default=0.5)
    parser.add_argument("--max-positive-windows", type=int)
    parser.add_argument("--use-duration-feature", action="store_true")
    parser.add_argument("--multi-scales", type=float, nargs="+")
    parser.add_argument("--multi-strides", type=float, nargs="+")
    parser.add_argument("--embedding-cache-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")


def _run_download_model(args: argparse.Namespace) -> None:
    from transformers import XCLIPModel, XCLIPProcessor

    XCLIPProcessor.from_pretrained(args.model_name)
    XCLIPModel.from_pretrained(args.model_name, use_safetensors=True)
    print(json.dumps({"model": args.model_name, "status": "downloaded"}))


def _run_eval(args: argparse.Namespace) -> None:
    from data import read_jsonl_manifest
    from video_clip.config import VideoClipConfig
    from video_clip.evaluate import evaluate_examples
    from video_clip.pipeline import VideoClipLocalizer

    examples = read_jsonl_manifest(args.manifest, video_root=args.video_root)
    threshold = args.threshold if args.threshold == "auto" else float(args.threshold)
    config = VideoClipConfig(
        model_name=args.model_name,
        clip_seconds=args.clip_seconds,
        stride_seconds=args.stride_seconds,
        frames_per_clip=args.frames_per_clip,
        batch_size=args.batch_size,
        score_threshold=threshold,
        min_segment_seconds=args.min_segment_seconds,
        merge_gap_seconds=args.merge_gap_seconds,
        smoothing_window=args.smoothing_window,
        top_k=args.top_k,
        device=args.device,
        embedding_cache_dir=str(args.embedding_cache_dir) if args.embedding_cache_dir else None,
        multi_scales=tuple(args.multi_scales) if args.multi_scales else None,
        multi_strides=tuple(args.multi_strides) if args.multi_strides else None,
        multi_merge_gaps=tuple(args.multi_merge_gaps) if args.multi_merge_gaps else None,
    )
    result = evaluate_examples(
        examples,
        localizer=VideoClipLocalizer.from_checkpoint(args.checkpoint, config=config),
        iou_thresholds=args.iou_thresholds,
        limit=args.limit,
        predictions_out=args.predictions_out,
        resume=args.resume,
    )
    print(json.dumps(asdict(result), indent=2))


def _run_explain(args: argparse.Namespace) -> None:
    from segments import TemporalSegment, temporal_iou
    from video_clip.config import VideoClipConfig
    from video_clip.pipeline import VideoClipLocalizer

    threshold = args.threshold if args.threshold == "auto" else float(args.threshold)
    config = VideoClipConfig(
        model_name=args.model_name,
        clip_seconds=args.clip_seconds,
        stride_seconds=args.stride_seconds,
        frames_per_clip=args.frames_per_clip,
        batch_size=args.batch_size,
        score_threshold=threshold,
        min_segment_seconds=args.min_segment_seconds,
        merge_gap_seconds=args.merge_gap_seconds,
        smoothing_window=args.smoothing_window,
        top_k=args.top_k,
        device=args.device,
        multi_scales=tuple(args.multi_scales) if args.multi_scales else None,
        multi_strides=tuple(args.multi_strides) if args.multi_strides else None,
        multi_merge_gaps=tuple(args.multi_merge_gaps) if args.multi_merge_gaps else None,
    )
    explanation = VideoClipLocalizer.from_checkpoint(args.checkpoint, config=config).explain(
        args.video, args.query
    )
    window_rows = [asdict(window) for window in explanation.windows]
    if not args.all_windows:
        window_rows = sorted(window_rows, key=lambda item: item["score"], reverse=True)[
            : args.top_windows
        ]
    payload = {
        "query": explanation.query,
        "video": str(explanation.video_path),
        "config": asdict(config),
        "cutoff": explanation.cutoff,
        "windows": window_rows,
        "active_windows": [
            asdict(window) for window in explanation.windows if window.active
        ],
        "raw_segments": [asdict(segment) for segment in explanation.raw_segments],
        "merged_segments": [asdict(segment) for segment in explanation.merged_segments],
        "final_segments": [asdict(segment) for segment in explanation.final_segments],
    }
    if args.ground_truth_start is not None and args.ground_truth_end is not None:
        target = TemporalSegment(args.ground_truth_start, args.ground_truth_end, 1.0)
        payload["final_iou_with_ground_truth"] = [
            temporal_iou(segment, target) for segment in explanation.final_segments
        ]
    print(json.dumps(payload, indent=2))


def _run_oracle(args: argparse.Namespace) -> None:
    from data import read_jsonl_manifest
    from video_clip.oracle import evaluate_oracle_proposals

    examples = read_jsonl_manifest(args.manifest, video_root=args.video_root)
    result = evaluate_oracle_proposals(
        examples,
        scales=tuple(args.multi_scales),
        strides=tuple(args.multi_strides),
        iou_thresholds=args.iou_thresholds,
        limit=args.limit,
        merged=not args.single_window_only,
    )
    print(json.dumps(asdict(result), indent=2))


def _run_segment_train(args: argparse.Namespace) -> None:
    from data import read_jsonl_manifest
    from video_clip.segment_reranker import train_segment_reranker
    from video_clip.xclip_backend import XCLIPEmbedder

    examples = read_jsonl_manifest(args.manifest, video_root=args.video_root)
    if args.limit is not None:
        examples = examples[: args.limit]
    embedder = XCLIPEmbedder(args.model_name, device=args.device, batch_size=args.embed_batch_size)
    summary = train_segment_reranker(
        examples,
        output=args.output,
        embedder=embedder,
        scales=tuple(args.multi_scales),
        strides=tuple(args.multi_strides),
        embedding_cache_dir=args.embedding_cache_dir,
        frames_per_clip=args.frames_per_clip,
        span_window_counts=tuple(args.span_window_counts),
        positive_iou_threshold=args.positive_iou_threshold,
        negative_iou_threshold=args.negative_iou_threshold,
        max_positive_segments=args.max_positive_segments,
        negative_ratio=args.negative_ratio,
        target_mode=args.target_mode,
        use_score_features=args.use_score_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    print(json.dumps(asdict(summary), indent=2, default=str))


def _run_segment_eval(args: argparse.Namespace) -> None:
    from data import read_jsonl_manifest
    from video_clip.segment_reranker import evaluate_segment_reranker
    from video_clip.xclip_backend import XCLIPEmbedder

    examples = read_jsonl_manifest(args.manifest, video_root=args.video_root)
    embedder = XCLIPEmbedder(args.model_name, device=args.device, batch_size=args.embed_batch_size)
    result = evaluate_segment_reranker(
        examples,
        checkpoint=args.checkpoint,
        embedder=embedder,
        scales=tuple(args.multi_scales),
        strides=tuple(args.multi_strides),
        embedding_cache_dir=args.embedding_cache_dir,
        frames_per_clip=args.frames_per_clip,
        span_window_counts=tuple(args.span_window_counts),
        iou_thresholds=args.iou_thresholds,
        top_k=args.top_k,
        limit=args.limit,
        predictions_out=args.predictions_out,
        device=args.device,
    )
    print(json.dumps(asdict(result), indent=2))


def _run_hard_negative_train(args: argparse.Namespace) -> None:
    from data import read_jsonl_manifest
    from video_clip.segment_reranker import train_hard_negative_segment_reranker
    from video_clip.xclip_backend import XCLIPEmbedder

    examples = read_jsonl_manifest(args.manifest, video_root=args.video_root)
    if args.limit is not None:
        examples = examples[: args.limit]
    embedder = XCLIPEmbedder(args.model_name, device=args.device, batch_size=args.embed_batch_size)
    summary = train_hard_negative_segment_reranker(
        examples,
        output=args.output,
        base_checkpoint=args.base_checkpoint,
        embedder=embedder,
        scales=tuple(args.multi_scales),
        strides=tuple(args.multi_strides),
        embedding_cache_dir=args.embedding_cache_dir,
        frames_per_clip=args.frames_per_clip,
        span_window_counts=tuple(args.span_window_counts),
        max_positive_segments=args.max_positive_segments,
        max_hard_negatives=args.max_hard_negatives,
        max_broad_negatives=args.max_broad_negatives,
        hard_negative_iou_threshold=args.hard_negative_iou_threshold,
        broad_negative_iou_threshold=args.broad_negative_iou_threshold,
        broad_negative_duration_ratio=args.broad_negative_duration_ratio,
        target_mode=args.target_mode,
        use_score_features=args.use_score_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    print(json.dumps(asdict(summary), indent=2, default=str))


def _run_boundary_train(args: argparse.Namespace) -> None:
    from data import read_jsonl_manifest
    from video_clip.segment_reranker import train_boundary_refiner
    from video_clip.xclip_backend import XCLIPEmbedder

    examples = read_jsonl_manifest(args.manifest, video_root=args.video_root)
    if args.limit is not None:
        examples = examples[: args.limit]
    embedder = XCLIPEmbedder(args.model_name, device=args.device, batch_size=args.embed_batch_size)
    summary = train_boundary_refiner(
        examples,
        output=args.output,
        scorer_checkpoint=args.checkpoint,
        embedder=embedder,
        scales=tuple(args.multi_scales),
        strides=tuple(args.multi_strides),
        embedding_cache_dir=args.embedding_cache_dir,
        frames_per_clip=args.frames_per_clip,
        span_window_counts=tuple(args.span_window_counts),
        min_train_iou=args.min_train_iou,
        max_segments_per_example=args.max_segments_per_example,
        use_score_features=args.use_score_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    print(json.dumps(asdict(summary), indent=2, default=str))


def _run_boundary_eval(args: argparse.Namespace) -> None:
    from data import read_jsonl_manifest
    from video_clip.segment_reranker import evaluate_boundary_refined_reranker
    from video_clip.xclip_backend import XCLIPEmbedder

    examples = read_jsonl_manifest(args.manifest, video_root=args.video_root)
    embedder = XCLIPEmbedder(args.model_name, device=args.device, batch_size=args.embed_batch_size)
    result = evaluate_boundary_refined_reranker(
        examples,
        checkpoint=args.checkpoint,
        boundary_checkpoint=args.boundary_checkpoint,
        embedder=embedder,
        scales=tuple(args.multi_scales),
        strides=tuple(args.multi_strides),
        embedding_cache_dir=args.embedding_cache_dir,
        frames_per_clip=args.frames_per_clip,
        span_window_counts=tuple(args.span_window_counts),
        iou_thresholds=args.iou_thresholds,
        top_k=args.top_k,
        boundary_alpha=args.boundary_alpha,
        max_boundary_offset=args.max_boundary_offset,
        limit=args.limit,
        predictions_out=args.predictions_out,
        device=args.device,
    )
    print(json.dumps(asdict(result), indent=2))


def _run_listwise_train(args: argparse.Namespace) -> None:
    from data import read_jsonl_manifest
    from video_clip.listwise_reranker import train_listwise_reranker
    from video_clip.xclip_backend import XCLIPEmbedder

    examples = read_jsonl_manifest(args.manifest, video_root=args.video_root)
    if args.limit is not None:
        examples = examples[: args.limit]
    embedder = XCLIPEmbedder(args.model_name, device=args.device, batch_size=args.embed_batch_size)
    summary = train_listwise_reranker(
        examples,
        output=args.output,
        base_checkpoint=args.base_checkpoint,
        embedder=embedder,
        scales=tuple(args.multi_scales),
        strides=tuple(args.multi_strides),
        embedding_cache_dir=args.embedding_cache_dir,
        frames_per_clip=args.frames_per_clip,
        span_window_counts=tuple(args.span_window_counts),
        candidate_pool_size=args.candidate_pool_size,
        oracle_pool_size=args.oracle_pool_size,
        target_temperature=args.target_temperature,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    print(json.dumps(asdict(summary), indent=2, default=str))


def _run_listwise_eval(args: argparse.Namespace) -> None:
    from data import read_jsonl_manifest
    from video_clip.listwise_reranker import evaluate_listwise_reranker
    from video_clip.xclip_backend import XCLIPEmbedder

    examples = read_jsonl_manifest(args.manifest, video_root=args.video_root)
    embedder = XCLIPEmbedder(args.model_name, device=args.device, batch_size=args.embed_batch_size)
    result = evaluate_listwise_reranker(
        examples,
        base_checkpoint=args.base_checkpoint,
        checkpoint=args.checkpoint,
        embedder=embedder,
        scales=tuple(args.multi_scales),
        strides=tuple(args.multi_strides),
        embedding_cache_dir=args.embedding_cache_dir,
        frames_per_clip=args.frames_per_clip,
        span_window_counts=tuple(args.span_window_counts),
        iou_thresholds=args.iou_thresholds,
        candidate_pool_size=args.candidate_pool_size,
        top_k=args.top_k,
        limit=args.limit,
        predictions_out=args.predictions_out,
        device=args.device,
    )
    print(json.dumps(asdict(result), indent=2))


def _run_context_train(args: argparse.Namespace) -> None:
    from data import read_jsonl_manifest
    from video_clip.context_reranker import train_contextual_reranker
    from video_clip.xclip_backend import XCLIPEmbedder

    examples = read_jsonl_manifest(args.manifest, video_root=args.video_root)
    if args.limit is not None:
        examples = examples[: args.limit]
    embedder = XCLIPEmbedder(args.model_name, device=args.device, batch_size=args.embed_batch_size)
    summary = train_contextual_reranker(
        examples,
        output=args.output,
        base_checkpoint=args.base_checkpoint,
        init_checkpoint=args.init_checkpoint,
        embedder=embedder,
        scales=tuple(args.multi_scales),
        strides=tuple(args.multi_strides),
        embedding_cache_dir=args.embedding_cache_dir,
        frames_per_clip=args.frames_per_clip,
        span_window_counts=tuple(args.span_window_counts),
        max_positive_segments=args.max_positive_segments,
        max_hard_negatives=args.max_hard_negatives,
        target_mode=args.target_mode,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    print(json.dumps(asdict(summary), indent=2, default=str))


def _run_context_eval(args: argparse.Namespace) -> None:
    from data import read_jsonl_manifest
    from video_clip.context_reranker import evaluate_contextual_reranker
    from video_clip.xclip_backend import XCLIPEmbedder

    examples = read_jsonl_manifest(args.manifest, video_root=args.video_root)
    embedder = XCLIPEmbedder(args.model_name, device=args.device, batch_size=args.embed_batch_size)
    result = evaluate_contextual_reranker(
        examples,
        base_checkpoint=args.base_checkpoint,
        checkpoint=args.checkpoint,
        embedder=embedder,
        scales=tuple(args.multi_scales),
        strides=tuple(args.multi_strides),
        embedding_cache_dir=args.embedding_cache_dir,
        frames_per_clip=args.frames_per_clip,
        span_window_counts=tuple(args.span_window_counts),
        iou_thresholds=args.iou_thresholds,
        top_k=args.top_k,
        limit=args.limit,
        predictions_out=args.predictions_out,
        device=args.device,
    )
    print(json.dumps(asdict(result), indent=2))


def _run_context_explain(args: argparse.Namespace) -> None:
    from video_clip.context_reranker import localize_contextual_reranker
    from video_clip.xclip_backend import XCLIPEmbedder

    embedder = XCLIPEmbedder(args.model_name, device=args.device, batch_size=args.embed_batch_size)
    result = localize_contextual_reranker(
        args.video,
        args.query,
        base_checkpoint=args.base_checkpoint,
        checkpoint=args.checkpoint,
        embedder=embedder,
        scales=tuple(args.multi_scales),
        strides=tuple(args.multi_strides),
        embedding_cache_dir=args.embedding_cache_dir,
        frames_per_clip=args.frames_per_clip,
        span_window_counts=tuple(args.span_window_counts),
        top_k=args.top_k,
        nms_iou_threshold=args.nms_iou_threshold,
        device=args.device,
    )
    print(json.dumps(asdict(result), indent=2, default=str))


def _run_train(args: argparse.Namespace) -> None:
    from data import read_jsonl_manifest
    from video_clip.config import VideoClipTrainingConfig
    from video_clip.train import train_projection_scorer
    from video_clip.xclip_backend import XCLIPEmbedder

    examples = read_jsonl_manifest(args.manifest, video_root=args.video_root)
    if args.limit is not None:
        examples = examples[: args.limit]
    embedder = XCLIPEmbedder(args.model_name, device=args.device, batch_size=args.embed_batch_size)
    config = VideoClipTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        negative_ratio=args.negative_ratio,
        positive_iou_threshold=args.positive_iou_threshold,
        positive_window_coverage_threshold=args.positive_window_coverage_threshold,
        max_positive_windows=args.max_positive_windows,
        use_duration_feature=args.use_duration_feature,
        device=args.device,
    )
    summary = train_projection_scorer(
        examples,
        output=args.output,
        embedder=embedder,
        config=config,
        clip_seconds=args.clip_seconds,
        stride_seconds=args.stride_seconds,
        frames_per_clip=args.frames_per_clip,
        multi_scales=tuple(args.multi_scales) if args.multi_scales else None,
        multi_strides=tuple(args.multi_strides) if args.multi_strides else None,
        embedding_cache_dir=args.embedding_cache_dir,
    )
    print(json.dumps(asdict(summary), indent=2, default=str))


if __name__ == "__main__":
    main()
