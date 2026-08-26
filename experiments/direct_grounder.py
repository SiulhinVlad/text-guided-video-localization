from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from data import VideoCaptionExample, read_jsonl_manifest
from metrics import summarize_recall
from segments import TemporalSegment, temporal_iou
from video_clip.train import _load_or_encode_video_windows
from video_clip.xclip_backend import XCLIPEmbedder


class DirectTemporalGrounder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        model_dim: int = 256,
        hidden_dim: int = 256,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.video_proj = nn.Linear(embedding_dim, model_dim)
        self.text_proj = nn.Linear(embedding_dim, model_dim)
        self.mix_proj = nn.Sequential(
            nn.Linear(model_dim * 4 + 2, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.start_head = nn.Linear(model_dim, 1)
        self.end_head = nn.Linear(model_dim, 1)

    def forward(
        self,
        window_embeddings: torch.Tensor,
        text_embedding: torch.Tensor,
        *,
        video_duration: float,
        window_centers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video = self.video_proj(window_embeddings.float())
        text = self.text_proj(text_embedding.float())
        if text.ndim == 1:
            text = text.unsqueeze(0).expand_as(video)
        duration = max(float(video_duration), 1e-6)
        center = window_centers.to(video.device, dtype=video.dtype).unsqueeze(-1)
        pos = torch.cat([center / duration, torch.sin(center / duration * np.pi).to(video.dtype)], dim=-1)
        mixed = self.mix_proj(torch.cat([video, text, torch.abs(video - text), video * text, pos], dim=-1))
        context = self.encoder(mixed.unsqueeze(0)).squeeze(0)
        return self.start_head(context).squeeze(-1), self.end_head(context).squeeze(-1)


@dataclass
class GrounderState:
    embedder: XCLIPEmbedder
    clip_seconds: float
    stride_seconds: float
    frames_per_clip: int
    embedding_cache_dir: Path | None
    text_cache: dict[str, torch.Tensor]
    video_cache: OrderedDict[Path, tuple[list, torch.Tensor]]

    def text(self, caption: str) -> torch.Tensor:
        if caption not in self.text_cache:
            self.text_cache[caption] = self.embedder.encode_text(caption)[0]
        return self.text_cache[caption]

    def video(self, video_path: Path) -> tuple[list, torch.Tensor]:
        if video_path not in self.video_cache:
            self.video_cache[video_path] = _load_or_encode_video_windows(
                video_path,
                embedder=self.embedder,
                clip_seconds=self.clip_seconds,
                stride_seconds=self.stride_seconds,
                frames_per_clip=self.frames_per_clip,
                embedding_cache_dir=self.embedding_cache_dir,
            )
            if len(self.video_cache) > 12:
                self.video_cache.popitem(last=False)
        else:
            self.video_cache.move_to_end(video_path)
        return self.video_cache[video_path]


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_train_parser(subparsers)
    add_eval_parser(subparsers)
    args = parser.parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "eval":
        run_eval(args)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--model-name", default="microsoft/xclip-base-patch32")
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--embedding-cache-dir", type=Path)
    parser.add_argument("--embed-batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")


def add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train")
    add_common_args(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)


def add_eval_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("eval")
    add_common_args(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--predictions-out", type=Path)


def make_state(args: argparse.Namespace) -> GrounderState:
    return GrounderState(
        embedder=XCLIPEmbedder(args.model_name, device=args.device, batch_size=args.embed_batch_size),
        clip_seconds=args.clip_seconds,
        stride_seconds=args.stride_seconds,
        frames_per_clip=args.frames_per_clip,
        embedding_cache_dir=args.embedding_cache_dir,
        text_cache={},
        video_cache=OrderedDict(),
    )


def run_train(args: argparse.Namespace) -> None:
    examples = read_jsonl_manifest(args.manifest, video_root=args.video_root)
    if args.limit is not None:
        examples = examples[: args.limit]

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = make_state(args)
    model = DirectTemporalGrounder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    final_loss = 0.0
    seen = 0
    model.train()
    for _epoch in range(args.epochs):
        running = 0.0
        epoch_seen = 0
        for example in tqdm(examples, desc="Training direct temporal grounder"):
            if example.video_path is None:
                continue
            windows, embeddings = state.video(example.video_path)
            if not windows:
                continue
            text_embedding = state.text(example.caption).to(device)
            centers = torch.as_tensor([(window.start + window.end) / 2.0 for window in windows], dtype=torch.float32)
            start_idx = nearest_index(centers.numpy(), example.start)
            end_idx = nearest_index(centers.numpy(), example.end)
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx

            start_logits, end_logits = model(
                embeddings.to(device),
                text_embedding,
                video_duration=max((window.end for window in windows), default=1.0),
                window_centers=centers.to(device),
            )
            labels = torch.as_tensor([start_idx], dtype=torch.long, device=device)
            end_labels = torch.as_tensor([end_idx], dtype=torch.long, device=device)
            loss = F.cross_entropy(start_logits.unsqueeze(0), labels) + F.cross_entropy(
                end_logits.unsqueeze(0), end_labels
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
            epoch_seen += 1
        final_loss = running / max(epoch_seen, 1)
        seen += epoch_seen

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "metadata": {
                "backend": "direct-temporal-grounder",
                "clip_seconds": args.clip_seconds,
                "stride_seconds": args.stride_seconds,
                "frames_per_clip": args.frames_per_clip,
                "epochs": args.epochs,
                "examples": len(examples),
            },
        },
        args.output,
    )
    print(json.dumps({"examples": len(examples), "updates": seen, "final_loss": final_loss, "checkpoint": str(args.output)}, indent=2))


def run_eval(args: argparse.Namespace) -> None:
    examples = read_jsonl_manifest(args.manifest, video_root=args.video_root)
    if args.limit is not None:
        examples = examples[: args.limit]

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = make_state(args)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DirectTemporalGrounder().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    predictions: list[list[TemporalSegment]] = []
    targets: list[TemporalSegment] = []
    output_handle = None
    if args.predictions_out is not None:
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.predictions_out.open("w", encoding="utf-8")
    try:
        for example in tqdm(examples, desc="Evaluating direct temporal grounder"):
            if example.video_path is None:
                continue
            windows, embeddings = state.video(example.video_path)
            if not windows:
                continue
            text_embedding = state.text(example.caption).to(device)
            centers = torch.as_tensor([(window.start + window.end) / 2.0 for window in windows], dtype=torch.float32)
            with torch.no_grad():
                start_logits, end_logits = model(
                    embeddings.to(device),
                    text_embedding,
                    video_duration=max((window.end for window in windows), default=1.0),
                    window_centers=centers.to(device),
                )
            segments = predict_segments(windows, start_logits.cpu(), end_logits.cpu(), top_k=args.top_k)
            target = TemporalSegment(example.start, example.end, 1.0)
            predictions.append(segments)
            targets.append(target)
            if output_handle is not None:
                output_handle.write(json.dumps(prediction_row(example, segments, target)) + "\n")
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


def predict_segments(windows: list, start_logits: torch.Tensor, end_logits: torch.Tensor, *, top_k: int) -> list[TemporalSegment]:
    n = len(windows)
    start_log_probs = F.log_softmax(start_logits, dim=0)
    end_log_probs = F.log_softmax(end_logits, dim=0)
    scores = start_log_probs[:, None] + end_log_probs[None, :]
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=0)
    scores = scores.masked_fill(~mask, -1e9)
    flat_scores, flat_indices = torch.topk(scores.flatten(), k=min(max(top_k * 8, top_k), n * n))
    segments: list[TemporalSegment] = []
    for score, flat_idx in zip(flat_scores.tolist(), flat_indices.tolist(), strict=True):
        start_idx = flat_idx // n
        end_idx = flat_idx % n
        segment = TemporalSegment(windows[start_idx].start, windows[end_idx].end, float(score))
        if all(temporal_iou(segment, existing) < 0.5 for existing in segments):
            segments.append(segment)
        if len(segments) >= top_k:
            break
    return segments


def prediction_row(example: VideoCaptionExample, predictions: list[TemporalSegment], target: TemporalSegment) -> dict:
    rows = [
        {"start": item.start, "end": item.end, "score": item.score, "iou_with_ground_truth": temporal_iou(item, target)}
        for item in predictions
    ]
    return {
        "video_id": example.video_id,
        "video_path": str(example.video_path) if example.video_path else None,
        "caption": example.caption,
        "ground_truth": {"start": target.start, "end": target.end, "duration": target.duration},
        "predictions": rows,
        "best_iou": max((item["iou_with_ground_truth"] for item in rows), default=0.0),
    }


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - float(target))))


if __name__ == "__main__":
    main()
