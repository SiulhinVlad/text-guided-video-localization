from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from data import VideoCaptionExample
from metrics import summarize_recall
from scoring import ProjectionScorer, load_scorer, save_scorer
from segments import TemporalSegment, non_max_suppression, temporal_iou
from video_clip.segment_reranker import (
    SegmentCandidate,
    SegmentEvaluationResult,
    _prediction_row,
    _video_duration,
    generate_segment_candidates,
    segment_embeddings,
    segment_features,
)
from video_clip.train import _load_or_encode_video_windows
from video_clip.windows import VideoWindow
from video_clip.xclip_backend import XCLIPEmbedder


@dataclass
class ListwiseTrainingSummary:
    examples: int
    groups: int
    candidates: int
    final_loss: float
    checkpoint: Path


@dataclass
class CandidatePool:
    candidates: list[SegmentCandidate]
    embeddings: torch.Tensor
    text_embedding: torch.Tensor
    features: torch.Tensor
    base_scores: torch.Tensor
    ious: torch.Tensor | None = None


def train_listwise_reranker(
    examples: list[VideoCaptionExample],
    *,
    output: Path,
    base_checkpoint: Path,
    embedder: XCLIPEmbedder,
    scales: tuple[float, ...],
    strides: tuple[float, ...],
    embedding_cache_dir: Path | None,
    frames_per_clip: int = 8,
    span_window_counts: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
    candidate_pool_size: int = 128,
    oracle_pool_size: int = 16,
    target_temperature: float = 0.08,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str | None = None,
) -> ListwiseTrainingSummary:
    if len(scales) != len(strides):
        raise ValueError("scales and strides must have matching lengths")

    model_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    base_scorer = load_scorer(base_checkpoint, map_location=model_device).to(model_device)
    base_scorer.eval()
    groups, targets = build_listwise_training_groups(
        examples,
        base_scorer=base_scorer,
        embedder=embedder,
        scales=scales,
        strides=strides,
        embedding_cache_dir=embedding_cache_dir,
        frames_per_clip=frames_per_clip,
        span_window_counts=span_window_counts,
        candidate_pool_size=candidate_pool_size,
        oracle_pool_size=oracle_pool_size,
        target_temperature=target_temperature,
        device=model_device,
    )
    if not groups:
        raise ValueError("No listwise training groups were produced.")

    model = ProjectionScorer(
        embedding_dim=groups[0].embeddings.shape[-1],
        projection_dim=256,
        hidden_dim=256,
        extra_feature_dim=groups[0].features.shape[-1],
    ).to(model_device)
    initialize_from_base_scorer(model, base_scorer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    final_loss = 0.0
    rng = np.random.default_rng(23)
    model.train()
    for _epoch in range(epochs):
        order = rng.permutation(len(groups))
        running = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.tensor(0.0, device=model_device)
            group_count = 0
            for group_index in order[start : start + batch_size]:
                group = groups[int(group_index)]
                target = targets[int(group_index)].to(model_device)
                logits = model(
                    group.embeddings.to(model_device),
                    group.text_embedding.to(model_device),
                    group.features.to(model_device),
                )
                loss = loss + F.kl_div(
                    F.log_softmax(logits, dim=0),
                    target,
                    reduction="sum",
                )
                group_count += 1
            loss = loss / max(group_count, 1)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * group_count
            seen += group_count
        final_loss = running / max(seen, 1)

    save_scorer(
        output,
        model.cpu(),
        metadata={
            "backend": "xclip-listwise-reranker",
            "base_checkpoint": str(base_checkpoint),
            "scales": list(scales),
            "strides": list(strides),
            "frames_per_clip": frames_per_clip,
            "span_window_counts": list(span_window_counts),
            "candidate_pool_size": candidate_pool_size,
            "oracle_pool_size": oracle_pool_size,
            "target_temperature": target_temperature,
            "examples": len(examples),
            "groups": len(groups),
            "candidates": int(sum(group.embeddings.shape[0] for group in groups)),
        },
    )
    return ListwiseTrainingSummary(
        examples=len(examples),
        groups=len(groups),
        candidates=int(sum(group.embeddings.shape[0] for group in groups)),
        final_loss=final_loss,
        checkpoint=output,
    )


def evaluate_listwise_reranker(
    examples: list[VideoCaptionExample],
    *,
    base_checkpoint: Path,
    checkpoint: Path,
    embedder: XCLIPEmbedder,
    scales: tuple[float, ...],
    strides: tuple[float, ...],
    embedding_cache_dir: Path | None,
    frames_per_clip: int,
    span_window_counts: tuple[int, ...],
    iou_thresholds: list[float],
    candidate_pool_size: int = 128,
    top_k: int = 5,
    limit: int | None = None,
    predictions_out: Path | None = None,
    device: str | None = None,
) -> SegmentEvaluationResult:
    selected = examples if limit is None else examples[:limit]
    model_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    base_scorer = load_scorer(base_checkpoint, map_location=model_device).to(model_device)
    base_scorer.eval()
    listwise_scorer = load_scorer(checkpoint, map_location=model_device).to(model_device)
    listwise_scorer.eval()

    state = CandidateState()
    predictions: list[list[TemporalSegment]] = []
    targets: list[TemporalSegment] = []
    skipped = 0

    output_handle = None
    if predictions_out is not None:
        predictions_out.parent.mkdir(parents=True, exist_ok=True)
        output_handle = predictions_out.open("w", encoding="utf-8")

    try:
        for example in tqdm(selected, desc="Evaluating listwise reranker"):
            if example.video_path is None:
                skipped += 1
                continue
            text_embedding = state.text_embedding(example.caption, embedder)
            pool = collect_candidate_pool(
                example,
                base_scorer=base_scorer,
                embedder=embedder,
                text_embedding=text_embedding,
                scales=scales,
                strides=strides,
                embedding_cache_dir=embedding_cache_dir,
                frames_per_clip=frames_per_clip,
                span_window_counts=span_window_counts,
                candidate_pool_size=candidate_pool_size,
                oracle_pool_size=0,
                include_ious=False,
                state=state,
                device=model_device,
            )
            scored = score_listwise_pool(
                listwise_scorer,
                pool,
                text_embedding,
                device=model_device,
            )
            final_segments = non_max_suppression(scored, iou_threshold=0.5, top_k=top_k)
            target = TemporalSegment(example.start, example.end, 1.0)
            predictions.append(final_segments)
            targets.append(target)
            if output_handle is not None:
                output_handle.write(json.dumps(_prediction_row(example, final_segments, target)) + "\n")
    finally:
        if output_handle is not None:
            output_handle.close()

    return SegmentEvaluationResult(
        metrics=summarize_recall(predictions, targets, iou_thresholds),
        evaluated=len(targets),
        skipped=skipped,
    )


def build_listwise_training_groups(
    examples: list[VideoCaptionExample],
    *,
    base_scorer: ProjectionScorer,
    embedder: XCLIPEmbedder,
    scales: tuple[float, ...],
    strides: tuple[float, ...],
    embedding_cache_dir: Path | None,
    frames_per_clip: int,
    span_window_counts: tuple[int, ...],
    candidate_pool_size: int,
    oracle_pool_size: int,
    target_temperature: float,
    device: torch.device,
) -> tuple[list[CandidatePool], list[torch.Tensor]]:
    state = CandidateState()
    groups: list[CandidatePool] = []
    targets: list[torch.Tensor] = []
    for example in tqdm(examples, desc="Building listwise reranker groups"):
        if example.video_path is None:
            continue
        text_embedding = state.text_embedding(example.caption, embedder)
        pool = collect_candidate_pool(
            example,
            base_scorer=base_scorer,
            embedder=embedder,
            text_embedding=text_embedding,
            scales=scales,
            strides=strides,
            embedding_cache_dir=embedding_cache_dir,
            frames_per_clip=frames_per_clip,
            span_window_counts=span_window_counts,
            candidate_pool_size=candidate_pool_size,
            oracle_pool_size=oracle_pool_size,
            include_ious=True,
            state=state,
            device=device,
        )
        if pool.ious is None or pool.embeddings.numel() == 0:
            continue
        target = F.softmax(pool.ious.float() / max(target_temperature, 1e-6), dim=0)
        pool.ious = None
        groups.append(pool)
        targets.append(target)
    return groups, targets


class CandidateState:
    def __init__(self) -> None:
        self.video_cache: dict[tuple[Path, float, float], tuple[list[VideoWindow], torch.Tensor]] = {}
        self.candidate_cache: dict[tuple[Path, float, float], list[SegmentCandidate]] = {}
        self.text_cache: dict[str, torch.Tensor] = {}
        self.group_text_cache: dict[int, torch.Tensor] = {}

    def text_embedding(self, caption: str, embedder: XCLIPEmbedder) -> torch.Tensor:
        if caption not in self.text_cache:
            self.text_cache[caption] = embedder.encode_text(caption)[0]
        return self.text_cache[caption]


def collect_candidate_pool(
    example: VideoCaptionExample,
    *,
    base_scorer: ProjectionScorer,
    embedder: XCLIPEmbedder,
    text_embedding: torch.Tensor,
    scales: tuple[float, ...],
    strides: tuple[float, ...],
    embedding_cache_dir: Path | None,
    frames_per_clip: int,
    span_window_counts: tuple[int, ...],
    candidate_pool_size: int,
    oracle_pool_size: int,
    include_ious: bool,
    state: CandidateState,
    device: torch.device,
) -> CandidatePool:
    rows: list[tuple[SegmentCandidate, torch.Tensor, torch.Tensor, float, float]] = []
    target = TemporalSegment(example.start, example.end, 1.0)
    for scale, stride in zip(scales, strides, strict=True):
        cache_key = (example.video_path, float(scale), float(stride))
        if cache_key not in state.video_cache:
            state.video_cache[cache_key] = _load_or_encode_video_windows(
                example.video_path,
                embedder=embedder,
                clip_seconds=scale,
                stride_seconds=stride,
                frames_per_clip=frames_per_clip,
                embedding_cache_dir=embedding_cache_dir,
            )
        windows, embeddings = state.video_cache[cache_key]
        if cache_key not in state.candidate_cache:
            state.candidate_cache[cache_key] = generate_segment_candidates(
                windows,
                scale=scale,
                span_window_counts=span_window_counts,
            )
        candidates = state.candidate_cache[cache_key]
        if not candidates:
            continue
        video_duration = _video_duration(windows)
        segment_rows = segment_embeddings(embeddings, candidates)
        feature_rows = segment_features(
            candidates,
            video_duration=video_duration,
            embeddings=embeddings,
            text_embedding=text_embedding,
            use_score_features=base_scorer.extra_feature_dim > 4,
        )
        base_scores = base_scorer.predict_proba(
            segment_rows.to(device),
            text_embedding.to(device),
            feature_rows.to(device),
        ).cpu()
        feature_rows = torch.cat([feature_rows, base_scores.unsqueeze(1)], dim=1)
        ious = [
            temporal_iou(TemporalSegment(candidate.start, candidate.end, 1.0), target)
            if include_ious
            else 0.0
            for candidate in candidates
        ]
        for candidate, embedding, features, score, iou in zip(
            candidates,
            segment_rows,
            feature_rows,
            base_scores,
            ious,
            strict=True,
        ):
            rows.append((candidate, embedding, features, float(score), float(iou)))

    if not rows:
        return CandidatePool(
            [],
            torch.empty(0, 512),
            text_embedding,
            torch.empty(0, 9),
            torch.empty(0),
            torch.empty(0) if include_ious else None,
        )

    score_order = np.argsort([row[3] for row in rows])[::-1][:candidate_pool_size]
    selected_indices = set(int(idx) for idx in score_order)
    if include_ious and oracle_pool_size > 0:
        iou_order = np.argsort([row[4] for row in rows])[::-1][:oracle_pool_size]
        selected_indices.update(int(idx) for idx in iou_order)
    selected = [rows[idx] for idx in sorted(selected_indices)]

    return CandidatePool(
        candidates=[row[0] for row in selected],
        embeddings=torch.stack([row[1] for row in selected]),
        text_embedding=text_embedding,
        features=torch.stack([row[2] for row in selected]),
        base_scores=torch.as_tensor([row[3] for row in selected], dtype=torch.float32),
        ious=torch.as_tensor([row[4] for row in selected], dtype=torch.float32) if include_ious else None,
    )


def score_listwise_pool(
    scorer: ProjectionScorer,
    pool: CandidatePool,
    text_embedding: torch.Tensor,
    *,
    device: torch.device,
) -> list[TemporalSegment]:
    if not pool.candidates:
        return []
    with torch.no_grad():
        logits = scorer(
            pool.embeddings.to(device),
            text_embedding.to(device),
            pool.features.to(device),
        )
        scores = torch.softmax(logits, dim=0).cpu()
    return [
        TemporalSegment(candidate.start, candidate.end, float(score))
        for candidate, score in zip(pool.candidates, scores, strict=True)
    ]


def initialize_from_base_scorer(model: ProjectionScorer, base_scorer: ProjectionScorer) -> None:
    model.frame_proj.load_state_dict(base_scorer.frame_proj.state_dict())
    model.text_proj.load_state_dict(base_scorer.text_proj.state_dict())

    base_first = base_scorer.score_head[0]
    model_first = model.score_head[0]
    shared_inputs = min(base_first.in_features, model_first.in_features)
    shared_outputs = min(base_first.out_features, model_first.out_features)
    with torch.no_grad():
        model_first.weight[:shared_outputs, :shared_inputs].copy_(
            base_first.weight[:shared_outputs, :shared_inputs]
        )
        model_first.bias[:shared_outputs].copy_(base_first.bias[:shared_outputs])
        if len(model.score_head) >= 4 and len(base_scorer.score_head) >= 4:
            model_last = model.score_head[3]
            base_last = base_scorer.score_head[3]
            shared_hidden = min(base_last.in_features, model_last.in_features)
            model_last.weight[:, :shared_hidden].copy_(base_last.weight[:, :shared_hidden])
            model_last.bias.copy_(base_last.bias)
