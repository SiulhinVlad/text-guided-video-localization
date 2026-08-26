from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoCaptionExample:
    video_id: str
    caption: str
    start: float
    end: float
    video_path: Path | None = None


def read_jsonl_manifest(path: Path, *, video_root: Path | None = None) -> list[VideoCaptionExample]:
    examples: list[VideoCaptionExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                examples.append(_example_from_row(row, video_root=video_root))
            except KeyError as exc:
                raise ValueError(f"Missing required key {exc!s} on line {line_no}") from exc
    return examples


def write_jsonl_manifest(path: Path, examples: Iterable[VideoCaptionExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(
                json.dumps(
                    {
                        "video_id": example.video_id,
                        "video_path": str(example.video_path) if example.video_path else None,
                        "caption": example.caption,
                        "start": example.start,
                        "end": example.end,
                    }
                )
                + "\n"
            )


def load_activitynet_captions(split: str = "train") -> Iterator[VideoCaptionExample]:
    from datasets import load_dataset

    dataset = load_dataset("friedrichor/ActivityNet_Captions", split=split)
    for row in dataset:
        yield from _activitynet_row_to_examples(row)


def _example_from_row(row: dict, *, video_root: Path | None) -> VideoCaptionExample:
    video_id = str(row["video_id"])
    video_path = Path(row["video_path"])
    if not video_path.is_absolute() and video_root is not None:
        video_path = video_root / video_path
    return VideoCaptionExample(
        video_id=video_id,
        video_path=video_path,
        caption=str(row["caption"]),
        start=float(row["start"]),
        end=float(row["end"]),
    )


def _activitynet_row_to_examples(row: dict) -> Iterator[VideoCaptionExample]:
    video_id = str(row["video_id"])
    video_path = Path(row["video"])
    captions = row["sentences"]
    timestamps = row["timestamps"]

    for idx, (caption, timestamp) in enumerate(zip(captions, timestamps, strict=False)):
        if len(timestamp) < 2:
            continue
        yield VideoCaptionExample(
            video_id=video_id,
            video_path=video_path,
            caption=str(caption),
            start=float(timestamp[0]),
            end=float(timestamp[1]),
        )
