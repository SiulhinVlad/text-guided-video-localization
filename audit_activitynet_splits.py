from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="append", nargs=2, metavar=("NAME", "MANIFEST"), required=True)
    parser.add_argument("--video-root", action="append", nargs=2, metavar=("NAME", "VIDEO_ROOT"), default=[])
    args = parser.parse_args()

    split_rows = {name: read_jsonl(Path(path)) for name, path in args.split}
    split_videos = {name: {row["video_id"] for row in rows} for name, rows in split_rows.items()}

    report = {
        "splits": {
            name: {
                "caption_rows": len(rows),
                "unique_video_ids": len(split_videos[name]),
                "duration_seconds": duration_summary(rows),
            }
            for name, rows in split_rows.items()
        },
        "video_roots": {
            name: {"mp4_files": count_mp4(Path(path))}
            for name, path in args.video_root
        },
        "overlaps": overlaps(split_videos),
    }
    print(json.dumps(report, indent=2))


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def duration_summary(rows: list[dict]) -> dict:
    durations = sorted(float(row["end"]) - float(row["start"]) for row in rows)
    if not durations:
        return {"min": None, "median": None, "max": None}
    return {
        "min": round(durations[0], 3),
        "median": round(durations[len(durations) // 2], 3),
        "max": round(durations[-1], 3),
    }


def count_mp4(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob("*.mp4"))


def overlaps(split_videos: dict[str, set[str]]) -> dict[str, int]:
    names = list(split_videos)
    result: dict[str, int] = {}
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            result[f"{left_name}__{right_name}"] = len(split_videos[left_name] & split_videos[right_name])
    return result


if __name__ == "__main__":
    main()
