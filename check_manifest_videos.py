from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from data import read_jsonl_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args()

    examples = read_jsonl_manifest(args.manifest, video_root=args.video_root)
    by_video = defaultdict(list)
    for example in examples:
        by_video[example.video_path].append(example)

    missing = []
    unreadable = []
    timestamp_problems = []
    durations = {}

    for video_path, video_examples in by_video.items():
        if video_path is None or not video_path.exists():
            missing.append(video_path)
            continue
        duration = get_video_duration(video_path)
        if duration <= 0:
            unreadable.append(video_path)
            continue
        durations[video_path] = duration
        for example in video_examples:
            if example.start < 0 or example.end <= example.start or example.end > duration + 2.0:
                timestamp_problems.append((example, duration))

    print("Manifest:", args.manifest)
    print("Video root:", args.video_root)
    print("Caption examples:", len(examples))
    print("Unique videos in manifest:", len(by_video))
    print("Missing videos:", len(missing))
    print("Unreadable videos:", len(unreadable))
    print("Timestamp problems:", len(timestamp_problems))
    print()

    print("First examples:")
    for example in examples[: args.show]:
        duration = durations.get(example.video_path)
        duration_text = f"{duration:.2f}s" if duration is not None else "unknown"
        print(f"- {example.video_path.name if example.video_path else None}")
        print(f"  caption: {example.caption}")
        print(f"  timestamp: {example.start:.2f}s - {example.end:.2f}s")
        print(f"  video duration: {duration_text}")

    if missing:
        print()
        print("Missing video files:")
        for path in missing[: args.show]:
            print("-", path)

    if timestamp_problems:
        print()
        print("Timestamp problems:")
        for example, duration in timestamp_problems[: args.show]:
            print(
                f"- {example.video_path.name}: {example.start:.2f}-{example.end:.2f}, "
                f"duration={duration:.2f}, caption={example.caption}"
            )

    suffix_counts = Counter(path.suffix.lower() for path in by_video if path is not None)
    print()
    print("Video extensions:", dict(suffix_counts))


def get_video_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return 0.0
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if fps <= 0 or frame_count <= 0:
        return 0.0
    return frame_count / fps


if __name__ == "__main__":
    main()
