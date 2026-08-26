from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from data import load_activitynet_captions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--caption-splits", nargs="+", default=["val1", "val2"])
    parser.add_argument("--fiftyone-dir", type=Path, default=Path("data/fiftyone"))
    parser.add_argument("--video-root", type=Path, default=Path("data/videos"))
    parser.add_argument("--manifest-out", type=Path, default=Path("data/activitynet_fiftyone.jsonl"))
    parser.add_argument("--exclude-video-root", type=Path, action="append", default=[])
    parser.add_argument("--manifest-mode", choices=["write", "append"], default="write")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import fiftyone as fo
    import fiftyone.zoo as foz

    args.fiftyone_dir.mkdir(parents=True, exist_ok=True)
    fo.config.dataset_zoo_dir = str(args.fiftyone_dir.resolve())

    dataset = foz.load_zoo_dataset(
        "activitynet-200",
        split=args.split,
        max_samples=args.max_samples,
        shuffle=True,
        seed=args.seed,
    )

    args.video_root.mkdir(parents=True, exist_ok=True)
    excluded_videos = existing_video_names(args.exclude_video_root)
    downloaded_videos = copy_videos(dataset, args.video_root, exclude_names=excluded_videos)
    captions = load_captions_for_videos(downloaded_videos, args.caption_splits)
    write_manifest(args.manifest_out, captions, mode=args.manifest_mode)

    print(
        json.dumps(
            {
                "downloaded_videos": len(downloaded_videos),
                "excluded_videos": len(excluded_videos),
                "caption_examples": len(captions),
                "fiftyone_dir": str(args.fiftyone_dir),
                "video_root": str(args.video_root),
                "manifest": str(args.manifest_out),
            },
            indent=2,
        )
    )


def existing_video_names(video_roots: list[Path]) -> set[str]:
    names: set[str] = set()
    for video_root in video_roots:
        if video_root.exists():
            names.update(path.name for path in video_root.glob("*.mp4"))
    return names


def copy_videos(dataset, video_root: Path, *, exclude_names: set[str] | None = None) -> set[str]:
    exclude_names = exclude_names or set()
    video_names: set[str] = set()
    for sample in dataset:
        source = Path(sample.filepath)
        video_name = source.name
        if video_name in exclude_names:
            continue
        destination = video_root / video_name
        if not destination.exists():
            shutil.copy2(source, destination)
        video_names.add(video_name)
    return video_names


def load_captions_for_videos(video_names: set[str], caption_splits: list[str]) -> list[dict]:
    rows: list[dict] = []
    for split in caption_splits:
        for example in load_activitynet_captions(split):
            if example.video_path is None:
                continue
            if example.video_path.name not in video_names:
                continue
            rows.append(
                {
                    "video_id": example.video_id,
                    "video_path": example.video_path.name,
                    "caption": example.caption,
                    "start": example.start,
                    "end": example.end,
                }
            )
    return rows


def write_manifest(path: Path, rows: list[dict], *, mode: str = "write") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if mode == "append" else "w"
    with path.open(open_mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
