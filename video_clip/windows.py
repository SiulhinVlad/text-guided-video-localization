from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)


@dataclass
class VideoWindow:
    frames: list[Image.Image]
    start: float
    end: float

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0


def sample_video_windows(
    video_path: Path,
    *,
    clip_seconds: float = 4.0,
    stride_seconds: float = 2.0,
    frames_per_clip: int = 8,
) -> list[VideoWindow]:
    if clip_seconds <= 0:
        raise ValueError("clip_seconds must be positive")
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")
    if frames_per_clip <= 0:
        raise ValueError("frames_per_clip must be positive")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or frame_count <= 0:
        capture.release()
        return []

    duration = frame_count / fps
    starts = list(np.arange(0.0, max(duration - clip_seconds, 0.0) + 1e-6, stride_seconds))
    if not starts:
        starts = [0.0]

    windows: list[VideoWindow] = []
    try:
        for start in starts:
            end = min(start + clip_seconds, duration)
            frames = _read_window_frames(capture, fps, start, end, frames_per_clip)
            if frames:
                windows.append(VideoWindow(frames=frames, start=float(start), end=float(end)))
    finally:
        capture.release()
    return windows


def sample_video_window_spans(
    video_path: Path,
    *,
    clip_seconds: float = 4.0,
    stride_seconds: float = 2.0,
) -> list[tuple[float, float]]:
    if clip_seconds <= 0:
        raise ValueError("clip_seconds must be positive")
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            return []
        duration = frame_count / fps
    finally:
        capture.release()

    starts = list(np.arange(0.0, max(duration - clip_seconds, 0.0) + 1e-6, stride_seconds))
    if not starts:
        starts = [0.0]
    return [(float(start), float(min(start + clip_seconds, duration))) for start in starts]


def _read_window_frames(
    capture: cv2.VideoCapture,
    fps: float,
    start: float,
    end: float,
    frames_per_clip: int,
) -> list[Image.Image]:
    if end <= start:
        end = start + 1.0 / fps
    times = np.linspace(start, end, frames_per_clip, endpoint=False)
    frames: list[Image.Image] = []
    for time_sec in times:
        frame_index = max(0, int(round(time_sec * fps)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = capture.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    if not frames:
        return []
    while len(frames) < frames_per_clip:
        frames.append(frames[-1].copy())
    return frames[:frames_per_clip]
