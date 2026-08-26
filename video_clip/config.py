from dataclasses import dataclass


@dataclass
class VideoClipConfig:
    model_name: str = "microsoft/xclip-base-patch32"
    clip_seconds: float = 4.0
    stride_seconds: float = 2.0
    frames_per_clip: int = 8
    batch_size: int = 4
    score_threshold: float | str = "auto"
    min_segment_seconds: float = 1.0
    merge_gap_seconds: float = 2.0
    smoothing_window: int = 3
    top_k: int = 3
    device: str | None = None
    embedding_cache_dir: str | None = None
    multi_scales: tuple[float, ...] | None = None
    multi_strides: tuple[float, ...] | None = None
    multi_merge_gaps: tuple[float, ...] | None = None


@dataclass
class VideoClipTrainingConfig:
    embedding_dim: int = 512
    projection_dim: int = 256
    hidden_dim: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 3
    batch_size: int = 128
    negative_ratio: int = 3
    positive_iou_threshold: float = 0.5
    positive_window_coverage_threshold: float = 0.5
    max_positive_windows: int | None = None
    use_duration_feature: bool = False
    device: str | None = None
