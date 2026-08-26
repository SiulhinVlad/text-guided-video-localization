from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image
from torch.nn import functional as F
from transformers import XCLIPModel, XCLIPProcessor


class XCLIPEmbedder:
    def __init__(
        self,
        model_name: str = "microsoft/xclip-base-patch32",
        *,
        device: str | None = None,
        batch_size: int = 4,
    ):
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.processor = XCLIPProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = XCLIPModel.from_pretrained(
            model_name, use_safetensors=True, local_files_only=True
        ).to(self.device)
        self.model.config.return_dict = True
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, texts: str | Sequence[str]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
        inputs = _to_device(inputs, self.device)
        embeddings = self.model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        return F.normalize(embeddings, dim=-1).cpu()

    @torch.no_grad()
    def encode_videos(self, clips: Sequence[list[Image.Image]]) -> torch.Tensor:
        if not clips:
            return torch.empty(0, self.model.config.projection_dim)
        batches = []
        for start in range(0, len(clips), self.batch_size):
            batch = clips[start : start + self.batch_size]
            inputs = self.processor(images=list(batch), return_tensors="pt")
            inputs = _to_device(inputs, self.device)
            embeddings = self._get_video_features(inputs["pixel_values"])
            batches.append(F.normalize(embeddings, dim=-1).cpu())
        return torch.cat(batches, dim=0)

    def _get_video_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, num_channels, height, width)
        vision_outputs = self.model.vision_model(pixel_values=pixel_values, return_dict=True)
        frame_embeddings = self.model.visual_projection(vision_outputs.pooler_output)
        frame_embeddings = frame_embeddings.view(batch_size, num_frames, -1)
        video_outputs = self.model.mit(frame_embeddings, return_dict=True)
        return video_outputs.pooler_output


def _to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
