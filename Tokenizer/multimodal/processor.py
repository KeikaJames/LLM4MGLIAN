# -*- coding: utf-8 -*-
"""Multimodal tokenizer processor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from Tokenizer.unified.encoded import EncodedToken

from .image_placeholders import (
    image_patch_count,
    image_placeholder_tokens,
)
from .video_placeholders import (
    video_patch_count,
    video_placeholder_tokens,
)
from .tokens import IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER


@dataclass
class MultimodalEncoding:
    input_ids: list[int]
    attention_mask: list[int]
    tokens: list[EncodedToken]
    image_token_spans: list[tuple[int, int]]
    video_token_spans: list[tuple[int, int]] = field(default_factory=list)
    images: list[Any] = field(default_factory=list)
    videos: list[Any] = field(default_factory=list)
    pixel_values: Any = None


class MultimodalProcessor:
    def __init__(
        self,
        tokenizer,
        image_processor=None,
        patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size

    def __call__(
        self,
        text: str,
        images: list[Any] | None = None,
        image_sizes: list[Any] | None = None,
        videos: list[Any] | None = None,
        video_sizes: list[Any] | None = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> MultimodalEncoding:
        image_list = list(images or [])
        sizes = list(image_sizes or [])
        video_list = list(videos or [])
        vsizes = list(video_sizes or [])

        n_img_holes = text.count(IMAGE_PLACEHOLDER)
        n_vid_holes = text.count(VIDEO_PLACEHOLDER)

        if len(image_list) != n_img_holes:
            raise ValueError(
                f"<image> placeholder count ({n_img_holes}) does not match "
                f"images length ({len(image_list)})"
            )
        if len(video_list) != n_vid_holes:
            raise ValueError(
                f"<video> placeholder count ({n_vid_holes}) does not match "
                f"videos length ({len(video_list)})"
            )
        if sizes and len(sizes) != n_img_holes:
            raise ValueError(
                f"<image> placeholder count ({n_img_holes}) does not match "
                f"image_sizes length ({len(sizes)})"
            )
        if vsizes and len(vsizes) != n_vid_holes:
            raise ValueError(
                f"<video> placeholder count ({n_vid_holes}) does not match "
                f"video_sizes length ({len(vsizes)})"
            )

        # Default sizes for missing entries: assume single-patch image/video.
        if not sizes and n_img_holes:
            sizes = [None] * n_img_holes
        if not vsizes and n_vid_holes:
            vsizes = [None] * n_vid_holes

        expanded = self._expand_images(text, sizes)
        expanded = self._expand_videos(expanded, vsizes)

        result = self.tokenizer.encode_with_spans(
            expanded, add_bos=add_bos, add_eos=add_eos
        )
        tokens = self._annotate_patches(result.tokens)
        img_spans = self._collect_spans(tokens, "<image_start>", "<image_end>")
        vid_spans = self._collect_spans(tokens, "<video_start>", "<video_end>")

        # Invariant: attention_mask aligns with input_ids
        assert len(result.attention_mask) == len(result.input_ids)

        processed_images = image_list
        if self.image_processor is not None and image_list:
            processed_images = self.image_processor(image_list)

        return MultimodalEncoding(
            input_ids=result.input_ids,
            attention_mask=result.attention_mask,
            tokens=tokens,
            image_token_spans=img_spans,
            video_token_spans=vid_spans,
            images=processed_images,
            videos=video_list,
        )

    # ---- expansion ----

    def _expand_images(self, text: str, sizes: list[Any]) -> str:
        parts = text.split(IMAGE_PLACEHOLDER)
        if len(parts) == 1:
            return text
        out = [parts[0]]
        for i, size in enumerate(sizes):
            n = self._patches_image(size)
            out.append("".join(image_placeholder_tokens(n)))
            out.append(parts[i + 1])
        return "".join(out)

    def _expand_videos(self, text: str, sizes: list[Any]) -> str:
        parts = text.split(VIDEO_PLACEHOLDER)
        if len(parts) == 1:
            return text
        out = [parts[0]]
        for i, size in enumerate(sizes):
            n = self._patches_video(size)
            out.append("".join(video_placeholder_tokens(n)))
            out.append(parts[i + 1])
        return "".join(out)

    def _patches_image(self, size: Any) -> int:
        if size is None:
            return 1
        if isinstance(size, dict):
            w = int(size.get("width") or size.get("w"))
            h = int(size.get("height") or size.get("h"))
        else:
            w, h = int(size[0]), int(size[1])
        return image_patch_count(w, h, self.patch_size, self.merge_size)

    def _patches_video(self, size: Any) -> int:
        if size is None:
            return 1
        if isinstance(size, dict):
            f = int(size.get("frames") or size.get("num_frames"))
            w = int(size.get("width") or size.get("w"))
            h = int(size.get("height") or size.get("h"))
        else:
            f, w, h = int(size[0]), int(size[1]), int(size[2])
        return video_patch_count(
            f, w, h, self.patch_size, self.temporal_patch_size, self.merge_size
        )

    # ---- annotation & spans ----

    def _annotate_patches(self, tokens: list[EncodedToken]) -> list[EncodedToken]:
        """Attach image_index/video_index metadata to patch tokens."""
        out: list[EncodedToken] = []
        img_idx = -1
        vid_idx = -1
        in_img = False
        in_vid = False
        for t in tokens:
            meta = t.metadata
            if t.token == "<image_start>":
                img_idx += 1
                in_img = True
                meta = {**(meta or {}), "image_index": img_idx}
            elif t.token == "<image_end>":
                meta = {**(meta or {}), "image_index": img_idx}
                in_img = False
            elif t.token == "<image_patch>" and in_img:
                meta = {**(meta or {}), "image_index": img_idx}
            elif t.token == "<video_start>":
                vid_idx += 1
                in_vid = True
                meta = {**(meta or {}), "video_index": vid_idx}
            elif t.token == "<video_end>":
                meta = {**(meta or {}), "video_index": vid_idx}
                in_vid = False
            elif t.token == "<video_patch>" and in_vid:
                meta = {**(meta or {}), "video_index": vid_idx}
            if meta is not t.metadata:
                out.append(
                    EncodedToken(
                        t.id, t.token, t.track, t.start, t.end, t.surface, meta
                    )
                )
            else:
                out.append(t)
        return out

    def _collect_spans(
        self, tokens: list[EncodedToken], start_tok: str, end_tok: str
    ) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        start: int | None = None
        for idx, t in enumerate(tokens):
            if t.token == start_tok:
                start = idx
            elif t.token == end_tok and start is not None:
                spans.append((start, idx + 1))
                # Validate that internal patch count matches inclusive span
                # length minus 2 (start/end markers themselves)
                start = None
        return spans
