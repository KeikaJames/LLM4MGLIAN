# -*- coding: utf-8 -*-

"""Shared multimodal CLI helpers for training entry points.

Centralises the construction of :class:`PILImageProcessor` /
:class:`OMVTConfig` so that ``train_rdt`` / ``train_vlm_align`` /
``train_omvt_ssl`` expose a consistent ``--multimodal`` + ``--image-size``
+ ``--n-image-tokens`` surface and remain in lockstep on defaults.
"""

from __future__ import annotations

import argparse
from typing import Any

from Model.config import OMVTConfig


def add_multimodal_args(p: argparse.ArgumentParser, *, default_image_size: int = 56) -> None:
    """Register the standard multimodal flags on a CLI parser."""

    p.add_argument(
        "--multimodal",
        action="store_true",
        help="enable pixel-aware path: loads images from JSONL rows via PIL",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=default_image_size,
        help="square image edge fed to OMVT (must be divisible by 4)",
    )
    p.add_argument(
        "--n-image-tokens",
        type=int,
        default=8,
        help="OMVT compress_to: number of <image_patch> slots per image",
    )
    p.add_argument(
        "--d-vision",
        type=int,
        default=64,
        help="OMVT hidden width (also the projector input dim)",
    )


def build_image_processor(args: argparse.Namespace) -> Any | None:
    """Construct a :class:`PILImageProcessor`, deferring the import.

    Returns ``None`` when ``--multimodal`` is not set so call sites can
    short-circuit. PIL is imported lazily to keep text-only runs free of
    the Pillow dependency.
    """

    if not getattr(args, "multimodal", False):
        return None
    from Tokenizer.multimodal import PILImageProcessor  # local import

    return PILImageProcessor(image_size=args.image_size)


def build_omvt_cfg(args: argparse.Namespace) -> OMVTConfig | None:
    """Construct a balanced :class:`OMVTConfig` from CLI args.

    The four patch grids cover the canonical OMVT layout: a half-half
    vertical/horizontal split, a quarter-square grid, and a single
    layout-level macro patch. Tweak via dataclass replace at the call
    site if a specialised tower geometry is needed.
    """

    if not getattr(args, "multimodal", False):
        return None
    s = args.image_size
    if s <= 0 or s % 4 != 0:
        raise ValueError(f"--image-size must be a positive multiple of 4, got {s}")
    return OMVTConfig(
        image_size=s,
        d_vision=args.d_vision,
        vertical_patch=(s // 2, s // 4),
        horizontal_patch=(s // 4, s // 2),
        square_patch=(s // 4, s // 4),
        layout_patch=(s, s),
        compress_to=args.n_image_tokens,
    )


__all__ = ["add_multimodal_args", "build_image_processor", "build_omvt_cfg"]
