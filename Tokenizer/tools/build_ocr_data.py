# -*- coding: utf-8 -*-

"""Build a multimodal JSONL shard from paired ``{image, label}`` files.

Walks a directory and pairs each ``<stem>.{png,jpg,jpeg,webp,bmp,tif,tiff}``
with an optional sibling ``<stem>.txt`` (plain UTF-8 text) or ``<stem>.json``
(``{"text": ..., "ocr_labels": [...], "reading_order": [...]}``). The output
JSONL conforms to the multimodal pretraining schema consumed by
``PretrainingCollator`` and the OMVT SSL trainer:

.. code-block:: json

    {
      "text": "<image> ...",
      "images": ["/abs/path/page0001.png"],
      "image_sizes": [[H, W]],
      "ocr_labels": [[int, int, ...]],
      "reading_order": [[int, int, ...]]
    }

Run ``--demo`` to materialise a 4-image synthetic dataset for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _iter_pairs(root: Path) -> Iterable[tuple[Path, Path | None]]:
    """Yield ``(image_path, label_path or None)`` for every image under ``root``."""

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in _IMAGE_EXTS:
            continue
        for ext in (".json", ".txt"):
            label = path.with_suffix(ext)
            if label.exists():
                yield path, label
                break
        else:
            yield path, None


def _row_from_pair(image: Path, label: Path | None, *, image_token: str) -> dict:
    """Compose one JSONL row from an image + optional sibling label file."""

    from PIL import Image

    with Image.open(image) as im:
        w, h = im.size
    row: dict = {
        "text": f"{image_token} ",
        "images": [str(image.resolve())],
        "image_sizes": [[int(h), int(w)]],
    }
    if label is None:
        return row
    if label.suffix == ".txt":
        text = label.read_text(encoding="utf-8").strip()
        row["text"] = f"{image_token} {text}"
        return row
    payload = json.loads(label.read_text(encoding="utf-8"))
    if "text" in payload:
        row["text"] = f"{image_token} {payload['text']}"
    if "ocr_labels" in payload:
        row["ocr_labels"] = [list(map(int, payload["ocr_labels"]))]
    if "reading_order" in payload:
        row["reading_order"] = [list(map(int, payload["reading_order"]))]
    return row


def _write_demo(root: Path, *, image_token: str) -> None:
    """Materialise a tiny dataset (4 PNGs + 4 labels) for smoke tests."""

    from PIL import Image, ImageDraw

    root.mkdir(parents=True, exist_ok=True)
    palette = [(220, 80, 60), (80, 200, 120), (60, 110, 220), (240, 200, 60)]
    for i, color in enumerate(palette):
        img = Image.new("RGB", (64, 64), color=color)
        ImageDraw.Draw(img).text((4, 4), f"#{i:02d}", fill=(255, 255, 255))
        img.save(root / f"sample_{i:02d}.png")
        (root / f"sample_{i:02d}.json").write_text(
            json.dumps(
                {
                    "text": f"sample {i}",
                    "ocr_labels": [i % 16, (i + 1) % 16, (i + 2) % 16],
                    "reading_order": [0, 1, 2],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a multimodal JSONL shard from image/label pairs")
    p.add_argument("--input", required=False, help="directory of {stem.png, stem.txt|json} pairs")
    p.add_argument("--output", required=True, help="destination .jsonl path")
    p.add_argument("--image-token", default="<image>", help="placeholder inserted into the text field")
    p.add_argument(
        "--demo",
        action="store_true",
        help="materialise a synthetic 4-image dataset at --input first (for smoke tests)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input:
        print("build_ocr_data: --input is required", file=sys.stderr)
        return 2
    root = Path(args.input)
    if args.demo:
        _write_demo(root, image_token=args.image_token)
    if not root.exists():
        print(f"build_ocr_data: input directory does not exist: {root}", file=sys.stderr)
        return 2

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for image, label in _iter_pairs(root):
            row = _row_from_pair(image, label, image_token=args.image_token)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"build_ocr_data: wrote {n} rows to {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
