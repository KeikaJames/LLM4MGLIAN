# -*- coding: utf-8 -*-

"""Tests for the ``build_ocr_data`` JSONL builder."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

try:
    from PIL import Image  # noqa: F401
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

from Tokenizer.tools.build_ocr_data import main


@unittest.skipIf(Image is None, "Pillow not installed")
class BuildOCRDataTest(unittest.TestCase):
    def test_demo_writes_consumable_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "imgs"
            out = Path(tmp) / "shard.jsonl"
            rc = main([
                "--input", str(data_root),
                "--output", str(out),
                "--demo",
            ])
            self.assertEqual(rc, 0)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertIn("<image>", row["text"])
                self.assertEqual(len(row["images"]), 1)
                self.assertTrue(os.path.isabs(row["images"][0]))
                self.assertEqual(len(row["image_sizes"][0]), 2)
                self.assertEqual(len(row["ocr_labels"][0]), 3)
                self.assertEqual(row["reading_order"], [[0, 1, 2]])

    def test_txt_label_becomes_text_field(self) -> None:
        from PIL import Image as _Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "a.png"
            _Image.new("RGB", (16, 16), (10, 20, 30)).save(img)
            (root / "a.txt").write_text("hello world", encoding="utf-8")
            out = root / "out.jsonl"
            rc = main(["--input", str(root), "--output", str(out)])
            self.assertEqual(rc, 0)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertIn("hello world", rows[0]["text"])
            self.assertNotIn("ocr_labels", rows[0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
