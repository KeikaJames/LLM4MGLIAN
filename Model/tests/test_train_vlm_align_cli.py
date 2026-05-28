# -*- coding: utf-8 -*-

"""CLI guard tests for ``scripts/train_vlm_align``.

Ensures bad CLI input fails cleanly (exit code 2 + stderr message) rather than
raising a raw traceback — in particular ``--image-size`` is validated before
``image_patch_count`` derives the default ``--n-image-tokens``.
"""

from __future__ import annotations

import contextlib
import io
import unittest

from scripts import train_vlm_align


class TrainVlmAlignCliGuardsTest(unittest.TestCase):
    def test_non_positive_image_size_fails_cleanly(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = train_vlm_align.main(["--image-size", "0", "--seq-len", "16"])
        self.assertEqual(rc, 2)
        self.assertIn("--image-size", stderr.getvalue())

    def test_non_multiple_of_four_image_size_fails_cleanly(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = train_vlm_align.main(["--image-size", "6", "--seq-len", "16"])
        self.assertEqual(rc, 2)
        self.assertIn("--image-size", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
