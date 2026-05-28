# -*- coding: utf-8 -*-

"""Unit tests for checkpoint RNG capture/restore (Python + NumPy + torch)."""

from __future__ import annotations

import random
import unittest

import torch

from Model.training.checkpoint import _restore_rng, _rng_state


class CheckpointRngStateTest(unittest.TestCase):
    def test_python_and_torch_rng_roundtrip(self) -> None:
        random.seed(1234)
        torch.manual_seed(1234)
        # Capture state, then draw a reference sequence.
        state = _rng_state()
        ref_py = [random.random() for _ in range(5)]
        ref_torch = torch.randint(0, 1_000_000, (5,)).tolist()

        # Advance the generators so they diverge from the captured state.
        for _ in range(10):
            random.random()
            torch.randint(0, 1_000_000, (3,))

        # Restoring must reproduce the exact reference sequence.
        _restore_rng(state)
        self.assertEqual([random.random() for _ in range(5)], ref_py)
        self.assertEqual(torch.randint(0, 1_000_000, (5,)).tolist(), ref_torch)

    def test_numpy_rng_roundtrip(self) -> None:
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            self.skipTest("numpy not installed")

        np.random.seed(99)
        self.assertIn("numpy", _rng_state())
        state = _rng_state()
        ref = np.random.rand(5).tolist()
        np.random.rand(20)  # advance
        _restore_rng(state)
        self.assertEqual(np.random.rand(5).tolist(), ref)


if __name__ == "__main__":
    unittest.main()
