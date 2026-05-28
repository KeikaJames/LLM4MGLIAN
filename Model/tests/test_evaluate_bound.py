# -*- coding: utf-8 -*-

"""Regression: ``evaluate`` must bound consumed batches by ``max_batches``
even when every batch has only ignore-index targets (M3 / Codex follow-up)."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from Model.config import IGNORE_INDEX, TrainingConfig
from Model.training.loop import evaluate


class _ConstLossModel(nn.Module):
    def forward(self, **kwargs):  # noqa: ANN003 - test stub
        return {"loss": torch.tensor(1.0)}


def _all_ignore_batches():
    """Infinite generator of batches whose shifted labels are all ignore."""
    while True:
        labels = torch.full((2, 5), IGNORE_INDEX, dtype=torch.long)
        yield {
            "input_ids": torch.zeros((2, 5), dtype=torch.long),
            "attention_mask": torch.ones((2, 5), dtype=torch.long),
            "labels": labels,
        }


class EvaluateBoundTest(unittest.TestCase):
    def test_all_ignore_batches_do_not_spin_forever(self) -> None:
        cfg = TrainingConfig(use_loss_chunking=False)
        # If the consumed-batch counter were only incremented on non-empty
        # batches, this call would never terminate on the infinite generator.
        out = evaluate(
            _ConstLossModel(),
            _all_ignore_batches(),
            cfg,
            device=torch.device("cpu"),
            max_batches=8,
        )
        self.assertEqual(out["eval_tokens"], 0.0)
        self.assertEqual(out["eval_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
