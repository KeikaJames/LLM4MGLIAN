# -*- coding: utf-8 -*-

"""CLI guard tests for ``scripts/train_rdt``.

These guards prevent the most dangerous foot-gun: kicking off what looks
like a real pretraining run but silently feeding the model random tokens
and emitting checkpoints with no useful signal.
"""

from __future__ import annotations

import unittest

from scripts import train_rdt


class TrainRdtCliGuardsTest(unittest.TestCase):
    def test_requires_data_or_smoke(self) -> None:
        # No --data, no --smoke ⇒ must abort before model construction.
        with self.assertRaises(SystemExit):
            train_rdt.main(["--config", "tiny"])

    def test_smoke_runs_without_data(self) -> None:
        # --smoke explicitly opts into the synthetic batch generator.
        rc = train_rdt.main(["--config", "tiny", "--smoke"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
