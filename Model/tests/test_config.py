# -*- coding: utf-8 -*-

"""Guards for ``Model.config`` special-token / segment contracts (L2)."""

from __future__ import annotations

import unittest

from Model.config import _FALLBACK_SEGMENT, _FALLBACK_SPECIAL_TOKENS


class SpecialTokenFallbackConsistencyTest(unittest.TestCase):
    def test_fallback_matches_canonical_vocab(self) -> None:
        try:
            from Tokenizer.unified.vocab import SEGMENT, SPECIAL_TOKENS
        except ImportError:  # pragma: no cover
            self.skipTest("Tokenizer package not importable")

        self.assertEqual(
            _FALLBACK_SPECIAL_TOKENS,
            dict(SPECIAL_TOKENS),
            "Model.config fallback SPECIAL_TOKENS drifted from "
            "Tokenizer/unified/vocab.py — keep them in sync.",
        )
        self.assertEqual(
            _FALLBACK_SEGMENT,
            {k: tuple(v) for k, v in SEGMENT.items()},
            "Model.config fallback SEGMENT drifted from "
            "Tokenizer/unified/vocab.py — keep them in sync.",
        )


if __name__ == "__main__":
    unittest.main()
