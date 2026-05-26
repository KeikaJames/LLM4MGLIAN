# -*- coding: utf-8 -*-

import unittest

from Tokenizer.evals.mongolian_boundary_recall import compute_metrics
from Tokenizer.morphbpe.offsets import MorphToken
from Tokenizer.traditional_mongolian import MVS


class BoundaryRecallEvalTest(unittest.TestCase):
    def test_boundary_rate_uses_measured_crossings(self):
        word = "ᠪᠢᠴᠢᠭ" + MVS + "ᠦᠨ"

        class CrossingTokenizer:
            def tokenize_word(self, text):
                return [MorphToken(text, 1, 0, len(text))]

        metrics = compute_metrics([word], CrossingTokenizer())
        self.assertGreater(metrics["morpheme_boundaries"], 0)
        self.assertGreater(metrics["crossed_boundary_count"], 0)
        self.assertLess(metrics["boundary_respecting_rate"], 1.0)

    def test_boundary_rate_is_perfect_for_char_tokens(self):
        word = "ᠪᠢᠴᠢᠭ" + MVS + "ᠦᠨ"
        metrics = compute_metrics([word])
        self.assertGreater(metrics["morpheme_boundaries"], 0)
        self.assertEqual(metrics["crossed_boundary_count"], 0)
        self.assertEqual(metrics["boundary_respecting_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
