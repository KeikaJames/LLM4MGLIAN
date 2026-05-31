# -*- coding: utf-8 -*-

"""Regression tests for ``derive_morph_info_from_tokens``.

These pin the contract that only ``mn`` and ``general`` are word tracks, while
``general_punct``, ``space`` and ``special`` reset the running word. After the
two-track redesign punctuation is routed to a dedicated ``general_punct`` track
(not ``general``), so common punctuation-heavy text such as ``hello,world`` or
``这。图`` keeps clean per-word morphology features instead of gluing words
together through contiguous offsets.
"""

import unittest
from dataclasses import dataclass

from Tokenizer.pretraining.morphology import (
    WORD_TRACKS,
    derive_morph_info_from_tokens,
)


@dataclass
class FakeToken:
    track: str
    start: int
    end: int


class WordTracksTest(unittest.TestCase):
    def test_word_tracks_exclude_general_punct(self):
        self.assertIn("mn", WORD_TRACKS)
        self.assertIn("general", WORD_TRACKS)
        self.assertNotIn("general_punct", WORD_TRACKS)
        self.assertNotIn("space", WORD_TRACKS)
        self.assertNotIn("special", WORD_TRACKS)


class DeriveMorphInfoTest(unittest.TestCase):
    def test_contiguous_general_pieces_share_word_and_bump_depth(self):
        # "hello" as five contiguous general byte pieces -> one word, depth 0..4.
        tokens = [FakeToken("general", i, i + 1) for i in range(5)]
        word_pos, depth = derive_morph_info_from_tokens(tokens)
        self.assertEqual(word_pos, [0, 0, 0, 0, 0])
        self.assertEqual(depth, [0, 1, 2, 3, 4])

    def test_general_punct_resets_word(self):
        # "hello,world": comma on the general_punct track must separate words.
        tokens = (
            [FakeToken("general", i, i + 1) for i in range(5)]
            + [FakeToken("general_punct", 5, 6)]
            + [FakeToken("general", 6 + i, 7 + i) for i in range(5)]
        )
        word_pos, depth = derive_morph_info_from_tokens(tokens)
        # hello -> word 0; comma -> reset (word 0, depth 0); world -> word 1.
        self.assertEqual(word_pos, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.assertEqual(depth, [0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4])

    def test_punct_between_cjk_words_resets(self):
        # 这。图 : letter / punct / letter -> two distinct words around the period.
        tokens = [
            FakeToken("general", 0, 1),
            FakeToken("general_punct", 1, 2),
            FakeToken("general", 2, 3),
        ]
        word_pos, depth = derive_morph_info_from_tokens(tokens)
        self.assertEqual(word_pos, [0, 0, 1])
        self.assertEqual(depth, [0, 0, 0])

    def test_space_and_special_reset(self):
        tokens = [
            FakeToken("general", 0, 1),
            FakeToken("space", 1, 2),
            FakeToken("general", 2, 3),
            FakeToken("special", -1, -1),
            FakeToken("general", 3, 4),
        ]
        word_pos, depth = derive_morph_info_from_tokens(tokens)
        self.assertEqual(word_pos, [0, 0, 1, 1, 2])
        self.assertEqual(depth, [0, 0, 0, 0, 0])

    def test_non_contiguous_general_pieces_start_new_word(self):
        # A gap in offsets (e.g. a dropped separator) opens a new word.
        tokens = [FakeToken("general", 0, 1), FakeToken("general", 5, 6)]
        word_pos, depth = derive_morph_info_from_tokens(tokens)
        self.assertEqual(word_pos, [0, 1])
        self.assertEqual(depth, [0, 0])

    def test_track_switch_starts_new_word(self):
        # Adjacent offsets but different word tracks must not merge.
        tokens = [FakeToken("mn", 0, 1), FakeToken("general", 1, 2)]
        word_pos, depth = derive_morph_info_from_tokens(tokens)
        self.assertEqual(word_pos, [0, 1])
        self.assertEqual(depth, [0, 0])


if __name__ == "__main__":
    unittest.main()
