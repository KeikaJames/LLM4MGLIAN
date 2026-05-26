# -*- coding: utf-8 -*-

import unittest

from .core_traditional_mongolian_suffixes import MVS, NNBSP
from .stemmer import MongolStemmer, control_boundaries, skeleton_with_map, strip_all


class TraditionalMongolianStemmerTest(unittest.TestCase):
    def setUp(self):
        self.stemmer = MongolStemmer()

    def test_genitive_suffix_with_mvs_boundary(self):
        word = "ᠪᠢᠴᠢᠭ" + MVS + "ᠦᠨ"
        result = self.stemmer.analyze(word)
        self.assertEqual(result.root, "ᠪᠢᠴᠢᠭ")
        self.assertEqual(result.suffixes, [MVS + "ᠦᠨ"])
        self.assertEqual(result.suffix_ids, ["GEN"])
        self.assertEqual(result.boundaries, [0, 5, 8])
        self.assertEqual(result.skeleton_boundaries, [0, 5, 7])

    def test_plural_case_stack_with_original_boundaries(self):
        word = "ᠪᠢᠴᠢᠭ" + MVS + "ᠦᠳ" + MVS + "ᠦᠨ"
        result = self.stemmer.analyze(word)
        self.assertEqual(result.root, "ᠪᠢᠴᠢᠭ")
        self.assertEqual(result.suffix_ids, ["PL_UD", "GEN"])
        self.assertEqual(result.suffix_types, ["plural", "case"])
        self.assertEqual(result.suffixes, [MVS + "ᠦᠳ", MVS + "ᠦᠨ"])
        self.assertEqual(result.skeleton_boundaries, [0, 5, 7, 9])

    def test_nnbsp_question_particle(self):
        word = "ᠨᠡᠷ" + NNBSP + "ᠦᠦ"
        result = self.stemmer.analyze(word)
        self.assertEqual(result.root, "ᠨᠡᠷ")
        self.assertEqual(result.suffix_ids, ["Q_UU"])
        self.assertEqual(result.suffix_types, ["particle"])
        self.assertEqual(result.suffixes, [NNBSP + "ᠦᠦ"])

    def test_encoding_mapping_output_shape_is_accepted(self):
        # Encoding Mapping normalizes MW/NNBSP suffix spacing to MVS Unicode.
        word = "ᠨᠡᠷ" + MVS + "ᠡ"
        result = self.stemmer.analyze(word)
        self.assertEqual(result.root, "ᠨᠡᠷ")
        self.assertEqual(result.suffix_ids, ["DAT_LOC"])
        self.assertGreaterEqual(result.confidence, 0.60)

    def test_skeleton_map_ignores_controls(self):
        word = "ᠨᠡᠷ" + NNBSP + "ᠦᠦ"
        skeleton, boundary_map = skeleton_with_map(word)
        self.assertEqual(skeleton, strip_all(word))
        self.assertEqual(skeleton, "ᠨᠡᠷᠦᠦ")
        self.assertEqual(boundary_map, [0, 1, 2, 3, 5, 6])
        self.assertEqual(control_boundaries(word), {3})


if __name__ == "__main__":
    unittest.main()
