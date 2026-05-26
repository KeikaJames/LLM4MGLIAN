# -*- coding: utf-8 -*-

import unittest

from Tokenizer.multimodal import (
    IMAGE_END,
    IMAGE_PATCH,
    IMAGE_PLACEHOLDER,
    IMAGE_START,
    expand_image_placeholders,
)
from Tokenizer.unified.dual_tokenizer import (
    DualTrackTokenizer,
    SPECIAL_TOKENS,
    build_misc_tokens,
    build_unified_vocab,
    segment_by_language,
)


class FakeMorphBPE:
    vocab = {"ᠮᠣᠩᠭᠣᠯ": 0, "ᠪᠢᠴᠢᠭ": 1}

    def encode(self, text):
        return [self.vocab[text]]


class FakeHFTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, text, add_special_tokens=False):
        if text in self._vocab:
            return [self._vocab[text]]
        return [self._vocab[ch] for ch in text if ch in self._vocab]


def build_fake_tokenizer():
    zh = FakeHFTokenizer({"这": 0, "张": 1, "图": 2})
    en = FakeHFTokenizer({"test": 0, "hello": 1})
    vocab = build_unified_vocab(
        morphbpe_vocab=FakeMorphBPE.vocab,
        chinese_tokens=["这", "张", "图"],
        english_tokens=["test", "hello"],
        misc_tokens=build_misc_tokens(),
    )
    return DualTrackTokenizer(vocab, FakeMorphBPE(), zh, en)


class DualTokenizerTest(unittest.TestCase):
    def test_segments_special_tokens_before_language_routing(self):
        spans = segment_by_language("这张图 " + IMAGE_PLACEHOLDER + " test")
        self.assertEqual(
            [(span.lang, span.text) for span in spans],
            [
                ("zh", "这张图"),
                ("space", " "),
                ("special", IMAGE_PLACEHOLDER),
                ("space", " "),
                ("en", "test"),
            ],
        )

    def test_encode_routes_tracks_to_global_id_ranges(self):
        tokenizer = build_fake_tokenizer()
        result = tokenizer.encode_with_spans("ᠮᠣᠩᠭᠣᠯ 这 test!", add_bos=True, add_eos=True)
        self.assertEqual(result.ids[0], SPECIAL_TOKENS["<bos>"])
        self.assertEqual(result.ids[-1], SPECIAL_TOKENS["<eos>"])
        self.assertIn(16, result.ids)
        self.assertIn(40000, result.ids)
        self.assertIn(55000, result.ids)
        self.assertIn(tokenizer.vocab["!"], result.ids)

    def test_multimodal_placeholder_and_patch_tokens_are_special(self):
        tokenizer = build_fake_tokenizer()
        expanded = expand_image_placeholders(IMAGE_PLACEHOLDER, patches_per_image=2)
        self.assertEqual(expanded, IMAGE_START + IMAGE_PATCH + IMAGE_PATCH + IMAGE_END)
        ids = tokenizer.encode(expanded)
        self.assertEqual(
            ids,
            [
                SPECIAL_TOKENS[IMAGE_START],
                SPECIAL_TOKENS[IMAGE_PATCH],
                SPECIAL_TOKENS[IMAGE_PATCH],
                SPECIAL_TOKENS[IMAGE_END],
            ],
        )

    def test_byte_fallback_round_trip(self):
        tokenizer = build_fake_tokenizer()
        ids = tokenizer.encode("🙂")
        self.assertGreater(len(ids), 1)
        self.assertEqual(tokenizer.decode(ids), "🙂")


if __name__ == "__main__":
    unittest.main()
