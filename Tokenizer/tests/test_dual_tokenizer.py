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
    SEGMENT,
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

    def test_segments_fullwidth_latin_and_digits_as_en_and_misc(self):
        spans = segment_by_language("Ａ３！test")
        self.assertEqual(
            [(span.lang, span.text) for span in spans],
            [
                ("en", "Ａ"),
                ("misc", "３！"),
                ("en", "test"),
            ],
        )

    def test_segments_cjk_punctuation_as_misc(self):
        spans = segment_by_language("这。图")
        self.assertEqual(
            [(span.lang, span.text) for span in spans],
            [
                ("zh", "这"),
                ("misc", "。"),
                ("zh", "图"),
            ],
        )

    def test_encode_routes_tracks_to_global_id_ranges(self):
        tokenizer = build_fake_tokenizer()
        result = tokenizer.encode_with_spans("ᠮᠣᠩᠭᠣᠯ 这 test!", add_bos=True, add_eos=True)
        self.assertEqual(result.ids[0], SPECIAL_TOKENS["<bos>"])
        self.assertEqual(result.ids[-1], SPECIAL_TOKENS["<eos>"])
        self.assertIn(SEGMENT["mongolian"][0], result.ids)
        self.assertIn(SEGMENT["chinese"][0], result.ids)
        self.assertIn(SEGMENT["english"][0], result.ids)
        self.assertIn(tokenizer.vocab["!"], result.ids)
        self.assertEqual(result.input_ids, result.ids)
        self.assertEqual(len(result.tokens), len(result.ids))
        self.assertEqual(result.tokens[0].start, -1)
        self.assertEqual(result.tokens[-1].end, -1)

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

    def test_decode_strips_hf_boundary_markers(self):
        # Simulate HF vocabs that keep SentencePiece "▁" and GPT-2 "Ġ"
        # word-boundary markers in their raw token strings (e.g. Llama/Qwen).
        zh = FakeHFTokenizer({"\u2581这": 0, "张图": 1})
        en = FakeHFTokenizer({"\u0120hello": 0, "world": 1})
        vocab = build_unified_vocab(
            morphbpe_vocab=FakeMorphBPE.vocab,
            chinese_tokens=["\u2581这", "张图"],
            english_tokens=["\u0120hello", "world"],
            misc_tokens=build_misc_tokens(),
        )
        tokenizer = DualTrackTokenizer(vocab, FakeMorphBPE(), zh, en)

        zh_ids = [vocab["zh▁\u2581这"], vocab["zh▁张图"]]
        en_ids = [vocab["en▁\u0120hello"], vocab["en▁world"]]

        self.assertEqual(tokenizer.decode(zh_ids), " 这张图")
        self.assertEqual(tokenizer.decode(en_ids), " helloworld")
        # No raw boundary glyphs should leak through.
        decoded = tokenizer.decode(zh_ids + en_ids)
        self.assertNotIn("\u2581", decoded)
        self.assertNotIn("\u0120", decoded)

    def test_token_level_offsets_cover_tracks(self):
        tokenizer = build_fake_tokenizer()
        result = tokenizer.encode_with_spans("这 test 🙂")
        by_track = [(tok.track, tok.start, tok.end) for tok in result.tokens]
        self.assertIn(("zh", 0, 1), by_track)
        self.assertIn(("space", 1, 2), by_track)
        self.assertIn(("en", 2, 6), by_track)
        self.assertTrue(any(tok.track == "misc" and tok.start == 7 for tok in result.tokens))


if __name__ == "__main__":
    unittest.main()
