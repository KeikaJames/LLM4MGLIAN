# -*- coding: utf-8 -*-
"""Tests for generic BPE: HF offset wrapper and byte fallback."""

from __future__ import annotations

import unittest

from Tokenizer.generic_bpe import (
    HFTrackTokenizer,
    decode_bytes,
    encode_byte_fallback,
    is_byte_token,
)


class _FakeFastTokenizer:
    """Mimics a HF fast tokenizer that supports return_offsets_mapping."""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        # Naive whitespace tokenization with real char offsets.
        ids = []
        offsets = []
        cursor = 0
        for word in text.split(" "):
            if not word:
                cursor += 1
                continue
            ids.append(self.vocab.get(word, self.vocab.get("<unk>", 0)))
            offsets.append((cursor, cursor + len(word)))
            cursor += len(word) + 1
        return {"input_ids": ids, "offset_mapping": offsets}

    def get_vocab(self):
        return self.vocab

    def encode(self, text, add_special_tokens=False):
        return self(text)["input_ids"]

    def convert_ids_to_tokens(self, idx):
        return self.id_to_token.get(idx, "<unk>")


class _FakeSlowTokenizer:
    """Mimics a HF slow tokenizer (no __call__ offsets support)."""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}

    def encode(self, text, add_special_tokens=False):
        return [
            self.vocab.get(w, self.vocab.get("<unk>", 0))
            for w in text.split(" ")
            if w
        ]

    def get_vocab(self):
        return self.vocab

    def convert_ids_to_tokens(self, idx):
        return self.id_to_token.get(idx, "<unk>")


class HFTrackOffsetTests(unittest.TestCase):
    def test_fast_offsets_used(self):
        vocab = {"hello": 1, "world": 2, "<unk>": 0}
        local_to_global = {1: 101, 2: 102}
        tok = HFTrackTokenizer(
            _FakeFastTokenizer(vocab),
            prefix="en▁",
            local_to_global=local_to_global,
            unk_id=999,
            track="en",
        )
        tokens = tok.encode_with_offsets("hello world", base_start=10)
        self.assertEqual([t.id for t in tokens], [101, 102])
        self.assertEqual(tokens[0].start, 10)
        self.assertEqual(tokens[0].end, 15)
        self.assertEqual(tokens[1].start, 16)
        self.assertEqual(tokens[1].end, 21)
        self.assertTrue(all(t.track == "en" for t in tokens))

    def test_slow_fallback(self):
        vocab = {"hello": 1, "world": 2, "<unk>": 0}
        local_to_global = {1: 101, 2: 102}
        tok = HFTrackTokenizer(
            _FakeSlowTokenizer(vocab),
            prefix="en▁",
            local_to_global=local_to_global,
            unk_id=999,
            track="en",
        )
        tokens = tok.encode_with_offsets("hello world")
        self.assertEqual(len(tokens), 2)
        # Coarse offsets must span the whole input monotonically.
        self.assertEqual(tokens[0].start, 0)
        self.assertGreater(tokens[-1].end, tokens[0].end)
        self.assertEqual(tokens[-1].end, len("hello world"))

    def test_unknown_local_id_falls_to_unk(self):
        vocab = {"hello": 1, "<unk>": 0}
        tok = HFTrackTokenizer(
            _FakeFastTokenizer(vocab),
            prefix="en▁",
            local_to_global={1: 101},
            unk_id=999,
            track="en",
        )
        tokens = tok.encode_with_offsets("hello banana")
        # 'banana' isn't in local_to_global so resolves to unk_id.
        self.assertEqual(tokens[0].id, 101)
        self.assertEqual(tokens[1].id, 999)


class ByteFallbackTests(unittest.TestCase):
    def test_emoji_roundtrip(self):
        vocab = {f"<0x{i:02X}>": 1000 + i for i in range(256)}
        vocab["<unk>"] = 0
        tokens = encode_byte_fallback("🙂", vocab, unk_id=0)
        byte_strs = [t.token for t in tokens]
        self.assertTrue(all(is_byte_token(t) for t in byte_strs))
        self.assertEqual(decode_bytes(byte_strs), "🙂")

    def test_cjk_punct_byte_fallback(self):
        vocab = {f"<0x{i:02X}>": 1000 + i for i in range(256)}
        vocab["<unk>"] = 0
        tokens = encode_byte_fallback("，", vocab, unk_id=0)
        self.assertTrue(all(is_byte_token(t.token) for t in tokens))
        # CJK comma is 3 bytes in UTF-8
        self.assertEqual(len(tokens), 3)
        self.assertEqual(decode_bytes([t.token for t in tokens]), "，")

    def test_direct_token_used_when_present(self):
        vocab = {"a": 7, "<unk>": 0}
        tokens = encode_byte_fallback("a", vocab, unk_id=0)
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].id, 7)
        self.assertEqual(tokens[0].token, "a")

    def test_offsets_monotonic(self):
        vocab = {f"<0x{i:02X}>": 1000 + i for i in range(256)}
        vocab["<unk>"] = 0
        tokens = encode_byte_fallback("🙂a，", vocab, unk_id=0, base_start=5)
        # All offsets must be >= base_start and non-decreasing.
        last = 5
        for t in tokens:
            self.assertGreaterEqual(t.start, last - 1)
            self.assertLessEqual(t.start, t.end)
            last = t.start

    def test_lone_surrogate_does_not_crash(self):
        # Regression: scraped text can contain lone surrogates (U+D800–U+DFFF).
        # Strict ``str.encode("utf-8")`` raises UnicodeEncodeError and would
        # abort the whole shard; surrogatepass must keep encoding going.
        vocab = {f"<0x{i:02X}>": 1000 + i for i in range(256)}
        vocab["<unk>"] = 0
        tokens = encode_byte_fallback("a\ud800b", vocab, unk_id=0)
        self.assertTrue(any(is_byte_token(t.token) for t in tokens))
        # 'a' + 3 surrogate bytes + 'b' worth of byte tokens, all finite ids.
        self.assertTrue(all(isinstance(t.id, int) for t in tokens))


if __name__ == "__main__":
    unittest.main()
