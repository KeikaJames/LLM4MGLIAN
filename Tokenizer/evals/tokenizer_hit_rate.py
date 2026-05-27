# -*- coding: utf-8 -*-
"""Tokenizer coverage and hit-rate metrics for mixed Mongolian/zh/en text."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from typing import Any, Iterable

from Tokenizer.evals.mongolian_boundary_recall import compute_metrics
from Tokenizer.morphbpe import MorphBPETrainer
from Tokenizer.traditional_mongolian.alphabet import is_mongolian_codepoint
from Tokenizer.unified.bundle import TokenizerBundle
from Tokenizer.unified.dual_tokenizer import (
    DualTrackTokenizer,
    build_misc_tokens,
    build_unified_vocab,
    char_lang,
)


class _LongestMatchTokenizer:
    def __init__(self, tokens: Iterable[str]):
        ordered = list(dict.fromkeys(tok for tok in tokens if tok))
        if "<unk>" not in ordered:
            ordered.insert(0, "<unk>")
        self._vocab = {tok: idx for idx, tok in enumerate(ordered)}
        self._unk_id = self._vocab["<unk>"]
        self._ordered = sorted(
            (tok for tok in self._vocab if tok != "<unk>"),
            key=len,
            reverse=True,
        )

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [idx for idx, _start, _end in self._tokenize(text)]

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[Any]]:
        pieces = self._tokenize(text)
        result: dict[str, list[Any]] = {
            "input_ids": [idx for idx, _start, _end in pieces]
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [(start, end) for _idx, start, end in pieces]
        return result

    def _tokenize(self, text: str) -> list[tuple[int, int, int]]:
        out: list[tuple[int, int, int]] = []
        cursor = 0
        while cursor < len(text):
            match = None
            for token in self._ordered:
                if text.startswith(token, cursor):
                    match = token
                    break
            if match is None:
                out.append((self._unk_id, cursor, cursor + 1))
                cursor += 1
                continue
            out.append((self._vocab[match], cursor, cursor + len(match)))
            cursor += len(match)
        return out


def iter_text(path: str) -> list[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if path.endswith(".jsonl"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(obj.get("text", ""))
            else:
                text = line
            if text:
                texts.append(text)
    return texts


def build_experimental_tokenizer(
    train_texts: list[str],
    vocab_size: int,
    min_pair_freq: int,
    seed_alphabet: bool = True,
) -> DualTrackTokenizer:
    morphbpe = MorphBPETrainer(
        vocab_size=vocab_size,
        min_pair_freq=min_pair_freq,
        seed_alphabet=seed_alphabet,
    ).train(train_texts)

    zh_counts: Counter[str] = Counter()
    en_counts: Counter[str] = Counter()
    for text in train_texts:
        for ch in text:
            if char_lang(ch) == "zh":
                zh_counts[ch] += 1
        for word in re.findall(r"[A-Za-z]+", text):
            en_counts[word] += 1

    zh_tokens = [tok for tok, _count in zh_counts.most_common()]
    en_tokens = [tok for tok, _count in en_counts.most_common()]
    vocab = build_unified_vocab(
        morphbpe_vocab=morphbpe.vocab,
        chinese_tokens=zh_tokens,
        english_tokens=en_tokens,
        misc_tokens=build_misc_tokens(),
    )
    return DualTrackTokenizer(
        unified_vocab=vocab,
        morphbpe=morphbpe,
        zh_hf_tokenizer=_LongestMatchTokenizer(zh_tokens),
        en_hf_tokenizer=_LongestMatchTokenizer(en_tokens),
    )


def mongolian_words(texts: Iterable[str]) -> list[str]:
    words = []
    for text in texts:
        for word in text.split():
            if any(is_mongolian_codepoint(ch) for ch in word):
                words.append(word)
    return words


def compute_hit_rate(texts: list[str], tokenizer: DualTrackTokenizer) -> dict[str, Any]:
    chars = 0
    tokens = 0
    unk_count = 0
    byte_fallback = 0
    by_track: Counter[str] = Counter()
    chars_by_track: Counter[str] = Counter()
    unk_by_track: Counter[str] = Counter()
    mn_word_total = 0
    mn_word_hit = 0
    failures = []

    for text in texts:
        chars += len(text)
        result = tokenizer.encode_with_spans(text)
        decoded = tokenizer.decode(result.input_ids)
        if decoded != text and len(failures) < 10:
            failures.append({"text": text, "decoded": decoded})

        tokens += len(result.tokens)
        for token in result.tokens:
            by_track[token.track] += 1
            chars_by_track[token.track] += max(0, token.end - token.start)
            if token.id == tokenizer.unk_id:
                unk_count += 1
                unk_by_track[token.track] += 1
            if token.track == "misc" and token.token.startswith("<0x"):
                byte_fallback += 1

        for span in result.spans:
            if span.lang != "mn":
                continue
            span_tokens = [
                token
                for token in result.tokens
                if token.start >= span.start and token.end <= span.end
            ]
            for word in span.text.split():
                mn_word_total += 1
            if span.text and all(token.id != tokenizer.unk_id for token in span_tokens):
                mn_word_hit += len(span.text.split())

    words = mongolian_words(texts)
    boundary = compute_metrics(words, tokenizer.morphbpe) if words else {}
    return {
        "texts": len(texts),
        "chars": chars,
        "tokens": tokens,
        "chars_per_token": chars / tokens if tokens else 0.0,
        "unk_count": unk_count,
        "unk_rate": unk_count / tokens if tokens else 0.0,
        "token_hit_rate": 1.0 - (unk_count / tokens if tokens else 0.0),
        "byte_fallback_tokens": byte_fallback,
        "byte_fallback_rate": byte_fallback / tokens if tokens else 0.0,
        "mongolian_words": mn_word_total,
        "mongolian_word_hit_rate": mn_word_hit / mn_word_total
        if mn_word_total
        else 1.0,
        "tokens_by_track": dict(by_track),
        "unk_by_track": dict(unk_by_track),
        "chars_per_token_by_track": {
            track: chars_by_track[track] / by_track[track]
            for track in by_track
            if by_track[track]
        },
        "boundary": boundary,
        "roundtrip_failures_sample": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help=".txt or .jsonl evaluation data")
    parser.add_argument("--train-input", help="optional .txt/.jsonl training data")
    parser.add_argument("--tokenizer-bundle", help="load a persisted TokenizerBundle")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--min-pair-freq", type=int, default=2)
    parser.add_argument("--no-seed-alphabet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    texts = iter_text(args.input)
    if args.tokenizer_bundle:
        tokenizer = TokenizerBundle.from_dir(args.tokenizer_bundle).tokenizer
    else:
        train_texts = iter_text(args.train_input) if args.train_input else texts
        tokenizer = build_experimental_tokenizer(
            train_texts,
            vocab_size=args.vocab_size,
            min_pair_freq=args.min_pair_freq,
            seed_alphabet=not args.no_seed_alphabet,
        )

    metrics = compute_hit_rate(texts, tokenizer)
    if args.json:
        json.dump(metrics, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return

    for key, value in metrics.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
