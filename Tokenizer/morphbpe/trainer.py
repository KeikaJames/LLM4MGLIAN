# -*- coding: utf-8 -*-
"""Simple boundary-constrained MorphBPE trainer."""

from __future__ import annotations

import json
from collections import Counter
from typing import Iterable

from Tokenizer.traditional_mongolian.stemmer import MongolStemmer
from Tokenizer.traditional_mongolian.unicode_norm import strip_all_with_map

from .offsets import Piece, split_on_ascii_space
from .tokenizer import MorphBPETokenizer


class MorphBPETrainer:
    def __init__(
        self,
        stemmer: MongolStemmer | None = None,
        vocab_size: int = 4096,
        min_pair_freq: int = 2,
    ):
        self.stemmer = stemmer or MongolStemmer()
        self.vocab_size = vocab_size
        self.min_pair_freq = min_pair_freq

    def train(self, texts: Iterable[str]) -> MorphBPETokenizer:
        words = self._initial_words(texts)
        vocab = {"<unk>": 0}
        for pieces, _forbidden in words:
            for piece in pieces:
                vocab.setdefault(piece.text, len(vocab))

        merges: dict[tuple[str, str], tuple[str, int]] = {}
        while len(vocab) < self.vocab_size:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for pieces, forbidden in words:
                for i in range(len(pieces) - 1):
                    if pieces[i].end in forbidden:
                        continue
                    pair_counts[(pieces[i].text, pieces[i + 1].text)] += 1
            if not pair_counts:
                break
            (left, right), freq = pair_counts.most_common(1)[0]
            if freq < self.min_pair_freq:
                break

            merged = left + right
            rank = len(merges)
            merges[(left, right)] = (merged, rank)
            vocab.setdefault(merged, len(vocab))
            words = [
                (self._merge_word(pieces, forbidden, left, right, merged), forbidden)
                for pieces, forbidden in words
            ]

        return MorphBPETokenizer(vocab=vocab, merges=merges, stemmer=self.stemmer)

    def save(self, tokenizer: MorphBPETokenizer, path: str) -> None:
        tokenizer.save(path)

    def train_to_file(self, texts: Iterable[str], path: str) -> MorphBPETokenizer:
        tokenizer = self.train(texts)
        tokenizer.save(path)
        return tokenizer

    def _initial_words(
        self, texts: Iterable[str]
    ) -> list[tuple[list[Piece], set[int]]]:
        result: list[tuple[list[Piece], set[int]]] = []
        for text in texts:
            for start, end in split_on_ascii_space(text):
                word = text[start:end]
                skeleton, boundary_map = strip_all_with_map(word)
                if not skeleton:
                    continue
                analysis = self.stemmer.analyze(word)
                forbidden = set(analysis.skeleton_boundaries[1:-1])
                pieces = [
                    Piece(ch, boundary_map[i], boundary_map[i + 1])
                    for i, ch in enumerate(skeleton)
                ]
                result.append((pieces, forbidden))
        return result

    def _merge_word(
        self,
        pieces: list[Piece],
        forbidden: set[int],
        left: str,
        right: str,
        merged: str,
    ) -> list[Piece]:
        out: list[Piece] = []
        i = 0
        while i < len(pieces):
            if (
                i < len(pieces) - 1
                and pieces[i].text == left
                and pieces[i + 1].text == right
                and pieces[i].end not in forbidden
            ):
                out.append(Piece(merged, pieces[i].start, pieces[i + 1].end))
                i += 2
            else:
                out.append(pieces[i])
                i += 1
        return out


def save_training_json(tokenizer: MorphBPETokenizer, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "vocab": tokenizer.vocab,
                "merges": [
                    {"pair": list(pair), "merged": merged, "rank": rank}
                    for pair, (merged, rank) in tokenizer.merges.items()
                ],
                "specials": {"unk": "<unk>"},
                "config": {"type": "morphbpe"},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
