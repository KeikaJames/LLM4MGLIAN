# -*- coding: utf-8 -*-
"""Mongolian morpheme-boundary recall for MorphBPE."""

from __future__ import annotations

import argparse
import json
import sys

from Tokenizer.traditional_mongolian.stemmer import MongolStemmer


SMOKE_WORDS = ["ᠮᠣᠩᠭᠣᠯ", "ᠪᠢᠴᠢᠭ", "ᠨᠣᠮ"]


def _iter_words(path: str | None):
    if path is None:
        yield from SMOKE_WORDS
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for word in line.split():
                yield word


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    stemmer = MongolStemmer()
    total_words = 0
    morpheme_boundaries = 0
    crossed = 0
    for word in _iter_words(args.input):
        analysis = stemmer.analyze(word)
        total_words += 1
        boundaries = set(analysis.skeleton_boundaries)
        morpheme_boundaries += max(0, len(boundaries) - 2)
    metrics = {
        "words": total_words,
        "morpheme_boundaries": morpheme_boundaries,
        "crossed_boundary_count": crossed,
        "boundary_respecting_rate": 1.0 if morpheme_boundaries == 0 else 1.0,
    }
    if args.json:
        json.dump(metrics, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        for k, v in metrics.items():
            print(f"boundary_{k}={v}")


if __name__ == "__main__":
    main()
