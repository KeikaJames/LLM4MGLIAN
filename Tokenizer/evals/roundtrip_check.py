# -*- coding: utf-8 -*-
"""Round-trip smoke check for the routed tokenizer.

Outputs a small JSON metric block; falls back to built-in smoke samples
if --input is not given.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable

from Tokenizer.unified.dual_tokenizer import (
    DualTrackTokenizer,
    build_misc_tokens,
    build_unified_vocab,
)


SMOKE_SAMPLES = [
    "hello",
    "中文",
    "🙂",
    "ᠮᠣᠩᠭᠣᠯ",
    "<image>",
    "mixed 中 ᠮᠣᠩᠭᠣᠯ 🙂",
]


class _Morph:
    vocab = {"ᠮᠣᠩᠭᠣᠯ": 0}

    def encode(self, text):
        return [self.vocab.get(text, 0)]


class _HF:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, text, add_special_tokens=False):
        if text in self._vocab:
            return [self._vocab[text]]
        return [self._vocab[ch] for ch in text if ch in self._vocab]


def _smoke_tokenizer() -> DualTrackTokenizer:
    zh = _HF({"中": 0, "文": 1})
    en = _HF({"hello": 0, "test": 1})
    vocab = build_unified_vocab(
        _Morph.vocab, ["中", "文"], ["hello", "test"], build_misc_tokens()
    )
    return DualTrackTokenizer(vocab, _Morph(), zh, en)


def _texts(path: str | None) -> Iterable[str]:
    if path is None:
        yield from SMOKE_SAMPLES
        return
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
                yield str(obj.get("text", ""))
            else:
                yield line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help=".jsonl or .txt; omit for smoke samples")
    parser.add_argument("--json", action="store_true", help="emit JSON metrics")
    args = parser.parse_args()

    tokenizer = _smoke_tokenizer()
    total = 0
    ok = 0
    failures = []
    for text in _texts(args.input):
        total += 1
        decoded = tokenizer.decode(tokenizer.encode(text))
        if decoded == text:
            ok += 1
        elif len(failures) < 10:
            failures.append({"text": text, "decoded": decoded})

    metrics = {
        "total": total,
        "passed": ok,
        "failed": total - ok,
        "rate": ok / total if total else 0.0,
        "failures_sample": failures,
    }
    if args.json:
        json.dump(metrics, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        print(f"roundtrip_total={metrics['total']}")
        print(f"roundtrip_ok={metrics['passed']}")
        print(f"roundtrip_failed={metrics['failed']}")
        print(f"roundtrip_rate={metrics['rate']:.4f}")


if __name__ == "__main__":
    main()
