# -*- coding: utf-8 -*-
"""Build minimum JSONL pretraining data with a tokenizer bundle."""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable

from Tokenizer.pretraining import PretrainingDataBuilder, encoded_sample_to_dict
from Tokenizer.unified.bundle import TokenizerBundle


def _iter_samples(path: str, builder: PretrainingDataBuilder):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if path.endswith(".jsonl"):
                yield builder.encode_jsonl_line(line)
            else:
                yield builder.encode_text(line, metadata={"type": "text"})


def _summary(samples: Iterable[dict], unk_id: int) -> dict:
    rows = list(samples)
    lengths = [len(row["input_ids"]) for row in rows]
    total_tokens = sum(lengths)
    unk_count = sum(row["input_ids"].count(unk_id) for row in rows)
    return {
        "num_samples": len(rows),
        "avg_len": (total_tokens / len(rows)) if rows else 0.0,
        "max_len": max(lengths) if lengths else 0,
        "unk_rate": (unk_count / total_tokens) if total_tokens else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-bundle", required=True)
    parser.add_argument("--input", required=True, help=".txt or .jsonl")
    parser.add_argument("--output", required=True, help="output JSONL")
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    bundle = TokenizerBundle.from_dir(args.tokenizer_bundle)
    builder = PretrainingDataBuilder(bundle, max_length=args.max_length)
    rows = [encoded_sample_to_dict(sample) for sample in _iter_samples(args.input, builder)]

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(_summary(rows, bundle.tokenizer.unk_id), ensure_ascii=False))


if __name__ == "__main__":
    main()
