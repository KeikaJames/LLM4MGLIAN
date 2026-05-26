# -*- coding: utf-8 -*-
"""Inspect routed tokenization output."""

from __future__ import annotations

import argparse

from Tokenizer.evals.roundtrip_check import _smoke_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--morphbpe", help="optional MorphBPE model path (smoke tokenizer ignores this)")
    args = parser.parse_args()
    tokenizer = _smoke_tokenizer()
    result = tokenizer.encode_with_spans(args.text)
    for token in result.tokens:
        print(f"{token.id}\t{token.track}\t{token.start}:{token.end}\t{token.token!r}")
    print(f"decoded={tokenizer.decode(result.input_ids)!r}")
    print(f"special_tokens_mask={result.special_tokens_mask}")


if __name__ == "__main__":
    main()
