# -*- coding: utf-8 -*-
"""Build and save the routed unified tokenizer."""

from __future__ import annotations

import argparse

from Tokenizer.unified.dual_tokenizer import build_dual_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("morphbpe")
    parser.add_argument("output")
    parser.add_argument("--zh-source", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--en-source", default="meta-llama/Llama-3.2-1B")
    args = parser.parse_args()
    try:
        tokenizer = build_dual_tokenizer(args.morphbpe, args.zh_source, args.en_source)
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}") from exc
    tokenizer.save(args.output)
    print(f"vocab_size={tokenizer.vocab_size}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
