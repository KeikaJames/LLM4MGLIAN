# -*- coding: utf-8 -*-
"""Build and save a routed tokenizer bundle."""

from __future__ import annotations

import argparse
import sys

from Tokenizer.unified.bundle import TokenizerBundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("morphbpe_pos", nargs="?", help="legacy MorphBPE path")
    parser.add_argument("output_pos", nargs="?", help="legacy output bundle dir")
    parser.add_argument("--morphbpe", help="MorphBPE JSON model path")
    parser.add_argument("--output", help="output tokenizer bundle directory")
    parser.add_argument("--zh-source", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--en-source", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--smoke-hf", action="store_true", help="use local fake HF tracks")
    args = parser.parse_args()

    morphbpe = args.morphbpe or args.morphbpe_pos
    output = args.output or args.output_pos
    if not morphbpe or not output:
        parser.error("provide --morphbpe and --output, or legacy positional morphbpe output")

    try:
        bundle = TokenizerBundle.from_files(
            morphbpe,
            zh_source=args.zh_source,
            en_source=args.en_source,
            use_smoke_hf=args.smoke_hf,
        )
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}") from exc
    bundle.save_dir(output)
    loaded = TokenizerBundle.from_dir(output)
    issues = loaded.validate()
    print(f"vocab_size={loaded.tokenizer.vocab_size}")
    print(f"saved={output}")
    if issues:
        print("validate=failed")
        for issue in issues:
            print(f"- {issue}")
        sys.exit(1)
    print("validate=ok")


if __name__ == "__main__":
    main()
