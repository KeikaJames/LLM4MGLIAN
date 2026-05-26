# -*- coding: utf-8 -*-
"""Gate a tokenizer bundle and input file for pretraining data construction."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from Tokenizer.pretraining import EncodedSample, PretrainingDataBuilder
from Tokenizer.unified.bundle import TokenizerBundle


def run_gate(bundle_dir: str, input_path: str, max_length: int) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    bundle = TokenizerBundle.from_dir(bundle_dir)
    for issue in bundle.validate():
        failures.append({"scope": "bundle", "message": issue})

    builder = PretrainingDataBuilder(bundle, max_length=max_length)
    samples: list[EncodedSample] = []
    total_tokens = 0
    unk_count = 0
    for idx, obj in enumerate(_iter_input(input_path)):
        text = str(obj.get("text", ""))
        try:
            sample = builder.encode_json_obj(obj)
        except Exception as exc:
            failures.append({"sample": idx, "message": f"encode failed: {exc}"})
            continue
        samples.append(sample)
        total_tokens += len(sample.input_ids)
        unk_count += sample.input_ids.count(bundle.tokenizer.unk_id)
        failures.extend(_validate_sample(bundle, sample, text, idx))

    metrics = {
        "total_tokens": total_tokens,
        "unk_rate": (unk_count / total_tokens) if total_tokens else 0.0,
        "max_len": max((len(sample.input_ids) for sample in samples), default=0),
    }
    return {
        "passed": not failures,
        "num_samples": len(samples),
        "metrics": metrics,
        "failures": failures,
    }


def _iter_input(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if path.endswith(".jsonl"):
                yield json.loads(line)
            else:
                yield {"type": "text", "text": line}


def _validate_sample(
    bundle: TokenizerBundle, sample: EncodedSample, text: str, idx: int
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    n = len(sample.input_ids)
    if len(sample.labels) != n or len(sample.attention_mask) != n:
        failures.append({"sample": idx, "message": "input_ids/labels/attention_mask length mismatch"})
    if len(sample.token_offsets) != n:
        failures.append({"sample": idx, "message": "token_offsets length mismatch"})
    vocab_size = bundle.tokenizer.vocab_size
    for pos, token_id in enumerate(sample.input_ids):
        if token_id < 0 or token_id >= vocab_size:
            failures.append({"sample": idx, "token": pos, "message": f"token id {token_id} out of range"})
    for pos, offset in enumerate(sample.token_offsets):
        start, end = int(offset[0]), int(offset[1])
        if (start, end) == (-1, -1):
            continue
        if start < 0 or end < start or end > len(text):
            failures.append({"sample": idx, "token": pos, "message": f"invalid offset {(start, end)}"})
    failures.extend(
        _validate_spans(
            bundle,
            sample,
            idx,
            "image_token_spans",
            "<image_start>",
            "<image_end>",
        )
    )
    failures.extend(
        _validate_spans(
            bundle,
            sample,
            idx,
            "video_token_spans",
            "<video_start>",
            "<video_end>",
        )
    )
    return failures


def _validate_spans(
    bundle: TokenizerBundle,
    sample: EncodedSample,
    sample_idx: int,
    key: str,
    start_token: str,
    end_token: str,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    start_id = bundle.tokenizer.vocab[start_token]
    end_id = bundle.tokenizer.vocab[end_token]
    for span_idx, span in enumerate(sample.modality_spans.get(key, [])):
        start, end = int(span[0]), int(span[1])
        if start < 0 or end > len(sample.input_ids) or end <= start:
            failures.append({"sample": sample_idx, "span": span_idx, "message": f"{key} out of bounds"})
            continue
        if end - start < 2:
            failures.append({"sample": sample_idx, "span": span_idx, "message": f"{key} missing start/end"})
            continue
        if sample.input_ids[start] != start_id or sample.input_ids[end - 1] != end_id:
            failures.append({"sample": sample_idx, "span": span_idx, "message": f"{key} markers mismatch"})
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-bundle", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_gate(args.tokenizer_bundle, args.input, args.max_length)
    if args.json:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        print(f"passed={str(result['passed']).lower()}")
        print(f"num_samples={result['num_samples']}")
        print(f"unk_rate={result['metrics']['unk_rate']:.6f}")
        for failure in result["failures"]:
            print(f"failure={failure}")
    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
