#!/usr/bin/env bash
# End-to-end multimodal smoke: PIL images → JSONL → builder → trainers.
# All paths use synthetic data and complete on CPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${OUT:-/tmp/llm4mglian_mm_smoke}"
rm -rf "$OUT"
mkdir -p "$OUT"

echo "==> [1/4] generate 4 synthetic images + raw JSONL"
PYTHONPATH=. python3 -m Tokenizer.tools.build_ocr_data \
    --input  "$OUT/imgs" \
    --output "$OUT/raw.jsonl" \
    --demo

echo "==> [2/4] encode JSONL through PretrainingDataBuilder (smoke bundle)"
PYTHONPATH=. python3 - <<PY
import json, tempfile, os, sys
from Tokenizer.tests.test_pretraining_builder import build_smoke_bundle
from Tokenizer.pretraining import PretrainingDataBuilder, encoded_sample_to_dict

bundle_dir = "$OUT/bundle"
os.makedirs(bundle_dir, exist_ok=True)
bundle = build_smoke_bundle(bundle_dir)
builder = PretrainingDataBuilder(bundle)
with open("$OUT/raw.jsonl") as fin, open("$OUT/encoded.jsonl", "w") as fout:
    for line in fin:
        line = line.strip()
        if not line: continue
        enc = builder.encode_jsonl_line(line)
        fout.write(json.dumps(encoded_sample_to_dict(enc), ensure_ascii=False) + "\n")
print("encoded ->", "$OUT/encoded.jsonl")
PY

echo "==> [3/4] train_omvt_ssl --data (real images, self-supervised orientation)"
PYTHONPATH=. python3 -m scripts.train_omvt_ssl \
    --data "$OUT/raw.jsonl" \
    --image-size 64 --steps 2 --batch-size 2 \
    --output "$OUT/omvt"

echo "==> [4/4] train_vlm_align --data (real images, full LM)"
PYTHONPATH=. python3 -m scripts.train_vlm_align \
    --data "$OUT/encoded.jsonl" \
    --image-size 64 --n-image-tokens 9 --seq-len 32 --batch-size 2 --steps 2 \
    --output "$OUT/vlm"

echo "All multimodal smoke runs OK."
