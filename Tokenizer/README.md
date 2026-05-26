# Tokenizer

Routed multi-track tokenizer scaffold for traditional Mongolian, Chinese,
English, misc byte fallback, and multimodal placeholders.

## Run tests

```bash
python3 -m unittest discover Tokenizer
```

## Segment text

```bash
python3 -m Tokenizer.unified.dual_tokenizer segment "ᠮᠣᠩᠭᠣᠯ 中文 test <image>"
```

## Build MorphBPE and unified tokenizer

```bash
python3 -m Tokenizer.tools.build_morphbpe --input corpus.jsonl --output morphbpe.json --vocab-size 4096
python3 -m Tokenizer.tools.build_unified_tokenizer morphbpe.json unified_tokenizer.json
```

The unified build command needs `transformers` for real HF vocab extraction; if
it is missing, the tool exits with a clear dependency error.

## Use MultimodalProcessor

```python
from Tokenizer.multimodal import MultimodalProcessor

processor = MultimodalProcessor(tokenizer)
encoding = processor(
    "Describe <image>",
    images=[image],
    image_sizes=[(224, 224)],
)
print(encoding.input_ids)
print(encoding.image_token_spans)
```

`MultimodalProcessor` expands placeholders and records token spans. It leaves
actual vision feature extraction to an optional external image processor.

## Further documentation

- [docs/tokenizer_architecture.md](docs/tokenizer_architecture.md) — overall design.
- [docs/offset_contract.md](docs/offset_contract.md) — EncodedToken offset semantics.
- [docs/multimodal_contract.md](docs/multimodal_contract.md) — image/video processor contract.
- [docs/training_pipeline.md](docs/training_pipeline.md) — full training pipeline.
- [docs/external_implementation_notes.md](docs/external_implementation_notes.md) — design influences.

## Build pretraining rows

```bash
python3 -m Tokenizer.tools.build_pretraining_data \
  --tokenizer-bundle artefacts/tokenizer_bundle \
  --input Tokenizer/data/sample_multimodal.jsonl \
  --output artefacts/pretrain.jsonl \
  --max-length 2048 \
  --pack \
  --pad-to-max-length

python3 -m Tokenizer.evals.pretraining_gate \
  --tokenizer-bundle artefacts/tokenizer_bundle \
  --input artefacts/pretrain.jsonl \
  --max-length 2048 \
  --json
```

The pretraining builder masks structural labels with `-100`, keeps
image/video spans aligned, and can pack text-only rows while leaving
multimodal rows standalone.

## Evaluation

```bash
python3 -m Tokenizer.evals.roundtrip_check --json
python3 -m Tokenizer.evals.offset_check --json
python3 -m Tokenizer.evals.chars_per_token --json
python3 -m Tokenizer.evals.mongolian_boundary_recall --json
python3 -m Tokenizer.evals.compare_baselines --json
```

All evals support `--input <path.jsonl|.txt>` and fall back to built-in smoke
samples when no input is provided.

## Normalize Mongolian via Rust bridge

```bash
python3 -m Tokenizer.tools.normalize_mongolian --nominal < raw.txt > clean.txt
```
