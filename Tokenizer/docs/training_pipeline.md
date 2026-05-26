# Training Pipeline

End-to-end recipe for producing the v0.8 unified tokenizer artefact.

## 1. Normalize raw Mongolian text

Use the Rust normalizer to fold Menksoft PUA and MW presentation
variants down to nominal Unicode before any tokenizer training touches
the text:

```bash
python -m Tokenizer.tools.normalize_mongolian --nominal \
    --input data/raw_mongolian.txt --output data/clean_mongolian.txt
```

The Python tool shells out to the Rust `normalize` example
(`cd "Encoding Mapping" && cargo run --example normalize -- --nominal`).
A PyO3 binding is a planned follow-up; the CLI bridge is the v0.8
contract.

## 2. Train MorphBPE

```bash
python -m Tokenizer.tools.build_morphbpe \
    --input Tokenizer/data/sample_text.jsonl \
    --output artefacts/morphbpe.json \
    --vocab-size 4096 \
    --min-pair-freq 2 \
    --min-boundary-confidence 0.60
```

`MorphBPETrainer` uses `MongolStemmer.analyze(word).skeleton_boundaries`
as forbidden merge boundaries only when the stemmer confidence meets the
configured threshold. Low-confidence analyses are treated as lexical words,
which prevents false roots like splitting `ᠮᠣᠩᠭᠣᠯ` at final `ᠯ`.
The trainer also ignores non-Mongolian whitespace-delimited words in mixed
corpora; Chinese, English, punctuation, and byte fallback tracks are trained
outside MorphBPE. The JSON output follows `serialization.SCHEMA_VERSION == 1`.

## 3. Build the unified tokenizer

Combine MorphBPE + HF zh + HF en + misc byte fallback through
`Tokenizer.tools.build_unified_tokenizer` (or programmatically via
`build_unified_vocab` + `DualTrackTokenizer`).

## 4. Encode mixed text

```python
from Tokenizer.unified.dual_tokenizer import DualTrackTokenizer
result = tokenizer.encode_with_spans("ᠮᠣᠩᠭᠣᠯ 文字 hello <image>")
result.input_ids
result.tokens         # list[EncodedToken]
result.spans          # language-routed spans
result.special_tokens_mask
```

## 5. Multimodal

```python
from Tokenizer.multimodal import MultimodalProcessor
proc = MultimodalProcessor(tokenizer)
enc = proc("describe <image>", images=[img], image_sizes=[(448, 448)])
```

## 6. Evaluate

```bash
python -m Tokenizer.evals.roundtrip_check --json
python -m Tokenizer.evals.offset_check --json
python -m Tokenizer.evals.chars_per_token --json
python -m Tokenizer.evals.mongolian_boundary_recall --json
python -m Tokenizer.evals.compare_baselines --json
```

All evals support `--input` (jsonl or txt) and fall back to built-in
smoke samples when no input is provided.

## 7. Run the test suite

```bash
scripts/test_all.sh
```
