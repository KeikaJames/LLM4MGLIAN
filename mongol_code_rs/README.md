# mongol_code_rs

Rust port work for `mongol_code`.

The tokenizer-facing goal is normalization: collapse legacy Mongolian encodings
into one standard Unicode stream before segmentation, tokenization, and vector
training.

Current status:

- Rust exposes `normalize_to_unicode`, `detect_encoding`, and
  `convert_menksoft_to_unicode`.
- Menksoft PUA text is normalized with native Rust code; there is no Dart
  subprocess bridge.
- GB/T 25914-2023 fixed sequences from the original implementation are included
  as Rust lookup data.
- Non-fixed Menksoft presentation glyphs are collapsed to their Unicode nominal
  characters for ML normalization.

Menksoft glyphs encode presentation forms, so full reversible rendering parity
still needs the contextual rule port. For tokenizer input, this crate should
prefer canonical Unicode normalization over preserving every presentation
variant.
