# Encoding Mapping

Rust encoding mapping and normalization for traditional Mongolian text.

The tokenizer-facing goal is normalization: collapse legacy Mongolian encodings
into one standard Unicode stream before segmentation, tokenization, and vector
training.

Current status:

- Rust exposes `normalize_to_unicode`, `detect_encoding`, and
  `normalize_to_nominal_unicode` for tokenizer input, plus
  `convert_menksoft_to_unicode`, `convert_mw_to_unicode`, and
  `convert_unicode_to_menksoft` for parity checks.
- Menksoft PUA text and Unicode/MW-style text are normalized with native Rust
  code; there is no Dart subprocess bridge.
- GB/T 25914-2023 fixed sequences and contextual Menksoft presentation variants
  covered by the historical word tests are included as Rust lookup/rule data.
- `standards/README.md` records the official standards-system reference.
- 210 Menksoft -> Unicode fixed sequences and 210 Unicode -> Menksoft fixed
  sequences are compiled into Rust lookup tables.
- 140 Menksoft -> Unicode historical parity samples and Onon-derived
  Unicode/MW/Menksoft samples run as Rust tests.

Standards reference:

- Official public standards page:
  `https://openstd.samr.gov.cn/bzgk/gb/newGbInfo?hcno=BD6429DE5A7FC782FAAE13938A07166E`
- Standard name: `信息技术 传统蒙古文名义字符、变形显现字符和控制字符使用规则`
- Standard status: current; release date: 2023-11-27; implementation date:
  2024-06-01.
- The official download endpoint requires CAPTCHA verification. Large standards
  PDFs are kept out of git; use the official page as the source of record.

Important encoding boundary:

- Nominal Unicode is the canonical tokenizer representation.
- Menksoft is a legacy PUA presentation-glyph encoding and maps into Unicode.
- Onon documents three relevant code modes: GB2010 (`MN`), Menksoft/Menk Code
  (`MK`/`MKL`), and 民委共享工程 (`MW`). The current Onon converter labels MW as
  a Unicode 2022-style standard stream rather than Menksoft PUA. The tokenizer
  should therefore call `normalize_to_nominal_unicode` so MW presentation
  selectors and Menksoft presentation glyphs collapse onto the same nominal
  Unicode letters.
- `convert_mw_to_unicode` cleans MW/Unicode transport noise by removing
  zero-width noise (`ZWJ`, `ZWNJ`, `ZWSP`, word joiner, BOM) and converting
  historical `NNBS` suffix spacing to `MVS`.
