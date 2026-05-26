# Encoding Mapping

Rust encoding mapping and normalization for traditional Mongolian text.

The tokenizer-facing goal is normalization: collapse legacy Mongolian encodings
into one standard Unicode stream before segmentation, tokenization, and vector
training.

Current status:

- Rust exposes `normalize_to_unicode`, `detect_encoding`, and
  `convert_menksoft_to_unicode`.
- Menksoft PUA text is normalized with native Rust code; there is no Dart
  subprocess bridge.
- GB/T 25914-2023 fixed sequences and contextual Menksoft presentation variants
  covered by the historical word tests are included as Rust lookup/rule data.
- `standards/README.md` records the official standards-system reference.
- 140 Menksoft -> Unicode historical parity samples run as Rust tests.

Standards reference:

- Official public standards page:
  `https://openstd.samr.gov.cn/bzgk/gb/newGbInfo?hcno=BD6429DE5A7FC782FAAE13938A07166E`
- Standard name: `信息技术 传统蒙古文名义字符、变形显现字符和控制字符使用规则`
- Standard status: current; release date: 2023-11-27; implementation date:
  2024-06-01.
- The official download endpoint requires CAPTCHA verification. Large standards
  PDFs are kept out of git; use the official page as the source of record.

Important encoding boundary:

- Unicode/GB/T 25914-2023 is the canonical tokenizer representation.
- Menksoft is a legacy PUA presentation-glyph encoding and maps into Unicode.
- 民委/共享工程编码 is a separate legacy source. It needs its own table or
  verified samples before it can be mapped safely; this crate must not pretend
  it is the same as Menksoft.
