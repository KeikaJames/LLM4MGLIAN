# mongol_code_rs

Rust port work for `mongol_code`.

`mongol_code` converts between standard Mongolian Unicode text and legacy
Menksoft Private Use Area glyph codes. This is not a simple one-to-one mapping:
the conversion depends on word position, FVS/MVS control characters, vowel
gender, fixed sequences from the GB/T 25914-2023 rules, and neighboring glyph
shape.

Current status:

- Rust exposes the same two public conversion functions:
  - `convert_unicode_to_menksoft`
  - `convert_menksoft_to_unicode`
- The Rust crate currently delegates to the restored Dart implementation as a
  parity bridge.
- The bridge is intentional scaffolding for the native Rust port: it gives the
  Rust API and tests a stable oracle while each contextual rule module is moved
  from Dart to Rust.

The Dart test suite is restored in `mongol_code-master/test` and should stay as
the compatibility baseline during the port.
