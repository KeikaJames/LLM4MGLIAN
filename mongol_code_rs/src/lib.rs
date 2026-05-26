//! Rust port of the `mongol_code` conversion API.
//!
//! The original Dart package converts between standard Mongolian Unicode and
//! legacy Menksoft Private Use Area glyph codes.  The conversion is contextual:
//! the same Mongolian letter can map to different Menksoft glyphs depending on
//! word position, variation selectors, vowel separator usage, and neighboring
//! glyph shapes.

mod bridge;

/// Convert standard Mongolian Unicode text to legacy Menksoft glyph codes.
///
/// This crate is being ported from the Dart implementation in this repository.
/// During the port, the Rust API delegates to the restored Dart implementation
/// so parity tests can validate behavior before replacing each rule module with
/// native Rust.
pub fn convert_unicode_to_menksoft(input: &str) -> String {
    bridge::convert("unicode-to-menksoft", input)
}

/// Convert legacy Menksoft glyph codes to standard Mongolian Unicode text.
///
/// See [`convert_unicode_to_menksoft`] for porting status.
pub fn convert_menksoft_to_unicode(input: &str) -> String {
    bridge::convert("menksoft-to-unicode", input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_unicode_to_menksoft_example() {
        assert_eq!(
            convert_unicode_to_menksoft("\u{182E}\u{1823}\u{1829}\u{182D}\u{1823}\u{182F}"),
            "\u{E2F2}\u{E289}\u{E2BC}\u{E2EC}\u{E289}\u{E2F9}"
        );
    }

    #[test]
    fn converts_menksoft_to_unicode_example() {
        assert_eq!(
            convert_menksoft_to_unicode("\u{E2C1}\u{E27F}\u{E317}\u{E27E}\u{E2E8}"),
            "\u{182A}\u{1822}\u{1834}\u{1822}\u{182D}"
        );
    }
}
