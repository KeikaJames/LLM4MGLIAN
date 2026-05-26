//! Native Rust normalization for traditional Mongolian text.
//!
//! The tokenizer-facing goal is to collapse legacy encodings such as Menksoft
//! PUA glyph codes into one standard Unicode stream before segmentation and
//! tokenization.  Menksoft glyphs encode presentation forms; this crate maps
//! them back to Unicode nominal characters, preserving MVS/FVS where the
//! available rules identify them.

mod fixed;
mod menksoft;
mod mongol;
mod unicode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Unicode,
    Menksoft,
    Mixed,
}

/// Detect whether the text contains standard Mongolian Unicode, Menksoft PUA,
/// or both.
pub fn detect_encoding(input: &str) -> Encoding {
    let has_menksoft = input.chars().any(|c| menksoft::is_menksoft(c as u32));
    let has_unicode = input.chars().any(|c| mongol::is_mongolian(c as u32));

    match (has_unicode, has_menksoft) {
        (true, true) => Encoding::Mixed,
        (false, true) => Encoding::Menksoft,
        _ => Encoding::Unicode,
    }
}

/// Normalize supported legacy Mongolian encodings to standard Unicode.
///
/// The normalized stream is the representation tokenizer and embedding stages
/// should consume.  Unsupported text is passed through unchanged.
pub fn normalize_to_unicode(input: &str) -> String {
    if input
        .chars()
        .any(|c| menksoft::is_menksoft_or_space(c as u32))
    {
        convert_menksoft_to_unicode(input)
    } else {
        normalize_unicode(input)
    }
}

/// Normalize already-Unicode text for tokenizer ingestion.
pub fn normalize_unicode(input: &str) -> String {
    unicode::clean_mw_unicode(input)
}

/// Convert MW/民委共享工程-style Unicode text into the crate's standard Unicode.
///
/// Onon's current MW output uses Mongolian Unicode plus standard variation
/// selectors rather than Menksoft PUA. This step cleans transport/control
/// noise and unifies historical NNBS suffix spacing to MVS while preserving
/// FVS. Use `normalize_to_nominal_unicode` when tokenizer input must collapse
/// presentation variants to one base-letter stream.
pub fn convert_mw_to_unicode(input: &str) -> String {
    unicode::clean_mw_unicode(input)
}

/// Normalize text to nominal Mongolian Unicode for tokenizer ingestion.
///
/// This collapses Menksoft PUA and MW/Unicode presentation variants into the
/// same base-letter stream by removing FVS controls after conversion. MVS is
/// preserved because it is a suffix separator, not a glyph-only selector.
pub fn normalize_to_nominal_unicode(input: &str) -> String {
    unicode::to_nominal_unicode(&normalize_to_unicode(input))
}

/// Convert legacy Menksoft PUA glyph codes to Unicode.
pub fn convert_menksoft_to_unicode(input: &str) -> String {
    if input.is_empty() {
        return String::new();
    }

    let mut output = Vec::new();
    let mut word = Vec::new();

    for codepoint in input.chars().map(|c| c as u32) {
        if menksoft::is_menksoft(codepoint) && !menksoft::is_space(codepoint) {
            word.push(codepoint);
            continue;
        }

        if !word.is_empty() {
            append_menksoft_word(&mut output, &word);
            word.clear();
        }

        if menksoft::is_space(codepoint) {
            word.push(codepoint);
        } else {
            output.push(normalize_non_menksoft_codepoint(codepoint));
        }
    }

    if !word.is_empty() {
        append_menksoft_word(&mut output, &word);
    }

    from_codepoints(&output)
}

/// Convert standard Mongolian Unicode to legacy Menksoft PUA glyph codes.
pub fn convert_unicode_to_menksoft(input: &str) -> String {
    unicode::to_menksoft(input)
}

fn append_menksoft_word(output: &mut Vec<u32>, word: &[u32]) {
    if let Some(mapped) = fixed::lookup_menksoft_word(word) {
        output.extend_from_slice(mapped);
        return;
    }

    for (index, &cp) in word.iter().enumerate() {
        if menksoft::is_space(cp) {
            let next_is_letter = word
                .get(index + 1)
                .copied()
                .map(menksoft::is_letter)
                .unwrap_or(false);
            output.push(if cp == menksoft::NONBREAKING_SPACE && next_is_letter {
                mongol::MVS
            } else {
                mongol::SPACE
            });
            continue;
        }

        if append_menksoft_variant(output, word, index, cp) {
            continue;
        }

        let Some(mapped) = menksoft::to_unicode_nominal(cp) else {
            output.push(cp);
            continue;
        };

        output.extend_from_slice(mapped);
    }
}

fn append_menksoft_variant(output: &mut Vec<u32>, word: &[u32], index: usize, cp: u32) -> bool {
    let above = previous_codepoint(word, index);
    let below = word.get(index + 1).copied().unwrap_or(0);
    let word_len = word.iter().filter(|&&c| !menksoft::is_space(c)).count();

    match cp {
        // Contextual MVS + A/E presentation glyphs.
        0xE26A if menksoft::is_letter(above) => push_all(output, &[mongol::MVS, mongol::A]),
        0xE274 if menksoft::is_letter(above) => push_all(output, &[mongol::MVS, mongol::E]),

        // Straight medial YA is dropped in vowel + YI diphthongs; otherwise it
        // carries FVS1.
        0xE321 if menksoft::is_vowel(above) && menksoft::is_long_tooth_i(below) => true,
        0xE321 => push_all(output, &[mongol::YA, mongol::FVS1]),
        0xE27E | 0xE27F if menksoft::is_i(above) => true,
        0xE27E | 0xE27F if menksoft::is_a(above) && menksoft::is_m(below) => {
            push_all(output, &[mongol::I, mongol::FVS3])
        }

        // Common final or medial variation selectors from GB/T presentation
        // glyphs.
        0xE286 | 0xE288 => push_all(output, &[mongol::O, mongol::FVS1]),
        0xE285 if word_len == 2 => push_all(output, &[mongol::O, mongol::FVS2]),
        0xE28D if word_len == 2 => push_all(output, &[mongol::U, mongol::FVS2]),
        0xE296 if word_len == 2 => push_all(output, &[mongol::OE, mongol::FVS3]),
        0xE2A3 if word_len == 2 => push_all(output, &[mongol::UE, mongol::FVS3]),
        0xE2A8 => push_all(output, &[mongol::UE, mongol::FVS2]),

        // Dotted D/G/N variants that the nominal table intentionally flattens.
        0xE310 if matches!(below, 0xE2A8 | 0xE2A9) => push_all(output, &[mongol::DA, mongol::FVS1]),
        0xE312 => push_all(output, &[mongol::DA, mongol::FVS1]),
        0xE313 if menksoft::is_consonant(below) => push_all(output, &[mongol::DA, mongol::FVS1]),
        0xE30C | 0xE30D => push_all(output, &[mongol::TA, mongol::FVS1]),
        0xE2B8 | 0xE2BA | 0xE2C0 if menksoft::is_vowel(above) && menksoft::is_vowel(below) => {
            push_all(output, &[mongol::NA, mongol::FVS2])
        }
        0xE2B7 | 0xE2B9 | 0xE2BF if menksoft::is_consonant(below) => {
            push_all(output, &[mongol::NA, mongol::FVS1])
        }
        0xE2E7 => push_all(output, &[mongol::GA, mongol::FVS1]),
        0xE2E8 if index + 1 == word.len() && last_non_i_vowel_is_masculine(word, index) => {
            push_all(output, &[mongol::GA, mongol::FVS2])
        }
        0xE2EB | 0xE2ED | 0xE2EF | 0xE2F0
            if menksoft::word_contains_masculine_vowel(word)
                && !menksoft::is_feminine_vowel_or_i(below)
                && (menksoft::is_vowel(above) || menksoft::is_vowel(below)) =>
        {
            push_all(output, &[mongol::GA, mongol::FVS2])
        }
        0xE2A9 | 0xE2AA if output.len() > 2 => push_all(output, &[mongol::UE, mongol::FVS1]),

        _ => false,
    }
}

fn last_non_i_vowel_is_masculine(word: &[u32], index: usize) -> bool {
    word[..index]
        .iter()
        .rev()
        .copied()
        .find(|&cp| menksoft::is_vowel(cp) && !menksoft::is_i(cp))
        .map(menksoft::is_masculine_vowel)
        .unwrap_or(false)
}

fn previous_codepoint(word: &[u32], index: usize) -> u32 {
    if index == 0 {
        return 0;
    }

    word[..index]
        .iter()
        .rev()
        .copied()
        .find(|&cp| !menksoft::is_space(cp))
        .unwrap_or(0)
}

fn push_all(output: &mut Vec<u32>, codepoints: &[u32]) -> bool {
    output.extend_from_slice(codepoints);
    true
}

fn normalize_non_menksoft_codepoint(codepoint: u32) -> u32 {
    match codepoint {
        mongol::NNBS => mongol::MVS,
        cp => cp,
    }
}

fn from_codepoints(codepoints: &[u32]) -> String {
    codepoints
        .iter()
        .filter_map(|&cp| char::from_u32(cp))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_menksoft_text() {
        assert_eq!(detect_encoding("\u{E2C1}\u{E27F}"), Encoding::Menksoft);
        assert_eq!(detect_encoding("\u{182A}\u{1822}"), Encoding::Unicode);
    }

    #[test]
    fn normalizes_menksoft_word_to_unicode() {
        assert_eq!(
            normalize_to_unicode("\u{E2C1}\u{E27F}\u{E317}\u{E27E}\u{E2E8}"),
            "\u{182A}\u{1822}\u{1834}\u{1822}\u{182D}"
        );
    }

    #[test]
    fn normalizes_menksoft_suffix_space_to_mvs() {
        assert_eq!(
            normalize_to_unicode("\u{E2C1}\u{E275}\u{E263}\u{E282}"),
            "\u{182A}\u{1821}\u{180E}\u{1822}"
        );
    }

    #[test]
    fn normalizes_historical_nnbs_to_mvs() {
        assert_eq!(
            normalize_to_unicode("\u{182A}\u{202F}\u{1822}"),
            "\u{182A}\u{180E}\u{1822}"
        );
    }

    #[test]
    fn cleans_zero_width_noise_in_unicode_pipeline() {
        assert_eq!(
            convert_mw_to_unicode(
                "\u{FEFF}\u{182A}\u{200B}\u{1820}\u{200C}\u{182D}\u{200D}\u{1830}\u{2060}\u{1822}"
            ),
            "\u{182A}\u{1820}\u{182D}\u{1830}\u{1822}"
        );
    }

    #[test]
    fn normalizes_mw_variants_to_nominal_unicode() {
        assert_eq!(
            normalize_to_nominal_unicode("\u{182A}\u{1820}\u{182D}\u{180D}\u{1830}\u{1822}"),
            "\u{182A}\u{1820}\u{182D}\u{1830}\u{1822}"
        );
    }

    #[test]
    fn converts_unicode_word_to_menksoft() {
        assert_eq!(
            convert_unicode_to_menksoft("\u{182A}\u{1822}\u{1834}\u{1822}\u{182D}"),
            "\u{E2C1}\u{E27F}\u{E317}\u{E27E}\u{E2E8}"
        );
    }

    #[test]
    fn converts_mw_variant_word_to_menksoft() {
        assert_eq!(
            convert_unicode_to_menksoft("\u{182A}\u{1820}\u{182D}\u{180D}\u{1830}\u{1822}"),
            "\u{E2C1}\u{E26D}\u{E2EE}\u{E301}\u{E27B}"
        );
    }

    #[test]
    fn converts_nirugu_to_menksoft() {
        assert_eq!(convert_unicode_to_menksoft("\u{180A}"), "\u{E23E}");
        assert_eq!(
            convert_unicode_to_menksoft("\u{1820}\u{180A}\u{1820}"),
            "\u{E266}\u{E23E}\u{E268}"
        );
    }
}
