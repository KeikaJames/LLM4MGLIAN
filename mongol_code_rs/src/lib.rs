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
    input
        .chars()
        .map(|c| match c as u32 {
            // Historical NNBSP usage in this codebase means suffix separator.
            mongol::NNBS => char::from_u32(mongol::MVS).unwrap(),
            cp => char::from_u32(cp).unwrap(),
        })
        .collect()
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

/// Unicode -> Menksoft is not needed for tokenizer normalization yet.
pub fn convert_unicode_to_menksoft(_input: &str) -> String {
    unimplemented!("Unicode to Menksoft is not part of the tokenizer normalizer yet")
}

fn append_menksoft_word(output: &mut Vec<u32>, word: &[u32]) {
    if let Some(mapped) = fixed::lookup_menksoft_word(word) {
        output.extend_from_slice(mapped);
        return;
    }

    let mut previous_was_letter = false;
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
            previous_was_letter = false;
            continue;
        }

        let Some(mapped) = menksoft::to_unicode_nominal(cp) else {
            output.push(cp);
            previous_was_letter = false;
            continue;
        };

        // Menksoft encodes some standalone final/medial glyphs.  For tokenizer
        // normalization, keep the nominal character and avoid inserting Nirugu
        // unless the glyph is explicitly a separator/punctuation.
        if previous_was_letter && menksoft::is_initial_isolate_glyph(cp) {
            output.push(mongol::SPACE);
        }
        output.extend_from_slice(mapped);
        previous_was_letter = menksoft::is_letter(cp);
    }
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
}
