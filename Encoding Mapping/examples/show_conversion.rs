use encoding_mapping::{
    convert_menksoft_to_unicode, convert_mw_to_unicode, convert_unicode_to_menksoft,
    normalize_to_nominal_unicode,
};

fn main() {
    let cases = [
        (
            "Menksoft -> Unicode",
            "\u{E2C1}\u{E27F}\u{E317}\u{E27E}\u{E2E8}",
            convert_menksoft_to_unicode("\u{E2C1}\u{E27F}\u{E317}\u{E27E}\u{E2E8}"),
        ),
        (
            "Unicode -> Menksoft",
            "\u{182A}\u{1822}\u{1834}\u{1822}\u{182D}",
            convert_unicode_to_menksoft("\u{182A}\u{1822}\u{1834}\u{1822}\u{182D}"),
        ),
        (
            "MW/Unicode variant -> nominal Unicode",
            "\u{182A}\u{1820}\u{182D}\u{180D}\u{1830}\u{1822}",
            normalize_to_nominal_unicode("\u{182A}\u{1820}\u{182D}\u{180D}\u{1830}\u{1822}"),
        ),
        (
            "Noisy MW -> Unicode",
            "\u{FEFF}\u{1828}\u{1821}\u{1837}\u{202F}\u{200D}\u{1821}\u{200B}",
            convert_mw_to_unicode(
                "\u{FEFF}\u{1828}\u{1821}\u{1837}\u{202F}\u{200D}\u{1821}\u{200B}",
            ),
        ),
        (
            "Menksoft variant -> nominal Unicode",
            "\u{E2C1}\u{E26D}\u{E2EE}\u{E301}\u{E27B}",
            normalize_to_nominal_unicode("\u{E2C1}\u{E26D}\u{E2EE}\u{E301}\u{E27B}"),
        ),
    ];

    for (label, input, output) in cases {
        println!("{label}");
        println!("  input:  {}  [{}]", input, codepoints(input));
        println!("  output: {}  [{}]", output, codepoints(&output));
    }
}

fn codepoints(text: &str) -> String {
    text.chars()
        .map(|ch| format!("U+{:04X}", ch as u32))
        .collect::<Vec<_>>()
        .join(" ")
}
