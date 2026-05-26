use encoding_mapping::{
    convert_menksoft_to_unicode, convert_mw_to_unicode, convert_unicode_to_menksoft,
    normalize_to_nominal_unicode,
};

#[test]
fn unicode_to_menksoft_matches_onon_samples() {
    let cases = [
        (
            "\u{182A}\u{1822}\u{1834}\u{1822}\u{182D}",
            "\u{E2C1}\u{E27F}\u{E317}\u{E27E}\u{E2E8}",
        ),
        (
            "\u{182E}\u{1823}\u{1829}\u{182D}\u{1823}\u{182F}",
            "\u{E2F2}\u{E289}\u{E2BC}\u{E2EC}\u{E289}\u{E2F9}",
        ),
        (
            "\u{182A}\u{1820}\u{182D}\u{180D}\u{1830}\u{1822}",
            "\u{E2C1}\u{E26D}\u{E2EE}\u{E301}\u{E27B}",
        ),
        (
            "\u{1828}\u{1821}\u{1837}\u{180E}\u{1821}",
            "\u{E2B1}\u{E276}\u{E325}\u{E274}",
        ),
    ];

    for (unicode, menksoft) in cases {
        assert_eq!(convert_unicode_to_menksoft(unicode), menksoft);
        assert_eq!(
            normalize_to_nominal_unicode(&convert_menksoft_to_unicode(menksoft)),
            normalize_to_nominal_unicode(unicode)
        );
    }
}

#[test]
fn mw_and_menksoft_collapse_to_same_nominal_unicode() {
    let mw = "\u{182A}\u{1820}\u{182D}\u{180D}\u{1830}\u{1822}";
    let menksoft = "\u{E2C1}\u{E26D}\u{E2EE}\u{E301}\u{E27B}";
    let nominal = "\u{182A}\u{1820}\u{182D}\u{1830}\u{1822}";

    assert_eq!(normalize_to_nominal_unicode(mw), nominal);
    assert_eq!(normalize_to_nominal_unicode(menksoft), nominal);
}

#[test]
fn mw_pipeline_unifies_nnbs_and_removes_zero_width_noise() {
    let noisy_mw = "\u{FEFF}\u{1828}\u{1821}\u{1837}\u{202F}\u{200D}\u{1821}\u{200B}";

    assert_eq!(
        convert_mw_to_unicode(noisy_mw),
        "\u{1828}\u{1821}\u{1837}\u{180E}\u{1821}"
    );
    assert_eq!(
        normalize_to_nominal_unicode(noisy_mw),
        "\u{1828}\u{1821}\u{1837}\u{180E}\u{1821}"
    );
}
