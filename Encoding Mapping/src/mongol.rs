pub const SPACE: u32 = 0x0020;
pub const MIDDLE_DOT: u32 = 0x00B7;
pub const PUNCTUATION_X: u32 = 0x00D7;
pub const REFERENCE_MARK: u32 = 0x203B;
pub const QUESTION_EXCLAMATION: u32 = 0x2048;
pub const EXCLAMATION_QUESTION: u32 = 0x2049;
pub const NNBS: u32 = 0x202F;
pub const ZWJ: u32 = 0x200D;

pub const LEFT_DOUBLE_ANGLE_BRACKET: u32 = 0x300A;
pub const RIGHT_DOUBLE_ANGLE_BRACKET: u32 = 0x300B;
pub const LEFT_ANGLE_BRACKET: u32 = 0x3008;
pub const RIGHT_ANGLE_BRACKET: u32 = 0x3009;
pub const FULLWIDTH_SEMICOLON: u32 = 0xFF1B;
pub const FULLWIDTH_EXCLAMATION: u32 = 0xFF01;
pub const FULLWIDTH_QUESTION: u32 = 0xFF1F;
pub const FULLWIDTH_LEFT_PARENTHESIS: u32 = 0xFF08;
pub const FULLWIDTH_RIGHT_PARENTHESIS: u32 = 0xFF09;

pub const VERTICAL_COMMA: u32 = 0xFE10;
pub const VERTICAL_COLON: u32 = 0xFE13;
pub const VERTICAL_EM_DASH: u32 = 0xFE31;
pub const VERTICAL_EN_DASH: u32 = 0xFE32;
pub const VERTICAL_LEFT_TORTOISE_SHELL_BRACKET: u32 = 0xFE39;
pub const VERTICAL_RIGHT_TORTOISE_SHELL_BRACKET: u32 = 0xFE3A;
pub const VERTICAL_LEFT_WHITE_CORNER_BRACKET: u32 = 0xFE43;
pub const VERTICAL_RIGHT_WHITE_CORNER_BRACKET: u32 = 0xFE44;

pub const BIRGA: u32 = 0x1800;
pub const ELLIPSIS: u32 = 0x1801;
pub const COMMA: u32 = 0x1802;
pub const FULL_STOP: u32 = 0x1803;
pub const COLON: u32 = 0x1804;
pub const FOUR_DOTS: u32 = 0x1805;
pub const TODO_SOFT_HYPHEN: u32 = 0x1806;
pub const SIBE_SYLLABLE_BOUNDARY_MARKER: u32 = 0x1807;
pub const MANCHU_COMMA: u32 = 0x1808;
pub const MANCHU_FULL_STOP: u32 = 0x1809;
pub const NIRUGU: u32 = 0x180A;
pub const FVS1: u32 = 0x180B;
pub const FVS2: u32 = 0x180C;
pub const FVS3: u32 = 0x180D;
pub const MVS: u32 = 0x180E;
pub const FVS4: u32 = 0x180F;
pub const DIGIT_ZERO: u32 = 0x1810;
pub const DIGIT_ONE: u32 = 0x1811;
pub const DIGIT_TWO: u32 = 0x1812;
pub const DIGIT_THREE: u32 = 0x1813;
pub const DIGIT_FOUR: u32 = 0x1814;
pub const DIGIT_FIVE: u32 = 0x1815;
pub const DIGIT_SIX: u32 = 0x1816;
pub const DIGIT_SEVEN: u32 = 0x1817;
pub const DIGIT_EIGHT: u32 = 0x1818;
pub const DIGIT_NINE: u32 = 0x1819;

pub const A: u32 = 0x1820;
pub const E: u32 = 0x1821;
pub const I: u32 = 0x1822;
pub const O: u32 = 0x1823;
pub const U: u32 = 0x1824;
pub const OE: u32 = 0x1825;
pub const UE: u32 = 0x1826;
pub const EE: u32 = 0x1827;
pub const NA: u32 = 0x1828;
pub const ANG: u32 = 0x1829;
pub const BA: u32 = 0x182A;
pub const PA: u32 = 0x182B;
pub const QA: u32 = 0x182C;
pub const GA: u32 = 0x182D;
pub const MA: u32 = 0x182E;
pub const LA: u32 = 0x182F;
pub const SA: u32 = 0x1830;
pub const SHA: u32 = 0x1831;
pub const TA: u32 = 0x1832;
pub const DA: u32 = 0x1833;
pub const CHA: u32 = 0x1834;
pub const JA: u32 = 0x1835;
pub const YA: u32 = 0x1836;
pub const RA: u32 = 0x1837;
pub const WA: u32 = 0x1838;
pub const FA: u32 = 0x1839;
pub const KA: u32 = 0x183A;
pub const KHA: u32 = 0x183B;
pub const TSA: u32 = 0x183C;
pub const ZA: u32 = 0x183D;
pub const HAA: u32 = 0x183E;
pub const ZRA: u32 = 0x183F;
pub const LHA: u32 = 0x1840;
pub const ZHI: u32 = 0x1841;
pub const CHI: u32 = 0x1842;

pub fn is_mongolian(codepoint: u32) -> bool {
    (0x1800..=0x18AA).contains(&codepoint)
}

pub fn is_control(codepoint: u32) -> bool {
    (FVS1..=FVS4).contains(&codepoint)
}

pub fn is_fvs(codepoint: u32) -> bool {
    matches!(codepoint, FVS1 | FVS2 | FVS3 | FVS4)
}

pub fn is_letter(codepoint: u32) -> bool {
    (A..=CHI).contains(&codepoint)
}

pub fn is_consonant(codepoint: u32) -> bool {
    (NA..=CHI).contains(&codepoint)
}

#[allow(dead_code)]
pub fn is_vowel(codepoint: u32) -> bool {
    (A..=EE).contains(&codepoint)
}

#[allow(dead_code)]
pub fn is_masculine_vowel(codepoint: u32) -> bool {
    matches!(codepoint, A | O | U)
}

#[allow(dead_code)]
pub fn is_feminine_vowel(codepoint: u32) -> bool {
    matches!(codepoint, E | EE | OE | UE)
}

pub fn needs_long_tooth_u(word: &[u32], index: usize) -> bool {
    if word.get(index) != Some(&OE) && word.get(index) != Some(&UE) {
        return false;
    }

    match index {
        0 => true,
        1 => is_consonant(word[0]),
        2 => is_consonant(word[0]) && is_fvs(word[1]),
        _ => false,
    }
}
