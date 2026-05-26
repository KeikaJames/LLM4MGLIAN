use crate::mongol;

pub const MENKSOFT_START: u32 = 0xE234;
pub const MENKSOFT_END: u32 = 0xE34F;
pub const UNKNOWN_SPACE: u32 = 0xE262;
pub const NONBREAKING_SPACE: u32 = 0xE263;

pub fn is_menksoft(codepoint: u32) -> bool {
    (MENKSOFT_START..=MENKSOFT_END).contains(&codepoint)
}

pub fn is_menksoft_or_space(codepoint: u32) -> bool {
    is_menksoft(codepoint) || is_space(codepoint)
}

pub fn is_space(codepoint: u32) -> bool {
    codepoint == UNKNOWN_SPACE || codepoint == NONBREAKING_SPACE
}

pub fn is_letter(codepoint: u32) -> bool {
    (0xE264..=MENKSOFT_END).contains(&codepoint)
}

pub fn is_consonant(codepoint: u32) -> bool {
    (0xE2B1..=0xE34F).contains(&codepoint)
}

pub fn is_vowel(codepoint: u32) -> bool {
    (0xE264..0xE2B1).contains(&codepoint)
}

pub fn is_masculine_vowel(codepoint: u32) -> bool {
    (0xE264..0xE270).contains(&codepoint) || (0xE283..0xE293).contains(&codepoint)
}

pub fn is_feminine_vowel(codepoint: u32) -> bool {
    (0xE270..0xE279).contains(&codepoint) || (0xE293..0xE2B1).contains(&codepoint)
}

pub fn is_feminine_vowel_or_i(codepoint: u32) -> bool {
    is_feminine_vowel(codepoint) || is_i(codepoint)
}

pub fn is_a(codepoint: u32) -> bool {
    (0xE264..=0xE26F).contains(&codepoint)
}

pub fn is_i(codepoint: u32) -> bool {
    (0xE279..0xE283).contains(&codepoint)
}

pub fn is_m(codepoint: u32) -> bool {
    (0xE2F1..=0xE2F6).contains(&codepoint)
}

pub fn is_long_tooth_i(codepoint: u32) -> bool {
    matches!(codepoint, 0xE27E | 0xE27F | 0xE280 | 0xE321)
}

pub fn word_contains_masculine_vowel(word: &[u32]) -> bool {
    word.iter().copied().any(is_masculine_vowel)
}

#[allow(dead_code)]
pub fn is_initial_isolate_glyph(codepoint: u32) -> bool {
    matches!(
        codepoint,
        0xE264 | 0xE265 | 0xE266 | 0xE267
            | 0xE270
            | 0xE271
            | 0xE272
            | 0xE279
            | 0xE27A
            | 0xE280
            | 0xE283
            | 0xE284
            | 0xE28B
            | 0xE28C
            | 0xE293
            | 0xE294
            | 0xE295
            | 0xE2A0
            | 0xE2A1
            | 0xE2A2
            | 0xE2AD
            | 0xE2AE
            | 0xE2B1..=0xE2B4
            | 0xE2C1
            | 0xE2C2
            | 0xE2C7
            | 0xE2C8
            | 0xE2C9
            | 0xE2CD
            | 0xE2CE..=0xE2D5
            | 0xE2E1..=0xE2E6
            | 0xE2F1
            | 0xE2F2
            | 0xE2F7
            | 0xE2F8
            | 0xE2FD
            | 0xE2FE
            | 0xE303
            | 0xE304
            | 0xE308
            | 0xE309
            | 0xE30E..=0xE310
            | 0xE315
            | 0xE319
            | 0xE31A
            | 0xE31E
            | 0xE321
            | 0xE322
            | 0xE323
            | 0xE329
            | 0xE32D
            | 0xE32E
            | 0xE332
            | 0xE333
            | 0xE334
            | 0xE339
            | 0xE33A
            | 0xE33F
            | 0xE342
            | 0xE345
            | 0xE348
            | 0xE34B
    )
}

pub fn to_unicode_nominal(codepoint: u32) -> Option<&'static [u32]> {
    Some(match codepoint {
        0xE234 => &[mongol::BIRGA],
        0xE235 => &[mongol::ELLIPSIS],
        0xE236 => &[mongol::COMMA],
        0xE237 => &[mongol::FULL_STOP],
        0xE238 => &[mongol::COLON],
        0xE239 => &[mongol::FOUR_DOTS],
        0xE23A => &[mongol::TODO_SOFT_HYPHEN],
        0xE23B => &[mongol::SIBE_SYLLABLE_BOUNDARY_MARKER],
        0xE23C => &[mongol::MANCHU_COMMA],
        0xE23D => &[mongol::MANCHU_FULL_STOP],
        0xE23E => &[mongol::NIRUGU],
        0xE23F => &[mongol::BIRGA, mongol::FVS1],
        0xE240 => &[mongol::BIRGA, mongol::FVS2],
        0xE241 => &[mongol::BIRGA, mongol::FVS3],
        0xE242 => &[mongol::BIRGA, mongol::FVS4],
        0xE243 => &[mongol::MIDDLE_DOT],
        0xE244 => &[mongol::DIGIT_ZERO],
        0xE245 => &[mongol::DIGIT_ONE],
        0xE246 => &[mongol::DIGIT_TWO],
        0xE247 => &[mongol::DIGIT_THREE],
        0xE248 => &[mongol::DIGIT_FOUR],
        0xE249 => &[mongol::DIGIT_FIVE],
        0xE24A => &[mongol::DIGIT_SIX],
        0xE24B => &[mongol::DIGIT_SEVEN],
        0xE24C => &[mongol::DIGIT_EIGHT],
        0xE24D => &[mongol::DIGIT_NINE],
        0xE24E => &[mongol::QUESTION_EXCLAMATION],
        0xE24F => &[mongol::EXCLAMATION_QUESTION],
        0xE250 => &[mongol::FULLWIDTH_EXCLAMATION],
        0xE251 => &[mongol::FULLWIDTH_QUESTION],
        0xE252 => &[mongol::FULLWIDTH_SEMICOLON],
        0xE253 => &[mongol::FULLWIDTH_LEFT_PARENTHESIS],
        0xE254 => &[mongol::FULLWIDTH_RIGHT_PARENTHESIS],
        0xE255 => &[mongol::LEFT_ANGLE_BRACKET],
        0xE256 => &[mongol::RIGHT_ANGLE_BRACKET],
        0xE257 => &[mongol::VERTICAL_LEFT_TORTOISE_SHELL_BRACKET],
        0xE258 => &[mongol::VERTICAL_RIGHT_TORTOISE_SHELL_BRACKET],
        0xE259 => &[mongol::LEFT_DOUBLE_ANGLE_BRACKET],
        0xE25A => &[mongol::RIGHT_DOUBLE_ANGLE_BRACKET],
        0xE25B => &[mongol::VERTICAL_LEFT_WHITE_CORNER_BRACKET],
        0xE25C => &[mongol::VERTICAL_RIGHT_WHITE_CORNER_BRACKET],
        0xE25D => &[mongol::VERTICAL_COMMA],
        0xE25E => &[mongol::PUNCTUATION_X],
        0xE25F => &[mongol::REFERENCE_MARK],
        0xE260 => &[mongol::VERTICAL_EN_DASH],
        0xE261 => &[mongol::VERTICAL_EM_DASH],
        0xE262 => &[mongol::SPACE],
        0xE263 => &[mongol::MVS],
        0xE264..=0xE26F => &[mongol::A],
        0xE270..=0xE278 => &[mongol::E],
        0xE279..=0xE282 => &[mongol::I],
        0xE283..=0xE28A => &[mongol::O],
        0xE28B..=0xE292 => &[mongol::U],
        0xE293..=0xE29F => &[mongol::OE],
        0xE2A0..=0xE2AC => &[mongol::UE],
        0xE2AD..=0xE2B0 => &[mongol::EE],
        0xE2B1..=0xE2BA | 0xE2BF | 0xE2C0 => &[mongol::NA],
        0xE2BB..=0xE2BE => &[mongol::ANG],
        0xE2C1..=0xE2C7 => &[mongol::BA],
        0xE2C8..=0xE2CD => &[mongol::PA],
        0xE2CE..=0xE2E0 => &[mongol::QA],
        0xE2E1..=0xE2F0 => &[mongol::GA],
        0xE2F1..=0xE2F6 => &[mongol::MA],
        0xE2F7..=0xE2FC => &[mongol::LA],
        0xE2FD..=0xE302 => &[mongol::SA],
        0xE303..=0xE307 => &[mongol::SHA],
        0xE308..=0xE30D => &[mongol::TA],
        0xE30E..=0xE314 => &[mongol::DA],
        0xE315..=0xE317 => &[mongol::CHA],
        0xE318..=0xE31D => &[mongol::JA],
        0xE31E..=0xE321 => &[mongol::YA],
        0xE322..=0xE328 => &[mongol::RA],
        0xE329..=0xE32C => &[mongol::WA],
        0xE32D..=0xE332 => &[mongol::FA],
        0xE333..=0xE338 => &[mongol::KA],
        0xE339..=0xE33E => &[mongol::KHA],
        0xE33F..=0xE341 => &[mongol::TSA],
        0xE342..=0xE344 => &[mongol::ZA],
        0xE345..=0xE347 => &[mongol::HAA],
        0xE348..=0xE34A => &[mongol::ZRA],
        0xE34B..=0xE34D => &[mongol::LHA],
        0xE34E => &[mongol::ZHI],
        0xE34F => &[mongol::CHI],
        _ => return None,
    })
}
