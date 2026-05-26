# -*- coding: utf-8 -*-
"""Core traditional Mongolian suffix inventory for tokenizer experiments."""

NNBSP = "\u202F"
NIRUGU = "\u180A"
FVS1 = "\u180B"
FVS2 = "\u180C"
FVS3 = "\u180D"
MVS = "\u180E"
FVS4 = "\u180F"

CONTROL_CHARS = {
    "NNBSP": NNBSP,
    "NIRUGU": NIRUGU,
    "FVS1": FVS1,
    "FVS2": FVS2,
    "FVS3": FVS3,
    "MVS": MVS,
    "FVS4": FVS4,
}


CASE_SUFFIXES = [
    {
        "id": "GEN",
        "name": "genitive",
        "surface": ["ᠤᠨ", "ᠦᠨ", "ᠶᠢᠨ", "ᠢᠨ", "ᠨᠤ", "ᠨᠦ"],
        "type": "case",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 2,
        "notes": "of",
    },
    {
        "id": "ACC",
        "name": "accusative",
        "surface": ["ᠶᠢ", "ᠢ"],
        "type": "case",
        "attaches_to": "noun",
        "harmony": "none",
        "order": 2,
        "notes": "direct object",
    },
    {
        "id": "DAT_LOC",
        "name": "dative-locative",
        "surface": ["ᠳᠤ", "ᠳᠦ", "ᠲᠤ", "ᠲᠦ", "ᠳᠤᠷ", "ᠳᠦᠷ", "ᠲᠤᠷ", "ᠲᠦᠷ", "ᠠ", "ᠡ"],
        "type": "case",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 2,
        "notes": "to/in/at",
    },
    {
        "id": "ABL",
        "name": "ablative",
        "surface": ["ᠠᠴᠠ", "ᠡᠴᠡ", "ᠴᠠ", "ᠴᠡ"],
        "type": "case",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 2,
        "notes": "from",
    },
    {
        "id": "INST",
        "name": "instrumental",
        "surface": ["ᠪᠠᠷ", "ᠪᠡᠷ", "ᠢᠶᠠᠷ", "ᠢᠶᠡᠷ", "ᠶᠠᠷ", "ᠶᠡᠷ"],
        "type": "case",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 2,
        "notes": "by/with",
    },
    {
        "id": "COM",
        "name": "comitative",
        "surface": ["ᠯᠤᠭ᠎ᠠ", "ᠯᠦᠭᠡ", "ᠲᠠᠢ", "ᠲᠡᠢ", "ᠲᠣᠢ"],
        "type": "case",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 2,
        "notes": "with",
    },
    {
        "id": "DIR",
        "name": "directive",
        "surface": ["ᠤᠷᠤᠭᠤ", "ᠦᠷᠦᠭᠦ", "ᠷᠤ", "ᠷᠦ"],
        "type": "case",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 2,
        "notes": "towards",
    },
    {
        "id": "TERM",
        "name": "terminative",
        "surface": ["ᠬᠦᠷᠲᠡᠯ᠎ᠡ", "ᠬᠦᠷᠲᠡᠯᠡ"],
        "type": "case",
        "attaches_to": "noun",
        "harmony": "none",
        "order": 2,
        "notes": "until",
    },
]


PLURAL_SUFFIXES = [
    {
        "id": "PL_NAR",
        "name": "plural-nar",
        "surface": ["ᠨᠠᠷ", "ᠨᠡᠷ"],
        "type": "plural",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 1,
        "notes": "human",
    },
    {
        "id": "PL_NUGUD",
        "name": "plural-nugud",
        "surface": ["ᠨᠤᠭᠤᠳ", "ᠨᠦᠭᠦᠳ"],
        "type": "plural",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 1,
        "notes": "general",
    },
    {
        "id": "PL_UD",
        "name": "plural-ud",
        "surface": ["ᠤᠳ", "ᠦᠳ"],
        "type": "plural",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 1,
        "notes": "general",
    },
    {
        "id": "PL_D",
        "name": "plural-d",
        "surface": ["ᠳ"],
        "type": "plural",
        "attaches_to": "noun",
        "harmony": "none",
        "order": 1,
        "notes": "short",
    },
    {
        "id": "PL_S",
        "name": "plural-s",
        "surface": ["ᠰ"],
        "type": "plural",
        "attaches_to": "noun",
        "harmony": "none",
        "order": 1,
        "notes": "short",
    },
    {
        "id": "PL_CHUD",
        "name": "plural-chud",
        "surface": ["ᠴᠤᠳ", "ᠴᠦᠳ"],
        "type": "plural",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 1,
        "notes": "collective",
    },
    {
        "id": "PL_DUD",
        "name": "plural-dud",
        "surface": ["ᠳᠤᠳ", "ᠳᠦᠳ"],
        "type": "plural",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 1,
        "notes": "collective",
    },
]


POSSESSIVE_SUFFIXES = [
    {
        "id": "POSS_1SG",
        "name": "possessive-1sg",
        "surface": ["ᠮᠢᠨᠢ", "ᠮᠢᠨ"],
        "type": "possessive",
        "attaches_to": "noun",
        "harmony": "none",
        "order": 3,
        "notes": "my",
    },
    {
        "id": "POSS_2SG",
        "name": "possessive-2sg",
        "surface": ["ᠴᠢᠨᠢ", "ᠴᠢᠨ"],
        "type": "possessive",
        "attaches_to": "noun",
        "harmony": "none",
        "order": 3,
        "notes": "your",
    },
    {
        "id": "POSS_3",
        "name": "possessive-3",
        "surface": ["ᠨᠢ", "ᠢᠨᠤ", "ᠶᠢᠨᠤ"],
        "type": "possessive",
        "attaches_to": "noun",
        "harmony": "none",
        "order": 3,
        "notes": "his/her/its",
    },
    {
        "id": "POSS_1PL",
        "name": "possessive-1pl",
        "surface": ["ᠮᠠᠨᠤ", "ᠮᠠᠨᠢ", "ᠪᠢᠳᠡᠨᠦ"],
        "type": "possessive",
        "attaches_to": "noun",
        "harmony": "none",
        "order": 3,
        "notes": "our",
    },
    {
        "id": "POSS_2PL",
        "name": "possessive-2pl",
        "surface": ["ᠲᠠᠨᠤ", "ᠲᠠᠨᠢ"],
        "type": "possessive",
        "attaches_to": "noun",
        "harmony": "none",
        "order": 3,
        "notes": "your-plural",
    },
    {
        "id": "REFL_POSS",
        "name": "reflexive-possessive",
        "surface": ["ᠪᠠᠨ", "ᠪᠡᠨ", "ᠢᠶᠠᠨ", "ᠢᠶᠡᠨ", "ᠶᠠᠨ", "ᠶᠡᠨ"],
        "type": "possessive",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 3,
        "notes": "own",
    },
]


VOICE_SUFFIXES = [
    {
        "id": "CAUS_GUL",
        "name": "causative",
        "surface": ["ᠭᠤᠯ", "ᠭᠦᠯ", "ᠤᠯ", "ᠦᠯ"],
        "type": "voice",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 0,
        "notes": "causative",
    },
    {
        "id": "CAUS_GA",
        "name": "causative",
        "surface": ["ᠭᠠ", "ᠭᠡ", "ᠭᠠᠯ", "ᠭᠡᠯ"],
        "type": "voice",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 0,
        "notes": "causative",
    },
    {
        "id": "PASS",
        "name": "passive",
        "surface": ["ᠭᠳᠠ", "ᠭᠳᠡ", "ᠳᠠ", "ᠳᠡ", "ᠲᠠ", "ᠲᠡ"],
        "type": "voice",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 0,
        "notes": "passive",
    },
    {
        "id": "RECIP",
        "name": "reciprocal",
        "surface": ["ᠯᠴᠠ", "ᠯᠴᠡ"],
        "type": "voice",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 0,
        "notes": "reciprocal",
    },
    {
        "id": "COOP",
        "name": "cooperative",
        "surface": ["ᠯᠳᠤ", "ᠯᠳᠦ", "ᠯᠳᠠ", "ᠯᠳᠡ"],
        "type": "voice",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 0,
        "notes": "cooperative",
    },
]


TENSE_SUFFIXES = [
    {
        "id": "FIN_PAST_BA",
        "name": "finite-past",
        "surface": ["ᠪᠠ", "ᠪᠡ"],
        "type": "tense",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "finite",
    },
    {
        "id": "FIN_PAST_LUGA",
        "name": "finite-past",
        "surface": ["ᠯᠤᠭ᠎ᠠ", "ᠯᠦᠭᠡ"],
        "type": "tense",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "finite",
    },
    {
        "id": "FIN_PRES_MUI",
        "name": "finite-present",
        "surface": ["ᠮᠤᠢ", "ᠮᠦᠢ"],
        "type": "tense",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "finite",
    },
    {
        "id": "FIN_PRES_NAM",
        "name": "finite-present",
        "surface": ["ᠨᠠᠮ", "ᠨᠡᠮ"],
        "type": "tense",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "finite",
    },
    {
        "id": "FIN_FUT_QU",
        "name": "finite-future",
        "surface": ["ᠬᠤ", "ᠬᠦ"],
        "type": "tense",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "finite/participle",
    },
]


MOOD_SUFFIXES = [
    {
        "id": "IMP_2",
        "name": "imperative-2",
        "surface": ["ᠠ", "ᠡ", ""],
        "type": "mood",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "imperative",
    },
    {
        "id": "IMP_POLITE",
        "name": "imperative-polite",
        "surface": ["ᠭᠠᠷᠠᠢ", "ᠭᠡᠷᠡᠢ", "ᠠᠷᠠᠢ", "ᠡᠷᠡᠢ"],
        "type": "mood",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "polite imperative",
    },
    {
        "id": "IMP_HON",
        "name": "imperative-honorific",
        "surface": ["ᠭᠲᠤᠨ", "ᠭᠲᠦᠨ", "ᠲᠤᠨ", "ᠲᠦᠨ"],
        "type": "mood",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "honorific imperative",
    },
    {
        "id": "OPT_1",
        "name": "optative-1",
        "surface": ["ᠶ᠎ᠠ", "ᠶ᠎ᠡ", "ᠰᠤᠭᠠᠢ", "ᠰᠦᠭᠡᠢ"],
        "type": "mood",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "let me/us",
    },
    {
        "id": "VOLITIVE",
        "name": "volitive",
        "surface": ["ᠰᠤᠭᠠᠢ", "ᠰᠦᠭᠡᠢ"],
        "type": "mood",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "volitive",
    },
    {
        "id": "PRECATIVE",
        "name": "precative",
        "surface": ["ᠲᠤᠭᠠᠢ", "ᠲᠦᠭᠡᠢ"],
        "type": "mood",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "may",
    },
    {
        "id": "PROHIBITIVE",
        "name": "prohibitive",
        "surface": ["ᠪᠤᠤ"],
        "type": "mood",
        "attaches_to": "verb",
        "harmony": "none",
        "order": 1,
        "notes": "negative imperative marker",
    },
]


ASPECT_SUFFIXES = [
    {
        "id": "PFV_GAD",
        "name": "perfective",
        "surface": ["ᠭᠠᠳ", "ᠭᠡᠳ", "ᠠᠳ", "ᠡᠳ"],
        "type": "aspect",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "perfective",
    },
    {
        "id": "HAB_DAG",
        "name": "habitual",
        "surface": ["ᠳᠠᠭ", "ᠳᠡᠭ", "ᠲᠠᠭ", "ᠲᠡᠭ"],
        "type": "aspect",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "habitual",
    },
    {
        "id": "PROG_JU",
        "name": "progressive-connective",
        "surface": ["ᠵᠤ", "ᠵᠦ", "ᠴᠤ", "ᠴᠦ"],
        "type": "aspect",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "connective",
    },
]


CONVERB_SUFFIXES = [
    {
        "id": "CVB_COORD_JU",
        "name": "converb-coordinate",
        "surface": ["ᠵᠤ", "ᠵᠦ", "ᠴᠤ", "ᠴᠦ"],
        "type": "converb",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "and",
    },
    {
        "id": "CVB_SIMUL_N",
        "name": "converb-simultaneous",
        "surface": ["ᠨ"],
        "type": "converb",
        "attaches_to": "verb",
        "harmony": "none",
        "order": 1,
        "notes": "while",
    },
    {
        "id": "CVB_COND_BAL",
        "name": "converb-conditional",
        "surface": ["ᠪᠠᠯ", "ᠪᠡᠯ", "ᠪᠤᠯ", "ᠪᠦᠯ"],
        "type": "converb",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "if",
    },
    {
        "id": "CVB_PFV_GAD",
        "name": "converb-perfective",
        "surface": ["ᠭᠠᠳ", "ᠭᠡᠳ", "ᠠᠳ", "ᠡᠳ"],
        "type": "converb",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "after",
    },
    {
        "id": "CVB_LIMIT_TALA",
        "name": "converb-limitative",
        "surface": ["ᠲᠠᠯ᠎ᠠ", "ᠲᠡᠯ᠎ᠡ"],
        "type": "converb",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "until",
    },
    {
        "id": "CVB_CONC_BACHU",
        "name": "converb-concessive",
        "surface": ["ᠪᠠᠴᠤ", "ᠪᠡᠴᠦ"],
        "type": "converb",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "although",
    },
    {
        "id": "CVB_PURP_RA",
        "name": "converb-purposive",
        "surface": ["ᠷᠠ", "ᠷᠡ"],
        "type": "converb",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "in order to",
    },
    {
        "id": "CVB_PREP_MAGCA",
        "name": "converb-immediate",
        "surface": ["ᠮᠠᠭᠴᠠ", "ᠮᠡᠭᠴᠡ"],
        "type": "converb",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "as soon as",
    },
]


PARTICIPLE_SUFFIXES = [
    {
        "id": "PTCP_PAST_GSAN",
        "name": "participle-past",
        "surface": ["ᠭᠰᠠᠨ", "ᠭᠰᠡᠨ", "ᠰᠠᠨ", "ᠰᠡᠨ"],
        "type": "participle",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "past",
    },
    {
        "id": "PTCP_FUT_QU",
        "name": "participle-future",
        "surface": ["ᠬᠤ", "ᠬᠦ"],
        "type": "participle",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "future",
    },
    {
        "id": "PTCP_HAB_DAG",
        "name": "participle-habitual",
        "surface": ["ᠳᠠᠭ", "ᠳᠡᠭ", "ᠲᠠᠭ", "ᠲᠡᠭ"],
        "type": "participle",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "habitual",
    },
    {
        "id": "PTCP_AGENT_GCHI",
        "name": "participle-agentive",
        "surface": ["ᠭᠴᠢ", "ᠴᠢ"],
        "type": "participle",
        "attaches_to": "verb",
        "harmony": "none",
        "order": 1,
        "notes": "agentive",
    },
    {
        "id": "PTCP_PRES_A",
        "name": "participle-present",
        "surface": ["ᠠ", "ᠡ"],
        "type": "participle",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "present",
    },
    {
        "id": "PTCP_POT_MAR",
        "name": "participle-potential",
        "surface": ["ᠮᠠᠷ", "ᠮᠡᠷ"],
        "type": "participle",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "potential",
    },
]


NEGATION_SUFFIXES = [
    {
        "id": "NEG_UGEI",
        "name": "negative",
        "surface": ["ᠦᠭᠡᠢ", "ᠤᠭᠠᠢ"],
        "type": "negation",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 2,
        "notes": "not",
    },
    {
        "id": "NEG_ES",
        "name": "negative-past",
        "surface": ["ᠡᠰᠡ", "ᠡᠰ"],
        "type": "negation",
        "attaches_to": "verb",
        "harmony": "none",
        "order": 2,
        "notes": "negative auxiliary",
    },
]


DERIVATIONAL_SUFFIXES = [
    {
        "id": "DER_N_ADJ_TAI",
        "name": "denominal-adjective",
        "surface": ["ᠲᠠᠢ", "ᠲᠡᠢ", "ᠲᠣᠢ"],
        "type": "derivational",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 0,
        "notes": "with",
    },
    {
        "id": "DER_N_ADJ_UGEI",
        "name": "denominal-negative-adjective",
        "surface": ["ᠦᠭᠡᠢ", "ᠤᠭᠠᠢ"],
        "type": "derivational",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 0,
        "notes": "without",
    },
    {
        "id": "DER_N_V_LA",
        "name": "denominal-verb",
        "surface": ["ᠯᠠ", "ᠯᠡ"],
        "type": "derivational",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 0,
        "notes": "verbalizer",
    },
    {
        "id": "DER_N_V_JI",
        "name": "denominal-verb",
        "surface": ["ᠵᠢ", "ᠴᠢ"],
        "type": "derivational",
        "attaches_to": "noun",
        "harmony": "none",
        "order": 0,
        "notes": "verbalizer",
    },
    {
        "id": "DER_V_N_L",
        "name": "deverbal-noun",
        "surface": ["ᠯ"],
        "type": "derivational",
        "attaches_to": "verb",
        "harmony": "none",
        "order": 0,
        "notes": "nominalizer",
    },
    {
        "id": "DER_V_N_GA",
        "name": "deverbal-noun",
        "surface": ["ᠭᠠ", "ᠭᠡ", "ᠠ", "ᠡ"],
        "type": "derivational",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 0,
        "notes": "nominalizer",
    },
    {
        "id": "DER_V_N_LGA",
        "name": "deverbal-noun",
        "surface": ["ᠯᠭ᠎ᠠ", "ᠯᠭᠡ"],
        "type": "derivational",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 0,
        "notes": "nominalizer",
    },
    {
        "id": "DER_AGENT_CHI",
        "name": "agentive",
        "surface": ["ᠴᠢ", "ᠭᠴᠢ"],
        "type": "derivational",
        "attaches_to": "any",
        "harmony": "none",
        "order": 0,
        "notes": "agent",
    },
    {
        "id": "DER_ABSTRACT_LIG",
        "name": "abstract",
        "surface": ["ᠯᠢᠭ"],
        "type": "derivational",
        "attaches_to": "any",
        "harmony": "none",
        "order": 0,
        "notes": "abstract",
    },
    {
        "id": "DER_QUALITY_TU",
        "name": "quality",
        "surface": ["ᠲᠤ", "ᠲᠦ", "ᠳᠤ", "ᠳᠦ"],
        "type": "derivational",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 0,
        "notes": "having quality",
    },
    {
        "id": "DER_DIM_QAN",
        "name": "diminutive",
        "surface": ["ᠬᠠᠨ", "ᠬᠡᠨ", "ᠬᠤᠨ", "ᠬᠦᠨ"],
        "type": "derivational",
        "attaches_to": "adj",
        "harmony": "all_variants",
        "order": 0,
        "notes": "diminutive",
    },
]


CORE_TRADITIONAL_MONGOLIAN_SUFFIXES = (
    CASE_SUFFIXES
    + PLURAL_SUFFIXES
    + POSSESSIVE_SUFFIXES
    + VOICE_SUFFIXES
    + TENSE_SUFFIXES
    + MOOD_SUFFIXES
    + ASPECT_SUFFIXES
    + CONVERB_SUFFIXES
    + PARTICIPLE_SUFFIXES
    + NEGATION_SUFFIXES
    + DERIVATIONAL_SUFFIXES
)

ALL_SUFFIXES = CORE_TRADITIONAL_MONGOLIAN_SUFFIXES

ALL_SUFFIXES_BY_ORDER = sorted(
    ALL_SUFFIXES,
    key=lambda item: (
        -item.get("order", 0),
        -max(len(surface) for surface in item.get("surface", [""])),
        item.get("id", ""),
    ),
)


def strip_controls(text):
    """Remove suffix-control characters for loose surface matching."""
    for ch in (FVS1, FVS2, FVS3, FVS4, MVS):
        text = text.replace(ch, "")
    return text.replace(NNBSP, "").replace(NIRUGU, "")


def with_nnbsp(suffix):
    return NNBSP + suffix if suffix else suffix


def iter_surfaces(include_nnbsp=False, include_empty=False):
    for item in ALL_SUFFIXES_BY_ORDER:
        for surface in item["surface"]:
            if not surface and not include_empty:
                continue
            yield item, with_nnbsp(surface) if include_nnbsp else surface


def validate_suffix_inventory():
    ids = [item["id"] for item in ALL_SUFFIXES]
    duplicate_ids = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicate_ids:
        raise ValueError(f"Duplicate suffix ids: {duplicate_ids}")

    for item in ALL_SUFFIXES:
        if not item.get("surface"):
            raise ValueError(f"Suffix {item['id']} has no surfaces")
        if any(not isinstance(surface, str) for surface in item["surface"]):
            raise TypeError(f"Suffix {item['id']} has a non-string surface")

    return True


validate_suffix_inventory()
