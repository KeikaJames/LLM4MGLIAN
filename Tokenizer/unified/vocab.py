# -*- coding: utf-8 -*-
"""Unified tokenizer id-space registry and vocabulary builders."""

from __future__ import annotations

try:
    from ..multimodal.tokens import MULTIMODAL_SPECIAL_TOKENS
except ImportError:  # pragma: no cover
    from Tokenizer.multimodal.tokens import MULTIMODAL_SPECIAL_TOKENS


BASE_SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "<img>": 4,
    "▁": 17,
    "◈": 18,
}

SPECIAL_TOKENS = {
    **BASE_SPECIAL_TOKENS,
    **MULTIMODAL_SPECIAL_TOKENS,
}

SEGMENT = {
    "special": (0, 256),
    "mongolian": (256, 24576),
    "chinese": (24576, 49152),
    "english": (49152, 63488),
    "misc": (63488, 65536),
}


def make_byte_tokens() -> list[str]:
    return [f"<0x{i:02X}>" for i in range(256)]


def build_misc_tokens() -> list[str]:
    punct = list("0123456789.,!?;:()[]{}\"'-—…。，！？；：（）《》“”‘’、·")
    mongolian_punct = list("᠀᠁᠂᠃᠄᠅᠆᠇᠈᠉")
    return list(dict.fromkeys(punct + mongolian_punct + make_byte_tokens()))


def build_unified_vocab(
    morphbpe_vocab: dict[str, int],
    chinese_tokens: list[str],
    english_tokens: list[str],
    misc_tokens: list[str] | None = None,
) -> dict[str, int]:
    unified: dict[str, int] = dict(SPECIAL_TOKENS)

    mn_lo, mn_hi = SEGMENT["mongolian"]
    next_id = mn_lo
    for token, _local_id in sorted(morphbpe_vocab.items(), key=lambda x: x[1]):
        if token in SPECIAL_TOKENS:
            continue
        if next_id >= mn_hi:
            break
        unified[token] = next_id
        next_id += 1

    zh_lo, zh_hi = SEGMENT["chinese"]
    next_id = zh_lo
    for token in chinese_tokens:
        if next_id >= zh_hi:
            break
        unified[f"zh▁{token}"] = next_id
        next_id += 1

    en_lo, en_hi = SEGMENT["english"]
    next_id = en_lo
    for token in english_tokens:
        if next_id >= en_hi:
            break
        unified[f"en▁{token}"] = next_id
        next_id += 1

    mi_lo, mi_hi = SEGMENT["misc"]
    next_id = mi_lo
    for token in misc_tokens or []:
        if next_id >= mi_hi:
            break
        if token in unified:
            continue
        unified[token] = next_id
        next_id += 1

    return unified
