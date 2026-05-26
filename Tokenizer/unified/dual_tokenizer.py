# -*- coding: utf-8 -*-
"""Dual-track tokenizer with one unified id space.

Tracks:
    mn    -> MorphBPE
    zh/en -> HuggingFace BPE tokenizer
    misc  -> byte fallback / punctuation / digits
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

try:
    from ..multimodal.tokens import MULTIMODAL_SPECIAL_TOKENS
    from ..traditional_mongolian.stemmer import MongolStemmer
except ImportError:  # pragma: no cover - supports direct script execution.
    from Tokenizer.multimodal.tokens import MULTIMODAL_SPECIAL_TOKENS
    from Tokenizer.traditional_mongolian.stemmer import MongolStemmer


BASE_SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "<img>": 4,
    "▁": 5,
    "◈": 6,
}

SPECIAL_TOKENS = {
    **BASE_SPECIAL_TOKENS,
    **MULTIMODAL_SPECIAL_TOKENS,
}

SEGMENT = {
    "special": (0, 16),
    "mongolian": (16, 40000),
    "chinese": (40000, 55000),
    "english": (55000, 63000),
    "misc": (63000, 64000),
}

SPECIAL_TOKEN_TEXTS = tuple(sorted(SPECIAL_TOKENS, key=len, reverse=True))

MONGOLIAN_RANGES = [
    (0x1800, 0x18AF),
    (0x11660, 0x1167F),
]

CJK_RANGES = [
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
]

LATIN_RANGES = [
    (0x0041, 0x005A),
    (0x0061, 0x007A),
    (0x00C0, 0x024F),
    (0x1E00, 0x1EFF),
]

FULLWIDTH_LATIN_RANGES = [
    (0xFF21, 0xFF3A),
    (0xFF41, 0xFF5A),
]

DIGIT_RANGES = [
    (0x0030, 0x0039),
    (0xFF10, 0xFF19),
]

CJK_SYMBOL_RANGES = [
    (0x3000, 0x303F),
    (0xFF00, 0xFFEF),
]

SPACE_CHARS = {
    " ",
    "\t",
    "\n",
    "\r",
    "\u00A0",
    "\u202F",
}


@dataclass(frozen=True)
class Span:
    lang: str
    text: str
    start: int
    end: int


@dataclass
class DualTrackResult:
    ids: list[int]
    spans: list[Span] = field(default_factory=list)


def in_ranges(cp: int, ranges: list[tuple[int, int]]) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def char_lang(ch: str) -> str:
    cp = ord(ch)

    if ch in SPACE_CHARS:
        return "space"
    if in_ranges(cp, MONGOLIAN_RANGES):
        return "mn"
    if in_ranges(cp, DIGIT_RANGES):
        return "misc"
    if in_ranges(cp, LATIN_RANGES) or in_ranges(cp, FULLWIDTH_LATIN_RANGES):
        return "en"
    if in_ranges(cp, CJK_SYMBOL_RANGES):
        return "misc"
    if in_ranges(cp, CJK_RANGES):
        return "zh"

    return "misc"


def special_at(text: str, start: int) -> str | None:
    for token in SPECIAL_TOKEN_TEXTS:
        if text.startswith(token, start):
            return token
    return None


def segment_by_language(text: str) -> list[Span]:
    if not text:
        return []

    spans: list[Span] = []
    start = 0
    cur_lang = ""
    i = 0

    while i < len(text):
        special = special_at(text, i)
        if special is not None:
            if cur_lang and start < i:
                spans.append(Span(cur_lang, text[start:i], start, i))
            end = i + len(special)
            spans.append(Span("special", text[i:end], i, end))
            i = end
            start = i
            cur_lang = ""
            continue

        lang = char_lang(text[i])
        if not cur_lang:
            cur_lang = lang
            start = i
        elif lang != cur_lang:
            spans.append(Span(cur_lang, text[start:i], start, i))
            start = i
            cur_lang = lang
        i += 1

    if cur_lang and start < len(text):
        spans.append(Span(cur_lang, text[start:], start, len(text)))

    return spans


class HFTrackTokenizer:
    def __init__(
        self,
        hf_tokenizer: Any,
        prefix: str,
        local_to_global: dict[int, int],
        unk_id: int,
    ):
        self.hf_tokenizer = hf_tokenizer
        self.prefix = prefix
        self.local_to_global = local_to_global
        self.unk_id = unk_id

    def encode(self, text: str) -> list[int]:
        local_ids = self.hf_tokenizer.encode(text, add_special_tokens=False)
        return [self.local_to_global.get(i, self.unk_id) for i in local_ids]


def make_byte_tokens() -> list[str]:
    return [f"<0x{i:02X}>" for i in range(256)]


def build_misc_tokens() -> list[str]:
    punct = list("0123456789.,!?;:()[]{}\"'-—…。，！？；：（）《》“”‘’、·")
    return list(dict.fromkeys(punct + make_byte_tokens()))


def extract_hf_vocab_tokens(
    model_name: str, lang: str, limit: int
) -> tuple[list[str], dict[str, int]]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("pip install transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    selected: list[str] = []
    for token, _local_id in sorted_vocab:
        cleaned = token.replace("Ġ", " ").replace("▁", " ")
        if not cleaned.strip():
            continue

        if lang == "zh":
            keep = any(char_lang(ch) == "zh" for ch in cleaned)
        elif lang == "en":
            keep = any(char_lang(ch) == "en" for ch in cleaned) and all(
                char_lang(ch) in {"en", "space", "misc"} for ch in cleaned
            )
        else:
            keep = False

        if keep:
            selected.append(token)
        if len(selected) >= limit:
            break

    return selected, vocab


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


class DualTrackTokenizer:
    def __init__(
        self,
        unified_vocab: dict[str, int],
        morphbpe: Any,
        zh_hf_tokenizer: Any,
        en_hf_tokenizer: Any,
    ):
        self.vocab = unified_vocab
        self.id_to_token = {idx: tok for tok, idx in unified_vocab.items()}
        self.unk_id = unified_vocab["<unk>"]
        self.morphbpe = morphbpe

        self.mn_local_to_global = {
            local_id: unified_vocab[token]
            for token, local_id in morphbpe.vocab.items()
            if token in unified_vocab
        }

        self.zh_local_to_global = self._build_hf_map(zh_hf_tokenizer, "zh")
        self.en_local_to_global = self._build_hf_map(en_hf_tokenizer, "en")

        self.zh = HFTrackTokenizer(
            zh_hf_tokenizer,
            prefix="zh▁",
            local_to_global=self.zh_local_to_global,
            unk_id=self.unk_id,
        )
        self.en = HFTrackTokenizer(
            en_hf_tokenizer,
            prefix="en▁",
            local_to_global=self.en_local_to_global,
            unk_id=self.unk_id,
        )

    def _build_hf_map(self, hf_tokenizer: Any, lang: str) -> dict[int, int]:
        result: dict[int, int] = {}
        for token, local_id in hf_tokenizer.get_vocab().items():
            global_id = self.vocab.get(f"{lang}▁{token}")
            if global_id is not None:
                result[local_id] = global_id
        return result

    @property
    def vocab_size(self) -> int:
        return max(self.vocab.values()) + 1

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        return self.encode_with_spans(text, add_bos=add_bos, add_eos=add_eos).ids

    def encode_with_spans(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> DualTrackResult:
        spans = segment_by_language(text)
        ids: list[int] = []

        if add_bos:
            ids.append(self.vocab["<bos>"])

        for span in spans:
            if span.lang == "special":
                ids.extend(self._encode_special(span.text))
            elif span.lang == "mn":
                ids.extend(self._encode_mongolian(span.text))
            elif span.lang == "zh":
                ids.extend(self.zh.encode(span.text))
            elif span.lang == "en":
                ids.extend(self.en.encode(span.text))
            elif span.lang == "space":
                ids.extend(self._encode_space(span.text))
            else:
                ids.extend(self._encode_misc(span.text))

        if add_eos:
            ids.append(self.vocab["<eos>"])

        return DualTrackResult(ids=ids, spans=spans)

    def _encode_special(self, text: str) -> list[int]:
        token_id = self.vocab.get(text)
        return [token_id if token_id is not None else self.unk_id]

    def _encode_mongolian(self, text: str) -> list[int]:
        local_ids = self.morphbpe.encode(text)
        return [self.mn_local_to_global.get(i, self.unk_id) for i in local_ids]

    def _encode_space(self, text: str) -> list[int]:
        space_id = self.vocab.get("▁", self.unk_id)
        return [space_id for _ in text]

    def _encode_misc(self, text: str) -> list[int]:
        ids: list[int] = []
        for ch in text:
            direct = self.vocab.get(ch)
            if direct is not None:
                ids.append(direct)
                continue

            for byte in ch.encode("utf-8"):
                ids.append(self.vocab.get(f"<0x{byte:02X}>", self.unk_id))
        return ids

    def decode(self, ids: list[int]) -> str:
        parts: list[str] = []
        byte_buf: list[int] = []

        def flush_bytes() -> None:
            nonlocal byte_buf
            if byte_buf:
                parts.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                byte_buf = []

        for idx in ids:
            token = self.id_to_token.get(idx, "")
            if token in {"<pad>", "<unk>", "<bos>", "<eos>", "<img>"}:
                flush_bytes()
                continue

            if token.startswith("<0x") and token.endswith(">") and len(token) == 6:
                try:
                    byte_buf.append(int(token[3:5], 16))
                    continue
                except ValueError:
                    pass

            flush_bytes()

            if token.startswith("zh▁"):
                parts.append(token[3:])
            elif token.startswith("en▁"):
                parts.append(token[3:])
            elif token == "▁":
                parts.append(" ")
            elif token == "◈":
                continue
            else:
                parts.append(token)

        flush_bytes()
        return "".join(parts)

    def save(self, path: str, config: dict[str, Any] | None = None) -> None:
        payload = {
            "vocab": self.vocab,
            "segment": SEGMENT,
            "special_tokens": SPECIAL_TOKENS,
            "config": config or {},
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def load_morphbpe_tokenizer(morphbpe_model_path: str, stemmer: MongolStemmer) -> Any:
    try:
        from Tokenizer.morphbpe import MorphBPETokenizer
    except ImportError as exc:
        raise ImportError("Tokenizer.morphbpe.MorphBPETokenizer is not implemented yet") from exc

    return MorphBPETokenizer.from_file(morphbpe_model_path, stemmer)


def build_dual_tokenizer(
    morphbpe_model_path: str,
    zh_source: str = "Qwen/Qwen2.5-0.5B",
    en_source: str = "meta-llama/Llama-3.2-1B",
    n_chinese: int = 15000,
    n_english: int = 8000,
) -> DualTrackTokenizer:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("pip install transformers") from exc

    stemmer = MongolStemmer()
    morphbpe = load_morphbpe_tokenizer(morphbpe_model_path, stemmer)

    zh_tokens, _ = extract_hf_vocab_tokens(zh_source, "zh", n_chinese)
    en_tokens, _ = extract_hf_vocab_tokens(en_source, "en", n_english)

    unified_vocab = build_unified_vocab(
        morphbpe_vocab=morphbpe.vocab,
        chinese_tokens=zh_tokens,
        english_tokens=en_tokens,
        misc_tokens=build_misc_tokens(),
    )

    zh_hf = AutoTokenizer.from_pretrained(zh_source)
    en_hf = AutoTokenizer.from_pretrained(en_source)

    return DualTrackTokenizer(
        unified_vocab=unified_vocab,
        morphbpe=morphbpe,
        zh_hf_tokenizer=zh_hf,
        en_hf_tokenizer=en_hf,
    )


def run_segment(text: str) -> None:
    for span in segment_by_language(text):
        print(f"[{span.lang}] {span.start}:{span.end} {span.text!r}")


def run_build(args: list[str]) -> None:
    morphbpe_path = args[2]
    out_path = args[3] if len(args) > 3 else "unified_tokenizer.json"

    tokenizer = build_dual_tokenizer(morphbpe_path)
    tokenizer.save(
        out_path,
        config={
            "type": "dual_track_tokenizer",
            "tracks": ["mn:morphbpe", "zh:hf", "en:hf", "misc:byte"],
        },
    )

    print(f"vocab_size={tokenizer.vocab_size}")
    print(f"saved={out_path}")


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m Tokenizer.unified.dual_tokenizer segment <text>")
        print("  python -m Tokenizer.unified.dual_tokenizer build <morphbpe.json> [out.json]")
        return

    cmd = sys.argv[1]
    if cmd == "segment":
        if len(sys.argv) < 3:
            print("Usage: python -m Tokenizer.unified.dual_tokenizer segment <text>")
            return
        run_segment(sys.argv[2])
        return

    if cmd == "build":
        if len(sys.argv) < 3:
            print("Usage: python -m Tokenizer.unified.dual_tokenizer build <morphbpe.json> [out.json]")
            return
        run_build(sys.argv)
        return

    print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
