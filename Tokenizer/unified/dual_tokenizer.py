# -*- coding: utf-8 -*-
"""Dual-track tokenizer with one unified id space.

Tracks:
    mn    -> MorphBPE
    zh/en -> HuggingFace BPE tokenizer
    misc  -> byte fallback / punctuation / digits
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

try:
    from ..generic_bpe import HFTrackTokenizer, encode_byte_fallback, is_byte_token
    from ..traditional_mongolian.stemmer import MongolStemmer
    from .encoded import DualTrackResult, EncodedToken
    from .vocab import (
        SEGMENT,
        SPECIAL_TOKENS,
        build_misc_tokens,
        build_unified_vocab,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from Tokenizer.generic_bpe import (
        HFTrackTokenizer,
        encode_byte_fallback,
        is_byte_token,
    )
    from Tokenizer.traditional_mongolian.stemmer import MongolStemmer
    from Tokenizer.unified.encoded import DualTrackResult, EncodedToken
    from Tokenizer.unified.vocab import (
        SEGMENT,
        SPECIAL_TOKENS,
        build_misc_tokens,
        build_unified_vocab,
    )

SPECIAL_TOKEN_TEXTS = tuple(sorted(SPECIAL_TOKENS, key=len, reverse=True))

MONGOLIAN_RANGES = [
    (0x1800, 0x18AF),
    (0x11660, 0x1167F),
]

MONGOLIAN_PUNCTUATION = {
    "\u1800",
    "\u1801",
    "\u1802",
    "\u1803",
    "\u1804",
    "\u1805",
    "\u1806",
    "\u1807",
    "\u1808",
    "\u1809",
}

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
    "\u00a0",
    "\u202f",
}
NNBSP = "\u202f"


@dataclass(frozen=True)
class Span:
    lang: str
    text: str
    start: int
    end: int


def in_ranges(cp: int, ranges: list[tuple[int, int]]) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def char_lang(ch: str) -> str:
    cp = ord(ch)

    if ch in SPACE_CHARS:
        return "space"
    if ch in MONGOLIAN_PUNCTUATION:
        return "misc"
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


def _is_mongolian_char(ch: str) -> bool:
    return in_ranges(ord(ch), MONGOLIAN_RANGES)


def contextual_char_lang(text: str, index: int) -> str:
    ch = text[index]
    if ch == NNBSP:
        prev_is_mn = index > 0 and _is_mongolian_char(text[index - 1])
        next_is_mn = index + 1 < len(text) and _is_mongolian_char(text[index + 1])
        if prev_is_mn and next_is_mn:
            return "mn"
    return char_lang(ch)


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

        lang = contextual_char_lang(text, i)
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


def _strip_hf_boundary_markers(text: str) -> str:
    # HuggingFace tokenizers keep word-boundary markers in their raw token
    # strings (SentencePiece "▁" / GPT-2 "Ġ"). Mirror HF's
    # convert_tokens_to_string behavior so decode() yields natural text
    # instead of artifacts like "▁hello" / "Ġhello".
    return text.replace("\u2581", " ").replace("\u0120", " ")


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
            track="zh",
        )
        self.en = HFTrackTokenizer(
            en_hf_tokenizer,
            prefix="en▁",
            local_to_global=self.en_local_to_global,
            unk_id=self.unk_id,
            track="en",
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
        return self.encode_with_spans(text, add_bos=add_bos, add_eos=add_eos).input_ids

    def encode_with_spans(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> DualTrackResult:
        spans = segment_by_language(text)
        tokens: list[EncodedToken] = []

        if add_bos:
            tokens.append(EncodedToken(self.vocab["<bos>"], "<bos>", "special", -1, -1))

        for span in spans:
            if span.lang == "special":
                tokens.extend(self._encode_special(span))
            elif span.lang == "mn":
                tokens.extend(self._encode_mongolian(span))
            elif span.lang == "zh":
                tokens.extend(self._encode_hf_track(self.zh, span))
            elif span.lang == "en":
                tokens.extend(self._encode_hf_track(self.en, span))
            elif span.lang == "space":
                tokens.extend(self._encode_space(span))
            else:
                tokens.extend(self._encode_misc(span))

        if add_eos:
            tokens.append(EncodedToken(self.vocab["<eos>"], "<eos>", "special", -1, -1))

        return DualTrackResult(
            input_ids=[token.id for token in tokens], tokens=tokens, spans=spans
        )

    def _encode_special(self, span: Span) -> list[EncodedToken]:
        token_id = self.vocab.get(span.text, self.unk_id)
        return [EncodedToken(token_id, span.text, "special", span.start, span.end)]

    def _encode_mongolian(self, span: Span) -> list[EncodedToken]:
        if hasattr(self.morphbpe, "encode_with_offsets"):
            local_tokens = self.morphbpe.encode_with_offsets(span.text)
            tokens: list[EncodedToken] = []
            for item in local_tokens:
                if isinstance(item, dict):
                    local_id = item["id"]
                    text = item["token"]
                    start = item["start"]
                    end = item["end"]
                else:
                    local_id = item.id
                    text = item.token
                    start = item.start
                    end = item.end
                tokens.append(
                    EncodedToken(
                        self.mn_local_to_global.get(local_id, self.unk_id),
                        text,
                        "mn",
                        span.start + start,
                        span.start + end,
                    )
                )
            return tokens

        local_ids = self.morphbpe.encode(span.text)
        return [
            EncodedToken(
                self.mn_local_to_global.get(i, self.unk_id),
                span.text,
                "mn",
                span.start,
                span.end,
            )
            for i in local_ids
        ]

    def _encode_hf_track(
        self, track_tokenizer: HFTrackTokenizer, span: Span
    ) -> list[EncodedToken]:
        encoded = track_tokenizer.encode_with_offsets(span.text, span.start)
        if not encoded and span.text:
            return encode_byte_fallback(
                span.text, self.vocab, self.unk_id, span.start, "misc"
            )

        tokens: list[EncodedToken] = []
        for token in encoded:
            if token.id != self.unk_id:
                tokens.append(token)
                continue

            rel_start = max(0, token.start - span.start)
            rel_end = min(len(span.text), max(rel_start, token.end - span.start))
            surface = span.text[rel_start:rel_end] or token.surface or token.token
            tokens.extend(
                encode_byte_fallback(
                    surface, self.vocab, self.unk_id, token.start, "misc"
                )
            )
        return tokens

    def _encode_space(self, span: Span) -> list[EncodedToken]:
        space_id = self.vocab.get("▁", self.unk_id)
        return [
            EncodedToken(space_id, "▁", "space", pos, pos + 1)
            for pos in range(span.start, span.end)
        ]

    def _encode_misc(self, span: Span) -> list[EncodedToken]:
        return encode_byte_fallback(
            span.text, self.vocab, self.unk_id, span.start, "misc"
        )

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

            if is_byte_token(token):
                try:
                    byte_buf.append(int(token[3:5], 16))
                    continue
                except ValueError:
                    pass

            flush_bytes()

            if token.startswith("zh▁"):
                parts.append(_strip_hf_boundary_markers(token[3:]))
            elif token.startswith("en▁"):
                parts.append(_strip_hf_boundary_markers(token[3:]))
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
        raise ImportError(
            "Tokenizer.morphbpe.MorphBPETokenizer is not implemented yet"
        ) from exc

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
        print(
            "  python -m Tokenizer.unified.dual_tokenizer build <morphbpe.json> [out.json]"
        )
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
            print(
                "Usage: python -m Tokenizer.unified.dual_tokenizer build <morphbpe.json> [out.json]"
            )
            return
        run_build(sys.argv)
        return

    print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
