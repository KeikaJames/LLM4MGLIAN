# -*- coding: utf-8 -*-
"""Persisted tokenizer bundle for reproducible pretraining data builds."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from typing import Any

from Tokenizer.morphbpe import MorphBPETokenizer
from Tokenizer.multimodal import MultimodalProcessor
from Tokenizer.traditional_mongolian.stemmer import MongolStemmer

from .dual_tokenizer import DualTrackTokenizer
from .vocab import SPECIAL_TOKENS, build_misc_tokens, build_unified_vocab


BUNDLE_VERSION = 1
CONFIG_NAME = "config.json"
MORPHBPE_NAME = "morphbpe.json"
VOCAB_NAME = "vocab.json"


@dataclass
class TokenizerBundleConfig:
    version: int
    morphbpe_file: str
    zh_source: str
    en_source: str
    patch_size: int = 14
    merge_size: int = 2
    temporal_patch_size: int = 2
    use_smoke_hf: bool = False


class _SmokeHFTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self._vocab = dict(vocab)
        self._id_to_token = {idx: tok for tok, idx in self._vocab.items()}

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def convert_ids_to_tokens(self, local_id: int) -> str:
        return self._id_to_token.get(local_id, str(local_id))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [local_id for local_id, _start, _end in self._tokenize(text)]

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[Any]]:
        pieces = self._tokenize(text)
        result: dict[str, list[Any]] = {
            "input_ids": [local_id for local_id, _start, _end in pieces]
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [(start, end) for _id, start, end in pieces]
        return result

    def _tokenize(self, text: str) -> list[tuple[int, int, int]]:
        out: list[tuple[int, int, int]] = []
        cursor = 0
        ordered = sorted(self._vocab, key=len, reverse=True)
        while cursor < len(text):
            match = None
            for token in ordered:
                if token and text.startswith(token, cursor):
                    match = token
                    break
            if match is None:
                cursor += 1
                continue
            out.append((self._vocab[match], cursor, cursor + len(match)))
            cursor += len(match)
        return out


class TokenizerBundle:
    tokenizer: DualTrackTokenizer
    processor: MultimodalProcessor
    config: TokenizerBundleConfig

    def __init__(
        self,
        tokenizer: DualTrackTokenizer,
        processor: MultimodalProcessor,
        config: TokenizerBundleConfig,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

    @classmethod
    def from_files(
        cls,
        morphbpe_path: str,
        zh_source: str = "Qwen/Qwen2.5-0.5B",
        en_source: str = "meta-llama/Llama-3.2-1B",
        patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
        use_smoke_hf: bool = False,
    ) -> "TokenizerBundle":
        config = TokenizerBundleConfig(
            version=BUNDLE_VERSION,
            morphbpe_file=morphbpe_path,
            zh_source=zh_source,
            en_source=en_source,
            patch_size=patch_size,
            merge_size=merge_size,
            temporal_patch_size=temporal_patch_size,
            use_smoke_hf=use_smoke_hf,
        )
        return cls._build(config, morphbpe_path=morphbpe_path, vocab=None)

    @classmethod
    def from_dir(cls, path: str) -> "TokenizerBundle":
        config_path = os.path.join(path, CONFIG_NAME)
        vocab_path = os.path.join(path, VOCAB_NAME)
        with open(config_path, "r", encoding="utf-8") as f:
            config = TokenizerBundleConfig(**json.load(f))
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = {str(token): int(idx) for token, idx in json.load(f).items()}
        morphbpe_path = os.path.join(path, config.morphbpe_file)
        return cls._build(config, morphbpe_path=morphbpe_path, vocab=vocab)

    @classmethod
    def _build(
        cls,
        config: TokenizerBundleConfig,
        morphbpe_path: str,
        vocab: dict[str, int] | None,
    ) -> "TokenizerBundle":
        stemmer = MongolStemmer()
        morphbpe = MorphBPETokenizer.from_file(morphbpe_path, stemmer)
        if vocab is None:
            zh_hf, en_hf, zh_tokens, en_tokens = _load_hf_tracks(config)
            vocab = build_unified_vocab(
                morphbpe.vocab, zh_tokens, en_tokens, build_misc_tokens()
            )
        else:
            zh_hf, en_hf = _load_hf_from_vocab(config, vocab)
        tokenizer = DualTrackTokenizer(vocab, morphbpe, zh_hf, en_hf)
        processor = MultimodalProcessor(
            tokenizer,
            patch_size=config.patch_size,
            merge_size=config.merge_size,
            temporal_patch_size=config.temporal_patch_size,
        )
        return cls(tokenizer=tokenizer, processor=processor, config=config)

    def save_dir(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        dest_morphbpe = os.path.join(path, MORPHBPE_NAME)
        source_morphbpe = self.config.morphbpe_file
        if os.path.abspath(source_morphbpe) != os.path.abspath(dest_morphbpe):
            if os.path.exists(source_morphbpe):
                shutil.copyfile(source_morphbpe, dest_morphbpe)
            else:
                self.tokenizer.morphbpe.save(dest_morphbpe)
        config = TokenizerBundleConfig(
            version=self.config.version,
            morphbpe_file=MORPHBPE_NAME,
            zh_source=self.config.zh_source,
            en_source=self.config.en_source,
            patch_size=self.config.patch_size,
            merge_size=self.config.merge_size,
            temporal_patch_size=self.config.temporal_patch_size,
            use_smoke_hf=self.config.use_smoke_hf,
        )
        with open(os.path.join(path, CONFIG_NAME), "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, ensure_ascii=False, indent=2)
        with open(os.path.join(path, VOCAB_NAME), "w", encoding="utf-8") as f:
            json.dump(self.tokenizer.vocab, f, ensure_ascii=False, indent=2)

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        return self.tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)

    def encode_with_spans(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ):
        return self.tokenizer.encode_with_spans(text, add_bos=add_bos, add_eos=add_eos)

    def encode_multimodal(
        self,
        text: str,
        images=None,
        image_sizes=None,
        videos=None,
        video_sizes=None,
        add_bos: bool = False,
        add_eos: bool = False,
    ):
        return self.processor(
            text,
            images=images,
            image_sizes=image_sizes,
            videos=videos,
            video_sizes=video_sizes,
            add_bos=add_bos,
            add_eos=add_eos,
        )

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.config.version != BUNDLE_VERSION:
            issues.append(f"unsupported config version: {self.config.version}")
        if not self.config.morphbpe_file:
            issues.append("config.morphbpe_file is empty")
        values = list(self.tokenizer.vocab.values())
        if len(values) != len(set(values)):
            issues.append("vocab contains duplicate ids")
        for token, expected_id in SPECIAL_TOKENS.items():
            actual = self.tokenizer.vocab.get(token)
            if actual != expected_id:
                issues.append(f"special token {token!r} has id {actual}, expected {expected_id}")
        if "<unk>" not in self.tokenizer.morphbpe.vocab:
            issues.append("morphbpe vocab is missing <unk>")
        try:
            result = self.encode_with_spans("ᠮᠣᠩᠭᠣᠯ 文字 test 🙂", add_bos=True, add_eos=True)
            if len(result.input_ids) != len(result.tokens):
                issues.append("encode_with_spans produced mismatched ids/tokens")
        except Exception as exc:  # pragma: no cover - reported as validation issue.
            issues.append(f"encode smoke failed: {exc}")
        try:
            mm = self.encode_multimodal(
                "文字 <image> test",
                images=["smoke-image"],
                image_sizes=[(14, 14)],
            )
            if len(mm.input_ids) != len(mm.attention_mask):
                issues.append("multimodal smoke produced mismatched ids/mask")
            if not mm.image_token_spans:
                issues.append("multimodal smoke produced no image span")
        except Exception as exc:  # pragma: no cover - reported as validation issue.
            issues.append(f"multimodal smoke failed: {exc}")
        return issues


def _load_hf_tracks(
    config: TokenizerBundleConfig,
) -> tuple[Any, Any, list[str], list[str]]:
    if config.use_smoke_hf:
        zh_tokens = ["文", "字", "这", "张", "图", "中"]
        en_tokens = ["test", "hello", "abc", "def", "text"]
        return (
            _SmokeHFTokenizer({tok: idx for idx, tok in enumerate(zh_tokens)}),
            _SmokeHFTokenizer({tok: idx for idx, tok in enumerate(en_tokens)}),
            zh_tokens,
            en_tokens,
        )
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required when use_smoke_hf=False; "
            "install transformers or build with --smoke-hf"
        ) from exc

    from .dual_tokenizer import extract_hf_vocab_tokens

    zh_tokens, _ = extract_hf_vocab_tokens(config.zh_source, "zh", 15000)
    en_tokens, _ = extract_hf_vocab_tokens(config.en_source, "en", 8000)
    return (
        AutoTokenizer.from_pretrained(config.zh_source),
        AutoTokenizer.from_pretrained(config.en_source),
        zh_tokens,
        en_tokens,
    )


def _load_hf_from_vocab(
    config: TokenizerBundleConfig, vocab: dict[str, int]
) -> tuple[Any, Any]:
    if config.use_smoke_hf:
        return (
            _SmokeHFTokenizer(_track_vocab_from_unified(vocab, "zh")),
            _SmokeHFTokenizer(_track_vocab_from_unified(vocab, "en")),
        )
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required when use_smoke_hf=False; "
            "install transformers or use a smoke-HF bundle"
        ) from exc
    return (
        AutoTokenizer.from_pretrained(config.zh_source),
        AutoTokenizer.from_pretrained(config.en_source),
    )


def _track_vocab_from_unified(vocab: dict[str, int], lang: str) -> dict[str, int]:
    prefix = f"{lang}▁"
    ordered = sorted(
        ((token[len(prefix):], idx) for token, idx in vocab.items() if token.startswith(prefix)),
        key=lambda item: item[1],
    )
    return {token: local_id for local_id, (token, _global_id) in enumerate(ordered)}
