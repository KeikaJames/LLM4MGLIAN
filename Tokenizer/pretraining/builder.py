# -*- coding: utf-8 -*-
"""Minimum pretraining sample encoder built on a TokenizerBundle."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from Tokenizer.unified.bundle import TokenizerBundle
from Tokenizer.unified.encoded import EncodedToken

IGNORE_INDEX = -100
DEFAULT_LABEL_IGNORE_TOKENS = {
    "<pad>",
    "<unk>",
    "<bos>",
    "<img>",
    "<image>",
    "<image_start>",
    "<image_patch>",
    "<image_end>",
    "<video>",
    "<video_start>",
    "<video_patch>",
    "<video_end>",
    "<audio>",
    "<audio_start>",
    "<audio_patch>",
    "<audio_end>",
    "<bbox>",
    "<ocr>",
    "<ocr_start>",
    "<ocr_end>",
    "<doc>",
    "<table>",
    "<layout>",
    "◈",
}


@dataclass
class EncodedSample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    token_offsets: list[tuple[int, int]]
    modality_spans: dict
    metadata: dict


class PretrainingDataBuilder:
    def __init__(
        self,
        bundle: TokenizerBundle,
        max_length: int = 4096,
        add_bos: bool = True,
        add_eos: bool = True,
        label_ignore_id: int = IGNORE_INDEX,
        label_ignore_tokens: set[str] | None = None,
    ):
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.bundle = bundle
        self.max_length = max_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.label_ignore_id = label_ignore_id
        self.label_ignore_tokens = set(label_ignore_tokens or DEFAULT_LABEL_IGNORE_TOKENS)

    def encode_text(self, text: str, metadata: dict | None = None) -> EncodedSample:
        result = self.bundle.encode_with_spans(
            text, add_bos=self.add_bos, add_eos=self.add_eos
        )
        sample = self._sample_from_tokens(
            result.input_ids,
            result.attention_mask,
            result.tokens,
            {"image_token_spans": [], "video_token_spans": []},
            metadata or {"type": "text"},
        )
        return self._truncate(sample)

    def encode_json_obj(self, obj: dict) -> EncodedSample:
        text = str(obj.get("text", ""))
        sample_type = str(obj.get("type", "text"))
        has_media = bool(obj.get("images") or obj.get("videos"))
        if sample_type in {"image_text", "ocr", "video_text"} or has_media:
            result = self.bundle.encode_multimodal(
                text,
                images=obj.get("images"),
                image_sizes=obj.get("image_sizes"),
                videos=obj.get("videos"),
                video_sizes=obj.get("video_sizes"),
                add_bos=self.add_bos,
                add_eos=self.add_eos,
            )
            sample = self._sample_from_tokens(
                result.input_ids,
                result.attention_mask,
                result.tokens,
                {
                    "image_token_spans": result.image_token_spans,
                    "video_token_spans": result.video_token_spans,
                },
                self._metadata(obj),
            )
            return self._truncate(sample)
        return self.encode_text(text, metadata=self._metadata(obj))

    def encode_jsonl_line(self, line: str) -> EncodedSample:
        return self.encode_json_obj(json.loads(line))

    def _sample_from_tokens(
        self,
        input_ids: list[int],
        attention_mask: list[int],
        tokens: list[EncodedToken],
        modality_spans: dict,
        metadata: dict,
    ) -> EncodedSample:
        return EncodedSample(
            input_ids=list(input_ids),
            attention_mask=list(attention_mask),
            labels=self._build_labels(input_ids, tokens),
            token_offsets=[(token.start, token.end) for token in tokens],
            modality_spans={
                "image_token_spans": [tuple(span) for span in modality_spans.get("image_token_spans", [])],
                "video_token_spans": [tuple(span) for span in modality_spans.get("video_token_spans", [])],
            },
            metadata=dict(metadata),
        )

    def _build_labels(
        self, input_ids: list[int], tokens: list[EncodedToken]
    ) -> list[int]:
        labels = list(input_ids)
        for i, token in enumerate(tokens):
            if token.token in self.label_ignore_tokens:
                labels[i] = self.label_ignore_id
        return labels

    def _truncate(self, sample: EncodedSample) -> EncodedSample:
        if len(sample.input_ids) <= self.max_length:
            return sample
        cutoff = self.max_length
        for key in ("image_token_spans", "video_token_spans"):
            for start, end in sample.modality_spans.get(key, []):
                if start < cutoff < end:
                    cutoff = start
        modality_spans = {
            key: [span for span in spans if span[1] <= cutoff]
            for key, spans in sample.modality_spans.items()
        }
        metadata = dict(sample.metadata)
        metadata["truncated"] = True
        return EncodedSample(
            input_ids=sample.input_ids[:cutoff],
            attention_mask=sample.attention_mask[:cutoff],
            labels=sample.labels[:cutoff],
            token_offsets=sample.token_offsets[:cutoff],
            modality_spans=modality_spans,
            metadata=metadata,
        )

    def _metadata(self, obj: dict) -> dict:
        keep = {
            "type",
            "ocr",
            "images",
            "image_sizes",
            "videos",
            "video_sizes",
            "source",
            "id",
        }
        meta = {key: obj[key] for key in keep if key in obj}
        if "type" not in meta:
            meta["type"] = "text"
        return meta


def encoded_sample_to_dict(sample: EncodedSample) -> dict[str, Any]:
    return asdict(sample)
