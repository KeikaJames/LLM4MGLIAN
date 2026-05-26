# -*- coding: utf-8 -*-
"""Simple sample packing for minimum pretraining data builds."""

from __future__ import annotations

from .builder import EncodedSample


def pack_samples(
    samples: list[EncodedSample], max_length: int, pad_id: int, eos_id: int
) -> list[EncodedSample]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    packed: list[EncodedSample] = []
    current = _empty_text_pack()
    count = 0

    def flush() -> None:
        nonlocal current, count
        if current.input_ids:
            current.metadata = {"type": "packed_text", "num_samples": count}
            packed.append(current)
        current = _empty_text_pack()
        count = 0

    for sample in samples:
        sample = _trim(sample, max_length)
        if _has_modality(sample):
            flush()
            packed.append(sample)
            continue

        extra_ids = list(sample.input_ids)
        extra_mask = list(sample.attention_mask)
        extra_labels = list(sample.labels)
        extra_offsets = list(sample.token_offsets)
        if current.input_ids:
            extra_ids = [eos_id] + extra_ids
            extra_mask = [1] + extra_mask
            extra_labels = [eos_id] + extra_labels
            extra_offsets = [(-1, -1)] + extra_offsets

        if len(current.input_ids) + len(extra_ids) > max_length:
            flush()
            extra_ids = list(sample.input_ids)
            extra_mask = list(sample.attention_mask)
            extra_labels = list(sample.labels)
            extra_offsets = list(sample.token_offsets)

        current.input_ids.extend(extra_ids[:max_length])
        current.attention_mask.extend(extra_mask[:max_length])
        current.labels.extend(extra_labels[:max_length])
        current.token_offsets.extend(extra_offsets[:max_length])
        count += 1

    flush()
    return packed


def _empty_text_pack() -> EncodedSample:
    return EncodedSample(
        input_ids=[],
        attention_mask=[],
        labels=[],
        token_offsets=[],
        modality_spans={"image_token_spans": [], "video_token_spans": []},
        metadata={"type": "packed_text", "num_samples": 0},
    )


def _has_modality(sample: EncodedSample) -> bool:
    return bool(
        sample.modality_spans.get("image_token_spans")
        or sample.modality_spans.get("video_token_spans")
    )


def _trim(sample: EncodedSample, max_length: int) -> EncodedSample:
    if len(sample.input_ids) <= max_length:
        return sample
    return EncodedSample(
        input_ids=sample.input_ids[:max_length],
        attention_mask=sample.attention_mask[:max_length],
        labels=sample.labels[:max_length],
        token_offsets=sample.token_offsets[:max_length],
        modality_spans=sample.modality_spans,
        metadata={**sample.metadata, "truncated": True},
    )
