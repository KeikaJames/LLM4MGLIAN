# -*- coding: utf-8 -*-
"""Utilities for feeding tokenizer pretraining JSONL into the model."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from Model.config import IGNORE_INDEX, PAD_ID
from Tokenizer.pretraining import derive_morph_info_from_offsets


class JsonlPretrainingDataset(Dataset):
    """Small JSONL dataset for rows emitted by build_pretraining_data."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


@dataclass
class PretrainingCollator:
    pad_id: int = PAD_ID
    ignore_index: int = IGNORE_INDEX
    pad_to_multiple_of: int | None = None
    include_metadata: bool = False

    def __call__(self, rows: Sequence[Any]) -> dict[str, Any]:
        if not rows:
            raise ValueError("rows cannot be empty")

        normalized = [_normalize_row(row) for row in rows]
        max_len = max(len(row["input_ids"]) for row in normalized)
        if max_len <= 0:
            raise ValueError("rows cannot contain empty input_ids")
        if self.pad_to_multiple_of is not None:
            if self.pad_to_multiple_of <= 0:
                raise ValueError("pad_to_multiple_of must be positive")
            remainder = max_len % self.pad_to_multiple_of
            if remainder:
                max_len += self.pad_to_multiple_of - remainder

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "word_pos": [],
            "morph_depth": [],
        }

        for row in normalized:
            pad_count = max_len - len(row["input_ids"])
            batch["input_ids"].append(row["input_ids"] + [self.pad_id] * pad_count)
            batch["attention_mask"].append(row["attention_mask"] + [0] * pad_count)
            batch["labels"].append(row["labels"] + [self.ignore_index] * pad_count)
            batch["word_pos"].append(row["word_pos"] + [0] * pad_count)
            batch["morph_depth"].append(row["morph_depth"] + [0] * pad_count)

        tensors: dict[str, Any] = {
            key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()
        }
        if self.include_metadata:
            tensors["metadata"] = [row.get("metadata", {}) for row in normalized]
            tensors["modality_spans"] = [
                row.get("modality_spans", {}) for row in normalized
            ]
        return tensors


def _normalize_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        row = row.__dict__

    required = ("input_ids", "attention_mask", "labels")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"missing pretraining fields: {', '.join(missing)}")

    out: dict[str, Any] = {
        "input_ids": [int(value) for value in row["input_ids"]],
        "attention_mask": [int(value) for value in row["attention_mask"]],
        "labels": [int(value) for value in row["labels"]],
        "metadata": dict(row.get("metadata") or {}),
        "modality_spans": dict(row.get("modality_spans") or {}),
    }
    n = len(out["input_ids"])
    if len(out["attention_mask"]) != n or len(out["labels"]) != n:
        raise ValueError("input_ids, attention_mask, and labels must align")

    word_pos = row.get("word_pos")
    morph_depth = row.get("morph_depth")
    if word_pos is None or morph_depth is None:
        token_offsets = row.get("token_offsets")
        if token_offsets is None:
            word_pos = list(range(n))
            morph_depth = [0] * n
        else:
            word_pos, morph_depth = derive_morph_info_from_offsets(token_offsets)

    out["word_pos"] = [int(value) for value in word_pos]
    out["morph_depth"] = [int(value) for value in morph_depth]
    if len(out["word_pos"]) != n or len(out["morph_depth"]) != n:
        raise ValueError("word_pos and morph_depth must align with input_ids")
    return out
