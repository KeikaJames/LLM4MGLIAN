# -*- coding: utf-8 -*-

"""JSONL datasets, sequence packing, and collators for pretraining."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from Model.config import IGNORE_INDEX, PAD_ID, TrainingConfig
from Tokenizer.pretraining import derive_morph_info_from_offsets


class JsonlPretrainingDataset(Dataset):
    """Eager JSONL dataset for rows emitted by ``build_pretraining_data``."""

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


class StreamingJsonlDataset(IterableDataset):
    """Sharded, rank-aware streaming JSONL reader.

    Files in ``paths`` are statically assigned to ranks (round-robin); within
    a rank, dataloader workers further partition the assigned files. A small
    in-memory shuffle buffer can be enabled.
    """

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        world_size: int = 1,
        rank: int = 0,
        shuffle_buffer: int = 0,
        seed: int = 0,
        infinite: bool = True,
    ) -> None:
        if not paths:
            raise ValueError("StreamingJsonlDataset requires at least one path")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if not (0 <= rank < world_size):
            raise ValueError("rank must be in [0, world_size)")
        self.paths = [Path(p) for p in paths]
        self.world_size = world_size
        self.rank = rank
        self.shuffle_buffer = max(0, int(shuffle_buffer))
        self.seed = int(seed)
        self.infinite = bool(infinite)

    def _files_for_worker(self) -> list[Path]:
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        rank_shards = self.paths[self.rank :: self.world_size]
        if not rank_shards:
            return []
        return rank_shards[worker_id::num_workers]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        files = self._files_for_worker()
        if not files:
            return

        rng = random.Random(self.seed + 1000 * self.rank + os.getpid())
        buffer: list[dict[str, Any]] = []

        def _emit(item: dict[str, Any]) -> Iterator[dict[str, Any]]:
            if self.shuffle_buffer <= 0:
                yield item
                return
            buffer.append(item)
            if len(buffer) >= self.shuffle_buffer:
                idx = rng.randrange(len(buffer))
                buffer[idx], buffer[-1] = buffer[-1], buffer[idx]
                yield buffer.pop()

        first_pass = True
        while first_pass or self.infinite:
            first_pass = False
            for path in files:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        yield from _emit(row)

        rng.shuffle(buffer)
        for item in buffer:
            yield item


@dataclass
class PretrainingCollator:
    pad_id: int = PAD_ID
    ignore_index: int = IGNORE_INDEX
    pad_to_multiple_of: int | None = None
    include_metadata: bool = False
    max_seq_len: int | None = None

    def __call__(self, rows: Sequence[Any]) -> dict[str, Any]:
        if not rows:
            raise ValueError("rows cannot be empty")

        normalized = [_normalize_row(row, max_seq_len=self.max_seq_len) for row in rows]
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


def _normalize_row(row: Any, *, max_seq_len: int | None) -> dict[str, Any]:
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

    if max_seq_len is not None and n > max_seq_len:
        for key in ("input_ids", "attention_mask", "labels", "word_pos", "morph_depth"):
            out[key] = out[key][:max_seq_len]
    return out


def _resolve_shards(spec: str | Sequence[str | Path]) -> list[Path]:
    if isinstance(spec, (str, Path)):
        path = Path(spec)
        if path.is_dir():
            return sorted(path.glob("*.jsonl"))
        if any(ch in str(spec) for ch in "*?["):
            return sorted(Path().glob(str(spec)))
        return [path]
    return [Path(p) for p in spec]


def build_dataloader(
    spec: str | Sequence[str | Path],
    cfg: TrainingConfig,
    *,
    world_size: int = 1,
    rank: int = 0,
    seed: int | None = None,
    pad_id: int = PAD_ID,
    ignore_index: int = IGNORE_INDEX,
    infinite: bool = True,
) -> DataLoader:
    """Construct a rank-aware streaming dataloader."""

    paths = _resolve_shards(spec)
    if not paths:
        raise FileNotFoundError(f"no shards resolved from spec: {spec!r}")

    dataset = StreamingJsonlDataset(
        paths,
        world_size=world_size,
        rank=rank,
        shuffle_buffer=cfg.shuffle_buffer,
        seed=seed if seed is not None else cfg.seed,
        infinite=infinite,
    )

    collator = PretrainingCollator(
        pad_id=pad_id,
        ignore_index=ignore_index,
        pad_to_multiple_of=None,
        max_seq_len=cfg.seq_len,
    )

    return DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collator,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )


def estimate_tokens_per_step(cfg: TrainingConfig, world_size: int = 1) -> int:
    return cfg.micro_batch_size * cfg.grad_accum_steps * cfg.seq_len * world_size


__all__ = [
    "JsonlPretrainingDataset",
    "PretrainingCollator",
    "StreamingJsonlDataset",
    "build_dataloader",
    "estimate_tokens_per_step",
]
