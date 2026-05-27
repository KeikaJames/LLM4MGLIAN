# -*- coding: utf-8 -*-

"""Checkpoint save / load (DDP and FSDP aware)."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from Model.training.dist import is_distributed, is_main_process


@dataclass
class CheckpointPayload:
    step: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    scheduler_state: dict[str, Any] | None
    rng_state: dict[str, Any]
    metadata: dict[str, Any]


def _unwrap(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        return model.module
    return model


def _is_fsdp(model: nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return isinstance(model, FSDP)
    except ImportError:  # pragma: no cover
        return False


def _get_model_state(model: nn.Module) -> dict[str, Any]:
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            return model.state_dict()
    return _unwrap(model).state_dict()


def _load_model_state(model: nn.Module, state: dict[str, Any]) -> None:
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            model.load_state_dict(state)
    else:
        _unwrap(model).load_state_dict(state)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    if "cpu" in state:
        torch.set_rng_state(state["cpu"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    metadata: dict[str, Any] | None = None,
    keep_last_n: int = 0,
) -> Path | None:
    """Save a step checkpoint under ``{output_dir}/step_{step:08d}/``."""

    out = Path(output_dir)
    step_dir = out / f"step_{step:08d}"

    model_state = _get_model_state(model)
    if not is_main_process():
        return None

    step_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, step_dir / "model.pt")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), step_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), step_dir / "scheduler.pt")
    torch.save(_rng_state(), step_dir / "rng.pt")
    torch.save(
        {"step": step, "metadata": metadata or {}},
        step_dir / "meta.pt",
    )

    if keep_last_n > 0:
        ckpts = sorted(out.glob("step_*"))
        for old in ckpts[:-keep_last_n]:
            shutil.rmtree(old, ignore_errors=True)

    latest = out / "latest"
    if latest.exists() or latest.is_symlink():
        try:
            latest.unlink()
        except OSError:
            shutil.rmtree(latest, ignore_errors=True)
    try:
        os.symlink(step_dir.name, latest)
    except OSError:
        # symlinks may not be available on some filesystems
        pass
    return step_dir


def load_checkpoint(path: str | Path) -> CheckpointPayload:
    p = Path(path)
    if p.name == "latest" or not p.exists():
        candidate = p.parent / "latest" if p.parent.exists() else None
        if candidate and candidate.exists():
            p = candidate.resolve()

    model_state = torch.load(p / "model.pt", map_location="cpu", weights_only=False)
    opt_path = p / "optimizer.pt"
    sched_path = p / "scheduler.pt"
    rng_path = p / "rng.pt"
    meta_path = p / "meta.pt"

    opt_state = (
        torch.load(opt_path, map_location="cpu", weights_only=False)
        if opt_path.exists()
        else None
    )
    sched_state = (
        torch.load(sched_path, map_location="cpu", weights_only=False)
        if sched_path.exists()
        else None
    )
    rng_state = (
        torch.load(rng_path, map_location="cpu", weights_only=False)
        if rng_path.exists()
        else {}
    )
    meta = (
        torch.load(meta_path, map_location="cpu", weights_only=False)
        if meta_path.exists()
        else {"step": 0, "metadata": {}}
    )

    return CheckpointPayload(
        step=int(meta.get("step", 0)),
        model_state=model_state,
        optimizer_state=opt_state,
        scheduler_state=sched_state,
        rng_state=rng_state,
        metadata=meta.get("metadata", {}),
    )


def resume_state(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> int:
    """Load a checkpoint into the given model/optimizer/scheduler."""

    payload = load_checkpoint(path)
    _load_model_state(model, payload.model_state)
    if optimizer is not None and payload.optimizer_state is not None:
        optimizer.load_state_dict(payload.optimizer_state)
    if scheduler is not None and payload.scheduler_state is not None:
        scheduler.load_state_dict(payload.scheduler_state)
    if payload.rng_state:
        _restore_rng(payload.rng_state)
    if is_distributed():
        torch.distributed.barrier()
    return payload.step


__all__ = [
    "CheckpointPayload",
    "load_checkpoint",
    "resume_state",
    "save_checkpoint",
]
