# -*- coding: utf-8 -*-

"""Single-step train / eval primitives."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from Model.config import IGNORE_INDEX, TrainingConfig
from Model.training.optim import recurrent_steps_for_step


@dataclass
class TrainState:
    step: int = 0
    tokens_seen: int = 0
    last_loss: float = float("nan")
    extra: dict[str, Any] = field(default_factory=dict)


def _autocast_ctx(precision: str, device_type: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device_type, dtype=dtype)


def _new_cuda_grad_scaler():
    """Build a CUDA ``GradScaler`` using the non-deprecated API when available."""

    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def _grad_scaler_for(model: nn.Module, cfg: TrainingConfig, device_type: str):
    """Return a fp16 ``GradScaler`` (sharded under FSDP) or ``None``.

    fp16 autocast without loss scaling silently underflows small gradients to
    zero and can diverge to NaN. bf16/fp32 have enough exponent range and need
    no scaler. CPU has no fp16 GradScaler support, so we leave it unscaled.
    """

    if cfg.precision != "fp16" or device_type != "cuda":
        return None
    # FSDP shards gradients across ranks; a plain GradScaler.unscale_ would see
    # only the local shard, so use the sharded variant when the model is FSDP.
    if hasattr(model, "clip_grad_norm_"):
        try:
            from torch.distributed.fsdp.sharded_grad_scaler import (
                ShardedGradScaler,
            )

            return ShardedGradScaler()
        except Exception:  # pragma: no cover - older torch without ShardedGradScaler
            return _new_cuda_grad_scaler()
    return _new_cuda_grad_scaler()


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            # Pixel batches arrive as ``{"images": ..., "vertical_patches": ...}``;
            # we have to recurse so every leaf tensor lands on the model's
            # device. Non-tensor leaves (rare) pass through untouched.
            out[k] = {
                kk: vv.to(device, non_blocking=True) if isinstance(vv, torch.Tensor) else vv
                for kk, vv in v.items()
            }
        else:
            out[k] = v
    return out


def train_one_step(
    model: nn.Module,
    batch_iter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    cfg: TrainingConfig,
    state: TrainState,
    *,
    device: torch.device,
    target_recurrent_steps: int | None = None,
) -> dict[str, float]:
    """Run one optimizer step (covers ``grad_accum_steps`` micro batches)."""

    model.train()
    optimizer.zero_grad(set_to_none=True)

    device_type = device.type if device.type in ("cuda", "cpu") else "cpu"
    loss_terms: list[torch.Tensor] = []
    token_count = 0

    # fp16 needs loss scaling; persist the scaler across steps via state.extra
    # so its dynamic scale factor is preserved (and can be checkpointed).
    scaler = state.extra.get("grad_scaler")
    if scaler is None and cfg.precision == "fp16":
        scaler = _grad_scaler_for(model, cfg, device_type)
        if scaler is not None:
            state.extra["grad_scaler"] = scaler

    rec_steps = None
    if target_recurrent_steps is not None:
        rec_steps = recurrent_steps_for_step(state.step, cfg, target_recurrent_steps)

    for _ in range(cfg.grad_accum_steps):
        batch = next(batch_iter)
        # Count tokens on the CPU mask **before** moving to device so the
        # ``.sum()`` does not force a host-sync against the GPU stream.
        cpu_mask = batch.get("attention_mask")
        if isinstance(cpu_mask, torch.Tensor):
            token_count += int(cpu_mask.sum().item())
        batch = _to_device(batch, device)

        with _autocast_ctx(cfg.precision, device_type):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
                word_pos=batch.get("word_pos"),
                morph_depth=batch.get("morph_depth"),
                pixel_values=batch.get("pixel_values"),
                steps=rec_steps,
                bptt_window=cfg.bptt_window,
                return_logits=not cfg.use_loss_chunking,
            )
        loss = out["loss"] / cfg.grad_accum_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Keep the per-microstep loss as a detached tensor; we only sync
        # to host once per optimizer step to avoid stalling the training
        # loop on the device queue.
        loss_terms.append(loss.detach())

    if scaler is not None:
        # Unscale before clipping so grad_clip is applied to true gradients.
        scaler.unscale_(optimizer)

    if cfg.grad_clip and cfg.grad_clip > 0:
        if hasattr(model, "clip_grad_norm_"):
            grad_norm = model.clip_grad_norm_(cfg.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip
            )
        grad_norm_val = float(grad_norm)
    else:
        grad_norm_val = float("nan")

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    scheduler.step()

    if loss_terms:
        # One host-sync per optimizer step instead of per micro-step.
        loss_sum = torch.stack(loss_terms).sum().item() * cfg.grad_accum_steps
    else:
        loss_sum = 0.0

    state.step += 1
    state.tokens_seen += token_count
    state.last_loss = loss_sum / max(1, cfg.grad_accum_steps)
    return {
        "loss": state.last_loss,
        "grad_norm": grad_norm_val,
        "lr": float(scheduler.get_last_lr()[0]),
        "tokens": float(token_count),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batches,
    cfg: TrainingConfig,
    *,
    device: torch.device,
    max_batches: int = 32,
) -> dict[str, float]:
    model.eval()
    device_type = device.type if device.type in ("cuda", "cpu") else "cpu"
    total_loss = 0.0
    total_targets = 0
    seen = 0
    for batch in batches:
        if seen >= max_batches:
            break
        batch = _to_device(batch, device)
        with _autocast_ctx(cfg.precision, device_type):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
                word_pos=batch.get("word_pos"),
                morph_depth=batch.get("morph_depth"),
                pixel_values=batch.get("pixel_values"),
                return_logits=not cfg.use_loss_chunking,
            )
        # The model's loss is a mean over valid (non-ignore-index) next-token
        # targets — labels are shifted internally (logits[:, :-1] vs
        # labels[:, 1:]), and image/pad positions are set to ignore_index.
        # Weight by that same count so the cross-batch average is exact rather
        # than biased by padded/masked positions (attention_mask overcounts).
        labels = batch["labels"]
        n_targets = int((labels[:, 1:] != IGNORE_INDEX).sum().item())
        if n_targets == 0:
            continue
        total_loss += float(out["loss"].item()) * n_targets
        total_targets += n_targets
        seen += 1
    avg = total_loss / max(1, total_targets)
    return {"eval_loss": avg, "eval_tokens": float(total_targets)}


__all__ = ["TrainState", "evaluate", "train_one_step"]
