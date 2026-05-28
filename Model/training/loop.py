# -*- coding: utf-8 -*-

"""Single-step train / eval primitives."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from Model.config import TrainingConfig
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
            )
        loss = out["loss"] / cfg.grad_accum_steps
        loss.backward()

        # Keep the per-microstep loss as a detached tensor; we only sync
        # to host once per optimizer step to avoid stalling the training
        # loop on the device queue.
        loss_terms.append(loss.detach())

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
    total_tokens = 0
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
            )
        tok = int(batch["attention_mask"].sum().item())
        total_loss += float(out["loss"].item()) * tok
        total_tokens += tok
        seen += 1
    avg = total_loss / max(1, total_tokens)
    return {"eval_loss": avg, "eval_tokens": float(total_tokens)}


__all__ = ["TrainState", "evaluate", "train_one_step"]
