# -*- coding: utf-8 -*-

"""VLM alignment: OMVT vision tower → projector → RDT.

Runs a synthetic end-to-end forward/backward where the RDT LM consumes
``<image_patch>`` slots filled by the OMVT compressed tokens.  The LM head
is fine-tuned by default; pass ``--freeze-rdt`` to train only the
projector/tower.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from Model.config import (
    BOS_ID,
    EOS_ID,
    IMAGE_PATCH_ID,
    OMVTConfig,
    TrainingConfig,
    tiny_config,
)
from Model.model import RDTForCausalLM
from Model.omvt import OMVTInjector
from Model.omvt.patcher import collate_omvt_batch
from Model.training import RankZeroLogger, build_optimizer, build_scheduler


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--image-size", type=int, default=56)
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--n-image-tokens", type=int, default=8)
    p.add_argument("--freeze-rdt", action="store_true")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output", default="outputs/vlm_align")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _build_omvt_cfg(args) -> OMVTConfig:
    return OMVTConfig(
        image_size=args.image_size,
        d_vision=64,
        vertical_patch=(args.image_size // 2, args.image_size // 4),
        horizontal_patch=(args.image_size // 4, args.image_size // 2),
        square_patch=(args.image_size // 4, args.image_size // 4),
        layout_patch=(args.image_size, args.image_size),
        compress_to=args.n_image_tokens,
    )


def _make_text_batch(args, vocab_floor=300, vocab_ceil=320):
    B, L, N = args.batch_size, args.seq_len, args.n_image_tokens
    rng = torch.Generator().manual_seed(args.seed)
    # layout: [BOS] <image_patch>*N <random text...> [EOS]
    text_len = L - N - 2
    if text_len <= 0:
        raise ValueError("seq_len must be greater than 2 + n_image_tokens")
    text_ids = torch.randint(vocab_floor, vocab_ceil, (B, text_len), generator=rng)
    input_ids = torch.full((B, L), 0, dtype=torch.long)
    input_ids[:, 0] = BOS_ID
    input_ids[:, 1 : 1 + N] = IMAGE_PATCH_ID
    input_ids[:, 1 + N : 1 + N + text_len] = text_ids
    input_ids[:, -1] = EOS_ID
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return input_ids, attention_mask, labels


def main(argv=None):
    args = parse_args(argv)
    # Fast-fail validation **before** any device alloc / model construction.
    # Mirrors the train_rdt CLI pattern: misconfigured runs should not pay
    # the cost of building the model only to crash inside the first step.
    if args.seq_len <= args.n_image_tokens + 2:
        print(
            "scripts/train_vlm_align: --seq-len must be > --n-image-tokens + 2 "
            f"(got seq_len={args.seq_len}, n_image_tokens={args.n_image_tokens})",
            file=sys.stderr,
        )
        return 2
    if args.image_size <= 0 or args.image_size % 4 != 0:
        print(
            "scripts/train_vlm_align: --image-size must be a positive multiple of 4",
            file=sys.stderr,
        )
        return 2

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rdt_cfg = tiny_config()
    # cap seq_len to the synthetic layout (tiny config is 2048 by default but
    # the smoke layout is much shorter)
    from dataclasses import replace
    rdt_cfg = replace(rdt_cfg, max_seq_len=args.seq_len)

    omvt_cfg = _build_omvt_cfg(args)

    model = RDTForCausalLM(rdt_cfg).to(device)
    # plug in matching-size OMVT injector (otherwise dispatcher would build
    # a default-sized one on first forward and fail on tiny synthetic inputs).
    model.vision._omvt_cfg = omvt_cfg
    model.vision.omvt = OMVTInjector(rdt_cfg, omvt_cfg).to(device)

    if args.freeze_rdt:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.vision.omvt.parameters():
            p.requires_grad_(True)

    train_cfg = TrainingConfig(
        train_data="",
        seq_len=args.seq_len,
        micro_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.05,
        max_steps=args.steps,
        warmup_steps=1,
        precision="fp32",
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    logger = RankZeroLogger(args.output, enable_tensorboard=False)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        input_ids, attention_mask, labels = _make_text_batch(args)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        images = torch.randn(
            args.batch_size,
            omvt_cfg.in_channels,
            omvt_cfg.image_size,
            omvt_cfg.image_size,
            device=device,
        )
        batch = collate_omvt_batch(images, omvt_cfg)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=dict(batch),
        )
        loss = out["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        logger.log(step, {"loss": float(loss.detach())})

    logger.close()
    print(f"VLM align smoke OK in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
