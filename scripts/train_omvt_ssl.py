# -*- coding: utf-8 -*-

"""OMVT vision-tower self-supervised pretraining entry point.

Runs the four SSL heads (OCR reconstruction, masked patch, orientation,
layout order) on synthetic batches.  Use this to smoke-test the tower /
heads / loss plumbing before plugging in a real OCR corpus.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from Model.config import OMVTConfig
from Model.omvt import (
    LayoutOrderHead,
    MaskedPatchHead,
    OCRReconstructionHead,
    OMVTVisionTower,
    OrientationHead,
    layout_order_loss,
    masked_patch_loss,
    ocr_reconstruction_loss,
    orientation_loss,
)
from Model.training import RankZeroLogger, build_optimizer
from Model.config import TrainingConfig


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--image-size", type=int, default=56)
    p.add_argument("--d-vision", type=int, default=64)
    p.add_argument("--compress-to", type=int, default=8)
    p.add_argument("--ocr-vocab", type=int, default=64)
    p.add_argument("--output", default="outputs/omvt_ssl")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _omvt_cfg(args) -> OMVTConfig:
    cfg = OMVTConfig(
        image_size=args.image_size,
        d_vision=args.d_vision,
        vertical_patch=(args.image_size // 2, args.image_size // 4),
        horizontal_patch=(args.image_size // 4, args.image_size // 2),
        square_patch=(args.image_size // 4, args.image_size // 4),
        layout_patch=(args.image_size, args.image_size),
        compress_to=args.compress_to,
    )
    return cfg


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = _omvt_cfg(args)
    tower = OMVTVisionTower(cfg).to(device)
    ocr_head = OCRReconstructionHead(cfg.d_vision, args.ocr_vocab).to(device)
    mp_head = MaskedPatchHead(cfg.d_vision, patch_pixels=cfg.square_patch[0] * cfg.square_patch[1] * cfg.in_channels).to(device)
    ori_head = OrientationHead(cfg.d_vision).to(device)
    layout_head = LayoutOrderHead(cfg.d_vision, max_positions=cfg.compress_to).to(device)

    modules = torch.nn.ModuleList([tower, ocr_head, mp_head, ori_head, layout_head])
    train_cfg = TrainingConfig(
        train_data="",
        seq_len=cfg.compress_to,
        micro_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.05,
        max_steps=args.steps,
        warmup_steps=1,
        precision="fp32",
    )
    optimizer = build_optimizer(modules, train_cfg)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    logger = RankZeroLogger(args.output, enable_tensorboard=False)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        images = torch.randn(args.batch_size, cfg.in_channels, cfg.image_size, cfg.image_size, device=device)
        feats = tower(images)["compressed"]  # [B, compress_to, d_vision]

        ocr_logits = ocr_head(feats)
        ocr_labels = torch.randint(0, args.ocr_vocab, (args.batch_size, cfg.compress_to), device=device)
        loss_ocr = ocr_reconstruction_loss(ocr_logits, ocr_labels)

        masked = mp_head(feats)
        true_patches = torch.randn_like(masked)
        loss_mp = masked_patch_loss(masked, true_patches)

        ori_logits = ori_head(feats)
        ori_labels = torch.randint(0, 4, (args.batch_size,), device=device)
        loss_ori = orientation_loss(ori_logits, ori_labels)

        layout_logits = layout_head(feats)
        layout_targets = torch.arange(cfg.compress_to, device=device).unsqueeze(0).expand(args.batch_size, -1)
        loss_layout = layout_order_loss(layout_logits, layout_targets)

        loss = (
            cfg.w_ocr * loss_ocr
            + cfg.w_masked_patch * loss_mp
            + cfg.w_orientation * loss_ori
            + cfg.w_layout_order * loss_layout
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modules.parameters(), 1.0)
        optimizer.step()

        logger.log(step, {
            "loss": float(loss.detach()),
            "ocr": float(loss_ocr.detach()),
            "mp": float(loss_mp.detach()),
            "ori": float(loss_ori.detach()),
            "layout": float(loss_layout.detach()),
        })

    logger.close()
    dt = time.time() - t0
    print(f"OMVT SSL smoke OK in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
