# -*- coding: utf-8 -*-

"""Smoke tests for OMVT tower, SSL heads, and RDT injection."""

import unittest

import torch

from Model.config import OMVTConfig, RDTConfig
from Model.model import RDTForCausalLM
from Model.omvt import (
    LayoutOrderHead,
    MaskedPatchHead,
    OCRReconstructionHead,
    OMVTInjector,
    OMVTVisionTower,
    OrientationHead,
    collate_omvt_batch,
    layout_order_loss,
    masked_patch_loss,
    ocr_reconstruction_loss,
    orientation_loss,
)


def _tiny_omvt(d_vision: int = 64, compress_to: int = 8) -> OMVTConfig:
    return OMVTConfig(
        image_size=56,
        vertical_patch=(28, 14),
        horizontal_patch=(14, 28),
        square_patch=(14, 14),
        layout_patch=(28, 28),
        d_vision=d_vision,
        vision_n_heads=4,
        vision_ffn_hidden=128,
        compress_to=compress_to,
        compressor_layers=1,
        compressor_heads=4,
        n_vertical_layers=1,
        n_horizontal_layers=1,
        n_local_attn_layers=1,
        n_layout_layers=1,
    )


def _tiny_rdt() -> RDTConfig:
    return RDTConfig(
        d_model=32,
        n_heads=4,
        head_dim=8,
        kv_lora_rank=8,
        rope_head_dim=4,
        nope_head_dim=4,
        ffn_hidden=64,
        ffn_multiple=32,
        n_prelude=1,
        n_coda=1,
        mamba_per_block=1,
        attn_per_block=1,
        recurrent_steps=2,
        mamba_d_state=8,
        mamba_expand=2,
        mamba_headdim=16,
        use_official_mamba=False,
        max_seq_len=64,
    )


class OMVTTowerSmokeTest(unittest.TestCase):
    def test_forward_backward(self):
        torch.manual_seed(0)
        cfg = _tiny_omvt()
        tower = OMVTVisionTower(cfg)
        imgs = torch.randn(2, 3, 56, 56)
        out = tower(imgs)
        self.assertEqual(out["compressed"].shape, (2, cfg.compress_to, cfg.d_vision))
        self.assertEqual(out["router_weights"].shape, (2, 4))
        self.assertTrue(torch.allclose(out["router_weights"].sum(-1), torch.ones(2), atol=1e-5))
        loss = out["compressed"].pow(2).mean()
        loss.backward()
        # at least one mixer should have non-zero grad
        grads = [p.grad for p in tower.parameters() if p.grad is not None]
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))


class OMVTSSLHeadsTest(unittest.TestCase):
    def test_all_heads_and_losses(self):
        torch.manual_seed(0)
        cfg = _tiny_omvt()
        tower = OMVTVisionTower(cfg)
        imgs = torch.randn(2, 3, 56, 56)
        out = tower(imgs)
        tokens = out["compressed"]

        ocr = OCRReconstructionHead(cfg.d_vision, vocab_size=128)
        patch = MaskedPatchHead(cfg.d_vision, patch_pixels=14 * 14 * 3)
        ori = OrientationHead(cfg.d_vision)
        layout = LayoutOrderHead(cfg.d_vision, max_positions=cfg.compress_to)

        ocr_logits = ocr(tokens)
        patch_logits = patch(tokens)
        ori_logits = ori(tokens)
        layout_logits = layout(tokens)

        ocr_target = torch.randint(0, 128, (2, cfg.compress_to))
        patch_target = torch.randn(2, cfg.compress_to, 14 * 14 * 3)
        patch_mask = torch.ones(2, cfg.compress_to, dtype=torch.bool)
        ori_target = torch.randint(0, 4, (2,))
        layout_target = torch.arange(cfg.compress_to).unsqueeze(0).expand(2, -1)

        loss_ocr = ocr_reconstruction_loss(ocr_logits, ocr_target)
        loss_patch = masked_patch_loss(patch_logits, patch_target, patch_mask)
        loss_ori = orientation_loss(ori_logits, ori_target)
        loss_layout = layout_order_loss(layout_logits, layout_target)

        total = loss_ocr + loss_patch + loss_ori + loss_layout
        self.assertTrue(torch.isfinite(total))
        total.backward()


class OMVTIntoRDTSmokeTest(unittest.TestCase):
    def test_omvt_compressed_injected_into_rdt(self):
        torch.manual_seed(0)
        omvt_cfg = _tiny_omvt(compress_to=4)
        rdt_cfg = _tiny_rdt()

        model = RDTForCausalLM(rdt_cfg)

        # Build OMVT injector and project into d_model.
        injector = OMVTInjector(rdt_cfg, omvt_cfg)
        imgs = torch.randn(1, 3, 56, 56)
        batch = collate_omvt_batch(imgs, omvt_cfg)
        visual_tokens = injector(batch)
        self.assertEqual(visual_tokens.shape, (1, omvt_cfg.compress_to, rdt_cfg.d_model))

        # Inject through VisionInjector dispatcher.
        ip = rdt_cfg.image_patch_id
        n_patches = omvt_cfg.compress_to
        seq = [rdt_cfg.bos_id, 300] + [ip] * n_patches + [301, rdt_cfg.eos_id]
        input_ids = torch.tensor([seq])
        labels = input_ids.clone()
        attn = torch.ones_like(input_ids)

        # Bypass model's MLP path: replace embed manually via VisionInjector dict path.
        # We feed dict pixel_values that triggers OMVT, then run forward.
        # Override the model's VisionInjector to use the matching OMVT config
        # (the dispatcher would otherwise lazily build a default-sized tower).
        model.vision._omvt_cfg = omvt_cfg
        model.vision.omvt = OMVTInjector(rdt_cfg, omvt_cfg)

        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            labels=labels,
            pixel_values=dict(batch),
        )
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()


if __name__ == "__main__":
    unittest.main()
