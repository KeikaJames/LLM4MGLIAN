# -*- coding: utf-8 -*-

import unittest

import torch
import torch.nn as nn

from Model.config import RDTConfig
import Model.layers.mamba3_layer as mamba3_layer
from Model.layers.mamba3_layer import Mamba3Layer
from Model.model import RDTForCausalLM
from Model.vision import inject_visual_features


def test_config() -> RDTConfig:
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


class ModelSmokeTest(unittest.TestCase):
    def test_tied_heads_share_embedding_after_init(self):
        cfg = test_config()
        model = RDTForCausalLM(cfg)

        self.assertEqual(model.lm_head.weight.data_ptr(), model.embed.weight.data_ptr())
        self.assertIsNotNone(model.reverse_head)
        self.assertEqual(
            model.reverse_head.weight.data_ptr(),
            model.embed.weight.data_ptr(),
        )

    def test_forward_backward_with_image_patch(self):
        torch.manual_seed(0)
        cfg = test_config()
        model = RDTForCausalLM(cfg, patch_pixels=4)

        input_ids = torch.tensor(
            [
                [
                    cfg.bos_id,
                    256,
                    cfg.image_start_id,
                    cfg.image_patch_id,
                    cfg.image_end_id,
                    cfg.eos_id,
                ]
            ]
        )
        labels = input_ids.clone()
        labels[input_ids == cfg.image_start_id] = cfg.ignore_index
        labels[input_ids == cfg.image_patch_id] = cfg.ignore_index
        labels[input_ids == cfg.image_end_id] = cfg.ignore_index

        out = model(
            input_ids=input_ids,
            labels=labels,
            pixel_values=torch.randn(1, 4),
        )

        self.assertEqual(out["logits"].shape, (1, 6, cfg.vocab_size))
        self.assertTrue(torch.isfinite(out["loss"]))

        out["loss"].backward()
        self.assertIsNotNone(model.embed.weight.grad)

    def test_all_ignored_labels_return_zero_loss(self):
        torch.manual_seed(0)
        cfg = test_config()
        model = RDTForCausalLM(cfg)

        input_ids = torch.tensor([[cfg.bos_id]])
        labels = torch.full_like(input_ids, cfg.ignore_index)

        out = model(input_ids=input_ids, labels=labels)

        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertEqual(float(out["loss"].detach()), 0.0)

        out["loss"].backward()
        self.assertIsNotNone(model.embed.weight.grad)

    def test_mamba_mask_skips_left_padding_state(self):
        torch.manual_seed(0)
        cfg = test_config()
        layer = Mamba3Layer(cfg)
        layer.eval()

        tokens = torch.randn(1, 2, cfg.d_model)
        padded = torch.cat([torch.zeros(1, 2, cfg.d_model), tokens], dim=1)
        mask = torch.tensor([[0, 0, 1, 1]])

        with torch.no_grad():
            padded_out = layer(padded, attn_mask=mask)
            short_out = layer(tokens)

        self.assertTrue(
            torch.allclose(padded_out[:, 2:], short_out, atol=1e-5, rtol=1e-5)
        )
        self.assertTrue(
            torch.equal(padded_out[:, :2], torch.zeros_like(padded_out[:, :2]))
        )

    def test_fallback_mamba_honors_configured_state_size(self):
        cfg = test_config()
        cfg.mamba_d_state = 24

        layer = Mamba3Layer(cfg)

        self.assertEqual(layer.backend, "fallback")
        self.assertEqual(layer.mamba.d_state, 24)

    def test_official_mamba_rejects_masked_state_updates(self):
        class FakeOfficialMamba3(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()

            def forward(self, x):
                return x

        old = mamba3_layer.OfficialMamba3
        mamba3_layer.OfficialMamba3 = FakeOfficialMamba3
        try:
            cfg = test_config()
            cfg.use_official_mamba = True
            layer = Mamba3Layer(cfg)

            x = torch.randn(1, 3, cfg.d_model)
            layer(x, attn_mask=torch.ones(1, 3, dtype=torch.long))

            with self.assertRaisesRegex(ValueError, "official Mamba backend"):
                layer(x, attn_mask=torch.tensor([[0, 1, 1]]))
        finally:
            mamba3_layer.OfficialMamba3 = old

    def test_unbatched_visual_features_require_single_batch(self):
        cfg = test_config()
        inputs = torch.zeros(2, 3, cfg.d_model)
        input_ids = torch.tensor(
            [
                [cfg.bos_id, cfg.image_patch_id, cfg.eos_id],
                [cfg.bos_id, cfg.image_patch_id, cfg.eos_id],
            ]
        )
        features = torch.randn(2, cfg.d_model)

        with self.assertRaisesRegex(ValueError, "batch size is 1"):
            inject_visual_features(inputs, input_ids, features, cfg.image_patch_id)

    def test_config_rejects_invalid_dropout(self):
        with self.assertRaisesRegex(ValueError, "dropout"):
            RDTConfig(dropout=1.0)


if __name__ == "__main__":
    unittest.main()
