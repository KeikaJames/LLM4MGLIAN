# -*- coding: utf-8 -*-

import unittest

import torch

from Model.config import RDTConfig
from Model.layers.mamba3_layer import Mamba3Layer
from Model.model import RDTForCausalLM


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

    def test_config_rejects_invalid_dropout(self):
        with self.assertRaisesRegex(ValueError, "dropout"):
            RDTConfig(dropout=1.0)


if __name__ == "__main__":
    unittest.main()
