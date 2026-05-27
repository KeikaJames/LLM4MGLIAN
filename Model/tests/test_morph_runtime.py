# -*- coding: utf-8 -*-

import unittest

import torch

from Model.config import RDTConfig
from Model.model import RDTForCausalLM
from Tokenizer.pretraining import derive_morph_info_from_boundary_ids


def _cfg() -> RDTConfig:
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


class DefaultMorphInfoTest(unittest.TestCase):
    def test_matches_python_reference(self):
        cfg = _cfg()
        model = RDTForCausalLM(cfg)
        wb = cfg.word_boundary_id
        mb = cfg.morpheme_boundary_id
        bos = cfg.bos_id
        eos = cfg.eos_id

        seqs = [
            [bos, wb, 300, mb, 301, wb, 302, eos],
            [bos, 300, wb, 301, mb, 302, mb, eos],
        ]
        input_ids = torch.tensor(seqs, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        word_pos, morph_depth = model._default_morph_info(input_ids, attention_mask)

        for i, seq in enumerate(seqs):
            ref_wp, ref_md = derive_morph_info_from_boundary_ids(
                seq, wb, mb, max_depth=cfg.max_morph_depth
            )
            self.assertEqual(word_pos[i].tolist(), ref_wp, msg=f"row {i}")
            self.assertEqual(morph_depth[i].tolist(), ref_md, msg=f"row {i}")

    def test_no_boundaries_yields_word_zero(self):
        cfg = _cfg()
        model = RDTForCausalLM(cfg)
        input_ids = torch.tensor([[300, 301, 302, 303]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        word_pos, morph_depth = model._default_morph_info(input_ids, attention_mask)
        self.assertEqual(word_pos.tolist(), [[0, 0, 0, 0]])
        self.assertEqual(morph_depth.tolist(), [[0, 0, 0, 0]])

    def test_morph_depth_clamps_to_max(self):
        cfg = _cfg()
        cfg.max_morph_depth = 3
        model = RDTForCausalLM(cfg)
        mb = cfg.morpheme_boundary_id
        wb = cfg.word_boundary_id
        ids = [[wb, mb, mb, mb, mb, mb, 300]]
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        _, morph_depth = model._default_morph_info(input_ids, attention_mask)
        self.assertLessEqual(int(morph_depth.max().item()), cfg.max_morph_depth - 1)


if __name__ == "__main__":
    unittest.main()
