# -*- coding: utf-8 -*-

"""Smoke test: one-step training + checkpoint save/load/resume."""

import tempfile
import unittest
from pathlib import Path

import torch

from Model.config import RDTConfig, TrainingConfig
from Model.model import RDTForCausalLM
from Model.training import (
    PretrainingCollator,
    TrainState,
    build_optimizer,
    build_scheduler,
    resume_state,
    save_checkpoint,
    train_one_step,
)


def _tiny_cfg() -> RDTConfig:
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


def _make_batch(model_cfg: RDTConfig, length: int = 8) -> dict:
    seq = [model_cfg.bos_id, 300, 301, 302, 303, 304, 305, model_cfg.eos_id][:length]
    row = {
        "input_ids": seq,
        "attention_mask": [1] * length,
        "labels": seq,
    }
    collator = PretrainingCollator()
    return collator([row, row])


class TrainingLoopSmokeTest(unittest.TestCase):
    def test_train_step_uses_chunked_loss_path(self):
        torch.manual_seed(0)
        cfg = _tiny_cfg()
        cfg.loss_chunk_size = 2

        class RecordingModel(RDTForCausalLM):
            def __init__(self, model_cfg):
                super().__init__(model_cfg)
                self.seen_return_logits: list[bool | None] = []

            def forward(self, *args, **kwargs):
                self.seen_return_logits.append(kwargs.get("return_logits"))
                return super().forward(*args, **kwargs)

        train_cfg = TrainingConfig(
            train_data="",
            seq_len=8,
            micro_batch_size=2,
            grad_accum_steps=1,
            num_workers=0,
            learning_rate=1e-3,
            max_steps=1,
            warmup_steps=1,
            precision="fp32",
            use_loss_chunking=True,
        )
        model = RecordingModel(cfg)
        optim = build_optimizer(model, train_cfg)
        sched = build_scheduler(optim, train_cfg)
        state = TrainState()
        batch = _make_batch(cfg)

        def _iter():
            while True:
                yield batch

        train_one_step(
            model,
            _iter(),
            optim,
            sched,
            train_cfg,
            state,
            device=torch.device("cpu"),
        )

        self.assertEqual(model.seen_return_logits, [False])

    def test_train_step_and_resume(self):
        torch.manual_seed(0)
        cfg = _tiny_cfg()
        train_cfg = TrainingConfig(
            train_data="",
            seq_len=8,
            micro_batch_size=2,
            grad_accum_steps=1,
            num_workers=0,
            learning_rate=1e-3,
            max_steps=4,
            warmup_steps=1,
            precision="fp32",
            grad_clip=1.0,
        )

        model = RDTForCausalLM(cfg)
        optim = build_optimizer(model, train_cfg)
        sched = build_scheduler(optim, train_cfg)
        state = TrainState()

        device = torch.device("cpu")
        batch = _make_batch(cfg)

        def _iter():
            while True:
                yield batch

        it = _iter()
        m1 = train_one_step(model, it, optim, sched, train_cfg, state, device=device)
        self.assertTrue(m1["loss"] == m1["loss"], "loss must not be NaN")
        self.assertEqual(state.step, 1)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ckpt_dir = save_checkpoint(out, state.step, model, optim, sched)
            self.assertIsNotNone(ckpt_dir)
            self.assertTrue((ckpt_dir / "model.pt").exists())

            # Take another step on the original model
            m2 = train_one_step(
                model, it, optim, sched, train_cfg, state, device=device
            )
            after_extra_loss = m2["loss"]

            # Fresh model + restored from checkpoint should match step==1 state
            model2 = RDTForCausalLM(cfg)
            optim2 = build_optimizer(model2, train_cfg)
            sched2 = build_scheduler(optim2, train_cfg)
            step = resume_state(ckpt_dir, model2, optim2, sched2)
            self.assertEqual(step, 1)

            state2 = TrainState(step=step)
            m3 = train_one_step(
                model2, it, optim2, sched2, train_cfg, state2, device=device
            )
            self.assertAlmostEqual(after_extra_loss, m3["loss"], places=3)


if __name__ == "__main__":
    unittest.main()
