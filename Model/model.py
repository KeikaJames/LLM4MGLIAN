# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.blocks import StandardBlock
from Model.config import RDTConfig
from Model.layers.rmsnorm import RMSNorm
from Model.recurrent import RecurrentCore
from Model.vision import VisionInjector


class RDTForCausalLM(nn.Module):
    def __init__(self, cfg: RDTConfig, patch_pixels: int = 14 * 14 * 3):
        super().__init__()

        self.cfg = cfg
        self.patch_pixels = patch_pixels

        self.embed = nn.Embedding(
            cfg.vocab_size,
            cfg.d_model,
            padding_idx=cfg.pad_id,
        )

        self.vision = VisionInjector(cfg, patch_pixels)

        self.prelude = nn.ModuleList(
            StandardBlock(cfg, layer_idx=i) for i in range(cfg.n_prelude)
        )

        self.recurrent = RecurrentCore(cfg)

        self.coda = nn.ModuleList(
            StandardBlock(cfg, layer_idx=cfg.n_prelude + i) for i in range(cfg.n_coda)
        )

        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.bidirectional = cfg.bidirectional

        if self.bidirectional:
            self.reverse_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        else:
            self.reverse_head = None

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight
            if self.reverse_head is not None:
                self.reverse_head.weight = self.embed.weight

        self.apply(self._init_weights)

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight
            if self.reverse_head is not None:
                self.reverse_head.weight = self.embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        word_pos: torch.Tensor | None = None,
        morph_depth: torch.Tensor | None = None,
        steps: int | None = None,
    ) -> dict[str, torch.Tensor | dict]:
        self._check_inputs(input_ids, attention_mask, labels)

        bsz, seq_len = input_ids.shape

        if attention_mask is None:
            attention_mask = (input_ids != self.cfg.pad_id).long()

        if word_pos is None or morph_depth is None:
            word_pos, morph_depth = self._default_morph_info(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        h = self.embed(input_ids)

        if pixel_values is not None:
            h = self.vision(h, input_ids, pixel_values)

        for block in self.prelude:
            h = block(
                h,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attention_mask,
                causal=True,
            )

        e0 = h

        h, rec_info = self.recurrent(
            e0,
            word_pos=word_pos,
            morph_depth=morph_depth,
            attn_mask=attention_mask,
            causal=True,
            steps=steps,
        )

        for block in self.coda:
            h = block(
                h,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attention_mask,
                causal=True,
            )

        h = self.final_norm(h)
        logits = self.lm_head(h)

        loss = None
        loss_parts: dict[str, float] = {}

        if labels is not None:
            loss, loss_parts = self._losses(h, logits, labels, rec_info)

        return {
            "loss": loss,
            "logits": logits,
            "loss_parts": loss_parts,
            "rec_info": rec_info,
        }

    def _losses(
        self,
        h: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        rec_info: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        forward = self._causal_loss(logits, labels)
        loss = forward
        parts = {"forward": float(forward.detach())}

        if self.reverse_head is not None:
            rev_logits = self.reverse_head(h)
            reverse = self._reverse_loss(rev_logits, labels)
            loss = loss + self.cfg.reverse_loss_weight * reverse
            parts["reverse"] = float(reverse.detach())

        ponder = rec_info.get("ponder_cost")
        if self.cfg.use_act and isinstance(ponder, torch.Tensor):
            loss = loss + self.cfg.act_ponder_cost * ponder
            parts["ponder"] = float(ponder.detach())

        return loss, parts

    def _causal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self._token_loss(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
        )

    def _reverse_loss(
        self,
        rev_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self._token_loss(
            rev_logits[:, 1:].reshape(-1, rev_logits.size(-1)),
            labels[:, :-1].reshape(-1),
        )

    def _token_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = targets != self.cfg.ignore_index
        if logits.numel() == 0 or not bool(valid.any()):
            return logits.sum() * 0.0

        return F.cross_entropy(
            logits,
            targets,
            ignore_index=self.cfg.ignore_index,
        )

    def _default_morph_info(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        word_pos = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
        )
        morph_depth = torch.zeros(bsz, seq_len, dtype=torch.long, device=device)

        word_pos = word_pos * attention_mask.to(device=device, dtype=torch.long)
        return word_pos, morph_depth

    def _check_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
    ) -> None:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, L]")

        if input_ids.numel() == 0:
            raise ValueError("input_ids cannot be empty")

        if input_ids.min().item() < 0:
            raise ValueError("input_ids contain negative ids")

        if input_ids.max().item() >= self.cfg.vocab_size:
            raise ValueError("input_ids contain ids outside vocab_size")

        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have shape [B, L]")

        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must have shape [B, L]")

        if input_ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    @torch.no_grad()
    def count_params(self, trainable_only: bool = False) -> int:
        seen: set[int] = set()
        total = 0

        for param in self.parameters():
            if trainable_only and not param.requires_grad:
                continue

            ptr = param.data_ptr()
            if ptr in seen:
                continue

            seen.add(ptr)
            total += param.numel()

        return total


def _check() -> None:
    from Model.config import tiny_config

    torch.manual_seed(0)

    cfg = tiny_config()
    model = RDTForCausalLM(cfg)

    print("RDTForCausalLM")
    print(f"  params: {model.count_params():,}")
    print(f"  actual_layers: {cfg.actual_layers}")
    print(f"  effective_depth: {cfg.effective_depth}")

    bsz, seq_len = 2, 32
    input_ids = torch.randint(256, 24576, (bsz, seq_len))
    input_ids[:, 0] = cfg.bos_id

    attention_mask = torch.ones(bsz, seq_len, dtype=torch.long)
    labels = input_ids.clone()

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    print("Text")
    print(f"  logits: {tuple(out['logits'].shape)}")
    print(f"  loss: {out['loss'].item():.6f}")
    print(f"  loss_parts: {out['loss_parts']}")
    print(f"  steps_used: {out['rec_info'].get('steps_used')}")

    out["loss"].backward()
    grad_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_sq += p.grad.norm().item() ** 2

    print(f"  grad_norm: {grad_sq**0.5:.6f}")

    ids2 = torch.tensor(
        [
            [
                cfg.bos_id,
                300,
                cfg.image_start_id,
                cfg.image_patch_id,
                cfg.image_patch_id,
                cfg.image_end_id,
                301,
                cfg.eos_id,
            ]
        ]
    )

    labels2 = ids2.clone()
    labels2[ids2 == cfg.image_start_id] = cfg.ignore_index
    labels2[ids2 == cfg.image_patch_id] = cfg.ignore_index
    labels2[ids2 == cfg.image_end_id] = cfg.ignore_index

    pixel_values = torch.randn(2, model.patch_pixels)

    out2 = model(
        input_ids=ids2,
        labels=labels2,
        pixel_values=pixel_values,
    )

    print("ImageText")
    print(f"  loss: {out2['loss'].item():.6f}")

    model.eval()
    with torch.no_grad():
        for steps in [2, 8]:
            out_step = model(input_ids=input_ids, steps=steps)
            print(
                f"  steps={steps}, logits_norm={out_step['logits'].norm().item():.6f}"
            )


if __name__ == "__main__":
    _check()
