# RDT Model — Pretraining Guide

Recurrent Depth Transformer with Mamba3 + MoE Latent Attention (MLA), an
ACT (PonderNet) controller, optional bidirectional auxiliary head, and a
two-tier vision pathway (MLP fallback + OMVT — Orientation-aware
Multiscript Vision Tower).

## 1. Module layout

```
Model/
  config.py            # RDTConfig, TrainingConfig, OMVTConfig, tiny/small/base/pretrain
  model.py             # RDTForCausalLM (LM head, vision injection, ACT)
  recurrent.py         # RecurrentCore (fixed + ACT loops, activation checkpointing)
  blocks.py            # StandardBlock, RecurrentBlock, AttnSubLayer, MambaSubLayer
  layers/
    mamba3_layer.py    # Official mamba_ssm.Mamba3 with NaiveSSM fallback
    attention_mla.py   # MLA + MorphologicalRoPE
    ...
  vision.py            # MLPVisionEncoder + dispatcher VisionInjector
  omvt/
    patcher.py         # MultiScalePatcher: vertical/horizontal/square/layout
    router.py          # GeometricRouter (sobel-based, LM-free)
    mixers.py          # VerticalSSM, HorizontalSSM, LocalAttention, LayoutMixer
    compressor.py      # PerceiverCompressor → fixed-N visual tokens
    tower.py           # OMVTVisionTower (full pipeline)
    injector.py        # OMVTInjector: tower + projector + <image_patch> replacement
    heads.py / losses.py  # OCR / masked-patch / orientation / layout-order SSL
  training/
    data.py            # JSONL + StreamingJsonlDataset + collator + dataloader
    optim.py           # AdamW (no-decay groups) + warmup/cosine
    dist.py            # init_distributed + wrap_ddp + wrap_fsdp
    checkpoint.py      # FSDP-aware save / resume
    loop.py            # train_one_step + evaluate (autocast + grad accum + clip)
    logging.py         # RankZeroLogger (+ optional tensorboard)
```

## 2. Configs

```python
from Model.config import tiny_config, small_config, base_config, pretrain_config
cfg = pretrain_config()  # ~1.1B params: d_model=2048, 16 heads, 8 recurrent steps
```

| Config     | d_model | heads | layers (prelude / coda) | recurrent steps | seq_len |
|------------|---------|-------|-------------------------|-----------------|---------|
| tiny       | 192     | 3     | 1 / 1                   | 4               | 2048    |
| small      | 768     | 12    | 2 / 2                   | 8               | 4096    |
| base       | 1536    | 16    | 3 / 3                   | 16              | 8192    |
| pretrain   | 2048    | 16    | 3 / 3                   | 8               | 4096    |

Activation-memory controls (in `RDTConfig`):

- `use_activation_checkpointing`, `grad_ckpt_recurrent`, `grad_ckpt_blocks`,
  `grad_ckpt_prelude_coda`.
- `bptt_window > 0` truncates BPTT (older recurrent steps are detached).
- `use_act` switches to PonderNet-style adaptive halting with
  `act_max_steps` upper bound; the loop runs the full bound without
  host-syncs so CUDA streams stay pipelined.

## 3. Training entry points

All three scripts are CLI-driven and accept `--smoke` for a synthetic
in-memory smoke run.

```bash
# Text RDT pretraining (single process)
python -m scripts.train_rdt --config pretrain \
    --data path/to/shards/*.jsonl --output runs/rdt

# DDP / FSDP
torchrun --nproc_per_node=8 scripts/train_rdt.py --config pretrain \
    --dist fsdp --precision bf16 --data path/to/shards/*.jsonl \
    --output runs/rdt

# OMVT vision-tower SSL (Phase 1 — OCR / masked-patch / orientation / layout-order)
python -m scripts.train_omvt_ssl --output runs/omvt_ssl

# OMVT → RDT alignment (Phase 3 — vision tokens injected at <image_patch>)
python -m scripts.train_vlm_align --output runs/vlm_align [--freeze-rdt]
```

Resume: `--resume runs/rdt/latest` (auto-detects FSDP / DDP / single).

## 4. Activation-memory strategy (P0)

The recurrent core dominates activation memory:
`recurrent_steps × block_layers × L × d_model`. We control it with three
levers, applied in order from cheapest to most aggressive:

1. **BPTT window** (`cfg.bptt_window=k`): steps before the last `k` run
   under `torch.no_grad`-style detach; activations released immediately.
2. **Recurrent checkpoint** (`grad_ckpt_recurrent=True`): wrap each step's
   block call in `torch.utils.checkpoint(use_reentrant=False)`; trades
   one extra forward for a `~recurrent_steps×` activation-memory cut.
3. **Block / prelude+coda checkpoint** (`grad_ckpt_blocks`,
   `grad_ckpt_prelude_coda`): per-block checkpointing for the static
   prelude/coda stack as well.

Equivalence with non-checkpointed training is covered by
`Model/tests/test_grad_ckpt.py` (loss and embed-grad atol=1e-5).

FSDP uses `transformer_auto_wrap_policy` over
`{StandardBlock, RecurrentBlock, AttnSubLayer, MambaSubLayer}` so the
recurrent steps do **not** share a single shard — that would erase the
sharding benefit.

## 5. Vision pathway

`VisionInjector` is a dispatcher:

- `pixel_values: Tensor`  → `MLPVisionEncoder` (smoke / fallback only).
- `pixel_values: Mapping` → `OMVTInjector` (production path).

### OMVT — Orientation-aware Multiscript Vision Tower

Three-phase roadmap:

1. **Phase 1 (SSL pretraining)** — `scripts/train_omvt_ssl.py`.
   Four heads: OCR reconstruction, masked-patch reconstruction,
   orientation (4-way), layout-order (permutation prediction).
2. **Phase 2 (text-only RDT pretraining)** — `scripts/train_rdt.py`.
   OMVT frozen / unused.
3. **Phase 3 (joint VLM alignment)** — `scripts/train_vlm_align.py`.
   `OMVTVisionTower → PerceiverCompressor → projector → <image_patch>`.
   RDT can be frozen with `--freeze-rdt`.

`GeometricRouter` is intentionally **LM-free**: it derives stream weights
from Sobel-edge statistics (vertical/horizontal/square/layout), so the
vision tower stays self-contained during Phase 1 SSL.

## 6. Testing & smoke

```bash
./scripts/test_all.sh       # Tokenizer + Model unittests + Rust normalizer
./scripts/smoke_all.sh      # 5 acceptance smoke runs (single + DDP + VLM + SSL)
```

Acceptance items covered by `smoke_all.sh`:

1. Text RDT pretraining (single process).
2. Text RDT pretraining (DDP × 2, gloo backend, runs on CPU).
3. VLM alignment with OMVT vision tower injected into RDT.
4. OMVT vision-tower SSL with all four heads + losses.
5. OMVT → RDT end-to-end (covered by #3 above).

## 7. Mamba official fail-fast

`Mamba3Layer._build_official` no longer silently swallows constructor
mismatches: a missing parameter raises with an "upgrade `mamba-ssm`"
hint. Production configs (`small_config`, `base_config`,
`pretrain_config`) all set `use_official_mamba=True`. The CPU-only smoke
configs and tests use the `NaiveSSM` fallback explicitly.
