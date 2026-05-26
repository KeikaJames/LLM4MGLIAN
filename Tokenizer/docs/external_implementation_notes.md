# External Implementation Notes

Design principles distilled from open-source tokenizers and multimodal
processors. We do not vendor code or weights; we record contracts and
patterns that shape this repo's API.

## HuggingFace tokenizers (`tokenizers.Encoding`)

- Source: https://huggingface.co/docs/tokenizers/api/encoding
- Encoding carries `ids`, `tokens`, `offsets`, `attention_mask`,
  `special_tokens_mask`, `type_ids`, `word_ids`, `sequence_ids`. We mirror
  this contract: `DualTrackResult` exposes `input_ids`, `tokens`,
  `attention_mask`, `special_tokens_mask` plus our routed `track`.
- Special tokens carry `offsets = (0, 0)` to signal "synthetic, not from
  text". We use `(-1, -1)` instead for BOS/EOS to make detection
  unambiguous against zero-length real spans; placeholder tokens (image
  patches) reuse the source `<image>` span and surface `image_index` in
  `metadata`.
- `char_to_token` / `token_to_chars` is feasible only when every token has
  a real character span. We document our offset contract (see
  `offset_contract.md`) so downstream layers can do the same lookups.

## tiktoken

- Source: https://github.com/openai/tiktoken
- BPE state is split into `mergeable_ranks` + a separate `special_tokens`
  dict; specials are matched out-of-band (regex split before BPE). We do
  the same: `vocab.SPECIAL_TOKENS` is reserved in segment 0 and routed
  before language detection.
- Rank-based BPE: merges are stored as `(piece -> rank)` and the lowest
  rank wins at each pair. Our `MorphBPETrainer` records merges in order
  and reconstructs ranks on load — the merge order *is* the rank.
- Byte-level fallback is guaranteed reversible & lossless. We adopt the
  same invariant in `generic_bpe.byte_fallback`: any input byte sequence
  encodes to vocab tokens and decodes back to the exact bytes; emoji
  round-trip is tested.

## SentencePiece

- Source: https://github.com/google/sentencepiece
- Trains directly from raw sentences; whitespace is a regular symbol
  (`▁`). We respect this: `▁` is reserved id 17, decoders treat it as a
  word boundary, and `_strip_hf_boundary_markers` normalizes `▁`/`Ġ`.
- NFKC normalization is part of the tokenizer, not a pre-step the caller
  must remember. Our Rust `normalize_to_nominal_unicode` plays the same
  role for Mongolian and is wired in via the CLI bridge so training
  pipelines cannot skip it.
- Vocabulary size is predetermined and the trainer optimizes inside that
  budget. Our `MorphBPETrainer.train(vocab_size=...)` stops at the target
  size and skips merges below `min_pair_freq` for stability.

## LLaVA processor

- Source: https://huggingface.co/docs/transformers/main/en/model_doc/llava
- The processor expands a single `<image>` placeholder into N image
  tokens based on `patch_size`, `image_seq_length`, and
  `num_additional_image_tokens` (CLS). We mirror this with
  `expand_image_placeholders_by_sizes` driven by `image_patch_count`.
- The processor validates that the text contains the right number of
  image placeholders for the supplied images. We raise `ValueError` when
  `image_sizes` length disagrees with `<image>` count.
- Image embeddings are merged into the LM after tokenization using the
  image-token positions. We therefore must return `image_token_spans` as
  `input_ids`-index ranges (not char ranges) so the model layer can locate
  the slot.

## Qwen2-VL processor

- Source: https://huggingface.co/docs/transformers/main/en/model_doc/qwen2_vl
- Vision tokens are bracketed by explicit start/end markers
  (`<|vision_start|>` / `<|vision_end|>`); the count between them depends
  on `image_grid_thw`. We adopt the same triple
  (`<image_start>`/`<image_patch>*N`/`<image_end>`) and document
  `image_token_spans = (start_index, end_index)` as the *inclusive* token
  range covering all three.
- Video is handled with the same scheme plus a temporal patch factor.
  Our `video_patch_count(num_frames, w, h, patch_size, temporal_patch_size,
  merge_size)` matches the formula
  `ceil(num_frames/temporal_patch_size) * (w/patch_size/merge_size) *
  (h/patch_size/merge_size)`.
- Dynamic resolution: the number of vision tokens is data-dependent.
  Processors must compute counts per-sample, not at config time. Our
  `MultimodalProcessor.__call__` accepts per-call `image_sizes` /
  `video_sizes` rather than baking patch counts into the tokenizer.

## vLLM multimodal processing

- Source: https://docs.vllm.ai/en/latest/features/multimodal_inputs.html
- vLLM enforces that the number of image placeholders in the prompt
  equals the number of supplied images (and exposes
  `limit_mm_per_prompt`). We validate the same invariant in the processor
  and surface a clear `ValueError`.
- The multi-modal data dict is keyed by modality (`"image"`, `"video"`)
  and accepts lists for batched inputs. Our processor accepts
  `images=[...]` / `videos=[...]` in the same shape so future vLLM-style
  serving is a thin wrapper.
- Allowed-media-domains and redirect controls are a security primitive,
  not a tokenizer concern. We note this for the future pipeline layer:
  the tokenizer must never fetch remote bytes itself; callers preload
  PIL images / bytes.
