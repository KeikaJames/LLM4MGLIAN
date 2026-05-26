# Tokenizer Architecture Notes

The unified tokenizer keeps one global id space and delegates text handling by
span type:

- `traditional_mongolian`: suffix inventory, Unicode control normalization, and
  reverse stemming for MorphBPE boundaries.
- `unified`: routing across Mongolian MorphBPE, Chinese/English HuggingFace BPE,
  special tokens, and byte fallback.
- `multimodal`: image placeholder and image patch token definitions.

Implementation references checked during scaffolding:

- HuggingFace tokenizers expose token offsets for mapping encoded tokens back to
  source spans:
  https://huggingface.co/docs/tokenizers/main/api/encoding
- tiktoken treats special tokens as explicit ids and requires explicit handling
  for special-token text:
  https://github.com/openai/tiktoken/blob/main/tiktoken/core.py
- Qwen2-VL uses distinct vision start/end and image placeholder token ids:
  https://huggingface.co/docs/transformers/v5.0.0/ko/model_doc/qwen2_vl
- LLaVA processors expose an image token and infer/expand the required image
  token count from patch and vision feature configuration:
  https://huggingface.co/docs/transformers/v4.48.0/model_doc/llava
- Qwen-VL/LLaVA/vLLM-style multimodal processing expands image placeholders
  into model-specific visual token slots and validates alignment between image
  placeholders and image features:
  https://docs.vllm.ai/en/latest/api/vllm/multimodal/processing.html
