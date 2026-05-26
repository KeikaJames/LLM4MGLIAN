# -*- coding: utf-8 -*-
"""HuggingFace-backed token track wrapper with offset fallback."""

from __future__ import annotations

from typing import Any

from Tokenizer.unified.encoded import EncodedToken


class HFTrackTokenizer:
    def __init__(
        self,
        hf_tokenizer: Any,
        prefix: str,
        local_to_global: dict[int, int],
        unk_id: int,
        track: str,
    ):
        self.hf_tokenizer = hf_tokenizer
        self.prefix = prefix
        self.local_to_global = local_to_global
        self.unk_id = unk_id
        self.track = track

    def encode(self, text: str) -> list[int]:
        return [tok.id for tok in self.encode_with_offsets(text)]

    def encode_with_offsets(self, text: str, base_start: int = 0) -> list[EncodedToken]:
        fast = self._try_fast_offsets(text, base_start)
        if fast is not None:
            return fast

        local_ids = self.hf_tokenizer.encode(text, add_special_tokens=False)
        if not local_ids:
            return []
        span_len = max(1, len(text) // len(local_ids))
        tokens: list[EncodedToken] = []
        cursor = 0
        for i, local_id in enumerate(local_ids):
            start_rel = cursor
            end_rel = len(text) if i == len(local_ids) - 1 else min(len(text), cursor + span_len)
            cursor = end_rel
            raw = self._id_to_token(local_id)
            tokens.append(
                EncodedToken(
                    self.local_to_global.get(local_id, self.unk_id),
                    self.prefix + raw,
                    self.track,
                    base_start + start_rel,
                    base_start + end_rel,
                )
            )
        return tokens

    def _try_fast_offsets(
        self, text: str, base_start: int
    ) -> list[EncodedToken] | None:
        if not callable(self.hf_tokenizer):
            return None
        try:
            encoded = self.hf_tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
        except (TypeError, ValueError, NotImplementedError):
            return None

        input_ids = self._field(encoded, "input_ids")
        offsets = self._field(encoded, "offset_mapping")
        if input_ids is None or offsets is None:
            return None

        tokens: list[EncodedToken] = []
        for local_id, (start, end) in zip(input_ids, offsets):
            if isinstance(local_id, list):
                return None
            raw = self._id_to_token(local_id)
            tokens.append(
                EncodedToken(
                    self.local_to_global.get(local_id, self.unk_id),
                    self.prefix + raw,
                    self.track,
                    base_start + int(start),
                    base_start + int(end),
                )
            )
        return tokens

    def _field(self, encoded: Any, name: str) -> Any:
        if isinstance(encoded, dict):
            return encoded.get(name)
        return getattr(encoded, name, None)

    def _id_to_token(self, local_id: int) -> str:
        converter = getattr(self.hf_tokenizer, "convert_ids_to_tokens", None)
        if converter is not None:
            token = converter(local_id)
            if isinstance(token, str):
                return token
        vocab = getattr(self.hf_tokenizer, "get_vocab", lambda: {})()
        for token, idx in vocab.items():
            if idx == local_id:
                return token
        return str(local_id)
