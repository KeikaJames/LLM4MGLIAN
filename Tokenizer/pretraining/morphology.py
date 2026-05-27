# -*- coding: utf-8 -*-
"""Model-side morphology features derived from tokenizer spans."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class TokenLike(Protocol):
    track: str
    start: int
    end: int


WORD_TRACKS = {"mn", "zh", "en"}


def derive_morph_info_from_tokens(
    tokens: Sequence[TokenLike],
) -> tuple[list[int], list[int]]:
    """Derive word position and intra-word subtoken depth from encoded tokens.

    Mongolian MorphBPE pieces that are contiguous in the original text share a
    word position and advance morph_depth. Chinese/English BPE pieces use the
    same convention. Spaces, specials, and punctuation reset the current word.
    """

    word_positions: list[int] = []
    morph_depths: list[int] = []

    cur_word = -1
    cur_depth = 0
    prev_track: str | None = None
    prev_end: int | None = None

    for token in tokens:
        track = token.track
        start = int(token.start)
        end = int(token.end)

        if start < 0 or end < 0:
            word_positions.append(max(cur_word, 0))
            morph_depths.append(0)
            prev_track = None
            prev_end = None
            continue

        if track not in WORD_TRACKS:
            word_positions.append(max(cur_word, 0))
            morph_depths.append(0)
            prev_track = None
            prev_end = None
            continue

        same_word = prev_track == track and prev_end is not None and start == prev_end
        if same_word:
            cur_depth += 1
        else:
            cur_word += 1
            cur_depth = 0

        word_positions.append(cur_word)
        morph_depths.append(cur_depth)
        prev_track = track
        prev_end = end

    return word_positions, morph_depths


def derive_morph_info_from_offsets(
    token_offsets: Sequence[tuple[int, int] | list[int]],
) -> tuple[list[int], list[int]]:
    """Best-effort fallback for legacy encoded rows without token tracks."""

    word_positions: list[int] = []
    morph_depths: list[int] = []

    cur_word = -1
    cur_depth = 0
    prev_end: int | None = None

    for offset in token_offsets:
        start, end = int(offset[0]), int(offset[1])
        if start < 0 or end < 0:
            word_positions.append(max(cur_word, 0))
            morph_depths.append(0)
            prev_end = None
            continue

        if prev_end is None or start != prev_end:
            cur_word += 1
            cur_depth = 0
        else:
            cur_depth += 1

        word_positions.append(cur_word)
        morph_depths.append(cur_depth)
        prev_end = end

    return word_positions, morph_depths
