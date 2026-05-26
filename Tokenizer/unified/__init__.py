# -*- coding: utf-8 -*-

__all__ = [
    "DualTrackResult",
    "DualTrackTokenizer",
    "Span",
    "build_dual_tokenizer",
    "build_misc_tokens",
    "build_unified_vocab",
    "segment_by_language",
]


def __getattr__(name):
    if name in __all__:
        from . import dual_tokenizer

        return getattr(dual_tokenizer, name)
    raise AttributeError(name)
