# -*- coding: utf-8 -*-
"""Multimodal placeholder and image patch tokens."""

IMAGE_PLACEHOLDER = "<image>"
IMAGE_START = "<image_start>"
IMAGE_PATCH = "<image_patch>"
IMAGE_END = "<image_end>"

MULTIMODAL_SPECIAL_TOKENS = {
    IMAGE_PLACEHOLDER: 7,
    IMAGE_START: 8,
    IMAGE_PATCH: 9,
    IMAGE_END: 10,
}


def expand_image_placeholders(text: str, patches_per_image: int) -> str:
    if patches_per_image < 1:
        raise ValueError("patches_per_image must be positive")

    replacement = IMAGE_START + (IMAGE_PATCH * patches_per_image) + IMAGE_END
    return text.replace(IMAGE_PLACEHOLDER, replacement)
