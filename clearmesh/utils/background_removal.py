"""Background removal for input images using rembg."""

import numpy as np
from PIL import Image


def remove_background(image: Image.Image) -> Image.Image:
    """Remove background from an input image.

    Uses rembg (U2-Net based) for high-quality background removal.
    Returns RGBA image with transparent background.

    Args:
        image: Input PIL Image (RGB or RGBA)

    Returns:
        RGBA image with background removed
    """
    from rembg import remove

    result = remove(image)

    # Ensure RGBA output
    if result.mode != "RGBA":
        result = result.convert("RGBA")

    return result


def remove_background_batch(images: list[Image.Image]) -> list[Image.Image]:
    """Remove background from multiple images.

    Args:
        images: List of PIL Images

    Returns:
        List of RGBA images with backgrounds removed
    """
    return [remove_background(img) for img in images]
