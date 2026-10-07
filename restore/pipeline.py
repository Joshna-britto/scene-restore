from .composite import composite
from .masks import prepare_mask


def restore_image(img, mask, inpainter, dilate=15, feather=6, shadow=False, seed=0, prompt=None):
    """img RGB uint8, mask 255=fill. Returns (result, final_mask, raw_generation)."""
    m = prepare_mask(mask, dilate, shadow, img)
    gen = inpainter(img, m, prompt, seed)
    return composite(img, gen, m, feather), m, gen
