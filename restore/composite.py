"""Paste-back: THE mechanism that guarantees undamaged pixels are never altered."""
import cv2
import numpy as np


def feather_alpha(mask, feather):
    """Alpha is exactly 0 outside the mask and ramps 0->1 over `feather` px *inside* it,
    so blending never leaks into undamaged pixels."""
    hard = (mask > 127).astype(np.uint8)
    if feather <= 0:
        return hard.astype(np.float32)
    d = cv2.distanceTransform(hard, cv2.DIST_L2, 5)
    return np.clip(d / float(feather), 0, 1).astype(np.float32)


def match_color(orig, gen, mask, ring=12):
    """Diffusion/VAE round-trips drift colour slightly. Estimate the per-channel offset on a thin
    ring just OUTSIDE the mask (where we know the truth) and apply it to the generated pixels."""
    hard = (mask > 127).astype(np.uint8)
    outer = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1,) * 2)) - hard
    if outer.sum() < 50:
        return gen
    o = orig[outer > 0].astype(np.float32).mean(0)
    g = gen[outer > 0].astype(np.float32).mean(0)
    return np.clip(gen.astype(np.float32) + (o - g), 0, 255).astype(np.uint8)


def composite(orig, gen, mask, feather=6, color_match=True):
    if color_match:
        gen = match_color(orig, gen, mask)
    a = feather_alpha(mask, feather)[..., None]
    out = np.rint(a * gen.astype(np.float32) + (1 - a) * orig.astype(np.float32)).astype(np.uint8)
    keep = mask <= 127
    out[keep] = orig[keep]  # hard guarantee: bit-exact outside the mask
    return out
