"""Mask utilities. Convention everywhere: uint8 HxW, 255 = region to regenerate, 0 = keep."""
import cv2
import numpy as np


def _k(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def random_damage_mask(h, w, seed=0, strokes=5, max_width=35):
    """Synthetic 'scratch/damage' mask: random polylines with random thickness."""
    rng = np.random.default_rng(seed)
    m = np.zeros((h, w), np.uint8)
    for _ in range(strokes):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        for _ in range(int(rng.integers(2, 6))):
            nx = int(np.clip(x + rng.integers(-w // 4, w // 4), 0, w - 1))
            ny = int(np.clip(y + rng.integers(-h // 4, h // 4), 0, h - 1))
            cv2.line(m, (x, y), (nx, ny), 255, int(rng.integers(8, max_width)))
            x, y = nx, ny
    return m


def box_mask(h, w, box):
    x0, y0, x1, y1 = box
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


def coco_object_mask(ann_json, image_id, category=None):
    """Union of COCO instance masks for an image (optionally one category, e.g. 'person')."""
    from pycocotools.coco import COCO
    coco = COCO(ann_json)
    cats = coco.getCatIds(catNms=[category]) if category else []
    ids = coco.getAnnIds(imgIds=[image_id], catIds=cats, iscrowd=None)
    m = None
    for a in coco.loadAnns(ids):
        am = coco.annToMask(a) * 255
        m = am if m is None else np.maximum(m, am)
    if m is None:
        raise ValueError("no matching annotations")
    return m.astype(np.uint8)


def grow_into_shadow(img, m, ring=40, drop=0.15):
    """Heuristic shadow handling: add pixels near the mask that are clearly darker than the
    scene further away AND connected to the mask. Removing a person but leaving their shadow
    is the classic failure case, so we grow the mask over it."""
    L = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    d1, d2 = cv2.dilate(m, _k(ring)), cv2.dilate(m, _k(2 * ring))
    near = cv2.bitwise_and(d1, cv2.bitwise_not(m))
    far = cv2.bitwise_and(d2, cv2.bitwise_not(d1))
    if not far.any():
        return m
    ref = np.median(L[far > 0])
    dark = (((L < ref * (1 - drop)) & (near > 0)).astype(np.uint8)) * 255
    _, lbl = cv2.connectedComponents(cv2.bitwise_or(dark, m))
    keep = np.unique(lbl[m > 0])
    keep = keep[keep != 0]
    return (np.isin(lbl, keep).astype(np.uint8)) * 255


def prepare_mask(mask, dilate=15, shadow=False, img=None):
    """Binarise, optionally grow over shadows, then dilate so the mask also covers the
    object's halo / damaged edge pixels (dilation is the single most important quality knob)."""
    m = (mask > 127).astype(np.uint8) * 255
    if shadow and img is not None:
        m = grow_into_shadow(img, m)
    if dilate > 0:
        m = cv2.dilate(m, _k(dilate))
    return m
