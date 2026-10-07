import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skimage import data
from restore.inpaint import OpenCVInpainter
from restore.masks import box_mask, random_damage_mask
from restore.metrics import max_diff_outside
from restore.pipeline import restore_image


def test_undamaged_pixels_bit_exact():
    img = data.astronaut()
    for mask in (box_mask(*img.shape[:2], (100, 100, 250, 300)), random_damage_mask(*img.shape[:2], seed=1)):
        out, m, _ = restore_image(img, mask, OpenCVInpainter())
        assert max_diff_outside(img, out, m) == 0


def test_hole_actually_changes():
    img = data.astronaut()
    mask = box_mask(*img.shape[:2], (100, 100, 250, 300))
    out, m, _ = restore_image(img, mask, OpenCVInpainter())
    assert np.abs(out.astype(int) - img.astype(int))[m > 127].mean() > 1
