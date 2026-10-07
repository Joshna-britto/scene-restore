"""Single-image restoration.  Examples:
  python scripts/run_image.py --demo --backend opencv
  python scripts/run_image.py --image photo.jpg --mask mask.png --backend sd
  python scripts/run_image.py --image photo.jpg --box 100 80 260 300 --backend lama --shadow
"""
import argparse, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from restore.inpaint import get_inpainter
from restore.masks import box_mask, random_damage_mask
from restore.metrics import max_diff_outside, ssim_full
from restore.pipeline import restore_image


def load_rgb(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def save(path, rgb):
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image"); ap.add_argument("--mask")
    ap.add_argument("--box", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--demo", action="store_true", help="bundled sample image + random damage mask")
    ap.add_argument("--backend", default="opencv", choices=["opencv", "sim", "lama", "sd"])
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    ap.add_argument("--dilate", type=int, default=15); ap.add_argument("--feather", type=int, default=6)
    ap.add_argument("--shadow", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt"); ap.add_argument("--out", default="outputs/image")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.demo:
        from skimage import data
        img = data.chelsea()
        mask = random_damage_mask(*img.shape[:2], seed=3)
    else:
        img = load_rgb(a.image)
        h, w = img.shape[:2]
        mask = box_mask(h, w, a.box) if a.box else cv2.imread(a.mask, 0)

    kw = {"model_id": a.model} if a.backend == "sd" else {}
    out, m, gen = restore_image(img, mask, get_inpainter(a.backend, **kw), a.dilate, a.feather,
                                a.shadow, a.seed, a.prompt)
    diff = np.abs(img.astype(int) - out.astype(int)).max(-1)
    save(f"{a.out}/00_input.png", img); save(f"{a.out}/01_mask.png", m)
    save(f"{a.out}/02_raw_generation.png", gen); save(f"{a.out}/03_result.png", out)
    save(f"{a.out}/04_changed_pixels.png", np.clip(diff * 4, 0, 255).astype(np.uint8))
    stats = {"backend": a.backend, "hole_pixels_pct": round(100 * float((m > 127).mean()), 2),
             "max_pixel_change_outside_mask": max_diff_outside(img, out, m),
             "ssim_input_vs_result": round(ssim_full(img, out), 4), "seed": a.seed}
    json.dump(stats, open(f"{a.out}/metrics.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
