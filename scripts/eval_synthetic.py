"""Ground-truth benchmark with NO downloads: a textured photo pans sideways while a coloured
box moves across it. The mask is the box; the true background behind it is known, so we can
score hole PSNR for real. Compares naive per-frame vs flow propagation.
  python scripts/eval_synthetic.py --backend opencv        (CPU, seconds)
  python scripts/eval_synthetic.py --backend sd            (GPU)"""
import argparse, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skimage import data
from restore.inpaint import get_inpainter
from restore.masks import prepare_mask
from restore.metrics import flicker, max_diff_outside, psnr_in
from restore.video import process_video, write_video


def make_clip(n=40, h=300, w=420):
    bg = data.coffee()
    bg = cv2.resize(bg, (w + 2 * n, h))
    gt, frames, masks = [], [], []
    for t in range(n):
        base = bg[:, 2 * t: 2 * t + w].copy()           # camera pans 2 px / frame
        x = 40 + 7 * t; y = 90 + (t % 10)                # foreground moves 7 px / frame
        f = base.copy(); f[y:y + 90, x:x + 70] = (200, 30, 30)
        m = np.zeros((h, w), np.uint8); m[y:y + 90, x:x + 70] = 255
        gt.append(base); frames.append(f); masks.append(m)
    return gt, frames, masks


def gt_temporal_error(outs, masks, shift=2):
    """Camera pans exactly `shift` px/frame, so out[t][x] should equal out[t-1][x+shift] in the hole.
    Mean abs error there = flicker measured against KNOWN motion (no estimated flow involved)."""
    e = []
    for t in range(1, len(outs)):
        a = outs[t][:, :-shift].astype(np.float32); b = outs[t - 1][:, shift:].astype(np.float32)
        sel = masks[t][:, :-shift] > 127
        if sel.any():
            e.append(np.abs(a[sel] - b[sel]).mean())
    return float(np.mean(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="opencv", help="opencv | sim (simulated stochastic, CPU) | lama | sd"); ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    ap.add_argument("--out", default="outputs/synthetic")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    gt, frames, masks = make_clip()
    pm = [prepare_mask(m, 15) for m in masks]
    inp = get_inpainter(a.backend, **({"model_id": a.model} if a.backend == "sd" else {}))
    res = {}
    for mode in ["per_frame", "flow"]:
        outs = process_video(frames, masks, inp, mode)
        write_video(f"{a.out}/{mode}.mp4", outs, 12)
        res[mode] = {"hole_psnr_vs_ground_truth_db": round(float(np.mean([psnr_in(o, g, m) for o, g, m in zip(outs, gt, pm)])), 2),
                     "flicker_warp_error": round(flicker(outs, pm), 3),
                     "temporal_error_vs_known_motion": round(gt_temporal_error(outs, pm), 2),
                     "max_change_outside_mask": max(max_diff_outside(f, o, m) for f, o, m in zip(frames, outs, pm))}
    write_video(f"{a.out}/input.mp4", frames, 12)
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
