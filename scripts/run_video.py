"""Video restoration.  Examples:
  python scripts/run_video.py --video clip.mp4 --mask mask.png --backend lama           # static damaged region
  python scripts/run_video.py --video clip.mp4 --mask-dir masks/ --backend sd           # per-frame masks (e.g. remove a walker)
  python scripts/run_video.py --video clip.mp4 --mask mask.png --mode per_frame         # naive baseline for comparison
"""
import argparse, glob, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from restore.inpaint import get_inpainter
from restore.masks import prepare_mask
from restore.metrics import flicker, max_diff_outside
from restore.video import process_video, read_video, write_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mask"); g.add_argument("--mask-dir")
    ap.add_argument("--backend", default="opencv", choices=["opencv", "sim", "lama", "sd"])
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    ap.add_argument("--mode", default="flow", choices=["flow", "per_frame"])
    ap.add_argument("--max-frames", type=int, default=120); ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--dilate", type=int, default=15); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/video")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    frames, fps = read_video(a.video, a.max_frames, a.max_side)
    h, w = frames[0].shape[:2]
    if a.mask:
        m = cv2.resize(cv2.imread(a.mask, 0), (w, h), interpolation=cv2.INTER_NEAREST)
        masks = [m] * len(frames)
    else:
        files = sorted(glob.glob(os.path.join(a.mask_dir, "*.png")))[:len(frames)]
        masks = [cv2.resize(cv2.imread(f, 0), (w, h), interpolation=cv2.INTER_NEAREST) for f in files]
        frames = frames[:len(masks)]

    kw = {"model_id": a.model} if a.backend == "sd" else {}
    outs = process_video(frames, masks, get_inpainter(a.backend, **kw), a.mode, a.dilate, seed=a.seed,
                         progress=lambda i, n: print(f"\rframe {i}/{n}", end=""))
    print()
    pm = [prepare_mask(m, a.dilate) for m in masks]
    stats = {"mode": a.mode, "backend": a.backend, "frames": len(outs),
             "max_pixel_change_outside_mask": max(max_diff_outside(f, o, m) for f, o, m in zip(frames, outs, pm)),
             "flicker_warp_error_in_hole": round(flicker(outs, pm), 3)}
    write_video(f"{a.out}/result_{a.mode}.mp4", outs, fps)
    write_video(f"{a.out}/input.mp4", frames, fps)
    json.dump(stats, open(f"{a.out}/metrics_{a.mode}.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
