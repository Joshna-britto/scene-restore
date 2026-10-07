import numpy as np
from skimage.metrics import structural_similarity as ssim

from .video import flow_to, warp


def max_diff_outside(orig, out, mask):
    """Largest per-pixel difference in the undamaged region. Must be 0."""
    d = np.abs(orig.astype(int) - out.astype(int))
    d[mask > 127] = 0
    return int(d.max())


def psnr_in(a, b, mask):
    sel = mask > 127
    mse = np.mean((a[sel].astype(np.float64) - b[sel].astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def ssim_full(a, b):
    return float(ssim(a, b, channel_axis=2, data_range=255))


def flicker(outs, masks):
    """Temporal warp error inside the hole: warp out[t-1] onto out[t] with optical flow and
    measure what the flow could NOT explain (lower = steadier). Mean abs error in 0..255."""
    errs = []
    for t in range(1, len(outs)):
        flow = flow_to(outs[t], outs[t - 1])
        w, valid = warp(outs[t - 1], flow)
        sel = (masks[t] > 127) & valid
        if sel.any():
            errs.append(np.abs(outs[t][sel].astype(np.float32) - w[sel].astype(np.float32)).mean())
    return float(np.mean(errs)) if errs else 0.0


def object_consistency(orig, out, mask, weights="yolov8n.pt"):
    """Optional: run a detector on input and output; detections away from the hole must be
    identical in count and position. Returns (count_in, count_out, mean_iou_of_matches)."""
    from ultralytics import YOLO
    model = YOLO(weights)

    def dets(img):
        r = model(img[..., ::-1], verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy(); c = r.boxes.cls.cpu().numpy()
        keep = []
        for bb, cc in zip(b, c):
            x0, y0, x1, y1 = map(int, bb)
            if (mask[y0:y1, x0:x1] > 127).mean() < 0.05:  # ignore boxes overlapping the hole
                keep.append((bb, cc))
        return keep

    a, b = dets(orig), dets(out)

    def iou(p, q):
        x0, y0 = max(p[0], q[0]), max(p[1], q[1]); x1, y1 = min(p[2], q[2]), min(p[3], q[3])
        i = max(0, x1 - x0) * max(0, y1 - y0)
        u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i
        return i / u if u > 0 else 0
    ious = [max([iou(p, q) for q, cq in b if cq == cp] or [0]) for p, cp in a]
    return len(a), len(b), float(np.mean(ious)) if ious else 1.0
