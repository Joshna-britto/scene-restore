"""Temporally consistent video inpainting WITHOUT re-sampling a generative model every frame
(which is what causes flicker).

Idea: generate once (keyframe), then PROPAGATE. For frame t:
  1. estimate optical flow  frame_t -> result_{t-1}  (hole region filled with the previous result,
     then the flow inside the hole is interpolated from the surrounding flow);
  2. backward-warp result_{t-1} into frame t  -> the hole content moves with the scene;
  3. pixels that fall outside the image (disocclusion) get a small classical fill;
  4. composite into the original frame (bit-exact outside the mask).
Because the hole content is *carried over* rather than re-generated, identity cannot drift and
texture cannot pop. A moving object's mask also reveals real background seen in earlier frames."""
import cv2
import numpy as np

from .composite import composite
from .masks import prepare_mask


def read_video(path, max_frames=None, max_side=None):
    cap, frames = cv2.VideoCapture(path), []
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    while True:
        ok, f = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if max_side and max(f.shape[:2]) > max_side:
            s = max_side / max(f.shape[:2])
            f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    return frames, fps


def write_video(path, frames, fps=24):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def flow_to(cur, prev):
    """Farneback flow F with cur(x) ~= prev(x + F(x)). Swap for RAFT (torchvision) for better quality."""
    g1, g0 = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (cur, prev))
    return cv2.calcOpticalFlowFarneback(g1, g0, None, 0.5, 5, 21, 5, 7, 1.5, 0)


def fill_flow(flow, hole, band=60):
    """Replace the (untrustworthy) flow inside `hole` by a robust affine motion model fitted to the
    flow in a band AROUND the hole (iteratively trimmed least squares). Handles camera pan/zoom/rotation
    well and, unlike diffusing boundary values, cannot be corrupted by one noisy boundary pixel.
    Local independent motion inside the hole is not modelled (use RAFT + learned propagation for that)."""
    hb = hole > 127
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1,) * 2)
    ring = (cv2.dilate(hole, k) > 0) & ~hb
    ys, xs = np.nonzero(ring)
    out = flow.copy()
    if len(ys) < 200:  # tiny ring (hole fills the frame): fall back to diffusion
        h8 = hb.astype(np.uint8) * 255
        for c in range(2):
            out[..., c] = cv2.inpaint(np.ascontiguousarray(flow[..., c], dtype=np.float32), h8, 7, cv2.INPAINT_TELEA)
        return out
    if len(ys) > 30000:
        idx = np.random.default_rng(0).choice(len(ys), 30000, replace=False)
        ys, xs = ys[idx], xs[idx]
    H, W = hole.shape
    A = np.stack([xs / W, ys / H, np.ones(len(xs))], 1)
    yh, xh = np.nonzero(hb)
    Ah = np.stack([xh / W, yh / H, np.ones(len(xh))], 1)
    for c in range(2):
        b, sel = flow[ys, xs, c], np.ones(len(ys), bool)
        for _ in range(4):
            coef = np.linalg.lstsq(A[sel], b[sel], rcond=None)[0]
            res = np.abs(A @ coef - b)
            sel = res < max(0.3, 2.5 * np.median(res[sel]))
        out[yh, xh, c] = Ah @ coef
    return out


def warp(img, flow):
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    mx, my = (gx + flow[..., 0]).astype(np.float32), (gy + flow[..., 1]).astype(np.float32)
    out = cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    valid = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    return out, valid


def process_video(frames, masks, inpainter, mode="flow", dilate=15, feather=6, seed=0, prompt=None, progress=None):
    """mode='flow'      : keyframe + flow propagation (ours)
       mode='per_frame' : independent inpainting every frame with a different seed (naive baseline)."""
    outs, prev = [], None
    for t, (f, m0) in enumerate(zip(frames, masks)):
        m = prepare_mask(m0, dilate)
        if mode == "per_frame" or t == 0:
            gen = inpainter(f, m, prompt, seed + (t if mode == "per_frame" else 0))
        else:
            # Flow is only trusted OUTSIDE the hole. (Copying prev into the hole would force zero flow
            # there, so use a cheap classical fill of the current frame instead, then re-interpolate
            # the flow inside a slightly enlarged hole from the trusted surroundings.)
            guess = cv2.inpaint(f, m, 5, cv2.INPAINT_TELEA)
            big = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
            flow = fill_flow(flow_to(guess, prev), big)
            gen, valid = warp(prev, flow)
            bad = ((m > 127) & ~valid).astype(np.uint8) * 255
            if bad.any():
                gen = cv2.inpaint(gen, bad, 5, cv2.INPAINT_TELEA)
        out = composite(f, gen, m, feather)
        outs.append(out)
        prev = out
        if progress:
            progress(t + 1, len(frames))
    return outs
