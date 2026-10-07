"""Inpainting backends. All share the signature  f(img_rgb, mask255, prompt, seed) -> full-size RGB.
Whatever they do outside the mask is irrelevant: composite.py throws it away."""
import cv2
import numpy as np
from PIL import Image

DEFAULT_PROMPT = "a clean, realistic, seamless continuation of the surrounding scene, high quality photo"
NEGATIVE = "person, people, human, animal, object, text, watermark, logo, blurry, low quality, distorted"


def _crop_box(mask, h, w, ctx=0.5, min_side=256):
    """Square crop around the hole with `ctx` extra context on each side -> the model sees the hole
    at high effective resolution instead of a tiny region of a downscaled image.
    Falls back to the whole image if the hole does not fit in a square crop."""
    ys, xs = np.where(mask > 127)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = max(int(max(y1 - y0, x1 - x0) * (1 + 2 * ctx)), min_side)
    if side > min(h, w):
        if max(y1 - y0, x1 - x0) > min(h, w):
            return 0, 0, h, w
        side = min(h, w)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    top = int(np.clip(cy - side // 2, 0, h - side))
    left = int(np.clip(cx - side // 2, 0, w - side))
    return top, left, side, side


class OpenCVInpainter:
    """Classical Telea inpainting: CPU-only baseline / fallback. Smears texture on big holes."""
    name = "opencv"

    def __call__(self, img, mask, prompt=None, seed=0):
        return cv2.inpaint(img, (mask > 127).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)


class SimulatedStochasticInpainter:
    """TEST-ONLY stand-in for a *stochastic* generator: Telea fill + seed-dependent low-frequency
    texture noise inside the hole. It lets the CPU benchmark show what happens when a model gives a
    different sample each frame. It is NOT a real generative model and is not used for any claim
    about visual quality."""
    name = "sim"

    def __call__(self, img, mask, prompt=None, seed=0):
        base = cv2.inpaint(img, (mask > 127).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
        rng = np.random.default_rng(seed)
        noise = cv2.GaussianBlur(rng.normal(0, 1, img.shape).astype(np.float32), (0, 0), 4) * 40
        out = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return np.where((mask > 127)[..., None], out, base)


class LamaInpainter:
    """LaMa (Fourier-conv, large-mask inpainting). Fast, deterministic, strong for object removal."""
    name = "lama"

    def __init__(self):
        from simple_lama_inpainting import SimpleLama
        self.model = SimpleLama()

    def __call__(self, img, mask, prompt=None, seed=0):
        h, w = img.shape[:2]
        out = np.array(self.model(Image.fromarray(img), Image.fromarray(mask)))
        if out.shape[:2] != (h, w):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return out


class SDInpainter:
    """Stable Diffusion inpainting through diffusers, with crop-around-hole and a FIXED seed."""
    name = "sd"

    def __init__(self, model_id="stable-diffusion-v1-5/stable-diffusion-inpainting", device=None,
                 steps=30, target=512, guidance=7.5):
        import torch
        from diffusers import AutoPipelineForInpainting
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=dtype).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.steps, self.target, self.guidance = steps, target, guidance

    def __call__(self, img, mask, prompt=None, seed=0):
        h, w = img.shape[:2]
        top, left, ch, cw = _crop_box(mask, h, w)
        crop, mcrop = img[top:top + ch, left:left + cw], mask[top:top + ch, left:left + cw]
        s = self.target / max(ch, cw)
        nw, nh = max(8, int(round(cw * s / 8)) * 8), max(8, int(round(ch * s / 8)) * 8)
        pil_i = Image.fromarray(crop).resize((nw, nh), Image.LANCZOS)
        pil_m = Image.fromarray(mcrop).resize((nw, nh), Image.NEAREST)
        gen = self.torch.Generator(device=self.device).manual_seed(int(seed))
        res = self.pipe(prompt=prompt or DEFAULT_PROMPT, negative_prompt=NEGATIVE, image=pil_i,
                        mask_image=pil_m, height=nh, width=nw, num_inference_steps=self.steps,
                        guidance_scale=self.guidance, generator=gen).images[0]
        full = img.copy()
        full[top:top + ch, left:left + cw] = np.array(res.resize((cw, ch), Image.LANCZOS))
        return full


def get_inpainter(name="opencv", **kw):
    return {"opencv": OpenCVInpainter, "sim": SimulatedStochasticInpainter, "lama": LamaInpainter, "sd": SDInpainter}[name](**kw)
