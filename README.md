# Scene-Consistent Inpainting (HNX26PSI06)

Fill missing / damaged / unwanted regions of **images and video** while leaving everything else
**bit-exactly unchanged** (same objects, positions, perspective, lighting), with **temporal consistency** for video.

## Core idea (3 mechanisms)
1. **Paste-back guarantee** (`restore/composite.py`): the generator's output is only ever used *inside* the mask.
   Outside the mask the output equals the input pixel-for-pixel (verified by `max_pixel_change_outside_mask = 0`
   in every run and by `tests/`). Identity, count and position of undamaged objects therefore cannot change.
2. **Context-aware generation** (`restore/inpaint.py`): crop around the hole (+50% context), generate at the model's
   native resolution, fixed seed, then resize back. Colour-offset correction on a ring just outside the hole removes
   VAE colour drift; inward feathering hides the seam. Optional shadow-aware mask growth (`--shadow`).
3. **Generate once, then propagate** (`restore/video.py`): keyframe is inpainted by the model; later frames reuse the
   previous result warped by optical flow (robust affine motion model fitted around the hole), so texture is
   *carried* instead of re-sampled -> no flicker, no identity drift. A moving object's mask reveals real background
   seen in earlier frames.

```
input + mask -> prepare_mask (binarise, [shadow grow], dilate)
   image: crop -> backend (opencv | lama | sd) -> uncrop -> colour match -> feathered composite -> result
   video: frame0 as image; frame t: flow(t->t-1) -> affine-fill flow in hole -> warp prev result
          -> Telea only for disoccluded pixels -> composite into ORIGINAL frame t
```

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # CPU-only is fine for the opencv backend and all tests
```
GPU recommended for `--backend sd`. First SD run downloads weights from Hugging Face
(`stabilityai/stable-diffusion-2-inpainting`; change with `--model`, verify the repo id is still available).

## Run
```bash
# Image
python scripts/run_image.py --demo --backend opencv                    # no downloads, CPU
python scripts/run_image.py --image photo.jpg --mask mask.png --backend sd
python scripts/run_image.py --image photo.jpg --box 100 80 260 300 --backend lama --shadow
# Video (static mask image, or --mask-dir with per-frame PNG masks)
python scripts/run_video.py --video clip.mp4 --mask mask.png --backend lama
python scripts/run_video.py --video clip.mp4 --mask mask.png --mode per_frame   # naive baseline to compare
# Benchmark with ground truth (no downloads)
python scripts/eval_synthetic.py --backend opencv
# Interactive demo
python app.py
# Tests
pytest -q tests
```
Outputs: `00_input, 01_mask, 02_raw_generation, 03_result, 04_changed_pixels` (x4 amplified diff; black outside the mask) + `metrics.json`.

## Evidence (reproduce: `python scripts/eval_synthetic.py --backend opencv`)
Synthetic clip: photo panning 2 px/frame, box moving across it, mask = box, true background known. 40 frames.

| metric (lower flicker / higher PSNR is better) | per-frame fill | **flow propagation (ours)** |
|---|---|---|
| hole PSNR vs ground truth | 15.8 dB | **28.6 dB** |
| temporal error vs known motion | 13.8 | **1.1** |
| max pixel change outside mask | 0 | 0 |

Files in `samples/`. Notes: this run uses the classical OpenCV backend so it runs anywhere. `--backend sim` is a
*simulated* stochastic generator (test-only) used to show the flicker mechanism on CPU; it makes no quality claim.
Add your own GPU numbers for `--backend sd/lama` here after running.

## Scope note
**Minimum viable (implemented):** image inpainting with exact outside-mask preservation, 3 backends, mask tools
(random damage, box, COCO objects), metrics, tests. **Stretch (implemented):** video propagation with flow,
shadow-aware mask growth, ground-truth temporal benchmark, Gradio app, optional YOLO object-consistency metric.
**Not done / limitations:** local independent motion inside the hole is not modelled (affine flow only), repeated
warping slowly blurs long clips (mitigation: re-anchor on clean frames / RAFT), reflections and heavy shadows are
handled only heuristically, no learned video-inpainting network (see ProPainter / E2FGVI).

## Resources declared
Libraries: NumPy, OpenCV, scikit-image (sample photos `chelsea`, `coffee`, `astronaut`), Pillow, PyTorch, diffusers,
Gradio, ultralytics (optional), pycocotools (optional). Models (optional, downloaded at runtime): Stable Diffusion 2
inpainting (stabilityai), LaMa via `simple-lama-inpainting`, YOLOv8n. Datasets (optional): COCO, Places.
AI assistance (Claude) was used to draft code; the team reviewed and can explain every module.
