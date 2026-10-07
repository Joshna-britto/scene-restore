
import os
import cv2
import numpy as np
import gradio as gr

from restore.inpaint import get_inpainter
from restore.pipeline import restore_image
from restore.video import read_video, write_video, process_video


_cache = {}


# =========================
# IMAGE RESTORATION
# =========================

def run_image(editor, backend, dilate, shadow, seed, prompt):

    if editor is None:
        raise gr.Error("Please upload an image first.")

    img = editor["background"]

    if hasattr(img, "convert"):
        img = np.array(img.convert("RGB"))
    else:
        img = np.array(img)[..., :3]

    layer = np.array(editor["layers"][0])
    mask = (layer[..., 3] > 0).astype(np.uint8) * 255

    if not mask.any():
        raise gr.Error("Please paint over the area you want to remove.")

    if backend not in _cache:
        _cache[backend] = get_inpainter(backend)

    out, final_mask, _ = restore_image(
        img,
        mask,
        _cache[backend],
        int(dilate),
        6,
        shadow,
        int(seed),
        prompt or None
    )

    return out, final_mask


# =========================
# VIDEO RESTORATION
# =========================

def run_video(
    video_path,
    mask_image,
    backend,
    mode,
    dilate,
    seed,
    prompt,
    max_frames
):

    if video_path is None:
        raise gr.Error("Please upload a video.")

    if mask_image is None:
        raise gr.Error("Please upload a mask image.")

    # Read video
    frames, fps = read_video(
        video_path,
        max_frames=int(max_frames),
        max_side=512
    )

    if not frames:
        raise gr.Error("Could not read the video.")

    h, w = frames[0].shape[:2]

    # Convert mask to grayscale
    mask = np.array(mask_image)

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask[..., :3], cv2.COLOR_RGB2GRAY)

    # Resize mask to video size
    mask = cv2.resize(
        mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    # Make mask binary
    mask = (mask > 127).astype(np.uint8) * 255

    if not mask.any():
        raise gr.Error("The mask is empty.")

    # Get inpainter
    if backend not in _cache:
        _cache[backend] = get_inpainter(backend)

    # Create same mask for every frame
    masks = [mask.copy() for _ in frames]

    # Output directory
    os.makedirs("outputs/gradio_video", exist_ok=True)

    output_path = "outputs/gradio_video/restored_video.mp4"

    # Progress callback
    def update_progress(current, total):
        if total > 0:
            progress(current / total, desc=f"Processing frame {current}/{total}")

    # Process video
    outs = process_video(
        frames,
        masks,
        _cache[backend],
        mode=mode,
        dilate=int(dilate),
        feather=6,
        seed=int(seed),
        prompt=prompt or None,
        progress=update_progress
    )

    # Write output
    write_video(
        output_path,
        outs,
        fps=fps
    )

    return output_path


# =========================
# GRADIO UI
# =========================

with gr.Blocks(title="SceneRestore") as demo:

    gr.Markdown(
        """
        # SceneRestore
        ### Scene-consistent image and video inpainting

        **Remove unwanted regions while preserving undamaged pixels.**

        For video, optical-flow propagation is used to reduce
        frame-to-frame flickering.
        """
    )

    # =====================
    # IMAGE TAB
    # =====================

    with gr.Tab("🖼️ Image Restoration"):

        gr.Markdown(
            """
            Upload an image and **paint over the person/object you want to remove**.
            """
        )

        image_editor = gr.ImageEditor(
            type="pil",
            label="Image + Brush Mask"
        )

        with gr.Row():

            image_backend = gr.Dropdown(
                ["opencv", "sd"],
                value="sd",
                label="Backend"
            )

            image_dilate = gr.Slider(
                0,
                40,
                value=15,
                step=1,
                label="Mask dilation (px)"
            )

            image_shadow = gr.Checkbox(
                label="Grow mask over shadows"
            )

            image_seed = gr.Number(
                value=0,
                label="Seed",
                precision=0
            )

        image_prompt = gr.Textbox(
            label="Prompt (SD only)",
            placeholder="Describe what should appear after removing the object..."
        )

        image_button = gr.Button(
            "Run Image Restoration",
            variant="primary"
        )

        with gr.Row():

            image_result = gr.Image(
                label="Result"
            )

            image_mask_result = gr.Image(
                label="Final mask used"
            )

        image_button.click(
            run_image,
            [
                image_editor,
                image_backend,
                image_dilate,
                image_shadow,
                image_seed,
                image_prompt
            ],
            [
                image_result,
                image_mask_result
            ]
        )


    # =====================
    # VIDEO TAB
    # =====================

    with gr.Tab("🎥 Video Restoration"):

        gr.Markdown(
            """
            ### Upload your own video

            1. Upload the video.
            2. Upload a **black-and-white mask image**.
            3. White = region to restore/remove.
            4. Black = region to keep.
            5. Use **Flow** mode for temporally consistent restoration.
            """
        )

        with gr.Row():

            video_input = gr.Video(
                label="Upload Video",
        
            )

            video_mask = gr.Image(
                label="Upload Mask",
                type="numpy"
            )

        with gr.Row():

            video_backend = gr.Dropdown(
                ["opencv", "sd"],
                value="sd",
                label="Backend"
            )

            video_mode = gr.Dropdown(
                ["flow", "per_frame"],
                value="flow",
                label="Video Mode"
            )

        with gr.Row():

            video_dilate = gr.Slider(
                0,
                40,
                value=15,
                step=1,
                label="Mask dilation (px)"
            )

            video_seed = gr.Number(
                value=0,
                label="Seed",
                precision=0
            )

            max_frames = gr.Slider(
                10,
                120,
                value=60,
                step=10,
                label="Maximum frames"
            )

        video_prompt = gr.Textbox(
            label="Prompt (SD only)",
            placeholder="Describe the background that should replace the unwanted region..."
        )

        video_button = gr.Button(
            "Run Video Restoration",
            variant="primary"
        )

        video_output = gr.Video(
            label="Restored Video"
        )

        video_button.click(
            run_video,
            [
                video_input,
                video_mask,
                video_backend,
                video_mode,
                video_dilate,
                video_seed,
                video_prompt,
                max_frames
            ],
            video_output
        )


if __name__ == "__main__":
    demo.launch(share=True)
