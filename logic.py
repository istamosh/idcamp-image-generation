import torch
import gc
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler
)
from PIL import Image, ImageFilter

# MODEL LOADER (JANGAN DIUBAH)
def load_models_cached():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models to {device}")

    pipe_txt2img = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.bfloat16, device_map="cuda"
    ).to(device)

    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    ).to(device)

    return pipe_txt2img, pipe_inpaint

# Ini mencegah error "Function not found" jika hanya mengerjakan Basic
def flush_memory(): pass
def set_scheduler(pipe, name): return pipe
def run_inpainting(pipe, img, mask, prompt, strength): return None
def prepare_outpainting(img, expand=128): return img, None


def generate_image(pipe, prompt, neg_prompt, seed, steps, cfg, num_images=1, scheduler_name="Euler A"):

    ### MULAI CODE ###

    # Setup Generator (Seed)
    generator = torch.Generator("cuda").manual_seed(int(seed)) if torch.cuda.is_available() else torch.Generator().manual_seed(int(seed))

    # Generate gambar standard
    image = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        guidance_scale=cfg,
        num_inference_steps=steps,
        generator=generator
    ).images[0]

    ### SELESAI CODE ###

    # Kembalikan dalam bentuk List agar kompatibel dengan UI (List isi 1 gambar)
    return [image]
