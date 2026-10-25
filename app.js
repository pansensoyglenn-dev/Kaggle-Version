window.__appLoaded = true;
const $ = (id) => document.getElementById(id);

const py = (s) => JSON.stringify(s.replace(/\s+/g, " ").trim());

function clamp(value, min, max, fallback) {
  const n = parseInt(value, 10);
  if (Number.isNaN(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

function safeFileName(name) {
  const cleaned = name.trim().replace(/[^\w.\-]/g, "_");
  if (!cleaned) return "output_6s.mp4";
  return cleaned.toLowerCase().endsWith(".mp4") ? cleaned : cleaned + ".mp4";
}

function buildStep1(v) {
  return `!pip install -q diffusers transformers accelerate safetensors

import gc
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, AutoencoderKL
from IPython.display import display

# Realism checkpoint
MODEL_ID = "SG161222/RealVisXL_V5.0"

prompt = ${py(v.prompt)}
negative_prompt = ${py(v.negative)}
SEED = ${v.seed}

print("Loading VAE and RealVisXL Pipeline...")

# Fixed VAE to prevent half-precision color corruption
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)

sdxl = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    low_cpu_mem_usage=True,
)

# Better scheduler for sharpness
sdxl.scheduler = DPMSolverMultistepScheduler.from_config(
    sdxl.scheduler.config,
    use_karras_sigmas=True,
)

# Memory optimizations
sdxl.enable_model_cpu_offload()
sdxl.vae.enable_slicing()
sdxl.vae.enable_tiling()

print("Generating realistic images...")
for i in range(3):
    g = torch.Generator("cpu").manual_seed(SEED + i)
    img = sdxl(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1344,
        height=768,
        num_inference_steps=28,
        guidance_scale=6.5,
        generator=g,
    ).images[0]

    filename = f"realistic_picture_{i + 1}.png"
    img.save(filename)
    print(f"Saved: {filename}")
    display(img.resize((672, 384)))

# Free memory so Step 2 has room to load the video model
del sdxl, vae
gc.collect()
torch.cuda.empty_cache()

print("Done.")
`;
}

function buildStep2(v) {
  return `!pip install -q diffusers transformers accelerate safetensors

import gc
import os
import shutil
import requests
import torch
import ipywidgets as widgets
from PIL import Image, ImageOps
from IPython.display import display
from google.colab import files, drive
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

SEED = ${v.seed}
PICK = ${v.pick}
N_CLIPS = ${v.clips}
MOTION = ${v.motion}
OUTPUT_NAME = ${py(v.outname)}
NOTIFY_TOPIC = ${py(v.ntfy)}

def notify(text):
    if not NOTIFY_TOPIC:
        return
    try:
        requests.post(f"https://ntfy.sh/{NOTIFY_TOPIC}", data=text.encode("utf-8"), headers={"Title": "Video job"}, timeout=10)
    except Exception:
        pass

# Show available images
for n in (1, 2, 3):
    path = f"realistic_picture_{n}.png"
    if os.path.exists(path):
        print(f"Picture {n}")
        display(Image.open(path).resize((672, 384)))

answer = input(f"Which picture do you want to animate? Type 1, 2 or 3, or 0 to upload your own (press Enter for {PICK}): ").strip()
if answer in ("0", "1", "2", "3"):
    PICK = int(answer)

if PICK == 0:
    uploaded = files.upload()
    picture_path = list(uploaded.keys())[0]
else:
    picture_path = f"realistic_picture_{PICK}.png"

print(f"Animating {picture_path}")
keyframe = ImageOps.fit(Image.open(picture_path).convert("RGB"), (1024, 576))

print("Loading Stable Video Diffusion...")
svd = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
    low_cpu_mem_usage=True,
)
svd.enable_model_cpu_offload()
svd.unet.enable_forward_chunking()

all_frames = []
current = keyframe

for n in range(N_CLIPS):
    print(f"Animating clip {n + 1}/{N_CLIPS}...")
    g = torch.Generator("cpu").manual_seed(SEED + n)
    out = svd(
        current,
        num_frames=25,
        decode_chunk_size=2,
        motion_bucket_id=MOTION,
        noise_aug_strength=0.02,
        fps=7,
        generator=g,
    ).frames[0]

    all_frames += out if n == 0 else out[1:]
    current = out[-1]

export_to_video(all_frames, "output_raw.mp4", fps=8)

del svd
gc.collect()
torch.cuda.empty_cache()

print("Post-processing with FFmpeg...")
!ffmpeg -y -loglevel error -i output_raw.mp4 \\
  -vf "minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:vsbmc=1,scale=1280:720:flags=lanczos,eq=contrast=1.05:saturation=1.08" \\
  -c:v libx264 -pix_fmt yuv420p -crf 18 {OUTPUT_NAME}

notify("Your video is ready. Go back to Colab.")
files.download(OUTPUT_NAME)
print(f"Done! Video saved as: {OUTPUT_NAME}")

try:
    drive.mount("/content/drive")
    save_dir = "/content/drive/MyDrive/Videos"
    os.makedirs(save_dir, exist_ok=True)

    button = widgets.Button(description="Save to Google Drive", button_style="success")
    message = widgets.Output()

    def save_video(_):
        with message:
            message.clear_output()
            shutil.copy(OUTPUT_NAME, save_dir)
            print(f"Saved to {save_dir}/{OUTPUT_NAME}")

    button.on_click(save_video)
    display(button, message)
except Exception as e:
    print("Google Drive was not connected, so there is no save button:", e)
`;
}

function buildKaggle(v) {
  return `!pip install -q diffusers transformers accelerate safetensors

import gc
import glob
import os
import subprocess
import requests
import torch
from PIL import Image, ImageOps
from diffusers import (
    StableDiffusionXLPipeline,
    StableVideoDiffusionPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)
from diffusers.utils import export_to_video

# Realism checkpoint
MODEL_ID = "SG161222/RealVisXL_V5.0"

prompt = ${py(v.prompt)}
negative_prompt = ${py(v.negative)}
SEED = ${v.seed}
PICK = ${v.pick}
N_CLIPS = ${v.clips}
MOTION = ${v.motion}
OUTPUT_NAME = ${py(v.outname)}
NOTIFY_TOPIC = ${py(v.ntfy)}

OUT_DIR = "/kaggle/working"


def notify(text):
    if not NOTIFY_TOPIC:
        return
    try:
        requests.post(f"https://ntfy.sh/{NOTIFY_TOPIC}", data=text.encode("utf-8"), headers={"Title": "Video job"}, timeout=10)
    except Exception:
        pass


try:
    if PICK == 0:
        found = []
        for ext in ("png", "jpg", "jpeg", "webp"):
            found += glob.glob(f"/kaggle/input/**/*.{ext}", recursive=True)
        if not found:
            raise RuntimeError("No picture found in /kaggle/input. Add your picture to the notebook as an input.")
        picture_path = found[0]
    else:
        print("Loading VAE and RealVisXL Pipeline...")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
        )
        sdxl = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
        sdxl.scheduler = DPMSolverMultistepScheduler.from_config(
            sdxl.scheduler.config,
            use_karras_sigmas=True,
        )
        sdxl.enable_model_cpu_offload()
        sdxl.vae.enable_slicing()
        sdxl.vae.enable_tiling()

        print("Generating realistic images...")
        for i in range(3):
            g = torch.Generator("cpu").manual_seed(SEED + i)
            img = sdxl(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=1344,
                height=768,
                num_inference_steps=28,
                guidance_scale=6.5,
                generator=g,
            ).images[0]
            img.save(f"{OUT_DIR}/realistic_picture_{i + 1}.png")
            print(f"Picture {i + 1} saved")

        picture_path = f"{OUT_DIR}/realistic_picture_{PICK}.png"

        del sdxl, vae
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Animating {picture_path}")
    keyframe = ImageOps.fit(Image.open(picture_path).convert("RGB"), (1024, 576))

    print("Loading Stable Video Diffusion...")
    svd = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
        low_cpu_mem_usage=True,
    )
    svd.enable_model_cpu_offload()
    svd.unet.enable_forward_chunking()

    all_frames = []
    current = keyframe
    for n in range(N_CLIPS):
        print(f"Animating clip {n + 1}/{N_CLIPS}...")
        g = torch.Generator("cpu").manual_seed(SEED + n)
        out = svd(
            current,
            num_frames=25,
            decode_chunk_size=2,
            motion_bucket_id=MOTION,
            noise_aug_strength=0.02,
            fps=7,
            generator=g,
        ).frames[0]
        all_frames += out if n == 0 else out[1:]
        current = out[-1]

    raw_path = f"{OUT_DIR}/output_raw.mp4"
    final_path = f"{OUT_DIR}/{OUTPUT_NAME}"
    export_to_video(all_frames, raw_path, fps=8)

    del svd
    gc.collect()
    torch.cuda.empty_cache()

    print("Post-processing with FFmpeg...")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", raw_path,
        "-vf", "minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:vsbmc=1,scale=1280:720:flags=lanczos,eq=contrast=1.05:saturation=1.08",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", final_path,
    ], check=True)
    os.remove(raw_path)

    print(f"Done. Video saved to {final_path}")
    notify("Your video is ready. Open the Kaggle notebook output to download it.")
except Exception as e:
    notify(f"The video job failed: {e}")
    raise
`;
}

$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  try {

  const values = {
    prompt: $("prompt").value,
    negative: $("negative").value,
    seed: clamp($("seed").value, 0, 2147483647, 42),
    clips: clamp($("clips").value, 1, 4, 2),
    pick: clamp($("pick").value, 0, 3, 1),
    motion: clamp($("motion").value, 30, 200, 90),
    outname: safeFileName($("outname").value),
    ntfy: $("ntfy").value.trim().replace(/[^\w-]/g, ""),
  };

  if (!values.prompt.trim()) {
    $("prompt").focus();
    return;
  }

  $("code1").value = buildStep1(values);
  $("code2").value = buildStep2(values);
  $("code3").value = buildKaggle(values);
  $("result").hidden = false;
  $("result").scrollIntoView({ behavior: "smooth" });
  $("status").hidden = true;
  } catch (err) {
    $("status").textContent = "Error: " + err.message;
    $("status").hidden = false;
  }
});

document.querySelectorAll(".copy").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const box = $(btn.dataset.target);
    const label = btn.dataset.label;
    try {
      await navigator.clipboard.writeText(box.value);
    } catch (err) {
      box.focus();
      box.select();
      document.execCommand("copy");
    }
    btn.textContent = "Copied!";
    setTimeout(() => (btn.textContent = label), 1500);
  });
});

document.querySelectorAll(".clear").forEach((btn) => {
  btn.addEventListener("click", () => {
    const box = $(btn.dataset.clear);
    box.value = "";
    box.focus();
  });
});

const savedFields = ["prompt", "negative", "seed", "pick", "clips", "motion", "outname", "ntfy"];
const storeKey = "videoPromptBuilder";

function loadSaved() {
  try {
    const data = JSON.parse(localStorage.getItem(storeKey) || "null");
    if (!data) return;
    savedFields.forEach((id) => {
      if (typeof data[id] === "string") $(id).value = data[id];
    });
  } catch (err) {}
}

$("save").addEventListener("click", () => {
  const btn = $("save");
  try {
    const data = {};
    savedFields.forEach((id) => (data[id] = $(id).value));
    localStorage.setItem(storeKey, JSON.stringify(data));
    btn.textContent = "Saved!";
  } catch (err) {
    btn.textContent = "Could not save";
  }
  setTimeout(() => (btn.textContent = "Save"), 1500);
});

loadSaved();
