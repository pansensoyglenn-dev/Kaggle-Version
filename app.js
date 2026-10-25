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
  if (!cleaned) return "output_realistic.mp4";
  return cleaned.toLowerCase().endsWith(".mp4") ? cleaned : cleaned + ".mp4";
}

function buildStep1(v) {
  return `!pip install -q diffusers transformers accelerate safetensors

import gc
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, AutoencoderKL
from IPython.display import display

assert torch.cuda.is_available(), "No GPU detected. In Colab: Runtime > Change runtime type > T4 GPU."
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB)")
if vram_gb < 14:
    print("WARNING: less than 14 GB VRAM. Step 1 may be slow or OOM. A T4 (16 GB) is recommended.")

MODEL_ID = "SG161222/RealVisXL_V5.0"

prompt = ${py(v.prompt)}
negative_prompt = ${py(v.negative)}
SEED = ${v.seed}

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

    filename = f"realistic_picture_{i + 1}.png"
    img.save(filename)
    print(f"Saved: {filename}")
    display(img.resize((672, 384)))

del sdxl, vae
gc.collect()
torch.cuda.empty_cache()

print("Done.")
print("RECOMMENDED: before running Step 2, do Runtime > Restart runtime, then re-run Step 1's")
print("pip/import lines only if Step 2 complains about memory. If Step 2 OOMs, restart and jump")
print("straight to Step 2 — the PNGs are already saved.")
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

assert torch.cuda.is_available(), "No GPU detected. In Colab: Runtime > Change runtime type > T4 GPU."
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB)")

SEED = ${v.seed}
PICK = ${v.pick}
N_CLIPS = min(${v.clips}, 2)
MOTION = ${v.motion}
OUTPUT_NAME = ${py(v.outname)}
NOTIFY_TOPIC = ${py(v.ntfy)}

if ${v.clips} > 2:
    print(f"N_CLIPS was ${v.clips}; capped to 2. Clips 3+ drift heavily and are usually unusable.")

def notify(text):
    if not NOTIFY_TOPIC:
        return
    try:
        requests.post(f"https://ntfy.sh/{NOTIFY_TOPIC}", data=text.encode("utf-8"), headers={"Title": "Video job"}, timeout=10)
    except Exception:
        pass

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
    if not os.path.exists(picture_path):
        raise FileNotFoundError(f"{picture_path} not found. Please run Step 1 first, or enter 0 to upload.")

print(f"Animating {picture_path}")
keyframe = ImageOps.fit(Image.open(picture_path).convert("RGB"), (1024, 576))

print("Loading Stable Video Diffusion...")
try:
    svd = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
        low_cpu_mem_usage=True,
    )
except torch.cuda.OutOfMemoryError:
    raise RuntimeError(
        "Out of memory loading SVD. If you just ran Step 1, do Runtime > Restart runtime "
        "and then re-run Step 2 alone. The PNGs are already on disk."
    )
svd.enable_model_cpu_offload()
svd.unet.enable_forward_chunking()

all_frames = []
current = keyframe

for n in range(N_CLIPS):
    print(f"Animating clip {n + 1}/{N_CLIPS}...")
    g = torch.Generator("cpu").manual_seed(SEED + n)
    try:
        out = svd(
            current,
            num_frames=25,
            decode_chunk_size=2,
            motion_bucket_id=MOTION,
            noise_aug_strength=0.05,
            fps=7,
            generator=g,
        ).frames[0]
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"OOM on clip {n + 1}. Try Runtime > Restart runtime, then run Step 2 alone, "
            "or lower MOTION / motion_bucket_id."
        )
    all_frames += out if n == 0 else out[1:]
    current = out[-1]

export_to_video(all_frames, "output_raw.mp4", fps=8)

del svd
gc.collect()
torch.cuda.empty_cache()

print("Post-processing with FFmpeg (fast framerate doubling)...")
vf = "framerate=fps=24,scale=1280:720:flags=lanczos,unsharp=5:5:0.6:5:5:0.3,eq=contrast=1.04:saturation=1.06"
cmd = [
    "ffmpeg", "-y", "-loglevel", "error", "-i", "output_raw.mp4",
    "-vf", vf,
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
    OUTPUT_NAME,
]
try:
    import subprocess
    subprocess.run(cmd, check=True)
    final_file = OUTPUT_NAME
except Exception as e:
    print(f"FFmpeg post-processing failed ({e}); falling back to raw video.")
    final_file = "output_raw.mp4"

notify(f"Your video is ready: {final_file}. Go back to Colab.")
files.download(final_file)
print(f"Done! Video saved as: {final_file}")

try:
    drive.mount("/content/drive")
    save_dir = "/content/drive/MyDrive/Videos"
    os.makedirs(save_dir, exist_ok=True)

    button = widgets.Button(description="Save to Google Drive", button_style="success")
    message = widgets.Output()

    def save_video(_):
        with message:
            message.clear_output()
            shutil.copy(final_file, save_dir)
            print(f"Saved to {save_dir}/{final_file}")

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

assert torch.cuda.is_available(), "No GPU detected. In Kaggle: Settings > Accelerator > GPU T4 x2."
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB)")

MODEL_ID = "SG161222/RealVisXL_V5.0"

prompt = ${py(v.prompt)}
negative_prompt = ${py(v.negative)}
SEED = ${v.seed}
PICK = ${v.pick}
N_CLIPS = min(${v.clips}, 2)
MOTION = ${v.motion}
OUTPUT_NAME = ${py(v.outname)}
NOTIFY_TOPIC = ${py(v.ntfy)}

USE_UPLOADED_IMAGE = (PICK == 0)
OUT_DIR = "/kaggle/working"

if ${v.clips} > 2:
    print(f"N_CLIPS was ${v.clips}; capped to 2. Clips 3+ drift heavily and are usually unusable.")


def notify(text):
    if not NOTIFY_TOPIC:
        return
    try:
        requests.post(f"https://ntfy.sh/{NOTIFY_TOPIC}", data=text.encode("utf-8"), headers={"Title": "Video job"}, timeout=10)
    except Exception:
        pass


try:
    if USE_UPLOADED_IMAGE:
        found = []
        for ext in ("png", "jpg", "jpeg", "webp"):
            found += glob.glob(f"/kaggle/input/**/*.{ext}", recursive=True)
        if not found:
            raise RuntimeError(
                "No picture found in /kaggle/input. Add your picture via 'Add Data' "
                "on the right sidebar, then re-run this cell."
            )
        picture_path = found[0]
        print(f"Using uploaded image: {picture_path}")
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
    try:
        svd = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16",
            low_cpu_mem_usage=True,
        )
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError(
            "OOM loading SVD. If the image-gen block just ran, stop this session and "
            "start a fresh Kaggle session with USE_UPLOADED_IMAGE = True."
        )
    svd.enable_model_cpu_offload()
    svd.unet.enable_forward_chunking()

    all_frames = []
    current = keyframe
    for n in range(N_CLIPS):
        print(f"Animating clip {n + 1}/{N_CLIPS}...")
        g = torch.Generator("cpu").manual_seed(SEED + n)
        try:
            out = svd(
                current,
                num_frames=25,
                decode_chunk_size=2,
                motion_bucket_id=MOTION,
                noise_aug_strength=0.05,
                fps=7,
                generator=g,
            ).frames[0]
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"OOM on clip {n + 1}. In Kaggle, try a fresh session or lower MOTION."
            )
        all_frames += out if n == 0 else out[1:]
        current = out[-1]

    raw_path = f"{OUT_DIR}/output_raw.mp4"
    final_path = f"{OUT_DIR}/{OUTPUT_NAME}"
    export_to_video(all_frames, raw_path, fps=8)

    del svd
    gc.collect()
    torch.cuda.empty_cache()

    print("Post-processing with FFmpeg (fast framerate doubling)...")
    vf = "framerate=fps=24,scale=1280:720:flags=lanczos,unsharp=5:5:0.6:5:5:0.3,eq=contrast=1.04:saturation=1.06"
    try:
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", raw_path,
            "-vf", vf,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
            final_path,
        ], check=True)
        os.remove(raw_path)
        final_msg = f"Video saved to {final_path}"
    except Exception as e:
        print(f"FFmpeg post-processing failed ({e}); keeping raw video.")
        final_msg = f"Raw video saved to {raw_path}"

    print(f"Done. {final_msg}")
    notify(f"Your video is ready. {final_msg}. Open the Kaggle notebook output to download it.")
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
    clips: clamp($("clips").value, 1, 2, 2),
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
