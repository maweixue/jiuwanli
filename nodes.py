import base64
import io
import json
import os
import time

import numpy as np
import torch
import requests
from PIL import Image

os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

SESSION = requests.Session()
SESSION.trust_env = False

BASE_URL = "https://api.bananarouter.com"

MODEL_DEFS = {
    "香蕉V2迷你": {
        "api": "gemini",
        "model_id": "gemini-3.1-flash-lite-image",
        "resolutions": ["auto", "512", "1K", "2K", "4K"],
        "qualities": [],
    },
    "香蕉V2": {
        "api": "gemini",
        "model_id": "gemini-3.1-flash-image-preview",
        "resolutions": ["auto", "512", "1K", "2K", "4K"],
        "qualities": [],
    },
    "香蕉pro": {
        "api": "gemini",
        "model_id": "gemini-3-pro-image",
        "resolutions": ["auto", "1K", "2K", "4K"],
        "qualities": [],
    },
    "gpt2.0": {
        "api": "gpt",
        "model_id": "gpt-image-2",
        "resolutions": ["auto", "1K", "2K", "4K"],
        "qualities": ["auto", "low", "medium", "high"],
    },
    "gpt2.5迷你": {
        "api": "gpt",
        "model_id": "gpt-image-2.5-sunburst",
        "resolutions": ["auto", "1K", "2K", "4K"],
        "qualities": ["auto", "low", "medium", "high", "xhigh", "max"],
    },
    "gpt2.5精细": {
        "api": "gpt",
        "model_id": "gpt-image-2.5-flare",
        "resolutions": ["auto", "1K", "2K", "4K"],
        "qualities": ["auto", "low", "medium", "high", "xhigh", "max"],
    },
}

ASPECT_OPTIONS = ["自动", "1:1", "16:9", "9:16", "3:2", "2:3",
                  "4:3", "3:4", "4:5", "5:4", "21:9"]

GPT_SIZE_LANDSCAPE = {"1K": "1536x1024", "2K": "2048x1152", "4K": "3840x2160"}
GPT_SIZE_PORTRAIT = {"1K": "1024x1536", "2K": "1152x2048", "4K": "2160x3840"}
GPT_SIZE_SQUARE = {"1K": "1024x1024", "2K": "2048x2048", "4K": "2048x2048"}

EMPTY_IMG = torch.zeros([1, 64, 64, 3], dtype=torch.float32)
POLL_INTERVAL = 5
POLL_MAX_WAIT = 600


def _ratio_of(r):
    a, b = r.split(":")
    return int(a) / int(b)


def _closest_aspect(w, h):
    target = w / h
    return min(ASPECT_OPTIONS[1:], key=lambda r: abs(_ratio_of(r) - target))


def _gpt_size(aspect_ratio, res):
    if res == "auto":
        return "auto"
    r = _ratio_of(aspect_ratio)
    if r > 1.2:
        return GPT_SIZE_LANDSCAPE.get(res, "1024x1024")
    if r < 0.83:
        return GPT_SIZE_PORTRAIT.get(res, "1024x1024")
    return GPT_SIZE_SQUARE.get(res, "1024x1024")


def _tensor_to_png_bytes(img_tensor):
    arr = img_tensor[0].cpu().float().numpy()
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _tensor_to_data_url(img_tensor):
    b64 = base64.b64encode(_tensor_to_png_bytes(img_tensor)).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _tensor_to_b64(img_tensor):
    return base64.b64encode(_tensor_to_png_bytes(img_tensor)).decode("utf-8")


def _pil_to_tensor(pil_img):
    arr = np.array(pil_img.convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _poll_task(task_id, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = time.time() + POLL_MAX_WAIT
    while time.time() < deadline:
        try:
            resp = SESSION.get(f"{BASE_URL}/v1/async-tasks/{task_id}",
                                headers=headers, timeout=5)
            data = resp.json()
        except Exception:
            time.sleep(POLL_INTERVAL)
            continue
        status = data.get("status")
        if status == "success":
            return data
        if status in ("failed", "expired", "cancelled", "canceled"):
            return data
        time.sleep(POLL_INTERVAL)
    return {"status": "poll_timeout", "statusMessage": "轮询超时"}


def _cancel_task(task_id, api_key):
    try:
        SESSION.delete(f"{BASE_URL}/v1/async-tasks/{task_id}",
                       headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
    except Exception:
        pass


def _download_to_tensor(url):
    last_err = None
    for attempt in range(5):
        try:
            resp = SESSION.get(url, timeout=30, stream=True)
            chunks = []
            for chunk in resp.iter_content(chunk_size=65536):
                chunks.append(chunk)
            resp.close()
            data = b"".join(chunks)
            pil = Image.open(io.BytesIO(data))
            return _pil_to_tensor(pil)
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise last_err


def _run_gemini_async(cfg, prompt, aspect_ratio, seed, resolution, api_key, images):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    parts = []
    for im in images:
        parts.append({"inlineData": {"mimeType": "image/png", "data": _tensor_to_b64(im)}})
    parts.append({"text": prompt})

    image_config = {"aspectRatio": aspect_ratio}
    if resolution and resolution != "auto":
        image_config["imageSize"] = resolution

    gen_config = {"responseModalities": ["IMAGE"], "imageConfig": image_config}
    if seed and seed > 0:
        gen_config["seed"] = seed

    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": gen_config,
    }
    url = f"{BASE_URL}/v1beta/models/{cfg['model_id']}:asyncGenerateContent"
    resp = SESSION.post(url, headers=headers, json=body)
    data = resp.json()
    if resp.status_code not in (200, 202) or "taskID" not in data:
        return (EMPTY_IMG, "", resp.text)
    task_id = data["taskID"]
    try:
        result = _poll_task(task_id, api_key)
    except BaseException:
        _cancel_task(task_id, api_key)
        raise
    return _finalize(result)


def _run_gpt_async(cfg, is_img2img, prompt, aspect_ratio, seed, resolution, quality, api_key, images):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    size = _gpt_size(aspect_ratio, resolution)

    if is_img2img and images:
        url = f"{BASE_URL}/v1/images/edits/async"
        body = {
            "model": cfg["model_id"], "prompt": prompt,
            "images": [_tensor_to_data_url(im) for im in images],
            "size": size, "quality": quality, "output_format": "png",
        }
    else:
        url = f"{BASE_URL}/v1/images/generations/async"
        body = {
            "model": cfg["model_id"], "prompt": prompt, "n": 1,
            "size": size, "quality": quality, "output_format": "png",
        }

    resp = SESSION.post(url, headers=headers, json=body)
    data = resp.json()
    if resp.status_code not in (200, 202) or "taskID" not in data:
        return (EMPTY_IMG, "", resp.text)
    task_id = data["taskID"]
    try:
        result = _poll_task(task_id, api_key)
    except BaseException:
        _cancel_task(task_id, api_key)
        raise
    return _finalize(result)


def _finalize(result):
    if result.get("status") != "success":
        msg = result.get("statusMessage") or result.get("error") or json.dumps(result, ensure_ascii=False)
        return (EMPTY_IMG, "", str(msg))
    images_meta = result.get("resultImages") or []
    if not images_meta:
        return (EMPTY_IMG, "", json.dumps(result, ensure_ascii=False))
    out_url = images_meta[0].get("url", "")
    try:
        img = _download_to_tensor(out_url)
    except Exception as e:
        return (EMPTY_IMG, out_url, str(e))
    return (img, out_url, json.dumps(result, ensure_ascii=False))


ON_ERROR_OPTIONS = ["报错时停止工作流", "报错时继续运行"]


def _run_image(cfg, is_img2img, prompt, aspect_ratio, seed, resolution, quality, api_key, on_error, **kw):
    images = []
    if is_img2img:
        for i in range(1, 11):
            im = kw.get(f"image_{i}")
            if im is not None:
                images.append(im)

    if aspect_ratio == "自动":
        if images:
            h, w = images[0].shape[1], images[0].shape[2]
            aspect_ratio = _closest_aspect(int(w), int(h))
        else:
            aspect_ratio = "1:1"

    skip = on_error == "报错时继续运行"
    try:
        if cfg["api"] == "gemini":
            return _run_gemini_async(cfg, prompt, aspect_ratio, seed, resolution, api_key, images)
        return _run_gpt_async(cfg, is_img2img, prompt, aspect_ratio, seed,
                              resolution, quality, api_key, images)
    except Exception as e:
        if skip:
            return (EMPTY_IMG, "", str(e))
        raise


def _make_image_node(label, cfg, is_img2img):
    api = cfg["api"]

    def INPUT_TYPES(cls):
        req = {
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "aspect_ratio": (ASPECT_OPTIONS, {"default": "自动"}),
            "resolution": (cfg["resolutions"], {"default": "auto"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            "api_key": ("STRING", {"default": ""}),
            "on_error": (ON_ERROR_OPTIONS, {"default": "报错时停止工作流"}),
        }
        if api == "gpt":
            req["quality"] = (cfg["qualities"], {"default": "auto"})

        optional = {}
        if is_img2img:
            for i in range(1, 11):
                optional[f"image_{i}"] = ("IMAGE",)

        return {"required": req, "optional": optional}

    def generate(self, prompt, aspect_ratio, resolution, seed, api_key, on_error, quality="auto", **kw):
        return _run_image(cfg, is_img2img, prompt, aspect_ratio, seed,
                          resolution, quality, api_key, on_error, **kw)

    cls_name = "jwl_" + label.replace(" ", "") + ("_I2I" if is_img2img else "_T2I")
    return type(cls_name, (), {
        "INPUT_TYPES": classmethod(INPUT_TYPES),
        "RETURN_TYPES": ("IMAGE", "STRING", "STRING"),
        "RETURN_NAMES": ("image", "url", "response"),
        "FUNCTION": "generate",
        "CATEGORY": "九万里",
        "generate": generate,
    })


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for name, cfg in MODEL_DEFS.items():
    for is_img2img in (False, True):
        cls = _make_image_node(name, cfg, is_img2img)
        NODE_CLASS_MAPPINGS[cls.__name__] = cls
        NODE_DISPLAY_NAME_MAPPINGS[cls.__name__] = f"九万里 {name} {'图生图' if is_img2img else '文生图'}"
