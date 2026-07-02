"""
FastAPI backend — serves the frontend and exposes a /api/predict endpoint.

Usage:
    python app.py
    # then open http://localhost:8000 in your browser
"""
import io
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
import timm
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from config import (
    CLASSES, CONFIG_PATH, DEVICE, IMAGE_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, MODEL_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_val_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
ALLOWED_MIME  = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif"}

# ── Model singleton ────────────────────────────────────────────────────────────

class _ModelHolder:
    model = None
    meta  = None
    ready = False


def _load_model():
    if not MODEL_PATH.exists():
        log.warning("Model file not found at %s — run python train.py first", MODEL_PATH)
        return

    try:
        meta = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
        model_name = meta.get("model_name", "efficientnet_b2")

        from train import SnakeClassifier
        model = SnakeClassifier(model_name=model_name).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()

        _ModelHolder.model = model
        _ModelHolder.meta  = meta
        _ModelHolder.ready = True
        log.info("Model loaded: %s  |  test_acc: %s%%  |  device: %s",
                 model_name, meta.get("test_accuracy", "?"), DEVICE)
    except Exception as exc:
        log.error("Failed to load model: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Snake Vision API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

STATIC_DIR   = Path("frontend/static")
FRONTEND_DIR = Path("frontend")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "Snake Vision API is running. POST an image to /api/predict"})


@app.get("/api/health")
async def health():
    return {
        "status": "ok" if _ModelHolder.ready else "model_not_loaded",
        "device": str(DEVICE),
        "model": _ModelHolder.meta.get("model_name") if _ModelHolder.meta else None,
        "test_accuracy": _ModelHolder.meta.get("test_accuracy") if _ModelHolder.meta else None,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    # ── Validate ──────────────────────────────────────────────────────────────
    if not _ModelHolder.ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first: python train.py"
        )

    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_MIME and not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {content_type}")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image too large (max 20 MB)")

    # ── Decode image ──────────────────────────────────────────────────────────
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, Exception) as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {exc}")

    # ── Inference ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    tensor = _val_tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = _ModelHolder.model(tensor).squeeze()
        venomous_prob = torch.sigmoid(logit).item()

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    label      = "venomous" if venomous_prob > 0.5 else "non_venomous"
    confidence = venomous_prob if label == "venomous" else 1.0 - venomous_prob

    return {
        "label":               label,
        "display":             "Venomous" if label == "venomous" else "Non-Venomous",
        "confidence":          round(confidence * 100, 1),
        "venomous_probability": round(venomous_prob * 100, 1),
        "safe_probability":     round((1 - venomous_prob) * 100, 1),
        "inference_ms":        elapsed_ms,
        "warning": (
            "DANGER: This snake may be venomous. Keep distance and seek expert help."
            if label == "venomous"
            else "This snake appears to be non-venomous, but always exercise caution."
        ),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
