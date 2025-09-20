from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, PlainTextResponse
from typing import Optional, Callable
import logging
import asyncio
import gc
import threading
from contextlib import asynccontextmanager
import os
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
remove_bg_fn: Optional[Callable] = None
model_loaded = False
model_loading = False
model_error = None
model_download_progress = None

# Path to your Git LFS model
MODEL_PATH = "backend/models/u2net_human_seg.pth"

def _lazy_import():
    global remove_bg_fn, model_loaded, model_loading, model_error, model_download_progress
    if remove_bg_fn is None and not model_loading:
        model_loading = True
        model_download_progress = "Initializing lightweight model..."
        try:
            # Ensure the model file exists
            if not os.path.exists(MODEL_PATH):
                model_download_progress = "Model file missing, running 'git lfs pull'..."
                logger.info("Model file not found. Pulling from Git LFS...")
                try:
                    subprocess.run(["git", "lfs", "pull"], check=True)
                except Exception as lfs_err:
                    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Failed to pull via Git LFS: {lfs_err}")

            logger.info("Loading U2NetP model via rembg...")
            model_download_progress = "Importing backend..."
            from backend.u2net_backend import remove_bg_bytes as _remove_bg_bytes, get_session
            remove_bg_fn = _remove_bg_bytes

            # Warm up rembg session
            model_download_progress = "Warming up model session..."
            try:
                get_session("u2netp")
            except Exception as warm_err:
                logger.warning(f"Warmup failed (will retry on first request): {warm_err}")

            model_loaded = True
            model_error = None
            model_download_progress = "Complete"
            logger.info("U2NetP model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            model_error = str(e)
            model_loaded = False
            model_download_progress = f"Error: {str(e)}"
        finally:
            model_loading = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting background lightweight model loading...")
    threading.Thread(target=_lazy_import, daemon=True).start()
    yield
    logger.info("Application shutdown")

app = FastAPI(
    title="Remove-IT (U2NetP)",
    version="1.0.0",
    description="Background removal API using lightweight U2NetP model",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Remove-IT Background Removal API", "status": "running"}

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"

@app.get("/model-status")
def model_status():
    if model_loaded:
        return {"status": "loaded", "ready": True, "progress": "Complete"}
    elif model_loading:
        return {"status": "loading", "ready": False, "progress": model_download_progress or "Loading..."}
    elif model_error:
        return {"status": "error", "ready": False, "error": model_error, "progress": model_download_progress}
    else:
        return {"status": "not_started", "ready": False, "progress": "Not started"}

@app.post("/download-model")
async def download_model():
    global model_loading, model_loaded, model_error
    
    if model_loaded:
        return {"status": "already_loaded", "message": "Model is already loaded"}
    
    if model_loading:
        return {"status": "loading", "message": "Model is already being loaded"}
    
    threading.Thread(target=_lazy_import, daemon=True).start()
    return {"status": "started", "message": "Model download started. Check /model-status for progress"}

@app.post("/remove")
async def remove_endpoint(
    file: UploadFile = File(...),
    model: str = "u2netp",
    feather: int = 2,
    erode: int = 1,
):
    logger.info(f"Processing file: {file.filename}, content_type: {file.content_type}")

    if model_loading:
        raise HTTPException(503, f"Model is still loading: {model_download_progress}. Please wait.")
    if model_error:
        raise HTTPException(503, f"Model failed to load: {model_error}")
    if not model_loaded or remove_bg_fn is None:
        logger.info("Model not loaded, attempting to load now...")
        if not model_loading:
            threading.Thread(target=_lazy_import, daemon=True).start()
        raise HTTPException(503, "Model not ready. Please retry after /download-model completes.")

    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Send a PNG/JPEG/WEBP image.")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large. Maximum size is 10MB.")

    loop = asyncio.get_event_loop()

    def process_image():
        return remove_bg_fn(
            contents,
            model="u2netp",  # force lightweight model
            feather=max(0, min(int(feather), 6)),
            erode=max(0, min(int(erode), 3)),
        )

    try:
        png_bytes = await asyncio.wait_for(
            loop.run_in_executor(None, process_image),
            timeout=300.0
        )
        logger.info(f"Successfully processed image, output size: {len(png_bytes)} bytes")
        return Response(content=png_bytes, media_type="image/png")

    except asyncio.TimeoutError:
        logger.error("Image processing timed out")
        raise HTTPException(504, "Image processing timed out. Try smaller image.")
    except Exception as e:
        logger.error(f"Background removal failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Background removal failed: {str(e)}")
    finally:
        gc.collect()
