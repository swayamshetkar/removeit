import os
from io import BytesIO
from typing import Literal, Tuple
import numpy as np
from PIL import Image
import cv2
from rembg import remove, new_session

ModelName = Literal["u2net", "u2netp", "u2net_human_seg"]

# Cache sessions (avoid reloading weights per request)
_sessions = {}

def get_session(model: ModelName) -> object:
    if model not in _sessions:
        _sessions[model] = new_session(model)
    return _sessions[model]

def _gauss(alpha: np.ndarray, sigma: int) -> np.ndarray:
    return cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)

def _defringe(alpha: np.ndarray, erode_px: int) -> np.ndarray:
    if erode_px <= 0:
        return alpha
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(alpha, kernel, iterations=erode_px)

def remove_bg_bytes(
    img_bytes: bytes,
    model: ModelName = "u2net_human_seg",
    feather: int = 2,
    erode: int = 1,
    max_side: int = 3000
) -> bytes:
    """Returns PNG bytes with RGBA (alpha matte)."""
    im = Image.open(BytesIO(img_bytes)).convert("RGBA")

    # Safety: very large photos → downscale to keep memory sane on Render free tier
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)

    # Run rembg
    session = get_session(model)
    cut = remove(im, session=session)
    if isinstance(cut, (bytes, bytearray)):
        cut = Image.open(BytesIO(cut)).convert("RGBA")

    rgba = np.array(cut)
    a = rgba[:, :, 3]

    # Edge cleanup
    if erode > 0:
        a = _defringe(a, erode_px=erode)
    if feather > 0:
        a = _gauss(a, sigma=feather)

    rgba[:, :, 3] = a.clip(0, 255).astype(np.uint8)

    out = Image.fromarray(rgba, "RGBA")
    buff = BytesIO()
    out.save(buff, format="PNG")  # keep alpha
    buff.seek(0)
    return buff.read()
