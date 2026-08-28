"""
Image utilities for tool-returned images.

Handles:
- Loading local image files referenced by tool outputs (e.g. browser_screenshot)
- Extracting image file paths from text like "Screenshot saved: /abs/path.png"
"""
import base64
import mimetypes
import os
import re
import logging

logger = logging.getLogger(__name__)

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MiB per image
MAX_IMAGES_PER_TOOL = 5

# Regex for absolute image file paths starting with /  (e.g. /home/.../shot.png)
_IMAGE_PATH_RE = re.compile(
    r'(/[^\s"\'`<>]+\.(?:png|jpg|jpeg|webp|gif|bmp))',
    re.IGNORECASE,
)


def _mime_for_ext(ext: str) -> str:
    ext = ext.lower()
    if ext in MIME_MAP:
        return MIME_MAP[ext]
    guessed, _ = mimetypes.guess_type(f"file{ext}")
    return guessed or "image/png"


def load_image_as_base64(path: str):
    """Load an image file and return dict with base64 data.

    Returns None if file doesn't exist, extension not allowed, too large, or read fails.
    Returned dict: {"base64": str, "mime_type": str, "source": str}
    """
    if not path or not isinstance(path, str):
        return None
    # Strip surrounding quotes/brackets that regex might include
    path = path.strip().strip('"\'`').strip()
    # Remove trailing punctuation like ) , .
    path = path.rstrip('.,;)')
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXTS:
        return None
    if not os.path.isfile(path):
        return None
    try:
        size = os.path.getsize(path)
        if size > MAX_IMAGE_BYTES:
            logger.warning(f"[IMAGE] Skipping {path}: {size} bytes > {MAX_IMAGE_BYTES}")
            return None
        if size == 0:
            return None
        with open(path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("ascii")
        mime = _mime_for_ext(ext)
        return {"base64": b64, "mime_type": mime, "source": path}
    except Exception as e:
        logger.warning(f"[IMAGE] Failed to load {path}: {e}")
        return None


def extract_image_paths(text: str):
    """Extract absolute image file paths from text."""
    if not text or not isinstance(text, str):
        return []
    matches = _IMAGE_PATH_RE.findall(text)
    # Deduplicate preserving order
    seen = set()
    out = []
    for m in matches:
        # Clean trailing chars again
        cleaned = m.strip().rstrip('.,;)')
        if cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def collect_images_from_text(text: str, limit: int = MAX_IMAGES_PER_TOOL):
    """Load images referenced by file paths in text.

    Returns list of image dicts, capped at limit.
    """
    paths = extract_image_paths(text)
    images = []
    for p in paths:
        if len(images) >= limit:
            break
        img = load_image_as_base64(p)
        if img:
            images.append(img)
        else:
            # Debug: path mentioned but not loadable
            logger.debug(f"[IMAGE] Path mentioned but not loaded: {p}")
    return images


def collect_images_from_result(result: dict, limit: int = MAX_IMAGES_PER_TOOL):
    """Scan a parsed tool result dict for image file paths in text content.

    The result is expected to have 'content': [{"type":"text","text":...}, ...]
    Returns list of loaded images.
    """
    if not isinstance(result, dict):
        return []
    # Already has images from MCP direct ImageContent? Preserve them but also scan text.
    # This helper only handles file-path images (screenshot case).
    texts = []
    for item in result.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
            texts.append(item["text"])
    # Also check stringified result fallback
    if not texts and result.get("content") and isinstance(result["content"], str):
        texts.append(result["content"])
    combined = "\n".join(texts)
    if not combined:
        # Fallback: try json dump of result
        try:
            import json as _json
            combined = _json.dumps(result, default=str)
        except Exception:
            pass
    return collect_images_from_text(combined, limit=limit)
