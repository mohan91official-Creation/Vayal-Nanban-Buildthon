"""Validation helpers for farmer-supplied crop photographs."""

from __future__ import annotations

from io import BytesIO
import warnings

from PIL import Image, UnidentifiedImageError


ALLOWED_IMAGE_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}
MAX_IMAGE_PIXELS = 40_000_000
MAX_DISPLAY_NAME_LENGTH = 120


def sanitize_filename(name: str) -> str:
    """Return a short display-only filename without path or control characters."""
    leaf = str(name or "").replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = "".join(character for character in leaf if character.isprintable()).strip()
    cleaned = cleaned[:MAX_DISPLAY_NAME_LENGTH]
    return cleaned if cleaned not in {"", ".", ".."} else "crop-photo"


def prepare_image_upload(name: str, data: bytes) -> dict[str, object]:
    """Verify image bytes and return safe metadata for display and model input."""
    if not data:
        raise ValueError("The uploaded image is empty.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                image_format = str(image.format or "").upper()
                width, height = image.size
                if image_format not in ALLOWED_IMAGE_FORMATS:
                    raise ValueError("Unsupported image format.")
                if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image dimensions are not supported.")
                image.verify()
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError("The uploaded file is not a safe, readable image.") from exc

    return {
        "name": sanitize_filename(name),
        "mime": ALLOWED_IMAGE_FORMATS[image_format],
        "data": data,
    }
