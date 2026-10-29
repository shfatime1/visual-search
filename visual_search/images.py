import io
import warnings
from PIL import Image, ImageOps, UnidentifiedImageError
from .config import MAX_IMAGE_BYTES


def prepare_image(content: bytes) -> bytes:
    """Validate actual bytes; strip metadata and normalize consistently for the model."""
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise ValueError("Image must be between 1 byte and 5 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as check:
                if check.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Use a JPEG, PNG, or WebP image.")
                if check.width * check.height > 20_000_000:
                    raise ValueError("Image exceeds 20 megapixels.")
                if getattr(check, "n_frames", 1) != 1:
                    raise ValueError("Animated images are not supported.")
                check.verify()
            with Image.open(io.BytesIO(content)) as source:
                img = ImageOps.exif_transpose(source).convert("RGBA")
                img.thumbnail((1120, 1120))
                background = Image.new("RGB", img.size, "white")
                background.paste(img, mask=img.getchannel("A"))
                out = io.BytesIO()
                background.save(out, format="JPEG", quality=92)
                return out.getvalue()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Invalid or unsafe image.") from exc
