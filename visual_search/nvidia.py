import asyncio
import base64
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import uuid
import httpx
import numpy as np
from .config import DIMENSIONS, ENDPOINT, MODEL

RECIPE = {"model": MODEL, "input_type": "passage", "modality": "image", "encoding_format": "float",
          "embedding_type": "float", "dimensions": DIMENSIONS, "longest_edge": 1120, "truncate": "NONE"}
CACHE_VERSION = "jpeg92-white-exif-v1"


class NvidiaError(RuntimeError):
    def __init__(self, message, status=502):
        super().__init__(message)
        self.status = status


def validate_vector(value):
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (DIMENSIONS,) or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
        raise ValueError("Invalid embedding vector")
    return vector


def cache_key(image):
    return hashlib.sha256(json.dumps(RECIPE, sort_keys=True).encode() + CACHE_VERSION.encode() + image).hexdigest()


def save_npy(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("wb") as f:
        np.save(f, value, allow_pickle=False)
    temporary.replace(path)


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class NvidiaEmbedder:
    def __init__(self, settings, client=None):
        self.settings = settings
        self.client = client
        self.lock = asyncio.Lock()
        self.cache_hit = False

    async def embed(self, image: bytes, force=False):
        key = cache_key(image)
        path = self.settings.embedding_cache_dir / f"{key}.npy"
        async with self.lock:
            self.cache_hit = False
            if path.exists() and not force:
                try:
                    result = validate_vector(np.load(path, allow_pickle=False))
                    self.cache_hit = True
                    return result
                except (ValueError, OSError):
                    pass
            if not self.settings.nvidia_key:
                raise NvidiaError("Set NVIDIA_API_KEY in .env before embedding images.", 503)
            if self.client:
                vector = await self._request(self.client, image)
            else:
                async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=15), trust_env=False) as client:
                    vector = await self._request(client, image)
            save_npy(path, vector)
            return vector

    async def _request(self, client, image):
        payload = {**RECIPE, "input": ["data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")]}
        for attempt in range(5):
            try:
                response = await client.post(ENDPOINT, json=payload, headers={"Authorization": "Bearer " + self.settings.nvidia_key})
            except httpx.TransportError as exc:
                if attempt == 4:
                    raise NvidiaError("NVIDIA connection failed after retries.", 503) from exc
                await asyncio.sleep(min(2 ** attempt + random.random(), 30))
                continue
            if response.status_code == 200:
                try:
                    data = response.json()["data"]
                    if len(data) != 1 or data[0].get("index", 0) != 0:
                        raise ValueError("Unexpected response items")
                    return validate_vector(data[0]["embedding"])
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    raise NvidiaError("NVIDIA returned a malformed embedding.") from exc
            retryable = response.status_code in {408, 429, 500, 502, 503, 504}
            if not retryable or attempt == 4:
                status = 503 if retryable else 502
                raise NvidiaError(f"NVIDIA returned HTTP {response.status_code}. Check credentials, quota, and model access.", status)
            delay = 2 ** attempt + random.random()
            retry = response.headers.get("Retry-After")
            if retry:
                try:
                    delay = max(delay, float(retry))
                except ValueError:
                    try:
                        delay = max(delay, (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds())
                    except (ValueError, TypeError):
                        pass
            if delay > 60:
                raise NvidiaError("NVIDIA requested a long cooldown. Retry later; completed images are cached.", 503)
            await asyncio.sleep(delay)
