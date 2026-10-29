import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
MODEL = "nvidia/nemotron-3.5-embed-1b"
ENDPOINT = "https://integrate.api.nvidia.com/v1/embeddings"
DIMENSIONS = 2048
MAX_IMAGE_BYTES = 5 * 1024 * 1024


def env_bool(name, default=False):
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes"}


def default_origin():
    return (os.getenv("UI_ORIGIN") or os.getenv("RENDER_EXTERNAL_URL") or "http://127.0.0.1:8000").rstrip("/")


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("VISUAL_SEARCH_DATA_DIR", str(ROOT / "data"))).resolve())
    artifacts: Path = field(default_factory=lambda: Path(os.getenv("VISUAL_SEARCH_INDEX_DIR", str(ROOT / "artifacts"))).resolve())
    cache_dir: Path | None = field(default_factory=lambda: Path(os.environ["VISUAL_SEARCH_CACHE_DIR"]).resolve() if os.getenv("VISUAL_SEARCH_CACHE_DIR") else None)
    nvidia_key: str = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", ""))
    api_key: str = field(default_factory=lambda: os.getenv("VISUAL_SEARCH_API_KEY", ""))
    ui_origin: str = field(default_factory=default_origin)
    public_ui: bool = field(default_factory=lambda: env_bool("PUBLIC_UI_ENABLED", env_bool("RENDER")))

    @property
    def embedding_cache_dir(self):
        return self.cache_dir if self.cache_dir is not None else self.artifacts / "cache"
