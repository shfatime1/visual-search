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


@dataclass
class Settings:
    data_dir: Path = ROOT / "data"
    artifacts: Path = ROOT / "artifacts"
    nvidia_key: str = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", ""))
    api_key: str = field(default_factory=lambda: os.getenv("VISUAL_SEARCH_API_KEY", ""))
    ui_origin: str = field(default_factory=lambda: os.getenv("UI_ORIGIN", "http://127.0.0.1:8000").rstrip("/"))
