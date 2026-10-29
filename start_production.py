"""Fail early on broken deployment artifacts, then run one production ASGI worker."""
import os
import uvicorn
from validate_index import validate
from visual_search.config import Settings


if __name__ == "__main__":
    settings = Settings()
    summary = validate(settings)
    if not settings.nvidia_key or not settings.api_key:
        raise SystemExit("Set NVIDIA_API_KEY and VISUAL_SEARCH_API_KEY in the hosting service's environment. Do not commit .env.")
    print(f"Validated {summary['indexed_products']} bundled embeddings; no NVIDIA indexing calls.", flush=True)
    # One worker keeps the per-process rate limiter shared by all requests.
    # Render's reverse proxy is the only public entry point. Never trust forwarded
    # client addresses when deploying this command directly on an exposed host.
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")),
                proxy_headers=os.getenv("RENDER") == "true",
                forwarded_allow_ips="*" if os.getenv("RENDER") == "true" else "127.0.0.1")
