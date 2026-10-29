"""Resumable indexing; requires a successful live test_nvidia.py receipt."""
import asyncio
import json
import sys
import numpy as np
from visual_search.catalog import load_catalog, fingerprint
from visual_search.config import Settings
from visual_search.images import prepare_image
from visual_search.nvidia import NvidiaEmbedder, RECIPE, CACHE_VERSION, cache_key, save_json, save_npy


async def main():
    settings = Settings()
    products, report = load_catalog(settings.data_dir)
    save_json(settings.artifacts / "validation_report.json", report)
    receipt_path = settings.artifacts / "nvidia_verified.json"
    if not receipt_path.exists():
        raise RuntimeError("Run python test_nvidia.py successfully before indexing.")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("recipe") != RECIPE or receipt.get("cache_version") != CACHE_VERSION:
        raise RuntimeError("Embedding configuration changed. Run python test_nvidia.py again.")
    if not products or receipt.get("image_cache_key") != cache_key(prepare_image(products[0]["_path"].read_bytes())):
        raise RuntimeError("Verification image changed. Run python test_nvidia.py again.")
    embedder, vectors = NvidiaEmbedder(settings), []
    for number, product in enumerate(products, 1):
        try:
            vector = await embedder.embed(prepare_image(product["_path"].read_bytes()))
            vectors.append(vector)
        except Exception:
            save_json(settings.artifacts / "progress.json", {"completed": number - 1, "total": len(products), "failed_id": product["id"]})
            raise
        save_json(settings.artifacts / "progress.json", {"completed": number, "total": len(products), "last_id": product["id"]})
        print(f"[{number}/{len(products)}] id={product['id']} {'cached' if embedder.cache_hit else 'embedded'}", flush=True)
    # Publish manifest last; hashes prevent accepting a partial pair after interruption.
    import hashlib
    save_npy(settings.artifacts / "embeddings.npy", np.stack(vectors))
    save_npy(settings.artifacts / "product_ids.npy", np.array([p["id"] for p in products], dtype=np.int64))
    save_json(settings.artifacts / "index.json", {"recipe": RECIPE, "cache_version": CACHE_VERSION,
        "catalog_fingerprint": fingerprint(products), "count": len(products),
        "files": {name: hashlib.sha256((settings.artifacts / name).read_bytes()).hexdigest() for name in ("embeddings.npy", "product_ids.npy")}})
    print(f"Index ready: {len(products)} products.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"Indexing stopped: {exc}", file=sys.stderr)
        sys.exit(1)
