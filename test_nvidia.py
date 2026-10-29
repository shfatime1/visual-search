"""Explicit one-image live verification. Never run automatically under pytest."""
import asyncio
import sys
from datetime import datetime, timezone
from visual_search.catalog import load_catalog
from visual_search.config import Settings
from visual_search.images import prepare_image
from visual_search.nvidia import NvidiaEmbedder, RECIPE, CACHE_VERSION, cache_key, save_json


async def main():
    settings = Settings()
    products, report = load_catalog(settings.data_dir)
    save_json(settings.artifacts / "validation_report.json", report)
    if not products:
        raise RuntimeError("No valid products")
    image = prepare_image(products[0]["_path"].read_bytes())
    vector = await NvidiaEmbedder(settings).embed(image, force=True)
    receipt = {"recipe": RECIPE, "cache_version": CACHE_VERSION, "image_cache_key": cache_key(image),
               "verified_at": datetime.now(timezone.utc).isoformat(), "product_id": products[0]["id"],
               "dimensions": len(vector)}
    save_json(settings.artifacts / "nvidia_verified.json", receipt)
    print(f"VERIFIED: product {products[0]['id']}; {len(vector)} finite embedding values. Cached for indexing.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        sys.exit(1)
