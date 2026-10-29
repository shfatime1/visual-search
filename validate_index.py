"""Offline deployment gate. Validates existing arrays; never requests embeddings."""
import json
from visual_search.catalog import load_catalog
from visual_search.config import Settings
from visual_search.engine import SearchEngine


def validate(settings=None, expected_count=194):
    settings = settings or Settings()
    products, report = load_catalog(settings.data_dir)
    if report["invalid_rows"] or len(products) != expected_count:
        raise ValueError(f"Expected {expected_count} valid original products, found {len(products)}")
    engine = SearchEngine.load(settings, products)
    return {"indexed_products": len(engine.products), "dimensions": engine.embeddings.shape[1],
            "data_dir": str(settings.data_dir), "index_dir": str(settings.artifacts),
            "cache_dir": str(settings.embedding_cache_dir), "nvidia_calls": 0}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
