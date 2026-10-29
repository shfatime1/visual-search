"""Exercise the running server and real index; optional live public-image URL check."""
import argparse
import json
from datetime import datetime, timezone
import httpx
from visual_search.catalog import load_catalog
from visual_search.config import Settings
from visual_search.engine import SearchEngine
from visual_search.nvidia import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Optional public image URL for a live download + NVIDIA search")
    args = parser.parse_args()
    settings = Settings()
    products, _ = load_catalog(settings.data_dir)
    engine = SearchEngine.load(settings, products)
    for index, product in enumerate(engine.products):
        results = engine.search(engine.embeddings[index], len(products))
        own = next(row for row in results if row["id"] == product["id"])
        assert abs(own["similarity_score"] - 1) < 1e-6
        assert abs(results[0]["similarity_score"] - 1) < 1e-6
    report = {"verified_at": datetime.now(timezone.utc).isoformat(), "self_matches": len(products), "api_checks": []}
    with httpx.Client(base_url=settings.ui_origin, timeout=180, trust_env=False) as client:
        health = client.get("/health").raise_for_status().json()
        assert health["status"] == "ok" and health["indexed_products"] == len(products), health
        for path in ("/", "/docs", "/openapi.json", "/static/app.js", "/static/style.css", "/products"):
            client.get(path).raise_for_status()
        report["api_checks"].append("health, catalog, UI assets, Swagger")
        headers = {"X-API-Key": settings.api_key}
        for product in [products[0], products[86], products[173]]:
            response = client.post("/search", headers=headers,
                files={"file": (product["_path"].name, product["_path"].read_bytes(), "image/jpeg")})
            results = response.raise_for_status().json()
            assert len(results) == 5 and results[0]["id"] == product["id"]
            assert abs(results[0]["similarity_score"] - 1) < 1e-6
            client.get(results[0]["image_url"]).raise_for_status()
        report["api_checks"].append("3 real uploads: default Top-5, exact self-match, image serving")
        assert client.post("/search/url", json={"image_url": "https://example.com/a.jpg"}).status_code == 401
        assert client.post("/search/url", headers=headers, json={"image_url": "http://127.0.0.1/a.jpg"}).status_code == 400
        assert client.post("/search/url", headers=headers, json={"image_url": "https://example.com/a.jpg", "top_k": len(products) + 1}).status_code == 422
        report["api_checks"].append("unauthenticated API, private URL, excessive Top-K rejected")
        token = client.get("/ui/session").raise_for_status().json()["csrf_token"]
        response = client.post("/ui/search", headers={"Origin": settings.ui_origin, "X-CSRF-Token": token},
            files={"file": ("image.jpg", products[0]["_path"].read_bytes(), "image/jpeg")}, data={"top_k": len(products)})
        results = response.raise_for_status().json()
        assert len(results) == len(products) and results[0]["id"] == products[0]["id"]
        report["api_checks"].append("local UI session/CSRF search and maximum Top-K")
        if args.url:
            response = client.post("/search/url", headers=headers, json={"image_url": args.url, "top_k": 5})
            if response.is_error:
                raise RuntimeError(f"Public URL search failed: HTTP {response.status_code}: {response.text}")
            results = response.json()
            assert len(results) == 5
            report["public_url_search"] = {"url": args.url, "top_result": results[0]}
    save_json(settings.artifacts / "local_verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
