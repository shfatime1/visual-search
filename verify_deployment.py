"""Verify the real Render deployment without logging credentials or session tokens."""
import argparse
import json
from urllib.parse import urlsplit
from datetime import datetime, timezone
import httpx
from visual_search.catalog import load_catalog
from visual_search.config import Settings
from visual_search.nvidia import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://visual-search-1slj.onrender.com")
    parser.add_argument("--health-only", action="store_true")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    if urlsplit(base).scheme != "https":
        raise SystemExit("Use an HTTPS deployment URL.")
    settings = Settings()
    with httpx.Client(base_url=base, timeout=180, follow_redirects=False, trust_env=False) as client:
        health_response = client.get("/health")
        health = health_response.json()
        print(json.dumps({"http_status": health_response.status_code, "health": health}, indent=2), flush=True)
        if args.health_only:
            return
        health_response.raise_for_status()
        assert health["indexed_products"] == 194 and health["status"] == "ok", "Deployment not ready"
        assert health["external_api_configured"] and health["public_ui_enabled"], "Server environment incomplete"
        products, _ = load_catalog(settings.data_dir)
        product = products[173]
        files = {"file": (product["_path"].name, product["_path"].read_bytes(), "image/jpeg")}
        headers = {"X-API-Key": settings.api_key}
        response = client.post("/search", headers=headers, files=files, data={"top_k": 5})
        response.raise_for_status()
        results = response.json()
        assert len(results) == 5 and results[0]["id"] == product["id"], "Unexpected product search result"
        assert results[0]["similarity_score"] > 0.99
        assert client.post("/search", files=files).status_code == 401
        session = client.get("/ui/session")
        session.raise_for_status()
        assert "Secure" in session.headers["set-cookie"] and "HttpOnly" in session.headers["set-cookie"]
        csrf = session.json()["csrf_token"]
        response = client.post("/ui/search", headers={"Origin": base, "X-CSRF-Token": csrf}, files=files, data={"top_k": 194})
        response.raise_for_status()
        assert len(response.json()) == 194 and response.json()[0]["id"] == product["id"]
        assert client.post("/ui/search", headers={"Origin": "https://attacker.example", "X-CSRF-Token": csrf}, files=files).status_code == 403
        url_response = client.post("/search/url", headers=headers, json={"image_url": "https://cdn.dummyjson.com/product-images/beauty/essence-mascara-lash-princess/1.webp", "top_k": 5})
        url_response.raise_for_status()
        assert url_response.json()[0]["id"] == 1
        report = {"verified_at": datetime.now(timezone.utc).isoformat(), "base_url": base, "health": health,
                  "upload_top_result": results[0], "url_top_result": url_response.json()[0],
                  "checks": ["194 indexed products", "authenticated upload Top-5", "unauthenticated API blocked",
                             "HTTPS browser session + CSRF", "browser Top-194", "cross-site request blocked", "public URL search"]}
        save_json(settings.artifacts / "deployment_verification.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
