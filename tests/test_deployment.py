"""Offline regressions for a deployment with no local cache or verification receipt."""
import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from fastapi.testclient import TestClient
from api import create_app
from validate_index import validate
from visual_search.catalog import load_catalog
from visual_search.config import ROOT, Settings, default_origin
from visual_search.engine import SearchEngine


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(ROOT / "data")[0]


@pytest.fixture
def isolated_bundle(tmp_path):
    destination = tmp_path / "release" / "artifacts"
    destination.mkdir(parents=True)
    for name in ("embeddings.npy", "product_ids.npy", "index.json"):
        shutil.copyfile(ROOT / "artifacts" / name, destination / name)
    return Settings(artifacts=destination, cache_dir=tmp_path / "query-cache", nvidia_key="", api_key="")


def test_bundled_index_validates_without_api_or_cache(isolated_bundle):
    with patch("httpx.AsyncClient.post", side_effect=AssertionError("No NVIDIA requests during validation")):
        summary = validate(isolated_bundle)
    assert summary["indexed_products"] == 194
    assert summary["dimensions"] == 2048
    assert summary["nvidia_calls"] == 0
    assert not isolated_bundle.embedding_cache_dir.exists()
    assert not (isolated_bundle.artifacts / "nvidia_verified.json").exists()


def test_fingerprint_portable_when_project_moves(tmp_path, catalog):
    from visual_search.catalog import fingerprint
    # Fingerprinting excludes absolute paths and hashes original image bytes.
    moved = []
    for p in catalog:
        target = tmp_path / p["_path"].name
        shutil.copyfile(p["_path"], target)
        moved.append({**p, "_path": target})
    assert fingerprint(moved) == fingerprint(catalog)


@pytest.mark.parametrize("filename", ["index.json", "embeddings.npy", "product_ids.npy"])
def test_missing_bundle_file_fails_explicitly(isolated_bundle, catalog, filename):
    (isolated_bundle.artifacts / filename).unlink()
    with pytest.raises(ValueError, match="Missing bundled index file"):
        SearchEngine.load(isolated_bundle, catalog)


def test_corrupt_vectors_rejected(isolated_bundle, catalog):
    with (isolated_bundle.artifacts / "embeddings.npy").open("ab") as f:
        f.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        SearchEngine.load(isolated_bundle, catalog)


def test_stale_catalog_rejected(isolated_bundle, catalog):
    changed = [{**catalog[0], "title": "changed"}, *catalog[1:]]
    with pytest.raises(ValueError, match="stale"):
        SearchEngine.load(isolated_bundle, changed)


def test_mismatched_manifest_count_rejected(isolated_bundle, catalog):
    path = isolated_bundle.artifacts / "index.json"
    manifest = json.loads(path.read_text())
    manifest["count"] = 193
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match"):
        SearchEngine.load(isolated_bundle, catalog)


def test_render_origin_defaults(monkeypatch):
    monkeypatch.delenv("UI_ORIGIN", raising=False)
    monkeypatch.setenv("RENDER_EXTERNAL_URL", "https://visual-search-1slj.onrender.com")
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.delenv("PUBLIC_UI_ENABLED", raising=False)
    assert default_origin() == "https://visual-search-1slj.onrender.com"
    assert Settings().public_ui
    monkeypatch.setenv("PUBLIC_UI_ENABLED", "false")
    assert not Settings().public_ui


@pytest.fixture
def public_client(catalog, tmp_path):
    settings = Settings(artifacts=tmp_path, nvidia_key="nvidia-test", api_key="api-test",
                        public_ui=True, ui_origin="https://visual-search-1slj.onrender.com")
    engine = SearchEngine(catalog[:3], np.array([[1., 0], [0, 1], [-1, 0]]), np.array([p["id"] for p in catalog[:3]]))
    class Embedder:
        async def embed(self, image):
            return np.array([1., 0])
    with patch("api.load_catalog", return_value=(catalog, {})):
        with TestClient(create_app(settings, engine, Embedder()), base_url=settings.ui_origin,
                        client=("203.0.113.20", 443)) as client:
            yield client


def test_public_ui_secure_cookie_search_and_external_auth(public_client, catalog):
    response = public_client.get("/ui/session")
    assert response.status_code == 200
    assert "Secure" in response.headers["set-cookie"]
    assert "HttpOnly" in response.headers["set-cookie"]
    headers = {"Origin": "https://visual-search-1slj.onrender.com", "X-CSRF-Token": response.json()["csrf_token"]}
    files = {"file": ("product.jpg", catalog[0]["_path"].read_bytes(), "image/jpeg")}
    assert public_client.post("/ui/search", headers=headers, files=files, data={"top_k": 1}).status_code == 200
    assert public_client.post("/search", files=files, data={"top_k": 1}).status_code == 401
    assert public_client.post("/ui/search", files=files, data={"top_k": 1}).status_code == 403


@pytest.mark.parametrize("headers", [{"Host": "attacker.example"}, {"Origin": "https://attacker.example"}, {"Sec-Fetch-Site": "cross-site"}])
def test_public_ui_cross_site_rejected(public_client, headers):
    assert public_client.get("/ui/session", headers=headers).status_code == 403


def test_public_session_rate_limit(public_client):
    for _ in range(20):
        assert public_client.get("/ui/session").status_code == 200
    assert public_client.get("/ui/session").status_code == 429


def test_missing_index_health_is_unhealthy(tmp_path, catalog):
    with patch("api.load_catalog", return_value=(catalog, {})):
        with TestClient(create_app(Settings(artifacts=tmp_path, nvidia_key="test"))) as client:
            response = client.get("/health")
            assert response.status_code == 503
            assert response.json()["indexed_products"] == 0
