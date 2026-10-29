import asyncio
import io
import socket
from unittest.mock import patch
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from api import create_app
from visual_search.catalog import load_catalog
from visual_search.config import ROOT, DIMENSIONS, Settings
from visual_search.engine import SearchEngine, cosine_rank
from visual_search.images import prepare_image
from visual_search.nvidia import NvidiaEmbedder, NvidiaError, RECIPE
from visual_search.remote import resolve_public, download_image


@pytest.fixture(scope="session")
def products():
    return load_catalog(ROOT / "data")[0]


@pytest.fixture
def client(tmp_path, products):
    # Actual catalog metadata; small deterministic vectors are test doubles only.
    engine = SearchEngine(products[:3], np.array([[3., 0], [1, 1], [-2, 0]]), np.array([p['id'] for p in products[:3]]))
    class FakeEmbedder:
        async def embed(self, image):
            return np.array([1., 0])
    settings = Settings(artifacts=tmp_path, api_key="test-key", nvidia_key="test-nvidia-key")
    with patch("api.load_catalog", return_value=(products, {})):
        with TestClient(create_app(settings, engine, FakeEmbedder()), base_url="http://127.0.0.1:8000", client=("127.0.0.1", 12345)) as test_client:
            yield test_client


def upload(products):
    return {"file": ("sample.jpg", products[0]["_path"].read_bytes(), "image/jpeg")}


def test_cosine_ranking_and_actual_scores():
    order, scores = cosine_rank([[0, 7], [9, 0], [-3, 0], [1, 1]], [2, 0], 4)
    assert order.tolist() == [1, 3, 0, 2]
    assert scores.tolist() == pytest.approx([1, 1 / np.sqrt(2), 0, -1])


def test_ties_are_stable():
    order, _ = cosine_rank([[1, 0], [2, 0]], [1, 0], 2)
    assert order.tolist() == [0, 1]


@pytest.mark.parametrize("k", [0, -1, 3])
def test_invalid_top_k(k):
    with pytest.raises(ValueError):
        cosine_rank([[1, 0], [0, 1]], [1, 0], k)


@pytest.mark.parametrize("query", [[0, 0], [float("nan"), 1], [1, 2, 3]])
def test_bad_vectors(query):
    with pytest.raises(ValueError):
        cosine_rank([[1, 0]], query, 1)


@pytest.mark.parametrize("endpoint", ["/search", "/search/url"])
def test_auth_required(client, products, endpoint):
    kwargs = {"files": upload(products)} if endpoint == "/search" else {"json": {"image_url": "https://example.com/a.jpg"}}
    assert client.post(endpoint, **kwargs).status_code == 401
    assert client.post(endpoint, headers={"X-API-Key": "wrong"}, **kwargs).status_code == 401


def test_search_response(client, products):
    response = client.post("/search", headers={"X-API-Key": "test-key"}, files=upload(products), data={"top_k": 3})
    assert response.status_code == 200, response.text
    rows = response.json()
    assert [r["id"] for r in rows] == [p["id"] for p in products[:3]]
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert rows[0]["similarity_score"] == pytest.approx(1)
    assert rows[2]["similarity_score"] == pytest.approx(-1)
    assert set(rows[0]) == {"rank", "id", "title", "brand", "category", "subcategory", "price_usd", "image_url", "similarity_score"}


def test_top_k_api_bound(client, products):
    assert client.post("/search", headers={"X-API-Key": "test-key"}, files=upload(products), data={"top_k": 4}).status_code == 422


def test_url_search_same_engine(client, products):
    with patch("api.download_image", return_value=products[0]["_path"].read_bytes()):
        response = client.post("/search/url", headers={"X-API-Key": "test-key"}, json={"image_url": "https://example.com/image.jpg", "top_k": 1})
    assert response.status_code == 200
    assert response.json()[0]["id"] == products[0]["id"]


def test_invalid_images(client):
    response = client.post("/search", headers={"X-API-Key": "test-key"}, files={"file": ("fake.jpg", b"not an image", "image/jpeg")}, data={"top_k": 1})
    assert response.status_code == 400
    response = client.post("/search", headers={"X-API-Key": "test-key"}, files={"file": ("a.svg", b"<svg/>", "image/svg+xml")}, data={"top_k": 1})
    assert response.status_code == 415


def test_oversized_request(client):
    response = client.post("/search", content=b"x" * (5 * 1024 * 1024 + 65537), headers={"X-API-Key": "test-key", "Content-Type": "application/octet-stream"})
    assert response.status_code == 413


def test_browser_session_csrf(client, products):
    assert client.post("/ui/search", files=upload(products), data={"top_k": 1}).status_code == 403
    session = client.get("/ui/session")
    assert "HttpOnly" in session.headers["set-cookie"] and "SameSite=strict" in session.headers["set-cookie"]
    token = session.json()["csrf_token"]
    headers = {"Origin": "http://127.0.0.1:8000", "X-CSRF-Token": token}
    assert client.post("/ui/search", headers=headers, files=upload(products), data={"top_k": 1}).status_code == 200
    headers["Origin"] = "https://attacker.example"
    assert client.post("/ui/search", headers=headers, files=upload(products), data={"top_k": 1}).status_code == 403


def test_local_sessions_reject_host_rebinding_and_cross_site(client):
    assert client.get("/ui/session", headers={"Host": "attacker.example"}).status_code == 403
    assert client.get("/ui/session", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403


def test_health_catalog_static_and_no_credentials(client, products):
    assert client.get("/health").json()["indexed_products"] == 3
    assert len(client.get("/products").json()) == 194
    assert client.get(products[0]["image_url"]).status_code == 200
    assert client.get("/images/.env").status_code == 404
    for path in ["/", "/static/app.js", "/health", "/openapi.json"]:
        text = client.get(path).text
        assert "test-key" not in text and "test-nvidia-key" not in text


def test_catalog_validation_preserves_missing_brands():
    products, report = load_catalog(ROOT / "data")
    assert len(products) == 194
    assert report["invalid_rows"] == []
    assert report["duplicate_ids"] == []
    assert len(report["missing_values"]["brand"]) == 92
    assert sum(p["brand"] == "" for p in products) == 92
    assert report["duplicate_image_content"] == [[96, 191], [97, 192]]


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "224.0.0.1", "::ffff:127.0.0.1", "2002:7f00:1::", "2001:0000:4136:e378:8000:63bf:3fff:fdd2"])
def test_ssrf_blocks_addresses(address):
    with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))]):
        with pytest.raises(ValueError):
            resolve_public("https://example.com/image.jpg")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://user:pass@example.com/a", "https://example.com:8443/a", "http://example.com/\r\nX:1"])
def test_ssrf_blocks_invalid_urls(url):
    with pytest.raises(ValueError):
        resolve_public(url)


def test_ssrf_rejects_mixed_public_and_private_dns():
    addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443)) for ip in ["93.184.216.34", "10.0.0.1"]]
    with patch("socket.getaddrinfo", return_value=addresses):
        with pytest.raises(ValueError, match="blocked"):
            resolve_public("https://example.com/image.jpg")


def test_ssrf_rechecks_redirect():
    from visual_search.remote import PinnedConnection
    class Response:
        status = 302
        def getheader(self, key, default=None):
            return "http://127.0.0.1/private" if key == "Location" else default
    def resolve(host, port, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34" if host == "example.com" else "127.0.0.1", port))]
    with patch("socket.getaddrinfo", side_effect=resolve), patch.object(PinnedConnection, "request"), patch.object(PinnedConnection, "getresponse", return_value=Response()):
        with pytest.raises(ValueError, match="blocked"):
            download_image("https://example.com/image.jpg")


def test_nvidia_schema_and_cache(tmp_path, products):
    calls = []
    def handler(request):
        import json
        body = json.loads(request.content)
        assert all(body[k] == v for k, v in RECIPE.items())
        assert body["input"][0].startswith("data:image/jpeg;base64,")
        calls.append(request)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.] * DIMENSIONS}]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            embedder = NvidiaEmbedder(Settings(artifacts=tmp_path, nvidia_key="fake"), http)
            image = prepare_image(products[0]["_path"].read_bytes())
            first = await embedder.embed(image)
            second = await embedder.embed(image)
            assert np.array_equal(first, second)
            assert embedder.cache_hit
    asyncio.run(run())
    assert len(calls) == 1


def test_nvidia_retry_rate_limit(tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"data": [{"embedding": [1.] * DIMENSIONS}]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            with patch("visual_search.nvidia.asyncio.sleep") as sleep:
                await NvidiaEmbedder(Settings(artifacts=tmp_path, nvidia_key="fake"), http).embed(b"test-only")
                assert sleep.await_count == 1
    asyncio.run(run())
    assert len(calls) == 2


@pytest.mark.parametrize("status", [401, 403, 400])
def test_nvidia_errors_are_sanitized(tmp_path, status):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(status, text="sensitive upstream content"))) as http:
            with pytest.raises(NvidiaError) as caught:
                await NvidiaEmbedder(Settings(artifacts=tmp_path, nvidia_key="fake"), http).embed(b"test-only")
            assert str(status) in str(caught.value)
            assert "sensitive" not in str(caught.value)
    asyncio.run(run())


def test_unavailable_index_returns_503(tmp_path, products):
    with TestClient(create_app(Settings(artifacts=tmp_path, api_key="test-key"))) as client:
        assert client.get("/health").json()["status"] == "not_ready"
        assert client.post("/search", headers={"X-API-Key": "test-key"}, files=upload(products)).status_code == 503
