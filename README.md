# Forma — Visual Search

A local FastAPI application that searches the supplied 194-product catalog with NVIDIA image embeddings and NumPy cosine similarity. The HTML/CSS/JavaScript UI supports drag-and-drop, previews, direct image URLs, adjustable Top-K, loading/error states, and product cards with actual similarity scores. No database, CLIP, OpenAI SDK, synthetic catalog, frontend build system, or deployment is used.

## Run locally in VS Code

Open this folder in VS Code. Python 3.11 or newer is supported; the supplied `.venv` was created with the machine's Python 3.12.10. The `.vscode` configuration selects the local environment and provides a FastAPI debugger configuration (requires the VS Code Python/Python Debugger extensions).

In a PowerShell terminal in this folder:

```powershell
# First-time environment setup (skip if .venv already exists)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# First time only: copy the example if you do not already have .env
# Copy-Item .env.example .env
# Edit .env locally and set both keys. Never commit the file.

python test_nvidia.py
python build_embeddings.py
python -m pytest
python -m uvicorn api:app --reload --no-proxy-headers
```

If PowerShell activation is unavailable, use `.\.venv\Scripts\python.exe` instead of `python` in every command. The default Uvicorn host and port are `127.0.0.1:8000`. The requested `python -m uvicorn api:app --reload` also works for direct local access; `--no-proxy-headers` makes the loopback-only UI boundary explicit.

- UI: http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

Use the exact configured `UI_ORIGIN` (default `http://127.0.0.1:8000`), including the port. `localhost` is a different origin. Stop the server with Ctrl+C. Restart it after rebuilding the index or changing `.env`; the index loads once at startup.

## Configuration

```dotenv
NVIDIA_API_KEY=your-nvidia-api-catalog-key
VISUAL_SEARCH_API_KEY=your-random-external-api-key
UI_ORIGIN=http://127.0.0.1:8000
```

The installation in this workspace already has a randomly generated `VISUAL_SEARCH_API_KEY` in `.env`; its value is deliberately not printed. Both keys stay server-side. For another installation, create a random application key locally, for example with `python -c "import secrets; print(secrets.token_urlsafe(32))"`, and put it in `.env`. Missing application credentials disable external search. Missing NVIDIA credentials prevent new embeddings; the UI and catalog can still load with a not-ready status.

## Original dataset and validation

`data/products.csv` and `data/images/` are the supplied originals, left unchanged. CSV `name` is exposed as response `title`; all other metadata is preserved. The archive at `../visual_search_task_data.zip` contains 195 files, all already extracted and byte-identical to this workspace.

```powershell
python inspect_dataset.py ..\visual_search_task_data.zip
```

The archive inspector verifies paths and file sizes, refuses conflicting overwrites, and extracts absent files. `test_nvidia.py` and `build_embeddings.py` validate the catalog and write `artifacts/validation_report.json`. Current data:

- 194 rows and 194 readable images; no missing files, invalid rows, duplicate IDs, duplicate CSV rows, duplicate image paths, or unreferenced images.
- 92 blank brands are preserved as empty strings. The UI labels them “Brand not provided.”
- Product IDs 96/191 and 97/192 share identical image bytes. Both products remain searchable; their image embeddings are cached once per distinct image.

Rows with invalid prices, unsafe/missing image paths, missing essential fields, or unreadable images are reported and excluded. The first valid occurrence of a duplicate ID is retained without modifying the CSV. Missing brands are permitted. Original files are never resized or rewritten.

## NVIDIA verification and indexing

The official schema is documented at [NVIDIA's model API reference](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-embed-1b-infer). Requests go directly to `https://integrate.api.nvidia.com/v1/embeddings` using `httpx`:

```json
{
  "model": "nvidia/nemotron-3.5-embed-1b",
  "input": ["data:image/jpeg;base64,<encoded image>"],
  "input_type": "passage",
  "modality": "image",
  "encoding_format": "float",
  "embedding_type": "float",
  "dimensions": 2048,
  "longest_edge": 1120,
  "truncate": "NONE"
}
```

**Image requests require `passage`, even for the image used to search.** `query` is for text. Both indexed and incoming images use the same preprocessing and request configuration: verify bytes, apply EXIF orientation, composite transparency on white, resize to at most 1120 pixels, encode JPEG at quality 92, and strip metadata. JPEG, PNG, and WebP uploads are supported; NVIDIA receives JPEG.

`test_nvidia.py` makes one live request (even if cached), validates the returned 2048 finite, nonzero values, caches the vector, and writes a verification receipt. `build_embeddings.py` refuses to run without a matching receipt and source image. It writes progress after each product and stops on an unrecoverable error. Re-run it to resume from cache.

Cache keys include normalized image bytes, the model/request parameters, and preprocessing version. Identical images reuse cached vectors. The client serializes calls, retries transport failures, HTTP 408/429, and transient 5xx responses up to five total attempts with exponential backoff and jitter. `Retry-After` is respected; a cooldown longer than 60 seconds stops the run with an actionable message. Authorization and validation errors fail immediately. Raw upstream responses and keys are not exposed.

Artifacts:

| File | Purpose |
| --- | --- |
| `artifacts/embeddings.npy` | Float32 matrix, one row per valid product |
| `artifacts/product_ids.npy` | Int64 product IDs aligned with matrix rows |
| `artifacts/index.json` | Model/preprocessing recipe, catalog fingerprint, array checksums |
| `artifacts/cache/*.npy` | Content-addressed vectors for resumable indexing/search |
| `artifacts/nvidia_verified.json` | Successful one-image live verification receipt |
| `artifacts/progress.json` | Completed count and last/failed product ID |
| `artifacts/validation_report.json` | Dataset audit, missing values, duplicate groups |

Array files use `allow_pickle=False` and atomic individual replacements. The manifest is published last. Startup rejects partial, stale, or corrupted arrays rather than returning misaligned products. After a failed rebuild, rerun the builder and restart the server. Credentials and all generated artifacts are excluded from version control.

## API

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/` | Browser UI |
| GET | `/health` | Process/index/configuration status; not a live NVIDIA probe |
| GET | `/products?offset=0&limit=194` | Valid original catalog, JSON array |
| POST | `/search` | Multipart `file`, optional `top_k` (default 5) |
| POST | `/search/url` | JSON `image_url`, optional `top_k` (default 5) |
| GET | `/images/{filename}` | Allowlisted catalog image |

Both `/search` routes require `X-API-Key: <VISUAL_SEARCH_API_KEY>`. In Swagger, click **Authorize** and enter that local application key. Never enter your NVIDIA key in Swagger. Top-K must be 1 through the indexed product count (194 for this dataset); out-of-range input returns 422. Results are a JSON array with these fields:

```json
[
  {
    "rank": 1,
    "id": 1,
    "title": "Essence Mascara Lash Princess",
    "brand": "Essence",
    "category": "Beauty & Personal Care",
    "subcategory": "Makeup",
    "price_usd": 9.99,
    "image_url": "/images/001_essence-mascara-lash-princess.jpg",
    "similarity_score": 1.0
  }
]
```

This example represents an exact-image match; actual values are computed from the embeddings. Cosine similarity is `dot(a, b) / (norm(a) * norm(b))`, sorted descending with stable ties in catalog order. Scores lie in [-1, 1] and are not probabilities. The UI displays four decimal places; JSON retains full computed precision.

Example Python client (keeps credentials out of shell history):

```python
from pathlib import Path
import os
import httpx
from dotenv import load_dotenv

load_dotenv()
headers = {"X-API-Key": os.environ["VISUAL_SEARCH_API_KEY"]}
with Path("data/images/001_essence-mascara-lash-princess.jpg").open("rb") as image:
    response = httpx.post(
        "http://127.0.0.1:8000/search",
        headers=headers,
        files={"file": ("image.jpg", image, "image/jpeg")},
        data={"top_k": 5}, timeout=180,
    )
response.raise_for_status()
print(response.json())
```

For URLs, use `httpx.post("http://127.0.0.1:8000/search/url", headers=headers, json={"image_url": "https://your-public-host/image.jpg", "top_k": 5}, timeout=180)`.

## Local UI and input safety

The UI calls separate `/ui/search` and `/ui/search/url` adapters backed by the identical search function and NumPy engine. A loopback-only `/ui/session` issues a short-lived signed HttpOnly, SameSite=Strict cookie and session CSRF token. Browser search checks the actual client address, configured Host, exact Origin, cookie signature/expiry, and CSRF header. No secret application API key is sent to JavaScript. This is intentionally a **local trusted-user UI**, not a remotely authenticated multi-user service. Do not place it behind a public reverse proxy without adding user authentication and revisiting trust boundaries.

Uploads have a 5 MiB image limit, bounded request bodies, MIME and decoded-format checks, a 20-megapixel limit, decompression-bomb protection, and rejection of animated/SVG files. Product image serving is allowlisted by the validated catalog. Search is limited to 20 requests per client per minute per process.

URL downloads allow only HTTP/HTTPS standard ports without credentials. Every host and redirect is resolved and checked for public addresses; private, loopback, link-local, multicast and mapped IPv6 addresses are rejected. Connections pin the validated IP while preserving the hostname for TLS verification, preventing a second DNS lookup from bypassing the check. Proxy environment settings are not used. Redirects, response type, size, and download time are bounded. The original URL is not handed to NVIDIA. DNS resolution uses the OS resolver, whose own timeout applies before socket/download timeouts.

## Tests and limitations

Local verification on 2026-10-29 passed all 39 automated tests and completed NVIDIA's one-image test, all 194 indexed products, 194 exact-image self-matches, three real uploaded-image API searches, default and maximum Top-K, protected API routes, and the UI session/CSRF adapter. A public URL search of the mascara image returned the correct product first with cosine similarity `0.9896297004200088`. Evidence is in `artifacts/local_verification.json`. The original ZIP comparison still reports all 195 files identical after implementation.

Requirements also resolved successfully for Python 3.11 (`artifacts/python311_resolution.json`); runtime tests used installed Python 3.12.10. JavaScript syntax checks, Python compilation, and `pip check` passed. Starlette emits one upstream deprecation warning about its httpx-based test client; the tests pass. Browser visual/interactive QA could not be completed: the browser host was unavailable, and desktop automation subsequently stopped because it could not determine Chrome's current URL reliably. HTTP checks confirmed that the UI assets and Swagger are served. No deployment was performed.

```powershell
python -m pytest
python test_nvidia.py    # Live NVIDIA test; consumes one API request
# With the local server running:
python verify_local.py
python verify_local.py --url https://cdn.dummyjson.com/product-images/beauty/essence-mascara-lash-princess/1.webp
```

Offline tests use actual catalog metadata with small deterministic embedding test doubles. These vectors are never written into the real product index. Tests cover cosine scores/ranking/ties, invalid Top-K/vectors, API authentication, JSON response fields, upload validation/size limits, local session/CSRF protection, private URL and redirect blocking, NVIDIA schema/caching/retries/errors, catalog validation, and unavailable indexes. `pytest.ini` excludes the live verification script from automatic collection.

The model is optimized for text/document-image retrieval, as described in [NVIDIA's model card](https://build.nvidia.com/nvidia/nemotron-3_5-embed-1b/modelcard). This application compares its image embeddings for product photos; relevance should be evaluated with representative user photos. Exact-image self-matches validate plumbing and index alignment, not general retrieval quality. API availability and quotas are external dependencies. Query vectors are cached locally; image uploads themselves are not persisted. Cache eviction, distributed rate limiting, multiple users, and deployment are outside this local application's scope.
