# Forma — Visual Search

Upload an image (or paste an image URL) and find the most similar products from a 194-product catalog.

**🔗 Live demo: https://visual-search-01.onrender.com**

## How it works

1. Each catalog image is converted to a 2048-dimension vector using NVIDIA's `nemotron-3.5-embed-1b` model.
2. Vectors are stored in three small files in `artifacts/` (committed to Git, ~1.5 MB total).
3. A search image is embedded the same way and compared to the catalog using cosine similarity (NumPy).
4. The top-K most similar products are returned with their similarity scores.

**Stack:** FastAPI, NumPy, httpx, plain HTML/CSS/JS. No database, no frontend build step.

## Features

- Drag-and-drop upload, image preview, or direct image URL
- Adjustable Top-K (1–194)
- Product cards with similarity scores
- Loading and error states
- Protected REST API

## Quick start (local)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env        # then set your keys
python test_nvidia.py              # one live NVIDIA test
python build_embeddings.py         # build the index
python -m pytest                   # run tests
python -m uvicorn api:app --reload --no-proxy-headers
```

Open http://127.0.0.1:8000 (Swagger: `/docs`, health: `/health`).

## Configuration (`.env`)

```dotenv
NVIDIA_API_KEY=your-nvidia-api-catalog-key
VISUAL_SEARCH_API_KEY=your-random-external-api-key
UI_ORIGIN=http://127.0.0.1:8000
```

Generate a random key: `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
Never commit `.env`.

## API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Web UI |
| GET | `/health` | Index and config status |
| GET | `/products` | Catalog list |
| POST | `/search` | Upload an image (`file`, optional `top_k`) |
| POST | `/search/url` | Search by `image_url` (optional `top_k`) |

Both search routes require the header `X-API-Key: <VISUAL_SEARCH_API_KEY>`.

Example response item:

```json
{
  "rank": 1,
  "id": 1,
  "title": "Essence Mascara Lash Princess",
  "brand": "Essence",
  "category": "Beauty & Personal Care",
  "price_usd": 9.99,
  "image_url": "/images/001_essence-mascara-lash-princess.jpg",
  "similarity_score": 1.0
}
```

Scores are cosine similarity in [-1, 1], not probabilities.

## Deploy on Render

| Setting | Value |
| --- | --- |
| Build command | `python -m pip install -r requirements.txt && python validate_index.py` |
| Start command | `python start_production.py` |
| Health check path | `/health` |

Set `NVIDIA_API_KEY` and `VISUAL_SEARCH_API_KEY` as secret environment variables. Do not run `build_embeddings.py` during deployment, because the index is already included in the repo.

Verify after deploying:

```powershell
python verify_deployment.py --health-only
python verify_deployment.py
```

## Security

- Uploads: max 5 MiB, JPEG/PNG/WebP only, 20 MP limit, no SVG or animated images.
- URL search blocks private/internal addresses (SSRF protection).
- Browser search uses a signed cookie, CSRF token, and rate limits (20 requests/min per client).
- API keys stay server-side and never reach the browser.

## Limitations

- The embedding model is optimized for text and document images, so check result quality with real user photos.
- Rate limiting is in-process. Use a single worker, or add shared limiting before scaling.
- Searching new images depends on NVIDIA API availability and quotas.
