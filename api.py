import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from visual_search.catalog import load_catalog, public_product
from visual_search.config import ROOT, MAX_IMAGE_BYTES, MODEL, Settings
from visual_search.engine import SearchEngine
from visual_search.images import prepare_image
from visual_search.nvidia import NvidiaEmbedder, NvidiaError
from visual_search.remote import download_image
from visual_search.security import BodyLimit, LocalSessions


class URLSearch(BaseModel):
    image_url: str = Field(min_length=1, max_length=2048)
    top_k: int = Field(default=5, ge=1, strict=True)


class Product(BaseModel):
    id: int
    title: str
    brand: str
    category: str
    subcategory: str
    price_usd: float
    image_url: str


class SearchResult(Product):
    rank: int
    similarity_score: float


def create_app(settings=None, engine=None, embedder=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        products, report = load_catalog(settings.data_dir)
        app.state.products = products
        app.state.report = report
        app.state.engine = engine
        app.state.index_error = None
        if engine is None:
            try:
                app.state.engine = SearchEngine.load(settings, products)
            except (OSError, ValueError, KeyError, IndexError) as exc:
                logging.getLogger(__name__).error("Index load failed from %s: %s", settings.artifacts, exc)
                app.state.index_error = "Product index unavailable. Deploy the validated bundled index and run validate_index.py; check server logs for the exact validation failure."
            else:
                logging.getLogger("uvicorn.error").info("Loaded %d product embeddings from %s", len(products), settings.artifacts)
        app.state.embedder = embedder or NvidiaEmbedder(settings)
        yield

    app = FastAPI(title="Visual Search", version="1.0.0", lifespan=lifespan)
    app.add_middleware(BodyLimit)
    sessions = LocalSessions(settings)
    gate = asyncio.Semaphore(2)
    key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def external_auth(request: Request, key: Annotated[str | None, Depends(key_header)]):
        if not settings.api_key:
            raise HTTPException(503, "External search is disabled until VISUAL_SEARCH_API_KEY is configured.")
        if not key or not hmac.compare_digest(key.encode(), settings.api_key.encode()):
            raise HTTPException(401, "Invalid or missing X-API-Key.")
        sessions.rate_limit(request)

    async def browser_auth(request: Request):
        sessions.authorize(request)
        sessions.rate_limit(request)

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path not in {"/docs", "/redoc", "/docs/oauth2-redirect"}:
            response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' blob:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        if request.url.path.startswith(("/ui/", "/search")):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(NvidiaError)
    async def nvidia_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    @app.get("/", include_in_schema=False)
    async def home():
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/health")
    async def health():
        count = len(app.state.engine.products) if app.state.engine else 0
        ready = count > 0 and bool(settings.nvidia_key)
        detail = app.state.index_error
        if not detail and not settings.nvidia_key:
            detail = "Configure NVIDIA_API_KEY in the server environment to embed search images."
        payload = {"status": "ok" if ready else "not_ready", "indexed_products": count,
                "catalog_products": len(app.state.products), "model": MODEL,
                "nvidia_configured": bool(settings.nvidia_key), "external_api_configured": bool(settings.api_key),
                "public_ui_enabled": settings.public_ui, "revision": os.getenv("RENDER_GIT_COMMIT", "local"),
                "detail": detail}
        return JSONResponse(payload, status_code=200 if ready else 503)

    @app.get("/products", response_model=list[Product])
    async def products(offset: int = Query(0, ge=0), limit: int = Query(194, ge=1, le=1000)):
        return [public_product(p) for p in app.state.products[offset:offset + limit]]

    @app.get("/images/{filename}", include_in_schema=False)
    async def product_image(filename: str):
        product = next((p for p in app.state.products if p["_path"].name == filename), None)
        if not product:
            raise HTTPException(404, "Product image not found.")
        return FileResponse(product["_path"], headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/ui/session", include_in_schema=False)
    async def session(request: Request):
        cookie, csrf = sessions.issue(request)
        response = JSONResponse({"csrf_token": csrf})
        response.set_cookie("visual_session", cookie, max_age=3600, httponly=True, samesite="strict", secure=settings.ui_origin.startswith("https:"), path="/ui")
        return response

    def require_engine(top_k):
        current = app.state.engine
        if current is None:
            raise HTTPException(503, app.state.index_error)
        if not 1 <= top_k <= len(current.products):
            raise HTTPException(422, f"top_k must be between 1 and {len(current.products)}.")
        return current

    async def search_bytes(content, top_k):
        current = require_engine(top_k)
        try:
            image = await asyncio.to_thread(prepare_image, content)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        async with gate:
            vector = await app.state.embedder.embed(image)
        return current.search(vector, top_k)

    async def upload_search(file, top_k):
        require_engine(top_k)
        try:
            if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
                raise HTTPException(415, "Use a JPEG, PNG, or WebP image.")
            content = await file.read(MAX_IMAGE_BYTES + 1)
        finally:
            await file.close()
        if len(content) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Image exceeds 5 MiB.")
        return await search_bytes(content, top_k)

    async def url_search(payload):
        require_engine(payload.top_k)
        try:
            content = await asyncio.to_thread(download_image, payload.image_url)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return await search_bytes(content, payload.top_k)

    @app.post("/search", response_model=list[SearchResult], dependencies=[Depends(external_auth)])
    async def search(file: UploadFile = File(...), top_k: int = Form(5, ge=1)):
        return await upload_search(file, top_k)

    @app.post("/search/url", response_model=list[SearchResult], dependencies=[Depends(external_auth)])
    async def search_url(payload: URLSearch):
        return await url_search(payload)

    @app.post("/ui/search", response_model=list[SearchResult], dependencies=[Depends(browser_auth)], include_in_schema=False)
    async def ui_search(file: UploadFile = File(...), top_k: int = Form(5, ge=1)):
        return await upload_search(file, top_k)

    @app.post("/ui/search/url", response_model=list[SearchResult], dependencies=[Depends(browser_auth)], include_in_schema=False)
    async def ui_url(payload: URLSearch):
        return await url_search(payload)

    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    return app


app = create_app()
