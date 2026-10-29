FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    VISUAL_SEARCH_CACHE_DIR=/tmp/visual-search-cache
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY visual_search ./visual_search
COPY static ./static
COPY data ./data
COPY artifacts/index.json artifacts/embeddings.npy artifacts/product_ids.npy ./artifacts/
COPY api.py validate_index.py start_production.py ./
# Pure local validation: no secrets or NVIDIA calls needed to build the image.
RUN python validate_index.py
USER 10001:10001
EXPOSE 8000
CMD ["python", "start_production.py"]
