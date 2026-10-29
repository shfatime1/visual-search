import hashlib
import json
import numpy as np
from .catalog import fingerprint, public_product
from .nvidia import RECIPE, CACHE_VERSION


def cosine_rank(embeddings, query, top_k=5):
    matrix = np.asarray(embeddings, dtype=np.float64)
    vector = np.asarray(query, dtype=np.float64)
    if matrix.ndim != 2 or vector.shape != (matrix.shape[1],) or not len(matrix):
        raise ValueError("Invalid embedding dimensions.")
    if not isinstance(top_k, int) or not 1 <= top_k <= len(matrix):
        raise ValueError(f"top_k must be between 1 and {len(matrix)}.")
    norms = np.linalg.norm(matrix, axis=1)
    norm = np.linalg.norm(vector)
    if not np.isfinite(matrix).all() or not np.isfinite(vector).all() or norm == 0 or (norms == 0).any():
        raise ValueError("Embeddings must be finite and nonzero.")
    scores = np.clip((matrix @ vector) / (norms * norm), -1, 1)
    order = np.argsort(-scores, kind="stable")[:top_k]
    return order, scores[order]


class SearchEngine:
    def __init__(self, products, embeddings, product_ids):
        by_id = {p["id"]: p for p in products}
        if len(product_ids) != len(embeddings) or len(set(product_ids.tolist())) != len(product_ids):
            raise ValueError("Index IDs and vectors do not align.")
        self.products = [by_id[int(pid)] for pid in product_ids]
        self.embeddings = embeddings
        cosine_rank(embeddings, embeddings[0], 1)

    def search(self, query, top_k):
        order, scores = cosine_rank(self.embeddings, query, top_k)
        return [{"rank": rank, **public_product(self.products[int(index)]), "similarity_score": float(score)}
                for rank, (index, score) in enumerate(zip(order, scores), 1)]

    @classmethod
    def load(cls, settings, products):
        for name in ("index.json", "embeddings.npy", "product_ids.npy"):
            if not (settings.artifacts / name).is_file():
                raise ValueError(f"Missing bundled index file: {name}. Deploy the validated artifacts directory; NVIDIA reindexing is not required.")
        manifest = json.loads((settings.artifacts / "index.json").read_text())
        if manifest.get("recipe") != RECIPE or manifest.get("cache_version") != CACHE_VERSION or manifest.get("catalog_fingerprint") != fingerprint(products):
            raise ValueError("Index is stale. Run build_embeddings.py and restart the server.")
        for name in ("embeddings.npy", "product_ids.npy"):
            if hashlib.sha256((settings.artifacts / name).read_bytes()).hexdigest() != manifest["files"][name]:
                raise ValueError("Index checksum mismatch. Rebuild the index.")
        ids = np.load(settings.artifacts / "product_ids.npy", allow_pickle=False)
        vectors = np.load(settings.artifacts / "embeddings.npy", allow_pickle=False)
        if (manifest.get("count") != len(products) or ids.dtype != np.dtype("int64")
                or vectors.dtype != np.dtype("float32") or ids.ndim != 1
                or ids.tolist() != [p["id"] for p in products]
                or vectors.shape != (len(products), RECIPE["dimensions"])):
            raise ValueError("Index does not match the catalog.")
        return cls(products, vectors, ids)
