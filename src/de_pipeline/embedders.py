"""Pluggable embedding backends.

* sentence-transformers (default when installed): all-MiniLM-L6-v2, 384-d neural embeddings,
  runs locally on CPU, no API key.
* lsa: TF-IDF + truncated SVD (Latent Semantic Analysis). Classic, dependency-light and fully
  offline; used as an automatic fallback when the neural model can't be installed/downloaded.

Every backend returns L2-normalised float32 vectors, so inner product == cosine similarity.
The chosen backend is persisted (artifacts/embedder.json) so queries are embedded with
exactly the same model as the corpus.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Config
from .log import get_logger

log = get_logger(__name__)


class Embedder(Protocol):
    model_id: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model_name, device="cpu")
        self.batch_size = batch_size
        self.model_id = f"st:{model_name}"
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        return _l2(self.model.encode(texts, batch_size=self.batch_size,
                                     show_progress_bar=False, convert_to_numpy=True))


class LSAEmbedder:
    """TF-IDF (uni+bi-grams) -> TruncatedSVD -> L2 norm. Fitted once on the corpus, then frozen."""

    def __init__(self, path: Path, dimensions: int = 256):
        self.path = path
        self.dimensions = dimensions
        self.pipeline = None
        self.model_id = ""
        self.dim = dimensions
        if path.exists():
            state = pickle.loads(path.read_bytes())
            self.pipeline, self.model_id, self.dim = state["pipeline"], state["model_id"], state["dim"]

    @property
    def is_fitted(self) -> bool:
        return self.pipeline is not None

    def fit(self, corpus: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline

        dims = min(self.dimensions, len(corpus) - 1)
        self.pipeline = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                            min_df=2 if len(corpus) >= 100 else 1, max_df=0.6 if len(corpus) >= 100 else 1.0,
                            stop_words="english", max_features=100_000),
            TruncatedSVD(n_components=dims, random_state=42),
        )
        self.pipeline.fit(corpus)
        fingerprint = hashlib.sha1("".join(sorted(corpus)).encode()).hexdigest()[:8]
        self.model_id, self.dim = f"lsa:tfidf-svd{dims}-{fingerprint}", dims
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(pickle.dumps({"pipeline": self.pipeline, "model_id": self.model_id,
                                            "dim": dims}))
        log.info("fitted LSA embedder %s on %d texts", self.model_id, len(corpus))

    def embed(self, texts: list[str]) -> np.ndarray:
        return _l2(self.pipeline.transform(texts))


def _state_file(cfg: Config) -> Path:
    return cfg.paths.artifacts_dir / "embedder.json"


def build_embedder(cfg: Config, corpus: list[str] | None = None) -> Embedder:
    """Create the embedder for indexing. `corpus` is required to (re)fit LSA."""
    e = cfg.embedding
    backend = e.backend
    if backend in ("auto", "sentence-transformers"):
        try:
            emb = SentenceTransformerEmbedder(e.model_name, e.batch_size)
            backend = "sentence-transformers"
        except Exception as exc:  # ImportError, or no network to fetch weights
            if e.backend == "sentence-transformers":
                raise
            log.warning("sentence-transformers unavailable (%s: %s) -> falling back to LSA",
                        type(exc).__name__, str(exc).splitlines()[0][:120])
            backend = "lsa"
    if backend == "lsa":
        emb = LSAEmbedder(cfg.paths.artifacts_dir / "lsa_embedder.pkl", e.lsa_dimensions)
        if corpus is not None:
            fp = hashlib.sha1("".join(sorted(corpus)).encode()).hexdigest()[:8]
            if not emb.is_fitted or not emb.model_id.endswith(fp):
                emb.fit(corpus)
        if not emb.is_fitted:
            raise RuntimeError("LSA embedder not fitted yet - run the embed stage first")
    elif backend != "sentence-transformers":
        raise ValueError(f"unknown embedding backend {backend}")
    _state_file(cfg).write_text(json.dumps({"backend": backend, "model_id": emb.model_id,
                                            "dim": emb.dim}))
    return emb


def load_query_embedder(cfg: Config) -> Embedder:
    """Re-create exactly the embedder the index was built with."""
    state = json.loads(_state_file(cfg).read_text())
    if state["backend"] == "sentence-transformers":
        return SentenceTransformerEmbedder(state["model_id"].removeprefix("st:"))
    return LSAEmbedder(cfg.paths.artifacts_dir / "lsa_embedder.pkl")
