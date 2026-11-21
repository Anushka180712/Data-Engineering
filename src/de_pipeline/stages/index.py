"""Stage 5 - Vector store load: gold.chunks + gold.embeddings -> vector DB.

The store is *synchronised* with the gold layer: current chunks are upserted, chunks that no
longer exist are deleted, and the collection is rebuilt if the embedding model changed.
"""
from __future__ import annotations

import json

import numpy as np

from .. import lakehouse
from ..config import Config
from ..log import get_logger
from ..vectorstores import META_COLS, get_store
from .embed import embedding_key

log = get_logger(__name__)


def run(cfg: Config, run_id: str) -> dict:
    model_id = json.loads((cfg.paths.artifacts_dir / "embedder.json").read_text())["model_id"]
    chunks = lakehouse.read_table(cfg, "gold", "chunks")
    embs = lakehouse.read_table(cfg, "gold", "embeddings")
    embs = embs[embs["model_id"] == model_id]
    chunks["embedding_key"] = [embedding_key(h, model_id) for h in chunks["text_hash"]]
    df = chunks.merge(embs[["embedding_key", "vector"]], on="embedding_key", how="inner")
    if len(df) != len(chunks):
        raise RuntimeError(f"{len(chunks) - len(df)} chunks have no embedding for {model_id}")

    vectors = np.vstack(df["vector"].to_numpy()).astype(np.float32)
    metas = df[META_COLS].assign(chunk_index=df["chunk_index"].astype(int)).to_dict("records")
    store = get_store(cfg)
    stats = store.sync(df["chunk_id"].tolist(), vectors, df["text"].tolist(), metas, model_id)
    log.info("%s: %s", cfg.vector_store.backend, stats)
    return {"backend": cfg.vector_store.backend, "collection": cfg.vector_store.collection,
            "model_id": model_id, **stats}
