"""Stage 4 - Embedding: gold.chunks -> gold.embeddings (incremental, content-addressed cache).

A chunk is embedded only if (text_hash, model_id) has never been embedded before, so re-runs
are cheap and switching models keeps both sets of vectors side by side.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd

from .. import lakehouse
from ..config import Config
from ..embedders import build_embedder
from ..log import get_logger

log = get_logger(__name__)


def embedding_key(text_hash: str, model_id: str) -> str:
    return hashlib.sha1(f"{model_id}|{text_hash}".encode()).hexdigest()


def run(cfg: Config, run_id: str) -> dict:
    chunks = lakehouse.read_table(cfg, "gold", "chunks")
    emb = build_embedder(cfg, corpus=chunks["embed_text"].tolist())
    chunks["embedding_key"] = [embedding_key(h, emb.model_id) for h in chunks["text_hash"]]

    existing: set[str] = set()
    if lakehouse.exists(cfg, "gold", "embeddings"):
        existing = set(lakehouse.read_table(cfg, "gold", "embeddings")["embedding_key"])
    cached = chunks["embedding_key"].isin(existing)
    todo = chunks[~cached].drop_duplicates("embedding_key")
    log.info("embedding %d new chunks with %s (%d cache hits)", len(todo), emb.model_id,
             int(cached.sum()))

    metrics = {"model_id": emb.model_id, "dim": emb.dim, "chunks": len(chunks),
               "newly_embedded": len(todo), "cache_hits": int(cached.sum()),
               "duplicate_texts_skipped": int((~cached).sum()) - len(todo)}
    if len(todo):
        t0 = time.perf_counter()
        bs = cfg.embedding.batch_size * 8
        vecs = np.vstack([emb.embed(todo["embed_text"].iloc[i:i + bs].tolist())
                          for i in range(0, len(todo), bs)])
        elapsed = time.perf_counter() - t0
        out = pd.DataFrame({
            "embedding_key": todo["embedding_key"].values,
            "text_hash": todo["text_hash"].values,
            "model_id": emb.model_id,
            "dim": np.int32(emb.dim),
            "vector": list(vecs.astype(np.float32)),
            "embedded_at": pd.Timestamp.now(tz="UTC"),
        })
        stats = lakehouse.write_table(
            cfg, "gold", "embeddings", out, mode="merge", merge_keys=["embedding_key"],
            description="Embedding cache keyed by (model_id, text_hash); vectors are L2-normalised")
        metrics.update(stats)
        metrics["chunks_per_sec"] = round(len(todo) / elapsed, 1)
    return metrics
