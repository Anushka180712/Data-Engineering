"""Vector store backends behind one small interface.

* chroma: persistent ChromaDB collection (HNSW, cosine), metadata filtering, upsert/delete sync
* faiss : exact inner-product index (IndexFlatIP) + Parquet id/metadata sidecar
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .config import Config

META_COLS = ["doc_id", "chunk_index", "title", "category"]


class VectorStore(Protocol):
    def sync(self, ids: list[str], vectors: np.ndarray, docs: list[str], metas: list[dict],
             model_id: str) -> dict: ...

    def query(self, vector: np.ndarray, k: int, where: dict | None = None) -> list[dict]: ...

    def count(self) -> int: ...


class ChromaStore:
    def __init__(self, path: Path, collection: str):
        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.PersistentClient(path=str(path),
                                                settings=Settings(anonymized_telemetry=False))
        self.name = collection

    def _collection(self, model_id: str | None = None, reset: bool = False):
        if reset:
            try:
                self.client.delete_collection(self.name)
            except Exception:
                pass
        meta = {"hnsw:space": "cosine"}
        if model_id:
            meta["model_id"] = model_id
        # embedding_function=None: we always supply our own vectors
        return self.client.get_or_create_collection(self.name, metadata=meta,
                                                    embedding_function=None)

    def sync(self, ids, vectors, docs, metas, model_id):
        col = self._collection()
        reset = (col.metadata or {}).get("model_id") not in (None, model_id)
        col = self._collection(model_id, reset=reset)
        current = set(col.get(include=[])["ids"])
        stale = list(current - set(ids))
        for i in range(0, len(stale), 5000):
            col.delete(ids=stale[i:i + 5000])
        bs = 2000
        for i in range(0, len(ids), bs):
            col.upsert(ids=ids[i:i + bs], embeddings=vectors[i:i + bs].tolist(),
                       documents=docs[i:i + bs], metadatas=metas[i:i + bs])
        return {"upserted": len(ids), "deleted_stale": len(stale), "reset": reset,
                "count": col.count()}

    def query(self, vector, k, where=None):
        res = self._collection().query(query_embeddings=[vector.tolist()], n_results=k,
                                       where=where or None,
                                       include=["documents", "metadatas", "distances"])
        return [{"chunk_id": cid, "score": round(1 - dist, 4), "text": doc, **meta}
                for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                                res["metadatas"][0], res["distances"][0])]

    def count(self):
        return self._collection().count()


class FaissStore:
    def __init__(self, path: Path, collection: str):
        self.dir = path / collection
        self.dir.mkdir(parents=True, exist_ok=True)
        self._index = None
        self._meta: pd.DataFrame | None = None

    def sync(self, ids, vectors, docs, metas, model_id):
        import faiss

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        faiss.write_index(index, str(self.dir / "index.faiss"))
        meta = pd.DataFrame(metas)
        meta.insert(0, "chunk_id", ids)
        meta["text"] = docs
        meta.to_parquet(self.dir / "meta.parquet", index=False)
        (self.dir / "info.json").write_text(json.dumps({"model_id": model_id, "count": len(ids)}))
        return {"upserted": len(ids), "deleted_stale": 0, "reset": True, "count": index.ntotal}

    def _load(self):
        import faiss

        if self._index is None:
            self._index = faiss.read_index(str(self.dir / "index.faiss"))
            self._meta = pd.read_parquet(self.dir / "meta.parquet")

    def query(self, vector, k, where=None):
        self._load()
        # exact search; with a filter, over-fetch then filter (fine at this scale)
        fetch = k if not where else min(self._index.ntotal, k * 50)
        scores, idx = self._index.search(vector.reshape(1, -1).astype(np.float32), fetch)
        out = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            row = self._meta.iloc[int(i)].to_dict()
            if where and any(row.get(key) != val for key, val in where.items()):
                continue
            out.append({**row, "score": round(float(s), 4)})
            if len(out) == k:
                break
        return out

    def count(self):
        self._load()
        return self._index.ntotal


def get_store(cfg: Config) -> VectorStore:
    vs = cfg.vector_store
    root = cfg.paths.vector_dir / vs.backend
    if vs.backend == "chroma":
        return ChromaStore(root, vs.collection)
    if vs.backend == "faiss":
        return FaissStore(root, vs.collection)
    raise ValueError(f"unknown vector store backend {vs.backend}")
