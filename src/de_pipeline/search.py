"""Semantic search + retrieval evaluation over the vector store."""
from __future__ import annotations

import json
import random
import time

from . import lakehouse
from .config import Config
from .embedders import load_query_embedder
from .vectorstores import get_store


class Searcher:
    def __init__(self, cfg: Config):
        self.embedder = load_query_embedder(cfg)
        self.store = get_store(cfg)

    def search(self, query: str, k: int = 5, category: str | None = None,
               dedupe_docs: bool = True) -> list[dict]:
        vec = self.embedder.embed([query])[0]
        where = {"category": category} if category else None
        hits = self.store.query(vec, k * 4 if dedupe_docs else k, where)
        if dedupe_docs:  # best chunk per article
            seen, out = set(), []
            for h in hits:
                if h["doc_id"] not in seen:
                    seen.add(h["doc_id"])
                    out.append(h)
            hits = out
        return hits[:k]


def evaluate(cfg: Config, n_queries: int = 200, k: int = 5, seed: int = 7) -> dict:
    """Self-supervised retrieval eval (no labelled queries needed).

    For a random sample of articles, use a *paraphrase-free proxy query*: a sentence taken
    from the middle of the article body. We then measure
      * hit_rate@k / MRR  - is the source article retrieved?
      * category_precision@k - share of retrieved articles in the query's category
    """
    silver = lakehouse.read_table(cfg, "silver", "articles")
    rng = random.Random(seed)
    sample = silver.sample(n=min(n_queries, len(silver)), random_state=seed)
    s = Searcher(cfg)
    hits = rr = cat_prec = 0.0
    t0 = time.perf_counter()
    for row in sample.itertuples(index=False):
        sents = [x for x in row.body.replace("\n\n", " ").split(". ") if len(x.split()) >= 8]
        query = rng.choice(sents) if sents else row.title
        res = s.search(query, k=k)
        ids = [r["doc_id"] for r in res]
        if row.doc_id in ids:
            hits += 1
            rr += 1 / (ids.index(row.doc_id) + 1)
        cat_prec += sum(r["category"] == row.category for r in res) / max(len(res), 1)
    n = len(sample)
    report = {"queries": n, "k": k, "hit_rate@k": round(hits / n, 3), "mrr@k": round(rr / n, 3),
              "category_precision@k": round(cat_prec / n, 3),
              "avg_latency_ms": round((time.perf_counter() - t0) / n * 1000, 1),
              "model_id": s.embedder.model_id, "vector_store": cfg.vector_store.backend}
    out = cfg.paths.artifacts_dir / "eval_report.json"
    out.write_text(json.dumps(report, indent=2))
    return report
