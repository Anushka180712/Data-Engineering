"""Stage 3 - Chunking: silver.articles -> gold.chunks.

Sentence-aware, token-budgeted sliding-window chunker (no framework dependency):
  * never cuts a sentence in half unless one sentence alone exceeds the budget
  * consecutive chunks overlap by ~`chunk_overlap_tokens` whole sentences, so context
    that straddles a boundary is retrievable from either side
  * tiny trailing chunks are merged into the previous chunk
  * contextual enrichment: the article title is prepended to the text that gets embedded
  * deterministic chunk_id = hash(doc_id, position, text) -> stable upserts downstream
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd

from .. import lakehouse, quality as dq
from ..config import ChunkingConfig, Config
from ..log import get_logger

log = get_logger(__name__)

_SENT_SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[A-Z0-9\"'(\[])")


def n_tokens(text: str) -> int:
    """Cheap whitespace token estimate (model-agnostic; ~0.75 of a BPE token count)."""
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    out = []
    for para in text.split("\n\n"):
        out.extend(s.strip() for s in _SENT_SPLIT.split(para) if s.strip())
    return out


def _hard_split(sentence: str, size: int) -> list[str]:
    words = sentence.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def chunk_text(text: str, size: int, overlap: int, min_tokens: int) -> list[str]:
    if overlap >= size:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
    sentences: list[str] = []
    for s in split_sentences(text):
        sentences.extend(_hard_split(s, size) if n_tokens(s) > size else [s])

    chunks: list[list[str]] = []
    window: list[str] = []
    window_tokens = 0
    for s in sentences:
        t = n_tokens(s)
        if window and window_tokens + t > size:
            chunks.append(window)
            # carry the last sentences (up to `overlap` tokens) into the next window
            carry, carry_tokens = [], 0
            for prev in reversed(window):
                pt = n_tokens(prev)
                if carry_tokens + pt > overlap:
                    break
                carry.insert(0, prev)
                carry_tokens += pt
            window, window_tokens = carry, carry_tokens
        window.append(s)
        window_tokens += t
    if window:
        # don't emit a window that is pure overlap with the previous chunk
        if not chunks or window[-1] != chunks[-1][-1]:
            chunks.append(window)

    texts = [" ".join(c) for c in chunks]
    if len(texts) > 1 and n_tokens(texts[-1]) < min_tokens:
        tail = chunks.pop()
        new_part = [s for s in tail if s not in chunks[-1]]
        chunks[-1].extend(new_part)
        texts = [" ".join(c) for c in chunks]
    return texts


def build_chunks(silver: pd.DataFrame, c: ChunkingConfig) -> pd.DataFrame:
    rows = []
    for doc in silver.itertuples(index=False):
        pieces = chunk_text(doc.body, c.chunk_size_tokens, c.chunk_overlap_tokens, c.min_chunk_tokens)
        for i, text in enumerate(pieces):
            embed_text = f"{doc.title}\n\n{text}" if c.prepend_title else text
            rows.append({
                "chunk_id": hashlib.sha1(f"{doc.doc_id}|{i}|{text}".encode()).hexdigest()[:20],
                "doc_id": doc.doc_id,
                "chunk_index": i,
                "n_chunks": len(pieces),
                "title": doc.title,
                "category": doc.category,
                "text": text,
                "embed_text": embed_text,
                "token_count": n_tokens(text),
                "text_hash": hashlib.sha256(embed_text.encode()).hexdigest(),
            })
    return pd.DataFrame(rows)


def run(cfg: Config, run_id: str) -> dict:
    silver = lakehouse.read_table(cfg, "silver", "articles")
    chunks = build_chunks(silver, cfg.chunking)
    results = dq.run_checks(chunks, [
        dq.row_count_between(len(silver)),           # at least one chunk per doc
        dq.unique("chunk_id"), dq.not_null("text"),
        dq.Check("max_tokens<=size*1.5",
                 lambda df: ((k := int((df.token_count > cfg.chunking.chunk_size_tokens * 1.5).sum())) == 0, k),
                 "warn"),
    ], "gold.chunks", cfg.quality.fail_on_error)
    lakehouse.record_dq(cfg, run_id, "gold.chunks", results)
    stats = lakehouse.write_table(
        cfg, "gold", "chunks", chunks, mode="overwrite",
        description="Retrieval units: sentence-aware overlapping chunks of silver.articles")
    tc = chunks["token_count"]
    metrics = {"docs": len(silver), "chunks": len(chunks),
               "chunks_per_doc": round(len(chunks) / max(len(silver), 1), 2),
               "tokens_p50": int(tc.median()), "tokens_p95": int(tc.quantile(0.95)),
               "tokens_max": int(tc.max()), **stats}
    log.info("gold.chunks: %d docs -> %d chunks (p50=%d tokens) v%s", len(silver), len(chunks),
             metrics["tokens_p50"], stats["version"])
    return metrics
