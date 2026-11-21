"""Stage 2 - Cleaning & transformation: bronze.articles -> silver.articles.

Transformations
  1. Repair mojibake (UTF-8 text that was mis-decoded as cp1252, e.g. "Â£" -> "£")
  2. Unicode NFKC normalisation, strip control characters, normalise whitespace/paragraphs
  3. Parse semi-structured text into columns: title (first line) + body (rest)
  4. Conform the label column (lower-case, trimmed) to the allowed category set
  5. Filter junk (empty / too-short bodies)
  6. Deduplicate on a content hash (the BBC dump contains ~100 exact duplicates)
  7. Derive features: word/char/paragraph counts, deterministic doc_id
Then validate with data-quality checks and MERGE (upsert) into silver on doc_id.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

import pandas as pd

from .. import lakehouse, quality as dq
from ..config import Config
from ..log import get_logger

log = get_logger(__name__)

_MOJIBAKE = re.compile(r"[ÂÃâ][\x80-\xbf€‚ƒ„…†‡ˆ‰Š‹ŒŽ‘’“”•–—˜™š›œžŸ¡-¿]")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def fix_mojibake(s: str) -> str:
    if not _MOJIBAKE.search(s):
        return s
    try:
        return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Fix fragment by fragment when the whole string can't be round-tripped
        return _MOJIBAKE.sub(lambda m: _try_fix(m.group(0)), s)


def _try_fix(frag: str) -> str:
    try:
        return frag.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return frag


def normalise_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", fix_mojibake(s))
    s = _CTRL.sub("", s).replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [re.sub(r"[ \t ]+", " ", p).strip() for p in re.split(r"\n\s*\n|\n", s)]
    return "\n\n".join(p for p in paragraphs if p)


def split_title_body(text: str) -> tuple[str, str]:
    title, _, body = text.partition("\n\n")
    return title.strip(), body.strip()


def stable_id(*parts: str, n: int = 16) -> str:
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:n]


def transform(bronze: pd.DataFrame, min_body_chars: int) -> tuple[pd.DataFrame, dict]:
    m = {"input_rows": len(bronze)}
    df = bronze.copy()
    df["text"] = df["raw_text"].astype(str).map(normalise_text)
    df[["title", "body"]] = df["text"].apply(lambda t: pd.Series(split_title_body(t)))
    df["category"] = df["raw_label"].astype(str).str.strip().str.lower()

    too_short = df["body"].str.len() < min_body_chars
    m["dropped_too_short"] = int(too_short.sum())
    df = df[~too_short]

    df["content_hash"] = df["body"].map(lambda b: hashlib.sha256(b.lower().encode()).hexdigest())
    dupes = df["content_hash"].duplicated(keep="first")
    m["dropped_duplicates"] = int(dupes.sum())
    df = df[~dupes]

    df["doc_id"] = df["content_hash"].str[:16]
    df["word_count"] = df["body"].str.split().str.len().astype("int32")
    df["char_count"] = df["body"].str.len().astype("int32")
    df["paragraph_count"] = (df["body"].str.count("\n\n") + 1).astype("int32")
    df["processed_at"] = pd.Timestamp.now(tz="UTC")
    out = df[["doc_id", "title", "body", "category", "word_count", "char_count",
              "paragraph_count", "content_hash", "source", "source_row", "batch_id",
              "processed_at"]].reset_index(drop=True)
    m["output_rows"] = len(out)
    return out, m


def silver_checks(cfg: Config) -> list[dq.Check]:
    return [
        dq.row_count_between(1),
        dq.not_null("doc_id"), dq.unique("doc_id"),
        dq.not_null("title"), dq.min_length("title", 3),
        dq.min_length("body", cfg.cleaning.min_body_chars),
        dq.accepted_values("category", cfg.cleaning.allowed_categories),
        dq.no_text_artifacts("body"),
    ]


def run(cfg: Config, run_id: str) -> dict:
    bronze = lakehouse.read_table(cfg, "bronze", "articles")
    # only the latest batch of each source is (re)processed
    latest = bronze.groupby("source")["batch_id"].transform("max")
    bronze = bronze[bronze["batch_id"] == latest]

    silver, metrics = transform(bronze, cfg.cleaning.min_body_chars)
    results = dq.run_checks(silver, silver_checks(cfg), "silver.articles", cfg.quality.fail_on_error)
    lakehouse.record_dq(cfg, run_id, "silver.articles", results)

    stats = lakehouse.write_table(
        cfg, "silver", "articles", silver, mode="merge", merge_keys=["doc_id"],
        update_when="t.content_hash != s.content_hash OR t.category != s.category OR t.title != s.title",
        description="Cleaned, conformed and deduplicated articles (one row per unique article)")
    metrics.update(stats)
    metrics["dq_failed"] = sum(not r["passed"] for r in results)
    metrics["category_counts"] = silver["category"].value_counts().to_dict()
    log.info("silver.articles: %d in -> %d out (%d dupes, %d short) v%s", metrics["input_rows"],
             metrics["output_rows"], metrics["dropped_duplicates"], metrics["dropped_too_short"],
             stats["version"])
    return metrics
