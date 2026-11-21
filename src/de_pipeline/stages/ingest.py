"""Stage 1 - Ingestion: land raw files immutably in the lake, then load them into bronze.

* Raw zone: <lake_uri>/raw/<source>/<batch_id>/<file> (local disk or S3; never modified)
* Idempotent: a file whose sha256 is already in the ingestion manifest is skipped
* Robust download: streaming, retries with exponential backoff, timeout
* Bronze keeps the data *as received* (all strings) + lineage columns
"""
from __future__ import annotations

import codecs
import hashlib
import io
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .. import lakehouse, storage
from ..config import Config, SourceConfig
from ..log import get_logger

log = get_logger(__name__)


def _cp1252_fallback(err: UnicodeDecodeError):
    """Decode bytes that are invalid UTF-8 as cp1252 (the source mixes encodings)."""
    bad = err.object[err.start:err.end]
    return bad.decode("cp1252", errors="replace"), err.end


codecs.register_error("cp1252_fallback", _cp1252_fallback)


def _download(uri: str, dest: Path, retries: int = 4, timeout: int = 60) -> None:
    for attempt in range(1, retries + 1):
        try:
            with requests.get(uri, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for block in r.iter_content(1 << 16):
                        f.write(block)
                tmp.rename(dest)
                return
        except requests.RequestException as e:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            log.warning("download failed (%s), retry %d/%d in %ds", e, attempt, retries, wait)
            time.sleep(wait)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_source(cfg: Config, src: SourceConfig, batch_id: str) -> Path:
    """Download/copy the source into local staging; it is promoted to the raw zone once new."""
    landing = cfg.paths.data_dir / "staging" / src.name / batch_id
    landing.mkdir(parents=True, exist_ok=True)
    dest = landing / (Path(src.uri).name or "data.csv")
    if src.type == "http_csv":
        log.info("downloading %s", src.uri)
        _download(src.uri, dest)
    elif src.type == "local_csv":
        shutil.copyfile(src.uri, dest)
    else:
        raise ValueError(f"unknown source type {src.type}")
    return dest


def read_raw_csv(path: Path) -> pd.DataFrame:
    text = path.read_bytes().decode("utf-8", errors="cp1252_fallback")
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


def run(cfg: Config, run_id: str) -> dict:
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metrics: dict = {"batch_id": batch_id, "sources": {}}
    for src in cfg.sources:
        raw_path = fetch_source(cfg, src, batch_id)
        sha = _sha256(raw_path)
        with lakehouse.metadata_db(cfg) as con:
            seen = con.execute("SELECT 1 FROM ingestion_manifest WHERE source=? AND sha256=?",
                               [src.name, sha]).fetchone()
        if seen and lakehouse.exists(cfg, "bronze", "articles"):
            log.info("[%s] unchanged (sha256=%s...) - skipping load", src.name, sha[:12])
            shutil.rmtree(raw_path.parent)
            metrics["sources"][src.name] = {"status": "skipped_unchanged", "sha256": sha}
            continue

        raw_uri = storage.put_file(cfg, raw_path, "raw", src.name, batch_id, raw_path.name)
        log.info("[%s] landed raw file -> %s", src.name, raw_uri)
        df = read_raw_csv(raw_path)
        bronze = pd.DataFrame({
            "raw_text": df[src.text_column],
            "raw_label": df[src.label_column] if src.label_column else "",
            "source": src.name,
            "source_row": range(len(df)),
            "batch_id": batch_id,
            "source_sha256": sha,
            "ingested_at": datetime.now(timezone.utc),
        })
        stats = lakehouse.write_table(
            cfg, "bronze", "articles", bronze, mode="append",
            description="Raw articles exactly as received, one row per source record, with lineage columns")
        with lakehouse.metadata_db(cfg) as con:
            con.execute("INSERT INTO ingestion_manifest VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        [src.name, src.uri, sha, raw_path.stat().st_size, len(df), raw_uri,
                         batch_id, datetime.now(timezone.utc)])
        size = raw_path.stat().st_size
        shutil.rmtree(raw_path.parent)  # staging is transient; the raw zone is the copy of record
        log.info("[%s] landed %d rows -> bronze.articles v%s", src.name, len(df), stats["version"])
        metrics["sources"][src.name] = {"status": "loaded", "rows": len(df), "sha256": sha,
                                        "bytes": size, "raw_uri": raw_uri, **stats}
    return metrics
