"""Lakehouse storage layer.

* Tables are stored in the **Delta Lake** open table format (via delta-rs, no Spark/JVM needed):
  ACID commits, schema enforcement, MERGE/upserts and time travel on plain Parquet files.
* Tables are organised in a **medallion architecture**: bronze (raw, as ingested) ->
  silver (cleaned, conformed, deduplicated) -> gold (consumption-ready: chunks, embeddings).
* A small **catalog** (in DuckDB) registers every table with its layer, location, version,
  row count and schema, the way Unity Catalog / Glue / Hive Metastore would.
* **DuckDB** is the SQL query engine over the Delta tables (`de-pipeline sql "..."`).
* DuckDB also stores operational metadata: pipeline runs, data-quality results, ingestion manifest.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone

import duckdb
import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

from .config import Config

LAYERS = ("bronze", "silver", "gold")

DDL = """
CREATE TABLE IF NOT EXISTS catalog (
    table_fqn VARCHAR PRIMARY KEY, layer VARCHAR, table_name VARCHAR, format VARCHAR,
    location VARCHAR, version BIGINT, num_rows BIGINT, schema_json VARCHAR,
    description VARCHAR, updated_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id VARCHAR, stage VARCHAR, status VARCHAR,
    started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ, duration_s DOUBLE, metrics JSON
);
CREATE TABLE IF NOT EXISTS dq_results (
    run_id VARCHAR, dataset VARCHAR, check_name VARCHAR, severity VARCHAR,
    passed BOOLEAN, observed VARCHAR, checked_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS ingestion_manifest (
    source VARCHAR, uri VARCHAR, sha256 VARCHAR, bytes BIGINT, rows BIGINT,
    raw_path VARCHAR, batch_id VARCHAR, ingested_at TIMESTAMPTZ
);
"""


# --------------------------------------------------------------------------- metadata DB
@contextmanager
def metadata_db(cfg: Config):
    con = duckdb.connect(str(cfg.paths.warehouse))
    try:
        con.execute(DDL)
        yield con
    finally:
        con.close()


def record_stage(cfg: Config, run_id: str, stage: str, status: str,
                 started: datetime, metrics: dict) -> None:
    finished = datetime.now(timezone.utc)
    with metadata_db(cfg) as con:
        con.execute("INSERT INTO pipeline_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [run_id, stage, status, started, finished,
                     (finished - started).total_seconds(), json.dumps(metrics, default=str)])


def record_dq(cfg: Config, run_id: str, dataset: str, results: list[dict]) -> None:
    now = datetime.now(timezone.utc)
    with metadata_db(cfg) as con:
        con.executemany("INSERT INTO dq_results VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [[run_id, dataset, r["check"], r["severity"], r["passed"],
                          str(r["observed"]), now] for r in results])


# --------------------------------------------------------------------------- delta tables
def table_path(cfg: Config, layer: str, name: str) -> str:
    assert layer in LAYERS, layer
    return str(cfg.paths.lakehouse_dir / layer / name)


def exists(cfg: Config, layer: str, name: str) -> bool:
    return DeltaTable.is_deltatable(table_path(cfg, layer, name))


def read_table(cfg: Config, layer: str, name: str, version: int | None = None) -> pd.DataFrame:
    return DeltaTable(table_path(cfg, layer, name), version=version).to_pandas()


def write_table(cfg: Config, layer: str, name: str, df: pd.DataFrame, *, mode: str = "overwrite",
                merge_keys: list[str] | None = None, update_when: str | None = None,
                description: str = "",
                partition_by: list[str] | None = None) -> dict:
    """Write a DataFrame to a Delta table.

    mode = "overwrite" | "append" | "merge". "merge" upserts on `merge_keys`
    (update matched rows, insert new ones), i.e. an idempotent incremental load.
    `update_when` optionally restricts updates, e.g. "t.content_hash != s.content_hash"
    so unchanged rows aren't rewritten (SCD type-1 on change only).
    Returns commit stats and registers/updates the table in the catalog.
    """
    path = table_path(cfg, layer, name)
    data = pa.Table.from_pandas(df, preserve_index=False)
    stats: dict = {"mode": mode}
    if mode == "merge" and DeltaTable.is_deltatable(path):
        predicate = " AND ".join(f"t.{k} = s.{k}" for k in merge_keys or [])
        res = (DeltaTable(path)
               .merge(data, predicate=predicate, source_alias="s", target_alias="t")
               .when_matched_update_all(predicate=update_when)
               .when_not_matched_insert_all()
               .execute())
        stats.update({k: res.get(k) for k in ("num_target_rows_inserted",
                                              "num_target_rows_updated") if k in res})
    else:
        write_deltalake(path, data, mode="append" if mode == "append" else "overwrite",
                        schema_mode="overwrite" if mode == "overwrite" else None,
                        partition_by=partition_by)
    stats.update(register(cfg, layer, name, description))
    return stats


def register(cfg: Config, layer: str, name: str, description: str = "") -> dict:
    dt = DeltaTable(table_path(cfg, layer, name))
    n_rows = dt.to_pyarrow_dataset().count_rows()
    schema = {f.name: str(f.type) for f in dt.schema().to_arrow()}
    fqn = f"{layer}.{name}"
    with metadata_db(cfg) as con:
        old = con.execute("SELECT description FROM catalog WHERE table_fqn = ?", [fqn]).fetchone()
        con.execute("DELETE FROM catalog WHERE table_fqn = ?", [fqn])
        con.execute("INSERT INTO catalog VALUES (?, ?, ?, 'delta', ?, ?, ?, ?, ?, ?)",
                    [fqn, layer, name, dt.table_uri, dt.version(), n_rows, json.dumps(schema),
                     description or (old[0] if old else ""), datetime.now(timezone.utc)])
    return {"table": fqn, "version": dt.version(), "rows": n_rows}


def history(cfg: Config, layer: str, name: str) -> list[dict]:
    return DeltaTable(table_path(cfg, layer, name)).history()


# --------------------------------------------------------------------------- SQL engine
def sql_connection(cfg: Config) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with every catalog table exposed as `<layer>_<name>`, plus metadata."""
    con = duckdb.connect()
    con.execute(f"ATTACH '{cfg.paths.warehouse.as_posix()}' AS meta (READ_ONLY)")
    for layer, name in con.execute("SELECT layer, table_name FROM meta.catalog").fetchall():
        ds = DeltaTable(table_path(cfg, layer, name)).to_pyarrow_dataset()
        con.register(f"{layer}_{name}", ds)
    for t in ("catalog", "pipeline_runs", "dq_results", "ingestion_manifest"):
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM meta.{t}")
    return con
