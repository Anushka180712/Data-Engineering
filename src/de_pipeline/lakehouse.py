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
import pyarrow.fs as pafs

from .storage import delta_storage_options, filesystem, lake_uri, to_hadoop_uri

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
    """<lake_uri>/<layer>/<table>, e.g. s3://lakehouse/silver/articles"""
    assert layer in LAYERS, layer
    return lake_uri(cfg, layer, name)


def _dt(cfg: Config, layer: str, name: str, version: int | None = None) -> DeltaTable:
    return DeltaTable(table_path(cfg, layer, name), version=version,
                      storage_options=delta_storage_options(cfg))


def exists(cfg: Config, layer: str, name: str) -> bool:
    """A Delta table exists iff its _delta_log/ holds a commit. Checked through the filesystem
    layer (cheap on S3: one LIST call)."""
    fs, base = filesystem(cfg)
    info = fs.get_file_info(pafs.FileSelector(f"{base}/{layer}/{name}/_delta_log",
                                              allow_not_found=True))
    return any(f.path.endswith(".json") for f in info)


def read_table(cfg: Config, layer: str, name: str, version: int | None = None) -> pd.DataFrame:
    return _dt(cfg, layer, name, version).to_pandas()


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
    so = delta_storage_options(cfg)
    if mode == "merge" and exists(cfg, layer, name):
        predicate = " AND ".join(f"t.{k} = s.{k}" for k in merge_keys or [])
        res = (DeltaTable(path, storage_options=so)
               .merge(data, predicate=predicate, source_alias="s", target_alias="t")
               .when_matched_update_all(predicate=update_when)
               .when_not_matched_insert_all()
               .execute())
        stats.update({k: res.get(k) for k in ("num_target_rows_inserted",
                                              "num_target_rows_updated") if k in res})
    else:
        write_deltalake(path, data, mode="append" if mode == "append" else "overwrite",
                        schema_mode="overwrite" if mode == "overwrite" else None,
                        partition_by=partition_by, storage_options=so)
    stats.update(register(cfg, layer, name, description))
    return stats


def _delta_type_str(t) -> str:
    """Delta Lake type JSON -> compact string, e.g. 'string', 'long', 'array<float>'."""
    if isinstance(t, str):
        return t
    if t.get("type") == "array":
        return f"array<{_delta_type_str(t['elementType'])}>"
    return t.get("type", "struct")


def register(cfg: Config, layer: str, name: str, description: str = "") -> dict:
    dt = _dt(cfg, layer, name)
    n_rows = dt.to_pyarrow_dataset().count_rows()
    schema = {f["name"]: _delta_type_str(f["type"])
              for f in json.loads(dt.schema().to_json())["fields"]}
    fqn = f"{layer}.{name}"
    with metadata_db(cfg) as con:
        old = con.execute("SELECT description FROM catalog WHERE table_fqn = ?", [fqn]).fetchone()
        con.execute("DELETE FROM catalog WHERE table_fqn = ?", [fqn])
        con.execute("INSERT INTO catalog VALUES (?, ?, ?, 'delta', ?, ?, ?, ?, ?, ?)",
                    [fqn, layer, name, dt.table_uri, dt.version(), n_rows, json.dumps(schema),
                     description or (old[0] if old else ""), datetime.now(timezone.utc)])
    return {"table": fqn, "version": dt.version(), "rows": n_rows}


def history(cfg: Config, layer: str, name: str) -> list[dict]:
    return _dt(cfg, layer, name).history()


# --------------------------------------------------------------------------- SQL engine
def sql_connection(cfg: Config) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with every catalog table exposed as `<layer>_<name>`, plus metadata."""
    con = duckdb.connect()
    con.execute(f"ATTACH '{cfg.paths.warehouse.as_posix()}' AS meta (READ_ONLY)")
    for layer, name in con.execute("SELECT layer, table_name FROM meta.catalog").fetchall():
        ds = _dt(cfg, layer, name).to_pyarrow_dataset()
        con.register(f"{layer}_{name}", ds)
    for t in ("catalog", "pipeline_runs", "dq_results", "ingestion_manifest"):
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM meta.{t}")
    return con


# --------------------------------------------------------------------------- metastore export
_DELTA_TO_HIVE = {"string": "STRING", "long": "BIGINT", "integer": "INT", "short": "SMALLINT",
                  "byte": "TINYINT", "double": "DOUBLE", "float": "FLOAT", "boolean": "BOOLEAN",
                  "timestamp": "TIMESTAMP", "timestamp_ntz": "TIMESTAMP_NTZ", "date": "DATE",
                  "binary": "BINARY"}


def _hive_type(delta_type: str) -> str:
    if delta_type.startswith("array<"):
        return f"ARRAY<{_hive_type(delta_type[6:-1])}>"
    return _DELTA_TO_HIVE.get(delta_type, delta_type.upper())


def metastore_ddl(cfg: Config) -> str:
    """Spark SQL / Hive Metastore / AWS Glue DDL that registers every lake table.

    Run it in spark-sql (with delta-spark), Databricks or EMR to expose the same physical
    files as `lakehouse_<layer>.<table>` - no data is copied, the metastore only stores
    schema + location (exactly what Hive's EXTERNAL TABLE always did).
    """
    with metadata_db(cfg) as con:
        rows = con.execute("SELECT layer, table_name, schema_json, description, num_rows, version "
                           "FROM catalog ORDER BY layer, table_name").fetchall()
    out = ["-- Generated by `de-pipeline catalog --ddl`.",
           "-- Registers the Delta tables of this lake in a Hive Metastore / Glue / Unity catalog.",
           "-- Requires delta-spark; for S3 configure the Hadoop S3A connector (fs.s3a.*).", ""]
    for layer in LAYERS:
        if any(r[0] == layer for r in rows):
            out.append(f"CREATE DATABASE IF NOT EXISTS lakehouse_{layer};")
    out.append("")
    for layer, name, schema_json, desc, n, version in rows:
        cols = ",\n".join(f"  `{c}` {_hive_type(t)}" for c, t in json.loads(schema_json).items())
        loc = to_hadoop_uri(table_path(cfg, layer, name))
        out.append(f"-- {layer}.{name}: {n} rows, Delta version {version}")
        out.append(f"CREATE TABLE IF NOT EXISTS lakehouse_{layer}.{name} (\n{cols}\n)\n"
                   f"USING DELTA\nLOCATION '{loc}'\nCOMMENT '{(desc or '').replace(chr(39), '')}';\n")
    return "\n".join(out)
