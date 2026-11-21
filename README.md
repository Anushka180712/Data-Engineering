# Data Engineering: from raw text to vectors

An end-to-end, runnable data engineering project. It takes a public news dataset from a raw HTTP
source all the way to a searchable **vector store**. Along the way the data passes through a
**data lake** (local disk or **S3**) holding a **Delta Lake lakehouse** with a medallion
architecture (bronze, silver, gold). The lake is registered in a catalog that can be exported to a
**Hive Metastore or AWS Glue**. The pipeline also has data-quality gates, SQL access, incremental
embeddings and retrieval evaluation.

Everything runs locally on a laptop. There are **no paid APIs and no API keys**.

```bash
pip install -e ".[dev,neural]"     # or `make install` for the dependency-light version
de-pipeline run                    # ingest -> clean -> chunk -> embed -> index  (~20 s)
de-pipeline search "rising oil prices hurt airline profits"
```

Want the lake on S3? `make lake-up && make run-s3` puts the same lake on MinIO, an S3-compatible
object store running in Docker.

---

## Architecture

```mermaid
flowchart LR
    subgraph SRC["Source"]
        A["BBC News CSV<br/>(HTTP)"]
    end

    subgraph LAKE["Data lake: local disk or S3 / MinIO  (lake_uri)"]
        direction LR
        RAW[/"raw/ zone<br/>immutable source files<br/>sha256 fingerprinted"/]
        subgraph LH["Lakehouse: Delta Lake tables (Parquet + transaction log)"]
            direction LR
            BR[("🥉 bronze.articles<br/>as received<br/>+ lineage cols")]
            SI[("🥈 silver.articles<br/>clean, conformed,<br/>deduplicated")]
            GC[("🥇 gold.chunks<br/>retrieval units")]
            GE[("🥇 gold.embeddings<br/>vector cache")]
            BR -- "clean + DQ gate<br/>MERGE on doc_id" --> SI
            SI -- "chunk" --> GC
            GC -- "embed new<br/>chunks only" --> GE
        end
        RAW --> BR
    end

    subgraph META["Catalog & metadata (DuckDB)"]
        CAT["catalog<br/>(layer, location,<br/>schema, version)"]
        OPS["pipeline_runs<br/>dq_results<br/>ingestion_manifest"]
    end

    HMS["Hive Metastore / Glue / Unity<br/>(DDL export, optional)"]
    ENG["SQL engines<br/>DuckDB here; Spark, Trino,<br/>Athena on the same files"]

    subgraph VS["Vector store"]
        CH[("ChromaDB<br/>HNSW + filters")]
        FA[("FAISS<br/>exact")]
    end

    S["semantic search<br/>+ retrieval eval"]

    A -- "download, retry,<br/>checksum" --> RAW
    LH -. "registered in" .-> CAT
    CAT -. "catalog --ddl" .-> HMS
    LH -.-> ENG
    GC & GE -- "sync: upsert +<br/>delete stale" --> CH
    GC & GE -.-> FA
    CH & FA --> S
```

Orchestration: each stage is an **idempotent function** `(config, run_id) -> metrics`. The bundled
CLI runs them as a DAG, and [`orchestration/airflow_dag.py`](orchestration/airflow_dag.py) wraps
the very same functions as Airflow tasks with retries.

```mermaid
flowchart LR
    ingest --> clean --> chunk --> embed --> index
```

---

## Data lake and lakehouse, explained

### What is a data lake?
A data lake is cheap, scalable **storage** where data lands in open file formats, usually before
anyone knows exactly how it will be used. Storage and compute are separate: files sit in the lake,
and any engine can read them.

A lake is built from a few layers:

| Layer | Question it answers | Options | In this project |
|---|---|---|---|
| **Storage** | Where do the bytes live? | HDFS, AWS S3, Azure ADLS, Google GCS, MinIO | **Local disk or S3 / MinIO** (one setting: `lake_uri`) |
| **File format** | How is one file laid out? | CSV, JSON, Avro, ORC, **Parquet** | Raw CSV in `raw/`, **Parquet** for every table |
| **Table format** | How do many files act as one table, with safe updates? | Plain Hive-style folders, **Delta Lake**, Apache Iceberg, Apache Hudi | **Delta Lake** |
| **Catalog / metastore** | Which tables exist, where are they, what is their schema? | **Hive Metastore**, AWS Glue, Unity Catalog, Nessie | DuckDB `catalog` table, with **Hive / Glue DDL export** |
| **Compute / query engine** | Who reads and transforms the data? | MapReduce, **Hive**, Spark, Trino/Presto, Athena, DuckDB | Python + **DuckDB** (Spark, Trino and Athena can read the same tables) |
| **Orchestration** | What runs when, in what order, with retries? | Oozie, Airflow, Dagster, Prefect | CLI DAG runner + **Airflow DAG** |

### From Hadoop to the lakehouse
1. **Hadoop era (around 2008 to 2015).** Data lived in **HDFS** on the same machines that ran
   **MapReduce** and later **Hive**. Hive added SQL on top, plus the **Hive Metastore**, which maps a
   table name to a folder of files and a schema. A Hive "table" is just a directory such as
   `/warehouse/articles/`. Storage and compute were tied to one cluster, so scaling storage meant
   buying compute too.
2. **Cloud data lake.** **S3** (or ADLS, GCS) replaced HDFS. Storage became effectively unlimited,
   pay-per-GB and separate from compute. Spark, Hive, Presto and Athena read S3 through the Hadoop
   **S3A** connector (`s3a://`). The Hive Metastore stayed, often as **AWS Glue**. The weakness also
   stayed: a folder of files has **no transactions**. A failed job can leave half-written data,
   readers can see partial writes, and updates or deletes mean rewriting whole partitions.
3. **Lakehouse (today).** Open **table formats** (Delta Lake, Iceberg, Hudi) add a **transaction
   log** next to the Parquet files. That brings warehouse features to lake storage: ACID commits,
   schema enforcement, `MERGE`, `UPDATE` and `DELETE`, **time travel**, and an audit history. The
   files stay open and stay in the lake. This project is built at this stage.

### How Hadoop / Hive concepts map to this project

| Hadoop / Hive world | Cloud equivalent | This project |
|---|---|---|
| HDFS (`hdfs://namenode/...`) | S3 / ADLS / GCS (`s3a://`) | `lake_uri`: `data/lake` or `s3://lakehouse` ([`storage.py`](src/de_pipeline/storage.py)) |
| HDFS landing directory | S3 raw bucket or prefix | `raw/<source>/<batch_id>/` zone, write-once |
| Hive external table on a folder | Glue table on an S3 prefix | Delta table at `<lake_uri>/<layer>/<table>` |
| Hive Metastore | AWS Glue Data Catalog / Unity Catalog | `catalog` table, plus `de-pipeline catalog --ddl` |
| `INSERT OVERWRITE` a partition | same | Delta `MERGE` upserts and `overwrite` as atomic commits |
| (no equivalent in Hive) | (no equivalent in plain S3) | Delta **time travel** and commit **history** |
| Hive / MapReduce / Spark jobs | EMR, Databricks, Athena | Python stages + DuckDB SQL |
| Oozie workflows | Airflow (MWAA), Step Functions | CLI DAG + [Airflow DAG](orchestration/airflow_dag.py) |

**Why there's no Hadoop cluster here:** about 2,000 articles is a laptop-sized dataset, and running
HDFS, YARN and a Hive server would add a lot of setup without teaching anything new. The design
carries over directly, though. The storage layer is an interface (local filesystem or S3), the
tables are open Delta tables, and `catalog --ddl` produces the exact statements that register them
in a Hive Metastore or Glue. On EMR or Databricks you would point `lake_uri` at your bucket, run
the same pipeline, and query the tables from Spark or Hive.

### The lake layout

```text
<lake_uri>/                         data/lake/   or   s3://lakehouse/
├── raw/bbc_news/<batch_id>/dataset.csv     # raw zone: original bytes, never modified
├── bronze/articles/                        # Delta table
│   ├── _delta_log/00000000000000000000.json    # transaction log (ACID, history, time travel)
│   └── part-00000-....snappy.parquet           # data files
├── silver/articles/
├── gold/chunks/
└── gold/embeddings/
```

### Medallion layers

| Layer | Table | Grain | Write pattern | What happens |
|---|---|---|---|---|
| Raw zone | `raw/...` | one file per batch | write-once | Original bytes, sha256-fingerprinted, logged in `ingestion_manifest` |
| 🥉 Bronze | `bronze.articles` | one row per source record | **append** | Data exactly as received (all strings) plus `source`, `batch_id`, `source_sha256`, `ingested_at` lineage |
| 🥈 Silver | `silver.articles` | one row per unique article | **MERGE** on `doc_id` (updates only when content changed) | Encoding repair, Unicode normalisation, title/body parsing, label conforming, filtering, dedup, derived features, **DQ gate** |
| 🥇 Gold | `gold.chunks` | one row per chunk | overwrite (fully derived) | Sentence-aware overlapping chunks, title-enriched text for embedding, deterministic `chunk_id` |
| 🥇 Gold | `gold.embeddings` | one row per (model, text) | **MERGE** on `embedding_key` | Content-addressed embedding cache. Re-runs embed only new or changed chunks |

**Why Delta Lake?** It is an open table format on plain Parquet that gives ACID commits, schema
enforcement, `MERGE` upserts and **time travel**. It's written here with
[delta-rs](https://github.com/delta-io/delta-rs), so there's no Spark or JVM to run locally. The
same tables can be read by Spark, Databricks, Trino, Athena or DuckDB. Iceberg would work equally
well; Delta was chosen because delta-rs makes it fully usable from plain Python.

### Running the lake on S3 (MinIO)

```bash
make lake-up       # docker compose: MinIO + create bucket "lakehouse"
make run-s3        # same pipeline, lake_uri = s3://lakehouse
# or run everything in containers:
docker compose run --rm pipeline
```
Open the MinIO console at http://localhost:9001 (login `minioadmin` / `minioadmin`) to browse
`raw/`, `bronze/`, `silver/` and `gold/`, including each table's `_delta_log/`.

For **AWS S3**, set `lake_uri: s3://your-bucket/lakehouse` and the usual `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY` and `AWS_REGION`, and leave `AWS_ENDPOINT_URL` unset. The S3 path is
covered by an automated test ([`tests/test_s3_lake.py`](tests/test_s3_lake.py)) that runs the
whole pipeline against an in-process S3 server.

### Catalog and Hive Metastore / Glue registration
Every table write registers the table (layer, location, format, Delta version, row count, schema
and description) in a `catalog` table:

```text
$ de-pipeline catalog
lake: s3://lakehouse

table            format  version  rows  description
---------------  ------  -------  ----  -----------------------------------------------------------
bronze.articles  delta   0        2225  Raw articles exactly as received, one row per source record...
silver.articles  delta   0        2091  Cleaned, conformed and deduplicated articles (one row per u...
gold.chunks      delta   0        6074  Retrieval units: sentence-aware overlapping chunks of silve...
gold.embeddings  delta   0        6024  Embedding cache keyed by (model_id, text_hash); vectors are...
```

`de-pipeline catalog --ddl` generates the statements that register the same files in a **Hive
Metastore, AWS Glue or Unity Catalog** from Spark SQL. No data is copied: the metastore stores only
the schema and location, exactly like a Hive external table.

```sql
CREATE DATABASE IF NOT EXISTS lakehouse_gold;

-- gold.embeddings: 6024 rows, Delta version 0
CREATE TABLE IF NOT EXISTS lakehouse_gold.embeddings (
  `embedding_key` STRING,
  `text_hash` STRING,
  `model_id` STRING,
  `dim` INT,
  `vector` ARRAY<FLOAT>,
  `embedded_at` TIMESTAMP
)
USING DELTA
LOCATION 's3a://lakehouse/gold/embeddings'
COMMENT 'Embedding cache keyed by (model_id, text_hash); vectors are L2-normalised';
```

### SQL over the lake
DuckDB is the query engine. Each table is exposed as `<layer>_<table>`:

```text
$ de-pipeline sql "SELECT category, count(*) docs, round(avg(word_count)) avg_words
                   FROM silver_articles GROUP BY 1 ORDER BY 2 DESC"
category       docs  avg_words
-------------  ----  ---------
business       500   323.0
sport          493   322.0
politics       395   453.0
entertainment  366   328.0
tech           337   512.0
```

### Time travel and audit
These come from Delta's commit log:

```text
$ de-pipeline history silver.articles
version  timestamp_utc        operation  metrics
-------  -------------------  ---------  ------------------------------------------------
1        2026-11-21 16:06:59  MERGE      {"num_source_rows": 2091, "num_target_rows_inserted": 0, ...
0        2026-11-21 16:05:56  WRITE      {"num_added_files": 1, ...
```
```python
from de_pipeline import lakehouse
from de_pipeline.config import load_config
lakehouse.read_table(load_config(), "silver", "articles", version=0)   # read an older snapshot
```

---

## Pipeline stages

### 1. Ingestion ([`stages/ingest.py`](src/de_pipeline/stages/ingest.py))
- Streaming HTTP download with timeout and **exponential-backoff retries**. Writes go to a `.part` file in local staging and are renamed atomically.
- The file is **sha256-fingerprinted**. If the same content was already loaded, the batch is skipped, so re-running never double-loads.
- New files are copied to the lake's **raw zone** (`<lake_uri>/raw/<source>/<batch_id>/`, local or S3), which is the copy of record, and then loaded into bronze.
- **Mixed-encoding handling:** the source file is mostly UTF-8 but contains stray cp1252 bytes. A custom codec error handler decodes each invalid byte as cp1252 instead of dropping or replacing it.

### 2. Cleaning and transformation ([`stages/clean.py`](src/de_pipeline/stages/clean.py))
- Repairs **mojibake** (`Â£3.2bn` becomes `£3.2bn`), applies Unicode NFKC, strips control characters and normalises whitespace and paragraphs.
- Parses semi-structured text into `title` and `body`, and conforms labels (`" TECH "` becomes `tech`).
- Filters junk and **deduplicates** on a content hash. The raw dump has **134 duplicate articles** out of 2,225.
- Derives `word_count`, `char_count`, `paragraph_count` and a deterministic `doc_id`.

### 3. Data-quality gate ([`quality.py`](src/de_pipeline/quality.py))
A small declarative check library in the style of Great Expectations or dbt tests: `not_null`, `unique`,
`accepted_values`, `min_length`, `row_count_between` and `no_encoding_artifacts`. Error-severity
failures **abort the run before bad data reaches silver**, and every result is persisted to `dq_results`.

```text
DQ [silver.articles] unique:doc_id               PASS (observed=0)
DQ [silver.articles] accepted_values:category    PASS (observed=ok)
DQ [silver.articles] no_encoding_artifacts:body  PASS (observed=0)
```

### 4. Chunking ([`stages/chunk.py`](src/de_pipeline/stages/chunk.py))
A sentence-aware, token-budgeted sliding window, written from scratch with no LangChain:
- Chunks never cut a sentence in half. Only a single sentence over the budget gets hard-split.
- Consecutive chunks share about 40 tokens of whole sentences, so facts on a boundary can be found from either side.
- Tiny tail chunks are merged into the previous chunk.
- **Contextual enrichment:** the article title is prepended to the text that gets embedded.
- Result: 2,091 articles become **6,074 chunks** (median 165 tokens, p95 179).

### 5. Embeddings ([`embedders.py`](src/de_pipeline/embedders.py), [`stages/embed.py`](src/de_pipeline/stages/embed.py))
| Backend | Model | Dim | Notes |
|---|---|---|---|
| `sentence-transformers` (default with `[neural]`) | `all-MiniLM-L6-v2` | 384 | Neural, runs on CPU, downloads once from Hugging Face |
| `lsa` (automatic fallback) | TF-IDF (1-2 grams) + TruncatedSVD | 256 | Classic, fully offline, fitted on the corpus and persisted |

- Vectors are L2-normalised, so inner product equals cosine similarity.
- Embeddings are **content-addressed** by `(model_id, sha256(text))`. Re-runs embed only new or changed chunks, and switching models keeps both vector sets.
- The chosen model is persisted so queries are always embedded with the same model as the corpus.

### 6. Vector store ([`vectorstores.py`](src/de_pipeline/vectorstores.py), [`stages/index.py`](src/de_pipeline/stages/index.py))
- **ChromaDB** (default) is persistent, uses an HNSW index with cosine distance, and supports metadata filters (`--category tech`).
- **FAISS** uses an exact `IndexFlatIP` with a Parquet metadata sidecar.
- The store is **synchronised** with gold: current chunks are upserted, deleted chunks are removed, and the collection is rebuilt automatically if the embedding model changes.

## Semantic search demo

```text
$ de-pipeline search "rising oil prices hurt airline profits" -k 3

1. [0.824] (business) Oil prices fall back from highs
     Oil prices retreated from four-month highs in early trading on Tuesday after producers' cartel
     Opec said it was now unlikely to cut production...
2. [0.792] (business) Oil prices reach three-month low
3. [0.787] (business) Crude oil prices back above $50

$ de-pipeline search "new mobile phone technology" --category tech -k 3
1. [0.753] (tech) Mobiles rack up 20 years of use
2. [0.727] (tech) Finding new homes for old phones
...
```

## Retrieval evaluation

`de-pipeline eval` runs a self-supervised benchmark that needs no hand-labelled queries. It samples 200
articles, uses a sentence from the middle of each one as the query, and checks whether that article
comes back.

| Embeddings | Vector store | hit_rate@5 | MRR@5 | category precision@5 |
|---|---|---|---|---|
| LSA (256-d) | Chroma (HNSW, approximate) | 0.955 | 0.869 | 0.834 |
| LSA (256-d) | FAISS (exact) | 0.955 | 0.898 | 0.868 |

The gap between the two rows shows the trade-off between approximate (HNSW) and exact nearest-neighbour search.
Run it with `[neural]` installed to compare MiniLM against LSA on the same benchmark.

## Run it

**Locally**
```bash
python -m venv .venv && source .venv/bin/activate
make install-neural        # or: make install   (no torch; uses the LSA embedder)
make run
make search Q="interest rates and inflation"
make eval
make test
```

**Docker (lake on MinIO / S3)**
```bash
docker compose run --rm pipeline                         # starts MinIO, creates the bucket, runs the pipeline
docker compose run --rm pipeline search "election campaign"
docker compose run --rm pipeline sql "SELECT * FROM catalog"
docker compose run --rm pipeline catalog --ddl           # Hive Metastore / Glue DDL
```

**Useful commands**
```bash
de-pipeline run --stages chunk,embed,index    # re-run part of the DAG
de-pipeline runs                              # recent stage runs, status, durations
de-pipeline history gold.embeddings           # Delta commit log
de-pipeline catalog --ddl                     # register tables in Hive Metastore / Glue
DE_VECTOR_STORE__BACKEND=faiss de-pipeline run --stages index   # any config key via env
```

## Configuration

Everything is in [`config/pipeline.yaml`](config/pipeline.yaml): sources, cleaning rules, chunk
sizes, embedding backend, vector store and `lake_uri` (local path or `s3://`). Any key can be overridden with an environment variable in
the form `DE_<SECTION>__<KEY>`. To add a source, append it under `sources:` (`http_csv` or
`local_csv`) and name its text and label columns.

## Project layout

```text
├── config/pipeline.yaml          # single source of configuration
├── src/de_pipeline/
│   ├── cli.py                    # de-pipeline run | search | eval | sql | catalog | history | runs
│   ├── pipeline.py               # DAG runner, run metadata
│   ├── storage.py                # lake storage: local filesystem or S3 / MinIO
│   ├── lakehouse.py              # Delta tables, catalog, Hive DDL export, DuckDB SQL, metadata
│   ├── quality.py                # data-quality checks
│   ├── embedders.py              # sentence-transformers / LSA
│   ├── vectorstores.py           # Chroma / FAISS
│   ├── search.py                 # semantic search + evaluation
│   └── stages/                   # ingest, clean, chunk, embed, index
├── orchestration/airflow_dag.py  # same stages as an Airflow DAG
├── tests/                        # unit + end-to-end tests (fixture data, runs offline)
├── Dockerfile, docker-compose.yml (pipeline + MinIO), Makefile
└── .github/workflows/ci.yml      # pytest on every push
```

## Engineering principles shown here

- **Idempotent, re-runnable stages:** checksum-based ingestion, MERGE upserts, deterministic IDs and content-addressed caches.
- **Immutability and lineage:** raw files are never modified, and every bronze row knows its source file and batch.
- **Fail fast on bad data:** quality gates run before data is promoted to the next layer.
- **Observability:** per-stage metrics, durations and status in `pipeline_runs`, plus Delta commit history.
- **Pluggable components:** the source, embedding backend and vector store are each one config line.
- **Storage-agnostic:** the same code runs on local disk and on S3, so moving to the cloud is a configuration change.
- **Tested:** unit tests, plus end-to-end tests that run the whole pipeline on fixture data for both vector stores and on S3.

## Dataset

BBC News dataset: 2,225 articles in 5 categories (D. Greene and P. Cunningham, *Practical Solutions to
the Problem of Diagonal Dominance in Kernel Document Clustering*, ICML 2006). It is used here for
non-commercial, educational purposes and is downloaded at runtime, not redistributed.
