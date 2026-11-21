"""End-to-end: fixture CSV -> bronze -> silver -> gold -> embeddings -> vector store -> search."""
import pytest

from de_pipeline import lakehouse
from de_pipeline.pipeline import run_pipeline
from de_pipeline.search import Searcher


@pytest.mark.parametrize("backend", ["chroma", "faiss"])
def test_end_to_end(cfg, backend):
    cfg.vector_store.backend = backend
    summary = run_pipeline(cfg)
    st = summary["stages"]
    assert st["ingest"]["sources"]["fixture"]["rows"] == 5
    assert st["clean"]["output_rows"] == 3              # 1 duplicate + 1 too short removed
    assert st["chunk"]["chunks"] >= 3
    assert st["index"]["count"] == st["chunk"]["chunks"]

    silver = lakehouse.read_table(cfg, "silver", "articles")
    assert "£2bn" in silver.loc[silver.category == "business", "body"].iloc[0]  # mojibake fixed
    assert set(silver.category) == {"business", "sport", "tech"}

    hits = Searcher(cfg).search("striker scored goals in the cup", k=2)
    assert hits[0]["category"] == "sport"
    assert Searcher(cfg).search("battery", k=3, category="tech")[0]["category"] == "tech"

    with lakehouse.metadata_db(cfg) as con:
        tables = {r[0] for r in con.execute("SELECT table_fqn FROM catalog").fetchall()}
        runs = con.execute("SELECT count(*) FROM pipeline_runs WHERE status='success'").fetchone()[0]
    assert tables == {"bronze.articles", "silver.articles", "gold.chunks", "gold.embeddings"}
    assert runs == 5


def test_rerun_is_idempotent(cfg):
    run_pipeline(cfg)
    second = run_pipeline(cfg)["stages"]
    assert second["ingest"]["sources"]["fixture"]["status"] == "skipped_unchanged"
    assert second["embed"]["newly_embedded"] == 0
    assert lakehouse.read_table(cfg, "bronze", "articles").shape[0] == 5   # no double load
    assert lakehouse.read_table(cfg, "silver", "articles").shape[0] == 3


def test_sql_over_lakehouse(cfg):
    run_pipeline(cfg)
    con = lakehouse.sql_connection(cfg)
    n = con.execute("SELECT count(DISTINCT doc_id) FROM gold_chunks").fetchall()[0][0]
    assert n == 3
