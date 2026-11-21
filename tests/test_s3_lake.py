"""Run the full pipeline against S3-compatible object storage (in-process moto S3 server).

Proves the lake is storage-agnostic: identical code on local disk and on s3:// (AWS S3 / MinIO).
"""
import pytest

moto_server = pytest.importorskip("moto.server")
boto3 = pytest.importorskip("boto3")

from de_pipeline import lakehouse  # noqa: E402
from de_pipeline.pipeline import run_pipeline  # noqa: E402
from de_pipeline.search import Searcher  # noqa: E402


@pytest.fixture
def s3(monkeypatch):
    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    for k, v in {"AWS_ENDPOINT_URL": endpoint, "AWS_ACCESS_KEY_ID": "test",
                 "AWS_SECRET_ACCESS_KEY": "test", "AWS_REGION": "us-east-1"}.items():
        monkeypatch.setenv(k, v)
    client = boto3.client("s3", endpoint_url=endpoint, region_name="us-east-1")
    client.create_bucket(Bucket="test-lake")
    yield client
    server.stop()


def test_pipeline_on_s3(cfg, s3):
    cfg.paths.lake_uri = "s3://test-lake/lakehouse"
    summary = run_pipeline(cfg)
    assert summary["stages"]["ingest"]["sources"]["fixture"]["raw_uri"].startswith(
        "s3://test-lake/lakehouse/raw/fixture/")

    keys = [o["Key"] for o in s3.list_objects_v2(Bucket="test-lake")["Contents"]]
    for prefix in ("lakehouse/raw/", "lakehouse/bronze/articles/_delta_log/",
                   "lakehouse/silver/articles/_delta_log/", "lakehouse/gold/chunks/",
                   "lakehouse/gold/embeddings/"):
        assert any(k.startswith(prefix) for k in keys), prefix
    assert not any((cfg.paths.data_dir / "lake").rglob("*"))   # nothing written to local lake

    assert len(lakehouse.read_table(cfg, "silver", "articles")) == 3
    assert Searcher(cfg).search("battery", k=1, category="tech")[0]["category"] == "tech"
    ddl = lakehouse.metastore_ddl(cfg)
    assert "LOCATION 's3a://test-lake/lakehouse/silver/articles'" in ddl
    assert "USING DELTA" in ddl

    # idempotent re-run against S3
    again = run_pipeline(cfg)["stages"]
    assert again["ingest"]["sources"]["fixture"]["status"] == "skipped_unchanged"
