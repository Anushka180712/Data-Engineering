import shutil
from pathlib import Path

import pytest
import yaml

from de_pipeline.config import load_config

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Isolated config: local fixture source, LSA embeddings, temp data dir."""
    monkeypatch.delenv("DE_CONFIG", raising=False)
    shutil.copy(FIXTURES / "sample_news.csv", tmp_path / "sample_news.csv")
    raw = {
        "paths": {"data_dir": "data", "lakehouse_dir": "data/lakehouse",
                  "warehouse": "data/warehouse.duckdb", "vector_dir": "data/vectorstore",
                  "artifacts_dir": "data/artifacts"},
        "sources": [{"name": "fixture", "type": "local_csv", "uri": "sample_news.csv",
                     "text_column": "news", "label_column": "type"}],
        "cleaning": {"min_body_chars": 100,
                     "allowed_categories": ["business", "entertainment", "politics", "sport", "tech"]},
        "chunking": {"chunk_size_tokens": 30, "chunk_overlap_tokens": 10, "min_chunk_tokens": 5},
        "embedding": {"backend": "lsa", "lsa_dimensions": 8},
        "vector_store": {"backend": "chroma", "collection": "test"},
        "quality": {"fail_on_error": True},
    }
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path, root=tmp_path)
