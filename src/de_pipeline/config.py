"""Typed configuration loaded from YAML, overridable via environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SOURCE_ROOT = Path(__file__).resolve().parents[2]   # repo root when installed editable


def default_config_path() -> Path:
    """$DE_CONFIG, else ./config/pipeline.yaml, else the repo's config (editable installs)."""
    if os.environ.get("DE_CONFIG"):
        return Path(os.environ["DE_CONFIG"])
    for candidate in (Path.cwd() / "config" / "pipeline.yaml",
                      SOURCE_ROOT / "config" / "pipeline.yaml"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("config/pipeline.yaml not found: run from the project root "
                            "or set DE_CONFIG=/path/to/pipeline.yaml")


@dataclass
class SourceConfig:
    name: str
    type: str
    uri: str
    text_column: str
    label_column: str | None = None
    license_note: str = ""


@dataclass
class Paths:
    lake_uri: str         # data lake root: a local directory or s3://bucket/prefix
    data_dir: Path        # local working dir (staging downloads)
    warehouse: Path       # DuckDB file: catalog + operational metadata
    vector_dir: Path
    artifacts_dir: Path

    @property
    def lake_is_remote(self) -> bool:
        return "://" in self.lake_uri


@dataclass
class CleaningConfig:
    min_body_chars: int = 200
    allowed_categories: list[str] = field(default_factory=list)


@dataclass
class ChunkingConfig:
    chunk_size_tokens: int = 180
    chunk_overlap_tokens: int = 40
    min_chunk_tokens: int = 30
    prepend_title: bool = True


@dataclass
class EmbeddingConfig:
    backend: str = "auto"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    lsa_dimensions: int = 256
    batch_size: int = 64


@dataclass
class VectorStoreConfig:
    backend: str = "chroma"
    collection: str = "chunks"


@dataclass
class QualityConfig:
    fail_on_error: bool = True


@dataclass
class Config:
    paths: Paths
    sources: list[SourceConfig]
    cleaning: CleaningConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    quality: QualityConfig


def _coerce(value: str, current: Any) -> Any:
    if isinstance(current, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(value)
    if isinstance(current, list):
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


def _apply_env_overrides(raw: dict) -> dict:
    """DE_<SECTION>__<KEY>=value overrides raw[section][key]."""
    for env_key, value in os.environ.items():
        if not env_key.startswith("DE_") or "__" not in env_key:
            continue
        section, key = env_key[3:].lower().split("__", 1)
        if isinstance(raw.get(section), dict):
            raw[section][key] = _coerce(value, raw[section].get(key, ""))
    return raw


def load_config(path: str | Path | None = None, root: Path | None = None) -> Config:
    path = Path(path) if path else default_config_path()
    # relative paths in the config resolve against the project root (the dir holding config/)
    root = root or path.resolve().parent.parent
    raw = _apply_env_overrides(yaml.safe_load(path.read_text()))

    def resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else root / pp

    p_raw = dict(raw["paths"])
    lake_uri = str(p_raw.pop("lake_uri")).rstrip("/")
    if "://" not in lake_uri:
        lake_uri = str(resolve(lake_uri))
        Path(lake_uri).mkdir(parents=True, exist_ok=True)
    paths = Paths(lake_uri=lake_uri, **{k: resolve(v) for k, v in p_raw.items()})
    for p in (paths.data_dir, paths.vector_dir, paths.artifacts_dir):
        p.mkdir(parents=True, exist_ok=True)

    sources = []
    for s in raw["sources"]:
        s = dict(s)
        if s["type"] == "local_csv":
            s["uri"] = str(resolve(s["uri"]))
        sources.append(SourceConfig(**s))

    return Config(
        paths=paths,
        sources=sources,
        cleaning=CleaningConfig(**raw.get("cleaning", {})),
        chunking=ChunkingConfig(**raw.get("chunking", {})),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        vector_store=VectorStoreConfig(**raw.get("vector_store", {})),
        quality=QualityConfig(**raw.get("quality", {})),
    )
