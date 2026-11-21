"""Minimal DAG runner. Each stage is an idempotent function (cfg, run_id) -> metrics.

The same stage functions are what an orchestrator (Airflow / Prefect / Dagster) would call;
see orchestration/ for an Airflow DAG wrapping them.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from . import lakehouse
from .config import Config
from .log import get_logger
from .stages import chunk, clean, embed, index, ingest

log = get_logger("pipeline")

STAGES = {
    "ingest": ingest.run,   # source -> landing zone -> bronze
    "clean": clean.run,     # bronze -> silver (+ DQ)
    "chunk": chunk.run,     # silver -> gold.chunks
    "embed": embed.run,     # gold.chunks -> gold.embeddings
    "index": index.run,     # gold -> vector store
}


def run_pipeline(cfg: Config, stages: list[str] | None = None) -> dict:
    stages = stages or list(STAGES)
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages: {unknown}; choose from {list(STAGES)}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    log.info("run %s: %s", run_id, " -> ".join(stages))
    summary = {"run_id": run_id, "stages": {}}
    for name in [s for s in STAGES if s in stages]:  # always topological order
        started = datetime.now(timezone.utc)
        log.info("=== stage %s ===", name)
        try:
            metrics = STAGES[name](cfg, run_id)
        except Exception as exc:
            lakehouse.record_stage(cfg, run_id, name, "failed", started, {"error": repr(exc)})
            log.exception("stage %s failed", name)
            raise
        lakehouse.record_stage(cfg, run_id, name, "success", started, metrics)
        summary["stages"][name] = metrics
    (cfg.paths.artifacts_dir / "last_run.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
