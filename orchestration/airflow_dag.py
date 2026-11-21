"""Airflow DAG wrapping the same idempotent stage functions the CLI runs.

Drop this file (and `pip install` this package) into an Airflow deployment. Each stage is a
task; the run_id is the Airflow run_id, so stage metrics land in the same `pipeline_runs`
table as CLI runs. Retries are safe because every stage is idempotent.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator

from de_pipeline import lakehouse
from de_pipeline.config import load_config
from de_pipeline.pipeline import STAGES


def _run_stage(stage: str, run_id: str, **_):
    cfg = load_config()
    started = datetime.now(timezone.utc)
    try:
        metrics = STAGES[stage](cfg, run_id)
    except Exception as exc:
        lakehouse.record_stage(cfg, run_id, stage, "failed", started, {"error": repr(exc)})
        raise
    lakehouse.record_stage(cfg, run_id, stage, "success", started, metrics)
    return metrics  # pushed to XCom for observability


with DAG(
    dag_id="news_to_vectors",
    description="BBC News: ingest -> bronze -> silver -> gold chunks -> embeddings -> vector store",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=2)},
    tags=["lakehouse", "embeddings", "rag"],
) as dag:
    tasks = [
        PythonOperator(task_id=stage, python_callable=_run_stage,
                       op_kwargs={"stage": stage, "run_id": "{{ run_id }}"})
        for stage in STAGES
    ]
    for upstream, downstream in zip(tasks, tasks[1:]):
        upstream >> downstream
