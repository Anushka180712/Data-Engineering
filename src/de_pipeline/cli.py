"""Command-line entry point: `de-pipeline <command>` (or `python -m de_pipeline`)."""
from __future__ import annotations

import argparse
import json
import sys
import textwrap

from . import lakehouse
from .config import load_config
from .pipeline import STAGES, run_pipeline


def _print_table(rows: list[tuple], headers: list[str], max_width: int = 60) -> None:
    rows = [[("" if v is None else str(v))[:max_width] for v in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)))


def cmd_run(cfg, args):
    stages = args.stages.split(",") if args.stages else None
    summary = run_pipeline(cfg, stages)
    print(json.dumps(summary, indent=2, default=str))


def cmd_search(cfg, args):
    from .search import Searcher

    hits = Searcher(cfg).search(args.query, k=args.k, category=args.category)
    print(f'\nTop {len(hits)} results for: "{args.query}"'
          + (f"  [category={args.category}]" if args.category else "") + "\n")
    for i, h in enumerate(hits, 1):
        print(f"{i}. [{h['score']:.3f}] ({h['category']}) {h['title']}")
        print(textwrap.indent(textwrap.fill(h["text"][:320] + "...", 96), "     "))
        print()


def cmd_eval(cfg, args):
    from .search import evaluate

    print(json.dumps(evaluate(cfg, n_queries=args.n, k=args.k), indent=2))


def cmd_sql(cfg, args):
    con = lakehouse.sql_connection(cfg)
    rel = con.sql(args.query)
    _print_table(rel.fetchall(), rel.columns)


def cmd_catalog(cfg, args):
    if args.ddl:
        ddl = lakehouse.metastore_ddl(cfg)
        out = cfg.paths.artifacts_dir / "metastore_ddl.sql"
        out.write_text(ddl)
        print(ddl)
        print(f"-- written to {out}", file=sys.stderr)
        return
    print(f"lake: {cfg.paths.lake_uri}\n")
    with lakehouse.metadata_db(cfg) as con:
        rows = con.execute("SELECT table_fqn, format, version, num_rows, description FROM catalog "
                           "ORDER BY CASE layer WHEN 'bronze' THEN 1 WHEN 'silver' THEN 2 ELSE 3 END, "
                           "table_name").fetchall()
    _print_table(rows, ["table", "format", "version", "rows", "description"], 80)


def cmd_history(cfg, args):
    layer, name = args.table.split(".")
    from datetime import datetime, timezone

    rows = [(h.get("version"),
             datetime.fromtimestamp(h["timestamp"] / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
             h.get("operation"),
             json.dumps(h.get("operationMetrics", {}))[:70])
            for h in lakehouse.history(cfg, layer, name)]
    _print_table(rows, ["version", "timestamp_utc", "operation", "metrics"], 80)


def cmd_runs(cfg, args):
    with lakehouse.metadata_db(cfg) as con:
        rows = con.execute("SELECT run_id, stage, status, round(duration_s, 2), started_at "
                           "FROM pipeline_runs ORDER BY started_at DESC LIMIT ?", [args.n]).fetchall()
    _print_table(rows, ["run_id", "stage", "status", "secs", "started_at"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="de-pipeline", description=__doc__)
    p.add_argument("--config", help="path to pipeline.yaml (default: config/pipeline.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline (all stages by default)")
    r.add_argument("--stages", help=f"comma-separated subset of {','.join(STAGES)}")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("search", help="semantic search over the vector store")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--category", help="metadata filter, e.g. tech")
    s.set_defaults(fn=cmd_search)

    e = sub.add_parser("eval", help="retrieval quality evaluation")
    e.add_argument("-n", type=int, default=200)
    e.add_argument("-k", type=int, default=5)
    e.set_defaults(fn=cmd_eval)

    q = sub.add_parser("sql", help="run SQL (DuckDB) over the lakehouse, e.g. 'SELECT * FROM silver_articles LIMIT 5'")
    q.add_argument("query")
    q.set_defaults(fn=cmd_sql)

    c = sub.add_parser("catalog", help="list lakehouse tables")
    c.add_argument("--ddl", action="store_true",
                   help="print Hive Metastore / Glue / Spark SQL DDL that registers the tables")
    c.set_defaults(fn=cmd_catalog)
    h = sub.add_parser("history", help="Delta commit history of a table, e.g. silver.articles")
    h.add_argument("table")
    h.set_defaults(fn=cmd_history)
    rr = sub.add_parser("runs", help="recent pipeline stage runs")
    rr.add_argument("-n", type=int, default=15)
    rr.set_defaults(fn=cmd_runs)

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    args.fn(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
