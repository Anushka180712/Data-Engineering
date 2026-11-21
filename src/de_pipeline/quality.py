"""Lightweight, declarative data-quality checks (a tiny Great-Expectations-style layer)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .log import get_logger

log = get_logger(__name__)


class DataQualityError(RuntimeError):
    pass


@dataclass
class Check:
    name: str
    fn: Callable[[pd.DataFrame], tuple[bool, object]]
    severity: str = "error"  # error | warn


def not_null(col: str) -> Check:
    return Check(f"not_null:{col}", lambda df: ((n := int(df[col].isna().sum())) == 0, n))


def unique(col: str) -> Check:
    return Check(f"unique:{col}", lambda df: ((n := int(df[col].duplicated().sum())) == 0, n))


def accepted_values(col: str, values: list[str]) -> Check:
    def fn(df):
        bad = sorted(set(df[col].dropna()) - set(values))
        return not bad, bad or "ok"
    return Check(f"accepted_values:{col}", fn)


def min_length(col: str, n: int, severity: str = "error") -> Check:
    return Check(f"min_length:{col}>={n}",
                 lambda df: ((k := int((df[col].str.len() < n).sum())) == 0, k), severity)


def row_count_between(lo: int, hi: int | None = None, severity: str = "error") -> Check:
    return Check(f"row_count_between:{lo}..{hi or 'inf'}",
                 lambda df: (lo <= len(df) and (hi is None or len(df) <= hi), len(df)), severity)


def no_text_artifacts(col: str, pattern: str = r"�|Â£|â€", severity: str = "warn") -> Check:
    return Check(f"no_encoding_artifacts:{col}",
                 lambda df: ((k := int(df[col].str.contains(pattern, regex=True).sum())) == 0, k),
                 severity)


def run_checks(df: pd.DataFrame, checks: list[Check], dataset: str,
               fail_on_error: bool = True) -> list[dict]:
    results = []
    for c in checks:
        passed, observed = c.fn(df)
        results.append({"check": c.name, "severity": c.severity,
                        "passed": bool(passed), "observed": observed})
        level = log.info if passed else (log.error if c.severity == "error" else log.warning)
        level("DQ %-40s %s (observed=%s)", f"[{dataset}] {c.name}", "PASS" if passed else "FAIL", observed)
    failed = [r for r in results if not r["passed"] and r["severity"] == "error"]
    if failed and fail_on_error:
        raise DataQualityError(f"{dataset}: {len(failed)} error-level checks failed: "
                               f"{[r['check'] for r in failed]}")
    return results
