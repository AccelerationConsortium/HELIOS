"""Export the scoreboard + recommendations to markdown and CSV strings.

Pure string builders -- no file I/O (the caller decides where to write), so the
toolkit/case-study deliverable can render results inline or persist them.
"""
from __future__ import annotations

import csv
import io
from collections import defaultdict
from typing import Any

from benchmarks.methods.recommend import Recommendation
from benchmarks.methods.scoreboard import MethodScore


def _fmt(value: float | None, ndigits: int = 4) -> str:
    if value is None:
        return "-"
    if value == float("inf"):
        return "inf"
    return f"{value:.{ndigits}f}"


def scoreboard_to_markdown(scores: list[MethodScore]) -> str:
    """Render the (problem x method) score table as a markdown table."""
    header = (
        "| problem | method | seeds | mean_regret | mean_auc | "
        "evals_to_target | hit_rate | robustness | cost_s | errors |"
    )
    sep = "|" + "|".join(["---"] * 10) + "|"
    rows = [header, sep]
    for s in scores:
        rows.append(
            f"| {s.problem_id} | {s.backend} | {s.n_seeds} | {_fmt(s.mean_regret)} | {_fmt(s.mean_auc)} | {_fmt(s.mean_evals_to_target, 1)} | {_fmt(s.target_hit_rate, 2)} | "
            f"{_fmt(s.regret_robustness)} | {_fmt(s.mean_cost_s, 5)} | {s.errors} |"
        )
    return "\n".join(rows)


def scoreboard_to_csv(scores: list[MethodScore]) -> str:
    """Render the score table as CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "problem_id",
            "backend",
            "n_seeds",
            "mean_regret",
            "mean_auc",
            "mean_evals_to_target",
            "target_hit_rate",
            "regret_robustness",
            "mean_cost_s",
            "errors",
        ]
    )
    for s in scores:
        writer.writerow(
            [
                s.problem_id,
                s.backend,
                s.n_seeds,
                s.mean_regret,
                s.mean_auc,
                s.mean_evals_to_target if s.mean_evals_to_target is not None else "",
                s.target_hit_rate,
                s.regret_robustness,
                s.mean_cost_s,
                s.errors,
            ]
        )
    return buf.getvalue()


def benchmark_family(score: MethodScore) -> str:
    """Classify a benchmark problem into the family that should drive conclusions."""
    tags = score.tags
    if tags.surface_class.startswith("early_stage"):
        return "early_stage_imperfect_data"
    if tags.has_categorical:
        return "mixed_categorical"
    if tags.n_dims <= 3 and tags.noise_std <= 0.0:
        return "clean_low_dim_bo"
    if tags.n_dims > 3 and tags.noise_std <= 0.0:
        return "clean_high_dim"
    return "analytic_other"


def family_summary(scores: list[MethodScore]) -> list[dict[str, Any]]:
    """Average method performance within each benchmark family."""
    grouped: dict[tuple[str, str], list[MethodScore]] = defaultdict(list)
    for score in scores:
        grouped[(benchmark_family(score), score.backend)].append(score)

    rows: list[dict[str, Any]] = []
    for (family, backend), items in sorted(grouped.items()):
        n = len(items)
        rows.append(
            {
                "family": family,
                "backend": backend,
                "n_problems": n,
                "mean_regret": sum(s.mean_regret for s in items) / n,
                "mean_auc": sum(s.mean_auc for s in items) / n,
                "target_hit_rate": sum(s.target_hit_rate for s in items) / n,
                "mean_cost_s": sum(s.mean_cost_s for s in items) / n,
                "errors": sum(s.errors for s in items),
            }
        )
    rows.sort(key=lambda row: (str(row["family"]), float(row["mean_regret"])))
    return rows


def family_summary_to_markdown(scores: list[MethodScore]) -> str:
    """Render family-level method performance as a markdown table."""
    header = (
        "| family | method | problems | mean_regret | mean_auc | "
        "hit_rate | cost_s | errors |"
    )
    sep = "|" + "|".join(["---"] * 8) + "|"
    rows = [header, sep]
    for row in family_summary(scores):
        rows.append(
            f"| {row['family']} | {row['backend']} | {row['n_problems']} | "
            f"{_fmt(float(row['mean_regret']))} | {_fmt(float(row['mean_auc']))} | "
            f"{_fmt(float(row['target_hit_rate']), 2)} | "
            f"{_fmt(float(row['mean_cost_s']), 5)} | {row['errors']} |"
        )
    return "\n".join(rows)


def family_summary_to_csv(scores: list[MethodScore]) -> str:
    """Render family-level method performance as CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "family",
            "backend",
            "n_problems",
            "mean_regret",
            "mean_auc",
            "target_hit_rate",
            "mean_cost_s",
            "errors",
        ]
    )
    for row in family_summary(scores):
        writer.writerow(
            [
                row["family"],
                row["backend"],
                row["n_problems"],
                row["mean_regret"],
                row["mean_auc"],
                row["target_hit_rate"],
                row["mean_cost_s"],
                row["errors"],
            ]
        )
    return buf.getvalue()


def recommendations_to_markdown(recs: list[Recommendation]) -> str:
    """Render the decision table (tag bucket -> ranked methods) as markdown."""
    lines = ["## Method recommendations by problem structure", ""]
    for rec in recs:
        lines.append(f"### {rec.bucket}")
        lines.append("")
        lines.append("| rank | method | score | mean_regret | hit_rate |")
        lines.append("|---|---|---|---|---|")
        for i, m in enumerate(rec.ranked, start=1):
            lines.append(
                f"| {i} | {m.backend} | {_fmt(m.score)} | "
                f"{_fmt(m.mean_regret)} | {_fmt(m.target_hit_rate, 2)} |"
            )
        lines.append("")
    return "\n".join(lines)


def recommendations_to_csv(recs: list[Recommendation]) -> str:
    """Render the decision table as CSV (one row per bucket-method-rank)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["bucket", "rank", "method", "score", "mean_regret", "target_hit_rate"]
    )
    for rec in recs:
        for i, m in enumerate(rec.ranked, start=1):
            writer.writerow(
                [rec.bucket, i, m.backend, m.score, m.mean_regret, m.target_hit_rate]
            )
    return buf.getvalue()
