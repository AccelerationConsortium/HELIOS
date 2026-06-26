"""Export the scoreboard + recommendations to markdown and CSV strings.

Pure string builders -- no file I/O (the caller decides where to write), so the
toolkit/case-study deliverable can render results inline or persist them.
"""
from __future__ import annotations

import csv
import io

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
            "| {pid} | {be} | {ns} | {reg} | {auc} | {ett} | {hit} | "
            "{rob} | {cost} | {err} |".format(
                pid=s.problem_id,
                be=s.backend,
                ns=s.n_seeds,
                reg=_fmt(s.mean_regret),
                auc=_fmt(s.mean_auc),
                ett=_fmt(s.mean_evals_to_target, 1),
                hit=_fmt(s.target_hit_rate, 2),
                rob=_fmt(s.regret_robustness),
                cost=_fmt(s.mean_cost_s, 5),
                err=s.errors,
            )
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
