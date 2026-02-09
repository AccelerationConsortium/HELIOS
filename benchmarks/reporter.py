"""Reporter — JSON + Markdown report generation for benchmark results.

Generates two output files:
- report.json: machine-readable full results
- report.md: human-readable summary with tables
"""
from __future__ import annotations

import json
import os
from typing import Any


class Reporter:
    """Generates benchmark reports in JSON and Markdown formats.

    Args:
        output_dir: Directory to write report files to.
    """

    def __init__(self, output_dir: str = "benchmark_results") -> None:
        self.output_dir = output_dir

    def generate(self, report: Any) -> tuple[str, str]:
        """Generate both JSON and Markdown reports.

        Returns (json_path, md_path).
        """
        os.makedirs(self.output_dir, exist_ok=True)
        json_path = self._write_json(report)
        md_path = self._write_markdown(report)
        return json_path, md_path

    def _write_json(self, report: Any) -> str:
        """Write machine-readable JSON report."""
        path = os.path.join(self.output_dir, "report.json")
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        return path

    def _write_markdown(self, report: Any) -> str:
        """Write human-readable Markdown report."""
        path = os.path.join(self.output_dir, "report.md")
        lines: list[str] = []

        lines.append("# OTbot Benchmark Report")
        lines.append("")
        lines.append(f"**Timestamp**: {report.timestamp}")
        lines.append(f"**Seed**: {report.seed}")
        lines.append(f"**Duration**: {report.duration_s:.1f}s")
        lines.append(f"**Total**: {report.total} scenarios | "
                     f"**Passed**: {report.passed} | "
                     f"**Failed**: {report.failed}")
        lines.append("")

        # Scoreboard
        if report.scoreboard:
            lines.append("## Scoreboard")
            lines.append("")
            lines.append("| Metric | Value | Status |")
            lines.append("|--------|-------|--------|")

            sb = report.scoreboard
            metrics_display = [
                ("Goal Success Rate", sb.get("goal_success_rate", 0),
                 ">=0.50", sb.get("goal_success_rate", 0) >= 0.5),
                ("Sample Efficiency", sb.get("sample_efficiency", 0),
                 ">0.00", sb.get("sample_efficiency", 0) > 0),
                ("Safety Violations", sb.get("safety_violations", 0),
                 "==0", sb.get("safety_violations", 0) == 0),
                ("Recovery Rate", sb.get("recovery_rate", 0),
                 ">=0.50", sb.get("recovery_rate", 0) >= 0.5),
                ("Stability", sb.get("stability", 0),
                 ">=0.80", sb.get("stability", 0) >= 0.8),
            ]

            for name, value, target, passed in metrics_display:
                icon = "PASS" if passed else "FAIL"
                if isinstance(value, int):
                    lines.append(f"| {name} | {value} | {icon} ({target}) |")
                else:
                    lines.append(f"| {name} | {value:.4f} | {icon} ({target}) |")

            lines.append("")

        # Results by category
        lines.append("## Results by Category")
        lines.append("")

        categories = {}
        for r in report.results:
            cat = r.scenario_id.split("_")[0]
            categories.setdefault(cat, []).append(r)

        category_names = {
            "c2": "C2: Metrics Store",
            "c3": "C3: Reviewer",
            "c4": "C4: Candidate Gen",
            "c5": "C5: Evolution Engine",
            "fault": "Fault Injection",
            "intel": "Intelligence Metrics",
        }

        for cat, results in sorted(categories.items()):
            passed_count = sum(1 for r in results if r.passed)
            total_count = len(results)
            cat_name = category_names.get(cat, cat)

            lines.append(f"### {cat_name} ({passed_count}/{total_count} passed)")
            lines.append("")
            lines.append("| Scenario | Status | Duration | Details |")
            lines.append("|----------|--------|----------|---------|")

            for r in results:
                icon = "PASS" if r.passed else "FAIL"
                detail = ""
                if r.error:
                    detail = r.error[:50]
                elif r.acceptance.details:
                    # Show first criterion detail
                    first_detail = next(iter(r.acceptance.details.values()), "")
                    detail = first_detail[:50]

                lines.append(
                    f"| {r.scenario_id} | {icon} | "
                    f"{r.duration_s:.2f}s | {detail} |"
                )

            lines.append("")

        # Failed scenarios detail
        failed = [r for r in report.results if not r.passed]
        if failed:
            lines.append("## Failed Scenarios")
            lines.append("")
            for r in failed:
                lines.append(f"### {r.scenario_id}")
                if r.error:
                    lines.append(f"**Error**: {r.error}")
                if r.acceptance.details:
                    lines.append("")
                    for crit, detail in r.acceptance.details.items():
                        passed = r.acceptance.criteria.get(crit, False)
                        icon = "PASS" if passed else "FAIL"
                        lines.append(f"- [{icon}] {crit}: {detail}")
                lines.append("")

        # Footer
        lines.append("---")
        lines.append(f"*Generated by OTbot Benchmark Framework v0.1.0*")

        with open(path, "w") as f:
            f.write("\n".join(lines))

        return path
