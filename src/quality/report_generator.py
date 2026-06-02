"""Generate HTML data quality reports with embedded charts.

Produces self-contained HTML reports that include:
  - Summary statistics table
  - Pixel intensity distribution charts
  - Metadata completeness heatmaps
  - Class balance bar charts
  - Outlier highlights
  - Warnings and recommendations

Reports are standalone HTML files with inline CSS and base64-encoded
chart images -- no external dependencies needed to view them.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _fig_to_base64(fig: Any) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return b64


def _generate_distribution_chart(
    stats: Any,
    title: str,
    xlabel: str,
    color: str = "#4fc3f7",
) -> str:
    """Generate a distribution histogram chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    if stats.histogram_bins and stats.histogram_counts:
        bins = stats.histogram_bins
        counts = stats.histogram_counts
        bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(counts))]
        ax.bar(bin_centers, counts, width=(bins[1] - bins[0]) * 0.9, color=color, alpha=0.8)

    ax.set_title(title, color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, color="#aaa")
    ax.set_ylabel("Count", color="#aaa")
    ax.tick_params(colors="#888")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add stats annotation
    text = (
        f"Mean: {stats.mean:.3f}\n"
        f"Std: {stats.std:.3f}\n"
        f"Median: {stats.median:.3f}"
    )
    ax.text(
        0.97, 0.95, text,
        transform=ax.transAxes, fontsize=9, color="#ccc",
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f3460", alpha=0.8),
    )

    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def _generate_completeness_chart(completeness: Dict[str, float]) -> str:
    """Generate a metadata completeness horizontal bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = list(completeness.keys())
    values = [completeness[f] * 100 for f in fields]

    fig, ax = plt.subplots(figsize=(8, max(3, len(fields) * 0.35)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    colors = ["#4caf50" if v >= 95 else "#ff9800" if v >= 80 else "#f44336" for v in values]
    y_pos = range(len(fields))
    ax.barh(y_pos, values, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fields, fontsize=9, color="#ccc")
    ax.set_xlabel("Completeness (%)", color="#aaa")
    ax.set_title("Metadata Field Completeness", color="white", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.axvline(x=95, color="#4caf50", linestyle="--", alpha=0.5, linewidth=1)
    ax.tick_params(colors="#888")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels
    for i, v in enumerate(values):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8, color="#ccc")

    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def _generate_class_balance_chart(class_counts: Dict[str, int]) -> str:
    """Generate a class distribution bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
    ax.bar(classes, counts, color=colors, alpha=0.85)
    ax.set_title("Class Distribution", color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Class", color="#aaa")
    ax.set_ylabel("Count", color="#aaa")
    ax.tick_params(colors="#888", axis="x", rotation=45)
    ax.tick_params(colors="#888", axis="y")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add count labels
    for i, (cls, cnt) in enumerate(zip(classes, counts)):
        ax.text(i, cnt + max(counts) * 0.02, str(cnt), ha="center", fontsize=9, color="#ccc")

    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Data Quality Report - {dataset_name}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f0f23; color: #e0e0e0; padding: 20px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ color: #4fc3f7; margin-bottom: 5px; font-size: 28px; }}
  h2 {{ color: #81d4fa; margin: 30px 0 15px; font-size: 20px; border-bottom: 1px solid #333; padding-bottom: 8px; }}
  .subtitle {{ color: #888; margin-bottom: 25px; font-size: 14px; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
  .stat-card {{ background: #1a1a2e; border: 1px solid #333; border-radius: 8px; padding: 15px; }}
  .stat-card .label {{ font-size: 12px; color: #888; text-transform: uppercase; }}
  .stat-card .value {{ font-size: 24px; color: #4fc3f7; font-weight: bold; margin-top: 5px; }}
  .chart-container {{ margin: 20px 0; text-align: center; }}
  .chart-container img {{ max-width: 100%; border-radius: 8px; border: 1px solid #333; }}
  .warning {{ background: #332200; border: 1px solid #ff9800; border-radius: 6px; padding: 12px; margin: 8px 0; }}
  .warning::before {{ content: "Warning: "; font-weight: bold; color: #ff9800; }}
  .recommendation {{ background: #002233; border: 1px solid #4fc3f7; border-radius: 6px; padding: 12px; margin: 8px 0; }}
  .recommendation::before {{ content: "Recommendation: "; font-weight: bold; color: #4fc3f7; }}
  table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #333; }}
  th {{ background: #16213e; color: #81d4fa; font-weight: 600; }}
  tr:hover {{ background: #1a1a2e; }}
  .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #333; color: #666; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Data Quality Report</h1>
  <p class="subtitle">{dataset_name} | Generated: {timestamp} | {total_samples} samples</p>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="label">Total Samples</div>
      <div class="value">{total_samples}</div>
    </div>
    {stats_cards}
  </div>

  {charts_html}

  {warnings_html}

  {recommendations_html}

  {tables_html}

  <div class="footer">
    Generated by DICOM Processing Pipeline - Data Quality Engine
  </div>
</div>
</body>
</html>"""


class ReportGenerator:
    """Generate HTML data quality reports.

    Takes a DataQualityReport and produces a self-contained HTML file
    with embedded charts and styling.

    Args:
        output_dir: Directory for report output.

    Example::

        generator = ReportGenerator(output_dir="/reports")
        generator.generate(quality_report, filename="quality_report.html")
    """

    def __init__(self, output_dir: str = ".") -> None:
        self.output_dir = output_dir

    def generate(
        self,
        report: Any,
        filename: str = "data_quality_report.html",
    ) -> str:
        """Generate an HTML report from a DataQualityReport.

        Args:
            report: DataQualityReport instance.
            filename: Output filename.

        Returns:
            Absolute path to the generated HTML file.
        """
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Build stats cards
        stats_cards = self._build_stats_cards(report)

        # Build chart sections
        charts_html = self._build_charts(report)

        # Warnings
        warnings_html = ""
        if report.warnings:
            warnings_html = "<h2>Warnings</h2>\n"
            for w in report.warnings:
                warnings_html += f'<div class="warning">{w}</div>\n'

        # Recommendations
        recommendations_html = ""
        if report.recommendations:
            recommendations_html = "<h2>Recommendations</h2>\n"
            for r in report.recommendations:
                recommendations_html += f'<div class="recommendation">{r}</div>\n'

        # Tables
        tables_html = self._build_tables(report)

        html = _HTML_TEMPLATE.format(
            dataset_name=report.dataset_name or "Dataset",
            timestamp=report.timestamp or datetime.now().isoformat(),
            total_samples=report.total_samples,
            stats_cards=stats_cards,
            charts_html=charts_html,
            warnings_html=warnings_html,
            recommendations_html=recommendations_html,
            tables_html=tables_html,
        )

        output_path = str(Path(self.output_dir) / filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("Quality report written to %s", output_path)
        return output_path

    def _build_stats_cards(self, report: Any) -> str:
        """Build summary statistics card HTML."""
        cards = []

        if report.outlier_report:
            cards.append(
                f'<div class="stat-card">'
                f'<div class="label">Outliers Detected</div>'
                f'<div class="value">{report.outlier_report.outlier_count}</div>'
                f'</div>'
            )

        if report.class_balance:
            cards.append(
                f'<div class="stat-card">'
                f'<div class="label">Num Classes</div>'
                f'<div class="value">{report.class_balance.num_classes}</div>'
                f'</div>'
            )
            cards.append(
                f'<div class="stat-card">'
                f'<div class="label">Imbalance Ratio</div>'
                f'<div class="value">{report.class_balance.imbalance_ratio:.1f}:1</div>'
                f'</div>'
            )

        if report.completeness_report:
            n_critical = len(report.completeness_report.critical_missing)
            cards.append(
                f'<div class="stat-card">'
                f'<div class="label">Missing Fields</div>'
                f'<div class="value">{n_critical}</div>'
                f'</div>'
            )

        return "\n".join(cards)

    def _build_charts(self, report: Any) -> str:
        """Build chart section HTML."""
        sections = []

        # Intensity distribution
        if report.intensity_stats and report.intensity_stats.count > 0:
            b64 = _generate_distribution_chart(
                report.intensity_stats,
                "Pixel Intensity Distribution (Mean per Image)",
                "Mean Intensity",
            )
            sections.append(
                f'<h2>Pixel Intensity Distribution</h2>\n'
                f'<div class="chart-container">'
                f'<img src="data:image/png;base64,{b64}" alt="Intensity Distribution">'
                f'</div>'
            )

        # Completeness
        if report.completeness_report and report.completeness_report.field_completeness:
            b64 = _generate_completeness_chart(report.completeness_report.field_completeness)
            sections.append(
                f'<h2>Metadata Completeness</h2>\n'
                f'<div class="chart-container">'
                f'<img src="data:image/png;base64,{b64}" alt="Completeness">'
                f'</div>'
            )

        # Class balance
        if report.class_balance and report.class_balance.class_counts:
            b64 = _generate_class_balance_chart(report.class_balance.class_counts)
            sections.append(
                f'<h2>Class Balance</h2>\n'
                f'<div class="chart-container">'
                f'<img src="data:image/png;base64,{b64}" alt="Class Balance">'
                f'</div>'
            )

        return "\n".join(sections)

    def _build_tables(self, report: Any) -> str:
        """Build data tables HTML."""
        tables = []

        # Distribution summary
        if any([
            report.modality_distribution,
            report.manufacturer_distribution,
            report.laterality_distribution,
        ]):
            tables.append("<h2>Data Distributions</h2>")

            for title, dist in [
                ("Modality", report.modality_distribution),
                ("Manufacturer", report.manufacturer_distribution),
                ("Laterality", report.laterality_distribution),
                ("View Position", report.view_distribution),
            ]:
                if dist:
                    table = f"<h3 style='color:#81d4fa;margin:15px 0 8px;font-size:16px'>{title}</h3>"
                    table += "<table><tr><th>Value</th><th>Count</th><th>Fraction</th></tr>"
                    total = sum(dist.values())
                    for k, v in sorted(dist.items(), key=lambda x: -x[1]):
                        frac = v / max(total, 1) * 100
                        table += f"<tr><td>{k}</td><td>{v}</td><td>{frac:.1f}%</td></tr>"
                    table += "</table>"
                    tables.append(table)

        return "\n".join(tables)
