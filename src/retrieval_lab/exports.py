"""Portable tables and figures from checked experiment evidence, never live timings."""

import io

import pandas as pd


def comparison_csv(report: dict) -> bytes:
    return pd.DataFrame(report["systems"]).to_csv(index=False).encode("utf-8")


def comparison_figure(report: dict, format: str = "png") -> bytes:
    if format not in {"png", "svg"}:
        raise ValueError("Use PNG or SVG")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    rows = report["systems"]
    labels = [f"{r['role']}: {r['system']}" for r in rows]
    fig = Figure(figsize=(10, 7), layout="constrained", facecolor="#f8fafc")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    for ax, field, title in zip(
        axes.flat,
        ["recall_at_10", "ndcg_at_10", "p95_ms", "query_stage_sampled_peak_rss_mib"],
        ["Recall@10", "nDCG@10", "Query p95 (ms)", "Sampled process peak RSS (MiB)"],
        strict=True,
    ):
        ax.barh(labels, [r[field] for r in rows], color=["#334155", "#087f8c"][: len(rows)])
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        if field in {"recall_at_10", "ndcg_at_10"}:
            ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.suptitle(f"RetrievalLab · {report['query_count']} paired queries", fontsize=15)
    fig.supxlabel(
        "Measured on one CPU workstation; timing is descriptive. See paired intervals in JSON.",
        fontsize=9,
    )
    stream = io.BytesIO()
    fig.savefig(stream, format=format, dpi=150)
    return stream.getvalue()
