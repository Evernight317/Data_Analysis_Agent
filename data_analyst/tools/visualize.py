"""Visualization tool for the Data Analysis Agent.

Generates static charts (PNG/SVG) using matplotlib and seaborn,
with Chinese font support.
"""

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file output
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from data_analyst.config import CHARTS_DIR, CHART_FORMAT, CHART_DPI, CHINESE_FONT
from data_analyst.tools.data_loader import get_dataframe


def _setup_chinese_font():
    """Configure matplotlib for Chinese font rendering."""
    plt.rcParams["font.sans-serif"] = [CHINESE_FONT, "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display


# Initialize on import
_setup_chinese_font()


# Chart type to handler mapping
_CHART_HANDLERS = {}


def chart_handler(chart_type: str):
    """Decorator to register a chart handler function."""
    def decorator(func):
        _CHART_HANDLERS[chart_type] = func
        return func
    return decorator


def create_chart(
    data_id: str,
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    hue: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    filename: str | None = None,
    save_format: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a chart from loaded data and save it to file.

    Args:
        data_id: Reference to previously loaded data.
        chart_type: Type of chart to create.
        x: Column name for x-axis.
        y: Column name for y-axis.
        hue: Column name for color grouping.
        title: Chart title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        filename: Output filename (without extension). Auto-generated if not specified.
        save_format: Output format ('png' or 'svg'). Uses config default if not specified.
        params: Additional chart-specific parameters.

    Returns:
        Dictionary with file path and chart info.
    """
    params = params or {}

    try:
        df = get_dataframe(data_id)
    except KeyError as e:
        return {"error": str(e)}

    # Validate columns
    for col_name, col_val in [("x", x), ("y", y), ("hue", hue)]:
        if col_val and col_val not in df.columns:
            return {"error": f"Column '{col_val}' not found. Available: {list(df.columns)}"}

    handler = _CHART_HANDLERS.get(chart_type)
    if handler is None:
        available = list(_CHART_HANDLERS.keys())
        return {"error": f"Unknown chart type: '{chart_type}'. Available: {available}"}

    try:
        fig, ax = handler(df, x=x, y=y, hue=hue, params=params)

        # Apply labels
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)

        # Tight layout
        fig.tight_layout()

        # Save
        fmt = save_format or CHART_FORMAT
        if filename is None:
            filename = f"chart_{chart_type}_{data_id}"
        filepath = CHARTS_DIR / f"{filename}.{fmt}"

        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=CHART_DPI, format=fmt, bbox_inches="tight")
        plt.close(fig)

        return {
            "chart_type": chart_type,
            "file_path": str(filepath),
            "format": fmt,
            "title": title or f"{chart_type} chart",
            "data_id": data_id,
        }

    except Exception as e:
        plt.close("all")
        return {"error": f"Chart creation failed: {str(e)}"}


# --- Chart Handlers ---

@chart_handler("line")
def _line_chart(df, x, y, hue, params):
    """Line chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if hue and hue in df.columns:
        for name, group in df.groupby(hue):
            ax.plot(group[x], group[y], marker="o", markersize=3, label=str(name))
        ax.legend(title=hue)
    else:
        ax.plot(df[x], df[y], marker="o", markersize=3)
    return fig, ax


@chart_handler("bar")
def _bar_chart(df, x, y, hue, params):
    """Bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if hue and hue in df.columns:
        pivot = df.pivot_table(index=x, columns=hue, values=y, aggfunc="mean")
        pivot.plot(kind="bar", ax=ax)
        ax.legend(title=hue)
    else:
        ax.bar(df[x].astype(str), df[y])
    plt.xticks(rotation=45, ha="right")
    return fig, ax


@chart_handler("scatter")
def _scatter_chart(df, x, y, hue, params):
    """Scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    size_col = params.get("size")
    if hue and hue in df.columns:
        scatter = ax.scatter(df[x], df[y], c=df[hue].astype("category").cat.codes,
                            cmap="viridis", alpha=0.6, s=50)
        # Add legend for hue
        categories = df[hue].astype("category").cat.categories
        handles = [plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=scatter.cmap(scatter.norm(i)),
                              markersize=8, label=str(cat))
                   for i, cat in enumerate(categories)]
        ax.legend(handles=handles, title=hue)
    else:
        ax.scatter(df[x], df[y], alpha=0.6, s=50)
    return fig, ax


@chart_handler("histogram")
def _histogram(df, x, y, hue, params):
    """Histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = params.get("bins", 30)
    kde = params.get("kde", True)
    if hue and hue in df.columns:
        for name, group in df.groupby(hue):
            ax.hist(group[x].dropna(), bins=bins, alpha=0.5, label=str(name), density=True)
            if kde:
                group[x].dropna().plot.kde(ax=ax, label=f"{name} (KDE)")
        ax.legend(title=hue)
    else:
        ax.hist(df[x].dropna(), bins=bins, alpha=0.7, density=True)
        if kde:
            df[x].dropna().plot.kde(ax=ax)
    return fig, ax


@chart_handler("box")
def _box_chart(df, x, y, hue, params):
    """Box plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if x and y:
        if hue and hue in df.columns:
            sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax)
        else:
            sns.boxplot(data=df, x=x, y=y, ax=ax)
    elif y:
        sns.boxplot(data=df, y=y, ax=ax)
    elif x:
        # x is the numeric column, no grouping
        sns.boxplot(data=df, x=x, ax=ax)
    plt.xticks(rotation=45, ha="right")
    return fig, ax


@chart_handler("violin")
def _violin_chart(df, x, y, hue, params):
    """Violin plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if x and y:
        if hue and hue in df.columns:
            sns.violinplot(data=df, x=x, y=y, hue=hue, ax=ax, split=True)
        else:
            sns.violinplot(data=df, x=x, y=y, ax=ax)
    elif y:
        sns.violinplot(data=df, y=y, ax=ax)
    plt.xticks(rotation=45, ha="right")
    return fig, ax


@chart_handler("heatmap")
def _heatmap(df, x, y, hue, params):
    """Heatmap (typically for correlation matrix)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    method = params.get("method", "pearson")
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) < 2:
        raise ValueError("Heatmap requires at least 2 numeric columns.")

    corr = df[numeric_cols].corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool)) if params.get("triangle", True) else None
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    return fig, ax


@chart_handler("pairplot")
def _pairplot_chart(df, x, y, hue, params):
    """Pair plot (scatter matrix)."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    max_cols = params.get("max_columns", 6)
    if len(numeric_cols) > max_cols:
        numeric_cols = numeric_cols[:max_cols]

    plot_df = df[numeric_cols].copy()
    if hue and hue in df.columns:
        plot_df[hue] = df[hue]

    g = sns.pairplot(plot_df, hue=hue if hue and hue in df.columns else None,
                     diag_kind="kde", plot_kws={"alpha": 0.5})
    g.fig.set_size_inches(12, 10)
    return g.fig, g.fig.axes[0]


@chart_handler("pie")
def _pie_chart(df, x, y, hue, params):
    """Pie chart."""
    fig, ax = plt.subplots(figsize=(8, 8))
    if y:
        # x is labels, y is values
        values = df[y]
        labels = df[x].astype(str) if x else df.index.astype(str)
    elif x:
        # x is categorical, count occurrences
        vc = df[x].value_counts()
        values = vc.values
        labels = vc.index.astype(str)
    else:
        raise ValueError("Pie chart requires at least one column (x for counts, or x+y for values).")

    # Limit to top N categories, group rest as "Other"
    top_n = params.get("top_n", 10)
    if len(values) > top_n:
        top_values = values[:top_n]
        other_sum = values[top_n:].sum()
        values = list(top_values) + [other_sum]
        labels = list(labels[:top_n]) + ["Other"]

    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90,
           pctdistance=0.85)
    ax.axis("equal")
    return fig, ax


@chart_handler("count")
def _count_chart(df, x, y, hue, params):
    """Count plot (bar chart of value counts)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    col = x or y
    if not col:
        raise ValueError("Count chart requires a column name (x or y).")
    if hue and hue in df.columns:
        sns.countplot(data=df, x=col, hue=hue, ax=ax)
    else:
        sns.countplot(data=df, x=col, ax=ax)
    plt.xticks(rotation=45, ha="right")
    return fig, ax


@chart_handler("area")
def _area_chart(df, x, y, hue, params):
    """Area chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if hue and hue in df.columns:
        for name, group in df.groupby(hue):
            ax.fill_between(group[x], group[y], alpha=0.3, label=str(name))
        ax.legend(title=hue)
    else:
        ax.fill_between(df[x], df[y], alpha=0.3)
    return fig, ax


def get_tool_def() -> dict:
    """Return the Claude tool definition for create_chart."""
    return {
        "name": "create_chart",
        "description": (
            "Create a chart/visualization from loaded data and save it as a static image (PNG or SVG). "
            "Supports: line, bar, scatter, histogram, box, violin, heatmap, pairplot, pie, count, area charts. "
            "Returns the file path of the saved chart."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "data_id": {
                    "type": "string",
                    "description": "Reference ID of the previously loaded dataset."
                },
                "chart_type": {
                    "type": "string",
                    "enum": ["line", "bar", "scatter", "histogram", "box", "violin",
                             "heatmap", "pairplot", "pie", "count", "area"],
                    "description": "Type of chart to create."
                },
                "x": {
                    "type": "string",
                    "description": "Column name for x-axis."
                },
                "y": {
                    "type": "string",
                    "description": "Column name for y-axis."
                },
                "hue": {
                    "type": "string",
                    "description": "Column name for color grouping (optional)."
                },
                "title": {
                    "type": "string",
                    "description": "Chart title."
                },
                "xlabel": {
                    "type": "string",
                    "description": "X-axis label."
                },
                "ylabel": {
                    "type": "string",
                    "description": "Y-axis label."
                },
                "filename": {
                    "type": "string",
                    "description": "Output filename (without extension). Auto-generated if not specified."
                },
                "save_format": {
                    "type": "string",
                    "enum": ["png", "svg"],
                    "description": "Output format. Defaults to config setting (png)."
                },
                "params": {
                    "type": "object",
                    "description": "Chart-specific parameters. Examples: "
                    "{'bins': 30, 'kde': true} for histogram, "
                    "{'method': 'spearman'} for heatmap, "
                    "{'top_n': 10} for pie, "
                    "{'max_columns': 6} for pairplot.",
                    "properties": {
                        "bins": {"type": "integer", "description": "Number of bins for histogram."},
                        "kde": {"type": "boolean", "description": "Show KDE overlay on histogram."},
                        "method": {"type": "string", "description": "Correlation method for heatmap (pearson/spearman)."},
                        "top_n": {"type": "integer", "description": "Top N categories for pie chart."},
                        "max_columns": {"type": "integer", "description": "Max columns for pairplot."},
                        "triangle": {"type": "boolean", "description": "Show only lower triangle for heatmap."}
                    }
                }
            },
            "required": ["data_id", "chart_type"]
        }
    }
