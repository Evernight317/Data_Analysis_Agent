"""Statistical analysis tool for the Data Analysis Agent.

Provides descriptive statistics, correlation analysis, distribution tests,
group-by aggregation, and hypothesis testing.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from data_analyst.tools.data_loader import get_dataframe


def analyze_stats(
    data_id: str,
    analysis_type: str,
    columns: list[str] | None = None,
    group_by: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Perform statistical analysis on loaded data.

    Args:
        data_id: Reference to previously loaded data.
        analysis_type: Type of analysis to perform. One of:
            - "descriptive": Basic descriptive statistics (mean, std, quartiles, etc.)
            - "correlation": Correlation matrix (Pearson or Spearman)
            - "distribution_test": Test if data follows normal distribution
            - "group_stats": Group-by aggregation statistics
            - "hypothesis_test": Statistical hypothesis tests (t-test, ANOVA, chi-square)
            - "frequency": Frequency distribution and value counts
            - "outlier_detection": Detect outliers using IQR or Z-score method
        columns: Target columns for analysis. If None, uses all applicable columns.
        group_by: Column to group by (for group_stats and some hypothesis tests).
        params: Additional parameters specific to each analysis type.

    Returns:
        Dictionary with analysis results.
    """
    params = params or {}

    try:
        df = get_dataframe(data_id)
    except KeyError as e:
        return {"error": str(e)}

    # Validate columns
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            return {"error": f"Columns not found: {missing}. Available: {list(df.columns)}"}
    else:
        columns = df.columns.tolist()

    try:
        handler = _ANALYSIS_HANDLERS.get(analysis_type)
        if handler is None:
            return {
                "error": f"Unknown analysis type: '{analysis_type}'. "
                         f"Available: {list(_ANALYSIS_HANDLERS.keys())}"
            }
        result = handler(df, columns, group_by, params)
        result["analysis_type"] = analysis_type
        result["data_id"] = data_id
        return result

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


# --- Analysis Handlers ---

def _descriptive(df: pd.DataFrame, columns: list[str], group_by: str | None, params: dict) -> dict:
    """Compute descriptive statistics."""
    target_cols = [c for c in columns if c in df.columns]
    numeric_cols = df[target_cols].select_dtypes(include="number").columns.tolist()
    cat_cols = df[target_cols].select_dtypes(include=["object", "string", "category"]).columns.tolist()

    result = {}

    if numeric_cols:
        desc = df[numeric_cols].describe().to_dict()
        result["numeric"] = {
            col: {k: round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v)
                  for k, v in stats.items()}
            for col, stats in desc.items()
        }
        # Add skewness and kurtosis
        result["skewness"] = {col: round(float(df[col].skew()), 4) for col in numeric_cols}
        result["kurtosis"] = {col: round(float(df[col].kurtosis()), 4) for col in numeric_cols}

    if cat_cols:
        result["categorical"] = {
            col: {
                "count": int(df[col].count()),
                "unique": int(df[col].nunique()),
                "top": str(df[col].mode().iloc[0]) if len(df[col].mode()) > 0 else None,
                "freq": int(df[col].value_counts().iloc[0]) if len(df[col]) > 0 else 0,
            }
            for col in cat_cols
        }

    return result


def _correlation(df: pd.DataFrame, columns: list[str], group_by: str | None, params: dict) -> dict:
    """Compute correlation matrix."""
    method = params.get("method", "pearson")  # pearson or spearman
    numeric_cols = df[columns].select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 2:
        return {"error": "Correlation requires at least 2 numeric columns."}

    corr = df[numeric_cols].corr(method=method)

    # Find strongest correlations (excluding self-correlation)
    strong_corrs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            val = corr.iloc[i, j]
            if abs(val) > 0.5:  # Report correlations > 0.5
                strong_corrs.append({
                    "var1": numeric_cols[i],
                    "var2": numeric_cols[j],
                    "correlation": round(float(val), 4),
                    "strength": "strong" if abs(val) > 0.7 else "moderate"
                })

    return {
        "method": method,
        "matrix": {col: {idx: round(float(val), 4) for idx, val in row.items()}
                   for col, row in corr.to_dict().items()},
        "strong_correlations": sorted(strong_corrs, key=lambda x: abs(x["correlation"]), reverse=True),
    }


def _distribution_test(df: pd.DataFrame, columns: list[str], group_by: str | None, params: dict) -> dict:
    """Test if data follows normal distribution."""
    test = params.get("test", "shapiro")  # shapiro or ks
    numeric_cols = df[columns].select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        return {"error": "Distribution test requires numeric columns."}

    results = {}
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) < 3:
            results[col] = {"error": "Insufficient data (need at least 3 values)"}
            continue

        if test == "shapiro":
            # Shapiro-Wilk test (max 5000 samples)
            sample = data if len(data) <= 5000 else data.sample(5000, random_state=42)
            stat, p_value = scipy_stats.shapiro(sample)
            test_name = "Shapiro-Wilk"
        elif test == "ks":
            # Kolmogorov-Smirnov test against normal distribution
            stat, p_value = scipy_stats.kstest(data, 'norm', args=(data.mean(), data.std()))
            test_name = "Kolmogorov-Smirnov"
        else:
            return {"error": f"Unknown test: {test}. Use 'shapiro' or 'ks'."}

        results[col] = {
            "test": test_name,
            "statistic": round(float(stat), 6),
            "p_value": round(float(p_value), 6),
            "is_normal": p_value > 0.05,
            "interpretation": f"Data {'appears normally' if p_value > 0.05 else 'does not appear normally'} distributed (p={p_value:.4f})"
        }

    return {"test_type": test, "results": results}


def _group_stats(df: pd.DataFrame, columns: list[str], group_by: str | None, params: dict) -> dict:
    """Compute group-by statistics."""
    if group_by is None:
        return {"error": "group_by column is required for group_stats analysis."}

    if group_by not in df.columns:
        return {"error": f"Group-by column '{group_by}' not found."}

    agg_funcs = params.get("agg_funcs", ["mean", "std", "min", "max", "count"])
    numeric_cols = df[columns].select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        return {"error": "No numeric columns found for aggregation."}

    grouped = df.groupby(group_by)[numeric_cols].agg(agg_funcs)

    # Flatten multi-level column names
    result_data = {}
    for col in numeric_cols:
        result_data[col] = {}
        for func in agg_funcs:
            try:
                val = grouped.loc[:, (col, func)]
                result_data[col][func] = {
                    str(k): round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v)
                    for k, v in val.items()
                }
            except (KeyError, ValueError):
                pass

    return {
        "group_by": group_by,
        "aggregations": agg_funcs,
        "results": result_data,
        "group_count": int(df[group_by].nunique()),
    }


def _hypothesis_test(df: pd.DataFrame, columns: list[str], group_by: str | None, params: dict) -> dict:
    """Perform hypothesis tests."""
    test_type = params.get("test", "ttest")  # ttest, anova, chi2

    if test_type == "ttest":
        # Two-sample t-test
        if len(columns) < 2:
            return {"error": "t-test requires at least 2 columns or a group_by column."}

        if group_by and group_by in df.columns:
            groups = df[group_by].unique()
            if len(groups) != 2:
                return {"error": f"t-test with group_by requires exactly 2 groups, found {len(groups)}."}
            col = columns[0]
            g1 = df[df[group_by] == groups[0]][col].dropna()
            g2 = df[df[group_by] == groups[1]][col].dropna()
            stat, p_value = scipy_stats.ttest_ind(g1, g2)
            return {
                "test": "Independent t-test",
                "column": col,
                "group_by": group_by,
                "groups": [str(g) for g in groups],
                "group_means": {str(groups[0]): round(float(g1.mean()), 4),
                                str(groups[1]): round(float(g2.mean()), 4)},
                "statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 6),
                "significant": p_value < 0.05,
                "interpretation": f"{'Significant' if p_value < 0.05 else 'No significant'} difference between groups (p={p_value:.4f})"
            }
        else:
            col1, col2 = columns[0], columns[1]
            d1 = df[col1].dropna()
            d2 = df[col2].dropna()
            stat, p_value = scipy_stats.ttest_ind(d1, d2)
            return {
                "test": "Independent t-test",
                "columns": [col1, col2],
                "means": {col1: round(float(d1.mean()), 4), col2: round(float(d2.mean()), 4)},
                "statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 6),
                "significant": p_value < 0.05,
            }

    elif test_type == "anova":
        if not group_by or group_by not in df.columns:
            return {"error": "ANOVA requires a group_by column."}
        col = columns[0]
        groups = df[group_by].unique()
        group_data = [df[df[group_by] == g][col].dropna().values for g in groups]
        stat, p_value = scipy_stats.f_oneway(*group_data)
        return {
            "test": "One-way ANOVA",
            "column": col,
            "group_by": group_by,
            "num_groups": len(groups),
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 6),
            "significant": p_value < 0.05,
            "interpretation": f"{'Significant' if p_value < 0.05 else 'No significant'} difference among groups (p={p_value:.4f})"
        }

    elif test_type == "chi2":
        if len(columns) < 2:
            return {"error": "Chi-square test requires 2 categorical columns."}
        col1, col2 = columns[0], columns[1]
        contingency = pd.crosstab(df[col1], df[col2])
        stat, p_value, dof, expected = scipy_stats.chi2_contingency(contingency)
        return {
            "test": "Chi-square test of independence",
            "columns": [col1, col2],
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 6),
            "degrees_of_freedom": int(dof),
            "significant": p_value < 0.05,
            "interpretation": f"{'Significant' if p_value < 0.05 else 'No significant'} association between variables (p={p_value:.4f})"
        }

    else:
        return {"error": f"Unknown test type: {test_type}. Available: ttest, anova, chi2"}


def _frequency(df: pd.DataFrame, columns: list[str], group_by: str | None, params: dict) -> dict:
    """Compute frequency distributions."""
    results = {}
    for col in columns:
        if col not in df.columns:
            continue
        vc = df[col].value_counts()
        total = len(df[col].dropna())
        results[col] = {
            "unique_count": int(df[col].nunique()),
            "total_count": total,
            "top_values": [
                {"value": str(k), "count": int(v), "percentage": round(float(v / total * 100), 2)}
                for k, v in vc.head(20).items()
            ],
        }
    return {"results": results}


def _outlier_detection(df: pd.DataFrame, columns: list[str], group_by: str | None, params: dict) -> dict:
    """Detect outliers using IQR or Z-score method."""
    method = params.get("method", "iqr")  # iqr or zscore
    threshold = params.get("threshold", 3.0 if method == "zscore" else 1.5)

    numeric_cols = df[columns].select_dtypes(include="number").columns.tolist()
    results = {}

    for col in numeric_cols:
        data = df[col].dropna()
        if method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outliers = data[(data < lower) | (data > upper)]
            results[col] = {
                "method": "IQR",
                "threshold": threshold,
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
                "outlier_count": int(len(outliers)),
                "outlier_percentage": round(float(len(outliers) / len(data) * 100), 2),
            }
        elif method == "zscore":
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = data[z_scores > threshold]
            results[col] = {
                "method": "Z-score",
                "threshold": threshold,
                "outlier_count": int(len(outliers)),
                "outlier_percentage": round(float(len(outliers) / len(data) * 100), 2),
            }

    return {"method": method, "results": results}


# Handler registry
_ANALYSIS_HANDLERS = {
    "descriptive": _descriptive,
    "correlation": _correlation,
    "distribution_test": _distribution_test,
    "group_stats": _group_stats,
    "hypothesis_test": _hypothesis_test,
    "frequency": _frequency,
    "outlier_detection": _outlier_detection,
}


def get_tool_def() -> dict:
    """Return the Claude tool definition for analyze_stats."""
    return {
        "name": "analyze_stats",
        "description": (
            "Perform statistical analysis on loaded data. Supports: "
            "descriptive statistics, correlation analysis, distribution tests (normality), "
            "group-by aggregation, hypothesis tests (t-test, ANOVA, chi-square), "
            "frequency distributions, and outlier detection."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "data_id": {
                    "type": "string",
                    "description": "Reference ID of the previously loaded dataset."
                },
                "analysis_type": {
                    "type": "string",
                    "enum": [
                        "descriptive", "correlation", "distribution_test",
                        "group_stats", "hypothesis_test", "frequency",
                        "outlier_detection"
                    ],
                    "description": "Type of statistical analysis to perform."
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Target columns for analysis. If omitted, uses all applicable columns."
                },
                "group_by": {
                    "type": "string",
                    "description": "Column to group by (required for group_stats, optional for hypothesis_test)."
                },
                "params": {
                    "type": "object",
                    "description": "Additional parameters. Examples: "
                    "{'method': 'spearman'} for correlation, "
                    "{'test': 'shapiro'} for distribution_test, "
                    "{'test': 'anova'} for hypothesis_test, "
                    "{'agg_funcs': ['mean', 'median']} for group_stats, "
                    "{'method': 'zscore', 'threshold': 3.0} for outlier_detection.",
                    "properties": {
                        "method": {
                            "type": "string",
                            "description": "Method variant (e.g., 'pearson'/'spearman' for correlation, 'iqr'/'zscore' for outliers)."
                        },
                        "test": {
                            "type": "string",
                            "description": "Test type (e.g., 'shapiro'/'ks' for distribution, 'ttest'/'anova'/'chi2' for hypothesis)."
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Threshold value (e.g., IQR multiplier or Z-score cutoff)."
                        },
                        "agg_funcs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Aggregation functions for group_stats (e.g., ['mean', 'std', 'count'])."
                        }
                    }
                }
            },
            "required": ["data_id", "analysis_type"]
        }
    }
