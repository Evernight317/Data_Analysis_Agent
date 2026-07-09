"""Data loading tool for the Data Analysis Agent.

Supports CSV, Excel, JSON, TSV formats with automatic encoding detection,
data summarization, and basic cleaning options.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from data_analyst.config import ensure_output_dirs

# --- Data Registry ---
# Stores loaded DataFrames in memory, keyed by data_id
_data_registry: dict[str, pd.DataFrame] = {}
_next_id = 1


def _generate_data_id() -> str:
    """Generate a unique data ID."""
    global _next_id
    data_id = f"data_{_next_id}"
    _next_id += 1
    return data_id


def get_dataframe(data_id: str) -> pd.DataFrame:
    """Retrieve a DataFrame from the registry by its ID.

    Raises KeyError if the data_id is not found.
    """
    if data_id not in _data_registry:
        available = list(_data_registry.keys())
        raise KeyError(
            f"Data ID '{data_id}' not found. Available: {available}"
        )
    return _data_registry[data_id]


def list_loaded_data() -> list[dict[str, Any]]:
    """Return summary of all loaded datasets."""
    result = []
    for data_id, df in _data_registry.items():
        result.append({
            "data_id": data_id,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
        })
    return result


def _summarize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Generate a comprehensive summary of a DataFrame."""
    summary = {
        "shape": list(df.shape),
        "rows": df.shape[0],
        "columns": list(df.columns),
        "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
        "missing_values": {
            col: int(df[col].isna().sum())
            for col in df.columns
            if df[col].isna().sum() > 0
        },
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    }

    # Numeric column statistics
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        desc = df[numeric_cols].describe().to_dict()
        # Convert numpy types to Python native
        summary["numeric_summary"] = {
            col: {k: round(float(v), 4) if isinstance(v, float) else int(v)
                  for k, v in stats.items()}
            for col, stats in desc.items()
        }

    # Categorical column top values
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if cat_cols:
        summary["categorical_summary"] = {}
        for col in cat_cols:
            vc = df[col].value_counts().head(5)
            summary["categorical_summary"][col] = {
                "unique_count": int(df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in vc.items()},
            }

    # Preview (first 5 rows)
    preview = df.head(5).to_dict(orient="list")
    # Truncate long string values
    for col, vals in preview.items():
        preview[col] = [
            (str(v)[:100] + "...") if isinstance(v, str) and len(str(v)) > 100 else v
            for v in vals
        ]
    summary["preview"] = preview

    return summary


def _detect_encoding(file_path: str) -> str:
    """Try to detect file encoding."""
    try:
        import chardet
        with open(file_path, "rb") as f:
            raw = f.read(10000)
            result = chardet.detect(raw)
            if result["confidence"] > 0.7:
                return result["encoding"]
    except ImportError:
        pass
    return "utf-8"


def load_data(
    file_path: str,
    encoding: str | None = None,
    sheet_name: str | int | None = None,
    clean_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load data from a file and register it in the data registry.

    Args:
        file_path: Path to the data file (CSV, Excel, JSON, TSV).
        encoding: File encoding. Auto-detected if not specified.
        sheet_name: Excel sheet name or index (for Excel files only).
        clean_options: Data cleaning options:
            - drop_na_rows: Drop rows with any missing values (bool)
            - drop_na_cols: Drop columns with any missing values (bool)
            - fill_na: Value to fill missing values with, or "mean"/"median"/"mode"
            - remove_duplicates: Remove duplicate rows (bool)
            - remove_outliers: Remove outliers using IQR method (bool or list of columns)

    Returns:
        Dictionary with data_id and data summary.
    """
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    suffix = path.suffix.lower()

    try:
        # Load based on file type
        if suffix == ".csv":
            enc = encoding or _detect_encoding(file_path)
            # Try different separators; use utf-8-sig to handle BOM
            if enc.lower().replace("-", "") in ("utf8", "utf8sig"):
                enc = "utf-8-sig"
            df = pd.read_csv(file_path, encoding=enc, sep=None, engine="python")
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(file_path, sheet_name=sheet_name or 0)
        elif suffix == ".json":
            df = pd.read_json(file_path)
        elif suffix == ".tsv":
            enc = encoding or _detect_encoding(file_path)
            df = pd.read_csv(file_path, encoding=enc, sep="\t")
        else:
            return {"error": f"Unsupported file format: {suffix}. Supported: .csv, .xlsx, .xls, .json, .tsv"}

        # Apply cleaning options
        if clean_options:
            df = _clean_data(df, clean_options)

        # Register the DataFrame
        data_id = _generate_data_id()
        _data_registry[data_id] = df

        # Generate summary
        summary = _summarize_dataframe(df)
        summary["data_id"] = data_id
        summary["file_path"] = str(path)
        summary["file_type"] = suffix

        return summary

    except Exception as e:
        return {"error": f"Failed to load data: {str(e)}"}


def _clean_data(df: pd.DataFrame, options: dict[str, Any]) -> pd.DataFrame:
    """Apply data cleaning operations."""
    df = df.copy()

    if options.get("drop_na_rows"):
        df = df.dropna()

    if options.get("drop_na_cols"):
        df = df.dropna(axis=1)

    fill_na = options.get("fill_na")
    if fill_na is not None:
        if fill_na == "mean":
            numeric_cols = df.select_dtypes(include="number").columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif fill_na == "median":
            numeric_cols = df.select_dtypes(include="number").columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif fill_na == "mode":
            for col in df.columns:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val.iloc[0])
        else:
            df = df.fillna(fill_na)

    if options.get("remove_duplicates"):
        df = df.drop_duplicates()

    if options.get("remove_outliers"):
        outlier_cols = options["remove_outliers"]
        if isinstance(outlier_cols, bool):
            outlier_cols = df.select_dtypes(include="number").columns.tolist()
        df = _remove_outliers_iqr(df, outlier_cols)

    return df


def _remove_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Remove outliers using the IQR method."""
    mask = pd.Series(True, index=df.index)
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            mask &= (df[col] >= lower) & (df[col] <= upper)
    return df[mask]


def get_tool_def() -> dict:
    """Return the Claude tool definition for load_data."""
    return {
        "name": "load_data",
        "description": (
            "Load data from a file (CSV, Excel, JSON, TSV) into the analysis workspace. "
            "Returns a data_id that can be used in subsequent analysis, visualization, and ML tools. "
            "Also returns a summary of the data including shape, columns, types, missing values, "
            "and a preview of the first 5 rows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the data file to load (CSV, Excel .xlsx/.xls, JSON, or TSV)."
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (e.g., 'utf-8', 'gbk', 'gb2312'). Auto-detected if not specified."
                },
                "sheet_name": {
                    "type": ["string", "integer"],
                    "description": "Excel sheet name or index (0-based). Only for Excel files."
                },
                "clean_options": {
                    "type": "object",
                    "description": "Data cleaning options to apply after loading.",
                    "properties": {
                        "drop_na_rows": {
                            "type": "boolean",
                            "description": "Drop rows with any missing values."
                        },
                        "drop_na_cols": {
                            "type": "boolean",
                            "description": "Drop columns with any missing values."
                        },
                        "fill_na": {
                            "type": ["string", "number"],
                            "description": "Fill missing values. Use a specific value, or 'mean', 'median', 'mode'."
                        },
                        "remove_duplicates": {
                            "type": "boolean",
                            "description": "Remove duplicate rows."
                        },
                        "remove_outliers": {
                            "type": ["boolean", "array"],
                            "description": "Remove outliers using IQR method. True for all numeric columns, or list of column names."
                        }
                    }
                }
            },
            "required": ["file_path"]
        }
    }
