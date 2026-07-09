"""Utility functions for the Data Analysis Agent."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def truncate_dict(data: dict, max_str_len: int = 200, max_list_len: int = 20) -> dict:
    """Truncate long strings and lists in a dict for display."""
    result = {}
    for key, value in data.items():
        if isinstance(value, str) and len(value) > max_str_len:
            result[key] = value[:max_str_len] + "..."
        elif isinstance(value, list) and len(value) > max_list_len:
            result[key] = value[:max_list_len] + [f"... ({len(value)} total)"]
        elif isinstance(value, dict):
            result[key] = truncate_dict(value, max_str_len, max_list_len)
        else:
            result[key] = value
    return result


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def safe_json_dumps(data: Any, **kwargs) -> str:
    """JSON dumps that handles non-serializable types."""
    return json.dumps(data, ensure_ascii=False, default=str, **kwargs)


def generate_sample_data(output_path: str, n_rows: int = 200) -> str:
    """Generate a sample CSV dataset for testing.

    Creates a housing price dataset with features like area, bedrooms,
    age, location, and price.
    """
    import numpy as np

    np.random.seed(42)

    data = {
        "面积": np.random.normal(100, 30, n_rows).clip(40, 250).round(1),
        "卧室数": np.random.choice([1, 2, 3, 4, 5], n_rows, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
        "楼龄": np.random.exponential(10, n_rows).clip(0, 50).round(1),
        "楼层": np.random.randint(1, 33, n_rows),
        "区域": np.random.choice(["朝阳区", "海淀区", "西城区", "东城区", "丰台区"], n_rows),
        "是否学区": np.random.choice([0, 1], n_rows, p=[0.7, 0.3]),
        "是否有地铁": np.random.choice([0, 1], n_rows, p=[0.4, 0.6]),
    }

    # Generate price based on features with some noise
    price = (
        data["面积"] * 500
        + data["卧室数"] * 50000
        - data["楼龄"] * 2000
        + data["是否学区"] * 200000
        + data["是否有地铁"] * 100000
        + np.random.normal(0, 50000, n_rows)
    )
    data["房价"] = price.clip(200000, 3000000).round(0).astype(int)

    df = pd.DataFrame(data)

    # Add some missing values
    mask = np.random.random((n_rows, len(df.columns))) < 0.02
    for i, col in enumerate(df.columns):
        df.loc[mask[:, i], col] = None

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path
