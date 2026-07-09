"""Configuration management for the Data Analysis Agent."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# --- API Configuration ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", None)
MODEL_NAME = os.getenv("MODEL_NAME", "claude-sonnet-4-6")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

# --- Agent Configuration ---
MAX_TURNS = int(os.getenv("MAX_TURNS", "20"))

# --- Output Configuration ---
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "output")))
CHARTS_DIR = OUTPUT_DIR / "charts"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

# --- Chart Configuration ---
CHART_FORMAT = os.getenv("CHART_FORMAT", "png")  # png or svg
CHART_DPI = int(os.getenv("CHART_DPI", "150"))

# --- Chinese Font Configuration ---
# Try common Chinese fonts on Windows
_CHINESE_FONTS = [
    "Microsoft YaHei",      # 微软雅黑
    "SimHei",               # 黑体
    "SimSun",               # 宋体
    "FangSong",             # 仿宋
    "KaiTi",                # 楷体
]

CHINESE_FONT = os.getenv("CHINESE_FONT", "Microsoft YaHei")


def ensure_output_dirs():
    """Create output directories if they don't exist."""
    for d in [CHARTS_DIR, MODELS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def validate_config():
    """Validate configuration and return list of issues."""
    issues = []
    if not ANTHROPIC_API_KEY:
        issues.append("ANTHROPIC_API_KEY is not set. Please set it in .env file or environment variable.")
    if CHART_FORMAT not in ("png", "svg"):
        issues.append(f"Invalid CHART_FORMAT: {CHART_FORMAT}. Must be 'png' or 'svg'.")
    return issues


def get_config_summary() -> dict:
    """Return a summary of current configuration."""
    return {
        "model": MODEL_NAME,
        "max_turns": MAX_TURNS,
        "output_dir": str(OUTPUT_DIR),
        "chart_format": CHART_FORMAT,
        "chart_dpi": CHART_DPI,
        "chinese_font": CHINESE_FONT,
        "api_key_set": bool(ANTHROPIC_API_KEY),
    }
