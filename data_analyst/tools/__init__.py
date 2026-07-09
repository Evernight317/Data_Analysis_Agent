"""Tool definitions for the Data Analysis Agent."""

from data_analyst.tools.data_loader import load_data, get_tool_def as load_data_def
from data_analyst.tools.stats import analyze_stats, get_tool_def as stats_def
from data_analyst.tools.visualize import create_chart, get_tool_def as visualize_def
from data_analyst.tools.ml_train import train_model, get_tool_def as ml_train_def
from data_analyst.tools.ml_predict import predict, get_tool_def as ml_predict_def

ALL_TOOLS = [load_data_def(), stats_def(), visualize_def(), ml_train_def(), ml_predict_def()]

TOOL_HANDLERS = {
    "load_data": load_data,
    "analyze_stats": analyze_stats,
    "create_chart": create_chart,
    "train_model": train_model,
    "predict": predict,
}
