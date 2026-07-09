# 📊 Data Analyst Agent

> A natural language-driven data analysis, statistical visualization, and machine learning prediction tool powered by Claude

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Interactive Mode](#interactive-mode)
  - [One-Shot Mode](#one-shot-mode)
  - [Generate Sample Data](#generate-sample-data)
  - [Built-in Commands](#built-in-commands)
- [Tool Reference](#tool-reference)
  - [Data Loading (load_data)](#data-loading-load_data)
  - [Statistical Analysis (analyze_stats)](#statistical-analysis-analyze_stats)
  - [Data Visualization (create_chart)](#data-visualization-create_chart)
  - [ML Training (train_model)](#ml-training-train_model)
  - [ML Prediction (predict)](#ml-prediction-predict)
- [Supported Data Formats](#supported-data-formats)
- [Supported Chart Types](#supported-chart-types)
- [Supported ML Algorithms](#supported-ml-algorithms)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [FAQ](#faq)

---

## Introduction

Data Analyst Agent is a standalone Python application that leverages Claude's Tool Calling capability to enable full-pipeline data analysis through natural language — from data loading and statistical analysis to chart generation and machine learning prediction.

No coding required. Simply describe your analysis needs in natural language (English or Chinese), and the agent automatically understands your intent, selects the appropriate tools, executes the analysis, and returns the results.

---

## Features

### 📂 Data Loading & Preprocessing
- Supports CSV, Excel (.xlsx/.xls), JSON, and TSV formats
- Automatic file encoding detection (including Chinese encodings GBK/GB2312)
- Data cleaning: missing value handling, deduplication, outlier removal
- Auto-generated data summary: row/column count, data types, missing values, statistical distribution, first 5 rows preview

### 📈 Statistical Analysis (7 analysis types)
- **Descriptive Statistics** — mean, std, quartiles, skewness, kurtosis
- **Correlation Analysis** — Pearson / Spearman correlation matrix, automatic strong correlation detection
- **Distribution Tests** — Shapiro-Wilk / Kolmogorov-Smirnov normality test
- **Group Statistics** — aggregation by categorical column
- **Hypothesis Testing** — independent t-test, one-way ANOVA, chi-square test
- **Frequency Distribution** — value counts and percentages
- **Outlier Detection** — IQR method / Z-score method

### 📊 Data Visualization (11 chart types)
- Line chart, bar chart, scatter plot, histogram
- Box plot, violin plot, heatmap
- Pairplot, pie chart, count plot, area chart
- CJK label support (Microsoft YaHei / SimHei)
- Output format: PNG (default) or SVG

### 🤖 Machine Learning (3 task types)
- **Regression** — Linear, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, SVR, KNN, Decision Tree
- **Classification** — Logistic, Random Forest, Gradient Boosting, SVC, KNN, Decision Tree, Naive Bayes
- **Clustering** — K-Means, DBSCAN, Agglomerative Clustering
- Automatic feature engineering: standardization, One-Hot encoding, missing value imputation
- Cross-validation and hyperparameter search (GridSearchCV)
- Model persistence: save/load trained models
- Prediction: batch and single-row, classification supports probability output

---

## Quick Start

### 1. Requirements

- Python 3.10+
- Anthropic API Key

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Copy `.env.example` to `.env` and fill in your API key:

```bash
cp .env.example .env
```

Edit `.env`:

```ini
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 4. Run

```bash
# Generate sample data
python main.py --sample

# Interactive mode
python main.py

# Preload a data file
python main.py -f output/sample_housing.csv
```

---

## Usage

### Interactive Mode

Launch the interactive dialogue interface and type analysis requests in natural language:

```bash
$ python main.py
```

```
📊 Data Analysis Agent
Data analysis agent - natural language driven data analysis, visualization & ML

Enter natural language instructions, type 'help' for help, 'quit' to exit

You: Load output/sample_housing.csv
```

The agent automatically calls the `load_data` tool and returns a data summary.

```
You: Analyze the correlation between all columns and price
```

The agent calls `analyze_stats` to perform correlation analysis.

```
You: Plot a correlation heatmap
```

The agent calls `create_chart` to generate a heatmap saved as PNG.

```
You: Predict price using random forest
```

The agent calls `train_model` to train a model and returns evaluation metrics.

```
You: Predict new data using the trained model
```

The agent calls the `predict` tool to execute predictions.

### One-Shot Mode

For scripted execution or quick queries:

```bash
python main.py -f data.csv -q "Analyze price drivers, plot a correlation heatmap, predict price with random forest"
```

### Generate Sample Data

```bash
python main.py --sample
```

Generates `output/sample_housing.csv` with 200 simulated housing records:
- Area, bedrooms, building age, floor, district, is_school_district, has_subway, price
- Contains a small number of missing values, ideal for demonstrating data cleaning and analysis

### Built-in Commands

In interactive mode, the following commands are available:

| Command | Description |
|---------|-------------|
| `help` | Show help information |
| `status` | Show current status (loaded datasets, models, etc.) |
| `sample` | Generate sample data |
| `reset` | Reset conversation history |
| `quit` / `exit` | Exit the program |

### Command-Line Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--file` | `-f` | Data file to preload |
| `--query` | `-q` | Single query (non-interactive mode) |
| `--model` | `-m` | Specify Claude model name |
| `--output` | `-o` | Output directory |
| `--sample` | | Generate sample data |
| `--api-key` | | API key (overrides .env config) |

---

## Tool Reference

### Data Loading (load_data)

Loads a file and registers the data into an in-memory workspace. Subsequent tools reference data by `data_id`.

**Supported formats**: CSV, Excel (.xlsx/.xls), JSON, TSV

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_path` | string | ✅ | Path to the data file |
| `encoding` | string | | File encoding; auto-detected by default |
| `sheet_name` | string/int | | Excel sheet name or index |
| `clean_options` | object | | Data cleaning options |

**Cleaning Options (clean_options)**:

| Option | Type | Description |
|--------|------|-------------|
| `drop_na_rows` | boolean | Drop rows with any missing values |
| `drop_na_cols` | boolean | Drop columns with any missing values |
| `fill_na` | string/number | Fill missing values: a specific value, "mean", "median", or "mode" |
| `remove_duplicates` | boolean | Remove duplicate rows |
| `remove_outliers` | boolean/array | Remove outliers via IQR: true = all numeric columns, or a list of column names |

**Natural language examples**:
```
Load data.csv and drop missing values
Load data.xlsx first sheet, fill missing values with mean
```

---

### Statistical Analysis (analyze_stats)

Performs statistical analysis on loaded data.

**Analysis types**:

| Type | Description | Key Parameters |
|------|-------------|----------------|
| `descriptive` | Descriptive statistics | — |
| `correlation` | Correlation analysis | `method`: pearson / spearman |
| `distribution_test` | Normality test | `test`: shapiro / ks |
| `group_stats` | Group-by aggregation | `group_by`: column name, `agg_funcs`: list of aggregation functions |
| `hypothesis_test` | Hypothesis testing | `test`: ttest / anova / chi2 |
| `frequency` | Frequency distribution | — |
| `outlier_detection` | Outlier detection | `method`: iqr / zscore, `threshold`: cutoff value |

**Natural language examples**:
```
Analyze basic statistics of the data
Analyze correlation between area and price using Spearman method
Test if price follows a normal distribution
Group by district and compute mean and std of price
Compare prices across districts using ANOVA
Detect which columns have outliers
```

---

### Data Visualization (create_chart)

Generates a static chart and saves it to the `output/charts/` directory.

**Chart types**:

| Type | Description | Required Params |
|------|-------------|----------------|
| `line` | Line chart | x, y |
| `bar` | Bar chart | x, y |
| `scatter` | Scatter plot | x, y |
| `histogram` | Histogram | x |
| `box` | Box plot | y (optional x for grouping) |
| `violin` | Violin plot | y (optional x for grouping) |
| `heatmap` | Heatmap | — (auto-uses numeric columns) |
| `pairplot` | Pairwise relationships | — |
| `pie` | Pie chart | x (counts) or x+y (values) |
| `count` | Count plot | x or y |
| `area` | Area chart | x, y |

**Common parameters**:

| Parameter | Description |
|-----------|-------------|
| `hue` | Color grouping column |
| `title` | Chart title |
| `xlabel` / `ylabel` | Axis labels |
| `filename` | Output filename (without extension) |
| `save_format` | png or svg |

**Natural language examples**:
```
Plot a histogram of price distribution
Plot a scatter of area vs price
Plot a box plot of price by district
Plot a correlation heatmap
Plot a pie chart of district proportions, show top 5 only
Plot a line chart of price over building age
```

---

### ML Training (train_model)

Trains a scikit-learn machine learning model with automatic feature engineering.

**Task types**:

| Type | Description | Required Params |
|------|-------------|----------------|
| `regression` | Regression | `target_column` |
| `classification` | Classification | `target_column` |
| `clustering` | Clustering | — |

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `algorithm` | string | Algorithm name; auto-selected if omitted |
| `feature_columns` | array | Feature columns; auto-selected if omitted |
| `tune_hyperparams` | boolean | Enable hyperparameter search; default false |
| `cv_folds` | int | Cross-validation folds; default 5 |
| `test_size` | float | Test set proportion; default 0.2 |
| `random_state` | int | Random seed; default 42 |

**Automatic feature engineering**:
- Numeric columns: median imputation + standardization
- Categorical columns: mode imputation + One-Hot encoding
- Automatically composed into an sklearn Pipeline

**Evaluation metrics**:
- Regression: R², RMSE, MAE, cross-validated R²
- Classification: Accuracy, F1 (weighted), Precision, Recall, AUC (binary), confusion matrix, classification report
- Clustering: Silhouette score, Calinski-Harabasz index

**Natural language examples**:
```
Predict price using random forest
Classify school district status using logistic regression
Cluster the data using K-Means
Predict price using gradient boosting with hyperparameter tuning
```

---

### ML Prediction (predict)

Makes predictions on new data using a previously trained model.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | string | ✅ Path to the saved model file (returned by train_model) |
| `data_id` | string | ID of loaded data (batch prediction) |
| `input_data` | object | Single-row data as a dict (one-shot prediction) |
| `return_proba` | boolean | Return classification probabilities; default false |

**Natural language examples**:
```
Predict using the trained model on the dataset
Predict the price of this house: area 100, 3 bedrooms, age 5, floor 10, school district, has subway
```

---

## Supported Data Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| CSV | `.csv` | Auto-detects encoding and delimiter; supports UTF-8 BOM |
| Excel | `.xlsx`, `.xls` | Requires openpyxl; can specify sheet |
| JSON | `.json` | Standard format |
| TSV | `.tsv` | Tab-separated |

---

## Supported Chart Types

| Chart | chart_type | Best For |
|-------|------------|----------|
| Line | `line` | Time series, trends |
| Bar | `bar` | Category comparison |
| Scatter | `scatter` | Two-variable relationship |
| Histogram | `histogram` | Single-variable distribution |
| Box | `box` | Grouped distribution & outliers |
| Violin | `violin` | Distribution shape comparison |
| Heatmap | `heatmap` | Correlation matrix |
| Pairplot | `pairplot` | Pairwise multi-variable relationships |
| Pie | `pie` | Proportional breakdown |
| Count | `count` | Category counts |
| Area | `area` | Cumulative trends |

---

## Supported ML Algorithms

### Regression

| Algorithm | Name | Description |
|-----------|------|-------------|
| Linear Regression | `linear` | Baseline linear model |
| Ridge | `ridge` | L2 regularization |
| Lasso | `lasso` | L1 regularization, feature selection |
| ElasticNet | `elasticnet` | L1 + L2 regularization |
| Random Forest | `random_forest` | Ensemble method (default) |
| Gradient Boosting | `gradient_boosting` | Ensemble method, high accuracy |
| SVR | `svr` | Non-linear regression |
| KNN Regressor | `knn` | Distance-weighted |
| Decision Tree | `decision_tree` | High interpretability |

### Classification

| Algorithm | Name | Description |
|-----------|------|-------------|
| Logistic Regression | `logistic` | Linear classifier |
| Random Forest | `random_forest` | Ensemble method (default) |
| Gradient Boosting | `gradient_boosting` | Ensemble method, high accuracy |
| SVC | `svc` | Non-linear classification |
| KNN Classifier | `knn` | Distance-based |
| Decision Tree | `decision_tree` | High interpretability |
| Naive Bayes | `naive_bayes` | Text / high-dimensional data |

### Clustering

| Algorithm | Name | Description |
|-----------|------|-------------|
| K-Means | `kmeans` | Distance-based (default) |
| DBSCAN | `dbscan` | Density-based, arbitrary shapes |
| Agglomerative | `agglomerative` | Hierarchical agglomeration |

---

## Configuration

All configuration is managed via the `.env` file or environment variables. Copy `.env.example` to `.env` and modify:

```ini
# === Anthropic API ===
ANTHROPIC_API_KEY=sk-ant-...           # Required: your API key
ANTHROPIC_BASE_URL=                    # Optional: custom API endpoint (proxy)

# === Model ===
MODEL_NAME=claude-sonnet-4-6           # Claude model name
MAX_TOKENS=4096                        # Max output tokens

# === Agent ===
MAX_TURNS=20                           # Max agent loop iterations

# === Output ===
OUTPUT_DIR=output                      # Output root directory
CHART_FORMAT=png                       # Chart format: png / svg
CHART_DPI=150                          # Chart DPI

# === CJK Font (Windows) ===
CHINESE_FONT=Microsoft YaHei           # CJK font name
```

---

## Project Structure

```
untitled/
├── main.py                          # CLI entry: interactive / one-shot mode
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variable template
├── .gitignore                       # Git ignore rules
├── README.md                        # Project readme
├── PROJECT_DOC_EN.md                # This document
│
├── data_analyst/                    # Main package
│   ├── __init__.py                  # Package init
│   ├── agent.py                     # Agent core: Claude API agentic loop
│   ├── config.py                    # Config management + .env loading
│   ├── utils.py                     # Utilities + sample data generator
│   │
│   └── tools/                       # Tool suite
│       ├── __init__.py              # Tool registry
│       ├── data_loader.py           # Data loading + Data Registry
│       ├── stats.py                 # Statistical analysis (7 types)
│       ├── visualize.py             # Data visualization (11 chart types)
│       ├── ml_train.py              # ML model training (3 task types)
│       └── ml_predict.py            # ML model prediction
│
└── output/                          # Output directory (auto-created)
    ├── charts/                      # Generated chart files
    ├── models/                      # Saved model files (.pkl)
    └── reports/                     # Analysis reports
```

---

## How It Works

### Agentic Tool-Use Loop

```
┌──────────────────────────────────────────────────┐
│  User input (natural language)                    │
│  "Load data.csv, analyze price trends, predict    │
│   with random forest"                             │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────┐
│  Claude LLM                                       │
│  Understands intent → Selects Tool → Generates    │
│  tool call parameters                             │
│                                                   │
│  Tool 1: load_data(file_path="data.csv")          │
│  Tool 2: analyze_stats(analysis_type="group_      │
│           stats", group_by="district", ...)        │
│  Tool 3: create_chart(chart_type="line", ...)     │
│  Tool 4: train_model(task_type="regression", ...) │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────┐
│  Tool Execution Layer                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │data_loader│ │  stats   │ │ visualize│          │
│  └──────────┘ └──────────┘ └──────────┘          │
│  ┌──────────┐ ┌──────────┐                        │
│  │ml_train  │ │ml_predict│                        │
│  └──────────┘ └──────────┘                        │
│                                                   │
│  Data Registry (in-memory)                        │
│  ┌──────────────────────────────────┐             │
│  │ data_1 → DataFrame (200×8)      │             │
│  │ data_2 → DataFrame (50×5)       │             │
│  └──────────────────────────────────┘             │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────┐
│  Results                                          │
│  ┌─ output/charts/chart_line_data_1.png           │
│  ├─ output/models/random_forest_regression_...pkl │
│  └─ Text interpretation (generated by Claude)     │
└──────────────────────────────────────────────────┘
```

### Data Passing Mechanism

1. `load_data` stores the DataFrame in an in-memory **Data Registry** and returns a `data_id`
2. Subsequent tools reference data by `data_id` — no raw data is passed around
3. Only data summaries (shape, columns, dtypes, preview) are sent in the LLM context, not raw data
4. Multiple datasets can be loaded simultaneously, each with its own `data_id`

### Security Design

- **Tool-Based** architecture: the LLM never directly generates or executes Python code
- All operations go through predefined Tool functions, each with internal parameter validation
- The LLM can only call registered tools — it cannot execute arbitrary code

---

## FAQ

### Q: Startup shows "ANTHROPIC_API_KEY is not set"

Make sure a `.env` file exists in the project root with a valid API key:

```ini
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Q: Chinese characters display as boxes in charts

Ensure `CHINESE_FONT` in `.env` is set to a font installed on your system. The default on Windows is `Microsoft YaHei`; you can also use `SimHei`.

### Q: Encoding error when loading CSV

For Chinese CSV files, manually specify the encoding:

```
Load data.csv with encoding gbk
```

### Q: Model training takes too long

- Reduce data volume: use `clean_options` when loading to filter data
- Disable hyperparameter search: don't use `tune_hyperparams`
- Choose a simpler algorithm: e.g., `linear` or `decision_tree`

### Q: How to use a saved model

Training returns a `model_path` that can be referenced in subsequent conversations:

```
Predict new data using model at output/models/random_forest_regression_data_1.pkl
```

### Q: How to specify API key on the command line

```bash
python main.py --api-key sk-ant-your-key-here
```

### Q: Can I use a custom API endpoint?

Yes. Set `ANTHROPIC_BASE_URL` in `.env`:

```ini
ANTHROPIC_BASE_URL=https://your-proxy.example.com
```

### Q: Where are output files saved?

By default in the project root's `output/` directory:
- Charts: `output/charts/`
- Models: `output/models/`

Can be changed via the `--output` argument or `OUTPUT_DIR` in `.env`.

---

## Example Session

```
$ python main.py

📊 Data Analysis Agent
Data analysis agent - natural language driven data analysis, visualization & ML

You: Load output/sample_housing.csv

🤔 Loaded data_1: 200 rows, 8 columns:
  area, bedrooms, building_age, floor, district, is_school_district, has_subway, price
  Missing values: area(7), bedrooms(1), building_age(5), ...

You: Drop missing values, then analyze correlation with price

🤔 Cleaned data: 170 rows. Correlation results (Pearson):
  - area ↔ price: 0.72 (strong positive)
  - is_school_district ↔ price: 0.58 (moderate positive)
  - building_age ↔ price: -0.35 (weak negative)

You: Plot a correlation heatmap

🤔 Heatmap saved: output/charts/chart_heatmap_data_1.png

You: Predict price using random forest

🤔 Random Forest regression model trained:
  - R²: 0.63
  - RMSE: 53,531
  - Cross-validated R²: 0.73
  Model saved: output/models/random_forest_regression_data_1.pkl

You: quit

Goodbye! 👋
```

---

## License

MIT License
