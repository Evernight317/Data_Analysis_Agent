# Data Analyst Agent — 数据分析智能体

> 基于自然语言交互的数据分析、统计可视化与机器学习预测工具

---

## 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [使用说明](#使用说明)
  - [交互式模式](#交互式模式)
  - [单次执行模式](#单次执行模式)
  - [生成示例数据](#生成示例数据)
  - [内置命令](#内置命令)
- [工具详细说明](#工具详细说明)
  - [数据加载 (load_data)](#数据加载-load_data)
  - [统计分析 (analyze_stats)](#统计分析-analyze_stats)
  - [数据可视化 (create_chart)](#数据可视化-create_chart)
  - [机器学习训练 (train_model)](#机器学习训练-train_model)
  - [机器学习预测 (predict)](#机器学习预测-predict)
- [支持的数据格式](#支持的数据格式)
- [支持的图表类型](#支持的图表类型)
- [支持的机器学习算法](#支持的机器学习算法)
- [配置说明](#配置说明)
- [项目结构](#项目结构)
- [工作原理](#工作原理)
- [常见问题](#常见问题)

---

## 项目简介

Data Analyst Agent 是一个独立的 Python 应用，利用 Claude 大语言模型的 Tool Calling 能力，让用户通过自然语言即可完成从数据加载、统计分析、图表绘制到机器学习预测的全流程数据分析工作。

无需编写代码，只需用中文或英文描述你的分析需求，Agent 会自动理解意图、选择合适的工具、执行分析并返回结果。

---

## 功能特性

### 数据加载与预处理
- 支持 CSV、Excel (.xlsx/.xls)、JSON、TSV 格式
- 自动检测文件编码（含中文编码 GBK/GB2312）
- 数据清洗：缺失值处理、去重、异常值移除
- 自动生成数据摘要：行数、列数、类型、缺失值、统计分布、前 5 行预览

### 统计分析（7 种分析类型）
- **描述性统计** — 均值、标准差、四分位数、偏度、峰度
- **相关性分析** — Pearson / Spearman 相关系数矩阵，自动识别强相关
- **分布检验** — Shapiro-Wilk / Kolmogorov-Smirnov 正态性检验
- **分组统计** — 按分类列分组的聚合统计
- **假设检验** — 独立样本 t 检验、单因素方差分析 (ANOVA)、卡方检验
- **频率分布** — 值计数与百分比
- **异常值检测** — IQR 方法 / Z-score 方法

### 数据可视化（11 种图表）
- 折线图、柱状图、散点图、直方图
- 箱线图、小提琴图、热力图
- 成对关系图 (pairplot)、饼图、计数图、面积图
- 中文标签支持（微软雅黑/黑体）
- 输出格式：PNG（默认）或 SVG

### 机器学习（3 大任务类型）
- **回归** — 线性回归、Ridge、Lasso、ElasticNet、随机森林、梯度提升、SVR、KNN、决策树
- **分类** — 逻辑回归、随机森林、梯度提升、SVC、KNN、决策树、朴素贝叶斯
- **聚类** — K-Means、DBSCAN、层次聚类
- 自动特征工程：标准化、One-Hot 编码、缺失值填充
- 交叉验证与超参数搜索 (GridSearchCV)
- 模型持久化：保存/加载训练好的模型
- 预测：批量预测与单条预测，分类支持概率输出

---

## 快速开始

### 1. 环境要求

- Python 3.10+
- Anthropic API Key

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```ini
ANTHROPIC_API_KEY=sk-ant-你的密钥
```

### 4. 运行

```bash
# 生成示例数据
python main.py --sample

# 交互式模式
python main.py

# 预加载数据文件
python main.py -f output/sample_housing.csv
```

---

## 使用说明

### 交互式模式

启动后进入交互式对话界面，用自然语言输入分析需求：

```bash
$ python main.py
```

```
Data Analysis Agent
数据分析智能体 - 自然语言驱动的数据分析、可视化与机器学习

输入自然语言指令进行分析，输入 'help' 查看帮助，'quit' 退出

You: 加载 output/sample_housing.csv
```

Agent 会自动调用 `load_data` 工具加载数据，返回数据摘要。

```
You: 分析各列与房价的相关性
```

Agent 会调用 `analyze_stats` 执行相关性分析。

```
You: 绘制相关性热力图
```

Agent 会调用 `create_chart` 生成热力图并保存为 PNG 文件。

```
You: 用随机森林预测房价
```

Agent 会调用 `train_model` 训练模型，返回评估指标。

```
You: 用训练好的模型预测新数据
```

Agent 会调用 `predict` 工具执行预测。

### 单次执行模式

适合脚本化运行或快速查询：

```bash
python main.py -f data.csv -q "分析房价影响因素，绘制相关性热力图，用随机森林预测房价"
```

### 生成示例数据

```bash
python main.py --sample
```

生成 `output/sample_housing.csv`，包含 200 条模拟房价数据：
- 面积、卧室数、楼龄、楼层、区域、是否学区、是否有地铁、房价
- 含少量缺失值，适合演示数据清洗和分析流程

### 内置命令

在交互式模式下可使用以下命令：

| 命令 | 说明 |
|------|------|
| `help` | 显示帮助信息 |
| `status` | 查看当前状态（已加载数据集、模型等） |
| `sample` | 生成示例数据 |
| `reset` | 重置对话历史 |
| `quit` / `exit` | 退出程序 |

### 命令行参数

| 参数 | 简写 | 说明 |
|------|------|------|
| `--file` | `-f` | 预加载的数据文件路径 |
| `--query` | `-q` | 单次查询（非交互模式） |
| `--model` | `-m` | 指定 Claude 模型名称 |
| `--output` | `-o` | 输出目录 |
| `--sample` | | 生成示例数据 |
| `--api-key` | | API Key（覆盖 .env 配置） |

---

## 工具详细说明

### 数据加载 (load_data)

加载文件并将数据注册到内存工作区，后续工具通过 `data_id` 引用数据。

**支持的格式**：CSV、Excel (.xlsx/.xls)、JSON、TSV

**参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file_path` | string | ✅ | 数据文件路径 |
| `encoding` | string | | 文件编码，默认自动检测 |
| `sheet_name` | string/int | | Excel 工作表名或索引 |
| `clean_options` | object | | 数据清洗选项 |

**清洗选项 (clean_options)**：

| 选项 | 类型 | 说明 |
|------|------|------|
| `drop_na_rows` | boolean | 删除含缺失值的行 |
| `drop_na_cols` | boolean | 删除含缺失值的列 |
| `fill_na` | string/number | 填充缺失值：指定值、"mean"、"median"、"mode" |
| `remove_duplicates` | boolean | 去重 |
| `remove_outliers` | boolean/array | IQR 移除异常值：true=所有数值列，或指定列名列表 |

**自然语言示例**：
```
加载数据 data.csv 并删除缺失值
加载 data.xlsx 的第一个工作表，用均值填充缺失值
```

---

### 统计分析 (analyze_stats)

对已加载的数据执行统计分析。

**分析类型**：

| 类型 | 说明 | 关键参数 |
|------|------|----------|
| `descriptive` | 描述性统计 | — |
| `correlation` | 相关性分析 | `method`: pearson / spearman |
| `distribution_test` | 正态分布检验 | `test`: shapiro / ks |
| `group_stats` | 分组统计 | `group_by`: 分组列名, `agg_funcs`: 聚合函数列表 |
| `hypothesis_test` | 假设检验 | `test`: ttest / anova / chi2 |
| `frequency` | 频率分布 | — |
| `outlier_detection` | 异常值检测 | `method`: iqr / zscore, `threshold`: 阈值 |

**自然语言示例**：
```
分析数据的基本统计信息
分析面积和房价的相关性，用 Spearman 方法
检验房价是否服从正态分布
按区域分组统计房价的均值和标准差
对比不同区域的房价是否有显著差异（方差分析）
检测哪些列有异常值
```

---

### 数据可视化 (create_chart)

生成静态图表并保存到 `output/charts/` 目录。

**图表类型**：

| 类型 | 说明 | 必要参数 |
|------|------|----------|
| `line` | 折线图 | x, y |
| `bar` | 柱状图 | x, y |
| `scatter` | 散点图 | x, y |
| `histogram` | 直方图 | x |
| `box` | 箱线图 | y（可选 x 分组） |
| `violin` | 小提琴图 | y（可选 x 分组） |
| `heatmap` | 热力图 | —（自动使用数值列） |
| `pairplot` | 成对关系图 | — |
| `pie` | 饼图 | x（计数）或 x+y（值） |
| `count` | 计数图 | x 或 y |
| `area` | 面积图 | x, y |

**通用参数**：

| 参数 | 说明 |
|------|------|
| `hue` | 颜色分组列 |
| `title` | 图表标题 |
| `xlabel` / `ylabel` | 轴标签 |
| `filename` | 输出文件名（不含扩展名） |
| `save_format` | png 或 svg |

**自然语言示例**：
```
绘制房价分布直方图
绘制面积与房价的散点图
绘制各区域房价的箱线图
绘制相关性热力图
绘制区域占比饼图，只显示前5个
绘制房价随楼龄变化的折线图
```

---

### 机器学习训练 (train_model)

训练 scikit-learn 机器学习模型，自动处理特征工程。

**任务类型**：

| 类型 | 说明 | 必要参数 |
|------|------|----------|
| `regression` | 回归 | `target_column` |
| `classification` | 分类 | `target_column` |
| `clustering` | 聚类 | — |

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `algorithm` | string | 算法名称，省略则自动选择 |
| `feature_columns` | array | 特征列，省略则自动选择 |
| `tune_hyperparams` | boolean | 是否超参数搜索，默认 false |
| `cv_folds` | int | 交叉验证折数，默认 5 |
| `test_size` | float | 测试集比例，默认 0.2 |
| `random_state` | int | 随机种子，默认 42 |

**自动特征工程**：
- 数值列：中位数填充缺失值 + 标准化
- 分类列：众数填充缺失值 + One-Hot 编码
- 自动组成 sklearn Pipeline

**评估指标**：
- 回归：R²、RMSE、MAE、交叉验证 R²
- 分类：Accuracy、F1 (weighted)、Precision、Recall、AUC (二分类)、混淆矩阵、分类报告
- 聚类：轮廓系数、Calinski-Harabasz 指数

**自然语言示例**：
```
用随机森林预测房价
用逻辑回归分类是否为学区房
对数据进行 K-Means 聚类
用梯度提升预测房价，并搜索最优超参数
```

---

### 机器学习预测 (predict)

使用已训练的模型对新数据进行预测。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `model_path` | string | 模型文件路径（训练时返回） |
| `data_id` | string | 已加载数据的 ID（批量预测） |
| `input_data` | object | 单条数据的字典（单次预测） |
| `return_proba` | boolean | 是否返回分类概率，默认 false |

**自然语言示例**：
```
用训练好的模型对数据集进行预测
预测这套房子的价格：面积100，卧室3间，楼龄5年，楼层10，学区房，有地铁
```

---

## 支持的数据格式

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| CSV | `.csv` | 自动检测编码和分隔符，支持 UTF-8 BOM |
| Excel | `.xlsx`, `.xls` | 需安装 openpyxl，可指定工作表 |
| JSON | `.json` | 标准格式 |
| TSV | `.tsv` | 制表符分隔 |

---

## 支持的图表类型

| 图表 | chart_type | 适用场景 |
|------|------------|----------|
| 折线图 | `line` | 时间序列、趋势变化 |
| 柱状图 | `bar` | 分类比较 |
| 散点图 | `scatter` | 两个变量关系 |
| 直方图 | `histogram` | 单变量分布 |
| 箱线图 | `box` | 分组分布与离群值 |
| 小提琴图 | `violin` | 分布形态比较 |
| 热力图 | `heatmap` | 相关性矩阵 |
| 成对关系图 | `pairplot` | 多变量两两关系 |
| 饼图 | `pie` | 占比分布 |
| 计数图 | `count` | 分类计数 |
| 面积图 | `area` | 累积趋势 |

---

## 支持的机器学习算法

### 回归 (regression)

| 算法 | algorithm 名称 | 说明 |
|------|----------------|------|
| 线性回归 | `linear` | 基础线性模型 |
| 岭回归 | `ridge` | L2 正则化 |
| Lasso 回归 | `lasso` | L1 正则化，特征选择 |
| 弹性网络 | `elasticnet` | L1 + L2 正则化 |
| 随机森林 | `random_forest` | 集成方法，默认选择 |
| 梯度提升 | `gradient_boosting` | 集成方法，高精度 |
| 支持向量回归 | `svr` | 非线性回归 |
| K 近邻回归 | `knn` | 距离加权 |
| 决策树回归 | `decision_tree` | 可解释性强 |

### 分类 (classification)

| 算法 | algorithm 名称 | 说明 |
|------|----------------|------|
| 逻辑回归 | `logistic` | 线性分类器 |
| 随机森林 | `random_forest` | 集成方法，默认选择 |
| 梯度提升 | `gradient_boosting` | 集成方法，高精度 |
| 支持向量机 | `svc` | 非线性分类 |
| K 近邻 | `knn` | 距离分类 |
| 决策树 | `decision_tree` | 可解释性强 |
| 朴素贝叶斯 | `naive_bayes` | 文本/高维数据 |

### 聚类 (clustering)

| 算法 | algorithm 名称 | 说明 |
|------|----------------|------|
| K-Means | `kmeans` | 基于距离，默认选择 |
| DBSCAN | `dbscan` | 密度聚类，发现任意形状 |
| 层次聚类 | `agglomerative` | 层次凝聚 |

---

## 配置说明

所有配置通过 `.env` 文件或环境变量管理。复制 `.env.example` 为 `.env` 进行修改：

```ini
# === Anthropic API ===
ANTHROPIC_API_KEY=sk-ant-...           # 必填：你的 API Key
ANTHROPIC_BASE_URL=                    # 可选：自定义 API 地址（代理）

# === 模型 ===
MODEL_NAME=claude-sonnet-4-6           # Claude 模型名称
MAX_TOKENS=4096                        # 最大输出 token 数

# === Agent ===
MAX_TURNS=20                           # Agent 最大循环轮次

# === 输出 ===
OUTPUT_DIR=output                      # 输出根目录
CHART_FORMAT=png                       # 图表格式：png / svg
CHART_DPI=150                          # 图表 DPI

# === 中文字体 (Windows) ===
CHINESE_FONT=Microsoft YaHei           # 中文字体名称
```

---

## 项目结构

```
untitled/
├── main.py                          # CLI 入口：交互式/单次执行模式
├── requirements.txt                 # Python 依赖
├── .env.example                     # 环境变量模板
├── .gitignore                       # Git 忽略规则
├── README.md                        # 项目说明
├── PROJECT_DOC.md                   # 本文档
│
├── data_analyst/                    # 主包
│   ├── __init__.py                  # 包初始化
│   ├── agent.py                     # Agent 核心：Claude API agentic loop
│   ├── config.py                    # 配置管理 + .env 加载
│   ├── utils.py                     # 辅助函数 + 示例数据生成器
│   │
│   └── tools/                       # 工具集
│       ├── __init__.py              # 工具注册中心
│       ├── data_loader.py           # 数据加载 + Data Registry
│       ├── stats.py                 # 统计分析（7 种）
│       ├── visualize.py             # 数据可视化（11 种图表）
│       ├── ml_train.py              # ML 模型训练（3 大任务）
│       └── ml_predict.py            # ML 模型预测
│
└── output/                          # 输出目录（自动创建）
    ├── charts/                      # 生成的图表文件
    ├── models/                      # 保存的模型文件 (.pkl)
    └── reports/                     # 分析报告
```

---

## 工作原理

### Agentic Tool-Use Loop

```
┌──────────────────────────────────────────────────┐
│  用户输入自然语言                                    │
│  "加载 data.csv，分析房价趋势，用随机森林预测"       │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────┐
│  Claude LLM                                       │
│  理解用户意图 → 选择 Tool → 生成 Tool 调用参数      │
│                                                   │
│  Tool 1: load_data(file_path="data.csv")          │
│  Tool 2: analyze_stats(analysis_type="group_      │
│           stats", group_by="区域", columns=...)    │
│  Tool 3: create_chart(chart_type="line", ...)     │
│  Tool 4: train_model(task_type="regression", ...) │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────┐
│  Tool 执行层                                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │data_loader│ │  stats   │ │ visualize│          │
│  └──────────┘ └──────────┘ └──────────┘          │
│  ┌──────────┐ ┌──────────┐                        │
│  │ml_train  │ │ml_predict│                        │
│  └──────────┘ └──────────┘                        │
│                                                   │
│  Data Registry (内存)                              │
│  ┌──────────────────────────────────┐             │
│  │ data_1 → DataFrame (200×8)      │             │
│  │ data_2 → DataFrame (50×5)       │             │
│  └──────────────────────────────────┘             │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────┐
│  结果返回                                          │
│  ┌─ output/charts/chart_line_data_1.png           │
│  ├─ output/models/random_forest_regression_...pkl │
│  └─ 文字解读（由 Claude 生成）                      │
└──────────────────────────────────────────────────┘
```

### 数据传递机制

1. `load_data` 加载文件后，DataFrame 存入内存中的 **Data Registry**，返回 `data_id`
2. 后续工具通过 `data_id` 引用数据，无需传递原始数据
3. LLM 上下文中只传数据摘要（shape、columns、dtypes、preview），不传原始数据
4. 多个数据集可同时加载，各自拥有独立的 `data_id`

### 安全设计

- 采用 **Tool-Based** 架构，LLM 不直接生成和执行 Python 代码
- 所有操作通过预定义的 Tool 函数执行，每个 Tool 内部有参数验证
- LLM 只能调用注册的工具，无法执行任意代码

---

## 常见问题

### Q: 启动时提示 "ANTHROPIC_API_KEY is not set"

确保项目根目录下有 `.env` 文件，且包含有效的 API Key：

```ini
ANTHROPIC_API_KEY=sk-ant-你的密钥
```

### Q: 图表中文显示为方框

确认 `.env` 中 `CHINESE_FONT` 设置为系统已安装的中文字体。Windows 默认配置为 `Microsoft YaHei`（微软雅黑），也可改为 `SimHei`（黑体）。

### Q: 加载 CSV 时编码错误

对于中文 CSV 文件，可手动指定编码：

```
加载数据 data.csv，编码为 gbk
```

### Q: 模型训练时间过长

- 减小数据量：加载时使用 `clean_options` 过滤数据
- 关闭超参数搜索：不使用 `tune_hyperparams`
- 选择更简单的算法：如 `linear` 或 `decision_tree`

### Q: 如何使用已保存的模型

训练模型后会返回 `model_path`，在后续对话中可直接引用：

```
用 output/models/random_forest_regression_data_1.pkl 这个模型预测新数据
```

### Q: 如何在命令行中指定 API Key

```bash
python main.py --api-key sk-ant-你的密钥
```

### Q: 支持自定义 API 地址吗

支持。在 `.env` 中设置 `ANTHROPIC_BASE_URL`：

```ini
ANTHROPIC_BASE_URL=https://your-proxy.example.com
```

### Q: 输出文件保存在哪里

默认保存在项目根目录的 `output/` 下：
- 图表：`output/charts/`
- 模型：`output/models/`

可通过 `--output` 参数或 `.env` 中的 `OUTPUT_DIR` 修改。

---

## 示例会话

```
$ python main.py

Data Analysis Agent
数据分析智能体 - 自然语言驱动的数据分析、可视化与机器学习

You: 加载 output/sample_housing.csv

已加载数据 data_1，共 200 行 8 列：
  面积, 卧室数, 楼龄, 楼层, 区域, 是否学区, 是否有地铁, 房价
  缺失值: 面积(7), 卧室数(1), 楼龄(5), ...

You: 删除缺失值，然后分析各列与房价的相关性

清洗后数据: 170 行。相关性分析结果（Pearson）：
  - 面积 ↔ 房价: 0.72 (强正相关)
  - 是否学区 ↔ 房价: 0.58 (中等正相关)
  - 楼龄 ↔ 房价: -0.35 (弱负相关)

You: 绘制相关性热力图

已生成热力图: output/charts/chart_heatmap_data_1.png

You: 用随机森林预测房价

随机森林回归模型训练完成：
  - R²: 0.63
  - RMSE: 53,531
  - 交叉验证 R²: 0.73
  模型已保存: output/models/random_forest_regression_data_1.pkl

You: quit

Goodbye! 
```

---

## 许可证

MIT License
