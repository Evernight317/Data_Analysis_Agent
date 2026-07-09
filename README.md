# 📊 Data Analysis Agent

数据分析智能体 — 基于自然语言的数据分析、可视化与机器学习预测工具。

## ✨ 功能特性

- **数据加载**: 支持 CSV、Excel、JSON、TSV 格式，自动检测编码
- **统计分析**: 描述性统计、相关性分析、分布检验、假设检验、分组统计、异常值检测
- **数据可视化**: 折线图、柱状图、散点图、直方图、箱线图、小提琴图、热力图、饼图等（PNG/SVG）
- **机器学习**: 回归、分类、聚类（scikit-learn），支持超参数搜索和交叉验证
- **自然语言交互**: 用中文或英文描述需求，Agent 自动选择并执行分析流程

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 Anthropic API Key：

```bash
cp .env.example .env
# 编辑 .env，设置 ANTHROPIC_API_KEY=sk-ant-...
```

### 3. 生成示例数据（可选）

```bash
python main.py --sample
```

### 4. 运行

**交互式模式**:
```bash
python main.py
```

**预加载数据文件**:
```bash
python main.py -f data.csv
```

**单次查询模式**:
```bash
python main.py -f data.csv -q "分析房价影响因素，绘制相关性热力图，用随机森林预测房价"
```

## 📖 使用示例

```
📊 Data Analysis Agent
> 加载 output/sample_housing.csv
> 分析数据的基本统计信息
> 绘制房价分布直方图
> 分析各列与房价的相关性
> 绘制相关性热力图
> 用随机森林预测房价
> 用训练好的模型预测新数据
```

## 🛠️ 命令行参数

| 参数 | 说明 |
|------|------|
| `-f, --file` | 预加载的数据文件路径 |
| `-q, --query` | 单次查询（非交互模式） |
| `-m, --model` | 指定 Claude 模型 |
| `-o, --output` | 输出目录 |
| `--sample` | 生成示例数据 |
| `--api-key` | API Key（覆盖 .env 配置） |

## 🏗️ 项目结构

```
data_analyst/
├── agent.py              # Agent 核心: Claude API agentic loop
├── config.py             # 配置管理
├── utils.py              # 辅助函数
└── tools/
    ├── data_loader.py    # 数据加载工具
    ├── stats.py          # 统计分析工具
    ├── visualize.py      # 可视化工具
    ├── ml_train.py       # ML 训练工具
    └── ml_predict.py     # ML 预测工具
```

## 🔧 支持的分析类型

### 统计分析
- `descriptive` - 描述性统计
- `correlation` - 相关性分析（Pearson/Spearman）
- `distribution_test` - 正态分布检验（Shapiro-Wilk/K-S）
- `group_stats` - 分组统计
- `hypothesis_test` - 假设检验（t-test/ANOVA/chi-square）
- `frequency` - 频率分布
- `outlier_detection` - 异常值检测（IQR/Z-score）

### 可视化
- `line` - 折线图
- `bar` - 柱状图
- `scatter` - 散点图
- `histogram` - 直方图
- `box` - 箱线图
- `violin` - 小提琴图
- `heatmap` - 热力图
- `pairplot` - 成对关系图
- `pie` - 饼图
- `count` - 计数图
- `area` - 面积图

### 机器学习
- **回归**: linear, ridge, lasso, elasticnet, random_forest, gradient_boosting, svr, knn, decision_tree
- **分类**: logistic, random_forest, gradient_boosting, svc, knn, decision_tree, naive_bayes
- **聚类**: kmeans, dbscan, agglomerative

## ⚙️ 配置

所有配置通过 `.env` 文件或环境变量管理：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ANTHROPIC_API_KEY` | - | Anthropic API Key（必填） |
| `ANTHROPIC_BASE_URL` | - | API 基础 URL（可选，用于代理） |
| `MODEL_NAME` | claude-sonnet-4-6 | Claude 模型名称 |
| `MAX_TOKENS` | 4096 | 最大输出 token 数 |
| `MAX_TURNS` | 20 | Agent 最大循环轮次 |
| `OUTPUT_DIR` | output | 输出目录 |
| `CHART_FORMAT` | png | 图表格式（png/svg） |
| `CHART_DPI` | 150 | 图表 DPI |
| `CHINESE_FONT` | Microsoft YaHei | 中文字体 |

## 📄 License

MIT
