"""Data Analysis Agent - CLI Entry Point.

Provides both interactive and single-shot execution modes.
"""

import argparse
import sys
import os
from pathlib import Path

# Fix Windows console encoding for Chinese characters
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table

from data_analyst.config import validate_config, get_config_summary, ensure_output_dirs
from data_analyst.agent import DataAnalystAgent
from data_analyst.tools.data_loader import list_loaded_data
from data_analyst.utils import generate_sample_data

console = Console(force_terminal=True)


def print_banner():
    """Print the application banner."""
    banner = Panel(
        "[bold cyan]Data Analysis Agent[/bold cyan]\n"
        "[dim]数据分析智能体 - 自然语言驱动的数据分析、可视化与机器学习[/dim]\n\n"
        "[dim]输入自然语言指令进行分析，输入 'help' 查看帮助，'quit' 退出[/dim]",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(banner)


def print_help():
    """Print help information."""
    help_text = """
## 使用说明

### 数据加载
- `加载 data.csv` - 加载CSV文件
- `加载 data.xlsx` - 加载Excel文件
- `加载 data.json` - 加载JSON文件

### 统计分析
- `分析数据的基本统计信息` - 描述性统计
- `分析各列的相关性` - 相关性分析
- `检验某列是否正态分布` - 分布检验
- `按区域分组统计房价` - 分组统计
- `检测异常值` - 异常值检测

### 可视化
- `绘制房价分布直方图` - 直方图
- `绘制面积与房价的散点图` - 散点图
- `绘制相关性热力图` - 热力图
- `绘制各区域房价箱线图` - 箱线图
- `绘制区域占比饼图` - 饼图

### 机器学习
- `用随机森林预测房价` - 回归预测
- `用逻辑回归分类是否学区房` - 分类
- `对数据进行聚类分析` - 聚类
- `用训练好的模型预测新数据` - 预测

### 其他命令
- `help` - 显示帮助
- `status` - 查看当前状态
- `sample` - 生成示例数据
- `reset` - 重置对话
- `quit` / `exit` - 退出程序
"""
    console.print(Markdown(help_text))


def print_status(agent: DataAnalystAgent):
    """Print current agent status."""
    summary = agent.get_conversation_summary()
    config = get_config_summary()

    table = Table(title="📊 Agent Status", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Model", config["model"])
    table.add_row("API Key", "✅ Set" if config["api_key_set"] else "❌ Not Set")
    table.add_row("Messages", str(summary["message_count"]))
    table.add_row("Output Dir", config["output_dir"])
    table.add_row("Chart Format", config["chart_format"])

    if summary["loaded_datasets"]:
        for ds in summary["loaded_datasets"]:
            table.add_row(
                f"Dataset: {ds['data_id']}",
                f"{ds['shape'][0]} rows × {ds['shape'][1]} cols"
            )
    else:
        table.add_row("Datasets", "None loaded")

    console.print(table)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Data Analysis Agent - 数据分析智能体",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Interactive mode
  python main.py -f data.csv                        # Load file and start interactive
  python main.py -f data.csv -q "分析数据趋势"       # Single-shot mode
  python main.py --sample                           # Generate sample data
        """
    )
    parser.add_argument("-f", "--file", help="Data file to preload")
    parser.add_argument("-q", "--query", help="Single query (non-interactive mode)")
    parser.add_argument("-m", "--model", help="Claude model name")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--sample", action="store_true", help="Generate sample data for testing")
    parser.add_argument("--api-key", help="Anthropic API key (overrides .env)")

    args = parser.parse_args()

    # Generate sample data if requested
    if args.sample:
        sample_path = str(Path("output/sample_housing.csv"))
        Path("output").mkdir(exist_ok=True)
        generate_sample_data(sample_path)
        console.print(f"[green]✅ Sample data generated: {sample_path}[/green]")
        return

    # Validate configuration
    issues = validate_config()
    api_key = args.api_key
    if not api_key and issues:
        for issue in issues:
            console.print(f"[red]❌ {issue}[/red]")
        console.print("\n[yellow]提示: 请创建 .env 文件并设置 ANTHROPIC_API_KEY[/yellow]")
        console.print("[yellow]参考 .env.example 文件[/yellow]")
        sys.exit(1)

    # Ensure output directories
    ensure_output_dirs()

    # Initialize agent
    try:
        agent = DataAnalystAgent(api_key=api_key, model=args.model)
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize agent: {e}[/red]")
        sys.exit(1)

    # Preload file if specified
    if args.file:
        console.print(f"[cyan]📂 Loading file: {args.file}[/cyan]")
        result = agent.run(f"加载文件 {args.file}")
        console.print(Markdown(result))

    # Single-shot mode
    if args.query:
        console.print()
        result = agent.run(args.query)
        console.print(Markdown(result))
        return

    # Interactive mode
    print_banner()

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye! [/dim]")
            break

        if not user_input:
            continue

        # Handle special commands
        cmd = user_input.lower().strip()

        if cmd in ("quit", "exit", "q"):
            console.print("[dim]Goodbye! [/dim]")
            break
        elif cmd == "help":
            print_help()
            continue
        elif cmd == "status":
            print_status(agent)
            continue
        elif cmd == "reset":
            agent.reset()
            console.print("[green]✅ Conversation reset.[/green]")
            continue
        elif cmd == "sample":
            sample_path = str(Path("output/sample_housing.csv"))
            generate_sample_data(sample_path)
            console.print(f"[green]✅ Sample data generated: {sample_path}[/green]")
            continue

        # Run agent
        try:
            with console.status("[bold cyan]🤔 Thinking...", spinner="dots"):
                result = agent.run(user_input)
            console.print(Markdown(result))
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️ Interrupted[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")


if __name__ == "__main__":
    main()
