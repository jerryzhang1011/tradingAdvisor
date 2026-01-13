# Trading Advisor

```
  ______               ___             ___       __      _                
 /_  __/________ _____/ (_)___  ____ _/   | ____/ /   __(_)________  _____
  / / / ___/ __ `/ __  / / __ \/ __ `/ /| |/ __  / | / / / ___/ __ \/ ___/
 / / / /  / /_/ / /_/ / / / / / /_/ / ___ / /_/ /| |/ / (__  ) /_/ / /    
/_/ /_/   \__,_/\__,_/_/_/ /_/\__, /_/  |_\__,_/ |___/_/____/\____/_/     
                             /____/                                       
```

**AI-Powered Multi-Agent Trading Advisory System**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

Trading Advisor simulates a trading firm's workflow with multiple specialized agents:
- Analysts: technical / fundamentals / news / sentiment
- Researchers: bull vs bear debate + synthesis
- Trader: converts research to an actionable plan
- Risk management: multi-perspective review
- Memory/reflection: learns from past outcomes (ChromaDB)

Supported LLM backends include OpenAI / Anthropic / Google Gemini / Ollama / OpenRouter (depending on your config).

## Agents workflow (high level)

```
Market data → Analyst reports → Bull/Bear debate → Trading plan → Risk debate → Final decision (BUY/SELL/HOLD)
```

- Analysts pull data (e.g. price/indicators, fundamentals, news, sentiment) and write reports.
- Researchers run a structured bull vs bear debate, then a manager synthesizes a view.
- Trader turns the view into a concrete action/plan.
- Risk team debates the plan (aggressive / conservative / neutral) and a risk manager makes the final call.

## Quick start

### Install

With `uv` (recommended):

```bash
git clone https://github.com/jerryzhang1011/tradingAdvisor.git
cd tradingAdvisor
uv sync
```

With `pip`:

```bash
git clone https://github.com/jerryzhang1011/tradingAdvisor.git
cd tradingAdvisor
pip install -e .
```

### Environment variables

Create a `.env` file in the project root (see `.env.example` if present):

```bash
# LLM API Keys (choose one or more based on your provider)
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key

# Data Source API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# Email Notifications (optional)
GMAIL_USER=your_gmail_address
GMAIL_APP_PW=your_gmail_app_password
```

## Usage

### CLI

```bash
python -m cli.main
# or
python -u main.py
```

### Python API (minimal)

```python
import datetime
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"

ta = TradingAgentsGraph(
    selected_analysts=["market", "social", "news", "fundamentals"],
    debug=True,
    config=config,
)

_, decision = ta.propagate(
    company_name="AAPL",
    trade_date=datetime.date.today().strftime("%Y-%m-%d"),
)

print(decision)  # BUY / SELL / HOLD
```

### Output

Results are typically written to:
- `./results/{TICKER}/{DATE}/reports/` (markdown reports)
- `./eval_results/{TICKER}/strategy_logs/` (full state logs)

## Disclaimer

This project is for educational and research purposes only and does not constitute financial advice. Trading involves risk; consult a qualified professional before making investment decisions. The authors are not responsible for financial losses incurred through use of this software.
