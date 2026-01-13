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

> GitHub: [jerryzhang1011/tradingAdvisor](https://github.com/jerryzhang1011/tradingAdvisor)

---

## 📖 Overview

Trading Advisor is an advanced AI-powered financial trading advisory system that leverages multiple specialized LLM agents working collaboratively to analyze markets and provide informed trading recommendations. The system simulates a professional trading firm's decision-making process with distinct teams of analysts, researchers, traders, and risk managers.

### Key Features

- 🤖 **Multi-Agent Architecture**: Specialized AI agents for different aspects of financial analysis
- 📊 **Comprehensive Analysis**: Technical, fundamental, sentiment, and news-based analysis
- 🔄 **Debate-Driven Decisions**: Bull vs Bear researcher debates for balanced perspectives
- ⚖️ **Risk Management**: Multi-perspective risk evaluation before final decisions
- 🧠 **Memory & Reflection**: Agents learn from past decisions to improve future performance
- 📧 **Email Notifications**: Automated trade signal alerts via Gmail
- 💻 **Rich CLI Interface**: Beautiful terminal UI with real-time progress tracking
- 🔌 **Multiple Data Sources**: Support for yfinance, Alpha Vantage, and more
- 🧩 **Flexible LLM Support**: OpenAI, Anthropic, Google Gemini, Ollama, and OpenRouter

---

## 🏗️ Architecture

### Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Trading Advisor Workflow                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  I. Analyst Team  →  II. Research Team  →  III. Trader  →  IV. Risk Mgmt   │
│                                                        →  V. Portfolio Mgmt │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agent Teams

#### 📈 I. Analyst Team
Gathers and analyzes raw market data from multiple perspectives:

| Agent | Role | Data Sources |
|-------|------|--------------|
| **Market Analyst** | Technical analysis using indicators (SMA, EMA, MACD, RSI, Bollinger Bands, ATR, VWMA) | yfinance, Alpha Vantage |
| **Social Media Analyst** | Social sentiment analysis | News APIs, Local data |
| **News Analyst** | Current events and insider activity analysis | Alpha Vantage, Google News |
| **Fundamentals Analyst** | Company financial health analysis (balance sheet, cash flow, income statements) | Alpha Vantage, OpenAI |

#### 🔬 II. Research Team
Debates investment merit through structured argumentation:

| Agent | Role |
|-------|------|
| **Bull Researcher** | Advocates for investing - highlights growth potential, competitive advantages |
| **Bear Researcher** | Advocates against investing - identifies risks, weaknesses, threats |
| **Research Manager** | Synthesizes debates and makes preliminary investment recommendations |

#### 💹 III. Trading Team
Formulates actionable trading plans:

| Agent | Role |
|-------|------|
| **Trader** | Creates specific trading recommendations based on research team's analysis |

#### ⚠️ IV. Risk Management Team
Evaluates risk from multiple perspectives through debate:

| Agent | Role |
|-------|------|
| **Aggressive (Risky) Analyst** | Advocates for higher-risk, higher-reward positions |
| **Conservative (Safe) Analyst** | Advocates for capital preservation and risk mitigation |
| **Neutral Analyst** | Provides balanced perspective weighing both sides |

#### 📋 V. Portfolio Management
Makes final trading decisions:

| Agent | Role |
|-------|------|
| **Risk Manager (Portfolio Manager)** | Final decision maker - synthesizes all inputs into BUY/SELL/HOLD recommendation |

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- API keys for your chosen LLM provider (OpenAI, Anthropic, or Google)
- Optional: Alpha Vantage API key for enhanced market data

### Install with pip

```bash
# Clone the repository
git clone https://github.com/jerryzhang1011/tradingAdvisor.git
cd tradingAdvisor

# Install dependencies
pip install -e .
```

### Install with uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/jerryzhang1011/tradingAdvisor.git
cd tradingAdvisor

# Install with uv
uv sync
```

### Environment Variables

Create a `.env` file in the project root (See an example in: .env.example):

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

---

## 💻 Usage

### CLI Interface

The easiest way to use Trading Advisor is through the interactive CLI:

```bash
python -m cli.main
or
python -u main.py
```

The CLI will guide you through:
1. **Ticker Symbol**: Enter the stock to analyze (e.g., AAPL, TSLA, SPY)
2. **Analysis Date**: Select the date for analysis
3. **Analyst Selection**: Choose which analysts to include
4. **Research Depth**: Configure debate rounds (1-5)
5. **LLM Provider**: Select OpenAI, Anthropic, Google, Ollama, or OpenRouter
6. **Model Selection**: Choose specific models for deep and quick thinking

### Programmatic Usage

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
import datetime

# Create custom config
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"
config["deep_think_llm"] = "gpt-4o"
config["quick_think_llm"] = "gpt-4o-mini"
config["max_debate_rounds"] = 3

# Configure data vendors
config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "alpha_vantage",
    "news_data": "alpha_vantage",
}

# Initialize Trading Advisor
ta = TradingAgentsGraph(
    selected_analysts=["market", "social", "news", "fundamentals"],
    debug=True,
    config=config
)

# Run analysis
final_state, decision = ta.propagate(
    company_name="AAPL",
    trade_date=datetime.date.today().strftime("%Y-%m-%d")
)

print(f"Trading Decision: {decision}")  # Output: BUY, SELL, or HOLD
```

### Email Notifications

To receive trade signal notifications via email:

```python
from emailAgent.tradingAgentEmail import send_trade_signal_email

# Send notifications for trading signals
signals = [("AAPL", "BUY"), ("TSLA", "SELL")]
send_trade_signal_email(signals)
```

---

## ⚙️ Configuration

### Default Configuration

```python
DEFAULT_CONFIG = {
    # Directories
    "project_dir": "...",
    "results_dir": "./results",
    "data_cache_dir": "...",
    
    # LLM Settings
    "llm_provider": "openai",           # Options: openai, anthropic, google, ollama, openrouter
    "deep_think_llm": "gpt-5-nano",     # Model for complex reasoning
    "quick_think_llm": "gpt-4o-mini",   # Model for quick decisions
    "deep_think_temperature": 0.35,
    "quick_think_temperature": 0.10,
    "backend_url": "https://api.openai.com/v1",
    
    # Debate Settings
    "max_debate_rounds": 3,             # Bull vs Bear debate rounds
    "max_risk_discuss_rounds": 3,       # Risk management debate rounds
    "max_recur_limit": 150,
    
    # Data Vendors
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: yfinance, alpha_vantage, local
        "technical_indicators": "yfinance",  # Options: yfinance, alpha_vantage, local
        "fundamental_data": "alpha_vantage", # Options: openai, alpha_vantage, local
        "news_data": "alpha_vantage",        # Options: openai, alpha_vantage, google, local
    },
}
```

### Supported LLM Providers

| Provider | Models | Configuration |
|----------|--------|---------------|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-5-nano, etc. | `llm_provider: "openai"` |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Opus, etc. | `llm_provider: "anthropic"` |
| **Google** | Gemini Pro, Gemini Ultra, etc. | `llm_provider: "google"` |
| **Ollama** | Any local model | `llm_provider: "ollama"` |
| **OpenRouter** | Various models | `llm_provider: "openrouter"` |

### Technical Indicators Available

| Category | Indicators |
|----------|------------|
| **Moving Averages** | 50 SMA, 200 SMA, 10 EMA |
| **MACD** | MACD, MACD Signal, MACD Histogram |
| **Momentum** | RSI |
| **Volatility** | Bollinger Bands (Upper, Middle, Lower), ATR |
| **Volume** | VWMA |

---

## 📁 Project Structure

```
tradingAdvisor/
├── cli/                           # CLI application
│   ├── main.py                    # CLI entry point with Rich UI
│   ├── models.py                  # CLI data models
│   ├── utils.py                   # CLI utilities
│   └── static/
│       └── welcome.txt            # ASCII art banner
│
├── tradingagents/                 # Core trading agents package
│   ├── agents/                    # All agent implementations
│   │   ├── analysts/              # Analyst agents
│   │   │   ├── market_analyst.py
│   │   │   ├── news_analyst.py
│   │   │   ├── social_media_analyst.py
│   │   │   └── fundamentals_analyst.py
│   │   ├── researchers/           # Research agents
│   │   │   ├── bull_researcher.py
│   │   │   └── bear_researcher.py
│   │   ├── managers/              # Manager agents
│   │   │   ├── research_manager.py
│   │   │   └── risk_manager.py
│   │   ├── trader/                # Trader agent
│   │   │   └── trader.py
│   │   ├── risk_mgmt/             # Risk management debators
│   │   │   ├── aggresive_debator.py
│   │   │   ├── conservative_debator.py
│   │   │   └── neutral_debator.py
│   │   └── utils/                 # Agent utilities
│   │       ├── agent_states.py
│   │       ├── agent_utils.py
│   │       ├── memory.py
│   │       └── *_tools.py
│   │
│   ├── dataflows/                 # Data fetching and processing
│   │   ├── alpha_vantage*.py      # Alpha Vantage integrations
│   │   ├── y_finance.py           # Yahoo Finance integration
│   │   ├── google.py              # Google News integration
│   │   ├── reddit_utils.py        # Local Reddit data parsing
│   │   └── config.py              # Data configuration
│   │
│   ├── graph/                     # LangGraph workflow
│   │   ├── trading_graph.py       # Main graph orchestrator
│   │   ├── setup.py               # Graph construction
│   │   ├── propagation.py         # State propagation
│   │   ├── reflection.py          # Memory/reflection system
│   │   ├── signal_processing.py   # Signal extraction
│   │   └── conditional_logic.py   # Workflow conditionals
│   │
│   └── default_config.py          # Default configuration
│
├── emailAgent/                    # Email notification system
│   └── tradingAgentEmail.py
│
├── myPosition/                    # Portfolio tracking (optional)
│   ├── myOrders.py
│   ├── myProfolio.py
│   └── order.py
│
├── main.py                        # Main entry point
├── pyproject.toml                 # Project dependencies (uv)
├── requirements.txt               # Dependencies (pip)
├── setup.py                       # Package setup
└── README.md                      # This file
```

---

## 🧠 Memory & Reflection System

Trading Advisor includes a sophisticated memory system that allows agents to learn from past decisions:

```python
# After a trade is executed and returns are known
ta.reflect_and_remember(returns_losses=1000)  # Positive = profit, Negative = loss
```

This updates the memory for:
- Bull Researcher: Learns when bullish arguments succeeded/failed
- Bear Researcher: Learns when bearish arguments succeeded/failed  
- Trader: Learns from trading decision outcomes
- Investment Judge: Improves debate synthesis
- Risk Manager: Refines risk assessment calibration

The memories are stored in ChromaDB vector database and retrieved for similar future situations.

---

## 📊 Output Example

After running an analysis, you'll receive:

1. **Analyst Reports**: Detailed analysis from each analyst
2. **Research Debate**: Bull vs Bear arguments and synthesis
3. **Trading Plan**: Trader's recommended action
4. **Risk Assessment**: Three-way risk debate and final decision
5. **Final Decision**: Clear BUY, SELL, or HOLD recommendation

Results are saved to:
- `./results/{TICKER}/{DATE}/reports/` - Individual report markdown files
- `./eval_results/{TICKER}/strategy_logs/` - Full state JSON logs

---

## 🔧 Dependencies

### Core Dependencies

| Package | Purpose |
|---------|---------|
| `langgraph` | Agent workflow orchestration |
| `langchain-openai` | OpenAI LLM integration |
| `langchain-anthropic` | Anthropic LLM integration |
| `langchain-google-genai` | Google Gemini integration |
| `chromadb` | Vector database for agent memory |
| `yfinance` | Yahoo Finance market data |
| `pandas` | Data manipulation |
| `stockstats` | Technical indicator calculations |

### CLI Dependencies

| Package | Purpose |
|---------|---------|
| `typer` | CLI framework |
| `rich` | Beautiful terminal output |
| `questionary` | Interactive prompts |

### Optional Dependencies

| Package | Purpose |
|---------|---------|
| `feedparser` | RSS feed parsing |
| `finnhub-python` | Alternative market data |
| `akshare` | Chinese market data |
| `tushare` | Chinese market data |

---

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not intended to be used as financial advice. Trading stocks involves risk, and you should consult with a qualified financial advisor before making any investment decisions. The authors are not responsible for any financial losses incurred through the use of this software.
