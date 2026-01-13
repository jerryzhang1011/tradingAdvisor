from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from emailAgent.tradingAgentEmail import send_trade_signal_email
from dotenv import load_dotenv
import datetime
import time
import chromadb
from chromadb.config import Settings

# Load environment variables from .env file
load_dotenv()


# Create a custom config
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-5-nano"  # Use a different model
config["quick_think_llm"] = "gpt-4o-mini"  # Use a different model
config["max_debate_rounds"] = 3  # Increase debate rounds

# Configure data vendors (default uses yfinance and alpha_vantage)
config["data_vendors"] = {
    "core_stock_apis": "yfinance",           # Options: yfinance, alpha_vantage, local
    "technical_indicators": "yfinance",      # Options: yfinance, alpha_vantage, local
    "fundamental_data": "alpha_vantage",     # Options: openai, alpha_vantage, local
    "news_data": "alpha_vantage",            # Options: openai, alpha_vantage, google, local
}


start_time = time.perf_counter()

tickers = ["AAPL"]
# tickers = []

results = []
signals_to_email = []

# Reuse a single Chroma client so we can reset the vector store between iterations
chroma_reset_client = chromadb.Client(Settings(allow_reset=True))
for ticker in tickers:
    # Initialize with custom config
    ta = TradingAgentsGraph(debug=True, config=config)

    # forward propagate
    _, decision = ta.propagate(ticker, datetime.date.today().strftime("%Y-%m-%d"))
    results.append((ticker, decision))

    log_line = f"{datetime.datetime.now().isoformat():<30} {ticker:<10} {decision:<10}\n"
    with open("/Users/jerryzhang/Desktop/tradingAgentsLogs.txt", "a") as f:
        f.write(log_line)

    if decision in {"BUY", "SELL"}:
        signals_to_email.append((ticker, decision))

    # Reset ChromaDB so each ticker runs against a clean memory state
    chroma_reset_client.reset()
    

if signals_to_email:
    send_trade_signal_email(signals_to_email)

print("\n".join(f"{result[0]}: {result[1]}" for result in results))

total_execution_time = time.perf_counter() - start_time
minutes, seconds = divmod(total_execution_time, 60)
print(f"Total execution time: {int(minutes)} min {seconds:.2f} sec")



# from myPosition.myOrders import myOrders

# unique_symbols = set()
# for order in myOrders:
#     unique_symbols.add(order.ticker)

# for symbol in unique_symbols:
#     print(symbol)


# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
