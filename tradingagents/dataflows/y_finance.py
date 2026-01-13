from typing import Annotated, Optional
from datetime import datetime, timedelta
from pathlib import Path
import os

import pandas as pd
from dateutil.relativedelta import relativedelta
import yfinance as yf

from .config import get_config
from .stockstats_utils import StockstatsUtils

def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):

    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    # Create ticker object
    ticker = yf.Ticker(symbol.upper())

    # Fetch historical data for the specified date range
    data = ticker.history(start=start_date, end=end_date)

    # Check if data is empty
    if data.empty:
        return (
            f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
        )

    # Remove timezone info from index for cleaner output
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # Round numerical values to 2 decimal places for cleaner display
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].round(2)

    # Convert DataFrame to CSV string
    csv_string = data.to_csv()

    # Add header information
    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string

def get_intraday_bars(
    symbol: Annotated[str, "ticker symbol of the company"],
    interval: Optional[str] = None,
    period: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Retrieve intraday OHLCV data for a ticker with caching and configurable interval.

    Args:
        symbol: Stock ticker symbol.
        interval: Bar interval supported by yfinance (e.g., '1m', '5m', '15m').
        period: Lookback period supported by yfinance (e.g., '1d', '5d', '60d').
        force_refresh: Skip cache and fetch fresh data.

    Returns:
        Pandas DataFrame indexed by naive timestamps with OHLCV columns.
    """
    config = get_config()
    resolved_interval = interval or config.get("bar_interval", "5m")
    resolved_period = period or config.get("intraday_lookback", "5d")
    cache_ttl_minutes = config.get("intraday_cache_ttl_minutes", 5)

    cache_dir = Path(config.get("data_cache_dir", "dataflows/data_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol.upper()}-intraday-{resolved_interval}-{resolved_period}.csv"

    now_utc = datetime.utcnow()

    if not force_refresh and cache_file.exists():
        cache_age = now_utc - datetime.utcfromtimestamp(cache_file.stat().st_mtime)
        if cache_age <= timedelta(minutes=cache_ttl_minutes):
            cached_df = pd.read_csv(cache_file, parse_dates=["Datetime"])
            cached_df.set_index("Datetime", inplace=True)
            return cached_df

    data = yf.download(
        symbol,
        interval=resolved_interval,
        period=resolved_period,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        try:
            unique_level = data.columns.get_level_values(-1).unique()
        except (AttributeError, IndexError):
            unique_level = []
        if len(unique_level) == 1:
            data = data.droplevel(-1, axis=1)
        else:
            data.columns = [
                "_".join([str(part) for part in col if part not in ("", None)])
                if isinstance(col, tuple)
                else col
                for col in data.columns
            ]

    if data.empty:
        raise ValueError(
            f"No intraday data returned for {symbol} (interval={resolved_interval}, period={resolved_period})"
        )

    if getattr(data.index, "tz", None) is not None:
        data = data.tz_convert("UTC")
        data.index = data.index.tz_localize(None)

    df = data.reset_index()

    if "Datetime" not in df.columns:
        datetime_col = df.columns[0]
        df.rename(columns={datetime_col: "Datetime"}, inplace=True)

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.to_csv(cache_file, index=False)
    df.set_index("Datetime", inplace=True)

    return df

def build_intraday_snapshot(
    symbol: Annotated[str, "ticker symbol of the company"],
    trade_date: Annotated[str, "current trading date, YYYY-mm-dd"],
    interval: Optional[str] = None,
    period: Optional[str] = None,
) -> dict:
    """
    Create a concise summary of the current trading session using intraday data.

    Returns a dictionary with raw values and a human-readable summary that can be
    injected into agent prompts.
    """
    config = get_config()
    resolved_interval = interval or config.get("bar_interval", "5m")
    resolved_period = period or config.get("intraday_lookback", "5d")

    try:
        df = get_intraday_bars(
            symbol,
            interval=resolved_interval,
            period=resolved_period,
        )
    except Exception as exc:
        return {
            "summary": f"Intraday snapshot unavailable for {symbol.upper()}: {exc}",
            "error": str(exc),
            "session_date": trade_date,
            "interval": resolved_interval,
            "period": resolved_period,
        }

    if df.empty:
        return {
            "summary": f"Intraday snapshot unavailable for {symbol.upper()}: no bars returned.",
            "error": "empty_intraday_frame",
            "session_date": trade_date,
            "interval": resolved_interval,
            "period": resolved_period,
        }

    df = df.sort_index()
    requested_session = datetime.strptime(trade_date, "%Y-%m-%d").date()
    session_df = df[df.index.date == requested_session]

    note = ""
    if session_df.empty:
        latest_session_date = df.index[-1].date()
        session_df = df[df.index.date == latest_session_date]
        note = (
            f"Requested trade date {trade_date} has no intraday bars; "
            f"using most recent session {latest_session_date} instead."
        )
        requested_session = latest_session_date

    if session_df.empty:
        return {
            "summary": f"Intraday snapshot unavailable for {symbol.upper()}: could not locate bars for latest session.",
            "error": "no_session_data",
            "session_date": trade_date,
            "interval": resolved_interval,
            "period": resolved_period,
        }

    latest_bar = session_df.iloc[-1]
    latest_ts = latest_bar.name

    session_open = float(session_df.iloc[0]["Open"])
    session_high = float(session_df["High"].max())
    session_low = float(session_df["Low"].min())
    session_volume = float(session_df["Volume"].sum())
    last_price = float(latest_bar["Close"])
    last_bar_volume = float(latest_bar["Volume"])

    prior_df = df[df.index < session_df.index[0]]
    if not prior_df.empty:
        previous_close = float(prior_df.iloc[-1]["Close"])
    else:
        previous_close = session_open

    price_change = last_price - previous_close
    change_pct = (price_change / previous_close * 100) if previous_close else 0.0

    minutes_active = (
        (latest_ts - session_df.index[0]).total_seconds() / 60.0
        if len(session_df.index) > 1
        else 0.0
    )

    summary_lines = [
        f"{symbol.upper()} intraday snapshot for {requested_session} (interval {resolved_interval}):",
        f"- Last trade {latest_ts.strftime('%H:%M:%S')} UTC @ {last_price:.2f} "
        f"({price_change:+.2f}, {change_pct:+.2f}% vs prior close {previous_close:.2f})",
        f"- Session open {session_open:.2f} • High {session_high:.2f} • Low {session_low:.2f}",
        f"- Volume {session_volume:,.0f} shares ({last_bar_volume:,.0f} on latest bar)",
    ]

    if minutes_active:
        summary_lines.append(f"- Session progressing for approx. {minutes_active:.0f} minutes so far")
    if note:
        summary_lines.append(note)

    summary = "\n".join(summary_lines)

    return {
        "summary": summary,
        "session_date": requested_session.isoformat(),
        "requested_date": trade_date,
        "interval": resolved_interval,
        "period": resolved_period,
        "as_of": latest_ts.isoformat(),
        "last_price": last_price,
        "price_change": price_change,
        "price_change_pct": change_pct,
        "previous_close": previous_close,
        "session_open": session_open,
        "session_high": session_high,
        "session_low": session_low,
        "session_volume": session_volume,
        "last_bar_volume": last_bar_volume,
        "minutes_since_open": minutes_active,
        "note": note,
    }


def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:

    best_ind_params = {
        # Moving Averages
        "close_50_sma": (
            "50 SMA: A medium-term trend indicator. "
            "Usage: Identify trend direction and serve as dynamic support/resistance. "
            "Tips: It lags price; combine with faster indicators for timely signals."
        ),
        "close_200_sma": (
            "200 SMA: A long-term trend benchmark. "
            "Usage: Confirm overall market trend and identify golden/death cross setups. "
            "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
        ),
        "close_10_ema": (
            "10 EMA: A responsive short-term average. "
            "Usage: Capture quick shifts in momentum and potential entry points. "
            "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
        ),
        # MACD Related
        "macd": (
            "MACD: Computes momentum via differences of EMAs. "
            "Usage: Look for crossovers and divergence as signals of trend changes. "
            "Tips: Confirm with other indicators in low-volatility or sideways markets."
        ),
        "macds": (
            "MACD Signal: An EMA smoothing of the MACD line. "
            "Usage: Use crossovers with the MACD line to trigger trades. "
            "Tips: Should be part of a broader strategy to avoid false positives."
        ),
        "macdh": (
            "MACD Histogram: Shows the gap between the MACD line and its signal. "
            "Usage: Visualize momentum strength and spot divergence early. "
            "Tips: Can be volatile; complement with additional filters in fast-moving markets."
        ),
        # Momentum Indicators
        "rsi": (
            "RSI: Measures momentum to flag overbought/oversold conditions. "
            "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
            "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
        ),
        # Volatility Indicators
        "boll": (
            "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
            "Usage: Acts as a dynamic benchmark for price movement. "
            "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
        ),
        "boll_ub": (
            "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
            "Usage: Signals potential overbought conditions and breakout zones. "
            "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
        ),
        "boll_lb": (
            "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
            "Usage: Indicates potential oversold conditions. "
            "Tips: Use additional analysis to avoid false reversal signals."
        ),
        "atr": (
            "ATR: Averages true range to measure volatility. "
            "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
            "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
        ),
        # Volume-Based Indicators
        "vwma": (
            "VWMA: A moving average weighted by volume. "
            "Usage: Confirm trends by integrating price action with volume data. "
            "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
        ),
        "mfi": (
            "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
            "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
            "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
        ),
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}"
        )

    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    # Optimized: Get stock data once and calculate indicators for all dates
    try:
        indicator_data = _get_stock_stats_bulk(symbol, indicator, curr_date)
        
        # Generate the date range we need
        current_dt = curr_date_dt
        date_values = []
        
        while current_dt >= before:
            date_str = current_dt.strftime('%Y-%m-%d')
            
            # Look up the indicator value for this date
            if date_str in indicator_data:
                indicator_value = indicator_data[date_str]
            else:
                indicator_value = "N/A: Not a trading day (weekend or holiday)"
            
            date_values.append((date_str, indicator_value))
            current_dt = current_dt - relativedelta(days=1)
        
        # Build the result string
        ind_string = ""
        for date_str, value in date_values:
            ind_string += f"{date_str}: {value}\n"
        
    except Exception as e:
        print(f"Error getting bulk stockstats data: {e}")
        # Fallback to original implementation if bulk method fails
        ind_string = ""
        curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        while curr_date_dt >= before:
            indicator_value = get_stockstats_indicator(
                symbol, indicator, curr_date_dt.strftime("%Y-%m-%d")
            )
            ind_string += f"{curr_date_dt.strftime('%Y-%m-%d')}: {indicator_value}\n"
            curr_date_dt = curr_date_dt - relativedelta(days=1)

    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )

    return result_str


def _get_stock_stats_bulk(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to calculate"],
    curr_date: Annotated[str, "current date for reference"]
) -> dict:
    """
    Optimized bulk calculation of stock stats indicators.
    Fetches data once and calculates indicator for all available dates.
    Returns dict mapping date strings to indicator values.
    """
    from .config import get_config
    import pandas as pd
    from stockstats import wrap
    import os
    
    config = get_config()
    online = config["data_vendors"]["technical_indicators"] != "local"
    
    if not online:
        # Local data path
        try:
            data = pd.read_csv(
                os.path.join(
                    config.get("data_cache_dir", "data"),
                    f"{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
                )
            )
            df = wrap(data)
        except FileNotFoundError:
            raise Exception("Stockstats fail: Yahoo Finance data not fetched yet!")
    else:
        # Online data fetching with caching
        today_date = pd.Timestamp.today()
        curr_date_dt = pd.to_datetime(curr_date)
        
        end_date = today_date
        start_date = today_date - pd.DateOffset(years=15)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        os.makedirs(config["data_cache_dir"], exist_ok=True)
        
        data_file = os.path.join(
            config["data_cache_dir"],
            f"{symbol}-YFin-data-{start_date_str}-{end_date_str}.csv",
        )
        
        if os.path.exists(data_file):
            data = pd.read_csv(data_file)
            data["Date"] = pd.to_datetime(data["Date"])
        else:
            data = yf.download(
                symbol,
                start=start_date_str,
                end=end_date_str,
                multi_level_index=False,
                progress=False,
                auto_adjust=True,
            )
            data = data.reset_index()
            data.to_csv(data_file, index=False)
        
        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    
    # Calculate the indicator for all rows at once
    df[indicator]  # This triggers stockstats to calculate the indicator
    
    # Create a dictionary mapping date strings to indicator values
    result_dict = {}
    for _, row in df.iterrows():
        date_str = row["Date"]
        indicator_value = row[indicator]
        
        # Handle NaN/None values
        if pd.isna(indicator_value):
            result_dict[date_str] = "N/A"
        else:
            result_dict[date_str] = str(indicator_value)
    
    return result_dict


def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
) -> str:

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    curr_date = curr_date_dt.strftime("%Y-%m-%d")

    try:
        indicator_value = StockstatsUtils.get_stock_stats(
            symbol,
            indicator,
            curr_date,
        )
    except Exception as e:
        print(
            f"Error getting stockstats indicator data for indicator {indicator} on {curr_date}: {e}"
        )
        return ""

    return str(indicator_value)


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None
):
    """Get balance sheet data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        
        if freq.lower() == "quarterly":
            data = ticker_obj.quarterly_balance_sheet
        else:
            data = ticker_obj.balance_sheet
            
        if data.empty:
            return f"No balance sheet data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Balance Sheet data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving balance sheet for {ticker}: {str(e)}"


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None
):
    """Get cash flow data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        
        if freq.lower() == "quarterly":
            data = ticker_obj.quarterly_cashflow
        else:
            data = ticker_obj.cashflow
            
        if data.empty:
            return f"No cash flow data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Cash Flow data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving cash flow for {ticker}: {str(e)}"


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None
):
    """Get income statement data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        
        if freq.lower() == "quarterly":
            data = ticker_obj.quarterly_income_stmt
        else:
            data = ticker_obj.income_stmt
            
        if data.empty:
            return f"No income statement data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Income Statement data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving income statement for {ticker}: {str(e)}"


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"]
):
    """Get insider transactions data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = ticker_obj.insider_transactions
        
        if data is None or data.empty:
            return f"No insider transactions data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Insider Transactions data for {ticker.upper()}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving insider transactions for {ticker}: {str(e)}"