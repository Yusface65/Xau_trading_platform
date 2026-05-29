"""
Gold XAU/USD Multi-Timeframe Trading Dashboard

A Streamlit dashboard for Gold (XAU/USD) trading signals with:
- 1H and 3M timeframes
- SMA, EMA, RSI, MACD, Bollinger Bands
- FVG zones detection
- ML prediction probabilities
"""

import logging
import sys

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import ta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# -------- Logging Configuration --------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# -------- Constants --------
VALID_PERIODS = [
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max",
    "7d", "60d", "90d",
]
VALID_INTERVALS = [
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo", "3m",
]
MIN_ROWS_FOR_INDICATORS = 200
MIN_ROWS_FOR_MODEL = 50


# -------- Input Validation --------
def validate_symbol(symbol: str) -> str:
    """Validate and sanitize the trading symbol."""
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string.")
    symbol = symbol.strip().upper()
    if len(symbol) > 20:
        raise ValueError(f"Symbol '{symbol}' is too long (max 20 characters).")
    return symbol


def validate_period(period: str) -> str:
    """Validate the data period parameter."""
    if not period or not isinstance(period, str):
        raise ValueError("Period must be a non-empty string.")
    period = period.strip().lower()
    if period not in VALID_PERIODS:
        raise ValueError(
            f"Invalid period '{period}'. Must be one of: {VALID_PERIODS}"
        )
    return period


def validate_interval(interval: str) -> str:
    """Validate the data interval parameter."""
    if not interval or not isinstance(interval, str):
        raise ValueError("Interval must be a non-empty string.")
    interval = interval.strip().lower()
    if interval not in VALID_INTERVALS:
        raise ValueError(
            f"Invalid interval '{interval}'. Must be one of: {VALID_INTERVALS}"
        )
    return interval


# -------- Fetch Data --------
@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance with validation and error handling.

    Args:
        symbol: The ticker symbol (e.g., "GC=F" for gold futures).
        period: The data period (e.g., "90d", "7d").
        interval: The data interval (e.g., "1h", "3m").

    Returns:
        A DataFrame with OHLCV columns.

    Raises:
        ValueError: If parameters are invalid or data is empty.
        ConnectionError: If the API request fails.
    """
    symbol = validate_symbol(symbol)
    period = validate_period(period)
    interval = validate_interval(interval)

    logger.info(
        "Fetching data for symbol=%s, period=%s, interval=%s",
        symbol, period, interval,
    )

    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
    except Exception as exc:
        logger.error(
            "API request failed for symbol=%s, period=%s, interval=%s: %s",
            symbol, period, interval, exc,
        )
        raise ConnectionError(
            f"Failed to fetch data from Yahoo Finance for '{symbol}': {exc}"
        ) from exc

    if df is None or df.empty:
        logger.warning(
            "No data returned for symbol=%s, period=%s, interval=%s",
            symbol, period, interval,
        )
        raise ValueError(
            f"No data returned for symbol '{symbol}' with period='{period}' "
            f"and interval='{interval}'. The market may be closed or the "
            f"symbol may be invalid."
        )

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error("Missing required columns: %s", missing_cols)
        raise ValueError(
            f"Data is missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[required_columns].copy()

    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    nan_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
    if nan_pct > 50:
        logger.warning("Data contains %.1f%% NaN values", nan_pct)

    logger.info("Successfully fetched %d rows of data for %s", len(df), symbol)
    return df


# -------- Indicators --------
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators on the DataFrame.

    Adds SMA50, SMA200, EMA20, RSI, MACD, MACD_Signal, BB_High, BB_Low.

    Args:
        df: DataFrame with at least a 'Close' column.

    Returns:
        DataFrame with indicator columns added.
    """
    if df is None or df.empty:
        logger.error("Cannot calculate indicators on empty DataFrame")
        raise ValueError("Cannot calculate indicators: DataFrame is empty.")

    if "Close" not in df.columns:
        logger.error("DataFrame missing 'Close' column for indicator calculation")
        raise ValueError("DataFrame must contain a 'Close' column.")

    if len(df) < MIN_ROWS_FOR_INDICATORS:
        logger.warning(
            "DataFrame has %d rows, which is less than the recommended %d "
            "for accurate SMA200 calculation. Indicators may be unreliable.",
            len(df), MIN_ROWS_FOR_INDICATORS,
        )

    df = df.copy()

    try:
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(df["Close"])
        df["BB_High"] = bb.bollinger_hband()
        df["BB_Low"] = bb.bollinger_lband()
    except Exception as exc:
        logger.error("Failed to calculate indicators: %s", exc)
        raise RuntimeError(
            f"Error calculating technical indicators: {exc}"
        ) from exc

    logger.info("Calculated indicators for DataFrame with %d rows", len(df))
    return df


# -------- FVG Detection --------
def detect_fvg(df: pd.DataFrame) -> list:
    """
    Detect Fair Value Gaps (FVG) in the price data.

    Args:
        df: DataFrame with 'High' and 'Low' columns.

    Returns:
        List of tuples (index, type, price1, price2) for detected FVGs.
    """
    if df is None or df.empty:
        logger.warning("Cannot detect FVG on empty DataFrame")
        return []

    if "High" not in df.columns or "Low" not in df.columns:
        logger.error("DataFrame missing 'High' or 'Low' columns for FVG detection")
        return []

    if len(df) < 3:
        logger.warning("Need at least 3 rows for FVG detection, got %d", len(df))
        return []

    fvg_list = []
    try:
        for i in range(2, len(df)):
            low_prev2 = df["Low"].iloc[i - 2]
            high_curr = df["High"].iloc[i]
            high_prev2 = df["High"].iloc[i - 2]
            low_curr = df["Low"].iloc[i]

            if pd.isna(low_prev2) or pd.isna(high_curr):
                continue
            if pd.isna(high_prev2) or pd.isna(low_curr):
                continue

            if low_prev2 > high_curr:
                fvg_list.append((i, "bullish", low_curr, high_prev2))
            elif high_prev2 < low_curr:
                fvg_list.append((i, "bearish", high_curr, low_prev2))
    except Exception as exc:
        logger.error("Error during FVG detection: %s", exc)
        return []

    logger.info("Detected %d FVG zones", len(fvg_list))
    return fvg_list


# -------- ML Model --------
def train_model(df: pd.DataFrame):
    """
    Train a RandomForest classifier to predict price direction.

    Args:
        df: DataFrame with indicator columns already calculated.

    Returns:
        Tuple of (trained model, feature list) or (None, None) on failure.
    """
    if df is None or df.empty:
        logger.error("Cannot train model on empty DataFrame")
        return None, None

    required_features = ["SMA50", "SMA200", "EMA20", "RSI", "MACD", "MACD_Signal"]
    missing = [f for f in required_features if f not in df.columns]
    if missing:
        logger.error("Missing features for model training: %s", missing)
        return None, None

    df = df.copy()

    try:
        df["Trend"] = np.where(df["SMA50"] > df["SMA200"], 1, -1)
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        features = ["SMA50", "SMA200", "EMA20", "RSI", "MACD_Hist", "Trend"]

        df = df.dropna(subset=features)

        if len(df) < MIN_ROWS_FOR_MODEL:
            logger.warning(
                "Insufficient data for model training: %d rows (need %d)",
                len(df), MIN_ROWS_FOR_MODEL,
            )
            return None, None

        df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
        df = df.dropna(subset=["Target"])

        if len(df) < MIN_ROWS_FOR_MODEL:
            logger.warning(
                "Insufficient data after target creation: %d rows", len(df)
            )
            return None, None

        X = df[features]
        y = df["Target"].astype(int)

        if y.nunique() < 2:
            logger.warning("Target variable has only one class; cannot train model")
            return None, None

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        logger.info(
            "Model trained successfully on %d samples with %d features",
            len(X), len(features),
        )
        return model, features

    except Exception as exc:
        logger.error("Model training failed: %s", exc)
        return None, None


# -------- Multi-Timeframe Signal --------
def multi_tf_signal(df_1h, df_3m, fvg_1h, fvg_3m, model, features):
    """
    Generate a trading signal by combining multi-timeframe analysis with ML.

    Args:
        df_1h: 1-hour DataFrame with indicators.
        df_3m: 3-minute DataFrame with indicators.
        fvg_1h: FVG list for 1H timeframe.
        fvg_3m: FVG list for 3M timeframe.
        model: Trained ML model (or None).
        features: List of feature column names (or None).

    Returns:
        Tuple of (signal_string, prediction_probability).
    """
    if model is None or features is None:
        logger.warning("Model not available; returning HOLD signal")
        return "HOLD", 0.5

    if df_1h is None or df_1h.empty or df_3m is None or df_3m.empty:
        logger.warning("Insufficient data for signal generation")
        return "HOLD", 0.5

    try:
        last_1h = df_1h.iloc[-1]
        last_3m = df_3m.iloc[-1]

        # Check for NaN in critical fields
        critical_fields = ["SMA50", "SMA200"]
        for field in critical_fields:
            if pd.isna(last_1h.get(field)) or pd.isna(last_3m.get(field)):
                logger.warning("NaN detected in %s; returning HOLD", field)
                return "HOLD", 0.5

        bullish_1h = last_1h["SMA50"] > last_1h["SMA200"]
        bearish_1h = last_1h["SMA50"] < last_1h["SMA200"]
        bullish_3m = last_3m["SMA50"] > last_3m["SMA200"]
        bearish_3m = last_3m["SMA50"] < last_3m["SMA200"]

        fvg_buy_1h = any(
            z[0] == len(df_1h) - 1 and z[1] == "bullish" for z in fvg_1h
        )
        fvg_sell_1h = any(
            z[0] == len(df_1h) - 1 and z[1] == "bearish" for z in fvg_1h
        )
        fvg_buy_3m = any(
            z[0] == len(df_3m) - 1 and z[1] == "bullish" for z in fvg_3m
        )
        fvg_sell_3m = any(
            z[0] == len(df_3m) - 1 and z[1] == "bearish" for z in fvg_3m
        )

        # Validate feature availability
        missing_features = [f for f in features if f not in last_3m.index]
        if missing_features:
            logger.warning("Missing features in data: %s", missing_features)
            return "HOLD", 0.5

        X_last = last_3m[features].values.reshape(1, -1)

        if np.isnan(X_last).any():
            logger.warning("NaN values in prediction features; returning HOLD")
            return "HOLD", 0.5

        pred_prob = model.predict_proba(X_last)[0][1]

        signal = "HOLD"
        if (
            bullish_1h and bullish_3m
            and fvg_buy_1h and fvg_buy_3m
            and pred_prob > 0.6
        ):
            signal = "BUY"
        elif (
            bearish_1h and bearish_3m
            and fvg_sell_1h and fvg_sell_3m
            and pred_prob > 0.6
        ):
            signal = "SELL"

        logger.info("Signal generated: %s (confidence: %.2f%%)", signal, pred_prob * 100)
        return signal, pred_prob

    except Exception as exc:
        logger.error("Error generating signal: %s", exc)
        return "HOLD", 0.5


# -------- Plot Charts --------
def plot_chart(df: pd.DataFrame, title: str, fvg_list: list) -> None:
    """
    Plot a price chart with indicators and FVG zones.

    Args:
        df: DataFrame with 'Close', 'SMA50', 'SMA200' columns.
        title: Chart title.
        fvg_list: List of FVG tuples for overlay.
    """
    if df is None or df.empty:
        st.warning(f"No data available to plot: {title}")
        logger.warning("Cannot plot chart '%s': empty DataFrame", title)
        return

    try:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="blue"))
        )
        if "SMA50" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["SMA50"], name="SMA50", line=dict(color="orange")
                )
            )
        if "SMA200" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["SMA200"], name="SMA200", line=dict(color="red")
                )
            )

        for fvg in fvg_list:
            if fvg[0] >= len(df):
                continue
            color = "green" if fvg[1] == "bullish" else "red"
            fig.add_shape(
                type="rect",
                x0=df.index[fvg[0]],
                x1=df.index[-1],
                y0=fvg[2],
                y1=fvg[3],
                fillcolor=color,
                opacity=0.2,
            )

        fig.update_layout(title=title)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        logger.error("Error plotting chart '%s': %s", title, exc)
        st.error(f"Failed to render chart: {title}. Error: {exc}")


# -------- Main App --------
def main():
    """Main entry point for the Streamlit dashboard."""
    st.set_page_config(page_title="XAU/USD Dashboard", layout="wide")
    st.title("Gold XAU/USD Multi-Timeframe Trading Dashboard")

    logger.info("Dashboard started")

    # Load 1H data
    try:
        df_1h = fetch_data("GC=F", "90d", "1h")
    except (ValueError, ConnectionError) as exc:
        st.error(f"Failed to load 1H data: {exc}")
        logger.error("1H data load failed: %s", exc)
        st.stop()

    # Load 3M data
    try:
        df_3m = fetch_data("GC=F", "7d", "3m")
    except (ValueError, ConnectionError) as exc:
        st.error(f"Failed to load 3M data: {exc}")
        logger.error("3M data load failed: %s", exc)
        st.stop()

    # Calculate indicators
    try:
        df_1h = calculate_indicators(df_1h)
    except (ValueError, RuntimeError) as exc:
        st.error(f"Failed to calculate 1H indicators: {exc}")
        logger.error("1H indicator calculation failed: %s", exc)
        st.stop()

    try:
        df_3m = calculate_indicators(df_3m)
    except (ValueError, RuntimeError) as exc:
        st.error(f"Failed to calculate 3M indicators: {exc}")
        logger.error("3M indicator calculation failed: %s", exc)
        st.stop()

    # Detect FVG
    fvg_1h = detect_fvg(df_1h)
    fvg_3m = detect_fvg(df_3m)

    # Train model
    model, features = train_model(df_3m)
    if model is None:
        st.warning(
            "ML model could not be trained due to insufficient data. "
            "Signal will default to HOLD."
        )

    # Generate signal
    signal, prob = multi_tf_signal(df_1h, df_3m, fvg_1h, fvg_3m, model, features)

    # Display Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        current_price = df_3m["Close"].iloc[-1]
        if pd.isna(current_price):
            st.metric("Current Price", "N/A")
            st.warning("Current price is unavailable")
        else:
            st.metric("Current Price", f"${current_price:,.2f}")
    with col2:
        st.metric("Signal", signal)
    with col3:
        st.metric("Prediction Confidence", f"{prob * 100:.2f}%")

    # Plot Charts
    st.subheader("1H Chart with FVG Zones")
    plot_chart(df_1h, "1H Chart with FVG", fvg_1h)

    st.subheader("3M Chart with FVG Zones")
    plot_chart(df_3m, "3M Chart with FVG", fvg_3m)

    # Footer
    st.markdown("---")
    st.caption(
        "Data sourced from Yahoo Finance. Signals are for informational purposes only "
        "and do not constitute financial advice."
    )
    logger.info("Dashboard rendered successfully")


if __name__ == "__main__":
    main()
