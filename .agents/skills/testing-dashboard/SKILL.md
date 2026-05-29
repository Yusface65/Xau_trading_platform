---
name: testing-xauusd-dashboard
description: Test the XAU/USD Streamlit trading dashboard end-to-end. Use when verifying dashboard UI, data fetching, indicators, or ML predictions.
---

# Testing the XAU/USD Dashboard

## Setup

```bash
pip install -r Xauusd/requirements.txt
streamlit run Xauusd/dashboard.py --server.port 8501 --server.headless true
```

Then navigate to http://localhost:8501 in the browser.

## Common Issues

- **yfinance MultiIndex columns**: Newer yfinance versions return DataFrames with multi-level columns `(price, ticker)`. If you see `ValueError: Data must be 1-dimensional`, the columns need flattening with `df.columns = df.columns.get_level_values(0)`.
- **Invalid intervals**: yfinance valid intervals are: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 4h, 1d, 5d, 1wk, 1mo, 3mo. The "3m" interval is NOT supported.
- **ta library expects 1D Series**: Use `.squeeze()` on DataFrame columns before passing to `ta` indicators.
- **Deprecated `@st.cache`**: Use `@st.cache_data` instead for Streamlit 1.18+.
- **Data loading time**: The dashboard fetches live data from Yahoo Finance on first load. Allow 10-15 seconds for initial render.

## What to Verify

1. **No errors/tracebacks** on the page
2. **Metrics section**: Current Price (formatted as `$X.XX`), Signal (BUY/SELL/HOLD), Prediction Confidence (XX.XX%)
3. **1H Chart**: Shows Close (blue), SMA50 (orange), SMA200 (red) lines with limited FVG zones
4. **5M Chart**: Same indicators on shorter timeframe
5. **FVG zones**: Should be limited (max 10 per chart) and not extend to the end of the chart
6. **Chart readability**: Price action should be clearly visible, not obscured by overlapping rectangles

## Devin Secrets Needed

None — the dashboard uses public Yahoo Finance data (GC=F gold futures symbol).
