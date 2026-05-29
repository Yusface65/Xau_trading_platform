# XAU/USD Multi-Timeframe Trading Dashboard

Streamlit dashboard for Gold (XAU/USD) trading signals with:

- 1H and 3M timeframes
- SMA, EMA, RSI, MACD, Bollinger Bands
- FVG (Fair Value Gap) zone detection
- ML prediction probabilities (Random Forest)

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure authentication

The dashboard is password-protected. Set `APP_PASSWORD` using one of:

- **Streamlit secrets** (recommended): copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set a strong password.
- **Environment variable**: `export APP_PASSWORD="your-strong-password"`
- **Streamlit Cloud**: add `APP_PASSWORD` in the app's Secrets settings.

### 3. Run locally

```bash
streamlit run dashboard.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. Sign in at https://streamlit.io/cloud with GitHub.
3. Click **New App**, select repo, branch, and `dashboard.py`.
4. In **Advanced settings → Secrets**, add `APP_PASSWORD = "your-strong-password"`.
5. Access your live dashboard via the generated URL.

## Project structure

```
├── dashboard.py                      # Main Streamlit app
├── requirements.txt                  # Pinned dependencies
├── .streamlit/secrets.toml.example   # Auth config template
├── .gitignore                        # Prevents committing secrets
└── README.md
```

## Security

- **Authentication**: password gate via `st.secrets` or environment variable.
- **Input validation**: allowlisted symbols, periods, and intervals.
- **Error handling**: external API failures show user-friendly messages (no stack traces).
- **Pinned dependencies**: version ranges prevent supply-chain drift.
- **Secrets exclusion**: `.gitignore` blocks `.streamlit/secrets.toml` from commits.
