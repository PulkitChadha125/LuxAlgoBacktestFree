# Fyers Streamlit Backtesting Dashboard

A simple, single-folder backtesting dashboard built for quick testing and YouTube demos.

This project uses:
- **Fyers API** for historical market data
- **backtesting.py** for strategy simulation
- **Streamlit** for UI/dashboard

---

## Project Structure

```text
Lux Algo Backtest/
├── app.py
├── FyresIntegration.py
├── FyersCredentials.csv
├── requirements.txt
└── README.md
```

---

## Features

- Sidebar controls:
  - Symbol input
  - Timeframe selector
  - Start Date / End Date
  - "Use Today for End Date" option
  - Initial Capital
  - EMA period and RSI period
  - Optimization inputs (EMA/RSI ranges)
- Backtest results:
  - Performance summary cards
  - Equity curve
  - Drawdown curve
  - Trades table
  - PnL distribution chart
- Exports:
  - Download Summary CSV
  - Download Trades CSV
  - Download Drawdown CSV
  - Download PnL Distribution CSV
  - Save all CSVs into a separate timestamped folder

---

## Strategy Rules

### Buy Entry
- Close > EMA
- Close breaks above previous 20-candle high

### Buy Exit
- RSI > 70 **OR**
- Close < EMA

### Sell Entry
- Close < EMA
- Close breaks below previous 20-candle low

### Sell Exit
- RSI < 30 **OR**
- Close > EMA

---

## Fyers Credentials Setup

Update `FyersCredentials.csv` with your actual values.

Required keys:
- `client_id`
- `secret_key`
- `redirect_uri`
- `totpkey`
- `FY_ID`
- `PIN`

The app reuses your login flow from `FyresIntegration.py`.

---

## Run the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

App URL (default): [http://localhost:8501](http://localhost:8501)

---

## Intraday Date Range Handling

Fyers intraday history requests are limited to around 100 days per API request.

This app automatically handles large ranges by:
1. Splitting your selected start/end date into 99-day chunks
2. Fetching each chunk separately
3. Merging and de-duplicating all candles

So you can request **1 year** intraday data directly from the UI.

---

## Optimization

If "Enable EMA/RSI Optimization" is selected, the app:
- Runs parameter optimization with `backtesting.py`
- Finds the best EMA and RSI combination (maximizing `Return [%]`)
- Shows optimization table
- Allows optimization CSV download

---

## Saved Results Folder

When you click **Save All CSVs to Separate Folder**, files are stored in:

```text
saved_results/backtest_YYYYMMDD_HHMMSS/
```

Possible files:
- `summary.csv`
- `trades.csv`
- `drawdown_curve.csv`
- `pnl_distribution.csv`

---

## Notes

- This project is intentionally simple (no separate backend/frontend folders).
- Good for learning, experimentation, and quick content demos.
- For production use, add stronger logging, retries, and secure secret handling.
