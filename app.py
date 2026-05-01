import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from backtesting import Backtest, Strategy
from datetime import date, datetime, timedelta


st.set_page_config(page_title="Fyers Backtest Dashboard", layout="wide")


def load_credentials(csv_path: Path) -> dict:
    if not csv_path.exists():
        raise FileNotFoundError(f"Credentials file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Title" not in df.columns or "Value" not in df.columns:
        raise ValueError("FyersCredentials.csv must contain Title and Value columns")

    creds = dict(zip(df["Title"], df["Value"]))
    required_keys = ["client_id", "secret_key", "redirect_uri", "totpkey", "FY_ID", "PIN"]
    missing = [k for k in required_keys if not str(creds.get(k, "")).strip()]
    if missing:
        raise ValueError(f"Missing values in FyersCredentials.csv: {', '.join(missing)}")
    return creds


@st.cache_resource(show_spinner=False)
def get_fyers_client(credentials_path: str):
    # Reuse your existing integration code.
    import FyresIntegration

    creds = load_credentials(Path(credentials_path))
    login_result = FyresIntegration.automated_login(
        client_id=str(creds["client_id"]),
        secret_key=str(creds["secret_key"]),
        FY_ID=str(creds["FY_ID"]),
        TOTP_KEY=str(creds["totpkey"]),
        PIN=str(creds["PIN"]),
        redirect_uri=str(creds["redirect_uri"]),
    )
    return login_result["fyers"]


def _download_single_chunk(fyers, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    payload = {
        "symbol": symbol,
        "resolution": str(timeframe),
        "date_format": "1",
        "range_from": start_date,
        "range_to": end_date,
        "cont_flag": "1",
    }
    response = fyers.history(data=payload)
    candles = response.get("candles", [])
    if not candles:
        raise ValueError(f"No candles returned from Fyers. Response: {response}")

    data = pd.DataFrame(candles, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    data["Date"] = pd.to_datetime(data["Date"], unit="s")
    data = data.sort_values("Date").reset_index(drop=True)
    return data


def build_intraday_chunks(start_date: str, end_date: str, chunk_days: int = 99):
    """
    Create sequential date windows from start_date to end_date.
    Example for 1 year: [d1-d99], [d100-d198], ... [last_chunk]
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    if start_dt > end_dt:
        raise ValueError("Start Date cannot be greater than End Date")

    chunks = []
    current_start = start_dt
    while current_start <= end_dt:
        current_end = min(current_start + timedelta(days=chunk_days), end_dt)
        chunks.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)
    return chunks


def download_fyers_data(fyers, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download historical data from Fyers.
    For intraday timeframes, fetch in chunks (<=100 days per call) and merge.
    """
    intraday_timeframes = {"1", "2", "3", "5", "10", "15", "20", "30", "45", "60", "120", "180", "240"}
    if str(timeframe) not in intraday_timeframes:
        return _download_single_chunk(fyers, symbol, timeframe, start_date, end_date)

    chunks = []
    for current_start, current_end in build_intraday_chunks(start_date, end_date, chunk_days=99):
        chunk_df = _download_single_chunk(
            fyers,
            symbol,
            timeframe,
            current_start.strftime("%Y-%m-%d"),
            current_end.strftime("%Y-%m-%d"),
        )
        chunks.append(chunk_df)

    merged = pd.concat(chunks, ignore_index=True)
    merged = merged.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return merged


def calculate_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_ema_array(close_values, period: int):
    close = pd.Series(close_values)
    return close.ewm(span=period, adjust=False).mean().to_numpy()


def calculate_rsi_array(close_values, period: int):
    close = pd.Series(close_values)
    return calculate_rsi(close, period).to_numpy()


def bullish_breakout_array(high_values, close_values):
    high = pd.Series(high_values)
    close = pd.Series(close_values)
    prev_20_high = high.rolling(20).max().shift(1)
    out = (close > prev_20_high) & (close.shift(1) <= prev_20_high)
    return out.fillna(False).to_numpy()


def bearish_breakdown_array(low_values, close_values):
    low = pd.Series(low_values)
    close = pd.Series(close_values)
    prev_20_low = low.rolling(20).min().shift(1)
    out = (close < prev_20_low) & (close.shift(1) >= prev_20_low)
    return out.fillna(False).to_numpy()


def add_indicators(df: pd.DataFrame, ema_period: int, rsi_period: int) -> pd.DataFrame:
    out = df.copy()
    out["EMA"] = out["Close"].ewm(span=ema_period, adjust=False).mean()
    out["RSI"] = calculate_rsi(out["Close"], rsi_period)

    prev_20_high = out["High"].rolling(20).max().shift(1)
    prev_20_low = out["Low"].rolling(20).min().shift(1)

    out["BullishBreakout"] = (out["Close"] > prev_20_high) & (out["Close"].shift(1) <= prev_20_high)
    out["BearishBreakdown"] = (out["Close"] < prev_20_low) & (out["Close"].shift(1) >= prev_20_low)
    return out


class LuxSimpleStrategy(Strategy):
    # These will be set from sidebar inputs before running Backtest.
    EMA_PERIOD = 50
    RSI_PERIOD = 14

    def init(self):
        self.ema = self.I(calculate_ema_array, self.data.Close, int(self.EMA_PERIOD))
        self.rsi = self.I(calculate_rsi_array, self.data.Close, int(self.RSI_PERIOD))
        self.bullish_breakout = self.I(bullish_breakout_array, self.data.High, self.data.Close)
        self.bearish_breakdown = self.I(bearish_breakdown_array, self.data.Low, self.data.Close)

    def next(self):
        close = float(self.data.Close[-1])
        ema = float(self.ema[-1])
        rsi = float(self.rsi[-1])

        if np.isnan(ema) or np.isnan(rsi):
            return

        # Exit conditions first.
        if self.position:
            if self.position.is_long and (rsi > 70 or close < ema):
                self.position.close()
                return
            if self.position.is_short and (rsi < 30 or close > ema):
                self.position.close()
                return

        # Entry conditions.
        if not self.position:
            if close > ema and bool(self.bullish_breakout[-1]):
                self.buy()
            elif close < ema and bool(self.bearish_breakdown[-1]):
                self.sell()


def run_backtest(df: pd.DataFrame, initial_capital: float):
    bt_df = df.set_index("Date").copy()
    bt = Backtest(
        bt_df,
        LuxSimpleStrategy,
        cash=float(initial_capital),
        commission=0.001,
        trade_on_close=True,
        exclusive_orders=True,
    )
    stats = bt.run()
    return stats


def run_optimization(
    df: pd.DataFrame,
    initial_capital: float,
    ema_min: int,
    ema_max: int,
    ema_step: int,
    rsi_min: int,
    rsi_max: int,
    rsi_step: int,
):
    bt_df = df.set_index("Date").copy()
    bt = Backtest(
        bt_df,
        LuxSimpleStrategy,
        cash=float(initial_capital),
        commission=0.001,
        trade_on_close=True,
        exclusive_orders=True,
    )

    stats, heatmap = bt.optimize(
        EMA_PERIOD=range(ema_min, ema_max + 1, ema_step),
        RSI_PERIOD=range(rsi_min, rsi_max + 1, rsi_step),
        maximize="Return [%]",
        return_heatmap=True,
        constraint=lambda p: p.EMA_PERIOD > p.RSI_PERIOD,
    )
    return stats, heatmap


def stats_to_summary(stats) -> dict:
    def to_num(v):
        if pd.isna(v):
            return None
        return float(v)

    return {
        "Final Equity": to_num(stats.get("Equity Final [$]")),
        "Total Return %": to_num(stats.get("Return [%]")),
        "Win Rate %": to_num(stats.get("Win Rate [%]")),
        "Total Trades": int(stats.get("# Trades", 0) or 0),
        "Max Drawdown %": to_num(stats.get("Max. Drawdown [%]")),
        "Profit Factor": to_num(stats.get("Profit Factor")),
    }


def main():
    st.title("Fyers Backtesting Dashboard")
    st.caption("Simple Streamlit UI + backtesting.py")

    credentials_file = Path("FyersCredentials.csv")

    with st.sidebar:
        st.header("Backtest Settings")
        symbol = st.text_input("Symbol", value="NSE:NIFTY50-INDEX")
        timeframe = st.selectbox("Time Frame", options=["5", "15", "30", "60", "D"], index=1)
        start_date = st.date_input("Start Date", value=pd.Timestamp("2025-01-01")).strftime("%Y-%m-%d")
        use_today = st.checkbox("Use Today for End Date", value=True)
        if use_today:
            end_date = date.today().strftime("%Y-%m-%d")
            st.text_input("End Date", value=end_date, disabled=True)
        else:
            end_date = st.date_input("End Date", value=pd.Timestamp("2025-12-31")).strftime("%Y-%m-%d")
        initial_capital = st.number_input("Initial Capital", min_value=10000.0, value=1000000.0, step=10000.0)

        st.markdown("---")
        ema_period = st.number_input("EMA Period", min_value=5, max_value=300, value=50, step=1)
        rsi_period = st.number_input("RSI Period", min_value=2, max_value=100, value=14, step=1)

        st.markdown("---")
        st.subheader("Optimization")
        enable_optimization = st.checkbox("Enable EMA/RSI Optimization", value=False)
        if enable_optimization:
            ema_min = st.number_input("EMA Min", min_value=5, max_value=300, value=20, step=1)
            ema_max = st.number_input("EMA Max", min_value=5, max_value=300, value=80, step=1)
            ema_step = st.number_input("EMA Step", min_value=1, max_value=50, value=5, step=1)
            rsi_min = st.number_input("RSI Min", min_value=2, max_value=100, value=8, step=1)
            rsi_max = st.number_input("RSI Max", min_value=2, max_value=100, value=30, step=1)
            rsi_step = st.number_input("RSI Step", min_value=1, max_value=20, value=2, step=1)

        run_clicked = st.button("Run Backtest", type="primary", use_container_width=True)

    if run_clicked:
        try:
            effective_start_date = start_date
            effective_end_date = end_date

            with st.spinner("Logging in to Fyers and downloading data..."):
                fyers = get_fyers_client(str(credentials_file))
                raw_df = download_fyers_data(
                    fyers, symbol, timeframe, effective_start_date, effective_end_date
                )

            intraday_timeframes = {"1", "2", "3", "5", "10", "15", "20", "30", "45", "60", "120", "180", "240"}
            if str(timeframe) in intraday_timeframes and (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days > 99:
                total_chunks = len(build_intraday_chunks(start_date, end_date, chunk_days=99))
                st.info(
                    "Large intraday range detected. Data was fetched in multiple Fyers chunks "
                    f"({total_chunks} requests) and merged automatically."
                )

            data_df = add_indicators(raw_df, ema_period=ema_period, rsi_period=rsi_period)

            LuxSimpleStrategy.EMA_PERIOD = int(ema_period)
            LuxSimpleStrategy.RSI_PERIOD = int(rsi_period)

            if enable_optimization:
                with st.spinner("Running optimization..."):
                    best_stats, heatmap = run_optimization(
                        data_df,
                        initial_capital=initial_capital,
                        ema_min=int(ema_min),
                        ema_max=int(ema_max),
                        ema_step=int(ema_step),
                        rsi_min=int(rsi_min),
                        rsi_max=int(rsi_max),
                        rsi_step=int(rsi_step),
                    )
                stats = best_stats
            else:
                with st.spinner("Running backtest..."):
                    stats = run_backtest(data_df, initial_capital=initial_capital)

            summary = stats_to_summary(stats)
            equity_curve = stats.get("_equity_curve", pd.DataFrame()).reset_index()
            trades_df = stats.get("_trades", pd.DataFrame()).copy()
            drawdown_export_df = pd.DataFrame()
            pnl_distribution_export_df = pd.DataFrame()

            if "index" in equity_curve.columns:
                equity_curve = equity_curve.rename(columns={"index": "Date"})

            if enable_optimization:
                st.subheader("Optimization Result")
                st.write(
                    f"Best EMA Period: **{int(stats.get('_strategy').EMA_PERIOD)}**, "
                    f"Best RSI Period: **{int(stats.get('_strategy').RSI_PERIOD)}**"
                )
                heatmap_df = heatmap.reset_index()
                heatmap_df.columns = ["EMA_PERIOD", "RSI_PERIOD", "Return_%"]
                st.dataframe(heatmap_df, use_container_width=True)
                st.download_button(
                    label="Download Optimization CSV",
                    data=heatmap_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"optimization_results_{str(uuid.uuid4())[:8]}.csv",
                    mime="text/csv",
                )

            st.subheader("Performance Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Return %", f"{summary['Total Return %']:.2f}" if summary["Total Return %"] is not None else "-")
            c2.metric("Final Equity", f"{summary['Final Equity']:.2f}" if summary["Final Equity"] is not None else "-")
            c3.metric("Win Rate %", f"{summary['Win Rate %']:.2f}" if summary["Win Rate %"] is not None else "-")

            c4, c5, c6 = st.columns(3)
            c4.metric("Total Trades", str(summary["Total Trades"]))
            c5.metric("Max Drawdown %", f"{summary['Max Drawdown %']:.2f}" if summary["Max Drawdown %"] is not None else "-")
            c6.metric("Profit Factor", f"{summary['Profit Factor']:.2f}" if summary["Profit Factor"] is not None else "-")

            st.subheader("Equity Curve")
            if not equity_curve.empty and "Equity" in equity_curve.columns:
                st.line_chart(equity_curve.set_index("Date")["Equity"])
            else:
                st.info("No equity curve data available.")

            st.subheader("Drawdown Curve")
            if not equity_curve.empty and "Equity" in equity_curve.columns:
                dd_df = equity_curve[["Date", "Equity"]].copy()
                dd_df["PeakEquity"] = dd_df["Equity"].cummax()
                dd_df["DrawdownPct"] = ((dd_df["Equity"] / dd_df["PeakEquity"]) - 1.0) * 100.0
                st.line_chart(dd_df.set_index("Date")["DrawdownPct"])
                drawdown_export_df = dd_df.copy()
            else:
                st.info("No drawdown data available.")

            st.subheader("Trades")
            if not trades_df.empty:
                trades_show = trades_df.rename(
                    columns={
                        "EntryTime": "Entry Time",
                        "ExitTime": "Exit Time",
                        "EntryPrice": "Entry Price",
                        "ExitPrice": "Exit Price",
                        "PnL": "PnL",
                        "ReturnPct": "Return %",
                        "Size": "Direction Size",
                    }
                )
                st.dataframe(trades_show, use_container_width=True)
            else:
                st.info("No trades were generated for this period.")

            st.subheader("PnL Distribution")
            if not trades_df.empty and "PnL" in trades_df.columns:
                pnl_series = pd.to_numeric(trades_df["PnL"], errors="coerce").dropna()
                if not pnl_series.empty:
                    counts, bin_edges = np.histogram(pnl_series, bins=20)
                    pnl_hist_df = pd.DataFrame(
                        {
                            "PnL Bin": [
                                f"{bin_edges[i]:.2f} to {bin_edges[i + 1]:.2f}"
                                for i in range(len(bin_edges) - 1)
                            ],
                            "Trades": counts,
                        }
                    ).set_index("PnL Bin")
                    st.bar_chart(pnl_hist_df["Trades"])
                    pnl_distribution_export_df = pnl_hist_df.reset_index().copy()
                else:
                    st.info("PnL values are not available for distribution chart.")
            else:
                st.info("No PnL distribution data available.")

            st.subheader("Save Results")
            run_id = str(uuid.uuid4())[:8]
            summary_df = pd.DataFrame([{
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": effective_start_date,
                "end_date": effective_end_date,
                "ema_period": ema_period,
                "rsi_period": rsi_period,
                "initial_capital": initial_capital,
                **summary,
            }])

            st.download_button(
                label="Download Summary CSV",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name=f"backtest_summary_{run_id}.csv",
                mime="text/csv",
            )

            if not trades_df.empty:
                st.download_button(
                    label="Download Trades CSV",
                    data=trades_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"backtest_trades_{run_id}.csv",
                    mime="text/csv",
                )
            if not drawdown_export_df.empty:
                st.download_button(
                    label="Download Drawdown CSV",
                    data=drawdown_export_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"backtest_drawdown_{run_id}.csv",
                    mime="text/csv",
                )
            if not pnl_distribution_export_df.empty:
                st.download_button(
                    label="Download PnL Distribution CSV",
                    data=pnl_distribution_export_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"backtest_pnl_distribution_{run_id}.csv",
                    mime="text/csv",
                )

            if st.button("Save All CSVs to Separate Folder", use_container_width=True):
                save_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = Path("saved_results") / f"backtest_{save_id}"
                save_dir.mkdir(parents=True, exist_ok=True)

                summary_df.to_csv(save_dir / "summary.csv", index=False)
                if not trades_df.empty:
                    trades_df.to_csv(save_dir / "trades.csv", index=False)
                if not drawdown_export_df.empty:
                    drawdown_export_df.to_csv(save_dir / "drawdown_curve.csv", index=False)
                if not pnl_distribution_export_df.empty:
                    pnl_distribution_export_df.to_csv(save_dir / "pnl_distribution.csv", index=False)

                st.success(f"Files saved in folder: {save_dir}")

            st.subheader("Raw Historical Data")
            st.dataframe(raw_df.tail(200), use_container_width=True)

        except Exception as exc:
            st.error(f"Error: {exc}")


if __name__ == "__main__":
    main()
