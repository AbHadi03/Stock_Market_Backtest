import pandas as pd
import numpy as np
import os

# ==========================
# USER INPUTS
# ==========================
INVESTMENT_PER_TRADE = 10000
TARGET_PCT = 0.05
RSI_PERIOD = 14
RSI_BUY_LEVEL = 30
CHARGES_PER_TRADE = 25

CSV_FOLDER = "data_stocks"

# ==========================
# UTILITY FUNCTIONS
# ==========================
def clean_yfinance_csv(df):
    df = df.dropna(subset=["Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"]).copy()
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def clean_index_name(filename):
    return (
        filename.replace(".csv", "")
        .replace("Historical Data", "")
        .replace("Daily", "")
        .replace("OHCL", "")
        .replace("OHLC", "")
        .strip()
    )

def calculate_xirr(cashflow_df):
    dates = cashflow_df["date"]
    amounts = cashflow_df["cashflow"]

    def npv(rate):
        return sum(
            amt / ((1 + rate) ** ((d - dates.iloc[0]).days / 365))
            for amt, d in zip(amounts, dates)
        )

    low, high = -0.99, 3.0
    for _ in range(100):
        mid = (low + high) / 2
        val = npv(mid)
        if abs(val) < 1e-6:
            return mid
        if val > 0:
            low = mid
        else:
            high = mid
    return mid

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# ==========================
# STORAGE
# ==========================
all_trades = []
capital_timeline = []
open_positions_snapshot = []

GLOBAL_OPEN_TRADES = []

last_logged_capital = None   # ✅ NEW

# ==========================
# BACKTEST ENGINE
# ==========================
for file in os.listdir(CSV_FOLDER):

    if not file.lower().endswith(".csv"):
        continue

    print(f"\nProcessing: {file}")

    index_name = clean_index_name(file)
    df = pd.read_csv(os.path.join(CSV_FOLDER, file))
    df = clean_yfinance_csv(df)

    df["RSI"] = calculate_rsi(df["Close"], RSI_PERIOD)

    stock_open_trades = []

    for _, row in df.iterrows():

        # ---------- EXIT ----------
        for trade in stock_open_trades.copy():
            if row["High"] >= trade["target_price"]:
                trade["exit_date"] = row["Date"]
                trade["exit_price"] = trade["target_price"]

                gross_pnl = trade["quantity"] * (
                    trade["exit_price"] - trade["entry_price"]
                )

                trade["charges"] = CHARGES_PER_TRADE
                trade["pnl"] = gross_pnl - CHARGES_PER_TRADE
                trade["holding_days"] = (
                    trade["exit_date"] - trade["entry_date"]
                ).days

                all_trades.append(trade)

                stock_open_trades.remove(trade)
                GLOBAL_OPEN_TRADES.remove(trade)

        # ---------- ENTRY ----------
        if pd.notna(row["RSI"]) and row["RSI"] < RSI_BUY_LEVEL:
            entry_price = row["Close"]
            quantity = INVESTMENT_PER_TRADE / entry_price

            trade = {
                "index": index_name,
                "entry_date": row["Date"],
                "entry_price": entry_price,
                "quantity": quantity,
                "target_price": entry_price * (1 + TARGET_PCT),
                "exit_date": None,
                "exit_price": None,
                "pnl": None,
                "charges": None,
                "holding_days": None,
            }

            stock_open_trades.append(trade)
            GLOBAL_OPEN_TRADES.append(trade)

        # ---------- ✅ FIXED PORTFOLIO CAPITAL ----------
        current_capital = len(GLOBAL_OPEN_TRADES) * INVESTMENT_PER_TRADE

        if last_logged_capital is None or current_capital != last_logged_capital:
            capital_timeline.append({
                "Date": row["Date"],
                "capital_deployed": current_capital
            })
            last_logged_capital = current_capital


# ==========================
# STORE OPEN POSITIONS
# ==========================
for trade in GLOBAL_OPEN_TRADES:
    open_positions_snapshot.append({
        "index": trade["index"],
        "entry_date": trade["entry_date"],
        "entry_price": trade["entry_price"],
        "quantity": trade["quantity"],
        "target_price": trade["target_price"],
        "current_price": trade["entry_price"],
        "unrealized_pnl": trade["quantity"] * (trade["entry_price"] - trade["entry_price"]),
        "holding_days": 0
    })

# ==========================
# RESULTS
# ==========================
trades_df = pd.DataFrame(all_trades)

if trades_df.empty:
    print("\n❌ No trades generated.")
    exit()

trades_df.to_csv("trade_log.csv", index=False)
pd.DataFrame(capital_timeline).to_csv("capital_timeline.csv", index=False)

capital_df = (
    pd.DataFrame(capital_timeline)
    .groupby("Date")
    .last()
    .reset_index()
)

max_capital = capital_df["capital_deployed"].max()

trades_df = trades_df.sort_values("exit_date")
trades_df["cum_pnl"] = trades_df["pnl"].cumsum()

# ==========================
# XIRR
# ==========================
cashflows = []

for _, row in trades_df.iterrows():
    cashflows.append((row["entry_date"], -INVESTMENT_PER_TRADE))
    cashflows.append(
        (row["exit_date"], row["quantity"] * row["exit_price"] - CHARGES_PER_TRADE)
    )

cashflow_df = pd.DataFrame(cashflows, columns=["date", "cashflow"]).sort_values("date")
xirr = calculate_xirr(cashflow_df)

# ==========================
# SUMMARY
# ==========================
print("\n==============================")
print("PORTFOLIO SUMMARY (RSI STRATEGY)")
print("==============================")
print("Total Trades Executed :", len(trades_df))
print("Total Charges Paid (₹):", len(trades_df) * CHARGES_PER_TRADE)
print("Net P&L (₹)           :", round(trades_df["pnl"].sum(), 2))
print("Win Rate (%)          :", round((trades_df["pnl"] > 0).mean() * 100, 2))
print("Avg Holding Days      :", round(trades_df["holding_days"].mean(), 2))
print("Max Capital Deployed  :", f"{max_capital / 1e5:.2f} L")
print("Strategy XIRR (%)     :", round(xirr * 100, 2))
print("\nSaved: trade_log.csv")

# ==========================
# OPEN POSITIONS REPORT
# ==========================
open_pos_df = pd.DataFrame(open_positions_snapshot)

if not open_pos_df.empty:
    open_pos_df.to_csv("open_positions.csv", index=False)

    print("\n==============================")
    print("OPEN POSITIONS (END OF BACKTEST)")
    print("==============================")
    print("Total Open Positions :", len(open_pos_df))
    print("\nSaved: open_positions.csv")
else:
    print("\nNo open positions at end of backtest.")
