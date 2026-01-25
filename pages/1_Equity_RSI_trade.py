import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import altair as alt

# Helper functions adapted from main_rsi.py
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
    if cashflow_df.empty:
        return 0.0
    dates = cashflow_df["date"]
    amounts = cashflow_df["cashflow"]

    def npv(rate):
        return sum(
            amt / ((1 + rate) ** ((d - dates.iloc[0]).days / 365))
            for amt, d in zip(amounts, dates)
        )

    low, high = -0.99, 10.0 # Increased range just in case
    for _ in range(100):
        mid = (low + high) / 2
        try:
            val = npv(mid)
        except OverflowError:
             val = float('inf')
        
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

# Page Config
st.set_page_config(layout="wide", page_title="Equity RSI Trade", page_icon="📈")

# Login Check
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("Please login first from the main page")
    st.stop()

st.title("Equity RSI Strategy")
st.markdown("---")

# Inputs
col1, col2, col3 = st.columns(3)

with col1:
    INVESTMENT_PER_TRADE = st.number_input(
        "Investment Per Trade", 
        value=10000.0, 
        step=1000.0,
        help="Amount of capital allocated to each individual trade."
    )
    TARGET_PCT = st.number_input(
        "Target Percentage", 
        value=0.05, 
        step=0.01,
        format="%.2f",
        help="Target profit percentage (e.g., 0.05 for 5%)."
    )

with col2:
    RSI_PERIOD = st.number_input(
        "RSI Period", 
        value=14, 
        step=1,
        help="The lookback period for calculating RSI."
    )
    RSI_BUY_LEVEL = st.number_input(
        "RSI Buy Level", 
        value=30, 
        step=1,
        help="RSI level below which a buy signal is triggered."
    )

with col3:
    CHARGES_PER_TRADE = st.number_input(
        "Charges Per Trade", 
        value=25.0, 
        step=5.0,
        help="Transaction costs or brokerage charges applied per trade (both entry and exit combined or as specified)."
    )
    
    # Get data folders
    base_dir = os.getcwd()
    try:
        data_folders = [d for d in os.listdir(base_dir) if os.path.isdir(d) and d.startswith("data")]
    except:
        data_folders = []
    
    if not data_folders:
        st.warning("No folders starting with 'data' found.")
        data_folders = ["data_stocks"] # Fallback

    CSV_FOLDER = st.selectbox(
        "Data Folder", 
        data_folders,
        help="Select the folder containing the stock CSV data files."
    )

if st.button("Run Backtest", type="primary"):
    if not os.path.exists(CSV_FOLDER):
        st.error(f"Folder '{CSV_FOLDER}' not found.")
        st.stop()
        
    st.info(f"Running backtest on '{CSV_FOLDER}'...")
    
    all_trades = []
    capital_timeline = []
    GLOBAL_OPEN_TRADES = []
    last_logged_capital = None
    
    progress_bar = st.progress(0)
    files = [f for f in os.listdir(CSV_FOLDER) if f.lower().endswith(".csv")]
    total_files = len(files)
    
    for i, file in enumerate(files):
        index_name = clean_index_name(file)
        try:
            df = pd.read_csv(os.path.join(CSV_FOLDER, file))
            df = clean_yfinance_csv(df)
            
            if len(df) < RSI_PERIOD:
                continue

            df["RSI"] = calculate_rsi(df["Close"], RSI_PERIOD)
            
            stock_open_trades = []
            
            for _, row in df.iterrows():
                # EXIT
                for trade in stock_open_trades.copy():
                    if row["High"] >= trade["target_price"]:
                        trade["exit_date"] = row["Date"]
                        trade["exit_price"] = trade["target_price"]
                        gross_pnl = trade["quantity"] * (trade["exit_price"] - trade["entry_price"])
                        trade["charges"] = CHARGES_PER_TRADE
                        trade["pnl"] = gross_pnl - CHARGES_PER_TRADE
                        trade["holding_days"] = (trade["exit_date"] - trade["entry_date"]).days
                        
                        all_trades.append(trade)
                        stock_open_trades.remove(trade)
                        GLOBAL_OPEN_TRADES.remove(trade)
                
                # ENTRY
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
                
                # CAPITAL TRACKING
                current_capital = len(GLOBAL_OPEN_TRADES) * INVESTMENT_PER_TRADE
                if last_logged_capital is None or current_capital != last_logged_capital:
                    capital_timeline.append({
                        "Date": row["Date"],
                        "capital_deployed": current_capital
                    })
                    last_logged_capital = current_capital
                    
        except Exception as e:
            # print(f"Error processing {file}: {e}")
            pass
        
        progress_bar.progress((i + 1) / total_files)
        
    trades_df = pd.DataFrame(all_trades)
    
    if trades_df.empty:
        st.warning("No trades generated.")
    else:
        trades_df = trades_df.sort_values("exit_date")
        trades_df["cum_pnl"] = trades_df["pnl"].cumsum()
        
        # --- Metrics Calculation ---
        total_trades = len(trades_df)
        total_charges = total_trades * CHARGES_PER_TRADE
        net_pnl = trades_df["pnl"].sum()
        win_rate = (trades_df["pnl"] > 0).mean() * 100
        avg_holding_days = trades_df["holding_days"].mean()
        
        capital_df = pd.DataFrame(capital_timeline)
        if not capital_df.empty:
            capital_df["Date"] = pd.to_datetime(capital_df["Date"])
            capital_df = capital_df.groupby("Date").last().reset_index()
            capital_df = capital_df.sort_values("Date")
            max_capital = capital_df["capital_deployed"].max()
        else:
            max_capital = 0

        # XIRR
        cashflows = []
        for _, row in trades_df.iterrows():
            cashflows.append((row["entry_date"], -INVESTMENT_PER_TRADE))
            cashflows.append((row["exit_date"], row["quantity"] * row["exit_price"] - CHARGES_PER_TRADE))
        
        cashflow_df = pd.DataFrame(cashflows, columns=["date", "cashflow"]).sort_values("date")
        xirr = calculate_xirr(cashflow_df)

        # --- Dashboard ---
        st.subheader("Strategy Performance Dashboard")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Trades", total_trades)
        m2.metric("Total Charges", f"₹{total_charges:,.2f}")
        m3.metric("Net P&L", f"₹{net_pnl:,.2f}", delta_color="normal" if net_pnl >= 0 else "inverse")
        m4.metric("Win Rate", f"{win_rate:.2f}%")
        
        m5, m6, m7 = st.columns(3)
        m5.metric("Avg Holding Days", f"{avg_holding_days:.2f} Days")
        m6.metric("Max Capital Deployed", f"₹{max_capital:,.2f}")
        m7.metric("Strategy XIRR", f"{xirr * 100:.2f}%")
        
        st.markdown("---")
        
        # --- Analysis Charts ---
        
        # 1. Monthly & Yearly PnL Summary
        st.subheader("Period Analysis")
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        trades_df['Year'] = trades_df['exit_date'].dt.year
        trades_df['Month'] = trades_df['exit_date'].dt.strftime('%b') # Jan, Feb...
        trades_df['Month_Num'] = trades_df['exit_date'].dt.month

        # Group by Year and Month
        pivot_df = trades_df.groupby(['Year', 'Month', 'Month_Num'])['pnl'].sum().reset_index()
        
        # Pivot: Index=Year, Columns=Month_Num (to sort), Values=pnl
        pivot_table = pivot_df.pivot(index='Year', columns='Month_Num', values='pnl').fillna(0)
        
        # Rename columns to Month names
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                     7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        pivot_table.columns = [month_map.get(c, c) for c in pivot_table.columns]
        
        # Ensure all 12 months are present
        for m in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
             if m not in pivot_table.columns:
                 pivot_table[m] = 0.0
        
        # Reorder columns
        pivot_table = pivot_table[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
        
        # Add Total Column
        pivot_table['Total'] = pivot_table.sum(axis=1)
        
        # Format numbers
        pivot_table = pivot_table.map(lambda x: f"{x:,.2f}")

        st.dataframe(pivot_table, width="stretch")

        # 2. Equity Curve
        st.subheader("Equity Curve")
        # Creating a daily equity curve
        equity_data = trades_df[['exit_date', 'cum_pnl']].copy()
        equity_data = equity_data.rename(columns={'exit_date': 'Date', 'cum_pnl': 'Cumulative PnL'})
        # Resample to ensure daily continuity if needed, strictly speaking this is realized PnL curve
        st.line_chart(equity_data, x='Date', y='Cumulative PnL')

        # 3. Capital Deployment Curve
        st.subheader("Capital Deployment Curve")
        if not capital_df.empty:
             st.area_chart(capital_df, x='Date', y='capital_deployed')
        else:
            st.info("No capital deployment data available.")

        # 4. Yearly PnL Bar Graph
        st.subheader("Yearly PnL Visualization")
        yearly_pnl = trades_df.groupby('Year')['pnl'].sum().reset_index()
        yearly_pnl['Year'] = yearly_pnl['Year'].astype(str)
        st.bar_chart(yearly_pnl, x='Year', y='pnl')

        # Download Data
        st.download_button(
            label="Download Trade Log",
            data=trades_df.to_csv(index=False).encode('utf-8'),
            file_name='trade_log.csv',
            mime='text/csv'
        )
