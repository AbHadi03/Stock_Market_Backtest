import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import altair as alt

# --- Helper Functions ---
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

    low, high = -0.99, 10.0
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

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Equity RSI Trade (Stock-wise)", page_icon="📈")

# --- Login Check ---
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("Please login first from the main page")
    st.stop()

st.title("Equity RSI Strategy - Single Stock")
st.markdown("---")

st.markdown("""
### Strategy Description
- **Concept**: Relative Strength Index (RSI) Mean Reversion (Single Stock).
- **Entry**: Buy when RSI < Buy Level (Over sold area).
- **Exit**: Sell when Price > Entry Price * (1 + Target %).
- **Goal**: Analyze the performance of the RSI strategy on a specific stock over a custom timeframe.
""")


# --- Inputs ---
col1, col2, col3 = st.columns(3)

with col1:
    INVESTMENT_PER_TRADE = st.number_input(
        "Investment Per Trade", 
        value=10000.0, 
        step=1000.0,
        help="Amount of capital allocated to each trade."
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
        help="Transaction costs applied per trade."
    )

st.markdown("### Stock Selection")
col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    START_DATE = st.date_input("Start Date", value=datetime.today() - timedelta(days=365*3))
with col_s2:
    END_DATE = st.date_input("End Date", value=datetime.today())

with col_s3:
    # Top 10 Indian Stocks (Market Cap approximate)
    STOCK_LIST = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", 
        "ICICIBANK.NS", "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", 
        "BHARTIARTL.NS", "KOTAKBANK.NS"
    ]
    SELECTED_STOCK = st.selectbox("Stock Name", STOCK_LIST)

# --- Run Backtest ---
if st.button("Run Backtest", type="primary"):
    st.info(f"Fetching data for {SELECTED_STOCK} from {START_DATE} to {END_DATE}...")
    
    try:
        # Fetch Data
        df = yf.download(SELECTED_STOCK, start=START_DATE, end=END_DATE, progress=False)
        
        if df.empty:
            st.error("No data fetched from Yahoo Finance. Please check inputs.")
            st.stop()

        # Handle MultiIndex columns if present (common in new yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        
        # Ensure 'Date' is datetime
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None) # Remove timezone if present
        
        if len(df) < RSI_PERIOD:
            st.warning("Not enough data points for the selected RSI period.")
            st.stop()

        # Calculate RSI
        df["RSI"] = calculate_rsi(df["Close"], RSI_PERIOD)

        # Backtest Logic
        all_trades = []
        open_trades = []
        capital_timeline = []
        last_logged_capital = None
        
        # We need a global sense of time for capital, but here it's just one stock.
        # Capital deployed is simply Open Trades * Investment
        
        for index, row in df.iterrows():
            # EXIT Logic
            for trade in open_trades.copy():
                if row["High"] >= trade["target_price"]:
                    trade["exit_date"] = row["Date"]
                    trade["exit_price"] = trade["target_price"]
                    gross_pnl = trade["quantity"] * (trade["exit_price"] - trade["entry_price"])
                    trade["charges"] = CHARGES_PER_TRADE
                    trade["pnl"] = gross_pnl - CHARGES_PER_TRADE
                    trade["holding_days"] = (trade["exit_date"] - trade["entry_date"]).days
                    
                    all_trades.append(trade)
                    open_trades.remove(trade)
            
            # ENTRY Logic
            if pd.notna(row["RSI"]) and row["RSI"] < RSI_BUY_LEVEL:
                entry_price = row["Close"]
                quantity = INVESTMENT_PER_TRADE / entry_price
                trade = {
                    "Stock": SELECTED_STOCK,
                    "entry_date": row["Date"],
                    "entry_rsi": row["RSI"],
                    "entry_price": entry_price,
                    "quantity": quantity,
                    "target_price": entry_price * (1 + TARGET_PCT),
                    "exit_date": None,
                    "exit_price": None,
                    "pnl": None,
                    "charges": None,
                    "holding_days": None,
                }
                open_trades.append(trade)
            
            # Capital Tracking (Snapshot at end of day)
            current_capital = len(open_trades) * INVESTMENT_PER_TRADE
            if last_logged_capital is None or current_capital != last_logged_capital:
                capital_timeline.append({
                    "Date": row["Date"],
                    "capital_deployed": current_capital
                })
                last_logged_capital = current_capital

        # Build Results DataFrame
        trades_df = pd.DataFrame(all_trades)
        
        if trades_df.empty:
            st.warning("No trades generated for this period.")
        else:
            trades_df = trades_df.sort_values("exit_date")
            trades_df["cum_pnl"] = trades_df["pnl"].cumsum()
            
            # --- Metrics ---
            total_trades = len(trades_df)
            total_charges = total_trades * CHARGES_PER_TRADE
            net_pnl = trades_df["pnl"].sum()
            win_rate = (trades_df["pnl"] > 0).mean() * 100
            avg_holding_days = trades_df["holding_days"].mean()
            
            capital_df = pd.DataFrame(capital_timeline)
            max_capital = 0
            if not capital_df.empty:
                max_capital = capital_df["capital_deployed"].max()

            # XIRR
            cashflows = []
            for _, row in trades_df.iterrows():
                cashflows.append((row["entry_date"], -INVESTMENT_PER_TRADE))
                cashflows.append((row["exit_date"], row["quantity"] * row["exit_price"] - CHARGES_PER_TRADE))
            
            cashflow_df = pd.DataFrame(cashflows, columns=["date", "cashflow"]).sort_values("date")
            xirr = calculate_xirr(cashflow_df)

            # --- Dashboard ---
            st.subheader(f"Performance: {SELECTED_STOCK}")
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
            
            # --- Trade History with Cumulative PnL ---
            st.subheader("Trade History")
            # Formatting for display
            display_df = trades_df.copy()
            display_df['entry_date'] = display_df['entry_date'].dt.date
            display_df['exit_date'] = display_df['exit_date'].dt.date
            cols_to_show = ['Stock', 'entry_date', 'entry_rsi', 'entry_price', 'quantity', 'exit_date', 'exit_price', 'holding_days', 'charges', 'pnl', 'cum_pnl']
            st.dataframe(display_df[cols_to_show], width='stretch')

            # --- Analysis Charts ---
            
            # 1. Period Analysis
            st.subheader("Period Analysis")
            trades_df['exit_dt'] = pd.to_datetime(trades_df['exit_date'])
            trades_df['Year'] = trades_df['exit_dt'].dt.year
            trades_df['Month_Num'] = trades_df['exit_dt'].dt.month
            
            pivot_df = trades_df.groupby(['Year', 'Month_Num'])['pnl'].sum().reset_index()
            pivot_table = pivot_df.pivot(index='Year', columns='Month_Num', values='pnl').fillna(0)
            
            month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                         7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
            pivot_table.columns = [month_map.get(c, c) for c in pivot_table.columns]
            
            # Ensure all months
            for m in month_map.values():
                if m not in pivot_table.columns:
                    pivot_table[m] = 0.0
            
            # Reorder
            pivot_table = pivot_table[list(month_map.values())]
            pivot_table['Total'] = pivot_table.sum(axis=1)
            pivot_table = pivot_table.map(lambda x: f"{x:,.2f}")
            st.dataframe(pivot_table, width='stretch')

            # 2. Equity Curve
            st.subheader("Equity Curve")
            equity_data = trades_df[['exit_date', 'cum_pnl']].sort_values("exit_date").copy()
            equity_data = equity_data.rename(columns={'exit_date': 'Date', 'cum_pnl': 'Cumulative PnL'})
            st.line_chart(equity_data, x='Date', y='Cumulative PnL')

            # 3. Capital Deployment
            st.subheader("Capital Deployment")
            if not capital_df.empty:
                 st.area_chart(capital_df, x='Date', y='capital_deployed')
            
            # 4. Yearly PnL
            st.subheader("Yearly PnL")
            yearly_pnl = trades_df.groupby('Year')['pnl'].sum().reset_index()
            yearly_pnl['Year'] = yearly_pnl['Year'].astype(str)
            st.bar_chart(yearly_pnl, x='Year', y='pnl')

    except Exception as e:
        st.error(f"An error occurred during execution: {e}")
        # st.exception(e) # Uncomment for debugging
