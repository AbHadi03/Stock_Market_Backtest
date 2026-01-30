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
st.set_page_config(layout="wide", page_title="Vishal Malkhan GFS Strategy", page_icon="📈")

# --- Login Check ---
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("Please login first from the main page")
    st.stop()

st.title("📈 Vishal Malkhan GFS Strategy")
st.markdown("---")

with st.expander("📖 Strategy Description: Vishal Malkhan GFS"):
    st.markdown("""
    - **GFS Concept**: Go-For-Sona (Trend Following with Multi-Timeframe RSI).
    - **Entry Conditions**:
        - **Monthly RSI** > Monthly Threshold (Higher timeframe trend).
        - **Weekly RSI** > Weekly Threshold (Medium timeframe trend).
        - **Daily RSI** < Daily Threshold (Daily dip in a bullish trend).
    - **Constraints**: 
        - Only **one trade** can be active at a time.
        - **Minimum Holding (T+2)**: Cannot sell on the very next day.
    - **Exit**: Sell when Price reaches the **Target %**.
    """)

# --- Inputs ---
col1, col2, col3 = st.columns(3)

with col1:
    INVEST_PER_TRADE = st.number_input("Investment Per Trade (INR)", value=10000.0, step=1000.0)
    TARGET_PCT = st.number_input("Target Percentage", value=0.05, step=0.01, format="%.2f", help="e.g. 0.05 for 5%")
    RSI_PERIOD = st.number_input("RSI Period", value=14, step=1)

with col2:
    MONTHLY_RSI_THRESH = st.number_input("Monthly RSI Threshold (Greater Than)", value=60, step=1)
    WEEKLY_RSI_THRESH = st.number_input("Weekly RSI Threshold (Greater Than)", value=60, step=1)
    DAILY_RSI_THRESH = st.number_input("Daily RSI Threshold (Less Than)", value=40, step=1)

with col3:
    CHARGES_PER_TRADE = st.number_input("Charges Per Trade (Rs)", value=25.0, step=5.0)
    SELECTED_STOCK = st.text_input("Stock Name", value="RELIANCE")

col_d1, col_d2 = st.columns(2)
with col_d1:
    START_DATE = st.date_input("Start Date", value=datetime.today() - timedelta(days=365*2))
with col_d2:
    END_DATE = st.date_input("End Date", value=datetime.today())

# --- Run Backtest ---
if st.button("Run Backtest", type="primary"):
    SELECTED_STOCK = SELECTED_STOCK.upper().strip()
    if not SELECTED_STOCK.endswith(".NS"):
        SELECTED_STOCK += ".NS"
    
    st.info(f"Computing GFS Strategy for {SELECTED_STOCK}...")
    
    try:
        # Fetch Data (with buffer for Monthly RSI)
        fetch_start = START_DATE - timedelta(days=365*2) # 2 years buffer for Monthly RSI stability
        df = yf.download(SELECTED_STOCK, start=fetch_start, end=END_DATE, progress=False)
        
        if df.empty:
            st.error("No data fetched. Check symbol/dates.")
            st.stop()
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Robustly ensure 'Date' is a column
        if 'Date' not in df.columns:
            df = df.reset_index()
        
        # Rename 'index' to 'Date' if necessary (happens with some yfinance versions)
        if 'Date' not in df.columns and 'index' in df.columns:
            df = df.rename(columns={'index': 'Date'})
            
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        # --- Multi-Timeframe RSI Calculation ---
        # 1. Daily RSI
        df['RSI_Daily'] = calculate_rsi(df['Close'], RSI_PERIOD)
        
        # 2. Weekly RSI - Use set_index for robust resampling
        df_w = df.set_index('Date').resample('W-MON').last().dropna().copy()
        df_w['RSI_Weekly'] = calculate_rsi(df_w['Close'], RSI_PERIOD)
        df_w = df_w.reset_index()
        
        # Map Weekly RSI back to main dataframe
        weekly_rsi_map = df_w.set_index('Date')['RSI_Weekly']
        df['RSI_Weekly'] = df['Date'].map(weekly_rsi_map).ffill()
        
        # 3. Monthly RSI
        df_m = df.set_index('Date').resample('ME').last().dropna().copy()
        df_m['RSI_Monthly'] = calculate_rsi(df_m['Close'], RSI_PERIOD)
        df_m = df_m.reset_index()
        
        monthly_rsi_map = df_m.set_index('Date')['RSI_Monthly']
        df['RSI_Monthly'] = df['Date'].map(monthly_rsi_map).ffill()
        
        # Filter to Start Date
        df_backtest = df[df['Date'] >= pd.to_datetime(START_DATE)].copy().reset_index(drop=True)
        
        if df_backtest.empty:
            st.error("No data points available after the start date.")
            st.stop()
            
        # --- Backtest Engine ---
        all_trades = []
        position = None # Single trade constraint
        total_pnl = 0
        
        for i, row in df_backtest.iterrows():
            current_date = row['Date']
            open_price = row['Open']
            high_price = row['High']
            close_price = row['Close']
            
            # EXIT Logic
            if position:
                duration_days = (current_date - position['entry_date']).days
                
                # Check T+2 constraint
                if duration_days >= 2:
                    exit_price = None
                    # Gap check at Open
                    if open_price >= position['target_price']:
                        exit_price = open_price
                    # Intra-day check
                    elif high_price >= position['target_price']:
                        exit_price = position['target_price']
                        
                    if exit_price:
                        # Close Trade
                        pnl = float(round((exit_price - position['entry_price']) * position['quantity'] - CHARGES_PER_TRADE, 2))
                        total_pnl = float(round(total_pnl + pnl, 2))
                        
                        all_trades.append({
                            'Entry Date': position['entry_date'],
                            'Exit Date': current_date,
                            'Entry Price': position['entry_price'],
                            'Exit Price': float(round(exit_price, 2)),
                            'Quantity': position['quantity'],
                            'PnL': pnl,
                            'Duration': duration_days
                        })
                        position = None
            
            # ENTRY Logic (If no active position)
            if not position:
                m_rsi = row['RSI_Monthly']
                w_rsi = row['RSI_Weekly']
                d_rsi = row['RSI_Daily']
                
                if pd.notna(m_rsi) and pd.notna(w_rsi) and pd.notna(d_rsi):
                    if m_rsi > MONTHLY_RSI_THRESH and w_rsi > WEEKLY_RSI_THRESH and d_rsi < DAILY_RSI_THRESH:
                        entry_price = float(round(close_price, 2))
                        qty = float(round(INVEST_PER_TRADE / entry_price, 2))
                        position = {
                            'entry_date': current_date,
                            'entry_price': entry_price,
                            'quantity': qty,
                            'target_price': float(round(entry_price * (1 + TARGET_PCT), 2))
                        }
        
        # Results Display
        if not all_trades and not position:
            st.warning("No trades generated with these conditions.")
        else:
            trades_df = pd.DataFrame(all_trades)
            
            # Metrics
            st.subheader("Performance Summary")
            m1, m2, m3, m4 = st.columns(4)
            total_trades_count = len(all_trades)
            net_pnl = total_pnl
            
            m1.metric("Total Trades", total_trades_count)
            m2.metric("Net P&L", f"₹{net_pnl:,.2f}")
            m3.metric("Profit per Trade", f"₹{float(round(net_pnl/total_trades_count, 2)):,.2f}" if total_trades_count > 0 else "₹0.00")
            m4.metric("Active Position", "Yes" if position else "No")
            
            # --- Open Trade Section ---
            st.subheader("Open Trade Status")
            if position:
                with st.container(border=True):
                    st.write(f"🟢 **Position Open**: Bought on {position['entry_date'].date()} at ₹{position['entry_price']}")
                    st.write(f"🎯 **Target Price**: ₹{position['target_price']}")
                    
                    # Current unrealized PnL
                    latest_close = df_backtest.iloc[-1]['Close']
                    unrealized = (latest_close - position['entry_price']) * position['quantity']
                    st.write(f"💰 **Unrealized PnL**: ₹{float(round(unrealized, 2)):,.2f} (Current Price: ₹{latest_close:,.2f})")
            else:
                st.info("No active trade at the moment. Strategy is searching for GFS conditions.")
            
            st.markdown("---")
            
            # --- Trade History ---
            st.subheader("Trade History")
            if not trades_df.empty:
                trades_df['Cumulative PnL'] = trades_df['PnL'].cumsum()
                # Format dates
                display_df = trades_df.copy()
                display_df['Entry Date'] = display_df['Entry Date'].dt.date
                display_df['Exit Date'] = display_df['Exit Date'].dt.date
                st.dataframe(display_df, width='stretch')
                
                # Equity Curve
                st.subheader("Equity Curve")
                equity_data = trades_df[['Exit Date', 'Cumulative PnL']].copy()
                st.line_chart(equity_data, x='Exit Date', y='Cumulative PnL')
            
    except Exception as e:
        st.error(f"Error: {e}")
