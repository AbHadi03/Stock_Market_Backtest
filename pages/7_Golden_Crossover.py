import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

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

def calculate_sma(series, period):
    """Calculate Simple Moving Average"""
    return series.rolling(window=period).mean()

def resample_to_timeframe(df, timeframe):
    """Resample daily data to Monthly or Weekly"""
    df = df.set_index('Date')
    
    if timeframe == "Monthly":
        resampled = df.resample('ME').agg({ # Using 'ME' for Month End to avoid warnings
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
    elif timeframe == "Weekly":
        resampled = df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
    else:  # Daily
        return df.reset_index()
    
    return resampled.reset_index()

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Golden Crossover Strategy", page_icon="📈")

# --- Login Check ---
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("Please login first from the main page")
    st.stop()

st.title("📈 Golden Crossover Strategy (SMA Crossover)")
st.markdown("---")

with st.expander("📖 Strategy Description & Rules"):
    st.markdown("""
    ### Golden Crossover Strategy - Simple Moving Average Crossover System
    
    This strategy is based on the crossover of two Simple Moving Averages (SMA) to identify trend changes.
    
    #### 🎯 Entry Rules
    1. **Buy Signal**: When the **Fast SMA** crosses above the **Slow SMA**.
    2. **Entry Price**: Buy at the closing price when the crossover condition is met.
    
    #### 🚪 Exit Rules
    The user can choose between two exit strategies:
    1. **Take Profit (TP) / Stop Loss (SL)**: 
       - Exit when price reaches the target profit percentage.
       - Optional: Exit when price falls below the stop loss percentage.
    2. **SMA Cross Down**:
       - Exit when the **Fast SMA** crosses below the **Slow SMA**.
       - *Note: Stop Loss is disabled in this mode.*
    
    #### ⏰ Timeframe Options
    - **Daily, Weekly, Monthly** options are available. SMAs are calculated based on the selected timeframe.
    
    #### 💡 T1 Day Selling Concept
    - Ensuring a minimum holding period to simulate realistic trading restrictions where applicable.
    """)

# --- Inputs ---
if 'p7_date_preset' not in st.session_state:
    st.session_state.p7_date_preset = "2 Years"

def update_dates():
    preset = st.session_state.p7_preset_radio
    end = datetime.today().date()
    if preset == "7 Days":
        start = end - timedelta(days=7)
    elif preset == "15 Days":
        start = end - timedelta(days=15)
    elif preset == "30 Days":
        start = end - timedelta(days=30)
    elif preset == "3 Months":
        start = end - timedelta(days=90)
    elif preset == "6 Months":
        start = end - timedelta(days=180)
    elif preset == "1 Year":
        start = end - timedelta(days=365)
    elif preset == "2 Years":
        start = end - timedelta(days=730)
    elif preset == "5 Years":
        start = end - timedelta(days=1825)
    else: # Custom
        return
        
    st.session_state.p7_start_date = start
    st.session_state.p7_end_date = end

# Initialize defaults if needed
if 'p7_start_date' not in st.session_state:
    st.session_state.p7_start_date = datetime.today().date() - timedelta(days=730)
if 'p7_end_date' not in st.session_state:
    st.session_state.p7_end_date = datetime.today().date()

# UI Layout
st.radio(
    "Quick Select Range", 
    ["7 Days", "15 Days", "30 Days", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"], 
    horizontal=True, 
    key="p7_preset_radio",
    on_change=update_dates,
    index=6
)

# Core Inputs
col1, col2, col3, col4 = st.columns(4)
with col1:
    SELECTED_STOCK = st.text_input("Stock Name", value="RELIANCE").upper().strip()
with col2:
    TIMEFRAME = st.selectbox("Timeframe", ["Daily", "Weekly", "Monthly"], index=0)
with col3:
    FAST_SMA_P = st.text_input("Fast SMA", value="9")
    try:
        FAST_SMA_P = int(FAST_SMA_P)
    except:
        st.error("Fast SMA must be an integer")
        st.stop()
with col4:
    SLOW_SMA_P = st.text_input("Slow SMA", value="20")
    try:
        SLOW_SMA_P = int(SLOW_SMA_P)
    except:
        st.error("Slow SMA must be an integer")
        st.stop()

col5, col6, col7, col8 = st.columns(4)
with col5:
    EXIT_STRATEGY = st.radio("Exit Strategy", ["Take Profit %", "Fast SMA cross down Slow SMA"])
with col6:
    INVESTMENT = st.number_input("Investment (INR)", value=10000.0, step=1000.0)
with col7:
    CHARGES_PER_TRADE = st.number_input("Charges Per Trade (Rs)", value=25.0, step=5.0)
with col8:
    ST_DATE = st.date_input("Start Date", key="p7_start_date")
    ED_DATE = st.date_input("End Date", key="p7_end_date", max_value=datetime.today().date())

# Conditional SL/TP Inputs
row3_col1, row3_col2, row3_col3 = st.columns(3)
if EXIT_STRATEGY == "Take Profit %":
    with row3_col1:
        TAKE_PROFIT_PCT = st.number_input("Take Profit %", value=10.0, step=0.5)
    with row3_col2:
        USE_STOP_LOSS = st.checkbox("Enable Stop Loss", value=True)
        STOP_LOSS_PCT = st.number_input("Stop Loss %", value=5.0, step=0.5, disabled=not USE_STOP_LOSS)
else:
    TAKE_PROFIT_PCT = 9999.0 # Effectively no TP
    USE_STOP_LOSS = False
    STOP_LOSS_PCT = 100.0 # Effectively no SL

# --- Run Backtest ---
if st.button("Run Backtest", type="primary"):
    # Auto-append .NS if not present
    if not SELECTED_STOCK.endswith(".NS") and "." not in SELECTED_STOCK:
        SYMBOL = SELECTED_STOCK + ".NS"
    else:
        SYMBOL = SELECTED_STOCK
    
    st.info(f"Fetching data for {SYMBOL}...")
    
    try:
        # Buffer for SMA calculation
        max_sma = max(FAST_SMA_P, SLOW_SMA_P)
        if TIMEFRAME == "Weekly":
            buffer_days = max_sma * 7 + 100
        elif TIMEFRAME == "Monthly":
            buffer_days = max_sma * 31 + 100
        else:
            buffer_days = max_sma * 1.5 + 50
        
        fetch_start = ST_DATE - timedelta(days=int(buffer_days))
        df = yf.download(SYMBOL, start=fetch_start, end=ED_DATE, progress=False)
        
        if df.empty:
            st.error("No data fetched. Check symbol.")
            st.stop()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        # Resample
        df_resampled = resample_to_timeframe(df.copy(), TIMEFRAME)
        
        # Calculate SMAs
        df_resampled['Fast_SMA'] = calculate_sma(df_resampled['Close'], FAST_SMA_P)
        df_resampled['Slow_SMA'] = calculate_sma(df_resampled['Close'], SLOW_SMA_P)
        
        # Filter to selected range
        df_resampled = df_resampled[df_resampled['Date'] >= pd.to_datetime(ST_DATE)].reset_index(drop=True)
        
        if df_resampled.empty or df_resampled['Slow_SMA'].isna().all():
            st.error("Insufficient data for SMA calculation in selected range.")
            st.stop()

        # Backtest Logic
        results = []
        position = None
        total_pnl = 0
        
        for i in range(1, len(df_resampled)):
            row = df_resampled.iloc[i]
            prev_row = df_resampled.iloc[i-1]
            
            if pd.isna(row['Slow_SMA']) or pd.isna(prev_row['Slow_SMA']):
                continue
                
            # Handle Position
            if position:
                exit_signal = False
                exit_price = 0
                exit_action = ""
                
                # Exit Strategy 1: TP/SL
                if EXIT_STRATEGY == "Take Profit %":
                    tp_price = position['entry_price'] * (1 + TAKE_PROFIT_PCT / 100)
                    sl_price = position['entry_price'] * (1 - STOP_LOSS_PCT / 100)
                    
                    if row['High'] >= tp_price:
                        exit_signal = True
                        exit_price = max(row['Open'], tp_price)
                        exit_action = "TAKE PROFIT"
                    elif USE_STOP_LOSS and row['Low'] <= sl_price:
                        exit_signal = True
                        exit_price = min(row['Open'], sl_price)
                        exit_action = "STOP LOSS"
                
                # Exit Strategy 2: Cross Down
                else:
                    if row['Fast_SMA'] < row['Slow_SMA'] and prev_row['Fast_SMA'] >= prev_row['Slow_SMA']:
                        exit_signal = True
                        exit_price = row['Close']
                        exit_action = "SMA CROSS DOWN"
                
                if exit_signal:
                    # T1 Selling Concept: Can't sell on same day of entry (in Daily timeframe)
                    if TIMEFRAME == "Daily" and row['Date'].date() <= position['entry_date'].date():
                        continue
                        
                    pnl = float(round((exit_price - position['entry_price']) * position['quantity'] - CHARGES_PER_TRADE, 2))
                    total_pnl = float(round(total_pnl + pnl, 2))
                    
                    duration = (row['Date'] - position['entry_date']).days
                    results.append({
                        'Symbol': SYMBOL,
                        'Entry Date': position['entry_date'],
                        'Entry Price': position['entry_price'],
                        'Quantity': position['quantity'],
                        'Invested': INVESTMENT,
                        'Exit Date': row['Date'],
                        'Exit Price': float(round(exit_price, 2)),
                        'Action': exit_action,
                        'PnL': pnl,
                        'Cumilative PnL': total_pnl,
                        'Duration': f"{duration} days"
                    })
                    position = None
            
            # Entry Signal: Fast SMA crossover Slow SMA
            if not position:
                if row['Fast_SMA'] > row['Slow_SMA'] and prev_row['Fast_SMA'] <= prev_row['Slow_SMA']:
                    position = {
                        'entry_date': row['Date'],
                        'entry_price': float(round(row['Close'], 2)),
                        'quantity': float(round(INVESTMENT / row['Close'], 2))
                    }
        
        # Store results
        st.session_state.p7_results = {
            'results_df': pd.DataFrame(results),
            'last_price_data': df_resampled,
            'position': position,
            'symbol': SYMBOL,
            'investment': INVESTMENT,
            'charges': CHARGES_PER_TRADE,
            'timeframe': TIMEFRAME,
            'start_date': ST_DATE,
            'end_date': ED_DATE
        }

    except Exception as e:
        st.error(f"Error: {e}")

# --- Display Results ---
if 'p7_results' in st.session_state:
    data = st.session_state.p7_results
    results_df = data['results_df']
    df_resampled = data['last_price_data']
    position = data['position']
    symbol = data['symbol']

    # Retrieve inputs used for this run
    run_start_date = data.get('start_date', ST_DATE)
    run_end_date = data.get('end_date', ED_DATE)
    run_investment = data.get('investment', INVESTMENT)
    run_charges = data.get('charges', CHARGES_PER_TRADE)
    run_timeframe = data.get('timeframe', TIMEFRAME)
    
    st.header(f"Performance: {symbol}")
    
    # Summary Metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    total_trades = len(results_df)
    win_count = len(results_df[results_df['PnL'] > 0]) if not results_df.empty else 0
    loss_count = total_trades - win_count
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    net_profit = float(round(results_df['PnL'].sum(), 2)) if not results_df.empty else 0.0
    roi = float(round((net_profit / run_investment) * 100, 2)) if run_investment > 0 else 0.0
    
    # XIRR
    xirr = 0.0
    if not results_df.empty:
        cashflows = []
        for _, row_res in results_df.iterrows():
            cashflows.append((row_res["Entry Date"], -row_res["Invested"]))
            # Exit Value = Qty * Exit Price - Charges (assuming charges are per total trade entry+exit as per reference)
            exit_val = (row_res["Quantity"] * row_res["Exit Price"]) - run_charges
            cashflows.append((row_res["Exit Date"], exit_val))
        
        cashflow_df = pd.DataFrame(cashflows, columns=["date", "cashflow"]).sort_values("date")
        xirr = float(round(calculate_xirr(cashflow_df), 4))

    col_m1.metric("Total PnL", f"₹{net_profit:,.2f}", delta_color="normal" if net_profit >= 0 else "inverse")
    col_m2.metric("Total Trades", total_trades)
    col_m3.metric("XIRR", f"{xirr * 100:.2f}%")
    col_m4.metric("ROI", f"{roi:.1f}%")
    
    col_m5, col_m6, col_m7 = st.columns(3)
    col_m5.metric("Win Rate", f"{win_rate:.1f}%")
    col_m6.metric("Wins / Losses", f"{win_count} / {loss_count}")
    
    # Links
    tv_symbol = symbol.replace(".NS", "")
    tv_url = f"https://www.tradingview.com/chart/?symbol=NSE%3A{tv_symbol}"
    st.link_button("View Chart on TradingView", tv_url)

    # --- Download Analytics ---
    download_data = {
        "Stock Name": [symbol],
        "Strategy Name": ["7 Golden SMA Strategy"],
        "Timeframe": [run_timeframe],
        "Start Date": [run_start_date],
        "End Date": [run_end_date],
        "Investment": [run_investment],
        "Exit Strategy": [EXIT_STRATEGY],
        "Total PnL": [net_profit],
        "Total Trades": [total_trades],
        "Win Rate": [f"{win_rate:.2f}%"],
        "XIRR": [f"{xirr * 100:.2f}%"],
        "ROI": [f"{roi:.2f}%"]
    }
    st.download_button(
        label="Download Strategy Analytics as CSV",
        data=pd.DataFrame(download_data).to_csv(index=False).encode('utf-8'),
        file_name=f"{symbol}_sma_analytics.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # --- Open Trades Table ---
    if position:
        st.subheader("📌 Open Trade")
        current_price = df_resampled.iloc[-1]['Close']
        open_pnl = float(round((current_price - position['entry_price']) * position['quantity'], 2))
        open_pnl_pct = float(round((current_price / position['entry_price'] - 1) * 100, 2))
        
        open_trade_data = {
            "Symbol": [symbol],
            "Entry Date": [position['entry_date'].date()],
            "Entry Price": [position['entry_price']],
            "Qty": [position['quantity']],
            "Current Price": [float(round(current_price, 2))],
            "PnL": [open_pnl],
            "PnL %": [f"{open_pnl_pct:.2f}%"]
        }
        st.dataframe(pd.DataFrame(open_trade_data), width='stretch', hide_index=True)
        st.markdown("---")

    if not results_df.empty:
        # Trade History
        st.subheader("Trade History")
        display_df = results_df.copy()
        display_df['Entry Date'] = display_df['Entry Date'].dt.date
        display_df['Exit Date'] = display_df['Exit Date'].dt.date
        st.dataframe(display_df, width='stretch')
        
        st.markdown("---")
        
        # --- Market Analysis Table ---
        st.subheader("Market Analysis")
        
        # 1. Price Change
        start_price = df_resampled.iloc[0]['Close']
        end_price = df_resampled.iloc[-1]['Close']
        price_change_pct = (end_price / start_price - 1) * 100
        
        # 2. Perfect Trade (Min to Max)
        min_row = df_resampled.loc[df_resampled['Low'].idxmin()]
        global_min = min_row['Low']
        min_date = min_row['Date']
        
        df_after_min = df_resampled[df_resampled['Date'] > min_date]
        
        if not df_after_min.empty:
            max_row = df_after_min.loc[df_after_min['High'].idxmax()]
            global_max_after_min = max_row['High']
            max_date = max_row['Date']
            
            quantity_perfect = run_investment / global_min
            pnl_perfect = float(round(quantity_perfect * global_max_after_min - run_investment - run_charges, 2))
            roi_perfect = float(round((pnl_perfect / run_investment) * 100, 2))
        else:
            global_max_after_min = global_min
            max_date = min_date
            pnl_perfect = 0.0
            roi_perfect = 0.0

        # 3. Buy & Hold
        quantity_bnh = run_investment / start_price
        pnl_bnh = float(round(quantity_bnh * end_price - run_investment - run_charges, 2))
        roi_bnh = float(round((pnl_bnh / run_investment) * 100, 2))
        
        analysis_data = {
            "Metric": [
                "Lowest Price (Buy Point)", 
                "Highest Price (Sell Point)", 
                "Perfect Trade (Min -> Max)", 
                "Buy & Hold (Start -> End)",
                "Price Change (Start -> End)"
            ],
            "Value": [
                f"₹{global_min:,.2f}",
                f"₹{global_max_after_min:,.2f}",
                f"PnL: ₹{pnl_perfect:,.2f}",
                f"PnL: ₹{pnl_bnh:,.2f}",
                f"{price_change_pct:.2f}%"
            ],
            "Date/ROI": [
                min_date.strftime('%Y-%m-%d'),
                max_date.strftime('%Y-%m-%d'),
                f"ROI: {roi_perfect:.2f}%",
                f"ROI: {roi_bnh:.2f}%",
                f"{df_resampled.iloc[0]['Date'].date()} to {df_resampled.iloc[-1]['Date'].date()}"
            ]
        }
        st.table(pd.DataFrame(analysis_data))
        
        st.markdown("---")
        
        # Charts
        st.subheader("Equity Curve")
        equity_df = results_df[['Exit Date', 'Cumilative PnL']].copy()
        equity_df = equity_df.rename(columns={'Exit Date': 'Date'})
        st.line_chart(equity_df, x='Date', y='Cumilative PnL')
