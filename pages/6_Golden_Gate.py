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
        resampled = df.resample('M').agg({
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
st.set_page_config(layout="wide", page_title="Golden Gate Strategy", page_icon="🌉")

# --- Login Check ---
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("Please login first from the main page")
    st.stop()

st.title("🌉 Golden Gate Strategy (SMA Alignment)")
st.markdown("---")

with st.expander("📖 Strategy Description & Rules"):
    st.markdown("""
    ### Golden Gate Strategy - SMA Alignment System
    
    This strategy is based on the alignment of Simple Moving Averages (SMA) to identify strong trending opportunities.
    
    #### 📊 Core Concept
    The strategy looks for a specific order of price and moving averages, indicating a strong uptrend:
    - **Price > 10 SMA > 20 SMA > 50 SMA > 100 SMA > 200 SMA**
    
    #### 🎯 Entry Rules
    1. **Buy Signal**: When the price and all SMAs are in perfect ascending order
    2. **Confirmation**: All conditions must be met simultaneously
    3. **Entry Price**: Buy at the closing price when conditions are met
    
    #### 🚪 Exit Rules
    1. **Take Profit (TP)**: Exit when price reaches the target profit percentage
    2. **Stop Loss (SL)**: Exit when price falls below the stop loss percentage
    3. **Re-Entry**: After exit, wait for the same alignment condition to occur again
    
    #### ⏰ Timeframe Options
    - **Daily**: SMAs calculated on daily candles
    - **Weekly**: SMAs calculated on weekly candles (resampled from daily data)
    - **Monthly**: SMAs calculated on monthly candles (resampled from daily data)
    
    #### 💡 Strategy Benefits
    - Captures strong trending moves
    - Clear entry and exit rules
    - Flexible timeframe selection
    - Risk management through TP/SL
    
    #### ⚠️ Important Notes
    - The strategy requires sufficient historical data for SMA calculation (at least 200 periods)
    - Higher timeframes (Weekly/Monthly) may generate fewer signals but potentially stronger trends
    - Always consider transaction costs in your analysis
    """)

# --- Inputs ---
# Date Presets logic
if 'date_preset' not in st.session_state:
    st.session_state.date_preset = "1 Year"

def update_dates():
    preset = st.session_state.p6_preset_radio
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
        
    st.session_state.p6_start_date = start
    st.session_state.p6_end_date = end

# Initialize defaults if needed
if 'p6_start_date' not in st.session_state:
    st.session_state.p6_start_date = datetime.today().date() - timedelta(days=730)
if 'p6_end_date' not in st.session_state:
    st.session_state.p6_end_date = datetime.today().date()

# Date Selection and SL Toggle
col_top1, col_top2 = st.columns([4, 1])

with col_top1:
    st.radio(
        "Quick Select Range", 
        ["7 Days", "15 Days", "30 Days", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"], 
        horizontal=True, 
        key="p6_preset_radio",
        on_change=update_dates,
        index=6  # Default 2 Years for better SMA calculation
    )

with col_top2:
    st.write("") # Spacer to align with radio label
    USE_STOP_LOSS = st.checkbox("Enable Stop Loss", value=True, help="Toggle to enable or disable stop loss exit logic.")

# Core Inputs Row 1
row1_col1, row1_col2, row1_col3 = st.columns(3)
with row1_col1:
    SELECTED_STOCK = st.text_input("Stock Name", value="RELIANCE", help="Enter stock symbol (e.g. TCS, INFY)")
with row1_col2:
    TIMEFRAME = st.selectbox(
        "Timeframe",
        ["Daily", "Weekly", "Monthly"],
        index=0,
        help="Select the timeframe for SMA calculation."
    )
with row1_col3:
    INVESTMENT = st.number_input(
        "Investment (INR)", 
        value=10000.0, 
        step=1000.0,
        help="Amount to invest per trade."
    )

# Core Inputs Row 2
row2_col1, row2_col2, row2_col3 = st.columns(3)
with row2_col1:
    TAKE_PROFIT_PCT = st.number_input(
        "Take Profit %", 
        value=10.0, 
        step=0.5,
        help="Target profit percentage to exit the trade."
    )
with row2_col2:
    STOP_LOSS_PCT = st.number_input(
        "Stop Loss %", 
        value=5.0, 
        step=0.5,
        help="Stop loss percentage to exit the trade.",
        disabled=not USE_STOP_LOSS
    )
with row2_col3:
    CHARGES_PER_TRADE = st.number_input(
        "Charges Per Trade (Rs)", 
        value=25.0, 
        step=5.0,
        help="Transaction charges per trade (entry + exit combined)."
    )

# Core Inputs Row 3
row3_col1, row3_col2, row3_col3 = st.columns(3)
with row3_col1:
    START_DATE = st.date_input("Start Date", key="p6_start_date")
with row3_col2:
    END_DATE = st.date_input("End Date", key="p6_end_date", max_value=datetime.today().date())
with row3_col3:
    st.write("") # Spacer for alignment
    st.write("")
    st.link_button("Chartlink Screener", "https://chartink.com/screener/golden-gate-by-sunil-minglani-2")

# --- Run Backtest ---
if st.button("Run Backtest", type="primary"):
    # Auto-append .NS if not present
    SELECTED_STOCK = SELECTED_STOCK.upper().strip()
    if not SELECTED_STOCK.endswith(".NS"):
        SELECTED_STOCK += ".NS"
    
    st.info(f"Fetching data for {SELECTED_STOCK} from {START_DATE} to {END_DATE}...")
    
    try:
        # Fetch Data with buffer for SMA calculation
        # Need extra data for 200-period SMA
        if TIMEFRAME == "Weekly":
            buffer_days = 200 * 7 + 100 # ~1500 days
        elif TIMEFRAME == "Monthly":
            buffer_days = 200 * 31 + 100 # ~6300 days
        else: # Daily
            buffer_days = 200 * 1.5 + 50 # ~350 days
        
        fetch_start_date = START_DATE - timedelta(days=int(buffer_days))
        
        df = yf.download(SELECTED_STOCK, start=fetch_start_date, end=END_DATE, progress=False)
        
        if df.empty:
            st.error("No data fetched from Yahoo Finance. Please check inputs.")
            st.session_state.p6_results = None
            st.stop()

        # Handle MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        # Resample based on timeframe
        df_resampled = resample_to_timeframe(df.copy(), TIMEFRAME)
        
        # Calculate SMAs on resampled data
        df_resampled['SMA_10'] = calculate_sma(df_resampled['Close'], 10)
        df_resampled['SMA_20'] = calculate_sma(df_resampled['Close'], 20)
        df_resampled['SMA_50'] = calculate_sma(df_resampled['Close'], 50)
        df_resampled['SMA_100'] = calculate_sma(df_resampled['Close'], 100)
        df_resampled['SMA_200'] = calculate_sma(df_resampled['Close'], 200)
        
        # Filter to user's selected date range
        df_resampled = df_resampled[df_resampled['Date'] >= pd.to_datetime(START_DATE)].reset_index(drop=True)
        
        if df_resampled.empty:
            st.error("No data available for the selected date range.")
            st.session_state.p6_results = None
            st.stop()
            
        # Check if SMA_200 is available for the desired window
        nan_sma_count = df_resampled['SMA_200'].isna().sum()
        if nan_sma_count == len(df_resampled):
            st.error(f"Insufficient historical data for {SELECTED_STOCK} on {TIMEFRAME} timeframe to calculate SMA 200. Please try a more recent 'Start Date' or a stock with a longer trading history.")
            st.session_state.p6_results = None
            st.stop()
        elif nan_sma_count > 0:
            st.warning(f"Note: Backtest will start from {df_resampled.iloc[nan_sma_count]['Date'].date()} as SMA 200 requires earlier historical data for calculation.")
        
        # --- Backtest Logic ---
        results = []
        position = None  # None or dict
        total_pnl = 0
        can_enter = True  # Flag to ensure the condition is broken before re-entry
        
        for i, row in df_resampled.iterrows():
            current_date = row['Date']
            close_price = row['Close']
            high_price = row['High']
            low_price = row['Low']
            open_price = row['Open']
            
            # Check if all SMAs are available
            if pd.isna(row['SMA_200']):
                continue
            
            # 1. Manage Open Position
            if position:
                entry_price = position['entry_price']
                
                # Calculate TP and SL prices
                tp_price = entry_price * (1 + TAKE_PROFIT_PCT / 100)
                sl_price = entry_price * (1 - STOP_LOSS_PCT / 100)
                
                exit_action = None
                exit_price = 0
                
                # Check for Stop Loss if enabled
                if USE_STOP_LOSS:
                    # Check if Gap down below SL at Open
                    if open_price <= sl_price:
                        exit_action = "STOP LOSS"
                        exit_price = open_price
                    # Check if Low hit SL during the period
                    elif low_price <= sl_price:
                        exit_action = "STOP LOSS"
                        exit_price = sl_price
                
                # Check for Take Profit
                if not exit_action:
                    # Check if Gap up above TP at Open
                    if open_price >= tp_price:
                        exit_action = "TAKE PROFIT"
                        exit_price = open_price
                    # Check if High hit TP during the period
                    elif high_price >= tp_price:
                        exit_action = "TAKE PROFIT"
                        exit_price = tp_price
                
                if exit_action:
                    # Execute Exit
                    pnl = float(round((exit_price - entry_price) * position['quantity'] - CHARGES_PER_TRADE, 2))
                    total_pnl = float(round(total_pnl + pnl, 2))
                    
                    results.append({
                        'Entry Date': position['entry_date'],
                        'Exit Date': current_date,
                        'Symbol': SELECTED_STOCK,
                        'Action': exit_action,
                        'Entry Price': float(round(entry_price, 2)),
                        'Exit Price': float(round(exit_price, 2)),
                        'Quantity': float(round(position['quantity'], 2)),
                        'Invested': float(round(INVESTMENT, 2)),
                        'PnL': pnl,
                        'Cumulative PnL': total_pnl,
                        'SMA_10_Entry': position['sma_10'],
                        'SMA_20_Entry': position['sma_20'],
                        'SMA_50_Entry': position['sma_50'],
                        'SMA_100_Entry': position['sma_100'],
                        'SMA_200_Entry': position['sma_200']
                    })
                    
                    position = None  # Close position
            
            # 2. Check for Entry Signal (if no position open)
            if not position:
                # Golden Gate Condition: Price > SMA_10 > SMA_20 > SMA_50 > SMA_100 > SMA_200
                condition = (
                    close_price > row['SMA_10'] and
                    row['SMA_10'] > row['SMA_20'] and
                    row['SMA_20'] > row['SMA_50'] and
                    row['SMA_50'] > row['SMA_100'] and
                    row['SMA_100'] > row['SMA_200']
                )
                
                if condition:
                    if can_enter:
                        # Buy at close
                        entry_price = float(round(close_price, 2))
                        quantity = float(round(INVESTMENT / entry_price, 2))
                        
                        position = {
                            'entry_date': current_date,
                            'entry_price': entry_price,
                            'quantity': quantity,
                            'sma_10': float(round(row['SMA_10'], 2)),
                            'sma_20': float(round(row['SMA_20'], 2)),
                            'sma_50': float(round(row['SMA_50'], 2)),
                            'sma_100': float(round(row['SMA_100'], 2)),
                            'sma_200': float(round(row['SMA_200'], 2))
                        }
                        can_enter = False  # Set to False upon entry
                else:
                    # Condition is broken, reset can_enter to True
                    can_enter = True
        
        # Handle open position at end
        open_trade_val = 0
        last_trade_position = None
        if position:
            open_trade_val = float(round(position['quantity'] * df_resampled.iloc[-1]['Close'], 2))
            last_trade_position = position
        
        # --- Store in Session State ---
        results_df = pd.DataFrame(results)
        
        st.session_state.p6_results = {
            'results_df': results_df,
            'open_trade_val': open_trade_val,
            'selected_stock': SELECTED_STOCK,
            'position': last_trade_position,
            'price_data': df_resampled,
            # Store inputs used for this run
            'start_date': START_DATE,
            'end_date': END_DATE,
            'investment': INVESTMENT,
            'tp_pct': TAKE_PROFIT_PCT,
            'use_stop_loss': USE_STOP_LOSS,
            'sl_pct': STOP_LOSS_PCT,
            'charges': CHARGES_PER_TRADE,
            'timeframe': TIMEFRAME
        }

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state.p6_results = None

# --- Display Results from Session State ---
if 'p6_results' in st.session_state and st.session_state.p6_results:
    data = st.session_state.p6_results
    results_df = data['results_df']
    open_trade_val = data['open_trade_val']
    stock_symbol = data['selected_stock']
    position = data['position']
    df_slice = data['price_data']
    
    # Retrieve inputs used for this run
    run_start_date = data.get('start_date', START_DATE)
    run_end_date = data.get('end_date', END_DATE)
    run_investment = data.get('investment', INVESTMENT)
    run_tp_pct = data.get('tp_pct', TAKE_PROFIT_PCT)
    run_use_sl = data.get('use_stop_loss', True)
    run_sl_pct = data.get('sl_pct', STOP_LOSS_PCT)
    run_charges = data.get('charges', CHARGES_PER_TRADE)
    run_timeframe = data.get('timeframe', TIMEFRAME)
    
    st.header(f"Performance: {stock_symbol}")
    
    # Metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    total_trades = len(results_df)
    win_count = len(results_df[results_df['Action'] == 'TAKE PROFIT']) if not results_df.empty else 0
    loss_count = len(results_df[results_df['Action'] == 'STOP LOSS']) if not results_df.empty else 0
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    net_profit = float(round(results_df['PnL'].sum(), 2)) if not results_df.empty else 0.0
    roi = float(round((net_profit / run_investment) * 100, 2)) if run_investment > 0 else 0.0
    
    # XIRR
    xirr = 0.0
    if not results_df.empty:
        cashflows = []
        for _, row_res in results_df.iterrows():
            cashflows.append((row_res["Entry Date"], -row_res["Invested"]))
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
    col_m7.metric("Open Value", f"₹{open_trade_val:,.2f}" if position else "₹0.00")

    # Links
    tv_symbol = stock_symbol.replace(".NS", "")
    tv_url = f"https://www.tradingview.com/chart/?symbol=NSE%3A{tv_symbol}"
    st.link_button("View Chart on TradingView", tv_url)

    # --- Download Analytics ---
    download_data = {
        "Stock Name": [stock_symbol],
        "Strategy Name": ["Golden Gate Strategy"],
        "Timeframe": [run_timeframe],
        "Start Date": [run_start_date],
        "End Date": [run_end_date],
        "Investment": [run_investment],
        "Take Profit %": [run_tp_pct],
        "Use Stop Loss": [run_use_sl],
        "Stop Loss %": [run_sl_pct if run_use_sl else "N/A"],
        "Charges Per Trade": [float(round(run_charges, 2))],
        "Total PnL": [float(round(net_profit, 2))],
        "Total Trades": [total_trades],
        "Win Rate": [f"{win_rate:.2f}%"],
        "Wins": [win_count],
        "Losses": [loss_count],
        "XIRR": [f"{xirr * 100:.2f}%"],
        "ROI": [f"{roi:.2f}%"],
        "Open Value": [float(round(open_trade_val, 2))]
    }
    download_df = pd.DataFrame(download_data)
    csv_data = download_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Strategy Analytics as CSV",
        data=csv_data,
        file_name=f"{stock_symbol}_golden_gate_analytics.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # --- Open Trades Table ---
    if position:
        st.subheader("📌 Open Trade")
        # Use the latest close price from the resampled data
        current_price = df_slice.iloc[-1]['Close']
        open_pnl = float(round((current_price - position['entry_price']) * position['quantity'], 2))
        open_pnl_pct = float(round((current_price / position['entry_price'] - 1) * 100, 2))
        
        open_trade_data = {
            "Symbol": [stock_symbol],
            "Entry Date": [position['entry_date'].date()],
            "Entry Price": [float(round(position['entry_price'], 2))],
            "Qty": [position['quantity']],
            "Current Price": [float(round(current_price, 2))],
            "Current Value": [float(round(position['quantity'] * current_price, 2))],
            "PnL": [open_pnl],
            "PnL %": [f"{open_pnl_pct:.2f}%"]
        }
        st.dataframe(pd.DataFrame(open_trade_data), width='stretch', hide_index=True)
        st.markdown("---")

    if not results_df.empty:
        # Trade History
        st.subheader("Trade History")
        results_df = results_df.sort_values("Entry Date")
        
        # Format for display
        display_df = results_df.copy()
        display_df['Entry Date'] = display_df['Entry Date'].dt.date
        display_df['Exit Date'] = display_df['Exit Date'].dt.date
        
        cols_to_show = ['Symbol', 'Entry Date', 'Entry Price', 'Quantity', 'Invested', 
                       'Exit Date', 'Exit Price', 'Action', 'PnL', 'Cumulative PnL']
        st.dataframe(display_df[cols_to_show], width='stretch')
        
        st.markdown("---")
        
        # --- Market Analysis Table ---
        st.subheader("Market Analysis")
        
        # 1. Price Change
        start_price = df_slice.iloc[0]['Close']
        end_price = df_slice.iloc[-1]['Close']
        price_change = end_price - start_price
        price_change_pct = (price_change / start_price) * 100
        
        # 2. Perfect Trade (Min to Max)
        min_row = df_slice.loc[df_slice['Low'].idxmin()]
        global_min = min_row['Low']
        min_date = min_row['Date']
        
        df_after_min = df_slice[df_slice['Date'] > min_date]
        
        if not df_after_min.empty:
            max_row = df_after_min.loc[df_after_min['High'].idxmax()]
            global_max_after_min = max_row['High']
            max_date = max_row['Date']
            
            quantity_perfect = run_investment / global_min
            gross_val_perfect = quantity_perfect * global_max_after_min
            pnl_perfect = float(round(gross_val_perfect - run_investment - run_charges, 2))
            roi_perfect = float(round((pnl_perfect / run_investment) * 100, 2))
        else:
            global_max_after_min = global_min
            max_date = min_date
            pnl_perfect = 0.0
            roi_perfect = 0.0

        # 3. Buy & Hold
        quantity_bnh = run_investment / start_price
        gross_val_bnh = quantity_bnh * end_price
        pnl_bnh = float(round(gross_val_bnh - run_investment - run_charges, 2))
        roi_bnh = float(round((pnl_bnh / run_investment) * 100, 2))
        
        # Construct DataFrame
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
                f"{df_slice.iloc[0]['Date'].date()} to {df_slice.iloc[-1]['Date'].date()}"
            ]
        }
        st.table(pd.DataFrame(analysis_data))
        
        st.markdown("---")
        
        # Charts
        st.subheader("Equity Curve")
        equity_df = results_df[['Exit Date', 'Cumulative PnL']].copy()
        equity_df = equity_df.rename(columns={'Exit Date': 'Date'})
        st.line_chart(equity_df, x='Date', y='Cumulative PnL')
        
        # SMA Analysis at Entry
        st.subheader("SMA Values at Entry Points")
        sma_cols = ['Entry Date', 'Entry Price', 'SMA_10_Entry', 'SMA_20_Entry', 
                   'SMA_50_Entry', 'SMA_100_Entry', 'SMA_200_Entry']
        sma_display = display_df[sma_cols].copy()
        st.dataframe(sma_display, width='stretch')
        
    else:
        st.warning("No trades generated with the current parameters. Try adjusting the date range or timeframe.")
