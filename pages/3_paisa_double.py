import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
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

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Paisa Double Strategy", page_icon="💰")

# --- Login Check ---
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("Please login first from the main page")
    st.stop()

st.title("💰 Paisa Double Strategy (Spot Martingale)")
st.markdown("---")

st.markdown("""
### Strategy Description
- **Concept**: Long-only Spot Martingale on Stocks.
- **Entry**: Buy at the Close price of the start date (or next day after an exit).
- **Exit - Take Profit**: Sell if price rises by **TP%**. -> **Reset** investment to Initial Amount.
- **Exit - Stop Loss**: Sell if price drops by **SL%**. -> **Multiply** investment by Multiplier (e.g., 2x) for next trade.
- **Goal**: Recover losses from previous trades using increased position size, then reset after a win.


**रणनीति विवरण (Hindi)**
- **अवधारणा**: स्टॉक्स पर केवल लॉन्ग (Long-only) स्पॉट ज़रेबंद (Martingale) रणनीति।
- **प्रविष्टि (Entry)**: शुरू करने की तारीख (या निकास के अगले दिन) के बंद भाव (Close Price) पर खरीदें।
- **निकास - लाभ लें (Take Profit)**: यदि कीमत **TP%** बढ़ जाती है -> निवेश को प्रारंभिक राशि पर **रीसेट** करें।
- **निकास - स्टॉप लॉस (Stop Loss)**: यदि कीमत **SL%** गिर जाती है -> अगले व्यापार के लिए निवेश को गुणक (Multiplier) (जैसे, 2x) से **गुणा** करें।
- **लक्ष्य**: घाटे की भरपाई के लिए स्थिति का आकार (Position Size) बढ़ाएं, और जीत के बाद रीसेट करें।
""")

# --- Inputs ---
col1, col2, col3 = st.columns(3)

with col1:
    INITIAL_INVESTMENT = st.number_input(
        "Initial Investment (INR)", 
        value=10000.0, 
        step=1000.0,
        help="Starting capital for the first trade."
    )
    MULTIPLIER = st.number_input(
        "Subsequent Multiplier", 
        value=2.0, 
        step=0.1,
        help="Multiplier for investment size after a Stop Loss hit."
    )

with col2:
    TAKE_PROFIT_PCT = st.number_input(
        "Take Profit %", 
        value=4.0, 
        step=0.5,
        help="Target profit percentage to close trade and reset cycle."
    )
    STOP_LOSS_PCT = st.number_input(
        "Stop Loss %", 
        value=2.0, 
        step=0.5,
        help="Loss percentage to close trade and increase next position size."
    )

with col3:
    CHARGES_PER_TRADE = st.number_input(
        "Charges Per Trade (Rs)", 
        value=25.0, 
        step=5.0,
        help="Flat fee per trade (deducted on both Entry and Exit, total)."
    )

st.markdown("### Stock Selection")
col_s1, col_s2, col_s3 = st.columns(3)

# Date Presets logic
if 'date_preset' not in st.session_state:
    st.session_state.date_preset = "1 Year"

def update_dates():
    preset = st.session_state.p3_preset_radio
    end = datetime.today()
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
    else: # Custom
        return
        
    st.session_state.p3_start_date = start
    st.session_state.p3_end_date = end

st.radio(
    "Quick Select Range", 
    ["7 Days", "15 Days", "30 Days", "3 Months", "6 Months", "1 Year"], 
    horizontal=True, 
    key="p3_preset_radio",
    on_change=update_dates,
    index=5 # Default 1 Year
)

# Initialize defaults if needed
if 'p3_start_date' not in st.session_state:
    st.session_state.p3_start_date = datetime.today() - timedelta(days=365)
if 'p3_end_date' not in st.session_state:
    st.session_state.p3_end_date = datetime.today()

with col_s1:
    START_DATE = st.date_input("Start Date", key="p3_start_date")
with col_s2:
    END_DATE = st.date_input("End Date", key="p3_end_date", max_value=datetime.today())

with col_s3:
    SELECTED_STOCK = st.text_input("Stock Name", value="RELIANCE", help="Enter stock symbol (e.g. TCS, INFY)")

# --- Run Backtest ---
if st.button("Run Backtest", type="primary"):
    # Auto-append .NS if not present
    SELECTED_STOCK = SELECTED_STOCK.upper().strip()
    if not SELECTED_STOCK.endswith(".NS"):
        SELECTED_STOCK += ".NS"
    
    st.info(f"Fetching data for {SELECTED_STOCK} from {START_DATE} to {END_DATE}...")
    
    try:
        # Fetch Data
        df = yf.download(SELECTED_STOCK, start=START_DATE, end=END_DATE, progress=False)
        
        if df.empty:
            st.error("No data fetched from Yahoo Finance. Please check inputs.")
            st.session_state.p3_results = None
            st.stop()

        # Handle MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        # --- Backtest Logic ---
        results = []
        
        # State variables
        current_investment = INITIAL_INVESTMENT
        position = None # None or dict
        total_charges = 0
        cycles_completed = 0
        current_cycle = 1 # Start with Cycle 1
        
        # Track max investment
        max_investment_used = 0
        max_inv_cycle = 0
        total_pnl = 0
        
        # Find start index
        start_date_ts = pd.to_datetime(START_DATE)
        df_slice = df[df['Date'] >= start_date_ts].copy().reset_index(drop=True)
        
        if df_slice.empty:
             st.error("No data available for the selected date range.")
             st.session_state.p3_results = None
             st.stop()

        # Iterate day by day
        for i, row in df_slice.iterrows():
            current_date = row['Date']
            open_price = row['Open']
            close_price = row['Close']
            high_price = row['High']
            low_price = row['Low']
            
            # 1. Manage Open Position
            if position:
                entry_price = position['entry_price']
                
                # Check Exit Conditions (Only after T+2 days)
                duration_days = (current_date - position['entry_date']).days
                
                exit_action = None
                exit_price = 0
                
                if duration_days >= 2:
                    # TP Price
                    tp_price = entry_price * (1 + TAKE_PROFIT_PCT / 100)
                    # SL Price
                    sl_price = entry_price * (1 - STOP_LOSS_PCT / 100)
                    
                    # Check if Gap down below SL at Open
                    if open_price <= sl_price:
                        exit_action = "STOP LOSS"
                        exit_price = open_price # Exit at Open if it gaps below SL
                    # Check if Low hit SL during the day
                    elif low_price <= sl_price:
                        exit_action = "STOP LOSS"
                        exit_price = sl_price
                    # Check if Gap up above TP at Open
                    elif open_price >= tp_price:
                        exit_action = "TAKE PROFIT"
                        exit_price = open_price # Exit at Open if it gaps above TP
                    # Check if High hit TP during the day
                    elif high_price >= tp_price:
                        exit_action = "TAKE PROFIT"
                        exit_price = tp_price
                
                if exit_action:
                    # Execute Exit
                    # PnL = (Exit Price - Entry Price) * Quantity - Charges
                    pnl = float(round((exit_price - entry_price) * position['quantity'] - CHARGES_PER_TRADE, 2))
                    total_pnl = float(round(total_pnl + pnl, 2))
                    total_charges = float(round(total_charges + CHARGES_PER_TRADE, 2))
                    
                    duration = (current_date - position['entry_date']).days
                    
                    results.append({
                        'Cycle': position['cycle_number'],
                        'Entry Date': position['entry_date'],
                        'Exit Date': current_date,
                        'Symbol': SELECTED_STOCK,
                        'Action': exit_action,
                        'Entry Price': float(round(entry_price, 2)),
                        'Exit Price': float(round(exit_price, 2)),
                        'Quantity': float(round(position['quantity'], 2)),
                        'Invested': float(round(position['invested_amount'], 2)),
                        'PnL': pnl,
                        'Cumulative PnL': total_pnl,
                        'Duration (Days)': duration
                    })
                    
                    # Next Investment Logic
                    if exit_action == "TAKE PROFIT":
                        # Reset cycle
                        current_investment = INITIAL_INVESTMENT
                        cycles_completed += 1
                        current_cycle += 1 # New cycle starts
                    else:
                        # SL Hit -> Martingale
                        current_investment *= MULTIPLIER
                    
                    position = None # Close position
                    
            # 2. Open New Position (if none open)
            elif not position:
                # Update Max Investment
                if current_investment > max_investment_used:
                    max_investment_used = float(round(current_investment, 2))
                    max_inv_cycle = current_cycle

                # Buy at Close of this candle
                
                entry_price = float(round(close_price, 2))
                quantity = float(round(current_investment / entry_price, 2))
                
                position = {
                    'entry_date': current_date,
                    'entry_price': entry_price,
                    'quantity': quantity,
                    'invested_amount': float(round(current_investment, 2)),
                    'cycle_number': current_cycle
                }
        
        # Capture open position at end
        open_trade_val = 0
        last_trade_position = None
        if position:
            open_trade_val = float(round(position['quantity'] * df_slice.iloc[-1]['Close'], 2))
            last_trade_position = position
        
        # --- Store in Session State ---
        results_df = pd.DataFrame(results)
        
        st.session_state.p3_results = {
            'results_df': results_df,
            'max_investment_used': max_investment_used,
            'max_inv_cycle': max_inv_cycle,
            'cycles_completed': cycles_completed,
            'current_investment': current_investment,
            'open_trade_val': open_trade_val,
            'selected_stock': SELECTED_STOCK,
            'position': last_trade_position,
            'price_data': df_slice,
            # Store inputs used for this run
            'start_date': START_DATE,
            'end_date': END_DATE,
            'initial_investment': INITIAL_INVESTMENT,
            'multiplier': MULTIPLIER,
            'tp_pct': TAKE_PROFIT_PCT,
            'sl_pct': STOP_LOSS_PCT,
            'charges': CHARGES_PER_TRADE
        }

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state.p3_results = None

# --- Display Results from Session State ---
if 'p3_results' in st.session_state and st.session_state.p3_results:
    data = st.session_state.p3_results
    results_df = data['results_df']
    max_investment_used = data['max_investment_used']
    max_inv_cycle = data['max_inv_cycle']
    cycles_completed = data['cycles_completed']
    current_investment = data['current_investment']
    open_trade_val = data['open_trade_val']
    stock_symbol = data['selected_stock']
    position = data['position']
    df_slice = data['price_data']
    
    # Retrieve inputs used for this run
    run_start_date = data.get('start_date', START_DATE)
    run_end_date = data.get('end_date', END_DATE)
    run_initial_inv = data.get('initial_investment', INITIAL_INVESTMENT)
    run_multiplier = data.get('multiplier', MULTIPLIER)
    run_tp_pct = data.get('tp_pct', TAKE_PROFIT_PCT)
    run_sl_pct = data.get('sl_pct', STOP_LOSS_PCT)
    run_charges = data.get('charges', CHARGES_PER_TRADE)
    
    st.header(f"Performance: {stock_symbol}")
    
    # Metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    total_trades = len(results_df)
    win_count = len(results_df[results_df['Action'] == 'TAKE PROFIT']) if not results_df.empty else 0
    loss_count = len(results_df[results_df['Action'] == 'STOP LOSS']) if not results_df.empty else 0
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    net_profit = float(round(results_df['PnL'].sum(), 2)) if not results_df.empty else 0.0
    roi = float(round((net_profit / max_investment_used) * 100, 2)) if max_investment_used > 0 else 0.0
    
    # XIRR
    xirr = 0.0
    if not results_df.empty:
        cashflows = []
        for _, row_res in results_df.iterrows():
            cashflows.append((row_res["Entry Date"], -row_res["Invested"]))
            exit_val = (row_res["Quantity"] * row_res["Exit Price"]) - CHARGES_PER_TRADE
            cashflows.append((row_res["Exit Date"], exit_val))
        
        cashflow_df = pd.DataFrame(cashflows, columns=["date", "cashflow"]).sort_values("date")
        xirr = float(round(calculate_xirr(cashflow_df), 4)) # Keeping 4 for raw rate, display as % will be rounded

    col_m1.metric("Total PnL", f"₹{net_profit:,.2f}", delta_color="normal" if net_profit >= 0 else "inverse")
    col_m2.metric("Total Trades", total_trades)
    col_m3.metric("XIRR", f"{xirr * 100:.2f}%")
    col_m4.metric("ROI (on Max Inv)", f"{roi:.1f}%")
    
    col_m5, col_m6, col_m7 = st.columns(3)
    col_m5.metric("Cycles Completed", cycles_completed)
    col_m6.metric("Max Investment", f"₹{max_investment_used:,.2f} (Cycle {max_inv_cycle})")
    col_m7.metric("Open Value", f"₹{open_trade_val:,.2f}" if position else "₹0.00")

    # TradingView Link
    tv_symbol = stock_symbol.replace(".NS", "")
    tv_url = f"https://www.tradingview.com/chart/?symbol=NSE%3A{tv_symbol}"
    st.link_button("View Chart on TradingView", tv_url)

    # --- Download Analytics ---
    download_data = {
        "Stock Name": [stock_symbol],
        "Strategy Name": ["Paisa Double Strategy"],
        "Start Date": [run_start_date],
        "End Date": [run_end_date],
        "Initial Investment": [run_initial_inv],
        "Multiplier": [run_multiplier],
        "Take Profit %": [run_tp_pct],
        "Stop Loss %": [run_sl_pct],
        "Charges Per Trade": [float(round(run_charges, 2))],
        "Total PnL": [float(round(net_profit, 2))],
        "Total Trades": [total_trades],
        "XIRR": [f"{xirr * 100:.2f}%"],
        "ROI (on Max Inv)": [f"{roi:.2f}%"],
        "Cycles Completed": [cycles_completed],
        "Max Investment": [float(round(max_investment_used, 2))],
        "Max Inv Cycle": [max_inv_cycle],
        "Open Value": [float(round(open_trade_val, 2))]
    }
    download_df = pd.DataFrame(download_data)
    csv_data = download_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Analytics as CSV",
        data=csv_data,
        file_name=f"{stock_symbol}_paisa_double_analytics.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    if not results_df.empty:
        # Tables and Charts
        
        # 1. Trade History
        st.subheader("Trade History")
        # Sort by Entry Date
        results_df = results_df.sort_values("Entry Date")
        
        # Filter by Cycle
        all_cycles = sorted(results_df['Cycle'].unique())
        cycle_filter = st.multiselect("Filter by Cycle Number", all_cycles)
        
        display_df = results_df.copy()
        if cycle_filter:
            display_df = display_df[display_df['Cycle'].isin(cycle_filter)]
        
        # --- Filtered Analytics ---
        if not display_df.empty:
            f_total_duration = display_df['Duration (Days)'].sum()
            f_total_pnl = display_df['PnL'].sum()
            f_max_inv = display_df['Invested'].max()
            f_roi = (f_total_pnl / f_max_inv) * 100 if f_max_inv > 0 else 0
            
            st.markdown("##### Filtered Data Analytics")
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("Total Duration (Days)", int(f_total_duration))
            fc2.metric("Total PnL", f"₹{float(round(f_total_pnl, 2)):,.2f}", delta_color="normal" if f_total_pnl >= 0 else "inverse")
            fc3.metric("ROI (on Filtered Max Inv)", f"{f_roi:.2f}%")
            st.write(f"*Filtered Max Investment: ₹{float(round(f_max_inv, 2)):,.2f}*")
        
        # Format
        display_df['Entry Date'] = display_df['Entry Date'].dt.date
        display_df['Exit Date'] = display_df['Exit Date'].dt.date
        
        cols_to_show = ['Symbol', 'Entry Date', 'Entry Price', 'Quantity', 'Invested', 'Exit Date', 'Exit Price', 'Action', 'PnL', 'Cumulative PnL', 'Duration (Days)', 'Cycle']
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
        # Find global min price and its date
        min_row = df_slice.loc[df_slice['Low'].idxmin()]
        global_min = min_row['Low']
        min_date = min_row['Date']
        
        # Slice df AFTER min date to find max
        df_after_min = df_slice[df_slice['Date'] > min_date]
        
        if not df_after_min.empty:
            max_row = df_after_min.loc[df_after_min['High'].idxmax()]
            global_max_after_min = max_row['High']
            max_date = max_row['Date']
            
            # Perfect Trade ROI
            quantity_perfect = INITIAL_INVESTMENT / global_min
            gross_val_perfect = quantity_perfect * global_max_after_min
            # Assume 1 cycle charges
            pnl_perfect = float(round(gross_val_perfect - INITIAL_INVESTMENT - CHARGES_PER_TRADE, 2))
            roi_perfect = float(round((pnl_perfect / INITIAL_INVESTMENT) * 100, 2))
        else:
            global_max_after_min = global_min # No movement after min?
            max_date = min_date
            pnl_perfect = 0.0
            roi_perfect = 0.0

        # 3. Buy & Hold
        quantity_bnh = INITIAL_INVESTMENT / start_price
        gross_val_bnh = quantity_bnh * end_price
        pnl_bnh = float(round(gross_val_bnh - INITIAL_INVESTMENT - CHARGES_PER_TRADE, 2))
        roi_bnh = float(round((pnl_bnh / INITIAL_INVESTMENT) * 100, 2))
        
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
        
        # 2. Charts (Sequential Layout)
        
        st.subheader("Equity Curve")
        equity_df = results_df[['Exit Date', 'Cumulative PnL']].copy()
        equity_df = equity_df.rename(columns={'Exit Date': 'Date'})
        st.line_chart(equity_df, x='Date', y='Cumulative PnL')
            
        st.subheader("Investment Size Progression")
        st.bar_chart(results_df, x='Entry Date', y='Invested')
    else:
        st.warning("No trades generated with the current parameters.")
