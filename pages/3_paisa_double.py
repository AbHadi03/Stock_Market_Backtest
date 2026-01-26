import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

with col_s1:
    START_DATE = st.date_input("Start Date", value=datetime.today() - timedelta(days=365))
with col_s2:
    END_DATE = st.date_input("End Date", value=datetime.today(), max_value=datetime.today())

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
        total_pnl = 0
        total_charges = 0
        cycles_completed = 0
        
        # Find start index
        start_date_ts = pd.to_datetime(START_DATE)
        df_slice = df[df['Date'] >= start_date_ts].copy().reset_index(drop=True)
        
        if df_slice.empty:
             st.error("No data available for the selected date range.")
             st.stop()

        # Iterate day by day
        for i, row in df_slice.iterrows():
            current_date = row['Date']
            close_price = row['Close']
            high_price = row['High']
            low_price = row['Low']
            
            # 1. Manage Open Position
            if position:
                entry_price = position['entry_price']
                
                # Check Exit Conditions
                
                # TP Price
                tp_price = entry_price * (1 + TAKE_PROFIT_PCT / 100)
                # SL Price
                sl_price = entry_price * (1 - STOP_LOSS_PCT / 100)
                
                exit_action = None
                exit_price = 0
                
                # Check if SL hit (Low <= SL) - Prioritize SL check if both hit in same candle? 
                # Usually checking Low first is safer.
                if low_price <= sl_price:
                    exit_action = "STOP LOSS"
                    exit_price = sl_price
                elif high_price >= tp_price:
                    exit_action = "TAKE PROFIT"
                    exit_price = tp_price
                
                if exit_action:
                    # Execute Exit
                    gross_val = position['quantity'] * exit_price
                    # We deduct charges once per "trade cycle" (Entry + Exit combined in input? 
                    # Prompt said "take it in the input and deduct it in trades". 
                    # Assuming CHARGES_PER_TRADE is total for the round trip.)
                    
                    pnl = gross_val - position['invested_amount'] - CHARGES_PER_TRADE
                    total_pnl += pnl
                    total_charges += CHARGES_PER_TRADE
                    
                    results.append({
                        'Entry Date': position['entry_date'],
                        'Exit Date': current_date,
                        'Symbol': SELECTED_STOCK,
                        'Action': exit_action,
                        'Entry Price': entry_price,
                        'Exit Price': exit_price,
                        'Quantity': position['quantity'],
                        'Invested': position['invested_amount'],
                        'PnL': pnl,
                        'Cumulative PnL': total_pnl
                    })
                    
                    # Next Investment Logic
                    if exit_action == "TAKE PROFIT":
                        # Reset cycle
                        current_investment = INITIAL_INVESTMENT
                        cycles_completed += 1
                    else:
                        # SL Hit -> Martingale
                        current_investment *= MULTIPLIER
                    
                    position = None # Close position
                    
            # 2. Open New Position (if none open)
            elif not position:
                # Buy at Close of this candle (simulating filling at close or next open)
                # Strategy says "buy ... on next day close price". 
                # So if we exited today, we are effectively flat. 
                # We can enter TODAY's Close for the NEXT trade, 
                # effectively carrying the position overnight starting today.
                
                entry_price = close_price
                quantity = current_investment / entry_price
                
                position = {
                    'entry_date': current_date,
                    'entry_price': entry_price,
                    'quantity': quantity,
                    'invested_amount': current_investment
                }
        
        # Capture open position at end
        open_trade_val = 0
        if position:
            open_trade_val = position['quantity'] * df_slice.iloc[-1]['Close']
        
        # --- Results Processing ---
        results_df = pd.DataFrame(results)
        
        st.header(f"Performance: {SELECTED_STOCK}")
        
        # Metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        total_trades = len(results_df)
        win_count = len(results_df[results_df['Action'] == 'TAKE PROFIT'])
        loss_count = len(results_df[results_df['Action'] == 'STOP LOSS'])
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        net_profit = results_df['PnL'].sum() if not results_df.empty else 0
        roi = (net_profit / INITIAL_INVESTMENT) * 100 # ROI on initial capital base
        
        col_m1.metric("Total PnL", f"₹{net_profit:,.2f}", delta_color="normal" if net_profit >= 0 else "inverse")
        col_m2.metric("Total Trades", total_trades)
        col_m3.metric("Win Rate", f"{win_rate:.1f}%")
        col_m4.metric("ROI (on Base)", f"{roi:.1f}%")
        
        col_m5, col_m6, col_m7 = st.columns(3)
        col_m5.metric("Cycles Completed", cycles_completed)
        col_m6.metric("Current Investment Requirement", f"₹{current_investment:,.2f}")
        col_m7.metric("Open Position Value", f"₹{open_trade_val:,.2f}" if position else "₹0.00")

        # TradingView Link
        tv_symbol = SELECTED_STOCK.replace(".NS", "")
        tv_url = f"https://www.tradingview.com/chart/?symbol=NSE%3A{tv_symbol}"
        st.link_button("View Chart on TradingView", tv_url)
        
        st.markdown("---")
        
        if not results_df.empty:
            # Tables and Charts
            
            # 1. Trade History
            st.subheader("Trade History")
            # Sort by Entry Date
            results_df = results_df.sort_values("Entry Date")
            
            # Format
            display_df = results_df.copy()
            display_df['Entry Date'] = display_df['Entry Date'].dt.date
            display_df['Exit Date'] = display_df['Exit Date'].dt.date
            
            st.dataframe(display_df, width='stretch')
            
            # 2. Charts
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                st.subheader("Equity Curve")
                equity_df = results_df[['Exit Date', 'Cumulative PnL']].copy()
                equity_df = equity_df.rename(columns={'Exit Date': 'Date'})
                st.line_chart(equity_df, x='Date', y='Cumulative PnL')
                
            with col_c2:
                st.subheader("Investment Size Progression")
                st.bar_chart(results_df, x='Entry Date', y='Invested')
                
            # 3. Plotly Candle Chart with Buy/Sell
            st.subheader("Price Chart & Trade Points")
            
            fig = make_subplots(rows=1, cols=1)
            
            # Candle
            fig.add_trace(go.Candlestick(
                x=df_slice['Date'],
                open=df_slice['Open'],
                high=df_slice['High'],
                low=df_slice['Low'],
                close=df_slice['Close'],
                name='Price'
            ))
            
            # Markers
            entries = results_df[['Entry Date', 'Entry Price']].copy()
            exits_tp = results_df[results_df['Action']=='TAKE PROFIT'][['Exit Date', 'Exit Price']]
            exits_sl = results_df[results_df['Action']=='STOP LOSS'][['Exit Date', 'Exit Price']]
            
            fig.add_trace(go.Scatter(
                x=entries['Entry Date'], y=entries['Entry Price'],
                mode='markers', name='Entry', marker=dict(color='blue', symbol='triangle-up', size=10)
            ))
            fig.add_trace(go.Scatter(
                x=exits_tp['Exit Date'], y=exits_tp['Exit Price'],
                mode='markers', name='Take Profit', marker=dict(color='green', symbol='circle', size=10)
            ))
            fig.add_trace(go.Scatter(
                x=exits_sl['Exit Date'], y=exits_sl['Exit Price'],
                mode='markers', name='Stop Loss', marker=dict(color='red', symbol='x', size=10)
            ))
            
            fig.update_layout(xaxis_rangeslider_visible=False, height=600)
            st.plotly_chart(fig, width='stretch')
            
        else:
            st.warning("No trades generated with the current parameters.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        # st.exception(e)
