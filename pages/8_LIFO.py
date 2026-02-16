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
        except (OverflowError, ZeroDivisionError):
             val = float('inf')
        
        if abs(val) < 1e-6:
            return mid
        if val > 0:
            low = mid
        else:
            high = mid
    return mid

# --- Page Config ---
st.set_page_config(layout="wide", page_title="LIFO Strategy", page_icon="🏗️")

# --- Login Check ---
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("Please login first from the main page")
    st.stop()

st.title("🏗️ LIFO Strategy (Last-In-First-Out Martingale)")
st.markdown("---")

with st.expander("📖 Strategy Description & Rules"):
    st.markdown("""
    ### LIFO Strategy - Last-In-First-Out Spot Martingale
    
    This strategy uses a grid-based buying approach and manages exits using a LIFO (Last-In-First-Out) stack.

    #### 🔄 Core Logic
    1. **Initial Buy**: Buy the first lot (L1) on the start date at the closing price.
    2. **Grid Buying**: If the stock price drops by the **Consecutive Trade %** from the last entry price, buy another lot (L2, L3, etc.) with the same investment amount.
    3. **LIFO Selling**:
        - Only the **most recent lot bought** (the one at the top of the stack) is checked for Take Profit.
        - If the latest lot hits the **Take Profit %** (and has been held for at least **T+2 days**), it is sold.
        - Once sold, the previous lot becomes active and is checked for its own Take Profit level.
    4. **Cycle Restart**: When all lots (including L1) are sold, the cycle is complete. A new L1 is bought at the current day's close to start a new cycle.
    5. **No Indicators**: This strategy does not use any technical indicators or stop losses.
    6. **CDSL Rule**: T+2 holding period is mandatory before any lot can be sold.
    """)

# --- Inputs ---
col1, col2, col3 = st.columns(3)

with col1:
    INITIAL_INVESTMENT = st.number_input(
        "Lot Investment (INR)", 
        value=10000.0, 
        step=1000.0,
        help="Capital allocated for each lot."
    )
    CONSECUTIVE_TRADE_PCT = st.number_input(
        "Consecutive Trade % (Drop)", 
        value=5.0, 
        step=0.5,
        help="Percentage drop from the last bought lot to trigger a new lot purchase."
    )

with col2:
    TAKE_PROFIT_PCT = st.number_input(
        "Take Profit %", 
        value=5.0, 
        step=0.5,
        help="Target profit percentage for each individual lot."
    )
    CHARGES_PER_TRADE = st.number_input(
        "Charges Per Trade (Rs)", 
        value=25.0, 
        step=5.0,
        help="Transaction costs per trade (Entry + Exit combined)."
    )

with col3:
    SELECTED_STOCK = st.text_input("Stock Name", value="RELIANCE", help="Enter stock symbol (e.g. TCS, INFY)")

st.markdown("### Date Selection")

# Date Presets logic
if 'p8_preset_radio' not in st.session_state:
    st.session_state.p8_preset_radio = "1 Year"

def update_dates():
    preset = st.session_state.p8_preset_radio
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
    else: # Custom
        return
        
    st.session_state.p8_start_date = start
    st.session_state.p8_end_date = end

st.radio(
    "Quick Select Range", 
    ["7 Days", "15 Days", "30 Days", "3 Months", "6 Months", "1 Year"], 
    horizontal=True, 
    key="p8_preset_radio",
    on_change=update_dates,
    index=5 # Default 1 Year
)

if 'p8_start_date' not in st.session_state:
    st.session_state.p8_start_date = datetime.today().date() - timedelta(days=365)
if 'p8_end_date' not in st.session_state:
    st.session_state.p8_end_date = datetime.today().date()

col_s1, col_s2 = st.columns(2)
with col_s1:
    START_DATE = st.date_input("Start Date", key="p8_start_date")
with col_s2:
    END_DATE = st.date_input("End Date", key="p8_end_date", max_value=datetime.today().date())

# --- Run Backtest ---
if st.button("Run LIFO Backtest", type="primary"):
    SELECTED_STOCK = SELECTED_STOCK.upper().strip()
    if not SELECTED_STOCK.endswith(".NS"):
        SELECTED_STOCK += ".NS"
    
    st.info(f"Computing LIFO Strategy for {SELECTED_STOCK}...")
    
    try:
        # Fetch Data
        df = yf.download(SELECTED_STOCK, start=START_DATE, end=END_DATE, progress=False)
        
        if df.empty:
            st.error("No data fetched. Check symbol/dates.")
            st.stop()
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        # --- Backtest Engine ---
        all_trades = []
        active_lots = [] # Stack of lots
        total_pnl = 0
        total_charges = 0
        
        current_cycle = 1
        
        for i, row in df.iterrows():
            current_date = row['Date']
            open_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            
            # 1. Check Exit (Only on the LAST lot in active_lots)
            if active_lots:
                latest_lot = active_lots[-1]
                duration_days = (current_date - latest_lot['entry_date']).days
                
                if duration_days >= 2:
                    exit_price = None
                    target_price = latest_lot['target_price']
                    
                    # Gap up at Open
                    if open_price >= target_price:
                        exit_price = open_price
                    # Intra-day High
                    elif high_price >= target_price:
                        exit_price = target_price
                        
                    if exit_price:
                        # Execute Exit for this lot
                        pnl = float(round((exit_price - latest_lot['entry_price']) * latest_lot['quantity'] - CHARGES_PER_TRADE, 2))
                        total_pnl = float(round(total_pnl + pnl, 2))
                        total_charges += CHARGES_PER_TRADE
                        
                        all_trades.append({
                            'Lot': f"L{latest_lot['lot_num']}",
                            'Cycle': latest_lot['cycle'],
                            'Entry Date': latest_lot['entry_date'],
                            'Exit Date': current_date,
                            'Entry Price': latest_lot['entry_price'],
                            'Exit Price': float(round(exit_price, 2)),
                            'Quantity': latest_lot['quantity'],
                            'Invested': float(round(latest_lot['invested'], 2)),
                            'PnL': pnl,
                            'Duration': duration_days
                        })
                        
                        # Remove lot from stack
                        active_lots.pop()
                        
                        # If cycle is complete, buy a new lot immediately at today's close
                        if not active_lots:
                            current_cycle += 1
                            # Immediate entry for new cycle
                            entry_price = float(round(close_price, 2))
                            qty = float(round(INITIAL_INVESTMENT / entry_price, 2))
                            active_lots.append({
                                'lot_num': 1,
                                'cycle': current_cycle,
                                'entry_date': current_date,
                                'entry_price': entry_price,
                                'quantity': qty,
                                'invested': INITIAL_INVESTMENT,
                                'target_price': float(round(entry_price * (1 + TAKE_PROFIT_PCT / 100), 2))
                            })
                        
                        # Important: Only one exit per day for now to keep things simple, 
                        # or we could loop while TP condition is met. 
                        # User requirement says "if price goes up... sold L3... check for TP on L2".
                        # Let's loop just in case multiple lots exit on same day high.
                        while active_lots:
                            next_lot = active_lots[-1]
                            # Recalculate duration for next lot
                            next_duration = (current_date - next_lot['entry_date']).days
                            if next_duration >= 2:
                                next_target = next_lot['target_price']
                                next_exit = None
                                # For subsequent exits on same day, we use the same high or open
                                if open_price >= next_target:
                                    next_exit = open_price
                                elif high_price >= next_target:
                                    next_exit = next_target
                                    
                                if next_exit:
                                    pnl = float(round((next_exit - next_lot['entry_price']) * next_lot['quantity'] - CHARGES_PER_TRADE, 2))
                                    total_pnl = float(round(total_pnl + pnl, 2))
                                    total_charges += CHARGES_PER_TRADE
                                    all_trades.append({
                                        'Lot': f"L{next_lot['lot_num']}",
                                        'Cycle': next_lot['cycle'],
                                        'Entry Date': next_lot['entry_date'],
                                        'Exit Date': current_date,
                                        'Entry Price': next_lot['entry_price'],
                                        'Exit Price': float(round(next_exit, 2)),
                                        'Quantity': next_lot['quantity'],
                                        'Invested': float(round(next_lot['invested'], 2)),
                                        'PnL': pnl,
                                        'Duration': next_duration
                                    })
                                    active_lots.pop()
                                    if not active_lots:
                                        current_cycle += 1
                                        # Entry for new cycle
                                        entry_p = float(round(close_price, 2))
                                        qty_n = float(round(INITIAL_INVESTMENT / entry_p, 2))
                                        active_lots.append({
                                            'lot_num': 1,
                                            'cycle': current_cycle,
                                            'entry_date': current_date,
                                            'entry_price': entry_p,
                                            'quantity': qty_n,
                                            'invested': INITIAL_INVESTMENT,
                                            'target_price': float(round(entry_p * (1 + TAKE_PROFIT_PCT / 100), 2))
                                        })
                                        break
                                else:
                                    break
                            else:
                                break
                
            # 2. Check Entry (Grid Buy)
            if not active_lots:
                # First ever entry
                entry_price = float(round(close_price, 2))
                qty = float(round(INITIAL_INVESTMENT / entry_price, 2))
                active_lots.append({
                    'lot_num': 1,
                    'cycle': current_cycle,
                    'entry_date': current_date,
                    'entry_price': entry_price,
                    'quantity': qty,
                    'invested': INITIAL_INVESTMENT,
                    'target_price': float(round(entry_price * (1 + TAKE_PROFIT_PCT / 100), 2))
                })
            else:
                # Check for grid entry from last bought lot
                latest_lot = active_lots[-1]
                required_price = latest_lot['entry_price'] * (1 - CONSECUTIVE_TRADE_PCT / 100)
                
                # Check if Close or Low drops below required price
                if low_price <= required_price:
                    # Buy at required price or Low? Usually buy at grid level or market close.
                    # As per requirement "check if stock goes 5% down then we take same investment again"
                    grid_entry_price = float(round(min(close_price, required_price), 2))
                    qty = float(round(INITIAL_INVESTMENT / grid_entry_price, 2))
                    active_lots.append({
                        'lot_num': len(active_lots) + 1,
                        'cycle': current_cycle,
                        'entry_date': current_date,
                        'entry_price': grid_entry_price,
                        'quantity': qty,
                        'invested': INITIAL_INVESTMENT,
                        'target_price': float(round(grid_entry_price * (1 + TAKE_PROFIT_PCT / 100), 2))
                    })

        # Final Processing
        trades_df = pd.DataFrame(all_trades)
        
        # Metrics
        st.subheader("Performance Summary")
        
        # Results metrics calculated in-situ or from df
        if not trades_df.empty:
            trades_df['Cumulative PnL'] = trades_df['PnL'].cumsum()
            total_trades_count = len(trades_df)
            net_pnl = trades_df['PnL'].sum()
            
            # Max capital calculation
            max_lots = 0
            # We would need to track LOT count per day to get true max capital, 
            # for now let's use the all_trades to infer or just show current status.
            # Better: let's calculate max lots from the loop if we were tracking it.
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Trades", total_trades_count)
            m2.metric("Net P&L", f"₹{net_pnl:,.2f}", delta_color="normal" if net_pnl >= 0 else "inverse")
            m3.metric("Total Charges", f"₹{total_charges:,.2f}")
            m4.metric("Active Lots", len(active_lots))
            
            # XIRR
            cashflows = []
            for _, t in trades_df.iterrows():
                cashflows.append((t['Entry Date'], -t['Invested']))
                cashflows.append((t['Exit Date'], t['Quantity'] * t['Exit Price'] - CHARGES_PER_TRADE))
            
            # Factor in open positions for XIRR
            latest_price = df.iloc[-1]['Close']
            for lot in active_lots:
                cashflows.append((lot['entry_date'], -lot['invested']))
                cashflows.append((df.iloc[-1]['Date'], lot['quantity'] * latest_price))

            cf_df = pd.DataFrame(cashflows, columns=["date", "cashflow"]).sort_values("date")
            xirr_val = calculate_xirr(cf_df)
            
            st.metric("Strategy XIRR", f"{xirr_val * 100:.2f}%")

            # Trade History Table
            st.subheader("Trade History")
            display_df = trades_df.copy()
            display_df['Entry Date'] = display_df['Entry Date'].dt.date
            display_df['Exit Date'] = display_df['Exit Date'].dt.date
            st.dataframe(display_df, width='stretch')

            # Equity Curve
            st.subheader("Equity Curve")
            st.line_chart(trades_df, x='Exit Date', y='Cumulative PnL')
        else:
            st.warning("No trades executed within parameters.")

        # Open Positions Section
        st.subheader("Current Open Lots")
        if active_lots:
            open_df = pd.DataFrame(active_lots)
            # Add Unrealized PnL
            curr_price = df.iloc[-1]['Close']
            open_df['Current Price'] = float(round(curr_price, 2))
            open_df['Unrealized PnL'] = (curr_price - open_df['entry_price']) * open_df['quantity']
            
            st.dataframe(open_df[['lot_num', 'cycle', 'entry_date', 'entry_price', 'quantity', 'invested', 'target_price', 'Unrealized_PnL' if 'Unrealized_PnL' in open_df else 'Unrealized PnL']], width='stretch')
        else:
            st.info("No active lots currently.")

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

