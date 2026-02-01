import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD histogram"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return histogram

def fetch_stock_data(ticker, start_date, end_date, interval='1d'):
    """Fetch stock data with error handling"""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        return df
    except:
        return None

def get_market_cap(ticker):
    """Get market cap in crores"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        market_cap_usd = info.get('marketCap', 0)
        # Convert USD to INR (approximate rate: 83)
        market_cap_inr = market_cap_usd * 91
        market_cap_crores = market_cap_inr / 10000000
        return market_cap_crores
    except:
        return 0

def validate_tickers(tickers, progress_bar):
    """Validate which tickers are available"""
    valid_tickers = []
    invalid_tickers = []
    
    for i, ticker in enumerate(tickers):
        progress_bar.progress((i + 1) / len(tickers), text=f"Validating {ticker}...")
        df = fetch_stock_data(ticker, datetime.today() - timedelta(days=30), datetime.today())
        if df is not None and len(df) > 0:
            valid_tickers.append(ticker)
        else:
            invalid_tickers.append(ticker)
    
    return valid_tickers, invalid_tickers

def run_screener(tickers, screening_date, progress_container, show_stage_results=False):
    """Run multi-stage screening on tickers and optionally return stage-wise results"""
    
    # Store results at each stage
    stage_results = {
        'stage_0_all': [],
        'stage_1_sma': [],
        'stage_2_volume': [],
        'stage_3_price_mcap': [],
        'stage_4_rsi': [],
        'stage_5_macd': [],
        'stage_6_momentum': []
    }
    
    total_tickers = len(tickers)
    
    # Calculate date ranges for different timeframes
    # Need enough historical data to calculate SMA 200 properly
    # Monthly: 200 months = ~16.7 years, add buffer = 18 years
    # Weekly: 200 weeks = ~3.85 years, add buffer = 5 years
    # Daily: 200 trading days = ~10 months (260 trading days/year), add buffer = 2 years
    start_date_monthly = screening_date - timedelta(days=365*18)  # 18 years for monthly SMA 200
    start_date_weekly = screening_date - timedelta(days=365*5)    # 5 years for weekly SMA 200
    start_date_daily = screening_date - timedelta(days=365*2)     # 2 years for daily SMA 200
    
    for idx, ticker in enumerate(tickers):
        progress_container.progress((idx + 1) / total_tickers, 
                                    text=f"Screening {ticker} ({idx+1}/{total_tickers})...")
        
        try:
            # Fetch multi-timeframe data
            df_monthly = fetch_stock_data(ticker, start_date_monthly, screening_date, interval='1mo')
            df_weekly = fetch_stock_data(ticker, start_date_weekly, screening_date, interval='1wk')
            df_daily = fetch_stock_data(ticker, start_date_daily, screening_date, interval='1d')
            
            # Validate minimum data requirements
            # Daily: Need at least 200 data points for SMA 200
            if df_daily is None or len(df_daily) < 200:
                continue
            
            # Weekly: Need at least 200 data points for SMA 200
            if df_weekly is None or len(df_weekly) < 200:
                continue
            
            # Monthly: Need at least 20 data points for SMA 20 (minimum requirement)
            if df_monthly is None or len(df_monthly) < 20:
                continue
            
            # Get current price
            current_price = df_daily.iloc[-1]['Close']
            
            # Record in stage 0 (all stocks with data)
            stage_results['stage_0_all'].append({
                'Ticker': ticker,
                'Price': current_price,
                'Date': screening_date
            })
            
            # Stage 1: SMA Filter
            # Daily SMAs
            df_daily['SMA_20'] = df_daily['Close'].rolling(20).mean()
            df_daily['SMA_50'] = df_daily['Close'].rolling(50).mean()
            df_daily['SMA_100'] = df_daily['Close'].rolling(100).mean()
            df_daily['SMA_200'] = df_daily['Close'].rolling(200).mean()
            
            daily_sma_20 = df_daily.iloc[-1]['SMA_20']
            daily_sma_50 = df_daily.iloc[-1]['SMA_50']
            daily_sma_100 = df_daily.iloc[-1]['SMA_100']
            daily_sma_200 = df_daily.iloc[-1]['SMA_200']
            
            daily_sma_check = (
                current_price > daily_sma_20 and
                current_price > daily_sma_50 and
                current_price > daily_sma_100 and
                current_price > daily_sma_200
            )
            
            if not daily_sma_check:
                continue
            
            # Weekly SMAs
            weekly_sma_check = True
            weekly_sma_20 = weekly_sma_50 = weekly_sma_100 = weekly_sma_200 = None
            
            if df_weekly is not None and len(df_weekly) >= 200:
                df_weekly['SMA_20'] = df_weekly['Close'].rolling(20).mean()
                df_weekly['SMA_50'] = df_weekly['Close'].rolling(50).mean()
                df_weekly['SMA_100'] = df_weekly['Close'].rolling(100).mean()
                df_weekly['SMA_200'] = df_weekly['Close'].rolling(200).mean()
                
                weekly_sma_20 = df_weekly.iloc[-1]['SMA_20']
                weekly_sma_50 = df_weekly.iloc[-1]['SMA_50']
                weekly_sma_100 = df_weekly.iloc[-1]['SMA_100']
                weekly_sma_200 = df_weekly.iloc[-1]['SMA_200']
                
                weekly_sma_check = (
                    current_price > weekly_sma_20 and
                    current_price > weekly_sma_50 and
                    current_price > weekly_sma_100 and
                    current_price > weekly_sma_200
                )
            else:
                continue
            
            if not weekly_sma_check:
                continue
            
            # Monthly SMAs
            monthly_sma_check = True
            has_monthly_sma_20 = False
            monthly_sma_20 = monthly_sma_50 = monthly_sma_100 = monthly_sma_200 = None
            
            if df_monthly is not None and len(df_monthly) >= 20:
                df_monthly['SMA_20'] = df_monthly['Close'].rolling(20).mean()
                monthly_sma_20 = df_monthly.iloc[-1]['SMA_20']
                has_monthly_sma_20 = True
                
                if pd.notna(monthly_sma_20):
                    monthly_sma_check = current_price > monthly_sma_20
                    
                    if len(df_monthly) >= 50:
                        df_monthly['SMA_50'] = df_monthly['Close'].rolling(50).mean()
                        monthly_sma_50 = df_monthly.iloc[-1]['SMA_50']
                        if pd.notna(monthly_sma_50):
                            monthly_sma_check = monthly_sma_check and (current_price > monthly_sma_50)
                    
                    if len(df_monthly) >= 100:
                        df_monthly['SMA_100'] = df_monthly['Close'].rolling(100).mean()
                        monthly_sma_100 = df_monthly.iloc[-1]['SMA_100']
                        if pd.notna(monthly_sma_100):
                            monthly_sma_check = monthly_sma_check and (current_price > monthly_sma_100)
                    
                    if len(df_monthly) >= 200:
                        df_monthly['SMA_200'] = df_monthly['Close'].rolling(200).mean()
                        monthly_sma_200 = df_monthly.iloc[-1]['SMA_200']
                        if pd.notna(monthly_sma_200):
                            monthly_sma_check = monthly_sma_check and (current_price > monthly_sma_200)
            
            if not has_monthly_sma_20 or not monthly_sma_check:
                continue
            
            # Record Stage 1 pass
            stage_results['stage_1_sma'].append({
                'Ticker': ticker,
                'Price': current_price,
                'Daily_SMA_20': daily_sma_20,
                'Daily_SMA_50': daily_sma_50,
                'Daily_SMA_100': daily_sma_100,
                'Daily_SMA_200': daily_sma_200,
                'Weekly_SMA_20': weekly_sma_20,
                'Weekly_SMA_50': weekly_sma_50,
                'Weekly_SMA_100': weekly_sma_100,
                'Weekly_SMA_200': weekly_sma_200,
                'Monthly_SMA_20': monthly_sma_20,
                'Monthly_SMA_50': monthly_sma_50,
                'Monthly_SMA_100': monthly_sma_100,
                'Monthly_SMA_200': monthly_sma_200,
                'Date': screening_date
            })
            
            # Stage 2: Volume Filter
            df_daily['Volume_SMA_20'] = df_daily['Volume'].rolling(20).mean()
            volume_sma_20 = df_daily.iloc[-1]['Volume_SMA_20']
            volume_check = volume_sma_20 > 100000
            
            if not volume_check:
                continue
            
            # Record Stage 2 pass
            stage_results['stage_2_volume'].append({
                'Ticker': ticker,
                'Price': current_price,
                'Volume_SMA_20': volume_sma_20,
                'Date': screening_date
            })
            
            # Stage 3: Price and Market Cap Filter
            if current_price >= 5000:
                continue
            
            market_cap = get_market_cap(ticker)
            if market_cap < 500:
                continue
            
            # Record Stage 3 pass
            stage_results['stage_3_price_mcap'].append({
                'Ticker': ticker,
                'Price': current_price,
                'Market_Cap_Cr': market_cap,
                'Date': screening_date
            })
            
            # Stage 4: RSI Filter
            df_daily['RSI'] = calculate_rsi(df_daily['Close'], 14)
            daily_rsi = df_daily.iloc[-1]['RSI']
            
            if pd.isna(daily_rsi) or daily_rsi <= 50:
                continue
            
            if df_weekly is not None and len(df_weekly) >= 14:
                df_weekly['RSI'] = calculate_rsi(df_weekly['Close'], 14)
                weekly_rsi = df_weekly.iloc[-1]['RSI']
                if pd.isna(weekly_rsi) or weekly_rsi <= 60:
                    continue
            else:
                continue
            
            if df_monthly is not None and len(df_monthly) >= 14:
                df_monthly['RSI'] = calculate_rsi(df_monthly['Close'], 14)
                monthly_rsi = df_monthly.iloc[-1]['RSI']
                if pd.isna(monthly_rsi) or monthly_rsi <= 60:
                    continue
            else:
                continue
            
            # Record Stage 4 pass
            stage_results['stage_4_rsi'].append({
                'Ticker': ticker,
                'Price': current_price,
                'Daily_RSI': daily_rsi,
                'Weekly_RSI': weekly_rsi,
                'Monthly_RSI': monthly_rsi,
                'Date': screening_date
            })
            
            # Stage 5: MACD Histogram Filter
            df_daily['MACD_Hist'] = calculate_macd(df_daily['Close'])
            daily_macd = df_daily.iloc[-1]['MACD_Hist']
            
            if pd.isna(daily_macd) or daily_macd <= 0:
                continue
            
            if df_weekly is not None and len(df_weekly) >= 26:
                df_weekly['MACD_Hist'] = calculate_macd(df_weekly['Close'])
                weekly_macd = df_weekly.iloc[-1]['MACD_Hist']
                if pd.isna(weekly_macd) or weekly_macd <= 0:
                    continue
            else:
                continue
            
            if df_monthly is not None and len(df_monthly) >= 26:
                df_monthly['MACD_Hist'] = calculate_macd(df_monthly['Close'])
                monthly_macd = df_monthly.iloc[-1]['MACD_Hist']
                if pd.isna(monthly_macd) or monthly_macd <= 0:
                    continue
            else:
                continue
            
            # Record Stage 5 pass
            stage_results['stage_5_macd'].append({
                'Ticker': ticker,
                'Price': current_price,
                'Daily_MACD': daily_macd,
                'Weekly_MACD': weekly_macd,
                'Monthly_MACD': monthly_macd,
                'Date': screening_date
            })
            
            # Stage 6: Calculate 15-day return
            if len(df_daily) >= 15:
                price_15_days_ago = df_daily.iloc[-15]['Close']
                return_15d = ((current_price - price_15_days_ago) / price_15_days_ago) * 100
            else:
                return_15d = 0
            
            if return_15d <= 0:
                continue
            
            # Record Stage 6 pass (final)
            stage_results['stage_6_momentum'].append({
                'Ticker': ticker,
                'Current_Price': current_price,
                'Market_Cap_Cr': market_cap,
                'Volume_SMA_20': volume_sma_20,
                'Daily_RSI': daily_rsi,
                'Weekly_RSI': weekly_rsi,
                'Monthly_RSI': monthly_rsi,
                'Daily_MACD': daily_macd,
                'Weekly_MACD': weekly_macd,
                'Monthly_MACD': monthly_macd,
                'Return_15D_Pct': return_15d,
                'Date': screening_date
            })
            
        except Exception as e:
            continue
    
    # Convert final results to DataFrame
    final_df = pd.DataFrame(stage_results['stage_6_momentum'])
    
    if show_stage_results:
        return final_df, stage_results
    else:
        return final_df


# --- Page Config ---
st.set_page_config(layout="wide", page_title="Smart Momentum Basket System", page_icon="🎯")

# --- Login Check ---
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("Please login first from the main page")
    st.stop()

st.title("🎯 Smart Momentum Basket System")
st.markdown("---")

with st.expander("📖 Strategy Description & Rules"):
    st.markdown("""
    ## Smart Momentum Basket System
    
    A comprehensive multi-stage screening and portfolio management strategy that identifies high-momentum stocks 
    and manages them with portfolio-level take-profit and stop-loss rules.
    
    ### Screening Stages
    
    1. **Multi-Timeframe SMA Filter**
       - Price must be above 20, 50, 100, and 200 SMA on Daily, Weekly, and Monthly timeframes
       - Stocks without Monthly SMA 20 or Weekly SMA 200 are eliminated
    
    2. **Volume Filter**
       - Daily Volume SMA 20 must be > 100,000
    
    3. **Price & Market Cap Filter**
       - Current Price < ₹5,000
       - Market Cap > ₹500 Crores
    
    4. **RSI Filter**
       - Monthly RSI > 60
       - Weekly RSI > 60
       - Daily RSI > 50
    
    5. **MACD Histogram Filter**
       - Monthly, Weekly, and Daily MACD Histogram must all be > 0
    
    6. **Momentum Ranking**
       - Stocks ranked by 15-day return percentage
       - Only positive returns are considered
    
    ### Portfolio Management
    
    - **Allocation**: Capital distributed proportionally based on each stock's contribution to total 15-day return
    - **Entry**: All selected stocks bought at close price on the same day
    - **Exit Rules** (T+2 compliance):
      - Take Profit: When total portfolio value reaches TP% (checked from T+2 onwards)
      - Stop Loss: When total portfolio value falls to SL% (optional, checked from T+2 onwards)
    - **Rebalancing**: On TP/SL hit, entire screener re-runs and portfolio rebalances
    
    ### Example Allocation
    
    If 6 stocks have 15-day returns of [15%, 13%, 11%, 7%, 4%, 3%]:
    - Total Return = 53%
    - Allocations = [28.3%, 24.5%, 20.7%, 13.2%, 7.5%, 5.6%]
    - For ₹1,00,000 investment: [₹28,300, ₹24,500, ₹20,700, ₹13,200, ₹7,500, ₹5,600]
    """)

# --- Inputs ---
st.markdown("### Strategy Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    TOTAL_INVESTMENT = st.number_input(
        "Total Investment (INR)", 
        value=100000.0, 
        step=10000.0,
        help="Total capital to invest across selected stocks."
    )
    TAKE_PROFIT_PCT = st.number_input(
        "Take Profit %", 
        value=10.0, 
        step=1.0,
        help="Portfolio-level take profit percentage."
    )

with col2:
    USE_STOP_LOSS = st.checkbox("Enable Stop Loss", value=True)
    STOP_LOSS_PCT = st.number_input(
        "Stop Loss %", 
        value=5.0, 
        step=1.0,
        help="Portfolio-level stop loss percentage.",
        disabled=not USE_STOP_LOSS
    )

with col3:
    CHARGES_PER_TRADE = st.number_input(
        "Charges Per Trade (Rs)", 
        value=25.0, 
        step=5.0,
        help="Transaction charges per stock trade."
    )

st.markdown("### Date Selection")

# Date Presets logic
if 'date_preset' not in st.session_state:
    st.session_state.date_preset = "1 Year"

def update_dates():
    preset = st.session_state.p5_preset_radio
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
    else:
        return
        
    st.session_state.p5_start_date = start
    st.session_state.p5_end_date = end

st.radio(
    "Quick Select Range", 
    ["7 Days", "15 Days", "30 Days", "3 Months", "6 Months", "1 Year"], 
    horizontal=True, 
    key="p5_preset_radio",
    on_change=update_dates,
    index=5
)

if 'p5_start_date' not in st.session_state:
    st.session_state.p5_start_date = datetime.today() - timedelta(days=365)
if 'p5_end_date' not in st.session_state:
    st.session_state.p5_end_date = datetime.today()

col_d1, col_d2 = st.columns(2)
with col_d1:
    START_DATE = st.date_input("Start Date", key="p5_start_date")
with col_d2:
    END_DATE = st.date_input("End Date", key="p5_end_date", max_value=datetime.today())

# --- Run Screener ---
if st.button("🔍 Run Screener and Backtest", type="primary"):
    
    # Load universe
    try:
        universe_df = pd.read_csv("bucket/universe.csv")
        all_tickers = universe_df['Ticker'].tolist()
        st.info(f"Loaded {len(all_tickers)} stocks from universe.")
    except Exception as e:
        st.error(f"Failed to load universe.csv: {e}")
        st.stop()
    
    # Step 1: Validate Tickers
    st.markdown("### Step 1: Validating Tickers")
    validation_progress = st.progress(0, text="Starting validation...")
    
    valid_tickers, invalid_tickers = validate_tickers(all_tickers, validation_progress)
    
    st.success(f"✅ Found {len(valid_tickers)} valid tickers")
    
    if invalid_tickers:
        with st.expander(f"⚠️ {len(invalid_tickers)} Invalid/Unavailable Tickers"):
            st.write(invalid_tickers)
    
    # Step 2: Run Screener on Start Date
    st.markdown(f"### Step 2: Multi-Stage Screening (Date: {START_DATE})")
    st.info(f"Running screener on data as of {START_DATE}")
    screening_progress = st.empty()
    
    qualified_stocks, stage_results = run_screener(valid_tickers, pd.to_datetime(START_DATE), screening_progress, show_stage_results=True)
    
    # Display stage-wise results
    st.markdown("#### Stage-wise Filtering Results")
    
    # Stage 0: All stocks with data
    st.markdown(f"**Stage 0: Initial Stocks** - {len(stage_results['stage_0_all'])} stocks")
    
    # Stage 1: SMA Filter
    if stage_results['stage_1_sma']:
        with st.expander(f"**Stage 1: SMA Filter** - {len(stage_results['stage_1_sma'])} stocks qualified"):
            sma_df = pd.DataFrame(stage_results['stage_1_sma'])
            sma_display_cols = ['Ticker', 'Price', 'Daily_SMA_20', 'Daily_SMA_50', 'Daily_SMA_100', 'Daily_SMA_200',
                               'Weekly_SMA_20', 'Weekly_SMA_50', 'Weekly_SMA_100', 'Weekly_SMA_200',
                               'Monthly_SMA_20', 'Monthly_SMA_50', 'Monthly_SMA_100', 'Monthly_SMA_200', 'Date']
            st.dataframe(sma_df[sma_display_cols], width='stretch')
            st.info(f"✅ {len(stage_results['stage_1_sma'])} stocks have price > all SMAs (Daily, Weekly, Monthly)")
    else:
        st.warning("❌ No stocks passed SMA filter")
        st.stop()
    
    # Stage 2: Volume Filter
    if stage_results['stage_2_volume']:
        with st.expander(f"**Stage 2: Volume Filter** - {len(stage_results['stage_2_volume'])} stocks qualified"):
            vol_df = pd.DataFrame(stage_results['stage_2_volume'])
            st.dataframe(vol_df, width='stretch')
            st.info(f"✅ {len(stage_results['stage_2_volume'])} stocks have Volume SMA 20 > 100,000")
    else:
        st.warning("❌ No stocks passed Volume filter")
        st.stop()
    
    # Stage 3: Price & Market Cap Filter
    if stage_results['stage_3_price_mcap']:
        with st.expander(f"**Stage 3: Price & Market Cap Filter** - {len(stage_results['stage_3_price_mcap'])} stocks qualified"):
            price_mcap_df = pd.DataFrame(stage_results['stage_3_price_mcap'])
            st.dataframe(price_mcap_df, width='stretch')
            st.info(f"✅ {len(stage_results['stage_3_price_mcap'])} stocks have Price < ₹5000 and Market Cap > ₹500 Cr")
    else:
        st.warning("❌ No stocks passed Price & Market Cap filter")
        st.stop()
    
    # Stage 4: RSI Filter
    if stage_results['stage_4_rsi']:
        with st.expander(f"**Stage 4: RSI Filter** - {len(stage_results['stage_4_rsi'])} stocks qualified"):
            rsi_df = pd.DataFrame(stage_results['stage_4_rsi'])
            st.dataframe(rsi_df, width='stretch')
            st.info(f"✅ {len(stage_results['stage_4_rsi'])} stocks have Monthly/Weekly RSI > 60 and Daily RSI > 50")
    else:
        st.warning("❌ No stocks passed RSI filter")
        st.stop()
    
    # Stage 5: MACD Filter
    if stage_results['stage_5_macd']:
        with st.expander(f"**Stage 5: MACD Histogram Filter** - {len(stage_results['stage_5_macd'])} stocks qualified"):
            macd_df = pd.DataFrame(stage_results['stage_5_macd'])
            st.dataframe(macd_df, width='stretch')
            st.info(f"✅ {len(stage_results['stage_5_macd'])} stocks have all MACD Histograms > 0")
    else:
        st.warning("❌ No stocks passed MACD filter")
        st.stop()
    
    # Stage 6: Momentum Filter (Final)
    if qualified_stocks.empty:
        st.error("❌ No stocks passed all screening criteria (including positive 15-day return).")
        st.stop()
    
    # Sort by 15-day return
    qualified_stocks = qualified_stocks.sort_values('Return_15D_Pct', ascending=False).reset_index(drop=True)
    
    with st.expander(f"**Stage 6: Momentum Filter (Final)** - {len(qualified_stocks)} stocks qualified", expanded=True):
        display_cols = ['Ticker', 'Current_Price', 'Market_Cap_Cr', 'Return_15D_Pct', 
                       'Daily_RSI', 'Weekly_RSI', 'Monthly_RSI', 'Daily_MACD', 'Weekly_MACD', 'Monthly_MACD', 'Date']
        st.dataframe(qualified_stocks[display_cols], width='stretch')
        st.success(f"✅ {len(qualified_stocks)} stocks have positive 15-day return!")
    

    # Step 3: Stock Selection
    st.markdown("### Step 3: Select Stocks for Portfolio")
    
    if len(qualified_stocks) < 5:
        st.warning(f"Only {len(qualified_stocks)} stocks qualified. Minimum 5 required.")
        num_stocks = len(qualified_stocks)
    else:
        num_stocks = st.number_input(
            "Number of stocks to select", 
            min_value=5, 
            max_value=len(qualified_stocks),
            value=min(10, len(qualified_stocks)),
            step=1
        )
    
    selected_stocks = qualified_stocks.head(num_stocks).copy()
    
    # Calculate allocation
    total_return = selected_stocks['Return_15D_Pct'].sum()
    selected_stocks['Contribution_Pct'] = (selected_stocks['Return_15D_Pct'] / total_return) * 100
    selected_stocks['Allocated_Amount'] = (selected_stocks['Contribution_Pct'] / 100) * TOTAL_INVESTMENT
    selected_stocks['Quantity'] = selected_stocks['Allocated_Amount'] / selected_stocks['Current_Price']
    
    st.markdown("### Portfolio Allocation")
    allocation_cols = ['Ticker', 'Return_15D_Pct', 'Contribution_Pct', 'Allocated_Amount', 'Quantity']
    st.dataframe(selected_stocks[allocation_cols], width='stretch')
    
    st.info(f"Total Allocation: ₹{selected_stocks['Allocated_Amount'].sum():,.2f}")
    
    # Store in session state for backtesting
    st.session_state.p5_screener_results = {
        'qualified_stocks': qualified_stocks,
        'selected_stocks': selected_stocks,
        'invalid_tickers': invalid_tickers,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'total_investment': TOTAL_INVESTMENT,
        'take_profit_pct': TAKE_PROFIT_PCT,
        'stop_loss_pct': STOP_LOSS_PCT if USE_STOP_LOSS else None,
        'charges_per_trade': CHARGES_PER_TRADE
    }
    
    # Step 4: Run Backtest
    st.markdown("### Step 4: Running Backtest")
    st.info("Running backtest with portfolio-level TP/SL and automatic rebalancing...")
    
    # Backtest logic
    all_cycles = []
    all_trades = []
    current_cycle = 1
    current_capital = TOTAL_INVESTMENT
    
    backtest_start = pd.to_datetime(START_DATE)
    backtest_end = pd.to_datetime(END_DATE)
    
    # Create date range for backtesting
    date_range = pd.date_range(start=backtest_start, end=backtest_end, freq='D')
    
    # Track current portfolio
    current_portfolio = None
    entry_date = None
    
    for current_date in date_range:
        # Skip weekends
        if current_date.weekday() >= 5:
            continue
        
        # If no portfolio, run screener and enter
        if current_portfolio is None:
            # Run screener for this date
            screener_results = run_screener(valid_tickers, current_date, st.empty())
            
            if screener_results.empty or len(screener_results) < 5:
                continue
            
            # Sort and select top stocks
            screener_results = screener_results.sort_values('Return_15D_Pct', ascending=False).reset_index(drop=True)
            selected = screener_results.head(num_stocks).copy()
            
            # Calculate allocation
            total_ret = selected['Return_15D_Pct'].sum()
            if total_ret <= 0:
                continue
            
            selected['Contribution_Pct'] = (selected['Return_15D_Pct'] / total_ret) * 100
            selected['Allocated_Amount'] = (selected['Contribution_Pct'] / 100) * current_capital
            selected['Entry_Price'] = selected['Current_Price']
            selected['Quantity'] = selected['Allocated_Amount'] / selected['Entry_Price']
            
            # Enter portfolio
            current_portfolio = selected.copy()
            entry_date = current_date
            
            # Record trades
            for _, stock in current_portfolio.iterrows():
                all_trades.append({
                    'Cycle': current_cycle,
                    'Ticker': stock['Ticker'],
                    'Entry_Date': entry_date,
                    'Entry_Price': stock['Entry_Price'],
                    'Quantity': stock['Quantity'],
                    'Invested': stock['Allocated_Amount'],
                    'Exit_Date': None,
                    'Exit_Price': None,
                    'PnL': None,
                    'Status': 'Open'
                })
            
            continue
        
        # Check exit conditions (only after T+2)
        days_held = (current_date - entry_date).days
        
        if days_held >= 2:
            # Fetch current prices for all stocks in portfolio
            portfolio_value = 0
            exit_prices = {}
            
            for _, stock in current_portfolio.iterrows():
                ticker = stock['Ticker']
                # Fetch current price
                df = fetch_stock_data(ticker, current_date - timedelta(days=5), current_date)
                if df is not None and len(df) > 0:
                    current_price = df.iloc[-1]['Close']
                    exit_prices[ticker] = current_price
                    portfolio_value += stock['Quantity'] * current_price
                else:
                    # If can't fetch, use entry price
                    exit_prices[ticker] = stock['Entry_Price']
                    portfolio_value += stock['Quantity'] * stock['Entry_Price']
            
            # Calculate total charges
            total_charges = len(current_portfolio) * CHARGES_PER_TRADE
            net_portfolio_value = portfolio_value - total_charges
            
            # Check TP/SL
            tp_value = current_capital * (1 + TAKE_PROFIT_PCT / 100)
            sl_value = current_capital * (1 - STOP_LOSS_PCT / 100) if USE_STOP_LOSS else 0
            
            exit_triggered = False
            exit_reason = None
            
            if net_portfolio_value >= tp_value:
                exit_triggered = True
                exit_reason = "TAKE PROFIT"
            elif USE_STOP_LOSS and net_portfolio_value <= sl_value:
                exit_triggered = True
                exit_reason = "STOP LOSS"
            
            if exit_triggered:
                # Exit all positions
                cycle_pnl = 0
                
                for i, (_, stock) in enumerate(current_portfolio.iterrows()):
                    ticker = stock['Ticker']
                    exit_price = exit_prices.get(ticker, stock['Entry_Price'])
                    pnl = (exit_price - stock['Entry_Price']) * stock['Quantity'] - CHARGES_PER_TRADE
                    cycle_pnl += pnl
                    
                    # Update trade record
                    for trade in all_trades:
                        if (trade['Cycle'] == current_cycle and 
                            trade['Ticker'] == ticker and 
                            trade['Status'] == 'Open'):
                            trade['Exit_Date'] = current_date
                            trade['Exit_Price'] = exit_price
                            trade['PnL'] = pnl
                            trade['Status'] = 'Closed'
                            break
                
                # Record cycle
                all_cycles.append({
                    'Cycle': current_cycle,
                    'Entry_Date': entry_date,
                    'Exit_Date': current_date,
                    'Exit_Reason': exit_reason,
                    'Investment': current_capital,
                    'Final_Value': net_portfolio_value,
                    'PnL': cycle_pnl,
                    'Duration_Days': days_held,
                    'Num_Stocks': len(current_portfolio)
                })
                
                # Update capital for next cycle
                current_capital = TOTAL_INVESTMENT  # Reset to initial investment
                current_cycle += 1
                current_portfolio = None
                entry_date = None
    
    # Handle open positions at end
    if current_portfolio is not None:
        st.markdown("### Open Trades")
        st.info(f"Portfolio entered on {entry_date.date()} is still open at backtest end date.")
        
        # Fetch final prices
        for i, (_, stock) in enumerate(current_portfolio.iterrows()):
            ticker = stock['Ticker']
            df = fetch_stock_data(ticker, backtest_end - timedelta(days=5), backtest_end)
            if df is not None and len(df) > 0:
                current_portfolio.at[i, 'Current_Price'] = df.iloc[-1]['Close']
                current_portfolio.at[i, 'Current_Value'] = df.iloc[-1]['Close'] * stock['Quantity']
                current_portfolio.at[i, 'Unrealized_PnL'] = (df.iloc[-1]['Close'] - stock['Entry_Price']) * stock['Quantity']
        
        open_cols = ['Ticker', 'Entry_Price', 'Quantity', 'Invested', 'Current_Price', 'Current_Value', 'Unrealized_PnL']
        st.dataframe(current_portfolio[open_cols], width='stretch')
        
        total_invested = current_portfolio['Invested'].sum()
        total_current = current_portfolio['Current_Value'].sum()
        total_unrealized = current_portfolio['Unrealized_PnL'].sum()
        
        st.metric("Total Invested", f"₹{total_invested:,.2f}")
        st.metric("Current Value", f"₹{total_current:,.2f}")
        st.metric("Unrealized PnL", f"₹{total_unrealized:,.2f}", 
                 delta_color="normal" if total_unrealized >= 0 else "inverse")
    
    # Display Results
    if all_cycles:
        st.markdown("---")
        st.markdown("### Backtest Results")
        
        cycles_df = pd.DataFrame(all_cycles)
        trades_df = pd.DataFrame([t for t in all_trades if t['Status'] == 'Closed'])
        
        # Overall Metrics
        total_cycles = len(cycles_df)
        total_pnl = cycles_df['PnL'].sum()
        total_trades = len(trades_df)
        avg_cycle_duration = cycles_df['Duration_Days'].mean()
        
        # Win rate
        winning_cycles = len(cycles_df[cycles_df['PnL'] > 0])
        win_rate = (winning_cycles / total_cycles * 100) if total_cycles > 0 else 0
        
        # XIRR calculation
        cashflows = []
        for _, cycle in cycles_df.iterrows():
            cashflows.append((cycle['Entry_Date'], -cycle['Investment']))
            cashflows.append((cycle['Exit_Date'], cycle['Final_Value']))
        
        cashflow_df = pd.DataFrame(cashflows, columns=['date', 'cashflow']).sort_values('date')
        xirr = calculate_xirr(cashflow_df)
        
        # Display metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Cycles", total_cycles)
        col_m2.metric("Total PnL", f"₹{total_pnl:,.2f}", delta_color="normal" if total_pnl >= 0 else "inverse")
        col_m3.metric("Win Rate", f"{win_rate:.1f}%")
        col_m4.metric("XIRR", f"{xirr * 100:.2f}%")
        
        col_m5, col_m6, col_m7 = st.columns(3)
        col_m5.metric("Total Trades", total_trades)
        col_m6.metric("Avg Cycle Duration", f"{avg_cycle_duration:.1f} days")
        col_m7.metric("ROI", f"{(total_pnl / TOTAL_INVESTMENT * 100):.2f}%")
        
        # Download Analytics
        download_data = {
            "Strategy Name": ["Smart Momentum Basket System"],
            "Start Date": [START_DATE],
            "End Date": [END_DATE],
            "Total Investment": [TOTAL_INVESTMENT],
            "Take Profit %": [TAKE_PROFIT_PCT],
            "Stop Loss %": [STOP_LOSS_PCT if USE_STOP_LOSS else "N/A"],
            "Charges Per Trade": [CHARGES_PER_TRADE],
            "Total Cycles": [total_cycles],
            "Total Trades": [total_trades],
            "Total PnL": [total_pnl],
            "Win Rate": [f"{win_rate:.2f}%"],
            "XIRR": [f"{xirr * 100:.2f}%"],
            "ROI": [f"{(total_pnl / TOTAL_INVESTMENT * 100):.2f}%"],
            "Avg Cycle Duration": [f"{avg_cycle_duration:.1f} days"]
        }
        download_df = pd.DataFrame(download_data)
        csv_data = download_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Analytics as CSV",
            data=csv_data,
            file_name=f"smart_momentum_basket_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # Cycle-wise Analytics
        st.subheader("Cycle-wise Performance")
        cycles_display = cycles_df.copy()
        cycles_display['Entry_Date'] = cycles_display['Entry_Date'].dt.date
        cycles_display['Exit_Date'] = cycles_display['Exit_Date'].dt.date
        st.dataframe(cycles_display, width='stretch')
        
        # Trade History
        st.subheader("Trade History")
        trades_display = trades_df.copy()
        trades_display['Entry_Date'] = pd.to_datetime(trades_display['Entry_Date']).dt.date
        trades_display['Exit_Date'] = pd.to_datetime(trades_display['Exit_Date']).dt.date
        trade_cols = ['Cycle', 'Ticker', 'Entry_Date', 'Entry_Price', 'Quantity', 
                     'Exit_Date', 'Exit_Price', 'PnL']
        st.dataframe(trades_display[trade_cols], width='stretch')
        
        # Charts
        st.markdown("---")
        
        # Equity Curve
        st.subheader("Equity Curve")
        cycles_df['Cumulative_PnL'] = cycles_df['PnL'].cumsum()
        equity_data = cycles_df[['Exit_Date', 'Cumulative_PnL']].copy()
        equity_data = equity_data.rename(columns={'Exit_Date': 'Date', 'Cumulative_PnL': 'Cumulative PnL'})
        st.line_chart(equity_data, x='Date', y='Cumulative PnL')
        
        # Cycle PnL Distribution
        st.subheader("Cycle PnL Distribution")
        cycles_df['Cycle_Label'] = 'Cycle ' + cycles_df['Cycle'].astype(str)
        st.bar_chart(cycles_df, x='Cycle_Label', y='PnL')
        
    else:
        st.warning("No complete cycles were executed during the backtest period.")
    
    st.success("✅ Backtest completed!")
