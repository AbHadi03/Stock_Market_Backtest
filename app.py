import streamlit as st
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Core.functions import login

st.set_page_config(layout="wide", page_title="Stock Market Strategy", page_icon="📈")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login(st)
else:
    st.title("Stock Market Strategy")
    st.markdown("---")
    
    st.markdown("""
    ### Welcome to the Stock Market Strategy Dashboard
    
    This application allows you to backtest and analyze various stock market strategies.
    
    #### Available Strategies:
    
    **1. Equity RSI Trade**
    - A Relativ Strength Index (RSI) based strategy.
    - Buys when RSI drops below a certain threshold (oversold).
    - Sells one a target percentage is reached.
    - Calculates P&L, Win Rate, and XIRR.

    **2. Equity RSI Trade (Stock-wise)**
    - Same RSI strategy but focused on single stock analysis.
    - Fetches live data from Yahoo Finance.
    - Allows custom date range backtesting.
    - Provides detailed trade history and capital deployment analysis.
    
    #### How it works:
    1. Navigate to the strategy page from the sidebar.
    2. Configure the strategy parameters (Investment amount, RSI period, etc.).
    3. Select the data folder.
    4. Run the backtest to see performance analytics including Equity Curve and Capital Deployment.
    
    Use the sidebar to navigate to specific strategies.
    """)
    
    st.sidebar.success(f"Logged in as {st.session_state.username}")
