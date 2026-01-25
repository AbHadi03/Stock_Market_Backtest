App Dashboard Link : https://equityanalyzer.streamlit.app/

====================================================
RSI EQUITY STRATEGY BACKTEST & ANALYSIS (PYTHON)
====================================================

Author : Nitin Joshi (Being System Trader)
Purpose : Strategy research, testing & portfolio capital analysis
Language : Python

----------------------------------------------------
INTRODUCTION
----------------------------------------------------
This folder contains a Python-based backtesting engine
designed to test a mean-reversion RSI equity strategy
across multiple stocks or indices.

This is NOT a signal provider.
It is a research and evaluation tool to understand:
- Strategy profitability
- Capital usage across portfolio
- Risk exposure over time
- Trade consistency and holding periods
- Portfolio-level performance metrics

----------------------------------------------------
STRATEGY DETAILS (CORE LOGIC)
----------------------------------------------------
This strategy uses the Relative Strength Index (RSI)
to detect oversold conditions and enter long trades.

----------------------------------------------------
1) INDICATOR USED
----------------------------------------------------
- RSI (Relative Strength Index)
- RSI Period = User Input (Default: 14)

----------------------------------------------------
2) ENTRY CONDITION
----------------------------------------------------
A BUY trade is triggered when:
- RSI falls below the RSI Buy Level (Default: 30)
- Entry is taken at the closing price of the same day

----------------------------------------------------
3) POSITION SIZING
----------------------------------------------------
- A fixed investment per trade is used (Default: ₹10,000)
- Quantity = Investment ÷ Entry Price
- Multiple trades across stocks can run simultaneously

----------------------------------------------------
4) EXIT CONDITION
----------------------------------------------------
- Exit occurs when price reaches a fixed percentage target
- Default Target = 5%
- No stop-loss is applied in this version

----------------------------------------------------
5) CHARGES
----------------------------------------------------
- A fixed brokerage/charge per trade is deducted
- Default = ₹50 per completed trade
- Net PnL reflects realistic trading costs

----------------------------------------------------
CAPITAL & PORTFOLIO LOGIC
----------------------------------------------------
This system tracks portfolio-level deployed capital,
not just stock-level exposure.

Capital Deployment Rules:
- Capital = Number of Open Trades × Investment Per Trade
- Tracks real-time portfolio capital usage
- Generates a capital timeline CSV
- Calculates maximum capital deployed during backtest

----------------------------------------------------
STRATEGY CHARACTERISTICS
----------------------------------------------------
- Swing / positional trading
- Mean-reversion logic
- Multiple overlapping trades allowed
- Portfolio-level capital tracking
- Suitable for:
- NSE Stocks
- ETFs
- Index instruments

----------------------------------------------------
USER-ADJUSTABLE PARAMETERS
----------------------------------------------------
Modify these inside the script:

INVESTMENT_PER_TRADE = 10000
TARGET_PCT = 0.05
RSI_PERIOD = 14
RSI_BUY_LEVEL = 30
CHARGES_PER_TRADE = 50
CSV_FOLDER = "data_nse500"

You can experiment with:
- Different RSI levels
- Different targets
- Different capital sizing
- Different stock universes

----------------------------------------------------
FOLDER STRUCTURE
----------------------------------------------------
main.py
→ Core RSI backtest engine

data_nse500/
→ Place NSE stock CSV files here

trade_log.csv
→ Completed trades report

capital_timeline.csv
→ Portfolio capital usage timeline

open_positions.csv
→ Open trades snapshot (if any)

----------------------------------------------------
REQUIREMENTS
----------------------------------------------------
- Python 3.9+ (Recommended Python 3.12)
- Libraries:
- pandas
- numpy
- os

Install dependencies:
pip install pandas numpy

----------------------------------------------------
HOW TO USE
----------------------------------------------------
Step 1:
Place historical stock CSV files inside:
data_nse500/

Step 2:
Run the script:
python main.py

Step 3:
View generated output files:
- trade_log.csv
- capital_timeline.csv
- open_positions.csv

----------------------------------------------------
OUTPUT FILES EXPLAINED
----------------------------------------------------

trade_log.csv
- Entry & exit dates
- Entry & exit prices
- Quantity traded
- Net PnL
- Charges paid
- Holding period
- Full trade transparency

capital_timeline.csv
- Date-wise portfolio capital deployed
- Used to compute maximum capital usage
- Helps analyze portfolio risk

open_positions.csv
- Lists trades still open at backtest end
- Shows unrealized positions snapshot

----------------------------------------------------
PERFORMANCE METRICS DISPLAYED
----------------------------------------------------
The script prints:
- Total trades executed
- Total charges paid
- Net PnL
- Win rate
- Average holding period
- Maximum capital deployed
- Portfolio XIRR (Return)

----------------------------------------------------
IMPORTANT NOTES
----------------------------------------------------
- For educational & research purposes only
- No guarantee of profitability
- Performance depends on:
- Market regime
- Parameter tuning
- Data quality

DO NOT deploy live without forward testing.

----------------------------------------------------
WHO THIS IS FOR
----------------------------------------------------
✔ Traders learning systematic trading
✔ Python strategy builders
✔ Portfolio researchers
✔ Data-driven traders
✔ Students of algorithmic trading

----------------------------------------------------
WHO THIS IS NOT FOR
----------------------------------------------------
✘ Signal hunters
✘ Guaranteed profit seekers
✘ Plug-and-play traders

----------------------------------------------------
LICENSE & USAGE
----------------------------------------------------
- Personal & educational use only
- Redistribution or resale prohibited
- Modification allowed for learning
- Provided AS-IS

----------------------------------------------------
FINAL NOTE
----------------------------------------------------
This project is designed to help you think like a
system trader — structured, disciplined,
portfolio-aware, and data-driven.

— Nitin Joshi
Being System Trader

====================================================
