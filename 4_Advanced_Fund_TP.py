from os.path import abspath, join, dirname
from sys import path, exc_info

base_dir = abspath(join(dirname(__file__), "../"))
path.append(base_dir)

import streamlit as st
import pandas as pd
import ccxt
import ccxt.async_support as ccxt_async
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import asyncio
from typing import Dict

st.set_page_config(layout="wide", page_title="Advanced Fund TP", page_icon="💰")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.error("Please login first from the main page")
    st.stop()

_exchange_pool = None

class ExchangePool:
    def __init__(self):
        self.exchanges = {}
    async def get_exchange(self, exchange_name: str, config=None):
        key = f"{exchange_name}_{str(config)}"
        if key not in self.exchanges:
            exchange_class = getattr(ccxt_async, exchange_name.lower())
            self.exchanges[key] = exchange_class(config or {'enableRateLimit': True})
        return self.exchanges[key]

def get_exchange_pool():
    global _exchange_pool
    if _exchange_pool:
        return _exchange_pool
    _exchange_pool = ExchangePool()
    return _exchange_pool

async def calculate_specific_fee(exchange_name: str, symbol: str, volume_usd: float) -> Dict:
    try:
        symbol_pair = f"{symbol}/USDT"
        exchange_pool = get_exchange_pool()
        exchange = await exchange_pool.get_exchange(exchange_name.lower())
        markets = await exchange.fetch_markets()
        market = next((m for m in markets if m['symbol'] == symbol_pair), None)
        
        if not market:
            return {'success': False, 'Error': f"Pair {symbol_pair} not supported on {exchange_name}"}
        
        return {
            'success': True,
            'data': {
                'timestamp': int(time.time() * 1000),
                'exchange': exchange_name,
                'symbol': symbol,
                'volume_usd': volume_usd,
                'maker_fee_usd': volume_usd * market['maker'],
                'taker_fee_usd': volume_usd * market['taker']
            }
        }
    except Exception as e:
        return {'success': False, 'Error': str(e)}

class CryptoBacktesterFundTP:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        
    def fetch_crypto_data(self, symbol, start_date, end_date):
        """Fetch crypto historical data from Binance"""
        try:
            start_ts = int(start_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1d', start_ts, limit=1000)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            
            return df.reset_index(drop=True)
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    def calculate_liquidation_price(self, entry_price, leverage, side='long'):
        """Calculate liquidation price for given leverage"""
        if side == 'long':
            return entry_price * (1 - 0.99/leverage)
        else:  # short
            return entry_price * (1 + 0.99/leverage)
    
    def calculate_take_profit_price(self, entry_price, leverage, take_profit_percent, trade_type):
        """Calculate take profit price based on investment profit percentage"""
        price_change_needed = take_profit_percent / leverage / 100
        
        if trade_type == 'long':
            return entry_price * (1 + price_change_needed)
        else:  # short
            return entry_price * (1 - price_change_needed)
    
    def run_backtest(self, df, start_date, leverage, take_profit_percent, stop_loss_percent, subsequent_multiplier, user_entry_price, initial_investment=100000, trade_type='long'):
        """Run the backtesting strategy with fund-based take profit"""
        results = []
        current_investment = initial_investment
        position_open = False
        entry_date = None
        liquidation_price = 0
        take_profit_price = 0
        total_trades = 0
        winning_trades = 0
        total_pnl = 0
        waiting_for_entry = True
        actual_entry_price = user_entry_price
        
        start_idx = df[df['timestamp'].dt.date >= start_date.date()].index[0] if len(df[df['timestamp'].dt.date >= start_date.date()]) > 0 else 0
        
        for i in range(start_idx, len(df)):
            current_date = df.iloc[i]['timestamp']
            current_price = df.iloc[i]['close']
            high_price = df.iloc[i]['high']
            low_price = df.iloc[i]['low']
            
            if not position_open:
                if waiting_for_entry:
                    if low_price <= user_entry_price <= high_price:
                        actual_entry_price = user_entry_price
                        waiting_for_entry = False
                    else:
                        continue
                else:
                    actual_entry_price = current_price
                
                entry_date = current_date
                liquidation_price = self.calculate_liquidation_price(actual_entry_price, leverage, trade_type)
                take_profit_price = self.calculate_take_profit_price(actual_entry_price, leverage, take_profit_percent, trade_type)
                
                if trade_type == 'long':
                    stop_loss_price = actual_entry_price * (1 - (stop_loss_percent / 100) / leverage)
                else:  # short
                    stop_loss_price = actual_entry_price * (1 + (stop_loss_percent / 100) / leverage)
                
                position_open = True
                entry_fee = current_investment * leverage * 0.0004
                
                action_name = 'BUY' if trade_type == 'long' else 'SHORT'
                
                results.append({
                    'date': current_date,
                    'action': action_name,
                    'price': actual_entry_price,
                    'investment': current_investment,
                    'leverage': leverage,
                    'liquidation_price': liquidation_price,
                    'take_profit_price': take_profit_price,
                    'quantity': (current_investment * leverage) / actual_entry_price,
                    'pnl': 0,
                    'cumulative_pnl': total_pnl,
                    'entry_fee': entry_fee,
                    'exit_fee': 0,
                    'entry_date': current_date
                })
                
            else:
                # Check conditions based on trade type
                if trade_type == 'long':
                    # LONG logic
                    if low_price <= stop_loss_price:
                        total_trades += 1
                        loss_amount = current_investment * (stop_loss_percent / 100)
                        pnl = -loss_amount
                        total_pnl += pnl
                        
                        quantity = (current_investment * leverage) / actual_entry_price
                        exit_fee = (quantity * stop_loss_price) * 0.0004
                        trade_duration = (current_date - entry_date).days
                        
                        results.append({
                            'date': current_date,
                            'action': 'STOP LOSS',
                            'price': stop_loss_price,
                            'investment': current_investment,
                            'leverage': leverage,
                            'liquidation_price': liquidation_price,
                            'take_profit_price': take_profit_price,
                            'quantity': quantity,
                            'pnl': pnl,
                            'cumulative_pnl': total_pnl,
                            'entry_fee': 0,
                            'exit_fee': exit_fee,
                            'trade_duration_days': trade_duration
                        })
                        
                        current_investment *= subsequent_multiplier
                        position_open = False
                        
                    elif low_price <= liquidation_price:
                        total_trades += 1
                        pnl = -current_investment
                        total_pnl += pnl
                        
                        trade_duration = (current_date - entry_date).days
                        
                        results.append({
                            'date': current_date,
                            'action': 'LIQUIDATED',
                            'price': liquidation_price,
                            'investment': current_investment,
                            'leverage': leverage,
                            'liquidation_price': liquidation_price,
                            'take_profit_price': take_profit_price,
                            'quantity': (current_investment * leverage) / actual_entry_price,
                            'pnl': pnl,
                            'cumulative_pnl': total_pnl,
                            'entry_fee': 0,
                            'exit_fee': 0,
                            'trade_duration_days': trade_duration
                        })
                        
                        current_investment *= subsequent_multiplier
                        position_open = False
                        
                    elif high_price >= take_profit_price:
                        total_trades += 1
                        winning_trades += 1
                        
                        quantity = (current_investment * leverage) / actual_entry_price
                        profit = current_investment * (take_profit_percent / 100)
                        
                        entry_fee = current_investment * leverage * 0.0004
                        exit_fee = (quantity * take_profit_price) * 0.0004
                        total_fees = entry_fee + exit_fee
                        
                        pnl = profit - total_fees
                        total_pnl += pnl
                        
                        trade_duration = (current_date - entry_date).days
                        
                        results.append({
                            'date': current_date,
                            'action': 'SELL (TP)',
                            'price': take_profit_price,
                            'investment': current_investment,
                            'leverage': leverage,
                            'liquidation_price': liquidation_price,
                            'take_profit_price': take_profit_price,
                            'quantity': quantity,
                            'pnl': pnl,
                            'cumulative_pnl': total_pnl,
                            'entry_fee': 0,
                            'exit_fee': exit_fee,
                            'trade_duration_days': trade_duration
                        })
                        
                        current_investment = initial_investment
                        position_open = False
                
                else:  # SHORT logic
                    if high_price >= stop_loss_price:
                        total_trades += 1
                        loss_amount = current_investment * (stop_loss_percent / 100)
                        pnl = -loss_amount
                        total_pnl += pnl
                        
                        quantity = (current_investment * leverage) / actual_entry_price
                        exit_fee = (quantity * stop_loss_price) * 0.0004
                        trade_duration = (current_date - entry_date).days
                        
                        results.append({
                            'date': current_date,
                            'action': 'STOP LOSS',
                            'price': stop_loss_price,
                            'investment': current_investment,
                            'leverage': leverage,
                            'liquidation_price': liquidation_price,
                            'take_profit_price': take_profit_price,
                            'quantity': quantity,
                            'pnl': pnl,
                            'cumulative_pnl': total_pnl,
                            'entry_fee': 0,
                            'exit_fee': exit_fee,
                            'trade_duration_days': trade_duration
                        })
                        
                        current_investment *= subsequent_multiplier
                        position_open = False
                        
                    elif high_price >= liquidation_price:
                        total_trades += 1
                        pnl = -current_investment
                        total_pnl += pnl
                        
                        trade_duration = (current_date - entry_date).days
                        
                        results.append({
                            'date': current_date,
                            'action': 'LIQUIDATED',
                            'price': liquidation_price,
                            'investment': current_investment,
                            'leverage': leverage,
                            'liquidation_price': liquidation_price,
                            'take_profit_price': take_profit_price,
                            'quantity': (current_investment * leverage) / actual_entry_price,
                            'pnl': pnl,
                            'cumulative_pnl': total_pnl,
                            'entry_fee': 0,
                            'exit_fee': 0,
                            'trade_duration_days': trade_duration
                        })
                        
                        current_investment *= subsequent_multiplier
                        position_open = False
                        
                    elif low_price <= take_profit_price:
                        total_trades += 1
                        winning_trades += 1
                        
                        quantity = (current_investment * leverage) / actual_entry_price
                        profit = current_investment * (take_profit_percent / 100)
                        
                        entry_fee = current_investment * leverage * 0.0004
                        exit_fee = (quantity * take_profit_price) * 0.0004
                        total_fees = entry_fee + exit_fee
                        
                        pnl = profit - total_fees
                        total_pnl += pnl
                        
                        trade_duration = (current_date - entry_date).days
                        
                        results.append({
                            'date': current_date,
                            'action': 'COVER (TP)',
                            'price': take_profit_price,
                            'investment': current_investment,
                            'leverage': leverage,
                            'liquidation_price': liquidation_price,
                            'take_profit_price': take_profit_price,
                            'quantity': quantity,
                            'pnl': pnl,
                            'cumulative_pnl': total_pnl,
                            'entry_fee': 0,
                            'exit_fee': exit_fee,
                            'trade_duration_days': trade_duration
                        })
                        
                        current_investment = initial_investment
                        position_open = False
        
        if position_open:
            final_price = df.iloc[-1]['close']
            quantity = (current_investment * leverage) / actual_entry_price
            
            if trade_type == 'long':
                unrealized_pnl = (final_price - actual_entry_price) * quantity
            else:  # short
                unrealized_pnl = (actual_entry_price - final_price) * quantity
            
            results.append({
                'date': df.iloc[-1]['timestamp'],
                'action': 'OPEN POSITION',
                'price': final_price,
                'investment': current_investment,
                'leverage': leverage,
                'liquidation_price': liquidation_price,
                'take_profit_price': take_profit_price,
                'quantity': quantity,
                'pnl': 0,
                'unrealized_pnl': unrealized_pnl,
                'cumulative_pnl': total_pnl
            })
        
        take_profit_hits = len([r for r in results if r['action'] in ['SELL (TP)', 'COVER (TP)']])
        cycles_completed = take_profit_hits
        
        return pd.DataFrame(results), {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'final_investment_size': current_investment,
            'position_open': position_open,
            'cycles_completed': cycles_completed,
            'take_profit_hits': take_profit_hits
        }

# UI
trade_type = st.sidebar.selectbox(
    "Trade Type",
    options=['long', 'short'],
    index=0
)

if trade_type == 'long':
    st.header("💰 Advanced Fund TP Long - Crypto Backtesting System")
else:
    st.header("💰 Advanced Fund TP Short - Crypto Backtesting System")

st.markdown("---")

st.sidebar.header("📊 Backtest Parameters")

symbol = st.sidebar.selectbox(
    "Select Symbol",
    options=['ETH/USDT', 'BTC/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT'],
    index=0
)

coin_name = symbol.split('/')[0]

start_date = st.sidebar.date_input(
    "Start Date",
    value=datetime(2024, 1, 1),
    min_value=datetime(2020, 1, 1),
    max_value=datetime.now()
)

entry_price = st.sidebar.number_input(
    "Entry Price (USD)",
    min_value=0.01,
    value=1.0,
    help=f"Bot will wait for this price to be hit before starting {trade_type.upper()} trades"
)

leverage = st.sidebar.selectbox(
    "Leverage",
    options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    index=4
)

initial_investment = st.sidebar.number_input(
    "Initial Investment (USD)",
    min_value=1,
    max_value=1000000,
    value=100000,
    step=1000
)

subsequent_multiplier = st.sidebar.number_input(
    "Subsequent Multiplier (after loss)",
    min_value=1.0,
    max_value=10.0,
    value=2.0,
    step=0.1
)

take_profit_percent = st.sidebar.number_input(
    "Take Profit Percentage (% of Investment)",
    min_value=10.0,
    value=100.0,
    step=10.0,
    help="Profit percentage on your investment amount"
)

stop_loss_percent = st.sidebar.number_input(
    "Stop Loss Percentage (% of investment)",
    min_value=1.0,
    max_value=100.0,
    value=100.0,
    step=1.0,
    help="100% means liquidation price"
)

if st.sidebar.button("🔄 Run Backtest", type="primary"):
    if entry_price <= 0:
        st.error("Please enter a valid entry price")
        st.stop()
        
    with st.spinner("Fetching data and running backtest..."):
        backtester = CryptoBacktesterFundTP()
        
        end_date = datetime.now()
        df = backtester.fetch_crypto_data(symbol, datetime.combine(start_date, datetime.min.time()), end_date)
        
        if df is not None and len(df) > 0:
            results_df, summary = backtester.run_backtest(
                df, 
                datetime.combine(start_date, datetime.min.time()), 
                leverage,
                take_profit_percent,
                stop_loss_percent,
                subsequent_multiplier,
                entry_price,
                initial_investment,
                trade_type
            )
            
            st.session_state.results_df = results_df
            st.session_state.summary = summary
            st.session_state.price_df = df
            st.session_state.backtest_params = {
                'start_date': start_date,
                'leverage': leverage,
                'take_profit_percent': take_profit_percent,
                'subsequent_multiplier': subsequent_multiplier,
                'initial_investment': initial_investment,
                'symbol': symbol,
                'coin_name': coin_name,
                'trade_type': trade_type
            }

# Display results if available
if 'results_df' in st.session_state:
    results_df = st.session_state.results_df
    summary = st.session_state.summary
    price_df = st.session_state.price_df
    params = st.session_state.backtest_params
    
    st.header("📈 Backtest Summary (This Back Testing Generated on 1 day price candle)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", summary['total_trades'])
    with col2:
        st.metric("Win Rate", f"{summary['win_rate']:.1f}%")
    with col3:
        st.metric("Total P&L", f"${summary['total_pnl']:,.2f}")
    with col4:
        roi = (summary['total_pnl'] / params['initial_investment']) * 100
        st.metric("ROI", f"{roi:.1f}%")
    
    col5, col6, col7, col8, col9 = st.columns(5)
    
    with col5:
        st.metric("Current Investment", f"${summary['final_investment_size']:,.0f}")
    with col6:
        current_quantity = results_df.iloc[-1]['quantity'] if summary['position_open'] and not results_df.empty else 0
        st.metric("Current Quantity", f"{current_quantity:.4f} {params['coin_name']}")
    with col7:
        st.metric("Position Status", "Open" if summary['position_open'] else "Closed")
    with col8:
        if summary['position_open'] and 'unrealized_pnl' in results_df.columns:
            unrealized = results_df.iloc[-1]['unrealized_pnl']
            st.metric("Unrealized P&L", f"${unrealized:,.2f}")
    with col9:
        current_price = price_df.iloc[-1]['close']
        st.metric(f"Current {params['coin_name']} Price", f"${current_price:,.2f}")
    
    # Trade Duration Analytics
    st.subheader("📊 Trade Duration Analytics")
    if not results_df.empty and 'action' in results_df.columns:
        completed_trades = results_df[results_df['action'].isin(['STOP LOSS', 'LIQUIDATED', 'SELL (TP)', 'COVER (TP)'])]
        if not completed_trades.empty:
            trades_list = list(completed_trades.iterrows())
            for row_start in range(0, len(trades_list), 5):
                row_trades = trades_list[row_start:row_start + 5]
                duration_cols = st.columns(len(row_trades))
                for col_idx, (_, trade) in enumerate(row_trades):
                    with duration_cols[col_idx]:
                        trade_num = row_start + col_idx + 1
                        st.metric(f"Trade {trade_num} ({trade['action']})", f"{trade['trade_duration_days']} days")
    else:
        st.write("No completed trades yet.")
    
    # Fee Analytics
    st.subheader("💰 Fee Analytics")
    total_entry_fees = results_df['entry_fee'].sum()
    total_exit_fees = results_df['exit_fee'].sum()
    total_fees = total_entry_fees + total_exit_fees
    
    fee_col1, fee_col2, fee_col3 = st.columns(3)
    with fee_col1:
        entry_label = "Total Buy Fees" if params['trade_type'] == 'long' else "Total Short Fees"
        st.metric(entry_label, f"${total_entry_fees:,.2f}")
    with fee_col2:
        exit_label = "Total Sell Fees" if params['trade_type'] == 'long' else "Total Cover Fees"
        st.metric(exit_label, f"${total_exit_fees:,.2f}")
    with fee_col3:
        st.metric("Total Fees Paid", f"${total_fees:,.2f}")
    
    # Price chart with trades
    chart_title = f'{params["symbol"]} Price with {params["trade_type"].upper()} Trades (Fund TP)'
    st.header("📊 Price Chart with Trades")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(chart_title, 'Cumulative P&L'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    fig.add_trace(
        go.Scatter(
            x=price_df['timestamp'],
            y=price_df['close'],
            mode='lines',
            name=f'{params["coin_name"]} Price',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    if not results_df.empty and 'action' in results_df.columns:
        if params['trade_type'] == 'long':
            entry_trades = results_df[results_df['action'] == 'BUY']
            tp_trades = results_df[results_df['action'] == 'SELL (TP)']
            entry_color = 'green'
            tp_color = 'blue'
            entry_symbol = 'triangle-up'
            tp_symbol = 'triangle-down'
            entry_name = 'Buy'
            tp_name = 'Take Profit'
        else:
            entry_trades = results_df[results_df['action'] == 'SHORT']
            tp_trades = results_df[results_df['action'] == 'COVER (TP)']
            entry_color = 'red'
            tp_color = 'green'
            entry_symbol = 'triangle-down'
            tp_symbol = 'triangle-up'
            entry_name = 'Short'
            tp_name = 'Take Profit (Cover)'
        
        if len(entry_trades) > 0:
            fig.add_trace(
                go.Scatter(
                    x=entry_trades['date'],
                    y=entry_trades['price'],
                    mode='markers',
                    name=entry_name,
                    marker=dict(color=entry_color, size=10, symbol=entry_symbol)
                ),
                row=1, col=1
            )
        
        if len(tp_trades) > 0:
            fig.add_trace(
                go.Scatter(
                    x=tp_trades['date'],
                    y=tp_trades['price'],
                    mode='markers',
                    name=tp_name,
                    marker=dict(color=tp_color, size=10, symbol=tp_symbol)
                ),
                row=1, col=1
            )
        
        liquidated_trades = results_df[results_df['action'] == 'LIQUIDATED']
        stop_loss_trades = results_df[results_df['action'] == 'STOP LOSS']
        
        if len(liquidated_trades) > 0:
            fig.add_trace(
                go.Scatter(
                    x=liquidated_trades['date'],
                    y=liquidated_trades['price'],
                    mode='markers',
                    name='Liquidated',
                    marker=dict(color='black', size=10, symbol='x')
                ),
                row=1, col=1
            )
        
        if len(stop_loss_trades) > 0:
            fig.add_trace(
                go.Scatter(
                    x=stop_loss_trades['date'],
                    y=stop_loss_trades['price'],
                    mode='markers',
                    name='Stop Loss',
                    marker=dict(color='orange', size=10, symbol='diamond')
                ),
                row=1, col=1
            )
    
    if not results_df.empty and 'date' in results_df.columns:
        fig.add_trace(
            go.Scatter(
                x=results_df['date'],
                y=results_df['cumulative_pnl'],
                mode='lines',
                name='Cumulative P&L',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
    
    fig.update_layout(height=800, showlegend=True)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="P&L (USD)", row=2, col=1)
    
    st.plotly_chart(fig, width='stretch')
    
    # Detailed trades table
    st.header("📋 Detailed Trades")
    
    display_df = results_df.copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['price'] = display_df['price'].round(2)
    display_df['investment'] = display_df['investment'].round(0)
    display_df['liquidation_price'] = display_df['liquidation_price'].round(2)
    display_df['take_profit_price'] = display_df['take_profit_price'].round(2)
    display_df['quantity'] = display_df['quantity'].round(4)
    display_df['pnl'] = display_df['pnl'].round(2)
    display_df['cumulative_pnl'] = display_df['cumulative_pnl'].round(2)
    
    st.dataframe(display_df, width='stretch')
    
    # Risk analysis
    st.header("⚠️ Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Investment Progression")
        if not results_df.empty and 'action' in results_df.columns:
            entry_action = 'BUY' if params['trade_type'] == 'long' else 'SHORT'
            investment_sizes = results_df[results_df['action'] == entry_action]['investment'].tolist()
        else:
            investment_sizes = []
        if investment_sizes:
            investment_df = pd.DataFrame({
                'Trade': range(1, len(investment_sizes) + 1),
                'Investment Size': investment_sizes
            })
            
            fig_inv = go.Figure()
            fig_inv.add_trace(
                go.Bar(
                    x=investment_df['Trade'],
                    y=investment_df['Investment Size'],
                    name='Investment Size',
                    marker_color='orange'
                )
            )
            fig_inv.update_layout(
                title="Investment Size per Trade",
                xaxis_title="Trade Number",
                yaxis_title="Investment Size (USD)"
            )
            st.plotly_chart(fig_inv, width='stretch')
    
    with col2:
        st.subheader("Current Position Info")
        if summary['position_open']:
            last_trade = results_df.iloc[-1]
            st.write(f"**Entry Price:** ${last_trade['price']:,.2f}")
            st.write(f"**Current Investment:** ${last_trade['investment']:,.0f}")
            st.write(f"**Liquidation Price:** ${last_trade['liquidation_price']:,.2f}")
            st.write(f"**Take Profit Price:** ${last_trade['take_profit_price']:,.2f}")
            st.write(f"**Quantity:** {last_trade['quantity']:.4f} {params['coin_name']}")
            
            if 'unrealized_pnl' in last_trade:
                st.write(f"**Unrealized P&L:** ${last_trade['unrealized_pnl']:,.2f}")
            
            # Distance calculations based on trade type
            current_price = price_df.iloc[-1]['close']
            if params['trade_type'] == 'long':
                liq_distance = ((current_price - last_trade['liquidation_price']) / current_price) * 100
                tp_distance = ((last_trade['take_profit_price'] - current_price) / current_price) * 100
            else:  # short
                liq_distance = ((last_trade['liquidation_price'] - current_price) / current_price) * 100
                tp_distance = ((current_price - last_trade['take_profit_price']) / current_price) * 100
            
            st.write(f"**Distance to Liquidation:** {liq_distance:.1f}%")
            st.write(f"**Distance to Take Profit:** {tp_distance:.1f}%")
        else:
            st.write("No open position")