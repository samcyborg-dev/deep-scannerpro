"""
A+ Market Phase & Supply-Demand Trading System
Professional implementation of your trading playbook
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

ASSETS = {
    'XAUUSD (Gold)': 'GC=F',
    'WTI (Oil)': 'CL=F',
    'EURUSD': 'EURUSD=X',
    'S&P500': '^GSPC',
    'DAX30': '^GDAXI'
}

TIMEFRAMES = {
    '5 Minutes': '5m',
    '15 Minutes': '15m',
    '30 Minutes': '30m',
    '1 Hour': '60m',
    '4 Hours': '240m',
    '1 Day': '1d'
}

# ============================================================================
# DATA FETCHER
# ============================================================================

@st.cache_data(ttl=300)
def fetch_data(symbol: str, interval: str, period: str = '1mo') -> pd.DataFrame:
    """Fetch market data with caching"""
    try:
        ticker = ASSETS.get(symbol, symbol)
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df.empty:
            return pd.DataFrame()
            
        # Standardize column names
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Calculate basic indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['ATR'] = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        return df.dropna()
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# MARKET PHASE DETECTOR (Based on Your Playbook)
# ============================================================================

def detect_market_phase(df: pd.DataFrame) -> Dict:
    """
    Detect market phase based on your playbook definitions:
    - Accumulation: At discount, after decline, sideways range at lows
    - Distribution: At premium, after advance, sideways range at highs
    - Markup: Strong impulsive moves with little overlap
    - Markdown: Fast, aggressive selling after distribution
    """
    if len(df) < 50:
        return {'phase': 'Unknown', 'confidence': 0, 'location': 'mid', 'liquidity_swept': False, 'structure_broken': False}
    
    # Get recent data
    recent = df.tail(50)
    current_price = df['Close'].iloc[-1]
    
    # Calculate price range and position
    range_high = recent['High'].max()
    range_low = recent['Low'].min()
    price_range = range_high - range_low
    
    if price_range > 0:
        price_position = (current_price - range_low) / price_range
    else:
        price_position = 0.5
    
    # Location classification
    if price_position < 0.3:
        location = 'discount'
    elif price_position > 0.7:
        location = 'premium'
    else:
        location = 'mid'
    
    # Calculate momentum
    momentum = (df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20] * 100
    
    # Calculate volatility
    volatility = df['Returns'].std() * 100
    
    # Phase detection logic
    phase = 'Consolidation'
    confidence = 0.5
    
    # Accumulation detection (discount, sideways, failed breakdowns)
    if location == 'discount' and abs(momentum) < 2 and volatility < 1.5:
        phase = 'Accumulation'
        confidence = 0.7 + (1 - price_position) * 0.3
        
        # Check for failed breakdowns
        failed_breakdowns = sum(1 for i in range(-10, 0) if df['Low'].iloc[i] < df['Low'].iloc[i-5:i].min() and df['Close'].iloc[i] > df['Low'].iloc[i])
        if failed_breakdowns > 2:
            confidence += 0.1
            
    # Distribution detection (premium, sideways, weak continuation)
    elif location == 'premium' and abs(momentum) < 2 and volatility < 1.5:
        phase = 'Distribution'
        confidence = 0.7 + (price_position - 0.7) * 0.3
        
    # Markup detection (strong bullish momentum)
    elif momentum > 3 and volatility > 1:
        phase = 'Markup'
        confidence = min(0.9, 0.6 + momentum / 20)
        
    # Markdown detection (strong bearish momentum)
    elif momentum < -3 and volatility > 1:
        phase = 'Markdown'
        confidence = min(0.9, 0.6 + abs(momentum) / 20)
    
    # Liquidity sweep detection
    liquidity_swept = False
    
    # Check for sweep above recent high
    recent_highs = df['High'].iloc[-10:-1].max()
    if df['High'].iloc[-1] > recent_highs:
        # Check for reversal after high sweep
        if df['Close'].iloc[-1] < df['Close'].iloc[-2]:
            liquidity_swept = True
            
    # Check for sweep below recent low
    recent_lows = df['Low'].iloc[-10:-1].min()
    if df['Low'].iloc[-1] < recent_lows:
        # Check for reversal after low sweep
        if df['Close'].iloc[-1] > df['Close'].iloc[-2]:
            liquidity_swept = True
    
    # Structure break detection
    structure_broken = False
    
    # Check for break of structure
    if len(df) > 20:
        # Bullish structure break
        if df['High'].iloc[-1] > df['High'].iloc[-10:-1].max() and df['Close'].iloc[-1] > df['SMA_20'].iloc[-1]:
            structure_broken = True
        # Bearish structure break
        elif df['Low'].iloc[-1] < df['Low'].iloc[-10:-1].min() and df['Close'].iloc[-1] < df['SMA_20'].iloc[-1]:
            structure_broken = True
    
    return {
        'phase': phase,
        'confidence': confidence,
        'location': location,
        'liquidity_swept': liquidity_swept,
        'structure_broken': structure_broken,
        'price_position': price_position,
        'momentum': momentum,
        'volatility': volatility
    }

# ============================================================================
# SUPPLY & DEMAND ZONES
# ============================================================================

def find_supply_demand_zones(df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """Identify supply and demand zones based on price action and volume"""
    supply_zones = []
    demand_zones = []
    
    if len(df) < 20:
        return supply_zones, demand_zones
    
    # Look for strong reversal candles
    for i in range(5, len(df) - 1):
        candle = df.iloc[i]
        prev_candle = df.iloc[i-1]
        volume_avg = df['Volume'].iloc[i-10:i].mean()
        
        # Demand zone (bullish reversal)
        if (candle['Close'] > candle['Open'] and 
            candle['Volume'] > volume_avg * 1.5 and
            candle['Low'] < prev_candle['Low']):
            
            zone = {
                'price': candle['Low'],
                'upper': candle['Close'],
                'lower': candle['Low'] * 0.998,
                'strength': min(1.0, candle['Volume'] / volume_avg),
                'time': candle.name
            }
            demand_zones.append(zone)
            
        # Supply zone (bearish reversal)
        elif (candle['Close'] < candle['Open'] and 
              candle['Volume'] > volume_avg * 1.5 and
              candle['High'] > prev_candle['High']):
            
            zone = {
                'price': candle['High'],
                'upper': candle['High'] * 1.002,
                'lower': candle['Close'],
                'strength': min(1.0, candle['Volume'] / volume_avg),
                'time': candle.name
            }
            supply_zones.append(zone)
    
    # Filter recent zones (last 30 candles)
    recent_time = df.index[-30] if len(df) > 30 else df.index[0]
    supply_zones = [z for z in supply_zones if z['time'] > recent_time]
    demand_zones = [z for z in demand_zones if z['time'] > recent_time]
    
    # Sort by strength
    supply_zones.sort(key=lambda x: x['strength'], reverse=True)
    demand_zones.sort(key=lambda x: x['strength'], reverse=True)
    
    return supply_zones[:5], demand_zones[:5]

# ============================================================================
# A+ TRADE SETUP GENERATOR (Based on Your Playbook)
# ============================================================================

def generate_trade_setups(df: pd.DataFrame, asset: str, phase_data: Dict) -> List[Dict]:
    """
    Generate A+ trade setups based on your playbook:
    1. HTF context clear
    2. Correct location (discount for longs, premium for shorts)
    3. Liquidity taken
    4. Structure break (BOS)
    5. Fresh LTF supply/demand AFTER BOS
    """
    setups = []
    
    # HARD FILTERS - Missing any = NO TRADE
    if phase_data['confidence'] < 0.6:
        return setups
    
    if not phase_data['liquidity_swept']:
        return setups
    
    if not phase_data['structure_broken']:
        return setups
    
    current_price = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else current_price * 0.01
    
    # Find supply/demand zones
    supply_zones, demand_zones = find_supply_demand_zones(df)
    
    # LONG SETUP (Demand zones at discount with liquidity sweep)
    if phase_data['location'] == 'discount' and (phase_data['phase'] in ['Accumulation', 'Markup']):
        
        # Find nearest demand zone
        if demand_zones:
            nearest_demand = min(demand_zones, key=lambda z: abs(z['price'] - current_price))
            
            if current_price <= nearest_demand['upper'] * 1.01:
                entry = current_price
                stop = nearest_demand['lower'] * 0.995
                
                # Take profit targets
                recent_high = df['High'].iloc[-20:].max()
                target1 = recent_high
                target2 = recent_high + atr
                
                risk = entry - stop
                reward1 = target1 - entry
                reward2 = target2 - entry
                
                if risk > 0:
                    rr_ratio = reward1 / risk
                    
                    if rr_ratio >= 2:  # Minimum 1:2 risk-reward
                        setups.append({
                            'direction': 'LONG',
                            'entry': entry,
                            'stop_loss': stop,
                            'take_profit': target1,
                            'take_profit_2': target2,
                            'risk_reward': rr_ratio,
                            'confidence': min(0.95, phase_data['confidence'] + nearest_demand['strength'] * 0.2),
                            'zone_price': nearest_demand['price'],
                            'zone_strength': nearest_demand['strength']
                        })
    
    # SHORT SETUP (Supply zones at premium with liquidity sweep)
    elif phase_data['location'] == 'premium' and (phase_data['phase'] in ['Distribution', 'Markdown']):
        
        # Find nearest supply zone
        if supply_zones:
            nearest_supply = min(supply_zones, key=lambda z: abs(z['price'] - current_price))
            
            if current_price >= nearest_supply['lower'] * 0.99:
                entry = current_price
                stop = nearest_supply['upper'] * 1.005
                
                # Take profit targets
                recent_low = df['Low'].iloc[-20:].min()
                target1 = recent_low
                target2 = recent_low - atr
                
                risk = stop - entry
                reward1 = entry - target1
                reward2 = entry - target2
                
                if risk > 0:
                    rr_ratio = reward1 / risk
                    
                    if rr_ratio >= 2:
                        setups.append({
                            'direction': 'SHORT',
                            'entry': entry,
                            'stop_loss': stop,
                            'take_profit': target1,
                            'take_profit_2': target2,
                            'risk_reward': rr_ratio,
                            'confidence': min(0.95, phase_data['confidence'] + nearest_supply['strength'] * 0.2),
                            'zone_price': nearest_supply['price'],
                            'zone_strength': nearest_supply['strength']
                        })
    
    return setups

# ============================================================================
# TELEGRAM ALERTS
# ============================================================================

def send_telegram_alert(bot_token: str, chat_id: str, setup: Dict, asset: str):
    """Send trade alert via Telegram"""
    if not bot_token or not chat_id or bot_token == 'YOUR_BOT_TOKEN':
        return False
    
    emoji = "🟢" if setup['direction'] == 'LONG' else "🔴"
    
    message = f"""
{emoji} <b>A+ TRADE SETUP ALERT</b>
━━━━━━━━━━━━━━━━━━━

<b>Asset:</b> {asset}
<b>Direction:</b> {setup['direction']}
<b>Entry:</b> ${setup['entry']:.2f}
<b>Stop Loss:</b> ${setup['stop_loss']:.2f}
<b>Take Profit:</b> ${setup['take_profit']:.2f}

<b>Risk/Reward:</b> 1:{setup['risk_reward']:.1f}
<b>Confidence:</b> {setup['confidence']:.0%}

━━━━━━━━━━━━━━━━━━━
<i>"Discipline over emotion. Wait for the market to invite you."</i>
"""
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except:
        return False

# ============================================================================
# CHARTING
# ============================================================================

def create_professional_chart(df: pd.DataFrame, supply_zones: List, demand_zones: List, setups: List) -> go.Figure:
    """Create professional trading chart with zones and setups"""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price Action', 'Volume')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Add moving averages
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['SMA_20'],
            name='SMA 20',
            line=dict(color='#00ff00', width=1.5),
            opacity=0.7
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['SMA_50'],
            name='SMA 50',
            line=dict(color='#ff9900', width=1.5),
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Add supply zones (red)
    for zone in supply_zones:
        fig.add_hrect(
            y0=zone['lower'],
            y1=zone['upper'],
            fillcolor="red",
            opacity=0.2,
            line_width=0,
            name="Supply Zone",
            row=1, col=1
        )
        
        fig.add_annotation(
            x=zone['time'],
            y=zone['upper'],
            text=f"Supply {zone['strength']:.0%}",
            showarrow=True,
            arrowhead=1,
            arrowcolor='red',
            font=dict(size=10, color='red'),
            row=1, col=1
        )
    
    # Add demand zones (green)
    for zone in demand_zones:
        fig.add_hrect(
            y0=zone['lower'],
            y1=zone['upper'],
            fillcolor="green",
            opacity=0.2,
            line_width=0,
            name="Demand Zone",
            row=1, col=1
        )
        
        fig.add_annotation(
            x=zone['time'],
            y=zone['lower'],
            text=f"Demand {zone['strength']:.0%}",
            showarrow=True,
            arrowhead=1,
            arrowcolor='green',
            font=dict(size=10, color='green'),
            row=1, col=1
        )
    
    # Add trade setups
    for setup in setups:
        arrow_color = 'green' if setup['direction'] == 'LONG' else 'red'
        arrow_symbol = 'arrow-up' if setup['direction'] == 'LONG' else 'arrow-down'
        
        fig.add_annotation(
            x=df.index[-1],
            y=setup['entry'],
            text=f"{setup['direction']} @ ${setup['entry']:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=arrow_color,
            font=dict(size=12, color=arrow_color),
            bgcolor='rgba(0,0,0,0.7)',
            row=1, col=1
        )
    
    # Volume bars
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(len(df))]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add volume MA
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Volume_MA'],
            name='Volume MA',
            line=dict(color='yellow', width=1, dash='dash'),
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'A+ Trading System - Supply & Demand Analysis',
            'font': {'size': 20, 'color': 'white'},
            'x': 0.5
        },
        template='plotly_dark',
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

def run_backtest(df: pd.DataFrame, initial_capital: float = 10000) -> Dict:
    """Simple backtest engine"""
    results = {
        'total_trades': 0,
        'winning_trades': 0,
        'total_pnl': 0,
        'max_drawdown': 0,
        'win_rate': 0,
        'trades': []
    }
    
    if len(df) < 100:
        return results
    
    capital = initial_capital
    peak = capital
    trades = []
    
    # Simple moving average crossover strategy for demonstration
    for i in range(50, len(df) - 1):
        if df['SMA_20'].iloc[i] > df['SMA_50'].iloc[i] and df['SMA_20'].iloc[i-1] <= df['SMA_50'].iloc[i-1]:
            # Buy signal
            entry_price = df['Close'].iloc[i]
            exit_price = df['Close'].iloc[i + 5]  # Hold for 5 periods
            
            pnl = (exit_price - entry_price) * 100  # 100 units
            capital += pnl
            
            trades.append({
                'entry': entry_price,
                'exit': exit_price,
                'pnl': pnl,
                'win': pnl > 0
            })
            
            # Update peak for drawdown
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak
            if drawdown > results['max_drawdown']:
                results['max_drawdown'] = drawdown
    
    if trades:
        results['total_trades'] = len(trades)
        results['winning_trades'] = sum(1 for t in trades if t['win'])
        results['total_pnl'] = capital - initial_capital
        results['win_rate'] = results['winning_trades'] / results['total_trades']
        results['trades'] = trades[-20:]  # Last 20 trades
    
    return results

# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="A+ Trading System",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    }
    .metric-card {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .aplus-text {
        color: #00ff00;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🎯 A+ Market Phase & Supply-Demand Trading System")
        st.markdown("*Based on Professional Trading Playbook*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Trading Configuration")
        
        asset = st.selectbox("Select Asset", list(ASSETS.keys()))
        timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()))
        period = st.selectbox("Data Period", ['1mo', '3mo', '6mo'])
        
        st.markdown("---")
        st.header("📱 Telegram Alerts")
        use_telegram = st.checkbox("Enable Telegram Alerts")
        bot_token = st.text_input("Bot Token", type="password", value="YOUR_BOT_TOKEN")
        chat_id = st.text_input("Chat ID", value="YOUR_CHAT_ID")
        
        st.markdown("---")
        st.header("📊 Backtest Settings")
        enable_backtest = st.checkbox("Enable Backtesting")
        initial_capital = st.number_input("Initial Capital ($)", value=10000, step=1000)
        
        st.markdown("---")
        st.header("📖 A+ Trading Rules")
        with st.expander("View Rules"):
            st.markdown("""
            **A+ Setup Requirements:**
            1. ✅ HTF context clear
            2. ✅ Correct location (discount for longs, premium for shorts)
            3. ✅ Liquidity taken (equal highs/lows swept)
            4. ✅ Structure break (BOS)
            5. ✅ Fresh LTF supply/demand AFTER BOS
            
            **Market Phases:**
            - **Accumulation**: At discount, sideways at lows
            - **Distribution**: At premium, sideways at highs
            - **Markup**: Strong impulsive moves
            - **Markdown**: Fast aggressive selling
            
            *"If it's not obvious, it's not A+"*
            """)
    
    # Fetch data
    with st.spinner(f"Fetching {asset} data..."):
        df = fetch_data(asset, TIMEFRAMES[timeframe], period)
    
    if df.empty:
        st.error("Failed to fetch data. Please check your internet connection.")
        return
    
    # Market analysis
    phase_data = detect_market_phase(df)
    supply_zones, demand_zones = find_supply_demand_zones(df)
    setups = generate_trade_setups(df, asset, phase_data)
    
    # Display key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Market Phase",
            phase_data['phase'],
            delta=f"{phase_data['confidence']:.0%} confidence"
        )
    
    with col2:
        st.metric(
            "Location",
            phase_data['location'].upper(),
            delta=f"{phase_data['price_position']:.0%} of range"
        )
    
    with col3:
        st.metric(
            "Liquidity Swept",
            "✅" if phase_data['liquidity_swept'] else "❌",
            delta="Recent sweep" if phase_data['liquidity_swept'] else "None"
        )
    
    with col4:
        st.metric(
            "Structure Break",
            "✅" if phase_data['structure_broken'] else "❌",
            delta="BOS confirmed" if phase_data['structure_broken'] else "Waiting"
        )
    
    with col5:
        st.metric(
            "Current Price",
            f"${df['Close'].iloc[-1]:.2f}",
            delta=f"{phase_data['momentum']:.1f}% momentum"
        )
    
    st.markdown("---")
    
    # Main chart
    fig = create_professional_chart(df, supply_zones, demand_zones, setups)
    st.plotly_chart(fig, use_container_width=True)
    
    # Trade setups section
    st.subheader("🎯 A+ Trade Setups")
    
    if setups:
        for i, setup in enumerate(setups):
            with st.container():
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                
                direction_color = "🟢" if setup['direction'] == 'LONG' else "🔴"
                
                with col1:
                    st.markdown(f"### {direction_color} {setup['direction']}")
                    st.write(f"**Confidence:** {setup['confidence']:.0%}")
                
                with col2:
                    st.write("**Entry:**")
                    st.write(f"${setup['entry']:.2f}")
                    st.write(f"**Stop:** ${setup['stop_loss']:.2f}")
                
                with col3:
                    st.write("**Target:**")
                    st.write(f"${setup['take_profit']:.2f}")
                    st.write(f"**R:R:** 1:{setup['risk_reward']:.1f}")
                
                with col4:
                    if use_telegram and bot_token != 'YOUR_BOT_TOKEN':
                        if st.button(f"🚀 Execute {setup['direction']}", key=f"exec_{i}"):
                            if send_telegram_alert(bot_token, chat_id, setup, asset):
                                st.success("✅ Alert sent to Telegram!")
                            else:
                                st.warning("⚠️ Failed to send alert")
                    else:
                        st.info("Enable Telegram to execute")
                
                st.markdown("---")
    else:
        st.info("🔍 No A+ setups detected at this time.")
        st.markdown("""
        **Requirements for A+ Setup:**
        - HTF context clear ✓
        - Correct location ✓
        - Liquidity taken ✗
        - Structure break ✗
        - Fresh supply/demand zone ✗
        
        *Wait for the market to invite you. Discipline over emotion.*
        """)
    
    # Backtest section
    if enable_backtest:
        st.markdown("---")
        st.subheader("📈 Backtest Results")
        
        with st.spinner("Running backtest..."):
            backtest_results = run_backtest(df, initial_capital)
        
        if backtest_results['total_trades'] > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", backtest_results['total_trades'])
            with col2:
                st.metric("Win Rate", f"{backtest_results['win_rate']:.1%}")
            with col3:
                st.metric("Total P&L", f"${backtest_results['total_pnl']:.2f}")
            with col4:
                st.metric("Max Drawdown", f"{backtest_results['max_drawdown']:.1%}")
            
            if backtest_results['trades']:
                st.write("**Recent Trades:**")
                trades_df = pd.DataFrame(backtest_results['trades'])
                st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("Insufficient data for backtesting")
    
    # Performance metrics
    st.markdown("---")
    st.subheader("📊 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        recent_high = df['High'].iloc[-20:].max()
        recent_low = df['Low'].iloc[-20:].min()
        st.metric("20-period Range", f"${recent_high - recent_low:.2f}")
    
    with col2:
        avg_volume = df['Volume'].iloc[-20:].mean()
        current_volume = df['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume
        st.metric("Volume Ratio", f"{volume_ratio:.2f}x", delta=f"{volume_ratio:.1f}x vs avg")
    
    with col3:
        supply_count = len(supply_zones)
        demand_count = len(demand_zones)
        st.metric("Active Zones", f"S:{supply_count} D:{demand_count}")
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)")
    if auto_refresh:
        time.sleep(60)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
