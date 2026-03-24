"""
Simplified Trading System - Light Version
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

# Configuration
ASSETS = {
    'XAUUSD': 'GC=F',
    'WTI': 'CL=F', 
    'EURUSD': 'EURUSD=X',
    'S&P500': '^GSPC',
    'DAX30': '^GDAXI'
}

TIMEFRAMES = {
    '15m': '15m',
    '1h': '60m',
    '4h': '240m',
    '1d': '1d'
}

class SimpleTradingSystem:
    def __init__(self, asset, timeframe):
        self.asset = asset
        self.timeframe = timeframe
        self.data = None
        
    def fetch_data(self):
        """Fetch market data"""
        ticker = ASSETS.get(self.asset, self.asset)
        self.data = yf.download(ticker, period='1mo', interval=self.timeframe, progress=False)
        
        # Calculate basic indicators
        self.data['EMA_20'] = self.data['Close'].ewm(span=20).mean()
        self.data['EMA_50'] = self.data['Close'].ewm(span=50).mean()
        self.data['ATR'] = self.data['High'].rolling(14).max() - self.data['Low'].rolling(14).min()
        
        return self.data
        
    def detect_market_phase(self):
        """Simple market phase detection"""
        if self.data is None or len(self.data) < 50:
            return "Unknown", 0
            
        close = self.data['Close'].values
        recent_range = self.data['High'].iloc[-50:].max() - self.data['Low'].iloc[-50:].min()
        current_price = close[-1]
        
        # Price location
        price_position = (current_price - self.data['Low'].iloc[-50:].min()) / recent_range
        
        # Momentum
        momentum = (close[-1] - close[-20]) / close[-20] * 100
        
        # Phase classification
        if price_position < 0.3 and momentum < 5:
            phase = "Accumulation"
            confidence = 0.7
        elif price_position > 0.7 and momentum < 5:
            phase = "Distribution" 
            confidence = 0.7
        elif momentum > 5:
            phase = "Markup"
            confidence = 0.8
        elif momentum < -5:
            phase = "Markdown"
            confidence = 0.8
        else:
            phase = "Consolidation"
            confidence = 0.5
            
        return phase, confidence
        
    def find_support_resistance(self):
        """Find simple support and resistance levels"""
        if self.data is None:
            return None, None
            
        # Simple pivot points
        high = self.data['High'].iloc[-20:].max()
        low = self.data['Low'].iloc[-20:].min()
        close = self.data['Close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        
        return s1, r1
        
    def generate_signals(self):
        """Generate trading signals based on your playbook"""
        if self.data is None or len(self.data) < 50:
            return []
            
        signals = []
        phase, confidence = self.detect_market_phase()
        s1, r1 = self.find_support_resistance()
        current_price = self.data['Close'].iloc[-1]
        
        # A+ criteria from your playbook
        # 1. HTF context clear
        if confidence < 0.6:
            return signals
            
        # 2. Correct location
        price_range = self.data['High'].iloc[-50:].max() - self.data['Low'].iloc[-50:].min()
        price_position = (current_price - self.data['Low'].iloc[-50:].min()) / price_range
        
        # 3. Liquidity sweep (simplified)
        high_sweep = self.data['High'].iloc[-1] > self.data['High'].iloc[-5:-1].max()
        low_sweep = self.data['Low'].iloc[-1] < self.data['Low'].iloc[-5:-1].min()
        
        # 4. Structure break
        ema_cross = self.data['EMA_20'].iloc[-1] > self.data['EMA_50'].iloc[-1] and \
                    self.data['EMA_20'].iloc[-2] <= self.data['EMA_50'].iloc[-2]
        
        # Long setup
        if phase in ['Accumulation', 'Markup'] and price_position < 0.3 and low_sweep:
            entry = current_price
            stop = s1 * 0.99 if s1 else current_price * 0.98
            target = r1 if r1 else current_price * 1.02
            
            risk = entry - stop
            reward = target - entry
            
            if risk > 0 and reward / risk >= 2:
                signals.append({
                    'direction': 'LONG',
                    'entry': entry,
                    'stop': stop,
                    'target': target,
                    'rr_ratio': reward / risk,
                    'phase': phase,
                    'confidence': confidence
                })
                
        # Short setup
        elif phase in ['Distribution', 'Markdown'] and price_position > 0.7 and high_sweep:
            entry = current_price
            stop = r1 * 1.01 if r1 else current_price * 1.02
            target = s1 if s1 else current_price * 0.98
            
            risk = stop - entry
            reward = entry - target
            
            if risk > 0 and reward / risk >= 2:
                signals.append({
                    'direction': 'SHORT',
                    'entry': entry,
                    'stop': stop,
                    'target': target,
                    'rr_ratio': reward / risk,
                    'phase': phase,
                    'confidence': confidence
                })
                
        return signals

def create_chart(data):
    """Create simple chart"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    # Add EMAs
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], name='EMA 20', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], name='EMA 50', line=dict(color='orange')))
    
    fig.update_layout(
        title='Trading Chart',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark',
        height=600
    )
    
    return fig

def send_telegram_alert(bot_token, chat_id, signal):
    """Send alert via Telegram"""
    if not bot_token or not chat_id:
        return False
        
    try:
        message = f"""
🚨 TRADE SIGNAL ALERT 🚨

Direction: {signal['direction']}
Entry: ${signal['entry']:.2f}
Stop Loss: ${signal['stop']:.2f}
Take Profit: ${signal['target']:.2f}
Risk/Reward: 1:{signal['rr_ratio']:.1f}
Market Phase: {signal['phase']}
Confidence: {signal['confidence']:.0%}

Discipline > Emotion
        """
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {'chat_id': chat_id, 'text': message, 'parse_mode': 'HTML'}
        response = requests.post(url, json=payload, timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    st.set_page_config(page_title="Trading System", layout="wide")
    st.title("🎯 Supply-Demand Trading System")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        asset = st.selectbox("Asset", list(ASSETS.keys()))
        timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()))
        
        st.markdown("---")
        st.header("Telegram Alerts")
        use_telegram = st.checkbox("Enable Alerts")
        bot_token = st.text_input("Bot Token", type="password")
        chat_id = st.text_input("Chat ID")
        
    # Main content
    system = SimpleTradingSystem(asset, TIMEFRAMES[timeframe])
    
    with st.spinner("Fetching data..."):
        data = system.fetch_data()
        
    if data.empty:
        st.error("Failed to fetch data")
        return
        
    # Market analysis
    phase, confidence = system.detect_market_phase()
    signals = system.generate_signals()
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Market Phase", phase)
    with col2:
        st.metric("Confidence", f"{confidence:.0%}")
    with col3:
        st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
    
    # Chart
    st.plotly_chart(create_chart(data), use_container_width=True)
    
    # Signals
    st.subheader("Trade Signals")
    if signals:
        for signal in signals:
            with st.expander(f"{'🟢' if signal['direction'] == 'LONG' else '🔴'} {signal['direction']} Signal - RR 1:{signal['rr_ratio']:.1f}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Entry: ${signal['entry']:.2f}")
                    st.write(f"Stop: ${signal['stop']:.2f}")
                with col2:
                    st.write(f"Target: ${signal['target']:.2f}")
                    st.write(f"Risk/Reward: 1:{signal['rr_ratio']:.1f}")
                
                if st.button(f"Execute {signal['direction']}", key=f"btn_{signal['direction']}"):
                    if use_telegram:
                        send_telegram_alert(bot_token, chat_id, signal)
                        st.success(f"Alert sent to Telegram!")
                    else:
                        st.success(f"Trade {signal['direction']} executed!")
    else:
        st.info("No A+ setups detected. Waiting for market to invite you...")
        st.markdown("""
        **A+ Setup Requirements:**
        - ✅ Clear market phase
        - ✅ Correct location (discount for longs, premium for shorts)
        - ✅ Liquidity swept
        - ✅ Structure break
        - ✅ Fresh supply/demand zone
        """)
        
    # Trade rules reminder
    with st.expander("📖 A+ Trading Rules"):
        st.markdown("""
        ### Non-Negotiable Rules:
        1. **Phase**: Only trade in clear accumulation/distribution or markup/markdown
        2. **Location**: Enter only at discount (longs) or premium (shorts)
        3. **Liquidity**: Wait for equal highs/lows to be swept
        4. **Structure**: Confirmation after break of structure
        5. **Risk**: Minimum 1:2 risk-to-reward ratio
        
        *If you can't label all four criteria → NO TRADE*
        """)

if __name__ == "__main__":
    main()
