import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

st.set_page_config(page_title="Grok World-Class Automation", layout="wide", initial_sidebar_state="expanded")
st.title("🚀 Grok World-Class Automation Dashboard")
st.markdown("**Professional Scanner • Advanced Backtester • Live Charts • Institutional Grade**")

# Primary symbols
symbols = ["GC=F", "EURUSD=X", "CL=F", "^GDAXI", "^GSPC", "BTC-USD", "NQ=F"]

st.sidebar.header("⚙️ Settings")
selected_symbols = st.sidebar.multiselect("Symbols", symbols, default=symbols)
timeframe = st.sidebar.selectbox("Timeframe", ["1h", "4h", "1d"], index=1)
capital = st.sidebar.number_input("Starting Capital ($)", value=100000, step=10000)
risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0)

# Safe indicators
def safe_sma(series, window):
    return pd.Series(series).rolling(window=window).mean()

def safe_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Scanner
def run_scanner():
    results = []
    for sym in selected_symbols:
        try:
            data = yf.download(sym, period="6mo", interval=timeframe, progress=False)
            if len(data) < 100:
                continue
                
            data['SMA50'] = safe_sma(data['Close'], 50)
            data['SMA200'] = safe_sma(data['Close'], 200)
            data['RSI'] = safe_rsi(data['Close'])
            data['VolAvg'] = data['Volume'].rolling(20).mean()
            
            latest = data.iloc[-1]
            prev = data.iloc[-2]
            
            golden_cross = (latest['SMA50'] > latest['SMA200']) and (prev['SMA50'] <= prev['SMA200'])
            volume_surge = latest['Volume'] > 1.5 * latest['VolAvg']
            rsi_condition = latest['RSI'] < 35
            
            if golden_cross and volume_surge and rsi_condition:
                size = round((capital * risk_pct / 100) / latest['Close'], 2)
                results.append({
                    "Symbol": sym,
                    "Signal": "BUY",
                    "Price": round(latest['Close'], 4),
                    "Size": size,
                    "RSI": round(latest['RSI'], 2),
                    "Reason": "4H Golden Cross + Volume + RSI"
                })
        except:
            continue
    return pd.DataFrame(results)

# Advanced Backtester
def run_backtest(sym):
    try:
        data = yf.download(sym, period="6mo", interval=timeframe, progress=False)
        if len(data) < 100:
            return None
            
        data['SMA50'] = safe_sma(data['Close'], 50)
        data['SMA200'] = safe_sma(data['Close'], 200)
        data['Signal'] = (data['SMA50'] > data['SMA200']).astype(int).shift(1).fillna(0)
        data['Returns'] = data['Close'].pct_change()
        data['Strategy'] = data['Signal'] * data['Returns']
        equity = (1 + data['Strategy']).cumprod()
        
        total_return = (equity.iloc[-1] - 1) * 100
        max_dd = ((equity / equity.cummax()) - 1).min() * 100
        win_rate = (data['Strategy'] > 0).mean() * 100
        sharpe = data['Strategy'].mean() / data['Strategy'].std() * np.sqrt(252) if data['Strategy'].std() != 0 else 0
        
        return {
            "Symbol": sym,
            "Total Return (%)": round(total_return, 2),
            "Max Drawdown (%)": round(max_dd, 2),
            "Win Rate (%)": round(win_rate, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Equity": equity
        }
    except:
        return None

# UI Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Scanner", "📈 Backtester", "📊 Live Charts", "📋 Performance Summary"])

with tab1:
    if st.button("🚀 Run Full Professional Scan"):
        with st.spinner("Scanning market..."):
            df = run_scanner()
            if not df.empty:
                st.success(f"Found {len(df)} high-probability signals!")
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No signals today.")

with tab2:
    st.subheader("Advanced Backtester")
    for sym in selected_symbols:
        result = run_backtest(sym)
        if result:
            st.write(f"**{sym}**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", f"{result['Total Return (%)']}%")
            col2.metric("Max Drawdown", f"{result['Max Drawdown (%)']}%")
            col3.metric("Win Rate", f"{result['Win Rate (%)']}%")
            col4.metric("Sharpe Ratio", result['Sharpe Ratio'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=result["Equity"].index, y=result["Equity"], name="Equity Curve"))
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Live Professional Charts")
    selected = st.selectbox("Select Symbol", selected_symbols)
    data = yf.download(selected, period="3mo", interval="1h", progress=False)
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    fig.update_layout(title=f"{selected} Professional Chart", xaxis_title="Date", yaxis_title="Price", height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Overall Performance Summary")
    summary_data = []
    for sym in selected_symbols:
        result = run_backtest(sym)
        if result:
            summary_data.append(result)
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Download button
        csv = summary_df.to_csv(index=False)
        st.download_button("📥 Download Full Report as CSV", csv, "backtest_report.csv", "text/csv")

st.caption("World-Class Automation Dashboard • Built live for you • March 2026")
