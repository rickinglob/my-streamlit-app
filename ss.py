import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import smtplib
import time
from email.mime.text import MIMEText
from datetime import datetime
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, TSIIndicator, KAMAIndicator
from ta.trend import MACD, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator

# Custom CSS (updated to style sidebar alerts)
st.markdown("""
    <style>
    .reportview-container {
        background: #1a1a1a;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 5px 15px;
    }
    .stSlider > div > div > div {
        background-color: #333;
        border-color: #555;
    }
    .stExpander > div > div > div {
        background-color: #2a2a2a;
        border-color: #555;
    }
    .stSelectbox > div > div > div {
        background-color: #2a2a2a;
        border-color: #555;
    }
    .stSidebar .sidebar-content {
        background-color: #2a2a2a;
        border-right: 1px solid #555;
    }
    .main-content {
        padding: 20px;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stWarning {
        background-color: #856404;
        color: #ffffff;
        padding: 10px;
        border-radius: 5px;
    }
    .sidebar .stWarning {
        margin: 5px 0;
    }
    /* New styles for indicator pills */
    .indicator-pill {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 12px;
        margin: 2px;
        font-size: 12px;
        font-weight: bold;
    }
    .indicator-buy {
        background-color: #4CAF50;
        color: white;
    }
    .indicator-sell {
        background-color: #f44336;
        color: white;
    }
    .indicator-neutral {
        background-color: #9e9e9e;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Expanded cache duration
@st.cache_data(ttl=900)
def get_asset_data(ticker, period="3mo", interval="1H"):
    try:      
        asset = yf.Ticker(ticker)
        data = asset.history(period=period, interval=interval)
        if data.empty:
            st.sidebar.warning(f"No data found for {ticker}. Check the ticker symbol.")
            return None
        return data
    except Exception as e:
        st.sidebar.error(f"Error fetching data for {ticker}: {e}")
        return None

@st.cache_data
def calculate_advanced_indicators(data, short_window=5, long_window=20, rsi_window=14, macd_fast=12, macd_slow=26, macd_signal=9):
    """Calculate comprehensive set of technical indicators using the TA library"""
    if data is None or data.empty or len(data) < 5:
        return None
    
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Traditional SMA-based momentum
    df["Short_SMA"] = df["Close"].rolling(window=short_window).mean()
    df["Long_SMA"] = df["Close"].rolling(window=long_window).mean()
    df["Momentum_SMA"] = df["Short_SMA"] - df["Long_SMA"]
    
    # RSI - Relative Strength Index
    rsi_indicator = RSIIndicator(df["Close"], window=rsi_window)
    df["RSI"] = rsi_indicator.rsi()
    
    # MACD - Moving Average Convergence Divergence
    macd_indicator = MACD(df["Close"], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
    df["MACD"] = macd_indicator.macd()
    df["MACD_Signal"] = macd_indicator.macd_signal()
    df["MACD_Histogram"] = macd_indicator.macd_diff()
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(df["High"], df["Low"], df["Close"], window=14, smooth_window=3)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()
    
    # Rate of Change (ROC)
    roc_indicator = ROCIndicator(df["Close"], window=12)
    df["ROC"] = roc_indicator.roc()
    
    # True Strength Index (TSI)
    tsi_indicator = TSIIndicator(df["Close"], window_slow=25, window_fast=13)
    df["TSI"] = tsi_indicator.tsi()
    
    # KAMA - Kaufman Adaptive Moving Average
    kama_indicator = KAMAIndicator(df["Close"], window=10, pow1=2, pow2=30)
    df["KAMA"] = kama_indicator.kama()
    
    # Bollinger Bands
    bollinger = BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_Upper"] = bollinger.bollinger_hband()
    df["BB_Lower"] = bollinger.bollinger_lband()
    df["BB_Middle"] = bollinger.bollinger_mavg()
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["BB_Percent"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    
    # Average True Range - volatility indicator
    atr_indicator = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14)
    df["ATR"] = atr_indicator.average_true_range()
    df["ATR_Percent"] = df["ATR"] / df["Close"] * 100
    
    # ADX - Average Directional Index - trend strength
    adx_indicator = ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
    df["ADX"] = adx_indicator.adx()
    df["DI_Plus"] = adx_indicator.adx_pos()
    df["DI_Minus"] = adx_indicator.adx_neg()
    
    # CCI - Commodity Channel Index
    cci_indicator = CCIIndicator(df["High"], df["Low"], df["Close"], window=20)
    df["CCI"] = cci_indicator.cci()
    
    # Volume-based indicators
    if "Volume" in df.columns and not df["Volume"].isnull().all():
        # Money Flow Index
        mfi_indicator = MFIIndicator(df["High"], df["Low"], df["Close"], df["Volume"], window=14)
        df["MFI"] = mfi_indicator.money_flow_index()
        
        # On-Balance Volume
        obv_indicator = OnBalanceVolumeIndicator(df["Close"], df["Volume"])
        df["OBV"] = obv_indicator.on_balance_volume()
        df["OBV_ROC"] = df["OBV"].pct_change(periods=10) * 100
    
    # Calculate Supertrend (custom implementation)
    atr_factor = 3
    df["basic_upper"] = (df["High"] + df["Low"]) / 2 + atr_factor * df["ATR"]
    df["basic_lower"] = (df["High"] + df["Low"]) / 2 - atr_factor * df["ATR"]
    
    # Initialize Supertrend columns
    df["supertrend_upper"] = df["basic_upper"]
    df["supertrend_lower"] = df["basic_lower"]
    df["supertrend"] = 0
    
    # First value
    if len(df) > 1:
        if df["Close"].iloc[0] <= df["basic_upper"].iloc[0]:
            df.loc[df.index[0], "supertrend"] = 1  # Uptrend
            df.loc[df.index[0], "supertrend_upper"] = np.nan
        else:
            df.loc[df.index[0], "supertrend"] = -1  # Downtrend
            df.loc[df.index[0], "supertrend_lower"] = np.nan
    
    # Calculate Supertrend
    for i in range(1, len(df)):
        curr_close = df["Close"].iloc[i]
        prev_upper = df["supertrend_upper"].iloc[i-1]
        curr_basic_upper = df["basic_upper"].iloc[i]
        prev_lower = df["supertrend_lower"].iloc[i-1]
        curr_basic_lower = df["basic_lower"].iloc[i]
        prev_supertrend = df["supertrend"].iloc[i-1]
        
        # Calculate upper band
        if (curr_basic_upper < prev_upper) or (df["Close"].iloc[i-1] > prev_upper):
            df.loc[df.index[i], "supertrend_upper"] = curr_basic_upper
        else:
            df.loc[df.index[i], "supertrend_upper"] = prev_upper
            
        # Calculate lower band
        if (curr_basic_lower > prev_lower) or (df["Close"].iloc[i-1] < prev_lower):
            df.loc[df.index[i], "supertrend_lower"] = curr_basic_lower
        else:
            df.loc[df.index[i], "supertrend_lower"] = prev_lower
            
        # Set supertrend value
        if (prev_supertrend == 1 and curr_close <= df["supertrend_upper"].iloc[i]):
            df.loc[df.index[i], "supertrend"] = 1
            df.loc[df.index[i], "supertrend_upper"] = np.nan
        elif (prev_supertrend == -1 and curr_close >= df["supertrend_lower"].iloc[i]):
            df.loc[df.index[i], "supertrend"] = -1
            df.loc[df.index[i], "supertrend_lower"] = np.nan
        elif (prev_supertrend == 1 and curr_close > df["supertrend_upper"].iloc[i]):
            df.loc[df.index[i], "supertrend"] = -1
            df.loc[df.index[i], "supertrend_lower"] = np.nan
        elif (prev_supertrend == -1 and curr_close < df["supertrend_lower"].iloc[i]):
            df.loc[df.index[i], "supertrend"] = 1
            df.loc[df.index[i], "supertrend_upper"] = np.nan

    # Exponential moving averages for trend analysis
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
    
    # Hull Moving Average (HMA) - responsive trend indicator
    wma_half = df["Close"].rolling(window=int(10/2)).apply(lambda x: 
                                   np.sum(np.arange(1, len(x)+1) * x) / np.sum(np.arange(1, len(x)+1)), raw=True)
    wma_full = df["Close"].rolling(window=10).apply(lambda x: 
                                   np.sum(np.arange(1, len(x)+1) * x) / np.sum(np.arange(1, len(x)+1)), raw=True)
    sqrt_length = int(np.sqrt(10))
    df["HMA"] = pd.Series(2 * wma_half - wma_full).rolling(window=sqrt_length).apply(lambda x: 
                                   np.sum(np.arange(1, len(x)+1) * x) / np.sum(np.arange(1, len(x)+1)), raw=True)
    
    # Calculate momentum metrics
    df["EMA_Cross"] = (df["EMA_9"] > df["EMA_21"]).astype(int)
    df["Golden_Cross"] = (df["EMA_50"] > df["EMA_200"]).astype(int)
    
    # Indicator combination metrics
    df["RSI_Stoch_Combined"] = (df["RSI"] + df["Stoch_K"]) / 2
    df["Trend_Strength"] = (df["ADX"] / 100) * (df["supertrend"] * 50 + 50)
    
    return df

def calculate_momentum_score(data, weights=None):
    """
    Calculate a comprehensive momentum score using multiple indicators.
    
    Parameters:
    - data: DataFrame with technical indicators
    - weights: Dictionary of weights for each indicator
    
    Returns:
    - momentum_score: Float between 0 and 100
    - signals: Dictionary of signals from different indicators
    """
    if data is None or data.empty or len(data) < 26:
        return 0, {}
    
    # Default weights if none provided
    if weights is None:
        weights = {
            "sma_momentum": 10,
            "macd": 15,
            "rsi": 15,
            "stochastic": 10,
            "supertrend": 15,
            "adx": 10,
            "cci": 5,
            "mfi": 10,
            "bb_position": 5,
            "ema_cross": 5
        }
    
    # Get latest values
    latest = data.iloc[-1]
    
    # Initialize signals dictionary
    signals = {}
    
    # SMA Momentum
    sma_momentum = latest["Momentum_SMA"] if not pd.isna(latest["Momentum_SMA"]) else 0
    sma_score = min(100, max(0, 50 + (sma_momentum / 2)))
    signals["SMA Momentum"] = {"value": f"{sma_momentum:.2f}", 
                               "signal": "buy" if sma_momentum > 0 else "sell"}
    
    # MACD
    macd_hist = latest["MACD_Histogram"] if not pd.isna(latest["MACD_Histogram"]) else 0
    macd_val = latest["MACD"] if not pd.isna(latest["MACD"]) else 0
    macd_score = min(100, max(0, 50 + (macd_hist * 50)))
    signals["MACD"] = {"value": f"{macd_val:.2f}", 
                       "signal": "buy" if macd_hist > 0 else "sell"}
    
    # RSI
    rsi_val = latest["RSI"] if not pd.isna(latest["RSI"]) else 50
    # RSI score - higher score for extremes (both overbought and oversold)
    rsi_score = min(100, max(0, 100 - abs(rsi_val - 50)))
    signals["RSI"] = {"value": f"{rsi_val:.2f}", 
                     "signal": "buy" if rsi_val < 30 else "sell" if rsi_val > 70 else "neutral"}
    
    # Stochastic
    stoch_k = latest["Stoch_K"] if not pd.isna(latest["Stoch_K"]) else 50
    stoch_d = latest["Stoch_D"] if not pd.isna(latest["Stoch_D"]) else 50
    stoch_score = min(100, max(0, 100 - abs(stoch_k - 50)))
    stoch_crossover = stoch_k > stoch_d
    signals["Stochastic"] = {"value": f"K:{stoch_k:.2f} D:{stoch_d:.2f}", 
                            "signal": "buy" if stoch_k < 20 and stoch_crossover else 
                                     "sell" if stoch_k > 80 and not stoch_crossover else "neutral"}
    
    # Supertrend
    supertrend_val = latest["supertrend"] if not pd.isna(latest["supertrend"]) else 0
    supertrend_score = 100 if supertrend_val == 1 else 0
    signals["Supertrend"] = {"value": "Bullish" if supertrend_val == 1 else "Bearish", 
                            "signal": "buy" if supertrend_val == 1 else "sell"}
    
    # ADX - trend strength
    adx_val = latest["ADX"] if not pd.isna(latest["ADX"]) else 0
    di_plus = latest["DI_Plus"] if not pd.isna(latest["DI_Plus"]) else 0
    di_minus = latest["DI_Minus"] if not pd.isna(latest["DI_Minus"]) else 0
    adx_score = min(100, adx_val)
    adx_direction = di_plus > di_minus
    signals["ADX"] = {"value": f"{adx_val:.2f}", 
                     "signal": "buy" if adx_val > 25 and adx_direction else 
                              "sell" if adx_val > 25 and not adx_direction else "neutral"}
    
    # CCI - Commodity Channel Index
    cci_val = latest["CCI"] if not pd.isna(latest["CCI"]) else 0
    cci_score = min(100, max(0, 50 + (cci_val / 2)))
    signals["CCI"] = {"value": f"{cci_val:.2f}", 
                     "signal": "buy" if cci_val > 100 else "sell" if cci_val < -100 else "neutral"}
    
    # MFI - Money Flow Index
    mfi_available = "MFI" in data.columns and not pd.isna(latest["MFI"])
    mfi_val = latest["MFI"] if mfi_available else 50
    mfi_score = min(100, max(0, 100 - abs(mfi_val - 50)))
    if mfi_available:
        signals["MFI"] = {"value": f"{mfi_val:.2f}", 
                         "signal": "buy" if mfi_val < 20 else "sell" if mfi_val > 80 else "neutral"}
    
    # Bollinger Bands position
    bb_available = "BB_Percent" in data.columns and not pd.isna(latest["BB_Percent"])
    bb_percent = latest["BB_Percent"] if bb_available else 0.5
    bb_score = min(100, max(0, 100 - (abs(bb_percent - 0.5) * 200)))
    if bb_available:
        signals["BB Position"] = {"value": f"{bb_percent:.2f}", 
                                  "signal": "buy" if bb_percent < 0.2 else "sell" if bb_percent > 0.8 else "neutral"}
    
    # EMA Cross
    ema_cross = latest["EMA_Cross"] if not pd.isna(latest["EMA_Cross"]) else 0
    golden_cross = latest["Golden_Cross"] if not pd.isna(latest["Golden_Cross"]) else 0
    ema_score = 100 if ema_cross == 1 and golden_cross == 1 else 75 if ema_cross == 1 else 25 if golden_cross == 1 else 0
    signals["EMA Cross"] = {"value": "Bullish" if ema_cross == 1 else "Bearish", 
                            "signal": "buy" if ema_cross == 1 else "sell"}
    
    # Calculate weighted score
    score_components = {
        "sma_momentum": sma_score,
        "macd": macd_score,
        "rsi": rsi_score,
        "stochastic": stoch_score,
        "supertrend": supertrend_score,
        "adx": adx_score,
        "cci": cci_score,
        "mfi": mfi_score if mfi_available else 50,
        "bb_position": bb_score if bb_available else 50,
        "ema_cross": ema_score
    }
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    normalized_weights = {k: v/total_weight for k, v in weights.items()}
    
    # Calculate final score
    final_score = sum(score_components[k] * normalized_weights[k] for k in weights.keys())
    
    # Add momentum direction indicator
    momentum_direction = "bullish" if final_score > 50 else "bearish" if final_score < 50 else "neutral"
    signals["Overall Momentum"] = {"value": f"{final_score:.2f}", "signal": 
                                   "buy" if momentum_direction == "bullish" else 
                                   "sell" if momentum_direction == "bearish" else "neutral"}
    
    # Return both the score and the signals
    return final_score, signals

def send_alert_email(subject, body, to_email, from_email, password):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(from_email, password)
            server.send_message(msg)
        st.sidebar.success(f"Email alert sent to {to_email}!")
    except Exception as e:
        st.sidebar.error(f"Failed to send email alert: {str(e)}")

# Add custom indicator weights adjustment section
def add_weight_adjustment_section():
    with st.expander("Customize Indicator Weights"):
        col1, col2 = st.columns(2)
        weights = {}
        with col1:
            weights["sma_momentum"] = st.slider("SMA Momentum", 0, 20, 10, 1)
            weights["macd"] = st.slider("MACD", 0, 20, 15, 1)
            weights["rsi"] = st.slider("RSI", 0, 20, 15, 1)
            weights["stochastic"] = st.slider("Stochastic", 0, 20, 10, 1)
            weights["supertrend"] = st.slider("Supertrend", 0, 20, 15, 1)
        with col2:
            weights["adx"] = st.slider("ADX", 0, 20, 10, 1)
            weights["cci"] = st.slider("CCI", 0, 20, 5, 1)
            weights["mfi"] = st.slider("MFI", 0, 20, 10, 1)
            weights["bb_position"] = st.slider("Bollinger Position", 0, 20, 5, 1)
            weights["ema_cross"] = st.slider("EMA Cross", 0, 20, 5, 1)
        
        return weights
            
def check_alerts(ticker, data, momentum_score, signals, 
                momentum_threshold, rsi_threshold_high, rsi_threshold_low, 
                enable_email_alerts, to_email, from_email, password):
    """Check for various alerts and display in sidebar."""
    if "alerts" not in st.session_state:
        st.session_state.alerts = []
    
    if data is None or data.empty:
        return
    
    latest_price = data["Close"].iloc[-1]
    latest_data = data.iloc[-1]
    
    with st.sidebar:
        # 1. Momentum Score Alert
        if abs(momentum_score - 50) > momentum_threshold:
            direction = "bullish" if momentum_score > 50 else "bearish"
            alert_msg = f"🚨 Momentum Alert for {ticker}: {direction.upper()} momentum ({momentum_score:.2f}) exceeds threshold!"
            if alert_msg not in st.session_state.alerts:
                st.session_state.alerts.append(alert_msg)
                st.warning(alert_msg)
                if enable_email_alerts and to_email and from_email and password:
                    send_alert_email(
                        f"Momentum Alert for {ticker}",
                        f"Strong {direction} momentum ({momentum_score:.2f}) detected for {ticker}. Price: ₹{latest_price:.2f}",
                        to_email, from_email, password
                    )

        # 2. RSI Alert
        rsi = latest_data["RSI"] if "RSI" in latest_data and not pd.isna(latest_data["RSI"]) else 50
        if rsi > rsi_threshold_high or rsi < rsi_threshold_low:
            condition = "overbought" if rsi > rsi_threshold_high else "oversold"
            alert_msg = f"🚨 RSI Alert for {ticker}: RSI {rsi:.2f} is {condition}!"
            if alert_msg not in st.session_state.alerts:
                st.session_state.alerts.append(alert_msg)
                st.warning(alert_msg)
                if enable_email_alerts and to_email and from_email and password:
                    send_alert_email(
                        f"RSI Alert for {ticker}",
                        f"RSI: {rsi:.2f} is {condition} for {ticker}. Price: ₹{latest_price:.2f}",
                        to_email, from_email, password
                    )
        
        # 3. Supertrend Signal Change
        if len(data) >= 2:
            current_supertrend = latest_data["supertrend"] if "supertrend" in latest_data and not pd.isna(latest_data["supertrend"]) else 0
            prev_supertrend = data["supertrend"].iloc[-2] if "supertrend" in data.columns and not pd.isna(data["supertrend"].iloc[-2]) else 0
            
            if current_supertrend != prev_supertrend:
                signal = "BUY" if current_supertrend == 1 else "SELL"
                alert_msg = f"🚨 Supertrend Signal Change for {ticker}: {signal} signal triggered!"
                if alert_msg not in st.session_state.alerts:
                    st.session_state.alerts.append(alert_msg)
                    st.warning(alert_msg)
                    if enable_email_alerts and to_email and from_email and password:
                        send_alert_email(
                            f"Supertrend Alert for {ticker}",
                            f"Supertrend {signal} signal for {ticker}. Price: ₹{latest_price:.2f}",
                            to_email, from_email, password
                        )
        
        # 4. MACD Crossover
        if len(data) >= 2:
            current_macd = latest_data["MACD"] if "MACD" in latest_data and not pd.isna(latest_data["MACD"]) else 0
            current_signal = latest_data["MACD_Signal"] if "MACD_Signal" in latest_data and not pd.isna(latest_data["MACD_Signal"]) else 0
            prev_macd = data["MACD"].iloc[-2] if "MACD" in data.columns and not pd.isna(data["MACD"].iloc[-2]) else 0
            prev_signal = data["MACD_Signal"].iloc[-2] if "MACD_Signal" in data.columns and not pd.isna(data["MACD_Signal"].iloc[-2]) else 0
            
            current_above = current_macd > current_signal
            prev_above = prev_macd > prev_signal
            
            if current_above != prev_above:
                signal = "BUY" if current_above else "SELL"
                alert_msg = f"🚨 MACD Crossover for {ticker}: {signal} signal!"
                if alert_msg not in st.session_state.alerts:
                    st.session_state.alerts.append(alert_msg)
                    st.warning(alert_msg)
                    if enable_email_alerts and to_email and from_email and password:
                        send_alert_email(
                            f"MACD Crossover for {ticker}",
                            f"MACD {signal} signal for {ticker}. Price: ₹{latest_price:.2f}",
                            to_email, from_email, password
                        )

def plot_advanced_stock_data(ticker, data):
    """Plots the stock data with advanced indicators."""
    if data is None or data.empty:
        st.warning(f"Could not display graph for {ticker}.")
        return
    
    # Calculate all indicators
    data_for_graph = calculate_advanced_indicators(data)
    if data_for_graph is None:
        st.warning(f"Insufficient data for {ticker} to calculate indicators.")
        return
    
    # Get momentum scores for the entire period
    momentum_scores = []
    signals_list = []
    for i in range(20, len(data_for_graph)):  # Start after we have enough data for indicators
        subset = data_for_graph.iloc[:i+1]
        score, signals = calculate_momentum_score(subset)
        momentum_scores.append(score)
        signals_list.append(signals.get("Overall Momentum", {}).get("signal", "neutral"))
    
    # Pad with NaN values for the start where we don't have enough data
    padding = [np.nan] * (len(data_for_graph) - len(momentum_scores))
    data_for_graph["Momentum_Score"] = padding + momentum_scores
    
    
    # Create our plot
    fig = go.Figure()
    
    # Price candles
    fig.add_trace(go.Candlestick(
        x=data_for_graph.index,
        open=data_for_graph['Open'],
        high=data_for_graph['High'],
        low=data_for_graph['Low'],
        close=data_for_graph['Close'],
        name="Price",
        increasing_line_color='#26a69a', 
        decreasing_line_color='#ef5350'
    ))

    # Add EMA lines
    fig.add_trace(go.Scatter(
        x=data_for_graph.index,
        y=data_for_graph['EMA_9'],
        mode='lines',
        name='EMA 9',
        line=dict(color='#f06292', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=data_for_graph.index,
        y=data_for_graph['EMA_21'],
        mode='lines',
        name='EMA 21',
        line=dict(color='#29b6f6', width=1)
    ))
    
    # Add Bollinger Bands
    if 'BB_Upper' in data_for_graph.columns and not data_for_graph['BB_Upper'].isnull().all():
        fig.add_trace(go.Scatter(
            x=data_for_graph.index,
            y=data_for_graph['BB_Upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='rgba(150, 150, 150, 0.5)', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=data_for_graph.index,
            y=data_for_graph['BB_Lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='rgba(150, 150, 150, 0.5)', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(150, 150, 150, 0.1)'
        ))
    
    # Add Supertrend
    if 'supertrend_upper' in data_for_graph.columns:
        fig.add_trace(go.Scatter(
            x=data_for_graph.index,
            y=data_for_graph['supertrend_upper'],
            mode='lines',
            name='Supertrend Upper',
            line=dict(color='rgba(255, 0, 0, 0.5)', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=data_for_graph.index,
            y=data_for_graph['supertrend_lower'],
            mode='lines',
            name='Supertrend Lower',
            line=dict(color='rgba(0, 255, 0, 0.5)', width=1)
        ))
    
    # Update layout
    fig.update_layout(
        title='Price Chart with Technical Indicators',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Add subplot for indicators
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI Plot
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(
            x=data_for_graph.index,
            y=data_for_graph['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='#7b1fa2', width=2)
        ))
        
        # Add RSI overbought/oversold lines
        rsi_fig.add_trace(go.Scatter(
            x=[data_for_graph.index[0], data_for_graph.index[-1]],
            y=[70, 70],
            mode='lines',
            name='Overbought',
            line=dict(color='rgba(255, 0, 0, 0.5)', width=1, dash='dash')
        ))
        
        rsi_fig.add_trace(go.Scatter(
            x=[data_for_graph.index[0], data_for_graph.index[-1]],
            y=[30, 30],
            mode='lines',
            name='Oversold',
            line=dict(color='rgba(0, 255, 0, 0.5)', width=1, dash='dash')
        ))
        
        rsi_fig.update_layout(
            title='RSI (Relative Strength Index)',
            xaxis_title='Date',
            yaxis_title='RSI Value',
            height=300,
            template='plotly_dark',
            yaxis=dict(range=[0, 100]),
            showlegend=False
        )
        
        st.plotly_chart(rsi_fig, use_container_width=True)
    
    with col2:
        # MACD Plot
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(
            x=data_for_graph.index,
            y=data_for_graph['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='#64b5f6', width=2)
        ))
        
        macd_fig.add_trace(go.Scatter(
            x=data_for_graph.index,
            y=data_for_graph['MACD_Signal'],
            mode='lines',
            name='Signal',
            line=dict(color='#ff7043', width=1)
        ))
        
        # Add MACD histogram
        macd_colors = ['rgba(0, 255, 0, 0.5)' if val >= 0 else 'rgba(255, 0, 0, 0.5)' 
                       for val in data_for_graph['MACD_Histogram']]
        
        macd_fig.add_trace(go.Bar(
            x=data_for_graph.index,
            y=data_for_graph['MACD_Histogram'],
            name='Histogram',
            marker_color=macd_colors
        ))
        
        macd_fig.update_layout(
            title='MACD (Moving Average Convergence Divergence)',
            xaxis_title='Date',
            yaxis_title='MACD Value',
            height=300,
            template='plotly_dark',
            showlegend=False
        )
        
        st.plotly_chart(macd_fig, use_container_width=True)
    
    # Momentum Score Plot
    momentum_fig = go.Figure()
    
    # Plot the momentum score
    momentum_fig.add_trace(go.Scatter(
        x=data_for_graph.index,
        y=data_for_graph['Momentum_Score'],
        mode='lines',
        name='Momentum Score',
        line=dict(color='#ffb300', width=2)
    ))
    
    # Add neutral line at 50
    momentum_fig.add_trace(go.Scatter(
        x=[data_for_graph.index[0], data_for_graph.index[-1]],
        y=[50, 50],
        mode='lines',
        name='Neutral',
        line=dict(color='rgba(150, 150, 150, 0.5)', width=1, dash='dash')
    ))
    
    momentum_fig.update_layout(
        title='Technical Momentum Score',
        xaxis_title='Date',
        yaxis_title='Score (0-100)',
        height=250,
        template='plotly_dark',
        yaxis=dict(range=[0, 100]),
        showlegend=False
    )
    
    st.plotly_chart(momentum_fig, use_container_width=True)

def display_strength_data(ticker, data, momentum_score, signals, session_state, unique_id):
    """Displays strength data and indicator values with unique button keys."""
    col1, col2, col3, col4 = st.columns([2, 4, 4, 2])
    
    with col1:
        st.write(f"**{ticker}**")
    with col2:
        # Display key signals
        signal_html = ""
        for key in ["MACD", "RSI", "Supertrend"]:
            if key in signals:
                signal_class = f"indicator-{signals[key]['signal']}"
                signal_html += f'<span class="indicator-pill {signal_class}">{key}: {signals[key]["value"]}</span> '
        
        st.markdown(signal_html, unsafe_allow_html=True)
    with col3:
        st.write(f"Momentum: **{momentum_score:.2f}**")
        # Color-coded momentum direction
        direction = signals["Overall Momentum"]["signal"] if "Overall Momentum" in signals else "neutral"
        direction_html = f'<span class="indicator-pill indicator-{direction}">Signal: {direction.upper()}</span>'
        st.markdown(direction_html, unsafe_allow_html=True)
    with col4:
        # Use unique_id to ensure unique key for each button
        # Store ticker and data directly in session state on button click
        button_key = f"graph_{ticker}_{unique_id}_{int(time.time() * 1000)}"
        if st.button("Show Graph", key=button_key, on_click=update_state, args=(ticker, data, session_state)):
            pass  # Callback handles the state update

def update_state(ticker, data, session_state):
    """Callback function to update session state."""
    session_state.selected_ticker = ticker
    session_state.selected_data = data
    session_state.show_graph = True
    time.sleep(0.5)  # Delay for state persistence

def close_graph(session_state):
    """Callback function to close the graph."""
    session_state.show_graph = False
    session_state.selected_ticker = None
    session_state.selected_data = None

# Main app function
def main():
    # Initialize session state
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = None
    if 'selected_data' not in st.session_state:
        st.session_state.selected_data = None
    if 'show_graph' not in st.session_state:
        st.session_state.show_graph = False
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'scan_id' not in st.session_state:
        st.session_state.scan_id = None
    
    st.title("Advanced Technical Momentum Scanner")
    
    # Sidebar settings
    st.sidebar.header("Scan Settings")
    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    ticker_list = ["^NSEI", "^NSEBANK", "^BSESN", "^NSMIDCP", "NIFTY_FIN_SERVICE.NS", "AARTIIND.NS", "ABB.NS", "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", "ADANIENSOL.NS", "ADANIENT.NS", "ADANIGREEN.NS",
    "ADANIPORTS.NS", "ALKEM.NS", "AMBUJACEM.NS", "ANGELONE.NS","APLAPOLLO.NS", "APOLLOHOSP.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASIANPAINT.NS", "ASTRAL.NS", "ATGL.NS", "AUBANK.NS", "AUROPHARMA.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS","BALKRISIND.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "BANKINDIA.NS", "BEL.NS", "BERGEPAINT.NS",
    "BHARATFORG.NS", "BHARTIARTL.NS", "BHEL.NS", "BIOCON.NS", "BOSCHLTD.NS", "BPCL.NS", "BRITANNIA.NS", "BSE.NS", "BSOFT.NS", "CAMS.NS", "CANBK.NS", "CDSL.NS", "CESC.NS", "CGPOWER.NS", "CHAMBLFERT.NS",
    "CHOLAFIN.NS", "CIPLA.NS", "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS", "CONCOR.NS", "CROMPTON.NS", "CUMMINSIND.NS", "CYIENT.NS", "DABUR.NS", "DALBHARAT.NS", "DEEPAKNTR.NS", "DELHIVERY.NS",
    "DIVISLAB.NS", "DIXON.NS", "DLF.NS", "DMART.NS", "DRREDDY.NS", "EICHERMOT.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", "GAIL.NS", "GLENMARK.NS", "GMRAIRPORT.NS", "GODREJCP.NS",
    "GODREJPROP.NS", "GRANULES.NS", "GRASIM.NS", "HAL.NS", "HAVELLS.NS", "HCLTECH.NS", "HDFCAMC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HFCL.NS", "HINDALCO.NS", "HINDCOPPER.NS",
    "HINDPETRO.NS", "HINDUNILVR.NS", "HUDCO.NS", "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS","IDEA.NS", "IDFCFIRSTB.NS", "IEX.NS", "IGL.NS", "IIFL.NS", "INDHOTEL.NS", "INDIANB.NS", "INDIGO.NS",
    "INDUSINDBK.NS", "INDUSTOWER.NS", "INFY.NS", "IOC.NS", "IRB.NS", "IRCTC.NS", "IREDA.NS", "IRFC.NS","ITC.NS", "JINDALSTEL.NS", "JIOFIN.NS", "JKCEMENT.NS", "JSL.NS", "JSWENERGY.NS", "JSWSTEEL.NS",
    "JUBLFOOD.NS", "KALYANKJIL.NS", "KOTAKBANK.NS", "KPITTECH.NS", "LAURUSLABS.NS", "LICHSGFIN.NS", "LICI.NS", "LODHA.NS", "LT.NS", "LTF.NS", "LTIM.NS", "LTTS.NS", "LUPIN.NS", "M&M.NS", "M&MFIN.NS",
    "MANAPPURAM.NS", "MARICO.NS", "MARUTI.NS", "MAXHEALTH.NS", "MCX.NS", "MFSL.NS", "MGL.NS", "MOTHERSON.NS", "MPHASIS.NS", "MRF.NS", "MUTHOOTFIN.NS", "NATIONALUM.NS", "NAUKRI.NS", "NBCC.NS",
    "NCC.NS", "NESTLEIND.NS", "NHPC.NS", "NMDC.NS", "NTPC.NS", "NYKAA.NS", "OBEROIRLTY.NS", "OFSS.NS", "OIL.NS", "ONGC.NS", "PAGEIND.NS", "PATANJALI.NS", "PAYTM.NS", "PEL.NS", "PERSISTENT.NS",
    "PETRONET.NS", "PFC.NS", "PHOENIXLTD.NS", "PIDILITIND.NS", "PIIND.NS", "PNB.NS", "POLICYBZR.NS","POLYCAB.NS", "POONAWALLA.NS", "POWERGRID.NS", "PRESTIGE.NS", "RAMCOCEM.NS", "RBLBANK.NS",
    "RECLTD.NS", "RELIANCE.NS", "SAIL.NS", "SBICARD.NS", "SBILIFE.NS", "SBIN.NS", "SHREECEM.NS", "SHRIRAMFIN.NS", "SIEMENS.NS", "SJVN.NS", "SOLARINDS.NS", "SONACOMS.NS", "SRF.NS", "SUNPHARMA.NS",
    "SUPREMEIND.NS", "SYNGENE.NS", "TATACHEM.NS", "TATACOMM.NS", "TATACONSUM.NS", "TATAELXSI.NS", "TATAMOTORS.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TATATECH.NS", "TCS.NS", "TECHM.NS", "TIINDIA.NS",
    "TITAGARH.NS", "TITAN.NS", "TORNTPHARM.NS", "TORNTPOWER.NS", "TRENT.NS", "TVSMOTOR.NS", "ULTRACEMCO.NS", "UNIONBANK.NS", "UNITDSPR.NS", "UPL.NS", "VBL.NS", "VEDL.NS", "VOLTAS.NS",
    "WIPRO.NS", "YESBANK.NS", "ZOMATO.NS", "ZYDUSLIFE.NS"]
    #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    # Period selection
    period = st.sidebar.selectbox("Time Period", 
                                 ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"],
                                 index=2)
    
    # Interval selection
    interval = st.sidebar.selectbox("Time Interval", 
                                   ["15m", "1h", "1d", "5d", "1wk", "1mo"],
                                   index=1)
    
    # Alert settings
    st.sidebar.header("Alert Settings")
    
    momentum_threshold = st.sidebar.slider("Momentum Threshold", 10, 40, 20, 5)
    rsi_threshold_high = st.sidebar.slider("RSI Overbought", 60, 90, 70, 5)
    rsi_threshold_low = st.sidebar.slider("RSI Oversold", 10, 40, 30, 5)
    
    # Email alerts
    enable_email_alerts = st.sidebar.checkbox("Enable Email Alerts")
    
    to_email = ""
    from_email = ""
    password = ""
    
    if enable_email_alerts:
        to_email = st.sidebar.text_input("Your Email")
        from_email = st.sidebar.text_input("Gmail Account")
        password = st.sidebar.text_input("App Password", type="password", 
                                        help="Use an App Password if you have 2FA enabled")
    
    # Clear alerts button
    if "alerts" in st.session_state and st.session_state.alerts:
        if st.sidebar.button("Clear Alerts"):
            st.session_state.alerts = []
            st.sidebar.success("Alerts cleared!")
    
    # Adjust indicator weights
    weights = add_weight_adjustment_section()
    
    # Add a container for the graph that's shown regardless of other state
    graph_container = st.container()
    
    # Run scanner
    run_scanner = st.button("Run Technical Scan")
    
    if run_scanner:
        progress_bar = st.progress(0)
        results = []
        
        # Generate a unique ID for this scan to ensure unique button keys
        scan_id = datetime.now().strftime("%Y%m%d%H%M%S")
        st.session_state.scan_id = scan_id
        
        for i, ticker in enumerate(ticker_list):
            progress_bar.progress((i + 1) / len(ticker_list))
            
            data = get_asset_data(ticker, period, interval)
            if data is not None and not data.empty:
                try:
                    data_with_indicators = calculate_advanced_indicators(data)
                    if data_with_indicators is not None and not data_with_indicators.empty:
                        momentum_score, signals = calculate_momentum_score(data_with_indicators, weights)
                        results.append({
                            "ticker": ticker,
                            "data": data_with_indicators,
                            "momentum_score": momentum_score,
                            "signals": signals
                        })
                        
                        # Check for alerts
                        check_alerts(ticker, data_with_indicators, momentum_score, signals,
                                   momentum_threshold, rsi_threshold_high, rsi_threshold_low,
                                   enable_email_alerts, to_email, from_email, password)
                except Exception as e:
                    st.error(f"Error processing {ticker}: {str(e)}")
        
        progress_bar.empty()
        
        if results:
            # Store results in session state
            st.session_state.results = results
            
            # Sort results by momentum score (descending)
            results.sort(key=lambda x: x["momentum_score"], reverse=True)
            
            # Display strongest bullish signals
            st.subheader("Strongest Bullish Signals")
            bullish_results = [r for r in results if r["momentum_score"] > 50]
            if bullish_results:
                for result in bullish_results[:5]:  # Show top 5
                    display_strength_data(result["ticker"], result["data"], 
                                        result["momentum_score"], result["signals"], 
                                        st.session_state, scan_id)
            else:
                st.info("No strong bullish signals detected.")
            
            # Display strongest bearish signals
            st.subheader("Strongest Bearish Signals")
            bearish_results = [r for r in results if r["momentum_score"] < 50]
            if bearish_results:
                bearish_sorted = sorted(bearish_results, key=lambda x: x["momentum_score"])
                for result in bearish_sorted[:5]:  # Show top 5
                    display_strength_data(result["ticker"], result["data"], 
                                        result["momentum_score"], result["signals"], 
                                        st.session_state, scan_id)
            else:
                st.info("No strong bearish signals detected.")
            
            # Display all results in a table
            st.subheader("All Results")
            table_data = []
            for result in results:
                signal = result["signals"].get("Overall Momentum", {}).get("signal", "neutral")
                rsi = result["signals"].get("RSI", {}).get("value", "N/A")
                macd = result["signals"].get("MACD", {}).get("value", "N/A")
                supertrend = result["signals"].get("Supertrend", {}).get("value", "N/A")
                table_data.append({
                    "Ticker": result["ticker"],
                    "Momentum Score": f"{result['momentum_score']:.2f}",
                    "Signal": signal.upper(),
                    "RSI": rsi,
                    "MACD": macd,
                    "Supertrend": supertrend
                })
            
            st.dataframe(pd.DataFrame(table_data).set_index("Ticker"), use_container_width=True)
        else:
            st.warning("No results found. Check your ticker symbols and try again.")
    
    # Display graph for selected ticker - this should happen AFTER all other UI elements
    with graph_container:
        
        if st.session_state.show_graph and st.session_state.selected_ticker and st.session_state.selected_data is not None and not st.session_state.selected_data.empty:

            latest_data = st.session_state.selected_data.iloc[-1] # Last row of DataFrame
            latest_open  = latest_data['Open']
            latest_high  = latest_data['High']
            latest_low   = latest_data['Low']
            latest_close = latest_data['Close']
        
            st.header(f"{st.session_state.selected_ticker}")
            st.subheader(f'O: {latest_open:.2f} - H: {latest_high:.2f} - L: {latest_low:.2f} - C: {latest_close:.2f}')
            try:
                plot_advanced_stock_data(st.session_state.selected_ticker, st.session_state.selected_data)
                if st.button("Close Graph", on_click=close_graph, args=(st.session_state,)):
                    pass  # Callback handles the state update
            except Exception as e:
                st.error(f"Graph Error: {str(e)}")

if __name__ == "__main__":
    main()