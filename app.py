import os
import time
import requests
import pandas as pd
import streamlit as st
from google import genai

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="ICT/SMC AI Agent Dashboard",
    page_icon="📈",
    layout="centered"
)

st.title("📈 ICT / SMC AI Trading Agent")
st.caption("Bar bar terminal-e code chalano charai ekhane shohoje market analyze korun.")

# ============================================================
# SIDEBAR CONFIGURATION
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key Input
    user_api_key = st.text_input(
        "Gemini API Key", 
        type="password", 
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Google AI Studio theke pawa API key ekhane din."
    )
    
    # Model Selection
    selected_model = st.selectbox(
        "Select Model", 
        ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"]
    )
    
    st.info("💡 Note: Error (429/503) ebola model ba API key bodle try korun.")

# ============================================================
# REAL-TIME MARKET DATA FETCHER
# ============================================================
def get_live_market_data(symbol="XAUUSD"):
    symbol = symbol.upper()
    crypto_map = {
        "BTC": "BTCUSDT",
        "BTCUSD": "BTCUSDT",
        "ETH": "ETHUSDT",
        "ETHUSD": "ETHUSDT",
    }

    if symbol in crypto_map:
        b_sym = crypto_map[symbol]
        url = f"https://api.binance.com/api/v3/klines?symbol={b_sym}&interval=15m&limit=50"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(
                data,
                columns=["t", "o", "h", "l", "c", "v", "ct", "q", "n", "tb", "tbq", "i"]
            )
            df["Open"] = df["o"].astype(float)
            df["High"] = df["h"].astype(float)
            df["Low"] = df["l"].astype(float)
            df["Close"] = df["c"].astype(float)
            return df[["Open", "High", "Low", "Close"]]
        except Exception as e:
            st.error(f"⚠️ Crypto data error: {e}")
            return pd.DataFrame()

    try:
        supported_oanda = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]
        if symbol in supported_oanda:
            tv_symbol = f"OANDA:{symbol}"
        else:
            tv_symbol = f"CAPITALCOM:{symbol}"

        url = f"https://scanner.tradingview.com/symbol?symbol={tv_symbol}&fields=close,open,high,low"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        res = response.json()

        c = float(res["close"])
        h = float(res["high"])
        l = float(res["low"])
        o = float(res["open"])

        df = pd.DataFrame(index=range(20))
        df["Close"] = c
        df["High"] = h
        df["Low"] = l
        df["Open"] = o
        return df
    except Exception as e:
        st.error(f"⚠️ Market Data Fetch Error: {e}")
        return pd.DataFrame()

# ============================================================
# ICT & SMC ENGINE
# ============================================================
def analyze_smc_structure(df):
    if df.empty:
        return {}

    live_price = float(df["Close"].iloc[-1])
    recent_high = float(df["High"].max())
    recent_low = float(df["Low"].min())
    eq = (recent_high + recent_low) / 2.0

    pd_zone = "DISCOUNT ZONE" if live_price < eq else "PREMIUM ZONE"

    ssl_sweep = live_price > recent_low and float(df["Low"].iloc[-1]) <= recent_low
    bsl_sweep = live_price < recent_high and float(df["High"].iloc[-1]) >= recent_high

    return {
        "live_price": live_price,
        "pd_zone": pd_zone,
        "high": recent_high,
        "low": recent_low,
        "eq": eq,
        "ssl_sweep": ssl_sweep,
        "bsl_sweep": bsl_sweep,
    }

# ============================================================
# GEMINI AI ANALYSIS
# ============================================================
def generate_ai_prediction(symbol, smc_data, api_key, model_name):
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an ICT/SMC market-analysis assistant.

Analyze the supplied market metrics for {symbol}.

Market data:
- Live Market Price: {smc_data['live_price']}
- PD Zone: {smc_data['pd_zone']}
- Recent High: {smc_data['high']}
- Recent Low: {smc_data['low']}
- Equilibrium (50%): {smc_data['eq']}
- SSL Swept: {smc_data['ssl_sweep']}
- BSL Swept: {smc_data['bsl_sweep']}

Use ICT/SMC concepts such as:
- Premium / Discount
- Buy-side liquidity
- Sell-side liquidity
- Liquidity sweeps
- Market structure
- Fair Value Gaps
- PD Arrays

Important:
Do not claim certainty.
The confidence percentage is a qualitative model estimate, not a statistically validated probability.

Return exactly this format:

🎯 INSTITUTIONAL ICT AI REPORT: {symbol}
---------------------------------------------------
- Action: [BUY / SELL / WAIT]
- Confidence: [e.g. 75%]
- Exact Entry: [Price or N/A]
- Stop Loss (SL): [Price or N/A]
- Take Profit 1 (TP1): [Price or N/A]
- Take Profit 2 (TP2): [Price or N/A]
- Take Profit 3 (TP3): [Price or N/A]

🧠 Rationale:
[2-4 lines explaining the market structure, liquidity sweep, FVG and PD Array context.]
"""

    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            error_text = str(e)
            if ("503" in error_text or "429" in error_text) and attempt < 3:
                wait_time = 4 * (attempt + 1)
                time.sleep(wait_time)
            else:
                return f"❌ Gemini API Error: {e}"

    return "❌ Gemini API Error: All retry attempts failed."

# ============================================================
# MAIN USER INTERFACE
# ============================================================
target_symbol = st.selectbox(
    "Select Currency Pair / Symbol",
    ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD"]
)

if st.button("🚀 Run Analysis", use_container_width=True):
    if not user_api_key:
        st.error("⚠️ Meherbaani kore Sidebar-e apnar Gemini API Key pradan korun.")
    else:
        with st.spinner(f"Fetching market data & generating AI report for {target_symbol}..."):
            df = get_live_market_data(target_symbol)
            if df.empty:
                st.error(f"❌ {target_symbol}-er jonno live price paoa jayni.")
            else:
                smc_data = analyze_smc_structure(df)
                if not smc_data:
                    st.error("❌ Market structure analyze korte birtho hoyeche.")
                else:
                    report = generate_ai_prediction(
                        target_symbol, 
                        smc_data, 
                        user_api_key, 
                        selected_model
                    )
                    st.success("Analysis Completed!")
                    st.text_area("Market Analysis Report", value=report, height=350)
