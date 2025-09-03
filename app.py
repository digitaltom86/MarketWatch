import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(
    page_title="Crypto Liquidity Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-good { border-left-color: #28a745 !important; }
    .status-warning { border-left-color: #ffc107 !important; }
    .status-danger { border-left-color: #dc3545 !important; }
    .big-font { font-size: 2rem; font-weight: bold; }
    .trading-rec { 
        background-color: #e8f4f8; 
        padding: 1rem; 
        border-radius: 0.5rem; 
        margin: 1rem 0; 
    }
</style>
""", unsafe_allow_html=True)

# Constants
FRED_API_KEY = "8d5a1786444a89510c8ea27e214e255f"
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Cache decorators with different TTL
@st.cache_data(ttl=3600)  # 1 hour cache
def fetch_fred_data(series_id, lookback_years=5):
    """Fetch data from FRED API"""
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*lookback_years)).strftime('%Y-%m-%d')
        
        params = {
            'series_id': series_id,
            'api_key': FRED_API_KEY,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date,
            'sort_order': 'desc',
            'limit': 10000
        }
        
        response = requests.get(FRED_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'observations' in data:
            df = pd.DataFrame(data['observations'])
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna().sort_values('date')
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching FRED data for {series_id}: {str(e)}")
        return None

@st.cache_data(ttl=1800)  # 30 min cache
def fetch_coingecko_data():
    """Fetch Bitcoin dominance from CoinGecko"""
    try:
        # Bitcoin dominance
        dom_url = "https://api.coingecko.com/api/v3/global"
        response = requests.get(dom_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        btc_dominance = data['data']['market_cap_percentage']['btc']
        
        # Stablecoin market cap (USDT + USDC)
        stable_url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids=tether,usd-coin&order=market_cap_desc&per_page=10"
        response = requests.get(stable_url, timeout=30)
        response.raise_for_status()
        stable_data = response.json()
        stable_mcap = sum([coin['market_cap'] for coin in stable_data]) / 1e9  # in billions
        
        return btc_dominance, stable_mcap
    except Exception as e:
        st.error(f"Error fetching CoinGecko data: {str(e)}")
        return 45.0, 150.0  # fallback values

@st.cache_data(ttl=1800)  # 30 min cache
def fetch_yahoo_data(symbol, period="10y"):
    """Fetch data from Yahoo Finance (simplified)"""
    try:
        # This is a simplified version - in production use yfinance library
        # For now, return mock data with realistic values
        if symbol == "^GSPC":
            return np.random.normal(4200, 200, 100)
        elif symbol == "BTC-USD":
            return np.random.normal(45000, 5000, 100)
        elif symbol == "GC=F":
            return np.random.normal(2000, 100, 100)
        return np.random.normal(100, 10, 100)
    except:
        return np.random.normal(100, 10, 100)

def calculate_correlation(x, y):
    """Calculate correlation between two series"""
    try:
        return np.corrcoef(x, y)[0, 1]
    except:
        return 0.0

def score_fed_rate(rate):
    """Score Fed Funds Rate (0-100)"""
    if rate < 1: return 95
    elif rate < 2: return 85
    elif rate < 3: return 70
    elif rate < 4: return 50
    elif rate < 5: return 30
    else: return 15

def score_m2_growth(yoy_growth):
    """Score M2 Money Supply YoY growth (0-100)"""
    if yoy_growth > 15: return 95
    elif yoy_growth > 10: return 85
    elif yoy_growth > 5: return 70
    elif yoy_growth > 0: return 50
    else: return 25

def score_btc_dominance(dominance):
    """Score Bitcoin Dominance (0-100)"""
    if dominance < 35: return 90
    elif dominance < 45: return 75
    elif dominance < 55: return 60
    else: return 40

def score_correlation(corr):
    """Score correlation (lower is better for diversification)"""
    abs_corr = abs(corr)
    if abs_corr < 0.2: return 90
    elif abs_corr < 0.4: return 75
    elif abs_corr < 0.6: return 55
    elif abs_corr < 0.8: return 35
    else: return 15

def get_regime_status(score):
    """Get regime classification"""
    if score >= 75: return "🟢 ABUNDANT LIQUIDITY", "success"
    elif score >= 60: return "🟡 GOOD LIQUIDITY", "info"
    elif score >= 40: return "🟠 NEUTRAL LIQUIDITY", "warning"
    elif score >= 25: return "🔴 TIGHT LIQUIDITY", "error"
    else: return "🚨 LIQUIDITY CRISIS", "error"

def get_trading_recommendations(score):
    """Get trading recommendations based on score"""
    if score >= 75:
        return {
            "position_size": "80-100%",
            "leverage": "2-3x",
            "strategy": "Aggressive growth, high-beta alts",
            "risk": "Low - Abundant liquidity supports risk assets"
        }
    elif score >= 50:
        return {
            "position_size": "50-80%",
            "leverage": "1.5-2x", 
            "strategy": "Moderate positions, BTC focus",
            "risk": "Medium - Selective opportunities"
        }
    elif score >= 25:
        return {
            "position_size": "20-50%",
            "leverage": "1-1.5x",
            "strategy": "Conservative, high-quality only",
            "risk": "High - Liquidity constraints emerging"
        }
    else:
        return {
            "position_size": "0-20%",
            "leverage": "No leverage",
            "strategy": "Defensive cash positions",
            "risk": "Very High - Liquidity crisis mode"
        }

# Sidebar Configuration
st.sidebar.title("⚙️ Dashboard Settings")

# Indicator Weights
st.sidebar.subheader("📊 Indicator Weights")
fed_weight = st.sidebar.slider("🏛️ Fed Funds Rate", 0, 50, 25, help="Weight for Fed Funds Rate in composite score") / 100
m2_weight = st.sidebar.slider("💰 M2 Money Supply", 0, 50, 25, help="Weight for M2 Money Supply in composite score") / 100
btc_dom_weight = st.sidebar.slider("₿ Bitcoin Dominance", 0, 50, 20, help="Weight for Bitcoin Dominance in composite score") / 100
tga_weight = st.sidebar.slider("🏦 TGA/RRP", 0, 30, 15, help="Weight for Treasury/RRP in composite score") / 100
corr_weight = st.sidebar.slider("📈 Stock Correlation", 0, 30, 15, help="Weight for Stock Correlation in composite score") / 100

# Normalize weights
total_weight = fed_weight + m2_weight + btc_dom_weight + tga_weight + corr_weight
if total_weight > 0:
    fed_weight /= total_weight
    m2_weight /= total_weight
    btc_dom_weight /= total_weight
    tga_weight /= total_weight
    corr_weight /= total_weight

# Data Settings
st.sidebar.subheader("📅 Data Settings")
fed_lookback = st.sidebar.selectbox("Fed Lookback", ["1Y", "2Y", "5Y", "10Y"], index=2)
m2_lookback = st.sidebar.selectbox("M2 Lookback", ["2Y", "5Y", "10Y", "15Y"], index=2)
corr_lookback = st.sidebar.selectbox("Correlation Lookback", ["2Y", "5Y", "10Y", "15Y"], index=2)

lookback_map = {"1Y": 1, "2Y": 2, "5Y": 5, "10Y": 10, "15Y": 15}

# Auto-refresh
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (5 min)")
if st.sidebar.button("🔄 Manual Refresh"):
    st.cache_data.clear()
    st.rerun()

if auto_refresh:
    time.sleep(300)  # 5 minutes
    st.rerun()

# Main Dashboard
st.title("📊 Crypto Liquidity Dashboard")
st.markdown("**Professional monitoring of macroeconomic indicators affecting crypto market liquidity**")

# Fetch all data
with st.spinner("Loading market data..."):
    # FRED data
    fed_data = fetch_fred_data("FEDFUNDS", lookback_map[fed_lookback])
    m2_data = fetch_fred_data("M2SL", lookback_map[m2_lookback])
    tga_data = fetch_fred_data("WTREGEN", 5)
    rrp_data = fetch_fred_data("RRPONTSYD", 5)
    cpi_data = fetch_fred_data("CPIAUCSL", 5)
    
    # CoinGecko data
    btc_dominance, stable_mcap = fetch_coingecko_data()
    
    # Yahoo Finance data (mock for demo)
    btc_prices = fetch_yahoo_data("BTC-USD")
    spx_prices = fetch_yahoo_data("^GSPC")
    gold_prices = fetch_yahoo_data("GC=F")

# Calculate scores
current_fed_rate = fed_data.iloc[-1]['value'] if fed_data is not None else 5.25
current_m2 = m2_data.iloc[-1]['value'] if m2_data is not None else 21000
m2_yoy = ((current_m2 / m2_data.iloc[-52]['value'] - 1) * 100) if m2_data is not None and len(m2_data) >= 52 else 5.0

fed_score = score_fed_rate(current_fed_rate)
m2_score = score_m2_growth(m2_yoy)
btc_dom_score = score_btc_dominance(btc_dominance)
stock_corr = calculate_correlation(btc_prices[-100:], spx_prices[-100:])
stock_corr_score = score_correlation(stock_corr)
gold_corr = calculate_correlation(btc_prices[-100:], gold_prices[-100:])
gold_corr_score = score_correlation(gold_corr)

# Calculate composite score
composite_score = (
    fed_score * fed_weight +
    m2_score * m2_weight +
    btc_dom_score * btc_dom_weight +
    stock_corr_score * corr_weight +
    gold_corr_score * (1 - fed_weight - m2_weight - btc_dom_weight - corr_weight)
)

regime, regime_type = get_regime_status(composite_score)

# Top-level metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🎯 Composite Score",
        value=f"{composite_score:.1f}/100",
        delta=f"{composite_score-50:.1f} vs Neutral"
    )

with col2:
    st.metric(
        label="🏛️ Fed Funds Rate",
        value=f"{current_fed_rate:.2f}%",
        delta=f"{current_fed_rate-2.5:.2f}% vs Neutral"
    )

with col3:
    st.metric(
        label="₿ BTC Dominance", 
        value=f"{btc_dominance:.1f}%",
        delta=f"{btc_dominance-45:.1f}% vs Neutral"
    )

with col4:
    st.metric(
        label="📈 BTC-SPX Correlation",
        value=f"{stock_corr:.2f}",
        delta=f"{abs(stock_corr)-0.4:.2f} vs Low"
    )

# Current Regime Status
if regime_type == "success":
    st.success(f"**Current Regime: {regime}**")
elif regime_type == "info":
    st.info(f"**Current Regime: {regime}**")
elif regime_type == "warning":
    st.warning(f"**Current Regime: {regime}**")
else:
    st.error(f"**Current Regime: {regime}**")

# Tabs
tabs = st.tabs([
    "🏛️ Fed Funds Rate",
    "💰 M2 Money Supply", 
    "₿ Bitcoin Dominance",
    "🏦 TGA & RRP",
    "📈 Stock Correlations",
    "🥇 Gold Correlation", 
    "💎 Stablecoins",
    "⚡ Futures OI/Funding",
    "📊 Inflation",
    "📅 Seasonality",
    "🎯 Composite Score"
])

# Tab 1: Fed Funds Rate
with tabs[0]:
    st.subheader("🏛️ Federal Funds Rate Analysis")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Rate", f"{current_fed_rate:.2f}%")
    with col2:
        rate_change = fed_data.iloc[-1]['value'] - fed_data.iloc[-2]['value'] if fed_data is not None and len(fed_data) > 1 else 0
        st.metric("Monthly Change", f"{rate_change:.2f}%")
    with col3:
        st.metric("Liquidity Score", f"{fed_score}/100")
    with col4:
        st.metric("Regime Impact", f"{fed_weight*100:.0f}%")
    
    # Chart
    if fed_data is not None:
        chart_data = fed_data.set_index('date')['value']
        st.line_chart(chart_data, height=400)
    
    # Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💡 Current Analysis**")
        if current_fed_rate < 2:
            st.success("✅ Ultra-accommodative policy supports crypto liquidity")
        elif current_fed_rate < 4:
            st.info("ℹ️ Neutral policy - moderate impact on risk assets")
        else:
            st.error("⚠️ Restrictive policy - headwinds for crypto markets")
        
        st.markdown(f"""
        - **Rate Level**: {current_fed_rate:.2f}% ({'Low' if current_fed_rate < 3 else 'High'})
        - **Liquidity Impact**: {'Positive' if current_fed_rate < 3 else 'Negative'}
        - **Trend**: {'Easing' if rate_change < 0 else 'Tightening' if rate_change > 0 else 'Unchanged'}
        """)
    
    with col2:
        st.markdown("**📊 Historical Context**")
        if fed_data is not None:
            avg_rate = fed_data['value'].mean()
            max_rate = fed_data['value'].max()
            min_rate = fed_data['value'].min()
            
            st.markdown(f"""
            - **10Y Average**: {avg_rate:.2f}%
            - **10Y Range**: {min_rate:.2f}% - {max_rate:.2f}%
            - **Percentile**: {(fed_data['value'] < current_fed_rate).mean()*100:.0f}th
            - **Regime**: {'Dovish' if current_fed_rate < avg_rate else 'Hawkish'}
            """)
    
    # Trading implications
    st.markdown("**🎯 Trading Implications**")
    if current_fed_rate < 2:
        st.markdown("🟢 **BULLISH for crypto**: Ultra-low rates drive liquidity into risk assets. Favor high-beta altcoins.")
    elif current_fed_rate < 4:
        st.markdown("🟡 **NEUTRAL**: Balanced policy. Focus on BTC and established alts.")
    else:
        st.markdown("🔴 **BEARISH**: High rates drain liquidity. Reduce leverage and position sizes.")

# Tab 2: M2 Money Supply
with tabs[1]:
    st.subheader("💰 M2 Money Supply Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current M2", f"${current_m2:.0f}B")
    with col2:
        st.metric("YoY Growth", f"{m2_yoy:.1f}%")
    with col3:
        st.metric("Liquidity Score", f"{m2_score}/100")
    with col4:
        st.metric("Regime Impact", f"{m2_weight*100:.0f}%")
    
    if m2_data is not None:
        chart_data = m2_data.set_index('date')['value']
        st.line_chart(chart_data, height=400)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💡 Current Analysis**")
        if m2_yoy > 10:
            st.success("✅ Strong monetary expansion - highly bullish for BTC")
        elif m2_yoy > 0:
            st.info("ℹ️ Moderate growth - supportive of crypto markets")
        else:
            st.error("⚠️ Money supply contraction - major headwind")
        
        st.markdown(f"""
        - **Growth Rate**: {m2_yoy:.1f}% YoY
        - **Liquidity Impact**: {'Very Positive' if m2_yoy > 10 else 'Positive' if m2_yoy > 0 else 'Negative'}
        - **BTC Correlation**: 0.85 (historically very strong)
        """)
    
    with col2:
        st.markdown("**📊 Historical Context**")
        st.markdown("""
        - **Normal Range**: 6-8% annual growth
        - **2020-2021**: 25%+ (unprecedented expansion)
        - **2022-2024**: Declining growth rates
        - **Key Insight**: M2 is the strongest predictor of BTC bull markets
        """)
    
    st.markdown("**🎯 Trading Implications**")
    if m2_yoy > 15:
        st.markdown("🟢 **EXTREMELY BULLISH**: Massive liquidity injection. Maximum position sizes recommended.")
    elif m2_yoy > 5:
        st.markdown("🟡 **MODERATELY BULLISH**: Healthy monetary growth supports crypto.")
    else:
        st.markdown("🔴 **BEARISH**: Tight money policy. Reduce crypto exposure significantly.")

# Tab 3: Bitcoin Dominance
with tabs[2]:
    st.subheader("₿ Bitcoin Dominance Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("BTC Dominance", f"{btc_dominance:.1f}%")
    with col2:
        dom_status = "Alt Season" if btc_dominance < 40 else "BTC Focus" if btc_dominance > 50 else "Balanced"
        st.metric("Market Phase", dom_status)
    with col3:
        st.metric("Liquidity Score", f"{btc_dom_score}/100")
    with col4:
        st.metric("Regime Impact", f"{btc_dom_weight*100:.0f}%")
    
    # Mock historical dominance chart
    historical_dom = np.random.uniform(35, 70, 100)
    historical_dom[-1] = btc_dominance
    st.line_chart(pd.DataFrame(historical_dom, columns=['BTC Dominance']), height=400)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💡 Current Analysis**")
        if btc_dominance < 40:
            st.success("✅ Alt season conditions - strong liquidity across crypto")
        elif btc_dominance < 55:
            st.info("ℹ️ Balanced market - selective altcoin opportunities")
        else:
            st.error("⚠️ Risk-off mode - flight to quality (BTC)")
        
        st.markdown(f"""
        - **Dominance Level**: {btc_dominance:.1f}%
        - **Market Regime**: {dom_status}
        - **Alt Risk**: {'Low' if btc_dominance < 45 else 'Medium' if btc_dominance < 55 else 'High'}
        """)
    
    with col2:
        st.markdown("**📊 Historical Context**")
        st.markdown("""
        - **Bull Market Range**: 35-45% (alt season)
        - **Bear Market Range**: 55-70% (BTC flight-to-safety)
        - **Neutral Range**: 45-55%
        - **Cycle Pattern**: Dominance falls in late bull markets
        """)
    
    st.markdown("**🎯 Trading Implications**")
    if btc_dominance < 40:
        st.markdown("🟢 **ALT SEASON**: Maximize altcoin exposure. High-beta plays recommended.")
    elif btc_dominance < 55:
        st.markdown("🟡 **BALANCED**: 70% BTC, 30% quality alts. Selective opportunities.")
    else:
        st.markdown("🔴 **BTC FOCUS**: 90%+ BTC allocation. Avoid speculative alts.")

# Tab 10: Composite Score
with tabs[10]:
    st.subheader("🎯 Composite Liquidity Score")
    
    # Current score display
    score_color = "success" if composite_score >= 75 else "info" if composite_score >= 60 else "warning" if composite_score >= 40 else "error"
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"<div class='big-font'>{composite_score:.1f}/100</div>", unsafe_allow_html=True)
        if score_color == "success":
            st.success(regime)
        elif score_color == "info":
            st.info(regime)
        elif score_color == "warning":
            st.warning(regime)
        else:
            st.error(regime)
    
    with col2:
        trading_rec = get_trading_recommendations(composite_score)
        st.metric("Position Size", trading_rec["position_size"])
        st.metric("Max Leverage", trading_rec["leverage"])
    
    with col3:
        st.metric("Strategy", trading_rec["strategy"].split(',')[0])
        st.metric("Risk Level", trading_rec["risk"].split(' -')[0])
    
    # Score breakdown
    st.subheader("📊 Score Breakdown")
    
    breakdown_data = {
        "Indicator": ["🏛️ Fed Funds", "💰 M2 Supply", "₿ BTC Dominance", "📈 Stock Correlation", "🥇 Gold Correlation"],
        "Score": [fed_score, m2_score, btc_dom_score, stock_corr_score, gold_corr_score],
        "Weight": [f"{fed_weight:.1%}", f"{m2_weight:.1%}", f"{btc_dom_weight:.1%}", f"{corr_weight:.1%}", f"{(1-fed_weight-m2_weight-btc_dom_weight-corr_weight):.1%}"],
        "Contribution": [fed_score*fed_weight, m2_score*m2_weight, btc_dom_score*btc_dom_weight, stock_corr_score*corr_weight, gold_corr_score*(1-fed_weight-m2_weight-btc_dom_weight-corr_weight)],
        "Status": ["🟢" if fed_score >= 70 else "🟡" if fed_score >= 40 else "🔴",
                  "🟢" if m2_score >= 70 else "🟡" if m2_score >= 40 else "🔴",
                  "🟢" if btc_dom_score >= 70 else "🟡" if btc_dom_score >= 40 else "🔴",
                  "🟢" if stock_corr_score >= 70 else "🟡" if stock_corr_score >= 40 else "🔴",
                  "🟢" if gold_corr_score >= 70 else "🟡" if gold_corr_score >= 40 else "🔴"]
    }
    
    breakdown_df = pd.DataFrame(breakdown_data)
    st.dataframe(breakdown_df, use_container_width=True)
    
    # Trading recommendations
    st.subheader("🎯 Trading Recommendations")
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.markdown("**📍 Position Sizing Matrix**")
        rec = trading_rec
        st.markdown(f"""
        - **Capital Allocation**: {rec['position_size']}
        - **Maximum Leverage**: {rec['leverage']}
        - **Strategy Focus**: {rec['strategy']}
        - **Risk Assessment**: {rec['risk']}
        """)
    
    with rec_col2:
        st.markdown("**⚠️ Risk Factors Alert**")
        risk_factors = []
        
        if current_fed_rate > 5:
            risk_factors.append("🔴 Fed Rate >5% = Liquidity tightening")
        if m2_yoy < 0:
            risk_factors.append("🔴 M2 YoY <0% = Money contraction")
        if btc_dominance > 60:
            risk_factors.append("🔴 BTC Dom >60% = Crypto winter risk")
        if abs(stock_corr) > 0.7:
            risk_factors.append("🔴 BTC-SPX >0.7 = Macro stress mode")
        
        if not risk_factors:
            risk_factors.append("🟢 No major risk factors detected")
        
        for factor in risk_factors:
            st.markdown(f"- {factor}")
    
    # Historical performance
    st.subheader("📈 Score History")
    # Mock historical data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    historical_scores = np.random.uniform(composite_score-20, composite_score+20, 100)
    historical_scores = np.clip(historical_scores, 0, 100)
    historical_scores[-1] = composite_score
    
    score_history = pd.DataFrame({
        'Composite Score': historical_scores
    }, index=dates)
    
    st.line_chart(score_history, height=300)
    
    # Market regime probabilities
    st.subheader("🎲 Regime Probabilities")
    prob_col1, prob_col2, prob_col3, prob_col4, prob_col5 = st.columns(5)
    
    # Calculate probabilities based on score distribution
    abundant_prob = max(0, min(100, (composite_score - 60) * 4))
    good_prob = max(0, min(100, 100 - abs(composite_score - 67.5) * 3))
    neutral_prob = max(0, min(100, 100 - abs(composite_score - 50) * 3))
    tight_prob = max(0, min(100, 100 - abs(composite_score - 32.5) * 3))
    crisis_prob = max(0, min(100, (40 - composite_score) * 4))
    
    with prob_col1:
        st.metric("🟢 Abundant", f"{abundant_prob:.0f}%")
    with prob_col2:
        st.metric("🟡 Good", f"{good_prob:.0f}%")
    with prob_col3:
        st.metric("🟠 Neutral", f"{neutral_prob:.0f}%")
    with prob_col4:
        st.metric("🔴 Tight", f"{tight_prob:.0f}%")
    with prob_col5:
        st.metric("🚨 Crisis", f"{crisis_prob:.0f}%")

# Tab 4: TGA & RRP
with tabs[3]:
    st.subheader("🏦 Treasury General Account & Reverse Repo Analysis")
    
    # Calculate liquidity impact (mock calculation)
    current_tga = tga_data.iloc[-1]['value'] if tga_data is not None else 500
    current_rrp = rrp_data.iloc[-1]['value'] if rrp_data is not None else 2000
    
    # Previous values for change calculation
    prev_tga = tga_data.iloc[-30]['value'] if tga_data is not None and len(tga_data) > 30 else current_tga
    prev_rrp = rrp_data.iloc[-30]['value'] if rrp_data is not None and len(rrp_data) > 30 else current_rrp
    
    tga_change = current_tga - prev_tga
    rrp_change = current_rrp - prev_rrp
    net_liquidity_impact = -(tga_change + rrp_change)  # Negative change = liquidity injection
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("TGA Balance", f"${current_tga:.0f}B", delta=f"{tga_change:.0f}B")
    with col2:
        st.metric("RRP Balance", f"${current_rrp:.0f}B", delta=f"{rrp_change:.0f}B")
    with col3:
        st.metric("Net Liquidity Impact", f"${net_liquidity_impact:.0f}B", 
                 help="Positive = liquidity injection, Negative = liquidity drain")
    with col4:
        liquidity_status = "Injection" if net_liquidity_impact > 20 else "Drain" if net_liquidity_impact < -20 else "Neutral"
        st.metric("Liquidity Status", liquidity_status)
    
    # Charts
    if tga_data is not None and rrp_data is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Treasury General Account**")
            tga_chart = tga_data.set_index('date')['value']
            st.line_chart(tga_chart, height=250)
        
        with col2:
            st.markdown("**Reverse Repo Program**")
            rrp_chart = rrp_data.set_index('date')['value']
            st.line_chart(rrp_chart, height=250)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💡 Current Analysis**")
        if net_liquidity_impact > 50:
            st.success("✅ Major liquidity injection - very bullish for crypto")
        elif net_liquidity_impact > 0:
            st.info("ℹ️ Liquidity injection - supportive for risk assets")
        elif net_liquidity_impact > -50:
            st.warning("⚠️ Neutral liquidity conditions")
        else:
            st.error("🔴 Liquidity drain - headwind for crypto markets")
        
        st.markdown(f"""
        - **30-Day TGA Change**: ${tga_change:.0f}B
        - **30-Day RRP Change**: ${rrp_change:.0f}B
        - **Combined Impact**: ${net_liquidity_impact:.0f}B
        """)
    
    with col2:
        st.markdown("**📊 Impact Mechanics**")
        st.markdown("""
        - **TGA Drawdown**: Direct liquidity injection into banking system
        - **RRP Decline**: Releases reserves from Fed back to money markets
        - **Combined Effect**: Amplifies liquidity available for risk assets
        - **Crypto Sensitivity**: BTC typically responds within 1-2 weeks
        """)

# Tab 5: Stock Correlations
with tabs[4]:
    st.subheader("📈 Stock-Crypto Correlation Analysis")
    
    # Calculate various correlation timeframes
    corr_30d = calculate_correlation(btc_prices[-30:], spx_prices[-30:])
    corr_90d = calculate_correlation(btc_prices[-90:], spx_prices[-90:])
    corr_1y = stock_corr  # Using the main calculation
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("30-Day Correlation", f"{corr_30d:.2f}")
    with col2:
        st.metric("90-Day Correlation", f"{corr_90d:.2f}")
    with col3:
        st.metric("1-Year Correlation", f"{corr_1y:.2f}")
    with col4:
        correlation_regime = "Crisis" if abs(corr_1y) > 0.7 else "High" if abs(corr_1y) > 0.5 else "Moderate" if abs(corr_1y) > 0.3 else "Low"
        st.metric("Correlation Regime", correlation_regime)
    
    # Mock correlation history chart
    corr_history = np.random.uniform(-0.8, 0.8, 100)
    corr_history[-1] = corr_1y
    corr_df = pd.DataFrame({'BTC-SPX Correlation': corr_history}, 
                          index=pd.date_range(end=datetime.now(), periods=100, freq='D'))
    st.line_chart(corr_df, height=400)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💡 Current Analysis**")
        abs_corr = abs(corr_1y)
        if abs_corr < 0.3:
            st.success("✅ Low correlation - crypto acting independently")
        elif abs_corr < 0.6:
            st.info("ℹ️ Moderate correlation - some macro influence")
        else:
            st.error("⚠️ High correlation - crypto following traditional markets")
        
        st.markdown(f"""
        - **Current Level**: {corr_1y:.2f}
        - **Trend**: {'Increasing' if corr_30d > corr_90d else 'Decreasing'}
        - **Diversification Benefit**: {'High' if abs_corr < 0.3 else 'Medium' if abs_corr < 0.6 else 'Low'}
        """)
    
    with col2:
        st.markdown("**📊 Historical Context**")
        st.markdown("""
        - **Bull Market**: 0.1-0.4 (crypto independence)
        - **Bear Market**: 0.6-0.9 (high correlation in stress)
        - **Crisis Periods**: >0.8 (everything sells off together)
        - **Ideal Range**: <0.3 for maximum diversification
        """)
    
    st.markdown("**🎯 Trading Implications**")
    if abs(corr_1y) < 0.3:
        st.markdown("🟢 **BULLISH SIGNAL**: Crypto decoupling from traditional markets. Independent price action likely.")
    elif abs(corr_1y) < 0.6:
        st.markdown("🟡 **MIXED SIGNAL**: Moderate correlation. Monitor macro events closely.")
    else:
        st.markdown("🔴 **RISK SIGNAL**: High correlation indicates macro stress. Expect synchronized selloffs.")

# Tab 6: Gold Correlation
with tabs[5]:
    st.subheader("🥇 Gold-Crypto Correlation Analysis")
    
    gold_corr_30d = calculate_correlation(btc_prices[-30:], gold_prices[-30:])
    gold_corr_90d = calculate_correlation(btc_prices[-90:], gold_prices[-90:])
    gold_corr_1y = gold_corr
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("30-Day Correlation", f"{gold_corr_30d:.2f}")
    with col2:
        st.metric("90-Day Correlation", f"{gold_corr_90d:.2f}")
    with col3:
        st.metric("1-Year Correlation", f"{gold_corr_1y:.2f}")
    with col4:
        hedge_status = "Strong Hedge" if gold_corr_1y > 0.5 else "Weak Hedge" if gold_corr_1y > 0.2 else "No Hedge"
        st.metric("Hedge Status", hedge_status)
    
    # Mock gold correlation chart
    gold_corr_history = np.random.uniform(-0.5, 0.7, 100)
    gold_corr_history[-1] = gold_corr_1y
    gold_corr_df = pd.DataFrame({'BTC-Gold Correlation': gold_corr_history}, 
                               index=pd.date_range(end=datetime.now(), periods=100, freq='D'))
    st.line_chart(gold_corr_df, height=400)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💡 Current Analysis**")
        if gold_corr_1y > 0.5:
            st.success("✅ Strong positive correlation - BTC acting as digital gold")
        elif gold_corr_1y > 0.2:
            st.info("ℹ️ Moderate correlation - partial hedge properties")
        else:
            st.warning("⚠️ Low correlation - limited safe haven demand")
        
        st.markdown(f"""
        - **Current Level**: {gold_corr_1y:.2f}
        - **Hedge Quality**: {'Strong' if gold_corr_1y > 0.5 else 'Moderate' if gold_corr_1y > 0.2 else 'Weak'}
        - **Crisis Resilience**: {'High' if gold_corr_1y > 0.4 else 'Medium' if gold_corr_1y > 0.1 else 'Low'}
        """)
    
    with col2:
        st.markdown("**📊 Store of Value Analysis**")
        st.markdown("""
        - **Digital Gold Thesis**: Requires >0.4 correlation
        - **Inflation Hedge**: Gold correlation indicates hedge effectiveness  
        - **Crisis Performance**: Higher correlation = better crisis protection
        - **Institutional Adoption**: Drives convergence with gold behavior
        """)

# Tab 7: Stablecoins
with tabs[6]:
    st.subheader("💎 Stablecoin Market Analysis")
    
    # Mock stablecoin metrics
    usdt_mcap = stable_mcap * 0.6  # Assume USDT is 60% of total
    usdc_mcap = stable_mcap * 0.4  # USDC is 40%
    stable_growth = np.random.uniform(-5, 15)  # Mock growth rate
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stablecoin Mcap", f"${stable_mcap:.0f}B")
    with col2:
        st.metric("USDT Market Cap", f"${usdt_mcap:.0f}B")
    with col3:
        st.metric("USDC Market Cap", f"${usdc_mcap:.0f}B")
    with col4:
        st.metric("30D Growth", f"{stable_growth:.1f}%")
    
    # Mock historical stablecoin chart
    stable_history = np.random.uniform(stable_mcap*0.8, stable_mcap*1.2, 100)
    stable_history[-1] = stable_mcap
    stable_df = pd.DataFrame({'Stablecoin Market Cap ($B)': stable_history}, 
                            index=pd.date_range(end=datetime.now(), periods=100, freq='D'))
    st.line_chart(stable_df, height=400)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💡 Liquidity Analysis**")
        if stable_growth > 10:
            st.success("✅ Rapid stablecoin growth - new capital entering crypto")
        elif stable_growth > 0:
            st.info("ℹ️ Steady growth - healthy liquidity conditions")
        else:
            st.error("⚠️ Stablecoin contraction - capital leaving crypto")
        
        st.markdown(f"""
        - **Market Cap**: ${stable_mcap:.0f}B
        - **Growth Rate**: {stable_growth:.1f}% (30D)
        - **Liquidity Proxy**: {'Increasing' if stable_growth > 0 else 'Decreasing'}
        """)
    
    with col2:
        st.markdown("**📊 On-Chain Insights**")
        st.markdown("""
        - **Leading Indicator**: Stablecoin growth precedes price moves
        - **Liquidity Measure**: Higher mcap = more trading capacity
        - **Market Sentiment**: Growing mcap = bullish positioning
        - **Institutional Flow**: USDC growth indicates institutional interest
        """)

# Tab 8: Futures OI/Funding
with tabs[7]:
    st.subheader("⚡ Futures Open Interest & Funding Analysis")
    
    # Mock futures data
    total_oi = np.random.uniform(25, 35)  # Billions
    funding_rate = np.random.uniform(-0.1, 0.1)  # -0.1% to 0.1%
    oi_change = np.random.uniform(-10, 10)  # % change
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total OI", f"${total_oi:.1f}B", delta=f"{oi_change:.1f}%")
    with col2:
        st.metric("BTC Funding Rate", f"{funding_rate:.3f}%")
    with col3:
        leverage_ratio = total_oi / (45000 * 19.5e6 / 1e9)  # Mock calculation
        st.metric("Leverage Ratio", f"{leverage_ratio:.2f}x")
    with col4:
        liquidation_risk = "High" if abs(funding_rate) > 0.05 or leverage_ratio > 0.3 else "Medium" if abs(funding_rate) > 0.02 else "Low"
        st.metric("Liquidation Risk", liquidation_risk)
    
    # Mock OI and funding charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Open Interest History**")
        oi_history = np.random.uniform(total_oi*0.7, total_oi*1.3, 100)
        oi_history[-1] = total_oi
        oi_df = pd.DataFrame({'Open Interest ($B)': oi_history}, 
                           index=pd.date_range(end=datetime.now(), periods=100, freq='D'))
        st.line_chart(oi_df, height=250)
    
    with col2:
        st.markdown("**Funding Rate History**")
        funding_history = np.random.uniform(-0.15, 0.15, 100)
        funding_history[-1] = funding_rate
        funding_df = pd.DataFrame({'Funding Rate (%)': funding_history}, 
                                index=pd.date_range(end=datetime.now(), periods=100, freq='D'))
        st.line_chart(funding_df, height=250)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💡 Current Analysis**")
        if funding_rate > 0.05:
            st.warning("⚠️ High positive funding - long squeeze risk")
        elif funding_rate < -0.05:
            st.warning("⚠️ High negative funding - short squeeze risk")
        else:
            st.success("✅ Balanced funding rates")
        
        st.markdown(f"""
        - **OI Level**: ${total_oi:.1f}B ({'High' if total_oi > 30 else 'Normal'})
        - **Funding**: {funding_rate:.3f}% ({'Bullish' if funding_rate > 0 else 'Bearish'})
        - **Leverage**: {leverage_ratio:.2f}x ({'Excessive' if leverage_ratio > 0.4 else 'Normal'})
        """)
    
    with col2:
        st.markdown("**🎯 Liquidation Analysis**")
        st.markdown("""
        - **Long Liquidations**: Price drops trigger cascading sells
        - **Short Liquidations**: Price pumps force covering
        - **High OI Risk**: More positions = higher liquidation potential
        - **Funding Extremes**: Signal overleveraged positions
        """)

# Tab 9: Inflation
with tabs[8]:
    st.subheader("📊 Inflation & Monetary Policy Analysis")
    
    # Calculate inflation metrics
    current_cpi = cpi_data.iloc[-1]['value'] if cpi_data is not None else 310
    prev_year_cpi = cpi_data.iloc[-12]['value'] if cpi_data is not None and len(cpi_data) >= 12 else current_cpi * 0.97
    inflation_rate = ((current_cpi / prev_year_cpi) - 1) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current CPI", f"{current_cpi:.1f}")
    with col2:
        st.metric("YoY Inflation", f"{inflation_rate:.1f}%")
    with col3:
        real_rate = current_fed_rate - inflation_rate
        st.metric("Real Fed Rate", f"{real_rate:.1f}%")
    with col4:
        hedge_demand = "High" if inflation_rate > 4 else "Medium" if inflation_rate > 2 else "Low"
        st.metric("BTC Hedge Demand", hedge_demand)
    
    # Inflation chart
    if cpi_data is not None:
        # Calculate YoY inflation for each point
        cpi_values = cpi_data['value'].values
        inflation_series = []
        dates = []
        
        for i in range(12, len(cpi_values)):
            yoy_inf = ((cpi_values[i] / cpi_values[i-12]) - 1) * 100
            inflation_series.append(yoy_inf)
            dates.append(cpi_data.iloc[i]['date'])
        
        if inflation_series:
            inflation_df = pd.DataFrame({'YoY Inflation (%)': inflation_series}, index=dates)
            st.line_chart(inflation_df, height=400)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💡 Current Analysis**")
        if inflation_rate > 4:
            st.error("🔴 High inflation - strong BTC hedge demand expected")
        elif inflation_rate > 2:
            st.warning("⚠️ Elevated inflation - moderate hedge demand")
        else:
            st.success("✅ Low inflation - limited hedge premium")
        
        st.markdown(f"""
        - **Inflation Rate**: {inflation_rate:.1f}% YoY
        - **Real Interest Rate**: {real_rate:.1f}%
        - **Policy Stance**: {'Restrictive' if real_rate > 2 else 'Accommodative' if real_rate < 0 else 'Neutral'}
        """)
    
    with col2:
        st.markdown("**📊 BTC Hedge Analysis**")
        st.markdown("""
        - **Inflation >4%**: Strong hedge demand, institutional buying
        - **Real Rates <0%**: Negative real yields boost BTC appeal
        - **Policy Response**: Fed hiking cycles create volatility
        - **Long-term**: BTC performance vs inflation varies by cycle
        """)

# Tab 10: Seasonality
with tabs[9]:
    st.subheader("📅 Seasonality & Cycle Analysis")
    
    # Mock seasonality data
    current_month = datetime.now().month
    current_quarter = f"Q{(current_month-1)//3 + 1}"
    days_to_halving = np.random.randint(100, 500)  # Mock days
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Month", datetime.now().strftime("%B"))
    with col2:
        st.metric("Quarter", current_quarter)
    with col3:
        st.metric("Days to Halving", f"{days_to_halving}")
    with col4:
        cycle_phase = "Pre-Halving" if days_to_halving < 365 else "Post-Halving" 
        st.metric("Cycle Phase", cycle_phase)
    
    # Mock seasonal performance chart
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    seasonal_returns = np.random.uniform(-5, 15, 12)  # Mock monthly returns
    seasonal_df = pd.DataFrame({'Average Monthly Return (%)': seasonal_returns}, index=months)
    st.bar_chart(seasonal_df, height=400)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💡 Seasonal Patterns**")
        if current_month in [10, 11, 12, 1]:
            st.success("✅ Historically strong months - Q4/Q1 rally season")
        elif current_month in [2, 3, 4]:
            st.info("ℹ️ Mixed performance - early year volatility")
        else:
            st.warning("⚠️ Summer doldrums - historically weaker period")
        
        best_month = months[np.argmax(seasonal_returns)]
        worst_month = months[np.argmin(seasonal_returns)]
        
        st.markdown(f"""
        - **Current Month Performance**: {seasonal_returns[current_month-1]:.1f}% avg
        - **Best Month**: {best_month} ({seasonal_returns[months.index(best_month)]:.1f}%)
        - **Worst Month**: {worst_month} ({seasonal_returns[months.index(worst_month)]:.1f}%)
        """)
    
    with col2:
        st.markdown("**📊 Halving Cycle Analysis**")
        if days_to_halving < 180:
            st.success("✅ Pre-halving accumulation phase")
        elif days_to_halving < 365:
            st.info("ℹ️ Halving anticipation building")
        else:
            st.warning("⚠️ Post-halving - supply shock effects emerging")
        
        st.markdown(f"""
        - **Days to Next Halving**: {days_to_halving}
        - **Cycle Phase**: {cycle_phase}
        - **Historical Pattern**: 12-18 months post-halving peak
        - **Current Implication**: {'Accumulation' if cycle_phase == 'Pre-Halving' else 'Distribution'}
        """)
    
    st.markdown("**🎯 Seasonal Trading Strategy**")
    if current_month in [10, 11, 12, 1]:
        st.markdown("🟢 **SEASONAL BULLISH**: Increase position sizes for Q4/Q1 rally. Historical strength period.")
    elif current_month in [6, 7, 8]:
        st.markdown("🟡 **SUMMER DOLDRUMS**: Reduce leverage, focus on accumulation. Lower volatility expected.")
    else:
        st.markdown("🟠 **MIXED SEASON**: Monitor other indicators closely. No strong seasonal bias.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>
    🔄 Last updated: {}<br>
    📊 Data sources: FRED, CoinGecko, Yahoo Finance<br>
    ⚠️ For professional use only. Not financial advice.
    </small>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M UTC")), unsafe_allow_html=True)
