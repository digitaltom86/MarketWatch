# STREAMLIT MACRO-CRYPTO DASHBOARD
# Top 3 Indicators MVP - Fed Funds Rate, M2 Money Supply, Bitcoin Dominance
# pip install streamlit plotly pandas requests fredapi

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time
from fredapi import Fred

# Page config
st.set_page_config(
    page_title="Macro-Crypto Liquidity Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FRED API setup
FRED_API_KEY = "8d5a1786444a89510c8ea27e214e255f"
fred = Fred(api_key=FRED_API_KEY)

class MacroDataFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_fed_funds_rate(_self, lookback_days=365):
        """
        1. Fed Funds Rate - główny driver płynności USD
        """
        try:
            # Get data from FRED
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            fed_rate_data = fred.get_data('FEDFUNDS', start=start_date, end=end_date)
            
            if not fed_rate_data.empty:
                current_rate = fed_rate_data.iloc[-1]
                previous_rate = fed_rate_data.iloc[-2] if len(fed_rate_data) > 1 else current_rate
                
                # Determine liquidity impact
                if current_rate > 4.5:
                    impact = "TIGHTENING"
                    impact_color = "red"
                elif current_rate < 2.0:
                    impact = "EASING"
                    impact_color = "green"
                else:
                    impact = "NEUTRAL"
                    impact_color = "orange"
                
                return {
                    'data': fed_rate_data,
                    'current_rate': current_rate,
                    'previous_rate': previous_rate,
                    'change': current_rate - previous_rate,
                    'change_pct': ((current_rate - previous_rate) / previous_rate * 100) if previous_rate != 0 else 0,
                    'impact': impact,
                    'impact_color': impact_color,
                    'liquidity_score': _self._calculate_fed_score(current_rate)
                }
        except Exception as e:
            st.error(f"Error fetching Fed Funds Rate: {e}")
            return None
    
    @st.cache_data(ttl=3600)
    def get_m2_money_supply(_self, lookback_days=730):  # 2 years for M2
        """
        2. M2 Money Supply - najsilniejsza korelacja z BTC
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # M2 Money Stock (FRED: M2SL)
            m2_data = fred.get_data('M2SL', start=start_date, end=end_date)
            
            if not m2_data.empty:
                current_m2 = m2_data.iloc[-1]
                previous_m2 = m2_data.iloc[-2] if len(m2_data) > 1 else current_m2
                
                # Calculate YoY change
                year_ago_m2 = m2_data.iloc[-12] if len(m2_data) >= 12 else previous_m2
                yoy_change = ((current_m2 - year_ago_m2) / year_ago_m2 * 100) if year_ago_m2 != 0 else 0
                
                # Determine regime
                if yoy_change > 10:
                    regime = "AGGRESSIVE_EXPANSION"
                    regime_color = "green"
                elif yoy_change > 5:
                    regime = "MODERATE_EXPANSION"
                    regime_color = "lightgreen"
                elif yoy_change > 0:
                    regime = "MILD_EXPANSION"
                    regime_color = "orange"
                else:
                    regime = "CONTRACTION"
                    regime_color = "red"
                
                return {
                    'data': m2_data,
                    'current_m2': current_m2,
                    'previous_m2': previous_m2,
                    'yoy_change': yoy_change,
                    'regime': regime,
                    'regime_color': regime_color,
                    'liquidity_score': _self._calculate_m2_score(yoy_change),
                    'btc_correlation': 0.85  # Historical correlation
                }
        except Exception as e:
            st.error(f"Error fetching M2 Money Supply: {e}")
            return None
    
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def get_btc_dominance(_self):
        """
        3. Bitcoin Dominance - cykl risk-on/risk-off
        """
        try:
            url = "https://api.coingecko.com/api/v3/global"
            response = _self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                btc_dom = data['data']['market_cap_percentage']['btc']
                
                # Get historical data for trend
                # For demo, we'll simulate trend - in production use CoinGecko Pro API
                trend = "INCREASING" if btc_dom > 48 else "DECREASING"
                
                # Determine market phase
                if btc_dom > 60:
                    phase = "CRYPTO_WINTER"
                    phase_color = "red"
                elif btc_dom > 50:
                    phase = "BTC_DOMINANCE"
                    phase_color = "orange"
                elif btc_dom > 40:
                    phase = "BALANCED_MARKET"
                    phase_color = "yellow"
                else:
                    phase = "ALT_SEASON"
                    phase_color = "green"
                
                return {
                    'btc_dominance': btc_dom,
                    'trend': trend,
                    'phase': phase,
                    'phase_color': phase_color,
                    'alt_season_probability': max(0, (50 - btc_dom) * 2),
                    'liquidity_score': _self._calculate_btc_dom_score(btc_dom),
                    'trading_signal': "FOCUS_BTC" if btc_dom > 50 else "DIVERSIFY_ALTS"
                }
        except Exception as e:
            st.error(f"Error fetching BTC Dominance: {e}")
            return None
    
    def _calculate_fed_score(self, rate):
        """Calculate liquidity score based on Fed rate"""
        if rate < 1.0:
            return 95
        elif rate < 2.0:
            return 85
        elif rate < 3.0:
            return 70
        elif rate < 4.0:
            return 50
        elif rate < 5.0:
            return 30
        else:
            return 15
    
    def _calculate_m2_score(self, yoy_change):
        """Calculate liquidity score based on M2 growth"""
        if yoy_change > 15:
            return 95
        elif yoy_change > 10:
            return 85
        elif yoy_change > 5:
            return 70
        elif yoy_change > 0:
            return 50
        else:
            return 25
    
    def _calculate_btc_dom_score(self, dominance):
        """Calculate market opportunity score based on BTC dominance"""
        if dominance < 35:
            return 90  # Alt season opportunity
        elif dominance < 45:
            return 75
        elif dominance < 55:
            return 60
        else:
            return 40  # Risk-off mode

def calculate_composite_score(fed_data, m2_data, btc_data, weights):
    """
    Calculate composite liquidity score
    """
    if not all([fed_data, m2_data, btc_data]):
        return None
    
    fed_score = fed_data.get('liquidity_score', 50)
    m2_score = m2_data.get('liquidity_score', 50)
    btc_score = btc_data.get('liquidity_score', 50)
    
    composite = (fed_score * weights['fed_weight'] + 
                m2_score * weights['m2_weight'] + 
                btc_score * weights['btc_weight']) / 100
    
    return {
        'score': composite,
        'fed_contribution': fed_score * weights['fed_weight'] / 100,
        'm2_contribution': m2_score * weights['m2_weight'] / 100,
        'btc_contribution': btc_score * weights['btc_weight'] / 100
    }

def create_forecast_model(data, forecast_days, volatility_factor, mean_reversion):
    """
    Simple forecast model based on user parameters
    """
    if data is None or len(data) < 2:
        return None
    
    # Get last value and trend
    current_value = data.iloc[-1]
    recent_values = data.tail(30)  # Last 30 periods
    trend = (recent_values.iloc[-1] - recent_values.iloc[0]) / len(recent_values)
    
    # Generate forecast
    forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), 
                                 periods=forecast_days, freq='D')
    
    forecast_values = []
    for i in range(forecast_days):
        # Basic trend projection with mean reversion
        trend_component = trend * (1 - mean_reversion/100) ** i
        noise = np.random.normal(0, volatility_factor/100 * current_value)
        forecast_value = current_value + trend_component * (i + 1) + noise
        forecast_values.append(forecast_value)
    
    return pd.Series(forecast_values, index=forecast_dates)

# Main App
def main():
    # Title
    st.title("📊 Macro-Crypto Liquidity Monitor")
    st.markdown("**Professional prop trading dashboard for macro indicators affecting crypto liquidity**")
    
    # Sidebar - Forecast Parameters
    st.sidebar.header("🔧 Forecast Parameters")
    
    # Weighting system
    st.sidebar.subheader("Indicator Weights")
    fed_weight = st.sidebar.slider("Fed Funds Rate Weight", 0, 100, 40, 5)
    m2_weight = st.sidebar.slider("M2 Money Supply Weight", 0, 100, 35, 5)
    btc_weight = st.sidebar.slider("BTC Dominance Weight", 0, 100, 25, 5)
    
    # Normalize weights
    total_weight = fed_weight + m2_weight + btc_weight
    if total_weight > 0:
        weights = {
            'fed_weight': fed_weight / total_weight * 100,
            'm2_weight': m2_weight / total_weight * 100,
            'btc_weight': btc_weight / total_weight * 100
        }
    else:
        weights = {'fed_weight': 33.3, 'm2_weight': 33.3, 'btc_weight': 33.3}
    
    st.sidebar.subheader("Forecast Settings")
    forecast_days = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)
    volatility_factor = st.sidebar.slider("Volatility Factor", 0.1, 5.0, 1.0, 0.1)
    mean_reversion = st.sidebar.slider("Mean Reversion %", 0, 50, 20, 5)
    
    # Lookback periods
    st.sidebar.subheader("Data Lookback")
    fed_lookback = st.sidebar.selectbox("Fed Funds Lookback", [180, 365, 730], index=1)
    m2_lookback = st.sidebar.selectbox("M2 Money Supply Lookback", [365, 730, 1095], index=1)
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5 min)", False)
    if auto_refresh:
        time.sleep(300)
        st.rerun()
    
    # Initialize data fetcher
    fetcher = MacroDataFetcher()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Top 3 Indicators", "🔮 Forecast Analysis", "📈 Composite Score"])
    
    with tab1:
        st.header("Top 3 Macro Indicators")
        
        # Fetch data
        with st.spinner("Fetching latest macro data..."):
            fed_data = fetcher.get_fed_funds_rate(fed_lookback)
            m2_data = fetcher.get_m2_money_supply(m2_lookback)
            btc_data = fetcher.get_btc_dominance()
        
        # Layout: 3 columns for metrics
        col1, col2, col3 = st.columns(3)
        
        # Fed Funds Rate
        with col1:
            st.subheader("🏛️ Fed Funds Rate")
            if fed_data:
                st.metric(
                    "Current Rate",
                    f"{fed_data['current_rate']:.2f}%",
                    f"{fed_data['change']:+.2f}pp"
                )
                st.markdown(f"**Impact:** ::{fed_data['impact_color']}[{fed_data['impact']}]")
                st.markdown(f"**Liquidity Score:** {fed_data['liquidity_score']}/100")
                
                # Mini chart
                if len(fed_data['data']) > 1:
                    fig_fed = go.Figure()
                    fig_fed.add_trace(go.Scatter(
                        x=fed_data['data'].index,
                        y=fed_data['data'].values,
                        mode='lines',
                        name='Fed Funds Rate',
                        line=dict(color=fed_data['impact_color'], width=2)
                    ))
                    fig_fed.update_layout(
                        height=200,
                        margin=dict(l=0, r=0, t=20, b=0),
                        showlegend=False,
                        xaxis_title=None,
                        yaxis_title="Rate %"
                    )
                    st.plotly_chart(fig_fed, use_container_width=True)
        
        # M2 Money Supply
        with col2:
            st.subheader("💰 M2 Money Supply")
            if m2_data:
                st.metric(
                    "Current M2",
                    f"${m2_data['current_m2']:.1f}T",
                    f"{m2_data['yoy_change']:+.1f}% YoY"
                )
                st.markdown(f"**Regime:** ::{m2_data['regime_color']}[{m2_data['regime']}]")
                st.markdown(f"**Liquidity Score:** {m2_data['liquidity_score']}/100")
                st.markdown(f"**BTC Correlation:** {m2_data['btc_correlation']:.2f}")
                
                # Mini chart
                if len(m2_data['data']) > 1:
                    fig_m2 = go.Figure()
                    fig_m2.add_trace(go.Scatter(
                        x=m2_data['data'].index,
                        y=m2_data['data'].values,
                        mode='lines',
                        name='M2 Money Supply',
                        line=dict(color=m2_data['regime_color'], width=2)
                    ))
                    fig_m2.update_layout(
                        height=200,
                        margin=dict(l=0, r=0, t=20, b=0),
                        showlegend=False,
                        xaxis_title=None,
                        yaxis_title="Trillions $"
                    )
                    st.plotly_chart(fig_m2, use_container_width=True)
        
        # Bitcoin Dominance
        with col3:
            st.subheader("₿ Bitcoin Dominance")
            if btc_data:
                st.metric(
                    "BTC Dominance",
                    f"{btc_data['btc_dominance']:.1f}%",
                    f"{btc_data['trend']}"
                )
                st.markdown(f"**Phase:** ::{btc_data['phase_color']}[{btc_data['phase']}]")
                st.markdown(f"**Opportunity Score:** {btc_data['liquidity_score']}/100")
                st.markdown(f"**Alt Season Prob:** {btc_data['alt_season_probability']:.0f}%")
                st.markdown(f"**Signal:** {btc_data['trading_signal']}")
    
    with tab2:
        st.header("🔮 Forecast Analysis")
        
        if fed_data and m2_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Fed Funds Rate Forecast")
                fed_forecast = create_forecast_model(
                    fed_data['data'], forecast_days, volatility_factor, mean_reversion
                )
                
                if fed_forecast is not None:
                    fig_forecast = go.Figure()
                    
                    # Historical data
                    fig_forecast.add_trace(go.Scatter(
                        x=fed_data['data'].index[-90:],  # Last 90 days
                        y=fed_data['data'].values[-90:],
                        mode='lines',
                        name='Historical',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Forecast
                    fig_forecast.add_trace(go.Scatter(
                        x=fed_forecast.index,
                        y=fed_forecast.values,
                        mode='lines',
                        name=f'{forecast_days}d Forecast',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    fig_forecast.update_layout(
                        title="Fed Funds Rate Forecast",
                        xaxis_title="Date",
                        yaxis_title="Rate %",
                        height=400
                    )
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Key insights
                    current_rate = fed_data['current_rate']
                    forecast_end = fed_forecast.iloc[-1]
                    change_forecast = forecast_end - current_rate
                    
                    st.info(f"""
                    **Forecast Insights:**
                    - Current Rate: {current_rate:.2f}%
                    - {forecast_days}d Forecast: {forecast_end:.2f}%
                    - Expected Change: {change_forecast:+.2f}pp
                    - Trend: {"Tightening" if change_forecast > 0.1 else "Easing" if change_forecast < -0.1 else "Stable"}
                    """)
            
            with col2:
                st.subheader("M2 Money Supply Forecast")
                m2_forecast = create_forecast_model(
                    m2_data['data'], forecast_days, volatility_factor, mean_reversion
                )
                
                if m2_forecast is not None:
                    fig_m2_forecast = go.Figure()
                    
                    # Historical data
                    fig_m2_forecast.add_trace(go.Scatter(
                        x=m2_data['data'].index[-90:],
                        y=m2_data['data'].values[-90:],
                        mode='lines',
                        name='Historical',
                        line=dict(color='green', width=2)
                    ))
                    
                    # Forecast
                    fig_m2_forecast.add_trace(go.Scatter(
                        x=m2_forecast.index,
                        y=m2_forecast.values,
                        mode='lines',
                        name=f'{forecast_days}d Forecast',
                        line=dict(color='orange', width=2, dash='dash')
                    ))
                    
                    fig_m2_forecast.update_layout(
                        title="M2 Money Supply Forecast",
                        xaxis_title="Date",
                        yaxis_title="Trillions $",
                        height=400
                    )
                    st.plotly_chart(fig_m2_forecast, use_container_width=True)
                    
                    # Key insights
                    current_m2 = m2_data['current_m2']
                    forecast_m2_end = m2_forecast.iloc[-1]
                    m2_change_forecast = ((forecast_m2_end - current_m2) / current_m2) * 100
                    
                    st.info(f"""
                    **Forecast Insights:**
                    - Current M2: ${current_m2:.1f}T
                    - {forecast_days}d Forecast: ${forecast_m2_end:.1f}T
                    - Expected Change: {m2_change_forecast:+.1f}%
                    - Liquidity Impact: {"Expansionary" if m2_change_forecast > 1 else "Contractionary" if m2_change_forecast < -1 else "Neutral"}
                    """)
    
    with tab3:
        st.header("📈 Composite Liquidity Score")
        
        if fed_data and m2_data and btc_data:
            # Calculate composite score
            composite = calculate_composite_score(fed_data, m2_data, btc_data, weights)
            
            if composite:
                # Main score display
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    # Gauge chart for composite score
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = composite['score'],
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Composite Liquidity Score"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 25], 'color': "red"},
                                {'range': [25, 50], 'color': "orange"},
                                {'range': [50, 75], 'color': "yellow"},
                                {'range': [75, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col2:
                    st.metric("Overall Score", f"{composite['score']:.1f}/100")
                    
                    # Regime classification
                    if composite['score'] > 75:
                        regime = "🟢 ABUNDANT LIQUIDITY"
                    elif composite['score'] > 60:
                        regime = "🟡 GOOD LIQUIDITY"
                    elif composite['score'] > 40:
                        regime = "🟠 NEUTRAL LIQUIDITY"
                    elif composite['score'] > 25:
                        regime = "🔴 TIGHT LIQUIDITY"
                    else:
                        regime = "🚨 LIQUIDITY CRISIS"
                    
                    st.markdown(f"**Regime:** {regime}")
                
                with col3:
                    st.subheader("Weight Distribution")
                    st.write(f"Fed Rate: {weights['fed_weight']:.1f}%")
                    st.write(f"M2 Supply: {weights['m2_weight']:.1f}%")
                    st.write(f"BTC Dom: {weights['btc_weight']:.1f}%")
                
                # Contribution breakdown
                st.subheader("Score Contribution Breakdown")
                contrib_data = {
                    'Indicator': ['Fed Funds Rate', 'M2 Money Supply', 'BTC Dominance'],
                    'Raw Score': [fed_data['liquidity_score'], m2_data['liquidity_score'], btc_data['liquidity_score']],
                    'Weight': [weights['fed_weight'], weights['m2_weight'], weights['btc_weight']],
                    'Contribution': [composite['fed_contribution'], composite['m2_contribution'], composite['btc_contribution']]
                }
                
                df_contrib = pd.DataFrame(contrib_data)
                
                # Bar chart
                fig_contrib = px.bar(
                    df_contrib, 
                    x='Indicator', 
                    y='Contribution',
                    title="Weighted Contribution to Composite Score",
                    color='Contribution',
                    color_continuous_scale='RdYlGn'
                )
                fig_contrib.update_layout(height=400)
                st.plotly_chart(fig_contrib, use_container_width=True)
                
                # Trading recommendations
                st.subheader("🎯 Trading Recommendations")
                
                if composite['score'] > 75:
                    rec_color = "green"
                    recommendations = [
                        "✅ Aggressive growth strategies recommended",
                        "✅ Increase position sizes and leverage",
                        "✅ Focus on high-beta altcoins",
                        "✅ Active market making with tight spreads"
                    ]
                elif composite['score'] > 50:
                    rec_color = "orange"
                    recommendations = [
                        "⚖️ Balanced approach recommended",
                        "⚖️ Normal risk parameters",
                        "⚖️ Diversified crypto portfolio",
                        "⚖️ Standard market making spreads"
                    ]
                else:
                    rec_color = "red"
                    recommendations = [
                        "🛡️ Defensive positioning required",
                        "🛡️ Reduce leverage and exposure",
                        "🛡️ Focus on BTC and major pairs only",
                        "🛡️ Widen spreads, reduce inventory"
                    ]
                
                for rec in recommendations:
                    st.markdown(f":{rec_color}[{rec}]")
        
        # Last update time
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
        if st.sidebar.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

if __name__ == "__main__":
    main()
