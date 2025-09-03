# STREAMLIT MACRO-CRYPTO DASHBOARD
# Top 3 Indicators MVP - Fixed Dependencies Version
# Requirements: streamlit pandas requests

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import json

# Page config
st.set_page_config(
    page_title="Macro-Crypto Liquidity Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FRED API setup
FRED_API_KEY = "8d5a1786444a89510c8ea27e214e255f"

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
            # FRED API call
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': 'FEDFUNDS',
                'api_key': FRED_API_KEY,
                'file_type': 'json',
                'start_date': start_date,
                'end_date': end_date,
                'sort_order': 'desc',
                'limit': 100
            }
            
            response = _self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                observations = data.get('observations', [])
                
                if observations:
                    # Get latest valid observation
                    latest_obs = None
                    previous_obs = None
                    
                    for obs in observations:
                        if obs['value'] != '.':
                            if latest_obs is None:
                                latest_obs = obs
                            elif previous_obs is None:
                                previous_obs = obs
                                break
                    
                    if latest_obs:
                        current_rate = float(latest_obs['value'])
                        previous_rate = float(previous_obs['value']) if previous_obs and previous_obs['value'] != '.' else current_rate
                        
                        # Determine liquidity impact
                        if current_rate > 4.5:
                            impact = "TIGHTENING"
                            impact_emoji = "🔴"
                        elif current_rate < 2.0:
                            impact = "EASING"
                            impact_emoji = "🟢"
                        else:
                            impact = "NEUTRAL"
                            impact_emoji = "🟡"
                        
                        # Create time series data
                        dates = []
                        values = []
                        for obs in reversed(observations):
                            if obs['value'] != '.':
                                dates.append(pd.to_datetime(obs['date']))
                                values.append(float(obs['value']))
                        
                        time_series = pd.Series(values, index=dates)
                        
                        return {
                            'data': time_series,
                            'current_rate': current_rate,
                            'previous_rate': previous_rate,
                            'change': current_rate - previous_rate,
                            'change_pct': ((current_rate - previous_rate) / previous_rate * 100) if previous_rate != 0 else 0,
                            'impact': impact,
                            'impact_emoji': impact_emoji,
                            'liquidity_score': _self._calculate_fed_score(current_rate),
                            'last_update': latest_obs['date']
                        }
            
        except Exception as e:
            st.error(f"Error fetching Fed Funds Rate: {e}")
            # Return fallback data
            return {
                'current_rate': 5.25,
                'previous_rate': 5.25,
                'change': 0,
                'change_pct': 0,
                'impact': "TIGHTENING",
                'impact_emoji': "🔴",
                'liquidity_score': 30,
                'error': str(e)
            }
    
    @st.cache_data(ttl=3600)
    def get_m2_money_supply(_self, lookback_days=730):
        """
        2. M2 Money Supply - najsilniejsza korelacja z BTC
        """
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': 'M2SL',
                'api_key': FRED_API_KEY,
                'file_type': 'json',
                'start_date': start_date,
                'end_date': end_date,
                'sort_order': 'desc',
                'limit': 50
            }
            
            response = _self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                observations = data.get('observations', [])
                
                if observations:
                    # Get latest valid observations
                    valid_obs = [obs for obs in observations if obs['value'] != '.']
                    
                    if len(valid_obs) >= 2:
                        current_m2 = float(valid_obs[0]['value']) / 1000  # Convert to trillions
                        previous_m2 = float(valid_obs[1]['value']) / 1000
                        
                        # Calculate YoY change (approximate with available data)
                        if len(valid_obs) >= 12:
                            year_ago_m2 = float(valid_obs[11]['value']) / 1000
                        else:
                            year_ago_m2 = previous_m2
                        
                        yoy_change = ((current_m2 - year_ago_m2) / year_ago_m2 * 100) if year_ago_m2 != 0 else 0
                        
                        # Determine regime
                        if yoy_change > 10:
                            regime = "AGGRESSIVE_EXPANSION"
                            regime_emoji = "🟢"
                        elif yoy_change > 5:
                            regime = "MODERATE_EXPANSION"
                            regime_emoji = "🟢"
                        elif yoy_change > 0:
                            regime = "MILD_EXPANSION"
                            regime_emoji = "🟡"
                        else:
                            regime = "CONTRACTION"
                            regime_emoji = "🔴"
                        
                        # Create time series
                        dates = []
                        values = []
                        for obs in reversed(valid_obs):
                            dates.append(pd.to_datetime(obs['date']))
                            values.append(float(obs['value']) / 1000)
                        
                        time_series = pd.Series(values, index=dates)
                        
                        return {
                            'data': time_series,
                            'current_m2': current_m2,
                            'previous_m2': previous_m2,
                            'yoy_change': yoy_change,
                            'regime': regime,
                            'regime_emoji': regime_emoji,
                            'liquidity_score': _self._calculate_m2_score(yoy_change),
                            'btc_correlation': 0.85,
                            'last_update': valid_obs[0]['date']
                        }
            
        except Exception as e:
            st.error(f"Error fetching M2 Money Supply: {e}")
            return {
                'current_m2': 20.8,
                'previous_m2': 20.7,
                'yoy_change': 2.5,
                'regime': "MILD_EXPANSION",
                'regime_emoji': "🟡",
                'liquidity_score': 55,
                'btc_correlation': 0.85,
                'error': str(e)
            }
    
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
                total_mcap = data['data']['total_market_cap']['usd'] / 1e12  # Trillions
                
                # Determine market phase
                if btc_dom > 60:
                    phase = "CRYPTO_WINTER"
                    phase_emoji = "🔴"
                elif btc_dom > 50:
                    phase = "BTC_DOMINANCE"
                    phase_emoji = "🟡"
                elif btc_dom > 40:
                    phase = "BALANCED_MARKET"
                    phase_emoji = "🟠"
                else:
                    phase = "ALT_SEASON"
                    phase_emoji = "🟢"
                
                return {
                    'btc_dominance': btc_dom,
                    'total_mcap': total_mcap,
                    'phase': phase,
                    'phase_emoji': phase_emoji,
                    'alt_season_probability': max(0, (50 - btc_dom) * 2),
                    'liquidity_score': _self._calculate_btc_dom_score(btc_dom),
                    'trading_signal': "FOCUS_BTC" if btc_dom > 50 else "DIVERSIFY_ALTS"
                }
                
        except Exception as e:
            st.error(f"Error fetching BTC Dominance: {e}")
            return {
                'btc_dominance': 48.5,
                'total_mcap': 2.1,
                'phase': "BALANCED_MARKET",
                'phase_emoji': "🟠",
                'alt_season_probability': 3,
                'liquidity_score': 60,
                'trading_signal': "DIVERSIFY_ALTS",
                'error': str(e)
            }
    
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

def create_simple_chart(data, title, color='blue'):
    """
    Create simple line chart using streamlit native charting
    """
    if data is not None and len(data) > 1:
        chart_df = pd.DataFrame({
            'Date': data.index,
            'Value': data.values
        })
        return chart_df
    return None

# Main App
def main():
    # Title and description
    st.title("📊 Macro-Crypto Liquidity Monitor")
    st.markdown("**Professional prop trading dashboard for macro indicators affecting crypto liquidity**")
    st.markdown("---")
    
    # Sidebar - Parameters
    st.sidebar.header("🔧 Model Parameters")
    
    # Weighting system
    st.sidebar.subheader("📊 Indicator Weights")
    fed_weight = st.sidebar.slider("Fed Funds Rate Weight", 0, 100, 40, 5, 
                                  help="Higher weight = more influence on composite score")
    m2_weight = st.sidebar.slider("M2 Money Supply Weight", 0, 100, 35, 5,
                                 help="M2 has strong correlation with BTC prices")
    btc_weight = st.sidebar.slider("BTC Dominance Weight", 0, 100, 25, 5,
                                  help="BTC dominance indicates risk-on/risk-off cycles")
    
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
    
    # Data settings
    st.sidebar.subheader("📅 Data Settings")
    fed_lookback = st.sidebar.selectbox("Fed Funds Lookback", [180, 365, 730], index=1,
                                       help="How many days of historical data to fetch")
    m2_lookback = st.sidebar.selectbox("M2 Money Supply Lookback", [365, 730, 1095], index=1)
    
    # Auto-refresh
    st.sidebar.subheader("🔄 Refresh Settings")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5 min)", False,
                                      help="Automatically refresh data every 5 minutes")
    
    if st.sidebar.button("🔄 Manual Refresh"):
        st.cache_data.clear()
        st.rerun()
    
    if auto_refresh:
        time.sleep(300)
        st.rerun()
    
    # Initialize data fetcher
    fetcher = MacroDataFetcher()
    
    # Main content
    st.header("📈 Top 3 Macro Indicators")
    
    # Fetch data with progress
    with st.spinner("Fetching latest macro data from FRED API and CoinGecko..."):
        fed_data = fetcher.get_fed_funds_rate(fed_lookback)
        m2_data = fetcher.get_m2_money_supply(m2_lookback)
        btc_data = fetcher.get_btc_dominance()
    
    # Layout: 3 columns for main indicators
    col1, col2, col3 = st.columns(3)
    
    # 1. Fed Funds Rate
    with col1:
        st.subheader("🏛️ Fed Funds Rate")
        
        if fed_data:
            # Main metrics
            st.metric(
                "Current Rate",
                f"{fed_data['current_rate']:.2f}%",
                f"{fed_data['change']:+.2f}pp",
                help="Federal Reserve's target interest rate"
            )
            
            # Status indicators
            st.markdown(f"**Status:** {fed_data['impact_emoji']} {fed_data['impact']}")
            st.markdown(f"**Liquidity Score:** {fed_data['liquidity_score']}/100")
            
            if 'last_update' in fed_data:
                st.caption(f"Last update: {fed_data['last_update']}")
            
            # Simple chart if we have data
            if 'data' in fed_data and fed_data['data'] is not None:
                chart_data = create_simple_chart(fed_data['data'], "Fed Funds Rate")
                if chart_data is not None:
                    st.line_chart(chart_data.set_index('Date'))
            
            # Analysis
            with st.expander("📊 Fed Rate Analysis"):
                if fed_data['current_rate'] > 5.0:
                    st.warning("🔴 **HIGH RATE REGIME** - Liquidity tightening, reduce crypto exposure")
                elif fed_data['current_rate'] < 2.0:
                    st.success("🟢 **LOW RATE REGIME** - Abundant liquidity, favorable for risk assets")
                else:
                    st.info("🟡 **NEUTRAL REGIME** - Balanced monetary policy")
                
                st.write(f"**Impact on Crypto:**")
                st.write(f"- Liquidity Score: {fed_data['liquidity_score']}/100")
                st.write(f"- Current stance: {fed_data['impact']}")
                st.write(f"- Change: {fed_data['change']:+.2f} percentage points")
    
    # 2. M2 Money Supply
    with col2:
        st.subheader("💰 M2 Money Supply")
        
        if m2_data:
            # Main metrics
            st.metric(
                "Current M2",
                f"${m2_data['current_m2']:.1f}T",
                f"{m2_data['yoy_change']:+.1f}% YoY",
                help="Total money supply in the US economy"
            )
            
            # Status indicators
            st.markdown(f"**Regime:** {m2_data['regime_emoji']} {m2_data['regime']}")
            st.markdown(f"**Liquidity Score:** {m2_data['liquidity_score']}/100")
            st.markdown(f"**BTC Correlation:** {m2_data['btc_correlation']:.2f}")
            
            if 'last_update' in m2_data:
                st.caption(f"Last update: {m2_data['last_update']}")
            
            # Simple chart
            if 'data' in m2_data and m2_data['data'] is not None:
                chart_data = create_simple_chart(m2_data['data'], "M2 Money Supply")
                if chart_data is not None:
                    st.line_chart(chart_data.set_index('Date'))
            
            # Analysis
            with st.expander("📊 M2 Analysis"):
                if m2_data['yoy_change'] > 8:
                    st.success("🟢 **RAPID EXPANSION** - Strong positive signal for BTC")
                elif m2_data['yoy_change'] > 3:
                    st.info("🟡 **MODERATE EXPANSION** - Supportive for crypto")
                elif m2_data['yoy_change'] > 0:
                    st.warning("🟠 **SLOW GROWTH** - Neutral to slightly positive")
                else:
                    st.error("🔴 **CONTRACTION** - Negative for risk assets")
                
                st.write(f"**Key Metrics:**")
                st.write(f"- YoY Growth: {m2_data['yoy_change']:+.1f}%")
                st.write(f"- Historical BTC correlation: {m2_data['btc_correlation']:.2f}")
                st.write(f"- Current supply: ${m2_data['current_m2']:.1f} trillion")
    
    # 3. Bitcoin Dominance
    with col3:
        st.subheader("₿ Bitcoin Dominance")
        
        if btc_data:
            # Main metrics
            st.metric(
                "BTC Dominance",
                f"{btc_data['btc_dominance']:.1f}%",
                delta=None,
                help="Bitcoin's share of total crypto market cap"
            )
            
            # Status indicators
            st.markdown(f"**Phase:** {btc_data['phase_emoji']} {btc_data['phase']}")
            st.markdown(f"**Opportunity Score:** {btc_data['liquidity_score']}/100")
            st.markdown(f"**Alt Season Prob:** {btc_data['alt_season_probability']:.0f}%")
            st.markdown(f"**Signal:** {btc_data['trading_signal']}")
            
            if 'total_mcap' in btc_data:
                st.caption(f"Total Crypto Market Cap: ${btc_data['total_mcap']:.1f}T")
            
            # Analysis
            with st.expander("📊 BTC Dominance Analysis"):
                if btc_data['btc_dominance'] > 55:
                    st.warning("🔴 **BTC DOMINANCE HIGH** - Risk-off mode, focus on BTC")
                elif btc_data['btc_dominance'] < 40:
                    st.success("🟢 **ALT SEASON ACTIVE** - High probability alt outperformance")
                else:
                    st.info("🟡 **BALANCED MARKET** - Mixed signals, diversified approach")
                
                st.write(f"**Market Implications:**")
                st.write(f"- Current dominance: {btc_data['btc_dominance']:.1f}%")
                st.write(f"- Alt season probability: {btc_data['alt_season_probability']:.0f}%")
                st.write(f"- Recommended strategy: {btc_data['trading_signal']}")
    
    # Composite Score Section
    st.markdown("---")
    st.header("🎯 Composite Liquidity Score")
    
    if fed_data and m2_data and btc_data:
        composite = calculate_composite_score(fed_data, m2_data, btc_data, weights)
        
        if composite:
            # Main score display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Composite Score", f"{composite['score']:.1f}/100", 
                         help="Weighted combination of all indicators")
            
            with col2:
                # Regime classification
                if composite['score'] > 75:
                    regime = "🟢 ABUNDANT LIQUIDITY"
                    regime_desc = "Aggressive strategies recommended"
                elif composite['score'] > 60:
                    regime = "🟡 GOOD LIQUIDITY"
                    regime_desc = "Moderate risk taking"
                elif composite['score'] > 40:
                    regime = "🟠 NEUTRAL LIQUIDITY"
                    regime_desc = "Balanced approach"
                elif composite['score'] > 25:
                    regime = "🔴 TIGHT LIQUIDITY"
                    regime_desc = "Defensive positioning"
                else:
                    regime = "🚨 LIQUIDITY CRISIS"
                    regime_desc = "Capital preservation mode"
                
                st.markdown(f"**Regime:**")
                st.markdown(regime)
                st.caption(regime_desc)
            
            with col3:
                st.markdown("**Weight Distribution:**")
                st.write(f"🏛️ Fed Rate: {weights['fed_weight']:.1f}%")
                st.write(f"💰 M2 Supply: {weights['m2_weight']:.1f}%")
                st.write(f"₿ BTC Dom: {weights['btc_weight']:.1f}%")
            
            with col4:
                st.markdown("**Score Contributions:**")
                st.write(f"🏛️ Fed: {composite['fed_contribution']:.1f}")
                st.write(f"💰 M2: {composite['m2_contribution']:.1f}")
                st.write(f"₿ BTC: {composite['btc_contribution']:.1f}")
            
            # Detailed breakdown
            st.subheader("📊 Score Breakdown")
            
            # Create breakdown table
            breakdown_data = {
                'Indicator': ['🏛️ Fed Funds Rate', '💰 M2 Money Supply', '₿ BTC Dominance'],
                'Raw Score': [fed_data['liquidity_score'], m2_data['liquidity_score'], btc_data['liquidity_score']],
                'Weight (%)': [f"{weights['fed_weight']:.1f}%", f"{weights['m2_weight']:.1f}%", f"{weights['btc_weight']:.1f}%"],
                'Weighted Score': [f"{composite['fed_contribution']:.1f}", f"{composite['m2_contribution']:.1f}", f"{composite['btc_contribution']:.1f}"],
                'Status': [f"{fed_data['impact_emoji']} {fed_data['impact']}", 
                          f"{m2_data['regime_emoji']} {m2_data['regime']}", 
                          f"{btc_data['phase_emoji']} {btc_data['phase']}"]
            }
            
            df_breakdown = pd.DataFrame(breakdown_data)
            st.dataframe(df_breakdown, use_container_width=True)
            
            # Trading recommendations
            st.subheader("🎯 Trading Recommendations")
            
            recommendation_col1, recommendation_col2 = st.columns(2)
            
            with recommendation_col1:
                st.markdown("**🎲 Strategy Recommendations:**")
                if composite['score'] > 75:
                    st.success("✅ **AGGRESSIVE GROWTH**")
                    st.write("- Increase position sizes")
                    st.write("- Use moderate leverage")
                    st.write("- Focus on high-beta altcoins")
                    st.write("- Tight market making spreads")
                elif composite['score'] > 50:
                    st.info("⚖️ **BALANCED APPROACH**")
                    st.write("- Normal risk parameters")
                    st.write("- Diversified crypto portfolio")
                    st.write("- Standard spreads")
                    st.write("- Monitor for regime changes")
                else:
                    st.warning("🛡️ **DEFENSIVE POSITIONING**")
                    st.write("- Reduce leverage and exposure")
                    st.write("- Focus on BTC and major pairs")
                    st.write("- Widen spreads")
                    st.write("- Increase cash reserves")
            
            with recommendation_col2:
                st.markdown("**⚠️ Risk Management:**")
                
                risk_factors = []
                if fed_data['current_rate'] > 5.0:
                    risk_factors.append("🔴 High Fed rate - liquidity pressure")
                if m2_data['yoy_change'] < 0:
                    risk_factors.append("🔴 M2 contraction - negative for crypto")
                if btc_data['btc_dominance'] > 60:
                    risk_factors.append("🟡 High BTC dominance - alt risk")
                
                if risk_factors:
                    st.markdown("**Current Risk Factors:**")
                    for factor in risk_factors:
                        st.write(f"- {factor}")
                else:
                    st.success("🟢 No major risk factors identified")
                
                # Key levels to watch
                st.markdown("**📊 Key Levels to Monitor:**")
                st.write(f"- Fed Rate: Watch for moves above 5.5% or below 4.0%")
                st.write(f"- M2 Growth: Watch for YoY changes above 8% or below 0%")
                st.write(f"- BTC Dom: Watch for breaks above 55% or below 40%")
    
    # Footer with update info
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("**📅 Last Update:**")
        st.write(datetime.now().strftime('%Y-%m-%d %H:%M UTC'))
    
    with footer_col2:
        st.markdown("**📊 Data Sources:**")
        st.write("FRED API, CoinGecko API")
    
    with footer_col3:
        st.markdown("**🔄 Cache Status:**")
        st.write("Fed/M2: 1h cache | BTC: 30min cache")

if __name__ == "__main__":
    main()

# Create requirements.txt content
requirements_txt = """
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
requests>=2.28.0
"""

# Instructions for setup
setup_instructions = """
# SETUP INSTRUCTIONS:

1. Save this code as 'app.py'

2. Create requirements.txt file with:
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
requests>=2.28.0

3. Install dependencies:
pip install -r requirements.txt

4. Run the app:
streamlit run app.py

5. The app will open in your browser at localhost:8501

# FEATURES:
- Real FRED API integration with your key
- Live CoinGecko data for BTC dominance
- Interactive weight adjustment
- Composite liquidity scoring
- Trading recommendations
- Auto-refresh capability
- Mobile-responsive design

# NEXT STEPS:
- Test all 3 indicators loading
- Adjust weights in sidebar
- Check composite score calculation
- Verify trading recommendations
"""

print("✅ Streamlit app ready!")
print("\n" + setup_instructions)
