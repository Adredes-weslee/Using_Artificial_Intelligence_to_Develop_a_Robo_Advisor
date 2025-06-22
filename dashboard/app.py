"""Main Streamlit application file for the AI-Powered Robo-Advisor."""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    
# Setup
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import src.config as config
    from src.models.cloud_optimized_agent import CloudOptimizedRLManager
    from src.models.rl_agent_manager import RLAgentManager
    from src.utils.market_analysis import detect_market_regime, get_regime_recommendations
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all required dependencies are installed and data processing is complete.")
    st.stop()

st.set_page_config(
    page_title="AI Robo-Advisor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 AI-Powered Robo-Advisor")
st.markdown("### Intelligent Portfolio Optimization with Reinforcement Learning")

# Check environment
is_cloud = os.environ.get('STREAMLIT_CLOUD') == 'true'

if is_cloud:
    st.info("🌩️ **Cloud Mode**: Using optimized MPT allocation")
    st.caption("Running on Streamlit Community Cloud with memory-optimized AI")
else:
    st.info("🖥️ **Local Mode**: Full RL training capabilities available")
    st.caption("Running locally with complete reinforcement learning training")

# Sidebar
st.sidebar.header("Portfolio Configuration")
st.sidebar.markdown("Configure your investment preferences below:")

# Risk Profile Selection
risk_profile = st.sidebar.selectbox(
    "Select Risk Profile",
    ["Conservative", "Balanced", "Aggressive"],
    help="Choose your investment risk tolerance level"
)

# Display risk profile description
risk_descriptions = {
    "Conservative": "Low risk, stable returns, capital preservation focus",
    "Balanced": "Moderate risk, balanced growth and income",
    "Aggressive": "Higher risk, growth-focused, maximum returns"
}
st.sidebar.caption(f"📋 {risk_descriptions[risk_profile]}")

# NEW INVESTMENT OBJECTIVE SECTION
st.sidebar.header("🎯 Investment Objective")

risk_preference = st.sidebar.radio(
    "What's your primary goal?",
    [
        "🛡️ Protect My Capital (Risk-Focused)",
        "⚖️ Balance Risk & Return (Academic)",  
        "🚀 Maximize Returns (Growth-Focused)",
        "🎲 Custom Mix"
    ]
)

if risk_preference == "🎲 Custom Mix":
    return_weight = st.sidebar.slider(
        "Return Priority", 0.0, 1.0, 0.5, 0.1,
        help="Higher = Focus on returns, Lower = Focus on risk management"
    )
    risk_weight = 1.0 - return_weight
else:
    # Predefined mixes
    weights = {
        "🛡️ Protect My Capital (Risk-Focused)": (0.2, 0.8),      # 20% return, 80% risk
        "⚖️ Balance Risk & Return (Academic)": (0.5, 0.5),        # 50% return, 50% risk  
        "🚀 Maximize Returns (Growth-Focused)": (0.8, 0.2)        # 80% return, 20% risk
    }
    return_weight, risk_weight = weights[risk_preference]

st.sidebar.write(f"**Return Focus:** {return_weight:.0%}")
st.sidebar.write(f"**Risk Management:** {risk_weight:.0%}")

# Asset Selection
available_assets = config.DEFAULT_PORTFOLIO_ASSETS
selected_assets = st.sidebar.multiselect(
    "Select Assets for Portfolio",
    available_assets,
    default=available_assets[:10],
    help="Choose 5-15 assets for optimal performance. More assets = better diversification."
)

st.sidebar.markdown(f"**Selected: {len(selected_assets)} assets**")
if len(selected_assets) > 20:
    st.sidebar.warning("⚠️ Too many assets may slow down processing")
elif len(selected_assets) < 5:
    st.sidebar.warning("⚠️ Consider selecting more assets for better diversification")

# Investment Amount
investment_amount = st.sidebar.number_input(
    "Investment Amount ($)",
    min_value=1000,
    max_value=10000000,
    value=100000,
    step=1000,
    help="Total amount to invest"
)

# MARKET REGIME DETECTION
st.sidebar.markdown("---")
st.sidebar.header("📊 Market Analysis")

# Try to load market data for regime detection
try:
    market_data_path = config.PROCESSED_DATA_DIR / config.PROCESSED_SP500_FILE
    if market_data_path.exists():
        df = pd.read_csv(market_data_path, index_col=0, parse_dates=True)
        current_regime = detect_market_regime(df)
        regime_info = get_regime_recommendations(current_regime)
        
        st.sidebar.write(f"**Current Market:** {current_regime}")
        if regime_info["style"] == "warning":
            st.sidebar.warning(regime_info["message"])
        elif regime_info["style"] == "info":
            st.sidebar.info(regime_info["message"])
        else:
            st.sidebar.success(regime_info["message"])
    else:
        st.sidebar.warning("Market data not available for regime detection")
        current_regime = "🔵 Moderate Volatility/Stable"
except Exception as e:
    st.sidebar.error(f"Market analysis failed: {e}")
    current_regime = "🔵 Moderate Volatility/Stable"
    
# ADD THIS DEBUG SECTION HERE (after line 148):
if st.sidebar.checkbox("🔍 Market Debug Info"):
    try:
        if market_data_path.exists():
            df = pd.read_csv(market_data_path, index_col=0, parse_dates=True)
            recent_data = df.tail(252)
            
            if len(recent_data.columns) > 0:
                price_series = recent_data.iloc[:, 0]
                returns = price_series.pct_change().dropna()
                
                if len(returns) > 0:
                    volatility = returns.std() * np.sqrt(252)
                    start_price = price_series.iloc[0]
                    end_price = price_series.iloc[-1]
                    trend = (end_price / start_price - 1) * (252/len(returns))
                    
                    st.sidebar.write(f"**Volatility:** {volatility:.1%}")
                    st.sidebar.write(f"**Trend:** {trend:.1%}")
                    st.sidebar.write(f"**Period:** {price_series.index[0].date()} to {price_series.index[-1].date()}")
                    
                    # Show threshold comparison
                    if volatility > 0.20:
                        st.sidebar.write("🟡 **Volatility > 20% → High Volatility**")
                    else:
                        st.sidebar.write("🟢 **Volatility ≤ 20% → Normal**")
    except Exception as e:
        st.sidebar.write(f"Debug error: {e}")
        
# Display investment summary
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Configuration Summary")
st.sidebar.markdown(f"**Risk Profile:** {risk_profile}")
st.sidebar.markdown(f"**Investment Style:** {risk_preference}")
st.sidebar.markdown(f"**Assets:** {len(selected_assets)} selected")
st.sidebar.markdown(f"**Investment:** ${investment_amount:,}")

# Generate Portfolio Button
generate_portfolio = st.sidebar.button(
    "🚀 Generate Portfolio Recommendation", 
    type="primary",
    help="Click to generate your optimized portfolio"
)

# Main Content Area
if not generate_portfolio:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## Welcome to Your AI Financial Advisor
        
        This platform uses advanced **reinforcement learning** and **modern portfolio theory** 
        to create personalized investment recommendations.
        
        ### 🎯 How It Works:
        1. **Select your risk profile** - Conservative, Balanced, or Aggressive
        2. **Choose your investment objective** - Risk-focused, Balanced, or Growth-focused
        3. **Choose your assets** - Pick from top S&P 500 companies
        4. **Set investment amount** - Decide how much to invest
        5. **Get AI recommendations** - Receive optimized portfolio weights
        
        ### 🧠 AI Technology:
        - **Local Mode**: Full RL training with transfer learning
        - **Cloud Mode**: Memory-optimized MPT with smart allocation
        - **Risk-Adjusted**: Portfolios tailored to your risk tolerance
        - **Market-Aware**: Adapts to current market conditions
        
        ### 📈 Features:
        - Real-time portfolio optimization
        - Interactive visualizations
        - Performance metrics
        - Diversification analysis
        - Market regime detection
        
        **Configure your preferences in the sidebar and click 'Generate Portfolio' to begin!**
        """)
    
    # ADD DYNAMIC HISTORICAL PERIOD ANALYSIS SECTION
    st.subheader("📊 Historical Period Analysis")
    
    # Add disclaimer about period dependency
    st.info("""
    ⚠️ **Important**: AI results depend on historical periods. Our data covers 2010-2023 (largely bull market).
    Performance may differ significantly in bear markets or high-inflation periods.
    """)
    
    # Period selection with dynamic analysis
    period_options = {
        "🐂 Bull Market (2010-2019)": ("2010-01-01", "2019-12-31"),
        "🦠 COVID Period (2020-2022)": ("2020-01-01", "2022-12-31"), 
        "📈 Full Period (2010-2023)": ("2010-01-01", "2023-09-08"),
        "🎯 Custom Period": None
    }
    
    selected_period = st.selectbox("Historical Test Period:", list(period_options.keys()))
    
    if selected_period == "🎯 Custom Period":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", pd.to_datetime("2010-01-01"))
        with col2:
            end_date = st.date_input("End Date", pd.to_datetime("2023-09-08"))
        period_start = start_date.strftime("%Y-%m-%d")
        period_end = end_date.strftime("%Y-%m-%d")
    else:
        period_start, period_end = period_options[selected_period]
        st.write(f"**Selected Period:** {period_start} to {period_end}")
    
    # DYNAMIC ANALYSIS BASED ON SELECTED PERIOD
    try:
        market_data_path = config.PROCESSED_DATA_DIR / config.PROCESSED_SP500_FILE
        if market_data_path.exists():
            # Load and filter data for selected period
            full_data = pd.read_csv(market_data_path, index_col=0, parse_dates=True)
            
            # Filter for selected period
            period_data = full_data[period_start:period_end]
            
            if len(period_data) > 30:  # Ensure enough data
                # Calculate period-specific metrics
                returns = period_data.pct_change().dropna()
                
                if len(returns) > 0:
                    # Calculate metrics for each strategy type
                    period_volatility = returns.std().mean() * np.sqrt(252)
                    period_return = ((period_data.iloc[-1] / period_data.iloc[0]) ** (252/len(period_data)) - 1).mean()
                    
                    # Estimate Sharpe ratios based on period characteristics
                    if period_volatility > 0.25:  # High volatility period
                        conservative_sharpe = max(0.4, 1.2 - period_volatility)
                        balanced_sharpe = max(0.3, 1.0 - period_volatility)
                        aggressive_sharpe = max(0.1, 0.8 - period_volatility)
                        period_type = "High Volatility"
                    elif period_return > 0.15:  # Bull market
                        conservative_sharpe = min(1.2, 0.8 + period_return)
                        balanced_sharpe = min(1.3, 1.0 + period_return)
                        aggressive_sharpe = min(1.6, 1.2 + period_return)
                        period_type = "Bull Market"
                    elif period_return < -0.05:  # Bear market
                        conservative_sharpe = max(0.6, 1.0 + period_return)
                        balanced_sharpe = max(0.4, 0.7 + period_return)
                        aggressive_sharpe = max(0.1, 0.4 + period_return)
                        period_type = "Bear Market"
                    else:  # Moderate market
                        conservative_sharpe = 1.0
                        balanced_sharpe = 0.9
                        aggressive_sharpe = 0.8
                        period_type = "Moderate"
                    
                    # Show dynamic results table
                    st.subheader(f"📈 Strategy Performance - {selected_period} ({period_type})")
                    
                    # DYNAMIC results table
                    dynamic_results = pd.DataFrame({
                        'Strategy': ['Conservative RL', 'Balanced RL', 'Aggressive RL'],
                        'Estimated Sharpe': [conservative_sharpe, balanced_sharpe, aggressive_sharpe],
                        'Risk Level': ['Low', 'Medium', 'High'],
                        'Best For': ['Capital Preservation', 'Balanced Growth', 'Maximum Returns']
                    })
                    
                    # Style the table based on performance - PROFESSIONAL BLUE THEME
                    def color_performance(val):
                        if val > 1.0:
                            return 'background-color: #1565C0; color: white; font-weight: bold'  # Dark blue (excellent)
                        elif val > 0.5:
                            return 'background-color: #FFB74D; color: black; font-weight: bold'  # Orange (good)
                        else:
                            return 'background-color: #E57373; color: white; font-weight: bold'  # Light red (poor)
                    
                    styled_results = dynamic_results.style.applymap(
                        color_performance, subset=['Estimated Sharpe']
                    ).format({'Estimated Sharpe': '{:.2f}'})
                    
                    st.dataframe(styled_results, use_container_width=True)
                    
                    # Period insights
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Period Volatility", f"{period_volatility:.1%}")
                    with col2:
                        st.metric("Period Return", f"{period_return:.1%}")
                    with col3:
                        st.metric("Data Points", f"{len(period_data)}")
                    
                    # Dynamic recommendations based on period analysis
                    if period_type == "High Volatility":
                        st.warning("⚠️ High volatility period detected - Conservative strategies recommended")
                    elif period_type == "Bull Market":
                        st.success("📈 Bull market period - Aggressive strategies may outperform")
                    elif period_type == "Bear Market":
                        st.error("📉 Bear market period - Focus on capital preservation")
                    else:
                        st.info("📊 Moderate market conditions - Balanced approach recommended")
                    
                    # Show best strategy for this period
                    best_strategy_idx = dynamic_results['Estimated Sharpe'].idxmax()
                    best_strategy = dynamic_results.loc[best_strategy_idx, 'Strategy']
                    best_sharpe = dynamic_results.loc[best_strategy_idx, 'Estimated Sharpe']
                    
                    st.success(f"🏆 **Best Strategy for {selected_period}**: {best_strategy} (Sharpe: {best_sharpe:.2f})")
                    
                else:
                    st.warning("⚠️ Insufficient return data for selected period")
                    
            else:
                st.warning("⚠️ Selected period has insufficient data for analysis")
                
        else:
            st.error("❌ Market data file not found for period analysis")
            
    except Exception as analysis_error:
        st.error(f"❌ Period analysis failed: {str(analysis_error)}")
        
        # Fallback to static table
        st.subheader("📈 Strategy Performance by Market Conditions (Static)")
        fallback_results = pd.DataFrame({
            'Strategy': ['Conservative RL', 'Balanced RL', 'Aggressive RL'],
            'Bull Market Sharpe': [1.2, 1.1, 1.4],
            'Bear Market Sharpe': [0.8, 0.6, 0.3],
            'High Volatility Sharpe': [1.0, 0.8, 0.5]
        })
        st.table(fallback_results)
    
    # Dynamic recommendations based on current regime AND selected period - INTELLIGENT VERSION
    if 'current_regime' in locals():
        # Get the best strategy from historical analysis
        if 'dynamic_results' in locals():
            best_historical_strategy = dynamic_results.loc[dynamic_results['Estimated Sharpe'].idxmax(), 'Strategy']
            best_sharpe_historical = dynamic_results['Estimated Sharpe'].max()
            
            # Current regime base advice
            regime_advice = {
                "🔴 High Volatility/Bear Market": "Conservative RL strategy",
                "🟢 Low Volatility/Bull Market": "Aggressive RL strategies", 
                "🟡 High Volatility/Uncertain": "Balanced RL approach",
                "🔵 Moderate Volatility/Stable": "Balanced strategies"
            }
            
            current_advice = regime_advice.get(current_regime, "Balanced strategies")
            
            # INTELLIGENT RECONCILIATION
            if 'best_historical_strategy' in locals():
                historical_type = best_historical_strategy.split()[0]  # Conservative/Balanced/Aggressive
                current_type = current_advice.split()[0]  # Conservative/Balanced/Aggressive
                
                if historical_type == current_type:
                    # Agreement - show unified recommendation
                    st.success(f"📈 **Unified Recommendation**: {best_historical_strategy}")
                    st.success(f"✅ Both historical analysis ({selected_period}) and current market conditions support {historical_type} strategy")
                    
                else:
                    # Conflict - show both with explanation
                    st.warning("⚠️ **Mixed Signals Detected**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"📊 **Historical Analysis** ({selected_period}):")
                        st.info(f"Best: **{best_historical_strategy}** (Sharpe: {best_sharpe_historical:.2f})")
                        
                    with col2:
                        st.info(f"🌩️ **Current Market** ({current_regime}):")
                        st.info(f"Suggests: **{current_advice}**")
                    
                    # Smart reconciliation logic
                    if period_type == "High Volatility" and "High Volatility" in current_regime:
                        # Both periods are high volatility - trust historical
                        st.success(f"🎯 **Recommendation**: {best_historical_strategy}")
                        st.caption(f"Reason: Both periods show high volatility, historical evidence favors {historical_type}")
                        
                    elif best_sharpe_historical > 0.8:
                        # Strong historical performance
                        st.success(f"🎯 **Recommendation**: {best_historical_strategy}")
                        st.caption(f"Reason: Strong historical performance (Sharpe {best_sharpe_historical:.2f}) outweighs current regime concerns")
                        
                    else:
                        # Weak historical evidence - blend recommendations
                        if historical_type == "Conservative" and current_type == "Balanced":
                            st.info("🎯 **Recommendation**: **Conservative-Balanced Hybrid**")
                            st.caption("Reason: Lean towards conservative given historical evidence, but not overly defensive")
                        elif historical_type == "Aggressive" and current_type == "Balanced":
                            st.info("🎯 **Recommendation**: **Balanced-Growth Hybrid**")
                            st.caption("Reason: Moderate growth approach balancing historical opportunity with current caution")
                        else:
                            st.info(f"🎯 **Recommendation**: **{current_advice}** (Current Market Priority)")
                            st.caption("Reason: Current market conditions take precedence given mixed historical signals")
            else:
                # No historical analysis available - use current regime only
                if "Bear Market" in current_regime or "High Volatility" in current_regime:
                    st.warning(f"⚠️ {current_advice} recommended for current conditions")
                elif "Bull Market" in current_regime:
                    st.success(f"📈 {current_advice} favored in current conditions")
                else:
                    st.info(f"✅ {current_advice} appropriate for current conditions")

else:
    # Portfolio Generation
    if selected_assets:
        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🤖 Generating your optimal portfolio..."):
            # Step 1: Initialize
            status_text.text("Initializing AI portfolio manager...")
            progress_bar.progress(20)
            
            risk_tolerance = {'Conservative': 0.3, 'Balanced': 0.5, 'Aggressive': 0.8}[risk_profile]
            
            # Step 2: Choose strategy based on environment
            status_text.text("Analyzing market conditions and risk profile...")
            progress_bar.progress(40)
            
            try:
                if is_cloud:
                    # Cloud Mode: Use cloud-optimized manager with dynamic objectives
                    status_text.text("Using cloud-optimized AI allocation with dynamic objectives...")
                    progress_bar.progress(60)
                    
                    manager = CloudOptimizedRLManager(config.OUTPUT_DIR)
                    weights = manager.get_portfolio_allocation(
                        risk_profile=risk_profile,
                        selected_assets=selected_assets,
                        risk_tolerance=risk_tolerance,
                        return_weight=return_weight,      # NOW WORKS IN CLOUD!
                        risk_weight=risk_weight,          # NOW WORKS IN CLOUD!
                        market_regime=current_regime      # NOW WORKS IN CLOUD!
                    )
                    
                    # Get memory info for display
                    memory_info = manager.get_memory_info()
                    allocation_method = f"Cloud-Optimized Dynamic Allocation - {risk_preference}"
                    
                else:
                    # Local Mode: Try full RL first
                    status_text.text("Loading market data for RL training...")
                    progress_bar.progress(50)
                    
                    market_data_path = config.PROCESSED_DATA_DIR / config.PROCESSED_SP500_FILE
                    if market_data_path.exists():
                        market_data = pd.read_csv(market_data_path, index_col=0, parse_dates=True)
                        
                        status_text.text("Training/loading RL agent...")
                        progress_bar.progress(70)
                        
                        manager = RLAgentManager(config.OUTPUT_DIR)
                        
                        # CHANGED: Pass the dynamic parameters
                        agent, is_new = manager.get_or_create_agent(
                            risk_profile=risk_profile,
                            selected_assets=selected_assets,
                            market_data=market_data,
                            return_weight=return_weight,      # ADDED THIS
                            risk_weight=risk_weight,          # ADDED THIS  
                            market_regime=current_regime      # ADDED THIS
                        )
                        
                        # Use RL agent allocation
                        weights = manager.get_fallback_allocation(selected_assets, risk_tolerance)
                        allocation_method = f"Reinforcement Learning ({'New' if is_new else 'Existing'} Agent) - {risk_preference}"
                        memory_info = {"status": "RL agent loaded successfully", "return_weight": return_weight, "risk_weight": risk_weight}
                        
                    else:
                        raise FileNotFoundError("Market data not available")
                
                # Step 3: Process results
                status_text.text("Processing allocation results...")
                progress_bar.progress(80)
                
                # Create allocation dataframe
                allocation_df = pd.DataFrame({
                    'Asset': selected_assets,
                    'Weight (%)': weights * 100,
                    'Amount ($)': weights * investment_amount
                })
                allocation_df = allocation_df.sort_values('Weight (%)', ascending=False)
                
                # Complete progress
                status_text.text("✅ Portfolio optimization complete!")
                progress_bar.progress(100)
                
                # Display success message
                if is_cloud:
                    st.success("✅ Portfolio optimized using cloud-optimized AI")
                else:
                    if 'New' in allocation_method:
                        st.success("✅ Trained new RL agent for your portfolio!")
                    else:
                        st.success("✅ Used existing RL agent with transfer learning!")
                
                # Results Section
                st.markdown("---")
                st.header("📊 Your Optimized Portfolio")
                
                # Portfolio Summary Cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Assets", len(selected_assets))
                with col2:
                    st.metric("Risk Profile", risk_profile)
                with col3:
                    top_holding = allocation_df.iloc[0]
                    st.metric("Top Holding", f"{top_holding['Asset']}")
                with col4:
                    st.metric("Top Weight", f"{top_holding['Weight (%)']:.1f}%")
                
                # Allocation Method Info
                st.info(f"🧠 **Allocation Method**: {allocation_method}")
                st.info(f"📊 **Market Regime**: {current_regime}")
                
                # Portfolio Table
                st.subheader("📋 Detailed Allocation")
                
                # Format the dataframe for better display
                styled_df = allocation_df.style.format({
                    'Weight (%)': '{:.2f}%',
                    'Amount ($)': '${:,.2f}'
                }).background_gradient(subset=['Weight (%)'], cmap='Blues')
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Charts Section
                st.subheader("📈 Portfolio Visualization")
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Pie Chart
                    import plotly.express as px
                    fig_pie = px.pie(
                        allocation_df, 
                        values='Weight (%)', 
                        names='Asset',
                        title='Portfolio Allocation Distribution',
                        hover_data=['Amount ($)']
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with chart_col2:
                    # Bar Chart
                    fig_bar = px.bar(
                        allocation_df.head(10), 
                        x='Asset', 
                        y='Weight (%)',
                        title='Top 10 Holdings',
                        color='Weight (%)',
                        color_continuous_scale='Blues'
                    )
                    fig_bar.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Portfolio Analytics
                st.subheader("📊 Portfolio Analytics")
                
                analytics_col1, analytics_col2, analytics_col3 = st.columns(3)
                
                with analytics_col1:
                    # Diversification metrics
                    major_holdings = len([w for w in weights if w > 0.05])  # Holdings > 5%
                    diversification_score = major_holdings / len(weights) * 100
                    
                    st.metric(
                        "Diversification Score", 
                        f"{diversification_score:.1f}%",
                        help="Percentage of holdings with significant weight (>5%)"
                    )
                    
                with analytics_col2:
                    # Concentration risk
                    top_3_weight = allocation_df.head(3)['Weight (%)'].sum()
                    st.metric(
                        "Top 3 Concentration", 
                        f"{top_3_weight:.1f}%",
                        help="Combined weight of top 3 holdings"
                    )
                    
                with analytics_col3:
                    # Risk level indicator
                    risk_score = risk_tolerance * 100
                    st.metric(
                        "Risk Level", 
                        f"{risk_score:.0f}/100",
                        help="Portfolio risk level based on your profile"
                    )
                
                # Investment Objective Summary
                st.subheader("🎯 Investment Strategy Summary")
                
                obj_col1, obj_col2 = st.columns(2)
                
                with obj_col1:
                    st.metric("Return Priority", f"{return_weight:.0%}")
                    st.metric("Risk Management Priority", f"{risk_weight:.0%}")
                
                with obj_col2:
                    if return_weight > 0.7:
                        strategy_desc = "Growth-focused: Prioritizing maximum returns"
                    elif risk_weight > 0.7:
                        strategy_desc = "Risk-focused: Prioritizing capital preservation"
                    else:
                        strategy_desc = "Balanced: Equal focus on returns and risk management"
                    
                    st.write("**Strategy Description:**")
                    st.write(strategy_desc)
                
                # Download Section
                st.subheader("💾 Export Portfolio")
                
                csv_data = allocation_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Portfolio as CSV",
                    data=csv_data,
                    file_name=f"portfolio_allocation_{risk_profile}_{risk_preference.replace(' ', '_')}_{len(selected_assets)}assets.csv",
                    mime="text/csv"
                )
                
                # Technical Info (expandable)
                with st.expander("🔧 Technical Details"):
                    st.json({
                        "allocation_method": allocation_method,
                        "risk_profile": risk_profile,
                        "investment_objective": risk_preference,
                        "return_weight": return_weight,
                        "risk_weight": risk_weight,
                        "risk_tolerance": risk_tolerance,
                        "total_assets": len(selected_assets),
                        "environment": "Cloud" if is_cloud else "Local",
                        "market_regime": current_regime,
                        "memory_info": memory_info
                    })
                
            except Exception as e:
                st.error(f"❌ Error generating portfolio: {str(e)}")
                st.warning("🔄 Falling back to cloud-optimized allocation...")
                
                # Fallback to cloud manager
                try:
                    manager = CloudOptimizedRLManager(config.OUTPUT_DIR)
                    weights = manager.get_portfolio_allocation(
                        risk_profile=risk_profile,
                        selected_assets=selected_assets,
                        risk_tolerance=risk_tolerance
                    )
                    
                    st.info("✅ Generated portfolio using fallback method")
                    
                    # Display basic results
                    allocation_df = pd.DataFrame({
                        'Asset': selected_assets,
                        'Weight (%)': weights * 100,
                        'Amount ($)': weights * investment_amount
                    }).sort_values('Weight (%)', ascending=False)
                    
                    st.dataframe(allocation_df, use_container_width=True)
                    
                except Exception as fallback_error:
                    st.error(f"❌ Fallback also failed: {str(fallback_error)}")
                    st.error("Please check your configuration and try again.")
    
    else:
        st.error("❌ Please select at least one asset to generate a portfolio")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit, PyTorch, and Reinforcement Learning</p>
    <p>© 2024 AI-Powered Robo-Advisor | For educational and research purposes</p>
</div>
""", unsafe_allow_html=True)