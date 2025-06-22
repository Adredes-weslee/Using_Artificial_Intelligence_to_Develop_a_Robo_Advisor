"""Streamlit Page: Portfolio Optimizer with PyTorch RL integration."""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import sys
import os

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import src.config as config
from src.models.cloud_optimized_agent import CloudOptimizedRLManager
from src.models.rl_agent_manager import RLAgentManager


try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    
st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide")
st.title("📈 AI-Powered Portfolio Optimizer")
st.markdown("### Generate optimal portfolio allocations using reinforcement learning")

# Check environment
is_cloud = os.environ.get('STREAMLIT_CLOUD') == 'true'

# Display mode
if is_cloud:
    st.info("🌩️ **Cloud Mode**: Using memory-optimized MPT allocation")
else:
    st.info("🖥️ **Local Mode**: Full RL training capabilities available")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_market_data_with_fallback(tickers: list) -> pd.DataFrame:
    """Fetches live data with a fallback to a processed file."""
    try:
        # Primary: Try to fetch live data
        st.info("📡 Fetching live market data...")
        live_data = yf.download(tickers, period="1y", progress=False)
        if live_data.empty:
            raise ValueError("No data returned from yfinance.")
        
        # Handle single ticker vs multiple tickers
        if len(tickers) == 1:
            return pd.DataFrame({tickers[0]: live_data['Adj Close']})
        else:
            return live_data['Adj Close']
            
    except Exception as e:
        # Secondary: Fallback to processed file
        st.warning(f"⚠️ Could not fetch live data ({str(e)[:50]}...). Using historical data.")
        fallback_path = config.PROCESSED_DATA_DIR / config.PROCESSED_SP500_FILE
        
        if fallback_path.exists():
            historical_data = pd.read_csv(fallback_path, index_col=0, parse_dates=True)
            # Filter for requested tickers
            available_tickers = [t for t in tickers if t in historical_data.columns]
            if available_tickers:
                return historical_data[available_tickers]
            else:
                st.error("❌ No data available for selected tickers")
                return pd.DataFrame()
        else:
            st.error("❌ No fallback data available. Please run data processing first.")
            return pd.DataFrame()

def load_rl_managers():
    """Load both RL managers."""
    try:
        cloud_manager = CloudOptimizedRLManager(config.OUTPUT_DIR)
        
        if not is_cloud:
            rl_manager = RLAgentManager(config.OUTPUT_DIR)
            return cloud_manager, rl_manager
        else:
            return cloud_manager, None
            
    except Exception as e:
        st.error(f"❌ Error loading RL managers: {e}")
        return None, None

# Load managers
cloud_manager, rl_manager = load_rl_managers()

if cloud_manager:
    # Check if risk assessment was done
    risk_from_profiler = st.session_state.get('recommended_profile', None)
    
    if risk_from_profiler:
        st.success(f"✅ Using risk profile from assessment: **{risk_from_profiler}**")
    
    # Portfolio Configuration
    st.subheader("🎯 Portfolio Configuration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        # Risk Profile
        risk_profile_options = ["Conservative", "Balanced", "Aggressive"]
        default_index = 0
        
        if risk_from_profiler and risk_from_profiler in risk_profile_options:
            default_index = risk_profile_options.index(risk_from_profiler)
        
        risk_profile = st.selectbox(
            "Select Risk Profile",
            risk_profile_options,
            index=default_index,
            help="Choose your investment risk tolerance"
        )
        
        # Risk tolerance display
        risk_tolerance_map = {'Conservative': 0.3, 'Balanced': 0.5, 'Aggressive': 0.8}
        risk_tolerance = risk_tolerance_map[risk_profile]
        st.metric("Risk Tolerance Score", f"{risk_tolerance:.1f}")
        
    with config_col2:
        # Investment amount
        investment_amount = st.number_input(
            "Investment Amount ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=5000,
            help="Total amount to invest"
        )
        
        # Portfolio size recommendation
        if is_cloud:
            max_assets = 10
            st.caption(f"⚠️ Cloud mode: Max {max_assets} assets recommended")
        else:
            max_assets = 25
            st.caption(f"💻 Local mode: Up to {max_assets} assets supported")
    
    # Asset Selection
    st.subheader("📊 Asset Selection")
    
    # Asset categories
    asset_categories = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
        'Finance': ['JPM', 'BAC', 'V', 'MA', 'GS', 'MS'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO'],
        'Consumer': ['PG', 'KO', 'HD', 'COST', 'DIS'],
        'Communication': ['VZ', 'T', 'NFLX'],
        'Energy': ['XOM', 'CVX']
    }
    
    # Asset selection tabs
    selection_method = st.radio(
        "Asset Selection Method",
        ["Quick Selection", "By Category", "Custom Selection"],
        horizontal=True
    )
    
    if selection_method == "Quick Selection":
        quick_option = st.selectbox(
            "Choose a preset portfolio",
            ["Conservative Mix", "Balanced Mix", "Growth Mix", "Tech Focus", "Dividend Focus"]
        )
        
        if quick_option == "Conservative Mix":
            selected_assets = ['AAPL', 'MSFT', 'JNJ', 'PG', 'KO', 'VZ', 'JPM']
        elif quick_option == "Balanced Mix":
            selected_assets = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ', 'HD', 'V', 'MA']
        elif quick_option == "Growth Mix":
            selected_assets = ['AAPL', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        elif quick_option == "Tech Focus":
            selected_assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']
        else:  # Dividend Focus
            selected_assets = ['JNJ', 'PG', 'KO', 'VZ', 'T', 'JPM', 'HD']
            
    elif selection_method == "By Category":
        selected_assets = []
        
        for category, assets in asset_categories.items():
            with st.expander(f"{category} Stocks"):
                category_selection = st.multiselect(
                    f"Select {category} assets:",
                    assets,
                    key=f"category_{category}"
                )
                selected_assets.extend(category_selection)
                
    else:  # Custom Selection
        available_assets = config.DEFAULT_PORTFOLIO_ASSETS
        selected_assets = st.multiselect(
            "Select assets for your portfolio:",
            available_assets,
            default=available_assets[:8],
            help=f"Choose up to {max_assets} assets for optimal performance"
        )
    
    # Remove duplicates and limit
    selected_assets = list(set(selected_assets))[:max_assets]
    
    # Display selection summary
    if selected_assets:
        st.success(f"✅ Selected {len(selected_assets)} assets: {', '.join(selected_assets)}")
        
        if len(selected_assets) > max_assets:
            st.warning(f"⚠️ Too many assets selected. Using first {max_assets}.")
            selected_assets = selected_assets[:max_assets]
            
    else:
        st.warning("⚠️ Please select at least one asset.")
    
    # Generate Portfolio Button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        generate_portfolio = st.button(
            "🚀 Generate Optimized Portfolio",
            type="primary",
            disabled=len(selected_assets) == 0,
            help="Generate AI-optimized portfolio allocation"
        )
    
    # Portfolio Generation
    if generate_portfolio and selected_assets:
        st.markdown("---")
        st.header("🤖 AI Portfolio Generation")
        
        # Progress tracking
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Load market data
                status_text.text("📡 Loading market data...")
                progress_bar.progress(20)
                
                market_data = get_market_data_with_fallback(selected_assets)
                
                if market_data.empty:
                    st.error("❌ Could not load market data for selected assets")
                    st.stop()
                
                # Step 2: Initialize AI manager
                status_text.text("🧠 Initializing AI portfolio manager...")
                progress_bar.progress(40)
                
                # Step 3: Generate allocation
                if is_cloud:
                    status_text.text("☁️ Generating cloud-optimized allocation...")
                    progress_bar.progress(70)
                    
                    weights = cloud_manager.get_portfolio_allocation(
                        risk_profile=risk_profile,
                        selected_assets=selected_assets,
                        risk_tolerance=risk_tolerance
                    )
                    allocation_method = "Cloud-Optimized MPT"
                    
                else:
                    status_text.text("🎯 Training/loading RL agent...")
                    progress_bar.progress(60)
                    
                    try:
                        agent, is_new = rl_manager.get_or_create_agent(
                            risk_profile=risk_profile,
                            selected_assets=selected_assets,
                            market_data=market_data
                        )
                        
                        status_text.text("📊 Generating RL-based allocation...")
                        progress_bar.progress(80)
                        
                        weights = rl_manager.get_fallback_allocation(selected_assets, risk_tolerance)
                        allocation_method = f"Reinforcement Learning ({'New' if is_new else 'Existing'} Agent)"
                        
                    except Exception as rl_error:
                        st.warning(f"⚠️ RL failed: {str(rl_error)[:50]}... Using fallback")
                        weights = cloud_manager.get_portfolio_allocation(
                            risk_profile=risk_profile,
                            selected_assets=selected_assets,
                            risk_tolerance=risk_tolerance
                        )
                        allocation_method = "MPT Fallback"
                
                # Step 4: Process results
                status_text.text("📈 Processing optimization results...")
                progress_bar.progress(90)
                
                # Create allocation dataframe
                allocation_df = pd.DataFrame({
                    'Asset': selected_assets,
                    'Weight (%)': weights * 100,
                    'Amount ($)': weights * investment_amount
                })
                allocation_df = allocation_df.sort_values('Weight (%)', ascending=False)
                
                # Complete
                status_text.text("✅ Portfolio optimization complete!")
                progress_bar.progress(100)
                
                # Display results
                st.success(f"✅ Portfolio optimized using {allocation_method}")
                
                # Results section
                st.markdown("---")
                st.header("📊 Your Optimized Portfolio")
                
                # Summary metrics
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                
                with summary_col1:
                    st.metric("Total Assets", len(selected_assets))
                with summary_col2:
                    st.metric("Risk Profile", risk_profile)
                with summary_col3:
                    top_holding = allocation_df.iloc[0]
                    st.metric("Top Holding", f"{top_holding['Asset']}")
                with summary_col4:
                    st.metric("Top Weight", f"{top_holding['Weight (%)']:.1f}%")
                
                # Allocation table
                st.subheader("📋 Portfolio Allocation")
                
                styled_df = allocation_df.style.format({
                    'Weight (%)': '{:.2f}%',
                    'Amount ($)': '${:,.2f}'
                }).background_gradient(subset=['Weight (%)'], cmap='RdYlGn')
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Visualizations
                st.subheader("📊 Portfolio Visualization")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    import plotly.express as px
                    
                    # Pie chart
                    fig_pie = px.pie(
                        allocation_df,
                        values='Weight (%)',
                        names='Asset',
                        title='Portfolio Distribution',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with viz_col2:
                    # Bar chart
                    fig_bar = px.bar(
                        allocation_df,
                        x='Asset',
                        y='Weight (%)',
                        title='Asset Weights',
                        color='Weight (%)',
                        color_continuous_scale='Blues'
                    )
                    fig_bar.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Portfolio analytics
                st.subheader("📈 Portfolio Analytics")
                
                analytics_col1, analytics_col2, analytics_col3 = st.columns(3)
                
                with analytics_col1:
                    # Diversification
                    major_holdings = len([w for w in weights if w > 0.05])
                    st.metric("Major Holdings (>5%)", f"{major_holdings}")
                    
                with analytics_col2:
                    # Concentration
                    top_5_weight = allocation_df.head(5)['Weight (%)'].sum()
                    st.metric("Top 5 Concentration", f"{top_5_weight:.1f}%")
                    
                with analytics_col3:
                    # Expected risk level
                    st.metric("Risk Level", f"{risk_tolerance * 100:.0f}/100")
                
                # Market data preview
                if not market_data.empty:
                    st.subheader("📊 Market Data Preview")
                    
                    # Show recent price movements
                    recent_data = market_data.tail(30)  # Last 30 days
                    
                    if len(recent_data) > 1:
                        # Calculate daily returns
                        returns = recent_data.pct_change().dropna()
                        
                        # Portfolio performance simulation
                        portfolio_returns = (returns * weights).sum(axis=1)
                        cumulative_returns = (1 + portfolio_returns).cumprod()
                        
                        # Plot portfolio performance
                        import plotly.graph_objects as go
                        
                        fig_perf = go.Figure()
                        fig_perf.add_trace(go.Scatter(
                            x=cumulative_returns.index,
                            y=cumulative_returns.values,
                            mode='lines',
                            name='Portfolio Performance',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig_perf.update_layout(
                            title='Simulated Portfolio Performance (Last 30 Days)',
                            xaxis_title='Date',
                            yaxis_title='Cumulative Return',
                            hovermode='x'
                        )
                        
                        st.plotly_chart(fig_perf, use_container_width=True)
                        
                        # Performance metrics
                        perf_col1, perf_col2, perf_col3 = st.columns(3)
                        
                        with perf_col1:
                            total_return = (cumulative_returns.iloc[-1] - 1) * 100
                            st.metric("30-Day Return", f"{total_return:.2f}%")
                            
                        with perf_col2:
                            volatility = portfolio_returns.std() * np.sqrt(252) * 100  # Annualized
                            st.metric("Annualized Volatility", f"{volatility:.2f}%")
                            
                        with perf_col3:
                            if volatility > 0:
                                sharpe = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))
                                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                            else:
                                st.metric("Sharpe Ratio", "N/A")
                
                # Export functionality
                st.subheader("💾 Export Portfolio")
                
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    csv_data = allocation_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv_data,
                        file_name=f"portfolio_{risk_profile}_{len(selected_assets)}assets.csv",
                        mime="text/csv"
                    )
                
                with export_col2:
                    # Portfolio summary for copying
                    summary_text = f"""Portfolio Summary:
Risk Profile: {risk_profile}
Total Assets: {len(selected_assets)}
Investment Amount: ${investment_amount:,}
Allocation Method: {allocation_method}

Asset Allocation:
{allocation_df.to_string(index=False)}"""
                    
                    st.download_button(
                        label="📄 Download Summary",
                        data=summary_text,
                        file_name=f"portfolio_summary_{risk_profile}.txt",
                        mime="text/plain"
                    )
                
                # Technical details
                with st.expander("🔧 Technical Details"):
                    st.json({
                        "allocation_method": allocation_method,
                        "risk_profile": risk_profile,
                        "risk_tolerance": risk_tolerance,
                        "total_assets": len(selected_assets),
                        "environment": "Cloud" if is_cloud else "Local",
                        "selected_assets": selected_assets,
                        "investment_amount": investment_amount
                    })
                
            except Exception as e:
                st.error(f"❌ Error generating portfolio: {str(e)}")
                st.error("Please check your configuration and try again.")
                
                # Show error details for debugging
                with st.expander("🐛 Error Details"):
                    st.code(str(e))
    
    # Tips and guidance
    st.markdown("---")
    st.subheader("💡 Portfolio Optimization Tips")
    
    tip_col1, tip_col2 = st.columns(2)
    
    with tip_col1:
        st.markdown("""
        **Asset Selection:**
        - Choose 5-15 assets for good diversification
        - Mix different sectors and asset classes
        - Consider correlation between assets
        - Include both growth and defensive stocks
        """)
    
    with tip_col2:
        st.markdown("""
        **Risk Management:**
        - Match portfolio risk to your tolerance
        - Regular rebalancing is important
        - Monitor market conditions
        - Review and adjust periodically
        """)

else:
    st.error("❌ Could not load portfolio optimization system.")
    st.error("Please ensure all dependencies are installed and models are trained.")

# Footer
st.markdown("---")
st.caption("Portfolio recommendations are for educational purposes only and should not be considered as financial advice.")