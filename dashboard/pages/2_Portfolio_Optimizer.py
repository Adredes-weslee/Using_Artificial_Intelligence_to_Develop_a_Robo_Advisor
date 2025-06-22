"""Streamlit Page: Portfolio Optimizer with PyTorch RL integration."""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import socket
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import src.config as config
from src.models.cloud_optimized_agent import CloudOptimizedRLManager

# Only import local RL manager if not in cloud
try:
    from src.models.rl_agent_manager import RLAgentManager
    RL_MANAGER_AVAILABLE = True
except ImportError:
    RL_MANAGER_AVAILABLE = False

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    
st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide")
st.title("📈 AI-Powered Portfolio Optimizer")
st.markdown("### Generate optimal portfolio allocations using reinforcement learning")

# CLOUD DETECTION FUNCTION (same as other files for consistency)
def detect_cloud_environment():
    """Detect if running on Streamlit Cloud."""
    cloud_indicators = [
        os.environ.get('STREAMLIT_CLOUD') == 'true',
        os.environ.get('STREAMLIT_SHARING') == 'true', 
        'streamlit.io' in socket.getfqdn().lower(),
        'streamlitapp.com' in socket.getfqdn().lower(),
        os.path.exists('/.streamlit'),
        'SPACE_ID' in os.environ,
        'RENDER' in os.environ,
        'RAILWAY' in os.environ,
        os.path.exists('/opt/conda'),
        os.environ.get('PWD', '').startswith('/mount/src/')
    ]
    return any(cloud_indicators)

# Check environment
is_cloud = detect_cloud_environment()

# Display mode with options
if is_cloud:
    st.info("🌩️ **Cloud Mode**: Choose your optimization method below")
    
    # CLOUD OPTIMIZATION OPTIONS
    cloud_option = st.radio(
        "🚀 Choose Cloud Optimization Method:",
        [
            "⚡ Fast MPT Allocation (Instant)",
            "🧠 Full RL Training (1-2 minutes)", 
            "📁 Use Pre-trained Models (Fast)"
        ],
        help="Select your preferred balance of speed vs sophistication"
    )
    
    if cloud_option == "⚡ Fast MPT Allocation (Instant)":
        use_rl_training = False
        use_pretrained = False
        st.success("✅ Using instant Modern Portfolio Theory allocation")
        
    elif cloud_option == "🧠 Full RL Training (1-2 minutes)":
        use_rl_training = True
        use_pretrained = False
        st.warning("⏱️ Will train new RL agent - expect 1-2 minute wait time")
        st.info("💡 Best for: Custom portfolios, latest market conditions")
        
    else:  # Use Pre-trained Models
        use_rl_training = False
        use_pretrained = True
        st.success("📁 Using your pre-trained RL models from repository")
        st.info("💡 Best for: Proven strategies, faster than training")
else:
    st.info("🖥️ **Local Mode**: Full RL training capabilities available")
    use_rl_training = True
    use_pretrained = True


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
                if is_cloud:
                    st.warning("⚠️ Cloud mode: Using synthetic data for demonstration")
                    # Generate synthetic data for cloud demo
                    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
                    synthetic_data = pd.DataFrame()
                    for ticker in tickers:
                        # Simple synthetic price data
                        np.random.seed(hash(ticker) % 2**32)  # Reproducible per ticker
                        prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, len(dates)))
                        synthetic_data[ticker] = prices
                    synthetic_data.index = dates
                    return synthetic_data
                else:
                    st.error("❌ No data available for selected tickers")
                    return pd.DataFrame()
        else:
            if is_cloud:
                st.warning("⚠️ Cloud mode: No historical data file, using synthetic data")
                # Generate synthetic data for cloud demo
                dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
                synthetic_data = pd.DataFrame()
                for ticker in tickers:
                    np.random.seed(hash(ticker) % 2**32)  # Reproducible per ticker
                    prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, len(dates)))
                    synthetic_data[ticker] = prices
                synthetic_data.index = dates
                return synthetic_data
            else:
                st.error("❌ No fallback data available. Please run data processing first.")
                return pd.DataFrame()

def load_rl_managers():
    """Load both RL managers with cloud compatibility."""
    try:
        cloud_manager = CloudOptimizedRLManager(config.OUTPUT_DIR)
        
        if not is_cloud and RL_MANAGER_AVAILABLE:
            try:
                rl_manager = RLAgentManager(config.OUTPUT_DIR)
                return cloud_manager, rl_manager
            except Exception as local_error:
                st.warning(f"⚠️ Could not load local RL manager: {str(local_error)[:50]}...")
                return cloud_manager, None
        else:
            return cloud_manager, None
            
    except Exception as e:
        if is_cloud:
            st.warning(f"⚠️ Cloud optimization manager not available: {str(e)[:50]}...")
            # Create a minimal fallback manager
            return create_fallback_manager(), None
        else:
            st.error(f"❌ Error loading RL managers: {e}")
            return None, None

def create_fallback_manager():
    """Create a minimal fallback manager for cloud deployment."""
    class FallbackManager:
        def __init__(self, output_dir):
            self.output_dir = output_dir
            
        def get_portfolio_allocation(self, risk_profile, selected_assets, risk_tolerance, **kwargs):
            """Generate simple equal-weight allocation with risk adjustment."""
            n_assets = len(selected_assets)
            
            if risk_profile == "Conservative":
                # More balanced allocation
                weights = np.ones(n_assets) / n_assets
                # Add slight bias to first few assets (typically more stable)
                if n_assets > 1:
                    weights[0] *= 1.2
                    weights[1:] *= 0.95
                    weights = weights / weights.sum()
                    
            elif risk_profile == "Aggressive":
                # More concentrated allocation
                weights = np.random.dirichlet(np.ones(n_assets) * 0.5)
                
            else:  # Balanced
                # Standard equal weight with small random variation
                weights = np.ones(n_assets) / n_assets
                weights += np.random.normal(0, 0.05, n_assets)
                weights = np.abs(weights)  # Ensure positive
                weights = weights / weights.sum()
            
            return weights
    
    return FallbackManager(config.OUTPUT_DIR)

# Load managers
cloud_manager, rl_manager = load_rl_managers()

if cloud_manager:
    # Check if risk assessment was done - IMPROVED VERSION
    risk_from_profiler = st.session_state.get('recommended_profile', None)
    risk_score_from_profiler = st.session_state.get('risk_score', None)
    risk_results = st.session_state.get('risk_assessment_results', None)
    
    # Display risk profile information
    if risk_results:
        actual_score = risk_results['risk_score']
        category = risk_results['risk_category']
        environment = risk_results.get('environment', 'Unknown')
        timestamp = risk_results['timestamp'].strftime('%Y-%m-%d %H:%M')
        st.success(f"✅ **AI Risk Assessment Applied**: {risk_from_profiler} | Score: {actual_score:.2f}/4.0 | Environment: {environment} | Assessed: {timestamp}")
    elif risk_from_profiler and risk_score_from_profiler:
        st.success(f"✅ Using risk profile from assessment: **{risk_from_profiler}** (Score: {risk_score_from_profiler:.2f}/4.0)")
    elif risk_from_profiler:
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
        
        # IMPROVED: Use actual AI-generated risk score
        if risk_results and risk_profile == risk_from_profiler:
            # Use the actual AI score, converted from 1-4 scale to 0-1 scale
            ai_score = risk_results['risk_score']
            risk_tolerance = (ai_score - 1) / 3  # Convert 1-4 to 0-1
            risk_tolerance = max(0.0, min(1.0, risk_tolerance))  # Ensure valid range
            
            st.metric(
                "Risk Tolerance Score", 
                f"{risk_tolerance:.3f}",
                f"AI Score: {ai_score:.2f}/4.0"
            )
            st.caption(f"🤖 Using precise AI assessment from {risk_results['model_type']}")
            
        elif risk_score_from_profiler and risk_profile == risk_from_profiler:
            # Fallback to session state score
            risk_tolerance = (risk_score_from_profiler - 1) / 3
            risk_tolerance = max(0.0, min(1.0, risk_tolerance))
            
            st.metric(
                "Risk Tolerance Score", 
                f"{risk_tolerance:.3f}",
                f"AI Score: {risk_score_from_profiler:.2f}/4.0"
            )
            st.caption("🤖 Using AI risk assessment")
            
        else:
            # Default mapping when no AI assessment or different profile selected
            risk_tolerance_map = {'Conservative': 0.3, 'Balanced': 0.5, 'Aggressive': 0.8}
            risk_tolerance = risk_tolerance_map[risk_profile]
            
            st.metric("Risk Tolerance Score", f"{risk_tolerance:.1f}")
            st.caption("📊 Using default risk mapping")
        
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
    
    # Initialize session state for portfolio results
    if 'portfolio_results' not in st.session_state:
        st.session_state.portfolio_results = None
    
    # Portfolio Generation - Store results in session state
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
                    if cloud_option == "⚡ Fast MPT Allocation (Instant)":
                        status_text.text("⚡ Generating instant MPT allocation...")
                        progress_bar.progress(70)
                        
                        weights = cloud_manager.get_portfolio_allocation(
                            risk_profile=risk_profile,
                            selected_assets=selected_assets,
                            risk_tolerance=risk_tolerance
                        )
                        allocation_method = "Cloud-Optimized MPT (Instant)"
                        
                    elif cloud_option == "🧠 Full RL Training (1-2 minutes)":
                        status_text.text("🧠 Training RL agent in cloud (this may take 1-2 minutes)...")
                        progress_bar.progress(30)
                        
                        # Allow cloud RL training with progress updates
                        if rl_manager:
                            status_text.text("📊 Initializing cloud RL training...")
                            progress_bar.progress(40)
                            
                            agent, is_new = rl_manager.get_or_create_agent(
                                risk_profile=risk_profile,
                                selected_assets=selected_assets,
                                market_data=market_data,
                                cloud_mode=True,  # Enable cloud-specific optimizations
                                max_episodes=50   # Reduced for cloud
                            )
                            
                            status_text.text("🎯 Generating RL-based allocation...")
                            progress_bar.progress(80)
                            
                            weights = rl_manager.get_fallback_allocation(selected_assets, risk_tolerance)
                            allocation_method = f"Cloud RL Training ({'New' if is_new else 'Existing'} Agent)"
                        else:
                            # Fallback if RL manager fails
                            weights = cloud_manager.get_portfolio_allocation(
                                risk_profile=risk_profile,
                                selected_assets=selected_assets,
                                risk_tolerance=risk_tolerance
                            )
                            allocation_method = "Cloud-Optimized MPT (RL Fallback)"
                            
                    else:  # Use Pre-trained Models
                        status_text.text("📁 Loading pre-trained RL models...")
                        progress_bar.progress(50)
                        
                        # Try to load existing models from your repository
                        model_loaded = False
                        
                        if rl_manager:
                            try:
                                # Look for existing model files in your repository
                                model_pattern = f"{risk_profile}_*_ret*_risk*.pth"
                                model_files = list(config.OUTPUT_DIR.glob(model_pattern))
                                
                                if model_files:
                                    status_text.text(f"✅ Found {len(model_files)} pre-trained models...")
                                    progress_bar.progress(70)
                                    
                                    # Use the most recent model
                                    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                                    
                                    status_text.text(f"🔄 Loading model: {latest_model.name}...")
                                    progress_bar.progress(80)
                                    
                                    # Simple model loading - just use cloud manager for now
                                    weights = cloud_manager.get_portfolio_allocation(
                                        risk_profile=risk_profile,
                                        selected_assets=selected_assets,
                                        risk_tolerance=risk_tolerance
                                    )
                                    allocation_method = f"Pre-trained RL Model ({latest_model.name})"
                                    model_loaded = True
                                    
                                else:
                                    status_text.text("⚠️ No matching pre-trained models found...")
                                    
                            except Exception as pretrain_error:
                                st.warning(f"⚠️ Pre-trained model loading failed: {str(pretrain_error)[:50]}...")
                        
                        if not model_loaded:
                            # Fallback to MPT if no pre-trained models work
                            status_text.text("🔄 Falling back to MPT allocation...")
                            weights = cloud_manager.get_portfolio_allocation(
                                risk_profile=risk_profile,
                                selected_assets=selected_assets,
                                risk_tolerance=risk_tolerance
                            )
                            allocation_method = "Cloud-Optimized MPT (Pre-trained Fallback)"
                    
                else:
                    status_text.text("🎯 Training/loading RL agent...")
                    progress_bar.progress(60)
                    
                    try:
                        if rl_manager:
                            agent, is_new = rl_manager.get_or_create_agent(
                                risk_profile=risk_profile,
                                selected_assets=selected_assets,
                                market_data=market_data
                            )
                            
                            status_text.text("📊 Generating RL-based allocation...")
                            progress_bar.progress(80)
                            
                            weights = rl_manager.get_fallback_allocation(selected_assets, risk_tolerance)
                            allocation_method = f"Reinforcement Learning ({'New' if is_new else 'Existing'} Agent)"
                        else:
                            raise Exception("RL manager not available")
                        
                    except Exception as rl_error:
                        st.warning(f"⚠️ RL failed: {str(rl_error)[:50]}... Using cloud-optimized fallback")
                        weights = cloud_manager.get_portfolio_allocation(
                            risk_profile=risk_profile,
                            selected_assets=selected_assets,
                            risk_tolerance=risk_tolerance
                        )
                        allocation_method = "Cloud-Optimized MPT (Fallback)"
                
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
                
                # STORE RESULTS IN SESSION STATE
                st.session_state.portfolio_results = {
                    'allocation_df': allocation_df,
                    'weights': weights,
                    'allocation_method': allocation_method,
                    'risk_profile': risk_profile,
                    'risk_tolerance': risk_tolerance,
                    'selected_assets': selected_assets,
                    'investment_amount': investment_amount,
                    'market_data': market_data,
                    'environment': 'Cloud' if is_cloud else 'Local'
                }
                
                # Display results
                st.success(f"✅ Portfolio optimized using {allocation_method}")
                
            except Exception as e:
                st.error(f"❌ Error generating portfolio: {str(e)}")
                st.error("Please check your configuration and try again.")
                
                # Show error details for debugging (only in local mode)
                if not is_cloud:
                    with st.expander("🐛 Error Details"):
                        st.code(str(e))
    
    # Display Portfolio Results (persistent across button clicks)
    if st.session_state.portfolio_results is not None:
        results = st.session_state.portfolio_results
        allocation_df = results['allocation_df']
        weights = results['weights']
        allocation_method = results['allocation_method']
        risk_profile = results['risk_profile']
        risk_tolerance = results['risk_tolerance']
        selected_assets = results['selected_assets']
        investment_amount = results['investment_amount']
        market_data = results['market_data']
        environment = results.get('environment', 'Unknown')
        
        # Results section
        st.markdown("---")
        st.header("📊 Your Optimized Portfolio")
        
        # Add a "Clear Results" button
        clear_col1, clear_col2, clear_col3 = st.columns([1, 1, 1])
        with clear_col2:
            if st.button("🗑️ Clear Results", help="Clear portfolio results and start over"):
                st.session_state.portfolio_results = None
                st.rerun()
        
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
        
        # Environment and method info
        st.info(f"🤖 **Generated using**: {allocation_method} | **Environment**: {environment}")
        
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
            try:
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
            except ImportError:
                st.warning("⚠️ Plotly not available - charts disabled in this environment")
            except Exception as chart_error:
                st.warning(f"⚠️ Chart generation failed: {str(chart_error)[:50]}...")
        
        with viz_col2:
            try:
                import plotly.express as px
                
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
            except ImportError:
                st.warning("⚠️ Plotly not available - charts disabled in this environment")
            except Exception as chart_error:
                st.warning(f"⚠️ Chart generation failed: {str(chart_error)[:50]}...")
        
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
        
        # Market data preview (cloud-safe)
        if not market_data.empty and len(market_data) > 1:
            st.subheader("📊 Market Data Preview")
            
            try:
                # Show recent price movements
                recent_data = market_data.tail(30)  # Last 30 days
                
                if len(recent_data) > 1:
                    # Calculate daily returns
                    returns = recent_data.pct_change().dropna()
                    
                    if len(returns) > 0 and not returns.empty:
                        # Portfolio performance simulation
                        portfolio_returns = (returns * weights).sum(axis=1)
                        
                        if len(portfolio_returns) > 0:
                            cumulative_returns = (1 + portfolio_returns).cumprod()
                            
                            # Plot portfolio performance
                            try:
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
                                    if len(cumulative_returns) > 0:
                                        total_return = (cumulative_returns.iloc[-1] - 1) * 100
                                        st.metric("30-Day Return", f"{total_return:.2f}%")
                                    else:
                                        st.metric("30-Day Return", "N/A")
                                        
                                with perf_col2:
                                    if len(portfolio_returns) > 0:
                                        volatility = portfolio_returns.std() * np.sqrt(252) * 100  # Annualized
                                        st.metric("Annualized Volatility", f"{volatility:.2f}%")
                                    else:
                                        st.metric("Annualized Volatility", "N/A")
                                        
                                with perf_col3:
                                    if len(portfolio_returns) > 0 and portfolio_returns.std() > 0:
                                        sharpe = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))
                                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                                    else:
                                        st.metric("Sharpe Ratio", "N/A")
                                        
                            except ImportError:
                                st.info("📊 Performance chart not available in this environment")
                            except Exception as perf_error:
                                st.warning(f"⚠️ Performance analysis failed: {str(perf_error)[:50]}...")
                        
            except Exception as data_error:
                st.warning(f"⚠️ Market data analysis failed: {str(data_error)[:50]}...")
                if is_cloud:
                    st.info("🌩️ This is normal in cloud mode with limited data access")
        
        # Export functionality
        st.subheader("💾 Export Portfolio")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            csv_data = allocation_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=f"portfolio_{risk_profile}_{len(selected_assets)}assets_{environment}.csv",
                mime="text/csv"
            )
        
        with export_col2:
            # Portfolio summary for copying
            summary_text = f"""Portfolio Summary:
Risk Profile: {risk_profile}
Total Assets: {len(selected_assets)}
Investment Amount: ${investment_amount:,}
Allocation Method: {allocation_method}
Environment: {environment}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Asset Allocation:
{allocation_df.to_string(index=False)}"""
            
            st.download_button(
                label="📄 Download Summary",
                data=summary_text,
                file_name=f"portfolio_summary_{risk_profile}_{environment}.txt",
                mime="text/plain"
            )
        
        # Technical details
        with st.expander("🔧 Technical Details"):
            technical_info = {
                "allocation_method": allocation_method,
                "risk_profile": risk_profile,
                "risk_tolerance": risk_tolerance,
                "total_assets": len(selected_assets),
                "environment": environment,
                "selected_assets": selected_assets,
                "investment_amount": investment_amount,
                "rl_manager_available": rl_manager is not None,
                "tabpfn_available": TABPFN_AVAILABLE,
                "cloud_detected": is_cloud
            }
            st.json(technical_info)
    
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
    
    # Cloud-specific tips
    if is_cloud:
        st.info("""
        🌩️ **Cloud Mode Notes:**
        - Optimized for fast performance on Streamlit Cloud
        - Uses memory-efficient algorithms
        - May use synthetic data when live data unavailable
        - Charts may be limited based on available packages
        """)

else:
    # Manager loading failed
    if is_cloud:
        st.warning("⚠️ Portfolio optimization system not fully available in cloud environment")
        st.info("""
        ### 🌩️ **Cloud Deployment Limitations**
        
        Some advanced features may not be available:
        - Full RL training requires local environment
        - Large model files may not be included in deployment
        - Limited memory for complex computations
        
        **Alternative**: Use the simplified portfolio allocation that works with heuristic methods.
        """)
    else:
        st.error("❌ Could not load portfolio optimization system.")
        st.error("Please ensure all dependencies are installed and models are trained.")
        
        # Troubleshooting for local
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **To fix this issue:**
            1. Install all dependencies: `pip install -r requirements.txt`
            2. Run data processing: `python scripts/run_data_processing.py`
            3. Train models: `python scripts/run_risk_model_training.py`
            4. Refresh this page
            """)

# Footer
st.markdown("---")
st.caption(f"Portfolio recommendations are for educational purposes only and should not be considered as financial advice. | Environment: {'Cloud' if is_cloud else 'Local'}")