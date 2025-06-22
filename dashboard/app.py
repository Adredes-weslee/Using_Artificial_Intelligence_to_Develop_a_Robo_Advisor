"""Main Streamlit application file for the AI-Powered Robo-Advisor."""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from src.utils.market_analysis import detect_market_regime, get_regime_recommendations  # ADD THIS LINE


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

# Display investment summary
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Configuration Summary")
st.sidebar.markdown(f"**Risk Profile:** {risk_profile}")
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
        2. **Choose your assets** - Pick from top S&P 500 companies
        3. **Set investment amount** - Decide how much to invest
        4. **Get AI recommendations** - Receive optimized portfolio weights
        
        ### 🧠 AI Technology:
        - **Local Mode**: Full RL training with transfer learning
        - **Cloud Mode**: Memory-optimized MPT with smart allocation
        - **Risk-Adjusted**: Portfolios tailored to your risk tolerance
        
        ### 📈 Features:
        - Real-time portfolio optimization
        - Interactive visualizations
        - Performance metrics
        - Diversification analysis
        
        **Configure your preferences in the sidebar and click 'Generate Portfolio' to begin!**
        """)

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
                    # Cloud Mode: Use cloud-optimized manager
                    status_text.text("Using cloud-optimized AI allocation...")
                    progress_bar.progress(60)
                    
                    manager = CloudOptimizedRLManager(config.OUTPUT_DIR)
                    weights = manager.get_portfolio_allocation(
                        risk_profile=risk_profile,
                        selected_assets=selected_assets,
                        risk_tolerance=risk_tolerance
                    )
                    
                    # Get memory info for display
                    memory_info = manager.get_memory_info()
                    allocation_method = "Cloud-Optimized MPT"
                    
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
                        agent, is_new = manager.get_or_create_agent(
                            risk_profile=risk_profile,
                            selected_assets=selected_assets,
                            market_data=market_data
                        )
                        
                        # Use RL agent allocation
                        weights = manager.get_fallback_allocation(selected_assets, risk_tolerance)
                        allocation_method = f"Reinforcement Learning ({'New' if is_new else 'Existing'} Agent)"
                        memory_info = {"status": "RL agent loaded successfully"}
                        
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
                
                # Download Section
                st.subheader("💾 Export Portfolio")
                
                csv_data = allocation_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Portfolio as CSV",
                    data=csv_data,
                    file_name=f"portfolio_allocation_{risk_profile}_{len(selected_assets)}assets.csv",
                    mime="text/csv"
                )
                
                # Technical Info (expandable)
                with st.expander("🔧 Technical Details"):
                    st.json({
                        "allocation_method": allocation_method,
                        "risk_tolerance": risk_tolerance,
                        "total_assets": len(selected_assets),
                        "environment": "Cloud" if is_cloud else "Local",
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