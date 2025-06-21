"""Launches the Streamlit dashboard application with intelligent RL management."""
import os
import sys
from pathlib import Path

def main():
    """Sets up the Python path and launches the Streamlit app."""
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    print("--- Starting AI Robo-Advisor Dashboard ---")
    
    # Import config to check paths
    import src.config as config
    
    # Check for required files
    required_files = [
        (config.PROCESSED_DATA_DIR / config.PROCESSED_SCF_FILE, "SCF processed data"),
        (config.PROCESSED_DATA_DIR / config.PROCESSED_SP500_FILE, "S&P 500 processed data"),
        (config.OUTPUT_DIR / config.RISK_MODEL_FILE, "Risk tolerance model")
    ]
    
    missing_files = []
    
    for file_path, description in required_files:
        if file_path.exists():
            print(f"✓ {description}: Found")
        else:
            print(f"❌ {description}: Missing")
            missing_files.append((file_path, description))
    
    # Check for optional RL models
    rl_models = ['Conservative', 'Balanced', 'Aggressive']
    rl_models_found = 0
    
    for profile in rl_models:
        model_files = list(config.OUTPUT_DIR.glob(f"{profile}_*.pth"))
        if model_files:
            rl_models_found += 1
            print(f"✓ {profile} RL model: Found")
        else:
            print(f"⚠️  {profile} RL model: Will train on demand")
    
    if missing_files:
        print(f"\n❌ {len(missing_files)} required files missing. Please run:")
        print("   1. python scripts/run_data_processing.py")
        print("   2. python scripts/run_risk_model_training.py")
        
        response = input("\nContinue anyway? (y/N): ").lower().strip()
        if response not in ['y', 'yes']:
            return
    
    # Check if dashboard exists, create if not
    dashboard_path = project_root / "dashboard" / "app.py"
    
    if not dashboard_path.exists():
        dashboard_path.parent.mkdir(exist_ok=True)
        create_streamlit_dashboard(dashboard_path)
        print(f"✓ Created dashboard: {dashboard_path}")
    
    # Set environment
    os.environ['PYTHONPATH'] = str(project_root)
    
    # Detect Streamlit Cloud
    is_streamlit_cloud = (
        os.environ.get('STREAMLIT_SHARING') == 'true' or
        'streamlit.io' in os.environ.get('STREAMLIT_SERVER_HEADLESS', '')
    )
    
    if is_streamlit_cloud:
        print("🌩️  Streamlit Cloud detected - Using cloud-optimized mode")
        os.environ['STREAMLIT_CLOUD'] = 'true'
    else:
        print("🖥️  Local mode - Full RL capabilities available")
    
    # Launch dashboard
    command = f"streamlit run {dashboard_path}"
    print(f"\nLaunching: {command}")
    print("🌐 Dashboard will open at: http://localhost:8501")
    
    try:
        os.system(command)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")

def create_streamlit_dashboard(app_path: Path):
    """Create optimized Streamlit dashboard."""
    code = '''
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Setup
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import src.config as config
    from src.models.cloud_optimized_agent import CloudOptimizedRLManager
    from src.models.rl_agent_manager import RLAgentManager
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

st.set_page_config(page_title="AI Robo-Advisor", page_icon="🤖", layout="wide")
st.title("🤖 AI-Powered Robo-Advisor")

# Check environment
is_cloud = os.environ.get('STREAMLIT_CLOUD') == 'true'

if is_cloud:
    st.info("🌩️ Cloud Mode: Using optimized MPT allocation")
else:
    st.info("🖥️ Local Mode: Full RL training available")

# Sidebar
st.sidebar.header("Portfolio Configuration")

risk_profile = st.sidebar.selectbox(
    "Risk Profile", 
    ["Conservative", "Balanced", "Aggressive"]
)

available_assets = config.DEFAULT_PORTFOLIO_ASSETS
selected_assets = st.sidebar.multiselect(
    "Select Assets",
    available_assets,
    default=available_assets[:10]
)

investment_amount = st.sidebar.number_input(
    "Investment Amount ($)",
    min_value=1000,
    value=100000,
    step=1000
)

# Generate portfolio
if st.sidebar.button("🚀 Generate Portfolio", type="primary"):
    if selected_assets:
        with st.spinner("Generating optimal portfolio..."):
            
            risk_tolerance = {'Conservative': 0.3, 'Balanced': 0.5, 'Aggressive': 0.8}[risk_profile]
            
            if is_cloud:
                # Use cloud-optimized manager
                manager = CloudOptimizedRLManager(config.OUTPUT_DIR)
                weights = manager.get_portfolio_allocation(
                    risk_profile, selected_assets, risk_tolerance
                )
                st.success("✅ Portfolio optimized using cloud AI")
                
            else:
                # Try full RL, fallback to cloud manager
                try:
                    market_data_path = config.PROCESSED_DATA_DIR / config.PROCESSED_SP500_FILE
                    if market_data_path.exists():
                        market_data = pd.read_csv(market_data_path, index_col=0, parse_dates=True)
                        
                        manager = RLAgentManager(config.OUTPUT_DIR)
                        agent, is_new = manager.get_or_create_agent(
                            risk_profile, selected_assets, market_data
                        )
                        weights = manager.get_fallback_allocation(selected_assets, risk_tolerance)
                        
                        if is_new:
                            st.success("✅ Trained new RL agent!")
                        else:
                            st.success("✅ Using existing RL agent!")
                    else:
                        raise FileNotFoundError("Market data not available")
                        
                except Exception as e:
                    st.warning(f"RL failed: {str(e)[:30]}... Using MPT fallback")
                    manager = CloudOptimizedRLManager(config.OUTPUT_DIR)
                    weights = manager.get_portfolio_allocation(
                        risk_profile, selected_assets, risk_tolerance
                    )
            
            # Display results
            allocation_df = pd.DataFrame({
                'Asset': selected_assets,
                'Weight (%)': weights * 100,
                'Amount ($)': weights * investment_amount
            }).sort_values('Weight (%)', ascending=False)
            
            st.subheader("📊 Portfolio Allocation")
            st.dataframe(allocation_df, use_container_width=True)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                import plotly.express as px
                fig = px.pie(allocation_df, values='Weight (%)', names='Asset')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(allocation_df.head(10), x='Asset', y='Weight (%)')
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Assets", len(selected_assets))
            with col2:
                st.metric("Risk Profile", risk_profile)
            with col3:
                top_holding = allocation_df.iloc[0]
                st.metric("Top Holding", f"{top_holding['Asset']} ({top_holding['Weight (%)']:.1f}%)")
    else:
        st.error("Please select at least one asset")

st.sidebar.markdown("---")
st.sidebar.markdown("**Built with ❤️ using Streamlit & AI**")
'''
    app_path.write_text(code.strip())

if __name__ == "__main__":
    main()