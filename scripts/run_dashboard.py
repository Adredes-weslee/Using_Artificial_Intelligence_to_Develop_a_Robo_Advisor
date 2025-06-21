"""Launches the Dash dashboard application with RL Agent Manager support."""
import os
import sys
from pathlib import Path
import warnings

def main():
    """Sets up the Python path and launches the Dash app."""
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Check if required files exist
    print("--- Starting Advanced Robo-Advisor Dashboard ---")
    print("\nChecking required files...")
    
    # Import config to check paths
    import src.config as config
    
    # Check for processed data files
    required_files = [
        (config.PROCESSED_DATA_DIR / config.PROCESSED_SCF_FILE, "SCF processed data"),
        (config.PROCESSED_DATA_DIR / config.PROCESSED_SP500_FILE, "S&P 500 processed data"),
        (config.OUTPUT_DIR / config.RISK_MODEL_FILE, "Risk tolerance model")
    ]
    
    # Check for RL models (optional but recommended)
    rl_model_files = [
        (config.OUTPUT_DIR / "Conservative_AAPL_AMZN_GOOGL_JPM_MSFT.pth", "Conservative RL model"),
        (config.OUTPUT_DIR / "Balanced_AAPL_AMZN_GOOGL_JPM_MSFT.pth", "Balanced RL model"),
        (config.OUTPUT_DIR / "Aggressive_AAPL_AMZN_GOOGL_JPM_MSFT.pth", "Aggressive RL model")
    ]
    
    missing_files = []
    optional_missing = []
    
    # Check required files
    for file_path, description in required_files:
        if file_path.exists():
            print(f"✓ {description}: {file_path}")
        else:
            print(f"❌ {description}: {file_path} (missing)")
            missing_files.append((file_path, description))
    
    # Check optional RL model files
    rl_models_found = 0
    for file_path, description in rl_model_files:
        if file_path.exists():
            print(f"✓ {description}: Found")
            rl_models_found += 1
        else:
            optional_missing.append((file_path, description))
    
    if rl_models_found == 0:
        print(f"⚠️  No pre-trained RL models found - will train on demand")
    else:
        print(f"✓ Found {rl_models_found}/3 pre-trained RL models")
    
    if missing_files:
        print(f"\n❌ Error: {len(missing_files)} required files are missing:")
        for file_path, description in missing_files:
            print(f"   - {description}: {file_path}")
        print("\nPlease run the following scripts first:")
        print("   1. python scripts/run_data_processing.py")
        print("   2. python scripts/run_risk_model_training.py")
        print("   3. python scripts/run_rl_agent_training.py (optional, for faster startup)")
        
        # Ask user if they want to continue
        response = input("\nDo you want to continue anyway? (y/N): ").lower().strip()
        if response not in ['y', 'yes']:
            print("Dashboard launch cancelled.")
            return
    
    # Check memory and warn about RL training
    print(f"\n" + "="*60)
    print("DEPLOYMENT CONSIDERATIONS")
    print("="*60)
    
    try:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        print(f"Available memory: {available_memory_gb:.1f} GB")
        
        if available_memory_gb < 4:
            print("⚠️  WARNING: Low memory detected!")
            print("   - RL training may fail on this system")
            print("   - Consider using pre-trained models only")
            print("   - Streamlit Community Cloud has ~1GB memory limit")
    except ImportError:
        print("💡 Install 'psutil' to check system memory")
    
    print("\nMemory Usage Guidelines:")
    print("  - Pre-trained models: ~100MB")
    print("  - RL training (10 assets, 50 episodes): ~2-4GB")
    print("  - For Streamlit Cloud: Use fallback to MPT optimization")
    
    # Check if dashboard file exists
    dashboard_candidates = [
        project_root / "notebooks" / "04_robo_advisor_dashboard_with_chatbot.py",
        project_root / "dashboard" / "app.py",
        project_root / "app.py"  # Common streamlit location
    ]
    
    app_path = None
    for candidate in dashboard_candidates:
        if candidate.exists():
            app_path = candidate
            break
    
    if not app_path:
        # Create a basic dashboard template
        print("\nDashboard script not found. Creating basic template...")
        dashboard_dir = project_root / "dashboard"
        dashboard_dir.mkdir(exist_ok=True)
        app_path = dashboard_dir / "app.py"
        
        # Create basic streamlit app template
        create_basic_dashboard_template(app_path)
        print(f"✓ Created basic dashboard template: {app_path}")
    
    print(f"\n✓ Dashboard script: {app_path}")
    
    # Set environment variables
    os.environ['PYTHONPATH'] = str(project_root)
    
    # Detect if running on Streamlit Cloud
    is_streamlit_cloud = (
        os.environ.get('STREAMLIT_SHARING') == 'true' or
        'streamlit.io' in os.environ.get('STREAMLIT_SERVER_HEADLESS', '')
    )
    
    if is_streamlit_cloud:
        print("\n🌩️  Detected Streamlit Community Cloud deployment")
        print("   - Memory-optimized mode activated")
        print("   - RL training disabled (will use MPT fallback)")
        os.environ['STREAMLIT_CLOUD'] = 'true'
    
    # Launch the dashboard
    if str(app_path).endswith('.py'):
        command = f"streamlit run {app_path}"
        print(f"\nLaunching Streamlit dashboard...")
        print(f"Command: {command}")
        
        if not is_streamlit_cloud:
            print("\n🌐 Dashboard will open at: http://localhost:8501")
            print("📝 Press Ctrl+C to stop the dashboard")
        
        try:
            os.system(command)
        except KeyboardInterrupt:
            print("\n\nDashboard stopped by user.")
        except Exception as e:
            print(f"\n❌ Error launching dashboard: {e}")
            print("\nTry running manually:")
            print(f"   cd {project_root}")
            print(f"   {command}")
    else:
        print("❌ Could not determine dashboard type")

def create_basic_dashboard_template(app_path: Path):
    """Create a basic Streamlit dashboard template with hybrid approach."""
    template_code = '''
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import src.config as config
    from src.models.cloud_optimized_agent import CloudOptimizedRLManager
    from src.models.rl_agent_manager import RLAgentManager
    from src.models.risk_profiler import load_compressed_model
    from src.utils.portfolio_math import mean_variance_optimization
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

st.set_page_config(
    page_title="AI Robo-Advisor",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI-Powered Robo-Advisor")
st.markdown("### Intelligent Portfolio Optimization with Reinforcement Learning")

# Check if running on Streamlit Cloud
is_cloud = os.environ.get('STREAMLIT_CLOUD') == 'true'

if is_cloud:
    st.warning("🌩️ Running on Streamlit Cloud - Using cloud-optimized AI")
else:
    st.info("🖥️ Running locally - Full RL capabilities available")

# Sidebar for user inputs
st.sidebar.header("Portfolio Configuration")

# Risk profile selection
risk_profile = st.sidebar.selectbox(
    "Select Risk Profile",
    ["Conservative", "Balanced", "Aggressive"],
    help="Choose your investment risk tolerance"
)

# Asset selection
available_assets = config.DEFAULT_PORTFOLIO_ASSETS
selected_assets = st.sidebar.multiselect(
    "Select Assets",
    available_assets,
    default=available_assets[:10],
    help="Choose 5-15 assets for optimal performance"
)

# Investment amount
investment_amount = st.sidebar.number_input(
    "Investment Amount ($)",
    min_value=1000,
    max_value=10000000,
    value=100000,
    step=1000
)

# Generate portfolio button
if st.sidebar.button("🚀 Generate Portfolio Recommendation", type="primary"):
    if selected_assets:
        st.header("📊 Portfolio Recommendation")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Initialize manager
        status_text.text("Initializing AI portfolio manager...")
        progress_bar.progress(25)
        
        # Risk tolerance mapping
        risk_tolerance_map = {'Conservative': 0.3, 'Balanced': 0.5, 'Aggressive': 0.8}
        risk_tolerance = risk_tolerance_map[risk_profile]
        
        # Step 2: Generate allocation based on environment
        status_text.text("Generating optimal portfolio allocation...")
        progress_bar.progress(50)
        
        if is_cloud:
            # Cloud mode: Use cloud-optimized manager
            st.info("🌩️ Using Cloud-Optimized AI Portfolio Manager")
            
            cloud_manager = CloudOptimizedRLManager(config.OUTPUT_DIR)
            weights = cloud_manager.get_portfolio_allocation(
                risk_profile=risk_profile,
                selected_assets=selected_assets,
                risk_tolerance=risk_tolerance
            )
            
            # Show memory info
            memory_info = cloud_manager.get_memory_info()
            st.sidebar.info(f"Memory usage: {memory_info['memory_mb']} MB")
            
        else:
            # Local mode: Try full RL agent first, fallback to cloud manager
            st.info("🤖 Using Full AI-Powered RL Agent")
            
            try:
                # Load market data for RL training
                market_data_path = config.PROCESSED_DATA_DIR / config.PROCESSED_SP500_FILE
                if market_data_path.exists():
                    market_data = pd.read_csv(market_data_path, index_col=0, parse_dates=True)
                    
                    # Use full RL manager
                    rl_manager = RLAgentManager(config.OUTPUT_DIR)
                    agent, is_new = rl_manager.get_or_create_agent(
                        risk_profile=risk_profile,
                        selected_assets=selected_assets,
                        market_data=market_data
                    )
                    
                    # Generate weights using RL agent
                    weights = rl_manager.get_fallback_allocation(selected_assets, risk_tolerance)
                    
                    if is_new:
                        st.success("✅ Trained new RL agent for your portfolio!")
                    else:
                        st.success("✅ Using optimized existing RL agent!")
                        
                else:
                    # Fallback to cloud manager if no market data
                    st.warning("📊 Market data not found, using MPT optimization")
                    cloud_manager = CloudOptimizedRLManager(config.OUTPUT_DIR)
                    weights = cloud_manager.get_portfolio_allocation(
                        risk_profile=risk_profile,
                        selected_assets=selected_assets,
                        risk_tolerance=risk_tolerance
                    )
                    
            except Exception as e:
                st.warning(f"⚠️ RL agent failed: {str(e)[:50]}...")
                st.info("🔄 Falling back to cloud-optimized manager")
                
                cloud_manager = CloudOptimizedRLManager(config.OUTPUT_DIR)
                weights = cloud_manager.get_portfolio_allocation(
                    risk_profile=risk_profile,
                    selected_assets=selected_assets,
                    risk_tolerance=risk_tolerance
                )
        
        # Step 3: Display results
        status_text.text("Preparing portfolio visualization...")
        progress_bar.progress(75)
        
        # Create allocation dataframe
        allocation_df = pd.DataFrame({
            'Asset': selected_assets,
            'Weight (%)': weights * 100,
            'Amount ($)': weights * investment_amount
        })
        allocation_df = allocation_df.sort_values('Weight (%)', ascending=False)
        
        # Display allocation table
        st.subheader("📋 Portfolio Allocation")
        st.dataframe(
            allocation_df.style.format({
                'Weight (%)': '{:.2f}%',
                'Amount ($)': '${:,.2f}'
            }),
            use_container_width=True
        )
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            import plotly.express as px
            fig_pie = px.pie(
                allocation_df, 
                values='Weight (%)', 
                names='Asset', 
                title='Portfolio Allocation'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            fig_bar = px.bar(
                allocation_df.head(10), 
                x='Asset', 
                y='Weight (%)',
                title='Top 10 Holdings'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Portfolio summary
        st.subheader("📈 Portfolio Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Assets", len(selected_assets))
        with col2:
            st.metric("Risk Profile", risk_profile)
        with col3:
            st.metric("Top Holding", f"{allocation_df.iloc[0]['Asset']} ({allocation_df.iloc[0]['Weight (%)']:.1f}%)")
        with col4:
            st.metric("Diversification", f"{len([w for w in weights if w > 0.05])}/{len(weights)} major holdings")
        
        # Complete progress
        progress_bar.progress(100)
        status_text.text("✅ Portfolio optimization complete!")
        
    else:
        st.error("❌ Please select at least one asset")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Portfolio Guidelines")
st.sidebar.markdown("""
- **Conservative**: Low risk, stable returns
- **Balanced**: Moderate risk, balanced growth
- **Aggressive**: Higher risk, growth focused
- **5-15 assets**: Optimal for performance
- **Cloud**: Uses MPT optimization
- **Local**: Full RL training available
""")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, PyTorch, and Reinforcement Learning")
'''
    
    app_path.write_text(template_code.strip())

if __name__ == "__main__":
    main()