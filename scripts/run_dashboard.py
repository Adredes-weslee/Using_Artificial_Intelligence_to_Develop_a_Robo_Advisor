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
    
    # Use existing dashboard
    dashboard_path = project_root / "dashboard" / "app.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard not found at: {dashboard_path}")
        print("Please ensure the dashboard directory exists with app.py")
        return
    
    print(f"✓ Found dashboard: {dashboard_path}")
    
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

if __name__ == "__main__":
    main()