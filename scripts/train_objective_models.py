"""Script to pre-train RL models for different investment objectives.

Run this locally to generate models for all objective combinations,
then upload the trained models to Streamlit Cloud.
"""
import os
import warnings
import numpy as np

# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import src.config as config
from src.models.rl_agent_manager import RLAgentManager

def train_all_objective_models():
    """Train RL agents for all combinations of risk profiles and objectives."""
    
    # Load market data
    market_data_path = config.PROCESSED_DATA_DIR / config.PROCESSED_SP500_FILE
    if not market_data_path.exists():
        print("❌ Market data not found. Please run data processing first.")
        return
    
    market_data = pd.read_csv(market_data_path, index_col=0, parse_dates=True)
    print(f"✅ Loaded market data: {market_data.shape}")
    
    # Initialize manager
    manager = RLAgentManager(config.OUTPUT_DIR)
    
    # Define training configurations
    models_to_train = [
        # Conservative variations
        {"risk_profile": "Conservative", "return_weight": 0.2, "risk_weight": 0.8, "name": "Risk-Focused"},
        {"risk_profile": "Conservative", "return_weight": 0.5, "risk_weight": 0.5, "name": "Academic"},
        {"risk_profile": "Conservative", "return_weight": 0.8, "risk_weight": 0.2, "name": "Growth-Focused"},
        
        # Balanced variations  
        {"risk_profile": "Balanced", "return_weight": 0.2, "risk_weight": 0.8, "name": "Risk-Focused"},
        {"risk_profile": "Balanced", "return_weight": 0.5, "risk_weight": 0.5, "name": "Academic"},
        {"risk_profile": "Balanced", "return_weight": 0.8, "risk_weight": 0.2, "name": "Growth-Focused"},
        
        # Aggressive variations
        {"risk_profile": "Aggressive", "return_weight": 0.2, "risk_weight": 0.8, "name": "Risk-Focused"},
        {"risk_profile": "Aggressive", "return_weight": 0.5, "risk_weight": 0.5, "name": "Academic"},
        {"risk_profile": "Aggressive", "return_weight": 0.8, "risk_weight": 0.2, "name": "Growth-Focused"},
    ]
    
    # Default asset selection for training
    selected_assets = config.DEFAULT_PORTFOLIO_ASSETS[:15]  # Use first 15 assets
    print(f"Training with assets: {selected_assets}")
    
    # Train all models
    total_models = len(models_to_train)
    
    for i, model_config in enumerate(models_to_train, 1):
        print(f"\n🤖 Training Model {i}/{total_models}")
        print(f"   Risk Profile: {model_config['risk_profile']}")
        print(f"   Objective: {model_config['name']}")
        print(f"   Return Weight: {model_config['return_weight']:.0%}")
        print(f"   Risk Weight: {model_config['risk_weight']:.0%}")
        
        try:
            # Train the agent
            agent, is_new = manager.get_or_create_agent(
                risk_profile=model_config['risk_profile'],
                selected_assets=selected_assets,
                market_data=market_data,
                return_weight=model_config['return_weight'],
                risk_weight=model_config['risk_weight'],
                market_regime="🔵 Moderate Volatility/Stable"
            )
            
            if is_new:
                print(f"   ✅ Successfully trained new {model_config['risk_profile']} agent")
            else:
                print(f"   ✅ Agent already exists or loaded from cache")
                
        except Exception as e:
            print(f"   ❌ Failed to train {model_config['risk_profile']} {model_config['name']}: {e}")
    
    print(f"\n🎉 Training complete! Models saved to: {config.OUTPUT_DIR}")
    print(f"📁 Upload the .pth files to your Streamlit Cloud repository")


if __name__ == "__main__":
    print("🚀 Starting RL model training for all investment objectives...")
    train_all_objective_models()