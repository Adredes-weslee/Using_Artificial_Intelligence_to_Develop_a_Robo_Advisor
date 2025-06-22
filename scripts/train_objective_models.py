"""Script to pre-train RL models for different investment objectives.

Run this locally to generate models for all objective combinations,
then upload the trained models to Streamlit Cloud.
"""
import os
import warnings
import numpy as np
import logging

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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model_path(models_dir: Path, risk_profile: str, selected_assets: list, 
                   return_weight: float, risk_weight: float) -> Path:
    """Generate model file path for given configuration (matching RL manager pattern)."""
    asset_key = "_".join(sorted(selected_assets))
    objective_suffix = f"ret{return_weight}_risk{risk_weight}"
    model_filename = f"{risk_profile}_{asset_key}_{objective_suffix}.pth"
    return models_dir / model_filename

def train_all_objective_models():
    """Train RL agents for all combinations of risk profiles and objectives."""
    
    # Load market data
    market_data_path = config.PROCESSED_DATA_DIR / config.PROCESSED_SP500_FILE
    if not market_data_path.exists():
        logger.error(f"Market data not found: {market_data_path}")
        return
    
    logger.info("🚀 Starting RL model training for all investment objectives...")
    market_data = pd.read_csv(market_data_path, index_col=0, parse_dates=True)
    logger.info(f"✅ Loaded market data: {market_data.shape}")
    
    # Select a consistent set of assets
    selected_assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 
                      'ADBE', 'CRM', 'INTC', 'JPM', 'BAC', 'V', 'MA', 'GS']
    
    print(f"Training with assets: {selected_assets}")
    
    # Define all model configurations
    models_to_train = [
        # Conservative profiles
        {"risk_profile": "Conservative", "return_weight": 0.2, "risk_weight": 0.8, "name": "Risk-Focused"},
        {"risk_profile": "Conservative", "return_weight": 0.5, "risk_weight": 0.5, "name": "Academic"},
        {"risk_profile": "Conservative", "return_weight": 0.8, "risk_weight": 0.2, "name": "Growth-Focused"},
        
        # Balanced profiles  
        {"risk_profile": "Balanced", "return_weight": 0.2, "risk_weight": 0.8, "name": "Risk-Focused"},
        {"risk_profile": "Balanced", "return_weight": 0.5, "risk_weight": 0.5, "name": "Academic"},
        {"risk_profile": "Balanced", "return_weight": 0.8, "risk_weight": 0.2, "name": "Growth-Focused"},
        
        # Aggressive profiles
        {"risk_profile": "Aggressive", "return_weight": 0.2, "risk_weight": 0.8, "name": "Risk-Focused"},
        {"risk_profile": "Aggressive", "return_weight": 0.5, "risk_weight": 0.5, "name": "Academic"},
        {"risk_profile": "Aggressive", "return_weight": 0.8, "risk_weight": 0.2, "name": "Growth-Focused"},
    ]
    
    # **KEY FIX: Clear all existing models first to prevent transfer learning**
    print("\n🗑️ Clearing existing models to force fresh training...")
    existing_models = list(config.OUTPUT_DIR.glob("*.pth"))
    for model_file in existing_models:
        model_file.unlink()
        print(f"   Deleted: {model_file.name}")
    
    # Train each model with fresh manager instances
    for i, model_config in enumerate(models_to_train, 1):
        print(f"\n🤖 Training Model {i}/{len(models_to_train)}")
        print(f"   Risk Profile: {model_config['risk_profile']}")
        print(f"   Objective: {model_config['name']}")
        print(f"   Return Weight: {model_config['return_weight']:.0%}")
        print(f"   Risk Weight: {model_config['risk_weight']:.0%}")
        
        try:
            # **KEY FIX: Create a fresh manager for each model to avoid caching**
            manager = RLAgentManager(config.OUTPUT_DIR)
            
            # Generate expected model path
            model_path = get_model_path(
                config.OUTPUT_DIR,
                model_config['risk_profile'], 
                selected_assets, 
                model_config['return_weight'], 
                model_config['risk_weight']
            )
            
            # Train the agent - since we cleared all models, this will always be new
            agent, is_new = manager.get_or_create_agent(
                risk_profile=model_config['risk_profile'],
                selected_assets=selected_assets,
                market_data=market_data,
                return_weight=model_config['return_weight'],
                risk_weight=model_config['risk_weight'],
                market_regime="🔵 Moderate Volatility/Stable"
            )
            
            # Verify the model file was actually created
            if model_path.exists():
                print(f"   ✅ Successfully created: {model_path.name}")
            else:
                print(f"   ⚠️ Model file not found: {model_path.name}")
                
        except Exception as e:
            logger.error(f"Failed to train {model_config['risk_profile']} {model_config['name']}: {e}")
            print(f"   ❌ Training failed: {e}")
    
    # Final verification
    final_models = list(config.OUTPUT_DIR.glob("*.pth"))
    print(f"\n🎉 Training complete! Created {len(final_models)} models:")
    for model_file in sorted(final_models):
        print(f"   📄 {model_file.name}")
    
    if len(final_models) == 9:
        print("✅ SUCCESS: All 9 objective-specific models created!")
    else:
        print(f"⚠️ WARNING: Expected 9 models, got {len(final_models)}")
    
    print(f"\n📁 Upload these .pth files to your Streamlit Cloud repository")


if __name__ == "__main__":
    print("🚀 Starting RL model training for all investment objectives...")
    train_all_objective_models()