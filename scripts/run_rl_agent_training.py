"""Executes the training pipeline for the RL portfolio agent with multi-profile support."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # ← ADD THIS LINE
import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import src.config as config
from src.models.rl_agent import (
    train_rl_agent, evaluate_rl_agent, plot_training_results, 
    plot_evaluation_results, calculate_performance_stats
)
from src.models.rl_agent_manager import RLAgentManager, get_recommended_assets_for_profile
from src.utils.portfolio_math import mean_variance_optimization

def main():
    """Main function to train and evaluate RL agents for all risk profiles."""
    print("--- Starting Multi-Profile RL Agent Training Pipeline ---")
    
    # Ensure output directory exists
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load processed market data
    market_data_path = config.PROCESSED_DATA_DIR / config.PROCESSED_SP500_FILE
    print(f"\nLoading processed market data from {market_data_path}")
    
    if not market_data_path.exists():
        print(f"❌ Processed market data not found: {market_data_path}")
        print("Please run the data processing pipeline first.")
        return
    
    try:
        # Load market data
        df = pd.read_csv(market_data_path, index_col=0, parse_dates=True)
        print(f"✓ Loaded market data with shape: {df.shape}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Assets: {df.shape[1]} stocks")
        
        # Initialize RL Agent Manager
        manager = RLAgentManager(config.OUTPUT_DIR)
        
        # Train agents for each risk profile
        risk_profiles = ['Conservative', 'Balanced', 'Aggressive']
        results_summary = {}
        
        # Around line 40-90, replace this entire section:

        for risk_profile in risk_profiles:
            print(f"\n" + "="*80)
            print(f"Training {risk_profile} RL Agent")
            print("="*80)
            
            # Get recommended assets for this risk profile
            recommended_assets = get_recommended_assets_for_profile(risk_profile, max_assets=15)
            available_assets = [asset for asset in recommended_assets if asset in df.columns]
            
            # FIX 2: REMOVE DUPLICATES
            available_assets = list(dict.fromkeys(available_assets))
            
            if len(available_assets) < 5:
                print(f"❌ Insufficient assets for {risk_profile} profile: {len(available_assets)}")
                # Fallback to default portfolio assets  
                fallback_assets = [asset for asset in config.DEFAULT_PORTFOLIO_ASSETS[:15] if asset in df.columns]
                available_assets = list(dict.fromkeys(fallback_assets))  # Remove duplicates
            
            print(f"Selected assets for {risk_profile} profile ({len(available_assets)}):")
            for i, asset in enumerate(available_assets, 1):
                print(f"  {i:2d}. {asset}")
            
            # Prepare data
            selected_data = df[available_assets].copy().dropna()
            print(f"Training data shape: {selected_data.shape}")
            
            if len(selected_data) < 500:
                print(f"❌ Insufficient data points: {len(selected_data)}")
                continue
            
            # FIX 3: CALCULATE MPT BENCHMARK WITH CORRECT SIGNATURE
            try:
                benchmark_risk_tolerance = {'Conservative': 0.3, 'Balanced': 0.5, 'Aggressive': 0.8}[risk_profile]
                
                print(f"Performing mean variance optimization for risk tolerance: {benchmark_risk_tolerance}")
                print(f"Assets: {available_assets}")
                
                # Call MPT with the EXACT signature from portfolio_math.py
                allocation, portfolio_performance = mean_variance_optimization(
                    risk_tolerance=benchmark_risk_tolerance,  # First parameter: float
                    tickers=available_assets,                 # Second parameter: List[str]  
                    price_data=selected_data                  # Third parameter: pd.DataFrame
                )
                
                # Extract weights from the allocation DataFrame
                mpt_weights_array = allocation['Weight'].values
                    
                print(f"Optimization successful. Weights sum: {mpt_weights_array.sum():.6f}")
                print(f"✓ MPT benchmark calculated for risk tolerance {benchmark_risk_tolerance}")
                
            except Exception as e:
                print(f"⚠️  MPT calculation failed: {e}, using equal weights")
                mpt_weights_array = np.ones(len(available_assets)) / len(available_assets)
            
            # Get or create agent using manager
            print(f"\nRequesting {risk_profile} agent from manager...")
            agent, is_new = manager.get_or_create_agent(
                risk_profile=risk_profile,
                selected_assets=available_assets,
                market_data=selected_data
            )
            
            # FIX 1: EVALUATE THE AGENT WITH CORRECT MODEL NAME AND PATH
            print(f"\nEvaluating {risk_profile} agent...")

            # Use the actual saved model filename format with FULL PATH
            asset_string = "_".join(sorted(available_assets))
            model_name = f"{risk_profile}_{asset_string}"  # Just the filename
            model_path = config.OUTPUT_DIR / f"{model_name}.pth"  # Full path with .pth extension

            print(f"Looking for model at: {model_path}")

            # Check if model file exists
            if not model_path.exists():
                print(f"❌ Model file not found: {model_path}")
                print(f"Available files in {config.OUTPUT_DIR}:")
                for file in config.OUTPUT_DIR.glob("*.pth"):
                    print(f"  - {file.name}")
                results_summary[risk_profile] = {'error': f'Model file not found: {model_path}'}
                continue

            try:
                evaluation_results, eval_agent = evaluate_rl_agent(
                    selected_data=selected_data,
                    model_name=str(model_path),  # Pass full path as string
                    mpt_weights=mpt_weights_array,
                    window_size=50,
                    rebalance_period=config.RL_MODEL_CONFIGS[risk_profile]['rebalance_frequency']
                )
                # Calculate performance statistics
                rl_stats = calculate_performance_stats(evaluation_results['returns_rl'])
                mpt_stats = calculate_performance_stats(evaluation_results['returns_mpt'])
                
                results_summary[risk_profile] = {
                    'rl_stats': rl_stats,
                    'mpt_stats': mpt_stats,
                    'assets_count': len(available_assets),
                    'is_new_model': is_new
                }
                
                print(f"\n{risk_profile} Performance:")
                print(f"  RL Sharpe Ratio: {rl_stats['sharpe_ratio']:.4f}")
                print(f"  MPT Sharpe Ratio: {mpt_stats['sharpe_ratio']:.4f}")
                print(f"  Performance: {'🏆 RL Wins' if rl_stats['sharpe_ratio'] > mpt_stats['sharpe_ratio'] else '📊 MPT Wins'}")
                
                # Save evaluation plots
                try:
                    plot_path = config.OUTPUT_DIR / f"evaluation_{risk_profile.lower()}.png"
                    plot_evaluation_results(evaluation_results, save_path=plot_path)
                    print(f"✓ Evaluation plot saved: {plot_path}")
                except Exception as e:
                    print(f"⚠️  Could not save evaluation plot: {e}")
                    
            except Exception as e:
                print(f"❌ Evaluation failed for {risk_profile}: {e}")
                results_summary[risk_profile] = {'error': str(e)}
        
        # Print comprehensive summary
        print(f"\n" + "="*80)
        print("TRAINING PIPELINE SUMMARY")
        print("="*80)
        
        cache_info = manager.get_cache_info()
        print(f"Agent Cache Status:")
        print(f"  Total agents: {cache_info['total_agents']}")
        print(f"  Conservative: {cache_info['conservative_agents']}")
        print(f"  Balanced: {cache_info['balanced_agents']}")
        print(f"  Aggressive: {cache_info['aggressive_agents']}")
        
        print(f"\nPerformance Summary:")
        print(f"{'Profile':<12} {'Assets':<7} {'RL Sharpe':<10} {'MPT Sharpe':<11} {'Winner':<8} {'Status'}")
        print("-" * 70)
        
        for profile, data in results_summary.items():
            if 'error' not in data:
                rl_sharpe = data['rl_stats']['sharpe_ratio']
                mpt_sharpe = data['mpt_stats']['sharpe_ratio']
                winner = 'RL' if rl_sharpe > mpt_sharpe else 'MPT'
                status = 'New' if data['is_new_model'] else 'Cached'
                assets = data['assets_count']
                
                print(f"{profile:<12} {assets:<7} {rl_sharpe:<10.4f} {mpt_sharpe:<11.4f} {winner:<8} {status}")
            else:
                print(f"{profile:<12} {'Error':<7} {data['error'][:40]}")
        
        print(f"\n✓ Multi-profile RL training pipeline completed!")
        print(f"✓ Models saved in: {config.OUTPUT_DIR}")
        
    except Exception as e:
        print(f"❌ Error during RL agent training: {e}")
        raise

    print("\n--- Multi-Profile RL Agent Training Pipeline Completed! ---")

if __name__ == "__main__":
    main()