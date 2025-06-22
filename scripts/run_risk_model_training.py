"""Executes the training pipeline for the risk tolerance prediction model."""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import src.config as config
from src.models.risk_profiler import (
    train_risk_tolerance_model, cross_validate_model, save_compressed_model,
    load_compressed_model, evaluate_model, plot_feature_importance
)

def main():
    """Main function to train the risk tolerance prediction model."""
    print("--- Starting Advanced Risk Tolerance Model Training ---")
    
    # Ensure output directory exists
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load processed data
    processed_scf_path = config.PROCESSED_DATA_DIR / config.PROCESSED_SCF_FILE
    print(f"\n📂 Loading data from {processed_scf_path}")
    
    if not processed_scf_path.exists():
        print(f"❌ Data not found. Run data processing first.")
        return
    
    try:
        df = pd.read_csv(processed_scf_path)
        print(f"✅ Loaded dataset: {df.shape}")
        
        # Prepare features and target
        feature_columns = [col for col in df.columns if col != config.TARGET_COLUMN]
        X = df[feature_columns]
        y = df[config.TARGET_COLUMN]
        
        print(f"\n📊 Dataset overview:")
        print(f"   Features: {len(feature_columns)}")
        print(f"   Samples: {len(df)}")
        print(f"   Target range: {y.min():.3f} to {y.max():.3f}")
        
        # Cross-validation
        print("\n" + "="*60)
        print("Step 1: Cross-Validation Analysis")
        print("="*60)
        
        cv_scores, cv_mean, cv_std = cross_validate_model(X, y)
        
        # Train final model
        print("\n" + "="*60)
        print("Step 2: Training Final Model")
        print("="*60)
        
        model, metrics = train_risk_tolerance_model(X, y)
        
        # Feature importance (if available)
        print("\n" + "="*60)
        print("Step 3: Model Analysis")
        print("="*60)
        
        plot_save_path = config.OUTPUT_DIR / "feature_importance.png"
        plot_feature_importance(model, feature_columns, save_path=plot_save_path)
        
        # Save model
        print("\n" + "="*60)
        print("Step 4: Saving Model")
        print("="*60)
        
        model_path = config.OUTPUT_DIR / config.RISK_MODEL_FILE
        save_compressed_model(model, model_path)
        
        # Verify loading
        print("\n🔍 Verifying model loading...")
        loaded_model = load_compressed_model(model_path)
        test_pred = loaded_model.predict(X.iloc[:1])
        print(f"✅ Model verification passed")
        
        # Final summary
        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)
        print(f"🧠 Model: {metrics['model_type']}")
        print(f"📊 Performance:")
        print(f"   CV RMSE: {cv_mean:.5f} (+/- {cv_std*2:.5f})")
        print(f"   Test R²: {metrics['test_r2']:.5f}")
        print(f"   Test RMSE: {metrics['test_rmse']:.5f}")
        print(f"💾 Saved to: {model_path}")
        
        if "TabPFN" in metrics['model_type']:
            print(f"🚀 GPU Acceleration: {'✅' if 'GPU' in metrics['model_type'] else '❌'}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

    print("\n--- Risk Tolerance Model Training Completed! ---")

if __name__ == "__main__":
    main()