"""Executes the training pipeline for the risk tolerance prediction model."""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import src.config as config
from src.models.risk_profiler import (
    train_extra_trees_model, cross_validate_model, save_compressed_model,
    load_compressed_model, evaluate_model, plot_feature_importance
)

def main():
    """Main function to train the risk tolerance prediction model."""
    print("--- Starting Risk Tolerance Model Training Pipeline ---")
    
    # Ensure output directory exists
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load processed SCF data
    processed_scf_path = config.PROCESSED_DATA_DIR / config.PROCESSED_SCF_FILE
    print(f"\nLoading processed SCF data from {processed_scf_path}")
    
    if not processed_scf_path.exists():
        print(f"❌ Processed SCF data not found: {processed_scf_path}")
        print("Please run the data processing pipeline first.")
        return
    
    try:
        df = pd.read_csv(processed_scf_path)
        print(f"✓ Loaded dataset with shape: {df.shape}")
        
        # Verify we have the target column
        if config.TARGET_COLUMN not in df.columns:
            print(f"❌ Target column '{config.TARGET_COLUMN}' not found in dataset")
            print(f"Available columns: {df.columns.tolist()}")
            return
        
        # Display basic statistics
        print(f"\nDataset Overview:")
        print(f"  Shape: {df.shape}")
        print(f"  Target column: {config.TARGET_COLUMN}")
        print(f"  Feature columns: {len(df.columns) - 1}")
        print(f"  Risk tolerance range: {df[config.TARGET_COLUMN].min():.4f} to {df[config.TARGET_COLUMN].max():.4f}")
        print(f"  Risk tolerance mean: {df[config.TARGET_COLUMN].mean():.4f}")
        
        # Prepare features and target
        feature_columns = [col for col in df.columns if col != config.TARGET_COLUMN]
        X = df[feature_columns]
        y = df[config.TARGET_COLUMN]
        
        print(f"\nFeature columns ({len(feature_columns)}):")
        for i, col in enumerate(feature_columns):
            print(f"  {i+1:2d}. {col}")
        
        # Cross-validate model to get baseline performance
        print("\n" + "="*60)
        print("Step 1: Cross-Validation Analysis")
        print("="*60)
        
        cv_scores, cv_mean, cv_std = cross_validate_model(X, y)
        print(f"Cross-validation RMSE: {cv_mean:.5f} (+/- {cv_std*2:.5f})")
        
        # Train the final model
        print("\n" + "="*60)
        print("Step 2: Training Final Model")
        print("="*60)
        
        model, train_score, test_score, train_rmse, test_rmse = train_extra_trees_model(
            X, y,
            n_estimators=config.N_ESTIMATORS_ETR,
            max_depth=config.MAX_DEPTH_ETR,
            criterion=config.CRITERION_ETR,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE
        )
        
        print(f"\nModel Training Results:")
        print(f"  Train R²: {train_score:.5f}")
        print(f"  Test R²:  {test_score:.5f}")
        print(f"  Train RMSE: {train_rmse:.5f}")
        print(f"  Test RMSE:  {test_rmse:.5f}")
        
        # Feature importance analysis
        print("\n" + "="*60)
        print("Step 3: Feature Importance Analysis")
        print("="*60)
        
        feature_importance = model.feature_importances_
        feature_names = feature_columns
        
        # Sort features by importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("Feature Importance Rankings:")
        for i, row in importance_df.iterrows():
            print(f"  {row.name+1:2d}. {row['feature']:12s}: {row['importance']:.4f}")
        
        # Save the trained model
        print("\n" + "="*60)
        print("Step 4: Saving Model")
        print("="*60)
        
        model_path = config.OUTPUT_DIR / config.RISK_MODEL_FILE
        save_compressed_model(model, model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Test loading the saved model
        print("\nTesting model loading...")
        loaded_model = load_compressed_model(model_path)
        
        # Quick prediction test
        test_prediction = loaded_model.predict(X.iloc[:1])
        original_prediction = model.predict(X.iloc[:1])
        
        if np.allclose(test_prediction, original_prediction):
            print("✓ Model loading test passed")
        else:
            print("❌ Model loading test failed")
        
        # Save feature importance plot
        try:
            plot_path = config.OUTPUT_DIR / "feature_importance.png"
            plot_feature_importance(model, feature_names, save_path=plot_path)
            print(f"✓ Feature importance plot saved to {plot_path}")
        except Exception as e:
            print(f"⚠️  Could not save feature importance plot: {e}")
        
        # Model summary
        print("\n" + "="*60)
        print("Model Training Summary")
        print("="*60)
        print(f"Model Type: Extra Trees Regressor")
        print(f"Hyperparameters:")
        print(f"  n_estimators: {config.N_ESTIMATORS_ETR}")
        print(f"  max_depth: {config.MAX_DEPTH_ETR}")
        print(f"  criterion: {config.CRITERION_ETR}")
        print(f"Performance:")
        print(f"  CV RMSE: {cv_mean:.5f}")
        print(f"  Test R²: {test_score:.5f}")
        print(f"  Test RMSE: {test_rmse:.5f}")
        print(f"Model saved to: {model_path}")
        
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        raise

    print("\n--- Risk Tolerance Model Training Pipeline Completed! ---")

if __name__ == "__main__":
    main()