"""Contains functions for training and evaluating the risk tolerance prediction model.

This module implements TabPFN foundation model for risk tolerance prediction,
providing state-of-the-art performance on tabular data with GPU acceleration.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor  # Fallback

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
    print("✅ TabPFN available - using foundation model")
except ImportError:
    TABPFN_AVAILABLE = False
    print("⚠️  TabPFN not available - falling back to Extra Trees")

from ..config import RANDOM_STATE, TEST_SIZE, TARGET_COLUMN


def get_best_model_for_system() -> str:
    """Determine the best model based on system capabilities."""
    if not TABPFN_AVAILABLE:
        return "extra_trees"
    
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU detected: {gpu_memory:.1f}GB VRAM")
        return "tabpfn_gpu"
    else:
        print("🖥️  CPU mode - TabPFN will work but slower")
        return "tabpfn_cpu"


# Update the get_tabpfn_model function (around line 50):
def get_tabpfn_model() -> Any:
    """Get TabPFN model with proper configuration for large datasets."""
    if not TABPFN_AVAILABLE:
        return None
    
    try:
        from tabpfn import TabPFNRegressor
        
        # Configure TabPFN with large dataset support (minimal parameters)
        model = TabPFNRegressor(
            ignore_pretraining_limits=True,  # ← CRITICAL FIX!
            device='cpu' if not torch.cuda.is_available() else 'cuda'
        )
        
        return model
    except Exception as e:
        print(f"⚠️ Error creating TabPFN model: {e}")
        return None


# Update the train_tabpfn_model function (around line 70):
def train_tabpfn_model(X_train: pd.DataFrame, y_train: pd.Series, 
                      use_gpu: bool = True) -> TabPFNRegressor:
    """Train TabPFN foundation model for risk tolerance prediction.
    
    Args:
        X_train: Training features
        y_train: Training targets
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Trained TabPFN model
    """
    print("🧠 Training TabPFN foundation model...")
    
    # Device selection
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"📱 Using device: {device}")
    
    # Initialize TabPFN regressor with large dataset support (minimal parameters)
    model = TabPFNRegressor(
        device=device,
        ignore_pretraining_limits=True  # ← CRITICAL FIX!
    )
    
    # Train the model
    print("🔄 Training in progress...")
    model.fit(X_train, y_train)
    print("✅ TabPFN training completed!")
    
    return model


def train_extra_trees_fallback(X_train: pd.DataFrame, y_train: pd.Series) -> ExtraTreesRegressor:
    """Fallback to Extra Trees if TabPFN unavailable."""
    from ..config import N_ESTIMATORS_ETR, MAX_DEPTH_ETR, CRITERION_ETR
    
    print("🌲 Training Extra Trees fallback model...")
    
    model = ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS_ETR,
        criterion=CRITERION_ETR,
        max_depth=MAX_DEPTH_ETR,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("✅ Extra Trees training completed")
    
    return model


def train_risk_tolerance_model(X: pd.DataFrame, y: pd.Series, 
                             test_size: float = TEST_SIZE,
                             random_state: int = RANDOM_STATE) -> Tuple[Any, Dict[str, float]]:
    """Train the best available model for risk tolerance prediction.
    
    Args:
        X: Feature matrix
        y: Target vector  
        test_size: Test split ratio
        random_state: Random seed
        
    Returns:
        Tuple of (trained_model, performance_metrics)
    """
    print("🎯 Starting risk tolerance model training...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"📊 Dataset split: {len(X_train)} train, {len(X_test)} test samples")
    
    # Determine best model
    model_type = get_best_model_for_system()
    
    # Train appropriate model
    if model_type == "tabpfn_gpu":
        model = train_tabpfn_model(X_train, y_train, use_gpu=True)
        model_name = "TabPFN (GPU)"
    elif model_type == "tabpfn_cpu":
        model = train_tabpfn_model(X_train, y_train, use_gpu=False)
        model_name = "TabPFN (CPU)"
    else:
        model = train_extra_trees_fallback(X_train, y_train)
        model_name = "Extra Trees"
    
    # Evaluate model
    print(f"📈 Evaluating {model_name} performance...")
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'model_type': model_name,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': mean_squared_error(y_train, y_pred_train, squared=False),
        'test_rmse': mean_squared_error(y_test, y_pred_test, squared=False)
    }
    
    # Print results
    print(f"\n📊 {model_name} Performance:")
    print(f"   Training R²:  {metrics['train_r2']:.5f}")
    print(f"   Test R²:      {metrics['test_r2']:.5f}")
    print(f"   Training RMSE: {metrics['train_rmse']:.5f}")
    print(f"   Test RMSE:     {metrics['test_rmse']:.5f}")
    
    return model, metrics


def cross_validate_model(X: pd.DataFrame, y: pd.Series, 
                        cv_folds: int = 5) -> Tuple[List[float], float, float]:
    """Perform cross-validation with intelligent model selection."""
    print(f"🔄 Performing {cv_folds}-fold cross-validation...")
    
    # Check dataset size and choose appropriate strategy
    n_samples = len(X)
    print(f"📊 Dataset size: {n_samples:,} samples")
    
    if n_samples > 10000:
        print("📊 Large dataset detected - using TabPFN with override limits")
    
    # Determine model type
    model_type = get_best_model_for_system()
    
    # Initialize model with large dataset support
    if model_type.startswith("tabpfn") and TABPFN_AVAILABLE:
        # For TabPFN with large datasets, use smaller CV folds or subsampling
        if n_samples > 20000:
            print("⚡ Large dataset: Using stratified subsampling for faster CV")
            # Option 1: Use subset for CV (recommended for speed)
            subset_size = min(15000, n_samples)
            indices = np.random.choice(n_samples, size=subset_size, replace=False)
            X_cv, y_cv = X.iloc[indices], y.iloc[indices]
            print(f"📊 Using {subset_size:,} samples for cross-validation")
        else:
            X_cv, y_cv = X, y
            
        # Use smaller k-fold for large datasets
        cv_folds = min(cv_folds, 3) if n_samples > 20000 else cv_folds
        print(f"🔄 Using {cv_folds}-fold cross-validation")
        
        # Get TabPFN model with large dataset support
        model = get_tabpfn_model()
        model_name = f"TabPFN (CPU)"
        
        if torch.cuda.is_available():
            model_name = f"TabPFN (GPU)"
    else:
        X_cv, y_cv = X, y
        from ..config import N_ESTIMATORS_ETR, MAX_DEPTH_ETR, CRITERION_ETR
        model = ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS_ETR,
            max_depth=MAX_DEPTH_ETR,
            criterion=CRITERION_ETR,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model_name = "Extra Trees"
    
    # Perform cross-validation
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    try:
        cv_scores = -cross_val_score(model, X_cv, y_cv, cv=kfold, 
                                   scoring='neg_mean_squared_error', n_jobs=1)
        
        # Convert to RMSE
        cv_rmse_scores = np.sqrt(cv_scores)
        cv_mean = cv_rmse_scores.mean()
        cv_std = cv_rmse_scores.std()
        
        print(f"📊 Cross-validation completed:")
        print(f"   CV RMSE: {cv_mean:.5f} (+/- {cv_std*2:.5f})")
        print(f"   Individual folds: {[f'{score:.5f}' for score in cv_rmse_scores]}")
        
        return cv_rmse_scores.tolist(), cv_mean, cv_std
        
    except Exception as e:
        print(f"❌ Cross-validation failed: {e}")
        print("🔄 Falling back to Extra Trees for cross-validation...")
        
        # Fallback to Extra Trees
        from ..config import N_ESTIMATORS_ETR, MAX_DEPTH_ETR, CRITERION_ETR
        et_model = ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS_ETR,
            max_depth=MAX_DEPTH_ETR,
            criterion=CRITERION_ETR,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        cv_scores = -cross_val_score(et_model, X, y, cv=kfold, 
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        cv_rmse_scores = np.sqrt(cv_scores)
        cv_mean = cv_rmse_scores.mean()
        cv_std = cv_rmse_scores.std()
        
        print(f"📊 Extra Trees CV RMSE: {cv_mean:.5f} (+/- {cv_std*2:.5f})")
        
        return cv_rmse_scores.tolist(), cv_mean, cv_std


def plot_feature_importance(model: Any, feature_names: List[str], 
                          save_path: Optional[Path] = None) -> None:
    """Plot feature importance (when available).
    
    Args:
        model: Trained model
        feature_names: List of feature names
        save_path: Optional path to save plot
    """
    if hasattr(model, 'feature_importances_'):
        # Extra Trees has feature importance
        importance_scores = pd.Series(model.feature_importances_, index=feature_names)
        importance_scores = importance_scores.sort_values(ascending=False)
        
        plt.figure(figsize=(10, 8))
        importance_scores.head(10).plot(kind='barh')
        plt.title('Feature Importance - Extra Trees Model')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Feature importance plot saved to {save_path}")
        
        plt.show()
    else:
        print("ℹ️  Feature importance not available for TabPFN (foundation model)")
        print("   TabPFN automatically learns optimal feature relationships")


def save_compressed_model(model: Any, filepath: Path) -> None:
    """Save model with compression."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Add model type info for loading
    model_info = {
        'model': model,
        'model_type': type(model).__name__,
        'tabpfn_available': TABPFN_AVAILABLE
    }
    
    joblib.dump(model_info, filepath)
    print(f"💾 Model saved to {filepath}")


def load_compressed_model(filepath: Path) -> Any:
    """Load model with type checking."""
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model_info = joblib.load(filepath)
    
    # Handle backward compatibility
    if isinstance(model_info, dict) and 'model' in model_info:
        model = model_info['model']
        model_type = model_info.get('model_type', 'Unknown')
        print(f"📂 Loaded {model_type} model from {filepath}")
    else:
        # Old format - just the model
        model = model_info
        print(f"📂 Loaded model from {filepath}")
    
    return model


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    
    return {
        'r2_score': r2_score(y_test, y_pred),
        'rmse': mean_squared_error(y_test, y_pred, squared=False),
        'mae': np.mean(np.abs(y_test - y_pred))
    }


# Backward compatibility functions
def train_extra_trees_model(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> Any:
    """Backward compatibility wrapper."""
    print("⚠️  Using deprecated function. Consider train_risk_tolerance_model()")
    return train_extra_trees_fallback(X_train, y_train)