"""Contains functions for training and evaluating the risk tolerance prediction model.

This module implements the complete machine learning pipeline for risk tolerance prediction,
including model comparison, hyperparameter tuning, and evaluation, based on the original notebook.
"""
import pandas as pd
import numpy as np
import joblib
import bz2
import _pickle as cPickle
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_STATE, TEST_SIZE, N_ESTIMATORS_ETR, MAX_DEPTH_ETR, CRITERION_ETR, TARGET_COLUMN


def prepare_training_data(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepares the features and target for model training.
    
    Args:
        df (pd.DataFrame): The processed SCF DataFrame.
        target_col (str): Name of the target column.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y).
    """
    print("Preparing training data...")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE, 
               random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        test_size (float): Proportion of data for testing.
        random_state (int): Random state for reproducibility.
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    print(f"Splitting data with test_size={test_size}, random_state={random_state}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def train_extra_trees_model(X_train: pd.DataFrame, y_train: pd.Series, 
                           n_estimators: int = N_ESTIMATORS_ETR,
                           criterion: str = CRITERION_ETR,
                           max_depth: int = MAX_DEPTH_ETR,
                           random_state: int = RANDOM_STATE) -> ExtraTreesRegressor:
    """Trains an Extra Trees Regressor model with the best parameters from notebook.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        n_estimators (int): Number of trees.
        criterion (str): Splitting criterion.
        max_depth (int): Maximum depth of trees.
        random_state (int): Random state for reproducibility.
        
    Returns:
        ExtraTreesRegressor: Trained model.
    """
    print("Training Extra Trees Regressor model...")
    print(f"Parameters: n_estimators={n_estimators}, criterion={criterion}, max_depth={max_depth}")
    
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model training completed")
    
    return model


def evaluate_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluates the trained model on training and test sets.
    
    Args:
        model: Trained model.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        
    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics.
    """
    print("Evaluating model performance...")
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': mean_squared_error(y_train, y_pred_train, squared=False),
        'test_rmse': mean_squared_error(y_test, y_pred_test, squared=False)
    }
    
    # Print results
    print(f"Training R²: {metrics['train_r2']:.5f}")
    print(f"Test R²: {metrics['test_r2']:.5f}")
    print(f"Training RMSE: {metrics['train_rmse']:.5f}")
    print(f"Test RMSE: {metrics['test_rmse']:.5f}")
    
    return metrics


def cross_validate_model(model: Any, X: pd.DataFrame, y: pd.Series, 
                        cv_folds: int = 10, random_state: int = RANDOM_STATE) -> Dict[str, float]:
    """Performs cross-validation on the model.
    
    Args:
        model: Model to cross-validate.
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        cv_folds (int): Number of cross-validation folds.
        random_state (int): Random state for reproducibility.
        
    Returns:
        Dict[str, float]: Cross-validation results.
    """
    print(f"Performing {cv_folds}-fold cross-validation...")
    
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Cross-validate RMSE (convert to positive values)
    cv_rmse_scores = -1 * cross_val_score(model, X, y, cv=kfold, 
                                         scoring='neg_root_mean_squared_error')
    
    # Cross-validate R²
    cv_r2_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    
    results = {
        'cv_rmse_mean': cv_rmse_scores.mean(),
        'cv_rmse_std': cv_rmse_scores.std(),
        'cv_r2_mean': cv_r2_scores.mean(),
        'cv_r2_std': cv_r2_scores.std()
    }
    
    print(f"Cross-validation RMSE: {results['cv_rmse_mean']:.5f} (+/- {results['cv_rmse_std']:.5f})")
    print(f"Cross-validation R²: {results['cv_r2_mean']:.5f} (+/- {results['cv_r2_std']:.5f})")
    
    return results


def analyze_feature_importance(model: ExtraTreesRegressor, feature_names: List[str], 
                              top_n: int = 10, save_plot: Optional[Path] = None) -> pd.Series:
    """Analyzes and visualizes feature importance from the trained model.
    
    Args:
        model (ExtraTreesRegressor): Trained Extra Trees model.
        feature_names (List[str]): List of feature names.
        top_n (int): Number of top features to display.
        save_plot (Optional[Path]): Path to save the plot.
        
    Returns:
        pd.Series: Feature importance scores.
    """
    print("Analyzing feature importance...")
    
    # Get feature importance
    importance_scores = pd.Series(model.feature_importances_, index=feature_names)
    importance_scores = importance_scores.sort_values(ascending=False)
    
    # Display top features
    print(f"Top {top_n} most important features:")
    for i, (feature, score) in enumerate(importance_scores.head(top_n).items(), 1):
        print(f"{i:2d}. {feature}: {score:.4f}")
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    importance_scores.head(top_n).plot(kind='barh')
    plt.title(f'Top {top_n} Feature Importances - Extra Trees Regressor')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_plot}")
    
    plt.show()
    
    return importance_scores


def hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series, 
                         cv_folds: int = 5, random_state: int = RANDOM_STATE,
                         verbose: bool = True) -> Tuple[ExtraTreesRegressor, Dict[str, Any]]:
    """Performs hyperparameter tuning for Extra Trees Regressor.
    
    Note: This is a simplified version. The full grid search from the notebook
    would take too long for regular execution.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        cv_folds (int): Number of cross-validation folds.
        random_state (int): Random state for reproducibility.
        verbose (bool): Whether to print verbose output.
        
    Returns:
        Tuple[ExtraTreesRegressor, Dict]: Best model and grid search results.
    """
    print("Performing hyperparameter tuning...")
    
    # Define parameter grid (simplified version)
    param_grid = {
        "n_estimators": [25, 50, 100],
        "criterion": ['squared_error', 'friedman_mse'],
        "max_depth": [25, 50, None]
    }
    
    # Initialize model
    model = ExtraTreesRegressor(random_state=random_state, n_jobs=-1)
    
    # Setup cross-validation
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        scoring='neg_root_mean_squared_error',
        cv=kfold, 
        verbose=1 if verbose else 0,
        n_jobs=-1
    )
    
    grid_result = grid_search.fit(X_train, y_train)
    
    print(f"Best RMSE: {-grid_result.best_score_:.5f}")
    print(f"Best parameters: {grid_result.best_params_}")
    
    if verbose:
        print("\nAll results:")
        means = -grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        
        for mean, std, param in zip(means, stds, params):
            print(f"RMSE: {mean:.5f} (+/- {std:.5f}) with: {param}")
    
    return grid_result.best_estimator_, grid_result.cv_results_


def save_model(model: Any, filepath: Path, compress: bool = True) -> None:
    """Saves the trained model to disk.
    
    Args:
        model: The trained model object.
        filepath (Path): Path where to save the model.
        compress (bool): Whether to use bz2 compression.
    """
    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if compress:
            # Save with bz2 compression (as done in notebook)
            with bz2.BZ2File(str(filepath) + '.pbz2', 'w') as f:
                cPickle.dump(model, f)
            print(f"Compressed model saved to {filepath}.pbz2")
        else:
            # Save with joblib
            joblib.dump(model, filepath)
            print(f"Model saved to {filepath}")
            
    except Exception as e:
        print(f"Error saving model: {e}")
        raise


def load_model(filepath: Path, compressed: bool = True) -> Any:
    """Loads a saved model from disk.
    
    Args:
        filepath (Path): Path to the saved model.
        compressed (bool): Whether the model is bz2 compressed.
        
    Returns:
        The loaded model object.
    """
    try:
        if compressed:
            # Load compressed model
            if not str(filepath).endswith('.pbz2'):
                filepath = Path(str(filepath) + '.pbz2')
            
            with bz2.BZ2File(str(filepath), 'rb') as f:
                model = cPickle.load(f)
            print(f"Compressed model loaded from {filepath}")
        else:
            # Load with joblib
            model = joblib.load(filepath)
            print(f"Model loaded from {filepath}")
        
        return model
        
    except FileNotFoundError:
        print(f"Model file not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def predict_risk_tolerance(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Makes risk tolerance predictions using the trained model.
    
    Args:
        model: Trained model.
        features (pd.DataFrame): Input features for prediction.
        
    Returns:
        np.ndarray: Predicted risk tolerance values.
    """
    predictions = model.predict(features)
    return predictions


def train_complete_pipeline(df: pd.DataFrame, save_model_path: Optional[Path] = None,
                           perform_tuning: bool = False) -> Tuple[Any, Dict[str, Any]]:
    """Runs the complete risk tolerance model training pipeline.
    
    This function orchestrates the entire training process as done in the notebook.
    
    Args:
        df (pd.DataFrame): Processed SCF dataset.
        save_model_path (Optional[Path]): Path to save the trained model.
        perform_tuning (bool): Whether to perform hyperparameter tuning.
        
    Returns:
        Tuple[Any, Dict]: Trained model and evaluation results.
    """
    print("Starting complete risk tolerance model training pipeline...")
    print("=" * 70)
    
    # Step 1: Prepare data
    X, y = prepare_training_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 2: Train model
    if perform_tuning:
        print("Training with hyperparameter tuning...")
        model, _ = hyperparameter_tuning(X_train, y_train)
    else:
        print("Training with best known parameters...")
        model = train_extra_trees_model(X_train, y_train)
    
    # Step 3: Evaluate model
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    cv_results = cross_validate_model(model, X, y)
    
    # Step 4: Analyze feature importance
    feature_importance = analyze_feature_importance(model, list(X.columns))
    
    # Step 5: Save model if requested
    if save_model_path:
        save_model(model, save_model_path)
    
    # Combine results
    results = {
        'model': model,
        'metrics': metrics,
        'cv_results': cv_results,
        'feature_importance': feature_importance
    }
    
    print("=" * 70)
    print("Risk tolerance model training completed successfully!")
    
    return model, results