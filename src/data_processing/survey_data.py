"""Handles cleaning and processing of the Survey of Consumer Finances data.

This module processes the SCF data to create the risk tolerance dataset exactly
as implemented in the original notebook, including proper feature engineering
and risk tolerance calculation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
from ..config import (
    SELECTED_SCF_FEATURES, RISK_FREE_ASSETS, RISKY_ASSETS, 
    FINAL_FEATURES, TARGET_COLUMN
)


def load_scf_data(filepath: Path) -> pd.DataFrame:
    """Loads the raw SCF dataset from CSV file.
    
    Args:
        filepath (Path): Path to the SCFP2019.csv file.
        
    Returns:
        pd.DataFrame: Raw SCF dataset.
        
    Raises:
        FileNotFoundError: If the SCF file is not found.
        Exception: If there's an error reading the file.
    """
    try:
        print(f"Loading SCF data from {filepath}...")
        dataset = pd.read_csv(filepath)
        print(f"Successfully loaded SCF data. Shape: {dataset.shape}")
        return dataset
    except FileNotFoundError:
        print(f"Error: SCF file not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error loading SCF data: {e}")
        raise


def select_working_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Selects the working features from the full SCF dataset.
    
    This function replicates the exact feature selection from the notebook.
    
    Args:
        dataset (pd.DataFrame): The full SCF dataset.
        
    Returns:
        pd.DataFrame: DataFrame with selected features only.
    """
    print("Selecting working features from SCF dataset...")
    
    # Check if all required features exist in the dataset
    missing_features = [feat for feat in SELECTED_SCF_FEATURES if feat not in dataset.columns]
    if missing_features:
        print(f"Warning: Missing features in dataset: {missing_features}")
    
    # Select available features
    available_features = [feat for feat in SELECTED_SCF_FEATURES if feat in dataset.columns]
    working_df = dataset[available_features].copy()
    
    print(f"Selected {len(available_features)} features for analysis")
    return working_df


def calculate_risk_tolerance(working_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates risk tolerance based on risky vs risk-free assets.
    
    This function implements the exact risk tolerance calculation from the notebook:
    Risk_tolerance = Risky_assets / (Risky_assets + Risk_free_assets)
    
    Args:
        working_df (pd.DataFrame): DataFrame with asset value columns.
        
    Returns:
        pd.DataFrame: DataFrame with calculated risk tolerance.
    """
    print("Calculating risk tolerance from asset allocations...")
    
    # Calculate Risk-Free Assets
    # These are considered safer investments
    risk_free_cols = [col for col in RISK_FREE_ASSETS if col in working_df.columns]
    print(f"Risk-free asset columns found: {risk_free_cols}")
    
    working_df['RiskFree'] = working_df[risk_free_cols].sum(axis=1)
    
    # Calculate Risky Assets  
    # These are considered higher-risk investments
    risky_cols = [col for col in RISKY_ASSETS if col in working_df.columns]
    print(f"Risky asset columns found: {risky_cols}")
    
    working_df['Risky'] = working_df[risky_cols].sum(axis=1)
    
    # Calculate Risk Tolerance as ratio
    # Risk_tolerance = Risky / (Risky + RiskFree)
    total_assets = working_df['Risky'] + working_df['RiskFree']
    
    # Handle division by zero cases
    working_df[TARGET_COLUMN] = np.where(
        total_assets > 0,
        working_df['Risky'] / total_assets,
        0.0  # If no assets, assume zero risk tolerance
    )
    
    print(f"Risk tolerance calculated. Range: {working_df[TARGET_COLUMN].min():.4f} to {working_df[TARGET_COLUMN].max():.4f}")
    return working_df


def clean_and_prepare_features(working_df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the data and prepares final features for modeling.
    
    This function replicates the exact data cleaning steps from the notebook
    and includes risk tolerance validation to ensure values are between 0 and 1.
    
    Args:
        working_df (pd.DataFrame): DataFrame with calculated risk tolerance.
        
    Returns:
        pd.DataFrame: Clean DataFrame ready for model training.
    """
    print("Cleaning and preparing features for modeling...")
    
    # Check for missing values before cleaning
    missing_before = working_df.isnull().sum().sum()
    print(f"Missing values before cleaning: {missing_before}")
    
    # Drop rows with missing values (as done in notebook)
    working_df_clean = working_df.dropna().copy()
    
    missing_after = working_df_clean.isnull().sum().sum()
    print(f"Missing values after cleaning: {missing_after}")
    print(f"Rows dropped: {len(working_df) - len(working_df_clean)}")
    
    # Validate risk tolerance values to ensure they are between 0 and 1
    working_df_clean = validate_risk_tolerance(working_df_clean)
    
    # Drop asset value columns and intermediate calculation columns
    # Keep only demographic, financial, and behavioral features + target
    columns_to_drop = (
        RISK_FREE_ASSETS + RISKY_ASSETS + ['RiskFree', 'Risky']
    )
    
    # Only drop columns that actually exist
    columns_to_drop = [col for col in columns_to_drop if col in working_df_clean.columns]
    working_df_clean.drop(columns=columns_to_drop, inplace=True)
    
    print(f"Dropped {len(columns_to_drop)} asset value columns")
    
    # Verify we have the expected final features and select only those
    expected_features = FINAL_FEATURES + [TARGET_COLUMN]
    actual_features = working_df_clean.columns.tolist()
    
    missing_final_features = [feat for feat in expected_features if feat not in actual_features]
    if missing_final_features:
        print(f"Warning: Missing expected features: {missing_final_features}")
    
    # Keep only the final features we need (those that exist)
    available_final_features = [feat for feat in expected_features if feat in actual_features]
    working_df_clean = working_df_clean[available_final_features].copy()
    
    print(f"Final dataset shape: {working_df_clean.shape}")
    print(f"Final features: {working_df_clean.columns.tolist()}")
    
    return working_df_clean

def get_data_summary(df: pd.DataFrame) -> dict:
    """Generates a summary of the processed SCF data.
    
    Args:
        df (pd.DataFrame): Processed SCF DataFrame.
        
    Returns:
        dict: Summary statistics and information.
    """
    summary = {
        'total_records': len(df),
        'features_count': len(df.columns) - 1,  # Exclude target column
        'target_column': TARGET_COLUMN,
        'risk_tolerance_stats': {
            'mean': df[TARGET_COLUMN].mean(),
            'std': df[TARGET_COLUMN].std(),
            'min': df[TARGET_COLUMN].min(),
            'max': df[TARGET_COLUMN].max(),
            'median': df[TARGET_COLUMN].median()
        },
        'zero_risk_tolerance_pct': (df[TARGET_COLUMN] == 0).mean() * 100,
        'low_risk_tolerance_pct': (df[TARGET_COLUMN] < 0.1).mean() * 100
    }
    
    return summary


def process_scf_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Main function to process the raw SCF data into model-ready format.
    
    This function orchestrates the complete SCF data processing pipeline
    exactly as implemented in the original notebook.
    
    Args:
        raw_df (pd.DataFrame): The raw pandas DataFrame from SCFP2019.csv.
        
    Returns:
        Tuple[pd.DataFrame, dict]: Processed DataFrame and summary statistics.
    """
    print("Starting complete SCF data processing pipeline...")
    print("=" * 60)
    
    # Step 1: Select working features
    working_df = select_working_features(raw_df)
    
    # Step 2: Calculate risk tolerance
    working_df = calculate_risk_tolerance(working_df)
    
    # Step 3: Clean and prepare final features
    final_df = clean_and_prepare_features(working_df)
    
    # Step 4: Generate summary
    summary = get_data_summary(final_df)
    
    print("=" * 60)
    print("SCF data processing completed successfully!")
    print(f"Final dataset: {summary['total_records']} records, {summary['features_count']} features")
    print(f"Risk tolerance - Mean: {summary['risk_tolerance_stats']['mean']:.4f}, "
          f"Std: {summary['risk_tolerance_stats']['std']:.4f}")
    print(f"Zero risk tolerance: {summary['zero_risk_tolerance_pct']:.1f}% of records")
    
    return final_df, summary


def save_processed_data(df: pd.DataFrame, filepath: Path) -> None:
    """Saves the processed SCF data to CSV file.
    
    Args:
        df (pd.DataFrame): Processed SCF DataFrame.
        filepath (Path): Path where to save the processed data.
    """
    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"Processed SCF data saved to {filepath}")
        
    except Exception as e:
        print(f"Error saving processed data: {e}")
        raise


def load_processed_data(filepath: Path) -> pd.DataFrame:
    """Loads previously processed SCF data.
    
    Args:
        filepath (Path): Path to the processed data file.
        
    Returns:
        pd.DataFrame: Processed SCF DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded processed SCF data from {filepath}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Processed data file not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error loading processed data: {e}")
        raise
    
    
def validate_risk_tolerance(df: pd.DataFrame) -> pd.DataFrame:
    """Validates and cleans risk tolerance values to ensure they are between 0 and 1.
    
    Args:
        df (pd.DataFrame): DataFrame with risk tolerance column.
        
    Returns:
        pd.DataFrame: DataFrame with validated risk tolerance values.
    """
    print(f"Risk tolerance validation - Original shape: {df.shape}")
    
    # Check for infinite values
    infinite_mask = np.isinf(df[TARGET_COLUMN])
    if infinite_mask.any():
        print(f"Found {infinite_mask.sum()} infinite risk tolerance values, removing...")
        df = df[~infinite_mask].copy()
    
    # Check for values outside [0, 1] range
    out_of_range_mask = (df[TARGET_COLUMN] < 0) | (df[TARGET_COLUMN] > 1)
    if out_of_range_mask.any():
        print(f"Found {out_of_range_mask.sum()} risk tolerance values outside [0, 1] range")
        print(f"Min: {df[TARGET_COLUMN].min():.6f}, Max: {df[TARGET_COLUMN].max():.6f}")
        
        # Clip values to [0, 1] range
        df[TARGET_COLUMN] = df[TARGET_COLUMN].clip(0, 1)
        print("Clipped values to [0, 1] range")
    
    # Remove any remaining NaN values after validation
    initial_count = len(df)
    df = df.dropna().copy()
    final_count = len(df)
    
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} rows with invalid data")
    
    print(f"Final risk tolerance stats:")
    print(f"  Count: {len(df)}")
    print(f"  Mean: {df[TARGET_COLUMN].mean():.4f}")
    print(f"  Std: {df[TARGET_COLUMN].std():.4f}")
    print(f"  Min: {df[TARGET_COLUMN].min():.4f}")
    print(f"  Max: {df[TARGET_COLUMN].max():.4f}")
    
    return df

