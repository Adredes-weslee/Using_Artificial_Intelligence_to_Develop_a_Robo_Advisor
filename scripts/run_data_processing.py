"""Executes the full data processing pipeline for both market and survey data."""
import pandas as pd
import sys
from pathlib import Path

# Add project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import src.config as config
from src.data_processing.market_data import get_sp500_tickers, fetch_sp500_data, clean_sp500_data
from src.data_processing.survey_data import (
    load_scf_data, select_working_features, calculate_risk_tolerance, 
    clean_and_prepare_features, get_data_summary
)

def main():
    """Main function to run all data processing."""
    print("--- Starting Full Data Processing Pipeline ---")
    
    # Ensure output directories exist
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Process S&P 500 Market Data
    print("\n" + "="*60)
    print("Step 1: Processing S&P 500 Market Data")
    print("="*60)
    
    try:
        # Get S&P 500 tickers from Wikipedia
        print("Fetching S&P 500 tickers from Wikipedia...")
        tickers = get_sp500_tickers()
        print(f"Found {len(tickers)} S&P 500 tickers")
        
        # Fetch stock data using yfinance
        print("Fetching stock price data from Yahoo Finance...")
        market_df_raw = fetch_sp500_data(tickers, "2000-01-01", "2023-12-31")
        
        if not market_df_raw.empty:
            print(f"Raw market data shape: {market_df_raw.shape}")
            
            # Clean and process the data
            print("Cleaning and processing market data...")
            market_df_processed = clean_sp500_data(market_df_raw, start_from_date='2010-01-03')
            
            # Save processed data
            market_output_path = config.PROCESSED_DATA_DIR / config.PROCESSED_SP500_FILE
            market_df_processed.to_csv(market_output_path)
            print(f"✓ Processed market data saved to {market_output_path}")
            print(f"  Final shape: {market_df_processed.shape}")
            print(f"  Date range: {market_df_processed.index.min()} to {market_df_processed.index.max()}")
            
        else:
            print("❌ Failed to fetch market data - skipping market data processing")
            
    except Exception as e:
        print(f"❌ Error processing market data: {e}")

    # 2. Process Survey of Consumer Finances (SCF) Data
    print("\n" + "="*60)
    print("Step 2: Processing Survey of Consumer Finances Data")
    print("="*60)
    
    try:
        # Load raw SCF data
        scf_raw_path = config.RAW_DATA_DIR / config.RAW_SCF_FILE
        print(f"Loading SCF data from {scf_raw_path}")
        
        if not scf_raw_path.exists():
            print(f"❌ SCF data file not found: {scf_raw_path}")
            print("Please ensure SCFP2019.csv is in the data/raw directory")
            return
            
        scf_df_raw = load_scf_data(scf_raw_path)
        print(f"Raw SCF data shape: {scf_df_raw.shape}")
        
        # Select working features
        print("Selecting working features...")
        working_df = select_working_features(scf_df_raw)
        print(f"Working features shape: {working_df.shape}")
        
        # Calculate risk tolerance
        print("Calculating risk tolerance...")
        working_df = calculate_risk_tolerance(working_df)
        
        # Clean and prepare final features
        print("Cleaning and preparing final features...")
        scf_df_processed = clean_and_prepare_features(working_df)
        
        # Get data summary
        summary = get_data_summary(scf_df_processed)
        print("\nData Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Save processed data
        scf_output_path = config.PROCESSED_DATA_DIR / config.PROCESSED_SCF_FILE
        scf_df_processed.to_csv(scf_output_path, index=False)
        print(f"✓ Processed SCF data saved to {scf_output_path}")
        print(f"  Final shape: {scf_df_processed.shape}")
        
    except Exception as e:
        print(f"❌ Error processing SCF data: {e}")

    print("\n" + "="*60)
    print("Data Processing Pipeline Completed!")
    print("="*60)

if __name__ == "__main__":
    main()