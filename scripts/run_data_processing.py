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

def process_existing_sp500_data():
    """Process existing S&P 500 CSV file with intelligent fallbacks."""
    try:
        # Check if we already have the file
        raw_file = config.RAW_DATA_DIR / 'S&P500.csv'
        
        if raw_file.exists():
            print(f"✅ Found existing S&P 500 data: {raw_file}")
            
            # Load existing data
            try:
                df = pd.read_csv(raw_file)
                print(f"📊 Loaded existing data: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Basic data processing (assuming it's already price data)
                # If your CSV has a date column, set it as index
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                elif df.index.name != 'Date':
                    # Try to convert index to datetime if it looks like dates
                    try:
                        df.index = pd.to_datetime(df.index)
                    except:
                        pass
                
                # Clean the data using existing function
                processed_data = clean_sp500_data(df, start_from_date='2010-01-03')
                
                # Save processed data
                output_file = config.PROCESSED_DATA_DIR / config.PROCESSED_SP500_FILE
                processed_data.to_csv(output_file)
                print(f"✅ Processed data saved to: {output_file}")
                print(f"  Final shape: {processed_data.shape}")
                print(f"  Columns: {list(processed_data.columns)[:10]}...")  # Show first 10 columns
                
                return processed_data
                
            except Exception as e:
                print(f"⚠️ Error processing existing file: {e}")
                print("🔄 Will try yfinance download as fallback...")
                return None
        else:
            print(f"📂 No existing S&P 500 file found at: {raw_file}")
            return None
        
    except Exception as e:
        print(f"❌ Error checking for existing data: {e}")
        return None

def download_sp500_with_yfinance():
    """Original yfinance download method as fallback."""
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
            
            return market_df_processed
        else:
            print("❌ Failed to fetch market data from yfinance")
            return None
            
    except Exception as e:
        print(f"❌ Error with yfinance download: {e}")
        return None

def main():
    """Main function to run all data processing with smart fallbacks."""
    print("--- Starting Enhanced Data Processing Pipeline ---")
    
    # Ensure output directories exist
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Process S&P 500 Market Data (with smart fallbacks)
    print("\n" + "="*60)
    print("Step 1: Processing S&P 500 Market Data")
    print("="*60)
    
    # Strategy 1: Try to use existing S&P 500 file
    market_data = process_existing_sp500_data()
    
    # Strategy 2: Fallback to yfinance if existing file failed
    if market_data is None:
        print("\n🔄 Falling back to yfinance download...")
        market_data = download_sp500_with_yfinance()
    
    # Strategy 3: Check if we have any processed data already
    if market_data is None:
        processed_file = config.PROCESSED_DATA_DIR / config.PROCESSED_SP500_FILE
        if processed_file.exists():
            print(f"\n✅ Using existing processed data: {processed_file}")
            try:
                market_data = pd.read_csv(processed_file, index_col=0, parse_dates=True)
                print(f"📊 Loaded processed data: {market_data.shape}")
            except Exception as e:
                print(f"⚠️ Error loading processed file: {e}")
    
    # Final status
    if market_data is not None:
        print("✅ Market data processing completed successfully!")
    else:
        print("⚠️ Market data processing failed - continuing with SCF data only")
        print("💡 The system can still function with just the risk model")

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
            print("Download from: https://www.federalreserve.gov/econres/scfindex.htm")
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
        
        print("✅ SCF data processing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error processing SCF data: {e}")
        print("🐛 Debug info:")
        import traceback
        traceback.print_exc()

    # 3. Final Summary
    print("\n" + "="*60)
    print("Data Processing Pipeline Summary")
    print("="*60)
    
    # Check what we have
    scf_processed = config.PROCESSED_DATA_DIR / config.PROCESSED_SCF_FILE
    sp500_processed = config.PROCESSED_DATA_DIR / config.PROCESSED_SP500_FILE
    
    print("📊 Data Processing Results:")
    if scf_processed.exists():
        print("  ✅ SCF Risk Data: Ready for TabPFN training")
    else:
        print("  ❌ SCF Risk Data: Missing - check SCFP2019.csv")
        
    if sp500_processed.exists():
        print("  ✅ S&P 500 Data: Ready for RL training")
    else:
        print("  ⚠️ S&P 500 Data: Missing - RL will use synthetic data")
    
    print("\n🚀 Next Steps:")
    if scf_processed.exists():
        print("  1. Run: python scripts/run_risk_model_training.py")
        if sp500_processed.exists():
            print("  2. Run: python scripts/run_rl_agent_training.py")
            print("  3. Run: python scripts/run_dashboard.py")
        else:
            print("  2. Skip RL training or fix S&P 500 data")
            print("  3. Run: python scripts/run_dashboard.py (risk profiler only)")
    else:
        print("  1. Fix SCF data first (download SCFP2019.csv)")
    
    print("\n--- Enhanced Data Processing Pipeline Completed! ---")

if __name__ == "__main__":
    main()