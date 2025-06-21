"""Handles fetching, cleaning, and processing of S&P 500 market data.

This module contains functions for downloading stock price data from Yahoo Finance,
processing the data by handling missing values, and preparing it for analysis.
"""
import pandas as pd
import yfinance as yf
import numpy as np
from pathlib import Path
from typing import List, Optional
import warnings
from ..config import MISSING_VALUE_THRESHOLD, SP500_WIKI_URL

warnings.filterwarnings('ignore')


def get_sp500_tickers() -> List[str]:
    """Scrapes the list of S&P 500 tickers from Wikipedia.
    
    Returns:
        List[str]: A list of S&P 500 stock tickers.
        
    Raises:
        Exception: If unable to fetch the ticker list from Wikipedia.
    """
    try:
        print("Fetching S&P 500 ticker list from Wikipedia...")
        data_table = pd.read_html(SP500_WIKI_URL)
        tickers = data_table[0]['Symbol'].tolist()
        print(f"Successfully fetched {len(tickers)} tickers")
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        # Fallback to a default list
        default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'BRK-B', 'UNH', 'JNJ', 'V']
        print(f"Using fallback ticker list with {len(default_tickers)} tickers")
        return default_tickers


def fetch_sp500_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches historical stock price data from Yahoo Finance.

    Args:
        tickers (List[str]): A list of stock tickers to fetch.
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the 'Adj Close' prices for the tickers.
        
    Raises:
        Exception: If there's an error fetching data from yfinance.
    """
    print(f"Fetching S&P 500 data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    # Initialize an empty DataFrame to store the 'Adj Close' columns for multiple stocks
    adj_close_data = pd.DataFrame()
    
    # Download data for each ticker individually to handle errors gracefully
    successful_tickers = []
    failed_tickers = []
    
    for i, ticker in enumerate(tickers):
        try:
            print(f"Downloading {ticker} ({i+1}/{len(tickers)})...")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if not data.empty and 'Adj Close' in data.columns:
                # Extract the 'Adj Close' column
                data_adj_close = data[['Adj Close']].copy()
                # Rename the column to match the ticker symbol
                data_adj_close.rename(columns={'Adj Close': ticker}, inplace=True)
                # Concatenate to the main DataFrame
                adj_close_data = pd.concat([adj_close_data, data_adj_close], axis=1)
                successful_tickers.append(ticker)
            else:
                failed_tickers.append(ticker)
                print(f"Warning: No data found for {ticker}")
                
        except Exception as e:
            failed_tickers.append(ticker)
            print(f"Error downloading {ticker}: {e}")
    
    if adj_close_data.empty:
        print("Warning: No data was successfully fetched from yfinance.")
        return pd.DataFrame()
    
    # Fix datetime index format
    adj_close_data.index = pd.to_datetime(adj_close_data.index, format='%Y-%m-%d')
    
    print(f"Successfully downloaded data for {len(successful_tickers)} tickers")
    if failed_tickers:
        print(f"Failed to download data for {len(failed_tickers)} tickers: {failed_tickers[:10]}{'...' if len(failed_tickers) > 10 else ''}")
    
    return adj_close_data


def clean_sp500_data(raw_df: pd.DataFrame, start_from_date: str = '2010-01-03') -> pd.DataFrame:
    """Processes the raw market data by handling missing values and filtering dates.

    Args:
        raw_df (pd.DataFrame): The raw pandas DataFrame of stock prices.
        start_from_date (str): Date to start the data from (default: '2010-01-03').

    Returns:
        pd.DataFrame: A cleaned pandas DataFrame ready for analysis.
    """
    if raw_df.empty:
        print("Warning: Empty DataFrame provided for cleaning")
        return pd.DataFrame()
    
    print("Processing S&P 500 market data...")
    
    # Slice dataframe to select stock data from the specified start date
    print(f"Filtering data from {start_from_date} onwards...")
    processed_df = raw_df[start_from_date:].copy()
    
    # Calculate the fraction of missing values for each column
    missing_fractions = processed_df.isnull().mean().sort_values(ascending=False)
    
    # Identify columns to drop (those with high percentage of missing values)
    drop_list = list(missing_fractions[missing_fractions > MISSING_VALUE_THRESHOLD].index)
    
    if drop_list:
        print(f"Dropping {len(drop_list)} columns with >{MISSING_VALUE_THRESHOLD*100}% missing values")
        processed_df.drop(labels=drop_list, axis=1, inplace=True)
    
    # Fill remaining missing values using linear interpolation
    print("Interpolating remaining missing values...")
    processed_df.interpolate(method='linear', axis=0, limit_direction='both', inplace=True)
    
    # Final check for any remaining missing values
    remaining_missing = processed_df.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"Warning: {remaining_missing} missing values remain after interpolation")
        # Forward fill and backward fill as last resort
        processed_df = processed_df.ffill().bfill()
    
    print(f"Market data processing complete. Final shape: {processed_df.shape}")
    return processed_df


def get_live_market_data(tickers: List[str], period: str = "60d") -> Optional[pd.DataFrame]:
    """Attempts to fetch recent market data for live recommendations.
    
    This function is designed for the Streamlit dashboard to get current market data
    with graceful fallback handling.
    
    Args:
        tickers (List[str]): List of stock tickers to fetch.
        period (str): Period for data (e.g., "60d", "1y").
        
    Returns:
        Optional[pd.DataFrame]: DataFrame with recent market data, or None if failed.
    """
    try:
        print(f"Attempting to fetch live market data for {len(tickers)} tickers...")
        data = yf.download(tickers, period=period, interval="1d", progress=False)
        
        if data.empty:
            print("No live data retrieved")
            return None
            
        # Extract Adj Close prices
        if len(tickers) == 1:
            # Handle single ticker case
            adj_close_data = data[['Adj Close']].copy()
            adj_close_data.columns = tickers
        else:
            # Handle multiple tickers
            adj_close_data = data['Adj Close'].copy()
        
        print(f"Successfully fetched live data. Shape: {adj_close_data.shape}")
        return adj_close_data
        
    except Exception as e:
        print(f"Error fetching live market data: {e}")
        return None


def create_ticker_options_for_dropdown(df: pd.DataFrame) -> List[dict]:
    """Creates a list of ticker options for dashboard dropdown menus.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock price data with tickers as columns.
        
    Returns:
        List[dict]: List of dictionaries with 'label' and 'value' keys for dropdown options.
    """
    options = []
    for ticker in df.columns:
        options.append({
            'label': ticker,
            'value': ticker
        })
    return options


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates percentage returns for the given price data.
    
    Args:
        df (pd.DataFrame): DataFrame with price data.
        
    Returns:
        pd.DataFrame: DataFrame with percentage returns.
    """
    return df.pct_change().dropna()


def calculate_market_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculates technical indicators and market features for RL training.
    
    Args:
        df (pd.DataFrame): DataFrame with price data.
        window (int): Window size for rolling calculations.
        
    Returns:
        pd.DataFrame: DataFrame with calculated features.
    """
    features_df = pd.DataFrame(index=df.index)
    
    for column in df.columns:
        price_series = df[column]
        
        # Moving averages
        features_df[f'{column}_MA_{window}'] = price_series.rolling(window=window).mean()
        
        # Volatility (rolling standard deviation)
        features_df[f'{column}_Vol_{window}'] = price_series.rolling(window=window).std()
        
        # Price momentum (rate of change)
        features_df[f'{column}_Momentum_{window}'] = price_series.pct_change(periods=window)
        
        # Relative position (current price vs rolling max/min)
        rolling_max = price_series.rolling(window=window).max()
        rolling_min = price_series.rolling(window=window).min()
        features_df[f'{column}_RelPos_{window}'] = (price_series - rolling_min) / (rolling_max - rolling_min + 1e-8)
    
    return features_df.dropna()