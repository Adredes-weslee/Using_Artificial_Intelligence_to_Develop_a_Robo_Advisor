"""Centralized configuration for the AI-Powered Robo-Advisor Project.

This module contains all configuration variables, such as file paths,
API parameters, and model hyperparameters, making it easy to manage
project settings from one place.
"""
from pathlib import Path

# --- Project & Data Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Filenames ---
RAW_SCF_FILE = "SCFP2019.csv"
PROCESSED_SCF_FILE = "attributes_risk_tolerance.csv"
RAW_SP500_FILE = "S&P500.csv"
PROCESSED_SP500_FILE = "sp500_processed.csv"

# --- Model Artifacts ---
RISK_MODEL_FILE = "risk_tolerance_model.pkl"
RL_AGENT_MODEL_FILE = "rl_agent_model.pth"  # Changed to PyTorch format

# --- Data Processing Parameters ---
# Features selected from the Survey of Consumer Finances (SCF) dataset
SELECTED_SCF_FEATURES = [
    'LIQ', 'MMA', 'CHECKING', 'SAVING', 'CALL', 'CDS', 'PREPAID', 'SAVBND', 'CASHLI',
    'NMMF', 'STOCKS', 'BOND', 'OTHMA', 'OTHFIN', 'RETQLIQ',
    'AGECL', 'HHSEX', 'EDCL', 'KIDS', 'MARRIED', 'HOUSECL', 'OCCAT2', 'LIFECL', 
    'INCCAT', 'NWCAT', 'WSAVED', 'SPENDMOR', 'KNOWL'
]

# Risk-free asset columns
RISK_FREE_ASSETS = ['LIQ', 'MMA', 'CHECKING', 'SAVING', 'CALL', 'CDS', 'PREPAID', 'SAVBND', 'CASHLI']

# Risky asset columns
RISKY_ASSETS = ['NMMF', 'STOCKS', 'BOND', 'OTHMA', 'OTHFIN', 'RETQLIQ']

# Final features for model training
FINAL_FEATURES = ['AGECL', 'HHSEX', 'EDCL', 'KIDS', 'MARRIED', 'HOUSECL', 'OCCAT2', 'LIFECL', 
                  'INCCAT', 'NWCAT', 'WSAVED', 'SPENDMOR', 'KNOWL']

TARGET_COLUMN = 'Risk_tolerance'

# --- Portfolio Configuration ---
# Core assets for portfolio optimization and RL training
DEFAULT_PORTFOLIO_ASSETS = [
    # Technology
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'ADBE', 'CRM', 'INTC',
    # Finance
    'JPM', 'BAC', 'V', 'MA', 'GS', 'MS',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR',
    # Consumer
    'PG', 'KO', 'PEP', 'HD', 'COST', 'DIS', 'NFLX',
    # Other Sectors
    'VZ', 'AVGO'
]

# Legacy compatibility
DEFAULT_SP500_TICKERS = ['GOOGL', 'META', 'GS', 'MS', 'GE', 'MSFT']

# Asset universe by sector for portfolio construction
ASSET_UNIVERSE = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'ADBE', 'CRM', 'INTC', 'ORCL', 'CSCO'],
    'Finance': ['JPM', 'BAC', 'V', 'MA', 'GS', 'MS', 'WFC', 'C', 'AXP', 'BLK'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'MRK', 'CVS'],
    'Consumer_Discretionary': ['AMZN', 'TSLA', 'HD', 'DIS', 'NKE', 'MCD', 'SBUX', 'TGT'],
    'Consumer_Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB'],
    'Communication': ['META', 'GOOGL', 'NFLX', 'DIS', 'VZ', 'T', 'CMCSA'],
    'Industrials': ['GE', 'BA', 'UPS', 'CAT', 'MMM', 'HON', 'LMT'],
    'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB'],
    'Utilities': ['NEE', 'DUK', 'SO', 'AEP'],
    'Real_Estate': ['PLD', 'AMT', 'CCI', 'EQIX']
}

# Risk tolerance levels for testing and UI
RISK_TOLERANCE_LEVELS = [0.2, 0.4, 0.5, 0.6, 0.8]

# --- RL Agent Configuration ---
# Multiple RL models for different risk profiles
RL_MODEL_CONFIGS = {
    'Conservative': {
        'model_file': 'rl_agent_conservative.pth',
        'reward_function': lambda reward, volatility: reward - 2 * volatility,
        'description': "Aims for stable returns with low volatility",
        'target_assets': 15,
        'rebalance_frequency': 30
    },
    'Balanced': {
        'model_file': 'rl_agent_balanced.pth',
        'reward_function': lambda reward, volatility: reward / (volatility + 1e-6),
        'description': "Balances return maximization with risk control",
        'target_assets': 20,
        'rebalance_frequency': 45
    },
    'Aggressive': {
        'model_file': 'rl_agent_aggressive.pth',
        'reward_function': lambda reward, volatility: reward,
        'description': "Focuses on maximizing returns, accepting higher volatility",
        'target_assets': 30,
        'rebalance_frequency': 60
    }
}

# Transfer learning configuration
TRANSFER_LEARNING_CONFIG = {
    'min_overlap_ratio': 0.6,
    'retrain_threshold': 0.4,
    'fine_tune_epochs': 10,
    'base_model_weight': 0.7
}

# --- Model Training Parameters ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS_ETR = 50
MAX_DEPTH_ETR = 25
CRITERION_ETR = 'squared_error'

# --- API and Data Parameters ---
SP500_START_DATE = '2000-01-01'
SP500_END_DATE = '2023-09-10'
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
MISSING_VALUE_THRESHOLD = 0.3

# --- RL Agent & Environment Parameters ---
LOOKBACK_WINDOW_SIZE = 50
INITIAL_BALANCE = 10000

# --- Cloud Optimization Configuration (Streamlit Only) ---
CLOUD_OPTIMIZATION_CONFIG = {
    'max_assets_for_rl': 10,
    'fallback_to_mpt': True,
    'memory_limit_mb': 512,
    'enable_transfer_learning': False,
    'max_episodes_cloud': 10
}

# Legacy Risk Profiles (for backward compatibility)
RISK_PROFILES = {
    'Conservative': {
        'reward_function': lambda reward, volatility: reward - 2 * volatility,
        'description': "Aims for stable returns with low volatility."
    },
    'Balanced': {
        'reward_function': lambda reward, volatility: reward / (volatility + 1e-6),
        'description': "Balances return maximization with risk control."
    },
    'Aggressive': {
        'reward_function': lambda reward, volatility: reward,
        'description': "Focuses on maximizing returns, accepting higher volatility."
    }
}