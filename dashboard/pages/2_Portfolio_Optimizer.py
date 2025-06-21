"""Streamlit Page: Portfolio Optimizer."""
import streamlit as st
import pandas as pd
import yfinance as yf
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import src.config as config
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈")
st.title("📈 Dynamic Portfolio Optimizer")

@st.cache_data(ttl=3600) # Cache for 1 hour
def get_market_data_with_fallback(tickers: list) -> pd.DataFrame:
    """Fetches live data with a fallback to a processed file."""
    try:
        # Primary: Try to fetch live data
        live_data = yf.download(tickers, period="60d")
        if live_data.empty:
            raise ValueError("No data returned from yfinance.")
        st.success("✅ Fetched live market data.")
        return live_data['Adj Close']
    except Exception as e:
        # Secondary: Fallback to processed file
        st.warning(f"⚠️ Could not fetch live data ({e}). Using last available data snapshot.")
        fallback_path = config.PROCESSED_DATA_DIR / config.PROCESSED_SP500_FILE
        return pd.read_csv(fallback_path, index_col='Date', parse_dates=True)

@st.cache_resource
def load_rl_model():
    """Loads the pre-trained RL agent model."""
    model_path = config.OUTPUT_DIR / config.RL_AGENT_MODEL_FILE
    if not model_path.exists():
        st.error("RL agent model not found. Please run the training script first.")
        return None
    return load_model(model_path)

rl_model = load_rl_model()
market_data = get_market_data_with_fallback(config.SP500_TICKERS)

if rl_model and not market_data.empty:
    st.subheader("Your Optimized Portfolio")
    
    risk_profile = st.selectbox("Select Your Risk Profile", options=list(config.RISK_PROFILES.keys()))

    if st.button("Generate My Portfolio"):
        # Get the latest state from the market data
        current_state = market_data.tail(config.LOOKBACK_WINDOW_SIZE).values
        current_state = current_state.reshape(1, config.LOOKBACK_WINDOW_SIZE, len(config.SP500_TICKERS))
        
        # Get prediction from the RL model
        action_q_values = rl_model.predict(current_state, verbose=0)[0]
        # In a real scenario, different policies would be executed here.
        # This is a simplified placeholder.
        best_action_idx = np.argmax(action_q_values)
        
        # This allocation is simplified. A real agent would output weights.
        st.info(f"Based on a **{risk_profile}** profile, the AI agent recommends the following allocation:")
        
        # Placeholder for portfolio weights output
        weights = [0.4, 0.3, 0.15, 0.1, 0.05] # Example
        portfolio_df = pd.DataFrame({
            'Asset': config.SP500_TICKERS,
            'Recommended Weight': weights
        })
        st.dataframe(portfolio_df.style.format({'Recommended Weight': '{:.2%}'}))


