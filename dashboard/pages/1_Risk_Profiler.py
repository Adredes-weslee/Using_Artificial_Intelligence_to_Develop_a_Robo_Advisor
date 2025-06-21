"""Streamlit Page: Risk Profiler."""
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import src.config as config

st.set_page_config(page_title="Risk Profiler", page_icon="🎯")
st.title("🎯 Investor Risk Profiler")
st.markdown("Answer the questions below to get your personalized risk tolerance score.")

@st.cache_resource
def load_risk_model():
    model_path = config.OUTPUT_DIR / config.RISK_MODEL_FILE
    if not model_path.exists():
        st.error("Risk model not found. Please run the training script first.")
        return None
    return joblib.load(model_path)

model = load_risk_model()

if model:
    # Create input fields for user
    # This is a simplified version. A real app would have more intuitive inputs.
    st.subheader("Your Financial & Demographic Profile")
    age = st.slider("Age", 18, 100, 40)
    net_worth = st.number_input("Net Worth (USD)", value=100000)
    income_cat = st.selectbox("Income Category (1=Low, 10=High)", list(range(1, 11)), index=4)
    # ... add more input fields for the other features ...

    if st.button("Calculate My Risk Score"):
        # Create a DataFrame from user inputs
        # Note: You need to create inputs for all features the model was trained on
        # This is a placeholder for the full feature set.
        input_data = pd.DataFrame({
            'AGE': [age], 'NWCAT': [5], 'INCCAT': [income_cat], 'NETWORTH': [net_worth],
            # Add placeholders for all other columns, matching the training columns
            # This is a key part of connecting the UI to the model.
        })
        # Important: Ensure the input_data columns match the model's training columns exactly
        
        # st.write("Model expects columns:", model.feature_names_in_)
        # st.write("Provided columns:", input_data.columns)

        # A real implementation needs to carefully construct the full feature vector.
        st.warning("This is a simplified demo. Full feature input is required for an accurate prediction.")
        # prediction = model.predict(input_data)
        # st.success(f"Your estimated risk tolerance score is: {prediction[0]:.2f} / 4.0")
