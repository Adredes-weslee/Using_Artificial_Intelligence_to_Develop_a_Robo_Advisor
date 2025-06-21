"""Main Streamlit application file for the AI-Powered Robo-Advisor."""
import streamlit as st

st.set_page_config(
    page_title="AI Robo-Advisor",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI-Powered Robo-Advisor")
st.markdown("""
Welcome to your personal AI-Powered Robo-Advisor. This platform leverages
machine learning and reinforcement learning to provide personalized financial advice.

**Please select a page from the sidebar to begin:**

1.  **Risk Profiler**: Answer a few questions to understand your personal investment risk tolerance.
2.  **Portfolio Optimizer**: Receive a dynamically optimized portfolio allocation based on your risk profile and the latest market conditions.
3.  **Chatbot**: Ask questions about financial concepts.
""")

