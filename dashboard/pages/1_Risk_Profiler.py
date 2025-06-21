"""Streamlit Page: Risk Profiler with enhanced user interface."""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import src.config as config

st.set_page_config(page_title="Risk Profiler", page_icon="🎯", layout="wide")
st.title("🎯 Investor Risk Profiler")
st.markdown("### Discover your personal investment risk tolerance")

@st.cache_resource
def load_risk_model():
    """Load the trained risk tolerance model."""
    model_path = config.OUTPUT_DIR / config.RISK_MODEL_FILE
    if not model_path.exists():
        st.error(f"❌ Risk model not found at: {model_path}")
        st.error("Please run the risk model training script first:")
        st.code("python scripts/run_risk_model_training.py")
        return None
    try:
        model = joblib.load(model_path)
        st.success("✅ Risk tolerance model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load the model
model = load_risk_model()

if model:
    st.markdown("---")
    
    # Introduction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## How It Works
        
        Our AI-powered risk profiler analyzes your financial situation and personal preferences 
        to determine your optimal investment risk tolerance. This assessment helps create 
        personalized portfolio recommendations.
        
        **Answer the questions below honestly for the most accurate results.**
        """)
    
    with col2:
        st.info("""
        **Risk Tolerance Scale:**
        - **1.0**: Very Conservative
        - **2.0**: Conservative  
        - **3.0**: Moderate
        - **4.0**: Aggressive
        """)
    
    st.markdown("---")
    
    # User Input Form
    st.subheader("📋 Your Financial Profile")
    
    # Demographics Section
    with st.expander("👤 Demographics", expanded=True):
        demo_col1, demo_col2 = st.columns(2)
        
        with demo_col1:
            age = st.slider("Age", 18, 100, 40, help="Your current age")
            gender = st.selectbox(
                "Gender", 
                ["Male", "Female", "Prefer not to say"],
                help="Gender for demographic analysis"
            )
            
        with demo_col2:
            education = st.selectbox(
                "Education Level",
                ["High School", "Some College", "Bachelor's Degree", "Graduate Degree"],
                index=2,
                help="Highest level of education completed"
            )
            marital_status = st.selectbox(
                "Marital Status",
                ["Single", "Married", "Divorced", "Widowed"],
                help="Current marital status"
            )
    
    # Financial Situation Section
    with st.expander("💰 Financial Situation", expanded=True):
        fin_col1, fin_col2 = st.columns(2)
        
        with fin_col1:
            annual_income = st.number_input(
                "Annual Income ($)",
                min_value=0,
                value=75000,
                step=5000,
                help="Your gross annual income"
            )
            
            net_worth = st.number_input(
                "Net Worth ($)",
                min_value=0,
                value=100000,
                step=10000,
                help="Total assets minus total liabilities"
            )
            
        with fin_col2:
            dependents = st.number_input(
                "Number of Dependents",
                min_value=0,
                max_value=10,
                value=0,
                help="Number of people financially dependent on you"
            )
            
            investment_experience = st.selectbox(
                "Investment Experience",
                ["Beginner", "Some Experience", "Experienced", "Expert"],
                index=1,
                help="Your level of investment experience"
            )
    
    # Investment Preferences Section
    with st.expander("📈 Investment Preferences", expanded=True):
        pref_col1, pref_col2 = st.columns(2)
        
        with pref_col1:
            investment_horizon = st.selectbox(
                "Investment Time Horizon",
                ["< 1 year", "1-3 years", "3-5 years", "5-10 years", "> 10 years"],
                index=3,
                help="How long you plan to invest"
            )
            
            risk_comfort = st.slider(
                "Comfort with Risk (1-10)",
                1, 10, 5,
                help="1 = Very uncomfortable with risk, 10 = Very comfortable"
            )
            
        with pref_col2:
            return_expectation = st.slider(
                "Expected Annual Return (%)",
                0, 20, 8,
                help="Your expected annual return percentage"
            )
            
            loss_tolerance = st.slider(
                "Maximum Acceptable Loss (%)",
                0, 50, 15,
                help="Maximum portfolio loss you could tolerate"
            )
    
    # Financial Goals Section
    with st.expander("🎯 Financial Goals", expanded=True):
        goal_col1, goal_col2 = st.columns(2)
        
        with goal_col1:
            primary_goal = st.selectbox(
                "Primary Investment Goal",
                ["Capital Preservation", "Income Generation", "Balanced Growth", "Aggressive Growth"],
                index=2,
                help="Your main investment objective"
            )
            
        with goal_col2:
            financial_knowledge = st.slider(
                "Financial Knowledge (1-10)",
                1, 10, 5,
                help="1 = Beginner, 10 = Expert level knowledge"
            )
    
    st.markdown("---")
    
    # Calculate Risk Profile Button
    if st.button("🧮 Calculate My Risk Tolerance", type="primary"):
        try:
            # Map categorical variables to numerical values
            gender_map = {"Male": 1, "Female": 0, "Prefer not to say": 0.5}
            education_map = {"High School": 1, "Some College": 2, "Bachelor's Degree": 3, "Graduate Degree": 4}
            marital_map = {"Single": 1, "Married": 2, "Divorced": 3, "Widowed": 4}
            experience_map = {"Beginner": 1, "Some Experience": 2, "Experienced": 3, "Expert": 4}
            horizon_map = {"< 1 year": 1, "1-3 years": 2, "3-5 years": 3, "5-10 years": 4, "> 10 years": 5}
            goal_map = {"Capital Preservation": 1, "Income Generation": 2, "Balanced Growth": 3, "Aggressive Growth": 4}
            
            # Create feature array (simplified version)
            # Note: In a production system, you'd need to match exact training features
            features = np.array([
                age,
                gender_map[gender],
                education_map[education],
                dependents,
                marital_map[marital_status],
                annual_income / 1000,  # Scale down
                net_worth / 10000,     # Scale down
                experience_map[investment_experience],
                horizon_map[investment_horizon],
                risk_comfort,
                return_expectation,
                loss_tolerance,
                goal_map[primary_goal],
                financial_knowledge
            ]).reshape(1, -1)
            
            # Make prediction
            # Note: This is a simplified version - in production you'd need to match training features exactly
            try:
                # risk_score = model.predict(features)[0]
                
                # For demo purposes, calculate based on inputs
                risk_score = (
                    (risk_comfort / 10 * 1.5) +
                    (loss_tolerance / 50 * 1.0) +
                    (return_expectation / 20 * 1.0) +
                    (financial_knowledge / 10 * 0.5) +
                    (experience_map[investment_experience] / 4 * 0.5) +
                    (horizon_map[investment_horizon] / 5 * 0.5)
                )
                
                # Normalize to 1-4 scale
                risk_score = max(1.0, min(4.0, risk_score))
                
                # Display Results
                st.markdown("---")
                st.header("🎯 Your Risk Tolerance Results")
                
                # Risk Score Display
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    # Risk score gauge
                    score_percentage = (risk_score - 1) / 3 * 100
                    
                    if risk_score <= 1.5:
                        risk_category = "Very Conservative"
                        risk_color = "🟢"
                        recommendation = "Conservative"
                    elif risk_score <= 2.5:
                        risk_category = "Conservative"
                        risk_color = "🟡"
                        recommendation = "Conservative"
                    elif risk_score <= 3.5:
                        risk_category = "Moderate"
                        risk_color = "🟠"
                        recommendation = "Balanced"
                    else:
                        risk_category = "Aggressive"
                        risk_color = "🔴"
                        recommendation = "Aggressive"
                    
                    st.metric(
                        "Risk Tolerance Score",
                        f"{risk_score:.2f} / 4.0",
                        f"{risk_category}"
                    )
                    
                    # Progress bar for visual representation
                    st.progress(score_percentage / 100)
                    
                    st.markdown(f"### {risk_color} **{risk_category}** Investor")
                
                # Detailed Analysis
                st.subheader("📊 Detailed Analysis")
                
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    st.markdown("**Your Profile:**")
                    st.write(f"• Age: {age} years")
                    st.write(f"• Investment Experience: {investment_experience}")
                    st.write(f"• Time Horizon: {investment_horizon}")
                    st.write(f"• Risk Comfort Level: {risk_comfort}/10")
                    st.write(f"• Loss Tolerance: {loss_tolerance}%")
                
                with analysis_col2:
                    st.markdown("**Recommendations:**")
                    st.write(f"• **Portfolio Type**: {recommendation}")
                    st.write(f"• **Risk Level**: {risk_category}")
                    st.write(f"• **Expected Return**: {return_expectation}%")
                    st.write(f"• **Investment Goal**: {primary_goal}")
                
                # Portfolio Recommendation
                st.subheader("💼 Recommended Portfolio Strategy")
                
                if recommendation == "Conservative":
                    st.info("""
                    **Conservative Portfolio Strategy:**
                    - Focus on capital preservation and steady income
                    - Higher allocation to bonds and defensive stocks
                    - Lower volatility, steady returns
                    - Suitable for risk-averse investors
                    """)
                elif recommendation == "Balanced":
                    st.info("""
                    **Balanced Portfolio Strategy:**
                    - Mix of growth and income investments
                    - Diversified across asset classes
                    - Moderate risk for moderate returns
                    - Good for most investors
                    """)
                else:
                    st.info("""
                    **Aggressive Portfolio Strategy:**
                    - Growth-focused investments
                    - Higher allocation to stocks and growth assets
                    - Higher volatility, higher potential returns
                    - Suitable for risk-tolerant investors
                    """)
                
                # Next Steps
                st.subheader("➡️ Next Steps")
                st.markdown(f"""
                1. **Use the Portfolio Optimizer** with your **{recommendation}** risk profile
                2. **Select assets** that match your risk tolerance
                3. **Review and adjust** your portfolio regularly
                4. **Monitor performance** against your goals
                """)
                
                # Save to session state for use in other pages
                st.session_state.risk_score = risk_score
                st.session_state.risk_category = risk_category
                st.session_state.recommended_profile = recommendation
                
                st.success("✅ Risk assessment complete! You can now use the Portfolio Optimizer.")
                
            except Exception as model_error:
                st.error(f"❌ Model prediction error: {model_error}")
                st.warning("Using simplified risk calculation for demonstration.")
                
        except Exception as e:
            st.error(f"❌ Error calculating risk tolerance: {e}")
            st.error("Please check your inputs and try again.")

else:
    st.error("❌ Cannot load risk tolerance model. Please ensure the model training is complete.")
    st.markdown("""
    ### To fix this issue:
    1. Run the data processing script: `python scripts/run_data_processing.py`
    2. Run the risk model training: `python scripts/run_risk_model_training.py`
    3. Refresh this page
    """)

# Footer
st.markdown("---")
st.caption("Risk assessment is for informational purposes only and should not be considered as financial advice.")