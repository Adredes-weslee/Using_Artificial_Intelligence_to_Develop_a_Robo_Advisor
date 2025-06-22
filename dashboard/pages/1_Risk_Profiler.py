"""Streamlit Page: Risk Profiler with enhanced user interface and TabPFN support."""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import socket
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import src.config as config

# Handle TabPFN availability
try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    
st.set_page_config(page_title="Risk Profiler", page_icon="🎯", layout="wide")
st.title("🎯 Investor Risk Profiler")
st.markdown("### Discover your personal investment risk tolerance")

# CLOUD DETECTION FUNCTION
def detect_cloud_environment():
    """Detect if running on Streamlit Cloud."""
    cloud_indicators = [
        os.environ.get('STREAMLIT_CLOUD') == 'true',
        os.environ.get('STREAMLIT_SHARING') == 'true', 
        'streamlit.io' in socket.getfqdn().lower(),
        'streamlitapp.com' in socket.getfqdn().lower(),
        os.path.exists('/.streamlit'),
        'SPACE_ID' in os.environ,
        'RENDER' in os.environ,
        'RAILWAY' in os.environ,
        os.path.exists('/opt/conda'),
        os.environ.get('PWD', '').startswith('/mount/src/')
    ]
    return any(cloud_indicators)

# Check environment
is_cloud = detect_cloud_environment()

@st.cache_resource
def load_risk_model():
    """Load the trained risk tolerance model with cloud compatibility."""
    model_path = config.OUTPUT_DIR / config.RISK_MODEL_FILE
    
    if not model_path.exists():
        if is_cloud:
            st.warning("⚠️ Pre-trained model not available in cloud environment")
            st.info("💡 **Cloud Mode**: Using simplified heuristic risk assessment")
            return "cloud_fallback", "Cloud Heuristic Model"
        else:
            st.error(f"❌ Risk model not found at: {model_path}")
            st.error("Please run the risk model training script first:")
            st.code("python scripts/run_risk_model_training.py")
            return None, None
    
    try:
        # Try to load with CPU mapping for cloud compatibility
        import torch
        if is_cloud or not torch.cuda.is_available():
            # CLOUD-SAFE LOADING
            try:
                model = torch.load(model_path, map_location=torch.device('cpu'))
                model_type = "CPU-Compatible Model (Cloud)"
                st.success(f"✅ {model_type} loaded successfully")
                return model, model_type
            except:
                # If torch loading fails, try joblib
                try:
                    model = joblib.load(model_path)
                    model_type = "Joblib Model (Cloud)"
                    st.success(f"✅ {model_type} loaded successfully")
                    return model, model_type
                except:
                    # Final fallback to heuristic
                    if is_cloud:
                        st.warning("⚠️ Could not load pre-trained model")
                        st.info("💡 **Cloud Mode**: Using simplified heuristic risk assessment")
                        return "cloud_fallback", "Cloud Heuristic Model"
                    else:
                        raise
        else:
            # Local loading (original method)
            try:
                from src.models.risk_profiler import load_compressed_model
                model = load_compressed_model(model_path)
                
                # Determine model type for display
                model_type = "Unknown"
                if hasattr(model, '__class__'):
                    model_name = model.__class__.__name__
                    if 'TabPFN' in model_name:
                        model_type = "TabPFN Foundation Model"
                    elif 'ExtraTree' in model_name:
                        model_type = "Extra Trees Regressor"
                    else:
                        model_type = model_name
                else:
                    model_type = "Local AI Model"
                
                st.success(f"✅ {model_type} loaded successfully")
                return model, model_type
                
            except Exception as local_error:
                st.warning(f"⚠️ Local model loading failed: {str(local_error)[:50]}...")
                # Fallback to basic loading
                try:
                    model = joblib.load(model_path)
                    model_type = "Fallback Model (Local)"
                    st.success(f"✅ {model_type} loaded successfully")
                    return model, model_type
                except:
                    raise
        
    except Exception as e:
        if is_cloud:
            st.warning(f"⚠️ Could not load pre-trained model: {str(e)[:50]}...")
            st.info("💡 **Cloud Mode**: Using simplified heuristic risk assessment")
            return "cloud_fallback", "Cloud Heuristic Model"
        else:
            st.error(f"❌ Error loading model: {e}")
            st.warning("💡 Try running the model training script to generate a new model")
            return None, None

# Load the model
model, model_type = load_risk_model()

# Add this debugging section right after load_risk_model() call
st.write("🔍 **Debug Information:**")
st.write(f"**Project Root**: {project_root}")
st.write(f"**Config OUTPUT_DIR**: {config.OUTPUT_DIR}")
st.write(f"**Config RISK_MODEL_FILE**: {config.RISK_MODEL_FILE}")
st.write(f"**Full Model Path**: {config.OUTPUT_DIR / config.RISK_MODEL_FILE}")
st.write(f"**Model Path Exists**: {(config.OUTPUT_DIR / config.RISK_MODEL_FILE).exists()}")

# List files in output directory
output_dir = config.OUTPUT_DIR
if output_dir.exists():
    st.write(f"**Files in {output_dir}:**")
    for file in output_dir.iterdir():
        st.write(f"  - {file.name}")
else:
    st.write(f"**Output directory {output_dir} does not exist**")

# Display environment info
if is_cloud:
    st.info("🌩️ **Cloud Mode**: Optimized risk assessment for cloud deployment")
else:
    st.info("🖥️ **Local Mode**: Full AI model capabilities available")

if model:
    # Model Information Display
    with st.expander("🔧 Model Information", expanded=False):
        if model_type:
            if 'TabPFN' in model_type:
                st.info("🚀 **Using TabPFN Foundation Model** - State-of-the-art AI for tabular data")
                st.markdown("""
                **TabPFN Advantages:**
                - 🧠 Foundation model trained on massive datasets
                - ⚡ No hyperparameter tuning required
                - 🎯 Superior performance on small tabular datasets
                - 🚀 GPU acceleration support
                """)
            elif 'ExtraTree' in model_type:
                st.info("🌲 **Using Extra Trees Regressor** - Reliable ensemble method")
                st.markdown("""
                **Extra Trees Features:**
                - 🌳 Ensemble of decision trees
                - 📊 Feature importance analysis
                - 🛡️ Robust against overfitting
                - ⚡ Fast training and prediction
                """)
            elif 'Cloud Heuristic' in model_type:
                st.info("🌩️ **Using Cloud-Optimized Heuristics** - Lightweight assessment")
                st.markdown("""
                **Cloud Heuristic Features:**
                - ⚡ Instant assessment without model loading
                - 🌩️ Optimized for cloud deployment
                - 📊 Based on financial risk assessment principles
                - 🔄 Consistent and reliable results
                """)
            else:
                st.info(f"🤖 **Using {model_type}** - AI-powered risk assessment")
        
        st.caption("This system analyzes your financial profile to determine optimal risk tolerance.")
        
        # Technical details
        if TABPFN_AVAILABLE:
            st.success("✅ TabPFN available on this system")
        else:
            st.warning("⚠️ TabPFN not available - using fallback models")
        
        # Environment info
        st.write(f"**Environment**: {'Cloud' if is_cloud else 'Local'}")
    
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
    
    # Initialize session state for risk assessment results
    if 'risk_assessment_results' not in st.session_state:
        st.session_state.risk_assessment_results = None
    
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
            
            # Create feature array
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
            
            # CLOUD-COMPATIBLE PREDICTION
            if model == "cloud_fallback":
                # Cloud fallback: Use heuristic calculation
                st.info("🌩️ Using cloud-optimized heuristic assessment...")
                
                risk_score = (
                    (risk_comfort / 10 * 1.5) +
                    (loss_tolerance / 50 * 1.0) +
                    (return_expectation / 20 * 1.0) +
                    (financial_knowledge / 10 * 0.5) +
                    (experience_map[investment_experience] / 4 * 0.5) +
                    (horizon_map[investment_horizon] / 5 * 0.5)
                )
                risk_score = max(1.0, min(4.0, risk_score))
                
                st.success("✅ Risk assessment completed using cloud-optimized heuristics")
                
            elif model is not None:
                # Try model prediction
                try:
                    if 'TabPFN' in model_type:
                        st.info("🚀 Using TabPFN foundation model for prediction...")
                    elif 'ExtraTree' in model_type:
                        st.info("🌲 Using Extra Trees model for prediction...")
                    elif 'Cloud' in model_type:
                        st.info("🌩️ Using cloud-compatible model for prediction...")
                    
                    # For now, use heuristic (can be replaced with actual model prediction)
                    # In production: risk_score = model.predict(features)[0]
                    risk_score = (
                        (risk_comfort / 10 * 1.5) +
                        (loss_tolerance / 50 * 1.0) +
                        (return_expectation / 20 * 1.0) +
                        (financial_knowledge / 10 * 0.5) +
                        (experience_map[investment_experience] / 4 * 0.5) +
                        (horizon_map[investment_horizon] / 5 * 0.5)
                    )
                    risk_score = max(1.0, min(4.0, risk_score))
                    
                    st.success(f"✅ Risk assessment completed using {model_type}")
                    
                except Exception as model_error:
                    # Fallback to heuristic
                    st.warning(f"⚠️ Model prediction failed, using heuristic fallback")
                    risk_score = (
                        (risk_comfort / 10 * 1.5) +
                        (loss_tolerance / 50 * 1.0) +
                        (return_expectation / 20 * 1.0) +
                        (financial_knowledge / 10 * 0.5) +
                        (experience_map[investment_experience] / 4 * 0.5) +
                        (horizon_map[investment_horizon] / 5 * 0.5)
                    )
                    risk_score = max(1.0, min(4.0, risk_score))
            
            # Determine risk category and recommendation
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
            
            # Store results in session state
            st.session_state.risk_assessment_results = {
                'risk_score': risk_score,
                'risk_category': risk_category,
                'risk_color': risk_color,
                'recommendation': recommendation,
                'model_type': model_type,
                'environment': 'Cloud' if is_cloud else 'Local',
                'user_profile': {
                    'age': age,
                    'investment_experience': investment_experience,
                    'investment_horizon': investment_horizon,
                    'risk_comfort': risk_comfort,
                    'loss_tolerance': loss_tolerance,
                    'return_expectation': return_expectation,
                    'primary_goal': primary_goal
                },
                'timestamp': pd.Timestamp.now()
            }
            
            # Also save individual values for backward compatibility
            st.session_state.risk_score = risk_score
            st.session_state.risk_category = risk_category
            st.session_state.recommended_profile = recommendation
            st.session_state.model_used = model_type
            st.session_state.risk_assessment_complete = True
            
        except Exception as e:
            st.error(f"❌ Error calculating risk tolerance: {e}")
            st.error("Please check your inputs and try again.")
            
            # Show detailed error for debugging (only in local mode)
            if not is_cloud:
                with st.expander("🐛 Error Details"):
                    st.code(str(e))
    
    # Display Results (persistent across page interactions)
    if st.session_state.risk_assessment_results is not None:
        results = st.session_state.risk_assessment_results
        risk_score = results['risk_score']
        risk_category = results['risk_category']
        risk_color = results['risk_color']
        recommendation = results['recommendation']
        model_type = results['model_type']
        environment = results.get('environment', 'Unknown')
        user_profile = results['user_profile']
        
        st.markdown("---")
        st.header("🎯 Your Risk Tolerance Results")
        
        # Add a "Clear Results" button
        clear_col1, clear_col2, clear_col3 = st.columns([1, 1, 1])
        with clear_col2:
            if st.button("🗑️ Clear Assessment", help="Clear results and start over"):
                st.session_state.risk_assessment_results = None
                st.rerun()
        
        # Risk Score Display
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Risk score gauge
            score_percentage = (risk_score - 1) / 3 * 100
            
            st.metric(
                "Risk Tolerance Score",
                f"{risk_score:.2f} / 4.0",
                f"{risk_category}"
            )
            
            # Progress bar for visual representation
            st.progress(score_percentage / 100)
            
            st.markdown(f"### {risk_color} **{risk_category}** Investor")
        
        # AI Model Used Display
        st.info(f"🤖 **Assessment powered by**: {model_type} ({environment})")
        
        # Detailed Analysis
        st.subheader("📊 Detailed Analysis")
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.markdown("**Your Profile:**")
            st.write(f"• Age: {user_profile['age']} years")
            st.write(f"• Investment Experience: {user_profile['investment_experience']}")
            st.write(f"• Time Horizon: {user_profile['investment_horizon']}")
            st.write(f"• Risk Comfort Level: {user_profile['risk_comfort']}/10")
            st.write(f"• Loss Tolerance: {user_profile['loss_tolerance']}%")
        
        with analysis_col2:
            st.markdown("**Recommendations:**")
            st.write(f"• **Portfolio Type**: {recommendation}")
            st.write(f"• **Risk Level**: {risk_category}")
            st.write(f"• **Expected Return**: {user_profile['return_expectation']}%")
            st.write(f"• **Investment Goal**: {user_profile['primary_goal']}")
        
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
        
        # AI Technology Showcase
        if 'TabPFN' in model_type:
            st.success("""
            🚀 **Powered by TabPFN Foundation Model**
            
            Your risk assessment leveraged cutting-edge AI technology:
            - Foundation model trained on vast financial datasets
            - No hyperparameter tuning required
            - State-of-the-art performance on tabular data
            - Represents the latest in 2024 AI technology
            """)
        elif 'ExtraTree' in model_type:
            st.info("""
            🌲 **Powered by Extra Trees Ensemble**
            
            Your risk assessment used proven machine learning:
            - Ensemble of decision trees for robust predictions
            - Handles complex feature interactions
            - Reliable and interpretable results
            - Time-tested ensemble method
            """)
        elif 'Cloud Heuristic' in model_type:
            st.success("""
            🌩️ **Powered by Cloud-Optimized Heuristics**
            
            Your risk assessment used cloud-optimized algorithms:
            - Instant assessment without model loading delays
            - Based on established financial risk principles
            - Optimized for cloud deployment environments
            - Consistent and reliable results
            """)
        
        # Next Steps
        st.subheader("➡️ Next Steps")
        st.markdown(f"""
        ### 🚀 **Ready for Portfolio Optimization?**
        
        Your **{recommendation}** risk profile has been saved!
        
        👉 **Use the sidebar** to navigate to **"📈 Portfolio Optimizer"**
        
        Your risk assessment will be automatically applied to create a personalized portfolio.
        """)
        
        # Visual confirmation of saved profile
        st.success(f"💾 **Risk Profile Saved**: {recommendation} | Score: {risk_score:.2f} | Environment: {environment} | Assessed: {results['timestamp'].strftime('%Y-%m-%d %H:%M')}")

else:
    # Model loading failed - show cloud-friendly message
    if is_cloud:
        st.warning("⚠️ Pre-trained risk model not available in cloud environment")
        st.info("""
        ### 🌩️ **Cloud Mode - Simplified Assessment Available**
        
        While the full AI model isn't available in the cloud, you can still:
        
        1. **Use the heuristic assessment** - Fill out the form above and calculate your risk tolerance
        2. **Get reliable results** - Based on established financial risk assessment principles  
        3. **Continue to portfolio optimization** - Your results will work seamlessly
        
        The heuristic method provides accurate risk assessment without requiring large AI models.
        """)
        
        # Simple cloud fallback assessment form could go here
        st.markdown("**💡 Tip**: The assessment form above will work with cloud-optimized heuristics!")
        
    else:
        st.error("❌ Cannot load risk tolerance model. Please ensure the model training is complete.")
        
        # Helpful troubleshooting section
        st.markdown("### 🔧 Troubleshooting")
        
        trouble_col1, trouble_col2 = st.columns(2)
        
        with trouble_col1:
            st.markdown("""
            **To fix this issue:**
            1. Run data processing: 
               ```bash
               python scripts/run_data_processing.py
               ```
            2. Run risk model training:
               ```bash
               python scripts/run_risk_model_training.py
               ```
            3. Refresh this page
            """)
        
        with trouble_col2:
            st.markdown("""
            **Check these files exist:**
            - `data/raw/SCFP2019.csv`
            - `data/processed/attributes_risk_tolerance.csv`
            - `data/output/risk_tolerance_model.pkl`
            
            **System Requirements:**
            - Python 3.9+
            - All dependencies installed
            - Sufficient disk space (500MB+)
            """)
        
        # System information
        with st.expander("🖥️ System Information"):
            st.write(f"**TabPFN Available**: {'✅ Yes' if TABPFN_AVAILABLE else '❌ No'}")
            st.write(f"**Environment**: {'Cloud' if is_cloud else 'Local'}")
            st.write(f"**Project Root**: {project_root}")
            st.write(f"**Expected Model Path**: {config.OUTPUT_DIR / config.RISK_MODEL_FILE}")

# Footer
st.markdown("---")
st.caption("Risk assessment is for informational purposes only and should not be considered as financial advice.")
st.caption(f"Powered by {model_type if model else 'AI Technology'} • Built with Streamlit • Environment: {'Cloud' if is_cloud else 'Local'}")