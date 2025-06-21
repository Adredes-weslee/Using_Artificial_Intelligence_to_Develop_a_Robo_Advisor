# 🤖 AI-Powered Robo-Advisor: Risk Profiling & Portfolio Optimization

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45.0-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6.1-F7931E.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-FF6F00.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An advanced financial platform that predicts an investor's risk tolerance using supervised learning, then leverages a pre-trained Deep Reinforcement Learning agent to recommend a personalized, dynamically optimized portfolio allocation via an interactive web application.

---

## 🎯 Project Overview

This project refactors a multi-stage financial modeling research process from a series of Jupyter notebooks into a robust, modular, and production-ready application. The system is designed to function as an AI-powered robo-advisor with two core AI components:

1.  **Risk Tolerance Prediction**: A supervised machine learning model (e.g., `RandomForestRegressor`) is trained **offline** on the **2019 Survey of Consumer Finances (SCF)** dataset to predict an individual's risk tolerance based on their demographic, financial, and behavioral attributes.
2.  **Dynamic Portfolio Optimization**: A **Deep Q-Learning (DQN)** reinforcement learning agent is trained **offline** to learn multiple allocation policies (e.g., Conservative, Balanced, Aggressive) based on different reward functions. It learns to map market states to an optimal portfolio allocation that aligns with a specific risk profile.

The final deliverable is an interactive Streamlit web application where a user can get their risk profile predicted. This profile then selects the appropriate pre-trained policy from the RL agent, which recommends a personalized and dynamically optimized portfolio based on the latest available market data.

### ✨ How This Project is Distinct

While both this project and the "ML Trading Strategist" use reinforcement learning, their objectives are fundamentally different:
* **ML Trading Strategist**: A tool for quants to **backtest trading algorithms** that decide *when to buy/sell* an asset.
* **AI-Powered Robo-Advisor**: A product for investors to get **personalized portfolio advice** that decides *how much of each asset* to hold based on personal risk appetite.

---

## 🏗️ Project Architecture

The architecture is designed for a robust deployment, separating heavy offline training from lightweight online inference.

### 📊 System Workflow

1.  **Market Data Ingestion (Offline)**: Historical price data for S&P 500 stocks is fetched and cleaned to create a historical dataset for training.
2.  **Risk Profile Data Processing (Offline)**: The comprehensive SCF dataset is processed to engineer a `risk_tolerance` target variable and select relevant features.
3.  **Risk Model Training (Offline)**: The supervised risk tolerance prediction model is trained on the SCF data and saved as a `.pkl` artifact.
4.  **Portfolio Agent Training (Offline)**: The DQN agent is trained on historical S&P 500 data to learn multiple optimal allocation policies (Conservative, Balanced, Aggressive). The single, multi-policy agent is saved as an `.h5` artifact.
5.  **Robo-Advisor Dashboard (Online)**: A multi-page Streamlit application is deployed to the cloud.
    * It **loads the pre-trained models** on startup.
    * On user request, it **fetches live market data** to determine the current market state.
    * A user inputs their financial details to get a **risk score** from the supervised model.
    * This score selects the corresponding policy from the RL agent, which uses the **current market state** to generate a personalized portfolio recommendation.

### ☁️ Deployment Strategy & Live Data Handling

To ensure reliability on a cloud platform like Streamlit Community Cloud, this application uses a **"Live with a Cached Fallback"** strategy.

1.  **Live Data First**: The dashboard attempts to fetch the latest market data using the `yfinance` API.
2.  **Graceful Fallback**: If the API fails (due to network issues or backend changes), the application will **not crash**. Instead, it loads the most recent data from the `sp500_processed.csv` file that was deployed with the app.
3.  **User Notification**: In case of a fallback, a prominent warning is displayed to the user, informing them that the recommendation is based on slightly stale data.

This makes the application both resilient and transparent.

### 📁 Directory Structure

```
ai-robo-advisor/
│
├── 📂 data/
│   ├── raw/
│   │   ├── SCFP2019.csv              # Raw survey data
│   │   └── S&P500.csv                # Raw S&P 500 price data
│   ├── processed/
│   │   ├── attributes_risk_tolerance.csv # Processed data for risk model
│   │   └── sp500_processed.csv       # Cleaned S&P 500 data, used as API fallback
│   └── output/
│       ├── risk_tolerance_model.pkl    # Trained supervised learning model
│       └── rl_agent_model.h5           # Trained multi-policy DQN agent
│
├── 📂 notebooks/                     # Original research notebooks for reference
│   ├── 01_fetching_S&P500_data.ipynb
│   ├── 02_predicting_risk_tolerance.ipynb
│   ├── 03_reinforcement_learning.ipynb
│   └── 04_robo_advisor_dashboard_with_chatbot.ipynb
│
├── 📂 src/
│   ├── init.py
│   ├── config.py                     # Project configuration (paths, model params)
│   ├── data_processing/
│   │   ├── init.py
│   │   ├── market_data.py            # S&P 500 data fetching and cleaning
│   │   └── survey_data.py            # SCF data processing and feature engineering
│   ├── models/
│   │   ├── init.py
│   │   ├── risk_profiler.py          # Supervised ML model training for risk tolerance
│   │   └── rl_agent.py               # DQN Agent and StockEnvironment classes
│   └── utils/
        ├── init.py
│       └── portfolio_math.py         # MVO and other financial calculations
│
├── 📂 dashboard/
│   ├── init.py
│   ├── app.py                        # Main Streamlit application file
│   └── pages/                        # Individual pages for the dashboard
│       ├── init.py
│       ├── 1_Risk_Profiler.py
│       ├── 2_Portfolio_Optimizer.py
│       └── 3_Chatbot.py
│
├── 📂 scripts/
│   ├── run_data_processing.py        # Executes all data processing pipelines
│   ├── run_risk_model_training.py    # Trains the risk tolerance prediction model
│   ├── run_rl_agent_training.py      # Trains the DQN portfolio allocation agent
│   └── run_dashboard.py              # Launches the Streamlit dashboard
│
├── 📄 requirements.txt                # Python package dependencies
├── 📄 README.md                       # This project overview document
└── 📄 .gitignore                      # Files to be ignored by Git
```

---

## ⚡ Quick Start Guide

### 🛠️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ai-robo-advisor
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

### ⚙️ Running the Full Pipeline

This process is divided into offline training (which you run once locally) and online deployment.

#### Offline Steps (Run Locally)

1.  **Process all data:**
    *This script prepares both the SCF survey data and the S&P 500 market data for training.*
    ```bash
    python scripts/run_data_processing.py
    ```

2.  **Train the machine learning models:**
    *First, train the risk profiler:*
    ```bash
    python scripts/run_risk_model_training.py
    ```
    *Then, train the RL portfolio agent (this may take a significant amount of time):*
    ```bash
    python scripts/run_rl_agent_training.py
    ```

#### Online Step (Deploy or Run Locally)

3.  **Launch the Robo-Advisor Dashboard:**
    *This will start the Streamlit server, load the pre-trained models, and open the application in your browser.*
    ```bash
    python scripts/run_dashboard.py
    ```
    🌐 **Dashboard**: Open `http://localhost:8501` in your browser.
