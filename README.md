# 🤖 AI-Powered Robo Advisor: Risk Profiling & Portfolio Optimization
> Predicting investor risk tolerance, allocating portfolio weights using MVO and Deep Reinforcement Learning, and building a live dashboard.

## 📘 Overview
This project delivers an AI-based robo advisor capable of:

- Predicting investor **risk tolerance** using demographic, financial, and behavioral attributes (from the **2019 Survey of Consumer Finances**) via **supervised machine learning**.
- Using **Deep Q-Learning (DQN)** to dynamically allocate portfolio weights across selected **S&P 500** stocks and compare with traditional **Mean Variance Optimization (MVO)**.
- Deploying a **Plotly Dash**-based robo advisor dashboard to interactively forecast risk tolerance and generate optimal portfolio allocations.

> 🏦 Client: Robo advisory firms and their investor users  
> ✨ Deployment: [Live Heroku Dashboard](https://wes-roboadvisor-dashboard-082860bb1de2.herokuapp.com/)

---

## 📂 Datasets

| Feature | Dataset | Description |
|--------:|--------:|-------------|
| Investor Attributes | `attributes_risk_tolerance.csv` | Truncated from SCFP2019 with 28 selected variables (15 asset types + 13 demographic/behavioral) |
| Full Survey | `SCFP2019.csv` | Complete dataset from the 2019 Survey of Consumer Finances |
| Market Data | `S&P500.csv` | Adjusted weekly close prices (2000–2023) scraped via yfinance for S&P 500 stocks |

> 🌐 Sources:  
> ▶️ [SCF 2019](https://www.federalreserve.gov/econres/scfindex.htm)  
> ▶️ [S&P 500 constituents](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)

---

## 🤔 Methodology

### ✏️ Risk Tolerance Prediction
- Target = risk tolerance (computed as ratio of risky to total financial assets)
- Features = 13 selected demographic, financial, behavioral predictors

**Models Used:**
- Linear Regression (with Lasso, ElasticNet)
- Support Vector Regression (SVR)
- Random Forest, Extra Trees
- Gradient Boosting, AdaBoost
- k-NN, Decision Trees
- Multi-layer Perceptron & Deep Neural Network

> ⚖️ Best model: **Extra Trees Regression** (R² = 0.9143, Test RMSE = 0.1152)

### 🤖 Deep Q-Learning (Reinforcement Learning)
- Uses a **Deep Q-Network (DQN)** agent to learn optimal portfolio allocation via Sharpe Ratio maximization
- Trains on simulated stock environment of 10 random S&P 500 stocks (2020–present)
- Rewards = Sharpe ratio; ANN trained to minimize Q-value MSE loss

> 📊 Compares DQN portfolio performance against static **MVO portfolios** using `cvxopt`

### 💻 Dashboard
- Built with **Plotly Dash**
- Risk tolerance model embedded for live prediction
- Interactive selection of S&P 500 tickers
- Generates optimal weights using MVO
- Integrated chatbot (planned for future expansion)

---

## 🔢 Model Performance

| Model | Test R² | Test RMSE |
|-------|---------|-----------|
| Extra Trees | **0.9143** | **0.1152** |
| Decision Tree | 0.8946 | 0.1277 |
| Random Forest | 0.9061 | 0.1206 |
| k-NN | 0.7685 | 0.1893 |
| SVR | 0.3707 | 0.3121 |
| DNN (Keras) | -0.1155 | 0.3398 |

> 🔹 Extensive grid search was performed to optimize Extra Trees hyperparameters.

---

## 🔹 Conclusions

1. **Extra Trees Regression** is highly effective at predicting individual investor risk tolerance.
2. **MVO** and **Reinforcement Learning** (DQN) are viable portfolio optimization methods, with DQN enabling **dynamic rebalancing**.
3. The deployed **dashboard** demonstrates practical integration of ML predictions and financial modeling.

---

## 💼 Recommendations

1. Expand feature selection in SCF data (e.g., use continuous variables for age, income).
2. Add support for other asset classes (international equities, crypto, bonds).
3. Integrate DQN agent with brokerage APIs for **live portfolio rebalancing**.

---

## 📊 Project Files

| Notebook | Purpose |
|----------|---------|
| `01_fetching_S&P500_data.ipynb` | Web scraping and formatting of S&P 500 prices |
| `02_predicting_risk_tolerance.ipynb` | Supervised ML modeling (regressors, evaluation) |
| `03_reinforcement_learning.ipynb` | DQN agent and simulation environment for portfolio allocation |
| `04_robo_advisor_dashboard_with_chatbot.ipynb` | Dash dashboard for prediction + allocation demo |

---

## 📖 References

1. https://www.federalreserve.gov/econres/scfindex.htm  
2. https://en.wikipedia.org/wiki/List_of_S%26P_500_companies  
3. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=95489  
4. https://books.google.com.sg/books?id=xS2pDAAAQBAJ  
5. https://www.mckinsey.com/industries/financial-services/our-insights/reimagining-customer-engagement-for-the-ai-bank-of-the-future  
6. https://www.cvxopt.org/  
7. https://towardsdatascience.com/introduction-to-deep-q-learning-lets-play-flappy-bird-e6195c3cecd1

---

**Author:** Wes Lee  
🔗 [LinkedIn](https://www.linkedin.com/in/wes-lee)  •  💻 Portfolio available upon request  
📄 License: MIT

---

