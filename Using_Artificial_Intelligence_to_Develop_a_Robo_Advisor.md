# The link for the robo advisor is https://wes-roboadvisor-dashboard-082860bb1de2.herokuapp.com/


**Objective:** 

Use artificial intelligence to develop a robo advisor and build a dashboard for investors to predict their risk tolerance and determine optimal portfolio weights for any selection of equities from the S&P 500. Use reinforcement learning for portfolio allocation as an alternative to allocating portfolio weights using mean variance optimization.


**Problem statement** 

1. Predict risk tolerance of an individual from various demographic, financial and behavioural attributes using supervised ML
2. Use reinforcement learning (value-based deep Q-network (DQN)) to dynamically change the portfolio allocation weights and compare cumulative return against the mean variance optimized (MVO) portfolio
3. Build a robo-advisor dashboard using Plotly Dash and implement the risk tolerance prediction model in the dashboard. Using the predicted value of risk tolerance and choosing from any selection of equities from the S&P 500, calculate the optimal portfolio weights using mean variance optimization (MVO) with the aid of the CVXOPT library (uses convex optimization)


**Primary Project Stakeholder:** 

Client investors of a robo advisory firm.


**Secondary Project Stakeholder:** 

The robo advisory firm. I prefer to put the clients first as the primary stakeholder since they generate the value for the business.


**Target Variable:** 

1. For the problem statement to be solved using supervised ML, the target variable is risk tolerance. This is first calculated from a ratio of risky to risk + risk-free assets. Selected demographic, financial and behavioral attributes of an individual are then used as the feature variables to predict the risk tolerance
2. Reinforcement learning is neither supervised nor unsupervised as it does not require labeled data or a training set. It relies on the ability to monitor the response to the actions of the learning agent. For the valued-based DQN, the reward is the sharpe ratio of the portfolio. Maximizing this reward will lead to portfolio allocation weights that maximize the return of the portfolio for a given level of risk


**Evaluation Metrics:**  

1. Evaluate the supervised ML models based on their R2 and RMSE values
2. For the value-based DQN, the loss function minimized by the ANN through gradient descent is the squared difference between the DQN’s estimate of the target and its estimate of the Q-value of the current state-action pair, Q(s,a:θ). We don't directly evaluate the value of the loss function, but since the Q-value is the expected reward for the state-action pair following a policy π, when the algorithm iteratively converges to the optimal Q-value, it learns an optimal policy which is how to act to maximize the return/reward in every state, which is to maximize the sharpe ratio of the portfolio. By maximizing the sharpe ratio of the portfolio, the RL agent also dynamically (automatically) change the portfolio allocation weights


**Notebooks**

1. 01_fetching_S&P500_data
2. 02_predicting_risk_tolerance
3. 03_reinforcement_learning
4. 04_robo_advisor_dashboard_with_chatbot


**README Overview**

1. Datasets
2. Methodolgy
3. Models Performance 
4. Conclusion
5. Recommendations
6. References


# 1. Datasets

1. 2019 Survey of Consumer Finances (SCF) [available from the Federal Reserve website.](https://www.federalreserve.gov/econres/scfindex.htm) This survey is conducted triennially (once every 3 years)
    - This full data set is saved as SCFP2019.csv
    - The truncated dataset used to build the robo advisor is saved as attributes_risk_tolerance.csv. Out of 351 variables from the full data set, 28 were selected with 15 representing various measures of asset values and the other 13 representing various demographic, financial and behavioural attributes

2. Adjusted closing prices for all S&P 500 stocks from 1st week of 2000 to 1st Week of September 2023 downloaded using yfinance library. [The full list of S&P 500 stocks was scraped from Wikipedia.](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
    - Saved as S&P500.csv

**Data Dictionary:**

|  Feature | Type | Dataset |                                                                                                                                    Description                                                                                                                                    |
|:--------:|:----:|:-------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|    LIQ   |      |         |                                                                                                          Total value of all types of transactions accounts, 2019 dollars                                                                                                          |
|    MMA   |      |         |                                                                                              Total value of money market deposit and money market mutual fund accounts, 2019 dollars                                                                                              |
| CHECKING |      |         |                                                                    Total value of checking accounts held by household, 2019 dollars (Money market accounts are not included in the value of checking accounts)                                                                    |
|  SAVING  |      |         |                                                                                      Total value of savings accounts held by household, 2019 dollars (Does not include money market accounts)                                                                                     |
|   CALL   |      |         |                                                                                                            Total value of call accounts held by household, 2019 dollars                                                                                                           |
|    CDS   |      |         |                                                                                                       Total value of certificates of deposit held by household, 2019 dollars                                                                                                      |
|  PREPAID |      |         |                                                                                                                   Amount in prepaid card accounts, 2019 dollars                                                                                                                   |
|  SAVBND  |      |         |                                                                                                            Total value of savings bonds held by household, 2019 dollars                                                                                                           |
|  CASHLI  |      |         |                                                                                                      Total cash value of whole life insurance held by household, 2019 dollars                                                                                                     |
|   NMMF   |      |         | Total value of directly held pooled investment funds held by household, 2019 dollars (Excludes money market mutual funds, but includes stock mutual funds, tax free bond mutual funds, government bond mutual funds, and combination and other mutual funds, such as hedge funds) |
|  STOCKS  |      |         |                                                                                                        Total value of directly held stocks held by household, 2019 dollars                                                                                                        |
|   BOND   |      |         |                                               Total value of directly held bonds held by household, 2019 dollars (Includes nontaxable bonds, mortgage bonds, government bonds, and 'other' bonds, such as coporate or foreign bonds)                                              |
|   OTHMA  |      |         |                                                    Total value of other managed assets held by household, 2019 dollars (Includes trusts, annuities and managed investment accounts in which the household has equity interest)                                                    |
|  OTHFIN  |      |         |                Total value of other financial assets, 2019 dollars (Includes: loans from the household to someone else, future proceed from lawsuits, royalties, futures, non-public stock, deferred compensation, oil, gas, and mineral investments., cash n.e.c)                |
|  RETQLIQ |      |         |                                                              Total value of quasi-liquid held by household, 2019 dollars (Includes IRAs, Keoghs, thrift-type accounts, and future and current account-type pensions)                                                              |
|   AGECL  |      |         |                                                                                         Age group of the reference person (1 = <35, 2 = 35-44, 3 = 45-54, 4 = 55-64, 5 = 65-74, 6 = >=75)                                                                                         |
|   HHSEX  |      |         |                                                                                                            Gender of household reference person (1 = male, 2 = female)                                                                                                            |
|   EDCL   |      |         |                                                                     Education category of reference person (1 = no high school diploma/GED, 2 = high school diploma/GED, 3 = some college, 4 = college degree)                                                                    |
|   KIDS   |      |         |                                                                                                                       Total number of children in household                                                                                                                       |
|  MARRIED |      |         |                                                                                 Marital status of reference person (1 = married/living with partner, 2 = neither married nor living with partner)                                                                                 |
|  HOUSECL |      |         |                                                                                     Home-ownership category of household (1 = owns ranch/farm/mobile home/house/condo/coop/etc., 2 = otherwise                                                                                    |
|  OCCAT2  |      |         |                          Occupation classification for reference person (1 = managerial/professional, 2 = technical/sales/services, 3 = other (incl. production/craft/repair workers, operators, laborers, farmers, foresters, fishers), 4 = not working)                         |
|  LIFECL  |      |         |       Lifecycle of reference person (1 = under 55 + not married/LWP + no children, 2 = under 55 + married/LWP + no children, 3 = under 55 + married/LWP + children, 4 = under 55 + not married/LWP + children, 5 = 55 or older and working, 6 = 55 or older and not working)      |
|  INCCAT  |      |         |                                                                                        Income percentile groups (1 = 0-20, 2 = 20-39.9, 3 = 40-59.9, 4 = 60-79.9, 5 = 80-89.9, 6 = 90-100)                                                                                        |
|   NWCAT  |      |         |                                                                                            Net worth percentile groups (1 = 0-24.9, 2 = 25-49.9, 3 = 50-74.9, 4 = 75-89.9, 5 = 90-100)                                                                                            |
|  WSAVED  |      |         |                                                                      Spent more/same/less than income in past year (1 = spending exceeded income, 2 = spending equaled income, 3 = spending less than income)                                                                     |
| SPENDMOR |      |         |                                                 Respondent would spend more if assets appreciated in value (1 = agree strongly, 2 = agree somewhat, 3 = neither agree nor disagree, 4 = disagree somewhat, 5 = disagree strongly)                                                 |
|   KNOWL  |      |         |                                                                                      Respondent's knowledge about personal finances (-1 is not at all knowledgeable. 10 is very knowledgeale)                                                                                     |


# 2. Approach

1. The 15 variables measuring asset values were used to calculate an a posteriori measure of risk tolerance
   - Risk tolerance calculated as the ratio of risky to (risky + risk-free) assets
   - Risk tolerance is the target value to predict
2. The 13 various demographic, financial and behavioural attributes used as predictor variables to predict risk tolerance
    - The models used to predict risk tolerance are:
       - Linear Regression (regularized and non-regularized)
       - k-Nearest Neighbours Regression
       - Support Vector Regression
       - Decision Tree Regression
       - Random Forest Regression
       - Extra Trees Regression
       - Gradient Boosted Regression (including AdaBoost)
       - Multi-layer Perceptron Regression
       - Sequential Deep Neural Network
3. For DQN, a random sample of adjusting closing prices of 10 equities from start of 2020 to present day used to train an RL Agent class (contains variable and member functions that perform Q-learning) in a StockEnvironment class (simulation environment for training the agent)
   - Deep Q-network is a value-based method that combines deep learning (using an ANN) with Q-learning, which sets the learning objective to optimize the estimates of Q-value
   - The deep Q-learning algorithm approximates the Q-values by learning a set of weights, θ, of a multilayered deep Q-network that maps states to actions and learns to find the right weights by iteratively adjusting them to maximize rewards (sharpe ratio)
   - Maximizing this reward will lead to portfolio allocation weights that maximize the return of the portfolio for a given level of risk
   - The ANN applies gradient descent to minimize a loss function (essentially MSE) which is the squared difference between the DQN’s estimate of the target and its estimate of the Q-value of the current state-action pair, Q(s,a:θ)
   - We don't directly evaluate the value of the loss function, but since the Q-value is the expected reward for the state-action pair following a policy π, when the algorithm iteratively converges to the optimal Q-value, it learns an optimal policy which is how to act to maximize the return/reward in every state, which is the sharpe ratio of the portfolio and also the portfolio allocation weights


# 3. Models Performance 

**Summary of Models' RMSE and R2 Scores:**

|               Model              | Train RMSE | Cross-validated RMSE | Test RMSE | Train R2 |  Test R2 |
|:--------------------------------:|:----------:|:--------------------:|:---------:|:--------:|:--------:|
|         Linear Regression        |   0.31748  |        0.31767       |  0.31909  |  0.34745 |  0.34220 |
|      Lasso Linear Regression     |   0.39302  |        0.39303       |  0.39346  |  0.00000 | -0.00014 |
|   ElasticNet Linear Regression   |   0.39302  |        0.39303       |  0.39346  |  0.00000 | -0.00014 |
|  k-Nearest Neighbors Regression  |   0.1316   |        0.20690       |  0.18929  |  0.88788 |  0.76851 |
|     Decision Tree Regression     |   0.08334  |        0.12626       |  0.12771  |  0.95504 |  0.89462 |
|     Support Vector Regression    |   0.30512  |        0.30960       |  0.31210  |  0.39729 |  0.37073 |
| Multi-layer Perceptron Regression|   0.29200  |        0.30258       |  0.30091  |  0.44802 |  0.41505 |
|        AdaBoost Regression       |   0.32353  |        0.32280       |  0.32449  |  0.32238 |  0.31975 |
|    Gradient Boosted Regression   |   0.30471  |        0.30688       |  0.30870  |  0.3989  |  0.38435 |
|     Random Forest Regression     |   0.08717  |        0.12156       |  0.12056  |  0.95081 |  0.90610 |
|      Extra Trees Regression      |   0.08334  |        0.11418       |  0.11539  |  0.95504 |  0.91398 |
|       Keras Neural Network       |   0.33789  |        0.35133       |  0.33981  | -0.11417 | -0.11547 |

**Summary of Best Model's RMSE and R2 Scores:**

|          Model         | n_estimators |   criterion   | max_depth | Train RMSE | Cross-validated RMSE | Test RMSE | Train R2 | Test R2 |
|:----------------------:|:------------:|:-------------:|:---------:|:----------:|:--------------------:|:---------:|:--------:|:-------:|
| Extra Trees Regression |      50      | squared_error |     25    |   0.08337  |        0.11407       |  0.11519  |  0.95501 | 0.91427 |

**Hyperparameters Grid Searched for Best Model**

1. Extra Trees Regression:
    - n_estimators: [10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    - criterion: ['squared_error', 'friedman_mse']
    - max_depth: [3, 5, 10, 25, 50, 75, 100]


# 4. Conclusion

1. Extra Trees Regressor used as best model to predict risk tolerance - although model is slightly overfitted
2. Mean variance optimized (MVO) portfolio weights can be determined using quadratic programming 
3. Reinforcement learning can be used to dynamically change portfolio allocation weights automatically
4. The robo advisor dashboard incorporating risk tolerance prediction and MVO portfolio weights has been deployed on Heroku at https://wes-roboadvisor-dashboard-082860bb1de2.herokuapp.com/


# 5. Recommendations

1. Allow for the selection of more features to predict risk tolerance. Include for the selection of continuous (ratio) values instead of ordinal values for features such as age, income, networth, etc.
2. Allow for more asset selection choices in the dashboard. US equities (NASDAQ, NYSE); non-US equities (FTSE100, CAC40, DAX); fixed income securities; cryptocurrencies (BTC, ETH, BNB)
3. Integrate DQN RL agent with an stock exchange’s API to automate portfolio rebalancing (not the same as an algorithmic trader as algo traders make use of high-frequency trading)


# 6. References

1. https://sda.berkeley.edu/sdaweb/analysis/?dataset=scfcomb2019
2. https://www.federalreserve.gov/econres/scfindex.htm
3. https://sda.berkeley.edu/sdaweb/docs/scfcomb2019/DOC/hcbkx01.htm#1.HEADING
4. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=95489
5. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4218181
6. https://books.google.com.sg/books?id=xS2pDAAAQBAJ&printsec=frontcover&redir_esc=y#v=onepage&q&f=false
7. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1669015
8. https://www.mckinsey.com/industries/financial-services/our-insights/reimagining-customer-engagement-for-the-ai-bank-of-the-future