"""Contains the Deep Q-Learning Agent and Stock Environment for Portfolio Optimization.

This module implements the exact reinforcement learning approach from the notebook,
using PyTorch instead of TensorFlow/Keras for neural network implementation.
"""
import numpy as np
import pandas as pd
import random
from typing import List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt

from ..config import RANDOM_STATE


class StockEnvironment:
    """Stock trading environment for reinforcement learning.
    
    This class replicates the exact StockEnvironment from the notebook,
    providing state data and reward calculations for the RL agent.
    """
    
    def __init__(self, data: pd.DataFrame, capital: float = 1e6):
        """Initialize a StockEnvironment instance.

        Args:
            data (pd.DataFrame): Stock price data.
            capital (float): Initial capital for the environment.
        """
        self.capital = capital
        self.data = data
        print(f"Initialized StockEnvironment with {len(data)} timesteps and {data.shape[1]} assets")

    def preprocess_state(self, state: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the state data if needed.
        
        Args:
            state (pd.DataFrame): Raw state data.
            
        Returns:
            pd.DataFrame: Preprocessed state data.
        """
        return state

    def get_state(self, t: int, lookback: int, is_cov_matrix: bool = True, 
                  is_raw_time_series: bool = False) -> pd.DataFrame:
        """Get the state data for a given time and lookback period.

        Args:
            t (int): Current time index.
            lookback (int): Number of historical time periods to consider.
            is_cov_matrix (bool): If True, return covariance matrix of returns.
            is_raw_time_series (bool): If True, return historical return data.

        Returns:
            pd.DataFrame: State data (covariance matrix or raw time series).
        """
        assert lookback <= t, f"Lookback {lookback} cannot be greater than time {t}"

        # Extract the historical data for the specified lookback period
        decision_making_state = self.data.iloc[t - lookback:t].copy()

        # If raw time series is requested, return preprocessed data
        if is_raw_time_series:
            return self.preprocess_state(decision_making_state)

        # If covariance matrix is requested, calculate and return it
        if is_cov_matrix:
            # Calculate percentage change in asset prices and drop NaN values
            decision_making_state = decision_making_state.pct_change().dropna()
            # Return the covariance matrix
            return decision_making_state.cov()

    def get_reward(self, action: np.ndarray, action_t: int, reward_t: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the reward based on the action taken and observed outcomes.

        Args:
            action (np.ndarray): Portfolio allocation weights for different assets.
            action_t (int): The time index when the action is taken.
            reward_t (int): The time index when the reward is observed.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Weighted returns and Sharpe ratios.
        """
        def portfolio(returns: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
            """Calculate portfolio statistics for given returns and weights.

            Args:
                returns (pd.DataFrame): Time series of asset returns.
                weights (np.ndarray): Portfolio allocation weights.

            Returns:
                np.ndarray: Portfolio statistics [return, volatility, Sharpe ratio].
            """
            weights = np.array(weights)
            rets = returns.mean()
            covs = returns.cov()
            P_ret = np.sum(rets * weights)
            P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))
            P_sharpe = P_ret / P_vol if P_vol != 0 else 0
            return np.array([P_ret, P_vol, P_sharpe])

        # Extract data for the specified time period
        data_period = self.data[action_t:reward_t].copy()
        weights = action

        # Calculate percentage returns for the selected time period
        returns = data_period.pct_change().dropna()

        # Calculate the Sharpe ratio for the portfolio
        sharpe = portfolio(returns, weights)[-1]
        
        # Create an array of Sharpe ratios with the same length as the number of assets
        sharpe_array = np.array([sharpe] * len(self.data.columns))
        
        # Calculate the weighted returns
        weighted_returns = np.dot(returns.values, weights)
        
        return weighted_returns, sharpe_array
    
    def calculate_dynamic_reward(self, returns: np.ndarray, weights: np.ndarray, 
                               return_weight: float = 0.5, risk_weight: float = 0.5,
                               market_regime: str = "Stable") -> float:
        """Calculate reward with configurable risk/return balance."""
        
        portfolio_return = np.mean(returns)
        portfolio_volatility = np.std(returns)
        
        # Base reward components
        return_component = portfolio_return
        risk_component = -portfolio_volatility  # Negative because we want to minimize risk
        
        # Market regime adjustments
        regime_multipliers = {
            "🔴 High Volatility/Bear Market": {"return": 0.7, "risk": 1.3},  # Emphasize risk management
            "🟢 Low Volatility/Bull Market": {"return": 1.3, "risk": 0.7},   # Emphasize returns
            "🟡 High Volatility/Uncertain": {"return": 0.9, "risk": 1.1},   # Slight risk focus
            "🔵 Moderate Volatility/Stable": {"return": 1.0, "risk": 1.0}    # Balanced
        }
        
        # Extract regime key for lookup
        for key in regime_multipliers.keys():
            if key.split()[0] in market_regime:
                multipliers = regime_multipliers[key]
                break
        else:
            multipliers = {"return": 1.0, "risk": 1.0}
        
        # Calculate final reward
        adjusted_return_weight = return_weight * multipliers["return"]
        adjusted_risk_weight = risk_weight * multipliers["risk"]
        
        reward = (adjusted_return_weight * return_component + 
                  adjusted_risk_weight * risk_component)
        
        return reward


class DQNNetwork(nn.Module):
    """Deep Q-Network implementation using PyTorch."""
    
    def __init__(self, input_shape: Tuple[int, int], portfolio_size: int, action_size: int = 3):
        """Initialize the DQN network.
        
        Args:
            input_shape: Shape of input (portfolio_size, portfolio_size)
            portfolio_size: Number of assets in portfolio
            action_size: Number of actions per asset (hold, buy, sell)
        """
        super(DQNNetwork, self).__init__()
        
        self.portfolio_size = portfolio_size
        self.action_size = action_size
        self.input_size = input_shape[0] * input_shape[1]
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Dense layers
        self.fc1 = nn.Linear(self.input_size, 100)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 50)
        self.dropout2 = nn.Dropout(0.5)
        
        # Output layers for each asset
        self.asset_heads = nn.ModuleList([
            nn.Linear(50, action_size) for _ in range(portfolio_size)
        ])
        
    def forward(self, x):
        """Forward pass through the network."""
        x = self.flatten(x)
        x = F.elu(self.fc1(x))
        x = self.dropout1(x)
        x = F.elu(self.fc2(x))
        x = self.dropout2(x)
        
        # Generate predictions for each asset
        outputs = []
        for head in self.asset_heads:
            outputs.append(head(x))
        
        return outputs


class Agent:
    """Deep Q-Learning Agent for portfolio management using PyTorch.
    
    This class replicates the exact Agent implementation from the notebook,
    but uses PyTorch instead of TensorFlow/Keras.
    """
    
    def __init__(self, portfolio_size: int, is_eval: bool = False, 
                 allow_short: bool = True, model_name: str = ''):
        """Initialize an Agent instance for portfolio management.

        Args:
            portfolio_size (int): Number of assets in the portfolio.
            is_eval (bool): Flag indicating whether the agent is in evaluation mode.
            allow_short (bool): Flag indicating whether short-selling is allowed.
            model_name (str): Name of the saved model to load (if in evaluation mode).
        """
        self.portfolio_size = portfolio_size
        self.allow_short = allow_short
        self.input_shape = (portfolio_size, portfolio_size)
        self.action_size = 3  # hold, buy or sell
        self.model_name = model_name
        
        self.memory4replay = []  # Memory for experience replay
        self.is_eval = is_eval  # Flag for evaluation mode

        # Hyperparameters (exactly as in notebook)
        self.alpha = 0.5  # Learning rate
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration factor
        self.epsilon_min = 0.01  # Minimum exploration factor
        self.epsilon_decay = 0.99  # Exploration decay rate
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize or load the model
        if is_eval and model_name:
            self.model = self._load_model(model_name)
            print(f"Loaded model: {model_name}")
        else:
            self.model = self._create_model()
            print("Created new PyTorch model for training")

    def _create_model(self) -> DQNNetwork:
        """Build and initialize the deep learning model for the agent.

        Returns:
            DQNNetwork: PyTorch neural network model.
        """
        model = DQNNetwork(self.input_shape, self.portfolio_size, self.action_size)
        model = model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        return model

    def _load_model(self, model_name: str) -> DQNNetwork:
        """Load a saved PyTorch model.
        
        Args:
            model_name: Name of the model file to load (can be full path or just filename)
            
        Returns:
            DQNNetwork: Loaded PyTorch model
        """
        # Check if model_name is already a full path
        if Path(model_name).is_absolute() and Path(model_name).exists():
            model_path = Path(model_name)
            print(f"Loading model from absolute path: {model_path}")
        else:
            # Try different relative path combinations
            model_path = Path(f"../data/{model_name}")
            if not model_path.exists():
                model_path = Path(f"../data/output/{model_name}")
            if not model_path.exists():
                # Try adding .pth extension if not present
                if not model_name.endswith('.pth'):
                    model_path = Path(f"../data/output/{model_name}.pth")
            
            print(f"Loading model from relative path: {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = DQNNetwork(self.input_shape, self.portfolio_size, self.action_size)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        
        # Initialize optimizer for potential further training
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        return model

    def nn_pred_to_weights(self, pred: List[torch.Tensor], 
                          allow_short: bool = True) -> Tuple[np.ndarray, Optional[float], float]:
        """Convert neural network predictions to portfolio weights.

        Args:
            pred (List[torch.Tensor]): Predictions from the neural network.
            allow_short (bool): Whether short selling is allowed.

        Returns:
            Tuple[np.ndarray, Optional[float], float]: Portfolio weights, saved minimum, saved sum.
        """
        # Initialize an array to store asset weights
        weights = np.zeros(len(pred))
        
        # Convert predictions to numpy and find argmax
        raw_weights = []
        for p in pred:
            raw_weights.append(torch.argmax(p, dim=-1).cpu().numpy().item())

        # Initialize a variable to save the minimum weight (for short selling)
        saved_min = None

        # Loop through the assets and assign weights based on predictions
        for e, r in enumerate(raw_weights):
            if r == 0:  # Hold
                weights[e] = 0
            elif r == 1:  # Buy
                weights[e] = abs(pred[e][0][r].cpu().numpy().item())
            else:  # Sell
                weights[e] = -abs(pred[e][0][r].cpu().numpy().item())

        # Adjust weights to ensure non-negative values if short selling is not allowed
        if not allow_short:
            weights += abs(np.min(weights))  # Ensure all weights are non-negative
            saved_min = abs(np.min(weights))  # Save the absolute minimum weight
            saved_sum = np.sum(weights)  # Calculate the sum of weights
        else:
            saved_sum = np.sum(np.abs(weights))

        # Normalize weights to ensure they sum to 1
        if saved_sum > 0:
            weights /= saved_sum
        else:
            weights = np.ones(len(weights)) / len(weights)  # Equal weights fallback
        
        return weights, saved_min, saved_sum

    def act(self, state: pd.DataFrame) -> Tuple[np.ndarray, Optional[float], float]:
        """Select an action (portfolio allocation weights) based on the current state.

        Args:
            state (pd.DataFrame): The current state containing asset covariance matrix.

        Returns:
            Tuple[np.ndarray, Optional[float], float]: Portfolio weights, saved min, saved sum.
        """
        if not self.is_eval and random.random() <= self.epsilon:
            # Generate a random portfolio allocation with a normal distribution
            weights = np.random.normal(0, 1, size=(self.portfolio_size,))
            
            saved_min = None
            
            # Adjust weights for non-negative values if short selling is not allowed
            if not self.allow_short:
                weights += abs(np.min(weights))  # Ensure all weights are non-negative
                saved_min = abs(np.min(weights))  # Save the absolute minimum weight
                
            saved_sum = np.sum(weights)  # Calculate the sum of weights
            if saved_sum > 0:
                weights /= saved_sum  # Normalize weights to sum to 1
            else:
                weights = np.ones(len(weights)) / len(weights)  # Equal weights fallback
            return weights, saved_min, saved_sum

        # Use the neural network to make predictions and convert them to weights
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state.values).unsqueeze(0).to(self.device)
            pred = self.model(state_tensor)
        
        return self.nn_pred_to_weights(pred, self.allow_short)

    def expReplay(self, batch_size: int) -> None:
        """Experience replay to update the agent's Q-function using stored memories.

        Args:
            batch_size (int): Size of the batch used for updating the Q-function.
        """
        def weights_to_nn_preds_with_reward(action_weights: np.ndarray, reward: np.ndarray, 
                                           Q_star: List[torch.Tensor] = None) -> List[torch.Tensor]:
            """Calculate Q-values with rewards based on action weights and Q* values."""
            if Q_star is None:
                Q_star = [torch.zeros(1, self.action_size).to(self.device) for _ in range(self.portfolio_size)]
                
            Q = [torch.zeros(1, self.action_size).to(self.device) for _ in range(self.portfolio_size)]
            
            for i in range(self.portfolio_size):
                if action_weights[i] == 0:  # Hold
                    Q[i][0][0] = reward[i] + self.gamma * torch.max(Q_star[i])
                elif action_weights[i] > 0:  # Buy
                    Q[i][0][1] = reward[i] + self.gamma * torch.max(Q_star[i])
                else:  # Sell
                    Q[i][0][2] = reward[i] + self.gamma * torch.max(Q_star[i])
            return Q

        def restore_Q_from_weights_and_stats(action: Tuple[np.ndarray, Optional[float], float]) -> np.ndarray:
            """Restore the action weights from the provided action tuple."""
            action_weights, action_min, action_sum = action[0], action[1], action[2]
            action_weights = action_weights * action_sum
            if action_min is not None:
                action_weights = action_weights - action_min
            return action_weights

        self.model.train()
        
        for (s, s_, action, reward, done) in self.memory4replay:
            # Restore action weights from the action tuple
            action_weights = restore_Q_from_weights_and_stats(action)

            # Convert states to tensors
            s_tensor = torch.FloatTensor(s.values).unsqueeze(0).to(self.device)
            s_tensor_ = torch.FloatTensor(s_.values).unsqueeze(0).to(self.device)

            # Calculate Q-values using Q-learning
            Q_learned_value = weights_to_nn_preds_with_reward(action_weights, reward)

            if not done:
                # Predict Q-values for the next state using the neural network
                with torch.no_grad():
                    Q_star = self.model(s_tensor_)
                # Update Q-learned value with Q_star if not in the terminal state
                Q_learned_value = weights_to_nn_preds_with_reward(action_weights, reward, Q_star)

            # Predict Q-values for the current state using the neural network
            Q_current_value = self.model(s_tensor)
            
            # Calculate target Q-values
            Q_target = []
            for i in range(self.portfolio_size):
                q_current = Q_current_value[i]
                q_learned = Q_learned_value[i]
                q_target = q_current * (1 - self.alpha) + q_learned * self.alpha
                Q_target.append(q_target)

            # Calculate loss and update model
            self.optimizer.zero_grad()
            loss = sum([self.criterion(Q_current_value[i], Q_target[i]) for i in range(self.portfolio_size)])
            loss.backward()
            self.optimizer.step()

        # Update exploration probability (epsilon) with epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath: Path) -> None:
        """Save the PyTorch model to file.
        
        Args:
            filepath: Path where to save the model
        """
        torch.save(self.model.state_dict(), filepath)
        print(f"PyTorch model saved to: {filepath}")


def train_rl_agent(selected_data: pd.DataFrame, window_size: int = 90, 
                   episode_count: int = 10, batch_size: int = 32, 
                   rebalance_period: int = 45, mpt_weights: np.ndarray = None,
                   save_path: Optional[Path] = None,
                   return_weight: float = 0.5, risk_weight: float = 0.5,  # ADDED THESE
                   market_regime: str = "Stable") -> Tuple[Agent, dict]:    # ADDED THIS
    """Train the RL agent using PyTorch with configurable objectives.

    Args:
        selected_data (pd.DataFrame): Selected stock price data.
        window_size (int): Number of market days to consider for state evaluation.
        episode_count (int): Number of training episodes.
        batch_size (int): Batch size for memory replay.
        rebalance_period (int): Period for portfolio rebalancing.
        mpt_weights (np.ndarray, optional): MPT benchmark weights for comparison.
        save_path (Path, optional): Path to save the trained model.
        return_weight (float): Weight for return component in reward (0.0 to 1.0).
        risk_weight (float): Weight for risk component in reward (0.0 to 1.0).
        market_regime (str): Current market regime for dynamic adjustments.

    Returns:
        Tuple[Agent, dict]: Trained agent and training history.
    """
    print(f"Training RL agent with PyTorch - {episode_count} episodes...")
    print(f"Objective: {return_weight:.0%} returns, {risk_weight:.0%} risk management")
    print(f"Market regime: {market_regime}")
    print(f"Window size: {window_size}, Batch size: {batch_size}, Rebalance period: {rebalance_period}")
    
    # Initialize agent and environment
    n_assets = selected_data.shape[1]
    agent = Agent(n_assets)
    env = StockEnvironment(selected_data)
    
    # Training history
    training_history = {
        'episode_returns': [],
        'episode_sharpe_ratios': [],
        'mpt_returns': [],
        'actions_history': []
    }
    
    # Training loop
    for e in range(1, episode_count + 1):
        agent.is_eval = False
        data_length = len(env.data)
        
        # Episode-specific tracking
        returns_history = []
        returns_history_mpt = []
        rewards_history = []
        rewards_history_mpt = []
        actions_to_show = []
        
        print(f"Episode {e}/{episode_count}, epsilon: {agent.epsilon:.4f}")

        # Initialize the state for the current episode
        s = env.get_state(np.random.randint(window_size + 1, data_length - window_size - 1), window_size)
        
        # Episode loop
        for t in range(window_size, data_length, rebalance_period):
            # Calculate the start date of the current period
            date1 = t - rebalance_period
            
            # Get the current state and select an action using the agent
            s_ = env.get_state(t, window_size)
            action = agent.act(s_)
            
            actions_to_show.append(action[0])
            
            # Calculate portfolio returns and rewards based on the selected action
            weighted_returns, reward = env.get_reward(action[0], date1, t)
            
            # ADDED: Apply dynamic reward if custom objective is selected
            if return_weight != 0.5 or risk_weight != 0.5:
                # Use dynamic reward calculation
                dynamic_reward = env.calculate_dynamic_reward(
                    weighted_returns, action[0], return_weight, risk_weight, market_regime
                )
                # Override the Sharpe ratio reward with our custom reward
                reward = np.array([dynamic_reward] * len(reward))
                if e == 1 and t == window_size:  # Print once per training session
                    print(f"Using dynamic reward system: return_weight={return_weight}, risk_weight={risk_weight}")
            
            if mpt_weights is not None:
                # Calculate returns and rewards for MPT weighted portfolio
                weighted_returns_mpt, reward_mpt = env.get_reward(mpt_weights, date1, t)
                rewards_history_mpt.append(reward_mpt)
                returns_history_mpt.extend(weighted_returns_mpt)
            
            rewards_history.append(reward)
            returns_history.extend(weighted_returns)

            # Check if the episode is done
            done = (t >= data_length - rebalance_period)
            
            # Append experience to agent's memory
            agent.memory4replay.append((s, s_, action, reward, done))
            
            # Perform experience replay when memory is sufficient
            if len(agent.memory4replay) >= batch_size:
                agent.expReplay(batch_size)
                agent.memory4replay = []
                
            s = s_

        # Calculate cumulative returns for the episode
        rl_result = np.array(returns_history).cumsum()
        
        # Store training metrics
        training_history['episode_returns'].append(rl_result)
        training_history['actions_history'].append(actions_to_show)
        
        if len(returns_history) > 0:
            sharpe_ratio = np.mean(returns_history) / np.std(returns_history) if np.std(returns_history) > 0 else 0
            training_history['episode_sharpe_ratios'].append(sharpe_ratio)
        
        if mpt_weights is not None:
            mpt_result = np.array(returns_history_mpt).cumsum()
            training_history['mpt_returns'].append(mpt_result)
        
        # Print episode summary
        final_return = rl_result[-1] if len(rl_result) > 0 else 0
        print(f"Episode {e} completed. Final return: {final_return:.4f}")

    # Save the trained model
    if save_path:
        model_path = save_path / f"model_rl{episode_count}.pth"
        agent.save_model(model_path)

    print("RL agent training completed!")
    return agent, training_history


def evaluate_rl_agent(selected_data: pd.DataFrame, model_name: str, 
                     mpt_weights: np.ndarray, window_size: int = 90,
                     rebalance_period: int = 45) -> Tuple[dict, Agent]:
    """Evaluate the trained RL agent against MPT benchmark.

    Args:
        selected_data (pd.DataFrame): Selected stock price data.
        model_name (str): Name of the saved model to load.
        mpt_weights (np.ndarray): MPT benchmark weights.
        window_size (int): Window size for state evaluation.
        rebalance_period (int): Rebalancing period.

    Returns:
        Tuple[dict, Agent]: Evaluation results and the agent.
    """
    print(f"Evaluating RL agent: {model_name}")
    
    # Initialize agent in evaluation mode
    n_assets = selected_data.shape[1]
    agent = Agent(n_assets, is_eval=True, model_name=model_name)
    env = StockEnvironment(selected_data)
    
    data_length = len(env.data)
    
    # Evaluation tracking
    actions_mpt, actions_rl = [], []
    returns_mpt, returns_rl = [], []

    # Evaluation loop
    for t in range(window_size, data_length, rebalance_period):
        date1 = t - rebalance_period
            
        # Get the state for the current time step
        s_ = env.get_state(t, window_size)
        
        # Select an action using the agent
        action = agent.act(s_)

        # Calculate portfolio returns for both portfolios
        weighted_returns, _ = env.get_reward(action[0], date1, t)
        weighted_returns_mpt, _ = env.get_reward(mpt_weights, date1, t)

        # Store results and actions
        returns_mpt.append(weighted_returns_mpt.tolist())
        returns_rl.append(weighted_returns.tolist())
        
        actions_mpt.append(mpt_weights)
        actions_rl.append(action[0])

    # Flatten the results
    returns_mpt_flat = [item for sublist in returns_mpt for item in sublist]
    returns_rl_flat = [item for sublist in returns_rl for item in sublist]
    
    # Calculate performance metrics
    evaluation_results = {
        'returns_mpt': returns_mpt_flat,
        'returns_rl': returns_rl_flat,
        'actions_mpt': actions_mpt,
        'actions_rl': actions_rl,
        'cumulative_mpt': np.array(returns_mpt_flat).cumsum(),
        'cumulative_rl': np.array(returns_rl_flat).cumsum()
    }
    
    print("RL agent evaluation completed!")
    return evaluation_results, agent


def calculate_performance_stats(returns: List[float]) -> dict:
    """Calculate performance statistics for a returns series.

    Args:
        returns (List[float]): List of returns.

    Returns:
        dict: Dictionary containing performance statistics.
    """
    returns_array = np.array(returns)
    
    # Calculate Sharpe ratio (annualized)
    mean_return = returns_array.mean()
    std_return = returns_array.std()
    sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return != 0 else 0
    
    # Calculate other metrics
    cumulative_return = returns_array.cumsum()[-1] if len(returns_array) > 0 else 0
    volatility = std_return
    
    return {
        'mean_return': mean_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'cumulative_return': cumulative_return
    }


def plot_training_results(training_history: dict, save_path: Optional[Path] = None) -> None:
    """Plot training results from the RL agent.

    Args:
        training_history (dict): Training history from train_rl_agent.
        save_path (Path, optional): Path to save the plots.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot cumulative returns
    for i, returns in enumerate(training_history['episode_returns']):
        axes[0].plot(returns, alpha=0.7, label=f'Episode {i+1}')
    
    axes[0].set_title('RL Agent Training - Cumulative Returns by Episode')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Cumulative Returns')
    axes[0].legend()
    
    # Plot Sharpe ratios
    episodes = range(1, len(training_history['episode_sharpe_ratios']) + 1)
    axes[1].plot(episodes, training_history['episode_sharpe_ratios'], 'bo-')
    axes[1].set_title('RL Agent Training - Sharpe Ratio by Episode')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Sharpe Ratio')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'rl_training_results.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_evaluation_results(evaluation_results: dict, save_path: Optional[Path] = None) -> None:
    """Plot evaluation results comparing RL agent to MPT benchmark."""
    plt.figure(figsize=(12, 6))
    
    # Plot cumulative returns comparison
    plt.plot(evaluation_results['cumulative_mpt'], 
             label='MPT Benchmark', color='grey', linestyle='--', linewidth=2)
    plt.plot(evaluation_results['cumulative_rl'], 
             label='Deep RL Portfolio', color='black', linestyle='-', linewidth=2)
    
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Cumulative Returns', fontsize=14)
    plt.title('Comparison of MVO Portfolio vs Deep RL Portfolio', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        # Fix: Use save_path directly, not as a directory
        if save_path.suffix == '.png':
            plot_path = save_path
        else:
            plot_path = save_path / 'rl_vs_mpt_comparison.png'
        
        # Ensure parent directory exists
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {plot_path}")
    
    plt.show()