"""Contains utility functions for financial calculations and portfolio optimization.

This module implements the mean variance optimization (MVO) and other financial
calculations used in the robo-advisor, based on the original notebook implementation.
"""
import numpy as np
import pandas as pd
import cvxopt as opt
from cvxopt import solvers
from typing import List, Tuple, Optional
import warnings

# Suppress CVXOPT output
solvers.options['show_progress'] = False
warnings.filterwarnings('ignore')


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculates percentage returns from price data.
    
    Args:
        prices (pd.DataFrame): DataFrame with price data.
        
    Returns:
        pd.DataFrame: DataFrame with percentage returns.
    """
    return prices.pct_change().dropna()


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculates the Sharpe ratio of a returns series.
    
    Args:
        returns (pd.Series): Series of returns.
        risk_free_rate (float): Risk-free rate (default: 0.0).
        
    Returns:
        float: Sharpe ratio.
    """
    excess_returns = returns - risk_free_rate
    if excess_returns.std() == 0:
        return 0.0
    return excess_returns.mean() / excess_returns.std()


def calculate_portfolio_metrics(weights: np.ndarray, returns: pd.DataFrame) -> dict:
    """Calculates portfolio metrics given weights and returns.
    
    Args:
        weights (np.ndarray): Portfolio weights.
        returns (pd.DataFrame): Asset returns.
        
    Returns:
        dict: Dictionary containing portfolio metrics.
    """
    portfolio_returns = returns.dot(weights)
    
    metrics = {
        'mean_return': portfolio_returns.mean(),
        'volatility': portfolio_returns.std(),
        'sharpe_ratio': calculate_sharpe_ratio(portfolio_returns),
        'cumulative_return': (1 + portfolio_returns).prod() - 1
    }
    
    return metrics


def mean_variance_optimization(risk_tolerance: float, tickers: List[str], 
                              price_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Performs mean variance optimization for portfolio allocation.
    
    This function implements the exact MVO logic from the original notebook,
    using CVXOPT for quadratic programming optimization.
    
    Args:
        risk_tolerance (float): Risk tolerance value (0 to 1).
        tickers (List[str]): List of stock tickers.
        price_data (pd.DataFrame): DataFrame with historical price data.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Allocation weights and portfolio performance.
    """
    print(f"Performing mean variance optimization for risk tolerance: {risk_tolerance}")
    print(f"Assets: {tickers}")
    
    # Select the subset of assets based on the provided tickers
    assets_selected = price_data.loc[:, tickers].copy()
    
    # Calculate returns and convert to numpy array
    returns = assets_selected.pct_change().dropna(axis=0)
    return_vec = np.array(returns).T
    
    # Number of assets
    n = len(return_vec)
    
    # Convert the array of returns into a NumPy matrix
    returns_matrix = np.asmatrix(return_vec)
    
    # Calculate one minus the risk tolerance (for optimization)
    mus = 1 - risk_tolerance
    
    # Convert return and covariance data to CVXOPT matrices for optimization
    S = opt.matrix(np.cov(return_vec))  # Covariance matrix of returns
    pbar = opt.matrix(np.mean(return_vec, axis=1))  # Mean returns
    
    # Create constraint matrices for the optimization problem
    G = -opt.matrix(np.eye(n))   # Negative n x n identity matrix (for non-negativity)
    h = opt.matrix(0.0, (n, 1))  # Lower bound of 0 for all weights
    A = opt.matrix(1.0, (1, n))  # Sum constraint matrix
    b = opt.matrix(1.0)          # Sum constraint (weights sum to 1)
    
    try:
        # Use quadratic programming to calculate efficient portfolio weights
        # Minimize: (1/2) * w^T * mus * S * w - pbar^T * w
        # Subject to: sum(w) = 1, w >= 0
        portfolios = solvers.qp(mus * S, -pbar, G, h, A, b)
        
        # Extract the allocation weights from the optimization result
        weights = np.array(portfolios['x']).flatten()
        
        # Create a DataFrame to store the asset allocation weights
        allocation = pd.DataFrame(data=weights, index=tickers, columns=['Weight'])
        
        # Calculate portfolio performance
        portfolio_performance = calculate_portfolio_performance(
            assets_selected, weights, risk_tolerance
        )
        
        print(f"Optimization successful. Weights sum: {weights.sum():.6f}")
        
        return allocation, portfolio_performance
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Return equal weights as fallback
        equal_weights = np.ones(n) / n
        allocation = pd.DataFrame(data=equal_weights, index=tickers, columns=['Weight'])
        
        portfolio_performance = calculate_portfolio_performance(
            assets_selected, equal_weights, risk_tolerance
        )
        
        print("Using equal weights as fallback")
        return allocation, portfolio_performance


def calculate_portfolio_performance(assets_data: pd.DataFrame, weights: np.ndarray, 
                                  risk_tolerance: float, initial_value: float = 100.0) -> pd.DataFrame:
    """Calculates portfolio performance over time.
    
    This function replicates the portfolio performance calculation from the notebook.
    
    Args:
        assets_data (pd.DataFrame): Historical price data for assets.
        weights (np.ndarray): Portfolio weights.
        risk_tolerance (float): Risk tolerance used.
        initial_value (float): Initial portfolio value.
        
    Returns:
        pd.DataFrame: Portfolio performance over time.
    """
    # Calculate weighted returns
    returns_final = np.array(assets_data) * np.array(weights)
    
    # Calculate the sum of weighted returns for each time period
    portfolio_returns = np.sum(returns_final, axis=1)
    
    # Create a DataFrame to store the portfolio performance
    performance_df = pd.DataFrame(
        portfolio_returns, 
        index=assets_data.index, 
        columns=['Portfolio_Value']
    )
    
    # Normalize to start at initial_value
    performance_df = performance_df - performance_df.iloc[0] + initial_value
    
    return performance_df


def get_efficient_frontier(tickers: List[str], price_data: pd.DataFrame, 
                          num_points: int = 50) -> pd.DataFrame:
    """Calculates the efficient frontier for the given assets.
    
    Args:
        tickers (List[str]): List of stock tickers.
        price_data (pd.DataFrame): Historical price data.
        num_points (int): Number of points on the efficient frontier.
        
    Returns:
        pd.DataFrame: Efficient frontier data.
    """
    risk_tolerances = np.linspace(0.01, 0.99, num_points)
    frontier_data = []
    
    for rt in risk_tolerances:
        try:
            allocation, _ = mean_variance_optimization(rt, tickers, price_data)
            weights = allocation['Weight'].values
            
            # Calculate portfolio metrics
            returns = calculate_returns(price_data[tickers])
            portfolio_returns = returns.dot(weights)
            
            frontier_data.append({
                'risk_tolerance': rt,
                'expected_return': portfolio_returns.mean(),
                'volatility': portfolio_returns.std(),
                'sharpe_ratio': calculate_sharpe_ratio(portfolio_returns)
            })
            
        except Exception as e:
            print(f"Error calculating efficient frontier point for RT={rt}: {e}")
            continue
    
    return pd.DataFrame(frontier_data)


def calculate_correlation_matrix(price_data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Calculates correlation matrix for the given tickers.
    
    Args:
        price_data (pd.DataFrame): Historical price data.
        tickers (List[str]): List of stock tickers.
        
    Returns:
        pd.DataFrame: Correlation matrix.
    """
    returns = calculate_returns(price_data[tickers])
    return returns.corr()


def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """Calculates Value at Risk (VaR) for a returns series.
    
    Args:
        returns (pd.Series): Series of returns.
        confidence_level (float): Confidence level (default: 0.05 for 95% VaR).
        
    Returns:
        float: VaR value.
    """
    return np.percentile(returns, confidence_level * 100)


def calculate_max_drawdown(performance: pd.Series) -> float:
    """Calculates maximum drawdown for a performance series.
    
    Args:
        performance (pd.Series): Portfolio performance series.
        
    Returns:
        float: Maximum drawdown value.
    """
    peak = performance.expanding().max()
    drawdown = (performance - peak) / peak
    return drawdown.min()


def validate_weights(weights: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Validates that portfolio weights are valid.
    
    Args:
        weights (np.ndarray): Portfolio weights.
        tolerance (float): Tolerance for validation.
        
    Returns:
        bool: True if weights are valid.
    """
    # Check if weights sum to 1
    if abs(weights.sum() - 1.0) > tolerance:
        return False
    
    # Check if all weights are non-negative
    if np.any(weights < -tolerance):
        return False
    
    return True


def rebalance_portfolio(current_weights: np.ndarray, target_weights: np.ndarray,
                       threshold: float = 0.05) -> bool:
    """Determines if portfolio needs rebalancing.
    
    Args:
        current_weights (np.ndarray): Current portfolio weights.
        target_weights (np.ndarray): Target portfolio weights.
        threshold (float): Threshold for rebalancing trigger.
        
    Returns:
        bool: True if rebalancing is needed.
    """
    weight_differences = np.abs(current_weights - target_weights)
    max_difference = np.max(weight_differences)
    
    return max_difference > threshold


def calculate_transaction_costs(current_weights: np.ndarray, target_weights: np.ndarray,
                               cost_rate: float = 0.001) -> float:
    """Calculates transaction costs for rebalancing.
    
    Args:
        current_weights (np.ndarray): Current portfolio weights.
        target_weights (np.ndarray): Target portfolio weights.
        cost_rate (float): Transaction cost rate.
        
    Returns:
        float: Total transaction costs.
    """
    weight_changes = np.abs(current_weights - target_weights)
    total_turnover = np.sum(weight_changes)
    
    return total_turnover * cost_rate


def optimize_portfolio_with_transaction_costs(risk_tolerance: float, tickers: List[str],
                                            price_data: pd.DataFrame, 
                                            current_weights: Optional[np.ndarray] = None,
                                            cost_rate: float = 0.001) -> Tuple[pd.DataFrame, float]:
    """Optimizes portfolio considering transaction costs.
    
    Args:
        risk_tolerance (float): Risk tolerance value.
        tickers (List[str]): List of stock tickers.
        price_data (pd.DataFrame): Historical price data.
        current_weights (Optional[np.ndarray]): Current portfolio weights.
        cost_rate (float): Transaction cost rate.
        
    Returns:
        Tuple[pd.DataFrame, float]: Optimal allocation and transaction costs.
    """
    # Get optimal allocation without transaction costs
    allocation, _ = mean_variance_optimization(risk_tolerance, tickers, price_data)
    target_weights = allocation['Weight'].values
    
    if current_weights is not None:
        # Calculate transaction costs
        transaction_costs = calculate_transaction_costs(
            current_weights, target_weights, cost_rate
        )
        
        # Check if rebalancing is beneficial
        if not rebalance_portfolio(current_weights, target_weights):
            print("No rebalancing needed - differences below threshold")
            allocation = pd.DataFrame(
                data=current_weights, index=tickers, columns=['Weight']
            )
            return allocation, 0.0
    else:
        transaction_costs = 0.0
    
    return allocation, transaction_costs