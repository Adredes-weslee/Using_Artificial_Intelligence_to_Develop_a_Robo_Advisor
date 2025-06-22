"""Market regime detection and analysis utilities."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def calculate_max_drawdown(price_series: pd.Series) -> float:
    """Calculate maximum drawdown of a price series."""
    peak = price_series.expanding().max()
    drawdown = (price_series - peak) / peak
    return drawdown.min()


def detect_market_regime(price_data: pd.DataFrame, lookback_days: int = 252) -> str:
    """Detect current market regime for dynamic risk adjustment."""
    
    # Calculate market indicators
    recent_data = price_data.tail(lookback_days)
    returns = recent_data.pct_change().dropna()
    
    # Market regime indicators
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    trend = (recent_data.iloc[-1] / recent_data.iloc[0] - 1) * (252/lookback_days)  # Annualized return
    max_drawdown = calculate_max_drawdown(recent_data.iloc[:, 0])  # Use first column as proxy
    
    # Regime classification
    if volatility > 0.25 and max_drawdown < -0.15:
        return "🔴 High Volatility/Bear Market"
    elif volatility < 0.15 and trend > 0.1:
        return "🟢 Low Volatility/Bull Market" 
    elif volatility > 0.20:
        return "🟡 High Volatility/Uncertain"
    else:
        return "🔵 Moderate Volatility/Stable"


def get_regime_recommendations(regime: str) -> Dict[str, str]:
    """Get investment recommendations based on market regime."""
    recommendations = {
        "🔴 High Volatility/Bear Market": {
            "message": "⚠️ Consider increasing risk management focus",
            "style": "warning",
            "advice": "Focus on capital preservation and defensive assets"
        },
        "🟢 Low Volatility/Bull Market": {
            "message": "💡 Growth-focused strategies may perform better",
            "style": "info", 
            "advice": "Consider higher allocation to growth assets"
        },
        "🟡 High Volatility/Uncertain": {
            "message": "⚠️ Proceed with caution - market uncertainty high",
            "style": "warning",
            "advice": "Balanced approach recommended"
        },
        "🔵 Moderate Volatility/Stable": {
            "message": "✅ Balanced strategies appropriate",
            "style": "success",
            "advice": "Standard risk-adjusted approach recommended"
        }
    }
    return recommendations.get(regime, recommendations["🔵 Moderate Volatility/Stable"])