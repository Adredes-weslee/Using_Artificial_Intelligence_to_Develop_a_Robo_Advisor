"""Market regime detection and analysis utilities."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def calculate_max_drawdown(price_series: pd.Series) -> float:
    """Calculate maximum drawdown of a price series."""
    try:
        # FIXED: Ensure we're working with a Series
        if isinstance(price_series, pd.DataFrame):
            price_series = price_series.iloc[:, 0]
        
        # Calculate rolling maximum (peak)
        peak = price_series.expanding().max()
        
        # Calculate drawdown
        drawdown = (price_series - peak) / peak
        
        # Return minimum drawdown (most negative value)
        return float(drawdown.min())  # FIXED: Convert to scalar
        
    except Exception as e:
        print(f"Max drawdown calculation error: {e}")
        return 0.0

def detect_market_regime(price_data: pd.DataFrame, lookback_days: int = 252) -> str:
    """Detect current market regime for dynamic risk adjustment."""
    
    try:
        # Get recent data
        recent_data = price_data.tail(lookback_days)
        
        if recent_data.empty or len(recent_data) < 10:
            return "🔵 Moderate Volatility/Stable"
        
        # FIXED: Handle DataFrame properly - use first column or specific column
        if isinstance(recent_data, pd.DataFrame):
            if len(recent_data.columns) == 1:
                price_series = recent_data.iloc[:, 0]
            else:
                # Use first column as market proxy (usually main index)
                price_series = recent_data.iloc[:, 0]
        else:
            price_series = recent_data
        
        # Calculate returns from the price series
        returns = price_series.pct_change().dropna()
        
        if len(returns) == 0:
            return "🔵 Moderate Volatility/Stable"
        
        # Market regime indicators - FIXED: Use scalar values
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # FIXED: Calculate trend using scalar division
        start_price = price_series.iloc[0]
        end_price = price_series.iloc[-1]
        trend = (end_price / start_price - 1) * (252/lookback_days)  # Annualized return
        
        # FIXED: Calculate max drawdown properly
        max_drawdown = calculate_max_drawdown(price_series)
        
        # Regime classification - FIXED: All comparisons now use scalars
        if volatility > 0.25 and max_drawdown < -0.15:
            return "🔴 High Volatility/Bear Market"
        elif volatility < 0.15 and trend > 0.1:
            return "🟢 Low Volatility/Bull Market" 
        elif volatility > 0.20:
            return "🟡 High Volatility/Uncertain"
        else:
            return "🔵 Moderate Volatility/Stable"
            
    except Exception as e:
        print(f"Market regime detection error: {e}")
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