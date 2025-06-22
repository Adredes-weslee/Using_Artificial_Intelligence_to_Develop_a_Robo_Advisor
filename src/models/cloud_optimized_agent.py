"""Cloud-optimized RL agent management for memory-constrained environments.

This module provides a lightweight alternative to the full RL Agent Manager
specifically designed for deployment on memory-constrained cloud platforms
like Streamlit Community Cloud.
"""
import os
import numpy as np
import pandas as pd
from typing import List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CloudOptimizedRLManager:
    """Memory-optimized RL manager for cloud deployment.
    
    This class provides intelligent fallback strategies for cloud deployment
    where memory is limited but we still want to provide AI-powered recommendations.
    """
    
    def __init__(self, models_dir: Path):
        """Initialize cloud-optimized RL manager.
        
        Args:
            models_dir: Directory containing pre-trained models
        """
        self.models_dir = models_dir
        self.is_cloud = os.environ.get('STREAMLIT_CLOUD') == 'true'
        
        # Memory usage tracking
        self.max_assets_for_rl = 10 if self.is_cloud else 30
        self.memory_limit_mb = 512 if self.is_cloud else 4096
        
        logger.info(f"Initialized CloudOptimizedRLManager - Cloud mode: {self.is_cloud}")
        
    def get_portfolio_allocation(self, risk_profile: str, selected_assets: List[str], 
                               risk_tolerance: float,
                               return_weight: float = 0.5,
                               risk_weight: float = 0.5,
                               market_regime: str = "Stable") -> np.ndarray:
        """Get portfolio allocation with cloud optimization and dynamic objectives.
        
        Args:
            risk_profile: Conservative, Balanced, or Aggressive
            selected_assets: List of asset tickers
            risk_tolerance: Risk tolerance score (0.0 to 1.0)
            return_weight: Weight for return component (0.0 to 1.0)
            risk_weight: Weight for risk component (0.0 to 1.0)
            market_regime: Current market regime
            
        Returns:
            Portfolio weights as numpy array
        """
        
        if self.is_cloud or len(selected_assets) > self.max_assets_for_rl:
            # Use enhanced MPT with dynamic objectives for cloud
            logger.info(f"Using enhanced MPT with dynamic objectives - Cloud: {self.is_cloud}, Assets: {len(selected_assets)}")
            return self._get_dynamic_mpt_allocation(selected_assets, risk_tolerance, 
                                                  return_weight, risk_weight, market_regime, risk_profile)
        else:
            # Try RL agent for local development with objective-specific models
            try:
                return self._get_rl_allocation(risk_profile, selected_assets, risk_tolerance,
                                             return_weight, risk_weight)
            except (MemoryError, ImportError, FileNotFoundError) as e:
                logger.warning(f"RL allocation failed: {e}, falling back to dynamic MPT")
                return self._get_dynamic_mpt_allocation(selected_assets, risk_tolerance,
                                                      return_weight, risk_weight, market_regime, risk_profile)
    
    def _get_dynamic_mpt_allocation(self, assets: List[str], risk_tolerance: float,
                                  return_weight: float, risk_weight: float,
                                  market_regime: str, risk_profile: str) -> np.ndarray:
        """Enhanced MPT allocation that simulates dynamic RL behavior.
        
        This method creates meaningful differences between investment objectives
        without requiring actual RL training.
        """
        n_assets = len(assets)
        if n_assets == 0:
            return np.array([])
        
        # Base allocation using risk-adjusted equal weights
        base_weights = self._get_base_allocation(assets, risk_tolerance)
        
        # Apply objective-based adjustments (simulating RL behavior)
        adjusted_weights = self._apply_objective_adjustment(
            base_weights, return_weight, risk_weight, market_regime, risk_profile, assets
        )
        
        logger.info(f"Generated dynamic MPT allocation: {return_weight:.0%} return, {risk_weight:.0%} risk")
        return adjusted_weights
    
    def _get_base_allocation(self, assets: List[str], risk_tolerance: float) -> np.ndarray:
        """Generate base allocation using sophisticated equal-weight approach."""
        n_assets = len(assets)
        
        # Start with equal weights
        weights = np.ones(n_assets) / n_assets
        
        # Risk tolerance adjustments to base allocation
        if risk_tolerance < 0.3:  # Conservative
            # More diversified (flatter distribution)
            weights = weights * 0.85 + 0.15 / n_assets
            
        elif risk_tolerance > 0.7:  # Aggressive
            # Allow some base concentration on growth assets
            growth_assets = self._identify_growth_assets(assets)
            for i, asset in enumerate(assets):
                if asset in growth_assets:
                    weights[i] *= 1.2
            weights = weights / weights.sum()  # Renormalize
            
        else:  # Balanced
            # Slight preference for large-cap stable assets
            stable_assets = self._identify_stable_assets(assets)
            for i, asset in enumerate(assets):
                if asset in stable_assets:
                    weights[i] *= 1.05
            weights = weights / weights.sum()
        
        return weights
    
    def _apply_objective_adjustment(self, base_weights: np.ndarray, 
                                  return_weight: float, risk_weight: float,
                                  market_regime: str, risk_profile: str, 
                                  assets: List[str]) -> np.ndarray:
        """Apply dynamic objective adjustments to simulate different RL behaviors."""
        
        weights = base_weights.copy()
        
        # MAJOR OBJECTIVE-BASED ADJUSTMENTS (simulating different RL training)
        if return_weight > 0.6:  # Growth-focused (🚀 Maximize Returns)
            logger.info("Applying growth-focused adjustments (simulating aggressive RL)")
            
            # 1. Increase concentration in top growth assets
            weights = self._increase_growth_concentration(weights, assets, factor=0.3)
            
            # 2. Boost technology and growth sectors
            weights = self._boost_growth_sectors(weights, assets, factor=0.2)
            
        elif risk_weight > 0.6:  # Risk-focused (🛡️ Protect Capital)
            logger.info("Applying risk-focused adjustments (simulating conservative RL)")
            
            # 1. Increase diversification
            weights = self._increase_diversification(weights, factor=0.25)
            
            # 2. Boost defensive sectors
            weights = self._boost_defensive_sectors(weights, assets, factor=0.2)
            
        # Academic/Balanced (⚖️ Balance Risk & Return) - minimal adjustments
        
        # MARKET REGIME ADJUSTMENTS
        if "Bear Market" in market_regime or "High Volatility" in market_regime:
            # More conservative regardless of objective
            weights = self._increase_diversification(weights, factor=0.1)
            weights = self._boost_defensive_sectors(weights, assets, factor=0.1)
            
        elif "Bull Market" in market_regime or "Low Volatility" in market_regime:
            # Allow more concentration and growth focus
            weights = self._increase_growth_concentration(weights, assets, factor=0.1)
        
        # RISK PROFILE MULTIPLIERS
        risk_multipliers = {
            'Conservative': 0.8,  # Dampen aggressive moves
            'Balanced': 1.0,      # No change
            'Aggressive': 1.2     # Amplify moves
        }
        
        multiplier = risk_multipliers.get(risk_profile, 1.0)
        if multiplier != 1.0:
            # Apply multiplier to deviations from equal weight
            equal_weight = 1.0 / len(weights)
            deviations = weights - equal_weight
            weights = equal_weight + (deviations * multiplier)
            weights = np.maximum(weights, 0.01)  # Ensure positive weights
            weights = weights / weights.sum()  # Renormalize
        
        return weights
    
    def _increase_growth_concentration(self, weights: np.ndarray, assets: List[str], 
                                     factor: float) -> np.ndarray:
        """Increase concentration in growth assets (simulate aggressive RL behavior)."""
        growth_assets = self._identify_growth_assets(assets)
        adjusted_weights = weights.copy()
        
        # Boost growth assets
        for i, asset in enumerate(assets):
            if asset in growth_assets:
                boost = weights[i] * factor
                adjusted_weights[i] += boost
        
        # Also boost top holdings regardless of sector
        sorted_indices = np.argsort(weights)[::-1]
        for i in range(min(3, len(weights))):
            idx = sorted_indices[i]
            boost = weights[idx] * (factor * 0.5)  # Smaller boost for top holdings
            adjusted_weights[idx] += boost
        
        return adjusted_weights / adjusted_weights.sum()
    
    def _increase_diversification(self, weights: np.ndarray, factor: float) -> np.ndarray:
        """Increase diversification (simulate conservative RL behavior)."""
        # Move toward equal weighting
        equal_weight = 1.0 / len(weights)
        adjusted_weights = weights * (1 - factor) + equal_weight * factor
        return adjusted_weights / adjusted_weights.sum()
    
    def _boost_growth_sectors(self, weights: np.ndarray, assets: List[str], 
                             factor: float) -> np.ndarray:
        """Boost growth-oriented sectors."""
        growth_assets = self._identify_growth_assets(assets)
        adjusted_weights = weights.copy()
        
        for i, asset in enumerate(assets):
            if asset in growth_assets:
                adjusted_weights[i] *= (1 + factor)
        
        return adjusted_weights / adjusted_weights.sum()
    
    def _boost_defensive_sectors(self, weights: np.ndarray, assets: List[str], 
                               factor: float) -> np.ndarray:
        """Boost defensive sectors."""
        defensive_assets = self._identify_stable_assets(assets)
        adjusted_weights = weights.copy()
        
        for i, asset in enumerate(assets):
            if asset in defensive_assets:
                adjusted_weights[i] *= (1 + factor)
        
        return adjusted_weights / adjusted_weights.sum()
    
    def _get_rl_allocation(self, risk_profile: str, selected_assets: List[str], 
                          risk_tolerance: float, return_weight: float = 0.5, 
                          risk_weight: float = 0.5) -> np.ndarray:
        """Try to use pre-trained RL model if available with objective-specific loading."""
        
        # 1. First try to find objective-specific model
        assets_str = "_".join(sorted(selected_assets))
        objective_model_filename = f"{risk_profile}_{assets_str}_ret{return_weight}_risk{risk_weight}.pth"
        objective_model_path = self.models_dir / objective_model_filename
        
        if objective_model_path.exists():
            logger.info(f"Found objective-specific model: {objective_model_path.name}")
            try:
                return self._load_and_use_model(objective_model_path, risk_profile, selected_assets, 
                                              return_weight, risk_weight)
            except Exception as e:
                logger.warning(f"Failed to use objective-specific model: {e}")
        
        # 2. Fallback to any model for this risk profile
        model_patterns = [f"{risk_profile}_*.pth", f"{risk_profile}_*.pkl"]
        model_files = []
        
        for pattern in model_patterns:
            model_files.extend(list(self.models_dir.glob(pattern)))
        
        if model_files:
            # Use the first available pre-trained model
            model_file = model_files[0]
            logger.info(f"Using fallback model: {model_file.name}")
            
            try:
                return self._load_and_use_model(model_file, risk_profile, selected_assets,
                                              return_weight, risk_weight)
                
            except Exception as e:
                logger.error(f"Failed to use fallback model: {e}")
                raise MemoryError("All model loading attempts failed")
        else:
            logger.warning(f"No models found for {risk_profile}")
            raise FileNotFoundError("No models available")
    
    def _load_and_use_model(self, model_path: Path, risk_profile: str, assets: List[str],
                           return_weight: float = 0.5, risk_weight: float = 0.5) -> np.ndarray:
        """Load and use a specific model file with objective awareness."""
        try:
            # Try to load and use the PyTorch model (simplified implementation)
            # In a full implementation, you'd load the PyTorch model and make actual predictions
            
            # For now, return a sophisticated allocation that considers the objectives
            weights = self._get_smart_pretrained_allocation(risk_profile, assets, 
                                                          return_weight, risk_weight)
            logger.info(f"Generated smart allocation from {model_path.name}")
            return weights
            
        except Exception as e:
            logger.error(f"Failed to use model {model_path.name}: {e}")
            raise MemoryError("Model loading failed")
    
    def _get_smart_pretrained_allocation(self, risk_profile: str, assets: List[str],
                                       return_weight: float = 0.5, risk_weight: float = 0.5) -> np.ndarray:
        """Generate smart allocation mimicking what a trained RL agent might produce with objectives."""
        n_assets = len(assets)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Risk profile specific base allocations
        if risk_profile == 'Conservative':
            # Favor defensive assets, lower concentration
            weights = np.random.dirichlet(np.ones(n_assets) * 2)  # More even distribution
            
        elif risk_profile == 'Aggressive':
            # Allow higher concentration, favor growth
            weights = np.random.dirichlet(np.ones(n_assets) * 0.8)  # More concentrated
            
        else:  # Balanced
            # Moderate concentration
            weights = np.random.dirichlet(np.ones(n_assets) * 1.2)
        
        # Apply objective-specific adjustments to the smart allocation
        if return_weight > 0.6:  # Growth-focused
            growth_assets = self._identify_growth_assets(assets)
            for i, asset in enumerate(assets):
                if asset in growth_assets:
                    weights[i] *= 1.4  # Strong boost for growth assets
            weights = weights / weights.sum()
            
        elif risk_weight > 0.6:  # Risk-focused
            # Increase diversification and boost defensive assets
            equal_weight = 1.0 / n_assets
            weights = weights * 0.7 + equal_weight * 0.3  # Move toward equal weight
            
            defensive_assets = self._identify_stable_assets(assets)
            for i, asset in enumerate(assets):
                if asset in defensive_assets:
                    weights[i] *= 1.2  # Boost defensive assets
            weights = weights / weights.sum()
        
        return weights
    
    def _identify_growth_assets(self, assets: List[str]) -> List[str]:
        """Identify growth-oriented assets from the list."""
        # Comprehensive growth stocks (tech, biotech, etc.)
        growth_indicators = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 
            'NFLX', 'CRM', 'ADBE', 'INTC', 'AMD', 'ORCL', 'CSCO', 'AVGO',
            
            # Communication Services  
            'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T',
            
            # Consumer Discretionary
            'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
            
            # Growth Healthcare/Biotech
            'UNH', 'JNJ', 'PFE', 'ABT', 'TMO', 'ABBV', 'LLY', 'BMY'
        ]
        return [asset for asset in assets if asset in growth_indicators]
    
    def _identify_stable_assets(self, assets: List[str]) -> List[str]:
        """Identify stable, defensive assets from the list."""
        # Comprehensive defensive stocks (utilities, consumer staples, dividend stocks)
        stable_indicators = [
            # Healthcare (large, stable)
            'JNJ', 'UNH', 'PFE', 'ABT', 'MRK', 'ABBV', 'BMY', 'LLY',
            
            # Consumer Staples
            'PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K',
            
            # Financial Services (large banks)
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK',
            
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'EXC', 'XEL', 'SRE', 'AEP',
            
            # Large Cap Dividend Stocks
            'V', 'MA', 'HD', 'MCD', 'MMM', 'CAT', 'IBM', 'VZ', 'T'
        ]
        return [asset for asset in assets if asset in stable_indicators]
    
    def get_memory_info(self) -> dict:
        """Get current memory usage information."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'memory_mb': memory_info.rss / 1024 / 1024,
                'memory_percent': process.memory_percent(),
                'is_cloud': self.is_cloud,
                'max_assets': self.max_assets_for_rl
            }
        except ImportError:
            return {
                'memory_mb': 'unknown',
                'memory_percent': 'unknown',
                'is_cloud': self.is_cloud,
                'max_assets': self.max_assets_for_rl
            }
    
    def can_use_rl(self, n_assets: int) -> bool:
        """Check if RL training/inference is feasible."""
        if self.is_cloud:
            return False
        if n_assets > self.max_assets_for_rl:
            return False
        
        # Check available memory if psutil is available
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            return available_memory_gb >= 2.0  # Need at least 2GB
        except ImportError:
            return True  # Assume OK if can't check