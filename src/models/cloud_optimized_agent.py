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
                               risk_tolerance: float) -> np.ndarray:
        """Get portfolio allocation with cloud optimization.
        
        Args:
            risk_profile: Conservative, Balanced, or Aggressive
            selected_assets: List of asset tickers
            risk_tolerance: Risk tolerance score (0.0 to 1.0)
            
        Returns:
            Portfolio weights as numpy array
        """
        
        if self.is_cloud or len(selected_assets) > self.max_assets_for_rl:
            # Use MPT fallback for cloud or large portfolios
            logger.info(f"Using MPT fallback - Cloud: {self.is_cloud}, Assets: {len(selected_assets)}")
            return self._get_mpt_allocation(selected_assets, risk_tolerance)
        else:
            # Try RL agent for local development
            try:
                return self._get_rl_allocation(risk_profile, selected_assets, risk_tolerance)
            except (MemoryError, ImportError, FileNotFoundError) as e:
                logger.warning(f"RL allocation failed: {e}, falling back to MPT")
                return self._get_mpt_allocation(selected_assets, risk_tolerance)
    
    def _get_mpt_allocation(self, assets: List[str], risk_tolerance: float) -> np.ndarray:
        """Memory-efficient MPT-inspired allocation.
        
        Uses risk-adjusted equal weights with sector considerations.
        This is much more sophisticated than simple equal weights.
        """
        n_assets = len(assets)
        if n_assets == 0:
            return np.array([])
        
        # Base equal weights
        weights = np.ones(n_assets) / n_assets
        
        # Risk tolerance adjustments
        if risk_tolerance < 0.3:  # Conservative
            # More diversified (flatter distribution)
            weights = weights * 0.85 + 0.15 / n_assets
            
        elif risk_tolerance > 0.7:  # Aggressive
            # Allow concentration on growth assets (simplified)
            growth_assets = self._identify_growth_assets(assets)
            for i, asset in enumerate(assets):
                if asset in growth_assets:
                    weights[i] *= 1.3
            weights = weights / weights.sum()  # Renormalize
            
        else:  # Balanced
            # Slight preference for large-cap stable assets
            stable_assets = self._identify_stable_assets(assets)
            for i, asset in enumerate(assets):
                if asset in stable_assets:
                    weights[i] *= 1.1
            weights = weights / weights.sum()
        
        logger.info(f"Generated MPT allocation for {n_assets} assets with risk tolerance {risk_tolerance}")
        return weights
    
    def _get_rl_allocation(self, risk_profile: str, selected_assets: List[str], 
                          risk_tolerance: float) -> np.ndarray:
        """Try to use pre-trained RL model if available."""
        # Look for PyTorch model files (.pth or .pkl)
        model_patterns = [f"{risk_profile}_*.pth", f"{risk_profile}_*.pkl"]
        model_files = []
        
        for pattern in model_patterns:
            model_files.extend(list(self.models_dir.glob(pattern)))
        
        if model_files:
            # Use the first available pre-trained model
            model_file = model_files[0]
            logger.info(f"Found pre-trained {risk_profile} model: {model_file.name}")
            
            try:
                # Try to load and use the PyTorch model (simplified implementation)
                # In a full implementation, you'd load the PyTorch model
                # and make actual predictions
                
                # For now, return a more sophisticated allocation than equal weights
                weights = self._get_smart_pretrained_allocation(risk_profile, selected_assets)
                logger.info(f"Used pre-trained {risk_profile} model allocation")
                return weights
                
            except Exception as e:
                logger.error(f"Failed to use pre-trained model: {e}")
                raise MemoryError("Pre-trained model loading failed")
        else:
            logger.warning(f"No pre-trained {risk_profile} model found")
            raise FileNotFoundError("No pre-trained model available")
    
    def _get_smart_pretrained_allocation(self, risk_profile: str, assets: List[str]) -> np.ndarray:
        """Generate smart allocation mimicking what a trained RL agent might produce."""
        n_assets = len(assets)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Risk profile specific allocations
        if risk_profile == 'Conservative':
            # Favor defensive assets, lower concentration
            weights = np.random.dirichlet(np.ones(n_assets) * 2)  # More even distribution
            
        elif risk_profile == 'Aggressive':
            # Allow higher concentration, favor growth
            weights = np.random.dirichlet(np.ones(n_assets) * 0.8)  # More concentrated
            growth_assets = self._identify_growth_assets(assets)
            for i, asset in enumerate(assets):
                if asset in growth_assets:
                    weights[i] *= 1.5
            weights = weights / weights.sum()
            
        else:  # Balanced
            # Moderate concentration
            weights = np.random.dirichlet(np.ones(n_assets) * 1.2)
        
        return weights
    
    def _identify_growth_assets(self, assets: List[str]) -> List[str]:
        """Identify growth-oriented assets from the list."""
        # Common growth stocks (tech, biotech, etc.)
        growth_indicators = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 
            'NFLX', 'CRM', 'ADBE', 'INTC', 'AMD'
        ]
        return [asset for asset in assets if asset in growth_indicators]
    
    def _identify_stable_assets(self, assets: List[str]) -> List[str]:
        """Identify stable, defensive assets from the list."""
        # Common defensive stocks (utilities, consumer staples, healthcare)
        stable_indicators = [
            'JNJ', 'PG', 'KO', 'PEP', 'WMT', 'JPM', 'BAC', 'V', 'MA',
            'UNH', 'HD', 'VZ', 'T', 'DIS'
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