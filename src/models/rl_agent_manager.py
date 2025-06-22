"""Manager for handling multiple RL agents with different asset configurations.

This module provides sophisticated agent management including transfer learning,
asset flexibility, and multi-risk-profile support for the robo-advisor system.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import logging

from ..config import RL_MODEL_CONFIGS, TRANSFER_LEARNING_CONFIG, DEFAULT_PORTFOLIO_ASSETS
from .rl_agent import Agent, train_rl_agent

logger = logging.getLogger(__name__)


class RLAgentManager:
    """Manages multiple RL agents for different risk profiles and asset configurations.
    
    This class handles:
    - Multiple risk profiles (Conservative, Balanced, Aggressive)
    - Dynamic asset selection with transfer learning
    - Model caching and reuse
    - Fallback strategies when RL is unavailable
    """
    
    def __init__(self, models_dir: Path):
        """Initialize the RL Agent Manager.
        
        Args:
            models_dir: Directory to store and load trained models
        """
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded agents and their asset mappings
        self.loaded_agents: Dict[str, Agent] = {}
        self.asset_mappings: Dict[str, List[str]] = {}
        
        logger.info(f"Initialized RLAgentManager with models directory: {models_dir}")
    
    def get_or_create_agent(self, risk_profile: str, selected_assets: List[str], 
                          market_data: pd.DataFrame,
                          return_weight: float = 0.5,    # ADDED THIS
                          risk_weight: float = 0.5,      # ADDED THIS
                          market_regime: str = "Stable") -> Tuple[Agent, bool]:  # ADDED THIS
        """Get existing agent or create/train new one with dynamic objectives.
        
        Args:
            risk_profile: Conservative, Balanced, or Aggressive
            selected_assets: List of asset tickers
            market_data: Historical price data
            return_weight: Weight for return component in reward (0.0 to 1.0)
            risk_weight: Weight for risk component in reward (0.0 to 1.0)
            market_regime: Current market regime for dynamic adjustments
            
        Returns:
            Tuple of (agent, is_newly_trained)
        """
        if risk_profile not in RL_MODEL_CONFIGS:
            raise ValueError(f"Unknown risk profile: {risk_profile}. "
                           f"Available: {list(RL_MODEL_CONFIGS.keys())}")
        
        asset_key = self._get_asset_key(selected_assets)
        # CHANGED: Include objective in agent key to distinguish different training objectives
        objective_key = f"ret{return_weight:.1f}_risk{risk_weight:.1f}_{market_regime.split()[0]}"
        agent_key = f"{risk_profile}_{asset_key}_{objective_key}"
        
        logger.info(f"Requesting agent for {risk_profile} profile with {len(selected_assets)} assets")
        logger.info(f"Objective: {return_weight:.0%} return, {risk_weight:.0%} risk, regime: {market_regime}")
        
        # Check if we already have this exact configuration
        if agent_key in self.loaded_agents:
            logger.info(f"Using cached agent: {agent_key}")
            return self.loaded_agents[agent_key], False
        
        # Check for compatible existing model (ignoring objective for transfer learning)
        compatible_agent = self._find_compatible_agent(risk_profile, selected_assets)
        
        if compatible_agent:
            # Use transfer learning
            logger.info("Found compatible agent, applying transfer learning...")
            new_agent = self._adapt_agent(compatible_agent, selected_assets, market_data, 
                                        risk_profile, return_weight, risk_weight, market_regime)
            self.loaded_agents[agent_key] = new_agent
            self.asset_mappings[agent_key] = selected_assets
            return new_agent, False
        else:
            # Train new agent from scratch
            logger.info("No compatible agent found, training new agent...")
            new_agent = self._train_new_agent(risk_profile, selected_assets, market_data,
                                            return_weight, risk_weight, market_regime)
            self.loaded_agents[agent_key] = new_agent
            self.asset_mappings[agent_key] = selected_assets
            return new_agent, True
    
    def _get_asset_key(self, assets: List[str]) -> str:
        """Create a consistent key for asset combinations."""
        return "_".join(sorted(assets))
    
    def _find_compatible_agent(self, risk_profile: str, selected_assets: List[str]) -> Optional[Agent]:
        """Find existing agent with sufficient asset overlap."""
        best_agent = None
        best_overlap_ratio = 0
        
        for key, agent in self.loaded_agents.items():
            if not key.startswith(risk_profile):
                continue
                
            existing_assets = self.asset_mappings.get(key, [])
            if not existing_assets:
                continue
                
            overlap = len(set(selected_assets) & set(existing_assets))
            overlap_ratio = overlap / len(selected_assets) if selected_assets else 0
            
            if overlap_ratio > best_overlap_ratio and overlap_ratio >= TRANSFER_LEARNING_CONFIG['min_overlap_ratio']:
                best_agent = agent
                best_overlap_ratio = overlap_ratio
        
        if best_agent:
            logger.info(f"Found compatible agent with {best_overlap_ratio:.2%} asset overlap")
        
        return best_agent
    
    def _adapt_agent(self, base_agent: Agent, new_assets: List[str], 
                    market_data: pd.DataFrame, risk_profile: str,
                    return_weight: float, risk_weight: float, market_regime: str) -> Agent:  # ADDED PARAMETERS
        """Adapt existing agent to new asset set using transfer learning."""
        try:
            # Create new agent with same architecture but different asset size
            new_agent = Agent(
                portfolio_size=len(new_assets),
                is_eval=False,
                allow_short=True
            )
            
            # If assets are subset/superset, we can do smart weight mapping
            if self._can_map_weights(base_agent, new_agent):
                self._transfer_weights(base_agent, new_agent, new_assets)
            
            # Fine-tune with new asset data
            config = RL_MODEL_CONFIGS[risk_profile]
            fine_tune_data = market_data[new_assets].copy()
            
            # Short fine-tuning session with dynamic objectives
            logger.info(f"Fine-tuning agent for {TRANSFER_LEARNING_CONFIG['fine_tune_epochs']} episodes...")
            fine_tuned_agent, _ = train_rl_agent(
                selected_data=fine_tune_data,
                window_size=50,
                episode_count=TRANSFER_LEARNING_CONFIG['fine_tune_epochs'],
                batch_size=16,
                rebalance_period=config['rebalance_frequency'],
                return_weight=return_weight,    # ADDED THIS
                risk_weight=risk_weight,        # ADDED THIS
                market_regime=market_regime     # ADDED THIS
            )
            
            return fine_tuned_agent
            
        except Exception as e:
            logger.error(f"Transfer learning failed: {e}, training from scratch")
            return self._train_new_agent(risk_profile, new_assets, market_data,
                                       return_weight, risk_weight, market_regime)
    
    def _can_map_weights(self, base_agent: Agent, new_agent: Agent) -> bool:
        """Check if we can map weights between agents."""
        # For now, simple check - could be more sophisticated
        size_diff = abs(base_agent.portfolio_size - new_agent.portfolio_size)
        return size_diff <= 10  # Allow up to 10 asset difference
    
    def _transfer_weights(self, base_agent: Agent, new_agent: Agent, new_assets: List[str]):
        """Transfer compatible weights from base agent to new agent."""
        try:
            # Copy compatible layers
            base_weights = base_agent.model.get_weights()
            new_weights = new_agent.model.get_weights()
            
            # Transfer shared layers (dense layers typically transferable)
            for i in range(min(len(base_weights), len(new_weights))):
                if base_weights[i].shape == new_weights[i].shape:
                    # Apply base model weight factor
                    new_weights[i] = (base_weights[i] * TRANSFER_LEARNING_CONFIG['base_model_weight'] + 
                                    new_weights[i] * (1 - TRANSFER_LEARNING_CONFIG['base_model_weight']))
            
            new_agent.model.set_weights(new_weights)
            logger.info("Successfully transferred weights between agents")
            
        except Exception as e:
            logger.warning(f"Weight transfer failed: {e}, using random initialization")
    
    def _train_new_agent(self, risk_profile: str, selected_assets: List[str], 
                        market_data: pd.DataFrame,
                        return_weight: float, risk_weight: float, market_regime: str) -> Agent:  # ADDED PARAMETERS
        """Train completely new agent for given assets with dynamic objectives."""
        config = RL_MODEL_CONFIGS[risk_profile]
        training_data = market_data[selected_assets].copy()
        
        logger.info(f"Training new {risk_profile} agent for {len(selected_assets)} assets...")
        logger.info(f"Dynamic objective: {return_weight:.0%} return, {risk_weight:.0%} risk")
        
        agent, _ = train_rl_agent(
            selected_data=training_data,
            window_size=50,
            episode_count=50,  # Full training
            batch_size=32,
            rebalance_period=config['rebalance_frequency'],
            return_weight=return_weight,    # ADDED THIS
            risk_weight=risk_weight,        # ADDED THIS
            market_regime=market_regime     # ADDED THIS
        )
        
        # Save model with asset-specific name and objective
        objective_suffix = f"ret{return_weight:.1f}_risk{risk_weight:.1f}"
        model_filename = f"{risk_profile}_{self._get_asset_key(selected_assets)}_{objective_suffix}.pth"
        model_path = self.models_dir / model_filename
        agent.save_model(model_path)  # Use the agent's PyTorch save method
        logger.info(f"Saved new agent model: {model_path}")
        
        return agent
    
    def get_fallback_allocation(self, selected_assets: List[str], 
                              risk_tolerance: float) -> np.ndarray:
        """Provide MPT-based fallback allocation when RL agent unavailable."""
        logger.info("Generating fallback allocation using MPT approach")
        
        # Use a simple equal-weight allocation as ultimate fallback
        if len(selected_assets) == 0:
            return np.array([])
        
        # Try MPT optimization, fall back to risk-adjusted equal weights
        try:
            from ..utils.portfolio_math import mean_variance_optimization
            
            # This would ideally use real market data for MPT
            # For now, use risk-adjusted equal weights
            weights = np.ones(len(selected_assets)) / len(selected_assets)
            
            # Adjust for risk tolerance
            if risk_tolerance < 0.3:  # Conservative
                # More equal distribution (less concentration)
                weights = weights * 0.8 + 0.2 / len(selected_assets)
            elif risk_tolerance > 0.7:  # Aggressive
                # Allow some concentration (simplified approach)
                # In practice, you'd want more sophisticated logic here
                pass
            
            logger.info(f"Generated fallback allocation for {len(selected_assets)} assets")
            return weights
            
        except Exception as e:
            logger.error(f"MPT fallback failed: {e}, using equal weights")
            # Ultimate fallback: equal weights
            return np.ones(len(selected_assets)) / len(selected_assets)
    
    def clear_cache(self):
        """Clear all cached agents and mappings."""
        self.loaded_agents.clear()
        self.asset_mappings.clear()
        logger.info("Cleared agent cache")
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get information about cached agents."""
        return {
            'total_agents': len(self.loaded_agents),
            'conservative_agents': len([k for k in self.loaded_agents.keys() if k.startswith('Conservative')]),
            'balanced_agents': len([k for k in self.loaded_agents.keys() if k.startswith('Balanced')]),
            'aggressive_agents': len([k for k in self.loaded_agents.keys() if k.startswith('Aggressive')])
        }


def create_agent_for_custom_portfolio(risk_profile: str, selected_assets: List[str], 
                                    market_data: pd.DataFrame, models_dir: Path,
                                    return_weight: float = 0.5, risk_weight: float = 0.5,  # ADDED THESE
                                    market_regime: str = "Stable") -> Agent:  # ADDED THIS
    """Convenience function to get appropriate agent for custom asset selection.
    
    Args:
        risk_profile: Risk profile (Conservative, Balanced, Aggressive)
        selected_assets: List of asset tickers
        market_data: Historical price data
        models_dir: Directory for model storage
        return_weight: Weight for return component in reward (0.0 to 1.0)
        risk_weight: Weight for risk component in reward (0.0 to 1.0)
        market_regime: Current market regime for dynamic adjustments
        
    Returns:
        Trained/adapted RL agent for the specified configuration
    """
    manager = RLAgentManager(models_dir)
    agent, is_new = manager.get_or_create_agent(risk_profile, selected_assets, market_data,
                                              return_weight, risk_weight, market_regime)
    
    if is_new:
        print(f"✓ Trained new {risk_profile} RL agent for {len(selected_assets)} assets")
        print(f"  Objective: {return_weight:.0%} return, {risk_weight:.0%} risk")
    else:
        print(f"✓ Using existing/adapted {risk_profile} RL agent")
        print(f"  Objective: {return_weight:.0%} return, {risk_weight:.0%} risk")
    
    return agent


def get_recommended_assets_for_profile(risk_profile: str, max_assets: int = 20) -> List[str]:
    """Get recommended asset selection based on risk profile.
    
    Args:
        risk_profile: Risk profile name
        max_assets: Maximum number of assets to return
        
    Returns:
        List of recommended asset tickers
    """
    from ..config import ASSET_UNIVERSE
    
    if risk_profile == 'Conservative':
        # Focus on stable sectors
        recommended = (ASSET_UNIVERSE['Utilities'] + 
                      ASSET_UNIVERSE['Consumer_Staples'] + 
                      ASSET_UNIVERSE['Healthcare'][:5])
    elif risk_profile == 'Balanced':
        # Mix across sectors
        recommended = (ASSET_UNIVERSE['Technology'][:5] + 
                      ASSET_UNIVERSE['Finance'][:5] + 
                      ASSET_UNIVERSE['Healthcare'][:5] + 
                      ASSET_UNIVERSE['Consumer_Staples'][:3] + 
                      ASSET_UNIVERSE['Industrials'][:2])
    else:  # Aggressive
        # Focus on growth sectors
        recommended = (ASSET_UNIVERSE['Technology'][:8] + 
                      ASSET_UNIVERSE['Consumer_Discretionary'][:6] + 
                      ASSET_UNIVERSE['Communication'][:4] + 
                      ASSET_UNIVERSE['Energy'][:2])
    
    return recommended[:max_assets]