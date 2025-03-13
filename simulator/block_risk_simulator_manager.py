"""
Module de gestion des blocs de trading pour l'optimisation des stratégies.
Contient les classes pour générer et gérer les blocs de conditions de trading
et les paramètres de risque pendant l'optimisation.
"""
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Importation des modules du système de trading
from .indicators import SignalGenerator, Block, Condition, Operator, LogicOperator
from .risk import PositionCalculator, RiskMode
from .simulator import Simulator, SimulationConfig
from .config import MarginMode, TradingMode

# Configuration du logging
logger = logging.getLogger(__name__)

class BlockManager:
    """Manages the generation of trading blocks based on trial parameters"""
    
    def __init__(self, trial, trading_config):
        self.trial = trial
        self.trading_config = trading_config
        self.structure_config = trading_config.strategy_structure
        self.available_indicators = trading_config.available_indicators
    
    def generate_blocks(self):
        """
        Generates buy and sell blocks based on trial parameters
        
        Returns:
            Tuple[List[Block], List[Block]]: (buy_blocks, sell_blocks)
        """
        # Determine the number of blocks to generate
        min_blocks = self.structure_config.min_blocks
        max_blocks = self.structure_config.max_blocks
        
        n_buy_blocks = self.trial.suggest_int("n_buy_blocks", min_blocks, max_blocks)
        n_sell_blocks = self.trial.suggest_int("n_sell_blocks", min_blocks, max_blocks)
        
        buy_blocks = self._generate_block_set(n_buy_blocks, "buy")
        sell_blocks = self._generate_block_set(n_sell_blocks, "sell")
        
        return buy_blocks, sell_blocks
    
    def _generate_block_set(self, n_blocks, prefix):
        """
        Generates a set of blocks with the given prefix
        
        Args:
            n_blocks: Number of blocks to generate
            prefix: Prefix for parameter names ("buy" or "sell")
            
        Returns:
            List[Block]: List of generated blocks
        """
        blocks = []
        
        for i in range(n_blocks):
            block_prefix = f"{prefix}_block_{i}"
            block = self._generate_single_block(block_prefix)
            blocks.append(block)
        
        return blocks
    
    def _generate_single_block(self, block_prefix):
        """
        Generates a single trading block
        
        Args:
            block_prefix: Prefix for parameter names
            
        Returns:
            Block: Generated block
        """
        min_conditions = self.structure_config.min_conditions_per_block
        max_conditions = self.structure_config.max_conditions_per_block
        
        # Determine number of conditions
        n_conditions = self.trial.suggest_int(f"{block_prefix}_n_conditions", min_conditions, max_conditions)
        
        # Generate conditions
        conditions = []
        logic_operators = []
        
        for j in range(n_conditions):
            cond_prefix = f"{block_prefix}_cond_{j}"
            condition = self._generate_condition(cond_prefix)
            conditions.append(condition)
            
            # Add logic operator if needed
            if j < n_conditions - 1:
                logic_op = self._get_logic_operator(f"{block_prefix}_logic_{j}")
                logic_operators.append(logic_op)
        
        return Block(conditions=conditions, logic_operators=logic_operators)
    
    def _generate_condition(self, cond_prefix):
        """
        Generates a single trading condition
        
        Args:
            cond_prefix: Prefix for parameter names
            
        Returns:
            Condition: Generated condition
        """
        # Choose indicators from available ones
        available_inds = list(self.available_indicators.keys())
        
        # First indicator
        ind1_type = self.trial.suggest_categorical(f"{cond_prefix}_ind1_type", available_inds)
        ind_config = self.available_indicators[ind1_type]
        
        min_period = ind_config.min_period
        max_period = ind_config.max_period
        step = ind_config.step
        
        period1 = self.trial.suggest_int(f"{cond_prefix}_period1", min_period, max_period, step=step)
        ind1 = f"{ind1_type}_{period1}"
        
        # Operator
        operators = [op.value for op in Operator]
        op = Operator(self.trial.suggest_categorical(f"{cond_prefix}_operator", operators))
        
        # Determine if we compare to another indicator or a value
        use_value = self.trial.suggest_float(f"{cond_prefix}_use_value", 0, 1) < self.structure_config.value_comparison_probability
        
        ind2 = None
        value = None
        
        if use_value:
            # Compare to a value
            if ind1_type == "RSI":
                # RSI values are between 0 and 100
                value = self.trial.suggest_float(f"{cond_prefix}_value", 
                                               self.structure_config.rsi_value_range[0],
                                               self.structure_config.rsi_value_range[1])
            else:
                # Use a multiplier range from structure config
                multiplier = self.trial.suggest_float(f"{cond_prefix}_multiplier", 
                                                   self.structure_config.general_value_range[0]/100,
                                                   self.structure_config.general_value_range[1]/100)
                value = multiplier  # Will be multiplied by price at runtime
        else:
            # Compare to another indicator
            ind2_type = self.trial.suggest_categorical(f"{cond_prefix}_ind2_type", available_inds)
            ind_config2 = self.available_indicators[ind2_type]
            
            min_period2 = ind_config2.min_period
            max_period2 = ind_config2.max_period
            step2 = ind_config2.step
            
            period2 = self.trial.suggest_int(f"{cond_prefix}_period2", min_period2, max_period2, step=step2)
            ind2 = f"{ind2_type}_{period2}"
        
        return Condition(indicator1=ind1, operator=op, indicator2=ind2, value=value)
    
    def _get_logic_operator(self, param_name):
        """
        Gets a logic operator based on the trial parameter
        
        Args:
            param_name: Parameter name
            
        Returns:
            LogicOperator: AND or OR
        """
        use_or = self.trial.suggest_categorical(param_name, [0, 1])
        return LogicOperator.OR if use_or else LogicOperator.AND


class RiskManager:
    """Manages risk parameters based on trial parameters"""
    
    def __init__(self, trial, trading_config):
        self.trial = trial
        self.trading_config = trading_config
        self.risk_config = trading_config.risk_config
        
        # Choose risk mode from available modes
        available_modes = [mode.value for mode in self.risk_config.available_modes]
        chosen_mode = self.trial.suggest_categorical("risk_mode", available_modes)
        self.risk_mode = RiskMode(chosen_mode)
    
    def get_config(self):
        """
        Gets risk configuration based on the chosen risk mode
        
        Returns:
            Dict: Configuration for PositionCalculator
        """
        config = {}
        
        # Base configuration common to all modes
        base_position_range = self.risk_config.position_size_range
        base_sl_range = self.risk_config.sl_range
        tp_mult_range = self.risk_config.tp_multiplier_range
        
        config["base_position"] = self.trial.suggest_float("base_position", base_position_range[0], base_position_range[1], log=True)
        config["base_sl"] = self.trial.suggest_float("base_sl", base_sl_range[0], base_sl_range[1], log=True)
        config["tp_multiplier"] = self.trial.suggest_float("tp_multiplier", tp_mult_range[0], tp_mult_range[1])
        
        # Mode-specific configuration
        if self.risk_mode == RiskMode.FIXED:
            mode_config = self.risk_config.mode_configs.get(RiskMode.FIXED)
            if mode_config:
                fixed_pos_range = mode_config.fixed_position_range
                fixed_sl_range = mode_config.fixed_sl_range
                fixed_tp_range = mode_config.fixed_tp_range
                
                config["base_position"] = self.trial.suggest_float("fixed_position", fixed_pos_range[0], fixed_pos_range[1], log=True)
                config["base_sl"] = self.trial.suggest_float("fixed_sl", fixed_sl_range[0], fixed_sl_range[1], log=True)
                config["tp_multiplier"] = self.trial.suggest_float("fixed_tp_mult", fixed_tp_range[0], fixed_tp_range[1])
        
        elif self.risk_mode == RiskMode.ATR_BASED:
            mode_config = self.risk_config.mode_configs.get(RiskMode.ATR_BASED)
            if mode_config:
                atr_period_range = mode_config.atr_period_range
                atr_mult_range = mode_config.atr_multiplier_range
                
                config["atr_period"] = self.trial.suggest_int("atr_period", atr_period_range[0], atr_period_range[1])
                config["atr_multiplier"] = self.trial.suggest_float("atr_multiplier", atr_mult_range[0], atr_mult_range[1])
        
        elif self.risk_mode == RiskMode.VOLATILITY_BASED:
            mode_config = self.risk_config.mode_configs.get(RiskMode.VOLATILITY_BASED)
            if mode_config:
                vol_period_range = mode_config.vol_period_range
                vol_mult_range = mode_config.vol_multiplier_range
                
                config["vol_period"] = self.trial.suggest_int("vol_period", vol_period_range[0], vol_period_range[1])
                config["vol_multiplier"] = self.trial.suggest_float("vol_multiplier", vol_mult_range[0], vol_mult_range[1])
        
        return config


class SimulationManager:
    """Manages simulation parameters based on trial parameters"""
    
    def __init__(self, trial, trading_config):
        self.trial = trial
        self.trading_config = trading_config
        self.sim_config = trading_config.sim_config
        
        # Choose leverage from available range
        leverage_range = self.sim_config.leverage_range
        self.leverage = self.trial.suggest_int("leverage", leverage_range[0], leverage_range[1], log=True)
        
        # Choose margin mode
        margin_modes = [mode.value for mode in self.sim_config.margin_modes]
        self.margin_mode = self.trial.suggest_categorical("margin_mode", margin_modes)
        
        # Choose trading mode
        trading_modes = [mode.value for mode in self.sim_config.trading_modes]
        self.trading_mode = self.trial.suggest_categorical("trading_mode", trading_modes)
    
    def get_simulator_config(self):
        """
        Gets simulation configuration based on trial parameters
        
        Returns:
            SimulationConfig: Configuration for Simulator
        """
        # Initial balance range
        balance_range = self.sim_config.initial_balance_range
        initial_balance = self.trial.suggest_float(
            "initial_balance", 
            balance_range[0], 
            balance_range[1], 
            log=True
        )
        
        # Fee and slippage
        # Instead of optimizing these, use fixed values from config
        fee_open = self.sim_config.fee 
        fee_close = self.sim_config.fee
        slippage = self.sim_config.slippage
        
        return SimulationConfig(
            initial_balance=initial_balance,
            fee_open=fee_open,
            fee_close=fee_close,
            slippage=slippage,
            tick_size=self.sim_config.tick_size,
            min_trade_size=self.sim_config.min_trade_size,
            max_trade_size=self.sim_config.max_trade_size,
            leverage=self.leverage,
            margin_mode=self.margin_mode,
            trading_mode=self.trading_mode
        )