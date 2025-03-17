"""
Module containing the simulation configuration classes.
This provides a flexible, serializable configuration for trading simulations.
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any, Union, List
import json
from datetime import datetime


class MarginMode(Enum):
    """Trading margin modes"""
    ISOLATED = 0
    CROSS = 1


class TradingMode(Enum):
    """Trading position modes"""
    ONE_WAY = 0  # Long OR short positions
    HEDGE = 1    # Long AND short positions


class TradeType(Enum):
    """Types of trades"""
    BUY = 1
    SELL = -1
    NONE = 0


@dataclass
class SimulationConfig:
    """Configuration parameters for trading simulations"""
    # Core parameters
    initial_balance: float = 10000.0
    fee_open: float = 0.001  # 0.1% per trade
    fee_close: float = 0.001
    slippage: float = 0.001
    tick_size: float = 0.01
    
    # Position sizing limits
    min_trade_size: float = 0.001
    max_trade_size: float = 100000.0
    
    # Trading parameters
    leverage: int = 1
    margin_mode: MarginMode = MarginMode.ISOLATED
    trading_mode: TradingMode = TradingMode.ONE_WAY
    
    # Additional custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Convert enum values if they're provided as integers"""
        if isinstance(self.margin_mode, int):
            self.margin_mode = MarginMode(self.margin_mode)
            
        if isinstance(self.trading_mode, int):
            self.trading_mode = TradingMode(self.trading_mode)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary, handling enums"""
        config_dict = asdict(self)
        
        # Convert enums to their values for serialization
        config_dict['margin_mode'] = self.margin_mode.value
        config_dict['trading_mode'] = self.trading_mode.value
        
        return config_dict
    
    def to_json(self) -> str:
        """Convert the config to a JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SimulationConfig':
        """Create a config from a dictionary"""
        # Make a copy to avoid modifying the original
        config = config_dict.copy()
        
        # Handle enum values
        if 'margin_mode' in config:
            config['margin_mode'] = MarginMode(config['margin_mode'])
            
        if 'trading_mode' in config:
            config['trading_mode'] = TradingMode(config['trading_mode'])
            
        return cls(**config)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SimulationConfig':
        """Create a config from a JSON string"""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    def save_to_file(self, filepath: str) -> None:
        """Save the config to a JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'SimulationConfig':
        """Load a config from a JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())
