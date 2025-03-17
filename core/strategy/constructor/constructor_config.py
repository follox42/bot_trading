"""
Centralized configuration for trading strategies.
Integrates all sub-configurations in a coherent way.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Union, Any

# Import configuration modules
from core.strategy.operand.operand import OperandStrategy
from core.strategy.indicators.indicators import IndicatorStrategy
from core.strategy.conditions.conditions_config import (
    ConditionConfig, BlockConfig, StrategyBlocksConfig,
    OperatorType, LogicOperatorType,
    PriceOperand, IndicatorOperand, ValueOperand
)
from core.strategy.operand.operand_config import Operand
from core.strategy.conditions.conditions import ConditionEvaluator
from core.strategy.risk.risk_config import RiskConfig, RiskModeType

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyError(Exception):
    """Base exception for all strategy-related errors"""
    pass


class ConfigurationError(StrategyError):
    """Error related to strategy configuration"""
    pass


class ValidationError(StrategyError):
    """Error in data or parameter validation"""
    pass


class SerializationError(StrategyError):
    """Error during serialization or deserialization"""
    pass


class FileIOError(StrategyError):
    """Error during file access"""
    pass


class ResultNotFoundError(StrategyError):
    """Error when a result is not found"""
    pass


class CalculationError(StrategyError):
    """Error during calculation"""
    pass


@dataclass
class StrategyConfig:
    """Complete configuration of a trading strategy with integrated sub-configurations"""
    
    # Strategy metadata
    name: str
    description: str = ""
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    
    # Component configurations (actual objects, not dicts)
    operand_list: OperandStrategy = None
    blocks_config: StrategyBlocksConfig = field(default_factory=StrategyBlocksConfig)
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    
    # Additional parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validation after initialization"""
        if not self.name:
            raise ValidationError("Strategy name cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the complete configuration to a dictionary.
        Intelligently handles conversion of sub-configurations.
        
        Returns:
            Dict: Dictionary representation of the configuration
            
        Raises:
            SerializationError: In case of serialization error
        """
        try:
            # Convert indicators
            indicators_dict = {}
            for indicator_id, indicator in self.operand_list.get_operands().items():
                try:
                    indicators_dict[indicator_id] = {
                        "type": indicator.__class__.__name__,
                        "params": indicator.params
                    }
                except Exception as e:
                    logger.warning(f"Error serializing indicator {indicator_id}: {str(e)}")
            
            # Convert condition blocks
            blocks_dict = {}
            try:
                if hasattr(self.blocks_config, "to_dict"):
                    blocks_dict = self.blocks_config.to_dict()
                else:
                    # Fallback if to_dict is not available
                    blocks_dict = asdict(self.blocks_config)
            except Exception as e:
                logger.warning(f"Error serializing blocks: {str(e)}")
            
            # Convert risk configuration
            risk_dict = {}
            try:
                if hasattr(self.risk_config, "to_dict"):
                    risk_dict = self.risk_config.to_dict()
                else:
                    risk_dict = asdict(self.risk_config)
            except Exception as e:
                logger.warning(f"Error serializing risk config: {str(e)}")
            
            return {
                # Metadata
                "id": self.id,
                "name": self.name,
                "description": self.description,
                "version": self.version,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "tags": self.tags,
                
                # Converted sub-configurations
                "indicators": indicators_dict,
                "blocks": blocks_dict,
                "risk": risk_dict,
                
                # Parameters
                "parameters": self.parameters
            }
        except Exception as e:
            raise SerializationError(f"Error converting to dictionary: {str(e)}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """
        Creates a configuration from a dictionary.
        Intelligently reconstructs sub-configurations.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            StrategyConfig: Reconstructed configuration
            
        Raises:
            SerializationError: In case of deserialization error
        """
        try:
            # Instantiate an indicator strategy
            strategy_name = data.get("name", "Imported Strategy")
            indicators_strategy = IndicatorStrategy(strategy_name)
            
            # Reconstruct indicators
            indicators_data = data.get("indicators", {})
            for indicator_id, indicator_config in indicators_data.items():
                try:
                    indicator_type = indicator_config.get("type")
                    if indicator_type in INDICATOR_REGISTRY:
                        indicator_class = INDICATOR_REGISTRY[indicator_type]
                        indicator_params = indicator_config.get("params", {})
                        indicator = indicator_class(**indicator_params)
                        indicators_strategy.add_indicator(indicator_id, indicator)
                    else:
                        logger.warning(f"Unknown indicator type: {indicator_type}")
                except Exception as e:
                    logger.warning(f"Error reconstructing indicator {indicator_id}: {str(e)}")
            
            # Reconstruct blocks configuration
            blocks_config = StrategyBlocksConfig()
            blocks_data = data.get("blocks", {})
            try:
                if hasattr(StrategyBlocksConfig, "from_dict"):
                    blocks_config = StrategyBlocksConfig.from_dict(blocks_data)
                else:
                    # Fallback for simple reconstruction
                    for key, value in blocks_data.items():
                        if hasattr(blocks_config, key):
                            setattr(blocks_config, key, value)
            except Exception as e:
                logger.warning(f"Error reconstructing blocks: {str(e)}")
            
            # Reconstruct risk configuration
            risk_config = RiskConfig()
            risk_data = data.get("risk", {})
            try:
                if hasattr(RiskConfig, "from_dict"):
                    risk_config = RiskConfig.from_dict(risk_data)
                else:
                    # Fallback for simple reconstruction
                    for key, value in risk_data.items():
                        if hasattr(risk_config, key):
                            setattr(risk_config, key, value)
            except Exception as e:
                logger.warning(f"Error reconstructing risk config: {str(e)}")
            
            return cls(
                id=data.get("id", str(uuid.uuid4())[:8]),
                name=data.get("name", "Imported Strategy"),
                description=data.get("description", ""),
                version=data.get("version", "1.0"),
                created_at=data.get("created_at", datetime.now().isoformat()),
                updated_at=data.get("updated_at", datetime.now().isoformat()),
                tags=data.get("tags", []),
                indicators_strategy=indicators_strategy,
                blocks_config=blocks_config,
                risk_config=risk_config,
                parameters=data.get("parameters", {})
            )
        except Exception as e:
            raise SerializationError(f"Error reconstructing from dictionary: {str(e)}")
    
    def to_json(self) -> str:
        """
        Converts the configuration to JSON.
        
        Returns:
            str: JSON representation of the configuration
            
        Raises:
            SerializationError: In case of JSON serialization error
        """
        try:
            return json.dumps(self.to_dict(), indent=4, ensure_ascii=False)
        except Exception as e:
            raise SerializationError(f"Error converting to JSON: {str(e)}")
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StrategyConfig':
        """
        Creates a configuration from a JSON string.
        
        Args:
            json_str: JSON string
            
        Returns:
            StrategyConfig: Reconstructed configuration
            
        Raises:
            SerializationError: In case of JSON deserialization error
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise SerializationError(f"Invalid JSON: {str(e)}")
        except Exception as e:
            raise SerializationError(f"Error reconstructing from JSON: {str(e)}")
    
    def save(self, filepath: str) -> bool:
        """
        Saves the configuration to a JSON file.
        
        Args:
            filepath: File path
            
        Returns:
            bool: True if save was successful
            
        Raises:
            FileIOError: In case of file writing error
        """
        try:
            # Update modification date
            self.updated_at = datetime.now().isoformat()
            
            # Create directory if necessary
            directory = os.path.dirname(os.path.abspath(filepath))
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Save as JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)
            
            logger.info(f"Strategy '{self.name}' successfully saved to {filepath}")
            return True
        except PermissionError:
            raise FileIOError(f"Permission denied when writing to {filepath}")
        except IsADirectoryError:
            raise FileIOError(f"{filepath} is a directory, cannot write file")
        except Exception as e:
            raise FileIOError(f"Error saving strategy to {filepath}: {str(e)}")
    
    @classmethod
    def load(cls, filepath: str) -> 'StrategyConfig':
        """
        Loads a configuration from a JSON file.
        
        Args:
            filepath: File path
            
        Returns:
            StrategyConfig: Loaded configuration
            
        Raises:
            FileIOError: In case of file reading error
            SerializationError: In case of deserialization error
        """
        try:
            if not os.path.exists(filepath):
                raise FileIOError(f"File {filepath} does not exist")
            
            if not os.path.isfile(filepath):
                raise FileIOError(f"{filepath} is not a file")
                
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            config = cls.from_dict(data)
            logger.info(f"Strategy '{config.name}' successfully loaded from {filepath}")
            return config
        except json.JSONDecodeError as e:
            raise SerializationError(f"Invalid JSON format in {filepath}: {str(e)}")
        except PermissionError:
            raise FileIOError(f"Permission denied when reading {filepath}")
        except Exception as e:
            raise FileIOError(f"Error loading configuration from {filepath}: {str(e)}")
    
    def clone(self) -> 'StrategyConfig':
        """
        Clones the current configuration by creating a deep copy.
        
        Returns:
            StrategyConfig: Clone of the configuration
            
        Raises:
            SerializationError: In case of error during cloning
        """
        try:
            # Simple cloning method via serialization/deserialization
            data = self.to_dict()
            
            # Modify fields for the clone
            data["id"] = str(uuid.uuid4())[:8]
            data["name"] = f"{self.name} (Clone)"
            data["created_at"] = datetime.now().isoformat()
            data["updated_at"] = datetime.now().isoformat()
            
            clone = self.__class__.from_dict(data)
            logger.info(f"Strategy '{self.name}' successfully cloned to '{clone.name}'")
            return clone
        except Exception as e:
            raise SerializationError(f"Error cloning strategy: {str(e)}")
    
    def validate(self) -> bool:
        """
        Validates the consistency of the configuration.
        
        Returns:
            bool: True if the configuration is valid
            
        Raises:
            ValidationError: If the configuration is not valid
        """
        errors = []
        
        # Validate metadata
        if not self.name:
            errors.append("Strategy name cannot be empty")
        
        # Validate indicators
        if not self.indicators_strategy.get_indicators():
            errors.append("Strategy contains no indicators")
        
        # Validate condition blocks
        if hasattr(self.blocks_config, "validate"):
            try:
                self.blocks_config.validate()
            except Exception as e:
                errors.append(f"Invalid blocks configuration: {str(e)}")
        elif not getattr(self.blocks_config, "entry_blocks", None):
            errors.append("No entry blocks defined")
        
        # Validate risk configuration
        if hasattr(self.risk_config, "validate"):
            try:
                self.risk_config.validate()
            except Exception as e:
                errors.append(f"Invalid risk configuration: {str(e)}")
        
        # If errors were detected, raise an exception
        if errors:
            raise ValidationError(f"Invalid configuration: {'; '.join(errors)}")
        
        return True
    
    def is_valid(self) -> bool:
        """
        Checks if the configuration is valid without raising exceptions.
        
        Returns:
            bool: True if the configuration is valid
        """
        try:
            return self.validate()
        except Exception:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Returns a report on the configuration status.
        
        Returns:
            Dict: Configuration status report
        """
        indicators = self.indicators_strategy.get_indicators()
        
        # Count blocks by type
        entry_blocks_count = len(getattr(self.blocks_config, "entry_blocks", []))
        exit_blocks_count = len(getattr(self.blocks_config, "exit_blocks", []))
        filter_blocks_count = len(getattr(self.blocks_config, "filter_blocks", []))
        
        # Get risk mode
        risk_mode = getattr(self.risk_config, "mode", None)
        risk_mode_str = risk_mode.value if hasattr(risk_mode, "value") else str(risk_mode)
        
        return {
            "name": self.name,
            "id": self.id,
            "version": self.version,
            "indicators_count": len(indicators),
            "indicators": list(indicators.keys()),
            "entry_blocks": entry_blocks_count,
            "exit_blocks": exit_blocks_count,
            "filter_blocks": filter_blocks_count,
            "risk_mode": risk_mode_str,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_valid": self.is_valid()
        }
    
    def update_metadata(self, **kwargs) -> None:
        """
        Updates the strategy metadata.
        
        Args:
            **kwargs: Fields to update (name, description, version, tags)
            
        Raises:
            ValidationError: If a value is invalid
        """
        for key, value in kwargs.items():
            if key == "name" and not value:
                raise ValidationError("Strategy name cannot be empty")
                
            if key in ["name", "description", "version", "tags"]:
                setattr(self, key, value)
                
        # Update modification date
        self.updated_at = datetime.now().isoformat()
        
        logger.info(f"Strategy '{self.name}' metadata updated")