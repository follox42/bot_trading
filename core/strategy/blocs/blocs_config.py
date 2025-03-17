"""
Configuration module for trading strategy blocks and conditions.
Allows organizing trading logic into structured, reusable components.
"""

import uuid
import json
import logging
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Union, Any, ClassVar

import sys
import os
# Ajouter le répertoire parent au chemin d'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
from numba import njit

# Import operand types
from core.strategy.conditions.conditions_config import (
    OperatorType, LogicOperatorType,
    PriceOperand, IndicatorOperand, ValueOperand,
    Condition
)
from core.strategy.constructor.constructor_config import StrategyError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases for clarity
OperandType = Union[PriceOperand, IndicatorOperand, ValueOperand]
BlockResult = np.ndarray  # Boolean array of block evaluation results

# === Exception Classes ===
class BlockError(StrategyError):
    """Base exception for all block-related errors"""
    pass


class BlockValidationError(BlockError):
    """Raised when block validation fails"""
    pass


class BlockConfigurationError(BlockError):
    """Raised when there's an error in block configuration"""
    pass


class BlockSerializationError(BlockError):
    """Raised when there's an error in block serialization/deserialization"""
    pass

class BlockType(Enum):
    """Types of blocks in a trading strategy"""
    LONG = "long"  # Entry signal blocks
    SHORT = "short"    # Exit signal blocks

class LogicOperatorType(Enum):
    """Types of logical operators"""
    AND = "and"
    OR = "or"

# === Core Classes ===
@dataclass
class BlockConfig:
    """
    Configuration for a block of conditions.
    
    A block combines multiple conditions with logical operators (AND, OR).
    """
    # Block identification
    name: str
    id: str = None  # Will be set in __post_init__
    description: str = ""
    order: int = 0  # Order number for this block
    
    # Block content
    conditions: List[Condition] = field(default_factory=list)
    logic_operators: List[LogicOperatorType] = field(default_factory=list)

    def __post_init__(self):
        """Validates block after initialization and sets the ID"""
        # Generate ID based on name and order if not provided
        if self.id is None:
            # Replace spaces and special characters with underscores
            sanitized_name = ''.join(c if c.isalnum() else '_' for c in self.name).lower()
            self.id = f"{sanitized_name}_{self.order}"
        
        self._validate()
    
    def _validate(self):
        """Validates the block's internal consistency"""
        # Check name
        if not self.name:
            self.name = f"Block {self.id}"
        
        # Check logic operators count
        if len(self.conditions) > 1 and len(self.logic_operators) != len(self.conditions) - 1:
            raise BlockValidationError(
                f"Block '{self.name}': Number of logic operators ({len(self.logic_operators)}) "
                f"must be equal to number of conditions ({len(self.conditions)}) - 1"
            )
    
    def add_condition(self, condition: ConditionConfig, logic_operator: LogicOperatorType = LogicOperatorType.AND) -> None:
        """
        Adds a condition to the block
        
        Args:
            condition: The condition to add
            logic_operator: The logical operator to connect with previous condition (if any)
            
        Raises:
            BlockConfigurationError: If the condition is invalid
        """
        if not isinstance(condition, ConditionConfig):
            raise BlockConfigurationError(
                f"Expected ConditionConfig, got {type(condition).__name__}"
            )
        
        # If this is not the first condition, add the logical operator
        if self.conditions:
            self.logic_operators.append(logic_operator)
        
        self.conditions.append(condition)
        logger.debug(f"Added condition to block '{self.name}': {condition}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts block to dictionary format
        
        Returns:
            Dict[str, Any]: Dictionary representation of the block
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "order": self.order,
            "block_type": self.block_type.value,
            "enabled": self.enabled,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "logic_operators": [op.value for op in self.logic_operators]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BlockConfig':
        """
        Creates a block from dictionary format
        
        Args:
            data: Dictionary representation of the block
            
        Returns:
            BlockConfig: Reconstructed block
            
        Raises:
            BlockSerializationError: If reconstruction fails
        """
        try:
            # Get basic fields
            name = data.get("name", f"Block_{uuid.uuid4()[:6]}")
            id = data.get("id", None)  # ID will be generated in __post_init__ if None
            description = data.get("description", "")
            order = data.get("order", 0)
            
            # Create the block
            block = cls(
                name=name,
                id=id,
                description=description,
                order=data.get("order", 0),
                conditions=[],  # Will be filled later
                logic_operators=[]  # Will be filled later
            )
            
            # Add conditions and logic operators
            conditions_data = data.get("conditions", [])
            logic_operators_data = data.get("logic_operators", [])
            
            for i, condition_data in enumerate(conditions_data):
                condition = ConditionConfig.from_dict(condition_data)
                
                # Add logic operator if not the first condition
                if i > 0:
                    if i - 1 < len(logic_operators_data):
                        logic_op = LogicOperatorType(logic_operators_data[i - 1])
                    else:
                        # Default to AND if missing
                        logger.warning(f"Missing logic operator for condition {i}, defaulting to AND")
                        logic_op = LogicOperatorType.AND
                    
                    block.logic_operators.append(logic_op)
                
                block.conditions.append(condition)
            
            return block
        except KeyError as e:
            raise BlockSerializationError(f"Missing required field in block data: {e}")
        except Exception as e:
            raise BlockSerializationError(f"Error reconstructing block: {str(e)}")
    
    def __str__(self) -> str:
        """String representation of the block"""
        return f"{self.name} ({len(self.conditions)} conditions, {self.block_type.value})"


@dataclass
class StrategyBlocksConfig:
    """
    Configuration for all blocks in a trading strategy.
    
    Organizes blocks into entry, exit, and filter categories.
    """
    # Block collections
    long_blocks: List[BlockConfig] = field(default_factory=list)
    short_blocks: List[BlockConfig] = field(default_factory=list)
    
    # Combination rules
    require_all_long_blocks: bool = False  # Whether all entry blocks must be true (AND) or any (OR)
    require_all_short_blocks: bool = False   # Whether all exit blocks must be true (AND) or any (OR)
    
    def validate(self) -> bool:
        """
        Validates the configuration
        
        Returns:
            bool: True if validation passes
            
        Raises:
            BlockValidationError: If validation fails
        """
        errors = []
        
        # Check if we have at least one entry block
        if not self.long_blocks:
            errors.append("No entry blocks defined")
        
        # Validate individual blocks
        for block in self.long_blocks + self.short_blocks:
            try:
                block._validate()
            except BlockValidationError as e:
                errors.append(f"Block '{block.name}': {str(e)}")
        
        if errors:
            raise BlockValidationError(f"Block configuration errors: {'; '.join(errors)}")
        
        return True
    
    def add_block(self, block: BlockConfig, order: int = None, type: BlockType = None) -> None:
        """
        Adds a block to the appropriate collection based on its type
        
        Args:
            block: The block to add
            
        Raises:
            BlockConfigurationError: If the block is invalid
        """
        if not isinstance(block, BlockConfig):
            raise BlockConfigurationError(
                f"Expected BlockConfig, got {type(block).__name__}"
            )
        if type is None:
            raise BlockConfigurationError(
                f"Type of bloc is requiered"
            )
        
        # Set the order if provided or use the current length of the corresponding block list
        if order is not None:
            block.order = order
        
        # Add to the appropriate collection
        if type == BlockType.LONG:
            if block.order == 0:  # Order not specified by user or in constructor
                block.order = len(self.long_blocks)
            self.long_blocks.append(block)
            logger.info(f"Added entry block: {block.name} (ID: {block.id})")
        elif type == BlockType.SHORT:
            if block.order == 0:  # Order not specified by user or in constructor
                block.order = len(self.short_blocks)
            self.short_blocks.append(block)
            logger.info(f"Added exit block: {block.name} (ID: {block.id})")
        else:
            raise BlockConfigurationError(f"Unknown block type: {type}")
        
        # Generate ID again since order may have changed
        if block.id is None or block.id.endswith("_0") and block.order != 0:
            sanitized_name = ''.join(c if c.isalnum() else '_' for c in block.name).lower()
            block.id = f"{sanitized_name}_{block.order}"
    
    def remove_block(self, block_id: str) -> bool:
        """
        Removes a block by ID
        
        Args:
            block_id: ID of the block to remove
            
        Returns:
            bool: True if block was found and removed
        """
        for block_list in [self.long_blocks, self.short_blocks]:
            for i, block in enumerate(block_list):
                if block.id == block_id:
                    block_list.pop(i)
                    logger.info(f"Removed block: {block.name}")
                    return True
        
        logger.warning(f"Block with ID {block_id} not found")
        return False
    
    def get_block(self, block_id: str) -> Optional[BlockConfig]:
        """
        Gets a block by ID
        
        Args:
            block_id: ID of the block to get
            
        Returns:
            Optional[BlockConfig]: The block if found, None otherwise
        """
        for block_list in [self.long_blocks, self.short_blocks]:
            for block in block_list:
                if block.id == block_id:
                    return block
        
        return None
    
    def get_all_blocks(self) -> List[BlockConfig]:
        """
        Gets all blocks
        
        Returns:
            List[BlockConfig]: All blocks
        """
        return self.long_blocks + self.short_blocks
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the configuration to dictionary format
        
        Returns:
            Dict[str, Any]: Dictionary representation of the configuration
        """
        return {
            "long_blocks": [block.to_dict() for block in self.long_blocks],
            "short_blocks": [block.to_dict() for block in self.short_blocks],
            "require_all_long_blocks": self.require_all_long_blocks,
            "require_all_short_blocks": self.require_all_short_blocks,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyBlocksConfig':
        """
        Creates a configuration from dictionary format
        
        Args:
            data: Dictionary representation of the configuration
            
        Returns:
            StrategyBlocksConfig: Reconstructed configuration
            
        Raises:
            BlockSerializationError: If reconstruction fails
        """
        try:
            # Create empty configuration
            config = cls()
            
            # Set combination rules
            config.require_all_long_blocks = data.get("require_all_long_blocks", False)
            config.require_all_short_blocks = data.get("require_all_short_blocks", False)
            
            # Add long blocks
            for block_data in data.get("long_blocks", []):
                try:
                    block = BlockConfig.from_dict(block_data)
                    config.long_blocks.append(block)
                except Exception as e:
                    logger.warning(f"Error reconstructing entry block: {str(e)}")
            
            # Add short blocks
            for block_data in data.get("short_blocks", []):
                try:
                    block = BlockConfig.from_dict(block_data)
                    config.short_blocks.append(block)
                except Exception as e:
                    logger.warning(f"Error reconstructing exit block: {str(e)}")

            return config
        except Exception as e:
            raise BlockSerializationError(f"Error reconstructing blocks configuration: {str(e)}")
    
    def to_json(self) -> str:
        """
        Converts the configuration to JSON
        
        Returns:
            str: JSON representation of the configuration
            
        Raises:
            BlockSerializationError: If serialization fails
        """
        try:
            return json.dumps(self.to_dict(), indent=4)
        except Exception as e:
            raise BlockSerializationError(f"Error converting to JSON: {str(e)}")
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StrategyBlocksConfig':
        """
        Creates a configuration from JSON
        
        Args:
            json_str: JSON representation of the configuration
            
        Returns:
            StrategyBlocksConfig: Reconstructed configuration
            
        Raises:
            BlockSerializationError: If deserialization fails
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise BlockSerializationError(f"Invalid JSON: {str(e)}")
        except Exception as e:
            raise BlockSerializationError(f"Error reconstructing from JSON: {str(e)}")
    
    def __str__(self) -> str:
        """String representation of the configuration"""
        return (
            f"Strategy Blocks: {len(self.long_blocks)} long, "
            f"{len(self.short_blocks)} short, "
        )


# Example usage
if __name__ == "__main__":
    # Create a simple entry block
    price_close = PriceOperand("close")
    ema_20 = IndicatorOperand("EMA20")
    
    # Create price > EMA20 condition
    condition1 = ConditionConfig(
        left_operand=price_close,
        operator=OperatorType.GREATER,
        right_operand=ema_20,
        name="Price > EMA20"
    )
    
    # Create EMA20 crosses above EMA50 condition
    ema_50 = IndicatorOperand("EMA50")
    condition2 = ConditionConfig(
        left_operand=ema_20,
        operator=OperatorType.CROSS_ABOVE,
        right_operand=ema_50,
        name="EMA20 crosses above EMA50"
    )
    
    # Create an entry block with both conditions (AND)
    entry_block = BlockConfig(
        name="Golden Cross Entry",
        description="Enter when price is above EMA20 AND EMA20 crosses above EMA50",
        conditions=[condition1, condition2],
        logic_operators=[LogicOperatorType.AND]
    )
    
    # Create an exit block
    exit_condition = ConditionConfig(
        left_operand=price_close,
        operator=OperatorType.LESS,
        right_operand=ema_20,
        name="Price < EMA20"
    )
    
    exit_block = BlockConfig(
        name="Exit below EMA20",
        description="Exit when price falls below EMA20",
        block_type=BlockType.EXIT,
        conditions=[exit_condition]
    )
    
    # Create strategy blocks configuration
    blocks_config = StrategyBlocksConfig()
    blocks_config.add_block(entry_block)
    blocks_config.add_block(exit_block)
    
    # Test serialization
    print("=== Strategy Blocks ===")
    print(blocks_config)
    
    # Serialize to JSON and back
    json_str = blocks_config.to_json()
    print("\n=== JSON ===")
    print(json_str)
    
    # Deserialize
    new_config = StrategyBlocksConfig.from_json(json_str)
    print("\n=== Reconstructed ===")
    print(new_config)
    
    # Validate
    try:
        new_config.validate()
        print("\nValidation: OK")
    except BlockValidationError as e:
        print(f"\nValidation failed: {str(e)}")