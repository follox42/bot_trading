from enum import Enum, auto
from dataclasses import dataclass, field
from typing import ClassVar, List, Dict, Callable, Optional, Any, Union, Tuple, Type
import numpy as np
from numba import njit
import logging

from core.strategy.operand.indicators_config import INDICATOR_REGISTRY
from core.strategy.operand.operand_config import OperandType
from core.strategy.blocs.blocs_config import BlockError

logger = logging.getLogger(__name__)

class ConditionError(BlockError):
    """Base exception for operand-related errors"""
    pass

class ResultNotFoundError(ConditionError):
    """Raised when a result is not found for an operand"""
    pass

class CalculationError(ConditionError):
    """Raised when an error occurs during calculation"""
    pass

class ValidationError(ConditionError):
    """Raised when validation fails"""
    pass

# Operator type enumerations
class OperatorType(Enum):
    """Types of operators for conditions"""
    GREATER = ">"
    LESS = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    CROSS_ABOVE = "CROSS_ABOVE"
    CROSS_BELOW = "CROSS_BELOW"

@dataclass
class Condition:
    """
    Configuration for a single trading condition
    
    A condition compares two operands (e.g., price > EMA50) or applies
    a special operator to a single operand.
    """
    # Required fields
    left_operand: OperandType
    operator: OperatorType
    right_operand: OperandType
    
    def __post_init__(self):
        """Validates condition after initialization"""
        self._validate()
        
        # Set default name if not provided
        if not self.name:
            left_name = getattr(self.left_operand, 'name', str(self.left_operand))
            op_name = getattr(self.operator, 'value', str(self.operator))
            
            if self.right_operand:
                right_name = getattr(self.right_operand, 'name', str(self.right_operand))
                self.name = f"{left_name} {op_name} {right_name}"
            else:
                self.name = f"{left_name} {op_name}"
    
    def _validate(self):
        """Validates the condition's internal consistency"""
        # Check if operator requires two operands
        binary_operators = [
            OperatorType.GREATER, OperatorType.LESS,
            OperatorType.GREATER_EQUAL, OperatorType.LESS_EQUAL,
            OperatorType.EQUAL, OperatorType.CROSS_ABOVE,
            OperatorType.CROSS_BELOW
        ]
        
        if self.operator in binary_operators and self.right_operand is None:
            raise BlockValidationError(
                f"Operator {self.operator.value} requires two operands"
            )
        
        # Validate lookback
        if not isinstance(self.lookback, int):
            raise BlockValidationError(f"Lookback must be an integer, got {type(self.lookback)}")
        
        if self.lookback < 0:
            raise BlockValidationError(f"Lookback must be non-negative, got {self.lookback}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts condition to dictionary format
        
        Returns:
            Dict[str, Any]: Dictionary representation of the condition
        """
        result = {
            "id": self.id,
            "name": self.name,
            "left_operand": self._operand_to_dict(self.left_operand),
            "operator": self.operator.value,
            "lookback": self.lookback
        }
        
        if self.right_operand:
            result["right_operand"] = self._operand_to_dict(self.right_operand)
        
        return result
    
    @staticmethod
    def _operand_to_dict(operand: OperandType) -> Dict[str, Any]:
        """Helper method to convert an operand to dictionary"""
        if hasattr(operand, 'to_dict'):
            return operand.to_dict()
        
        if isinstance(operand, PriceOperand):
            return {
                "type": "price",
                "price_type": operand.price_type
            }
        elif isinstance(operand, IndicatorOperand):
            return {
                "type": "indicator",
                "indicator_id": operand.indicator_id,
                "shift": getattr(operand, 'shift', 0)
            }
        elif isinstance(operand, ValueOperand):
            return {
                "type": "value",
                "value": operand.value
            }
        else:
            return {"type": "unknown", "value": str(operand)}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConditionConfig':
        """
        Creates a condition from dictionary format
        
        Args:
            data: Dictionary representation of the condition
            
        Returns:
            ConditionConfig: Reconstructed condition
            
        Raises:
            BlockSerializationError: If reconstruction fails
        """
        try:
            # Reconstruct operands
            left_operand = cls._dict_to_operand(data["left_operand"])
            
            # Get operator
            operator_str = data["operator"]
            try:
                operator = OperatorType(operator_str)
            except ValueError:
                raise BlockSerializationError(f"Unknown operator: {operator_str}")
            
            # Get optional fields
            right_operand = None
            if "right_operand" in data:
                right_operand = cls._dict_to_operand(data["right_operand"])
            
            lookback = data.get("lookback", 0)
            name = data.get("name", "")
            id = data.get("id", str(uuid.uuid4())[:8])
            
            return cls(
                left_operand=left_operand,
                operator=operator,
                right_operand=right_operand,
                lookback=lookback,
                name=name,
                id=id
            )
        except KeyError as e:
            raise BlockSerializationError(f"Missing required field in condition data: {e}")
        except Exception as e:
            raise BlockSerializationError(f"Error reconstructing condition: {str(e)}")
    
    @staticmethod
    def _dict_to_operand(data: Dict[str, Any]) -> OperandType:
        """Helper method to convert dictionary to operand"""
        operand_type = data.get("type", "").lower()
        
        if operand_type == "price":
            return PriceOperand(data["price_type"])
        elif operand_type == "indicator":
            return IndicatorOperand(
                data["indicator_id"],
                data.get("shift", 0)
            )
        elif operand_type == "value":
            return ValueOperand(data["value"])
        else:
            raise BlockSerializationError(f"Unknown operand type: {operand_type}")
    
    def __str__(self) -> str:
        """String representation of the condition"""
        if self.name:
            return self.name
        
        # Build representation on the fly
        left_str = str(self.left_operand)
        op_str = self.operator.value
        
        if self.right_operand:
            right_str = str(self.right_operand)
            return f"{left_str} {op_str} {right_str}"
        else:
            return f"{left_str} {op_str}"

# Usage example
if __name__ == "__main__":
    # Create simulated data
    close_prices = np.array([10, 11, 12, 11, 10, 9, 10, 11, 12, 13])
    data = {
        'close': close_prices,
        'open': close_prices - 0.5,
        'high': close_prices + 1,
        'low': close_prices - 1
    }
    
    # Create a context with indicators
    context = {
        'EMA20': np.array([10.5, 10.7, 10.9, 11.0, 10.8, 10.5, 10.3, 10.5, 10.8, 11.2]),
        'SMA10': np.array([10.0, 10.2, 10.5, 10.7, 10.8, 10.7, 10.5, 10.5, 10.7, 11.0])
    }
    
    # Create a simple condition: price > EMA20
    price_operand = PriceOperand('close')
    ema_operand = IndicatorOperand('EMA20')
    condition1 = Condition(price_operand, OperatorType.GREATER, ema_operand)
    
    # Create another condition: SMA10 > 10.5
    sma_operand = IndicatorOperand('SMA10')
    value_operand = ValueOperand(10.5)
    condition2 = Condition(sma_operand, OperatorType.GREATER, value_operand)
    
    # Create a composite condition: (price > EMA20) AND (SMA10 > 10.5)
    composite = CompositeCondition([condition1, condition2], [LogicOperatorType.AND])
    
    # Evaluate the conditions
    result1 = condition1.evaluate(data, context)
    result2 = condition2.evaluate(data, context)
    composite_result = composite.evaluate(data, context)
    
    # Display the results
    print(f"Condition 1 ({condition1}): {result1}")
    print(f"Condition 2 ({condition2}): {result2}")
    print(f"Composite ({composite}): {composite_result}")
    
    # JSON serialization
    json_data = composite.to_json()
    print(f"JSON: {json_data}")
    
    # Deserialization
    reconstructed = CompositeCondition.from_json(json_data)
    print(f"Reconstructed: {reconstructed}")
    reconstructed_result = reconstructed.evaluate(data, context)
    print(f"Reconstructed result: {reconstructed_result}")
    
    # Verify that the results are identical
    print(f"Results match: {np.array_equal(composite_result, reconstructed_result)}")