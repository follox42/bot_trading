from enum import Enum, auto
from dataclasses import dataclass, field
from typing import ClassVar, List, Dict, Callable, Optional, Any, Union, Tuple, Type
import numpy as np
from numba import njit
import logging

# Import BlockError as the base exception
from core.strategy.blocs.blocs_config import BlockError

logger = logging.getLogger(__name__)

# Define a proper error hierarchy
class OperandError(BlockError):
    """Base exception for operand-related errors"""
    pass

class CalculationError(OperandError):
    """Raised when an error occurs during calculation"""
    pass

class ValidationError(OperandError):
    """Raised when validation fails"""
    pass

class PriceError(OperandError):
    """Raised for price operand related errors"""
    pass

class ValueError(OperandError):
    """Raised for value operand related errors"""
    pass

# Enum defining the possible parameter types for operand
class ParamType(Enum):
    INT = "int"     # Integer parameter type
    FLOAT = "float" # Floating-point parameter type
    STRING = "str"  # String parameter type
    BOOLEAN = "bool" # Boolean parameter type
    
    def convert(self, value):
        """
        Convert a value to the appropriate Python type based on the parameter type
        
        Args:
            value: The value to convert
            
        Returns:
            The converted value or None if input is None
        """
        if value is None:
            return None
        
        converters = {
            self.INT: int,
            self.FLOAT: float,
            self.STRING: str,
            self.BOOLEAN: bool
        }
        return converters[self](value)

    def to_json(self):
        """Convert the enum to a JSON serializable format"""
        return self.value
    
@dataclass
class Parameter:
    """
    Class representing a parameter with its metadata
    """
    name: str           # Parameter name
    type: ParamType     # Parameter type (INT, FLOAT, etc.)
    required: bool = True   # Whether the parameter is required

    def validate(self, value):
        """
        Validate and convert a parameter value
        """
        if value is None:
            if self.required is None:
                raise ValidationError(f"Required parameter: {self.name}")
        
        try:
            return self.type.convert(value)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid value '{value}' for parameter '{self.name}' of type {self.type.value}: {str(e)}")
    
    def to_json(self):
        """Convert parameter to JSON serializable format"""
        return {
            "name": self.name,
            "type": self.type.to_json(),
            "required": self.required
        }
    
    @classmethod
    def from_json(cls, data):
        """Create parameter from JSON data"""
        try:
            return cls(
                name=data["name"],
                type=ParamType(data["type"]),
                required=data["required"]
            )
        except KeyError as e:
            raise ValidationError(f"Missing required field in parameter JSON: {str(e)}")
        except ValueError as e:
            raise ValidationError(f"Invalid value in parameter JSON: {str(e)}")

class Operand:
    """
    Base class for operand
    
    Class attributes:
        name: The indicator name
        parameters: List of Parameter objects defining the indicator inputs
        function: The calculation function for the operand
    """
    name: ClassVar[str] = ""
    parameters: ClassVar[List[Parameter]] = []
    function: ClassVar[Callable] = None
    
    def __init__(self, **kwargs):
        """
        Initialize the operand with parameter values
        
        Args:
            **kwargs: Parameter values passed as keyword arguments
            
        Raises:
            NotImplementedError: If the calculation function is not defined
        """
        # Validate and store parameters
        self.params = {}
        for param in self.__class__.parameters:
            self.params[param.name] = param.validate(kwargs.get(param.name))
        
        # Check if function is implemented
        if self.__class__.function is None:
            raise NotImplementedError(f"Function not defined for operand {self.__class__.name}")
    
    def calculate(self, data):
        """
        Calculate the operand using the stored function and parameters
        the function: 
            Args:
                **kwargs: Parameter values passed as keyword arguments

            Returns:
                Numpy array of the operand
        """
        try:
            return self.__class__.function(data, **self.params)
        except Exception as e:
            raise CalculationError(f"Error calculating {self.__class__.name}: {str(e)}")
    
    def __repr__(self):
        """String representation of the operand with its parameters"""
        params_str = ", ".join([f"{k}={v}" for k, v in self.params.items()])
        return f"{self.__class__.name}({params_str})"
    
    def to_json(self):
        """
        Convert operand to JSON serializable format
        
        Returns:
            Dictionary representation of the operand
        """
        return {
            "type": self.__class__.name,
            "params": self.params
        }
    
    @classmethod
    def from_json(cls, data):
        """
        Create operand from JSON data
        
        Args:
            data: Dictionary with operand data
            
        Returns:
            Operand instance
        """
        try:
            operand_type = data["type"]
            params = data["params"]
            
            # Get the operand class from registry
            from core.strategy.operand.indicators_config import INDICATOR_REGISTRY
            
            if operand_type not in INDICATOR_REGISTRY:
                raise ValidationError(f"Unknown operand type: {operand_type}")
            
            operand_class = INDICATOR_REGISTRY[operand_type]
            return operand_class(**params)
        except KeyError as e:
            raise ValidationError(f"Missing required field in operand JSON: {str(e)}")
        except Exception as e:
            raise ValidationError(f"Error creating operand from JSON: {str(e)}")

class PriceOperand(Operand):
    """
    Operand representing a price type
    """
    name = "PRICE"
    parameters = [
        Parameter("price_type", ParamType.STRING, required=True)
    ]
    
    def __init__(self, price_type: str):
        """
        Initializes a price operand
        
        Args:
            price_type: Price type ('open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4')
        """
        self.price_type = price_type
        super().__init__(price_type=price_type)
    
    def calculate(self, data: np.ndarray, context: Dict[str, np.ndarray] = None) -> np.ndarray:
        """
        Calculates price values according to the requested type
        
        Args:
            data: Input data (dictionary or array)
            context: Optional context with pre-calculated results
            
        Returns:
            Numpy array with price values
        """
        # If data is a dictionary, assume it contains 'open', 'high', 'low', 'close' keys
        if isinstance(data, dict):
            if self.price_type == 'open':
                return data.get('open', data.get('close', np.array([])))
            elif self.price_type == 'high':
                return data.get('high', data.get('close', np.array([])))
            elif self.price_type == 'low':
                return data.get('low', data.get('close', np.array([])))
            elif self.price_type == 'close':
                return data.get('close', np.array([]))
            elif self.price_type == 'hl2':
                high = data.get('high', data.get('close', np.array([])))
                low = data.get('low', data.get('close', np.array([])))
                return (high + low) / 2
            elif self.price_type == 'hlc3':
                high = data.get('high', data.get('close', np.array([])))
                low = data.get('low', data.get('close', np.array([])))
                close = data.get('close', np.array([]))
                return (high + low + close) / 3
            elif self.price_type == 'ohlc4':
                open_price = data.get('open', data.get('close', np.array([])))
                high = data.get('high', data.get('close', np.array([])))
                low = data.get('low', data.get('close', np.array([])))
                close = data.get('close', np.array([]))
                return (open_price + high + low + close) / 4
            else:
                raise PriceError(f"Unknown price type: {self.price_type}")
        else:
            # If data is an array, assume it's the close series
            return data
    
    def to_json(self) -> Dict[str, Any]:
        """Serializes to JSON"""
        result = super().to_json()
        result["price_type"] = self.price_type
        return result
    
class ValueOperand(Operand):
    """
    Operand representing a constant value
    """
    name = "VALUE"
    parameters = [
        Parameter("value", ParamType.FLOAT, required=True)
    ]
    
    def __init__(self, value: float):
        """
        Initializes a constant value operand
        
        Args:
            value: Fixed numeric value
        """
        self.value = value
        super().__init__(value=value)
    
    def calculate(self, data: np.ndarray, context: Dict[str, np.ndarray] = None) -> np.ndarray:
        """
        Returns an array of the same length as data with the constant value
        
        Args:
            data: Input data (used only for length)
            context: Context (not used)
            
        Returns:
            Numpy array with constant values
        """
        if isinstance(data, dict) and 'close' in data:
            length = len(data['close'])
        elif isinstance(data, np.ndarray):
            length = len(data)
        else:
            raise ValueError(f"Data should be a dictionary with close column or an ndarray but got {type(data)}")
            
        return np.full(length, self.value)
    
    def to_json(self) -> Dict[str, Any]:
        """Serializes to JSON"""
        result = super().to_json()
        result["value"] = self.value
        return result

# Operand type for annotations
from core.strategy.operand.indicators_config import IndicatorOperand
OperandType = Union[PriceOperand, IndicatorOperand, ValueOperand]