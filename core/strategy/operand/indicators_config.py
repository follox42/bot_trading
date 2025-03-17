from enum import Enum
from dataclasses import dataclass
from typing import ClassVar, List, Callable, Optional, Any
import numpy as np
from numba import njit

from core.strategy.operand.operand_config import Operand, Parameter, ParamType
from core.strategy.blocs.blocs_config import BlockError

# Define a proper error hierarchy
class OperandError(BlockError):
    """Base exception for all operand-related errors"""
    pass

class IndicatorError(OperandError):
    """Base exception for indicator-related errors"""
    pass

class CalculationError(IndicatorError):
    """Raised when an error occurs during indicator calculation"""
    pass

class ValidationError(IndicatorError):
    """Raised when indicator validation fails"""
    pass

class IndicatorNotFoundError(IndicatorError):
    """Raised when an indicator is not found in the registry"""
    pass

class IndicatorOperand(Operand):
    """
    Base class for technical indicators
    
    Class attributes:
        name: The indicator name
        parameters: List of Parameter objects defining the indicator inputs
        function: The calculation function for the indicator
    """
    name: ClassVar[str] = ""
    parameters: ClassVar[List[Parameter]] = []
    function: ClassVar[Callable] = None
    
    def __init__(self, **kwargs):
        """
        Initialize the indicator with parameter values
        
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
            raise NotImplementedError(f"Function not defined for indicator {self.__class__.name}")
    
    def calculate(self, data):
        """
        Calculate the indicator using the stored function and parameters
        """
        try:
            return self.__class__.function(data, **self.params)
        except Exception as e:
            raise CalculationError(f"Error calculating {self.__class__.name}: {str(e)}")

    @classmethod
    def from_json(cls, data):
        """
        Create indicator from JSON data
        
        Args:
            data: Dictionary with indicator data
            
        Returns:
            Indicator instance
        """
        try:
            indicator_type = data["type"]
            params = data["params"]
            
            # Get the indicator class from registry
            if indicator_type not in INDICATOR_REGISTRY:
                raise IndicatorNotFoundError(f"Unknown indicator type: {indicator_type}")
            
            indicator_class = INDICATOR_REGISTRY[indicator_type]
            return indicator_class(**params)
        except KeyError as e:
            raise ValidationError(f"Missing required field in indicator JSON: {str(e)}")
        except Exception as e:
            raise ValidationError(f"Error creating indicator from JSON: {str(e)}")
    
@njit(cache=True)
def calculate_ema(data: np.ndarray, period: int, alpha: Optional[float] = None) -> np.ndarray:
    """
    Calculate the Exponential Moving Average (EMA)
    
    The EMA gives more weight to recent prices compared to a simple moving average.
    
    Args:
        data: Source data array
        period: EMA period
        alpha: Smoothing coefficient (if None, uses 2/(period+1))
        
    Returns:
        Array containing the EMA values
    """
    n = len(data)
    ema = np.zeros(n, dtype=np.float64)
    
    # If alpha is not specified, use standard formula
    if alpha is None:
        alpha = 2.0 / (period + 1)
    
    # Initialize: first value is the initial value
    ema[0] = data[0]
    
    # Calculate EMA
    for i in range(1, n):
        ema[i] = data[i] * alpha + ema[i-1] * (1 - alpha)
    
    return ema
class EMA(IndicatorOperand):
    """
    Exponential Moving Average (EMA) indicator
    
    The EMA is a type of weighted moving average that gives more weight to recent data points.
    """
    name = "EMA"  # Indicator name
    parameters = [
        Parameter("period", ParamType.INT, required=True),  # Period for the EMA 
        Parameter("alpha", ParamType.FLOAT, required=False) # Optional smoothing factor
    ]
    function = staticmethod(calculate_ema)  # Static reference to the calculation function

@njit(cache=True)
def calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate the Simple Moving Average (SMA).
    
    Args:
        data: Source data array
        period: SMA period
        
    Returns:
        Array containing the SMA values
    """
    n = len(data)
    sma = np.zeros(n, dtype=np.float64)
    
    # Optimized method: cumulative calculation
    # Initialize: first (period-1) values remain at 0
    if n >= period:
        # Calculate first SMA
        sma[period-1] = np.sum(data[:period]) / period
        
        # Calculate remaining values in an optimized way
        for i in range(period, n):
            sma[i] = sma[i-1] + (data[i] - data[i-period]) / period
    
    return sma
class SMA(IndicatorOperand):
    """
    Simple Moving Average (SMA) indicator
    
    The SMA calculates the arithmetic mean of a given set of prices over a specified period, giving equal weight to each price point.
    """
    name = "SMA"  # Indicator name
    parameters = [
        Parameter("period", ParamType.INT, required=True)  # Period for the SMA
    ]
    function = staticmethod(calculate_sma)  # Static reference to the calculation function

INDICATOR_REGISTRY = {"EMA": EMA,
                      "SMA": SMA}

# Add a new indicator:
#   1. Create its numba function
#   2. Create its indicator class
#   3. Register it in the INDICATOR_REGISTRY dictionary