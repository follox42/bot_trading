"""
Module for managing and calculating trading strategy components.
Handles both indicators and operands for consistent evaluation.
"""

import numpy as np
from typing import Dict, Union, Any, Optional, List, Tuple
import logging
from datetime import datetime

# Import BlockError as the base exception
from core.strategy.blocs.blocs_config import BlockError

# Import from the updated error hierarchy
from core.strategy.operand.operand_config import (
    OperandError, ValidationError, CalculationError,
    Operand, PriceOperand, ValueOperand, OperandType
)

# Import from indicators config
from core.strategy.operand.indicators_config import (
    IndicatorOperand, INDICATOR_REGISTRY,
    IndicatorError, IndicatorNotFoundError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultNotFoundError(BlockError):
    """Raised when a result is not found for an indicator"""
    pass

class OperandStrategy:
    """
    A unified class to manage and calculate both indicators and operands 
    for a trading strategy.
    
    This class replaces the original IndicatorStrategy by providing a common
    interface for working with both indicators and operands.
    """
    def __init__(self, name: str):
        """
        Initialize a new strategy
        
        Args:
            name: Strategy name
        """
        self.name = name
        self.elements = {}    # Dictionary to store all elements (indicators/operands) with their IDs
        self.results = {}     # Dictionary to store calculation results
        self.data_length = 0  # Length of the data used in last calculation
        self.last_updated = None
    
    def add_operand(self, element_id: str, element: OperandType):
        """
        Add an operand (indicator or value or price) to the strategy
        
        Args:
            element_id: Unique identifier for this element usualy indicator_period_val...
            element: The element instance to add
            
        Returns:
            Self for method chaining
            
        Raises:
            ValidationError: If the element ID is invalid or already exists
        """
        # Validate element_id
        if not element_id or not isinstance(element_id, str):
            raise ValidationError("Element ID must be a non-empty string")
        
        if element_id in self.elements:
            raise ValidationError(f"Element ID '{element_id}' already exists in the strategy")
        
        # Validate element is a proper operand type
        if not isinstance(element, Operand):
            raise ValidationError(f"Expected Operand subclass, got {type(element).__name__}")
        
        self.elements[element_id] = element
        logger.info(f"Added {type(element).__name__} '{element_id}' ({element}) to strategy '{self.name}'")
        return self
    
    def remove_element(self, element_id: str):
        """
        Remove an element from the strategy
        
        Args:
            element_id: The element identifier to remove
            
        Returns:
            Self for method chaining
            
        Raises:
            ValidationError: If the element ID is not found
        """
        if element_id not in self.elements:
            raise ValidationError(f"No element found with ID '{element_id}'")
        
        element = self.elements[element_id]
        del self.elements[element_id]
        
        # Also remove results if they exist
        if element_id in self.results:
            del self.results[element_id]
        
        logger.info(f"Removed {type(element).__name__} '{element_id}' ({element}) from strategy '{self.name}'")
        return self
    
    def get_element(self, element_id: str) -> OperandType:
        """
        Get an element by its ID
        
        Args:
            element_id: The element identifier
            
        Returns:
            The element instance
            
        Raises:
            ValidationError: If the element ID is not found
        """
        if element_id not in self.elements:
            raise ValidationError(f"No element found with ID '{element_id}'")
        
        return self.elements[element_id]
    
    def get_elements(self) -> Dict[str, OperandType]:
        """
        Get all elements in the strategy
        
        Returns:
            Dictionary of all elements
        """
        return self.elements.copy()
    
    def get_elements_by_type(self, element_type: type) -> Dict[str, OperandType]:
        """
        Get elements of a specific type
        
        Args:
            element_type: The type of element to filter for
            
        Returns:
            Dictionary of elements of the specified type
        """
        return {id: elem for id, elem in self.elements.items() if isinstance(elem, element_type)}
    
    def has_element(self, element_id: str) -> bool:
        """
        Check if an element exists in the strategy
        
        Args:
            element_id: The element identifier
            
        Returns:
            True if the element exists, False otherwise
        """
        return element_id in self.elements
    
    def calculate_all(self, data: Union[np.ndarray, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Calculate all elements on the provided data in the correct order
        
        Args:
            data: Input data (array or dictionary)
            
        Returns:
            Dictionary with calculation results
            
        Raises:
            CalculationError: If an error occurs during calculation
        """
        # Validate input data
        if isinstance(data, dict):
            if 'close' not in data:
                raise ValidationError("Input data dictionary must contain 'close' key")
            array_length = len(data['close'])
        elif isinstance(data, np.ndarray):
            array_length = len(data)
        else:
            try:
                data = np.array(data, dtype=np.float64)
                array_length = len(data)
            except Exception as e:
                raise ValidationError(f"Invalid data format. Expected numpy array or dictionary: {str(e)}")
        
        if array_length == 0:
            raise ValidationError("Empty data provided for calculation")
        
        # Reset results
        self.results = {}
        error_elements = []
        
        # Create calculation order by element type
        calculation_order = [
            # First indicators (IndicatorOperand)
            (IndicatorOperand, "indicator"),
            # Then price operands
            (PriceOperand, "price"),
            # Then value operands
            (ValueOperand, "value"),
            # Finally any remaining custom operands
            (Operand, "custom")
        ]
        
        # Process each type in order
        for element_type, type_name in calculation_order:
            elements_of_type = self.get_elements_by_type(element_type)
            if element_type == Operand:  # For custom operands, exclude already processed types
                elements_of_type = {
                    id: elem for id, elem in elements_of_type.items() 
                    if not isinstance(elem, (IndicatorOperand, PriceOperand, ValueOperand))
                }
            
            for element_id, element in elements_of_type.items():
                try:
                    # Try with context first, fall back to without context
                    try:
                        self.results[element_id] = element.calculate(data, self.results)
                    except (TypeError, ValueError):
                        # If that fails, try without context
                        self.results[element_id] = element.calculate(data)
                    
                    logger.debug(f"Calculated {type_name} '{element_id}' ({element})")
                except Exception as e:
                    error_elements.append((element_id, str(e)))
                    logger.error(f"Error calculating {type_name} '{element_id}': {str(e)}")
        
        # Update metadata
        self.last_updated = datetime.now()
        self.data_length = array_length
        
        # Report errors if any
        if error_elements:
            error_msg = "; ".join([f"{id}: {err}" for id, err in error_elements])
            raise CalculationError(f"Errors calculating elements: {error_msg}")
        
        logger.info(f"Calculated all elements for strategy '{self.name}' (data length: {self.data_length})")
        return self.results
    
    def has_results(self, element_id: str = None) -> bool:
        """
        Check if results are available for an element or any element
        
        Args:
            element_id: Optional element ID to check, or None to check if any results exist
            
        Returns:
            True if results exist, False otherwise
        """
        if element_id is None:
            return len(self.results) > 0
        
        return element_id in self.results
    
    def get_result(self, element_id: str, default: Any = None) -> Union[np.ndarray, Any]:
        """
        Get the calculation result for a specific element
        
        Args:
            element_id: The element identifier
            default: Optional default value to return if result doesn't exist
            
        Returns:
            The calculation result or default value
            
        Raises:
            ResultNotFoundError: If the element result is not found and no default is provided
        """
        # Check if the element ID exists in the strategy
        if element_id not in self.elements:
            if default is not None:
                logger.warning(f"Element '{element_id}' not found in strategy '{self.name}', returning default value")
                return default
            raise ValidationError(f"No element found with ID '{element_id}'")
        
        # Check if results exist for this element
        if element_id not in self.results:
            if default is not None:
                logger.warning(f"No results for element '{element_id}', returning default value. Did you run calculate_all?")
                return default
            raise ResultNotFoundError(f"No results found for element '{element_id}'. Did you run calculate_all?")
        
        return self.results[element_id]
    
    def get_results(self) -> Dict[str, np.ndarray]:
        """
        Get all calculation results
        
        Returns:
            Dictionary with all results
        """
        return self.results.copy()