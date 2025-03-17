"""
Module de gestion des positions et du risque pour les stratégies de trading.
Ce module fournit des méthodes optimisées pour calculer les tailles de position,
les niveaux de stop loss et de take profit.
"""

import numpy as np
from numba import njit, float64
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from enum import Enum

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('position_calculator.log', mode='a'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('position_calculator')

class RiskMode(Enum):
    """Types de gestion du risque"""
    FIXED = "fixed"
    ATR_BASED = "atr_based" 
    VOLATILITY_BASED = "volatility_based"

class PositionCalculator:
    """
    Gestionnaire de calcul des positions et du risque.
    """
    
    def __init__(self, mode: RiskMode = RiskMode.FIXED, config: Dict = None):
        """
        Initialise le calculateur de position.
        
        Args:
            mode: Mode de calcul du risque 
            config: Configuration des paramètres de risque
        """
        self.mode = mode
        self.config = config or {}
        
        # Paramètres par défaut
        self.base_position = self.config.get('base_position', 0.1)  # 10% du capital
        self.base_sl = self.config.get('base_sl', 0.01)  # 1% du prix
        self.tp_multiplier = self.config.get('tp_multiplier', 2.0)  # 2x le SL
        
        # Paramètres spécifiques au mode ATR
        self.atr_period = self.config.get('atr_period', 14)
        self.atr_multiplier = self.config.get('atr_multiplier', 1.5)
        
        # Paramètres spécifiques au mode volatilité
        self.vol_period = self.config.get('vol_period', 20)
        self.vol_multiplier = self.config.get('vol_multiplier', 1.0)
        
        # Limites
        self.position_size_range = self.config.get('position_size_range', (0.01, 1.0))
        self.sl_range = self.config.get('sl_range', (0.001, 0.1))
    
    def calculate_risk_parameters(
        self,
        prices: np.ndarray,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcule les paramètres de position et de risque pour chaque point de données.
        
        Args:
            prices: Array des prix
            high: Array des prix hauts (optionnel, requis pour le mode ATR)
            low: Array des prix bas (optionnel, requis pour le mode ATR)
            
        Returns:
            Tuple contenant (position_sizes, sl_levels, tp_levels) pour chaque point de données
        """
        if self.mode == RiskMode.FIXED:
            return self._fixed_parameters(prices)
        elif self.mode == RiskMode.ATR_BASED:
            if high is None or low is None:
                raise ValueError("Les données high et low sont requises pour le mode ATR")
            return self._atr_parameters(prices, high, low)
        elif self.mode == RiskMode.VOLATILITY_BASED:
            return self._volatility_parameters(prices)
        else:
            raise ValueError(f"Mode de gestion du risque non reconnu: {self.mode}")
    
    def _fixed_parameters(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcule les paramètres fixes pour chaque point de données.
        
        Args:
            prices: Array des prix
            
        Returns:
            Tuple contenant (position_sizes, sl_levels, tp_levels)
        """
        n = len(prices)
        position_sizes = np.full(n, self.base_position)
        sl_levels = np.full(n, self.base_sl)
        tp_levels = np.full(n, self.base_sl * self.tp_multiplier)
        
        return position_sizes, sl_levels, tp_levels
    
    def _atr_parameters(
        self,
        prices: np.ndarray,
        high: np.ndarray,
        low: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcule les paramètres basés sur l'ATR pour chaque point de données.
        
        Args:
            prices: Array des prix
            high: Array des prix hauts 
            low: Array des prix bas
            
        Returns:
            Tuple contenant (position_sizes, sl_levels, tp_levels)
        """
        atr = self._calculate_atr(high, low, prices, self.atr_period)
        
        # Calcul des paramètres
        return self._dynamic_parameters_from_volatility(prices, atr / prices, self.atr_multiplier)
    
    def _volatility_parameters(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcule les paramètres basés sur la volatilité pour chaque point de données.
        
        Args:
            prices: Array des prix
            
        Returns:
            Tuple contenant (position_sizes, sl_levels, tp_levels)
        """
        # Calcul de la volatilité à partir des rendements
        volatility = self._calculate_volatility(prices, self.vol_period)
        
        # Calcul des paramètres
        return self._dynamic_parameters_from_volatility(prices, volatility, self.vol_multiplier)
    
    def _dynamic_parameters_from_volatility(
        self,
        prices: np.ndarray,
        volatility: np.ndarray,
        multiplier: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcule les paramètres dynamiques à partir d'une mesure de volatilité.
        
        Args:
            prices: Array des prix
            volatility: Mesure de volatilité (en %)
            multiplier: Facteur multiplicateur pour la volatilité
            
        Returns:
            Tuple contenant (position_sizes, sl_levels, tp_levels)
        """
        n = len(prices)
        
        # Application du multiplicateur
        vol_adjusted = volatility * multiplier
        
        # Stop-loss basé sur la volatilité (avec bornes)
        # MODIFICATION: Remplacer np.clip par une boucle avec min/max
        sl_levels = np.zeros(n, dtype=np.float64)
        for i in range(n):
            sl_levels[i] = min(self.sl_range[1], max(self.sl_range[0], vol_adjusted[i]))
        
        # Take-profit comme multiple du stop-loss
        tp_levels = sl_levels * self.tp_multiplier
        
        # Taille de position inversement proportionnelle au risque
        position_sizes = np.zeros(n, dtype=np.float64)
        
        # Éviter les divisions par zéro
        safe_vol = np.maximum(vol_adjusted, 1e-6)
        
        # Calculer les positions
        # MODIFICATION: Remplacer np.clip par une boucle avec min/max
        for i in range(n):
            position_sizes[i] = min(self.position_size_range[1], 
                                    max(self.position_size_range[0], 
                                        self.base_position / safe_vol[i]))
        
        return position_sizes, sl_levels, tp_levels
    
    @staticmethod
    @njit(cache=True)
    def _calculate_atr(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int
    ) -> np.ndarray:
        """
        Calcul optimisé de l'ATR (Average True Range).
        
        Args:
            high: Array des prix hauts
            low: Array des prix bas
            close: Array des prix de clôture
            period: Période pour le calcul
            
        Returns:
            Array de l'ATR
        """
        tr = np.zeros_like(high, dtype=np.float64)
        atr = np.zeros_like(high, dtype=np.float64)
        
        # Calcul du True Range
        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        # Calcul de l'ATR initial
        if period <= len(high):
            atr[period] = np.mean(tr[1:period+1])
        
        # Calcul de l'ATR par moyenne mobile exponentielle
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        return atr
    
    @staticmethod
    @njit(cache=True)
    def _calculate_volatility(prices: np.ndarray, period: int) -> np.ndarray:
        """
        Calcul optimisé de la volatilité à partir des rendements.
        
        Args:
            prices: Array des prix
            period: Période pour le calcul
            
        Returns:
            Array de la volatilité
        """
        n = len(prices)
        returns = np.zeros(n, dtype=np.float64)
        volatility = np.zeros(n, dtype=np.float64)
        
        # Calcul des rendements
        for i in range(1, n):
            if prices[i-1] > 0:
                returns[i] = (prices[i] - prices[i-1]) / prices[i-1]
        
        # Calcul de la volatilité glissante
        for i in range(period, n):
            volatility[i] = np.std(returns[i-period+1:i+1])
        
        # Remplir les valeurs initiales
        for i in range(period):
            volatility[i] = volatility[period]
        
        return volatility
    
    def update_parameters(self, config: Dict) -> None:
        """
        Met à jour les paramètres du calculateur de position.
        
        Args:
            config: Nouveaux paramètres de configuration
        """
        self.config.update(config)
        
        # Mise à jour des paramètres de base
        self.base_position = self.config.get('base_position', self.base_position)
        self.base_sl = self.config.get('base_sl', self.base_sl)
        self.tp_multiplier = self.config.get('tp_multiplier', self.tp_multiplier)
        
        # Mise à jour des paramètres spécifiques au mode
        self.atr_period = self.config.get('atr_period', self.atr_period)
        self.atr_multiplier = self.config.get('atr_multiplier', self.atr_multiplier)
        self.vol_period = self.config.get('vol_period', self.vol_period)
        self.vol_multiplier = self.config.get('vol_multiplier', self.vol_multiplier)
        
        # Mise à jour des limites
        self.position_size_range = self.config.get('position_size_range', self.position_size_range)
        self.sl_range = self.config.get('sl_range', self.sl_range)
    
    def set_mode(self, mode: RiskMode) -> None:
        """
        Change le mode de calcul du risque.
        
        Args:
            mode: Nouveau mode de gestion du risque
        """
        self.mode = mode
        logger.info(f"Mode de gestion du risque changé pour: {mode.value}")
    
    @staticmethod
    def calculate_dynamic_position_sizing(
        equity: float,
        price: float,
        risk_percent: float, 
        sl_percent: float,
        leverage: float = 1.0
    ) -> float:
        """
        Calcule la taille de position en fonction du capital, du risque et du stop loss.
        
        Args:
            equity: Capital disponible
            price: Prix actuel
            risk_percent: Pourcentage du capital à risquer (0.01 = 1%)
            sl_percent: Pourcentage de stop loss (0.01 = 1%)
            leverage: Effet de levier (1.0 = pas de levier)
            
        Returns:
            Taille de position en unités (ex: BTC)
        """
        # Calcul du montant à risquer
        risk_amount = equity * risk_percent
        
        # Calcul du montant par unité risqué au stop loss
        risk_per_unit = price * sl_percent
        
        if risk_per_unit <= 0:
            return 0.0
        
        # Calcul de la taille de position
        position_size = risk_amount / risk_per_unit
        
        # Application du levier
        position_size *= leverage
        
        return position_size