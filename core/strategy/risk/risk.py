"""
Module de calcul des paramètres de risque optimisé avec Numba.
Chaque fonction est spécifique à un type de risque.
"""

import numpy as np
import pandas as pd
from numba import njit, prange, float64
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

from core.strategy.risk.risk_config import RiskConfig, RiskModeType

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@njit(cache=True)
def calculate_fixed_risk_params(prices: np.ndarray, position_size: float, stop_loss: float, 
                              take_profit: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule les paramètres de risque fixes.
    
    Args:
        prices: Array des prix
        position_size: Taille de position (pourcentage du capital)
        stop_loss: Stop loss (pourcentage du prix)
        take_profit: Take profit (pourcentage du prix)
        
    Returns:
        Tuple de (position_sizes, sl_levels, tp_levels)
    """
    n = len(prices)
    position_sizes = np.full(n, position_size, dtype=np.float64)
    sl_levels = np.full(n, stop_loss, dtype=np.float64)
    tp_levels = np.full(n, take_profit, dtype=np.float64)
    
    return position_sizes, sl_levels, tp_levels


@njit(cache=True)
def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Calcule l'Average True Range (ATR).
    
    Args:
        high: Array des prix hauts
        low: Array des prix bas
        close: Array des prix de clôture
        period: Période pour le calcul de l'ATR
        
    Returns:
        Array de l'ATR
    """
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    atr = np.zeros(n, dtype=np.float64)
    
    # Calcul du True Range
    tr[0] = high[0] - low[0]  # Premier jour
    
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    # Calcul de l'ATR (moyenne mobile exponentielle)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = ((period - 1) * atr[i-1] + tr[i]) / period
    
    return atr


@njit(cache=True)
def calculate_atr_risk_params(prices: np.ndarray, high: np.ndarray, low: np.ndarray, 
                            atr_period: int, atr_multiplier: float, risk_per_trade: float,
                            tp_multiplier: float, max_position_size: float, 
                            min_position_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule les paramètres de risque basés sur l'ATR.
    
    Args:
        prices: Array des prix de clôture
        high: Array des prix hauts
        low: Array des prix bas
        atr_period: Période pour le calcul de l'ATR
        atr_multiplier: Multiplicateur appliqué à l'ATR
        risk_per_trade: Pourcentage du capital risqué par trade
        tp_multiplier: Multiplicateur pour le take profit
        max_position_size: Taille maximale de position
        min_position_size: Taille minimale de position
        
    Returns:
        Tuple de (position_sizes, sl_levels, tp_levels)
    """
    n = len(prices)
    position_sizes = np.zeros(n, dtype=np.float64)
    sl_levels = np.zeros(n, dtype=np.float64)
    tp_levels = np.zeros(n, dtype=np.float64)
    
    # Calcul de l'ATR
    atr = calculate_atr(high, low, prices, atr_period)
    
    # Calcul des niveaux de stop loss (en pourcentage)
    for i in range(n):
        # Éviter les divisions par zéro
        if prices[i] > 0:
            sl_distance = atr[i] * atr_multiplier / prices[i]
            sl_levels[i] = sl_distance
            tp_levels[i] = sl_distance * tp_multiplier
            
            # Calcul de la taille de position
            if sl_distance > 0:
                # Formule: position_size = (capital * risk_per_trade) / (prix * sl_distance)
                # Mais comme on travaille en pourcentages, on simplifie:
                position_sizes[i] = min(max_position_size, max(min_position_size, risk_per_trade / sl_distance))
            else:
                position_sizes[i] = min_position_size
        else:
            sl_levels[i] = 0
            tp_levels[i] = 0
            position_sizes[i] = min_position_size
    
    return position_sizes, sl_levels, tp_levels


@njit(cache=True)
def calculate_volatility(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calcule la volatilité (écart-type des rendements).
    
    Args:
        prices: Array des prix
        period: Période pour le calcul de la volatilité
        
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
    
    # Calcul de la volatilité (écart-type mobile)
    for i in range(period, n):
        sum_squared_dev = 0.0
        mean_return = 0.0
        
        # Calcul de la moyenne
        for j in range(i - period + 1, i + 1):
            mean_return += returns[j]
        
        mean_return /= period
        
        # Calcul de la somme des carrés des écarts
        for j in range(i - period + 1, i + 1):
            dev = returns[j] - mean_return
            sum_squared_dev += dev * dev
        
        # Écart-type
        volatility[i] = np.sqrt(sum_squared_dev / period)
    
    # Remplir les premières valeurs
    for i in range(period):
        volatility[i] = volatility[period]
    
    return volatility


@njit(cache=True)
def calculate_volatility_risk_params(prices: np.ndarray, 
                                  vol_period: int, vol_multiplier: float, risk_per_trade: float,
                                  tp_multiplier: float, max_position_size: float, 
                                  min_position_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule les paramètres de risque basés sur la volatilité.
    
    Args:
        prices: Array des prix
        vol_period: Période pour le calcul de la volatilité
        vol_multiplier: Multiplicateur appliqué à la volatilité
        risk_per_trade: Pourcentage du capital risqué par trade
        tp_multiplier: Multiplicateur pour le take profit
        max_position_size: Taille maximale de position
        min_position_size: Taille minimale de position
        
    Returns:
        Tuple de (position_sizes, sl_levels, tp_levels)
    """
    n = len(prices)
    position_sizes = np.zeros(n, dtype=np.float64)
    sl_levels = np.zeros(n, dtype=np.float64)
    tp_levels = np.zeros(n, dtype=np.float64)
    
    # Calcul de la volatilité
    volatility = calculate_volatility(prices, vol_period)
    
    # Calcul des niveaux de stop loss et tailles de position
    for i in range(n):
        # Niveau de stop loss basé sur la volatilité
        sl_distance = volatility[i] * vol_multiplier
        sl_levels[i] = sl_distance
        tp_levels[i] = sl_distance * tp_multiplier
        
        # Calcul de la taille de position
        if sl_distance > 0:
            position_sizes[i] = min(max_position_size, max(min_position_size, risk_per_trade / sl_distance))
        else:
            position_sizes[i] = min_position_size
    
    return position_sizes, sl_levels, tp_levels


@njit(cache=True)
def calculate_equity_percent_risk_params(prices: np.ndarray,
                                      risk_percent: float, max_position_size: float,
                                      tp_multiplier: float, min_stop_distance: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule les paramètres de risque basés sur un pourcentage du capital.
    
    Args:
        prices: Array des prix
        risk_percent: Pourcentage du capital risqué par trade
        max_position_size: Taille maximale de position
        tp_multiplier: Multiplicateur pour le take profit
        min_stop_distance: Distance minimale du stop loss
        
    Returns:
        Tuple de (position_sizes, sl_levels, tp_levels)
    """
    n = len(prices)
    position_sizes = np.zeros(n, dtype=np.float64)
    sl_levels = np.zeros(n, dtype=np.float64)
    tp_levels = np.zeros(n, dtype=np.float64)
    
    # Calcul des paramètres pour chaque point
    for i in range(n):
        # Niveau de stop loss fixe (minimum)
        sl_distance = max(min_stop_distance, 0.01)  # Au moins 1%
        sl_levels[i] = sl_distance
        tp_levels[i] = sl_distance * tp_multiplier
        
        # Calcul de la taille de position
        if sl_distance > 0:
            position_sizes[i] = min(max_position_size, risk_percent / sl_distance)
        else:
            position_sizes[i] = 0
    
    return position_sizes, sl_levels, tp_levels


@njit(cache=True)
def calculate_kelly_risk_params(prices: np.ndarray, win_rate: float, win_loss_ratio: float,
                              fraction: float, max_position_size: float,
                              min_position_size: float) -> np.ndarray:
    """
    Calcule la taille de position selon le critère de Kelly.
    
    Args:
        prices: Array des prix
        win_rate: Taux de réussite des trades
        win_loss_ratio: Ratio gain moyen / perte moyenne
        fraction: Fraction du critère de Kelly à utiliser (0.5 = Half Kelly)
        max_position_size: Taille maximale de position
        min_position_size: Taille minimale de position
        
    Returns:
        Array des tailles de position
    """
    n = len(prices)
    position_sizes = np.zeros(n, dtype=np.float64)
    
    # Formule de Kelly: f* = (p * b - q) / b
    # où p = probabilité de gain, q = probabilité de perte (1-p), b = ratio gain/perte
    
    kelly_size = 0.0
    if win_loss_ratio > 0:
        kelly_size = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Appliquer la fraction et les limites
        kelly_size = kelly_size * fraction
        kelly_size = min(max_position_size, max(min_position_size, kelly_size))
    else:
        kelly_size = min_position_size
    
    # Taille de position constante
    position_sizes.fill(kelly_size)
    
    return position_sizes


class RiskManager:
    """
    Gestionnaire de risque qui calcule les paramètres de risque selon la configuration.
    """
    
    def __init__(self, config: RiskConfig):
        """
        Initialise le gestionnaire de risque.
        
        Args:
            config: Configuration du risque
        """
        self.config = config
    
    def calculate_risk_params(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les paramètres de risque pour les données fournies.
        
        Args:
            data: DataFrame avec au moins une colonne 'close', idéalement 'high' et 'low' aussi
            
        Returns:
            DataFrame avec les colonnes ajoutées pour position_size, sl_level, tp_level
        """
        # Vérifier les colonnes minimales requises
        if 'close' not in data.columns:
            raise ValueError("Le DataFrame doit contenir au moins une colonne 'close'")
        
        # Conversion en arrays NumPy pour Numba
        close = data['close'].values.astype(np.float64)
        high = data['high'].values.astype(np.float64) if 'high' in data.columns else close
        low = data['low'].values.astype(np.float64) if 'low' in data.columns else close
        
        # Appel de la fonction appropriée selon le mode de risque
        position_sizes, sl_levels, tp_levels = self._calculate_by_mode(close, high, low)
        
        # Création d'un DataFrame résultat
        result = data.copy()
        result['position_size'] = position_sizes
        result['sl_level'] = sl_levels
        result['tp_level'] = tp_levels
        
        return result
    
    def _calculate_by_mode(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcule les paramètres de risque selon le mode configuré.
        
        Args:
            close: Array des prix de clôture
            high: Array des prix hauts
            low: Array des prix bas
            
        Returns:
            Tuple de (position_sizes, sl_levels, tp_levels)
        """
        mode = self.config.mode
        params = self.config.params
        
        if mode == RiskModeType.FIXED:
            return calculate_fixed_risk_params(
                close, 
                params.position_size, 
                params.stop_loss, 
                params.take_profit
            )
        elif mode == RiskModeType.ATR_BASED:
            return calculate_atr_risk_params(
                close, high, low,
                params.atr_period,
                params.atr_multiplier,
                params.risk_per_trade,
                params.tp_multiplier,
                params.max_position_size,
                params.min_position_size
            )
        elif mode == RiskModeType.VOLATILITY_BASED:
            return calculate_volatility_risk_params(
                close,
                params.vol_period,
                params.vol_multiplier,
                params.risk_per_trade,
                params.tp_multiplier,
                params.max_position_size,
                params.min_position_size
            )
        elif mode == RiskModeType.EQUITY_PERCENT:
            return calculate_equity_percent_risk_params(
                close,
                params.risk_percent,
                params.max_position_size,
                params.tp_multiplier,
                params.min_stop_distance
            )
        elif mode == RiskModeType.KELLEY:
            # Pour Kelly, on a besoin des statistiques historiques des trades
            # Ici on utilise des valeurs par défaut, mais idéalement on les calculerait
            # à partir des trades passés
            position_sizes = calculate_kelly_risk_params(
                close,
                win_rate=0.5,  # À calculer à partir des trades historiques
                win_loss_ratio=1.5,  # À calculer à partir des trades historiques
                fraction=params.fraction,
                max_position_size=params.max_position_size,
                min_position_size=params.min_position_size
            )
            
            # Pour les SL et TP, on utilise des valeurs fixes
            sl_levels = np.full_like(close, 0.02, dtype=np.float64)
            tp_levels = np.full_like(close, 0.04, dtype=np.float64)
            
            return position_sizes, sl_levels, tp_levels
        
        # Fallback sur le mode fixe
        return calculate_fixed_risk_params(close, 0.1, 0.02, 0.04)
    
    def update_config(self, new_config: RiskConfig):
        """
        Met à jour la configuration du gestionnaire.
        
        Args:
            new_config: Nouvelle configuration
        """
        self.config = new_config


def convert_dataframe_to_np_arrays(data: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Convertit un DataFrame pandas en arrays NumPy pour les calculs Numba.
    
    Args:
        data: DataFrame avec au moins les colonnes: close, [high, low]
        
    Returns:
        Dict: Dictionnaire d'arrays NumPy
    """
    result = {}
    
    # Colonnes essentielles
    if 'close' in data.columns:
        result['close'] = data['close'].values.astype(np.float64)
    else:
        raise ValueError("Le DataFrame doit contenir au moins une colonne 'close'")
    
    # Colonnes optionnelles
    for col in ['high', 'low', 'open', 'volume']:
        if col in data.columns:
            result[col] = data[col].values.astype(np.float64)
    
    return result