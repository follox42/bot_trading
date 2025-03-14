from enum import Enum
import numpy as np
import pandas as pd
import optuna
import json
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Union, Any
from numba import njit, prange, types, float64, int32, int64
from numba.typed import Dict, List
from tqdm import tqdm
import traceback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import cProfile
import pstats
import scipy
import logging
import psutil
import sqlite3
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import threading
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import math
import queue
import random 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from multiprocessing import Queue, Event
from multiprocessing.managers import DictProxy, SyncManager
from diagnostic_tools import create_diagnostic_tools, DiagnosticTools

# Au début du script
import gc
gc.enable()
gc.set_threshold(100, 5, 5)  # Rend le GC plus agressif

import psutil

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization_log.txt', mode='w'),
    ]
)

def set_process_priority(pid=None, priority=psutil.ABOVE_NORMAL_PRIORITY_CLASS):
    """Augmente la priorité du processus"""
    try:
        p = psutil.Process(pid or os.getpid())
        p.nice(priority)
    except Exception as e:
        print(f"Impossible de modifier la priorité : {e}")

# Utiliser dans prepare_optimization :
set_process_priority()  # Augmenter la priorité du processus principal

from model.Environnment_work import LoggerTradingEnv, TradingConfig, DataConfig
# =========================================================
# 1. Configuration des indicateurs et de la strategy
# =========================================================
'''
Block
   ---> Conditions
                ---> IndicatorType
                ---> Operator
                ---> IndicatorType or float
   ---> LogicOperator
'''
class IndicatorType(Enum):
    """Types d'indicateurs techniques disponibles"""
    EMA = "EMA"
    SMA = "SMA"
    RSI = "RSI"
    ATR = "ATR"
    MACD = "MACD"
    BOLL = "BOLL"
    STOCH = "STOCH"

class Operator(Enum):
    """Types d'opérateurs de comparaison disponibles"""
    GREATER = ">"
    LESS = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    CROSS_ABOVE = "CROSS_ABOVE"
    CROSS_BELOW = "CROSS_BELOW"

@dataclass
class Condition:
    """Condition de trading unique"""
    indicator1: IndicatorType  # Nom du premier indicateur (ex: "EMA_10")
    operator: Operator  # Opérateur de comparaison
    indicator2: Optional[IndicatorType] = None  # Nom du second indicateur (optionnel)
    value: Optional[float] = None  # Valeur fixe pour comparaison (optionnel)

class LogicOperator(Enum):
    """Types d'opérateurs logiques"""
    AND = "and"
    OR = "or"

@dataclass
class Block:
    """Bloc de conditions de trading avec validation et fonctionnalités étendues"""
    conditions: List[Condition]
    logic_operators: List[LogicOperator]
    
    def __post_init__(self):
        """Validation après initialisation"""
        # Validation du nombre d'opérateurs logiques
        if len(self.logic_operators) != len(self.conditions) and len(self.logic_operators) != len(self.conditions) - 1:
            raise ValueError(
                f"Nombre incorrect d'opérateurs logiques. Attendu {len(self.conditions)} ou {len(self.conditions) - 1}, "
                f"reçu {len(self.logic_operators)}"
            )
        
        # Vérification des conditions
        for condition in self.conditions:
            if condition.indicator2 is None and condition.value is None:
                raise ValueError("Une condition doit avoir soit un second indicateur soit une valeur")
            if condition.indicator2 is not None and condition.value is not None:
                raise ValueError("Une condition ne peut pas avoir à la fois un indicateur et une valeur")
    
    def add_condition(self, condition: Condition, logic_operator: LogicOperator = LogicOperator.AND):
        """Ajoute une condition au bloc avec un opérateur logique"""
        self.conditions.append(condition)
        if len(self.conditions) > 1:
            self.logic_operators.append(logic_operator)
    
    def remove_condition(self, index: int):
        """Supprime une condition et son opérateur logique associé"""
        if 0 <= index < len(self.conditions):
            self.conditions.pop(index)
            if len(self.logic_operators) > 0:
                # Supprime l'opérateur correspondant ou le précédent si c'est le dernier
                op_index = min(index, len(self.logic_operators) - 1)
                self.logic_operators.pop(op_index)
    
    def get_indicators(self) -> Set[str]:
        """Retourne l'ensemble des indicateurs utilisés dans le bloc"""
        indicators = set()
        for condition in self.conditions:
            indicators.add(condition.indicator1)
            if condition.indicator2 is not None:
                indicators.add(condition.indicator2)
        return indicators
    
    def to_dict(self) -> Dict:
        """Convertit le bloc en dictionnaire pour la sérialisation"""
        return {
            'conditions': [
                {
                    'indicator1': c.indicator1,
                    'operator': c.operator.value,
                    'indicator2': c.indicator2,
                    'value': c.value
                } for c in self.conditions
            ],
            'logic_operators': [op.value for op in self.logic_operators]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Block':
        """Crée un bloc à partir d'un dictionnaire"""
        conditions = [
            Condition(
                indicator1=c['indicator1'],
                operator=Operator(c['operator']),
                indicator2=c.get('indicator2'),
                value=c.get('value')
            ) for c in data['conditions']
        ]
        logic_operators = [LogicOperator(op) for op in data['logic_operators']]
        return cls(conditions=conditions, logic_operators=logic_operators)
    
    def validate_indicators(self, available_indicators: Set[str]) -> bool:
        """Vérifie si tous les indicateurs requis sont disponibles"""
        required_indicators = self.get_indicators()
        return all(ind in available_indicators for ind in required_indicators)
    
    def __str__(self) -> str:
        """Représentation string lisible du bloc"""
        if not self.conditions:
            return "Bloc vide"
        
        parts = []
        for i, condition in enumerate(self.conditions):
            # Formatage de la condition
            if condition.indicator2 is not None:
                cond_str = f"{condition.indicator1} {condition.operator.value} {condition.indicator2}"
            else:
                cond_str = f"{condition.indicator1} {condition.operator.value} {condition.value}"
            
            parts.append(cond_str)
            
            # Ajout de l'opérateur logique
            if i < len(self.logic_operators):
                parts.append(self.logic_operators[i].value)

        return " ".join(parts)
    
    def copy(self) -> 'Block':
        """Crée une copie profonde du bloc"""
        return Block(
            conditions=[
                Condition(
                    indicator1=c.indicator1,
                    operator=c.operator,
                    indicator2=c.indicator2,
                    value=c.value
                ) for c in self.conditions
            ],
            logic_operators=self.logic_operators.copy()
        )
    
@dataclass
class IndicatorConfig:
    type: IndicatorType
    min_period: int
    max_period: int
    step: int = 1
    price_type: str = "close"

    def __post_init__(self):
        # Ajuste max_period pour être divisible par step
        self.max_period = self.min_period + ((self.max_period - self.min_period) // self.step) * self.step

# =========================================================
# 2. Configuration des indicateurs et des recherche optuna
# =========================================================
@dataclass
class RiskConfig:
    """Configuration des paramètres de gestion du risque"""
    # Plages pour le position sizing en pourcentage du capital
    position_size_range: Tuple[float, float] = (0.01, 1.0)  # 1% à 100%
    position_step: float = 0.01
    
    # Plages pour les stop loss en pourcentage du prix d'entrée
    sl_range: Tuple[float, float] = (0.001, 0.1)  # 0.1% à 10%
    sl_step: float = 0.001
    
    # Take profit comme multiplicateur du stop loss
    tp_multiplier_range: Tuple[float, float] = (1.0, 10.0)  # 1x à 10x du SL
    tp_multiplier_step: float = 0.1
    
    # Paramètres ATR pour le mode dynamique
    atr_period_range: Tuple[int, int] = (5, 30)
    atr_multiplier_range: Tuple[float, float] = (0.5, 3.0)
    
    # Paramètres de volatilité pour le mode dynamique
    vol_period_range: Tuple[int, int] = (10, 50)
    vol_multiplier_range: Tuple[float, float] = (0.5, 3.0)

@dataclass
class StrategyStructureConfig:
    """Configuration de la structure de la stratégie"""
    # Limites pour les blocs de trading
    max_blocks: int = 3
    min_blocks: int = 1
    max_conditions_per_block: int = 3
    min_conditions_per_block: int = 1
    
    # Probabilités de génération
    cross_signals_probability: float = 0.3  # Probabilité d'utiliser des signaux de croisement
    value_comparison_probability: float = 0.4  # Probabilité de comparer avec une valeur fixe
    
    # Plages de valeurs pour les comparaisons
    rsi_value_range: Tuple[float, float] = (20.0, 80.0)
    price_value_range: Tuple[float, float] = (0.0, 1000.0)
    general_value_range: Tuple[float, float] = (-100.0, 100.0)

    indicators_list: Tuple = field(default_factory = tuple)

@dataclass
class OptimizationConfig:
    """Configuration des paramètres d'optimisation"""
    n_trials: int = 100
    n_jobs: int = -1  # -1 pour utiliser tous les cœurs disponibles
    timeout: Optional[int] = None  # Timeout en secondes
    gc_after_trial: bool = True
    
    # Paramètres du sampler Optuna
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    
    # Critères d'arrêt précoce
    early_stopping_n_trials: Optional[int] = None
    early_stopping_threshold: float = 0.0
    min_trades: int = 10

@dataclass
class SimulationConfig:
    """Configuration des paramètres de simulation"""
    initial_balance: float = 10000.0
    fee_open: float = 0.001  # 0.1% par trade
    fee_close: float = 0.001
    slippage: float = 0.001
    min_trade_size: float = 5.0
    max_trade_size: float = 100000.0
    tick_size: float = 0.01
    leverage_range: Tuple[int, int] = (1, 10)
    maintenance_margin: float = 0.01

@dataclass
class GeneralConfig:
    """Configuration générale regroupant tous les paramètres"""
    # Utilisation de default_factory au lieu de valeurs par défaut directes
    risk: RiskConfig = field(default_factory=RiskConfig)
    structure: StrategyStructureConfig = field(default_factory=StrategyStructureConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    
    # Pour les indicateurs, nous utilisons aussi default_factory
    indicators: Dict[str, IndicatorConfig] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.indicators:
            # Configuration par défaut des indicateurs
            self.indicators = {
                "EMA": IndicatorConfig(
                    type=IndicatorType.EMA,
                    min_period=3,
                    max_period=200,
                    step=2,
                    price_type="close"
                ),
                "SMA": IndicatorConfig(
                    type=IndicatorType.SMA,
                    min_period=5,
                    max_period=200,
                    step=5,
                    price_type="close"
                ),
                "RSI": IndicatorConfig(
                    type=IndicatorType.RSI,
                    min_period=2,
                    max_period=30,
                    step=1,
                    price_type="close"
                ),
                "ATR": IndicatorConfig(
                    type=IndicatorType.ATR,
                    min_period=5,
                    max_period=30,
                    step=1
                ),
                "MACD": IndicatorConfig(
                    type=IndicatorType.MACD,
                    min_period=12,
                    max_period=26,
                    step=2,
                    price_type="close"
                ),
                "BOLL": IndicatorConfig(
                    type=IndicatorType.BOLL,
                    min_period=10,
                    max_period=50,
                    step=5,
                    price_type="close"
                ),
                "STOCH": IndicatorConfig(
                    type=IndicatorType.STOCH,
                    min_period=5,
                    max_period=30,
                    step=1,
                    price_type="close"
                )
            }

        self.structure.indicators_list = tuple(self.indicators.keys())

    def validate(self) -> bool:
        """Valide la cohérence de la configuration"""
        try:
            # Validation des plages de valeurs
            assert self.risk.position_size_range[0] < self.risk.position_size_range[1]
            assert self.risk.sl_range[0] < self.risk.sl_range[1]
            assert self.risk.tp_multiplier_range[0] < self.risk.tp_multiplier_range[1]
            
            # Validation des périodes des indicateurs
            for ind_config in self.indicators.values():
                assert ind_config.min_period < ind_config.max_period
                assert ind_config.step > 0
            
            # Validation de la structure
            assert self.structure.min_blocks <= self.structure.max_blocks
            assert self.structure.min_conditions_per_block <= self.structure.max_conditions_per_block
            
            # Validation des probabilités
            assert 0 <= self.structure.cross_signals_probability <= 1
            assert 0 <= self.structure.value_comparison_probability <= 1
            
            # Validation des paramètres de simulation
            assert self.simulation.initial_balance > 0
            assert self.simulation.fee_open >= 0
            assert self.simulation.fee_close >= 0
            assert self.simulation.slippage >= 0
            assert self.simulation.min_trade_size > 0
            assert self.simulation.max_trade_size > self.simulation.min_trade_size
            
            return True
            
        except AssertionError as e:
            print(f"Erreur de validation: {e}")
            return False
    
    def get_optimization_params(self) -> dict:
        """Retourne les paramètres d'optimisation pour Optuna"""
        return {
            "n_trials": self.optimization.n_trials,
            "n_jobs": self.optimization.n_jobs,
            "timeout": self.optimization.timeout,
            "gc_after_trial": self.optimization.gc_after_trial,
            "sampler_params": {
                "n_startup_trials": self.optimization.n_startup_trials,
                "n_ei_candidates": self.optimization.n_ei_candidates
            }
        }

# =========================================================
# 3. Fonctions Numba optimisées pour les indicateurs
# =========================================================
# ===== Indicateurs de base =====
@njit(cache=True, fastmath=True)
def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calcul optimisé de l'EMA avec précision double.
    """
    alpha = 2.0 / (period + 1.0)
    ema = np.zeros_like(prices, dtype=np.float64)
    ema[0] = prices[0]
    
    for i in prange(1, len(prices)):
        ema[i] = prices[i] * alpha + ema[i-1] * (1 - alpha)
    
    return ema

@njit(cache=True, fastmath=True)
def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calcul optimisé de la SMA avec précision double.
    """
    sma = np.zeros_like(prices, dtype=np.float64)
    for i in prange(period - 1, len(prices)):
        sma[i] = np.mean(prices[i-period+1:i+1])
    return sma

@njit(cache=True, fastmath=True)
def calculate_rsi(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calcul optimisé du RSI avec précision double.
    """
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0).astype(np.float64)
    losses = np.where(deltas < 0, -deltas, 0.0).astype(np.float64)
    
    avg_gain = np.zeros_like(prices, dtype=np.float64)
    avg_loss = np.zeros_like(prices, dtype=np.float64)
    
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    
    for i in prange(period+1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i-1]) / period
    
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))

@njit(cache=True)
def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Calcul optimisé de l'ATR avec précision double.
    """
    tr = np.zeros_like(high, dtype=np.float64)
    atr = np.zeros_like(high, dtype=np.float64)
    
    for i in prange(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr[period] = np.mean(tr[1:period+1])
    for i in prange(period+1, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

# ===== Calculateur des indicateurs en parallel =====
@njit(cache=True)
def parallel_indicator_calculation(prices: np.ndarray, high: np.ndarray, low: np.ndarray, indicator_configs):
    """
    Calcul parallèle des indicateurs avec précision double.
    """
    max_indicators = 50
    indicators = np.zeros((max_indicators, len(prices)), dtype=np.float64)
    indicator_names = []
    
    current_idx = 0
    for i in prange(len(indicator_configs)):
        ind_type = indicator_configs[i][0]
        period = indicator_configs[i][1]
        
        if current_idx >= max_indicators:
            break
        
        if ind_type.item() == 'EMA':
            indicators[current_idx] = calculate_ema(prices.astype(np.float64), period)
            indicator_names.append(f"EMA_{period}")
            current_idx += 1
        
        elif ind_type.item() == 'SMA':
            indicators[current_idx] = calculate_sma(prices.astype(np.float64), period)
            indicator_names.append(f"SMA_{period}")
            current_idx += 1
        
        elif ind_type.item() == 'RSI':
            indicators[current_idx] = calculate_rsi(prices.astype(np.float64), period)
            indicator_names.append(f"RSI_{period}")
            current_idx += 1
        
        elif ind_type.item() == 'ATR':
            if len(high) > 0 and len(low) > 0:
                indicators[current_idx] = calculate_atr(
                    high.astype(np.float64),
                    low.astype(np.float64),
                    prices.astype(np.float64),
                    period
                )
                indicator_names.append(f"ATR_{period}")
                current_idx += 1
    
    return indicators[:current_idx], indicator_names

# ===== Creations des blocks, conditions, et signaux =====

@njit(cache=True)
def generate_signals_fast(
    indicators_array: np.ndarray,
    buy_blocks: list,       # List of arrays for each buy block ([ind1_idx, op_code, ind2_idx, value, logic_next])
    sell_blocks: list,      # List of arrays for each sell block
) -> np.ndarray:
    """
    Optimized version of signal generation using simplified block structure.
    
    Each block is an array of conditions, where each condition is represented by:
    [ind1_idx, op_code, ind2_idx, value, logic_next]
    
    Logical relationships between blocks are defined by separate arrays (0=AND, 1=OR).
    """
    data_length = indicators_array.shape[1]
    signals = np.zeros(data_length, dtype=np.int32)
    
    # Process buy blocks
    if len(buy_blocks) > 0:
        # Array to store results for each buy block
        buy_block_results = np.zeros((len(buy_blocks), data_length), dtype=np.bool_)
        
        # Process each buy block
        for block_idx, block in enumerate(buy_blocks):
            # Start with evaluating the first condition of the block
            block_result = np.ones(data_length, dtype=np.bool_)
            
            # Process each condition in the block
            for cond_idx in range(len(block)):
                ind1_idx = int(block[cond_idx, 0])
                op_code = int(block[cond_idx, 1])
                ind2_idx = int(block[cond_idx, 2])
                value = block[cond_idx, 3]
                
                # Get indicator values
                ind1 = indicators_array[ind1_idx]
                
                # Evaluate condition based on its type
                condition_result = np.zeros(data_length, dtype=np.bool_)
                
                if ind2_idx >= 0:
                    # Comparison between indicators
                    ind2 = indicators_array[ind2_idx]
                    if op_code == 0:  # >
                        condition_result = ind1 > ind2
                    elif op_code == 1:  # <
                        condition_result = ind1 < ind2
                    elif op_code == 2:  # >=
                        condition_result = ind1 >= ind2
                    elif op_code == 3:  # <=
                        condition_result = ind1 <= ind2
                    elif op_code == 4:  # ==
                        condition_result = np.abs(ind1 - ind2) < 1e-10
                    elif op_code == 5:  # CROSS_ABOVE
                        condition_result[1:] = (ind1[:-1] <= ind2[:-1]) & (ind1[1:] > ind2[1:])
                    elif op_code == 6:  # CROSS_BELOW
                        condition_result[1:] = (ind1[:-1] >= ind2[:-1]) & (ind1[1:] < ind2[1:])
                else:
                    # Comparison with fixed value
                    if op_code == 0:  # >
                        condition_result = ind1 > value
                    elif op_code == 1:  # <
                        condition_result = ind1 < value
                    elif op_code == 2:  # >=
                        condition_result = ind1 >= value
                    elif op_code == 3:  # <=
                        condition_result = ind1 <= value
                    elif op_code == 4:  # ==
                        condition_result = np.abs(ind1 - value) < 1e-10
                    elif op_code == 5:  # CROSS_ABOVE
                        condition_result[1:] = (ind1[:-1] <= value) & (ind1[1:] > value)
                    elif op_code == 6:  # CROSS_BELOW
                        condition_result[1:] = (ind1[:-1] >= value) & (ind1[1:] < value)
                
                # First condition or combine with previous result according to logic_next from previous condition
                if cond_idx == 0:
                    block_result = condition_result
                else:
                    # Get the logical operator from the previous condition's logic_next field
                    logic_operator = int(block[cond_idx-1, 4])
                    if logic_operator == 0:  # AND
                        block_result = block_result & condition_result
                    else:  # OR (logic_operator == 1)
                        block_result = block_result | condition_result
            
            # Store the block result
            buy_block_results[block_idx] = block_result
        
        # Combine all buy blocks (OR between blocks)
        buy_signal = np.zeros(data_length, dtype=np.bool_)
        for i in range(len(buy_blocks)):
            buy_signal = buy_signal | buy_block_results[i]
        
        # Apply buy signals
        signals[buy_signal] = 1
    
    # Process sell blocks (similar logic)
    if len(sell_blocks) > 0:
        # Array to store results for each sell block
        sell_block_results = np.zeros((len(sell_blocks), data_length), dtype=np.bool_)
        
        # Process each sell block
        for block_idx, block in enumerate(sell_blocks):
            # Start with evaluating the first condition of the block
            block_result = np.ones(data_length, dtype=np.bool_)
            
            # Process each condition in the block
            for cond_idx in range(len(block)):
                ind1_idx = int(block[cond_idx, 0])
                op_code = int(block[cond_idx, 1])
                ind2_idx = int(block[cond_idx, 2])
                value = block[cond_idx, 3]
                
                # Get indicator values
                ind1 = indicators_array[ind1_idx]
                
                # Evaluate condition based on its type
                condition_result = np.zeros(data_length, dtype=np.bool_)
                
                if ind2_idx >= 0:
                    # Comparison between indicators
                    ind2 = indicators_array[ind2_idx]
                    if op_code == 0:  # >
                        condition_result = ind1 > ind2
                    elif op_code == 1:  # <
                        condition_result = ind1 < ind2
                    elif op_code == 2:  # >=
                        condition_result = ind1 >= ind2
                    elif op_code == 3:  # <=
                        condition_result = ind1 <= ind2
                    elif op_code == 4:  # ==
                        condition_result = np.abs(ind1 - ind2) < 1e-10
                    elif op_code == 5:  # CROSS_ABOVE
                        condition_result[1:] = (ind1[:-1] <= ind2[:-1]) & (ind1[1:] > ind2[1:])
                    elif op_code == 6:  # CROSS_BELOW
                        condition_result[1:] = (ind1[:-1] >= ind2[:-1]) & (ind1[1:] < ind2[1:])
                else:
                    # Comparison with fixed value
                    if op_code == 0:  # >
                        condition_result = ind1 > value
                    elif op_code == 1:  # <
                        condition_result = ind1 < value
                    elif op_code == 2:  # >=
                        condition_result = ind1 >= value
                    elif op_code == 3:  # <=
                        condition_result = ind1 <= value
                    elif op_code == 4:  # ==
                        condition_result = np.abs(ind1 - value) < 1e-10
                    elif op_code == 5:  # CROSS_ABOVE
                        condition_result[1:] = (ind1[:-1] <= value) & (ind1[1:] > value)
                    elif op_code == 6:  # CROSS_BELOW
                        condition_result[1:] = (ind1[:-1] >= value) & (ind1[1:] < value)
                
                # First condition or combine with previous result according to logic_next from previous condition
                if cond_idx == 0:
                    block_result = condition_result
                else:
                    # Get the logical operator from the previous condition's logic_next field
                    logic_operator = int(block[cond_idx-1, 4])
                    if logic_operator == 0:  # AND
                        block_result = block_result & condition_result
                    else:  # OR (logic_operator == 1)
                        block_result = block_result | condition_result
            
            # Store the block result
            sell_block_results[block_idx] = block_result
        
        # Combine all sell blocks (OR between blocks)
        sell_signal = np.zeros(data_length, dtype=np.bool_)
        for i in range(len(sell_blocks)):
            sell_signal = sell_signal | sell_block_results[i]
        
        # Apply sell signals, but don't overwrite buy signals
        # Only apply sell signals where there are no buy signals
        sell_mask = sell_signal & ~(signals == 1)
        signals[sell_mask] = -1
    
    return signals

# =========================================================
# 4. Simulation réaliste
# =========================================================
from numba import njit
import numpy as np
from typing import Tuple, Union

# Fonctions utilitaires optimisées pour Numba
from typing import Tuple, Union
import numpy as np
from numba import njit

import numpy as np
from numba import njit
from typing import Tuple

import numpy as np
from numba import njit
from numba.types import float64, int64, boolean
from typing import Tuple
# =========================================================
# 5. Trials de optuna
# =========================================================
# ===== Les class de gestion optuna et la strategy avec le risk, les blocks et la strategy =====
class Strategy:
    def __init__(self):
        self.indicators_array = None
        self.current_idx = 0
        self.indicator_map = {}
        self.buy_blocks = []
        self.sell_blocks = []
        self.block_logic = []
        
        # Logger par défaut si strategy_logger n'est pas disponible
        self.logger = logging.getLogger('Strategy')
        if 'strategy_logger' in globals():
            self.logger = logging

    def prepare_indicators_config(self):
        """
        Convertit la configuration des indicateurs en tableau de configuration
        utilisable par les fonctions de calcul parallèle, en conservant l'ordre d'apparition.
        """
        required_indicators = []  # Liste ordonnée des indicateurs
        seen_indicators = set()   # Set pour éviter les doublons
        self.logger.debug("\n>> Analyse des indicateurs requis")

        # Parcours des blocs buy et sell pour récupérer les indicateurs dans l'ordre
        for block in self.buy_blocks + self.sell_blocks:
            for condition in block.conditions:
                for indicator in [condition.indicator1, condition.indicator2]:  # Garde l'ordre
                    if indicator and indicator not in seen_indicators:
                        try:
                            ind_type, period_str = indicator.split('_')
                            period = int(period_str)
                            required_indicators.append((ind_type, period))  # Ajout en respectant l'ordre
                        except ValueError:
                            self.logger.warning(f"  * Indicateur invalide ignoré: {indicator}")
                            continue

        # Si aucun indicateur trouvé, ajouter les indicateurs par défaut
        if not required_indicators:
            self.logger.info("  * Ajout des indicateurs par défaut")
            default_indicators = [('EMA', 10), ('SMA', 20), ('RSI', 14)]
            required_indicators.extend(default_indicators)
            seen_indicators.update(f"{ind}_{per}" for ind, per in default_indicators)

        # Préparation des configurations
        configs = []
        for ind_type, period in required_indicators:
            if ind_type in [e.value for e in IndicatorType]:
                configs.append((np.array(str(ind_type), dtype="U10"), period))
                self.logger.debug(f"  * Ajout configuration: {ind_type}_{period}")

        # Si aucun indicateur valide, fallback aux valeurs par défaut
        if not configs:
            self.logger.error("Aucune configuration valide d'indicateur!")
            configs = [
                (np.array("EMA", dtype="U10"), 10),
                (np.array("SMA", dtype="U10"), 20),
                (np.array("RSI", dtype="U10"), 14)
            ]
            self.logger.info("Utilisation des configurations par défaut")

        return configs

    def calculate_indicators(self, prices: np.ndarray, high: Optional[np.ndarray] = None,
                           low: Optional[np.ndarray] = None) -> None:
        """Calcul optimisé des indicateurs"""
        try:
            # Réinitialisation du dictionnaire de mapping
            self.indicator_map = {}
            
            # Préparer la configuration des indicateurs requis
            indicator_configs = self.prepare_indicators_config()

            # Conversion explicite en tableaux NumPy
            prices_array = np.ascontiguousarray(prices)
            high_array = np.ascontiguousarray(high) if high is not None else np.array([], dtype=prices.dtype)
            low_array = np.ascontiguousarray(low) if low is not None else np.array([], dtype=prices.dtype)

            # Calcul parallèle uniquement des indicateurs requis
            self.indicators_array, indicator_names = parallel_indicator_calculation(
                prices_array, high_array, low_array, indicator_configs
            )

            # Mise à jour du mapping
            for idx, name in enumerate(indicator_names):
                self.indicator_map[name] = idx

            self.logger.debug(f"Calcul effectué pour {len(indicator_names)} indicateurs")
            
        except Exception as e:
            self.logger.error(f"Erreur dans le calcul des indicateurs: {str(e)}")
            raise

    def cleanup(self):
        """Libère la mémoire"""
        if hasattr(self, 'indicators_array'):
            del self.indicators_array
        self.indicators_array = None
        self.indicator_map.clear()
        self.current_idx = 0
  
    def compile_conditions(self, conditions, logic_operators) -> Tuple[np.ndarray, np.ndarray]:
        """Compile une liste de conditions en array numpy"""
        n_conditions = len(conditions)
        conditions_array = np.zeros((n_conditions, 4))  # [idx1, op_code, idx2, value]
        logic_array = np.zeros(max(0, len(logic_operators)), dtype=np.int32)

        for i, condition in enumerate(conditions):
            conditions_array[i, 0] = self.indicator_map[condition.indicator1]
            conditions_array[i, 1] = self._operator_to_code(condition.operator)
            conditions_array[i, 2] = self.indicator_map.get(condition.indicator2, -1)
            conditions_array[i, 3] = condition.value if condition.value is not None else np.nan
            
            if i < len(logic_operators):
                logic_array[i] = 1 if logic_operators[i] == LogicOperator.OR else 0
                
        return conditions_array, logic_array
    
    def compile_blocks(self, blocks) -> Dict[str, np.ndarray]:
        """
        Compile les blocs en une structure simple et optimisée.
        Chaque bloc est représenté par un tableau distinct contenant ses conditions.
        """
        compiled_blocks = []
        
        # Traitement de chaque bloc
        for i, block in enumerate(blocks):
            # Création d'un tableau pour les conditions du bloc
            # Format: [ind1_idx, op_code, ind2_idx, value, logic_next]
            block_conditions = np.zeros((len(block.conditions), 5), dtype=np.float64)
            
            for j, condition in enumerate(block.conditions):
                # Indices des indicateurs
                ind1_idx = self.indicator_map[condition.indicator1]
                ind2_idx = self.indicator_map.get(condition.indicator2, -1)
                
                # Code de l'opérateur et valeur
                op_code = self._operator_to_code(condition.operator)
                value = condition.value if condition.value is not None else np.nan
                
                # Relation logique avec la condition suivante
                logic_next = -1  # Défaut: dernière condition
                if j < len(block.logic_operators):
                    logic_next = 1 if block.logic_operators[j] == LogicOperator.OR else 0
                
                # Stockage de la condition
                block_conditions[j] = [ind1_idx, op_code, ind2_idx, value, logic_next]
            
            # Ajout du bloc compilé
            compiled_blocks.append(block_conditions)

        return compiled_blocks
        
    def generate_signals(self, prices: np.ndarray, high: Optional[np.ndarray] = None,
                        low: Optional[np.ndarray] = None) -> np.ndarray:
        """Génère les signaux de trading de manière optimisée"""
        if self.indicators_array is None:
            self.calculate_indicators(prices, high, low)
        
        # Compilation des conditions
        buy_blocks = self.compile_blocks(self.buy_blocks)
        sell_blocks = self.compile_blocks(self.sell_blocks)

        # Génération des signaux
        return generate_signals_fast(
            self.indicators_array,
            buy_blocks,
            sell_blocks
        )
    
    def add_block(self, block: Block, is_buy: bool = True):
        """Ajoute un bloc et enregistre les indicateurs requis"""
        if is_buy:
            self.buy_blocks.append(block)
        else:
            self.sell_blocks.append(block)
             
    def _operator_to_code(self, operator: Operator) -> int:
        """Convertit un opérateur en code numérique"""
        operator_mapping = {
            Operator.GREATER: 0,
            Operator.LESS: 1,
            Operator.GREATER_EQUAL: 2,
            Operator.LESS_EQUAL: 3,
            Operator.EQUAL: 4,
            Operator.CROSS_ABOVE: 5,
            Operator.CROSS_BELOW: 6
        }
        return operator_mapping[operator]
    
    def _convert_block_for_numba(self, block: Block) -> Tuple[List[Tuple], List[int]]:
        """
        Convertit un bloc en format compatible Numba.
        
        Args:
            block: Bloc de conditions à convertir
            
        Returns:
            Tuple contenant:
              - Liste des conditions au format Numba
              - Liste des opérateurs logiques au format Numba
        """
        conditions = []
        for condition in block.conditions:
            # Conversion de l'opérateur en code numérique
            operator_code = self._operator_to_code(condition.operator)
            
            # Création du tuple de condition
            condition_tuple = (
                condition.indicator1,
                operator_code,
                condition.indicator2,
                condition.value
            )
            conditions.append(condition_tuple)
            
        # Conversion des opérateurs logiques
        logic_operators = [1 if op == LogicOperator.OR else 0 
                         for op in block.logic_operators]
            
        return conditions, logic_operators

class BlockManager:
    """Gestionnaire de blocs de trading optimisé pour GeneralConfig"""
    
    def __init__(self, trial: optuna.Trial, structure: StrategyStructureConfig, indicators: Dict[str, IndicatorConfig]):
        self.trial = trial
        self.structure = structure
        self.indicators = indicators
        
        # Blocs de trading
        self.buy_blocks: List[Block] = []
        self.sell_blocks: List[Block] = []
        
        # Statistiques
        self.total_conditions = 0
        self.used_indicators = set()
    
    def generate_blocks(self) -> Tuple[List[Block], List[Block]]:
        """Génère les blocs d'achat et de vente en parallèle"""
        # Lancement des générations en parallèle
        self.buy_blocks = self._generate_buy_blocks()
        self.sell_blocks = self._generate_sell_blocks()
        
        return self.buy_blocks, self.sell_blocks
    
    def _generate_buy_blocks(self) -> List[Block]:
        """Génère les blocs d'achat en utilisant la configuration"""
        n_blocks = self.trial.suggest_int('n_buy_blocks', 
                                        self.structure.min_blocks,
                                        self.structure.max_blocks)
        blocks = []
        
        for b in range(n_blocks):
            n_conditions = self.trial.suggest_int(
                f'buy_block_{b}_conditions',
                self.structure.min_conditions_per_block,
                self.structure.max_conditions_per_block
            )
            
            conditions = []
            logic_operators = []
            
            for c in range(n_conditions):
                # Création de la condition avec la configuration
                condition = self._create_condition('buy', b, c)
                conditions.append(condition)
                
                # Ajout de l'opérateur logique si nécessaire
                if c < n_conditions and not (c == n_conditions-1 and b == n_blocks-1):
                    logic_op = LogicOperator(
                        self.trial.suggest_categorical(
                            f'buy_b{b}_c{c}_logic',
                            ['and', 'or']
                        )
                    )
                    logic_operators.append(logic_op)

            block = Block(conditions=conditions, logic_operators=logic_operators)
            blocks.append(block)
            
        return blocks
    
    def _generate_sell_blocks(self) -> List[Block]:
        """
        Génère les blocs de vente en utilisant la configuration.
        Cette méthode est similaire à _generate_buy_blocks mais pour les signaux de vente.
        """
        n_blocks = self.trial.suggest_int('n_sell_blocks', 
                                        self.structure.min_blocks,
                                        self.structure.max_blocks)
        blocks = []
        
        for b in range(n_blocks):
            # Suggère le nombre de conditions pour ce bloc de vente
            n_conditions = self.trial.suggest_int(
                f'sell_block_{b}_conditions',
                self.structure.min_conditions_per_block,
                self.structure.max_conditions_per_block
            )
            
            conditions = []
            logic_operators = []
            
            # Génère chaque condition du bloc
            for c in range(n_conditions):
                # Création de la condition avec la configuration
                condition = self._create_condition('sell', b, c)
                conditions.append(condition)
                
                # Ajout de l'opérateur logique entre les conditions si ce n'est pas la dernière
                if c < n_conditions and not (c == n_conditions-1 and b == n_blocks-1):
                    logic_op = LogicOperator(
                        self.trial.suggest_categorical(
                            f'sell_b{b}_c{c}_logic',
                            ['and', 'or']
                        )
                    )
                    logic_operators.append(logic_op)

            # Création du bloc avec ses conditions et opérateurs
            block = Block(conditions=conditions, logic_operators=logic_operators)
            blocks.append(block)

        return blocks

    def _create_condition(self, prefix: str, block_idx: int, cond_idx: int) -> Condition:
        """Crée une condition de trading en utilisant la configuration"""

        # Sélection du type d'indicateur
        ind1_type = self.trial.suggest_categorical(
            f'{prefix}_b{block_idx}_c{cond_idx}_ind1_type',
            self.structure.indicators_list
        )
        
        # Configuration de l'indicateur
        ind_config = self.indicators[ind1_type]
        ind1_period = self.trial.suggest_int(
            f'{prefix}_b{block_idx}_c{cond_idx}_ind1_period',
            ind_config.min_period,
            ind_config.max_period,
            ind_config.step
        )
        
        # Sélection du type de comparaison
        use_value = random.randint(0,100) < self.structure.value_comparison_probability * 100
        
        crossover_operators = ['CROSS_ABOVE', 'CROSS_BELOW']
        comparison_operators = ['>', '<', '>=', '<=', '==']
        
        # Calculate weights
        crossover_weight = int(self.structure.cross_signals_probability * 100)
        comparison_weight = int(self.structure.value_comparison_probability * 100)
        
        # Generate weighted list
        operator_list = (
            crossover_operators * crossover_weight +
            comparison_operators * comparison_weight
        )

        operator = Operator(self.trial.suggest_categorical(
            f'{prefix}_b{block_idx}_c{cond_idx}_operator',
            operator_list
        ))

        
        # Création de la condition finale
        if use_value:
            value = self._suggest_value_for_indicator(f'{prefix}_b{block_idx}_c{cond_idx}_value',ind1_type)
            return Condition(
                indicator1=f"{ind1_type}_{ind1_period}",
                operator=operator,
                value=value
            )
        else:
            ind2_type = self.trial.suggest_categorical(
                f'{prefix}_b{block_idx}_c{cond_idx}_ind2_type',
                self.structure.indicators_list
            )
            ind2_config = self.indicators[ind2_type]
            ind2_period = self.trial.suggest_int(
                f'{prefix}_b{block_idx}_c{cond_idx}_ind2_period',
                ind2_config.min_period,
                ind2_config.max_period,
                ind2_config.step
            )
            return Condition(
                indicator1=f"{ind1_type}_{ind1_period}",
                operator=operator,
                indicator2=f"{ind2_type}_{ind2_period}"
            )
    
    def _suggest_value_for_indicator(self,name, indicator_type: str) -> float:
        """Suggère une valeur appropriée pour le type d'indicateur"""
        if indicator_type == 'RSI':
            return self.trial.suggest_float(name, 
                                          self.structure.rsi_value_range[0],
                                          self.structure.rsi_value_range[1])
        elif indicator_type in ['EMA', 'SMA']:
            return self.trial.suggest_float(name,
                                          self.structure.price_value_range[0],
                                          self.structure.price_value_range[1])
        else:
            return self.trial.suggest_float(name,
                                          self.structure.general_value_range[0],
                                          self.structure.general_value_range[1])

    def validate_blocks(self, available_indicators: Set[str]) -> bool:
        """
        Validates the generated trading blocks against strategy configuration constraints.
        
        Args:
            available_indicators (Set[str]): Set of calculated indicator names
        
        Returns:
            bool: True if blocks are valid, False otherwise
        """
        try:
            # Validate buy blocks
            if not self._validate_block_group(
                self.buy_blocks, 
                'buy', 
                available_indicators
            ):
                print("❌ Buy blocks validation failed")
                return False
            
            # Validate sell blocks
            if not self._validate_block_group(
                self.sell_blocks, 
                'sell', 
                available_indicators
            ):
                print("❌ Sell blocks validation failed")
                return False
            
            # Additional global validation checks
            total_buy_conditions = sum(len(block.conditions) for block in self.buy_blocks)
            total_sell_conditions = sum(len(block.conditions) for block in self.sell_blocks)
            
            # Check total number of blocks
            if not (self.structure.min_blocks <= len(self.buy_blocks) <= self.structure.max_blocks):
                print(f"❌ Number of buy blocks ({len(self.buy_blocks)}) outside allowed range")
                return False
            
            if not (self.structure.min_blocks <= len(self.sell_blocks) <= self.structure.max_blocks):
                print(f"❌ Number of sell blocks ({len(self.sell_blocks)}) outside allowed range")
                return False
            
            return True
        
        except Exception as e:
            print(f"🔥 Unexpected error during block validation: {e}")
            return False

    def _validate_block_group(
        self, 
        blocks: List[Block], 
        block_type: str, 
        available_indicators: Set[str]
    ) -> bool:
        """
        Validates a group of blocks (buy or sell) against configuration constraints.
        
        Args:
            blocks (List[Block]): Blocks to validate
            block_type (str): 'buy' or 'sell'
            available_indicators (Set[str]): Set of calculated indicator names
        
        Returns:
            bool: True if blocks are valid, False otherwise
        """
        for block_idx, block in enumerate(blocks):
            # Validate number of conditions per block
            if not (self.structure.min_conditions_per_block 
                    <= len(block.conditions) 
                    <= self.structure.max_conditions_per_block):
                print(f"❌ {block_type.capitalize()} Block {block_idx} has invalid number of conditions")
                return False
            
            # Validate block's indicators
            block_indicators = set()
            for condition_idx, condition in enumerate(block.conditions):
                # Check indicator1 exists
                if condition.indicator1.split("_")[0] not in available_indicators:
                    print(f"❌ {block_type.capitalize()} Block {block_idx}, Condition {condition_idx}: 11 "
                        f"Indicator {condition.indicator1.split("_")[0]} not available")
                    return False
                block_indicators.add(condition.indicator1)
                
                # Check indicator2 exists if comparison is between indicators
                if condition.indicator2 and condition.indicator2.split("_")[0] not in available_indicators:
                    print(f"❌ {block_type.capitalize()} Block {block_idx}, Condition {condition_idx}: "
                        f"Indicator {condition.indicator2.split("_")[0]} not available")
                    return False
                if condition.indicator2:
                    block_indicators.add(condition.indicator2)
            
            # Validate logical structure
            if len(block.conditions) > 1 and len(block.logic_operators) != len(block.conditions) - 1:
                print(f"❌ {block_type.capitalize()} Block {block_idx}: "
                    "Incorrect number of logical operators")
                return False
        
        return True

class RiskManager:
    """Gestionnaire de risque avec optimisation via Optuna"""
    
    def __init__(self, trial: optuna.Trial, config: RiskConfig):
        """
        Initialise le gestionnaire de risque avec un trial Optuna.
        
        Args:
            trial: Trial Optuna pour l'optimisation
            config: Configuration des paramètres de risque
        """
        self.trial = trial
        self.config = config
        
        # Suggestions des paramètres de base via Optuna
        self.risk_mode = self.trial.suggest_categorical(
            'risk_mode', 
            ['fixed', 'dynamic_atr', 'dynamic_vol']
        )
        
        # Position size de base en pourcentage
        self.base_position = self.trial.suggest_float(
            'base_position',
            self.config.position_size_range[0],
            self.config.position_size_range[1],
            step=self.config.position_step
        )
        
        # Stop loss de base en pourcentage
        self.base_sl = self.trial.suggest_float(
            'base_sl',
            self.config.sl_range[0],
            self.config.sl_range[1],
            step=self.config.sl_step
        )
        
        # Multiplicateur de take profit
        self.tp_mult = self.trial.suggest_float(
            'tp_mult',
            self.config.tp_multiplier_range[0],
            self.config.tp_multiplier_range[1],
            step=self.config.tp_multiplier_step
        )
        
        # Paramètres spécifiques au mode
        if self.risk_mode == 'dynamic_atr':
            self.atr_period = self.trial.suggest_int(
                'atr_period',
                self.config.atr_period_range[0],
                self.config.atr_period_range[1]
            )
            self.atr_multiplier = self.trial.suggest_float(
                'atr_multiplier',
                self.config.atr_multiplier_range[0],
                self.config.atr_multiplier_range[1]
            )
            
        elif self.risk_mode == 'dynamic_vol':
            self.vol_period = self.trial.suggest_int(
                'vol_period',
                self.config.vol_period_range[0],
                self.config.vol_period_range[1]
            )
            self.vol_multiplier = self.trial.suggest_float(
                'vol_multiplier',
                self.config.vol_multiplier_range[0],
                self.config.vol_multiplier_range[1]
            )
    
    def calculate_parameters(
        self,
        prices: np.ndarray,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcule les paramètres de trading selon le mode choisi.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (position_sizes en %, sl_levels en %, tp_levels en %)
        """
        if self.risk_mode == 'fixed':
            return self._fixed_parameters(prices)
        elif self.risk_mode == 'dynamic_atr':
            if high is None or low is None:
                raise ValueError("High et Low requis pour le mode ATR")
            return self._atr_parameters(prices, high, low)
        else:  # dynamic_vol
            return self._volatility_parameters(prices)
    
    def _fixed_parameters(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Paramètres fixes"""
        n = len(prices)
        position_sizes = np.full(n, self.base_position)
        sl_levels = np.full(n, self.base_sl)
        tp_levels = np.full(n, self.base_sl * self.tp_mult)
        return position_sizes, sl_levels, tp_levels
    
    def _atr_parameters(
        self,
        prices: np.ndarray,
        high: np.ndarray,
        low: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Paramètres basés sur l'ATR"""
        n = len(prices)
        atr = calculate_atr(high, low, prices, self.atr_period)
        
        # Normalisation ATR en pourcentage
        atr_pct = atr / prices
        
        # Stop loss basé sur l'ATR
        sl_levels = np.clip(
            atr_pct * self.atr_multiplier,
            self.config.sl_range[0],
            self.config.sl_range[1]
        )
        
        # Take profit comme multiple du stop loss
        tp_levels = sl_levels * self.tp_mult
        
        # Position size inversement proportionnel au risque
        position_sizes = np.clip(
            self.base_position / (atr_pct * self.atr_multiplier),
            self.config.position_size_range[0],
            self.config.position_size_range[1]
        )
        
        return position_sizes, sl_levels, tp_levels
    
    def _volatility_parameters(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Paramètres basés sur la volatilité"""
        n = len(prices)
        
        # Calcul de la volatilité en pourcentage
        returns = np.diff(prices) / prices[:-1]
        volatility = np.zeros(n)
        vol_period = self.vol_period
        
        # Calcul de la volatilité glissante
        for i in range(vol_period, n):
            volatility[i] = np.std(returns[i-vol_period:i])
        
        # Remplir les premières valeurs
        volatility[:vol_period] = volatility[vol_period]
        
        # Stop loss basé sur la volatilité
        sl_levels = np.clip(
            volatility * self.vol_multiplier,
            self.config.sl_range[0],
            self.config.sl_range[1]
        )
        
        # Take profit comme multiple du stop loss
        tp_levels = sl_levels * self.tp_mult
        
        # Position size inversement proportionnel à la volatilité
        position_sizes = np.clip(
            self.base_position / (volatility * self.vol_multiplier),
            self.config.position_size_range[0],
            self.config.position_size_range[1]
        )
        
        return position_sizes, sl_levels, tp_levels
    
    def get_parameters(self) -> Dict:
        """Retourne les paramètres actuels"""
        params = {
            'risk_mode': self.risk_mode,
            'base_position': self.base_position,
            'base_sl': self.base_sl,
            'tp_mult': self.tp_mult
        }
        
        if self.risk_mode == 'dynamic_atr':
            params.update({
                'atr_period': self.atr_period,
                'atr_multiplier': self.atr_multiplier
            })
        elif self.risk_mode == 'dynamic_vol':
            params.update({
                'vol_period': self.vol_period,
                'vol_multiplier': self.vol_multiplier
            })
            
        return params

# ===== Un logger optionelle =====
def log_simulation_results(strategy_logger, results: Tuple) -> None:
    """Log les résultats de la simulation à partir du tuple de résultats"""
    roi, win_rate, total_trades, max_drawdown, avg_profit_per_trade, profit_factor, liquidation_rate = results
    
    strategy_logger.logger.info("\n>> Résultats de simulation:")
    strategy_logger.logger.info(f"  * ROI: {roi*100:.2f}%")
    strategy_logger.logger.info(f"  * Win Rate: {win_rate*100:.2f}%")
    strategy_logger.logger.info(f"  * Total Trades: {total_trades}")
    strategy_logger.logger.info(f"  * Max Drawdown: {max_drawdown*100:.2f}%")
    strategy_logger.logger.info(f"  * Profit Moyen par Trade: {avg_profit_per_trade*100:.2f}%")
    strategy_logger.logger.info(f"  * Profit Factor: {profit_factor:.2f}")

# ===== Calcule du score finale multiparametre =====
def calculate_score(results: Tuple) -> float:
    """
    Calcule un score optimisé avec normalisation améliorée.
    """
    roi, win_rate, total_trades, max_drawdown, avg_profit_per_trade, liquidation_rate, max_profit, max_loss, profit_factor = results
    
    # Validation initiale
    if total_trades == 0:
        return float('-inf')
    
    # Normalisation du nombre de trades sur une échelle logarithmique
    # Permet une progression plus douce entre 300 et 10000 trades
    trade_score = np.log1p(total_trades) / np.log1p(10000)
    if trade_score < np.log1p(300) / np.log1p(10000):
        return float('-inf')
    
    # ROI annualisé avec normalisation logarithmique
    # Transforme un ROI de 20% en score ~0.5, 50% en ~0.7, 100% en ~0.85
    roi_score = np.log1p(max(0, roi * 100)) / np.log1p(100)
    
    # Win rate normalisé avec une courbe sigmoïde
    # Centre la distribution autour de 55% avec transition douce
    win_score = 1 / (1 + np.exp(-0.1 * (win_rate * 100 - 55)))
    
    # Drawdown inversé et normalisé
    # Un drawdown de 10% donne 0.9, 20% donne 0.8, etc.
    dd_score = max(0, 1 - abs(max_drawdown))
    
    # Profit factor normalisé logarithmiquement
    # Transforme PF de 1.5 en ~0.4, 2.0 en ~0.6, 3.0 en ~0.8
    pf_score = np.log1p(max(0, profit_factor - 1)) / np.log1p(2)
    
    # Profit moyen par trade normalisé
    # Centre autour de 0.5% de profit moyen
    avg_profit_score = np.tanh(avg_profit_per_trade * 100)
    
    # Pondérations des composantes
    weights = {
        'roi': 2.5,        # ROI est crucial
        'win_rate': 0.5,   # Important mais pas décisif
        'drawdown': 2.0,   # Protection du capital
        'profit_factor': 2.0,  # Consistance
        'trade_freq': 1.0,  # Volume suffisant
        'avg_profit': 1.0   # Qualité des trades
    }
    
    # Score composé avec pondérations
    base_score = (
        roi_score * weights['roi'] +
        win_score * weights['win_rate'] +
        dd_score * weights['drawdown'] +
        pf_score * weights['profit_factor'] +
        trade_score * weights['trade_freq'] +
        avg_profit_score * weights['avg_profit']
    ) / sum(weights.values())  # Normalisation par la somme des poids
    
    # Bonus multiplicatifs progressifs
    bonus = 1.0
    
    # Bonus pour ROI exceptionnel (>50% par an)
    if roi > 0.5:
        bonus += 0.2 * np.tanh(roi - 0.5)
    
    # Bonus pour drawdown contrôlé (<15%)
    if abs(max_drawdown) < 0.15:
        bonus += 0.1 * (1 - abs(max_drawdown) / 0.15)
    
    # Bonus pour profit factor élevé (>2.5)
    if profit_factor > 2.5:
        bonus += 0.15 * np.tanh(profit_factor - 2.5)
    
    final_score = base_score * bonus
    
    # Les scores devraient maintenant être dans une plage plus intuitive :
    # 0.2-0.4 : Stratégie basique mais viable
    # 0.4-0.6 : Bonne stratégie
    # 0.6-0.8 : Excellente stratégie
    # >0.8 : Stratégie exceptionnelle
    return final_score

# ===== La partit principale qui organise tout et gere chaque trial =====
def objective(trial: optuna.Trial, prices: np.ndarray, high: np.ndarray, low: np.ndarray, config: GeneralConfig, diagnostics: DiagnosticTools) -> float:
    """
    Fonction objective optimisée utilisant BlockManager et adaptée pour GeneralConfig.
    Cette fonction évalue chaque essai de stratégie en générant des blocs de trading
    et en simulant leurs performances.
    """
    try:
        # Initialisation du BlockManager avec le trial et la configuration
        block_manager = BlockManager(trial, config.structure, config.indicators)
        
        # Génération des blocs d'achat et de vente de manière parallèle
        buy_blocks, sell_blocks = block_manager.generate_blocks()
        
        # Création de la stratégie avec la configuration
        strategy = Strategy()
        
        # Ajout des blocs à la stratégie
        for block in buy_blocks:
            strategy.add_block(block, is_buy=True)
        for block in sell_blocks:
            strategy.add_block(block, is_buy=False)
        
        '''print("\n\n")
        for block in strategy.buy_blocks:
            print(block) 
        print("\n") 
        for block in strategy.sell_blocks:
            print(block)'''

        # Génération des signaux
        signals = strategy.generate_signals(prices, high, low)
        
        # Préparation des paramètres de risk management
        risk_manager = RiskManager(trial, config.risk)
        position_sizes, sl_levels, tp_levels = risk_manager.calculate_parameters(
            prices=prices,
            high=high,
            low=low
        )
        
        leverage = trial.suggest_int(
                'leverage',
                config.simulation.leverage_range[0],
                config.simulation.leverage_range[1]
            )
        # Simulation avec les paramètres de la configuration générale
        simulation_results, metrics = simulate_realistic_trading(
            prices=prices,
            signals=signals,
            position_size=position_sizes,
            sl_pct=sl_levels,
            tp_pct=tp_levels,
            leverage=np.full_like(prices, leverage, dtype=np.float32),
            initial_balance=config.simulation.initial_balance,
            slippage=config.simulation.slippage,
            fee_open=config.simulation.fee_open,
            fee_close=config.simulation.fee_close,
            margin_mode=trial.suggest_categorical('margin_mode', [0, 1]),
            trading_mode=trial.suggest_categorical('trading_mode', [0, 1]),
            tick_size=config.simulation.tick_size,
            safety_buffer=0.01,  # Buffer de sécurité pour éviter les liquidations
        )

        # Extraction et stockage des résultats
        roi, win_rate, total_trades, max_drawdown, avg_profit_per_trade, liquidation_rate, max_profit, max_loss, profit_factor = simulation_results
        
        account_history, long_history, short_history = metrics

        # Ajoutez des informations supplémentaires pour le monitoring
        buy_signals = np.sum(signals == 1)
        sell_signals = np.sum(signals == -1)
        neutral_signals = np.sum(signals == 0)

        # Calculez quelques statistiques pour enrichir les diagnostics
        signal_density = (buy_signals + sell_signals) / len(signals) * 100
        trade_frequency = total_trades / len(signals) * 100

        # Créez un dictionnaire enrichi de métriques
        enhanced_metrics = {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'neutral_signals': neutral_signals,
            'signal_density': signal_density,
            'trade_frequency': trade_frequency,
            'data_length': len(signals),
            **{k: v for k, v in zip(['roi', 'win_rate', 'total_trades', 'max_drawdown', 'avg_profit', 'liquidation_rate', 'max_profit', 'max_loss', 'profit_factor'], simulation_results)},
        }

        # Monitoring avec diagnostics verbeux
        diagnostics.monitor_simulation_results(
            test_id=trial.number,
            prices=prices,
            signals=signals,
            account_history=account_history,
            long_history=long_history,
            short_history=short_history,
            metrics=enhanced_metrics,  # Utilisez les métriques enrichies
            force_log=True # Force le logging tous les 10 essais pour plus de contexte
        )

        buy_signals = np.sum(signals == 1)
        sell_signals = np.sum(signals == -1)
        neutral_signals = np.sum(signals == 0)

        # Stockage des métriques dans le trial
        trial.set_user_attr('roi', float(roi))
        trial.set_user_attr('win_rate', float(win_rate))
        trial.set_user_attr('total_trades', float(total_trades))
        trial.set_user_attr('max_drawdown', float(max_drawdown))
        trial.set_user_attr('profit_factor', float(profit_factor))
        trial.set_user_attr('avg_profit', float(avg_profit_per_trade))
        trial.set_user_attr('buy_signals', int(buy_signals))
        trial.set_user_attr('sell_signals', int(sell_signals))
        trial.set_user_attr('neutral_signals', int(neutral_signals))
        

        score = calculate_score(simulation_results)
        return score if score is not None else float('-inf')
        
    except Exception as e:
        print(f"Erreur dans la fonction objective: {e}")
        traceback.print_exc()
        return float('-inf')


# =========================================================
# 5. Setup de optuna multiprocessor
# =========================================================
def cleanup_shared_data(shared_data: Dict) -> None:
    """
    Nettoie les ressources de mémoire partagée de manière sécurisée.
    
    Cette fonction s'assure que toutes les ressources de mémoire partagée sont
    correctement libérées, même en cas d'erreur pendant le nettoyage.
    
    Args:
        shared_data: Dictionnaire contenant les informations de mémoire partagée
                    avec les clés 'prices', 'high', 'low'
    """
    # Logger pour tracer les opérations de nettoyage
    logger = logging.getLogger('cleanup')
    
    if not shared_data:
        logger.warning("Aucune donnée partagée à nettoyer")
        return
        
    # Liste des clés attendues dans shared_data
    expected_keys = ['prices', 'high', 'low']
    cleanup_errors = []
    
    for key in expected_keys:
        if key not in shared_data:
            logger.warning(f"Clé manquante dans shared_data: {key}")
            continue
            
        try:
            # Récupération des informations de la mémoire partagée
            shm_info = shared_data[key]
            shm_name = shm_info.get('name')
            
            if not shm_name:
                logger.warning(f"Nom de mémoire partagée manquant pour {key}")
                continue
                
            try:
                # Tentative de récupération de la mémoire partagée
                shm = SharedMemory(name=shm_name)
                
                try:
                    # Fermeture et suppression de la mémoire partagée
                    shm.close()
                    shm.unlink()
                    logger.debug(f"Mémoire partagée nettoyée avec succès: {shm_name}")
                    
                except Exception as e:
                    cleanup_errors.append(f"Erreur lors du nettoyage de {shm_name}: {str(e)}")
                    
            except FileNotFoundError:
                logger.info(f"Mémoire partagée déjà supprimée: {shm_name}")
                
            except Exception as e:
                cleanup_errors.append(f"Erreur lors de l'accès à {shm_name}: {str(e)}")
                
        except Exception as e:
            cleanup_errors.append(f"Erreur inattendue pour {key}: {str(e)}")
    
    # Suppression des références dans le dictionnaire
    shared_data.clear()
    
    # Si des erreurs se sont produites, les logger et potentiellement les remonter
    if cleanup_errors:
        error_msg = "\n".join(cleanup_errors)
        logger.error(f"Erreurs lors du nettoyage:\n{error_msg}")
        
        # En mode debug, on peut vouloir lever une exception
        if logger.getEffectiveLevel() <= logging.DEBUG:
            raise RuntimeError(f"Erreurs lors du nettoyage de la mémoire partagée:\n{error_msg}")
            
    logger.info("Nettoyage des données partagées terminé")        

class DiverseTPESampler(optuna.samplers.TPESampler):
    """Sampler personnalisé qui maintient une diversité élevée dans la recherche"""
    
    def __init__(self, diversity_weight=0.5, population_size=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diversity_weight = diversity_weight
        self.population_size = population_size
        self.population = []
        self.min_distance_threshold = 0.05  # Distance minimum entre solutions
    
    def sample_relative(self, study, trial, search_space):
        # Mise à jour de la population avec les meilleurs trials
        self._update_population(study)
        
        # Stratégies de diversification
        if len(self.population) >= self.population_size:
            if random.random() < 0.4:  # 40% chance de diversification
                # Soit on fait un croisement
                if random.random() < 0.5:
                    return self._crossover()
                # Soit on fait une mutation
                else:
                    return self._mutate_random()
        
        # Sinon on utilise TPE avec une composante de diversité
        params = super().sample_relative(study, trial, search_space)
        
        # On vérifie si les paramètres sont assez différents des existants
        if self._is_too_similar(params):
            # Si trop similaire, on ajoute de la diversité
            params = self._add_diversity(params)
        
        return params
    
    def _update_population(self, study):
        """Met à jour la population avec les meilleurs trials en maintenant la diversité"""
        # Trie les trials par score
        sorted_trials = sorted(
            [t for t in study.trials if t.value is not None],
            key=lambda t: t.value if t.value is not None else float('-inf'),
            reverse=True
        )
        
        # Réinitialise la population
        self.population = []
        
        # Ajoute les trials en maintenant la diversité
        for trial in sorted_trials:
            if len(self.population) >= self.population_size:
                break
            
            if not any(self._calculate_distance(trial.params, p.params) < self.min_distance_threshold 
                      for p in self.population):
                self.population.append(trial)
    
    def _crossover(self):
        """Croise deux solutions de manière intelligente"""
        if len(self.population) < 2:
            return {}
        
        # Sélection des parents avec un biais vers les meilleurs
        weights = [math.exp(i.value if i.value is not None else float('-inf')) 
                  for i in self.population]
        parents = random.choices(self.population, weights=weights, k=2)
        
        child = {}
        # Croisement intelligent par paramètre
        for param in parents[0].params:
            if random.random() < 0.5:
                child[param] = parents[0].params[param]
            else:
                child[param] = parents[1].params[param]
            
            # Ajout d'une petite mutation
            if isinstance(child[param], (int, float)):
                mutation = random.gauss(0, abs(child[param]) * 0.1)
                child[param] += mutation
        
        return child
    
    def _mutate_random(self):
        """Mutation d'une solution existante"""
        if not self.population:
            return {}
        
        # Sélection d'une solution à muter
        base_solution = random.choice(self.population)
        mutated = base_solution.params.copy()
        
        # Mutation plus agressive
        for param in mutated:
            if random.random() < 0.5:  # 50% chance de mutation
                if isinstance(mutated[param], (int, float)):
                    # Mutation plus large
                    mutation_scale = abs(mutated[param]) * random.uniform(0.3, 1.0)
                    mutated[param] += random.gauss(0, mutation_scale)
                elif isinstance(mutated[param], str):
                    # Pour les paramètres catégoriels, changement aléatoire
                    possible_values = self._get_categorical_choices(param)
                    if possible_values:
                        mutated[param] = random.choice(possible_values)
        
        return mutated
    
    def _calculate_distance(self, params1, params2):
        """Calcule la distance normalisée entre deux ensembles de paramètres"""
        if not params1 or not params2:
            return float('inf')
            
        distance = 0
        common_params = set(params1.keys()) & set(params2.keys())
        
        if not common_params:
            return float('inf')
        
        for param in common_params:
            val1 = params1[param]
            val2 = params2[param]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Normalisation pour les valeurs numériques
                max_val = max(abs(val1), abs(val2))
                if max_val != 0:
                    distance += abs(val1 - val2) / max_val
            else:
                # Pour les valeurs non numériques
                distance += 1 if val1 != val2 else 0
        
        return distance / len(common_params)
    
    def _is_too_similar(self, params):
        """Vérifie si les paramètres sont trop similaires à ceux existants"""
        return any(self._calculate_distance(params, p.params) < self.min_distance_threshold 
                  for p in self.population)
    
    def _add_diversity(self, params):
        """Ajoute de la diversité aux paramètres"""
        diversified = params.copy()
        
        for param, value in diversified.items():
            if isinstance(value, (int, float)):
                # Ajout d'une perturbation significative
                noise = random.gauss(0, abs(value) * self.diversity_weight)
                diversified[param] = value + noise
        
        return diversified
    
    def _get_categorical_choices(self, param_name):
        """Récupère les choix possibles pour un paramètre catégoriel"""
        # À implémenter selon vos besoins spécifiques
        categorical_choices = {
            'margin_mode': [0, 1],
            'risk_management': ['fixed', 'atr_based', 'volatility_based']
            # Ajoutez d'autres paramètres catégoriels selon vos besoins
        }
        return categorical_choices.get(param_name, [])
    
class FastExplorationSampler(optuna.samplers.TPESampler):
    def __init__(self, n_regions=5, samples_per_region=20):
        super().__init__()
        # Nombre de régions différentes à explorer
        self.n_regions = n_regions
        # Nombre d'essais par région avant de passer à une autre
        self.samples_per_region = samples_per_region
        # Stockage des régions explorées
        self.explored_regions = []
        # Compteur d'échantillons dans la région actuelle
        self.current_region_samples = 0
        
    def sample_relative(self, study, trial, search_space):
        # Si nous avons assez exploré la région actuelle, on en cherche une nouvelle
        if self.current_region_samples >= self.samples_per_region:
            self._jump_to_new_region(study)
            self.current_region_samples = 0
        
        # 80% du temps on explore la région actuelle
        if random.random() < 0.8:
            params = super().sample_relative(study, trial, search_space)
        # 20% du temps on fait une exploration complètement aléatoire
        else:
            params = self._random_exploration(search_space)
            
        self.current_region_samples += 1
        return params

    def _jump_to_new_region(self, study):
        """Saute vers une nouvelle région éloignée des précédentes"""
        # Mémorise le centre de la région actuelle
        if study.best_trial:
            current_region = study.best_trial.params
            self.explored_regions.append(current_region)
        
        # Génère une nouvelle région éloignée des précédentes
        max_attempts = 10
        for _ in range(max_attempts):
            new_params = self._generate_distant_params()
            if self._is_far_from_explored(new_params):
                return new_params
                
        return self._completely_random_params()

    def _is_far_from_explored(self, params, min_distance=0.3):
        """Vérifie si les nouveaux paramètres sont assez éloignés des régions explorées"""
        for region in self.explored_regions:
            distance = self._calculate_parameter_distance(params, region)
            if distance < min_distance:
                return False
        return True

    def _calculate_parameter_distance(self, params1, params2):
        """Calcule une distance normalisée entre deux ensembles de paramètres"""
        total_diff = 0
        n_params = 0
        
        for key in params1:
            if key in params2:
                # Normalisation selon le type de paramètre
                if isinstance(params1[key], (int, float)):
                    diff = abs(params1[key] - params2[key]) / max(abs(params1[key]), abs(params2[key]))
                else:
                    diff = 0 if params1[key] == params2[key] else 1
                total_diff += diff
                n_params += 1
                
        return total_diff / n_params if n_params > 0 else 1.0

def smart_trial_filter(storage_url: str, max_trials: int = 5000):
    """
    Filtre intelligent des trials qui préserve la diversité des recherches.
    """
    # Création de deux connexions distinctes
    db_path = storage_url.replace('sqlite:///', '')
    conn = sqlite3.connect(db_path)
    vacuum_conn = sqlite3.connect(db_path)  # Connexion séparée pour VACUUM
    
    try:
        # 1. Récupération des trials
        trials_data = conn.execute("""
            SELECT 
                t.number,
                t.state,
                v.value,
                t.datetime_start,
                t.datetime_complete,
                GROUP_CONCAT(DISTINCT tp.param_name || ':' || tp.param_value) as params,
                GROUP_CONCAT(DISTINCT tu.key || ':' || tu.value_json) as user_attrs
            FROM trials t
            LEFT JOIN trial_values v ON t.trial_id = v.trial_id
            LEFT JOIN trial_params tp ON t.trial_id = tp.trial_id
            LEFT JOIN trial_user_attributes tu ON t.trial_id = tu.trial_id
            WHERE t.state = 'COMPLETE'
            GROUP BY t.trial_id
        """).fetchall()

        if not trials_data:
            print("Aucun trial à filtrer")
            return

        # 2. Création des clusters
        features = []
        valid_trials = []
        
        for trial in trials_data:
            try:
                value = float(trial[2]) if trial[2] is not None else -999
                
                # Parsing des paramètres
                params = {}
                if trial[5]:
                    for param in trial[5].split(','):
                        if ':' in param:
                            key, val = param.split(':', 1)
                            try:
                                params[key] = float(val)
                            except:
                                continue

                # Parsing des attributs utilisateur
                user_attrs = {}
                if trial[6]:
                    for attr in trial[6].split(','):
                        if ':' in attr:
                            key, val = attr.split(':', 1)
                            try:
                                user_attrs[key] = float(json.loads(val))
                            except:
                                continue

                feature_vec = [
                    value,
                    user_attrs.get('win_rate', 0),
                    user_attrs.get('total_trades', 0),
                    user_attrs.get('max_drawdown', 1),
                    params.get('position_size', 0),
                    params.get('sl_pct', 0)
                ]
                
                features.append(feature_vec)
                valid_trials.append(trial)
            except:
                continue

        if not features:
            print("Pas assez de données valides pour le clustering")
            return

        # Normalisation et clustering
        features = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        n_clusters = min(10, len(features) // 10)  # max 10 clusters ou 1 cluster par 10 trials
        if n_clusters < 2:
            n_clusters = 2

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)

        # 3. Sélection des trials à conserver
        trials_to_keep = set()
        
        # Par cluster
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_trials = [valid_trials[i] for i in cluster_indices]
            
            # Trier par valeur (ROI/score)
            cluster_trials.sort(key=lambda x: float(x[2]) if x[2] is not None else float('-inf'), 
                              reverse=True)
            
            # Garder les meilleurs de chaque cluster
            keep_count = max(1, max_trials // n_clusters)
            for trial in cluster_trials[:keep_count]:
                trials_to_keep.add(trial[0])  # number

        # 4. Ajouter les trials récents
        recent_trials = conn.execute("""
            SELECT t.number
            FROM trials t
            WHERE t.state = 'COMPLETE'
            ORDER BY t.datetime_complete DESC
            LIMIT ?
        """, (max_trials // 10,)).fetchall()

        for trial in recent_trials:
            trials_to_keep.add(trial[0])

        # 5. Ajouter les meilleurs trials
        best_trials = conn.execute("""
            SELECT t.number
            FROM trials t
            JOIN trial_values v ON t.trial_id = v.trial_id
            WHERE t.state = 'COMPLETE'
            ORDER BY v.value DESC
            LIMIT ?
        """, (max_trials // 10,)).fetchall()

        for trial in best_trials:
            trials_to_keep.add(trial[0])

        # 6. Nettoyage avec transaction
        if trials_to_keep:
            trials_to_keep_str = ','.join(str(t) for t in trials_to_keep)
            
            conn.execute("BEGIN TRANSACTION")
            try:
                # Récupérer les trial_ids correspondants
                trial_ids = conn.execute(f"""
                    SELECT trial_id 
                    FROM trials 
                    WHERE number IN ({trials_to_keep_str})
                """).fetchall()
                
                if trial_ids:
                    trial_ids_str = ','.join(str(t[0]) for t in trial_ids)
                    
                    # Suppression des trials non conservés
                    conn.execute(f"""
                        DELETE FROM trials 
                        WHERE trial_id NOT IN ({trial_ids_str})
                        AND state = 'COMPLETE'
                    """)

                    # Nettoyage des tables associées
                    for table in [
                        'trial_params',
                        'trial_values',
                        'trial_user_attributes',
                        'trial_system_attributes',
                        'trial_intermediate_values'
                    ]:
                        try:
                            conn.execute(f"""
                                DELETE FROM {table}
                                WHERE trial_id NOT IN ({trial_ids_str})
                            """)
                        except:
                            continue

                conn.commit()
                
                # VACUUM avec la connexion séparée après commit
                vacuum_conn.execute("VACUUM")

                print(f"\n📊 Base de données optimisée:")
                print(f"  • {len(trials_to_keep)}/{len(trials_data)} trials conservés")
                print(f"  • {n_clusters} clusters préservés")
                print(f"  • {len(recent_trials)} trials récents conservés")
                print(f"  • {len(best_trials)} meilleurs trials conservés")

            except Exception as e:
                conn.rollback()
                raise e

        gc.collect()

    except Exception as e:
        print(f"Erreur lors du filtrage: {e}")
        traceback.print_exc()
    finally:
        conn.close()
        vacuum_conn.close()

def init_worker(shared_data):
    """Initialise les données partagées pour chaque worker"""
    global _shared_data
    _shared_data = shared_data

class BatchProgress:
    """Gestionnaire amélioré de l'affichage des progrès"""
    
    def __init__(self, n_batches: int, trials_per_batch: int):
        self.n_batches = n_batches
        self.trials_per_batch = trials_per_batch
        self.progress = {i: 0 for i in range(n_batches)}
        self.success_count = {i: 0 for i in range(n_batches)}
        self.fail_count = {i: 0 for i in range(n_batches)}
        self.best_values = {i: float('-inf') for i in range(n_batches)}
        self.last_update = time.time()
        self.update_interval = 0.5  # Intervalle minimum entre les mises à jour

    def update(self, batch_results: dict) -> None:
        """Met à jour les progrès d'un batch"""
        if not batch_results:
            return

        for batch_id, data in batch_results.items():
            if not isinstance(data, dict):
                continue
                
            bid = int(batch_id.split('_')[1])
            self.success_count[bid] = data.get('success_count', 0)
            self.fail_count[bid] = data.get('fail_count', 0)
            self.progress[bid] = self.success_count[bid] + self.fail_count[bid]
            
            value = data.get('best_value', float('-inf'))
            if value > self.best_values[bid]:
                self.best_values[bid] = value

    def should_update_display(self) -> bool:
        """Vérifie si l'affichage doit être mis à jour"""
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.last_update = current_time
            return True
        return False

    def get_progress_str(self) -> str:
        """Génère une chaîne formatée des progrès"""
        output = ["\033[2J\033[H"]  # Clear screen and move cursor to top
        output.append("\n🔄 Optimisation en cours...\n")
        
        total_success = 0
        total_fail = 0
        global_best = float('-inf')
        
        for batch_id in range(self.n_batches):
            # Calcul des métriques
            progress = self.progress[batch_id]
            success = self.success_count[batch_id]
            fail = self.fail_count[batch_id]
            best = self.best_values[batch_id]
            
            # Mise à jour des totaux
            total_success += success
            total_fail += fail
            global_best = max(global_best, best)
            
            # Création de la barre de progression
            bar_length = 30
            filled = int(bar_length * progress / self.trials_per_batch)
            bar = ('█' * filled + '-' * (bar_length - filled))
            
            # Coloration selon le statut
            if fail > 0:
                bar_color = '\033[33m'  # Jaune si erreurs
            elif progress == self.trials_per_batch:
                bar_color = '\033[32m'  # Vert si terminé
            else:
                bar_color = '\033[34m'  # Bleu si en cours
                
            # Formatage de la ligne
            status = f"Lot {batch_id:2d} [{bar_color}{bar}\033[0m] {progress}/{self.trials_per_batch}"
            metrics = f"✅ {success} ❌ {fail}"
            if best > float('-inf'):
                metrics += f" 🏆 {best:.4f}"
            
            output.append(f"{status} | {metrics}")
        
        # Résumé global
        total_trials = sum(self.progress.values())
        total_target = self.n_batches * self.trials_per_batch
        progress_pct = (total_trials / total_target) * 100
        
        output.append("\n📊 Progression globale:")
        output.append(f"  • Progression: {total_trials}/{total_target} ({progress_pct:.1f}%)")
        output.append(f"  • Réussis: {total_success} ({total_success/max(1,total_trials)*100:.1f}%)")
        output.append(f"  • Échoués: {total_fail} ({total_fail/max(1,total_trials)*100:.1f}%)")
        if global_best > float('-inf'):
            output.append(f"  • Meilleur score: {global_best:.4f}")
        
        return "\n".join(output)

def run_batch(
    batch_id: int,
    n_trials: int,
    study_name: str,
    config: GeneralConfig,
    storage_url: str,
    progress_queue: Queue,
    progress_dict: Dict
) -> Optional[Dict[str, Any]]:
    """Version avec typage correct des objets multiprocessing"""
    try:
        # Initialisation des ressources
        prices_shm = SharedMemory(name=_shared_data['prices']['name'])
        high_shm = SharedMemory(name=_shared_data['high']['name'])
        low_shm = SharedMemory(name=_shared_data['low']['name'])

        try:
            # Setup des arrays
            prices = np.ndarray(
                _shared_data['prices']['shape'],
                dtype=_shared_data['prices']['dtype'],
                buffer=prices_shm.buf
            )
            high = np.ndarray(
                _shared_data['high']['shape'],
                dtype=_shared_data['high']['dtype'],
                buffer=high_shm.buf
            )
            low = np.ndarray(
                _shared_data['low']['shape'],
                dtype=_shared_data['low']['dtype'],
                buffer=low_shm.buf
            )

            # Configuration du stockage
            storage = optuna.storages.RDBStorage(
                url=storage_url,
                engine_kwargs={
                    'connect_args': {'timeout': 300},
                    'pool_size': 1
                }
            )

            study = optuna.load_study(study_name=study_name, storage=storage)
            successful_trials = 0
            failed_trials = 0
            best_value = float('-inf')
            last_update = time.time()

            # Create a diagnostic tools instance
            diagnostics = create_diagnostic_tools(
                enabled=True,
                output_dir='diagnostics',
                log_all_tests=True,
                custom_thresholds={
                    'roi_min': -1.02,         # ROI less than -100%
                    'roi_max': 5.0,         # ROI greater than 1000%
                    'drawdown_max': 1.1,     # Max drawdown greater than 110%
                    'win_rate_min': -1.0,    # Win rate less than 1%
                    'win_rate_max': 2.0,    # Win rate greater than 99%
                    'profit_factor_max': 100.0,  # Profit factor too high
                    'trades_min': -1,        # Too few trades
                },
                verbose=True  # Ajoutez ce paramètre
            )

            # Exécution des trials
            for trial_idx in range(n_trials):
                try:
                    trial = study.ask()
                    value = objective(trial, prices, high, low, config, diagnostics)
                    study.tell(trial, value)
                    
                    successful_trials += 1
                    best_value = max(best_value, value if value != float('-inf') else best_value)

                except Exception as e:
                    failed_trials += 1
                    print(f"\n❌ Erreur dans trial {trial_idx} du lot {batch_id}: {e}")

                # Mise à jour des progrès
                current_time = time.time()
                if current_time - last_update >= 0.2:
                    progress_data = {
                        'batch_id': batch_id,
                        'progress': trial_idx + 1,
                        'success_count': successful_trials,
                        'fail_count': failed_trials,
                        'best_value': best_value
                    }
                    progress_queue.put(progress_data)
                    progress_dict[f'batch_{batch_id}'] = progress_data
                    last_update = current_time

            # Mise à jour finale
            final_data = {
                'batch_id': batch_id,
                'progress': n_trials,
                'success_count': successful_trials,
                'fail_count': failed_trials,
                'best_value': best_value,
                'status': 'complete'
            }
            progress_queue.put(final_data)
            progress_dict[f'batch_{batch_id}'] = final_data

            return final_data

        finally:
            prices_shm.close()
            high_shm.close()
            low_shm.close()

    except Exception as e:
        print(f"\n❌ Erreur fatale dans le lot {batch_id}: {e}")
        return None

def display_progress(
    progress_queue: Queue,
    progress_dict: Dict,
    batch_progress: BatchProgress,
    stop_event: Event
) -> None:
    """Processus d'affichage avec typage correct"""
    last_display = time.time()
    display_interval = 0.5

    while not stop_event.is_set():
        try:
            # Collecte des mises à jour
            while not progress_queue.empty():
                update = progress_queue.get_nowait()
                if update:
                    batch_id = update['batch_id']
                    progress_dict[f'batch_{batch_id}'] = update

            # Mise à jour de la progression
            batch_progress.update(progress_dict)

            # Affichage périodique
            current_time = time.time()
            if current_time - last_display >= display_interval:
                print(batch_progress.get_progress_str())
                last_display = current_time

            time.sleep(0.1)
            
        except Exception as e:
            print(f"Erreur dans l'affichage: {e}")
            continue

def prepare_optimization(config: GeneralConfig, data_path: str, name: str = "trading_strategy") -> optuna.Study:
    """Version optimisée avec gestion correcte du multiprocessing"""
    
    # Configuration du stockage
    storage = optuna.storages.RDBStorage(
        url="sqlite:///optimization.db",
        engine_kwargs={
            'connect_args': {'timeout': 300},
            'isolation_level': None
        }
    )

    # Nettoyage préalable de la base
    print("\n🔍 Analyse et optimisation des trials existants...")
    smart_trial_filter(
        storage_url="sqlite:///optimization.db",
        max_trials=5000
    )

    # Configuration du sampler
    sampler = FastExplorationSampler(
        n_regions=5,              # Explore 5 régions différentes
        samples_per_region=20     # 20 essais par région avant de changer
    )
    study = optuna.create_study(storage=storage, sampler=sampler, 
                              study_name=name, direction="maximize",
                              load_if_exists=True)
    
    shared_data = init_shared_data(data_path)
    n_jobs = min(config.optimization.n_jobs if config.optimization.n_jobs > 0 
                 else mp.cpu_count(), mp.cpu_count())
    batch_size = max(10, config.optimization.n_trials // n_jobs)

    try:
        with mp.Manager() as manager:
            # Création des objets partagés
            progress_queue = manager.Queue()
            progress_dict = manager.dict()
            stop_event = manager.Event()
            
            # Configuration du système de progrès
            batch_progress = BatchProgress(n_jobs, batch_size)
            
            # Démarrage du processus d'affichage
            display_process = mp.Process(
                target=display_progress,
                args=(progress_queue, progress_dict, batch_progress, stop_event)
            )
            display_process.start()

            # Exécution des lots
            with ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=init_worker,
                initargs=(shared_data,)
            ) as executor:
                futures = []
                
                # Création des futures
                for i in range(n_jobs):
                    future = executor.submit(
                        run_batch,
                        i, batch_size, name, config,
                        storage.url, progress_queue, progress_dict
                    )
                    futures.append(future)

                # Attente des résultats
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            batch_id = result['batch_id']
                            progress_dict[f'batch_{batch_id}'] = result
                    except Exception as e:
                        print(f"\n❌ Erreur dans un lot: {e}")

            # Arrêt propre
            stop_event.set()
            display_process.join(timeout=1)
            if display_process.is_alive():
                display_process.terminate()

    finally:
        cleanup_shared_data(shared_data)

    return study

def init_shared_data(data_path: str) -> Dict:
    """Initialise les données partagées en précision double"""
    df = pd.read_csv(data_path)
    prices = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)

    def create_shared_array(array: np.ndarray, name_prefix: str) -> Dict:
        shm = SharedMemory(create=True, size=array.nbytes, name=f"{name_prefix}_{os.getpid()}")
        shared_array = np.ndarray(array.shape, dtype=np.float64, buffer=shm.buf)
        shared_array[:] = array[:]
        return {
            'name': shm.name,
            'shm': shm,
            'shape': array.shape,
            'dtype': np.float64
        }

    return {
        'prices': create_shared_array(prices, 'prices'),
        'high': create_shared_array(high, 'high'),
        'low': create_shared_array(low, 'low')
    }

# =========================================================
# 6. Analyse et Visualisation des Résultats
# =========================================================
class StrategyAnalyzer:
    def __init__(self, study: optuna.study, config: StrategyStructureConfig):
        self.study = study
        self.config = config
    
    def plot_multidimensional_exploration(self) -> None:
        """
        Visualisation multidimensionnelle avec gestion des paramètres manquants
        """
        # Récupération des trials terminés
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        # Création d'un ensemble de tous les paramètres possibles
        all_params = set()
        for trial in completed_trials:
            all_params.update(trial.params.keys())
        param_names = sorted(list(all_params))
        print(param_names)
        # Création de la matrice X avec gestion des valeurs manquantes
        X = []
        for trial in completed_trials:
            # Pour chaque paramètre possible, prendre sa valeur ou 0 si absent
            param_values = []
            for param in param_names:
                value = trial.params.get(param, 0)
                # Conversion des valeurs catégorielles en numériques
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except:
                        # Pour les valeurs catégorielles, utiliser un hash normalisé
                        value = hash(value) % 100 / 100.0
                param_values.append(value)
            X.append(param_values)

        X = np.array(X)
        print(X.shape)
        print(np.isnan(X))
        scores = np.array([t.value if t.value is not None else float('-inf') for t in completed_trials])
        trials = np.array([t.number for t in completed_trials])

        # Normalisation des données
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Réduction de dimensionnalité
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        
        tsne = TSNE(n_components=2, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled)

        # Création de la figure avec sous-graphiques
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{'type': 'scene'}, {'type': 'scatter'}],
                [{'colspan': 2}, None]
            ],
            subplot_titles=(
                'Exploration 3D (PCA)',
                'Projection 2D (t-SNE)',
                'Évolution des composantes principales'
            )
        )

        # 1. Vue 3D avec PCA
        fig.add_trace(
            go.Scatter3d(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                z=X_pca[:, 2],
                mode='markers+lines',
                marker=dict(
                    size=6,
                    color=scores,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Score')
                ),
                line=dict(
                    color='red',
                    width=2
                ),
                text=[f"Trial {t}<br>Score: {s:.4f}" for t, s in zip(trials, scores)],
                name='Exploration PCA'
            ),
            row=1, col=1
        )

        # 2. Vue 2D avec t-SNE
        fig.add_trace(
            go.Scatter(
                x=X_tsne[:, 0],
                y=X_tsne[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=scores,
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[f"Trial {t}<br>Score: {s:.4f}" for t, s in zip(trials, scores)],
                name='Projection t-SNE'
            ),
            row=1, col=2
        )

        # 3. Évolution temporelle des composantes
        for i in range(3):
            fig.add_trace(
                go.Scatter(
                    x=trials,
                    y=X_pca[:, i],
                    mode='lines',
                    name=f'PC{i+1}',
                    line=dict(width=2)
                ),
                row=2, col=1
            )

        # Ajout des informations sur la variance expliquée
        explained_variance = pca.explained_variance_ratio_
        variance_text = (
            f"Variance expliquée:<br>"
            f"PC1: {explained_variance[0]:.2%}<br>"
            f"PC2: {explained_variance[1]:.2%}<br>"
            f"PC3: {explained_variance[2]:.2%}"
        )

        fig.add_annotation(
            text=variance_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=12)
        )

        # Configuration du layout
        fig.update_layout(
            height=1200,
            width=1600,
            title_text="Visualisation multidimensionnelle de l'exploration des paramètres",
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3"
            ),
            showlegend=True,
            # Ajout des marges pour éviter la superposition
            margin=dict(r=100, l=50, t=100, b=50)
        )
        
        fig.show()

        # Création et affichage de la carte de corrélation
        self.plot_correlation_map(param_names, X, scores)

    def plot_correlation_map(self, param_names: List[str], X: np.ndarray, scores: np.ndarray) -> None:
        """
        Crée et affiche une carte de corrélation entre les paramètres et les scores.
        """
        corr_matrix = np.corrcoef(np.column_stack([X, scores]).T)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=param_names + ['Score'],
            y=param_names + ['Score'],
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title='Carte des corrélations entre paramètres et score',
            width=800,
            height=800
        )
        
        fig.show()

    def plot_optimization_history(self) -> None:
        """Affiche l'historique d'optimisation"""
        # Filtrer les valeurs valides
        valid_trials = [t.value for t in self.study.trials if t.value is not None and t.value != float('-inf')]
        
        # Si aucune valeur valide, afficher un message
        if not valid_trials:
            print("Aucune valeur d'optimisation valide trouvée.")
            return
        
        # Valeurs d'optimisation
        fig = go.Figure()
        
        # Tous les trials (pour les points)
        values = [t.value if t.value is not None and t.value != float('-inf') else 0 for t in self.study.trials]
        trials = list(range(len(values)))
        
        # Calcul des meilleures valeurs
        best_values = []
        current_best = float('-inf')
        for val in values:
            if val > current_best:
                current_best = val
            best_values.append(current_best)
        
        # Tracé des valeurs
        fig.add_trace(go.Scatter(
            x=trials,
            y=values,
            mode='markers',
            name='Trial Value',
            marker=dict(
                size=8, 
                opacity=0.5,
                # Changer la couleur pour les valeurs invalides
                color=[
                    'red' if t.value is None or t.value == float('-inf') else 'blue' 
                    for t in self.study.trials
                ]
            )
        ))
        
        # Tracé de la meilleure valeur
        fig.add_trace(go.Scatter(
            x=trials,
            y=best_values,
            mode='lines',
            name='Best Value',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Historique d\'optimisation',
            xaxis_title='Numéro d\'essai',
            yaxis_title='Score',
            showlegend=True
        )
        
        fig.show()

    def plot_metric_distributions(self) -> None:
        """
        Visualisation améliorée des distributions de métriques de performance avec statistiques
        """
        import numpy as np
        
        def calculate_stats(values):
            """Calcule les statistiques clés pour une série de valeurs"""
            if not values:
                return None
            return {
                'median': np.median(values),
                'q1': np.percentile(values, 25),
                'q3': np.percentile(values, 75),
                'min': np.min(values),
                'max': np.max(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        # Configuration des métriques avec leurs propriétés
        primary_metrics = {
            'ROI': {
                'values': [t.user_attrs['roi'] * 100 for t in self.study.trials 
                        if 'roi' in t.user_attrs and t.value != float('-inf')],
                'suffix': '%',
                'color': '#00b894',  # Vert menthe
                'yaxis': 'y'
            },
            'Win Rate': {
                'values': [min(t.user_attrs['win_rate'] * 100, 100) for t in self.study.trials 
                        if 'win_rate' in t.user_attrs and t.value != float('-inf')],
                'suffix': '%',
                'color': '#00cec9',  # Bleu clair
                'yaxis': 'y'
            },
            'Max Drawdown': {
                'values': [min(t.user_attrs['max_drawdown'] * 100, 100) for t in self.study.trials 
                        if 'max_drawdown' in t.user_attrs and t.value != float('-inf')],
                'suffix': '%',
                'color': '#d63031',  # Rouge
                'yaxis': 'y'
            }
        }
        
        secondary_metrics = {
            'Daily Trades': {
                'values': [t.user_attrs.get('trades_per_day', 0) for t in self.study.trials 
                        if 'trades_per_day' in t.user_attrs and t.value != float('-inf')],
                'suffix': '',
                'color': '#0984e3',  # Bleu
                'yaxis': 'y2'
            }
        }
        
        # Création de la figure
        fig = go.Figure()
        
        # Fonction pour ajouter un boxplot avec ses annotations
        def add_boxplot_with_stats(metric_name, metric_data, stats):
            # Ajout du boxplot
            fig.add_trace(go.Box(
                y=metric_data['values'],
                name=metric_name,
                boxmean=True,  # Ajoute un marqueur pour la moyenne
                boxpoints='outliers',  # Montre les points aberrants
                fillcolor=metric_data['color'],
                line=dict(color=metric_data['color']),
                yaxis=metric_data['yaxis'],
                hoverlabel=dict(
                    bgcolor=metric_data['color'],
                    font=dict(size=12, color='white')
                ),
                hovertemplate=(
                    f"<b>{metric_name}</b><br>" +
                    f"Médiane: %{{median:.2f}}{metric_data['suffix']}<br>" +
                    f"Moyenne: {stats['mean']:.2f}{metric_data['suffix']}<br>" +
                    f"Écart-type: {stats['std']:.2f}{metric_data['suffix']}<br>" +
                    f"Q1: %{{q1:.2f}}{metric_data['suffix']}<br>" +
                    f"Q3: %{{q3:.2f}}{metric_data['suffix']}<br>" +
                    f"Min: %{{min:.2f}}{metric_data['suffix']}<br>" +
                    f"Max: %{{max:.2f}}{metric_data['suffix']}<br>" +
                    "<extra></extra>"
                )
            ))
            
            # Ajout de l'annotation avec les statistiques
            y_pos = stats['max'] + abs(stats['max'] * 0.05)
            fig.add_annotation(
                x=metric_name,
                y=y_pos,
                text=(f"<b>{metric_name}</b><br>" +
                    f"Med: {stats['median']:.2f}{metric_data['suffix']}<br>" +
                    f"Moy: {stats['mean']:.2f}{metric_data['suffix']}"),
                showarrow=False,
                font=dict(
                    family="Arial",
                    size=10,
                    color=metric_data['color']
                ),
                align='center',
                xanchor='center',
                yanchor='bottom',
                xref='x',
                yref='y2' if metric_data['yaxis'] == 'y2' else 'y',
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor=metric_data['color'],
                borderwidth=1,
                borderpad=4,
                yshift=10
            )
        
        # Ajout des métriques primaires
        for metric_name, metric_data in primary_metrics.items():
            if metric_data['values']:
                stats = calculate_stats(metric_data['values'])
                add_boxplot_with_stats(metric_name, metric_data, stats)
        
        # Ajout des métriques secondaires
        for metric_name, metric_data in secondary_metrics.items():
            if metric_data['values']:
                stats = calculate_stats(metric_data['values'])
                add_boxplot_with_stats(metric_name, metric_data, stats)
        
        # Configuration du layout
        fig.update_layout(
            title=dict(
                text='Distribution des Métriques de Performance',
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(
                    family="Arial",
                    size=24,
                    color="#2d3436"
                )
            ),
            yaxis=dict(
                title=dict(
                    text='Pourcentage (%)',
                    font=dict(family="Arial", color="#2d3436", size=14)
                ),
                tickfont=dict(family="Arial", color="#2d3436", size=12),
                gridcolor='rgba(189, 195, 199, 0.4)',
                zeroline=True,
                zerolinecolor='rgba(189, 195, 199, 0.8)',
                zerolinewidth=2
            ),
            yaxis2=dict(
                title=dict(
                    text='Nombre de trades par jour',
                    font=dict(family="Arial", color="#0984e3", size=14)
                ),
                tickfont=dict(family="Arial", color="#0984e3", size=12),
                overlaying='y',
                side='right',
                gridcolor='rgba(189, 195, 199, 0.2)',
                zeroline=True,
                zerolinecolor='rgba(189, 195, 199, 0.8)',
                zerolinewidth=2
            ),
            showlegend=False,
            width=1600,
            height=800,
            template='plotly_white',
            boxmode='group',
            margin=dict(l=80, r=80, t=100, b=80),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Configuration des axes
        fig.update_xaxes(
            showgrid=True,
            gridcolor='rgba(189, 195, 199, 0.4)',
            showline=True,
            linecolor='rgba(189, 195, 199, 0.8)',
            linewidth=2,
            tickfont=dict(family="Arial", size=12, color="#2d3436")
        )
        
        # Affichage de la figure
        fig.show()

    def plot_param_importances(self) -> None:
        """
        Visualisation améliorée de l'importance des paramètres
        """
        # Calcul des importances
        try:
            importance = optuna.importance.get_param_importances(self.study)
            
            # Filtrer les paramètres avec une importance > 0
            importance = {k: v for k, v in importance.items() if v > 0}
            
            # Trier par importance
            params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            param_names = [p[0] for p in params]
            param_values = [p[1] for p in params]
            
            # Création du graphique
            fig = go.Figure()
            
            # Ajout des barres avec gradient de couleur
            fig.add_trace(go.Bar(
                x=param_values,
                y=param_names,
                orientation='h',
                marker=dict(
                    color=param_values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Importance",
                        thickness=20
                    )
                ),
                hovertemplate="Paramètre: %{y}<br>Importance: %{x:.3f}<extra></extra>"
            ))
            
            # Mise en page
            fig.update_layout(
                title={
                    'text': 'Importance Relative des Paramètres',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=24)
                },
                xaxis_title='Importance',
                yaxis_title='Paramètre',
                width=1600,
                height=max(800, len(param_names) * 30),
                template='plotly_white',
                margin=dict(l=200, r=50, t=100, b=50)  # Augmentation de la marge gauche pour les noms de paramètres
            )
            
            fig.show()
            
        except Exception as e:
            print(f"Erreur lors du calcul des importances : {e}")
            traceback.print_exc()

    def plot_strategy_composition(self) -> None:
        """
        Analyse visuelle comprehensive de la composition des stratégies
        """
        # Filtrer les meilleurs trials valides
        top_trials = [
            t for t in self.study.trials 
            if t.value is not None and t.value != float('-inf')
        ]
        
        # Trier et prendre les top 10
        top_trials = sorted(top_trials, key=lambda t: t.value, reverse=True)[:10]
        
        # Si aucun trial valide, afficher un message
        if not top_trials:
            print("Aucun trial d'optimisation valide trouvé.")
            return
        
        # Analyse des paramètres
        stats = {
            'Indicateurs d\'Achat': {},
            'Indicateurs de Vente': {},
            'Opérateurs d\'Achat': {},
            'Opérateurs de Vente': {},
            'Périodes d\'Achat': [],
            'Périodes de Vente': [],
            'Blocs d\'Achat': [],
            'Blocs de Vente': [],
            'Position Size': [],
            'Stop Loss': [],
            'Take Profit': [],
            'ROI': [],
            'Win Rate': [],
            'Max Drawdown': [],
            'Trades par Jour': []
        }
        
        # Collecte des statistiques
        for trial in top_trials:
            # Métriques de performance
            stats['ROI'].append(trial.user_attrs.get('roi', 0) * 100)
            stats['Win Rate'].append(trial.user_attrs.get('win_rate', 0) * 100)
            stats['Max Drawdown'].append(trial.user_attrs.get('max_drawdown', 0) * 100)
            stats['Trades par Jour'].append(trial.user_attrs.get('trades_per_day', 0))
            
            # Paramètres de trading
            stats['Position Size'].append(trial.params.get('position_size', 0) * 100)
            stats['Stop Loss'].append(trial.params.get('sl_pct', 0) * 100)
            stats['Take Profit'].append(trial.params.get('tp_multiplier', 0) * trial.params.get('sl_pct', 0) * 100)
            
            # Analyse des blocs et conditions
            n_buy_blocks = trial.params.get('n_buy_blocks', 1)
            n_sell_blocks = trial.params.get('n_sell_blocks', 1)
            stats['Blocs d\'Achat'].append(n_buy_blocks)
            stats['Blocs de Vente'].append(n_sell_blocks)
            
            # Analyse des blocs d'achat
            for b in range(n_buy_blocks):
                for c in range(trial.params.get(f'buy_block_{b}_conditions', 1)):
                    # Indicateurs d'achat
                    ind_type = trial.params.get(f'buy_b{b}_c{c}_ind1_type', 'EMA')
                    stats['Indicateurs d\'Achat'][ind_type] = stats['Indicateurs d\'Achat'].get(ind_type, 0) + 1
                    
                    # Opérateurs d'achat
                    op = trial.params.get(f'buy_b{b}_c{c}_operator', '>')
                    stats['Opérateurs d\'Achat'][op] = stats['Opérateurs d\'Achat'].get(op, 0) + 1
                    
                    # Périodes d'achat
                    period = trial.params.get(f'buy_b{b}_c{c}_ind1_period', 14)
                    stats['Périodes d\'Achat'].append(period)
            
            # Analyse des blocs de vente
            for b in range(n_sell_blocks):
                for c in range(trial.params.get(f'sell_block_{b}_conditions', 1)):
                    ind_type = trial.params.get(f'sell_b{b}_c{c}_ind1_type', 'EMA')
                    stats['Indicateurs de Vente'][ind_type] = stats['Indicateurs de Vente'].get(ind_type, 0) + 1
                    
                    op = trial.params.get(f'sell_b{b}_c{c}_operator', '<')
                    stats['Opérateurs de Vente'][op] = stats['Opérateurs de Vente'].get(op, 0) + 1
                    
                    period = trial.params.get(f'sell_b{b}_c{c}_ind1_period', 14)
                    stats['Périodes de Vente'].append(period)
        
        # Création des visualisations
        fig = make_subplots(
            rows=3, cols=3, 
            subplot_titles=(
                'Indicateurs d\'Achat', 
                'Indicateurs de Vente', 
                'Blocs de Conditions',
                'Opérateurs d\'Achat', 
                'Opérateurs de Vente', 
                'Périodes d\'Indicateurs',
                'Métriques de Performance', 
                'Paramètres de Trading', 
                'Distribution des Scores'
            )
        )
        
        # 1. Indicateurs d'Achat
        fig.add_trace(
            go.Bar(
                x=list(stats['Indicateurs d\'Achat'].keys()),
                y=list(stats['Indicateurs d\'Achat'].values()),
                marker_color='#00b894',
                name='Indicateurs d\'Achat'
            ),
            row=1, col=1
        )
        
        # 2. Indicateurs de Vente
        fig.add_trace(
            go.Bar(
                x=list(stats['Indicateurs de Vente'].keys()),
                y=list(stats['Indicateurs de Vente'].values()),
                marker_color='#0984e3',
                name='Indicateurs de Vente'
            ),
            row=1, col=2
        )
        
        # 3. Blocs de Conditions
        fig.add_trace(
            go.Box(
                y=stats['Blocs d\'Achat'],
                name='Blocs Achat',
                marker_color='#a29bfe'
            ),
            row=1, col=3
        )
        fig.add_trace(
            go.Box(
                y=stats['Blocs de Vente'],
                name='Blocs Vente',
                marker_color='#6a89cc'
            ),
            row=1, col=3
        )
        
        # 4. Opérateurs d'Achat
        fig.add_trace(
            go.Bar(
                x=list(stats['Opérateurs d\'Achat'].keys()),
                y=list(stats['Opérateurs d\'Achat'].values()),
                marker_color='#fab1a0',
                name='Opérateurs d\'Achat'
            ),
            row=2, col=1
        )
        
        # 5. Opérateurs de Vente
        fig.add_trace(
            go.Bar(
                x=list(stats['Opérateurs de Vente'].keys()),
                y=list(stats['Opérateurs de Vente'].values()),
                marker_color='#fdcb6e',
                name='Opérateurs de Vente'
            ),
            row=2, col=2
        )
        
        # 6. Périodes d'Indicateurs
        fig.add_trace(
            go.Box(
                y=stats['Périodes d\'Achat'],
                name='Périodes Achat',
                marker_color='#00cec9'
            ),
            row=2, col=3
        )
        fig.add_trace(
            go.Box(
                y=stats['Périodes de Vente'],
                name='Périodes Vente',
                marker_color='#6c5ce7'
            ),
            row=2, col=3
        )
        
        # 7. Métriques de Performance
        performance_metrics = ['ROI', 'Win Rate', 'Max Drawdown', 'Trades par Jour']
        colors = ['#00b894', '#0984e3', '#d63031', '#6a89cc']
        
        for i, (metric, color) in enumerate(zip(performance_metrics, colors), 1):
            fig.add_trace(
                go.Box(
                    y=stats[metric],
                    name=metric,
                    marker_color=color
                ),
                row=3, col=1
            )
        
        # 8. Paramètres de Trading
        trading_params = ['Position Size', 'Stop Loss', 'Take Profit']
        trading_colors = ['#00cec9', '#d63031', '#6c5ce7']
        
        for i, (param, color) in enumerate(zip(trading_params, trading_colors), 1):
            fig.add_trace(
                go.Box(
                    y=stats[param],
                    name=param,
                    marker_color=color
                ),
                row=3, col=2
            )
        
        # 9. Distribution des Scores
        values = [t.value for t in self.study.trials if t.value is not None]
        fig.add_trace(
            go.Histogram(
                x=values,
                name='Distribution des Scores',
                marker_color='#00b894'
            ),
            row=3, col=3
        )
        
        # Mise en page
        fig.update_layout(
            height=1600, 
            width=1600, 
            title_text="Composition Détaillée des Stratégies",
            showlegend=True,
            template='plotly_white'
        )
        
        fig.show()
        
    def generate_strategy_report(self, filepath: str) -> None:
        """Génère un rapport détaillé de la stratégie"""
        best_trial = self.study.best_trial
        best_params = best_trial.params
        
        report = {
            "optimization_summary": {
                "n_trials": len(self.study.trials),
                "best_value": self.study.best_value,
                "best_trial_number": best_trial.number,
                "optimization_duration": None  # À implémenter
            },
            "best_strategy": {
                "parameters": best_params,
                "metrics": {
                    "roi": best_trial.user_attrs['roi'],
                    "win_rate": best_trial.user_attrs['win_rate'],
                    "total_trades": best_trial.user_attrs['total_trades'],
                    "max_drawdown": best_trial.user_attrs['max_drawdown'],
                    "sharpe": best_trial.user_attrs['sharpe'],
                    "trades_per_day": best_trial.user_attrs.get('trades_per_day', 0)
                }
            },
            "strategy_analysis": {
                "n_buy_blocks": sum(1 for k in best_params if k.startswith('buy_block_')),
                "n_sell_blocks": sum(1 for k in best_params if k.startswith('sell_block_')),
                "total_conditions": sum(
                    best_params[k] for k in best_params 
                    if k.endswith('_conditions')
                ),
                "indicators_used": list(set(
                    value.split('_')[0] for key, value in best_params.items()
                    if ('_ind1' in key or '_ind2' in key) and isinstance(value, str)
                ))
            },
            "config_summary": {
                "indicators": {
                    name: {
                        "min_period": config.min_period,
                        "max_period": config.max_period,
                        "step": config.step
                    } for name, config in self.config.indicators.items()
                },
                "max_blocks": self.config.max_blocks,
                "max_conditions": self.config.max_conditions_per_block
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=4)
            
        print(f"Rapport sauvegardé: {filepath}")

def main():
    # Création de la configuration complète
    config = GeneralConfig(
        # Risk Management Configuration
        risk=RiskConfig(
            position_size_range = (0.01, 1.0),  # 1% à 100%
            position_step= 0.01,
            
            # Plages pour les stop loss en pourcentage du prix d'entrée
            sl_range = (0.001, 1.0),  # 0.1% à 10%
            sl_step = 0.001,
            
            # Take profit comme multiplicateur du stop loss
            tp_multiplier_range= (0.1, 10.0),  # 1x à 10x du SL
            tp_multiplier_step= 0.1,
            
            # Paramètres ATR pour le mode dynamique
            atr_period_range= (2, 30),
            atr_multiplier_range = (0.5, 3.0),
            
            # Paramètres de volatilité pour le mode dynamique
            vol_period_range= (1, 50),
            vol_multiplier_range = (0.5, 3.0)
        ),

        # Strategy Structure Configuration
        structure=StrategyStructureConfig(
            max_blocks=3,  # Maximum 3 condition blocks per signal type
            min_blocks=1,  # Minimum 1 condition block per signal type
            max_conditions_per_block=3,  # Maximum 3 conditions per block
            min_conditions_per_block=1,  # Minimum 1 condition per block
            cross_signals_probability=0.5,  # 30% chance of using crossover signals
            value_comparison_probability=0.1,  # 40% chance of comparing with fixed values
            rsi_value_range=(0.0, 100.0),  # RSI comparison values between 20 and 80
            price_value_range=(0.0, 100000.0),  # Price comparison values between 0 and 1000
            general_value_range=(-1000.0, 1000.0)  # General indicator comparison values
        ),

        # Optimization Configuration
        optimization=OptimizationConfig(
            min_trades=10,
            n_trials=10000,  # Run 1000 optimization trials
            n_jobs= 21,  # Use all available CPU cores
            timeout=36000,  # 1 hour maximum optimization time
            gc_after_trial=True,  # Run garbage collection after each trial
            n_startup_trials=100,  # Initial random trials before optimization
            n_ei_candidates=10,  # Number of candidates for expected improvement
            early_stopping_n_trials=50,  # Check for early stopping every 50 trials
            early_stopping_threshold=None  # Minimum improvement threshold for early stopping
        ),

        # Simulation Configuration
        simulation=SimulationConfig(
            initial_balance=1000.0,  # Starting with $1000
            fee_open=0.001,  # 0.1% fee for opening positions
            fee_close=0.001,  # 0.1% fee for closing positions
            slippage=0.001,  # 0.1% slippage per trade
            tick_size=0.0001,  # Minimum price movement of 0.0001 BTC
            leverage_range=(1, 125),  # Leverage between 1x and 10x
        ),

        # Technical Indicators Configuration
        indicators={
                "EMA": IndicatorConfig(
                    type=IndicatorType.EMA,
                    min_period=2,
                    max_period=2000,
                    step=2,
                    price_type="close"
                ),
                "SMA": IndicatorConfig(
                    type=IndicatorType.SMA,
                    min_period=2,
                    max_period=2000,
                    step=5,
                    price_type="close"
                ),
                "RSI": IndicatorConfig(
                    type=IndicatorType.RSI,
                    min_period=2,
                    max_period=300,
                    step=1,
                    price_type="close"
                ),
                "ATR": IndicatorConfig(
                    type=IndicatorType.ATR,
                    min_period=5,
                    max_period=300,
                    step=1
                )
        }
    )
    '''"MACD": IndicatorConfig(
                type=IndicatorType.MACD,
                min_period=12,
                max_period=26,
                step=2,
                price_type="close"
            ),
            "BOLL": IndicatorConfig(
                type=IndicatorType.BOLL,
                min_period=10,
                max_period=50,
                step=5,
                price_type="close"
            ),
            "STOCH": IndicatorConfig(
                type=IndicatorType.STOCH,
                min_period=5,
                max_period=30,
                step=1,
                price_type="close"
            )'''
    # Validation de la configuration
    if not config.validate():
        print("Configuration invalide!")
        return
        
    try:
        # Lancement de l'optimisation avec la nouvelle configuration
        study = prepare_optimization(config, "data/BTC_USDT_1m_binance_2020-01-01.csv", "1min")
        
        if study is not None:
            analyzer = StrategyAnalyzer(study, config.structure)
            
            # Générer les visualisations
            analyzer.plot_multidimensional_exploration()
            analyzer.plot_optimization_history()
            analyzer.plot_param_importances()
            analyzer.plot_metric_distributions()
            analyzer.plot_strategy_composition()
        
    except Exception as e:
        print(f"Erreur lors de l'optimisation : {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()