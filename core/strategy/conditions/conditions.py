"""
Module d'évaluation des conditions de trading optimisé avec Numba.
Permet l'évaluation efficace des blocs de conditions pour générer des signaux.
"""

import numpy as np
import pandas as pd
from numba import njit, prange, float64, boolean, int32
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from time import time
import math

from core.strategy.conditions.conditions_config import (
    ConditionConfig, BlockConfig, StrategyBlocksConfig,
    OperatorType, LogicOperatorType,
    PriceOperand, IndicatorOperand, ValueOperand
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== FONCTIONS D'ÉVALUATION NUMBA ==========
@njit(cache=True)
def evaluate_condition_array(
    left_array: np.ndarray,
    right_array: np.ndarray,
    operator_code: int,
    lookback: int
) -> np.ndarray:
    """
    Évalue une condition entre deux arrays numériques.
    
    Args:
        left_array: Array gauche
        right_array: Array droit
        operator_code: Code de l'opérateur (0=>, 1=<, 2=>=, 3=<=, 4===, 5=CROSS_ABOVE, 6=CROSS_BELOW)
        lookback: Décalage (lookback) à appliquer
        
    Returns:
        Array de booléens indiquant si la condition est vraie pour chaque point
    """
    n = len(left_array)
    result = np.zeros(n, dtype=np.bool_)
    
    # Appliquer le lookback si nécessaire
    if lookback > 0 and lookback < n:
        # Créer des copies avec décalage
        left_shifted = np.zeros(n, dtype=np.float64)
        left_shifted[lookback:] = left_array[:-lookback]
        left_array = left_shifted
    
    # Évaluer la condition selon l'opérateur
    if operator_code == 0:  # GREATER (>)
        result = left_array > right_array
    elif operator_code == 1:  # LESS (<)
        result = left_array < right_array
    elif operator_code == 2:  # GREATER_EQUAL (>=)
        result = left_array >= right_array
    elif operator_code == 3:  # LESS_EQUAL (<=)
        result = left_array <= right_array
    elif operator_code == 4:  # EQUAL (==)
        # Utiliser une tolérance pour les comparaisons d'égalité de nombres à virgule flottante
        tolerance = 1e-8
        result = np.abs(left_array - right_array) < tolerance
    elif operator_code == 5:  # CROSS_ABOVE
        # Initialiser tout à False
        result = np.zeros(n, dtype=np.bool_)
        
        # Vérifier chaque point à partir du second
        for i in range(1, n):
            # CROSS_ABOVE: left était <= right et maintenant left > right
            result[i] = (left_array[i-1] <= right_array[i-1]) and (left_array[i] > right_array[i])
    elif operator_code == 6:  # CROSS_BELOW
        # Initialiser tout à False
        result = np.zeros(n, dtype=np.bool_)
        
        # Vérifier chaque point à partir du second
        for i in range(1, n):
            # CROSS_BELOW: left était >= right et maintenant left < right
            result[i] = (left_array[i-1] >= right_array[i-1]) and (left_array[i] < right_array[i])
    
    return result


@njit(cache=True)
def evaluate_condition_value(
    left_array: np.ndarray,
    value: float,
    operator_code: int,
    lookback: int
) -> np.ndarray:
    """
    Évalue une condition entre un array et une valeur fixe.
    
    Args:
        left_array: Array gauche
        value: Valeur fixe droite
        operator_code: Code de l'opérateur
        lookback: Décalage (lookback) à appliquer
        
    Returns:
        Array de booléens indiquant si la condition est vraie pour chaque point
    """
    n = len(left_array)
    result = np.zeros(n, dtype=np.bool_)
    
    # Appliquer le lookback si nécessaire
    if lookback > 0 and lookback < n:
        # Créer une copie avec décalage
        left_shifted = np.zeros(n, dtype=np.float64)
        left_shifted[lookback:] = left_array[:-lookback]
        left_array = left_shifted
    
    # Évaluer la condition selon l'opérateur
    if operator_code == 0:  # GREATER (>)
        result = left_array > value
    elif operator_code == 1:  # LESS (<)
        result = left_array < value
    elif operator_code == 2:  # GREATER_EQUAL (>=)
        result = left_array >= value
    elif operator_code == 3:  # LESS_EQUAL (<=)
        result = left_array <= value
    elif operator_code == 4:  # EQUAL (==)
        # Utiliser une tolérance pour les comparaisons d'égalité de nombres à virgule flottante
        tolerance = 1e-8
        result = np.abs(left_array - value) < tolerance
    elif operator_code == 5:  # CROSS_ABOVE
        # Initialiser tout à False
        result = np.zeros(n, dtype=np.bool_)
        
        # Vérifier chaque point à partir du second
        for i in range(1, n):
            # CROSS_ABOVE: left était <= value et maintenant left > value
            result[i] = (left_array[i-1] <= value) and (left_array[i] > value)
    elif operator_code == 6:  # CROSS_BELOW
        # Initialiser tout à False
        result = np.zeros(n, dtype=np.bool_)
        
        # Vérifier chaque point à partir du second
        for i in range(1, n):
            # CROSS_BELOW: left était >= value et maintenant left < value
            result[i] = (left_array[i-1] >= value) and (left_array[i] < value)
    
    return result


@njit(cache=True)
def combine_conditions(
    condition_results: np.ndarray,
    logic_operators: np.ndarray
) -> np.ndarray:
    """
    Combine les résultats des conditions avec les opérateurs logiques.
    
    Args:
        condition_results: Array 2D des résultats de conditions [n_conditions, n_points]
        logic_operators: Array des opérateurs logiques (0=AND, 1=OR)
        
    Returns:
        Array de booléens indiquant si le bloc est vrai pour chaque point
    """
    n_conditions, n_points = condition_results.shape
    
    if n_conditions == 0:
        # Pas de conditions, retourner tout False
        return np.zeros(n_points, dtype=np.bool_)
    
    if n_conditions == 1:
        # Une seule condition, retourner directement son résultat
        return condition_results[0]
    
    # Initialiser le résultat avec la première condition
    result = condition_results[0].copy()
    
    # Combiner avec les conditions suivantes
    for i in range(1, n_conditions):
        if logic_operators[i-1] == 0:  # AND
            result = result & condition_results[i]
        else:  # OR
            result = result | condition_results[i]
    
    return result


@njit(cache=True)
def evaluate_blocks(
    block_results: np.ndarray,
    require_all: bool
) -> np.ndarray:
    """
    Évalue les résultats de plusieurs blocs.
    
    Args:
        block_results: Array 2D des résultats de blocs [n_blocks, n_points]
        require_all: Si True, tous les blocs doivent être vrais (AND); sinon au moins un (OR)
        
    Returns:
        Array de booléens indiquant si les blocs sont vrais pour chaque point
    """
    n_blocks, n_points = block_results.shape
    
    if n_blocks == 0:
        # Pas de blocs, retourner tout False
        return np.zeros(n_points, dtype=np.bool_)
    
    if n_blocks == 1:
        # Un seul bloc, retourner directement son résultat
        return block_results[0]
    
    # Initialiser le résultat
    result = np.zeros(n_points, dtype=np.bool_)
    
    if require_all:
        # Tous les blocs doivent être vrais (AND)
        result.fill(True)  # Commencer avec tout True
        for i in range(n_blocks):
            result = result & block_results[i]
    else:
        # Au moins un bloc doit être vrai (OR)
        result.fill(False)  # Commencer avec tout False
        for i in range(n_blocks):
            result = result | block_results[i]
    
    return result


@njit(cache=True)
def generate_signals(
    entry_results: np.ndarray,
    exit_results: np.ndarray,
    filter_results: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Génère les signaux de trading à partir des résultats des blocs.
    
    Args:
        entry_results: Array des résultats d'entrée
        exit_results: Array des résultats de sortie
        filter_results: Array des résultats de filtre (optionnel)
        
    Returns:
        Array des signaux (1=entrée, -1=sortie, 0=neutre)
    """
    n_points = len(entry_results)
    signals = np.zeros(n_points, dtype=np.int32)
    
    # Appliquer les filtres si présents
    if filter_results is not None:
        valid_points = filter_results
    else:
        valid_points = np.ones(n_points, dtype=np.bool_)
    
    # Générer les signaux
    for i in range(n_points):
        if not valid_points[i]:
            signals[i] = 0  # Point filtré
        elif entry_results[i]:
            signals[i] = 1  # Signal d'entrée
        elif exit_results[i]:
            signals[i] = -1  # Signal de sortie
        else:
            signals[i] = 0  # Pas de signal
    
    return signals



class ConditionEvaluator:
    """
    Évaluateur optimisé de conditions et de blocs pour générer des signaux.
    """
    
    def __init__(self, blocks_config: StrategyBlocksConfig = None):
        """
        Initialise l'évaluateur de conditions.
        
        Args:
            blocks_config: Configuration des blocs de stratégie (optionnel)
        """
        self.blocks_config = blocks_config or StrategyBlocksConfig()
        self.data_cache = {}  # Cache pour éviter de recalculer les mêmes arrays
    
    def evaluate_strategy(self, data: pd.DataFrame) -> np.ndarray:
        """
        Évalue la stratégie complète et génère les signaux.
        
        Args:
            data: DataFrame avec les données (prix et indicateurs)
            
        Returns:
            Array des signaux (1=entrée, -1=sortie, 0=neutre)
        """
        start_time = time()
        
        # Effacer le cache de données
        self.data_cache = {}
        
        # Vérifier si le DataFrame a un index de temps
        has_time_index = isinstance(data.index, pd.DatetimeIndex)
        
        # Précharger les données de prix
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                self.data_cache[col] = data[col].values.astype(np.float64)
        
        # Précharger tous les indicateurs
        for col in data.columns:
            if col not in price_columns:
                self.data_cache[col] = data[col].values.astype(np.float64)
        
        # Évaluer les blocs d'entrée
        entry_blocks_results = self._evaluate_blocks(self.blocks_config.entry_blocks, data)
        entry_results = evaluate_blocks(
            entry_blocks_results,
            self.blocks_config.require_all_entry_blocks
        )
        
        # Évaluer les blocs de sortie
        exit_blocks_results = self._evaluate_blocks(self.blocks_config.exit_blocks, data)
        exit_results = evaluate_blocks(
            exit_blocks_results,
            self.blocks_config.require_all_exit_blocks
        )
        
        # Évaluer les blocs de filtre
        filter_results = None
        if self.blocks_config.filter_blocks:
            filter_blocks_results = self._evaluate_blocks(self.blocks_config.filter_blocks, data)
            filter_results = evaluate_blocks(
                filter_blocks_results,
                self.blocks_config.require_all_filter_blocks
            )
        
        # Générer les signaux
        signals = generate_signals(entry_results, exit_results, filter_results)
        
        logger.debug(f"Évaluation de la stratégie en {time() - start_time:.3f}s")
        
        return signals
    
    def _evaluate_blocks(self, blocks: List[BlockConfig], data: pd.DataFrame) -> np.ndarray:
        """
        Évalue une liste de blocs de conditions.
        
        Args:
            blocks: Liste des blocs à évaluer
            data: DataFrame avec les données
            
        Returns:
            Array 2D des résultats de blocs [n_blocks, n_points]
        """
        n_blocks = len(blocks)
        n_points = len(data)
        
        if n_blocks == 0:
            return np.zeros((0, n_points), dtype=np.bool_)
        
        # Array pour stocker les résultats des blocs
        block_results = np.zeros((n_blocks, n_points), dtype=np.bool_)
        
        # Évaluer chaque bloc
        for i, block in enumerate(blocks):
            block_results[i] = self._evaluate_block(block, data)
        
        return block_results
    
    def _evaluate_block(self, block: BlockConfig, data: pd.DataFrame) -> np.ndarray:
        """
        Évalue un bloc de conditions.
        
        Args:
            block: Bloc à évaluer
            data: DataFrame avec les données
            
        Returns:
            Array des résultats du bloc
        """
        n_conditions = len(block.conditions)
        n_points = len(data)
        
        if n_conditions == 0:
            return np.zeros(n_points, dtype=np.bool_)
        
        # Array pour stocker les résultats des conditions
        condition_results = np.zeros((n_conditions, n_points), dtype=np.bool_)
        
        # Array pour stocker les opérateurs logiques
        logic_operators = np.zeros(len(block.logic_operators), dtype=np.int32)
        for i, op in enumerate(block.logic_operators):
            logic_operators[i] = 1 if op == LogicOperatorType.OR else 0
        
        # Évaluer chaque condition
        for i, condition in enumerate(block.conditions):
            condition_results[i] = self._evaluate_condition(condition, data)
        
        # Combiner les résultats des conditions
        return combine_conditions(condition_results, logic_operators)
    
    def _evaluate_condition(self, condition: ConditionConfig, data: pd.DataFrame) -> np.ndarray:
        """
        Évalue une condition individuelle.
        
        Args:
            condition: Condition à évaluer
            data: DataFrame avec les données
            
        Returns:
            Array des résultats de la condition
        """
        left_array = self._get_operand_array(condition.left_operand, data)
        
        # Convertir l'opérateur en code numérique
        operator_code = self._operator_to_code(condition.operator)
        
        # Évaluer selon le type d'opérande droit
        if condition.right_operand is not None:
            right_array = self._get_operand_array(condition.right_operand, data)
            return evaluate_condition_array(left_array, right_array, operator_code, condition.lookback)
        else:
            # Par défaut, valeur 0 si pas d'opérande droit
            value = 0.0
            # Si ValueOperand était utilisé, on aurait une valeur fixe
            # C'est un cas qui ne devrait pas arriver avec la vérification dans __post_init__
            return evaluate_condition_value(left_array, value, operator_code, condition.lookback)
    
    def _get_operand_array(self, operand: Union[PriceOperand, IndicatorOperand, ValueOperand], data: pd.DataFrame) -> np.ndarray:
        """
        Récupère l'array de valeurs correspondant à un opérande.
        
        Args:
            operand: Opérande (price, indicator, value)
            data: DataFrame avec les données
            
        Returns:
            Array des valeurs de l'opérande
        """
        if isinstance(operand, PriceOperand):
            # Récupérer l'array du prix depuis le cache ou le DataFrame
            price_type = operand.price_type
            
            if price_type in self.data_cache:
                return self.data_cache[price_type]
            
            if price_type in data.columns:
                array = data[price_type].values.astype(np.float64)
                self.data_cache[price_type] = array
                return array
            
            # Fallback sur close si le prix spécifié n'est pas disponible
            if 'close' in self.data_cache:
                return self.data_cache['close']
            
            if 'close' in data.columns:
                array = data['close'].values.astype(np.float64)
                self.data_cache['close'] = array
                return array
            
            # Si même close n'est pas disponible, utiliser la première colonne numérique
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    array = data[col].values.astype(np.float64)
                    self.data_cache[col] = array
                    return array
            
            # Si aucune donnée appropriée, retourner un array de zéros
            return np.zeros(len(data), dtype=np.float64)
            
        elif isinstance(operand, IndicatorOperand):
            # Pour les indicateurs, utiliser le nom complet
            indicator_name = operand.get_full_name()
            
            if indicator_name in self.data_cache:
                return self.data_cache[indicator_name]
            
            if indicator_name in data.columns:
                array = data[indicator_name].values.astype(np.float64)
                self.data_cache[indicator_name] = array
                return array
            
            # Si l'indicateur n'est pas disponible directement, chercher une correspondance partielle
            for col in data.columns:
                if indicator_name in col:
                    array = data[col].values.astype(np.float64)
                    self.data_cache[indicator_name] = array
                    return array
            
            # Si l'indicateur n'est toujours pas trouvé, retourner un array de zéros
            logger.warning(f"Indicateur {indicator_name} non trouvé dans les données")
            return np.zeros(len(data), dtype=np.float64)
            
        elif isinstance(operand, ValueOperand):
            # Pour une valeur fixe, créer un array constant
            return np.full(len(data), operand.value, dtype=np.float64)
            
        else:
            # Type non pris en charge, retourner un array de zéros
            logger.error(f"Type d'opérande non pris en charge: {type(operand)}")
            return np.zeros(len(data), dtype=np.float64)
    
    def _operator_to_code(self, operator: OperatorType) -> int:
        """
        Convertit un opérateur en code numérique pour les fonctions Numba.
        
        Args:
            operator: Opérateur à convertir
            
        Returns:
            int: Code numérique de l'opérateur
        """
        if operator == OperatorType.GREATER:
            return 0
        elif operator == OperatorType.LESS:
            return 1
        elif operator == OperatorType.GREATER_EQUAL:
            return 2
        elif operator == OperatorType.LESS_EQUAL:
            return 3
        elif operator == OperatorType.EQUAL:
            return 4
        elif operator == OperatorType.CROSS_ABOVE:
            return 5
        elif operator == OperatorType.CROSS_BELOW:
            return 6
        else:
            logger.warning(f"Opérateur non reconnu: {operator}, utilisation de GREATER par défaut")
            return 0
    
    def update_blocks_config(self, blocks_config: StrategyBlocksConfig):
        """
        Met à jour la configuration des blocs.
        
        Args:
            blocks_config: Nouvelle configuration
        """
        self.blocks_config = blocks_config
        self.data_cache = {}  # Effacer le cache


# ======= Fonctions utilitaires pour les tests =======
def test_condition_evaluator(blocks_config: StrategyBlocksConfig, data: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Teste l'évaluateur de conditions sur des données fournies.
    
    Args:
        blocks_config: Configuration des blocs
        data: DataFrame avec les données
        
    Returns:
        Tuple(np.ndarray, pd.DataFrame): Signaux générés et DataFrame avec les signaux ajoutés
    """
    evaluator = ConditionEvaluator(blocks_config)
    signals = evaluator.evaluate_strategy(data)
    
    # Ajouter les signaux au DataFrame
    result = data.copy()
    result['signal'] = signals
    
    # Ajouter des colonnes utiles pour l'analyse
    result['entry'] = np.where(signals == 1, 1, 0)
    result['exit'] = np.where(signals == -1, 1, 0)
    
    return signals, result


def plot_signals(data: pd.DataFrame, title: str = "Signaux de trading"):
    """
    Affiche un graphique des signaux de trading.
    
    Args:
        data: DataFrame avec les colonnes 'close' et 'signal'
        title: Titre du graphique
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Tracer le prix
        ax.plot(data.index, data['close'], label='Prix', color='blue', alpha=0.6)
        
        # Tracer les signaux d'entrée et de sortie
        entries = data[data['signal'] == 1].index
        exits = data[data['signal'] == -1].index
        
        ax.scatter(entries, data.loc[entries, 'close'], marker='^', color='green', s=100, label='Entrée')
        ax.scatter(exits, data.loc[exits, 'close'], marker='v', color='red', s=100, label='Sortie')
        
        # Ajouter les détails du graphique
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Prix')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format de date
        if isinstance(data.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib non disponible, impossible d'afficher le graphique")