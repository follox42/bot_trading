"""
Module de génération de signaux optimisé pour le backtesting de stratégies de trading.
Séparation claire entre la génération des signaux et le calcul des indicateurs techniques.
"""
import numpy as np
from numba import njit, prange
from typing import Dict, List, Tuple, Optional, Union, Set
import logging
from dataclasses import dataclass, field
from enum import Enum
import time

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('signal_generator.log', mode='a'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('signal_generator')

# ===== Définition des structures de données =====
class IndicatorType(Enum):
    """Types d'indicateurs techniques disponibles"""
    EMA = "EMA"
    SMA = "SMA"
    RSI = "RSI"
    ATR = "ATR"
    MACD = "MACD"
    BOLL = "BOLL"
    STOCH = "STOCH"
    VWAP = "VWAP"
    MFIMACD = "MFIMACD"

class Operator(Enum):
    """Types d'opérateurs de comparaison disponibles"""
    GREATER = ">"
    LESS = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    CROSS_ABOVE = "CROSS_ABOVE"
    CROSS_BELOW = "CROSS_BELOW"

class LogicOperator(Enum):
    """Types d'opérateurs logiques"""
    AND = "and"
    OR = "or"

@dataclass
class Condition:
    """Condition de trading unique"""
    indicator1: str  # Nom du premier indicateur (ex: "EMA_10")
    operator: Operator  # Opérateur de comparaison
    indicator2: Optional[str] = None  # Nom du second indicateur (optionnel)
    value: Optional[float] = None  # Valeur fixe pour comparaison (optionnel)

@dataclass
class Block:
    """Bloc de conditions de trading avec validation et fonctionnalités étendues"""
    conditions: List[Condition]
    logic_operators: List[LogicOperator]
    
    def __post_init__(self):
        """Validation après initialisation"""
        # Validation du nombre d'opérateurs logiques
        if len(self.conditions) > 1 and len(self.logic_operators) != len(self.conditions) - 1:
            raise ValueError(
                f"Nombre incorrect d'opérateurs logiques. Attendu {len(self.conditions) - 1}, "
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
            if index < len(self.logic_operators):
                self.logic_operators.pop(index)
            elif len(self.logic_operators) > 0:
                self.logic_operators.pop(-1)
    
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

@dataclass
class IndicatorConfig:
    """Configuration d'un indicateur technique"""
    type: IndicatorType
    min_period: int
    max_period: int
    step: int = 1
    price_type: str = "close"

# ===== Fonctions Numba optimisées pour indicateurs techniques =====
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
    
    # Première valeur calculable à partir de l'indice period-1
    for i in prange(period-1, len(prices)):
        sma[i] = np.mean(prices[i-period+1:i+1])
    
    return sma

@njit(cache=True, fastmath=True)
def calculate_rsi(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calcul optimisé du RSI avec précision double.
    """
    deltas = np.zeros_like(prices, dtype=np.float64)
    deltas[1:] = np.diff(prices)
    
    gains = np.zeros_like(deltas)
    losses = np.zeros_like(deltas)
    
    gains[deltas > 0] = deltas[deltas > 0]
    losses[deltas < 0] = -deltas[deltas < 0]
    
    avg_gain = np.zeros_like(prices, dtype=np.float64)
    avg_loss = np.zeros_like(prices, dtype=np.float64)
    
    # Calcul de la moyenne initiale
    if period <= len(prices):
        avg_gain[period] = np.mean(gains[1:period+1])
        avg_loss[period] = np.mean(losses[1:period+1])
    
    # Calcul du RSI par une moyenne mobile exponentielle
    for i in prange(period+1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i]) / period
    
    # Calcul du ratio RS
    rs = np.zeros_like(prices, dtype=np.float64)
    for i in prange(period, len(prices)):
        if avg_loss[i] == 0:
            rs[i] = 100.0  # Éviter division par zéro
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    # Calcul du RSI
    rsi = np.zeros_like(prices, dtype=np.float64)
    for i in prange(period, len(prices)):
        rsi[i] = 100.0 - (100.0 / (1.0 + rs[i]))
    
    return rsi

@njit(cache=True)
def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Calcul optimisé de l'ATR avec précision double.
    """
    tr = np.zeros_like(high, dtype=np.float64)
    atr = np.zeros_like(high, dtype=np.float64)
    
    # Calcul du True Range
    for i in prange(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    # Calcul de l'ATR initial
    if period <= len(high):
        atr[period] = np.mean(tr[1:period+1])
    
    # Calcul de l'ATR par une moyenne mobile exponentielle
    for i in prange(period+1, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

@njit(cache=True)
def parallel_indicator_calculation(prices: np.ndarray, high: np.ndarray, low: np.ndarray, 
                                 volumes: Optional[np.ndarray], indicator_configs):
    """
    Calcul parallèle des indicateurs avec précision double.
    
    Args:
        prices: Array des prix de clôture
        high: Array des prix hauts
        low: Array des prix bas
        volumes: Array des volumes (optionnel)
        indicator_configs: Liste de tuples (nom_indicateur, période, ...)
    
    Returns:
        Tuple contenant (indicators_array, indicator_names)
    """
    max_indicators = 100  # Augmenté pour supporter plus d'indicateurs
    indicators = np.zeros((max_indicators, len(prices)), dtype=np.float64)
    indicator_names = []
    
    current_idx = 0
    
    # Vérifier si volumes est None et créer un array de placeholder si nécessaire
    if volumes is None:
        volumes = np.ones_like(prices)
    
    for i in prange(len(indicator_configs)):
        ind_type = indicator_configs[i][0]
        period = indicator_configs[i][1]
        
        if current_idx >= max_indicators:
            break
            
        # === EMA ===
        if ind_type == 'EMA':
            indicators[current_idx] = calculate_ema(prices, period)
            indicator_names.append(f"EMA_{period}")
            current_idx += 1
            
        # === SMA ===
        elif ind_type == 'SMA':
            indicators[current_idx] = calculate_sma(prices, period)
            indicator_names.append(f"SMA_{period}")
            current_idx += 1
            
        # === RSI ===
        elif ind_type == 'RSI':
            indicators[current_idx] = calculate_rsi(prices, period)
            indicator_names.append(f"RSI_{period}")
            current_idx += 1
            
        # === ATR ===
        elif ind_type == 'ATR':
            if len(high) > 0 and len(low) > 0:
                indicators[current_idx] = calculate_atr(high, low, prices, period)
                indicator_names.append(f"ATR_{period}")
                current_idx += 1
    
    return indicators[:current_idx], indicator_names

# ===== Fonctions indépendantes pour Numba =====
@njit(cache=True)
def _operator_to_code(operator_value: str) -> int:
    """
    Convertit un opérateur en code numérique
    
    Args:
        operator_value: Valeur de l'opérateur (chaîne)
        
    Returns:
        int: Code de l'opérateur
    """
    # Mapping des opérateurs vers des codes numériques
    if operator_value == ">":         # GREATER
        return 0
    elif operator_value == "<":       # LESS
        return 1
    elif operator_value == ">=":      # GREATER_EQUAL
        return 2
    elif operator_value == "<=":      # LESS_EQUAL
        return 3
    elif operator_value == "==":      # EQUAL
        return 4
    elif operator_value == "CROSS_ABOVE":
        return 5
    elif operator_value == "CROSS_BELOW":
        return 6
    else:
        # Valeur par défaut
        return 0

@njit(cache=True)
def _generate_signals_fast(indicators_array, buy_blocks, sell_blocks) -> np.ndarray:
    """
    Version optimisée de la génération de signaux
    
    Args:
        indicators_array: Array des indicateurs calculés
        buy_blocks: Liste des blocs d'achat compilés
        sell_blocks: Liste des blocs de vente compilés
        
    Returns:
        Array des signaux (1 pour achat, -1 pour vente, 0 pour neutre)
    """
    if indicators_array is None:
        raise ValueError("Les indicateurs doivent être calculés avant de générer les signaux")
    
    data_length = indicators_array.shape[1]
    signals = np.zeros(data_length, dtype=np.int32)
    
    # Traiter les blocs d'achat
    if buy_blocks:
        # Array pour stocker les résultats de chaque bloc d'achat
        buy_block_results = np.zeros((len(buy_blocks), data_length), dtype=np.bool_)
        
        # Traiter chaque bloc d'achat
        for block_idx, block in enumerate(buy_blocks):
            # Commencer par évaluer la première condition du bloc
            block_result = np.ones(data_length, dtype=np.bool_)
            
            # Traiter chaque condition dans le bloc
            for cond_idx in range(len(block)):
                ind1_idx = int(block[cond_idx, 0])
                op_code = int(block[cond_idx, 1])
                ind2_idx = int(block[cond_idx, 2])
                value = block[cond_idx, 3]
                
                # Obtenir les valeurs des indicateurs
                ind1 = indicators_array[ind1_idx]
                
                # Évaluer la condition selon son type
                condition_result = np.zeros(data_length, dtype=np.bool_)
                
                if ind2_idx >= 0:
                    # Comparaison entre indicateurs
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
                    # Comparaison avec valeur fixe
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
                
                # Première condition ou combiner avec le résultat précédent selon logic_next
                if cond_idx == 0:
                    block_result = condition_result
                else:
                    # Obtenir l'opérateur logique de la condition précédente
                    logic_operator = int(block[cond_idx-1, 4])
                    if logic_operator == 0:  # AND
                        block_result = block_result & condition_result
                    else:  # OR (logic_operator == 1)
                        block_result = block_result | condition_result
            
            # Stocker le résultat du bloc
            buy_block_results[block_idx] = block_result
        
        # Combiner tous les blocs d'achat (OR entre les blocs)
        buy_signal = np.zeros(data_length, dtype=np.bool_)
        for i in range(len(buy_blocks)):
            buy_signal = buy_signal | buy_block_results[i]
        
        # Appliquer les signaux d'achat
        signals[buy_signal] = 1
    
    # Traiter les blocs de vente (logique similaire)
    if sell_blocks:
        # Array pour stocker les résultats de chaque bloc de vente
        sell_block_results = np.zeros((len(sell_blocks), data_length), dtype=np.bool_)
        
        # Traiter chaque bloc de vente
        for block_idx, block in enumerate(sell_blocks):
            # Commencer par évaluer la première condition du bloc
            block_result = np.ones(data_length, dtype=np.bool_)
            
            # Traiter chaque condition dans le bloc
            for cond_idx in range(len(block)):
                ind1_idx = int(block[cond_idx, 0])
                op_code = int(block[cond_idx, 1])
                ind2_idx = int(block[cond_idx, 2])
                value = block[cond_idx, 3]
                
                # Obtenir les valeurs des indicateurs
                ind1 = indicators_array[ind1_idx]
                
                # Évaluer la condition selon son type
                condition_result = np.zeros(data_length, dtype=np.bool_)
                
                if ind2_idx >= 0:
                    # Comparaison entre indicateurs
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
                    # Comparaison avec valeur fixe
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
                
                # Première condition ou combiner avec le résultat précédent selon logic_next
                if cond_idx == 0:
                    block_result = condition_result
                else:
                    # Obtenir l'opérateur logique de la condition précédente
                    logic_operator = int(block[cond_idx-1, 4])
                    if logic_operator == 0:  # AND
                        block_result = block_result & condition_result
                    else:  # OR (logic_operator == 1)
                        block_result = block_result | condition_result
            
            # Stocker le résultat du bloc
            sell_block_results[block_idx] = block_result
        
        # Combiner tous les blocs de vente (OR entre les blocs)
        sell_signal = np.zeros(data_length, dtype=np.bool_)
        for i in range(len(sell_blocks)):
            sell_signal = sell_signal | sell_block_results[i]
        
        # Appliquer les signaux de vente, mais ne pas écraser les signaux d'achat
        # Appliquer les signaux de vente seulement où il n'y a pas de signaux d'achat
        sell_mask = sell_signal & ~(signals == 1)
        signals[sell_mask] = -1
    
    return signals

# ===== Classe génératrice de signaux =====
class SignalGenerator:
    """
    Classe optimisée pour la génération de signaux de trading
    basée sur des conditions et des blocs.
    """
    
    def __init__(self):
        """
        Initialise le générateur de signaux.
        """
        self.indicators_array = None
        self.indicator_map = {}
        self.buy_blocks = []
        self.sell_blocks = []
        
        # Logger pour tracer l'exécution
        self.logger = logger
    
    def add_block(self, block: Block, is_buy: bool = True):
        """
        Ajoute un bloc de trading.
        
        Args:
            block: Bloc de conditions à ajouter
            is_buy: True pour un bloc d'achat, False pour un bloc de vente
        """
        if is_buy:
            self.buy_blocks.append(block)
        else:
            self.sell_blocks.append(block)
    
    def prepare_indicators_config(self, indicator_params):
        """
        Convertit les paramètres d'indicateurs en tableau de configuration
        utilisable par les fonctions de calcul parallèle.
        
        Args:
            indicator_params: Liste de tuples (nom_indicateur, période)
            
        Returns:
            Liste de tuples (type_indicateur, période) pour parallel_indicator_calculation
        """
        required_indicators = []
        seen_indicators = set()
        
        # Parcourir les indicateurs requis et les ajouter à la liste
        for ind_name, period in indicator_params:
            if (ind_name, period) not in seen_indicators:
                required_indicators.append((ind_name, period))
                seen_indicators.add((ind_name, period))
        
        # Si aucun indicateur trouvé, ajouter les indicateurs par défaut
        if not required_indicators:
            default_indicators = [('EMA', 10), ('SMA', 20), ('RSI', 14)]
            required_indicators.extend(default_indicators)
        
        return required_indicators

    def calculate_indicators(self, prices: np.ndarray, high: Optional[np.ndarray] = None,
                           low: Optional[np.ndarray] = None, volumes: Optional[np.ndarray] = None,
                           indicator_params=None) -> None:
        """
        Calcul optimisé des indicateurs
        
        Args:
            prices: Array des prix
            high: Array des prix hauts (optionnel)
            low: Array des prix bas (optionnel)
            volumes: Array des volumes (optionnel)
            indicator_params: Liste de tuples (nom_indicateur, période)
        """
        start_time = time.time()
        
        try:
            # Réinitialisation du dictionnaire de mapping
            self.indicator_map = {}
            
            # Préparer la configuration des indicateurs requis
            if indicator_params is None:
                # Collecter les indicateurs requis par les blocs
                indicator_params = []
                for block in self.buy_blocks + self.sell_blocks:
                    for condition in block.conditions:
                        if '_' in condition.indicator1:
                            ind_type, period_str = condition.indicator1.split('_')
                            try:
                                period = int(period_str)
                                indicator_params.append((ind_type, period))
                            except ValueError:
                                continue
                        
                        if condition.indicator2 and '_' in condition.indicator2:
                            ind_type, period_str = condition.indicator2.split('_')
                            try:
                                period = int(period_str)
                                indicator_params.append((ind_type, period))
                            except ValueError:
                                continue
            
            indicator_configs = self.prepare_indicators_config(indicator_params)
            
            # Vérifier si on a des données
            if len(prices) == 0:
                raise ValueError("Aucune donnée de prix fournie pour le calcul des indicateurs")

            # Conversion explicite en tableaux NumPy
            prices_array = np.ascontiguousarray(prices)
            high_array = np.ascontiguousarray(high) if high is not None else np.array([], dtype=prices.dtype)
            low_array = np.ascontiguousarray(low) if low is not None else np.array([], dtype=prices.dtype)
            volumes_array = np.ascontiguousarray(volumes) if volumes is not None else None

            # Calcul parallèle uniquement des indicateurs requis
            self.indicators_array, indicator_names = parallel_indicator_calculation(
                prices_array, high_array, low_array, volumes_array, indicator_configs
            )

            # Mise à jour du mapping
            for idx, name in enumerate(indicator_names):
                self.indicator_map[name] = idx

            self.logger.debug(f"Calcul effectué pour {len(indicator_names)} indicateurs en {time.time() - start_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Erreur dans le calcul des indicateurs: {str(e)}")
            raise

    def cleanup(self):
        """Libère la mémoire"""
        if hasattr(self, 'indicators_array'):
            del self.indicators_array
        self.indicators_array = None
        self.indicator_map.clear()
  
    def compile_blocks(self, blocks) -> List[np.ndarray]:
        """
        Compile les blocs en une structure simple et optimisée.
        Chaque bloc est représenté par un tableau distinct contenant ses conditions.
        
        Args:
            blocks: Liste des blocs à compiler
            
        Returns:
            Liste de tableaux numpy représentant les blocs
        """
        compiled_blocks = []
        
        # Traitement de chaque bloc
        for block in blocks:
            # Création d'un tableau pour les conditions du bloc
            # Format: [ind1_idx, op_code, ind2_idx, value, logic_next]
            block_conditions = np.zeros((len(block.conditions), 5), dtype=np.float64)
            
            for j, condition in enumerate(block.conditions):
                # Indices des indicateurs
                ind1_idx = self.indicator_map.get(condition.indicator1, -1)
                ind2_idx = self.indicator_map.get(condition.indicator2, -1) if condition.indicator2 else -1
                
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
                        low: Optional[np.ndarray] = None, volumes: Optional[np.ndarray] = None,
                        indicator_params=None) -> np.ndarray:
        """
        Génère les signaux de trading de manière optimisée
        
        Args:
            prices: Array des prix
            high: Array des prix hauts (optionnel)
            low: Array des prix bas (optionnel)
            volumes: Array des volumes (optionnel)
            indicator_params: Liste de tuples (nom_indicateur, période)
            
        Returns:
            Array des signaux (1 pour achat, -1 pour vente, 0 pour neutre)
        """
        start_time = time.time()
        
        # Calcul des indicateurs si nécessaire
        if self.indicators_array is None:
            self.calculate_indicators(prices, high, low, volumes, indicator_params)
        
        # Vérifier si nous avons des blocs de trading
        if not self.buy_blocks and not self.sell_blocks:
            # Pas de blocs définis, retourner un tableau de zéros
            self.logger.warning("Aucun bloc de trading défini. Génération de signaux nuls.")
            return np.zeros(len(prices), dtype=np.int32)
        
        # Compilation des conditions
        buy_blocks = self.compile_blocks(self.buy_blocks)
        sell_blocks = self.compile_blocks(self.sell_blocks)
        
        # Vérifier si les blocs compilés sont vides
        if not buy_blocks and not sell_blocks:
            # Blocs vides après compilation, retourner un tableau de zéros
            self.logger.warning("Blocs de trading vides après compilation. Génération de signaux nuls.")
            return np.zeros(len(prices), dtype=np.int32)

        # Génération des signaux via la fonction indépendante Numba
        signals = _generate_signals_fast(self.indicators_array, buy_blocks, sell_blocks)
        
        self.logger.debug(f"Génération de {len(signals)} signaux en {time.time() - start_time:.3f}s")
        
        return signals
             
    def _operator_to_code(self, operator: Operator) -> int:
        """Convertit un opérateur en code numérique"""
        # Appel à la fonction indépendante
        return _operator_to_code(operator.value)