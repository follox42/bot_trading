"""
Module d'optimisation des stratégies de trading avec flexibilité améliorée.
Supporte différentes méthodes d'optimisation, modèles de scoring personnalisables,
et une meilleure gestion des ressources système.
"""
import os
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler, NSGAIISampler
from optuna.pruners import MedianPruner, PercentilePruner, HyperbandPruner
import json
import time
import gc
import traceback
import logging
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Manager, shared_memory
import concurrent.futures
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable
import warnings
import psutil
import random
import math
from functools import partial
import pickle

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy_optimizer.log', mode='a'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('strategy_optimizer')

# Importation des modules personnalisés
from simulator.indicators import SignalGenerator, Block, Condition, Operator, LogicOperator
from simulator.risk import PositionCalculator, RiskMode
from simulator.simulator import Simulator, SimulationConfig
from simulator.config import (
    OptimizationConfig, StudyStatus, ScoringFormula, OptimizationMethod, PrunerMethod,
    ScoringWeights, create_default_optimization_config
)
from simulator.study_config_definitions import (
    OPTIMIZATION_METHODS, PRUNER_METHODS, SCORING_FORMULAS, AVAILABLE_METRICS
)

# Initialisation des variables globales pour le multiprocessing
_shared_data = None
_trading_config = None
_optimization_progress = {}

def init_worker(shared_data_info, trading_config_dict):
    """
    Initialise un worker de multiprocessing.
    
    Args:
        shared_data_info: Informations sur les données partagées
        trading_config_dict: Configuration de trading au format dict
    """
    global _shared_data, _trading_config
    try:
        _shared_data = shared_data_info
        
        # Deserialize the trading_config_dict if it's a string
        if isinstance(trading_config_dict, str):
            trading_config_dict = json.loads(trading_config_dict)
            
        from simulator.config import TradingConfig
        _trading_config = TradingConfig.from_dict(trading_config_dict)
        
        # Check if initialization was successful
        if _shared_data is None or _trading_config is None:
            logger.error("Worker initialization failed - data not properly set")
            
    except Exception as e:
        logger.error(f"Error in worker initialization: {str(e)}")
        traceback.print_exc()

class GroupingTrialWrapper:
    """
    Wrapper pour un trial Optuna qui gère l'enregistrement des groupes de paramètres.
    Cela permet d'éviter l'échantillonnage indépendant et améliore la cohérence des paramètres.
    """
    
    def __init__(self, trial, register_group_func):
        """
        Initialise le wrapper.
        
        Args:
            trial: Trial Optuna original
            register_group_func: Fonction pour enregistrer un paramètre dans un groupe
        """
        self.trial = trial
        self.register_group_func = register_group_func
        self.param_history = {}
    
    def suggest_categorical(self, name, choices, group_name=None):
        """Wrapper pour suggest_categorical avec gestion des groupes"""
        # Register with group if available
        if hasattr(self.trial, "register_param_group") and callable(self.register_group_func):
            group = self.register_group_func(name, group_name)
            self.trial.register_param_group(name, group)
        
        # Store parameter history
        self.param_history[name] = {"type": "categorical", "choices": choices}
        
        return self.trial.suggest_categorical(name, choices)
    
    def suggest_int(self, name, low, high, step=1, log=False, group_name=None):
        """Wrapper pour suggest_int avec gestion des groupes"""
        # Register with group if available
        if hasattr(self.trial, "register_param_group") and callable(self.register_group_func):
            group = self.register_group_func(name, group_name)
            self.trial.register_param_group(name, group)
            
        # Store parameter history
        self.param_history[name] = {"type": "int", "low": low, "high": high, "step": step, "log": log}
        
        return self.trial.suggest_int(name, low, high, step=step, log=log)
    
    def suggest_float(self, name, low, high, step=None, log=False, group_name=None):
        """Wrapper pour suggest_float avec gestion des groupes"""
        # Register with group if available
        if hasattr(self.trial, "register_param_group") and callable(self.register_group_func):
            group = self.register_group_func(name, group_name)
            self.trial.register_param_group(name, group)
            
        # Store parameter history
        self.param_history[name] = {"type": "float", "low": low, "high": high, "step": step, "log": log}
        
        return self.trial.suggest_float(name, low, high, step=step, log=log)

class BlockManager:
    """Manages the generation of trading blocks based on trial parameters"""
    
    def __init__(self, trial, trading_config):
        """
        Initialize the block manager with a trial and trading configuration.
        
        Args:
            trial: Optuna trial or GroupingTrialWrapper
            trading_config: Trading configuration
        """
        self.trial = trial
        self.trading_config = trading_config
        self.structure_config = trading_config.strategy_structure
        self.available_indicators = trading_config.available_indicators
    
    def generate_blocks(self):
        """
        Generates buy and sell blocks based on trial parameters
        
        Returns:
            Tuple[List[Block], List[Block]]: (buy_blocks, sell_blocks)
        """
        # Determine the number of blocks to generate
        min_blocks = self.structure_config.min_blocks
        max_blocks = self.structure_config.max_blocks
        
        n_buy_blocks = self.trial.suggest_int("n_buy_blocks", min_blocks, max_blocks)
        n_sell_blocks = self.trial.suggest_int("n_sell_blocks", min_blocks, max_blocks)
        
        buy_blocks = self._generate_block_set(n_buy_blocks, "buy")
        sell_blocks = self._generate_block_set(n_sell_blocks, "sell")
        
        return buy_blocks, sell_blocks
    
    def _generate_block_set(self, n_blocks, prefix):
        """
        Generates a set of blocks with the given prefix
        
        Args:
            n_blocks: Number of blocks to generate
            prefix: Prefix for parameter names ("buy" or "sell")
            
        Returns:
            List[Block]: List of generated blocks
        """
        blocks = []
        
        for i in range(n_blocks):
            block_prefix = f"{prefix}_block_{i}"
            block = self._generate_single_block(block_prefix)
            blocks.append(block)
        
        return blocks
    
    def _generate_single_block(self, block_prefix):
        """
        Generates a single trading block
        
        Args:
            block_prefix: Prefix for parameter names
            
        Returns:
            Block: Generated block
        """
        min_conditions = self.structure_config.min_conditions_per_block
        max_conditions = self.structure_config.max_conditions_per_block
        
        # Determine number of conditions
        n_conditions = self.trial.suggest_int(f"{block_prefix}_n_conditions", min_conditions, max_conditions)
        
        # Generate conditions
        conditions = []
        logic_operators = []
        
        for j in range(n_conditions):
            cond_prefix = f"{block_prefix}_cond_{j}"
            condition = self._generate_condition(cond_prefix)
            conditions.append(condition)
            
            # Add logic operator if needed
            if j < n_conditions - 1:
                logic_op = self._get_logic_operator(f"{block_prefix}_logic_{j}")
                logic_operators.append(logic_op)
        
        return Block(conditions=conditions, logic_operators=logic_operators)
    
    def _generate_condition(self, cond_prefix):
        """
        Generates a single trading condition
        
        Args:
            cond_prefix: Prefix for parameter names
            
        Returns:
            Condition: Generated condition
        """
        # Choose indicators from available ones
        available_inds = list(self.available_indicators.keys())
        if not available_inds:
            # Fallback to default indicators if none are available
            available_inds = ["EMA", "SMA", "RSI"]
        
        # First indicator
        ind1_type = self.trial.suggest_categorical(f"{cond_prefix}_ind1_type", available_inds)
        ind_config = self.available_indicators.get(ind1_type)
        
        # Use default values if config is missing
        if ind_config is None:
            min_period, max_period, step = 5, 50, 5
        else:
            min_period = ind_config.min_period
            max_period = ind_config.max_period
            step = ind_config.step
        
        period1 = self.trial.suggest_int(f"{cond_prefix}_period1", min_period, max_period, step=step)
        ind1 = f"{ind1_type}_{period1}"
        
        # Operator
        operators = [op.value for op in Operator]
        op = Operator(self.trial.suggest_categorical(f"{cond_prefix}_operator", operators))
        
        # Determine if we compare to another indicator or a value
        use_value = self.trial.suggest_float(f"{cond_prefix}_use_value", 0, 1) < self.structure_config.value_comparison_probability
        
        ind2 = None
        value = None
        
        if use_value:
            # Compare to a value
            if ind1_type == "RSI":
                # RSI values are between 0 and 100
                value = self.trial.suggest_float(f"{cond_prefix}_value", 
                                               self.structure_config.rsi_value_range[0],
                                               self.structure_config.rsi_value_range[1])
            else:
                # Use a multiplier range from structure config
                multiplier = self.trial.suggest_float(f"{cond_prefix}_multiplier", 
                                                   self.structure_config.general_value_range[0]/100,
                                                   self.structure_config.general_value_range[1]/100)
                value = multiplier  # Will be multiplied by price at runtime
        else:
            # Compare to another indicator
            ind2_type = self.trial.suggest_categorical(f"{cond_prefix}_ind2_type", available_inds)
            ind_config2 = self.available_indicators.get(ind2_type)
            
            # Use default values if config is missing
            if ind_config2 is None:
                min_period2, max_period2, step2 = 5, 50, 5
            else:
                min_period2 = ind_config2.min_period
                max_period2 = ind_config2.max_period
                step2 = ind_config2.step
            
            period2 = self.trial.suggest_int(f"{cond_prefix}_period2", min_period2, max_period2, step=step2)
            ind2 = f"{ind2_type}_{period2}"
        
        return Condition(indicator1=ind1, operator=op, indicator2=ind2, value=value)
    
    def _get_logic_operator(self, param_name):
        """
        Gets a logic operator based on the trial parameter
        
        Args:
            param_name: Parameter name
            
        Returns:
            LogicOperator: AND or OR
        """
        use_or = self.trial.suggest_categorical(param_name, [0, 1])
        return LogicOperator.OR if use_or else LogicOperator.AND

class RiskManager:
    """Manages risk parameters based on trial parameters"""
    
    def __init__(self, trial, trading_config):
        """
        Initialize the risk manager with a trial and trading configuration.
        
        Args:
            trial: Optuna trial or GroupingTrialWrapper
            trading_config: Trading configuration
        """
        self.trial = trial
        self.trading_config = trading_config
        self.risk_config = trading_config.risk_config
        
        # Choose risk mode from available modes
        available_modes = [mode.value for mode in self.risk_config.available_modes]
        if not available_modes:
            # Default if none are available
            available_modes = [RiskMode.FIXED.value]
            
        chosen_mode = self.trial.suggest_categorical("risk_mode", available_modes)
        self.risk_mode = RiskMode(chosen_mode)
    
    def get_config(self):
        """
        Gets risk configuration based on the chosen risk mode
        
        Returns:
            Dict: Configuration for PositionCalculator
        """
        config = {}
        
        # Base configuration common to all modes
        base_position_range = self.risk_config.position_size_range
        base_sl_range = self.risk_config.sl_range
        tp_mult_range = self.risk_config.tp_multiplier_range
        
        config["base_position"] = self.trial.suggest_float("base_position", base_position_range[0], base_position_range[1], log=True)
        config["base_sl"] = self.trial.suggest_float("base_sl", base_sl_range[0], base_sl_range[1], log=True)
        config["tp_multiplier"] = self.trial.suggest_float("tp_multiplier", tp_mult_range[0], tp_mult_range[1])
        
        # Mode-specific configuration
        if self.risk_mode == RiskMode.FIXED:
            mode_config = self.risk_config.mode_configs.get(RiskMode.FIXED)
            if mode_config:
                fixed_pos_range = mode_config.fixed_position_range
                fixed_sl_range = mode_config.fixed_sl_range
                fixed_tp_range = mode_config.fixed_tp_range
                
                config["base_position"] = self.trial.suggest_float("fixed_position", fixed_pos_range[0], fixed_pos_range[1], log=True)
                config["base_sl"] = self.trial.suggest_float("fixed_sl", fixed_sl_range[0], fixed_sl_range[1], log=True)
                config["tp_multiplier"] = self.trial.suggest_float("fixed_tp_mult", fixed_tp_range[0], fixed_tp_range[1])
        
        elif self.risk_mode == RiskMode.ATR_BASED:
            mode_config = self.risk_config.mode_configs.get(RiskMode.ATR_BASED)
            if mode_config:
                atr_period_range = mode_config.atr_period_range
                atr_mult_range = mode_config.atr_multiplier_range
                
                config["atr_period"] = self.trial.suggest_int("atr_period", atr_period_range[0], atr_period_range[1])
                config["atr_multiplier"] = self.trial.suggest_float("atr_multiplier", atr_mult_range[0], atr_mult_range[1])
        
        elif self.risk_mode == RiskMode.VOLATILITY_BASED:
            mode_config = self.risk_config.mode_configs.get(RiskMode.VOLATILITY_BASED)
            if mode_config:
                vol_period_range = mode_config.vol_period_range
                vol_mult_range = mode_config.vol_multiplier_range
                
                config["vol_period"] = self.trial.suggest_int("vol_period", vol_period_range[0], vol_period_range[1])
                config["vol_multiplier"] = self.trial.suggest_float("vol_multiplier", vol_mult_range[0], vol_mult_range[1])
        
        return config

class SimulationManager:
    """Manages simulation parameters based on trial parameters"""
    
    def __init__(self, trial, trading_config):
        """
        Initialize the simulation manager with a trial and trading configuration.
        
        Args:
            trial: Optuna trial or GroupingTrialWrapper
            trading_config: Trading configuration
        """
        self.trial = trial
        self.trading_config = trading_config
        self.sim_config = trading_config.sim_config
        
        # Choose leverage from available range
        leverage_range = self.sim_config.leverage_range
        self.leverage = self.trial.suggest_int("leverage", leverage_range[0], leverage_range[1], log=True)
        
        # Choose margin mode
        margin_modes = [mode.value for mode in self.sim_config.margin_modes]
        if not margin_modes:
            margin_modes = [0]  # Default to Isolated if none available
        self.margin_mode = self.trial.suggest_categorical("margin_mode", margin_modes)
        
        # Choose trading mode
        trading_modes = [mode.value for mode in self.sim_config.trading_modes]
        if not trading_modes:
            trading_modes = [0]  # Default to One-way if none available
        self.trading_mode = self.trial.suggest_categorical("trading_mode", trading_modes)
    
    def get_simulator_config(self):
        """
        Gets simulation configuration based on trial parameters
        
        Returns:
            SimulationConfig: Configuration for Simulator
        """
        # Initial balance range
        balance_range = self.sim_config.initial_balance_range
        initial_balance = self.trial.suggest_float(
            "initial_balance", 
            balance_range[0], 
            balance_range[1], 
            log=True
        )
        
        # Fee and slippage
        # Instead of optimizing these, use fixed values from config
        fee_open = self.sim_config.fee 
        fee_close = self.sim_config.fee
        slippage = self.sim_config.slippage
        
        return SimulationConfig(
            initial_balance=initial_balance,
            fee_open=fee_open,
            fee_close=fee_close,
            slippage=slippage,
            tick_size=self.sim_config.tick_size,
            min_trade_size=self.sim_config.min_trade_size,
            max_trade_size=self.sim_config.max_trade_size,
            leverage=self.leverage,
            margin_mode=self.margin_mode,
            trading_mode=self.trading_mode
        )

class ScoreCalculator:
    """Calcule et évalue les scores des stratégies"""

    def __init__(self, score_config=None):
        """
        Initialise le calculateur de score
        
        Args:
            score_config: Configuration du scoring
        """
        # Use default if none provided
        if score_config is None:
            score_config = SCORING_FORMULAS["standard"].copy()
            
        self.score_config = score_config
        self.metrics_info = AVAILABLE_METRICS
        
        # Validation des poids
        self._validate_weights()
        
        # Construction du transformateur de score
        self.transform_score = self.score_config.get("transformation", 
                                                   lambda x: x * 10.0)
    
    def _validate_weights(self):
        """Valide les poids et ajoute les poids manquants par défaut"""
        weights = self.score_config.get("weights", {})
        
        # Vérifier que tous les poids sont des métriques valides
        for metric, weight in list(weights.items()):
            if metric not in self.metrics_info:
                # Supprimer les métriques invalides
                logger.warning(f"Métrique inconnue dans la configuration de score: {metric}")
                del weights[metric]
        
        # S'assurer qu'il y a au moins une métrique
        if not weights:
            logger.warning("Aucun poids valide défini, utilisation des poids par défaut")
            weights = {
                "roi": 2.5,
                "win_rate": 0.5,
                "max_drawdown": 2.0,
                "profit_factor": 2.0,
                "total_trades": 1.0
            }
        
        self.score_config["weights"] = weights
    
    def calculate_score(self, metrics: Dict[str, float]) -> float:
        """
        Calcule un score pondéré à partir des métriques et des poids
        
        Args:
            metrics: Dictionnaire des métriques de performance
            
        Returns:
            float: Score final
        """
        # Validation initiale
        if metrics.get('total_trades', 0) == 0:
            return float('-inf')
        
        # Normalisation des métriques
        normalized_metrics = {}
        weights = self.score_config["weights"]
        
        for metric_name, metric_info in self.metrics_info.items():
            if metric_name in metrics and metric_name in weights:
                raw_value = metrics[metric_name]
                normalizer = metric_info.get("normalization", lambda x: x)
                
                # Normalisation
                normalized_value = normalizer(raw_value)
                normalized_metrics[metric_name] = normalized_value
        
        # Calcul du score final pondéré
        score = 0.0
        total_weight = sum(weights.values())
        
        if total_weight <= 0:
            logger.warning("Somme des poids nulle ou négative, utilisation de poids égaux")
            total_weight = len(weights)
            weights = {k: 1.0 for k in weights}
        
        for metric, weight in weights.items():
            if metric in normalized_metrics:
                score += normalized_metrics[metric] * (weight / total_weight)
        
        # Transformation finale pour ajuster l'échelle
        final_score = self.transform_score(score)
        
        return final_score
    
    @classmethod
    def get_available_formulas(cls) -> Dict[str, Dict]:
        """
        Retourne les formules de scoring disponibles
        
        Returns:
            Dict: Dictionnaire des formules de scoring
        """
        return SCORING_FORMULAS
    
    @classmethod
    def get_available_metrics(cls) -> Dict[str, Dict]:
        """
        Retourne les métriques disponibles pour le scoring
        
        Returns:
            Dict: Dictionnaire des métriques
        """
        return AVAILABLE_METRICS
    
    @classmethod
    def create_from_formula(cls, formula_name: str, custom_weights: Dict = None) -> 'ScoreCalculator':
        """
        Crée un calculateur de score à partir d'une formule prédéfinie
        
        Args:
            formula_name: Nom de la formule
            custom_weights: Poids personnalisés (pour la formule 'custom')
            
        Returns:
            ScoreCalculator: Instance configurée
        """
        if formula_name not in SCORING_FORMULAS:
            logger.warning(f"Formule {formula_name} non trouvée, utilisation de la formule standard")
            formula_name = "standard"
        
        formula = SCORING_FORMULAS[formula_name].copy()
        
        if formula_name == "custom" and custom_weights:
            formula["weights"] = custom_weights
        
        return cls(formula)

class ProcessManager:
    """Gère le multiprocessing et les ressources système"""
    
    def __init__(self, 
                n_jobs: int = -1, 
                backend: str = "multiprocessing",
                memory_limit: float = 0.8,
                cpu_priority: int = 0):
        """
        Initialise le gestionnaire de processus
        
        Args:
            n_jobs: Nombre de processus (-1 pour auto)
            backend: Backend de parallélisation ("multiprocessing" ou "threading")
            memory_limit: Limite de mémoire comme fraction de la mémoire totale
            cpu_priority: Priorité CPU (0=normal, -20=highest, 19=lowest) - Ignoré sur Windows
        """
        self.n_jobs = n_jobs
        self.backend = backend
        self.memory_limit = memory_limit
        
        import os
        import platform
        import psutil
        
        # Détecter le nombre de cœurs logiques
        self.logical_cores = os.cpu_count() or 1
        self.physical_cores = psutil.cpu_count(logical=False) or 1
        
        # Ajuster n_jobs si automatique
        if self.n_jobs < 1:
            self.n_jobs = max(1, self.logical_cores - 1)  # Default: tous les cœurs sauf 1
        
        # Limite supérieure pour n_jobs
        self.n_jobs = min(self.n_jobs, self.logical_cores)
        
        # Définir la priorité CPU (si possible) - Compatibilité multi-plateforme
        try:
            process = psutil.Process()
            system = platform.system()
            
            if system == "Windows":
                # Sur Windows, les valeurs sont différentes - utiliser les priorités de Windows
                priority_map = {
                    -20: psutil.HIGH_PRIORITY_CLASS,    # Haute priorité 
                    -10: psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                    0: psutil.NORMAL_PRIORITY_CLASS,    # Priorité normale
                    10: psutil.BELOW_NORMAL_PRIORITY_CLASS,
                    19: psutil.IDLE_PRIORITY_CLASS      # Basse priorité
                }
                
                # Trouver la priorité la plus proche dans le mapping
                windows_priority = None
                if cpu_priority <= -15:
                    windows_priority = psutil.HIGH_PRIORITY_CLASS
                elif cpu_priority <= -5:
                    windows_priority = psutil.ABOVE_NORMAL_PRIORITY_CLASS
                elif cpu_priority <= 5:
                    windows_priority = psutil.NORMAL_PRIORITY_CLASS
                elif cpu_priority <= 15:
                    windows_priority = psutil.BELOW_NORMAL_PRIORITY_CLASS
                else:
                    windows_priority = psutil.IDLE_PRIORITY_CLASS
                
                process.nice(windows_priority)
            else:
                # Sur Unix/Linux, utiliser nice normalement
                import os
                os.nice(cpu_priority)
        except (ImportError, AttributeError, PermissionError, OSError) as e:
            # Ne pas échouer si la fonction n'est pas disponible ou génère une erreur
            import logging
            logging.getLogger('strategy_optimizer').warning(f"Impossible d'ajuster la priorité CPU: {str(e)}")
    
    def create_executor(self):
        """
        Crée un executor pour le traitement parallèle
        
        Returns:
            concurrent.futures.Executor: L'executor créé
        """
        if self.backend == "threading":
            return concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs)
        else:  # Default to multiprocessing
            return concurrent.futures.ProcessPoolExecutor(
                max_workers=self.n_jobs,
                mp_context=mp.get_context('spawn')  # More stable than fork
            )
    
    def parallel_map(self, func: Callable, iterable: List, **kwargs) -> List:
        """
        Applique une fonction en parallèle sur un itérable
        
        Args:
            func: Fonction à appliquer
            iterable: Itérable d'entrées
            **kwargs: Arguments supplémentaires pour func
            
        Returns:
            List: Résultats de l'application parallèle
        """
        if self.n_jobs == 1:
            # Mode séquentiel
            return [func(item, **kwargs) for item in iterable]
        
        # Mode parallèle
        with self.create_executor() as executor:
            if kwargs:
                func = partial(func, **kwargs)
            return list(executor.map(func, iterable))
    
    def get_resource_usage(self) -> Dict[str, float]:
        """
        Retourne l'utilisation actuelle des ressources
        
        Returns:
            Dict: Informations sur l'utilisation des ressources
        """
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Mémoire
        memory = psutil.virtual_memory()
        memory_used_pct = memory.percent / 100.0
        
        return {
            "cpu_percent": cpu_percent,
            "memory_used_pct": memory_used_pct,
            "memory_available_gb": memory.available / (1024**3),
            "memory_total_gb": memory.total / (1024**3)
        }
    
    def should_throttle(self) -> bool:
        """
        Vérifie si le système doit ralentir en raison de la charge
        
        Returns:
            bool: True si le système est surchargé
        """
        resources = self.get_resource_usage()
        
        # Vérifier si la mémoire dépasse la limite
        if resources["memory_used_pct"] > self.memory_limit:
            return True
        
        # Autres critères (CPU, etc.)
        return False
    
    def adaptive_sleep(self) -> None:
        """Dort pendant une durée adaptative basée sur la charge"""
        if self.should_throttle():
            resource_usage = self.get_resource_usage()
            
            # Calcul d'un temps de sommeil adaptatif basé sur la charge mémoire
            memory_load = resource_usage["memory_used_pct"]
            excess_load = max(0, memory_load - self.memory_limit)
            
            # Temps de sommeil progressif
            sleep_time = min(30, 0.1 + (excess_load * 10))  # max 30 secondes
            
            logger.warning(f"Ressources système limitées (mémoire: {memory_load:.1%}), "
                         f"attente de {sleep_time:.1f}s")
            
            time.sleep(sleep_time)
            gc.collect()  # Forcer le garbage collector

class StrategyOptimizer:
    """
    Optimiseur de stratégies amélioré avec une meilleure gestion des paramètres et des performances.
    """
    
    def __init__(self, study_manager: 'IntegratedStudyManager'):
        """
        Initialise l'optimiseur de stratégies amélioré.
        
        Args:
            study_manager: Gestionnaire d'études
        """
        self.study_manager = study_manager
        self.config = create_default_optimization_config()
        self.shared_data = None
        self.score_calculator = None
        
        # Configuration du logger
        self.logger = logging.getLogger('strategy_optimizer')
        
        # Création du groupe de paramètres pour TPESampler
        self.param_groups = {}
        self.group_counter = 0
        
        # Process manager for resource handling
        self.process_manager = ProcessManager()
    
    def configure(self, config_dict: Dict) -> None:
        """
        Configure l'optimiseur avec les paramètres spécifiés
        
        Args:
            config_dict: Dictionnaire de configuration ou instance de OptimizationConfig
        """
        # If config_dict is a string, parse it as JSON
        if isinstance(config_dict, str):
            try:
                config_dict = json.loads(config_dict)
            except json.JSONDecodeError:
                logger.error("Invalid JSON configuration string provided")
                return
        
        # If it's an OptimizationConfig instance, extract the dict
        if isinstance(config_dict, OptimizationConfig):
            config_dict = config_dict.to_dict()
        
        # Paramètres de base
        if 'n_trials' in config_dict:
            self.config.n_trials = config_dict['n_trials']
        if 'timeout' in config_dict:
            self.config.timeout = config_dict['timeout']
        if 'gc_after_trial' in config_dict:
            self.config.gc_after_trial = config_dict['gc_after_trial']
        
        # Méthode d'optimisation
        if 'optimization_method' in config_dict:
            # Convert string to OptimizationMethod enum if needed
            if isinstance(config_dict['optimization_method'], str):
                try:
                    self.config.optimization_method = OptimizationMethod(config_dict['optimization_method'])
                except ValueError:
                    logger.warning(f"Invalid optimization method: {config_dict['optimization_method']}, using default")
            else:
                self.config.optimization_method = config_dict['optimization_method']
        
        if 'method_params' in config_dict:
            # Fusion tout en préservant nos paramètres spéciaux
            special_params = {
                k: v for k, v in self.config.method_params.items() 
                if k in ['group_related_params', 'consider_magic_clip']
            }
            self.config.method_params.update(config_dict['method_params'])
            self.config.method_params.update(special_params)
        
        # Pruning
        if 'enable_pruning' in config_dict:
            self.config.enable_pruning = config_dict['enable_pruning']
        if 'pruner_method' in config_dict:
            # Convert string to PrunerMethod enum if needed
            if isinstance(config_dict['pruner_method'], str):
                try:
                    self.config.pruner_method = PrunerMethod(config_dict['pruner_method'])
                except ValueError:
                    logger.warning(f"Invalid pruner method: {config_dict['pruner_method']}, using default")
            else:
                self.config.pruner_method = config_dict['pruner_method']
                
        if 'pruner_params' in config_dict:
            self.config.pruner_params = config_dict['pruner_params']
        
        # Early stopping
        if 'early_stopping_n_trials' in config_dict:
            self.config.early_stopping_n_trials = config_dict['early_stopping_n_trials']
        if 'early_stopping_threshold' in config_dict:
            self.config.early_stopping_threshold = config_dict['early_stopping_threshold']
        
        # Configuration du scoring
        if 'scoring_formula' in config_dict:
            # Convert string to ScoringFormula enum if needed
            if isinstance(config_dict['scoring_formula'], str):
                try:
                    self.config.scoring_formula = ScoringFormula(config_dict['scoring_formula'])
                except ValueError:
                    logger.warning(f"Invalid scoring formula: {config_dict['scoring_formula']}, using default")
            else:
                self.config.scoring_formula = config_dict['scoring_formula']
                
        if 'custom_weights' in config_dict:
            if isinstance(config_dict['custom_weights'], dict):
                self.config.custom_weights = ScoringWeights.from_dict(config_dict['custom_weights'])
            else:
                self.config.custom_weights = config_dict['custom_weights']
        
        # Limites de trading
        if 'min_trades' in config_dict:
            self.config.min_trades = config_dict['min_trades']
        
        # Multiprocessing
        if 'n_jobs' in config_dict:
            self.config.n_jobs = config_dict['n_jobs']
            # Update process manager
            self.process_manager = ProcessManager(n_jobs=config_dict['n_jobs'])
            
        if 'memory_limit' in config_dict:
            self.config.memory_limit = config_dict['memory_limit']
            # Update process manager
            self.process_manager.memory_limit = config_dict['memory_limit']
        
        # Checkpoints
        if 'save_checkpoints' in config_dict:
            self.config.save_checkpoints = config_dict['save_checkpoints']
        if 'checkpoint_every' in config_dict:
            self.config.checkpoint_every = config_dict['checkpoint_every']
        
        # Debug
        if 'debug' in config_dict:
            self.config.debug = config_dict['debug']
        
        # Initialiser le calculateur de score
        self._init_score_calculator()
        
        if self.config.debug:
            self.logger.info(f"Configuration de l'optimiseur: {config_dict}")
    
    def _init_score_calculator(self):
        """Initialise le calculateur de score"""
        formula_name = self.config.scoring_formula.value if isinstance(self.config.scoring_formula, ScoringFormula) else self.config.scoring_formula
        
        custom_weights = None
        if formula_name == "custom" and hasattr(self.config, "custom_weights"):
            if isinstance(self.config.custom_weights, ScoringWeights):
                custom_weights = self.config.custom_weights.to_dict()
            else:
                custom_weights = self.config.custom_weights
        
        self.score_calculator = ScoreCalculator.create_from_formula(
            formula_name, custom_weights
        )
    
    def prepare_optimization(self, study_name: str, optimization_config: Dict = None) -> Optional[Dict]:
        """
        Prépare la configuration d'optimisation pour une étude.
        
        Args:
            study_name: Nom de l'étude
            optimization_config: Configuration d'optimisation (optionnel)
            
        Returns:
            Optional[Dict]: Configuration d'optimisation ou None en cas d'erreur
        """
        try:
            # Vérifier si l'étude existe
            if not self.study_manager.study_exists(study_name):
                self.logger.error(f"L'étude '{study_name}' n'existe pas")
                return None
            
            # Récupérer la configuration de trading
            trading_config = self.study_manager.get_trading_config(study_name)
            if trading_config is None:
                self.logger.error(f"Impossible de récupérer la configuration de trading pour l'étude '{study_name}'")
                return None
            
            # Créer une configuration d'optimisation par défaut
            default_config = create_default_optimization_config().to_dict()
            
            # Fusion avec la configuration fournie
            if optimization_config:
                for key, value in optimization_config.items():
                    if key == 'custom_weights' and key in default_config:
                        # Fusion spéciale pour custom_weights
                        if isinstance(value, dict):
                            default_config[key].update(value)
                    else:
                        default_config[key] = value
            
            # Ajout de la date de création
            default_config['creation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Sauvegarde de la configuration
            self.study_manager.save_optimization_config(study_name, default_config)
            
            # Configuration de l'optimiseur
            self.configure(default_config)
            
            return default_config
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la préparation de l'optimisation: {str(e)}")
            traceback.print_exc()
            return None
    
    def prepare_data(self, data_path: str) -> None:
        """
        Prépare les données pour l'optimisation.
        
        Args:
            data_path: Chemin vers les données
        """
        try:
            # Chargement des données
            df = pd.read_csv(data_path)
            
            # Vérification des colonnes minimales requises
            required_columns = ['close', 'high', 'low']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing required columns: {missing_columns}. Trying to adapt...")
                
                # Try to adapt if columns are missing but we can map them
                if 'close' not in df.columns and 'Close' in df.columns:
                    df['close'] = df['Close']
                if 'high' not in df.columns and 'High' in df.columns:
                    df['high'] = df['High']
                if 'low' not in df.columns and 'Low' in df.columns:
                    df['low'] = df['Low']
                
                # Check again after adaptation
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Les données doivent contenir au moins les colonnes: {required_columns}")
            
            # Conversion en arrays NumPy
            prices = df['close'].values.astype(np.float64)
            high = df['high'].values.astype(np.float64)
            low = df['low'].values.astype(np.float64)
            volumes = df['volume'].values.astype(np.float64) if 'volume' in df.columns else None
            
            # Création de la mémoire partagée
            self.shared_data = {}
            
            # Fonction pour créer une mémoire partagée
            def create_shared_array(array, name):
                try:
                    shm = shared_memory.SharedMemory(create=True, size=array.nbytes, name=name)
                    shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
                    shared_array[:] = array[:]
                    return {
                        'shm': shm,
                        'name': shm.name,
                        'shape': array.shape,
                        'dtype': str(array.dtype)
                    }
                except Exception as e:
                    logger.error(f"Error creating shared memory for {name}: {e}")
                    raise
            
            # Création des arrays partagés
            pid = os.getpid()
            ts = int(time.time())
            self.shared_data['prices'] = create_shared_array(prices, f"prices_{pid}_{ts}")
            self.shared_data['high'] = create_shared_array(high, f"high_{pid}_{ts}")
            self.shared_data['low'] = create_shared_array(low, f"low_{pid}_{ts}")
            
            if volumes is not None:
                self.shared_data['volumes'] = create_shared_array(volumes, f"volumes_{pid}_{ts}")
                
            # Log success
            logger.info(f"Data prepared and shared memory created successfully. "
                       f"Shape: {prices.shape}, Columns: {df.columns.tolist()}")
            
        except Exception as e:
            logger.error(f"Error in prepare_data: {e}")
            traceback.print_exc()
            raise
    
    def cleanup_shared_data(self) -> None:
        """Nettoie les ressources de mémoire partagée"""
        if self.shared_data:
            for key, data in self.shared_data.items():
                try:
                    if 'shm' in data:
                        data['shm'].close()
                        data['shm'].unlink()
                except Exception as e:
                    self.logger.warning(f"Erreur lors du nettoyage de la mémoire partagée {key}: {e}")
            
            self.shared_data = None
            
            # Force GC to collect any lingering resources
            gc.collect()
    
    def _create_sampler(self):
        """
        Crée le sampler pour l'optimisation selon la configuration
        
        Returns:
            optuna.samplers.BaseSampler: Le sampler configuré
        """
        method = self.config.optimization_method
        method_name = method.value if isinstance(method, OptimizationMethod) else method
        
        method_info = OPTIMIZATION_METHODS.get(method_name, OPTIMIZATION_METHODS["tpe"])
        sampler_class = method_info["sampler_class"]
        
        # Paramètres de base
        params = {
            name: param["default"] 
            for name, param in method_info["params"].items()
        }
        
        # Fusion avec les paramètres personnalisés
        for name, value in self.config.method_params.items():
            if name in params or name in ["group_related_params", "consider_magic_clip"]:
                params[name] = value
        
        # Paramètres spéciaux pour le TPESampler
        if sampler_class == TPESampler:
            # Suppression des paramètres spéciaux qui ne sont pas pour le sampler
            params.pop("group_related_params", None)
            
            # Configuration pour meilleure exploration
            if "consider_magic_clip" in params:
                consider_magic = params.pop("consider_magic_clip")
                if consider_magic:
                    params["consider_magic_clip"] = True
            
            # Remplacer par EnhancedTPESampler personnalisé pour certains cas
            if self.config.method_params.get("group_related_params", False):
                # Supprimer multivariate des params pour éviter l'envoi en double
                params.pop("multivariate", None)
                return self._create_enhanced_tpe_sampler(**params)
        
        # Création du sampler
        return sampler_class(**params)

    def _create_enhanced_tpe_sampler(self, n_startup_trials=10, n_ei_candidates=24, seed=None, **kwargs):
        """
        Crée un TPESampler amélioré avec gestion des groupes et des paramètres conditionnels.
        
        Args:
            n_startup_trials: Nombre de trials de démarrage
            n_ei_candidates: Nombre de candidats pour l'expected improvement
            seed: Graine aléatoire
            **kwargs: Autres paramètres pour TPESampler
        
        Returns:
            optuna.samplers.TPESampler: Sampler TPE configuré
        """
        class EnhancedTPESampler(TPESampler):
            """TPESampler amélioré avec gestion des groupes de paramètres"""
            
            def __init__(self, *args, **kwargs):
                # Extraire les paramètres personnalisés avant de les passer au parent
                self.group_related_params = kwargs.pop('group_related_params', True)
                super().__init__(*args, **kwargs)
                self.param_groups = {}
                
            def register_param_group(self, param_name, group_name):
                """Enregistre un paramètre dans un groupe"""
                if group_name not in self.param_groups:
                    self.param_groups[group_name] = []
                
                if param_name not in self.param_groups[group_name]:
                    self.param_groups[group_name].append(param_name)
                    
            def sample_independent(self, study, trial, param_name, param_distribution):
                """
                Surcharge pour améliorer l'échantillonnage des paramètres indépendants
                """
                # Check if this parameter belongs to a group
                group = None
                for group_name, params in self.param_groups.items():
                    if param_name in params:
                        group = group_name
                        break
                
                # If it belongs to a group, use the first parameter's distribution
                # as a guide for the others
                if group and len(self.param_groups[group]) > 1:
                    # Find if any other parameter from this group has been sampled
                    for other_param in self.param_groups[group]:
                        if other_param != param_name and other_param in trial.params:
                            # Use knowledge from the other parameter to guide this one
                            return self._sample_using_group_knowledge(
                                study, trial, param_name, param_distribution, 
                                other_param, trial.params[other_param]
                            )
                
                # Fall back to standard sampling
                return super().sample_independent(study, trial, param_name, param_distribution)
            
            def _sample_using_group_knowledge(self, study, trial, param_name, param_distribution,
                                           other_param, other_value):
                """
                Sample a parameter using knowledge from another parameter in the same group
                """
                # This is where advanced sampling logic would go
                # For simplicity, we'll just use the standard method for now
                return super().sample_independent(study, trial, param_name, param_distribution)
        
        # Créer sampler avec tous les paramètres appropriés
        return EnhancedTPESampler(
            n_startup_trials=n_startup_trials,
            n_ei_candidates=n_ei_candidates, 
            seed=seed,
            multivariate=True,
            warn_independent_sampling=False,
            group_related_params=True,   # Ce paramètre sera extrait dans __init__
            **kwargs
        )

    def _create_pruner(self):
        """
        Crée le pruner pour l'optimisation selon la configuration
        
        Returns:
            optuna.pruners.BasePruner: Le pruner configuré
        """
        if not self.config.enable_pruning:
            return None
            
        pruner = self.config.pruner_method
        pruner_name = pruner.value if isinstance(pruner, PrunerMethod) else pruner
        
        if not pruner_name or pruner_name == "none":
            return None
            
        pruner_info = PRUNER_METHODS.get(pruner_name, PRUNER_METHODS["median"])
        pruner_class = pruner_info["pruner_class"]
        
        if pruner_class is None:
            return None
            
        default_params = {
            name: param["default"] 
            for name, param in pruner_info["params"].items()
        }
        
        # Fusion avec les paramètres personnalisés
        params = default_params.copy()
        for name, value in self.config.pruner_params.items():
            if name in default_params:
                params[name] = value
        
        # Création du pruner
        return pruner_class(**params)
    
    def _register_parameter_group(self, param_name, group_name=None):
        """
        Enregistre un paramètre dans un groupe pour l'échantillonnage cohérent
        
        Args:
            param_name: Nom du paramètre
            group_name: Nom du groupe (optionnel)
        
        Returns:
            str: Le nom du groupe
        """
        if not group_name:
            # Trouver le préfixe du paramètre (ex: "buy_block_1")
            parts = param_name.split('_')
            if len(parts) >= 3 and (parts[0] == 'buy' or parts[0] == 'sell') and parts[1] == 'block':
                group_name = f"{parts[0]}_{parts[1]}_{parts[2]}"
            elif param_name.startswith('risk_'):
                group_name = 'risk_params'
            elif param_name.startswith('indicator_'):
                group_name = 'indicator_params'
            else:
                # Créer un nouveau groupe si aucun correspondant
                self.group_counter += 1
                group_name = f"group_{self.group_counter}"
        
        # Ajouter au dictionnaire des groupes
        if group_name not in self.param_groups:
            self.param_groups[group_name] = []
        
        if param_name not in self.param_groups[group_name]:
            self.param_groups[group_name].append(param_name)
        
        return group_name    

    def objective(self, trial):
        """
        Fonction objectif améliorée pour Optuna avec gestion de groupes de paramètres
        et meilleure vérification des données partagées.
        
        Args:
            trial: Trial Optuna
        
        Returns:
            float: Score d'optimisation
        """
        # Utilise les variables globales définies dans init_worker
        global _shared_data, _trading_config
        
        # Vérifier que les données sont bien partagées
        if _shared_data is None or _trading_config is None:
            self.logger.error("Les données partagées ne sont pas initialisées correctement")
            
            # Attendre un peu et réessayer (mécanisme de retry)
            retry_count = 3
            for i in range(retry_count):
                time.sleep(0.5)  # Attendre 500ms
                if _shared_data is not None and _trading_config is not None:
                    self.logger.info(f"Données partagées récupérées après {i+1} tentatives")
                    break
            else:
                # Si toujours pas initialisé après les tentatives
                self.logger.error(f"Échec de récupération des données partagées après {retry_count} tentatives")
                return float('-inf')
                
        try:
            # Vérifier que les clés nécessaires sont présentes
            required_keys = ['prices', 'high', 'low']
            for key in required_keys:
                if key not in _shared_data:
                    self.logger.error(f"Clé '{key}' manquante dans les données partagées")
                    return float('-inf')
            
            # Récupération des données partagées avec gestion des erreurs améliorée
            try:
                prices_shm = shared_memory.SharedMemory(name=_shared_data['prices']['name'])
                high_shm = shared_memory.SharedMemory(name=_shared_data['high']['name'])
                low_shm = shared_memory.SharedMemory(name=_shared_data['low']['name'])
            except Exception as e:
                self.logger.error(f"Erreur lors de l'accès à la mémoire partagée: {e}")
                return float('-inf')
                
            # Conversion en arrays NumPy
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
            
            # Récupération des volumes si disponibles
            volumes = None
            volumes_shm = None
            if 'volumes' in _shared_data:
                try:
                    volumes_shm = shared_memory.SharedMemory(name=_shared_data['volumes']['name'])
                    volumes = np.ndarray(
                        _shared_data['volumes']['shape'],
                        dtype=_shared_data['volumes']['dtype'],
                        buffer=volumes_shm.buf
                    )
                except Exception as e:
                    self.logger.warning(f"Erreur lors de l'accès aux volumes: {e}")
                    # On continue sans les volumes
            
            try:
                # Pour le TPE sampler amélioré, enregistrer les groupes de paramètres
                if self.config.method_params.get("group_related_params", False) and hasattr(trial, "register_param_group"):
                    # Nettoyer les groupes
                    self.param_groups = {}
                    self.group_counter = 0
                
                # Création des gestionnaires avec gestion de groupes pour TPESampler
                # Échantillonnage de paramètres de structure avec groupes cohérents
                block_manager = BlockManager(GroupingTrialWrapper(trial, self._register_parameter_group), _trading_config)
                risk_manager = RiskManager(GroupingTrialWrapper(trial, self._register_parameter_group), _trading_config)
                sim_manager = SimulationManager(GroupingTrialWrapper(trial, self._register_parameter_group), _trading_config)
                
                # Génération des blocs de trading
                buy_blocks, sell_blocks = block_manager.generate_blocks()
                
                # Création du générateur de signaux
                signal_generator = SignalGenerator()
                
                # Ajout des blocs au générateur
                for block in buy_blocks:
                    signal_generator.add_block(block, is_buy=True)
                
                for block in sell_blocks:
                    signal_generator.add_block(block, is_buy=False)
                
                # Génération des signaux
                signals = signal_generator.generate_signals(prices, high, low, volumes)
                
                # Configuration du calculateur de position
                position_calculator = PositionCalculator(
                    mode=risk_manager.risk_mode,
                    config=risk_manager.get_config()
                )
                
                # Calcul des paramètres de risque
                position_sizes, sl_levels, tp_levels = position_calculator.calculate_risk_parameters(
                    prices=prices,
                    high=high,
                    low=low
                )
                
                # Simulation du trading
                simulator = Simulator(config=sim_manager.get_simulator_config())
                
                # Exécution de la simulation
                results = simulator.run(
                    prices=prices,
                    signals=signals,
                    position_sizes=position_sizes,
                    sl_levels=sl_levels,
                    tp_levels=tp_levels,
                    leverage_levels=np.full_like(prices, sim_manager.leverage, dtype=np.float64)
                )
                
                # Extraction des résultats
                performance = results.get('performance', {})
                roi = performance.get('roi', 0)
                win_rate = performance.get('win_rate', 0)
                total_trades = performance.get('total_trades', 0)
                max_drawdown = performance.get('max_drawdown', 1)
                avg_profit_per_trade = performance.get('avg_profit_per_trade', 0)
                liquidation_rate = performance.get('liquidation_rate', 0)
                max_profit = performance.get('max_profit_trade', 0)
                max_loss = performance.get('max_loss_trade', 0)
                profit_factor = performance.get('profit_factor', 0)
                
                # Stockage des métriques dans le trial
                trial.set_user_attr('roi', float(roi))
                trial.set_user_attr('win_rate', float(win_rate))
                trial.set_user_attr('total_trades', float(total_trades))
                trial.set_user_attr('max_drawdown', float(max_drawdown))
                trial.set_user_attr('profit_factor', float(profit_factor))
                trial.set_user_attr('avg_profit', float(avg_profit_per_trade))
                trial.set_user_attr('liquidation_rate', float(liquidation_rate))
                trial.set_user_attr('max_profit', float(max_profit))
                trial.set_user_attr('max_loss', float(max_loss))
                
                # Calcul des ratios avancés
                trades_per_day = total_trades / (len(prices) / 1440)  # Approximation pour des données 1m
                trial.set_user_attr('trades_per_day', float(trades_per_day))
                
                # Vérifications préliminaires
                if total_trades < self.config.min_trades:
                    return float('-inf')
                
                # Préparer les métriques pour le calcul du score
                metrics = {
                    'roi': roi,
                    'win_rate': win_rate,
                    'max_drawdown': max_drawdown,
                    'profit_factor': profit_factor,
                    'total_trades': total_trades,
                    'avg_profit': avg_profit_per_trade,
                    'trades_per_day': trades_per_day,
                    'max_consecutive_losses': results.get('max_consecutive_losses', 0) if 'max_consecutive_losses' in results else 0
                }
                
                # Calcul du score avec le calculateur configuré
                score = self.score_calculator.calculate_score(metrics)
                
                # Libération des ressources
                signal_generator.cleanup()
                
                # Collecte garbage si configuré ainsi
                if self.config.gc_after_trial:
                    gc.collect()
                
                return score
                
            finally:
                # Nettoyage des ressources partagées
                prices_shm.close()
                high_shm.close()
                low_shm.close()
                
                if volumes_shm is not None:
                    volumes_shm.close()
                    
        except Exception as e:
            self.logger.error(f"Erreur dans la fonction objective: {str(e)}")
            traceback.print_exc()
            return float('-inf')
    
    def _validate_shared_data(self):
        """
        Valide que les données partagées sont correctement initialisées
        
        Returns:
            bool: True si les données sont valides, False sinon
        """
        if not self.shared_data:
            self.logger.error("Aucune donnée partagée initialisée")
            return False
        
        required_keys = ['prices', 'high', 'low']
        for key in required_keys:
            if key not in self.shared_data:
                self.logger.error(f"Donnée partagée '{key}' manquante")
                return False
            
            data = self.shared_data[key]
            if 'name' not in data or 'shape' not in data or 'dtype' not in data:
                self.logger.error(f"Informations manquantes pour la donnée '{key}'")
                return False
                
            # Vérifier que la mémoire partagée est accessible
            try:
                shm = shared_memory.SharedMemory(name=data['name'])
                shm.close()
            except Exception as e:
                self.logger.error(f"Erreur lors de l'accès à la mémoire partagée '{key}': {e}")
                return False
        
        self.logger.info("Validation des données partagées: OK")
        return True

    
    def run_optimization(self, study_name: str, data_path: str) -> bool:
        """
        Lance l'optimisation de la stratégie avec les améliorations.
        
        Args:
            study_name: Nom de l'étude
            data_path: Chemin vers les données
            
        Returns:
            bool: True si l'optimisation a réussi, False sinon
        """
        try:
            # Vérifier si l'étude existe
            if not self.study_manager.study_exists(study_name):
                self.logger.error(f"L'étude '{study_name}' n'existe pas")
                return False
            
            # Récupérer la configuration de trading
            trading_config = self.study_manager.get_trading_config(study_name)
            if trading_config is None:
                self.logger.error(f"Impossible de récupérer la configuration de trading pour l'étude '{study_name}'")
                return False
            
            # Récupérer la configuration d'optimisation
            optim_config = self.study_manager.get_optimization_config(study_name)
            if optim_config is None:
                # Créer une configuration par défaut
                optim_config = self.prepare_optimization(study_name)
                if optim_config is None:
                    self.logger.error(f"Impossible de créer une configuration d'optimisation pour l'étude '{study_name}'")
                    return False
            
            # Configuration de l'optimiseur
            self.configure(optim_config)
            
            # Préparation des données
            self.prepare_data(data_path)
            
            # Configuration du storage
            storage_path = os.path.join(os.path.dirname(data_path), f"{study_name}_optimization.db")
            storage_url = f"sqlite:///{storage_path}"
            storage = optuna.storages.RDBStorage(
                url=storage_url,
                engine_kwargs={
                    'connect_args': {'timeout': 300},
                    'pool_size': 1
                }
            )
            
            # Préparation du sampler
            sampler = self._create_sampler()
            
            # Préparation du pruner
            pruner = self._create_pruner() if self.config.enable_pruning else None
            
            # Création ou chargement de l'étude
            study = optuna.create_study(
                storage=storage,
                sampler=sampler,
                pruner=pruner,
                study_name=f"{study_name}_opt",
                direction="maximize",
                load_if_exists=True
            )
            
            # Déterminer le mode d'exécution (parallèle ou séquentiel)
            n_jobs = self.config.n_jobs
            
            # Pour éviter les problèmes de multiprocessing, limiter à 1 processus si données partagées
            # ou n_jobs est invalide
            if not self.shared_data or n_jobs < 1:
                self.logger.warning(f"Utilisation du mode séquentiel (n_jobs=1) pour éviter les problèmes de données partagées")
                n_jobs = 1
            
            self.logger.info(f"Exécution de l'optimisation avec {n_jobs} processus")
            
            # Fonctions de callback
            callbacks = []
            
            # Callback de progression
            def log_progress(study, trial):
                if trial.value is not None and trial.value > float('-inf'):
                    trial_metrics = {
                        key: trial.user_attrs.get(key, "N/A") 
                        for key in ['roi', 'win_rate', 'total_trades', 'max_drawdown']
                    }
                    
                    metrics_str = ", ".join([f"{k}: {v}" for k, v in trial_metrics.items()])
                    self.logger.info(f"Trial {trial.number}: score={trial.value:.4f}, {metrics_str}")
                    
                    # Mettre à jour la progression dans le dictionnaire global
                    from ui.components.studies.optimization_panel import optimization_progress
                    if study_name in optimization_progress:
                        optimization_progress[study_name]['completed'] = trial.number
                        optimization_progress[study_name]['best_value'] = study.best_value
                        if trial.user_attrs:
                            optimization_progress[study_name]['best_metrics'] = {
                                key: trial.user_attrs.get(key, None)
                                for key in ['roi', 'win_rate', 'total_trades', 'max_drawdown']
                            }
            
            callbacks.append(log_progress)
            
            # Callback de sauvegarde intermédiaire (checkpoint)
            if self.config.save_checkpoints:
                def save_checkpoint(study, trial):
                    if trial.number > 0 and trial.number % self.config.checkpoint_every == 0:
                        self.save_best_strategies(study_name, study, top_n=3)
                        self.logger.info(f"Checkpoint sauvegardé après le trial {trial.number}")
                
                callbacks.append(save_checkpoint)
            
            # Callback pour la gestion des ressources système
            def manage_resources(study, trial):
                # Examiner l'utilisation des ressources
                try:
                    memory_usage = psutil.virtual_memory().percent / 100.0
                    if memory_usage > self.config.memory_limit:
                        self.logger.warning(f"Utilisation mémoire élevée ({memory_usage:.1%}), forcage GC")
                        gc.collect()
                        time.sleep(1)  # Pause pour laisser le GC faire son travail
                except:
                    pass
            
            callbacks.append(manage_resources)
            
            def check_stop_flag(study, trial):
                """Vérifie si l'optimisation a été marquée pour arrêt"""
                global _optimization_progress
                
                if study_name in _optimization_progress and _optimization_progress.get(study_name, {}).get('status') == 'stopped':
                    self.logger.info(f"Optimisation '{study_name}' arrêtée manuellement après l'essai {trial.number}")
                    # Lever une exception pour arrêter le processus d'optimisation
                    raise optuna.exceptions.OptunaError(f"Optimisation '{study_name}' arrêtée manuellement.")

            callbacks.append(check_stop_flag)

            # Si exécution séquentielle (n_jobs=1), utiliser directement l'objectif
            # sans worker init
            if n_jobs == 1:
                # Configuration des variables globales directement
                global _shared_data, _trading_config
                _shared_data = self.shared_data
                _trading_config = trading_config
                
                # Vérification des données partagées
                self.logger.info("Mode séquentiel: initialisation directe des données partagées")
                
                # Test de validation des données
                if not self._validate_shared_data():
                    self.logger.error("Validation des données partagées échouée. Impossible de poursuivre.")
                    return False
                
                # Exécuter l'optimisation en mode séquentiel
                self.logger.info("Démarrage de l'optimisation en mode séquentiel")
                
                # Exécuter l'optimisation
                start_time = time.time()
                try:
                    study.optimize(
                        self.objective,
                        n_trials=self.config.n_trials,
                        timeout=self.config.timeout,
                        n_jobs=1,  # Mode séquentiel forcé
                        callbacks=callbacks,
                        gc_after_trial=self.config.gc_after_trial,
                        catch=(Exception,)
                    )
                    
                    execution_time = time.time() - start_time
                    self.logger.info(f"Optimisation terminée en {execution_time:.2f} secondes ({execution_time/60:.2f} minutes)")
                    
                    # Affichage des résultats
                    if study.best_trial:
                        best_trial = study.best_trial
                        self.logger.info(f"Meilleur trial: {best_trial.number}")
                        self.logger.info(f"Meilleur score: {best_trial.value}")
                        
                        if 'roi' in best_trial.user_attrs:
                            self.logger.info(f"ROI: {best_trial.user_attrs['roi']*100:.2f}%")
                            self.logger.info(f"Win Rate: {best_trial.user_attrs['win_rate']*100:.2f}%")
                            self.logger.info(f"Trades: {best_trial.user_attrs['total_trades']}")
                            self.logger.info(f"Max Drawdown: {best_trial.user_attrs['max_drawdown']*100:.2f}%")
                    
                    # Sauvegarde des résultats
                    self.save_best_strategies(study_name, study)
                    
                    # Mise à jour des statistiques
                    from ui.components.studies.optimization_panel import optimization_progress
                    if study_name in optimization_progress:
                        optimization_progress[study_name]['status'] = 'completed'
                        optimization_progress[study_name]['completed'] = self.config.n_trials
                        optimization_progress[study_name]['best_value'] = study.best_value
                    
                    return True
                
                except Exception as e:
                    self.logger.error(f"Erreur pendant l'optimisation: {str(e)}")
                    traceback.print_exc()
                    
                    # Mettre à jour le statut dans le dictionnaire global
                    from ui.components.studies.optimization_panel import optimization_progress
                    if study_name in optimization_progress:
                        optimization_progress[study_name]['status'] = 'error'
                        optimization_progress[study_name]['error_message'] = str(e)
                    
                    return False
            else:
                # Mode parallèle avec multiprocessing - besoin de fixer problème de shared_data
                self.logger.info(f"Mode parallèle: préparation de l'initialisation des workers pour {n_jobs} processus")
                
                # Créer une fonction d'initialisation pour les workers qui contient un mécanisme
                # de vérification et d'attente
                def init_worker_robust(shared_data_info, trading_config_dict):
                    """Version robuste de init_worker avec vérification"""
                    global _shared_data, _trading_config
                    _shared_data = shared_data_info
                    _trading_config = TradingConfig.from_dict(trading_config_dict)
                    
                    # Vérifier l'initialisation
                    if not _shared_data or not _trading_config:
                        import time
                        # Attendre un peu et réessayer
                        for _ in range(5):  # 5 tentatives
                            time.sleep(0.5)
                            if _shared_data and _trading_config:
                                break
                    
                    # Log le statut d'initialisation
                    import logging
                    logger = logging.getLogger('worker_init')
                    logger.info(f"Worker initialisé: shared_data={'OK' if _shared_data else 'NON'}, "
                            f"trading_config={'OK' if _trading_config else 'NON'}")
                
                # Préparation des données pour le mode parallèle
                init_args = (self.shared_data, trading_config.to_dict())
                
                # Exécuter l'optimisation avec init robuste
                start_time = time.time()
                try:
                    study.optimize(
                        self.objective,
                        n_trials=self.config.n_trials,
                        timeout=self.config.timeout,
                        n_jobs=n_jobs,
                        callbacks=callbacks,
                        gc_after_trial=self.config.gc_after_trial,
                        catch=(Exception,),
                        # Utiliser un mécanisme de partage de données qui a fait ses preuves
                        # avec la nouvelle fonction d'initialisation
                        multiprocessing_options={
                            'initializer': init_worker_robust,
                            'initargs': init_args,
                            'context': 'spawn'  # Utiliser 'spawn' pour être sûr
                        }
                    )
                    
                    execution_time = time.time() - start_time
                    self.logger.info(f"Optimisation terminée en {execution_time:.2f} secondes ({execution_time/60:.2f} minutes)")
                    
                    # Affichage des résultats
                    if study.best_trial:
                        best_trial = study.best_trial
                        self.logger.info(f"Meilleur trial: {best_trial.number}")
                        self.logger.info(f"Meilleur score: {best_trial.value}")
                        
                        if 'roi' in best_trial.user_attrs:
                            self.logger.info(f"ROI: {best_trial.user_attrs['roi']*100:.2f}%")
                            self.logger.info(f"Win Rate: {best_trial.user_attrs['win_rate']*100:.2f}%")
                            self.logger.info(f"Trades: {best_trial.user_attrs['total_trades']}")
                            self.logger.info(f"Max Drawdown: {best_trial.user_attrs['max_drawdown']*100:.2f}%")
                    
                    # Sauvegarde des résultats
                    self.save_best_strategies(study_name, study)
                    
                    # Mise à jour des statistiques
                    from ui.components.studies.optimization_panel import optimization_progress
                    if study_name in optimization_progress:
                        optimization_progress[study_name]['status'] = 'completed'
                        optimization_progress[study_name]['completed'] = self.config.n_trials
                        optimization_progress[study_name]['best_value'] = study.best_value
                    
                    return True
                
                except Exception as e:
                    self.logger.error(f"Erreur pendant l'optimisation: {str(e)}")
                    traceback.print_exc()
                    
                    # Mettre à jour le statut dans le dictionnaire global
                    from ui.components.studies.optimization_panel import optimization_progress
                    if study_name in optimization_progress:
                        optimization_progress[study_name]['status'] = 'error'
                        optimization_progress[study_name]['error_message'] = str(e)
                    
                    return False
                    
        except Exception as e:
            self.logger.error(f"Erreur lors de l'optimisation de la stratégie: {str(e)}")
            traceback.print_exc()
            return False
        finally:
            # Nettoyage des ressources
            self.cleanup_shared_data()
            gc.collect()

    def save_best_strategies(self, study_name: str, study: optuna.Study, top_n: int = 5) -> bool:
        """
        Sauvegarde les meilleures stratégies d'une étude Optuna.
        
        Args:
            study_name: Nom de l'étude
            study: Étude Optuna
            top_n: Nombre de meilleures stratégies à sauvegarder
            
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        try:
            # Récupération des meilleurs trials
            best_trials = sorted(
                [t for t in study.trials if t.value is not None and t.value > float('-inf')],
                key=lambda t: t.value if t.value is not None else float('-inf'),
                reverse=True
            )[:top_n]  # Top N
            
            if not best_trials:
                self.logger.error(f"Aucun trial valide trouvé pour l'étude '{study_name}'")
                return False
            
            # Récupération de la configuration de trading
            trading_config = self.study_manager.get_trading_config(study_name)
            if trading_config is None:
                self.logger.error(f"Impossible de récupérer la configuration de trading pour l'étude '{study_name}'")
                return False
            
            # Sauvegarde des résultats d'optimisation
            from datetime import datetime
            
            optimization_results = {
                'study_name': study_name,
                'optimization_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'n_trials': len(study.trials),
                'best_trial_id': best_trials[0].number,
                'best_score': best_trials[0].value,
                'optimization_config': {
                    'method': self.config.optimization_method.value 
                        if hasattr(self.config.optimization_method, 'value') else self.config.optimization_method,
                    'method_params': self.config.method_params,
                    'enable_pruning': self.config.enable_pruning,
                    'pruner_method': self.config.pruner_method.value 
                        if hasattr(self.config.pruner_method, 'value') else self.config.pruner_method,
                    'scoring_formula': self.config.scoring_formula.value 
                        if hasattr(self.config.scoring_formula, 'value') else self.config.scoring_formula,
                    'custom_weights': self.config.custom_weights.to_dict() 
                        if hasattr(self.config.custom_weights, 'to_dict') else self.config.custom_weights
                },
                'best_trials': [{
                    'trial_id': t.number,
                    'score': t.value,
                    'params': t.params,
                    'metrics': {
                        k: v for k, v in t.user_attrs.items()
                        if k in ['roi', 'win_rate', 'total_trades', 'max_drawdown', 
                               'profit_factor', 'avg_profit', 'trades_per_day',
                               'liquidation_rate', 'max_profit', 'max_loss']
                    }
                } for t in best_trials]
            }
            
            self.study_manager.save_optimization_results(study_name, optimization_results)
            
            # Sauvegarde des meilleures stratégies
            class DummyTrial:
                """Classe pour simuler un Trial Optuna à partir de paramètres existants"""
                
                def __init__(self, params: Dict):
                    """
                    Initialise un faux trial avec des paramètres existants.
                    
                    Args:
                        params: Paramètres du trial
                    """
                    self.params = params
                
                def suggest_categorical(self, name: str, choices: List):
                    """Retourne le paramètre existant ou le premier choix"""
                    return self.params.get(name, choices[0])
                
                def suggest_int(self, name: str, low: int, high: int, step: int = 1):
                    """Retourne le paramètre existant ou la valeur basse"""
                    return self.params.get(name, low)
                
                def suggest_float(self, name: str, low: float, high: float, step: float = None, log: bool = False):
                    """Retourne le paramètre existant ou la valeur basse"""
                    return self.params.get(name, low)
            
            saved_strategies = 0
            for i, trial in enumerate(best_trials):
                try:
                    # Création du gestionnaire de blocs
                    block_manager = BlockManager(DummyTrial(trial.params), trading_config)
                    buy_blocks, sell_blocks = block_manager.generate_blocks()
                    
                    # Vérifier si les blocs générés sont valides
                    if not buy_blocks and not sell_blocks:
                        self.logger.warning(f"Trial {trial.number} n'a pas généré de blocs valides. Ajout d'un bloc par défaut.")
                        # Ajouter un bloc par défaut par sécurité
                        from simulator.indicators import Condition, Operator, Block
                        empty_condition = Condition(
                            indicator1="EMA_10",
                            operator=Operator.GREATER,
                            indicator2="EMA_20"
                        )
                        empty_block = Block(conditions=[empty_condition], logic_operators=[])
                        buy_blocks = [empty_block]
                    
                    # Création du gestionnaire de risque
                    risk_manager = RiskManager(DummyTrial(trial.params), trading_config)
                    
                    # Création du générateur de signaux
                    signal_generator = SignalGenerator()
                    
                    # Ajout des blocs
                    for block in buy_blocks:
                        signal_generator.add_block(block, is_buy=True)
                    
                    for block in sell_blocks:
                        signal_generator.add_block(block, is_buy=False)
                    
                    # Création du calculateur de position
                    position_calculator = PositionCalculator(
                        mode=risk_manager.risk_mode,
                        config=risk_manager.get_config()
                    )
                    
                    # Sauvegarde de la stratégie
                    rank = i + 1
                    
                    performance = {
                        'name': f'Optimized Strategy {rank}',
                        'source': 'Optimization',
                        'trial_id': trial.number,
                        'score': trial.value
                    }
                    
                    # Ajout des métriques
                    for key, value in trial.user_attrs.items():
                        if key in ['roi', 'win_rate', 'total_trades', 'max_drawdown', 
                                 'profit_factor', 'avg_profit', 'trades_per_day',
                                 'liquidation_rate', 'max_profit', 'max_loss']:
                            performance[key] = value
                    
                    # Conversion en pourcentages pour certaines métriques
                    for key in ['roi', 'win_rate', 'max_drawdown']:
                        if key in performance:
                            performance[f'{key}_pct'] = performance[key] * 100
                    
                    self.study_manager.save_strategy(
                        study_name=study_name,
                        strategy_rank=rank,
                        signal_generator=signal_generator,
                        position_calculator=position_calculator,
                        performance=performance
                    )
                    
                    saved_strategies += 1
                    
                    # Nettoyage
                    signal_generator.cleanup()
                except Exception as e:
                    self.logger.error(f"Erreur lors de la sauvegarde de la stratégie {i+1}: {str(e)}")
                    continue
            
            self.logger.info(f"{saved_strategies} meilleures stratégies sauvegardées pour l'étude '{study_name}'")
            
            # Mise à jour du statut de l'étude
            self.study_manager.update_study_status(study_name, "optimized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde des meilleures stratégies: {str(e)}")
            traceback.print_exc()
            return False

    @staticmethod
    def get_optimization_progress(study_name: str = None) -> Dict:
        """
        Récupère la progression des optimisations en cours.
        
        Args:
            study_name: Nom de l'étude (optionnel)
            
        Returns:
            Dict: État de progression
        """
        global _optimization_progress
        
        if study_name:
            return _optimization_progress.get(study_name, {})
        else:
            return _optimization_progress
    
    @staticmethod
    def stop_optimization(study_name: str) -> bool:
        """
        Arrête une optimisation en cours.
        
        Args:
            study_name: Nom de l'étude
            
        Returns:
            bool: True si l'arrêt a réussi
        """
        global _optimization_progress
        
        if study_name in _optimization_progress:
            _optimization_progress[study_name]['status'] = 'stopped'
            logger.info(f"Optimisation '{study_name}' marquée pour arrêt")
            return True
        else:
            # Initialize the entry if it doesn't exist yet
            _optimization_progress[study_name] = {'status': 'stopped'}
            logger.info(f"Optimisation '{study_name}' marquée pour arrêt (création de l'entrée)")
            return True

# Pour une compatibilité avec d'anciens scripts
def get_optimization_progress(study_name: str = None) -> Dict:
    """Wrapper pour la méthode statique de StrategyOptimizer"""
    return StrategyOptimizer.get_optimization_progress(study_name)