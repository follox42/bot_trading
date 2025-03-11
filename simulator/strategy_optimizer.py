"""
Correction du problème de multiprocessing dans l'optimiseur intégré et ajout du logger.
"""
import os
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler, RandomSampler
import json
import time
import gc
import traceback
import logging
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Manager, shared_memory
import concurrent.futures
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import warnings
import psutil
import random
import math

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
from indicators import SignalGenerator, Block, Condition, Operator, LogicOperator
from risk import PositionCalculator, RiskMode
from simulator import Simulator, SimulationConfig
from config import (
    FlexibleTradingConfig, create_flexible_default_config, 
    MarginMode, TradingMode, convert_to_simulator_config, 
    create_position_calculator_config
)

# Initialisation des variables globales pour le multiprocessing
_shared_data = None
_trading_config = None

# Fonction pour initialiser les workers
def init_worker(shared_data_info, trading_config_dict):
    """
    Initialise un worker de multiprocessing.
    
    Args:
        shared_data_info: Informations sur les données partagées
        trading_config_dict: Configuration de trading au format dict
    """
    global _shared_data, _trading_config
    _shared_data = shared_data_info
    _trading_config = FlexibleTradingConfig.from_dict(trading_config_dict)

class BlockManager:
    """Manages the generation of trading blocks based on trial parameters"""
    
    def __init__(self, trial, trading_config):
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
        
        # First indicator
        ind1_type = self.trial.suggest_categorical(f"{cond_prefix}_ind1_type", available_inds)
        ind_config = self.available_indicators[ind1_type]
        
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
                value = self.trial.suggest_float(f"{cond_prefix}_value", 20, 80)
            else:
                # Use a multiplier of current price for other indicators
                multiplier = self.trial.suggest_float(f"{cond_prefix}_multiplier", 0.5, 1.5)
                value = multiplier  # Will be multiplied by price at runtime
        else:
            # Compare to another indicator
            ind2_type = self.trial.suggest_categorical(f"{cond_prefix}_ind2_type", available_inds)
            ind_config2 = self.available_indicators[ind2_type]
            
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
        self.trial = trial
        self.trading_config = trading_config
        self.risk_config = trading_config.risk_config
        
        # Choose risk mode from available modes
        available_modes = [mode.value for mode in self.risk_config.available_modes]
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
        self.trial = trial
        self.trading_config = trading_config
        self.sim_config = trading_config.sim_config
        
        # Choose leverage from available range
        leverage_range = self.sim_config.leverage_range
        self.leverage = self.trial.suggest_int("leverage", leverage_range[0], leverage_range[1], log=True)
        
        # Choose margin mode
        margin_modes = [mode.value for mode in self.sim_config.margin_modes]
        self.margin_mode = self.trial.suggest_categorical("margin_mode", margin_modes)
        
        # Choose trading mode
        trading_modes = [mode.value for mode in self.sim_config.trading_modes]
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

class IntegratedStrategyOptimizer:
    """
    Optimiseur de stratégies intégré avec le gestionnaire d'études.
    Utilise les configurations des études pour l'optimisation.
    Version corrigée pour le multiprocessing.
    """
    
    def __init__(self, study_manager: 'IntegratedStudyManager'):
        """
        Initialise l'optimiseur de stratégies avec le gestionnaire d'études.
        
        Args:
            study_manager: Gestionnaire d'études
        """
        self.study_manager = study_manager
        self.shared_data = None
        self.debug = False
        
        # Attributs pour l'optimisation
        self.n_trials = 100
        self.n_jobs = -1
        self.timeout = None
        self.gc_after_trial = True
        self.sampler_type = 'tpe'
        self.n_startup_trials = 10
        self.n_ei_candidates = 24
        self.early_stopping_n_trials = None
        self.early_stopping_threshold = 0.0
        self.min_trades = 10
        self.score_weights = {
            'roi': 2.5,
            'win_rate': 0.5,
            'max_drawdown': 2.0,
            'profit_factor': 2.0,
            'total_trades': 1.0,
            'avg_profit': 1.0
        }
    
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
                logger.error(f"L'étude '{study_name}' n'existe pas")
                return None
            
            # Récupérer la configuration de trading
            trading_config = self.study_manager.get_trading_config(study_name)
            if trading_config is None:
                logger.error(f"Impossible de récupérer la configuration de trading pour l'étude '{study_name}'")
                return None
            
            # Configuration d'optimisation par défaut
            default_config = {
                'n_trials': 100,
                'n_jobs': -1,
                'timeout': None,
                'gc_after_trial': True,
                'sampler': 'tpe',
                'n_startup_trials': 10,
                'n_ei_candidates': 24,
                'early_stopping_n_trials': None,
                'early_stopping_threshold': 0.0,
                'min_trades': 10,
                'score_weights': {
                    'roi': 2.5,
                    'win_rate': 0.5,
                    'max_drawdown': 2.0,
                    'profit_factor': 2.0,
                    'total_trades': 1.0,
                    'avg_profit': 1.0
                }
            }
            
            # Fusion avec la configuration fournie
            if optimization_config:
                for key, value in optimization_config.items():
                    default_config[key] = value
            
            # Ajout de la date de création
            default_config['creation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Sauvegarde de la configuration
            self.study_manager.save_optimization_config(study_name, default_config)
            
            return default_config
            
        except Exception as e:
            logger.error(f"Erreur lors de la préparation de l'optimisation: {str(e)}")
            traceback.print_exc()
            return None
    
    def prepare_data(self, data_path: str) -> None:
        """
        Prépare les données pour l'optimisation.
        
        Args:
            data_path: Chemin vers les données
        """
        # Chargement des données
        df = pd.read_csv(data_path)
        
        # Vérification des colonnes minimales requises
        required_columns = ['close', 'high', 'low']
        if not all(col in df.columns for col in required_columns):
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
            shm = shared_memory.SharedMemory(create=True, size=array.nbytes, name=name)
            shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
            shared_array[:] = array[:]
            return {
                'shm': shm,
                'name': shm.name,
                'shape': array.shape,
                'dtype': array.dtype
            }
        
        # Création des arrays partagés
        self.shared_data['prices'] = create_shared_array(prices, f"prices_{os.getpid()}")
        self.shared_data['high'] = create_shared_array(high, f"high_{os.getpid()}")
        self.shared_data['low'] = create_shared_array(low, f"low_{os.getpid()}")
        
        if volumes is not None:
            self.shared_data['volumes'] = create_shared_array(volumes, f"volumes_{os.getpid()}")
    
    def cleanup_shared_data(self) -> None:
        """Nettoie les ressources de mémoire partagée"""
        if self.shared_data:
            for key, data in self.shared_data.items():
                try:
                    data['shm'].close()
                    data['shm'].unlink()
                except Exception as e:
                    logger.warning(f"Erreur lors du nettoyage de la mémoire partagée {key}: {e}")
            
            self.shared_data = None
    
    def objective(self, trial):
        """
        Fonction objectif pour Optuna avec configuration flexible.
        
        Args:
            trial: Trial Optuna
        
        Returns:
            float: Score d'optimisation
        """
        # Utilise les variables globales définies dans init_worker
        global _shared_data, _trading_config
        
        # Vérifier que les données sont bien partagées
        if _shared_data is None or _trading_config is None:
            logger.error("Les données partagées ne sont pas initialisées correctement")
            return float('-inf')
            
        try:
            # Récupération des données partagées
            prices_shm = shared_memory.SharedMemory(name=_shared_data['prices']['name'])
            high_shm = shared_memory.SharedMemory(name=_shared_data['high']['name'])
            low_shm = shared_memory.SharedMemory(name=_shared_data['low']['name'])
            
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
                volumes_shm = shared_memory.SharedMemory(name=_shared_data['volumes']['name'])
                volumes = np.ndarray(
                    _shared_data['volumes']['shape'],
                    dtype=_shared_data['volumes']['dtype'],
                    buffer=volumes_shm.buf
                )
            
            try:
                # Création des gestionnaires
                block_manager = BlockManager(trial, _trading_config)
                risk_manager = RiskManager(trial, _trading_config)
                sim_manager = SimulationManager(trial, _trading_config)
                
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
                if total_trades < self.min_trades:
                    return float('-inf')
                
                # Préparer les métriques pour le calcul du score
                metrics = {
                    'roi': roi,
                    'win_rate': win_rate,
                    'max_drawdown': max_drawdown,
                    'profit_factor': profit_factor,
                    'total_trades': total_trades,
                    'avg_profit': avg_profit_per_trade,
                    'trades_per_day': trades_per_day
                }
                
                score = self._calculate_score(metrics, self.score_weights)
                
                # Libération des ressources
                signal_generator.cleanup()
                
                # Collecte garbage si configuré ainsi
                if self.gc_after_trial:
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
            logger.error(f"Erreur dans la fonction objective: {str(e)}")
            traceback.print_exc()
            return float('-inf')
    
    def _calculate_score(self, metrics: Dict[str, float], weights: Dict[str, float]) -> float:
        """
        Calcule un score pondéré à partir des métriques et des poids.
        
        Args:
            metrics: Dictionnaire des métriques de performance
            weights: Dictionnaire des poids pour chaque métrique
            
        Returns:
            float: Score final
        """
        # Validation initiale
        if metrics['total_trades'] == 0:
            return float('-inf')
        
        # Normalisation des métriques
        normalized_metrics = {}
        
        # ROI - normalisation sigmoïde
        roi = metrics.get('roi', 0)
        normalized_metrics['roi'] = min(1.0, max(0, (1 / (1 + np.exp(-roi * 2)) - 0.5) * 2))
        
        # Win rate - déjà entre 0 et 1
        normalized_metrics['win_rate'] = metrics.get('win_rate', 0)
        
        # Drawdown - transformation pour pénaliser les grands drawdowns
        max_dd = metrics.get('max_drawdown', 1)
        normalized_metrics['max_drawdown'] = max(0.0, 1.0 - max_dd)
        
        # Profit factor - transformation log
        pf = metrics.get('profit_factor', 0)
        normalized_metrics['profit_factor'] = min(1.0, max(0.0, np.log(pf + 0.1) / 2.0)) if pf > 0 else 0
        
        # Nombre de trades - échelle logarithmique
        trades = metrics.get('total_trades', 0)
        normalized_metrics['total_trades'] = min(1.0, np.log(trades + 1) / np.log(1001))
        
        # Profit moyen par trade
        avg_profit = metrics.get('avg_profit', 0)
        normalized_metrics['avg_profit'] = min(1.0, max(0.0, avg_profit * 10 + 0.5))
        
        # Trades par jour - bonus pour la fréquence
        trades_per_day = metrics.get('trades_per_day', 0)
        normalized_metrics['trades_per_day'] = min(1.0, trades_per_day / 10.0)
        
        # Calcul du score final pondéré
        score = 0.0
        total_weight = sum(weights.values())
        
        for metric, weight in weights.items():
            if metric in normalized_metrics:
                score += normalized_metrics[metric] * (weight / total_weight)
        
        # Transformation non-linéaire pour différencier les bonnes stratégies
        return (score ** 1.2) * 10.0  # Échelle finale de 0 à 10
    
    def run_optimization(self, study_name: str, data_path: str) -> bool:
        """
        Lance l'optimisation de la stratégie.
        
        Args:
            study_name: Nom de l'étude
            data_path: Chemin vers les données
            
        Returns:
            bool: True si l'optimisation a réussi, False sinon
        """
        try:
            # Vérifier si l'étude existe
            if not self.study_manager.study_exists(study_name):
                logger.error(f"L'étude '{study_name}' n'existe pas")
                return False
            
            # Récupérer la configuration de trading
            trading_config = self.study_manager.get_trading_config(study_name)
            if trading_config is None:
                logger.error(f"Impossible de récupérer la configuration de trading pour l'étude '{study_name}'")
                return False
            
            # Récupérer la configuration d'optimisation
            optim_config = self.study_manager.get_optimization_config(study_name)
            if optim_config is None:
                # Créer une configuration par défaut
                optim_config = self.prepare_optimization(study_name)
                if optim_config is None:
                    logger.error(f"Impossible de créer une configuration d'optimisation pour l'étude '{study_name}'")
                    return False
            
            # Attributs d'optimisation
            self.n_trials = optim_config.get('n_trials', 100)
            self.n_jobs = optim_config.get('n_jobs', -1)
            self.timeout = optim_config.get('timeout', None)
            self.gc_after_trial = optim_config.get('gc_after_trial', True)
            self.sampler_type = optim_config.get('sampler', 'tpe')
            self.n_startup_trials = optim_config.get('n_startup_trials', 10)
            self.n_ei_candidates = optim_config.get('n_ei_candidates', 24)
            self.early_stopping_n_trials = optim_config.get('early_stopping_n_trials', None)
            self.early_stopping_threshold = optim_config.get('early_stopping_threshold', 0.0)
            self.min_trades = optim_config.get('min_trades', 10)
            self.score_weights = optim_config.get('score_weights', {
                'roi': 2.5,
                'win_rate': 0.5,
                'max_drawdown': 2.0,
                'profit_factor': 2.0,
                'total_trades': 1.0,
                'avg_profit': 1.0
            })
            self.debug = optim_config.get('debug', False)
            
            # Préparation des données
            self.prepare_data(data_path)
            
            # Configuration du storage
            storage_url = f"sqlite:///{study_name}_optimization.db"
            storage = optuna.storages.RDBStorage(
                url=storage_url,
                engine_kwargs={
                    'connect_args': {'timeout': 300},
                    'pool_size': 1
                }
            )
            
            # Configuration du sampler
            if self.sampler_type == 'random':
                sampler = RandomSampler()
            else:  # TPE par défaut
                sampler = TPESampler(
                    n_startup_trials=self.n_startup_trials,
                    n_ei_candidates=self.n_ei_candidates
                )
            
            # Configuration du pruner (arrêt précoce)
            pruner = None
            if self.early_stopping_n_trials:
                pruner = optuna.pruners.MedianPruner(
                    n_startup_trials=self.early_stopping_n_trials,
                    n_warmup_steps=0,
                    interval_steps=1,
                    n_min_trials=self.early_stopping_n_trials
                )
            
            # Création ou chargement de l'étude
            study = optuna.create_study(
                storage=storage,
                sampler=sampler,
                pruner=pruner,
                study_name=f"{study_name}_opt",
                direction="maximize",
                load_if_exists=True
            )
            
            # Forcer le mode séquentiel pour les tests
            logger.info(f"Exécution en mode séquentiel avec {self.n_trials} trials")
            
            # Initialiser les variables globales
            global _shared_data, _trading_config
            _shared_data = self.shared_data
            _trading_config = trading_config
            
            # Exécuter l'optimisation en mode séquentiel
            start_time = time.time()
            study.optimize(
                self.objective,
                n_trials=self.n_trials,
                n_jobs=1,  # Forcer à 1 pour éviter les problèmes de multiprocessing
                callbacks=[lambda study, trial: logger.info(f"Trial {trial.number}: {trial.value}") if trial.value is not None and trial.value > float('-inf') else None],
                gc_after_trial=self.gc_after_trial,
                catch=(Exception,)
            )
            execution_time = time.time() - start_time
            
            logger.info(f"Optimisation terminée en {execution_time:.2f} secondes ({execution_time/60:.2f} minutes)")
            
            # Affichage des résultats
            best_trial = study.best_trial
            logger.info(f"Meilleur trial: {best_trial.number}")
            logger.info(f"Meilleur score: {best_trial.value}")
            
            if 'roi' in best_trial.user_attrs:
                logger.info(f"ROI: {best_trial.user_attrs['roi']*100:.2f}%")
                logger.info(f"Win Rate: {best_trial.user_attrs['win_rate']*100:.2f}%")
                logger.info(f"Trades: {best_trial.user_attrs['total_trades']}")
                logger.info(f"Max Drawdown: {best_trial.user_attrs['max_drawdown']*100:.2f}%")
            
            # Sauvegarde des résultats
            self.save_best_strategies(study_name, study)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation de la stratégie: {str(e)}")
            traceback.print_exc()
            return False
        finally:
            # Nettoyage des ressources
            self.cleanup_shared_data()
            gc.collect()

    def save_best_strategies(self, study_name: str, study: optuna.Study) -> bool:
        """
        Sauvegarde les meilleures stratégies d'une étude Optuna.
        
        Args:
            study_name: Nom de l'étude
            study: Étude Optuna
            
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        try:
            # Récupération des meilleurs trials
            best_trials = sorted(
                [t for t in study.trials if t.value is not None],
                key=lambda t: t.value if t.value is not None else float('-inf'),
                reverse=True
            )[:10]  # Top 10
            
            if not best_trials:
                logger.error(f"Aucun trial valide trouvé pour l'étude '{study_name}'")
                return False
            
            # Récupération de la configuration de trading
            trading_config = self.study_manager.get_trading_config(study_name)
            if trading_config is None:
                logger.error(f"Impossible de récupérer la configuration de trading pour l'étude '{study_name}'")
                return False
            
            # Sauvegarde des résultats d'optimisation
            optimization_results = {
                'study_name': study_name,
                'optimization_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'n_trials': len(study.trials),
                'best_trial_id': best_trials[0].number,
                'best_score': best_trials[0].value,
                'best_trials': [{
                    'trial_id': t.number,
                    'score': t.value,
                    'params': t.params,
                    'metrics': {
                        k: v for k, v in t.user_attrs.items()
                        if k in ['roi', 'win_rate', 'total_trades', 'max_drawdown', 'profit_factor', 'avg_profit']
                    }
                } for t in best_trials]
            }
            
            self.study_manager.save_optimization_results(study_name, optimization_results)
            
            # Sauvegarde des meilleures stratégies
            for i, trial in enumerate(best_trials[:5]):  # Enregistrer les 5 meilleures
                # Création du gestionnaire de blocs
                block_manager = BlockManager(DummyTrial(trial.params), trading_config)
                buy_blocks, sell_blocks = block_manager.generate_blocks()
                
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
                    if key in ['roi', 'win_rate', 'total_trades', 'max_drawdown', 'profit_factor', 'avg_profit']:
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
                
                # Nettoyage
                signal_generator.cleanup()
            
            logger.info(f"Meilleures stratégies sauvegardées pour l'étude '{study_name}'")
            
            # Mise à jour du statut de l'étude
            self.study_manager.update_study_status(study_name, "optimized")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des meilleures stratégies: {str(e)}")
            traceback.print_exc()
            return False

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