"""
Module qui définit la tâche d'optimisation pour un trial individuel.
Cette classe est conçue pour être facilement sérialisable et utilisable
dans un contexte multiprocessus.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import time
import traceback
from typing import Dict, Any, Optional, Union, Tuple

from core.strategy.constructor.constructor import StrategyConstructor
from core.optimization.search_config import SearchSpace
from core.optimization.score_config import ScoreCalculator
from core.optimization.selector import create_strategy_from_trial

logger = logging.getLogger(__name__)

class OptimizerTask:
    """
    Classe représentant une tâche d'optimisation pour un trial spécifique.
    Cette classe est conçue pour être exécutée dans un processus séparé.
    """
    
    def __init__(
        self,
        trial_id: int,
        search_space: Union[Dict, SearchSpace],
        study_path: str,
        data_path: str,
        scoring_formula: str = "standard",
        min_trades: int = 10,
        seed: Optional[int] = None,
        debug: bool = False
    ):
        """
        Initialise une tâche d'optimisation.
        
        Args:
            trial_id: Identifiant du trial
            search_space: Espace de recherche (dictionnaire ou objet SearchSpace)
            study_path: Chemin vers le répertoire de l'étude
            data_path: Chemin vers les données pour le backtest
            scoring_formula: Formule de scoring à utiliser
            min_trades: Nombre minimum de trades pour qu'un trial soit valide
            seed: Graine pour la reproductibilité
            debug: Mode debug
        """
        self.trial_id = trial_id
        self.search_space = search_space if isinstance(search_space, SearchSpace) else SearchSpace.from_dict(search_space)
        self.study_path = study_path
        self.data_path = data_path
        self.scoring_formula = scoring_formula
        self.min_trades = min_trades
        self.seed = seed
        self.debug = debug
        
        # État interne
        self.start_time = None
        self.end_time = None
        self.status = "pending"
        self.result = None
        self.error = None
        
        # Outils de scoring
        self.score_calculator = ScoreCalculator(self.scoring_formula)
    
    def run(self) -> Dict[str, Any]:
        """
        Exécute la tâche d'optimisation.
        
        Returns:
            Dict: Résultats de l'optimisation
        """
        self.start_time = time.time()
        self.status = "running"
        
        try:
            # Configuration de la graine aléatoire si spécifiée
            if self.seed is not None:
                np.random.seed(self.seed + self.trial_id)
            
            # Chargement des données
            data = self._load_data()
            if data is None:
                raise ValueError(f"Impossible de charger les données depuis {self.data_path}")
            
            # Création d'un trial factice pour simuler Optuna
            trial = self._create_dummy_trial()
            
            # Création de la stratégie à partir du trial
            constructor = create_strategy_from_trial(trial, self.search_space)
            constructor.set_name(f"Strategy_Trial_{self.trial_id}")
            
            # Génération des signaux
            signals, data_with_signals = constructor.generate_signals(data)
            
            # Simulation de trading
            simulation_results = self._run_simulation(data_with_signals, signals)
            
            # Calcul du score
            score, metrics, is_valid = self._calculate_score(simulation_results)
            
            # Sauvegarde des résultats
            saved_files, backtest_id, strategy_id = self._save_results(
                constructor, simulation_results, is_valid
            )
            
            # Préparation du résultat
            self.result = {
                'trial_id': self.trial_id,
                'params': trial.params,
                'score': score,
                'metrics': metrics,
                'performance': simulation_results.get('performance', {}),
                'strategy_id': strategy_id,
                'backtest_id': backtest_id,
                'saved_files': saved_files,
                'valid': is_valid,
                'execution_time': time.time() - self.start_time
            }
            
            self.status = "completed"
            self.end_time = time.time()
            return self.result
        
        except Exception as e:
            self.status = "error"
            self.error = str(e)
            self.end_time = time.time()
            
            logger.error(f"Erreur dans le trial {self.trial_id}: {str(e)}")
            traceback.print_exc()
            
            return {
                'trial_id': self.trial_id,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'status': 'error',
                'execution_time': time.time() - self.start_time
            }
    
    def _load_data(self) -> Optional[pd.DataFrame]:
        """
        Charge les données pour le backtest.
        
        Returns:
            Optional[pd.DataFrame]: DataFrame avec les données ou None
        """
        try:
            if os.path.exists(self.data_path):
                data = pd.read_csv(self.data_path, index_col=0)
                try:
                    data.index = pd.to_datetime(data.index)
                except:
                    pass
                return data
            else:
                logger.error(f"Fichier de données introuvable: {self.data_path}")
                return None
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
            return None
    
    def _create_dummy_trial(self) -> Any:
        """
        Crée un trial factice pour simuler Optuna.
        
        Returns:
            Any: Objet trial factice
        """
        class DummyTrial:
            def __init__(self, trial_id):
                self.number = trial_id
                self.params = {}
                self.user_attrs = {}
            
            def suggest_categorical(self, name, choices):
                import random
                value = random.choice(choices)
                self.params[name] = value
                return value
            
            def suggest_int(self, name, low, high, step=1, log=False):
                import random
                import math
                if log:
                    value = int(math.exp(random.uniform(math.log(max(1, low)), math.log(high))))
                else:
                    value = random.randrange(low, high+1, step)
                self.params[name] = value
                return value
            
            def suggest_float(self, name, low, high, step=None, log=False):
                import random
                import math
                if log:
                    value = math.exp(random.uniform(math.log(max(1e-10, low)), math.log(high)))
                else:
                    if step:
                        n_steps = int((high - low) / step)
                        random_step = random.randint(0, n_steps)
                        value = low + random_step * step
                    else:
                        value = random.uniform(low, high)
                self.params[name] = value
                return value
            
            def set_user_attr(self, key, value):
                self.user_attrs[key] = value
        
        return DummyTrial(self.trial_id)
    
    def _run_simulation(self, data_with_signals: pd.DataFrame, signals: np.ndarray) -> Dict[str, Any]:
        """
        Exécute la simulation de trading.
        
        Args:
            data_with_signals: DataFrame avec les signaux et indicateurs
            signals: Array des signaux de trading
        
        Returns:
            Dict: Résultats de la simulation
        """
        from core.simulation.simulation_config import SimulationConfig
        from core.simulation.simulator import Simulator
        
        sim_config = SimulationConfig()
        simulator = Simulator(sim_config)
        
        position_sizes = data_with_signals['position_size'].values if 'position_size' in data_with_signals.columns else None
        sl_levels = data_with_signals['sl_level'].values if 'sl_level' in data_with_signals.columns else None
        tp_levels = data_with_signals['tp_level'].values if 'tp_level' in data_with_signals.columns else None
        
        results = simulator.run(
            prices=data_with_signals['close'].values,
            signals=signals,
            position_sizes=position_sizes,
            sl_levels=sl_levels,
            tp_levels=tp_levels
        )
        
        return results
    
    def _calculate_score(self, simulation_results: Dict[str, Any]) -> Tuple[float, Dict[str, float], bool]:
        """
        Calcule le score de la stratégie.
        
        Args:
            simulation_results: Résultats de la simulation
        
        Returns:
            Tuple[float, Dict[str, float], bool]: (score, métriques, validité)
        """
        performance = simulation_results.get('performance', {})
        total_trades = performance.get('total_trades', 0)
        
        if total_trades < self.min_trades:
            logger.info(f"Trial {self.trial_id}: Nombre de trades insuffisant ({total_trades} < {self.min_trades})")
            metrics = {
                'roi': performance.get('roi', 0),
                'win_rate': performance.get('win_rate', 0),
                'max_drawdown': performance.get('max_drawdown', 1),
                'profit_factor': performance.get('profit_factor', 0),
                'total_trades': total_trades,
                'sharpe_ratio': performance.get('sharpe_ratio', 0),
                'trades_per_day': performance.get('trades_per_day', 0),
                'avg_profit': performance.get('avg_profit_per_trade', 0),
                'max_consecutive_losses': performance.get('max_consecutive_losses', 0)
            }
            return float('-inf'), metrics, False
        
        metrics = {
            'roi': performance.get('roi', 0),
            'win_rate': performance.get('win_rate', 0),
            'max_drawdown': performance.get('max_drawdown', 1),
            'profit_factor': performance.get('profit_factor', 0),
            'total_trades': total_trades,
            'sharpe_ratio': performance.get('sharpe_ratio', 0),
            'trades_per_day': performance.get('trades_per_day', 0),
            'avg_profit': performance.get('avg_profit_per_trade', 0),
            'max_consecutive_losses': performance.get('max_consecutive_losses', 0)
        }
        
        score = self.score_calculator.calculate_score(metrics)
        return score, metrics, True
    
    def _save_results(
        self, 
        constructor: StrategyConstructor, 
        simulation_results: Dict[str, Any],
        is_valid: bool
    ) -> Tuple[List[str], str, str]:
        """
        Sauvegarde les résultats de la simulation.
        
        Args:
            constructor: Constructeur de stratégie
            simulation_results: Résultats de la simulation
            is_valid: Si la stratégie est valide
        
        Returns:
            Tuple[List[str], str, str]: (fichiers sauvegardés, ID du backtest, ID de la stratégie)
        """
        strategy_id = constructor.config.id
        backtest_id = f"trial_{self.trial_id}_{int(time.time())}"
        saved_files = []
        
        if self.study_path:
            try:
                # Création des répertoires
                strategy_dir = os.path.join(self.study_path, "strategies", strategy_id)
                os.makedirs(os.path.dirname(strategy_dir), exist_ok=True)
                os.makedirs(strategy_dir, exist_ok=True)
                
                backtest_dir = os.path.join(strategy_dir, "backtests")
                os.makedirs(backtest_dir, exist_ok=True)
                
                # Sauvegarde de la configuration de la stratégie
                config_path = os.path.join(strategy_dir, "config.json")
                constructor.save(config_path)
                saved_files.append(config_path)
                
                # Sauvegarde des résultats du backtest
                base_filepath = os.path.join(backtest_dir, backtest_id)
                from core.simulation.simulator import Simulator
                sim_config = SimulationConfig()
                simulator = Simulator(sim_config)
                
                history_files = simulator.save_to_csv(base_filepath)
                saved_files.extend(history_files)
                
                # Sauvegarde du résumé du backtest
                json_path = os.path.join(backtest_dir, f"{backtest_id}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    backtest_summary = {
                        "backtest_id": backtest_id,
                        "strategy_id": strategy_id,
                        "date": datetime.now().isoformat(),
                        "performance": simulation_results.get('performance', {}),
                        "execution_time": simulation_results.get('execution_time', 0),
                        "valid": is_valid,
                        "trial_id": self.trial_id
                    }
                    json.dump(backtest_summary, f, indent=2)
                saved_files.append(json_path)
                
                logger.info(f"Trial {self.trial_id}: Résultats sauvegardés dans {len(saved_files)} fichiers")
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde des résultats: {str(e)}")
        
        return saved_files, backtest_id, strategy_id