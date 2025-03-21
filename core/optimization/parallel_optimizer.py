"""Optimiseur parallèle pour les stratégies de trading.Responsable de l'optimisation des paramètres des stratégies."""
import os
import json
import logging
import traceback
import time
import uuid
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import multiprocessing as mp
from multiprocessing import Manager, Queue, Event
import numpy as np
import pandas as pd
import optuna
import gc
import psutil
import platform
import pickle
from core.optimization.search_config import SearchSpace, get_predefined_search_space
from core.optimization.score_config import ScoreCalculator

logger = logging.getLogger(__name__)

class ProcessMessage:
    def __init__(self, process_id: int, message_type: str, content: Any, timestamp: float = None):
        self.process_id = process_id
        self.message_type = message_type
        self.content = content
        self.timestamp = timestamp or time.time()
        
    def to_dict(self) -> Dict:
        return {
            "process_id": self.process_id,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp
        }

class OptimizationConfig:
    """Configuration for trading strategy optimization."""
    def __init__(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        search_space: Optional[Union[Dict, SearchSpace]] = None,
        optimization_method: str = "tpe",
        method_params: Optional[Dict] = None,
        enable_pruning: bool = True,
        pruner_method: str = "median",
        pruner_params: Optional[Dict] = None,
        early_stopping_n_trials: Optional[int] = None,
        scoring_formula: str = "standard",
        min_trades: int = 10,
        n_jobs: int = -1,
        memory_limit: float = 0.8,
        save_checkpoints: bool = True,
        checkpoint_every: int = 10,
        debug: bool = False,
        silent: bool = False
    ):
        """
        Initialize the optimization configuration.
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds (optional)
            search_space: Search space
            optimization_method: Optimization method
            method_params: Parameters for the optimization method
            enable_pruning: Enable pruning
            pruner_method: Pruning method
            pruner_params: Parameters for the pruning method
            early_stopping_n_trials: Number of trials for early stopping
            scoring_formula: Scoring formula
            min_trades: Minimum number of trades for a valid trial
            n_jobs: Number of parallel processes (-1 for auto)
            memory_limit: Memory limit as a fraction of total memory
            save_checkpoints: Save checkpoints
            checkpoint_every: Checkpoint frequency
            debug: Debug mode
            silent: Disable stdout output from processes
        """
        self.n_trials = n_trials
        self.timeout = timeout
        if search_space is None:
            from core.optimization.search_config import get_predefined_search_space
            self.search_space = get_predefined_search_space("default")
        elif isinstance(search_space, dict):
            from core.optimization.search_config import SearchSpace
            self.search_space = SearchSpace.from_dict(search_space)
        else:
            self.search_space = search_space
        self.optimization_method = optimization_method
        self.method_params = method_params or {}
        self.enable_pruning = enable_pruning
        self.pruner_method = pruner_method
        self.pruner_params = pruner_params or {}
        self.early_stopping_n_trials = early_stopping_n_trials
        self.scoring_formula = scoring_formula
        self.min_trades = min_trades
        self.n_jobs = n_jobs
        self.memory_limit = memory_limit
        self.save_checkpoints = save_checkpoints
        self.checkpoint_every = checkpoint_every
        self.debug = debug
        self.silent = silent
        
    def to_dict(self) -> Dict:
        """Convert the configuration to a dictionary"""
        result = {
            "n_trials": self.n_trials,
            "timeout": self.timeout,
            "search_space": self.search_space.to_dict() if hasattr(self.search_space, 'to_dict') else self.search_space,
            "optimization_method": self.optimization_method,
            "method_params": self.method_params,
            "enable_pruning": self.enable_pruning,
            "pruner_method": self.pruner_method,
            "pruner_params": self.pruner_params,
            "early_stopping_n_trials": self.early_stopping_n_trials,
            "scoring_formula": self.scoring_formula,
            "min_trades": self.min_trades,
            "n_jobs": self.n_jobs,
            "memory_limit": self.memory_limit,
            "save_checkpoints": self.save_checkpoints,
            "checkpoint_every": self.checkpoint_every,
            "debug": self.debug,
            "silent": self.silent
        }
        return result
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'OptimizationConfig':
        """Create a configuration from a dictionary"""
        search_space = data.get("search_space")
        if isinstance(search_space, dict):
            from core.optimization.search_config import SearchSpace
            search_space = SearchSpace.from_dict(search_space)
        return cls(
            n_trials=data.get("n_trials", 100),
            timeout=data.get("timeout"),
            search_space=search_space,
            optimization_method=data.get("optimization_method", "tpe"),
            method_params=data.get("method_params", {}),
            enable_pruning=data.get("enable_pruning", True),
            pruner_method=data.get("pruner_method", "median"),
            pruner_params=data.get("pruner_params", {}),
            early_stopping_n_trials=data.get("early_stopping_n_trials"),
            scoring_formula=data.get("scoring_formula", "standard"),
            min_trades=data.get("min_trades", 10),
            n_jobs=data.get("n_jobs", -1),
            memory_limit=data.get("memory_limit", 0.8),
            save_checkpoints=data.get("save_checkpoints", True),
            checkpoint_every=data.get("checkpoint_every", 10),
            debug=data.get("debug", False),
            silent=data.get("silent", False)
        )

def process_trial_standalone(
    trial_id: int,
    db_path: str,
    config_dict: Dict,
    message_queue: mp.Queue,
    stop_event: mp.Event,
    process_id: int
):
    """
    Standalone function to process a trial in a separate process.
    This avoids issues with pickling the main class.
    """
    try:
        print(f"Process {process_id}: Starting trial {trial_id}")
        message_queue.put(ProcessMessage(
            process_id=process_id,
            message_type="progress",
            content=f"Process {process_id} starting trial {trial_id}"
        ))
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        print(f"Process {process_id}: Connecting to database {db_path}")
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"Process {process_id}: Database tables: {tables}")
            message_queue.put(ProcessMessage(
                process_id=process_id,
                message_type="progress",
                content=f"Process {process_id} found tables: {[t[0] for t in tables]}"
            ))
        except Exception as e:
            print(f"Process {process_id}: Database connection error: {str(e)}")
            message_queue.put(ProcessMessage(
                process_id=process_id,
                message_type="error",
                content=f"Database error: {str(e)}"
            ))
            raise
            
        try:
            prices = pd.read_sql('SELECT * FROM data_close', conn)['close'].values
            high = pd.read_sql('SELECT * FROM data_high', conn)['high'].values
            low = pd.read_sql('SELECT * FROM data_low', conn)['low'].values
            print(f"Process {process_id}: Loaded price data: {len(prices)} points")
            try:
                volume = pd.read_sql('SELECT * FROM data_volume', conn)['volume'].values
                print(f"Process {process_id}: Loaded volume data: {len(volume)} points")
            except Exception as e:
                print(f"Process {process_id}: No volume data available: {str(e)}")
                volume = None
        except Exception as e:
            print(f"Process {process_id}: Error loading price data: {str(e)}")
            message_queue.put(ProcessMessage(
                process_id=process_id,
                message_type="error",
                content=f"Error loading price data: {str(e)}"
            ))
            raise
            
        study_path = None
        try:
            metadata_df = pd.read_sql('SELECT * FROM metadata', conn)
            if 'value' in metadata_df.columns:
                metadata_str = metadata_df.iloc[0]['value']
                metadata = json.loads(metadata_str)
                if 'study_path' in metadata:
                    study_path = metadata['study_path']
                    print(f"Process {process_id}: Found study path: {study_path}")
        except Exception as e:
            print(f"Process {process_id}: Error reading metadata: {str(e)}")
            
        conn.close()
        
        class DummyTrial:
            def __init__(self, trial_id, params=None):
                self.params = params or {}
                self.number = trial_id
                self.user_attrs = {}
                
            def suggest_categorical(self, name, choices):
                if name in self.params:
                    return self.params[name]
                import random
                return random.choice(choices)
                
            def suggest_int(self, name, low, high, step=1, log=False):
                if name in self.params:
                    return self.params[name]
                import random
                step = max(1, int(step)) if step else 1
                if log:
                    import math
                    return int(math.exp(random.uniform(math.log(max(1, low)), math.log(high))))
                else:
                    return random.randrange(low, high+1, step)
                    
            def suggest_float(self, name, low, high, step=None, log=False):
                if name in self.params:
                    return self.params[name]
                import random
                if log:
                    import math
                    return math.exp(random.uniform(math.log(max(1e-10, low)), math.log(high)))
                else:
                    if step:
                        n_steps = int((high - low) / step)
                        random_step = random.randint(0, n_steps)
                        return low + random_step * step
                    else:
                        return random.uniform(low, high)
                        
            def set_user_attr(self, key, value):
                self.user_attrs[key] = value
                
        try:
            from core.optimization.search_config import SearchSpace
            from core.optimization.selector import ParameterSelector
            from core.optimization.score_config import ScoreCalculator
            
            trial = DummyTrial(trial_id)
            search_space = SearchSpace.from_dict(config_dict['search_space'])
            selector = ParameterSelector(search_space)
            params = selector.suggest_all_parameters(trial)
            
            print(f"Process {process_id}: Generated {len(params)} parameters for trial {trial_id}")
            message_queue.put(ProcessMessage(
                process_id=process_id,
                message_type="progress",
                content=f"Process {process_id} generated {len(params)} parameters for trial {trial_id}"
            ))
        except Exception as e:
            print(f"Process {process_id}: Error generating parameters: {str(e)}")
            message_queue.put(ProcessMessage(
                process_id=process_id,
                message_type="error",
                content=f"Parameter error: {str(e)}"
            ))
            raise
            
        try:
            from core.strategy.constructor.constructor import StrategyConstructor
            from core.optimization.selector import create_strategy_from_trial
            
            constructor = create_strategy_from_trial(trial, search_space)
            print(f"Process {process_id}: Created strategy for trial {trial_id}")
            message_queue.put(ProcessMessage(
                process_id=process_id,
                message_type="progress",
                content=f"Process {process_id} created strategy for trial {trial_id}"
            ))
        except Exception as e:
            print(f"Process {process_id}: Error creating strategy: {str(e)}")
            message_queue.put(ProcessMessage(
                process_id=process_id,
                message_type="error",
                content=f"Strategy creation error: {str(e)}"
            ))
            raise
            
        try:
            df = pd.DataFrame({
                'close': prices,
                'high': high,
                'low': low
            })
            signals, data_with_signals = constructor.generate_signals(df)
            print(f"Process {process_id}: Generated signals for trial {trial_id}")
            message_queue.put(ProcessMessage(
                process_id=process_id,
                message_type="progress",
                content=f"Process {process_id} generated signals for trial {trial_id}"
            ))
        except Exception as e:
            print(f"Process {process_id}: Error generating signals: {str(e)}")
            message_queue.put(ProcessMessage(
                process_id=process_id,
                message_type="error",
                content=f"Signal generation error: {str(e)}"
            ))
            raise
            
        position_sizes = data_with_signals['position_size'].values if 'position_size' in data_with_signals.columns else None
        sl_levels = data_with_signals['sl_level'].values if 'sl_level' in data_with_signals.columns else None
        tp_levels = data_with_signals['tp_level'].values if 'tp_level' in data_with_signals.columns else None
        
        try:
            from core.simulation.simulation_config import SimulationConfig
            from core.simulation.simulator import Simulator
            
            sim_config = SimulationConfig()
            simulator = Simulator(sim_config)
            results = simulator.run(
                prices=prices,
                signals=signals,
                position_sizes=position_sizes,
                sl_levels=sl_levels,
                tp_levels=tp_levels
            )
            print(f"Process {process_id}: Completed simulation for trial {trial_id}")
            message_queue.put(ProcessMessage(
                process_id=process_id,
                message_type="progress",
                content=f"Process {process_id} completed simulation for trial {trial_id}"
            ))
        except Exception as e:
            print(f"Process {process_id}: Error running simulation: {str(e)}")
            message_queue.put(ProcessMessage(
                process_id=process_id,
                message_type="error",
                content=f"Simulation error: {str(e)}"
            ))
            raise
            
        performance = results.get('performance', {})
        total_trades = performance.get('total_trades', 0)
        min_trades = config_dict.get('min_trades', 10)
        print(f"Process {process_id}: Trial {trial_id} had {total_trades} trades (min required: {min_trades})")
        
        # Sauvegarder les fichiers même pour les trials invalides
        saved_files = []
        backtest_id = None
        strategy_id = constructor.config.id if hasattr(constructor, 'config') else f"strategy_trial_{trial_id}"
        
        if study_path:
            try:
                backtest_id = f"trial_{trial_id}_{int(time.time())}"
                backtest_dir = os.path.join(study_path, "strategies", strategy_id, "backtests")
                os.makedirs(os.path.dirname(backtest_dir), exist_ok=True)
                os.makedirs(backtest_dir, exist_ok=True)
                base_filepath = os.path.join(backtest_dir, backtest_id)
                saved_files = simulator.save_to_csv(base_filepath)
                print(f"Process {process_id}: Saved {len(saved_files)} history files for trial {trial_id}")
                message_queue.put(ProcessMessage(
                    process_id=process_id,
                    message_type="progress",
                    content=f"Trial {trial_id}: Saved {len(saved_files)} history files"
                ))
            except Exception as e:
                print(f"Process {process_id}: Error saving history files: {str(e)}")
                message_queue.put(ProcessMessage(
                    process_id=process_id,
                    message_type="error",
                    content=f"Error saving histories for trial {trial_id}: {str(e)}"
                ))
        
        # Si nombre de trades insuffisant, on attribue un score -inf mais on sauvegarde quand même
        if total_trades < min_trades:
            message_queue.put(ProcessMessage(
                process_id=process_id,
                message_type="progress",
                content=f"Trial {trial_id}: Insufficient trades ({total_trades} < {min_trades}), scoring as -inf"
            ))
            
            # Création des métriques et résultat avec score -inf
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
            
            result_msg = {
                'trial_id': trial_id,
                'params': params,
                'score': float('-inf'),  # Score à -inf
                'metrics': metrics,
                'performance': performance,
                'strategy_id': strategy_id,
                'backtest_id': backtest_id,
                'saved_files': saved_files,
                'valid': False  # Indicateur d'invalidité
            }
            
            message_queue.put(ProcessMessage(
                process_id=process_id,
                message_type="result",
                content=result_msg
            ))
            
            message_queue.put(ProcessMessage(
                process_id=process_id,
                message_type="progress",
                content=f"Trial {trial_id}: Invalid (insufficient trades), score = -inf"
            ))
            
            print(f"Process {process_id}: Trial {trial_id} marked as invalid but saved with score -inf")
            return
            
        try:
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
            
            formula = config_dict.get('scoring_formula', 'standard')
            score_calculator = ScoreCalculator(formula)
            score = score_calculator.calculate_score(metrics)
            print(f"Process {process_id}: Trial {trial_id} scored {score:.4f}")
        except Exception as e:
            print(f"Process {process_id}: Error calculating score: {str(e)}")
            message_queue.put(ProcessMessage(
                process_id=process_id,
                message_type="error",
                content=f"Score calculation error: {str(e)}"
            ))
            raise
            
        result_msg = {
            'trial_id': trial_id,
            'params': params,
            'score': score,
            'metrics': metrics,
            'performance': performance,
            'strategy_id': strategy_id,
            'backtest_id': backtest_id,
            'saved_files': saved_files,
            'valid': True
        }
        
        message_queue.put(ProcessMessage(
            process_id=process_id,
            message_type="result",
            content=result_msg
        ))
        
        message_queue.put(ProcessMessage(
            process_id=process_id,
            message_type="progress",
            content=f"Trial {trial_id}: Score = {score:.4f}, ROI = {metrics['roi']:.4f}, Trades = {total_trades}"
        ))
        
        print(f"Process {process_id}: Trial {trial_id} completed successfully")
    except Exception as e:
        print(f"Process {process_id}: Error in trial {trial_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        error_msg = f"Error in trial {trial_id}: {str(e)}\n{traceback.format_exc()}"
        message_queue.put(ProcessMessage(
            process_id=process_id,
            message_type="error",
            content=error_msg
        ))
    finally:
        gc.collect()

class ParallelOptimizer:
    """Optimiseur parallèle pour les stratégies de trading"""
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialise l'optimiseur parallèle.
        Args:
            config: Configuration d'optimisation
        """
        self.config = config or OptimizationConfig()
        self.manager = Manager()
        self.optimization_progress = self.manager.dict()
        self.message_queue = self.manager.Queue()
        self.stop_event = self.manager.Event()
        self.messages = []
        self.process_results = {}
        
    def create_shared_db(self, data: pd.DataFrame, db_path: str, study_path: str = None, force_recreate: bool = False) -> str:
        """
        Creates a SQLite database for sharing data between processes.
        Args:
            data: DataFrame with the data
            db_path: Path to the database
            study_path: Path to the study directory (optional)
            force_recreate: Force recreation of the database even if it exists
        Returns:
            str: Path to the database
        """
        if os.path.exists(db_path) and not force_recreate:
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [t[0] for t in cursor.fetchall()]
                required_tables = ['data_close', 'data_high', 'data_low', 'metadata']
                if all(table in tables for table in required_tables):
                    cursor.execute("SELECT COUNT(*) FROM data_close")
                    count = cursor.fetchone()[0]
                    if count > 0:
                        if study_path:
                            try:
                                cursor.execute("SELECT value FROM metadata")
                                metadata_str = cursor.fetchone()[0]
                                metadata = json.loads(metadata_str)
                                if 'study_path' not in metadata or metadata['study_path'] != study_path:
                                    metadata['study_path'] = study_path
                                    cursor.execute("UPDATE metadata SET value = ?",
                                        (json.dumps(metadata),))
                                    conn.commit()
                            except Exception as e:
                                logger.warning(f"Impossible de mettre à jour le chemin d'étude dans les métadonnées: {str(e)}")
                        conn.close()
                        logger.info(f"Réutilisation de la base de données existante: {db_path}")
                        return db_path
                conn.close()
            except Exception as e:
                logger.warning(f"Erreur lors de la vérification de la base de données existante: {str(e)}")
                
        conn = sqlite3.connect(db_path)
        required_columns = ['close', 'high', 'low']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}. Attempting to adapt...")
            if 'close' not in data.columns and 'Close' in data.columns:
                data['close'] = data['Close']
            if 'high' not in data.columns and 'High' in data.columns:
                data['high'] = data['High']
            if 'low' not in data.columns and 'Low' in data.columns:
                data['low'] = data['Low']
            if 'volume' not in data.columns and 'Volume' in data.columns:
                data['volume'] = data['Volume']
                
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Data must contain at least these columns: {required_columns}")
                
        for col in ['close', 'high', 'low']:
            pd.DataFrame({col: data[col].values}).to_sql(
                f'data_{col}',
                conn,
                if_exists='replace',
                index=False
            )
            
        if 'volume' in data.columns:
            pd.DataFrame({'volume': data['volume'].values}).to_sql(
                'data_volume',
                conn,
                if_exists='replace',
                index=False
            )
            
        metadata = {
            'shape': len(data),
            'columns': list(data.columns),
            'timestamp': datetime.now().isoformat(),
            'study_path': study_path,
            'db_id': str(uuid.uuid4())
        }
        metadata_df = pd.DataFrame({'value': [json.dumps(metadata)]})
        metadata_df.to_sql('metadata', conn, if_exists='replace', index=False)
        
        conn.close()
        logger.info(f"Base de données créée avec succès: {db_path}")
        return db_path

    def run_optimization(self, study_path: str, data: Union[str, pd.DataFrame]) -> Tuple[bool, Optional[Dict]]:
        """
        Runs the optimization for a given study.
        Args:
            study_path: Path to the study directory
            data: DataFrame or path to CSV file
        Returns:
            Tuple[bool, Optional[Dict]]: (success, results)
        """
        try:
            if not os.path.exists(study_path):
                logger.error(f"Study path '{study_path}' does not exist")
                return False, None
                
            if isinstance(data, str):
                try:
                    data = pd.read_csv(data, index_col=0)
                    try:
                        data.index = pd.to_datetime(data.index)
                    except:
                        pass
                except Exception as e:
                    logger.error(f"Error loading data: {str(e)}")
                    return False, None
                    
            study_name = os.path.basename(study_path)
            
            # Import du gestionnaire des configurations d'optimisation
            from core.optimization.optimization_range_manager import create_optimization_range_manager
            
            # Récupération de la configuration de recherche
            file_name = "search_config"
            range_manager = create_optimization_range_manager(study_path)
            search_space = range_manager.load_config(file_name)
            
            if not search_space:
                logger.error(f"Impossible de charger l'espace de recherche pour l'étude '{study_name}'")
                return False, None
            
            from core.study.study_config import StudyConfig
            study_config_path = os.path.join(study_path, "config.json")
            try:
                with open(study_config_path, 'r') as f:
                    study_config_dict = json.load(f)
                study_config = StudyConfig.from_dict(study_config_dict)
            except Exception as e:
                logger.warning(f"Impossible de charger la configuration de l'étude: {str(e)}")
                study_config = None
                
            self.optimization_progress[study_name] = {
                'status': 'running',
                'start_time': datetime.now().isoformat(),
                'completed': 0,
                'total': self.config.n_trials,
                'best_value': None,
                'best_metrics': {},
                'error_message': None,
                'trial_results': {},
                'messages': [],
                'last_update': datetime.now().isoformat()
            }
            
            # Mise à jour de l'espace de recherche dans la configuration d'optimisation
            self.config.search_space = search_space
            
            optim_dir = os.path.join(study_path, "optimizations")
            os.makedirs(optim_dir, exist_ok=True)
            
            optuna_db_path = None
            if study_config and hasattr(study_config.metadata, 'optuna_db_path'):
                optuna_db_path = study_config.metadata.optuna_db_path
                
            if not optuna_db_path:
                optuna_db_file = os.path.join(optim_dir, "optuna.db")
                optuna_db_path = f"sqlite:///{optuna_db_file}"
                
            if study_config:
                study_config.metadata.optuna_db_path = optuna_db_path
                try:
                    with open(study_config_path, 'w') as f:
                        json.dump(study_config.to_dict(), f, indent=2)
                except Exception as e:
                    logger.warning(f"Impossible de mettre à jour la configuration de l'étude: {str(e)}")
                    
            existing_data_db = None
            for file in os.listdir(optim_dir):
                if file.startswith("study_data_") and file.endswith(".db"):
                    existing_data_db = os.path.join(optim_dir, file)
                    logger.info(f"Base de données de données trouvée: {existing_data_db}")
                    break
                    
            if existing_data_db:
                db_path = existing_data_db
                self.create_shared_db(data, db_path, study_path, force_recreate=False)
            else:
                db_path = os.path.join(optim_dir, f"study_data_{study_name}.db")
                self.create_shared_db(data, db_path, study_path, force_recreate=True)
                
            n_jobs = self.config.n_jobs
            if n_jobs < 1:
                n_jobs = max(1, mp.cpu_count() - 1)
            n_jobs = min(n_jobs, mp.cpu_count(), self.config.n_trials)
            
            logger.info(f"Starting optimization for '{study_name}' with {n_jobs} processes")
            
            try:
                import optuna
                from optuna.samplers import TPESampler
                storage = optuna.storages.RDBStorage(optuna_db_path)
                optuna_study_name = f"optimization_{int(time.time())}"
                optuna_study = optuna.create_study(
                    study_name=optuna_study_name,
                    storage=storage,
                    load_if_exists=False,
                    direction="maximize",
                    sampler=TPESampler()
                )
                self.optimization_progress[study_name]['optuna_study_name'] = optuna_study_name
                logger.info(f"Created Optuna study '{optuna_study_name}' with storage {optuna_db_path}")
            except Exception as e:
                logger.warning(f"Impossible d'initialiser Optuna: {str(e)}. L'optimisation utilisera le mode standard.")
                optuna_study = None
                
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                except:
                    pass
                    
            self.stop_event.clear()
            config_dict = self.config.to_dict()
            
            processes = []
            for i in range(min(self.config.n_trials, n_jobs * 2)):
                process = mp.Process(
                    target=process_trial_standalone,
                    args=(i, db_path, config_dict, self.message_queue, self.stop_event, i)
                )
                processes.append(process)
                
            active_processes = 0
            for i in range(min(n_jobs, len(processes))):
                try:
                    processes[i].start()
                    active_processes += 1
                except Exception as e:
                    logger.error(f"Erreur au démarrage du processus initial {i}: {str(e)}")
                    processes[i] = None
                    
            best_score = float('-inf')
            best_trial = None
            trial_results = {}
            next_trial_id = len(processes)
            completed_trials = 0
            
            def process_messages():
                nonlocal best_score, best_trial, completed_trials
                print(f"Processing messages (queue size: {self.message_queue.qsize()} messages)")
                message_processed = 0
                while not self.message_queue.empty():
                    try:
                        msg = self.message_queue.get_nowait()
                        message_processed += 1
                        self.messages.append(msg)
                        print(f"Processed message: {msg.message_type} from process {msg.process_id}")
                        
                        if study_name in self.optimization_progress:
                            self.optimization_progress[study_name]['last_update'] = datetime.now().isoformat()
                            
                            if msg.message_type == "progress":
                                self.optimization_progress[study_name]['messages'].append(str(msg.content))
                                print(f"Added progress message: {msg.content}")
                                
                            elif msg.message_type == "result":
                                result = msg.content
                                trial_id = result['trial_id']
                                score = result['score']
                                trial_results[trial_id] = result
                                print(f"Result for trial {trial_id}: score={score}")
                                
                                # Sauvegarder le backtest même pour les essais invalides
                                if 'backtest_id' in result and result['backtest_id'] and study_path:
                                    try:
                                        strategy_id = result.get('strategy_id')
                                        if not strategy_id:
                                            strategy_id = f"strategy_trial_{trial_id}"
                                        backtest_id = result['backtest_id']
                                        
                                        # Ajouter un indicateur d'invalidité si présent dans le résultat
                                        simulation_results = {
                                            'performance': result.get('performance', {}),
                                            'execution_time': 0
                                        }
                                        
                                        if 'valid' in result and not result['valid']:
                                            simulation_results['valid'] = False
                                        
                                        self.save_backtest_results(
                                            study_path=study_path,
                                            strategy_id=strategy_id,
                                            backtest_id=backtest_id,
                                            simulation_results=simulation_results,
                                            config={
                                                'trial_id': trial_id,
                                                'params': result.get('params', {}),
                                                'optimization': {
                                                    'n_trials': self.config.n_trials,
                                                    'scoring_formula': config_dict.get('scoring_formula', 'standard'),
                                                    'search_space_name': getattr(self.config.search_space, 'name', 'unknown')
                                                }
                                            }
                                        )
                                        
                                        if 'saved_files' in result:
                                            json_path = os.path.join(
                                                study_path, "strategies", strategy_id, "backtests", f"{backtest_id}.json"
                                            )
                                            if json_path not in result['saved_files']:
                                                result['saved_files'].append(json_path)
                                    except Exception as e:
                                        logger.warning(f"Erreur lors de la sauvegarde du JSON pour l'essai {trial_id}: {str(e)}")
                                
                                # Ne pas enregistrer les scores -inf dans Optuna, mais compter quand même l'essai
                                if optuna_study and score != float('-inf'):
                                    try:
                                        dummy_trial = optuna_study.ask()
                                        optuna_study.tell(dummy_trial._trial_id, score)
                                        for param_name, param_value in result['params'].items():
                                            optuna_study._storage.set_trial_param(
                                                dummy_trial._trial_id, param_name, param_value,
                                                param_distribution_type="none"
                                            )
                                        for metric_name, metric_value in result['metrics'].items():
                                            optuna_study._storage.set_trial_user_attr(
                                                dummy_trial._trial_id, f"metric_{metric_name}", metric_value
                                            )
                                    except Exception as e:
                                        logger.warning(f"Impossible d'enregistrer le résultat dans Optuna: {str(e)}")
                                
                                # Ne mettre à jour le meilleur score que si l'essai est valide
                                if score != float('-inf') and score > best_score:
                                    best_score = score
                                    best_trial = result
                                    self.optimization_progress[study_name]['best_value'] = score
                                    self.optimization_progress[study_name]['best_metrics'] = result['metrics']
                                
                                # Toujours incrémenter le compteur et sauvegarder le résultat
                                completed_trials += 1
                                self.optimization_progress[study_name]['completed'] = completed_trials
                                self.optimization_progress[study_name]['trial_results'][trial_id] = {
                                    'score': score,
                                    'metrics': result['metrics'],
                                    'timestamp': datetime.now().isoformat(),
                                    'valid': result.get('valid', True)  # Ajout de l'indicateur d'invalidité
                                }
                                
                            elif msg.message_type == "error":
                                error_msg = f"ERROR in process {msg.process_id}: {msg.content}"
                                self.optimization_progress[study_name]['messages'].append(error_msg)
                                print(f"Error message: {error_msg}")
                    except Exception as e:
                        print(f"Error processing message: {str(e)}")
                        logger.error(f"Error processing messages: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        
                if message_processed > 0:
                    print(f"Processed {message_processed} messages")
                    
            start_time = time.time()
            try:
                while active_processes > 0 and completed_trials < self.config.n_trials:
                    process_messages()
                    
                    if self.stop_event.is_set():
                        logger.info(f"Stopping optimization for '{study_name}' as requested")
                        break
                        
                    if self.config.timeout and time.time() - start_time > self.config.timeout:
                        logger.info(f"Timeout reached for optimization '{study_name}'")
                        break
                        
                    for i, process in enumerate(processes):
                        try:
                            if hasattr(process, '_popen') and process._popen is not None:
                                if not process.is_alive():
                                    process.join()
                                    active_processes -= 1
                                    
                                    if next_trial_id < self.config.n_trials:
                                        new_process = mp.Process(
                                            target=process_trial_standalone,
                                            args=(next_trial_id, db_path, config_dict, self.message_queue, self.stop_event, i)
                                        )
                                        processes[i] = new_process
                                        try:
                                            new_process.start()
                                            active_processes += 1
                                            next_trial_id += 1
                                        except Exception as e:
                                            logger.error(f"Erreur au démarrage du processus pour l'essai {next_trial_id}: {str(e)}")
                        except Exception as e:
                            logger.error(f"Erreur lors de la gestion du processus {i}: {str(e)}")
                            
                    time.sleep(0.1)
                    
                process_messages()
                
                for process in processes:
                    if process and process.is_alive():
                        process.join(timeout=2)
                        if process.is_alive():
                            process.terminate()
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt detected, stopping processes")
                self.stop_event.set()
                for process in processes:
                    if process and process.is_alive():
                        process.terminate()
                process_messages()
                
            execution_time = time.time() - start_time
            
            best_trials = []
            if trial_results:
                best_trials = sorted(
                    [trial_results[tid] for tid in trial_results if trial_results[tid].get('valid', trial_results[tid].get('score', float('-inf')) != float('-inf'))],
                    key=lambda t: t['score'],
                    reverse=True
                )[:5]
                
            saved_strategies = []
            if best_trials:
                saved_strategies = self._save_best_trials(study_path, best_trials)
                
            self.optimization_progress[study_name]['status'] = 'completed'
            self.optimization_progress[study_name]['end_time'] = datetime.now().isoformat()
            self.optimization_progress[study_name]['execution_time'] = execution_time
            
            results = {
                'study_name': study_name,
                'optimization_date': datetime.now().isoformat(),
                'execution_time': execution_time,
                'n_trials': self.config.n_trials,
                'completed_trials': completed_trials,
                'best_trial_id': best_trial['trial_id'] if best_trial else None,
                'best_score': best_trial['score'] if best_trial else None,
                'best_trials': saved_strategies,
                'optimization_config': self.config.to_dict(),
                'messages': self.optimization_progress[study_name]['messages'][-20:],
                'optuna_study_name': self.optimization_progress[study_name].get('optuna_study_name')
            }
            
            results_path = os.path.join(optim_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
                
            latest_path = os.path.join(optim_dir, "latest.json")
            with open(latest_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Optimization completed: {completed_trials} trials in {execution_time:.2f} seconds")
            return True, results
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            traceback.print_exc()
            if study_name in self.optimization_progress:
                self.optimization_progress[study_name]['status'] = 'error'
                self.optimization_progress[study_name]['error_message'] = str(e)
            return False, None
        finally:
            for process in processes:
                try:
                    if process and hasattr(process, '_popen') and process._popen is not None and process.is_alive():
                        process.join(timeout=2)
                        if process.is_alive():
                            process.terminate()
                except Exception as e:
                    logger.error(f"Erreur lors de l'arrêt d'un processus: {str(e)}")
            gc.collect()

    def _save_best_trials(self, study_path: str, best_trials: List[Dict], top_n: int = 5) -> List[Dict]:
        """
        Saves the best strategies from an optimization.
        Args:
            study_path: Path to the study
            best_trials: List of best trials
            top_n: Number of top strategies to save
        Returns:
            List[Dict]: Information about the saved strategies
        """
        try:
            if not best_trials:
                logger.warning(f"No valid trials found for study")
                return []
                
            strategies_dir = os.path.join(study_path, "strategies")
            os.makedirs(strategies_dir, exist_ok=True)
            
            saved_strategies = []
            for i, trial in enumerate(best_trials[:top_n]):
                try:
                    class DummyTrial:
                        def __init__(self, params: Dict):
                            self.params = params
                            self.user_attrs = {}
                            self.number = trial['trial_id']
                            
                        def suggest_categorical(self, name: str, choices: List):
                            return self.params.get(name, choices[0])
                            
                        def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False):
                            step = int(step)
                            return self.params.get(name, low)
                            
                        def suggest_float(self, name: str, low: float, high: float, step: float = None, log: bool = False):
                            return self.params.get(name, low)
                            
                        def set_user_attr(self, key, value):
                            self.user_attrs[key] = value
                            
                    from core.optimization.search_config import SearchSpace
                    from core.optimization.selector import create_strategy_from_trial
                    
                    search_space = SearchSpace.from_dict(self.config.search_space)
                    dummy_trial = DummyTrial(trial['params'])
                    constructor = create_strategy_from_trial(dummy_trial, search_space)
                    
                    signal_generator = constructor.get_signal_generator()
                    position_calculator = constructor.get_position_calculator()
                    
                    strategy_id = f"opt_{trial['trial_id']}_{int(time.time())}"
                    strategy_dir = os.path.join(strategies_dir, strategy_id)
                    os.makedirs(strategy_dir, exist_ok=True)
                    os.makedirs(os.path.join(strategy_dir, "backtests"), exist_ok=True)
                    
                    config_path = os.path.join(strategy_dir, "config.json")
                    constructor.config.id = strategy_id
                    constructor.save(config_path)
                    
                    signal_path = os.path.join(strategy_dir, "signal_generator.pkl")
                    with open(signal_path, 'wb') as f:
                        pickle.dump(signal_generator, f)
                        
                    position_path = os.path.join(strategy_dir, "position_calculator.pkl")
                    with open(position_path, 'wb') as f:
                        pickle.dump(position_calculator, f)
                        
                    rank = i + 1
                    performance = {
                        'name': f'Optimized Strategy {rank}',
                        'source': 'Optimization',
                        'trial_id': trial['trial_id'],
                        'score': trial['score']
                    }
                    
                    metrics = trial['metrics']
                    for key in metrics:
                        performance[key] = metrics[key]
                        
                    for key in ['roi', 'win_rate', 'max_drawdown']:
                        if key in performance:
                            performance[f'{key}_pct'] = performance[key] * 100
                            
                    metadata = {
                        "id": strategy_id,
                        "rank": rank,
                        "name": f"Optimized Strategy {rank}",
                        "description": f"Strategy optimized with trial {trial['trial_id']}",
                        "creation_date": datetime.now().isoformat(),
                        "performance": performance,
                        "trial_id": trial['trial_id'],
                        "tags": ["optimized", f"trial_{trial['trial_id']}"]
                    }
                    
                    metadata_path = os.path.join(strategy_dir, "metadata.json")
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2)
                        
                    saved_strategies.append({
                        'trial_id': trial['trial_id'],
                        'strategy_id': strategy_id,
                        'rank': rank,
                        'score': float(trial['score']),
                        'params': trial['params'],
                        'metrics': {
                            k: float(v) for k, v in metrics.items()
                        }
                    })
                except Exception as e:
                    logger.error(f"Error saving strategy {i+1}: {str(e)}")
                    traceback.print_exc()
                    continue
                    
            logger.info(f"{len(saved_strategies)} best strategies saved for study")
            return saved_strategies
        except Exception as e:
            logger.error(f"Error saving best strategies: {str(e)}")
            traceback.print_exc()
            return []

    def save_backtest_results(self, study_path: str, strategy_id: str, simulation_results: Dict,
                             config: Dict = None, backtest_id: Optional[str] = None) -> str:
        """
        Sauvegarde les résultats de simulation avec la configuration complète.
        Args:
            study_path: Chemin vers le répertoire de l'étude
            strategy_id: Identifiant de la stratégie
            simulation_results: Résultats de la simulation
            config: Configuration utilisée (optionnel)
            backtest_id: Identifiant du backtest (optionnel)
        Returns:
            str: Identifiant du backtest sauvegardé
        """
        if not simulation_results:
            raise ValueError("Aucun résultat de simulation à sauvegarder")
            
        strategy_dir = os.path.join(study_path, "strategies", strategy_id)
        if not os.path.exists(strategy_dir):
            os.makedirs(strategy_dir, exist_ok=True)
            
        backtest_dir = os.path.join(strategy_dir, "backtests")
        os.makedirs(backtest_dir, exist_ok=True)
        
        if not backtest_id:
            backtest_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        study_name = os.path.basename(study_path)
        
        results = {
            "backtest_id": backtest_id,
            "strategy_id": strategy_id,
            "study_name": study_name,
            "date": datetime.now().isoformat(),
            "performance": simulation_results.get('performance', {}),
            "config": config or {},
            "execution_info": {
                "execution_time": simulation_results.get('execution_time', 0),
                "platform": {
                    "python_version": platform.python_version() if 'platform' in globals() else "unknown",
                    "system": platform.system() if 'platform' in globals() else "unknown",
                    "cpu_count": os.cpu_count()
                }
            }
        }
        
        # Ajouter l'information de validité si présente
        if 'valid' in simulation_results:
            results['valid'] = simulation_results['valid']
            
        backtest_path = os.path.join(backtest_dir, f"{backtest_id}.json")
        with open(backtest_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Résultats de backtest sauvegardés: {backtest_path}")
        return backtest_id

    def get_optimization_progress(self, study_name: str = None) -> Dict:
        """
        Gets the optimization progress.
        Args:
            study_name: Name of the study (optional)
        Returns:
            Dict: Progress state
        """
        if study_name:
            if study_name in self.optimization_progress:
                return dict(self.optimization_progress[study_name])
            return {}
            
        result = {}
        for name in self.optimization_progress.keys():
            result[name] = dict(self.optimization_progress[name])
        return result

    def stop_optimization(self, study_name: str) -> bool:
        """
        Stops an ongoing optimization.
        Args:
            study_name: Name of the study
        Returns:
            bool: True if stopping was successful
        """
        self.stop_event.set()
        if study_name in self.optimization_progress:
            self.optimization_progress[study_name]['status'] = 'stopped'
            logger.info(f"Optimization '{study_name}' marked for stopping")
            return True
        else:
            self.optimization_progress[study_name] = {'status': 'stopped'}
            logger.info(f"Optimization '{study_name}' marked for stopping (creating entry)")
            return True

    def get_latest_messages(self, study_name: str = None, limit: int = 20) -> List[str]:
        """
        Gets the latest progress messages.
        Args:
            study_name: Name of the study (optional)
            limit: Maximum number of messages to return
        Returns:
            List[str]: List of latest messages
        """
        if study_name and study_name in self.optimization_progress:
            messages = self.optimization_progress[study_name].get('messages', [])
            return messages[-limit:] if messages else []
            
        all_messages = []
        for name, progress in self.optimization_progress.items():
            if 'messages' in progress:
                for msg in progress['messages'][-5:]:
                    all_messages.append(f"[{name}] {msg}")
        return all_messages[-limit:]
        
    def get_trial_info(self, study_name: str, trial_id: int) -> Optional[Dict]:
        """
        Gets information about a specific trial.
        Args:
            study_name: Name of the study
            trial_id: ID of the trial
        Returns:
            Optional[Dict]: Trial information or None
        """
        if study_name in self.optimization_progress:
            trial_results = self.optimization_progress[study_name].get('trial_results', {})
            if str(trial_id) in trial_results:
                return trial_results[str(trial_id)]
        return None

def create_optimizer(config: Optional[Dict] = None) -> ParallelOptimizer:
    """
    Creates an optimizer with the given configuration.
    Args:
        config: Optimization configuration (optional)
    Returns:
        ParallelOptimizer: Optimizer instance
    """
    if config is None:
        return ParallelOptimizer()
        
    if isinstance(config, dict):
        config = OptimizationConfig.from_dict(config)
        
    return ParallelOptimizer(config)