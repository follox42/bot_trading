"""
Gestionnaire de stratégies de trading.
Responsable de la création, sauvegarde et chargement des stratégies.
"""

import os
import json
import logging
import traceback
import uuid
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

from core.strategy.constructor.constructor import StrategyConstructor
from core.strategy.constructor.constructor_config import StrategyConfig
from core.strategy.risk.risk_config import RiskConfig, RiskModeType
from core.simulation.simulation_config import SimulationConfig
from core.simulation.simulator import Simulator

logger = logging.getLogger(__name__)

class StrategyManager:
    """
    Gestionnaire de stratégies qui coordonne le chargement, l'exécution et
    l'analyse des stratégies en séparant ces fonctionnalités du constructeur.
    """
    
    def __init__(self, study_path: Optional[str] = None):
        """
        Initialise le gestionnaire de stratégies.
        Args:
            study_path: Chemin vers le répertoire de l'étude (optionnel)
        """
        self.study_path = study_path
        self.constructor = None
        self.simulator = None
        self.simulation_results = None
        self.simulation_history = None
    
    def set_study_path(self, study_path: str) -> None:
        """
        Définit le chemin de l'étude.
        Args:
            study_path: Chemin vers le répertoire de l'étude
        """
        self.study_path = study_path
    
    def create_strategy(self, name: str, **kwargs) -> StrategyConstructor:
        """
        Crée une nouvelle stratégie.
        Args:
            name: Nom de la stratégie
            **kwargs: Paramètres additionnels pour la création
        Returns:
            constructor: Constructeur de stratégie créé
        """
        try:
            self.constructor = StrategyConstructor()
            self.constructor.set_name(name)
            
            if 'description' in kwargs:
                self.constructor.set_description(kwargs['description'])
            
            if 'tags' in kwargs and isinstance(kwargs['tags'], list):
                for tag in kwargs['tags']:
                    self.constructor.add_tag(tag)
            
            if 'preset' in kwargs:
                self.constructor.apply_preset(kwargs['preset'])
            
            if 'indicators_preset' in kwargs:
                from core.strategy.constructor.constructor import create_strategy_from_presets
                preset_constructor = create_strategy_from_presets(
                    name=name,
                    indicators_preset=kwargs['indicators_preset'],
                    conditions_preset=kwargs.get('conditions_preset'),
                    risk_preset=kwargs.get('risk_preset')
                )
                current_id = self.constructor.config.id
                self.constructor = preset_constructor
                self.constructor.config.id = current_id
            
            logger.info(f"Nouvelle stratégie '{name}' créée avec succès")
            return self.constructor
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de la stratégie {name}: {str(e)}")
            raise
    
    def save_strategy(self, strategy_id: Optional[str] = None) -> str:
        """
        Sauvegarde la stratégie courante dans le stockage de l'étude.
        Args:
            strategy_id: Identifiant de la stratégie (si None, utilise l'ID existant)
        Returns:
            str: Identifiant de la stratégie sauvegardée
        """
        if self.constructor is None:
            raise ValueError("Aucune stratégie chargée ou créée")
        
        if not self.study_path:
            raise ValueError("Aucun chemin d'étude défini")
        
        if strategy_id:
            self.constructor.config.id = strategy_id
        
        sid = self.constructor.config.id or str(uuid.uuid4())[:8]
        if not sid:
            sid = str(uuid.uuid4())[:8]
            self.constructor.config.id = sid
        
        strategies_dir = os.path.join(self.study_path, "strategies")
        os.makedirs(strategies_dir, exist_ok=True)
        
        strategy_dir = os.path.join(strategies_dir, sid)
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Sauvegarde des éléments de la stratégie
        # 1. Configuration
        config_path = os.path.join(strategy_dir, "config.json")
        self.constructor.save(config_path)
        
        # 2. Générateur de signaux (ConditionEvaluator)
        signal_generator = self.constructor.get_signal_generator()
        signal_path = os.path.join(strategy_dir, "signal_generator.pkl")
        with open(signal_path, 'wb') as f:
            pickle.dump(signal_generator, f)
        
        # 3. Gestionnaire de risque (RiskManager)
        position_calculator = self.constructor.get_position_calculator()
        position_path = os.path.join(strategy_dir, "position_calculator.pkl")
        with open(position_path, 'wb') as f:
            pickle.dump(position_calculator, f)
        
        # 4. Métadonnées
        metadata = {
            "id": sid,
            "name": self.constructor.config.name,
            "description": self.constructor.config.description,
            "creation_date": datetime.now().isoformat(),
            "tags": self.constructor.config.tags,
            "ranks": {}
        }
        
        metadata_path = os.path.join(strategy_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Stratégie '{self.constructor.config.name}' sauvegardée avec succès")
        return sid
    
    def load_strategy(self, strategy_id: str) -> Optional[StrategyConstructor]:
        """
        Charge une stratégie depuis le stockage de l'étude.
        Args:
            strategy_id: Identifiant de la stratégie
        Returns:
            Optional[StrategyConstructor]: Constructeur chargé ou None
        """
        if not self.study_path:
            raise ValueError("Aucun chemin d'étude défini")
        
        strategies_dir = os.path.join(self.study_path, "strategies")
        strategy_dir = os.path.join(strategies_dir, strategy_id)
        
        if not os.path.exists(strategy_dir):
            logger.warning(f"Stratégie '{strategy_id}' non trouvée")
            return None
        
        try:
            # Chargement de la configuration
            config_path = os.path.join(strategy_dir, "config.json")
            if os.path.exists(config_path):
                self.constructor = StrategyConstructor.load(config_path)
            else:
                # Si pas de fichier de configuration, créer un constructeur minimal
                self.constructor = StrategyConstructor()
                self.constructor.config.id = strategy_id
            
            # Chargement des métadonnées
            metadata_path = os.path.join(strategy_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                self.constructor.set_name(metadata.get("name", f"Strategy {strategy_id}"))
                self.constructor.set_description(metadata.get("description", ""))
                for tag in metadata.get("tags", []):
                    self.constructor.add_tag(tag)
            
            logger.info(f"Stratégie '{self.constructor.config.name}' chargée avec succès")
            return self.constructor
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la stratégie {strategy_id}: {str(e)}")
            traceback.print_exc()
            return None
    
    def list_strategies(self) -> List[Dict]:
        """
        Liste toutes les stratégies disponibles pour l'étude.
        Returns:
            List[Dict]: Liste des stratégies avec leurs métadonnées
        """
        if not self.study_path:
            raise ValueError("Aucun chemin d'étude défini")
        
        strategies_dir = os.path.join(self.study_path, "strategies")
        if not os.path.exists(strategies_dir):
            os.makedirs(strategies_dir)
            return []
        
        strategies = []
        
        for strategy_id in os.listdir(strategies_dir):
            strategy_dir = os.path.join(strategies_dir, strategy_id)
            if not os.path.isdir(strategy_dir):
                continue
            
            metadata_path = os.path.join(strategy_dir, "metadata.json")
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    strategies.append(metadata)
                except Exception as e:
                    logger.warning(f"Erreur lors de la lecture des métadonnées de {strategy_id}: {str(e)}")
                    strategies.append({
                        "id": strategy_id,
                        "name": f"Strategy {strategy_id}",
                        "description": "Métadonnées non disponibles",
                        "creation_date": "unknown"
                    })
        
        return strategies
    
    def delete_strategy(self, strategy_id: str) -> bool:
        """
        Supprime une stratégie.
        Args:
            strategy_id: Identifiant de la stratégie
        Returns:
            bool: True si la suppression a réussi
        """
        if not self.study_path:
            raise ValueError("Aucun chemin d'étude défini")
        
        strategy_dir = os.path.join(self.study_path, "strategies", strategy_id)
        
        if not os.path.exists(strategy_dir):
            logger.warning(f"Stratégie '{strategy_id}' non trouvée")
            return False
        
        try:
            import shutil
            shutil.rmtree(strategy_dir)
            logger.info(f"Stratégie '{strategy_id}' supprimée avec succès")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de la stratégie {strategy_id}: {str(e)}")
            return False
    
    def setup_simulator(self, config: Optional[SimulationConfig] = None) -> Simulator:
        """
        Configure le simulateur pour la stratégie courante.
        Args:
            config: Configuration de simulation (optionnel)
        Returns:
            Simulator: Instance du simulateur configuré
        """
        self.simulator = Simulator(config)
        return self.simulator
    
    def run_simulation(
        self,
        data: pd.DataFrame,
        config: Optional[SimulationConfig] = None
    ) -> Dict:
        """
        Exécute une simulation avec la stratégie courante.
        Args:
            data: DataFrame avec les données OHLC
            config: Configuration de simulation (optionnel)
        Returns:
            Dict: Résultats de la simulation
        """
        if self.constructor is None:
            raise ValueError("Aucune stratégie chargée pour la simulation")
        
        if self.simulator is None or config is not None:
            self.setup_simulator(config)
        
        try:
            signals, data_with_signals = self.constructor.generate_signals(data)
            
            position_sizes = data_with_signals['position_size'].values if 'position_size' in data_with_signals.columns else None
            sl_levels = data_with_signals['sl_level'].values if 'sl_level' in data_with_signals.columns else None
            tp_levels = data_with_signals['tp_level'].values if 'tp_level' in data_with_signals.columns else None
            
            results = self.simulator.run(
                prices=data_with_signals['close'].values,
                signals=signals,
                position_sizes=position_sizes,
                sl_levels=sl_levels,
                tp_levels=tp_levels
            )
            
            self.simulation_results = results
            self.simulation_history = data_with_signals
            
            logger.info(f"Simulation exécutée avec succès: ROI={results['performance']['roi_pct']:.2f}%, " +
                       f"Trades={results['performance']['total_trades']}, " +
                       f"Win Rate={results['performance']['win_rate_pct']:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de la simulation: {str(e)}")
            traceback.print_exc()
            raise
    
    def save_backtest_results(self, backtest_id: Optional[str] = None) -> str:
        """
        Sauvegarde les résultats de la dernière simulation.
        Args:
            backtest_id: Identifiant du backtest (optionnel)
        Returns:
            str: Identifiant du backtest sauvegardé
        """
        if not self.simulation_results:
            raise ValueError("Aucun résultat de simulation à sauvegarder")
        
        if not self.constructor:
            raise ValueError("Aucune stratégie chargée")
        
        if not self.study_path:
            raise ValueError("Aucun chemin d'étude défini")
        
        strategy_id = self.constructor.config.id
        strategy_dir = os.path.join(self.study_path, "strategies", strategy_id)
        
        if not os.path.exists(strategy_dir):
            raise ValueError(f"Stratégie '{strategy_id}' non trouvée")
        
        backtest_dir = os.path.join(strategy_dir, "backtests")
        os.makedirs(backtest_dir, exist_ok=True)
        
        if not backtest_id:
            backtest_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backtest_path = os.path.join(backtest_dir, f"{backtest_id}.json")
        
        results = self.simulation_results.copy()
        
        # Ajouter des métadonnées
        results["backtest_id"] = backtest_id
        results["strategy_id"] = strategy_id
        results["date"] = datetime.now().isoformat()
        
        with open(backtest_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Résultats de backtest sauvegardés: {backtest_path}")
        
        # Sauvegarder l'historique des données avec signaux si disponible
        if self.simulation_history is not None:
            history_path = os.path.join(backtest_dir, f"{backtest_id}_history.csv")
            self.simulation_history.to_csv(history_path, index=False)
        
        return backtest_id
    
    def list_backtests(self, strategy_id: Optional[str] = None) -> List[Dict]:
        """
        Liste tous les backtests disponibles pour une stratégie.
        Args:
            strategy_id: Identifiant de la stratégie (utilise la stratégie courante si None)
        Returns:
            List[Dict]: Liste des backtests avec leurs métadonnées
        """
        if not strategy_id and not self.constructor:
            raise ValueError("Aucune stratégie spécifiée ou chargée")
        
        if not self.study_path:
            raise ValueError("Aucun chemin d'étude défini")
        
        if not strategy_id:
            strategy_id = self.constructor.config.id
        
        backtest_dir = os.path.join(self.study_path, "strategies", strategy_id, "backtests")
        
        if not os.path.exists(backtest_dir):
            return []
        
        backtests = []
        
        for filename in os.listdir(backtest_dir):
            if not filename.endswith('.json') or filename.endswith('_history.json'):
                continue
            
            backtest_path = os.path.join(backtest_dir, filename)
            
            try:
                with open(backtest_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                backtest_id = os.path.splitext(filename)[0]
                
                summary = {
                    "id": backtest_id,
                    "date": results.get("date", "unknown"),
                    "strategy_id": strategy_id,
                    "performance": results.get("performance", {})
                }
                
                backtests.append(summary)
                
            except Exception as e:
                logger.warning(f"Erreur lors de la lecture du backtest {filename}: {str(e)}")
        
        # Trier par date (plus récent d'abord)
        backtests.sort(key=lambda x: x.get("date", ""), reverse=True)
        
        return backtests
    
    def load_backtest(self, backtest_id: str, strategy_id: Optional[str] = None) -> Optional[Dict]:
        """
        Charge les résultats d'un backtest.
        Args:
            backtest_id: Identifiant du backtest
            strategy_id: Identifiant de la stratégie (utilise la stratégie courante si None)
        Returns:
            Optional[Dict]: Résultats du backtest ou None
        """
        if not strategy_id and not self.constructor:
            raise ValueError("Aucune stratégie spécifiée ou chargée")
        
        if not self.study_path:
            raise ValueError("Aucun chemin d'étude défini")
        
        if not strategy_id:
            strategy_id = self.constructor.config.id
        
        backtest_path = os.path.join(self.study_path, "strategies", strategy_id, "backtests", f"{backtest_id}.json")
        
        if not os.path.exists(backtest_path):
            logger.warning(f"Backtest '{backtest_id}' non trouvé")
            return None
        
        try:
            with open(backtest_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            self.simulation_results = results
            
            # Charger également l'historique si disponible
            history_path = os.path.join(os.path.dirname(backtest_path), f"{backtest_id}_history.csv")
            if os.path.exists(history_path):
                self.simulation_history = pd.read_csv(history_path)
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du backtest {backtest_id}: {str(e)}")
            return None
    
    def generate_performance_report(self, output_path: Optional[str] = None) -> Dict:
        """
        Génère un rapport de performance détaillé pour la dernière simulation.
        Args:
            output_path: Chemin pour sauvegarder le rapport (optionnel)
        Returns:
            Dict: Rapport de performance
        """
        if not self.simulation_results:
            raise ValueError("Aucun résultat de simulation pour générer un rapport")
        
        try:
            perf = self.simulation_results['performance']
            
            report = {
                "strategy_info": {
                    "name": self.constructor.config.name if self.constructor else "Unknown",
                    "id": self.constructor.config.id if self.constructor else "Unknown",
                    "description": self.constructor.config.description if self.constructor else "",
                    "tags": self.constructor.config.tags if self.constructor else []
                },
                "summary": {
                    "roi": f"{perf['roi_pct']:.2f}%",
                    "total_trades": perf['total_trades'],
                    "win_rate": f"{perf['win_rate_pct']:.2f}%",
                    "max_drawdown": f"{perf['max_drawdown_pct']:.2f}%",
                    "profit_factor": f"{perf['profit_factor']:.2f}",
                    "final_balance": f"{perf['final_balance']:.2f}"
                },
                "detailed_metrics": {
                    **perf
                },
                "timestamp": datetime.now().isoformat()
            }
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Rapport de performance sauvegardé dans {output_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport: {str(e)}")
            raise

def create_strategy_manager_for_study(study_path: str) -> StrategyManager:
    """
    Crée un gestionnaire de stratégies pour une étude spécifique.
    Args:
        study_path: Chemin vers le répertoire de l'étude
    Returns:
        StrategyManager: Gestionnaire de stratégies initialisé
    """
    return StrategyManager(study_path)