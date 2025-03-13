"""
Module de gestion centralisée des études de trading avec intégration des composants.
Ce module étend le gestionnaire d'études standard avec une meilleure intégration
des configurations, optimisations et visualisations.
"""
import os
import json
import shutil
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set, Union
import pandas as pd
import traceback
import hashlib
from concurrent.futures import ThreadPoolExecutor

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_study_manager.log', mode='a'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('integrated_study_manager')

class IntegratedStudyManager:
    """
    Gestionnaire intégré des études de trading avec une meilleure intégration 
    entre les différents composants du système.
    
    Cette classe fournit une interface complète pour:
    - Créer, modifier et supprimer des études
    - Gérer les configurations et métadonnées
    - Gérer les stratégies associées aux études
    - Gérer les optimisations et les backtests
    - Gérer les données utilisées par les études
    """
    
    def __init__(self, base_dir: str = "studies"):
        """
        Initialise le gestionnaire d'études.
        
        Args:
            base_dir: Répertoire de base pour le stockage des études
        """
        self.base_dir = base_dir
        self.studies_list = None
        self.cached_studies = {}
        
        # Créer le répertoire de base s'il n'existe pas
        os.makedirs(base_dir, exist_ok=True)
        
        # Charger la liste des études
        self.reload_studies_list()
        
        logger.info(f"IntegratedStudyManager initialisé avec {len(self.get_studies_list())} études")
    
    def reload_studies_list(self) -> None:
        """Recharge la liste des études depuis le système de fichiers"""
        from simulator.config import load_study_list, update_study_list
        
        self.studies_list = load_study_list(self.base_dir)
        
        # Si la liste est vide ou inexistante, la reconstruire
        if not self.studies_list or not self.studies_list.studies:
            self.studies_list = update_study_list(self.base_dir)
    
    def get_studies_list(self) -> List[Dict]:
        """
        Récupère la liste des études disponibles.
        
        Returns:
            List[Dict]: Liste des métadonnées de base des études
        """
        if self.studies_list is None:
            self.reload_studies_list()
        
        return self.studies_list.studies if self.studies_list else []
    
    def study_exists(self, study_name: str) -> bool:
        """
        Vérifie si une étude existe.
        
        Args:
            study_name: Nom de l'étude à vérifier
            
        Returns:
            bool: True si l'étude existe, False sinon
        """
        study_path = os.path.join(self.base_dir, study_name)
        return os.path.exists(study_path) and os.path.isdir(study_path)
    
    def get_study_dir(self, study_name: str) -> str:
        """
        Récupère le chemin du répertoire d'une étude.
        
        Args:
            study_name: Nom de l'étude
            
        Returns:
            str: Chemin du répertoire de l'étude
        """
        return os.path.join(self.base_dir, study_name)
    
    def create_study(
        self, 
        study_name: str, 
        metadata: Dict, 
        trading_config: Optional['FlexibleTradingConfig'] = None
    ) -> bool:
        """
        Crée une nouvelle étude avec les métadonnées et la configuration spécifiées.
        
        Args:
            study_name: Nom de l'étude à créer
            metadata: Métadonnées de base de l'étude
            trading_config: Configuration de trading (optionnel)
            
        Returns:
            bool: True si la création a réussi, False sinon
        """
        from simulator.config import (
            StudyMetadata, DataMetadata, DataSource,
            save_study_metadata, save_study_list
        )
        from simulator.config import (
            create_trading_default_config,
            save_trading_config_to_file,
            save_optimization_config_to_file,
            create_default_optimization_config
        )
        
        if self.study_exists(study_name):
            logger.warning(f"L'étude '{study_name}' existe déjà")
            return False
        
        try:
            # Créer le répertoire de l'étude
            study_dir = self.get_study_dir(study_name)
            os.makedirs(study_dir, exist_ok=True)
            
            # Créer les sous-répertoires
            os.makedirs(os.path.join(study_dir, "data"), exist_ok=True)
            os.makedirs(os.path.join(study_dir, "strategies"), exist_ok=True)
            os.makedirs(os.path.join(study_dir, "backtests"), exist_ok=True)
            os.makedirs(os.path.join(study_dir, "optimizations"), exist_ok=True)
            
            # Créer les métadonnées complètes
            study_metadata = StudyMetadata(
                name=study_name,
                description=metadata.get("description", ""),
                asset=metadata.get("asset", ""),
                timeframe=metadata.get("timeframe", ""),
                exchange=metadata.get("exchange", "")
            )
            
            # Mise à jour des métadonnées de données si fournies
            if "data_file_path" in metadata:
                data_file_path = metadata["data_file_path"]
                data_metadata = metadata.get("data_metadata", {})
                
                if os.path.exists(data_file_path):
                    # Calculer le checksum du fichier
                    file_checksum = self._calculate_file_checksum(data_file_path)
                    
                    # Copier le fichier dans le répertoire de l'étude
                    data_dir = os.path.join(study_dir, "data")
                    filename = os.path.basename(data_file_path)
                    local_path = os.path.join(data_dir, filename)
                    shutil.copy2(data_file_path, local_path)
                    
                    # Analyser les données et mettre à jour les métadonnées
                    try:
                        df = pd.read_csv(local_path)
                        rows_count = len(df)
                        columns = df.columns.tolist()
                    except Exception as e:
                        logger.error(f"Erreur lors de l'analyse des données: {e}")
                        rows_count = 0
                        columns = []
                    
                    # Mettre à jour les métadonnées de données
                    study_metadata.data = DataMetadata(
                        source=DataSource.STUDY,
                        file_path=os.path.join("data", filename),
                        exchange=metadata.get("exchange", ""),
                        symbol=metadata.get("asset", ""),
                        timeframe=metadata.get("timeframe", ""),
                        rows_count=rows_count,
                        columns=columns,
                        checksum=file_checksum,
                        start_date=data_metadata.get("start_date"),
                        end_date=data_metadata.get("end_date")
                    )
            
            # Sauvegarder les métadonnées
            save_study_metadata(study_metadata, study_dir)
            
            # Sauvegarder la configuration de trading
            if trading_config is None:
                trading_config = create_trading_default_config()
            
            config_path = os.path.join(study_dir, "trading_config.json")
            save_trading_config_to_file(trading_config, config_path)
            
            # Créer une configuration d'optimisation par défaut
            optim_config = create_default_optimization_config()
            optim_path = os.path.join(study_dir, "optimization_config.json")
            save_optimization_config_to_file(optim_config, optim_path)
            
            # Mettre à jour la liste des études
            if self.studies_list:
                self.studies_list.add_study(study_metadata)
                save_study_list(self.studies_list, self.base_dir)
            
            logger.info(f"Étude '{study_name}' créée avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'étude '{study_name}': {e}")
            traceback.print_exc()
            
            # Nettoyage en cas d'erreur
            try:
                study_dir = self.get_study_dir(study_name)
                if os.path.exists(study_dir):
                    shutil.rmtree(study_dir)
            except Exception as cleanup_error:
                logger.error(f"Erreur lors du nettoyage après échec: {cleanup_error}")
            
            return False
    
    def delete_study(self, study_name: str) -> bool:
        """
        Supprime une étude existante.
        
        Args:
            study_name: Nom de l'étude à supprimer
            
        Returns:
            bool: True si la suppression a réussi, False sinon
        """
        from simulator.config import save_study_list
        
        if not self.study_exists(study_name):
            logger.warning(f"L'étude '{study_name}' n'existe pas")
            return False
        
        try:
            # Supprimer le répertoire de l'étude
            study_dir = self.get_study_dir(study_name)
            shutil.rmtree(study_dir)
            
            # Mettre à jour la liste des études
            if self.studies_list:
                self.studies_list.remove_study(study_name)
                save_study_list(self.studies_list, self.base_dir)
            
            # Supprimer de la cache
            if study_name in self.cached_studies:
                del self.cached_studies[study_name]
            
            logger.info(f"Étude '{study_name}' supprimée avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de l'étude '{study_name}': {e}")
            return False
    
    def get_study_metadata(self, study_name: str) -> Optional['StudyMetadata']:
        """
        Récupère les métadonnées complètes d'une étude.
        
        Args:
            study_name: Nom de l'étude
            
        Returns:
            Optional[StudyMetadata]: Métadonnées de l'étude ou None en cas d'erreur
        """
        from simulator.config import load_study_metadata
        
        if not self.study_exists(study_name):
            logger.warning(f"L'étude '{study_name}' n'existe pas")
            return None
        
        try:
            # Charger les métadonnées
            study_dir = self.get_study_dir(study_name)
            metadata = load_study_metadata(study_dir)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des métadonnées: {e}")
            return None
    
    def update_study_metadata(self, study_name: str, updates: Dict) -> bool:
        """
        Met à jour les métadonnées d'une étude.
        
        Args:
            study_name: Nom de l'étude
            updates: Dictionnaire des mises à jour
            
        Returns:
            bool: True si la mise à jour a réussi, False sinon
        """
        from simulator.config import (
            load_study_metadata, save_study_metadata, save_study_list
        )
        
        if not self.study_exists(study_name):
            logger.warning(f"L'étude '{study_name}' n'existe pas")
            return False
        
        try:
            # Charger les métadonnées actuelles
            study_dir = self.get_study_dir(study_name)
            metadata = load_study_metadata(study_dir)
            
            if metadata is None:
                logger.error(f"Impossible de charger les métadonnées de l'étude '{study_name}'")
                return False
            
            # Mettre à jour les champs de base
            for field in ["description", "asset", "timeframe", "exchange"]:
                if field in updates:
                    setattr(metadata, field, updates[field])
            
            # Mettre à jour les tags et catégories
            if "tags" in updates:
                metadata.tags = updates["tags"]
            
            if "categories" in updates:
                metadata.categories = updates["categories"]
            
            # Mettre à jour les paramètres personnalisés
            if "custom_params" in updates:
                metadata.custom_params.update(updates["custom_params"])
            
            # Mettre à jour la date de dernière modification
            metadata.update_last_modified()
            
            # Sauvegarder les métadonnées
            success = save_study_metadata(metadata, study_dir)
            
            # Mettre à jour la liste des études
            if success and self.studies_list:
                self.studies_list.add_study(metadata)
                save_study_list(self.studies_list, self.base_dir)
            
            return success
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des métadonnées: {e}")
            return False
    
    def update_study_status(self, study_name: str, status: Union[str, 'StudyStatus']) -> bool:
        """
        Met à jour le statut d'une étude.
        
        Args:
            study_name: Nom de l'étude
            status: Nouveau statut (chaîne ou énumération)
            
        Returns:
            bool: True si la mise à jour a réussi, False sinon
        """
        from simulator.config import (
            StudyStatus, load_study_metadata, save_study_metadata, save_study_list
        )
        
        if not self.study_exists(study_name):
            logger.warning(f"L'étude '{study_name}' n'existe pas")
            return False
        
        try:
            # Convertir le statut en énumération si nécessaire
            if isinstance(status, str):
                status = StudyStatus(status)
            
            # Charger les métadonnées actuelles
            study_dir = self.get_study_dir(study_name)
            metadata = load_study_metadata(study_dir)
            
            if metadata is None:
                logger.error(f"Impossible de charger les métadonnées de l'étude '{study_name}'")
                return False
            
            # Mettre à jour le statut
            metadata.status = status
            
            # Mettre à jour la date de dernière modification
            metadata.update_last_modified()
            
            # Sauvegarder les métadonnées
            success = save_study_metadata(metadata, study_dir)
            
            # Mettre à jour la liste des études
            if success and self.studies_list:
                self.studies_list.add_study(metadata)
                save_study_list(self.studies_list, self.base_dir)
            
            return success
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du statut: {e}")
            return False
    
    def get_trading_config(self, study_name: str) -> Optional['FlexibleTradingConfig']:
        """
        Récupère la configuration de trading d'une étude.
        
        Args:
            study_name: Nom de l'étude
            
        Returns:
            Optional[FlexibleTradingConfig]: Configuration de trading ou None en cas d'erreur
        """
        from simulator.config import load_trading_config_from_file
        
        if not self.study_exists(study_name):
            logger.warning(f"L'étude '{study_name}' n'existe pas")
            return None
        
        try:
            # Charger la configuration
            study_dir = self.get_study_dir(study_name)
            config_path = os.path.join(study_dir, "trading_config.json")
            
            if not os.path.exists(config_path):
                logger.warning(f"Fichier de configuration de trading introuvable pour '{study_name}'")
                return None
            
            return load_trading_config_from_file(config_path)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la configuration de trading: {e}")
            return None
    
    def update_trading_config(self, study_name: str, config: 'FlexibleTradingConfig') -> bool:
        """
        Met à jour la configuration de trading d'une étude.
        
        Args:
            study_name: Nom de l'étude
            config: Nouvelle configuration de trading
            
        Returns:
            bool: True si la mise à jour a réussi, False sinon
        """
        from simulator.config import save_trading_config_to_file
        from simulator.config import load_study_metadata, save_study_metadata
        
        if not self.study_exists(study_name):
            logger.warning(f"L'étude '{study_name}' n'existe pas")
            return False
        
        try:
            # Sauvegarder la configuration
            study_dir = self.get_study_dir(study_name)
            config_path = os.path.join(study_dir, "trading_config.json")
            
            success = save_trading_config_to_file(config, config_path)
            
            # Mettre à jour la date de dernière modification des métadonnées
            if success:
                metadata = load_study_metadata(study_dir)
                if metadata:
                    metadata.update_last_modified()
                    save_study_metadata(metadata, study_dir)
                    
                    # Mettre à jour la liste des études
                    if self.studies_list:
                        self.studies_list.add_study(metadata)
                        from simulator.config import save_study_list
                        save_study_list(self.studies_list, self.base_dir)
            
            return success
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de la configuration de trading: {e}")
            return False
    
    def get_optimization_config(self, study_name: str) -> Optional['OptimizationConfig']:
        """
        Récupère la configuration d'optimisation d'une étude.
        
        Args:
            study_name: Nom de l'étude
            
        Returns:
            Optional[OptimizationConfig]: Configuration d'optimisation ou None en cas d'erreur
        """
        from simulator.config import load_optimization_config_from_file
        
        if not self.study_exists(study_name):
            logger.warning(f"L'étude '{study_name}' n'existe pas")
            return None
        
        try:
            # Charger la configuration
            study_dir = self.get_study_dir(study_name)
            config_path = os.path.join(study_dir, "optimization_config.json")
            
            if not os.path.exists(config_path):
                logger.warning(f"Fichier de configuration d'optimisation introuvable pour '{study_name}'")
                return None
            
            return load_optimization_config_from_file(config_path)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la configuration d'optimisation: {e}")
            return None
    
    def save_optimization_config(self, study_name: str, config: Union['OptimizationConfig', Dict]) -> bool:
        """
        Sauvegarde la configuration d'optimisation d'une étude.
        
        Args:
            study_name: Nom de l'étude
            config: Configuration d'optimisation (objet ou dictionnaire)
            
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        from simulator.config import OptimizationConfig, save_optimization_config_to_file
        
        if not self.study_exists(study_name):
            logger.warning(f"L'étude '{study_name}' n'existe pas")
            return False
        
        try:
            # Convertir le dictionnaire en objet si nécessaire
            if isinstance(config, dict):
                config_obj = OptimizationConfig.from_dict(config)
            else:
                config_obj = config
            
            # Sauvegarder la configuration
            study_dir = self.get_study_dir(study_name)
            config_path = os.path.join(study_dir, "optimization_config.json")
            
            return save_optimization_config_to_file(config_obj, config_path)
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la configuration d'optimisation: {e}")
            return False
    
    def get_optimization_results(self, study_name: str) -> Optional[Dict]:
        """
        Récupère les résultats d'optimisation d'une étude.
        
        Args:
            study_name: Nom de l'étude
            
        Returns:
            Optional[Dict]: Résultats d'optimisation ou None en cas d'erreur
        """
        if not self.study_exists(study_name):
            logger.warning(f"L'étude '{study_name}' n'existe pas")
            return None
        
        try:
            # Charger les résultats
            study_dir = self.get_study_dir(study_name)
            results_path = os.path.join(study_dir, "optimizations", "results.json")
            
            if not os.path.exists(results_path):
                logger.warning(f"Résultats d'optimisation introuvables pour '{study_name}'")
                return None
            
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des résultats d'optimisation: {e}")
            return None
    
    def save_optimization_results(self, study_name: str, results: Dict) -> bool:
        """
        Sauvegarde les résultats d'optimisation d'une étude.
        
        Args:
            study_name: Nom de l'étude
            results: Résultats d'optimisation
            
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        from simulator.config import (
            OptimizationMetadata, StudyStatus,
            load_study_metadata, save_study_metadata, save_study_list
        )
        
        if not self.study_exists(study_name):
            logger.warning(f"L'étude '{study_name}' n'existe pas")
            return False
        
        try:
            # Sauvegarder les résultats
            study_dir = self.get_study_dir(study_name)
            optim_dir = os.path.join(study_dir, "optimizations")
            os.makedirs(optim_dir, exist_ok=True)
            
            results_path = os.path.join(optim_dir, "results.json")
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            
            # Mettre à jour les métadonnées d'optimisation
            metadata = load_study_metadata(study_dir)
            if metadata:
                # Extraire les informations des résultats
                best_trial_id = results.get("best_trial_id", -1)
                best_score = results.get("best_score", 0.0)
                n_trials = results.get("n_trials", 0)
                optimization_date = results.get("optimization_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                
                # Mettre à jour les métadonnées
                metadata.optimization = OptimizationMetadata(
                    last_optimization=optimization_date,
                    trials_count=n_trials,
                    best_trial_id=best_trial_id,
                    best_score=best_score,
                    optimization_config=results.get("optimization_config", {})
                )
                
                # Mettre à jour le statut si nécessaire
                if metadata.status == StudyStatus.CREATED:
                    metadata.status = StudyStatus.OPTIMIZED
                
                # Sauvegarder les métadonnées
                metadata.update_last_modified()
                save_study_metadata(metadata, study_dir)
                
                # Mettre à jour la liste des études
                if self.studies_list:
                    self.studies_list.add_study(metadata)
                    save_study_list(self.studies_list, self.base_dir)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats d'optimisation: {e}")
            return False
    
    def save_strategy(
        self, 
        study_name: str, 
        strategy_rank: int, 
        signal_generator: 'SignalGenerator',
        position_calculator: 'PositionCalculator',
        performance: Dict[str, Any]
    ) -> bool:
        """
        Sauvegarde une stratégie pour une étude.
        
        Args:
            study_name: Nom de l'étude
            strategy_rank: Rang de la stratégie (1 = meilleure)
            signal_generator: Générateur de signaux
            position_calculator: Calculateur de position
            performance: Métriques de performance
            
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        from simulator.config import (
            load_study_metadata, save_study_metadata, save_study_list,
            PerformanceMetrics
        )
        
        if not self.study_exists(study_name):
            logger.warning(f"L'étude '{study_name}' n'existe pas")
            return False
        
        try:
            # Préparer le dossier de stratégies
            study_dir = self.get_study_dir(study_name)
            strategies_dir = os.path.join(study_dir, "strategies")
            os.makedirs(strategies_dir, exist_ok=True)
            
            # Générer l'ID de la stratégie
            strategy_id = f"strategy_{strategy_rank:03d}"
            strategy_dir = os.path.join(strategies_dir, strategy_id)
            os.makedirs(strategy_dir, exist_ok=True)
            
            # Sauvegarder les blocs d'achat et de vente
            buy_blocks = []
            for block in signal_generator.buy_blocks:
                buy_blocks.append(block.to_dict())
            
            sell_blocks = []
            for block in signal_generator.sell_blocks:
                sell_blocks.append(block.to_dict())
            
            # Sauvegarder la configuration de risque
            risk_config = {
                "mode": position_calculator.mode.value,
                "config": position_calculator.config
            }
            
            # Sauvegarder la stratégie complète
            strategy_data = {
                "id": strategy_id,
                "rank": strategy_rank,
                "name": performance.get("name", f"Strategy {strategy_rank}"),
                "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "buy_blocks": buy_blocks,
                "sell_blocks": sell_blocks,
                "risk": risk_config,
                "performance": performance,
                "source": performance.get("source", "Manual"),
                "trial_id": performance.get("trial_id")
            }
            
            # Sauvegarder en JSON
            strategy_path = os.path.join(strategy_dir, "strategy.json")
            with open(strategy_path, 'w', encoding='utf-8') as f:
                json.dump(strategy_data, f, indent=4, ensure_ascii=False)
            
            # Mettre à jour les métadonnées de l'étude
            metadata = load_study_metadata(study_dir)
            if metadata:
                # Vérifier si cette stratégie doit être définie comme la meilleure
                is_best = (strategy_rank == 1)
                
                # Mettre à jour les métadonnées des stratégies
                count = metadata.strategies.strategies_count + 1 if strategy_id not in [s.get("id") for s in metadata.strategies.strategies] else metadata.strategies.strategies_count
                
                metadata.strategies.strategies_count = count
                
                # Ajouter ou mettre à jour cette stratégie dans la liste
                strategy_info = {
                    "id": strategy_id,
                    "rank": strategy_rank,
                    "name": performance.get("name", f"Strategy {strategy_rank}"),
                    "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": performance.get("source", "Manual"),
                    "trial_id": performance.get("trial_id"),
                    "performance": {
                        "roi": performance.get("roi", 0),
                        "win_rate": performance.get("win_rate", 0),
                        "total_trades": performance.get("total_trades", 0),
                        "max_drawdown": performance.get("max_drawdown", 0)
                    }
                }
                
                # Mettre à jour la liste des stratégies
                for i, strat in enumerate(metadata.strategies.strategies):
                    if strat.get("id") == strategy_id:
                        metadata.strategies.strategies[i] = strategy_info
                        break
                else:
                    metadata.strategies.strategies.append(strategy_info)
                
                # Mettre à jour la meilleure stratégie si nécessaire
                if is_best:
                    metadata.strategies.best_strategy_id = strategy_id
                    
                    # Mettre à jour les métriques de performance globales
                    metadata.performance = PerformanceMetrics(
                        roi=performance.get("roi", 0),
                        win_rate=performance.get("win_rate", 0),
                        total_trades=performance.get("total_trades", 0),
                        max_drawdown=performance.get("max_drawdown", 0),
                        profit_factor=performance.get("profit_factor", 0),
                        avg_profit=performance.get("avg_profit", 0),
                        sharpe_ratio=performance.get("sharpe_ratio", None),
                        sortino_ratio=performance.get("sortino_ratio", None),
                        calmar_ratio=performance.get("calmar_ratio", None)
                    )
                
                # Sauvegarder les métadonnées
                metadata.update_last_modified()
                save_study_metadata(metadata, study_dir)
                
                # Mettre à jour la liste des études
                if self.studies_list:
                    self.studies_list.add_study(metadata)
                    save_study_list(self.studies_list, self.base_dir)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la stratégie: {e}")
            traceback.print_exc()
            return False
    
    def load_strategy(self, study_name: str, strategy_id: str) -> Optional[Dict]:
        """
        Charge une stratégie depuis une étude.
        
        Args:
            study_name: Nom de l'étude
            strategy_id: ID de la stratégie
            
        Returns:
            Optional[Dict]: Stratégie chargée ou None en cas d'erreur
        """
        if not self.study_exists(study_name):
            logger.warning(f"L'étude '{study_name}' n'existe pas")
            return None
        
        try:
            # Charger la stratégie
            study_dir = self.get_study_dir(study_name)
            strategy_path = os.path.join(study_dir, "strategies", strategy_id, "strategy.json")
            
            if not os.path.exists(strategy_path):
                logger.warning(f"Stratégie '{strategy_id}' introuvable pour l'étude '{study_name}'")
                return None
            
            with open(strategy_path, 'r', encoding='utf-8') as f:
                strategy_data = json.load(f)
            
            return strategy_data
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la stratégie: {e}")
            return None
    
    def create_signal_generator_from_strategy(self, strategy_data: Dict) -> Optional['SignalGenerator']:
        """
        Crée un générateur de signaux à partir des données de stratégie.
        
        Args:
            strategy_data: Données de la stratégie
            
        Returns:
            Optional[SignalGenerator]: Générateur de signaux ou None en cas d'erreur
        """
        from simulator.indicators import SignalGenerator, Block
        
        try:
            signal_generator = SignalGenerator()
            
            # Charger les blocs d'achat
            if "buy_blocks" in strategy_data:
                for block_data in strategy_data["buy_blocks"]:
                    block = Block.from_dict(block_data)
                    signal_generator.add_block(block, is_buy=True)
            
            # Charger les blocs de vente
            if "sell_blocks" in strategy_data:
                for block_data in strategy_data["sell_blocks"]:
                    block = Block.from_dict(block_data)
                    signal_generator.add_block(block, is_buy=False)
            
            return signal_generator
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du générateur de signaux: {e}")
            return None
    
    def create_position_calculator_from_strategy(self, strategy_data: Dict) -> Optional['PositionCalculator']:
        """
        Crée un calculateur de position à partir des données de stratégie.
        
        Args:
            strategy_data: Données de la stratégie
            
        Returns:
            Optional[PositionCalculator]: Calculateur de position ou None en cas d'erreur
        """
        from simulator.risk import PositionCalculator, RiskMode
        
        try:
            if "risk" not in strategy_data:
                logger.warning("Données de risque manquantes dans la stratégie")
                return None
            
            risk_data = strategy_data["risk"]
            risk_mode = RiskMode(risk_data["mode"])
            risk_config = risk_data["config"]
            
            return PositionCalculator(mode=risk_mode, config=risk_config)
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du calculateur de position: {e}")
            return None
    
    def save_backtest_results(
        self, 
        study_name: str, 
        strategy_id: str, 
        backtest_id: str, 
        results: Dict
    ) -> bool:
        """
        Sauvegarde les résultats d'un backtest.
        
        Args:
            study_name: Nom de l'étude
            strategy_id: ID de la stratégie
            backtest_id: ID du backtest
            results: Résultats du backtest
            
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        from simulator.config import (
            StudyStatus, load_study_metadata, save_study_metadata, save_study_list
        )
        
        if not self.study_exists(study_name):
            logger.warning(f"L'étude '{study_name}' n'existe pas")
            return False
        
        try:
            # Préparer le dossier de backtests
            study_dir = self.get_study_dir(study_name)
            backtest_dir = os.path.join(study_dir, "backtests", backtest_id)
            os.makedirs(backtest_dir, exist_ok=True)
            
            # Sauvegarder les résultats
            results_path = os.path.join(backtest_dir, "results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            
            # Mettre à jour les métadonnées de l'étude
            metadata = load_study_metadata(study_dir)
            if metadata:
                # Mettre à jour les métadonnées de backtests
                count = metadata.backtests.backtests_count + 1 if backtest_id not in [b.get("id") for b in metadata.backtests.backtests] else metadata.backtests.backtests_count
                
                metadata.backtests.backtests_count = count
                metadata.backtests.last_backtest = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Ajouter ou mettre à jour ce backtest dans la liste
                backtest_info = {
                    "id": backtest_id,
                    "strategy_id": strategy_id,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "performance": {
                        "roi": results.get("performance", {}).get("roi", 0),
                        "win_rate": results.get("performance", {}).get("win_rate", 0),
                        "total_trades": results.get("performance", {}).get("total_trades", 0),
                        "max_drawdown": results.get("performance", {}).get("max_drawdown", 0)
                    }
                }
                
                # Mettre à jour la liste des backtests
                for i, bt in enumerate(metadata.backtests.backtests):
                    if bt.get("id") == backtest_id:
                        metadata.backtests.backtests[i] = backtest_info
                        break
                else:
                    metadata.backtests.backtests.append(backtest_info)
                
                # Mettre à jour le statut si nécessaire
                if metadata.status == StudyStatus.OPTIMIZED:
                    metadata.status = StudyStatus.BACKTESTED
                
                # Sauvegarder les métadonnées
                metadata.update_last_modified()
                save_study_metadata(metadata, study_dir)
                
                # Mettre à jour la liste des études
                if self.studies_list:
                    self.studies_list.add_study(metadata)
                    save_study_list(self.studies_list, self.base_dir)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats de backtest: {e}")
            return False
    
    def load_backtest_results(self, study_name: str, backtest_id: str) -> Optional[Dict]:
        """
        Charge les résultats d'un backtest.
        
        Args:
            study_name: Nom de l'étude
            backtest_id: ID du backtest
            
        Returns:
            Optional[Dict]: Résultats du backtest ou None en cas d'erreur
        """
        if not self.study_exists(study_name):
            logger.warning(f"L'étude '{study_name}' n'existe pas")
            return None
        
        try:
            # Charger les résultats
            study_dir = self.get_study_dir(study_name)
            results_path = os.path.join(study_dir, "backtests", backtest_id, "results.json")
            
            if not os.path.exists(results_path):
                logger.warning(f"Résultats de backtest '{backtest_id}' introuvables pour l'étude '{study_name}'")
                return None
            
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des résultats de backtest: {e}")
            return None
    
    def get_study_data_file(self, study_name: str) -> Optional[str]:
        """
        Récupère le chemin du fichier de données associé à une étude.
        
        Args:
            study_name: Nom de l'étude
            
        Returns:
            Optional[str]: Chemin du fichier de données ou None en cas d'erreur
        """
        from simulator.config import load_study_metadata
        
        if not self.study_exists(study_name):
            logger.warning(f"L'étude '{study_name}' n'existe pas")
            return None
        
        try:
            # Charger les métadonnées
            study_dir = self.get_study_dir(study_name)
            metadata = load_study_metadata(study_dir)
            
            if metadata is None:
                logger.error(f"Impossible de charger les métadonnées de l'étude '{study_name}'")
                return None
            
            # Récupérer le chemin du fichier
            if metadata.data.file_path:
                file_path = os.path.join(study_dir, metadata.data.file_path)
                
                if os.path.exists(file_path):
                    return file_path
                else:
                    logger.warning(f"Fichier de données introuvable: {file_path}")
                    return None
            else:
                logger.warning(f"Aucun fichier de données associé à l'étude '{study_name}'")
                return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du fichier de données: {e}")
            return None
    
    def import_data_file(self, study_name: str, file_path: str) -> bool:
        """
        Importe un fichier de données dans une étude.
        
        Args:
            study_name: Nom de l'étude
            file_path: Chemin du fichier à importer
            
        Returns:
            bool: True si l'importation a réussi, False sinon
        """
        from simulator.config import (
            DataMetadata, DataSource, load_study_metadata, 
            save_study_metadata, save_study_list
        )
        
        if not self.study_exists(study_name):
            logger.warning(f"L'étude '{study_name}' n'existe pas")
            return False
        
        if not os.path.exists(file_path):
            logger.warning(f"Fichier introuvable: {file_path}")
            return False
        
        try:
            # Charger les métadonnées
            study_dir = self.get_study_dir(study_name)
            metadata = load_study_metadata(study_dir)
            
            if metadata is None:
                logger.error(f"Impossible de charger les métadonnées de l'étude '{study_name}'")
                return False
            
            # Calculer le checksum du fichier
            file_checksum = self._calculate_file_checksum(file_path)
            
            # Copier le fichier dans le répertoire de l'étude
            data_dir = os.path.join(study_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            filename = os.path.basename(file_path)
            local_path = os.path.join(data_dir, filename)
            shutil.copy2(file_path, local_path)
            
            # Analyser les données et mettre à jour les métadonnées
            try:
                df = pd.read_csv(local_path)
                rows_count = len(df)
                columns = df.columns.tolist()
                
                # Déterminer les dates si la colonne timestamp est présente
                start_date = None
                end_date = None
                
                if 'timestamp' in df.columns:
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        start_date = df['timestamp'].min().strftime("%Y-%m-%d")
                        end_date = df['timestamp'].max().strftime("%Y-%m-%d")
                    except:
                        pass
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse des données: {e}")
                rows_count = 0
                columns = []
                start_date = None
                end_date = None
            
            # Mettre à jour les métadonnées de données
            metadata.data = DataMetadata(
                source=DataSource.STUDY,
                file_path=os.path.join("data", filename),
                exchange=metadata.exchange,
                symbol=metadata.asset,
                timeframe=metadata.timeframe,
                rows_count=rows_count,
                columns=columns,
                checksum=file_checksum,
                start_date=start_date,
                end_date=end_date
            )
            
            # Sauvegarder les métadonnées
            metadata.update_last_modified()
            save_study_metadata(metadata, study_dir)
            
            # Mettre à jour la liste des études
            if self.studies_list:
                self.studies_list.add_study(metadata)
                save_study_list(self.studies_list, self.base_dir)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'importation du fichier de données: {e}")
            return False
    
    def list_strategies(self, study_name: str) -> List[Dict]:
        """
        Liste toutes les stratégies d'une étude.
        
        Args:
            study_name: Nom de l'étude
            
        Returns:
            List[Dict]: Liste des informations sur les stratégies
        """
        from simulator.config import load_study_metadata
        
        if not self.study_exists(study_name):
            logger.warning(f"L'étude '{study_name}' n'existe pas")
            return []
        
        try:
            # Charger les métadonnées
            study_dir = self.get_study_dir(study_name)
            metadata = load_study_metadata(study_dir)
            
            if metadata is None:
                logger.error(f"Impossible de charger les métadonnées de l'étude '{study_name}'")
                return []
            
            # Récupérer la liste des stratégies
            return metadata.strategies.strategies
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la liste des stratégies: {e}")
            return []
    
    def get_best_strategy_id(self, study_name: str) -> Optional[str]:
        """
        Récupère l'ID de la meilleure stratégie d'une étude.
        
        Args:
            study_name: Nom de l'étude
            
        Returns:
            Optional[str]: ID de la meilleure stratégie ou None en cas d'erreur
        """
        from simulator.config import load_study_metadata
        
        if not self.study_exists(study_name):
            logger.warning(f"L'étude '{study_name}' n'existe pas")
            return None
        
        try:
            # Charger les métadonnées
            study_dir = self.get_study_dir(study_name)
            metadata = load_study_metadata(study_dir)
            
            if metadata is None:
                logger.error(f"Impossible de charger les métadonnées de l'étude '{study_name}'")
                return None
            
            # Récupérer l'ID de la meilleure stratégie
            return metadata.strategies.best_strategy_id
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la meilleure stratégie: {e}")
            return None
    
    def search_studies(
        self, 
        query: str = None, 
        tags: List[str] = None, 
        categories: List[str] = None,
        status: Union[str, 'StudyStatus'] = None
    ) -> List[Dict]:
        """
        Recherche des études selon différents critères.
        
        Args:
            query: Chaîne de recherche dans le nom ou la description
            tags: Liste des tags à rechercher
            categories: Liste des catégories à rechercher
            status: Statut à filtrer
            
        Returns:
            List[Dict]: Liste des métadonnées des études correspondantes
        """
        from simulator.config import StudyStatus
        
        studies = self.get_studies_list()
        results = []
        
        # Conversion du statut en chaîne si nécessaire
        status_str = status.value if isinstance(status, StudyStatus) else status
        
        for study in studies:
            # Filtre par query
            if query and query.lower() not in study.get("name", "").lower() and query.lower() not in study.get("description", "").lower():
                continue
            
            # Filtre par tags
            if tags and not all(tag in study.get("tags", []) for tag in tags):
                continue
            
            # Filtre par catégories
            if categories and not all(cat in study.get("categories", []) for cat in categories):
                continue
            
            # Filtre par statut
            if status_str and study.get("status") != status_str:
                continue
            
            results.append(study)
        
        return results
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """
        Calcule le checksum SHA-256 d'un fichier.
        
        Args:
            file_path: Chemin du fichier
            
        Returns:
            str: Checksum SHA-256
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Lire le fichier par morceaux pour les gros fichiers
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()

# Pour la compatibilité avec le code existant
def get_integrated_study_manager(base_dir: str = "studies") -> IntegratedStudyManager:
    """
    Récupère une instance du gestionnaire d'études intégré.
    
    Args:
        base_dir: Répertoire de base pour le stockage des études
        
    Returns:
        IntegratedStudyManager: Instance du gestionnaire d'études
    """
    return IntegratedStudyManager(base_dir)