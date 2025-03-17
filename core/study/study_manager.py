"""
Gestionnaire d'études de trading.
Responsable de la coordination des fonctionnalités liées aux études,
en déléguant les opérations de base de données à DBOperations.
"""
import os
import json
import logging
import shutil
from typing import Dict, List, Optional, Union, Any

from core.study.study_config import StudyStatus
from core.optimization.search_config import SearchSpace, get_predefined_search_space
from core.db_study.db_operations import create_db_operations, DBOperations
from data.data_manager import get_data_manager

logger = logging.getLogger(__name__)

class StudyManager:
    """
    Gestionnaire d'études qui coordonne les fonctionnalités liées aux études,
    en déléguant les opérations de base de données à DBOperations.
    """
    
    def __init__(self, base_dir: str = "studies", db_url: Optional[str] = None):
        """
        Initialise le gestionnaire d'études.
        
        Args:
            base_dir: Répertoire de base pour les études
            db_url: URL de connexion à la base de données (facultatif)
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # Création du gestionnaire d'opérations DB
        self.db = create_db_operations(db_url)
        
        # Gestionnaire de données
        self.data_manager = get_data_manager()
    
    def create_study(
        self,
        name: str,
        description: str = "",
        timeframe: str = "1h",
        asset: str = "BTC/USDT",
        exchange: str = "binance",
        tags: List[str] = None,
        search_space_type: str = "default"
    ) -> Optional[str]:
        """
        Crée une nouvelle étude.
        
        Args:
            name: Nom de l'étude
            description: Description de l'étude
            timeframe: Timeframe de l'étude
            asset: Actif étudié
            exchange: Exchange utilisé
            tags: Tags associés à l'étude
            search_space_type: Type d'espace de recherche
        
        Returns:
            Optional[str]: Nom de l'étude créée ou None en cas d'erreur
        """
        try:
            # Prépare la configuration des données
            from data.data_config import MarketDataConfig, Timeframe, Exchange
            data_config = MarketDataConfig()
            data_config.symbol = asset
            try:
                data_config.timeframe = Timeframe(timeframe)
            except:
                data_config.timeframe = timeframe
            try:
                data_config.exchange = Exchange(exchange)
            except:
                data_config.exchange = exchange
            
            # Crée le répertoire de l'étude
            study_dir = os.path.join(self.base_dir, name)
            for subdir in ["data", "strategies", "backtests", "optimizations"]:
                os.makedirs(os.path.join(study_dir, subdir), exist_ok=True)
            
            # Prépare l'espace de recherche
            search_space = get_predefined_search_space(search_space_type)
            search_space_dict = search_space.to_dict()
            
            # Sauvegarde l'espace de recherche sur le disque (pour la rétrocompatibilité)
            optuna_dir = os.path.join(study_dir, "optimizations")
            search_space_path = os.path.join(optuna_dir, "search_config.json")
            with open(search_space_path, 'w', encoding='utf-8') as f:
                json.dump(search_space_dict, f, indent=4, ensure_ascii=False)
            
            # Crée l'étude dans la base de données
            result = self.db.create_study(
                name=name,
                description=description,
                asset=asset,
                timeframe=timeframe,
                exchange=exchange,
                tags=tags or [],
                data_config=data_config.to_dict(),
                search_space_config=search_space_dict,
                study_path=study_dir
            )
            
            if result:
                logger.info(f"Étude '{name}' créée avec succès")
                return name
            else:
                logger.error(f"Erreur lors de la création de l'étude '{name}'")
                return None
        
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'étude '{name}': {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_study(self, study_name: str) -> Optional[Dict]:
        """
        Récupère les informations d'une étude.
        
        Args:
            study_name: Nom de l'étude
        
        Returns:
            Optional[Dict]: Informations de l'étude ou None
        """
        return self.db.get_study(study_name)
    
    def delete_study(self, study_name: str) -> bool:
        """
        Supprime une étude existante.
        
        Args:
            study_name: Nom de l'étude à supprimer
        
        Returns:
            bool: True si la suppression a réussi
        """
        # Récupère le chemin de l'étude avant de la supprimer de la DB
        study_path = self.db.get_study_path(study_name)
        
        # Supprime l'étude de la base de données
        if self.db.delete_study(study_name):
            # Supprime également le répertoire de l'étude
            if study_path and os.path.exists(study_path):
                try:
                    shutil.rmtree(study_path, ignore_errors=True)
                    logger.info(f"Répertoire de l'étude '{study_name}' supprimé: {study_path}")
                except Exception as e:
                    logger.warning(f"Erreur lors de la suppression du répertoire: {str(e)}")
            
            return True
        
        return False
    
    def list_studies(self) -> List[Dict]:
        """
        Liste toutes les études disponibles avec leurs métadonnées.
        
        Returns:
            List[Dict]: Liste des études
        """
        # Récupère les études depuis la base de données
        studies = self.db.list_studies()
        
        # Enrichit les informations avec les données disponibles
        for study in studies:
            study_name = study.get("name")
            if study_name:
                # Compte le nombre de fichiers de données
                data_list = self.data_manager.get_data_for_study(study_name)
                study["data_count"] = len(data_list)
                
                # Détermine le type d'espace de recherche
                study_info = self.db.get_study(study_name)
                if study_info and "search_space_config" in study_info and study_info["search_space_config"]:
                    search_space_config = study_info["search_space_config"]
                    study["has_search_space"] = True
                    study["search_space_type"] = search_space_config.get("name", "custom")
                else:
                    study["has_search_space"] = False
                    study["search_space_type"] = "unknown"
        
        return studies
    
    def update_study_status(self, study_name: str, status: Union[str, StudyStatus]) -> bool:
        """
        Met à jour le statut d'une étude.
        
        Args:
            study_name: Nom de l'étude
            status: Nouveau statut
        
        Returns:
            bool: True si la mise à jour a réussi
        """
        # Convertit le statut si nécessaire
        if isinstance(status, StudyStatus):
            status_value = status.value
        else:
            try:
                # Vérifie que c'est un statut valide
                status_value = StudyStatus(status).value
            except ValueError:
                logger.error(f"Statut invalide: {status}")
                return False
        
        return self.db.update_study_status(study_name, status_value)
    
    def update_study_search_space(self, study_name: str, search_space: Union[Dict, SearchSpace]) -> bool:
        """
        Met à jour l'espace de recherche d'une étude.
        
        Args:
            study_name: Nom de l'étude
            search_space: Nouvel espace de recherche
        
        Returns:
            bool: True si la mise à jour a réussi
        """
        # Convertit l'espace de recherche si nécessaire
        if isinstance(search_space, SearchSpace):
            search_space_dict = search_space.to_dict()
        else:
            search_space_dict = search_space
        
        # Met à jour l'espace de recherche dans la base de données
        if self.db.update_study_search_space(study_name, search_space_dict):
            # Sauvegarde également l'espace de recherche sur le disque (pour la rétrocompatibilité)
            study_path = self.db.get_study_path(study_name)
            if study_path:
                optuna_dir = os.path.join(study_path, "optimizations")
                os.makedirs(optuna_dir, exist_ok=True)
                search_space_path = os.path.join(optuna_dir, "search_config.json")
                with open(search_space_path, 'w', encoding='utf-8') as f:
                    json.dump(search_space_dict, f, indent=4, ensure_ascii=False)
            
            return True
        
        return False
    
    def get_study_search_space(self, study_name: str) -> Optional[SearchSpace]:
        """
        Récupère l'espace de recherche d'une étude.
        
        Args:
            study_name: Nom de l'étude
        
        Returns:
            Optional[SearchSpace]: Espace de recherche ou None
        """
        # Récupère l'étude
        study = self.db.get_study(study_name)
        if not study:
            logger.warning(f"L'étude '{study_name}' n'existe pas")
            return None
        
        # Extrait l'espace de recherche
        if "search_space_config" not in study or not study["search_space_config"]:
            logger.warning(f"L'étude '{study_name}' n'a pas d'espace de recherche configuré")
            return None
        
        try:
            return SearchSpace.from_dict(study["search_space_config"])
        except Exception as e:
            logger.error(f"Erreur lors de la conversion de l'espace de recherche: {str(e)}")
            return None
    
    def get_study_path(self, study_name: str) -> Optional[str]:
        """
        Récupère le chemin du répertoire d'une étude.
        
        Args:
            study_name: Nom de l'étude
        
        Returns:
            Optional[str]: Chemin du répertoire ou None
        """
        return self.db.get_study_path(study_name)
    
    def save_strategy(self, study_name: str, strategy_id: str, config: Dict) -> bool:
        """
        Sauvegarde une stratégie pour une étude.
        
        Args:
            study_name: Nom de l'étude
            strategy_id: Identifiant de la stratégie
            config: Configuration de la stratégie
        
        Returns:
            bool: True si la sauvegarde a réussi
        """
        # Sauvegarde la stratégie dans la base de données
        if self.db.save_strategy(study_name, strategy_id, config):
            # Sauvegarde également la configuration sur le disque (pour la rétrocompatibilité)
            study_path = self.db.get_study_path(study_name)
            if study_path:
                strategies_dir = os.path.join(study_path, "strategies")
                os.makedirs(strategies_dir, exist_ok=True)
                strategy_dir = os.path.join(strategies_dir, strategy_id)
                os.makedirs(strategy_dir, exist_ok=True)
                config_path = os.path.join(strategy_dir, "config.json")
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
            
            return True
        
        return False
    
    def list_strategies(self, study_name: str) -> List[Dict]:
        """
        Liste toutes les stratégies d'une étude.
        
        Args:
            study_name: Nom de l'étude
        
        Returns:
            List[Dict]: Liste des stratégies
        """
        return self.db.list_strategies(study_name)
    
    def save_backtest(self, study_name: str, strategy_id: str, backtest_id: str, results: Dict) -> bool:
        """
        Sauvegarde les résultats d'un backtest.
        
        Args:
            study_name: Nom de l'étude
            strategy_id: Identifiant de la stratégie
            backtest_id: Identifiant du backtest
            results: Résultats du backtest
        
        Returns:
            bool: True si la sauvegarde a réussi
        """
        # Sauvegarde le backtest dans la base de données
        if self.db.save_backtest(study_name, strategy_id, backtest_id, results):
            # Sauvegarde également les résultats sur le disque (pour la rétrocompatibilité)
            study_path = self.db.get_study_path(study_name)
            if study_path:
                backtest_dir = os.path.join(study_path, "strategies", strategy_id, "backtests")
                os.makedirs(backtest_dir, exist_ok=True)
                backtest_path = os.path.join(backtest_dir, f"{backtest_id}.json")
                with open(backtest_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
            
            return True
        
        return False
    
    def list_backtests(self, study_name: str, strategy_id: Optional[str] = None) -> List[Dict]:
        """
        Liste tous les backtests d'une étude ou d'une stratégie.
        
        Args:
            study_name: Nom de l'étude
            strategy_id: Identifiant de la stratégie (optionnel)
        
        Returns:
            List[Dict]: Liste des backtests
        """
        return self.db.list_backtests(study_name, strategy_id)

def create_study_manager(base_dir: str = "studies", db_url: Optional[str] = None) -> StudyManager:
    """
    Crée une instance du gestionnaire d'études.
    
    Args:
        base_dir: Répertoire de base pour les études
        db_url: URL de connexion à la base de données (facultatif)
    
    Returns:
        StudyManager: Instance du gestionnaire
    """
    return StudyManager(base_dir, db_url)