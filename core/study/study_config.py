"""
Module de configuration pour les études de trading.
Définit les classes de données et énumérations nécessaires pour une gestion flexible des études.
"""
import os
import json
import hashlib
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Set

# Import des modules existants pour meilleure intégration
from data.data_config import MarketDataConfig, Exchange, Timeframe
from core.strategy.constructor.constructor_config import StrategyConfig
from core.simulation.simulation_config import SimulationConfig

class StudyStatus(Enum):
    """Statut possible pour une étude"""
    CREATED = "created"      # Étude nouvellement créée
    OPTIMIZED = "optimized"  # Étude optimisée
    BACKTESTED = "backtested" # Étude avec backtest
    LIVE = "live"            # Étude en production
    ARCHIVED = "archived"    # Étude archivée
    ERROR = "error"          # Étude en erreur

@dataclass
class StudyMetadata:
    """Métadonnées complètes d'une étude"""
    # Informations de base
    name: str
    description: Optional[str] = None
    
    # Configuration des données de marché
    data_config: MarketDataConfig = field(default_factory=MarketDataConfig)
    optuna_db_path: Optional[str] = None
    
    # Dates et statut
    creation_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    last_modified: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    status: StudyStatus = StudyStatus.CREATED
    
    # Tags et catégorisation
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    
    # Paramètres supplémentaires
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialisation post-création"""
        # Pour faciliter l'accès direct à certaines propriétés
        self._asset = None
        self._timeframe = None
        self._exchange = None
    
    @property
    def asset(self) -> str:
        """Retourne le symbole de l'actif"""
        return self.data_config.symbol
    
    @property
    def timeframe(self) -> str:
        """Retourne le timeframe sous forme de chaîne"""
        if isinstance(self.data_config.timeframe, Timeframe):
            return self.data_config.timeframe.value
        return str(self.data_config.timeframe)
    
    @property
    def exchange(self) -> str:
        """Retourne l'exchange sous forme de chaîne"""
        if isinstance(self.data_config.exchange, Exchange):
            return self.data_config.exchange.value
        return str(self.data_config.exchange)
    
    def to_dict(self) -> Dict:
        """Convertit les métadonnées en dictionnaire pour la sérialisation"""
        metadata_dict = {
            "name": self.name,
            "description": self.description,
            "data_config": self.data_config.to_dict(),
            "creation_date": self.creation_date,
            "last_modified": self.last_modified,
            "status": self.status.value,
            "tags": self.tags,
            "categories": self.categories,
            "custom_params": self.custom_params
        }
        return metadata_dict
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StudyMetadata':
        """Crée un objet StudyMetadata à partir d'un dictionnaire"""
        data_copy = data.copy()
        
        # Convertir les chaînes en énumérations
        if "status" in data_copy:
            data_copy["status"] = StudyStatus(data_copy["status"])
        
        # Convertir les structures imbriquées
        if "data_config" in data_copy:
            data_copy["data_config"] = MarketDataConfig.from_dict(data_copy["data_config"])
        
        return cls(**data_copy)
    
    def update_last_modified(self) -> None:
        """Met à jour la date de dernière modification"""
        self.last_modified = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@dataclass
class StudyConfig:
    """Configuration complète d'une étude de trading"""
    # Métadonnées
    metadata: StudyMetadata
    
    # Configuration de trading (stratégie)
    trading_config: StrategyConfig = field(default_factory=StrategyConfig)
    
    # Configuration de simulation
    simulation_config: SimulationConfig = field(default_factory=SimulationConfig)
    
    # Configuration d'optimisation
    optimization_config: Dict[str, Any] = field(default_factory=dict)
    
    search_space_config: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convertit la configuration en dictionnaire"""
        return {
            "metadata": self.metadata.to_dict(),
            "trading_config": self.trading_config.to_dict(),
            "simulation_config": self.simulation_config.to_dict(),
            "optimization_config": self.optimization_config,
            "search_space": self.search_space_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StudyConfig':
        """Crée une configuration à partir d'un dictionnaire"""
        # Créer les sous-configurations
        metadata = StudyMetadata.from_dict(data.get("metadata", {}))
        
        trading_config = None
        if "trading_config" in data:
            trading_config = StrategyConfig.from_dict(data["trading_config"])
        
        simulation_config = None
        if "simulation_config" in data:
            simulation_config = SimulationConfig.from_dict(data["simulation_config"])
        
        # Créer l'objet StudyConfig
        config = cls(metadata=metadata)
        
        if trading_config:
            config.trading_config = trading_config
        
        if simulation_config:
            config.simulation_config = simulation_config
        
        if "optimization_config" in data:
            config.optimization_config = data["optimization_config"]
        
        if "search_space_config" in data:
            config.search_space = data["search_space_config"]
        return config

@dataclass
class Study:
    """Représentation complète d'une étude de trading avec ses fichiers et données"""
    # Configuration de l'étude
    config: StudyConfig
    
    # Chemins des fichiers
    base_dir: str = "studies"
    
    # Métadonnées de stratégies et backtests
    strategies: List[Dict[str, Any]] = field(default_factory=list)
    backtests: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def name(self) -> str:
        """Retourne le nom de l'étude"""
        return self.config.metadata.name
    
    @property
    def status(self) -> StudyStatus:
        """Retourne le statut de l'étude"""
        return self.config.metadata.status
    
    def get_study_dir(self) -> str:
        """Retourne le chemin du répertoire de l'étude"""
        return os.path.join(self.base_dir, self.name)
    
    def save(self) -> str:
        """
        Sauvegarde l'étude sur disque
        
        Returns:
            str: Chemin du répertoire de l'étude
        """
        # Créer le répertoire de l'étude
        study_dir = self.get_study_dir()
        os.makedirs(study_dir, exist_ok=True)
        
        # Créer les sous-répertoires
        for subdir in ["data", "strategies", "backtests", "optimizations"]:
            os.makedirs(os.path.join(study_dir, subdir), exist_ok=True)
        
        # Mise à jour de la date de dernière modification
        self.config.metadata.update_last_modified()
        
        # Sauvegarder la configuration
        config_path = os.path.join(study_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, indent=4, ensure_ascii=False)
        
        return study_dir
    
    @classmethod
    def load(cls, study_name: str, base_dir: str = "studies") -> Optional['Study']:
        """
        Charge une étude depuis un répertoire
        
        Args:
            study_name: Nom de l'étude
            base_dir: Répertoire de base
            
        Returns:
            Optional[Study]: Étude chargée ou None si non trouvée
        """
        study_dir = os.path.join(base_dir, study_name)
        
        if not os.path.isdir(study_dir):
            return None
        
        try:
            # Charger la configuration
            config_path = os.path.join(study_dir, "config.json")
            if not os.path.exists(config_path):
                return None
                
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
                
            study_config = StudyConfig.from_dict(config_dict)
            
            # Créer l'objet Study
            study = cls(config=study_config, base_dir=base_dir)
            
            # TODO: Charger les stratégies et backtests
            # (pourrait être implémenté selon les besoins)
            
            return study
            
        except Exception as e:
            print(f"Erreur lors du chargement de l'étude: {e}")
            return None
    
    def update_status(self, status: StudyStatus) -> None:
        """
        Met à jour le statut de l'étude
        
        Args:
            status: Nouveau statut
        """
        self.config.metadata.status = status
        self.config.metadata.update_last_modified()
        self.save()

@dataclass
class StudyList:
    """Liste des études disponibles avec métadonnées de base"""
    studies: List[Dict] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def add_study(self, metadata: StudyMetadata) -> None:
        """
        Ajoute une étude à la liste
        
        Args:
            metadata: Métadonnées de l'étude
        """
        # Extraction des informations essentielles
        study_info = {
            "name": metadata.name,
            "description": metadata.description,
            "asset": metadata.asset,
            "timeframe": metadata.timeframe,
            "exchange": metadata.exchange,
            "creation_date": metadata.creation_date,
            "last_modified": metadata.last_modified,
            "status": metadata.status.value,
            "tags": metadata.tags,
            "categories": metadata.categories
        }
        
        # Mise à jour si l'étude existe déjà
        for i, study in enumerate(self.studies):
            if study.get("name") == metadata.name:
                self.studies[i] = study_info
                self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return
        
        # Ajout d'une nouvelle étude
        self.studies.append(study_info)
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def remove_study(self, study_name: str) -> bool:
        """
        Supprime une étude de la liste
        
        Args:
            study_name: Nom de l'étude à supprimer
            
        Returns:
            bool: True si l'étude a été supprimée
        """
        for i, study in enumerate(self.studies):
            if study.get("name") == study_name:
                del self.studies[i]
                self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return True
        
        return False
    
    def to_dict(self) -> Dict:
        """Convertit la liste en dictionnaire pour la sérialisation"""
        return {
            "studies": self.studies,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StudyList':
        """Crée un objet StudyList à partir d'un dictionnaire"""
        return cls(
            studies=data.get("studies", []),
            last_updated=data.get("last_updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
    
    def save(self, base_dir: str) -> str:
        """
        Sauvegarde la liste des études dans un fichier
        
        Args:
            base_dir: Répertoire de base
            
        Returns:
            str: Chemin du fichier sauvegardé
        """
        # Création du répertoire si nécessaire
        os.makedirs(base_dir, exist_ok=True)
        
        # Mise à jour de la date
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Sauvegarde du fichier
        file_path = os.path.join(base_dir, "studies_list.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)
            
        return file_path
    
    @classmethod
    def load(cls, base_dir: str) -> 'StudyList':
        """
        Charge la liste des études depuis un fichier
        
        Args:
            base_dir: Répertoire de base
            
        Returns:
            StudyList: Liste des études
        """
        file_path = os.path.join(base_dir, "studies_list.json")
        
        if not os.path.exists(file_path):
            return cls()
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            return cls.from_dict(data)
        except Exception as e:
            print(f"Erreur lors du chargement de la liste des études: {e}")
            return cls()