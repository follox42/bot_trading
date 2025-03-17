"""
Module de configuration centralisée pour le système de trading.
Version avec paramètres flexibles pour permettre une meilleure adaptabilité.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Union, Tuple
import logging
import json
import os
from datetime import datetime
from enum import Enum, auto
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler, NSGAIISampler
from optuna.pruners import MedianPruner, PercentilePruner, HyperbandPruner
import numpy as np

# Importation des modules du système de trading
from .indicators import IndicatorType, Operator, LogicOperator
from .risk import RiskMode
from .simulator import SimulationConfig
"""
    _____________________________________  Data  _____________________________________
"""
@dataclass
class DataMetadata:
    """Métadonnées concernant les données utilisées"""
    file_path: Optional[str] = None
    exchange: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    rows_count: int = 0
    columns: List[str] = field(default_factory=list)
    checksum: Optional[str] = None  # Pour vérifier l'intégrité des données
    
    def to_dict(self) -> Dict:
        """Convertit les métadonnées en dictionnaire"""
        return {
            "source": self.source.value,
            "file_path": self.file_path,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "rows_count": self.rows_count,
            "columns": self.columns,
            "checksum": self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DataMetadata':
        """Crée un objet DataMetadata à partir d'un dictionnaire"""
        metadata = cls()
        
        if "source" in data:
            metadata.source = DataSource(data["source"])
        
        for field in ["file_path", "exchange", "symbol", "timeframe", 
                     "start_date", "end_date", "rows_count", "checksum"]:
            if field in data:
                setattr(metadata, field, data[field])
        
        if "columns" in data:
            metadata.columns = data["columns"]
        
        return metadata

"""
    _____________________________________  Study  _____________________________________
"""
class StudyStatus(Enum):
    """Statuts possibles pour une étude"""
    CREATED = "created"
    OPTIMIZED = "optimized"
    BACKTESTED = "backtested"
    LIVE = "live"
    ARCHIVED = "archived"
    ERROR = "error"

@dataclass
class DataMetadata:
    """Métadonnées concernant les données utilisées"""
    file_path: Optional[str] = None
    exchange: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    rows_count: int = 0
    columns: List[str] = field(default_factory=list)
    checksum: Optional[str] = None  # Pour vérifier l'intégrité des données
    
    def to_dict(self) -> Dict:
        """Convertit les métadonnées en dictionnaire"""
        return {
            "source": self.source.value,
            "file_path": self.file_path,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "rows_count": self.rows_count,
            "columns": self.columns,
            "checksum": self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DataMetadata':
        """Crée un objet DataMetadata à partir d'un dictionnaire"""
        metadata = cls()
        
        if "source" in data:
            metadata.source = DataSource(data["source"])
        
        for field in ["file_path", "exchange", "symbol", "timeframe", 
                     "start_date", "end_date", "rows_count", "checksum"]:
            if field in data:
                setattr(metadata, field, data[field])
        
        if "columns" in data:
            metadata.columns = data["columns"]
        
        return metadata

@dataclass
class StudyMetadata:
    """Métadonnées complètes d'une étude"""
    # Informations de base
    name: str
    description: Optional[str] = None
    asset: Optional[str] = None
    timeframe: Optional[str] = None
    exchange: Optional[str] = None
    
    # Dates et statut
    creation_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    last_modified: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    status: StudyStatus = StudyStatus.CREATED
    
    # Métadonnées spécifiques
    data: DataMetadata = field(default_factory=DataMetadata)
    optimization: OptimizationMetadata = field(default_factory=OptimizationMetadata)
    strategies: StrategyMetadata = field(default_factory=StrategyMetadata)
    backtests: BacktestMetadata = field(default_factory=BacktestMetadata)
    
    # Métriques globales
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    
    # Tags et catégorisation
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    
    # Paramètres supplémentaires
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convertit les métadonnées en dictionnaire pour la sérialisation"""
        return {
            "name": self.name,
            "description": self.description,
            "asset": self.asset,
            "timeframe": self.timeframe,
            "exchange": self.exchange,
            "creation_date": self.creation_date,
            "last_modified": self.last_modified,
            "status": self.status.value,
            "data": self.data.to_dict(),
            "optimization": self.optimization.to_dict(),
            "strategies": self.strategies.to_dict(),
            "backtests": self.backtests.to_dict(),
            "performance": self.performance.to_dict(),
            "tags": self.tags,
            "categories": self.categories,
            "custom_params": self.custom_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StudyMetadata':
        """Crée un objet StudyMetadata à partir d'un dictionnaire"""
        # Champs obligatoires
        if "name" not in data:
            raise ValueError("Le champ 'name' est obligatoire pour les métadonnées d'étude")
        
        metadata = cls(name=data["name"])
        
        # Champs simples
        for field in ["description", "asset", "timeframe", "exchange", 
                     "creation_date", "last_modified"]:
            if field in data:
                setattr(metadata, field, data[field])
        
        # Status
        if "status" in data:
            metadata.status = StudyStatus(data["status"])
        
        # Sous-objets
        if "data" in data:
            metadata.data = DataMetadata.from_dict(data["data"])
        
        if "optimization" in data:
            metadata.optimization = OptimizationMetadata.from_dict(data["optimization"])
        
        if "strategies" in data:
            metadata.strategies = StrategyMetadata.from_dict(data["strategies"])
        
        if "backtests" in data:
            metadata.backtests = BacktestMetadata.from_dict(data["backtests"])
        
        if "performance" in data:
            metadata.performance = PerformanceMetrics.from_dict(data["performance"])
        
        # Collections
        if "tags" in data:
            metadata.tags = data["tags"]
        
        if "categories" in data:
            metadata.categories = data["categories"]
        
        if "custom_params" in data:
            metadata.custom_params = data["custom_params"]
        
        return metadata
    
    def update_last_modified(self) -> None:
        """Met à jour la date de dernière modification"""
        self.last_modified = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_study_metadata(metadata: StudyMetadata, study_dir: str) -> bool:
    """
    Sauvegarde les métadonnées d'une étude dans un fichier JSON
    
    Args:
        metadata: Métadonnées de l'étude
        study_dir: Répertoire de l'étude
        
    Returns:
        bool: True si la sauvegarde a réussi
    """
    try:
        # Création du répertoire si nécessaire
        os.makedirs(study_dir, exist_ok=True)
        
        # Mise à jour de la date de dernière modification
        metadata.update_last_modified()
        
        # Chemin du fichier de métadonnées
        metadata_path = os.path.join(study_dir, "metadata.json")
        
        # Sérialisation et sauvegarde
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=4, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des métadonnées: {str(e)}")
        return False

def load_study_metadata(study_dir: str) -> Optional[StudyMetadata]:
    """
    Charge les métadonnées d'une étude depuis un fichier JSON
    
    Args:
        study_dir: Répertoire de l'étude
        
    Returns:
        Optional[StudyMetadata]: Métadonnées de l'étude ou None en cas d'erreur
    """
    try:
        # Chemin du fichier de métadonnées
        metadata_path = os.path.join(study_dir, "metadata.json")
        
        if not os.path.exists(metadata_path):
            print(f"Le fichier de métadonnées n'existe pas: {metadata_path}")
            return None
        
        # Chargement et désérialisation
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return StudyMetadata.from_dict(data)
    except Exception as e:
        print(f"Erreur lors du chargement des métadonnées: {str(e)}")
        return None

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
            "has_optimization": metadata.optimization.trials_count > 0,
            "strategies_count": metadata.strategies.strategies_count,
            "backtests_count": metadata.backtests.backtests_count,
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
        study_list = cls()
        
        if "studies" in data:
            study_list.studies = data["studies"]
        
        if "last_updated" in data:
            study_list.last_updated = data["last_updated"]
        
        return study_list

def save_study_list(study_list: StudyList, base_dir: str) -> bool:
    """
    Sauvegarde la liste des études dans un fichier JSON
    
    Args:
        study_list: Liste des études
        base_dir: Répertoire de base
        
    Returns:
        bool: True si la sauvegarde a réussi
    """
    try:
        # Création du répertoire si nécessaire
        os.makedirs(base_dir, exist_ok=True)
        
        # Chemin du fichier de liste
        list_path = os.path.join(base_dir, "studies_list.json")
        
        # Sérialisation et sauvegarde
        with open(list_path, 'w', encoding='utf-8') as f:
            json.dump(study_list.to_dict(), f, indent=4, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de la liste des études: {str(e)}")
        return False

def load_study_list(base_dir: str) -> StudyList:
    """
    Charge la liste des études depuis un fichier JSON
    
    Args:
        base_dir: Répertoire de base
        
    Returns:
        StudyList: Liste des études
    """
    try:
        # Chemin du fichier de liste
        list_path = os.path.join(base_dir, "studies_list.json")
        
        if not os.path.exists(list_path):
            return StudyList()
        
        # Chargement et désérialisation
        with open(list_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return StudyList.from_dict(data)
    except Exception as e:
        print(f"Erreur lors du chargement de la liste des études: {str(e)}")
        return StudyList()

def update_study_list(base_dir: str) -> StudyList:
    """
    Met à jour la liste des études en parcourant le répertoire
    
    Args:
        base_dir: Répertoire de base des études
        
    Returns:
        StudyList: Liste des études mise à jour
    """
    study_list = StudyList()
    
    try:
        # Parcourir les sous-répertoires d'études
        for study_name in os.listdir(base_dir):
            study_dir = os.path.join(base_dir, study_name)
            
            # Vérifier que c'est un répertoire
            if not os.path.isdir(study_dir):
                continue
            
            # Charger les métadonnées de l'étude
            metadata = load_study_metadata(study_dir)
            if metadata:
                study_list.add_study(metadata)
        
        # Sauvegarder la liste mise à jour
        save_study_list(study_list, base_dir)
        
        return study_list
    except Exception as e:
        print(f"Erreur lors de la mise à jour de la liste des études: {str(e)}")
        return study_list
    
"""
    _________________________________  Trading  _________________________________
"""
class MarginMode(Enum):
    """Types de modes de marge"""
    ISOLATED = 0
    CROSS = 1

class TradingMode(Enum):
    """Types de modes de trading"""
    ONE_WAY = 0
    HEDGE = 1

@dataclass
class IndicatorConfig:
    """Configuration d'un indicateur technique"""
    type: IndicatorType
    min_period: int
    max_period: int
    step: int = 1
    price_type: str = "close"

@dataclass
class RiskModeConfig:
    """Configuration spécifique pour chaque mode de risque"""
    # Paramètres pour le mode FIXED
    fixed_position_range: Tuple[float, float] = (0.01, 1.0)
    fixed_sl_range: Tuple[float, float] = (0.001, 0.1)
    fixed_tp_range: Tuple[float, float] = (0.001, 0.5)
    
    # Paramètres pour le mode ATR_BASED
    atr_period_range: Tuple[int, int] = (5, 30)
    atr_multiplier_range: Tuple[float, float] = (0.5, 5.0)
    
    # Paramètres pour le mode VOLATILITY_BASED
    vol_period_range: Tuple[int, int] = (10, 50)
    vol_multiplier_range: Tuple[float, float] = (0.5, 5.0)

@dataclass
class RiskConfig:
    """Configuration de la gestion du risque avec paramètres flexibles"""
    # Liste des modes disponibles
    available_modes: List[RiskMode] = field(default_factory=lambda: list(RiskMode))
    
    # Configurations par mode
    mode_configs: Dict[RiskMode, RiskModeConfig] = field(default_factory=dict)
    
    # Paramètres généraux de gestion du risque
    position_size_range: Tuple[float, float] = (0.01, 1.0)
    sl_range: Tuple[float, float] = (0.001, 0.1)
    tp_multiplier_range: Tuple[float, float] = (1.0, 5.0)
    
    def __post_init__(self):
        # Initialisation des configurations par mode si nécessaire
        if not self.mode_configs:
            for mode in self.available_modes:
                self.mode_configs[mode] = RiskModeConfig()

@dataclass
class SimConfig:
    """Configuration de simulation étendue"""
    initial_balance_range: Tuple[float, float] = (1000.0, 100000.0)
    fee: float = 0.01
    slippage: float = 0.01
    tick_size: float = 0.001
    leverage_range: Tuple[int, int] = (1, 125)
    margin_modes: List[MarginMode] = field(default_factory=lambda: list(MarginMode))
    trading_modes: List[TradingMode] = field(default_factory=lambda: list(TradingMode))
    min_trade_size: float = 0.001
    max_trade_size: float = 100000.0

class SimulationConfig:
    """Configuration des paramètres de simulation"""
    initial_balance: float = 10000.0
    fee_open: float = 0.001  # 0.1% par trade
    fee_close: float = 0.001
    slippage: float = 0.001
    tick_size: float = 0.01
    min_trade_size: float = 0.001
    max_trade_size: float = 100000.0
    leverage: int = 1
    margin_mode: int = 0  # 0=Isolated, 1=Cross
    trading_mode: int = 0  # 0=One-way, 1=Hedge
    
@dataclass
class StrategyStructureConfig:
    """Configuration de la structure des stratégies"""
    max_blocks: int = 3
    min_blocks: int = 1
    max_conditions_per_block: int = 3
    min_conditions_per_block: int = 1
    cross_signals_probability: float = 0.3
    value_comparison_probability: float = 0.4
    
    # Plages de valeurs pour les comparaisons
    rsi_value_range: Tuple[float, float] = (20.0, 80.0)
    price_value_range: Tuple[float, float] = (0.0, 1000.0)
    general_value_range: Tuple[float, float] = (-100.0, 100.0)

@dataclass
class TradingConfig:
    """Configuration globale flexible du système de trading"""
    available_indicators: Dict[str, IndicatorConfig] = field(default_factory=dict)
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    sim_config: SimConfig = field(default_factory=SimConfig)
    strategy_structure: StrategyStructureConfig = field(default_factory=StrategyStructureConfig)
    
    def add_indicator(self, name: str, config: IndicatorConfig) -> None:
        """Ajoute un indicateur à la configuration"""
        self.available_indicators[name] = config
    
    def get_indicator_types(self) -> List[str]:
        """Récupère la liste des types d'indicateurs disponibles"""
        return list(self.available_indicators.keys())
    
    def to_dict(self) -> Dict:
        """Convertit la configuration en dictionnaire pour la sérialisation"""
        return {
            'available_indicators': {
                name: {
                    'type': config.type.value,
                    'min_period': config.min_period,
                    'max_period': config.max_period,
                    'step': config.step,
                    'price_type': config.price_type
                } for name, config in self.available_indicators.items()
            },
            'risk_config': {
                'available_modes': [mode.value for mode in self.risk_config.available_modes],
                'mode_configs': {
                    mode.value: {
                        'fixed_position_range': cfg.fixed_position_range,
                        'fixed_sl_range': cfg.fixed_sl_range,
                        'fixed_tp_range': cfg.fixed_tp_range,
                        'atr_period_range': cfg.atr_period_range,
                        'atr_multiplier_range': cfg.atr_multiplier_range,
                        'vol_period_range': cfg.vol_period_range,
                        'vol_multiplier_range': cfg.vol_multiplier_range
                    } for mode, cfg in self.risk_config.mode_configs.items()
                },
                'position_size_range': self.risk_config.position_size_range,
                'sl_range': self.risk_config.sl_range,
                'tp_multiplier_range': self.risk_config.tp_multiplier_range
            },
            'sim_config': {
                'initial_balance_range': self.sim_config.initial_balance_range,
                'fee': self.sim_config.fee,
                'slippage': self.sim_config.slippage,
                'tick_size': self.sim_config.tick_size,
                'leverage_range': self.sim_config.leverage_range,
                'margin_modes': [mode.value for mode in self.sim_config.margin_modes],
                'trading_modes': [mode.value for mode in self.sim_config.trading_modes],
                'min_trade_size': self.sim_config.min_trade_size,
                'max_trade_size': self.sim_config.max_trade_size
            },
            'strategy_structure': {
                'max_blocks': self.strategy_structure.max_blocks,
                'min_blocks': self.strategy_structure.min_blocks,
                'max_conditions_per_block': self.strategy_structure.max_conditions_per_block,
                'min_conditions_per_block': self.strategy_structure.min_conditions_per_block,
                'cross_signals_probability': self.strategy_structure.cross_signals_probability,
                'value_comparison_probability': self.strategy_structure.value_comparison_probability,
                'rsi_value_range': self.strategy_structure.rsi_value_range,
                'price_value_range': self.strategy_structure.price_value_range,
                'general_value_range': self.strategy_structure.general_value_range
            }
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradingConfig':
        """Crée une configuration à partir d'un dictionnaire"""
        config = cls()
        
        # Chargement des indicateurs
        if 'available_indicators' in data:
            for name, ind_data in data['available_indicators'].items():
                config.available_indicators[name] = IndicatorConfig(
                    type=IndicatorType(ind_data['type']),
                    min_period=ind_data['min_period'],
                    max_period=ind_data['max_period'],
                    step=ind_data.get('step', 1),
                    price_type=ind_data.get('price_type', 'close')
                )
        
        # Chargement de la configuration de risque
        if 'risk_config' in data:
            rc = data['risk_config']
            
            # Chargement des modes disponibles
            if 'available_modes' in rc:
                config.risk_config.available_modes = [RiskMode(mode) for mode in rc['available_modes']]
            
            # Chargement des configurations par mode
            if 'mode_configs' in rc:
                for mode_str, mode_cfg in rc['mode_configs'].items():
                    mode = RiskMode(mode_str)
                    config.risk_config.mode_configs[mode] = RiskModeConfig(
                        fixed_position_range=mode_cfg.get('fixed_position_range', (0.01, 1.0)),
                        fixed_sl_range=mode_cfg.get('fixed_sl_range', (0.001, 0.1)),
                        fixed_tp_range=mode_cfg.get('fixed_tp_range', (0.001, 0.5)),
                        atr_period_range=mode_cfg.get('atr_period_range', (5, 30)),
                        atr_multiplier_range=mode_cfg.get('atr_multiplier_range', (0.5, 5.0)),
                        vol_period_range=mode_cfg.get('vol_period_range', (10, 50)),
                        vol_multiplier_range=mode_cfg.get('vol_multiplier_range', (0.5, 5.0))
                    )
            
            # Chargement des paramètres généraux de risque
            config.risk_config.position_size_range = rc.get('position_size_range', (0.01, 1.0))
            config.risk_config.sl_range = rc.get('sl_range', (0.001, 0.1))
            config.risk_config.tp_multiplier_range = rc.get('tp_multiplier_range', (1.0, 5.0))
        
        # Chargement de la configuration de simulation
        if 'sim_config' in data:
            sc = data['sim_config']
            
            # Création de la configuration de simulation
            config.sim_config = SimConfig(
                initial_balance_range=sc.get('initial_balance_range', (1000.0, 100000.0)),
                fee=sc.get('fee', 0.01),
                slippage=sc.get('slippage', 0.01),
                tick_size=sc.get('tick_size', 0.001),
                leverage_range=sc.get('leverage_range', (1, 125)),
                margin_modes=[MarginMode(mode) for mode in sc.get('margin_modes', [0, 1])],
                trading_modes=[TradingMode(mode) for mode in sc.get('trading_modes', [0, 1])],
                min_trade_size=sc.get('min_trade_size', 0.001),
                max_trade_size=sc.get('max_trade_size', 100000.0)
            )
        
        # Chargement de la structure de stratégie
        if 'strategy_structure' in data:
            ss = data['strategy_structure']
            config.strategy_structure = StrategyStructureConfig(
                max_blocks=ss.get('max_blocks', 3),
                min_blocks=ss.get('min_blocks', 1),
                max_conditions_per_block=ss.get('max_conditions_per_block', 3),
                min_conditions_per_block=ss.get('min_conditions_per_block', 1),
                cross_signals_probability=ss.get('cross_signals_probability', 0.3),
                value_comparison_probability=ss.get('value_comparison_probability', 0.4),
                rsi_value_range=ss.get('rsi_value_range', (20.0, 80.0)),
                price_value_range=ss.get('price_value_range', (0.0, 1000.0)),
                general_value_range=ss.get('general_value_range', (-100.0, 100.0))
            )
        
        return config

def create_trading_default_config() -> TradingConfig:
        """
        Crée une configuration de trading flexible par défaut
        
        Returns:
            FlexibleTradingConfig: Configuration par défaut
        """
        from simulator.indicators import IndicatorConfig
        
        # Configuration des indicateurs disponibles
        available_indicators = {
            "EMA": IndicatorConfig(min_period=5, max_period=200, step=5, description="Moyenne Mobile Exponentielle", price_type="close"),
            "SMA": IndicatorConfig(min_period=5, max_period=200, step=5, description="Moyenne Mobile Simple", price_type="close"),
            "RSI": IndicatorConfig(min_period=7, max_period=30, step=1, description="Relative Strength Index", price_type="close"),
            "MACD": IndicatorConfig(min_period=12, max_period=26, step=1, description="MACD", price_type="close"),
            "ATR": IndicatorConfig(min_period=7, max_period=30, step=1, description="Average True Range", price_type="close")
        }
        
        # Configuration de la structure des stratégies
        strategy_structure = StrategyStructureConfig(
            min_blocks=1,
            max_blocks=3,
            min_conditions_per_block=1,
            max_conditions_per_block=3,
            cross_signals_probability=0.3,
            value_comparison_probability=0.4,
            rsi_value_range=(20, 80),
            price_value_range=(0, 1000),
            general_value_range=(-100, 100)
        )
        
        # Configuration du risque
        risk_config = RiskConfig(
            position_size_range=(0.01, 0.1),  # 1% à 10% du capital
            sl_range=(0.005, 0.03),  # 0.5% à 3% stop loss
            tp_multiplier_range=(1.5, 3.0),  # TP = 1.5 à 3.0 fois le SL
            available_modes=[RiskMode.FIXED, RiskMode.ATR_BASED],
            mode_configs={
                RiskMode.FIXED: FixedRiskConfig(
                    fixed_position_range=(0.01, 0.1),
                    fixed_sl_range=(0.005, 0.03),
                    fixed_tp_range=(0.0075, 0.09)  # 1.5 à 3 fois le SL
                ),
                RiskMode.ATR_BASED: AtrRiskConfig(
                    atr_period_range=(7, 20),
                    atr_multiplier_range=(0.5, 2.0)
                )
            }
        )
        
        # Configuration de la simulation
        sim_config = SimulationConfig(
            initial_balance_range=(1000, 10000),
            fee=0.001,  # 0.1%
            slippage=0.0005,  # 0.05%
            leverage_range=(1, 10),
            margin_modes=[MarginMode.ISOLATED],
            trading_modes=[TradingMode.ONE_WAY],
            tick_size=0.01,
            min_trade_size=0.001,
            max_trade_size=10000.0
        )
        
        return FlexibleTradingConfig(
            available_indicators=available_indicators,
            strategy_structure=strategy_structure,
            risk_config=risk_config,
            sim_config=sim_config
        )

def load_trading_config_from_file(file_path: str) -> TradingConfig:
    """Charge une configuration flexible depuis un fichier JSON"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return TradingConfig.from_dict(data)
    except Exception as e:
        logging.error(f"Erreur lors du chargement de la configuration: {str(e)}")
        return create_trading_default_config()

def save_trading_config_to_file(config: TradingConfig, file_path: str) -> bool:
    """Sauvegarde une configuration flexible dans un fichier JSON"""
    try:
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Sauvegarder le fichier
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=4, ensure_ascii=False)
        
        return True
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde de la configuration: {str(e)}")
        return False

def convert_to_simulator_config(sim_config: SimConfig) -> SimulationConfig:
    """
    Convertit la configuration de simulation flexible en configuration pour le simulateur
    en utilisant des valeurs moyennes ou par défaut
    """
    return SimulationConfig(
        initial_balance=(sim_config.initial_balance_range[0] + sim_config.initial_balance_range[1]),
        fee_open=sim_config.fee,
        fee_close=sim_config.fee,
        slippage=sim_config.slippage,
        tick_size=sim_config.tick_size,
        min_trade_size=sim_config.min_trade_size,
        max_trade_size=sim_config.max_trade_size,
        leverage=min(10, sim_config.leverage_range[1]),  # Valeur par défaut raisonnable
        margin_mode=0,  # Mode isolé par défaut
        trading_mode=0   # Mode unidirectionnel par défaut
    )

def create_position_calculator_config(risk_config: RiskConfig, mode: RiskMode = RiskMode.FIXED) -> Dict:
    """
    Crée une configuration pour le PositionCalculator à partir de la configuration de risque flexible
    
    Args:
        risk_config: Configuration de risque flexible
        mode: Mode de risque à utiliser
        
    Returns:
        Dict: Configuration pour le PositionCalculator
    """
    # Configuration de base
    config = {
        'base_position': (risk_config.position_size_range[0] + risk_config.position_size_range[1]) / 2,
        'base_sl': (risk_config.sl_range[0] + risk_config.sl_range[1]) / 2,
        'tp_multiplier': (risk_config.tp_multiplier_range[0] + risk_config.tp_multiplier_range[1]) / 2,
        'position_size_range': risk_config.position_size_range,
        'sl_range': risk_config.sl_range
    }
    
    # Ajout des paramètres spécifiques au mode
    if mode == RiskMode.FIXED:
        mode_config = risk_config.mode_configs.get(RiskMode.FIXED, RiskModeConfig())
        config.update({
            'base_position': (mode_config.fixed_position_range[0] + mode_config.fixed_position_range[1]) / 2,
            'base_sl': (mode_config.fixed_sl_range[0] + mode_config.fixed_sl_range[1]) / 2
        })
    elif mode == RiskMode.ATR_BASED:
        mode_config = risk_config.mode_configs.get(RiskMode.ATR_BASED, RiskModeConfig())
        config.update({
            'atr_period': int((mode_config.atr_period_range[0] + mode_config.atr_period_range[1]) / 2),
            'atr_multiplier': (mode_config.atr_multiplier_range[0] + mode_config.atr_multiplier_range[1]) / 2
        })
    elif mode == RiskMode.VOLATILITY_BASED:
        mode_config = risk_config.mode_configs.get(RiskMode.VOLATILITY_BASED, RiskModeConfig())
        config.update({
            'vol_period': int((mode_config.vol_period_range[0] + mode_config.vol_period_range[1]) / 2),
            'vol_multiplier': (mode_config.vol_multiplier_range[0] + mode_config.vol_multiplier_range[1]) / 2
        })
    
    return config

"""
    _________________________________  Optimization  _________________________________
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Union, Tuple, Callable
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler, NSGAIISampler
from optuna.pruners import MedianPruner, PercentilePruner, HyperbandPruner

class OptimizationMethod(Enum):
    """Méthodes d'optimisation disponibles"""
    TPE = "tpe"
    RANDOM = "random"
    CMAES = "cmaes"
    NSGAII = "nsgaii"

class PrunerMethod(Enum):
    """Méthodes de pruning disponibles"""
    MEDIAN = "median"
    PERCENTILE = "percentile"
    HYPERBAND = "hyperband"
    NONE = "none"

class ScoringFormula(Enum):
    """Formules de scoring disponibles"""
    STANDARD = "standard"
    CONSISTENCY = "consistency"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    VOLUME = "volume"
    CUSTOM = "custom"

@dataclass
class ScoringWeights:
    """Poids pour les différentes métriques dans le calcul du score"""
    roi: float = 2.5
    win_rate: float = 0.5
    max_drawdown: float = 2.0
    profit_factor: float = 2.0
    total_trades: float = 1.0
    avg_profit: float = 1.0
    trades_per_day: float = 0.0
    max_consecutive_losses: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convertit les poids en dictionnaire"""
        return {
            'roi': self.roi,
            'win_rate': self.win_rate,
            'max_drawdown': self.max_drawdown,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'avg_profit': self.avg_profit,
            'trades_per_day': self.trades_per_day,
            'max_consecutive_losses': self.max_consecutive_losses
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'ScoringWeights':
        """Crée un objet ScoringWeights à partir d'un dictionnaire"""
        weights = cls()
        for key, value in data.items():
            if hasattr(weights, key):
                setattr(weights, key, value)
        return weights

@dataclass
class OptimizationMethodConfig:
    """Configuration d'une méthode d'optimisation"""
    name: str
    description: str
    sampler_class: Any
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convertit la configuration en dictionnaire"""
        return {
            "name": self.name,
            "description": self.description,
            "params": self.params
        }

@dataclass
class PrunerMethodConfig:
    """Configuration d'une méthode de pruning"""
    name: str
    description: str
    pruner_class: Any
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convertit la configuration en dictionnaire"""
        return {
            "name": self.name,
            "description": self.description,
            "params": self.params
        }

@dataclass
class OptimizationConfig:
    """Configuration des paramètres d'optimisation pour les stratégies de trading"""
    
    # Paramètres de base
    n_trials: int = 500
    timeout: Optional[int] = None
    n_jobs: int = -1  # -1 pour utiliser tous les cœurs disponibles
    memory_limit: float = 0.8  # Utilisation maximale de la mémoire (proportion)
    
    # Méthode d'optimisation
    optimization_method: OptimizationMethod = OptimizationMethod.TPE
    method_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_startup_trials": 10,
        "n_ei_candidates": 24,
        "multivariate": True,
        "group_related_params": True,
        "consider_magic_clip": True
    })
    
    # Pruning et early stopping
    enable_pruning: bool = False
    pruner_method: PrunerMethod = PrunerMethod.NONE
    pruner_params: Dict[str, Any] = field(default_factory=dict)
    early_stopping_n_trials: Optional[int] = None
    early_stopping_threshold: float = 0.0
    
    # Scoring
    scoring_formula: ScoringFormula = ScoringFormula.STANDARD
    custom_weights: ScoringWeights = field(default_factory=ScoringWeights)
    
    # Limites
    min_trades: int = 10  # Nombre minimum de trades pour qu'une stratégie soit valide
    
    # Paramètres avancés
    gc_after_trial: bool = True  # Garbage collection après chaque essai
    save_checkpoints: bool = True  # Sauvegarde des points de contrôle
    checkpoint_every: int = 10  # Fréquence des checkpoints (tous les N essais)
    debug: bool = False  # Mode debug
    
    def to_dict(self) -> Dict:
        """Convertit la configuration en dictionnaire pour la sérialisation"""
        return {
            'n_trials': self.n_trials,
            'timeout': self.timeout,
            'n_jobs': self.n_jobs,
            'memory_limit': self.memory_limit,
            'optimization_method': self.optimization_method.value,
            'method_params': self.method_params,
            'enable_pruning': self.enable_pruning,
            'pruner_method': self.pruner_method.value if self.pruner_method != PrunerMethod.NONE else None,
            'pruner_params': self.pruner_params,
            'early_stopping_n_trials': self.early_stopping_n_trials,
            'early_stopping_threshold': self.early_stopping_threshold,
            'scoring_formula': self.scoring_formula.value,
            'custom_weights': self.custom_weights.to_dict(),
            'min_trades': self.min_trades,
            'gc_after_trial': self.gc_after_trial,
            'save_checkpoints': self.save_checkpoints,
            'checkpoint_every': self.checkpoint_every,
            'debug': self.debug
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OptimizationConfig':
        """Crée une configuration à partir d'un dictionnaire"""
        config = cls()
        
        # Paramètres de base
        for param in ['n_trials', 'timeout', 'n_jobs', 'memory_limit',
                     'enable_pruning', 'early_stopping_n_trials', 'early_stopping_threshold',
                     'min_trades', 'gc_after_trial', 'save_checkpoints', 'checkpoint_every', 'debug']:
            if param in data:
                setattr(config, param, data[param])
        
        # Méthode d'optimisation
        if 'optimization_method' in data:
            config.optimization_method = OptimizationMethod(data['optimization_method'])
        
        # Paramètres de méthode
        if 'method_params' in data:
            # Préservation des paramètres spéciaux par défaut
            special_params = {
                k: v for k, v in config.method_params.items()
                if k in ['group_related_params', 'consider_magic_clip']
            }
            config.method_params = data['method_params']
            # Ajout des paramètres spéciaux s'ils ne sont pas présents
            for k, v in special_params.items():
                if k not in config.method_params:
                    config.method_params[k] = v
        
        # Pruner
        if 'pruner_method' in data and data['pruner_method']:
            config.pruner_method = PrunerMethod(data['pruner_method'])
        else:
            config.pruner_method = PrunerMethod.NONE
        
        # Paramètres de pruner
        if 'pruner_params' in data:
            config.pruner_params = data['pruner_params']
        
        # Formule de scoring
        if 'scoring_formula' in data:
            config.scoring_formula = ScoringFormula(data['scoring_formula'])
        
        # Poids personnalisés
        if 'custom_weights' in data:
            config.custom_weights = ScoringWeights.from_dict(data['custom_weights'])
        
        return config
    
    def get_sampler_instance(self) -> optuna.samplers.BaseSampler:
        """Crée une instance du sampler configuré"""
        method_info = OPTIMIZATION_METHODS_CONFIG.get(self.optimization_method.value)
        if not method_info:
            raise ValueError(f"Méthode d'optimisation non reconnue: {self.optimization_method}")
        
        # Paramètres par défaut
        params = {}
        for param_name, param_info in method_info.params.items():
            params[param_name] = param_info['default']
        
        # Fusion avec les paramètres personnalisés
        for param_name, param_value in self.method_params.items():
            if param_name in params:
                params[param_name] = param_value
        
        # Paramètres spéciaux pour TPE
        if self.optimization_method == OptimizationMethod.TPE:
            # Suppression des paramètres spéciaux qui ne sont pas pour le sampler
            params.pop("group_related_params", None)
            params.pop("consider_magic_clip", None)
        
        return method_info.sampler_class(**params)
    
    def get_pruner_instance(self) -> Optional[optuna.pruners.BasePruner]:
        """Crée une instance du pruner configuré si le pruning est activé"""
        if not self.enable_pruning or self.pruner_method == PrunerMethod.NONE:
            return None
        
        pruner_info = PRUNER_METHODS_CONFIG.get(self.pruner_method.value)
        if not pruner_info:
            raise ValueError(f"Méthode de pruning non reconnue: {self.pruner_method}")
        
        # Paramètres par défaut
        params = {}
        for param_name, param_info in pruner_info.params.items():
            params[param_name] = param_info['default']
        
        # Fusion avec les paramètres personnalisés
        for param_name, param_value in self.pruner_params.items():
            if param_name in params:
                params[param_name] = param_value
        
        return pruner_info.pruner_class(**params)

@dataclass
class ScoringFormulaConfig:
    """Configuration d'une formule de scoring"""
    name: str
    description: str
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    transformation: Optional[Callable] = None
    
    def to_dict(self) -> Dict:
        """Convertit la configuration en dictionnaire"""
        return {
            "name": self.name,
            "description": self.description,
            "weights": self.weights.to_dict()
        }

def create_default_optimization_config() -> OptimizationConfig:
    """
    Crée une configuration d'optimisation par défaut
    
    Returns:
        OptimizationConfig: Configuration d'optimisation par défaut
    """
    return OptimizationConfig()

def save_optimization_config_to_file(config: OptimizationConfig, file_path: str) -> bool:
    """
    Sauvegarde une configuration d'optimisation dans un fichier JSON
    
    Args:
        config: Configuration d'optimisation
        file_path: Chemin du fichier
        
    Returns:
        bool: True si la sauvegarde a réussi
    """
    import os
    import json
    
    try:
        # Création du répertoire si nécessaire
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Sérialisation et sauvegarde
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=4, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de la configuration: {str(e)}")
        return False

def load_optimization_config_from_file(file_path: str) -> OptimizationConfig:
    """
    Charge une configuration d'optimisation depuis un fichier JSON
    
    Args:
        file_path: Chemin du fichier
        
    Returns:
        OptimizationConfig: Configuration d'optimisation chargée
    """
    import json
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return OptimizationConfig.from_dict(data)
    except Exception as e:
        print(f"Erreur lors du chargement de la configuration: {str(e)}")
        return create_default_optimization_config()
    

"""
    _________________________________  Metadata  _________________________________
"""


class DataSource(Enum):
    """Sources de données possibles"""
    STUDY = "study"  # Données intégrées à l'étude
    LOCAL = "local"  # Fichier local spécifique
    EXCHANGE = "exchange"  # Téléchargé directement depuis l'exchange
    CUSTOM = "custom"  # Source personnalisée

@dataclass
class OptimizationMetadata:
    """Métadonnées concernant l'optimisation d'une étude"""
    last_optimization: Optional[str] = None  # Date de la dernière optimisation
    trials_count: int = 0
    best_trial_id: Optional[int] = None
    best_score: Optional[float] = None
    optimization_config: Optional[Dict] = None  # Configuration utilisée pour l'optimisation
    optimization_duration: Optional[float] = None  # Durée en secondes
    
    def to_dict(self) -> Dict:
        """Convertit les métadonnées en dictionnaire"""
        return {
            "last_optimization": self.last_optimization,
            "trials_count": self.trials_count,
            "best_trial_id": self.best_trial_id,
            "best_score": self.best_score,
            "optimization_config": self.optimization_config,
            "optimization_duration": self.optimization_duration
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OptimizationMetadata':
        """Crée un objet OptimizationMetadata à partir d'un dictionnaire"""
        metadata = cls()
        
        for field in ["last_optimization", "trials_count", "best_trial_id", 
                     "best_score", "optimization_config", "optimization_duration"]:
            if field in data:
                setattr(metadata, field, data[field])
        
        return metadata

@dataclass
class StrategyMetadata:
    """Métadonnées concernant les stratégies d'une étude"""
    strategies_count: int = 0
    best_strategy_id: Optional[str] = None
    strategies: List[Dict] = field(default_factory=list)  # Liste des métadonnées des stratégies
    
    def to_dict(self) -> Dict:
        """Convertit les métadonnées en dictionnaire"""
        return {
            "strategies_count": self.strategies_count,
            "best_strategy_id": self.best_strategy_id,
            "strategies": self.strategies
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyMetadata':
        """Crée un objet StrategyMetadata à partir d'un dictionnaire"""
        metadata = cls()
        
        if "strategies_count" in data:
            metadata.strategies_count = data["strategies_count"]
        
        if "best_strategy_id" in data:
            metadata.best_strategy_id = data["best_strategy_id"]
        
        if "strategies" in data:
            metadata.strategies = data["strategies"]
        
        return metadata

@dataclass
class BacktestMetadata:
    """Métadonnées concernant les backtests d'une étude"""
    last_backtest: Optional[str] = None  # Date du dernier backtest
    backtests_count: int = 0
    best_backtest_id: Optional[str] = None
    backtests: List[Dict] = field(default_factory=list)  # Liste des métadonnées des backtests
    
    def to_dict(self) -> Dict:
        """Convertit les métadonnées en dictionnaire"""
        return {
            "last_backtest": self.last_backtest,
            "backtests_count": self.backtests_count,
            "best_backtest_id": self.best_backtest_id,
            "backtests": self.backtests
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BacktestMetadata':
        """Crée un objet BacktestMetadata à partir d'un dictionnaire"""
        metadata = cls()
        
        for field in ["last_backtest", "backtests_count", "best_backtest_id"]:
            if field in data:
                setattr(metadata, field, data[field])
        
        if "backtests" in data:
            metadata.backtests = data["backtests"]
        
        return metadata

@dataclass
class PerformanceMetrics:
    """Métriques de performance standard"""
    roi: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    avg_profit: float = 0.0
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convertit les métriques en dictionnaire"""
        result = {}
        for field in ["roi", "win_rate", "total_trades", "max_drawdown", 
                     "profit_factor", "avg_profit", "sharpe_ratio", 
                     "sortino_ratio", "calmar_ratio"]:
            value = getattr(self, field)
            if value is not None:
                result[field] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PerformanceMetrics':
        """Crée un objet PerformanceMetrics à partir d'un dictionnaire"""
        metrics = cls()
        
        for field in ["roi", "win_rate", "total_trades", "max_drawdown", 
                     "profit_factor", "avg_profit", "sharpe_ratio", 
                     "sortino_ratio", "calmar_ratio"]:
            if field in data:
                setattr(metrics, field, data[field])
        
        return metrics


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('config_manager.log', mode='a'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('config_manager')

if __name__ == "__main__":
    # Exemple d'utilisation
    config = create_flexible_default_config()
    
    # Affichage des plages de paramètres
    print("Configuration flexible par défaut:")
    print(f"Modes de risque disponibles: {[mode.value for mode in config.risk_config.available_modes]}")
    print(f"Plage de taille de position: {config.risk_config.position_size_range}")
    print(f"Plage de stop loss: {config.risk_config.sl_range}")
    print(f"Plage de multiplicateur de take profit: {config.risk_config.tp_multiplier_range}")
    print(f"Plage de levier: {config.sim_config.leverage_range}")
    print(f"Modes de marge disponibles: {[mode.value for mode in config.sim_config.margin_modes]}")
    print(f"Modes de trading disponibles: {[mode.value for mode in config.sim_config.trading_modes]}")
    
    # Sauvegarde de la configuration
    save_flexible_config_to_file(config, "configs/flexible_config.json")
    
    # Conversion en configuration de simulateur
    sim_config = convert_to_simulator_config(config.sim_config)
    print(f"\nConfiguration du simulateur:")
    print(f"Balance initiale: {sim_config.initial_balance}")
    print(f"Frais d'ouverture: {sim_config.fee_open}")
    print(f"Levier: {sim_config.leverage}")
    
    # Création de configuration pour PositionCalculator
    for mode in RiskMode:
        pc_config = create_position_calculator_config(config.risk_config, mode)
        print(f"\nConfiguration pour {mode.value}:")
        for key, value in pc_config.items():
            print(f"- {key}: {value}")