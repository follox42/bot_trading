
import os
import ccxt
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable, Tuple

class Exchange(Enum):
    """Enum des exchanges supportés"""
    BITGET = "bitget"
    BINANCE = "binance"
    
    @classmethod
    def from_string(cls, value: str) -> 'Exchange':
        """Convertit une chaîne en valeur d'enum"""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Exchange non supporté: {value}. Supportés: {[e.value for e in cls]}")
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration pour cet exchange"""
        configs = {
            Exchange.BITGET: {
                'class': ccxt.bitget,
                'options': {'defaultType': 'future'}
            },
            Exchange.BINANCE: {
                'class': ccxt.binance,
                'options': {'defaultType': 'future'}
            }
        }
        return configs[self]

class Timeframe(Enum):
    """Enum des timeframes supportés"""
    M1 = "1m"    # 1 minute
    M5 = "5m"    # 5 minutes
    M15 = "15m"  # 15 minutes
    M30 = "30m"  # 30 minutes
    H1 = "1h"    # 1 heure
    H4 = "4h"    # 4 heures
    D1 = "1d"    # 1 jour
    
    @classmethod
    def from_string(cls, value: str) -> 'Timeframe':
        """Convertit une chaîne en valeur d'enum"""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Timeframe non supporté: {value}. Supportés: {[t.value for t in cls]}")
    
    def to_milliseconds(self) -> int:
        """Convertit le timeframe en millisecondes"""
        timeframe_multipliers = {
            Timeframe.M1: 60 * 1000,           # 1 minute
            Timeframe.M5: 5 * 60 * 1000,       # 5 minutes
            Timeframe.M15: 15 * 60 * 1000,     # 15 minutes
            Timeframe.M30: 30 * 60 * 1000,     # 30 minutes
            Timeframe.H1: 60 * 60 * 1000,      # 1 heure
            Timeframe.H4: 4 * 60 * 60 * 1000,  # 4 heures
            Timeframe.D1: 24 * 60 * 60 * 1000  # 1 jour
        }
        return timeframe_multipliers[self]
    
    def to_timedelta(self) -> timedelta:
        """Convertit le timeframe en objet timedelta"""
        timeframe_deltas = {
            Timeframe.M1: timedelta(minutes=1),
            Timeframe.M5: timedelta(minutes=5),
            Timeframe.M15: timedelta(minutes=15),
            Timeframe.M30: timedelta(minutes=30),
            Timeframe.H1: timedelta(hours=1),
            Timeframe.H4: timedelta(hours=4),
            Timeframe.D1: timedelta(days=1)
        }
        return timeframe_deltas[self]
    
    def expected_candles_between(self, start_date: datetime, end_date: datetime) -> int:
        """
        Calcule le nombre de bougies attendues entre deux dates pour ce timeframe
        
        Args:
            start_date (datetime): Date de début
            end_date (datetime): Date de fin
            
        Returns:
            int: Nombre de bougies attendues
        """
        delta = end_date - start_date
        delta_seconds = delta.total_seconds()
        timeframe_seconds = self.to_timedelta().total_seconds()
        
        # Ajouter 1 car nous comptons aussi la bougie à la date de début
        return int(delta_seconds / timeframe_seconds) + 1

class IntegrityCheck(Enum):
    """Options pour la vérification d'intégrité"""
    IGNORE = "ignore"            # Ignore les problèmes d'intégrité
    WARN = "warn"                # Affiche un avertissement
    ERROR = "error"              # Lève une exception
    AUTO_REPAIR = "auto_repair"  # Tente de réparer automatiquement (re-télécharge)

class DataIntegrityError(Exception):
    """Exception levée en cas de problème d'intégrité des données"""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}

@dataclass
class IntegrityCheckResult:
    """Résultat d'une vérification d'intégrité"""
    is_valid: bool = True
    missing_rows: int = 0
    expected_rows: int = 0
    actual_rows: int = 0
    missing_columns: List[str] = field(default_factory=list)
    has_gaps: bool = False
    gap_details: List[Dict[str, Any]] = field(default_factory=list)
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le résultat en dictionnaire"""
        return asdict(self)
    
    @property
    def summary(self) -> str:
        """Résumé des problèmes d'intégrité"""
        if self.is_valid:
            return "Les données sont valides."
        
        issues = []
        if self.missing_rows > 0:
            issues.append(f"Lignes manquantes: {self.missing_rows}/{self.expected_rows} ({self.missing_rows/self.expected_rows*100:.1f}%)")
        
        if self.missing_columns:
            issues.append(f"Colonnes manquantes: {', '.join(self.missing_columns)}")
        
        if self.has_gaps:
            gaps_count = len(self.gap_details)
            issues.append(f"Discontinuités trouvées: {gaps_count}")
        
        return "Problèmes d'intégrité: " + "; ".join(issues)

@dataclass
class ProgressInfo:
    """Informations sur la progression du téléchargement"""
    progress: int = 0
    batch_count: int = 0
    max_batches: int = 0
    elapsed_time: float = 0.0
    remaining_time: Optional[float] = None

@dataclass
class MarketDataConfig:
    """Configuration pour les données de marché"""
    exchange: Exchange = Exchange.BITGET
    symbol: str = "BTC/USDT"
    timeframe: Timeframe = Timeframe.M1
    limit: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    data_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), 'data', 'historical'))
    integrity_check: IntegrityCheck = IntegrityCheck.WARN
    expected_columns: List[str] = field(default_factory=lambda: ["timestamp", "open", "high", "low", "close", "volume"])
    
    def __post_init__(self):
        """Validation et conversion des dates"""
        # Convertir les dates en objets datetime si elles sont présentes
        self._start_datetime = None
        self._end_datetime = None
        
        if self.start_date:
            self._start_datetime = datetime.strptime(self.start_date, "%Y-%m-%d")
        
        if self.end_date:
            self._end_datetime = datetime.strptime(self.end_date, "%Y-%m-%d")
    
    @property
    def start_datetime(self) -> Optional[datetime]:
        """Date de début en objet datetime"""
        return self._start_datetime
    
    @property
    def end_datetime(self) -> Optional[datetime]:
        """Date de fin en objet datetime"""
        return self._end_datetime
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MarketDataConfig':
        """Crée une configuration à partir d'un dictionnaire"""
        # Conversion des valeurs en types appropriés
        if 'exchange' in config_dict and isinstance(config_dict['exchange'], str):
            config_dict['exchange'] = Exchange.from_string(config_dict['exchange'])
        
        if 'timeframe' in config_dict and isinstance(config_dict['timeframe'], str):
            config_dict['timeframe'] = Timeframe.from_string(config_dict['timeframe'])
            
        if 'integrity_check' in config_dict and isinstance(config_dict['integrity_check'], str):
            config_dict['integrity_check'] = IntegrityCheck(config_dict['integrity_check'])
            
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire"""
        config_dict = asdict(self)
        
        # Conversion des enums en chaînes pour la sérialisation
        if 'exchange' in config_dict and isinstance(self.exchange, Exchange):
            config_dict['exchange'] = self.exchange.value
            
        if 'timeframe' in config_dict and isinstance(self.timeframe, Timeframe):
            config_dict['timeframe'] = self.timeframe.value
            
        if 'integrity_check' in config_dict and isinstance(self.integrity_check, IntegrityCheck):
            config_dict['integrity_check'] = self.integrity_check.value
            
        # Retirer les champs privés
        keys_to_remove = [k for k in config_dict.keys() if k.startswith('_')]
        for k in keys_to_remove:
            del config_dict[k]
            
        return config_dict
    
    def get_output_path(self) -> str:
        """Génère le chemin du fichier de sortie basé sur la configuration"""
        filename_parts = [
            self.symbol.replace('/', '_'),
            self.timeframe.value,
            self.exchange.value
        ]
        
        if self.start_date:
            start_date_str = self.start_date.replace('-', '')
            filename_parts.append(start_date_str)
        elif self.limit:
            filename_parts.append(str(self.limit))

        return os.path.join(self.data_dir, f"{'_'.join(filename_parts)}.csv")