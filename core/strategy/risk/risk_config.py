"""
Configuration des modes de risque pour les stratégies de trading.
Définit précisément chaque type de risque avec ses paramètres spécifiques.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Union, Tuple


class RiskModeType(Enum):
    """Types de modes de gestion du risque"""
    FIXED = "fixed"                   # Taille de position fixe
    ATR_BASED = "atr_based"           # Basé sur l'ATR
    VOLATILITY_BASED = "vol_based"    # Basé sur la volatilité
    EQUITY_PERCENT = "equity_percent" # Pourcentage de l'equity
    KELLEY = "kelley"                 # Critère de Kelly


@dataclass
class FixedRiskParams:
    """Paramètres pour le mode de risque fixe"""
    position_size: float = 0.1      # 10% du capital
    stop_loss: float = 0.02         # 2% de stop loss
    take_profit: float = 0.04       # 4% de take profit (stop_loss * 2)
    trail_stop_enabled: bool = False
    trail_stop_activation: float = 0.01  # 1% de profit pour activer le trailing stop


@dataclass
class AtrRiskParams:
    """Paramètres pour le mode de risque basé sur l'ATR"""
    atr_period: int = 14            # Période pour le calcul de l'ATR
    atr_multiplier: float = 1.5     # Multiplicateur appliqué à l'ATR
    risk_per_trade: float = 0.01    # 1% du capital risqué par trade
    tp_multiplier: float = 2.0      # TP = 2 fois le SL
    max_position_size: float = 0.2  # 20% du capital max par position
    min_position_size: float = 0.01 # 1% du capital min par position


@dataclass
class VolatilityRiskParams:
    """Paramètres pour le mode de risque basé sur la volatilité"""
    vol_period: int = 20            # Période pour le calcul de la volatilité
    vol_multiplier: float = 1.0     # Multiplicateur appliqué à la volatilité
    risk_per_trade: float = 0.01    # 1% du capital risqué par trade
    tp_multiplier: float = 2.0      # TP = 2 fois le SL
    max_position_size: float = 0.2  # 20% du capital max par position
    min_position_size: float = 0.01 # 1% du capital min par position


@dataclass
class EquityPercentRiskParams:
    """Paramètres pour le mode de risque basé sur un pourcentage du capital"""
    risk_percent: float = 0.01      # 1% du capital risqué par trade
    max_position_size: float = 0.2  # 20% du capital max par position
    tp_multiplier: float = 2.0      # TP = 2 fois le SL
    min_stop_distance: float = 0.005  # 0.5% distance min du stop loss


@dataclass
class KelleyRiskParams:
    """Paramètres pour le mode de risque basé sur le critère de Kelly"""
    fraction: float = 0.5           # Fraction du critère de Kelly (0.5 = Half Kelly)
    window_size: int = 50           # Fenêtre pour le calcul des probabilités
    max_position_size: float = 0.25 # 25% du capital max par position
    min_position_size: float = 0.01 # 1% du capital min par position
    min_win_rate: float = 0.40      # Win rate minimum requis pour utiliser Kelly


class RiskConfig:
    """Configuration du risque pour une stratégie de trading"""
    
    def __init__(self, mode: Union[str, RiskModeType] = RiskModeType.FIXED, **kwargs):
        """
        Initialise la configuration du risque.
        
        Args:
            mode: Mode de risque à utiliser
            **kwargs: Paramètres spécifiques au mode de risque
        """
        # Convertir en RiskModeType si nécessaire
        if isinstance(mode, str):
            mode = RiskModeType(mode)
            
        self.mode = mode
        
        # Initialiser les paramètres selon le mode
        if mode == RiskModeType.FIXED:
            self.params = FixedRiskParams(**kwargs)
        elif mode == RiskModeType.ATR_BASED:
            self.params = AtrRiskParams(**kwargs)
        elif mode == RiskModeType.VOLATILITY_BASED:
            self.params = VolatilityRiskParams(**kwargs)
        elif mode == RiskModeType.EQUITY_PERCENT:
            self.params = EquityPercentRiskParams(**kwargs)
        elif mode == RiskModeType.KELLEY:
            self.params = KelleyRiskParams(**kwargs)
        else:
            self.params = FixedRiskParams()
    
    def update_params(self, **kwargs):
        """Met à jour les paramètres spécifiques au mode actuel"""
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
    
    def to_dict(self) -> Dict:
        """Convertit la configuration en dictionnaire"""
        return {
            "mode": self.mode.value,
            "params": self.params.__dict__
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RiskConfig':
        """Crée une configuration à partir d'un dictionnaire"""
        mode = RiskModeType(data.get("mode", RiskModeType.FIXED.value))
        params = data.get("params", {})
        
        return cls(mode=mode, **params)
    
    def get_param_description(self) -> Dict:
        """
        Retourne la description des paramètres pour le mode actuel
        
        Returns:
            Dict: Description des paramètres (min, max, description, etc.)
        """
        if self.mode == RiskModeType.FIXED:
            return {
                "position_size": {
                    "type": "float",
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "description": "Pourcentage du capital à utiliser pour chaque position"
                },
                "stop_loss": {
                    "type": "float",
                    "min": 0.005,
                    "max": 0.05,
                    "step": 0.001,
                    "description": "Pourcentage de perte maximale acceptée par trade"
                },
                "take_profit": {
                    "type": "float",
                    "min": 0.01,
                    "max": 0.15,
                    "step": 0.005,
                    "description": "Pourcentage de gain cible par trade"
                },
                "trail_stop_enabled": {
                    "type": "bool",
                    "description": "Activer le trailing stop"
                },
                "trail_stop_activation": {
                    "type": "float",
                    "min": 0.005,
                    "max": 0.05,
                    "step": 0.001,
                    "description": "Profit nécessaire pour activer le trailing stop"
                }
            }
        elif self.mode == RiskModeType.ATR_BASED:
            return {
                "atr_period": {
                    "type": "int",
                    "min": 5,
                    "max": 30,
                    "step": 1,
                    "description": "Période pour le calcul de l'ATR"
                },
                "atr_multiplier": {
                    "type": "float",
                    "min": 0.5,
                    "max": 4.0,
                    "step": 0.1,
                    "description": "Multiplicateur appliqué à l'ATR pour le stop loss"
                },
                "risk_per_trade": {
                    "type": "float",
                    "min": 0.005,
                    "max": 0.03,
                    "step": 0.001,
                    "description": "Pourcentage du capital risqué par trade"
                },
                "tp_multiplier": {
                    "type": "float",
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.1,
                    "description": "Multiplicateur du stop loss pour le take profit"
                },
                "max_position_size": {
                    "type": "float",
                    "min": 0.05,
                    "max": 0.5,
                    "step": 0.05,
                    "description": "Taille maximale de position en % du capital"
                },
                "min_position_size": {
                    "type": "float",
                    "min": 0.01,
                    "max": 0.1,
                    "step": 0.01,
                    "description": "Taille minimale de position en % du capital"
                }
            }
        elif self.mode == RiskModeType.VOLATILITY_BASED:
            return {
                "vol_period": {
                    "type": "int",
                    "min": 10,
                    "max": 50,
                    "step": 1,
                    "description": "Période pour le calcul de la volatilité"
                },
                "vol_multiplier": {
                    "type": "float",
                    "min": 0.5,
                    "max": 3.0,
                    "step": 0.1,
                    "description": "Multiplicateur appliqué à la volatilité pour le stop loss"
                },
                "risk_per_trade": {
                    "type": "float",
                    "min": 0.005,
                    "max": 0.03,
                    "step": 0.001,
                    "description": "Pourcentage du capital risqué par trade"
                },
                "tp_multiplier": {
                    "type": "float",
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.1,
                    "description": "Multiplicateur du stop loss pour le take profit"
                },
                "max_position_size": {
                    "type": "float",
                    "min": 0.05,
                    "max": 0.5,
                    "step": 0.05,
                    "description": "Taille maximale de position en % du capital"
                },
                "min_position_size": {
                    "type": "float",
                    "min": 0.01,
                    "max": 0.1,
                    "step": 0.01,
                    "description": "Taille minimale de position en % du capital"
                }
            }
        elif self.mode == RiskModeType.EQUITY_PERCENT:
            return {
                "risk_percent": {
                    "type": "float",
                    "min": 0.005,
                    "max": 0.03,
                    "step": 0.001,
                    "description": "Pourcentage du capital risqué par trade"
                },
                "max_position_size": {
                    "type": "float",
                    "min": 0.05,
                    "max": 0.5,
                    "step": 0.05,
                    "description": "Taille maximale de position en % du capital"
                },
                "tp_multiplier": {
                    "type": "float",
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.1,
                    "description": "Multiplicateur du stop loss pour le take profit"
                },
                "min_stop_distance": {
                    "type": "float",
                    "min": 0.001,
                    "max": 0.02,
                    "step": 0.001,
                    "description": "Distance minimale du stop loss en % du prix"
                }
            }
        elif self.mode == RiskModeType.KELLEY:
            return {
                "fraction": {
                    "type": "float",
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "description": "Fraction du critère de Kelly à utiliser (0.5 = Half Kelly)"
                },
                "window_size": {
                    "type": "int",
                    "min": 20,
                    "max": 100,
                    "step": 5,
                    "description": "Nombre de trades précédents à considérer pour le calcul"
                },
                "max_position_size": {
                    "type": "float",
                    "min": 0.05,
                    "max": 0.5,
                    "step": 0.05,
                    "description": "Taille maximale de position en % du capital"
                },
                "min_position_size": {
                    "type": "float",
                    "min": 0.01,
                    "max": 0.1,
                    "step": 0.01,
                    "description": "Taille minimale de position en % du capital"
                },
                "min_win_rate": {
                    "type": "float",
                    "min": 0.3,
                    "max": 0.6,
                    "step": 0.01,
                    "description": "Taux de réussite minimum pour appliquer Kelly"
                }
            }
        else:
            return {}


# Configurations prédéfinies pour différents profils de risque
RISK_PRESETS = {
    RiskModeType.FIXED: {
        "conservative": FixedRiskParams(position_size=0.05, stop_loss=0.01, take_profit=0.02),
        "moderate": FixedRiskParams(position_size=0.1, stop_loss=0.02, take_profit=0.04),
        "aggressive": FixedRiskParams(position_size=0.2, stop_loss=0.03, take_profit=0.09)
    },
    RiskModeType.ATR_BASED: {
        "conservative": AtrRiskParams(atr_period=14, atr_multiplier=1.0, risk_per_trade=0.005),
        "moderate": AtrRiskParams(atr_period=14, atr_multiplier=1.5, risk_per_trade=0.01),
        "aggressive": AtrRiskParams(atr_period=14, atr_multiplier=2.0, risk_per_trade=0.02)
    },
    RiskModeType.VOLATILITY_BASED: {
        "conservative": VolatilityRiskParams(vol_period=20, vol_multiplier=0.8, risk_per_trade=0.005),
        "moderate": VolatilityRiskParams(vol_period=20, vol_multiplier=1.0, risk_per_trade=0.01),
        "aggressive": VolatilityRiskParams(vol_period=20, vol_multiplier=1.5, risk_per_trade=0.02)
    },
    RiskModeType.EQUITY_PERCENT: {
        "conservative": EquityPercentRiskParams(risk_percent=0.005, max_position_size=0.1, tp_multiplier=1.5),
        "moderate": EquityPercentRiskParams(risk_percent=0.01, max_position_size=0.2, tp_multiplier=2.0),
        "aggressive": EquityPercentRiskParams(risk_percent=0.02, max_position_size=0.3, tp_multiplier=3.0)
    },
    RiskModeType.KELLEY: {
        "conservative": KelleyRiskParams(fraction=0.3, max_position_size=0.15, min_win_rate=0.45),
        "moderate": KelleyRiskParams(fraction=0.5, max_position_size=0.25, min_win_rate=0.40),
        "aggressive": KelleyRiskParams(fraction=0.7, max_position_size=0.35, min_win_rate=0.35)
    }
}


def create_risk_config(mode: Union[str, RiskModeType], profile: str = "moderate") -> RiskConfig:
    """
    Crée une configuration de risque selon un profil prédéfini.
    
    Args:
        mode: Mode de risque (RiskModeType ou string)
        profile: Profil de risque ('conservative', 'moderate', 'aggressive')
        
    Returns:
        RiskConfig: Configuration de risque
    """
    # Convertir en RiskModeType si nécessaire
    if isinstance(mode, str):
        mode = RiskModeType(mode)
        
    # Vérifier si le profil existe pour ce mode
    if mode in RISK_PRESETS and profile in RISK_PRESETS[mode]:
        params = RISK_PRESETS[mode][profile]
        return RiskConfig(mode=mode, **params.__dict__)
    
    # Utiliser la configuration par défaut si le profil n'existe pas
    return RiskConfig(mode=mode)