"""
Module de configuration centralisée pour le système de trading.
Version avec paramètres flexibles pour permettre une meilleure adaptabilité.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Union, Tuple
import logging
import json
import os
from enum import Enum, auto

# Importation des modules du système de trading
from indicators import IndicatorType, Operator, LogicOperator
from risk import RiskMode
from simulator import SimulationConfig

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
class FlexibleTradingConfig:
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
    def from_dict(cls, data: Dict) -> 'FlexibleTradingConfig':
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

def create_flexible_default_config() -> FlexibleTradingConfig:
    """Crée une configuration flexible par défaut avec les indicateurs standard"""
    config = FlexibleTradingConfig()
    
    # Ajout des indicateurs standard
    for ind_type in IndicatorType:
        if ind_type == IndicatorType.EMA:
            config.add_indicator(ind_type.value, IndicatorConfig(
                type=ind_type,
                min_period=5,
                max_period=200,
                step=5
            ))
        elif ind_type == IndicatorType.SMA:
            config.add_indicator(ind_type.value, IndicatorConfig(
                type=ind_type,
                min_period=5,
                max_period=200,
                step=5
            ))
        elif ind_type == IndicatorType.RSI:
            config.add_indicator(ind_type.value, IndicatorConfig(
                type=ind_type,
                min_period=7,
                max_period=30,
                step=1
            ))
        elif ind_type == IndicatorType.ATR:
            config.add_indicator(ind_type.value, IndicatorConfig(
                type=ind_type,
                min_period=7,
                max_period=30,
                step=1
            ))
        elif ind_type == IndicatorType.MACD:
            config.add_indicator(ind_type.value, IndicatorConfig(
                type=ind_type,
                min_period=12,  # Fast period
                max_period=26,  # Slow period
                step=1
            ))
        elif ind_type == IndicatorType.BOLL:
            config.add_indicator(ind_type.value, IndicatorConfig(
                type=ind_type,
                min_period=20,
                max_period=20,
                step=1
            ))
        elif ind_type == IndicatorType.STOCH:
            config.add_indicator(ind_type.value, IndicatorConfig(
                type=ind_type,
                min_period=14,
                max_period=14,
                step=1
            ))
        elif ind_type == IndicatorType.VWAP:
            config.add_indicator(ind_type.value, IndicatorConfig(
                type=ind_type,
                min_period=1,
                max_period=1,
                step=1
            ))
        elif ind_type == IndicatorType.MFIMACD:
            config.add_indicator(ind_type.value, IndicatorConfig(
                type=ind_type,
                min_period=12,  # Fast period
                max_period=26,  # Slow period
                step=1
            ))
    
    # Configuration spécifique pour chaque mode de risque
    config.risk_config.mode_configs = {
        RiskMode.FIXED: RiskModeConfig(
            fixed_position_range=(0.01, 1.0),
            fixed_sl_range=(0.001, 0.1),
            fixed_tp_range=(0.002, 0.5)
        ),
        RiskMode.ATR_BASED: RiskModeConfig(
            atr_period_range=(5, 30),
            atr_multiplier_range=(0.5, 5.0)
        ),
        RiskMode.VOLATILITY_BASED: RiskModeConfig(
            vol_period_range=(10, 50),
            vol_multiplier_range=(0.5, 5.0)
        )
    }
    
    # Configuration de simulation avec plage de levier étendue
    config.sim_config = SimConfig(
        initial_balance_range=(1000.0, 100000.0),
        fee=0.01,
        slippage=0.01,
        tick_size=0.001,
        leverage_range=(1, 125),  # Levier jusqu'à 125
        margin_modes=[MarginMode.ISOLATED, MarginMode.CROSS],  # Les deux modes de marge
        trading_modes=[TradingMode.ONE_WAY, TradingMode.HEDGE]  # Les deux modes de trading
    )
    
    return config

def load_flexible_config_from_file(file_path: str) -> FlexibleTradingConfig:
    """Charge une configuration flexible depuis un fichier JSON"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return FlexibleTradingConfig.from_dict(data)
    except Exception as e:
        logging.error(f"Erreur lors du chargement de la configuration: {str(e)}")
        return create_flexible_default_config()

def save_flexible_config_to_file(config: FlexibleTradingConfig, file_path: str) -> bool:
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