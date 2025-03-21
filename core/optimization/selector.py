"""
Module pour l'exploration des espaces de recherche lors de l'optimisation des stratégies.
Permet la sélection des paramètres et la création de stratégies.
"""
import logging
from typing import Dict, List, Any, Optional, Union, Callable
import random
from core.optimization.search_config import SearchSpace, ParameterRange
from core.strategy.constructor.constructor import StrategyConstructor
from core.strategy.indicators.indicators_config import IndicatorType, IndicatorConfig
from core.strategy.conditions.conditions_config import (
    ConditionConfig, BlockConfig, OperatorType, LogicOperatorType,
    PriceOperand, IndicatorOperand, ValueOperand
)
from core.strategy.risk.risk_config import RiskConfig, RiskModeType

logger = logging.getLogger(__name__)

class ParameterSelector:
    """
    Classe pour sélectionner les paramètres à partir d'un espace de recherche
    et les injecter efficacement dans un trial Optuna.
    """
    def __init__(self, search_space: SearchSpace):
        """
        Initialise le sélecteur avec un espace de recherche.
        Args:
            search_space: L'espace de recherche à explorer
        """
        self.search_space = search_space
        self.selected_params = {}
        self.selected_indicators = []
        self.selected_risk_mode = None

    def select_indicators_and_risk(self, max_indicators=2):
        """
        Choisit les indicateurs et le mode de risque à utiliser dans la stratégie.
        Args:
            max_indicators: Nombre maximum d'indicateurs à sélectionner
        """
        # Sélectionner un sous-ensemble d'indicateurs disponibles
        available_indicators = list(self.search_space.selected_indicators.keys())
        if not available_indicators:
            return
        
        # Sélectionner aléatoirement un nombre limité d'indicateurs
        selected_count = min(max_indicators, len(available_indicators))
        self.selected_indicators = random.sample(available_indicators, selected_count)
        
        # Sélectionner un mode de risque
        if "risk_mode" in self.search_space.risk_params:
            risk_param = self.search_space.risk_params["risk_mode"]
            if risk_param.param_type == "categorical" and risk_param.choices:
                self.selected_risk_mode = random.choice(risk_param.choices)
            else:
                self.selected_risk_mode = RiskModeType.FIXED.value
        else:
            self.selected_risk_mode = RiskModeType.FIXED.value
            
        return self.selected_indicators, self.selected_risk_mode

    def suggest_parameter(self, trial, param_name: str, param_range: ParameterRange):
        """
        Suggère une valeur pour un paramètre spécifique à partir de l'espace de recherche.
        Args:
            trial: Trial Optuna
            param_name: Nom du paramètre
            param_range: Configuration de la plage du paramètre
        Returns:
            Any: Valeur suggérée pour le paramètre
        """
        value = None
        if param_range.param_type == "int":
            value = trial.suggest_int(
                param_name,
                param_range.min_value,
                param_range.max_value,
                step=param_range.step if param_range.step else 1,
                log=param_range.log_scale
            )
        elif param_range.param_type == "float":
            value = trial.suggest_float(
                param_name,
                param_range.min_value,
                param_range.max_value,
                step=param_range.step,
                log=param_range.log_scale
            )
        elif param_range.param_type == "categorical":
            value = trial.suggest_categorical(param_name, param_range.choices)
        
        if value is not None:
            self.selected_params[param_name] = value
        
        return value

    def suggest_parameters_for_strategy(self, trial) -> Dict[str, Any]:
        """
        Suggère uniquement les paramètres nécessaires pour les indicateurs et le mode de risque sélectionnés.
        Args:
            trial: Trial Optuna
        Returns:
            Dict[str, Any]: Dictionnaire des paramètres suggérés
        """
        self.selected_params = {}
        
        # Sélectionner les indicateurs et le mode de risque si ce n'est pas déjà fait
        if not self.selected_indicators:
            self.select_indicators_and_risk()
        
        # Suggérer risk_mode si présent dans l'espace de recherche
        if "risk_mode" in self.search_space.risk_params:
            self.suggest_parameter(trial, "risk_mode", self.search_space.risk_params["risk_mode"])
        else:
            # Si risk_mode n'est pas défini dans l'espace de recherche, on l'ajoute directement
            self.selected_params["risk_mode"] = self.selected_risk_mode
        
        # Suggérer les paramètres pour les indicateurs sélectionnés
        for ind_type in self.selected_indicators:
            ind_params = self.search_space.selected_indicators.get(ind_type, {})
            for param_name, param_range in ind_params.items():
                full_name = f"{ind_type}_{param_name}"
                self.suggest_parameter(trial, full_name, param_range)
        
        # Suggérer les paramètres pour le mode de risque sélectionné
        for param_name, param_range in self.search_space.risk_params.items():
            if param_name.startswith(f"{self.selected_risk_mode}_"):
                self.suggest_parameter(trial, param_name, param_range)
        
        return self.selected_params

    def reset(self):
        """Réinitialise l'état du sélecteur"""
        self.selected_params = {}
        self.selected_indicators = []
        self.selected_risk_mode = None


def create_strategy_from_trial(trial, search_space: SearchSpace) -> 'StrategyConstructor':
    """
    Crée une stratégie à partir d'un trial Optuna et un espace de recherche.
    Args:
        trial: Trial Optuna contenant les paramètres
        search_space: Espace de recherche définissant les paramètres à explorer
    Returns:
        StrategyConstructor: Instance du constructeur de stratégie
    """
    selector = ParameterSelector(search_space)
    params = selector.suggest_parameters_for_strategy(trial)
    
    constructor = StrategyConstructor()
    constructor.set_name(f"Strategy_Trial_{trial.number}")
    
    # Initialiser les conteneurs
    indicator_configs = {}
    
    # Ajouter les indicateurs
    for ind_type, ind_params in search_space.selected_indicators.items():
        ind_config_params = {}
        
        # Extraire les paramètres pour cet indicateur
        for param_name in ind_params:
            full_param_name = f"{ind_type}_{param_name}"
            if full_param_name in params:
                ind_config_params[param_name] = params[full_param_name]
        
        if ind_config_params:
            indicator_type = IndicatorType(ind_type)
            indicator_config = IndicatorConfig(indicator_type, **ind_config_params)
            indicator_name = f"{ind_type.lower()}_{len(indicator_configs)}"
            constructor.add_indicator(indicator_name, indicator_config)
            indicator_configs[ind_type] = {
                'name': indicator_name,
                'config': indicator_config
            }
    
    # Créer les blocs d'entrée et de sortie
    entry_blocks = []
    exit_blocks = []
    
    # Créer des blocs à partir des indicateurs disponibles
    if "EMA" in indicator_configs and "SMA" in indicator_configs:
        # Stratégie de croisement EMA/SMA
        ema_info = indicator_configs["EMA"]
        sma_info = indicator_configs["SMA"]
        
        ema_operand = IndicatorOperand(
            indicator_type=IndicatorType.EMA,
            indicator_name=ema_info['name'],
            indicator_params={}
        )
        
        sma_operand = IndicatorOperand(
            indicator_type=IndicatorType.SMA,
            indicator_name=sma_info['name'],
            indicator_params={}
        )
        
        # Condition d'entrée: EMA croise au-dessus de SMA
        entry_condition = ConditionConfig(
            left_operand=ema_operand,
            operator=OperatorType.CROSS_ABOVE,
            right_operand=sma_operand
        )
        
        # Condition de sortie: EMA croise en-dessous de SMA
        exit_condition = ConditionConfig(
            left_operand=ema_operand,
            operator=OperatorType.CROSS_BELOW,
            right_operand=sma_operand
        )
        
        entry_blocks.append(BlockConfig(
            conditions=[entry_condition],
            name="EMA Cross Above SMA"
        ))
        
        exit_blocks.append(BlockConfig(
            conditions=[exit_condition],
            name="EMA Cross Below SMA"
        ))
        
    elif "RSI" in indicator_configs:
        # Stratégie RSI
        rsi_info = indicator_configs["RSI"]
        
        rsi_operand = IndicatorOperand(
            indicator_type=IndicatorType.RSI,
            indicator_name=rsi_info['name'],
            indicator_params={}
        )
        
        # Obtenir les paramètres RSI
        rsi_params = {}
        for param_name, param_value in params.items():
            if param_name.startswith("RSI_"):
                rsi_param = param_name.split("RSI_")[1]
                rsi_params[rsi_param] = param_value
        
        oversold_value = rsi_params.get("oversold", 30.0)
        overbought_value = rsi_params.get("overbought", 70.0)
        
        # Condition d'entrée: RSI sous le niveau de survente
        entry_condition = ConditionConfig(
            left_operand=rsi_operand,
            operator=OperatorType.LESS,
            right_operand=ValueOperand(value=oversold_value)
        )
        
        # Condition de sortie: RSI au-dessus du niveau de surachat
        exit_condition = ConditionConfig(
            left_operand=rsi_operand,
            operator=OperatorType.GREATER,
            right_operand=ValueOperand(value=overbought_value)
        )
        
        entry_blocks.append(BlockConfig(
            conditions=[entry_condition],
            name="RSI Oversold"
        ))
        
        exit_blocks.append(BlockConfig(
            conditions=[exit_condition],
            name="RSI Overbought"
        ))
        
    elif "MACD" in indicator_configs:
        # Stratégie MACD
        macd_info = indicator_configs["MACD"]
        
        macd_line = IndicatorOperand(
            indicator_type=IndicatorType.MACD,
            indicator_name=macd_info['name'],
            indicator_params={"field": "line"}
        )
        
        macd_signal = IndicatorOperand(
            indicator_type=IndicatorType.MACD,
            indicator_name=macd_info['name'],
            indicator_params={"field": "signal"}
        )
        
        # Condition d'entrée: Ligne MACD croise au-dessus de la ligne de signal
        entry_condition = ConditionConfig(
            left_operand=macd_line,
            operator=OperatorType.CROSS_ABOVE,
            right_operand=macd_signal
        )
        
        # Condition de sortie: Ligne MACD croise en-dessous de la ligne de signal
        exit_condition = ConditionConfig(
            left_operand=macd_line,
            operator=OperatorType.CROSS_BELOW,
            right_operand=macd_signal
        )
        
        entry_blocks.append(BlockConfig(
            conditions=[entry_condition],
            name="MACD Cross Above Signal"
        ))
        
        exit_blocks.append(BlockConfig(
            conditions=[exit_condition],
            name="MACD Cross Below Signal"
        ))
    
    # Si aucune condition d'entrée/sortie spécifique, créer des conditions par défaut
    if not entry_blocks:
        price_operand = PriceOperand(price_type="close")
        entry_blocks.append(BlockConfig(
            conditions=[ConditionConfig(
                left_operand=price_operand,
                operator=OperatorType.GREATER,
                right_operand=ValueOperand(value=0.0)
            )],
            name="Default Entry"
        ))
    
    if not exit_blocks:
        price_operand = PriceOperand(price_type="close")
        exit_blocks.append(BlockConfig(
            conditions=[ConditionConfig(
                left_operand=price_operand,
                operator=OperatorType.LESS,
                right_operand=ValueOperand(value=0.0)
            )],
            name="Default Exit"
        ))
    
    # Ajouter les blocs au constructeur
    for block in entry_blocks:
        constructor.add_entry_block(block)
    
    for block in exit_blocks:
        constructor.add_exit_block(block)
    
    # Configurer la gestion du risque
    risk_mode = params.get("risk_mode", RiskModeType.FIXED.value)
    risk_params = {}
    
    # Extraire les paramètres de risque en fonction du mode
    for param_name, param_value in params.items():
        if param_name.startswith(f"{risk_mode}_"):
            risk_param = param_name.split(f"{risk_mode}_")[1]
            risk_params[risk_param] = param_value
    
    # S'assurer que nous avons les paramètres de base pour le mode sélectionné
    if risk_mode == RiskModeType.FIXED.value and not risk_params:
        # Valeurs par défaut pour le mode fixe
        risk_params = {
            "position_size": 0.1,
            "stop_loss": 0.02,
            "take_profit": 0.04
        }
    elif risk_mode == RiskModeType.ATR_BASED.value and not risk_params:
        # Valeurs par défaut pour le mode ATR
        risk_params = {
            "atr_period": 14,
            "atr_multiplier": 1.5,
            "risk_per_trade": 0.01,
            "tp_multiplier": 2.0
        }
    
    risk_config = RiskConfig(mode=risk_mode, **risk_params)
    constructor.set_risk_config(risk_config)
    
    # Stocker tous les paramètres dans la stratégie
    for key, value in params.items():
        constructor.set_parameter(key, value)
    
    # Stocker l'ID de la stratégie dans les attributs du trial
    trial.set_user_attr("strategy_id", constructor.config.id)
    
    return constructor