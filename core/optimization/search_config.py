"""
Configuration modulaire pour l'optimisation des stratégies de trading.
S'adapte automatiquement aux indicateurs et modes de risque disponibles dans le système.
"""
import json
import inspect
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union, Any, Type, Set
from dataclasses import dataclass, field, is_dataclass, asdict

# Import des définitions des indicateurs et des risques
from core.strategy.indicators.indicators_config import (
    IndicatorType, IndicatorConfig, 
    EMAParams, SMAParams, RSIParams, MACDParams, ATRParams, 
    BOLLParams, STOCHParams
)

from core.strategy.risk.risk_config import (
    RiskModeType, RiskConfig,
    FixedRiskParams, AtrRiskParams, VolatilityRiskParams,
    EquityPercentRiskParams, KelleyRiskParams
)


@dataclass
class ParameterRange:
    """
    Définition d'une plage de valeurs pour un paramètre d'optimisation.
    """
    name: str
    param_type: str  # "int", "float", "categorical"
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    step: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    default_value: Optional[Any] = None
    
    def __post_init__(self):
        """Validation après initialisation"""
        if self.param_type in ["int", "float"] and (self.min_value is None or self.max_value is None):
            raise ValueError(f"Les paramètres min et max sont obligatoires pour le type {self.param_type}")
        if self.param_type == "categorical" and not self.choices:
            raise ValueError("La liste des choix est obligatoire pour le type categorical")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire"""
        result = {
            "name": self.name,
            "param_type": self.param_type
        }
        
        if self.param_type in ["int", "float"]:
            result["min_value"] = self.min_value
            result["max_value"] = self.max_value
            if self.step is not None:
                result["step"] = self.step
            if self.log_scale:
                result["log_scale"] = self.log_scale
                
        if self.param_type == "categorical":
            result["choices"] = self.choices
            
        if self.default_value is not None:
            result["default_value"] = self.default_value
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterRange':
        """Crée une configuration à partir d'un dictionnaire"""
        return cls(
            name=data.get("name", ""),
            param_type=data.get("param_type", "int"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            step=data.get("step"),
            choices=data.get("choices"),
            log_scale=data.get("log_scale", False),
            default_value=data.get("default_value")
        )


class SearchSpaceError(Exception):
    """Exception spécifique aux erreurs d'espace de recherche"""
    pass


@dataclass
class SearchSpace:
    """
    Espace de recherche pour l'optimisation des stratégies de trading.
    S'adapte automatiquement aux indicateurs et modes de risque disponibles.
    """
    name: str
    description: str = ""
    selected_indicators: Dict[str, Dict[str, ParameterRange]] = field(default_factory=dict)
    risk_params: Dict[str, ParameterRange] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialisation après la création"""
        # S'assurer que les dictionnaires sont initialisés
        if not self.selected_indicators:
            self.selected_indicators = {}
        if not self.risk_params:
            self.risk_params = {}
    
    def get_required_params(self, indicator_type: Union[str, IndicatorType]) -> Dict[str, Dict[str, Any]]:
        """
        Récupère les paramètres requis pour un type d'indicateur donné.
        
        Args:
            indicator_type: Type d'indicateur (string ou IndicatorType)
            
        Returns:
            Dict[str, Dict[str, Any]]: Information sur les paramètres requis
        """
        # Convertir en objet IndicatorType si nécessaire
        if isinstance(indicator_type, str):
            try:
                indicator_type = IndicatorType(indicator_type)
            except ValueError:
                raise ValueError(f"Type d'indicateur inconnu: {indicator_type}")
        
        # Obtenir la classe de paramètres pour ce type d'indicateur
        param_class = get_param_class_for_indicator(indicator_type)
        if not param_class:
            return {}
        
        # Extraire les informations sur les paramètres
        return get_indicator_param_info(param_class)
    
    def add_indicator(self, indicator_type: Union[str, IndicatorType], 
                      params: Optional[Dict[str, ParameterRange]] = None) -> None:
        """
        Ajoute un indicateur à l'espace de recherche avec ses paramètres.
        Si les paramètres ne sont pas fournis, utilise les paramètres par défaut.
        Vérifie que tous les paramètres requis sont présents.
        
        Args:
            indicator_type: Type d'indicateur à ajouter (string ou IndicatorType)
            params: Paramètres personnalisés (optionnel)
            
        Raises:
            SearchSpaceError: Si des paramètres requis sont manquants
        """
        # Convertir en objet IndicatorType si nécessaire
        if isinstance(indicator_type, str):
            try:
                indicator_type = IndicatorType(indicator_type)
            except ValueError:
                raise ValueError(f"Type d'indicateur inconnu: {indicator_type}")
        
        # Normaliser le nom de l'indicateur
        indicator_name = indicator_type.value
        
        # Obtenir les paramètres requis pour ce type d'indicateur
        required_params_info = self.get_required_params(indicator_type)
        required_param_names = list(required_params_info.keys())
        
        # Si des paramètres sont fournis, vérifier qu'ils contiennent tous les paramètres requis
        if params is not None:
            missing_params = [param for param in required_param_names if param not in params]
            if missing_params:
                raise SearchSpaceError(
                    f"Paramètres manquants pour l'indicateur {indicator_name}: {', '.join(missing_params)}\n"
                    f"Paramètres requis: {', '.join(required_param_names)}"
                )
            
            # Ajouter l'indicateur avec les paramètres fournis
            self.selected_indicators[indicator_name] = params
        else:
            # Utiliser les paramètres par défaut
            default_params = get_indicator_default_params(indicator_type)
            self.selected_indicators[indicator_name] = default_params
    
    def remove_indicator(self, indicator_type: Union[str, IndicatorType]) -> None:
        """
        Supprime un indicateur de l'espace de recherche.
        
        Args:
            indicator_type: Type d'indicateur à supprimer (string ou IndicatorType)
        """
        # Convertir en objet IndicatorType si nécessaire
        if isinstance(indicator_type, str):
            try:
                indicator_type = IndicatorType(indicator_type)
            except ValueError:
                raise ValueError(f"Type d'indicateur inconnu: {indicator_type}")
        
        # Normaliser le nom de l'indicateur
        indicator_name = indicator_type.value
        
        # Supprimer l'indicateur s'il existe
        if indicator_name in self.selected_indicators:
            del self.selected_indicators[indicator_name]
    
    def get_required_risk_params(self, risk_mode: Union[str, RiskModeType]) -> Dict[str, Dict[str, Any]]:
        """
        Récupère les paramètres requis pour un mode de gestion du risque donné.
        
        Args:
            risk_mode: Mode de gestion du risque (string ou RiskModeType)
            
        Returns:
            Dict[str, Dict[str, Any]]: Information sur les paramètres requis
        """
        # Convertir en objet RiskModeType si nécessaire
        if isinstance(risk_mode, str):
            try:
                risk_mode = RiskModeType(risk_mode)
            except ValueError:
                raise ValueError(f"Mode de risque inconnu: {risk_mode}")
        
        # Obtenir la classe de paramètres pour ce mode de risque
        param_class = get_param_class_for_risk(risk_mode)
        if not param_class:
            return {}
        
        # Extraire les informations sur les paramètres
        return get_risk_param_info(param_class)
    
    def configure_risk(self, risk_mode: Union[str, RiskModeType] = None, 
                       params: Optional[Dict[str, ParameterRange]] = None) -> None:
        """
        Configure les paramètres de gestion du risque.
        
        Args:
            risk_mode: Mode de gestion du risque à configurer (si None, configure tous les modes)
            params: Paramètres personnalisés (optionnel)
            
        Raises:
            SearchSpaceError: Si des paramètres requis sont manquants
        """
        # Si aucun mode spécifié, ajouter les paramètres généraux de risk_mode
        if risk_mode is None:
            # Ajouter le paramètre de sélection du mode de risque
            risk_modes = [mode.value for mode in RiskModeType]
            self.risk_params["risk_mode"] = ParameterRange(
                name="risk_mode",
                param_type="categorical",
                choices=risk_modes,
                default_value=RiskModeType.FIXED.value
            )
            
            # Ajouter les paramètres pour tous les modes de risque
            for mode in RiskModeType:
                self._add_risk_mode_params(mode, params)
            
            return
        
        # Convertir en objet RiskModeType si nécessaire
        if isinstance(risk_mode, str):
            try:
                risk_mode = RiskModeType(risk_mode)
            except ValueError:
                raise ValueError(f"Mode de risque inconnu: {risk_mode}")
        
        # Ajouter les paramètres spécifiques à ce mode de risque
        self._add_risk_mode_params(risk_mode, params)
    
    def _add_risk_mode_params(self, risk_mode: RiskModeType, 
                              params: Optional[Dict[str, ParameterRange]] = None) -> None:
        """
        Ajoute les paramètres pour un mode de gestion du risque spécifique.
        
        Args:
            risk_mode: Mode de gestion du risque
            params: Paramètres personnalisés (optionnel)
            
        Raises:
            SearchSpaceError: Si des paramètres requis sont manquants
        """
        # Obtenir les paramètres requis pour ce mode de risque
        required_params_info = self.get_required_risk_params(risk_mode)
        required_param_names = list(required_params_info.keys())
        
        # Mode de risque sous forme de chaîne
        mode_name = risk_mode.value
        
        # Si des paramètres sont fournis, vérifier qu'ils contiennent tous les paramètres requis
        if params is not None:
            missing_params = [param for param in required_param_names if param not in params]
            if missing_params:
                raise SearchSpaceError(
                    f"Paramètres manquants pour le mode de risque {mode_name}: {', '.join(missing_params)}\n"
                    f"Paramètres requis: {', '.join(required_param_names)}"
                )
            
            # Ajouter les paramètres avec préfixe
            for param_name, param_range in params.items():
                prefixed_name = f"{mode_name}_{param_name}"
                param_range.name = prefixed_name  # Mettre à jour le nom
                self.risk_params[prefixed_name] = param_range
        else:
            # Obtenir les paramètres par défaut pour ce mode de risque
            risk_params = get_risk_default_params(risk_mode)
            
            # Ajouter les paramètres
            for param_name, param_range in risk_params.items():
                self.risk_params[param_name] = param_range
    
    def get_selected_indicators(self) -> List[IndicatorType]:
        """
        Récupère la liste des indicateurs sélectionnés.
        
        Returns:
            List[IndicatorType]: Liste des types d'indicateurs sélectionnés
        """
        return [IndicatorType(name) for name in self.selected_indicators.keys()]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit l'espace de recherche en dictionnaire.
        
        Returns:
            Dict[str, Any]: Représentation dictionnaire de l'espace de recherche
        """
        return {
            "name": self.name,
            "description": self.description,
            "selected_indicators": {
                ind_name: {param_name: param.to_dict() for param_name, param in params.items()}
                for ind_name, params in self.selected_indicators.items()
            },
            "risk_params": {
                param_name: param.to_dict() for param_name, param in self.risk_params.items()
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convertit l'espace de recherche en JSON.
        
        Args:
            indent: Niveau d'indentation pour le JSON
            
        Returns:
            str: Représentation JSON de l'espace de recherche
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchSpace':
        """
        Crée un espace de recherche à partir d'un dictionnaire.
        
        Args:
            data: Dictionnaire de configuration
            
        Returns:
            SearchSpace: Espace de recherche
        """
        space = cls(
            name=data.get("name", "Custom Space"),
            description=data.get("description", "")
        )
        
        # Charger les indicateurs sélectionnés
        for ind_name, params_dict in data.get("selected_indicators", {}).items():
            space.selected_indicators[ind_name] = {}
            for param_name, param_dict in params_dict.items():
                space.selected_indicators[ind_name][param_name] = ParameterRange.from_dict(param_dict)
        
        # Charger les paramètres de risque
        for param_name, param_dict in data.get("risk_params", {}).items():
            space.risk_params[param_name] = ParameterRange.from_dict(param_dict)
        
        return space
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SearchSpace':
        """
        Crée un espace de recherche à partir d'une chaîne JSON.
        
        Args:
            json_str: Chaîne JSON
            
        Returns:
            SearchSpace: Espace de recherche
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def list_all_parameters(self) -> Dict[str, Dict[str, ParameterRange]]:
        """
        Liste tous les paramètres configurés dans l'espace de recherche.
        Utile pour le sélecteur qui doit parcourir tous les paramètres.
        
        Returns:
            Dict[str, Dict[str, ParameterRange]]: Tous les paramètres par catégorie
        """
        return {
            "indicators": self.selected_indicators,
            "risk": self.risk_params
        }


def get_indicator_param_info(param_class: Type) -> Dict[str, Dict[str, Any]]:
    """
    Extrait des informations sur les paramètres d'une classe de paramètres d'indicateur.
    Utilise l'introspection pour découvrir les types et valeurs par défaut.
    
    Args:
        param_class: Classe de paramètres (e.g., EMAParams, RSIParams)
        
    Returns:
        Dict[str, Dict[str, Any]]: Informations sur les paramètres
    """
    param_info = {}
    
    # Si ce n'est pas une dataclass, on ne peut pas extraire facilement les paramètres
    if not is_dataclass(param_class):
        return param_info
    
    # Extraire les annotations de type et les valeurs par défaut
    for field_name, field_type in param_class.__annotations__.items():
        if field_name.startswith('_'):  # Ignorer les champs privés
            continue
        
        field_default = None
        for field_obj in param_class.__dataclass_fields__.values():
            if field_obj.name == field_name and field_obj.default is not field_obj.default_factory:
                field_default = field_obj.default
        
        # Déterminer le type de paramètre
        param_type = None
        min_val = None
        max_val = None
        step = None
        choices = None
        
        if field_type == int:
            param_type = "int"
            min_val = 1
            max_val = 100
            step = 1
        elif field_type == float:
            param_type = "float"
            min_val = 0.0
            max_val = 10.0
            step = 0.1
        elif field_type == str:
            param_type = "categorical"
            # Pour les chaînes, essayer de déterminer les choix possibles
            if field_name == 'source':
                choices = ["close", "open", "high", "low", "hl2", "hlc3", "ohlc4"]
            else:
                choices = [field_default] if field_default else [""]
        elif field_type == bool:
            param_type = "categorical"
            choices = [True, False]
        
        # Ajustements spécifiques au nom du paramètre
        if field_name == 'period':
            min_val = 2
            max_val = 100
        elif field_name == 'overbought':
            min_val = 50.0
            max_val = 90.0
            step = 1.0
        elif field_name == 'oversold':
            min_val = 10.0
            max_val = 50.0
            step = 1.0
        
        # Créer l'info du paramètre
        if param_type:
            param_info[field_name] = {
                "type": param_type,
                "default": field_default,
                "min_val": min_val,
                "max_val": max_val,
                "step": step,
                "choices": choices
            }
    
    return param_info


def get_risk_param_info(param_class: Type) -> Dict[str, Dict[str, Any]]:
    """
    Extrait des informations sur les paramètres d'une classe de paramètres de risque.
    Utilise l'introspection pour découvrir les types et valeurs par défaut.
    
    Args:
        param_class: Classe de paramètres (e.g., FixedRiskParams, AtrRiskParams)
        
    Returns:
        Dict[str, Dict[str, Any]]: Informations sur les paramètres
    """
    # Utiliser la même logique que pour les indicateurs
    return get_indicator_param_info(param_class)


def get_param_class_for_indicator(indicator_type: IndicatorType) -> Optional[Type]:
    """
    Récupère la classe de paramètres associée à un type d'indicateur.
    
    Args:
        indicator_type: Type d'indicateur
        
    Returns:
        Optional[Type]: Classe de paramètres ou None si non trouvée
    """
    # Mapping des types d'indicateurs vers leurs classes de paramètres
    indicator_param_classes = {
        IndicatorType.EMA: EMAParams,
        IndicatorType.SMA: SMAParams,
        IndicatorType.RSI: RSIParams,
        IndicatorType.MACD: MACDParams,
        IndicatorType.ATR: ATRParams,
        IndicatorType.BOLL: BOLLParams,
        IndicatorType.BB: BOLLParams,  # Alias
        IndicatorType.STOCH: STOCHParams
    }
    
    return indicator_param_classes.get(indicator_type)


def get_param_class_for_risk(risk_mode: RiskModeType) -> Optional[Type]:
    """
    Récupère la classe de paramètres associée à un mode de gestion du risque.
    
    Args:
        risk_mode: Mode de gestion du risque
        
    Returns:
        Optional[Type]: Classe de paramètres ou None si non trouvée
    """
    # Mapping des modes de risque vers leurs classes de paramètres
    risk_param_classes = {
        RiskModeType.FIXED: FixedRiskParams,
        RiskModeType.ATR_BASED: AtrRiskParams,
        RiskModeType.VOLATILITY_BASED: VolatilityRiskParams,
        RiskModeType.EQUITY_PERCENT: EquityPercentRiskParams,
        RiskModeType.KELLEY: KelleyRiskParams
    }
    
    return risk_param_classes.get(risk_mode)


def get_indicator_default_params(indicator_type: IndicatorType) -> Dict[str, ParameterRange]:
    """
    Récupère les paramètres par défaut pour un type d'indicateur.
    
    Args:
        indicator_type: Type d'indicateur
        
    Returns:
        Dict[str, ParameterRange]: Paramètres par défaut pour l'indicateur
    """
    params = {}
    
    # Récupérer la classe de paramètres pour ce type d'indicateur
    param_class = get_param_class_for_indicator(indicator_type)
    if not param_class:
        return params
    
    # Extraire les informations sur les paramètres
    param_info = get_indicator_param_info(param_class)
    
    # Créer les plages de paramètres
    for param_name, info in param_info.items():
        param_type = info["type"]
        default_value = info["default"]
        
        if param_type == "int":
            params[param_name] = ParameterRange(
                name=param_name,
                param_type="int",
                min_value=info["min_val"],
                max_value=info["max_val"],
                step=info["step"],
                default_value=default_value
            )
        elif param_type == "float":
            params[param_name] = ParameterRange(
                name=param_name,
                param_type="float",
                min_value=info["min_val"],
                max_value=info["max_val"],
                step=info["step"],
                default_value=default_value
            )
        elif param_type == "categorical":
            params[param_name] = ParameterRange(
                name=param_name,
                param_type="categorical",
                choices=info["choices"],
                default_value=default_value
            )
    
    return params


def get_risk_default_params(risk_mode: RiskModeType) -> Dict[str, ParameterRange]:
    """
    Récupère les paramètres par défaut pour un mode de gestion du risque.
    
    Args:
        risk_mode: Mode de gestion du risque
        
    Returns:
        Dict[str, ParameterRange]: Paramètres par défaut pour le mode de risque
    """
    params = {}
    
    # Récupérer la classe de paramètres pour ce mode de risque
    param_class = get_param_class_for_risk(risk_mode)
    if not param_class:
        return params
    
    # Extraire les informations sur les paramètres
    param_info = get_risk_param_info(param_class)
    
    # Créer les plages de paramètres
    for param_name, info in param_info.items():
        param_type = info["type"]
        default_value = info["default"]
        
        # Préfixer les paramètres avec le nom du mode pour éviter les collisions
        prefixed_name = f"{risk_mode.value}_{param_name}"
        
        if param_type == "int":
            params[prefixed_name] = ParameterRange(
                name=prefixed_name,
                param_type="int",
                min_value=info["min_val"],
                max_value=info["max_val"],
                step=info["step"],
                default_value=default_value
            )
        elif param_type == "float":
            params[prefixed_name] = ParameterRange(
                name=prefixed_name,
                param_type="float",
                min_value=info["min_val"],
                max_value=info["max_val"],
                step=info["step"],
                default_value=default_value
            )
        elif param_type == "categorical":
            params[prefixed_name] = ParameterRange(
                name=prefixed_name,
                param_type="categorical",
                choices=info["choices"],
                default_value=default_value
            )
    
    return params


def get_available_indicators() -> List[IndicatorType]:
    """
    Récupère la liste de tous les indicateurs disponibles.
    
    Returns:
        List[IndicatorType]: Liste des types d'indicateurs disponibles
    """
    # Récupérer directement les membres de l'enum
    return list(IndicatorType)


def get_available_risk_modes() -> List[RiskModeType]:
    """
    Récupère la liste de tous les modes de gestion du risque disponibles.
    
    Returns:
        List[RiskModeType]: Liste des modes de risque disponibles
    """
    # Récupérer directement les membres de l'enum
    return list(RiskModeType)


def create_default_search_space() -> SearchSpace:
    """
    Crée un espace de recherche par défaut avec des indicateurs courants.
    
    Returns:
        SearchSpace: Espace de recherche par défaut
    """
    space = SearchSpace(
        name="default",
        description="Espace de recherche par défaut avec une sélection d'indicateurs courants"
    )
    
    # Ajouter quelques indicateurs courants
    space.add_indicator(IndicatorType.EMA)
    space.add_indicator(IndicatorType.RSI)
    
    # Configurer les paramètres de risque
    space.configure_risk()
    
    return space


def create_trend_following_search_space() -> SearchSpace:
    """
    Crée un espace de recherche pour les stratégies de suivi de tendance.
    
    Returns:
        SearchSpace: Espace de recherche pour le suivi de tendance
    """
    space = SearchSpace(
        name="trend_following",
        description="Stratégie de suivi de tendance avec EMA, MACD et ATR"
    )
    
    # Ajouter les indicateurs pertinents pour le suivi de tendance
    space.add_indicator(IndicatorType.EMA)
    space.add_indicator(IndicatorType.MACD)
    space.add_indicator(IndicatorType.ATR)
    
    # Configurer les paramètres de risque
    space.configure_risk(RiskModeType.ATR_BASED)
    
    return space


def create_mean_reversion_search_space() -> SearchSpace:
    """
    Crée un espace de recherche pour les stratégies de retour à la moyenne.
    
    Returns:
        SearchSpace: Espace de recherche pour le retour à la moyenne
    """
    space = SearchSpace(
        name="mean_reversion",
        description="Stratégie de retour à la moyenne avec RSI et Bollinger Bands"
    )
    
    # Ajouter les indicateurs pertinents pour le retour à la moyenne
    space.add_indicator(IndicatorType.RSI)
    space.add_indicator(IndicatorType.BB)
    space.add_indicator(IndicatorType.STOCH)
    
    # Configurer les paramètres de risque
    space.configure_risk()
    
    return space


# Dictionnaire des espaces de recherche prédéfinis
PREDEFINED_SEARCH_SPACES = {
    "default": create_default_search_space(),
    "trend_following": create_trend_following_search_space(),
    "mean_reversion": create_mean_reversion_search_space()
}


def get_predefined_search_space(name: str) -> SearchSpace:
    """
    Récupère un espace de recherche prédéfini.
    
    Args:
        name: Nom de l'espace de recherche
        
    Returns:
        SearchSpace: Espace de recherche prédéfini
        
    Raises:
        ValueError: Si l'espace de recherche n'existe pas
    """
    if name in PREDEFINED_SEARCH_SPACES:
        # Créer une copie profonde
        import copy
        return copy.deepcopy(PREDEFINED_SEARCH_SPACES[name])
    else:
        raise ValueError(f"Espace de recherche '{name}' non trouvé. Options disponibles: {', '.join(PREDEFINED_SEARCH_SPACES.keys())}")


def list_predefined_search_spaces() -> Dict[str, str]:
    """
    Liste tous les espaces de recherche prédéfinis.
    
    Returns:
        Dict[str, str]: Dictionnaire {nom: description}
    """
    return {name: space.description for name, space in PREDEFINED_SEARCH_SPACES.items()}


# Module pour le sélecteur amélioré qui utilise les paramètres configurés
class StrategyParameterSelector:
    """
    Classe utilitaire pour sélectionner et valider des paramètres
    pour la création de stratégies à partir d'un espace de recherche.
    """
    def __init__(self, search_space: SearchSpace):
        """
        Initialise le sélecteur avec un espace de recherche.
        
        Args:
            search_space: L'espace de recherche contenant les paramètres
        """
        self.search_space = search_space
        self.params = {}
    
    def list_indicator_params(self, indicator_type: IndicatorType) -> Dict[str, ParameterRange]:
        """
        Liste les paramètres configurés pour un indicateur.
        
        Args:
            indicator_type: Type d'indicateur
            
        Returns:
            Dict[str, ParameterRange]: Paramètres configurés
            
        Raises:
            SearchSpaceError: Si l'indicateur n'est pas dans l'espace de recherche
        """
        indicator_name = indicator_type.value
        if indicator_name not in self.search_space.selected_indicators:
            raise SearchSpaceError(f"L'indicateur {indicator_name} n'est pas configuré dans l'espace de recherche")
        
        return self.search_space.selected_indicators[indicator_name]
    
    def list_risk_params(self) -> Dict[str, ParameterRange]:
        """
        Liste les paramètres de risque configurés.
        
        Returns:
            Dict[str, ParameterRange]: Paramètres de risque
        """
        return self.search_space.risk_params
    
    def suggest_all_parameters(self, trial) -> Dict[str, Any]:
        """
        Suggère des valeurs pour tous les paramètres configurés.
        
        Args:
            trial: Trial Optuna
            
        Returns:
            Dict[str, Any]: Paramètres suggérés
        """
        params = {}
        
        # Parcourir tous les indicateurs et leurs paramètres
        for ind_name, ind_params in self.search_space.selected_indicators.items():
            for param_name, param_range in ind_params.items():
                full_name = f"{ind_name.lower()}_{param_name}"
                
                if param_range.param_type == "int":
                    params[full_name] = trial.suggest_int(
                        full_name,
                        param_range.min_value,
                        param_range.max_value,
                        step=param_range.step or 1,
                        log=param_range.log_scale
                    )
                elif param_range.param_type == "float":
                    params[full_name] = trial.suggest_float(
                        full_name,
                        param_range.min_value,
                        param_range.max_value,
                        step=param_range.step,
                        log=param_range.log_scale
                    )
                elif param_range.param_type == "categorical":
                    params[full_name] = trial.suggest_categorical(full_name, param_range.choices)
        
        # Parcourir tous les paramètres de risque
        for param_name, param_range in self.search_space.risk_params.items():
            if param_range.param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_range.min_value,
                    param_range.max_value,
                    step=param_range.step or 1,
                    log=param_range.log_scale
                )
            elif param_range.param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_range.min_value,
                    param_range.max_value,
                    step=param_range.step,
                    log=param_range.log_scale
                )
            elif param_range.param_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, param_range.choices)
        
        return params


# Exemple d'utilisation si exécuté directement
if __name__ == "__main__":
    print("=== Indicateurs disponibles ===")
    for ind in get_available_indicators():
        print(f"- {ind.value}")
    
    print("\n=== Modes de risque disponibles ===")
    for mode in get_available_risk_modes():
        print(f"- {mode.value}")
    
    print("\n=== Exemple de listing des paramètres requis ===")
    ema_params = SearchSpace("test").get_required_params(IndicatorType.EMA)
    print(f"Paramètres requis pour EMA:")
    for name, info in ema_params.items():
        print(f"- {name}: type={info['type']}, default={info['default']}")
    
    print("\n=== Création d'un espace de recherche personnalisé ===")
    try:
        # Création avec erreur volontaire (paramètres manquants)
        space = SearchSpace("my_strategy", "Ma stratégie personnalisée")
        
        # Ceci va lever une exception car nous ne fournissons pas tous les paramètres requis
        print("Tentative d'ajout d'un indicateur sans tous les paramètres requis:")
        space.add_indicator(IndicatorType.EMA, {
            # 'period' est manquant !
            'source': ParameterRange(
                name='source',
                param_type='categorical',
                choices=["close", "open", "high", "low"]
            )
        })
    except SearchSpaceError as e:
        print(f"Erreur capturée (c'est normal): {e}")
    
    # Ajout correct
    print("\nAjout correct d'un indicateur avec tous les paramètres requis:")
    space = SearchSpace("my_strategy", "Ma stratégie personnalisée")
    space.add_indicator(IndicatorType.EMA, {
        'period': ParameterRange(
            name='period',
            param_type='int',
            min_value=5,
            max_value=50,
            step=1
        ),
        'source': ParameterRange(
            name='source',
            param_type='categorical',
            choices=["close", "open", "high", "low"]
        )
    })
    
    print("Indicateur EMA ajouté avec succès!")
    
    # Utilisation du sélecteur
    print("\n=== Utilisation du sélecteur de paramètres ===")
    selector = StrategyParameterSelector(space)
    print("Paramètres EMA disponibles:")
    ema_params = selector.list_indicator_params(IndicatorType.EMA)
    for name, param in ema_params.items():
        print(f"- {name}: {param.to_dict()}")