"""
Configuration et calcul des scores pour l'évaluation des stratégies de trading.
Définit des formules de scoring personnalisables et des métriques normalisées.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ===== Définition des métriques disponibles =====
class MetricType(Enum):
    """Types de métriques pour l'évaluation des stratégies"""
    ROI = "roi"                          # Return on Investment
    WIN_RATE = "win_rate"                # Taux de victoire
    TOTAL_TRADES = "total_trades"        # Nombre total de trades
    MAX_DRAWDOWN = "max_drawdown"        # Drawdown maximal
    PROFIT_FACTOR = "profit_factor"      # Facteur de profit
    SHARPE_RATIO = "sharpe_ratio"        # Ratio de Sharpe
    TRADES_PER_DAY = "trades_per_day"    # Nombre de trades par jour
    AVG_PROFIT = "avg_profit"            # Profit moyen par trade
    MAX_CONSECUTIVE_LOSSES = "max_consecutive_losses"  # Pertes consécutives maximales
    RECOVERY_FACTOR = "recovery_factor"  # Facteur de récupération (ROI / Max DD)
    RISK_REWARD_RATIO = "risk_reward_ratio"  # Ratio risque/récompense moyen

# ===== Configuration des métriques =====
@dataclass
class MetricConfig:
    """
    Configuration d'une métrique d'évaluation.
    Définit comment la métrique est calculée et normalisée.
    """
    name: str
    description: str = ""
    higher_is_better: bool = True
    normalization: Optional[Callable] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default_weight: float = 1.0
    
    def __post_init__(self):
        """Validation et initialisation par défaut"""
        # Si aucune fonction de normalisation n'est fournie, en créer une par défaut
        if self.normalization is None:
            # Normalisation par défaut selon si higher_is_better
            if self.higher_is_better:
                # Pour les métriques où "plus c'est mieux"
                if self.min_value is not None and self.max_value is not None:
                    # Normalisation min-max
                    self.normalization = lambda x: max(0, min(1, (x - self.min_value) / (self.max_value - self.min_value)))
                else:
                    # Normalisation sigmoïde
                    self.normalization = lambda x: 1 / (1 + np.exp(-x))
            else:
                # Pour les métriques où "moins c'est mieux"
                if self.min_value is not None and self.max_value is not None:
                    # Normalisation min-max inversée
                    self.normalization = lambda x: max(0, min(1, 1 - (x - self.min_value) / (self.max_value - self.min_value)))
                else:
                    # Normalisation sigmoïde inversée
                    self.normalization = lambda x: 1 / (1 + np.exp(x))

@dataclass
class ScoringWeights:
    """Poids pour les différentes métriques dans le calcul du score"""
    roi: float = 2.5
    win_rate: float = 0.5
    max_drawdown: float = 2.0
    profit_factor: float = 2.0
    total_trades: float = 1.0
    sharpe_ratio: float = 0.0
    trades_per_day: float = 0.0
    avg_profit: float = 0.0
    max_consecutive_losses: float = 0.0
    recovery_factor: float = 0.0
    risk_reward_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convertit les poids en dictionnaire"""
        return {
            "roi": self.roi,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "sharpe_ratio": self.sharpe_ratio,
            "trades_per_day": self.trades_per_day,
            "avg_profit": self.avg_profit,
            "max_consecutive_losses": self.max_consecutive_losses,
            "recovery_factor": self.recovery_factor,
            "risk_reward_ratio": self.risk_reward_ratio
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'ScoringWeights':
        """Crée des poids à partir d'un dictionnaire"""
        weights = cls()
        
        for key, value in data.items():
            if hasattr(weights, key):
                setattr(weights, key, value)
            else:
                logger.warning(f"Poids inconnu: {key}")
        
        return weights

@dataclass
class ScoringFormula:
    """
    Formule pour calculer le score global d'une stratégie.
    Définit comment les métriques sont combinées.
    """
    name: str
    description: str = ""
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    transformation: Optional[Callable] = None
    
    def __post_init__(self):
        """Initialisation par défaut"""
        if self.transformation is None:
            # Transformation par défaut pour ajuster l'échelle du score
            self.transformation = lambda x: x * 10.0
    
    def to_dict(self) -> Dict:
        """Convertit la formule en dictionnaire"""
        return {
            "name": self.name,
            "description": self.description,
            "weights": self.weights.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ScoringFormula':
        """Crée une formule à partir d'un dictionnaire"""
        weights = ScoringWeights.from_dict(data.get("weights", {}))
        
        return cls(
            name=data.get("name", "Custom"),
            description=data.get("description", ""),
            weights=weights
        )

# ===== Configuration et normalisation des métriques =====
METRIC_CONFIGS = {
    # ROI (Return on Investment)
    MetricType.ROI.value: MetricConfig(
        name="ROI",
        description="Return on Investment (pourcentage)",
        higher_is_better=True,
        normalization=lambda x: 1 / (1 + np.exp(-10 * x)),  # Sigmoïde adaptée
        default_weight=2.5
    ),
    
    # Win Rate
    MetricType.WIN_RATE.value: MetricConfig(
        name="Win Rate",
        description="Taux de victoire des trades",
        higher_is_better=True,
        min_value=0.0,
        max_value=1.0,
        normalization=lambda x: 2 * x - 0.5 if x > 0.5 else x / 2,  # Favorise les win rates > 50%
        default_weight=0.5
    ),
    
    # Total Trades
    MetricType.TOTAL_TRADES.value: MetricConfig(
        name="Total Trades",
        description="Nombre total de trades exécutés",
        higher_is_better=True,
        normalization=lambda x: min(1.0, x / 100.0),  # Valeur max à 100 trades
        default_weight=1.0
    ),
    
    # Max Drawdown
    MetricType.MAX_DRAWDOWN.value: MetricConfig(
        name="Maximum Drawdown",
        description="Perte maximale par rapport au pic précédent",
        higher_is_better=False,
        min_value=0.0,
        max_value=1.0,
        normalization=lambda x: max(0, 1 - 4 * x),  # Pénalise les drawdowns élevés
        default_weight=2.0
    ),
    
    # Profit Factor
    MetricType.PROFIT_FACTOR.value: MetricConfig(
        name="Profit Factor",
        description="Ratio des profits sur les pertes",
        higher_is_better=True,
        normalization=lambda x: min(1.0, x / 3.0),  # Valeur max à PF=3
        default_weight=2.0
    ),
    
    # Sharpe Ratio
    MetricType.SHARPE_RATIO.value: MetricConfig(
        name="Sharpe Ratio",
        description="Mesure de la performance ajustée au risque",
        higher_is_better=True,
        normalization=lambda x: min(1.0, max(0, x / 3.0)),  # Valeur max à SR=3
        default_weight=0.0
    ),
    
    # Trades Per Day
    MetricType.TRADES_PER_DAY.value: MetricConfig(
        name="Trades Per Day",
        description="Nombre moyen de trades par jour",
        higher_is_better=True,
        min_value=0.0,
        max_value=10.0,
        normalization=lambda x: 0.5 + 0.5 * min(1.0, x / 3.0),  # Optimale entre 1-3 trades par jour
        default_weight=0.0
    ),
    
    # Average Profit Per Trade
    MetricType.AVG_PROFIT.value: MetricConfig(
        name="Average Profit",
        description="Profit moyen par trade",
        higher_is_better=True,
        normalization=lambda x: 1 / (1 + np.exp(-20 * x)),  # Sigmoïde adaptée
        default_weight=0.0
    ),
    
    # Max Consecutive Losses
    MetricType.MAX_CONSECUTIVE_LOSSES.value: MetricConfig(
        name="Max Consecutive Losses",
        description="Nombre maximal de pertes consécutives",
        higher_is_better=False,
        min_value=0,
        max_value=20,
        normalization=lambda x: max(0, 1 - x / 20),  # Pénalise les longues séries de pertes
        default_weight=0.0
    ),
    
    # Recovery Factor
    MetricType.RECOVERY_FACTOR.value: MetricConfig(
        name="Recovery Factor",
        description="Ratio du ROI sur le drawdown maximal",
        higher_is_better=True,
        normalization=lambda x: min(1.0, x / 5.0),  # Valeur max à RF=5
        default_weight=0.0
    ),
    
    # Risk-Reward Ratio
    MetricType.RISK_REWARD_RATIO.value: MetricConfig(
        name="Risk-Reward Ratio",
        description="Ratio moyen de la récompense sur le risque",
        higher_is_better=True,
        normalization=lambda x: min(1.0, x / 3.0),  # Valeur max à RR=3
        default_weight=0.0
    )
}

# ===== Formules de scoring prédéfinies =====
PREDEFINED_FORMULAS = {
    "standard": ScoringFormula(
        name="Standard",
        description="Formule standard équilibrée entre performance et risque",
        weights=ScoringWeights(
            roi=2.5,
            win_rate=0.5,
            max_drawdown=2.0,
            profit_factor=2.0,
            total_trades=1.0
        ),
        transformation=lambda x: x * 10.0
    ),
    
    "performance": ScoringFormula(
        name="Performance",
        description="Formule axée sur la performance pure",
        weights=ScoringWeights(
            roi=4.0,
            win_rate=0.5,
            max_drawdown=1.0,
            profit_factor=2.5,
            total_trades=0.5
        ),
        transformation=lambda x: x * 10.0
    ),
    
    "conservative": ScoringFormula(
        name="Conservative",
        description="Formule axée sur la gestion du risque",
        weights=ScoringWeights(
            roi=1.5,
            win_rate=1.0,
            max_drawdown=3.0,
            profit_factor=1.5,
            total_trades=1.0,
            max_consecutive_losses=1.0
        ),
        transformation=lambda x: x * 10.0
    ),
    
    "frequency": ScoringFormula(
        name="Frequency",
        description="Formule favorisant les stratégies à haute fréquence",
        weights=ScoringWeights(
            roi=2.0,
            win_rate=1.0,
            max_drawdown=1.5,
            profit_factor=1.5,
            total_trades=2.0,
            trades_per_day=1.0
        ),
        transformation=lambda x: x * 10.0
    ),
    
    "comprehensive": ScoringFormula(
        name="Comprehensive",
        description="Formule utilisant toutes les métriques disponibles",
        weights=ScoringWeights(
            roi=2.0,
            win_rate=1.0,
            max_drawdown=1.5,
            profit_factor=1.5,
            total_trades=1.0,
            sharpe_ratio=1.0,
            trades_per_day=0.5,
            avg_profit=1.0,
            max_consecutive_losses=0.5,
            recovery_factor=1.0,
            risk_reward_ratio=1.0
        ),
        transformation=lambda x: x * 10.0
    )
}

# ===== Calculateur de score =====
class ScoreCalculator:
    """
    Calculateur de score pour les stratégies de trading.
    Applique une formule de scoring aux métriques d'une stratégie.
    """
    
    def __init__(self, formula: Optional[Union[str, ScoringFormula]] = None):
        """
        Initialise le calculateur de score.
        
        Args:
            formula: Formule de scoring ou nom d'une formule prédéfinie
        """
        self.metric_configs = METRIC_CONFIGS
        
        # Sélection de la formule
        if formula is None:
            self.formula = PREDEFINED_FORMULAS["standard"]
        elif isinstance(formula, str):
            if formula in PREDEFINED_FORMULAS:
                self.formula = PREDEFINED_FORMULAS[formula]
            else:
                logger.warning(f"Formule '{formula}' non trouvée, utilisation de la formule standard")
                self.formula = PREDEFINED_FORMULAS["standard"]
        else:
            self.formula = formula
    
    def calculate_score(self, metrics: Dict[str, float]) -> float:
        """
        Calcule le score d'une stratégie à partir de ses métriques.
        
        Args:
            metrics: Dictionnaire des métriques mesurées
            
        Returns:
            float: Score final
        """
        # Validation initiale
        if not metrics or "total_trades" not in metrics or metrics["total_trades"] <= 0:
            return float('-inf')
        
        # Normalisation des métriques
        normalized_metrics = {}
        weights = self.formula.weights.to_dict()
        
        for metric_name, metric_config in self.metric_configs.items():
            if metric_name in metrics and metric_name in weights and weights[metric_name] > 0:
                raw_value = metrics[metric_name]
                
                # Normalisation
                normalized_value = metric_config.normalization(raw_value)
                normalized_metrics[metric_name] = normalized_value
        
        # Calcul du score final pondéré
        score = 0.0
        total_weight = sum(weights[k] for k in normalized_metrics.keys())
        
        if total_weight <= 0:
            logger.warning("Somme des poids nulle ou négative, utilisation de poids égaux")
            total_weight = len(normalized_metrics)
            weights = {k: 1.0 for k in normalized_metrics.keys()}
        
        for metric, norm_value in normalized_metrics.items():
            score += norm_value * (weights[metric] / total_weight)
        
        # Transformation finale pour ajuster l'échelle
        final_score = self.formula.transformation(score)
        
        return final_score
    
    def normalize_metric(self, metric_name: str, value: float) -> float:
        """
        Normalise une métrique individuelle.
        
        Args:
            metric_name: Nom de la métrique
            value: Valeur brute
            
        Returns:
            float: Valeur normalisée
        """
        if metric_name not in self.metric_configs:
            logger.warning(f"Métrique '{metric_name}' non reconnue, aucune normalisation appliquée")
            return value
        
        return self.metric_configs[metric_name].normalization(value)
    
    def get_formula_info(self) -> Dict:
        """
        Récupère les informations sur la formule utilisée.
        
        Returns:
            Dict: Informations sur la formule
        """
        return {
            "name": self.formula.name,
            "description": self.formula.description,
            "weights": self.formula.weights.to_dict()
        }
    
    def set_formula(self, formula: Union[str, ScoringFormula]) -> None:
        """
        Modifie la formule utilisée.
        
        Args:
            formula: Nouvelle formule ou nom d'une formule prédéfinie
        """
        if isinstance(formula, str):
            if formula in PREDEFINED_FORMULAS:
                self.formula = PREDEFINED_FORMULAS[formula]
            else:
                logger.warning(f"Formule '{formula}' non trouvée, aucun changement")
        else:
            self.formula = formula
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Met à jour les poids de la formule actuelle.
        
        Args:
            new_weights: Nouveaux poids
        """
        weights_dict = self.formula.weights.to_dict()
        
        for key, value in new_weights.items():
            if key in weights_dict:
                weights_dict[key] = value
        
        self.formula.weights = ScoringWeights.from_dict(weights_dict)
    
    @staticmethod
    def list_predefined_formulas() -> Dict[str, str]:
        """
        Liste les formules prédéfinies disponibles.
        
        Returns:
            Dict[str, str]: Dictionnaire {nom: description}
        """
        return {name: formula.description for name, formula in PREDEFINED_FORMULAS.items()}
    
    @staticmethod
    def list_available_metrics() -> Dict[str, str]:
        """
        Liste les métriques disponibles.
        
        Returns:
            Dict[str, str]: Dictionnaire {nom: description}
        """
        return {name: config.description for name, config in METRIC_CONFIGS.items()}
    
    @staticmethod
    def create_custom_formula(
        name: str,
        weights: Dict[str, float],
        description: str = "Formule personnalisée"
    ) -> ScoringFormula:
        """
        Crée une formule personnalisée.
        
        Args:
            name: Nom de la formule
            weights: Poids des métriques
            description: Description de la formule
            
        Returns:
            ScoringFormula: Formule personnalisée
        """
        return ScoringFormula(
            name=name,
            description=description,
            weights=ScoringWeights.from_dict(weights)
        )