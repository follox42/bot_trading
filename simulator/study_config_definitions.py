"""
Fichier contenant les définitions des indicateurs, modes de risque, et autres paramètres
pour la création et l'optimisation des études de trading.

Ce fichier centralise toutes les configurations pour faciliter la maintenance et l'extension du système.
"""
from simulator.indicators import IndicatorType, Operator, LogicOperator, Block, Condition
from simulator.risk import RiskMode
from simulator.config import MarginMode, TradingMode, OptimizationMethod, PrunerMethod, ScoringFormula
from simulator.simulator import SimulationConfig

from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler, NSGAIISampler
from optuna.pruners import MedianPruner, PercentilePruner, HyperbandPruner
import numpy as np

# Définition des catégories d'indicateurs
INDICATOR_CATEGORIES = {
    "trend": {
        "label": "Indicateurs de Tendance",
        "indicators": [IndicatorType.EMA, IndicatorType.SMA, IndicatorType.MACD]
    },
    "oscillators": {
        "label": "Oscillateurs",
        "indicators": [IndicatorType.RSI, IndicatorType.STOCH]
    },
    "volatility": {
        "label": "Indicateurs de Volatilité",
        "indicators": [IndicatorType.ATR, IndicatorType.BOLL]
    },
    "volume": {
        "label": "Indicateurs de Volume",
        "indicators": [IndicatorType.VWAP, IndicatorType.MFIMACD]
    }
}

# Définitions des indicateurs avec leurs paramètres par défaut
INDICATOR_DEFAULTS = {
    IndicatorType.EMA: {
        "min_period": 5,
        "max_period": 200,
        "step": 5,
        "description": "Moyenne Mobile Exponentielle",
        "price_type": "close"
    },
    IndicatorType.SMA: {
        "min_period": 5,
        "max_period": 200,
        "step": 5,
        "description": "Moyenne Mobile Simple",
        "price_type": "close"
    },
    IndicatorType.MACD: {
        "min_period": 12,
        "max_period": 26,
        "step": 1,
        "description": "Convergence/Divergence de Moyennes Mobiles",
        "price_type": "close"
    },
    IndicatorType.RSI: {
        "min_period": 7,
        "max_period": 30,
        "step": 1,
        "description": "Relative Strength Index",
        "price_type": "close"
    },
    IndicatorType.STOCH: {
        "min_period": 5,
        "max_period": 14,
        "step": 1,
        "description": "Stochastique",
        "price_type": "close"
    },
    IndicatorType.ATR: {
        "min_period": 7,
        "max_period": 30,
        "step": 1,
        "description": "Average True Range",
        "price_type": "close"
    },
    IndicatorType.BOLL: {
        "min_period": 20,
        "max_period": 20,
        "step": 1,
        "description": "Bandes de Bollinger",
        "price_type": "close"
    },
    IndicatorType.VWAP: {
        "min_period": 1,
        "max_period": 1,
        "step": 1,
        "description": "Volume Weighted Average Price",
        "price_type": "close"
    },
    IndicatorType.MFIMACD: {
        "min_period": 12,
        "max_period": 26,
        "step": 1,
        "description": "Money Flow Index MACD",
        "price_type": "close"
    }
}

# Paramètres des modes de risque
RISK_MODES = {
    RiskMode.FIXED: {
        "name": "FIXED",
        "label": "Mode à risque fixe",
        "controls": [
            {
                "id": "fixed-pos",
                "label": "Position Size (%)",
                "min": 1,
                "max": 10, 
                "step": 0.1,
                "unit": "%",
                "width": 6
            },
            {
                "id": "fixed-sl",
                "label": "Stop Loss (%)",
                "min": 0.5,
                "max": 2,
                "step": 0.1,
                "unit": "%",
                "width": 6
            },
            {
                "id": "fixed-tp",
                "label": "Take Profit (multiplicateur de SL)",
                "min": 1.5,
                "max": 3,
                "step": 0.1,
                "unit": "x",
                "width": 12
            }
        ]
    },
    RiskMode.ATR_BASED: {
        "name": "ATR",
        "label": "Mode basé sur l'ATR",
        "controls": [
            {
                "id": "atr-period",
                "label": "Période ATR",
                "min": 5,
                "max": 20,
                "step": 1,
                "width": 6
            },
            {
                "id": "atr-mult",
                "label": "Multiplicateur ATR",
                "min": 0.5,
                "max": 2,
                "step": 0.1,
                "unit": "×",
                "width": 6
            }
        ]
    },
    RiskMode.VOLATILITY_BASED: {
        "name": "VOLATILITY",
        "label": "Mode basé sur la volatilité",
        "controls": [
            {
                "id": "vol-period",
                "label": "Période de volatilité",
                "min": 10,
                "max": 30,
                "step": 1,
                "width": 6
            },
            {
                "id": "vol-mult",
                "label": "Multiplicateur de volatilité",
                "min": 0.5,
                "max": 2,
                "step": 0.1,
                "unit": "×",
                "width": 6
            }
        ]
    }
}

# Paramètres généraux de risque
GENERAL_RISK_PARAMS = {
    "position_size": {
        "label": "Taille de position (%)",
        "min": 1, 
        "max": 10,
        "step": 0.1,
        "unit": "%"
    },
    "stop_loss": {
        "label": "Stop Loss (%)",
        "min": 0.5,
        "max": 3,
        "step": 0.1,
        "unit": "%"
    },
    "take_profit": {
        "label": "Take Profit (multiplicateur de SL)",
        "min": 1.5,
        "max": 3,
        "step": 0.1,
        "unit": "x"
    }
}

# Paramètres de simulation
SIMULATION_PARAMS = {
    "balance": {
        "label": "Balance initiale",
        "min": 1000,
        "max": 10000,
        "step": 100,
        "unit": "$"
    },
    "leverage": {
        "label": "Levier",
        "min": 1,
        "max": 10,
        "step": 1,
        "unit": "×"
    },
    "fee": {
        "label": "Frais de trading",
        "value": 0.1,
        "min": 0,
        "max": 1,
        "step": 0.01,
        "unit": "%"
    },
    "slippage": {
        "label": "Slippage",
        "value": 0.05,
        "min": 0,
        "max": 1,
        "step": 0.01,
        "unit": "%"
    },
    "margin_modes": [
        {"label": "Isolé", "value": MarginMode.ISOLATED.value},
        {"label": "Cross", "value": MarginMode.CROSS.value}
    ],
    "trading_modes": [
        {"label": "One-way", "value": TradingMode.ONE_WAY.value},
        {"label": "Hedge", "value": TradingMode.HEDGE.value}
    ],
    "min_trade_size": {
        "label": "Taille minimale de trade",
        "value": 0.001,
        "min": 0.0001,
        "max": 10,
        "step": 0.001,
        "unit": "BTC"
    },
    "max_trade_size": {
        "label": "Taille maximale de trade",
        "value": 1000,
        "min": 0.1,
        "max": 1000000,
        "step": 10,
        "unit": "USDT"
    }
}

# Opérateurs disponibles pour les conditions
AVAILABLE_OPERATORS = {
    Operator.GREATER: {
        "symbol": ">",
        "description": "Supérieur à"
    },
    Operator.LESS: {
        "symbol": "<",
        "description": "Inférieur à"
    },
    Operator.GREATER_EQUAL: {
        "symbol": ">=",
        "description": "Supérieur ou égal à"
    },
    Operator.LESS_EQUAL: {
        "symbol": "<=",
        "description": "Inférieur ou égal à"
    },
    Operator.EQUAL: {
        "symbol": "==",
        "description": "Égal à"
    },
    Operator.CROSS_ABOVE: {
        "symbol": "CROSS_ABOVE",
        "description": "Croise au-dessus"
    },
    Operator.CROSS_BELOW: {
        "symbol": "CROSS_BELOW",
        "description": "Croise en-dessous"
    }
}

# Opérateurs logiques pour les blocs
LOGIC_OPERATORS = {
    LogicOperator.AND: {
        "symbol": "ET",
        "description": "Toutes les conditions doivent être vraies"
    },
    LogicOperator.OR: {
        "symbol": "OU",
        "description": "Au moins une condition doit être vraie"
    }
}

# Paramètres pour la structure de stratégie
STRATEGY_STRUCTURE_PARAMS = {
    "blocks": {
        "min_blocks": {
            "label": "Nombre minimum de blocs",
            "value": 1,
            "min": 1,
            "max": 10,
            "step": 1
        },
        "max_blocks": {
            "label": "Nombre maximum de blocs",
            "value": 3, 
            "min": 1,
            "max": 10,
            "step": 1
        }
    },
    "conditions": {
        "min_conditions": {
            "label": "Conditions min. par bloc",
            "value": 1,
            "min": 1,
            "max": 5,
            "step": 1
        },
        "max_conditions": {
            "label": "Conditions max. par bloc",
            "value": 3,
            "min": 1,
            "max": 10,
            "step": 1
        }
    },
    "probabilities": {
        "cross_probability": {
            "label": "Probabilité croisement de signaux",
            "value": 0.3,
            "min": 0,
            "max": 1,
            "step": 0.05
        },
        "value_comparison_probability": {
            "label": "Probabilité comparaison de valeur",
            "value": 0.4,
            "min": 0,
            "max": 1,
            "step": 0.05
        }
    },
    "value_ranges": {
        "rsi_range": {
            "label": "Valeurs RSI",
            "min": 20,
            "max": 80,
            "step": 1
        },
        "price_range": {
            "label": "Valeurs de prix (multiplicateurs)",
            "min": 0,
            "max": 1000,
            "step": 1
        },
        "general_range": {
            "label": "Valeurs générales",
            "min": -100,
            "max": 100,
            "step": 1
        }
    }
}

# Configuration de simulation par défaut
DEFAULT_SIM_CONFIG = SimulationConfig(
    initial_balance=10000.0,
    fee_open=0.001,
    fee_close=0.001,
    slippage=0.001,
    tick_size=0.01,
    min_trade_size=0.001,
    max_trade_size=100000.0,
    leverage=1,
    margin_mode=MarginMode.ISOLATED.value,
    trading_mode=TradingMode.ONE_WAY.value
)

# Définir les sources de données disponibles
DATA_SOURCES = [
    {"label": "Télécharger automatiquement", "value": "auto"},
    {"label": "Fichier local", "value": "local"},
    {"label": "API Exchange", "value": "api"}
]

# Définir les périodes de données disponibles
DATA_PERIODS = [
    {"label": "1 mois", "value": "1m"},
    {"label": "3 mois", "value": "3m"},
    {"label": "6 mois", "value": "6m"},
    {"label": "1 an", "value": "1y"},
    {"label": "2 ans", "value": "2y"},
    {"label": "5 ans", "value": "5y"},
    {"label": "Tout l'historique", "value": "all"}
]

# Types d'études
STUDY_TYPES = [
    {"label": "Standard", "value": "standard"},
    {"label": "Expérimentale", "value": "experimental"},
    {"label": "Production", "value": "production"}
]

# Poids des scores pour l'optimisation par défaut
DEFAULT_SCORE_WEIGHTS = {
    "roi": 2.5,
    "win_rate": 0.5,
    "max_drawdown": 2.0,
    "profit_factor": 2.0,
    "total_trades": 1.0,
    "avg_profit": 1.0
}

# Définition des méthodes d'optimisation disponibles
OPTIMIZATION_METHODS = {
    OptimizationMethod.TPE.value: {
        "name": "TPE (Tree-structured Parzen Estimator)",
        "description": "Méthode bayésienne efficace qui apprend des résultats précédents",
        "sampler_class": TPESampler,
        "params": {
            "n_startup_trials": {"type": "int", "default": 10, "min": 5, "max": 50},
            "n_ei_candidates": {"type": "int", "default": 24, "min": 10, "max": 100},
            "multivariate": {"type": "bool", "default": True},
            "seed": {"type": "int", "default": None}
        }
    },
    OptimizationMethod.RANDOM.value: {
        "name": "Random Search",
        "description": "Exploration aléatoire de l'espace de paramètres",
        "sampler_class": RandomSampler,
        "params": {
            "seed": {"type": "int", "default": None}
        }
    },
    OptimizationMethod.CMAES.value: {
        "name": "CMA-ES (Covariance Matrix Adaptation)",
        "description": "Algorithme évolutionnaire pour l'optimisation en espace continu",
        "sampler_class": CmaEsSampler,
        "params": {
            "n_startup_trials": {"type": "int", "default": 10, "min": 5, "max": 50},
            "seed": {"type": "int", "default": None}
        }
    },
    OptimizationMethod.NSGAII.value: {
        "name": "NSGA-II (Multi-objectif)",
        "description": "Algorithme génétique pour optimisation multi-objectif",
        "sampler_class": NSGAIISampler,
        "params": {
            "population_size": {"type": "int", "default": 50, "min": 10, "max": 200},
            "seed": {"type": "int", "default": None}
        }
    }
}

# Définition des pruners disponibles
PRUNER_METHODS = {
    PrunerMethod.MEDIAN.value: {
        "name": "Median Pruning",
        "description": "Arrête les trials sous-performants par rapport à la médiane",
        "pruner_class": MedianPruner,
        "params": {
            "n_startup_trials": {"type": "int", "default": 5, "min": 1, "max": 50},
            "n_warmup_steps": {"type": "int", "default": 0, "min": 0, "max": 10},
            "interval_steps": {"type": "int", "default": 1, "min": 1, "max": 10}
        }
    },
    PrunerMethod.PERCENTILE.value: {
        "name": "Percentile Pruning",
        "description": "Arrête les trials dans le bas du percentile spécifié",
        "pruner_class": PercentilePruner,
        "params": {
            "percentile": {"type": "float", "default": 25.0, "min": 0.0, "max": 100.0},
            "n_startup_trials": {"type": "int", "default": 5, "min": 1, "max": 50},
            "n_warmup_steps": {"type": "int", "default": 0, "min": 0, "max": 10}
        }
    },
    PrunerMethod.HYPERBAND.value: {
        "name": "Hyperband Pruning",
        "description": "Stratégie d'arrêt précoce pour optimiser le budget d'exploration",
        "pruner_class": HyperbandPruner,
        "params": {
            "min_resource": {"type": "int", "default": 1, "min": 1, "max": 10},
            "max_resource": {"type": "int", "default": 100, "min": 10, "max": 1000},
            "reduction_factor": {"type": "int", "default": 3, "min": 2, "max": 10}
        }
    },
    PrunerMethod.NONE.value: {
        "name": "Pas de pruning",
        "description": "Aucun arrêt anticipé des essais",
        "pruner_class": None,
        "params": {}
    }
}

# Métriques disponibles pour le scoring
AVAILABLE_METRICS = {
    "roi": {
        "name": "ROI",
        "description": "Retour sur investissement",
        "higher_is_better": True,
        "default_weight": 2.5,
        "min_weight": 0.0,
        "max_weight": 5.0,
        "normalization": lambda x: min(1.0, max(0, (1 / (1 + np.exp(-x * 2)) - 0.5) * 2))
    },
    "win_rate": {
        "name": "Win Rate",
        "description": "Taux de réussite des trades",
        "higher_is_better": True,
        "default_weight": 0.5,
        "min_weight": 0.0,
        "max_weight": 5.0,
        "normalization": lambda x: x  # Déjà entre 0 et 1
    },
    "max_drawdown": {
        "name": "Max Drawdown",
        "description": "Baisse maximale du capital depuis le sommet",
        "higher_is_better": False,
        "default_weight": 2.0,
        "min_weight": 0.0,
        "max_weight": 5.0,
        "normalization": lambda x: max(0.0, 1.0 - x)  # Transformation pour pénaliser les grands drawdowns
    },
    "profit_factor": {
        "name": "Profit Factor",
        "description": "Ratio profits bruts / pertes brutes",
        "higher_is_better": True,
        "default_weight": 2.0,
        "min_weight": 0.0,
        "max_weight": 5.0,
        "normalization": lambda x: min(1.0, max(0.0, np.log(x + 0.1) / 2.0)) if x > 0 else 0
    },
    "total_trades": {
        "name": "Total Trades",
        "description": "Nombre total de trades exécutés",
        "higher_is_better": True,
        "default_weight": 1.0,
        "min_weight": 0.0,
        "max_weight": 5.0,
        "normalization": lambda x: min(1.0, np.log(x + 1) / np.log(1001))
    },
    "avg_profit": {
        "name": "Avg Profit/Trade",
        "description": "Profit moyen par trade",
        "higher_is_better": True,
        "default_weight": 1.0,
        "min_weight": 0.0,
        "max_weight": 5.0,
        "normalization": lambda x: min(1.0, max(0.0, x * 10 + 0.5))
    },
    "sharpe_ratio": {
        "name": "Sharpe Ratio",
        "description": "Rendement ajusté au risque",
        "higher_is_better": True,
        "default_weight": 1.0,
        "min_weight": 0.0,
        "max_weight": 5.0,
        "normalization": lambda x: min(1.0, max(0.0, x / 3.0))
    },
    "sortino_ratio": {
        "name": "Sortino Ratio",
        "description": "Rendement ajusté au risque à la baisse",
        "higher_is_better": True,
        "default_weight": 1.0,
        "min_weight": 0.0,
        "max_weight": 5.0,
        "normalization": lambda x: min(1.0, max(0.0, x / 3.0))
    },
    "max_consecutive_losses": {
        "name": "Max Consecutive Losses",
        "description": "Maximum de pertes consécutives",
        "higher_is_better": False,
        "default_weight": 0.5,
        "min_weight": 0.0,
        "max_weight": 5.0,
        "normalization": lambda x: max(0.0, 1.0 - min(1.0, x / 20.0))
    },
    "trades_per_day": {
        "name": "Trades/Day",
        "description": "Nombre de trades par jour",
        "higher_is_better": True,
        "default_weight": 0.5,
        "min_weight": 0.0,
        "max_weight": 5.0,
        "normalization": lambda x: min(1.0, x / 10.0)
    }
}

# Formules de scoring prédéfinies
SCORING_FORMULAS = {
    ScoringFormula.STANDARD.value: {
        "name": "Standard",
        "description": "Formule standard pondérant le ROI, win rate, drawdown et profit factor",
        "weights": {
            "roi": 2.5,
            "win_rate": 0.5,
            "max_drawdown": 2.0,
            "profit_factor": 2.0,
            "total_trades": 1.0,
            "avg_profit": 1.0
        },
        "transformation": lambda score: (score ** 1.2) * 10.0  # Transformation non-linéaire
    },
    ScoringFormula.CONSISTENCY.value: {
        "name": "Consistency",
        "description": "Favorise la stabilité et régularité des résultats",
        "weights": {
            "roi": 1.5,
            "win_rate": 2.5,
            "max_drawdown": 3.0,
            "profit_factor": 2.0,
            "total_trades": 1.5,
            "max_consecutive_losses": 2.0
        },
        "transformation": lambda score: (score ** 1.1) * 10.0
    },
    ScoringFormula.AGGRESSIVE.value: {
        "name": "Aggressive",
        "description": "Priorise le ROI et le profit factor pour des stratégies à haut rendement",
        "weights": {
            "roi": 4.0,
            "win_rate": 0.2,
            "profit_factor": 3.0,
            "avg_profit": 2.0,
            "total_trades": 0.5
        },
        "transformation": lambda score: (score ** 1.3) * 10.0
    },
    ScoringFormula.CONSERVATIVE.value: {
        "name": "Conservative",
        "description": "Favorise le win rate et minimise les drawdowns",
        "weights": {
            "roi": 1.0,
            "win_rate": 3.0,
            "max_drawdown": 4.0,
            "profit_factor": 2.0,
            "max_consecutive_losses": 2.0
        },
        "transformation": lambda score: (score ** 1.0) * 10.0
    },
    ScoringFormula.VOLUME.value: {
        "name": "Volume",
        "description": "Priorise les stratégies à volume élevé de trades",
        "weights": {
            "roi": 1.5,
            "total_trades": 3.0,
            "trades_per_day": 3.0,
            "win_rate": 1.0,
            "avg_profit": 1.0
        },
        "transformation": lambda score: (score ** 1.2) * 10.0
    },
    ScoringFormula.CUSTOM.value: {
        "name": "Custom",
        "description": "Formule personnalisée configurable par l'utilisateur",
        "weights": {},  # Sera configuré par l'utilisateur
        "transformation": lambda score: (score ** 1.2) * 10.0
    }
}

# Pour la compatibilité avec le code existant qui utilise ces noms
OPTIMIZATION_METHODS_CONFIG = OPTIMIZATION_METHODS
PRUNER_METHODS_CONFIG = PRUNER_METHODS