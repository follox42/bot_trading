"""
Exemple de code démontrant comment créer un espace de recherche
et générer des stratégies à partir de celui-ci.

Ce script montre le flux de travail complet d'optimisation de stratégie:
1. Créer un espace de recherche personnalisé
2. Générer des stratégies à partir de l'espace de recherche
3. Évaluer les stratégies
"""
import random
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os
# Ajouter le répertoire parent au chemin d'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.optimization.search_config import (
    SearchSpace, ParameterRange, 
    get_predefined_search_space,
    get_available_indicators
)
from core.optimization.selector import ParameterSelector, create_strategy_from_trial
from core.strategy.constructor.constructor import StrategyConstructor
from core.strategy.indicators.indicators_config import IndicatorType
from core.strategy.risk.risk_config import RiskModeType

def print_search_space_details(search_space):
    """Affiche tous les détails d'un espace de recherche"""
    print("\n===== DÉTAILS DE L'ESPACE DE RECHERCHE =====")
    print(f"Nom: {search_space.name}")
    print(f"Description: {search_space.description}")
    
    print("\n--- INDICATEURS SÉLECTIONNÉS ---")
    for ind_name, params in search_space.selected_indicators.items():
        print(f"\nIndicateur: {ind_name}")
        for param_name, param_range in params.items():
            print(f"  - {param_name}:")
            print(f"    • Type: {param_range.param_type}")
            if param_range.param_type in ["int", "float"]:
                print(f"    • Plage: {param_range.min_value} à {param_range.max_value}")
                if param_range.step:
                    print(f"    • Pas: {param_range.step}")
                print(f"    • Échelle logarithmique: {param_range.log_scale}")
            elif param_range.param_type == "categorical":
                print(f"    • Choix: {param_range.choices}")
            if param_range.default_value is not None:
                print(f"    • Valeur par défaut: {param_range.default_value}")
    
    print("\n--- PARAMÈTRES DE RISQUE ---")
    for param_name, param_range in search_space.risk_params.items():
        print(f"\nParamètre: {param_name}")
        print(f"  - Type: {param_range.param_type}")
        if param_range.param_type in ["int", "float"]:
            print(f"  - Plage: {param_range.min_value} à {param_range.max_value}")
            if param_range.step:
                print(f"  - Pas: {param_range.step}")
            print(f"  - Échelle logarithmique: {param_range.log_scale}")
        elif param_range.param_type == "categorical":
            print(f"  - Choix: {param_range.choices}")
        if param_range.default_value is not None:
            print(f"  - Valeur par défaut: {param_range.default_value}")
    
    print("\n--- MÉTHODES IMPORTANTES ---")
    print("• add_indicator(indicator_type, params): Ajoute un indicateur")
    print("• remove_indicator(indicator_type): Supprime un indicateur")
    print("• configure_risk(risk_mode, params): Configure les paramètres de risque")
    print("• to_dict(): Convertit l'espace en dictionnaire")
    print("• to_json(): Convertit l'espace en JSON")
    print("• from_dict(data): Crée un espace à partir d'un dictionnaire")
    print("• from_json(json_str): Crée un espace à partir de JSON")
    
    print("\n=== STRUCTURE POUR MIN/MAX BLOCS ===")
    print("Note: La structure SearchSpace ne contient pas directement de paramètres")
    print("pour min_blocks et max_blocks. Ces paramètres devraient être ajoutés manuellement")
    print("à l'espace de recherche si nécessaires.")
    
    # Pour les blocs entry/exit
    has_longblock_params = any(key.startswith("longblock_") for key in search_space.risk_params)
    has_shortblock_params = any(key.startswith("shortblock_") for key in search_space.risk_params)
    
    if has_longblock_params or has_shortblock_params:
        print("\n--- PARAMÈTRES DE BLOCS TROUVÉS ---")
        for param_name, param_range in search_space.risk_params.items():
            if param_name.startswith("longblock_") or param_name.startswith("shortblock_"):
                print(f"• {param_name}: {param_range.param_type}")
                if param_range.param_type in ["int", "float"]:
                    print(f"  Min: {param_range.min_value}, Max: {param_range.max_value}")
    else:
        print("\nAucun paramètre de bloc (longblock_min_blocks, etc.) n'a été trouvé.")
        print("Vous pouvez les ajouter comme ceci:")
        print("""
search_space.risk_params["longblock_min_blocks"] = ParameterRange(
    name="longblock_min_blocks",
    param_type="int",
    min_value=1,
    max_value=3,
    default_value=1
)

search_space.risk_params["longblock_max_blocks"] = ParameterRange(
    name="longblock_max_blocks",
    param_type="int",
    min_value=1,
    max_value=5,
    default_value=3
)
        """)


def create_custom_search_space():
    """
    Crée un espace de recherche personnalisé pour l'optimisation.
    
    Returns:
        SearchSpace: Espace de recherche personnalisé pour les stratégies de suivi de tendance
    """
    # Créer un nouvel espace de recherche avec un nom et une description
    space = SearchSpace(
        name="custom_trend_following",
        description="Stratégie personnalisée de suivi de tendance avec EMAs et MACD"
    )
    
    # Ajouter l'indicateur EMA avec des plages de paramètres personnalisées
    space.add_indicator(
        IndicatorType.EMA, 
        {
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
        }
    )
    
    # Ajouter l'indicateur SMA avec des plages de paramètres personnalisées
    space.add_indicator(
        IndicatorType.SMA,
        {
            'period': ParameterRange(
                name='period',
                param_type='int',
                min_value=10,
                max_value=200,
                step=5
            ),
            'source': ParameterRange(
                name='source',
                param_type='categorical',
                choices=["close"]
            )
        }
    )
    
    # Ajouter l'indicateur MACD avec des plages de paramètres personnalisées
    space.add_indicator(
        IndicatorType.MACD,
        {
            'fast_period': ParameterRange(
                name='fast_period',
                param_type='int',
                min_value=8,
                max_value=20,
                step=1
            ),
            'slow_period': ParameterRange(
                name='slow_period',
                param_type='int',
                min_value=20,
                max_value=40,
                step=1
            ),
            'signal_period': ParameterRange(
                name='signal_period',
                param_type='int',
                min_value=5,
                max_value=15,
                step=1
            ),
            'source': ParameterRange(
                name='source',
                param_type='categorical',
                choices=["close"]
            )
        }
    )
    
    # Configurer les paramètres de risque
    space.configure_risk(RiskModeType.ATR_BASED)
    print(space)
    return space


class DummyTrial:
    """
    Un trial factice qui imite l'objet Trial d'Optuna pour les tests.
    Utilisé pour simuler les suggestions de paramètres sans Optuna.
    """
    def __init__(self, trial_id):
        self.number = trial_id
        self.params = {}
        self.user_attrs = {}
    
    def suggest_categorical(self, name, choices):
        value = random.choice(choices)
        self.params[name] = value
        return value
    
    def suggest_int(self, name, low, high, step=1, log=False):
        if log:
            value = int(np.exp(random.uniform(np.log(max(1, low)), np.log(high))))
        else:
            value = random.randrange(low, high+1, step)
        self.params[name] = value
        return value
    
    def suggest_float(self, name, low, high, step=None, log=False):
        if log:
            value = np.exp(random.uniform(np.log(max(1e-10, low)), np.log(high)))
        else:
            if step:
                n_steps = int((high - low) / step)
                random_step = random.randint(0, n_steps)
                value = low + random_step * step
            else:
                value = random.uniform(low, high)
        self.params[name] = value
        return value
    
    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


def generate_strategies_example():
    """
    Exemple qui démontre comment créer et utiliser des espaces de recherche
    pour générer des stratégies de trading.
    """
    print("=== Indicateurs Disponibles ===")
    for indicator in get_available_indicators():
        print(f"- {indicator.value}")
    
    print("\n=== Création d'un Espace de Recherche Personnalisé ===")
    custom_space = create_custom_search_space()
    print(f"Espace de recherche personnalisé créé: {custom_space.name}")
    print(f"Description: {custom_space.description}")
    print(f"Indicateurs sélectionnés: {list(custom_space.selected_indicators.keys())}")
    
    print("\n=== Utilisation d'un Espace de Recherche Prédéfini ===")
    trend_space = get_predefined_search_space("trend_following")
    print(f"Utilisation de l'espace prédéfini: {trend_space.name}")
    print(f"Description: {trend_space.description}")
    print(f"Indicateurs sélectionnés: {list(trend_space.selected_indicators.keys())}")
    
    print("\n=== Génération de Stratégies à partir de l'Espace de Recherche ===")
    # Générer 3 stratégies à partir de l'espace personnalisé
    print("\nGénération de 3 stratégies à partir de l'espace personnalisé:")
    for i in range(3):
        # Créer un trial factice (dans une utilisation réelle, ce serait un trial Optuna)
        trial = DummyTrial(i)
        
        # Générer une stratégie à partir du trial et de l'espace de recherche
        strategy = create_strategy_from_trial(trial, custom_space)
        
        print(f"\nStratégie {i+1}:")
        print(f"Nom: {strategy.config.name}")
        print(f"ID: {strategy.config.id}")
        
        # Afficher les indicateurs sélectionnés
        indicators = strategy.config.indicators_manager.list_indicators()
        print(f"Indicateurs sélectionnés ({len(indicators)}):")
        for name, config in indicators.items():
            print(f"  - {name} ({config.type.value})")
            for param_name, param_value in config.params.__dict__.items():
                if not param_name.startswith('_') and param_name != 'offset':
                    print(f"    • {param_name}: {param_value}")
        
        # Afficher les blocs d'entrée
        entry_blocks = strategy.config.blocks_config.entry_blocks
        print(f"Blocs d'Entrée (Long) ({len(entry_blocks)}):")
        for idx, block in enumerate(entry_blocks):
            print(f"  - Bloc {idx+1}: {block.name}")
            for j, condition in enumerate(block.conditions):
                print(f"    • Condition {j+1}: {condition}")
        
        # Afficher les blocs de sortie
        exit_blocks = strategy.config.blocks_config.exit_blocks
        print(f"Blocs de Sortie (Short) ({len(exit_blocks)}):")
        for idx, block in enumerate(exit_blocks):
            print(f"  - Bloc {idx+1}: {block.name}")
            for j, condition in enumerate(block.conditions):
                print(f"    • Condition {j+1}: {condition}")
        
        # Afficher la configuration de risque
        risk_config = strategy.config.risk_config
        print(f"Gestion du Risque: {risk_config.mode.value}")
        for param_name, param_value in risk_config.params.__dict__.items():
            if not param_name.startswith('_'):
                print(f"  • {param_name}: {param_value}")
        
        # Afficher les paramètres suggérés
        print(f"Paramètres suggérés ({len(trial.params)}):")
        for param_name, param_value in trial.params.items():
            print(f"  • {param_name}: {param_value}")
    
    # Exemple d'exportation de l'espace de recherche en JSON
    print("\n=== Exportation de l'Espace de Recherche en JSON ===")
    json_str = custom_space.to_json()
    print(json_str[:500] + "... (truncated)")
    
    # Exemple d'importation d'un espace de recherche à partir de JSON
    print("\n=== Importation d'un Espace de Recherche à partir de JSON ===")
    imported_space = SearchSpace.from_json(json_str)
    print(f"Espace importé: {imported_space.name}")
    print(f"Indicateurs: {list(imported_space.selected_indicators.keys())}")


if __name__ == "__main__":
    generate_strategies_example()