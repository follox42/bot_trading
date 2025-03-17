"""
Exemple d'utilisation du système refactorisé avec séparation claire des responsabilités.
"""

import os
import pandas as pd

# --------------------
# Gestion des études
# --------------------
from core.study.study_manager import StudyManager

# Initialiser le gestionnaire d'études
study_manager = StudyManager(base_dir="studies")

# Créer une nouvelle étude
study_name = study_manager.create_study(
    name="BTC_Strategy",
    description="Étude de stratégie sur BTC/USDT",
    timeframe="1h",
    asset="BTC/USDT",
    tags=["crypto", "bitcoin"]
)

# Obtenir le chemin vers le répertoire de l'étude
study_path = study_manager.get_study_path(study_name)

# --------------------
# Gestion des données
# --------------------
from data.data_manager import get_data_manager

# Initialiser le gestionnaire de données
data_manager = get_data_manager()

# Télécharger des données pour l'étude
data = data_manager.download_data(
    exchange="bitget",
    symbol="BTC/USDT",
    timeframe="1h",
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# Associer les données à l'étude
data_manager.associate_study_with_data(
    study_name=study_name,
    exchange="bitget",
    symbol="BTC/USDT",
    timeframe="1h"
)

# Charger les données associées à l'étude
df = data_manager.load_study_data(study_name)

# --------------------
# Gestion des stratégies
# --------------------
from core.strategy.strategy_manager import create_strategy_manager_for_study

# Créer un gestionnaire de stratégies pour l'étude
strategy_manager = create_strategy_manager_for_study(study_path)

# Créer une stratégie
constructor = strategy_manager.create_strategy(
    name="MA Crossover",
    description="Stratégie de croisement de moyennes mobiles",
    indicators_preset="trend_following"
)

# Configurer la stratégie
constructor.add_indicator("EMA", "fast_ema", {"period": 10})
constructor.add_indicator("EMA", "slow_ema", {"period": 20})

# Créer des conditions pour les signaux d'entrée
constructor.add_entry_condition(
    left_indicator="fast_ema",
    operator="crosses_above",
    right_indicator="slow_ema"
)

# Créer des conditions pour les signaux de sortie
constructor.add_exit_condition(
    left_indicator="fast_ema",
    operator="crosses_below",
    right_indicator="slow_ema"
)

# Sauvegarder la stratégie
strategy_id = strategy_manager.save_strategy()

# Exécuter un backtest
simulation_results = strategy_manager.run_simulation(df)

# Sauvegarder les résultats du backtest
backtest_id = strategy_manager.save_backtest_results()

# Générer un rapport de performance
performance_report = strategy_manager.generate_performance_report()

# --------------------
# Optimisation de stratégies
# --------------------
from core.optimization.parallel_optimizer import create_optimizer, OptimizationConfig
from core.optimization.search_config import get_predefined_search_space

# Configurer l'optimisation
optimization_config = OptimizationConfig(
    n_trials=100,
    search_space=get_predefined_search_space("trend_following"),
    scoring_formula="standard",
    n_jobs=4  # Utiliser 4 processus en parallèle
)

# Créer l'optimiseur
optimizer = create_optimizer(optimization_config)

# Lancer l'optimisation
success, results = optimizer.run_optimization(study_path, df)

if success:
    print(f"Optimisation terminée avec succès. Meilleur score: {results['best_score']}")
    
    # Charger la meilleure stratégie
    best_strategy_id = results['best_trials'][0]['strategy_id']
    strategy_manager.load_strategy(best_strategy_id)
    
    # Exécuter un backtest avec la meilleure stratégie
    simulation_results = strategy_manager.run_simulation(df)
    
    # Sauvegarder les résultats
    backtest_id = strategy_manager.save_backtest_results("optimized_backtest")
else:
    print("L'optimisation a échoué")