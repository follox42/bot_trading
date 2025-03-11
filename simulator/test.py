"""
Fichier de test pour le système de trading avec configuration flexible
et gestionnaire d'études intégré.

Ce fichier démontre le flux de travail complet :
1. Création de configuration flexible
2. Création d'études avec le gestionnaire
3. Optimisation de stratégies
4. Backtest des stratégies optimisées
5. Comparaison des résultats
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import logging

# Import des modules du système de trading
from config import (
    FlexibleTradingConfig, create_flexible_default_config,
    MarginMode, TradingMode, RiskModeConfig
)
from study_manager import IntegratedStudyManager
from strategy_optimizer import IntegratedStrategyOptimizer
from risk import RiskMode
from indicators import IndicatorType

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_trading_system.log', mode='a'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('test_trading_system')

def create_test_data(n_samples=5000, output_file="test_data.csv"):
    """
    Crée des données de test pour l'optimisation.
    
    Args:
        n_samples: Nombre de points de données
        output_file: Fichier de sortie
    
    Returns:
        str: Chemin du fichier de données
    """
    logger.info(f"Création de données de test avec {n_samples} points")
    
    # Paramètres de simulation
    np.random.seed(42)
    initial_price = 1000.0
    vol = 0.01  # Volatilité
    
    # Générer une série de prix avec tendance et oscillations
    times = np.arange(n_samples)
    # Tendance à long terme (composante sinusoïdale)
    trend = 0.1 * np.sin(times * 2 * np.pi / n_samples) * initial_price
    # Oscillations à moyen terme
    oscil_med = 0.05 * np.sin(times * 2 * np.pi / 500) * initial_price
    # Oscillations à court terme
    oscil_short = 0.02 * np.sin(times * 2 * np.pi / 100) * initial_price
    # Bruit aléatoire
    noise = np.random.normal(0, vol, n_samples) * initial_price
    
    # Combine all components
    price_series = initial_price + trend + oscil_med + oscil_short + np.cumsum(noise) * 0.1
    
    # Calculer OHLC
    opens = price_series.copy()
    closes = np.roll(price_series, -1)  # Décalage d'un intervalle
    highs = np.maximum(opens, closes) * (1 + np.random.uniform(0, 0.005, n_samples))
    lows = np.minimum(opens, closes) * (1 - np.random.uniform(0, 0.005, n_samples))
    volumes = np.random.lognormal(10, 1, n_samples)
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='1h'),
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes[:-1].tolist() + [price_series[-1]],  # Ajuster la dernière valeur
        'volume': volumes
    })
    
    # Enregistrer le DataFrame au format CSV
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    df.to_csv(output_file, index=False)
    
    logger.info(f"Données de test enregistrées dans {output_file}")
    
    # Afficher un aperçu des données
    plt.figure(figsize=(10, 6))
    plt.plot(df['close'])
    plt.title("Prix de clôture simulés")
    plt.xlabel("Temps")
    plt.ylabel("Prix")
    plt.grid(True)
    plt.savefig("test_data_chart.png")
    plt.close()
    
    return output_file

def create_custom_trading_config():
    """
    Crée une configuration de trading personnalisée avec des paramètres flexibles.
    
    Returns:
        FlexibleTradingConfig: Configuration de trading personnalisée
    """
    logger.info("Création d'une configuration de trading personnalisée")
    
    # Partir de la configuration par défaut
    config = create_flexible_default_config()
    
    # Personnalisation des indicateurs
    # Ajouter ou modifier les indicateurs selon les besoins
    for ind_type in IndicatorType:
        if ind_type == IndicatorType.EMA:
            config.available_indicators[ind_type.value].min_period = 3
            config.available_indicators[ind_type.value].max_period = 200
            config.available_indicators[ind_type.value].step = 5
        elif ind_type == IndicatorType.RSI:
            config.available_indicators[ind_type.value].min_period = 5
            config.available_indicators[ind_type.value].max_period = 30
            config.available_indicators[ind_type.value].step = 1
    
    # Configuration du risque
    # Personnaliser les plages pour chaque mode de risque
    config.risk_config.mode_configs[RiskMode.FIXED] = RiskModeConfig(
        fixed_position_range=(0.05, 0.5),     # 5% à 50% du capital
        fixed_sl_range=(0.01, 0.1),           # 1% à 10% de stop loss
        fixed_tp_range=(0.02, 0.3)            # 2% à 30% de take profit
    )
    
    config.risk_config.mode_configs[RiskMode.ATR_BASED] = RiskModeConfig(
        atr_period_range=(7, 21),             # Période ATR de 7 à 21
        atr_multiplier_range=(1.0, 3.0)       # Multiplicateur ATR de 1.0 à 3.0
    )
    
    config.risk_config.mode_configs[RiskMode.VOLATILITY_BASED] = RiskModeConfig(
        vol_period_range=(10, 30),            # Période de volatilité de 10 à 30
        vol_multiplier_range=(1.0, 3.0)       # Multiplicateur de volatilité de 1.0 à 3.0
    )
    
    # Plages de levier et modes de trading/marge
    config.sim_config.leverage_range = (1, 100)       # Levier de 1x à 100x
    
    # Configuration de la structure de stratégie
    config.strategy_structure.max_blocks = 3           # Maximum 3 blocs par stratégie
    config.strategy_structure.max_conditions_per_block = 3  # Maximum 3 conditions par bloc
    config.strategy_structure.cross_signals_probability = 0.4  # 40% de probabilité d'utiliser des signaux de croisement
    
    logger.info("Configuration personnalisée créée avec succès")
    return config

def test_study_creation(study_manager, data_file):
    """
    Teste la création d'une étude avec une configuration personnalisée.
    
    Args:
        study_manager: Gestionnaire d'études
        data_file: Fichier de données
        
    Returns:
        str: Nom de l'étude créée
    """
    # Création d'une configuration personnalisée
    trading_config = create_custom_trading_config()
    
    # Métadonnées de l'étude
    study_name = f"test_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metadata = {
        "asset": "TEST/USD",
        "timeframe": "1h",
        "exchange": "test",
        "description": "Étude de test avec configuration flexible",
        "data_file": data_file
    }
    
    # Création de l'étude
    success = study_manager.create_study(study_name, metadata, trading_config)
    
    if success:
        logger.info(f"Étude '{study_name}' créée avec succès")
    else:
        logger.error(f"Échec de la création de l'étude '{study_name}'")
        sys.exit(1)
    
    return study_name

def test_optimization(study_manager, optimizer, study_name, data_file):
    """
    Teste l'optimisation d'une stratégie.
    
    Args:
        study_manager: Gestionnaire d'études
        optimizer: Optimiseur de stratégies
        study_name: Nom de l'étude
        data_file: Fichier de données
    """
    # Configuration d'optimisation personnalisée
    optimization_config = {
        'n_trials': 20,                 # Nombre d'essais (réduit pour le test)
        'n_jobs': 2,                    # Nombre de workers parallèles
        'timeout': 300,                 # Timeout de 5 minutes
        'score_weights': {              # Poids personnalisés pour le score
            'roi': 3.0,                 # Rendement (plus important)
            'win_rate': 1.5,            # Taux de réussite
            'max_drawdown': 2.5,        # Drawdown maximum (très important)
            'profit_factor': 2.0,       # Facteur de profit
            'total_trades': 0.5,        # Nombre de trades (moins important)
            'avg_profit': 1.0           # Profit moyen par trade
        },
        'min_trades': 5                 # Minimum 5 trades pour un essai valide
    }
    
    # Préparation de l'optimisation
    optimizer.prepare_optimization(study_name, optimization_config)
    
    # Exécution de l'optimisation
    logger.info(f"Démarrage de l'optimisation pour l'étude '{study_name}'")
    success = optimizer.run_optimization(study_name, data_file)
    
    if success:
        logger.info(f"Optimisation de l'étude '{study_name}' terminée avec succès")
    else:
        logger.error(f"Échec de l'optimisation de l'étude '{study_name}'")

def test_backtest(study_manager, study_name, data_file):
    """
    Teste le backtest des stratégies optimisées.
    
    Args:
        study_manager: Gestionnaire d'études
        study_name: Nom de l'étude
        data_file: Fichier de données
    """
    # Charger les données
    data = pd.read_csv(data_file)
    
    # Liste des stratégies de l'étude
    strategies = study_manager.list_strategies(study_name)
    
    if not strategies:
        logger.error(f"Aucune stratégie trouvée pour l'étude '{study_name}'")
        return
    
    # Exécuter le backtest pour chaque stratégie
    for strategy in strategies[:3]:  # Tester seulement les 3 meilleures
        rank = strategy['rank']
        logger.info(f"Backtest de la stratégie {rank} de l'étude '{study_name}'")
        
        results = study_manager.run_backtest(study_name, rank, data)
        
        if results:
            perf = results.get('performance', {})
            logger.info(f"Résultats de backtest pour la stratégie {rank}:")
            logger.info(f"  ROI: {perf.get('roi_pct', 0):.2f}%")
            logger.info(f"  Win Rate: {perf.get('win_rate_pct', 0):.2f}%")
            logger.info(f"  Trades: {perf.get('total_trades', 0)}")
            logger.info(f"  Max Drawdown: {perf.get('max_drawdown_pct', 0):.2f}%")
            logger.info(f"  Profit Factor: {perf.get('profit_factor', 0):.2f}")
        else:
            logger.error(f"Échec du backtest pour la stratégie {rank}")

def test_strategy_comparison(study_manager, study_name):
    """
    Teste la comparaison des stratégies optimisées.
    
    Args:
        study_manager: Gestionnaire d'études
        study_name: Nom de l'étude
    """
    # Liste des stratégies de l'étude
    strategies = study_manager.list_strategies(study_name)
    
    if len(strategies) < 2:
        logger.error(f"Pas assez de stratégies pour une comparaison dans l'étude '{study_name}'")
        return
    
    # Sélectionner les stratégies à comparer
    strategy_ranks = [s['rank'] for s in strategies[:3]]  # Les 3 meilleures
    
    logger.info(f"Comparaison des stratégies {strategy_ranks} de l'étude '{study_name}'")
    
    # Exécuter la comparaison
    comparison = study_manager.compare_strategies(study_name, strategy_ranks)
    
    if comparison:
        logger.info(f"Comparaison des stratégies terminée avec succès")
        
        # Afficher les résultats de la comparaison
        for strategy in comparison.get('strategies', []):
            rank = strategy.get('rank')
            perf = strategy.get('performance', {})
            logger.info(f"Stratégie {rank}:")
            logger.info(f"  ROI: {perf.get('roi_pct', 0):.2f}%")
            logger.info(f"  Win Rate: {perf.get('win_rate_pct', 0):.2f}%")
    else:
        logger.error(f"Échec de la comparaison des stratégies")

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Test du système de trading")
    parser.add_argument("--data", type=str, help="Fichier de données (si non spécifié, des données de test seront générées)")
    parser.add_argument("--study-dir", type=str, default="studies", help="Répertoire des études")
    args = parser.parse_args()
    
    # Création des données de test si nécessaire
    data_file = args.data if args.data else create_test_data(n_samples=2000, output_file="data/test_data.csv")
    
    # Initialisation du gestionnaire d'études
    study_manager = IntegratedStudyManager(base_dir=args.study_dir)
    
    # Initialisation de l'optimiseur
    optimizer = IntegratedStrategyOptimizer(study_manager)
    
    # Test de création d'étude
    study_name = test_study_creation(study_manager, data_file)
    
    # Test d'optimisation
    test_optimization(study_manager, optimizer, study_name, data_file)
    
    # Test de backtest
    test_backtest(study_manager, study_name, data_file)
    
    # Test de comparaison de stratégies
    test_strategy_comparison(study_manager, study_name)
    
    # Listage des études
    logger.info("Liste des études disponibles:")
    studies = study_manager.list_studies()
    for study in studies:
        logger.info(f"  {study['name']} - {study['asset']} ({study['status']}) - {study['strategies_count']} stratégies")
    
    logger.info("Tests terminés avec succès!")

if __name__ == "__main__":
    main()