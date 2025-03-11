import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_performance_data(days=7, start_value=10000, volatility=0.01):
    """
    Génère des données de performance simulées pour l'affichage
    
    Args:
        days: Nombre de jours de données
        start_value: Valeur initiale de l'équité
        volatility: Volatilité des rendements
        
    Returns:
        DataFrame avec les dates, valeurs et équité
    """
    # Données fixes pour le débogage
    data = {
        'date': ['03/03', '04/03', '05/03', '06/03', '07/03', '08/03', '09/03'],
        'value': [0.0, 0.5, -0.2, 1.1, 2.8, 2.2, 3.5],
        'equity': [10000, 10050, 10030, 10141, 10421, 10361, 10511]
    }
    
    df = pd.DataFrame(data)
    
    print("Performance data generated:", df)
    
    return df

def generate_strategy_data(num_strategies=3):
    """
    Génère des données de stratégie simulées
    
    Args:
        num_strategies: Nombre de stratégies à générer
        
    Returns:
        Liste de dictionnaires contenant les informations de stratégie
    """
    strategy_names = ["EMA Cross", "RSI Reversion", "BB Squeeze", "MACD Divergence", 
                     "Ichimoku Cloud", "Pivot Point Reversal", "Triple Moving Average"]
    
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT", "DOT/USDT"]
    
    strategy_types = ["trend", "reversion", "volatility", "momentum"]
    
    statuses = ["active", "paused", "in_development"]
    
    strategies = []
    
    for i in range(num_strategies):
        # Générer une performance aléatoire biaisée vers le positif
        performance = np.random.normal(2.0, 3.0)  # Moyenne de 2%, écart-type de 3%
        
        strategy = {
            "id": i + 1,
            "name": np.random.choice(strategy_names),
            "symbol": np.random.choice(symbols),
            "performance": round(performance, 1),
            "status": np.random.choice(statuses, p=[0.7, 0.2, 0.1]),  # Plus de chances d'être active
            "type": np.random.choice(strategy_types)
        }
        
        strategies.append(strategy)
    
    return strategies

def generate_log_entries(num_entries=7):
    """
    Génère des entrées de log simulées
    
    Args:
        num_entries: Nombre d'entrées à générer
        
    Returns:
        Liste de dictionnaires contenant les entrées de log
    """
    log_messages = [
        {"level": "INFO", "message": "Démarrage du système de trading"},
        {"level": "INFO", "message": "Chargement des données BTC/USDT"},
        {"level": "WARNING", "message": "Latence élevée avec Binance API"},
        {"level": "INFO", "message": "Stratégie EMA Cross exécutée avec succès"},
        {"level": "ERROR", "message": "Échec de connexion à Bitget API"},
        {"level": "INFO", "message": "Nouvelle version disponible v1.0.1"},
        {"level": "INFO", "message": "Ordre BTC/USDT exécuté à 58245.50"},
        {"level": "INFO", "message": "Initialisation de la stratégie RSI"},
        {"level": "WARNING", "message": "Mémoire système faible: 85% utilisée"},
        {"level": "INFO", "message": "Téléchargement des données historiques terminé"},
        {"level": "INFO", "message": "Position fermée: profit +2.3%"},
        {"level": "ERROR", "message": "Échec du calcul de l'indicateur MACD"},
        {"level": "INFO", "message": "Backtest terminé avec succès: 15% ROI"},
        {"level": "WARNING", "message": "Signal contradictoire détecté"}
    ]
    
    # Heure actuelle
    current_time = datetime.now()
    
    # Générer les logs
    logs = []
    
    for i in range(num_entries):
        # Sélectionner un message aléatoire
        log_entry = np.random.choice(log_messages)
        
        # Générer une heure aléatoire dans les 60 dernières minutes
        log_time = current_time - timedelta(minutes=np.random.randint(1, 60))
        
        logs.append({
            "timestamp": log_time.strftime("%H:%M:%S"),
            "level": log_entry["level"],
            "message": log_entry["message"]
        })
    
    # Trier par heure décroissante (plus récent en premier)
    logs.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return logs

def generate_system_stats():
    """
    Génère des statistiques système simulées
    
    Returns:
        Dictionnaire avec les statistiques
    """
    return {
        "activeStrategies": np.random.randint(1, 5),
        "totalBacktests": np.random.randint(30, 100),
        "runningOptimizations": np.random.randint(0, 3),
        "dataLastUpdate": datetime.now().strftime("%H:%M:%S")
    }