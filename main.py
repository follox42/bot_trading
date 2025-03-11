import threading
import time
import sys
import os

# Ajout du répertoire courant au PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports des modules de l'application
import config
from logger.logger import CentralizedLogger, LogLevel, LoggerType
from ui.app import create_app


def simulate_system_activity(central_logger):
    """Simule l'activité du système pour générer des logs variés"""
    system_logger = central_logger.get_logger("main", LoggerType.SYSTEM)
    
    actions = [
        ("Vérification des données BTC/USDT", LogLevel.INFO),
        ("Mise à jour des prix en direct", LogLevel.INFO),
        ("Signal d'achat détecté sur ETH/USDT", LogLevel.INFO),
        ("Ordre d'achat exécuté: 0.5 ETH à 3450 USDT", LogLevel.INFO),
        ("Latence élevée détectée avec l'API Binance", LogLevel.WARNING),
        ("Échec de connexion à l'API Bitget", LogLevel.ERROR),
        ("Stratégie RSI Reversion initialisée", LogLevel.INFO),
        ("Backtest terminé avec succès: 15% ROI", LogLevel.INFO),
        ("Téléchargement des données historiques terminé", LogLevel.INFO),
        ("Position fermée: profit +2.3%", LogLevel.INFO),
        ("Mise à jour du système disponible", LogLevel.WARNING),
        ("Opération d'optimisation démarrée", LogLevel.INFO),
        ("Mémoire système faible: 85% utilisée", LogLevel.WARNING),
        ("Nouvelle stratégie créée: MACD Cross", LogLevel.INFO)
    ]
    
    # Choix aléatoire d'une action
    import random
    action, level = random.choice(actions)
    
    # Logger l'action dans le logger approprié
    if "Stratégie" in action or "achat" in action or "vente" in action:
        strat_logger = central_logger.get_strategy_logger("auto")
        strat_logger.log(level.value, action)
    elif "données" in action or "Téléchargement" in action:
        data_logger = central_logger.get_data_logger("downloader")
        data_logger.log(level.value, action)
    elif "Backtest" in action:
        backtest_logger = central_logger.get_backtest_logger("test_runner")
        backtest_logger.log(level.value, action)
    else:
        system_logger.log(level.value, action)

def start_simulation_thread(central_logger):
    """Démarre un thread qui simule l'activité du système"""
    def run_simulation():
        while True:
            simulate_system_activity(central_logger)
            time.sleep(15)  # Une activité toutes les 15 secondes
    
    simulation_thread = threading.Thread(target=run_simulation, daemon=True)
    simulation_thread.start()
    
    system_logger = central_logger.get_logger("main", LoggerType.SYSTEM)
    system_logger.info("Thread de simulation démarré")

def main():
    """Fonction principale de l'application"""
    # Créer les répertoires nécessaires
    config.setup_directories()
    
    # Initialiser le logger centralisé
    central_logger = CentralizedLogger(
        base_dir=config.LOG_DIR, 
        console_output=config.LOG_CONSOLE_OUTPUT
    )
    
    # Obtenir le logger système
    system_logger = central_logger.get_logger("main", LoggerType.SYSTEM)
    system_logger.info('=== DÉMARRAGE DU SYSTÈME DE TRADING NEXUS ===')
    
    # Créer et lancer l'application Dash
    app = create_app(central_logger)
    
    system_logger.info(f'Serveur Dash démarré sur http://127.0.0.1:8050/')
    
    # Lancer l'application
    app.run_server(debug=config.DEBUG_MODE)

if __name__ == "__main__":
    main()