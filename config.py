# Configuration globale pour l'application Trading Nexus
import os
from datetime import datetime

# Paramètres de l'application
APP_NAME = "Trading Nexus"
APP_VERSION = "1.0.0 Alpha"
DEBUG_MODE = True

# Paramètres du système de log
LOG_DIR = os.path.join(os.getcwd(), "logs")
LOG_CONSOLE_OUTPUT = True

# Paramètres de l'interface
UI_THEME = "CYBORG"  # Thème Dash Bootstrap Components
UI_REFRESH_INTERVAL = 1000  # Intervalle de rafraîchissement en ms

# Constantes du système
SYSTEM_START_TIME = datetime.now()

# Configuration des API de trading (à remplacer par vos clés)
API_KEYS = {
    "binance": {
        "api_key": os.environ.get("BINANCE_API_KEY", ""),
        "api_secret": os.environ.get("BINANCE_API_SECRET", "")
    },
    "bitget": {
        "api_key": os.environ.get("BITGET_API_KEY", ""),
        "api_secret": os.environ.get("BITGET_SECRET", ""),
        "passphrase": os.environ.get("BITGET_PASSPHRASE", "")
    }
}

# Définir les répertoires
def setup_directories():
    """Crée les répertoires nécessaires s'ils n'existent pas déjà."""
    directories = [
        LOG_DIR,
        os.path.join(LOG_DIR, "strategy"),
        os.path.join(LOG_DIR, "data"),
        os.path.join(LOG_DIR, "optimization"),
        os.path.join(LOG_DIR, "system"),
        os.path.join(LOG_DIR, "ui"),
        os.path.join(LOG_DIR, "api"),
        os.path.join(LOG_DIR, "backtest"),
        os.path.join(LOG_DIR, "live_trading")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Séquence de démarrage pour l'animation d'intro
BOOT_SEQUENCE = [
    "INITIALIZING TRADING SYSTEM...",
    "LOADING CORE MODULES...",
    "[ OK ] Data module loaded",
    "[ OK ] Optimization engine loaded",
    "[ OK ] Strategy framework loaded",
    "[ OK ] Backtest engine loaded",
    "CHECKING MARKET CONNECTIONS...",
    "[ OK ] Binance API connected",
    "[ OK ] Bitget API connected",
    "[ OK ] Database connection established",
    "INITIALIZING TRADING ALGORITHMS...",
    "[ OK ] All systems operational",
    f"TRADING NEXUS v{APP_VERSION} READY"
]