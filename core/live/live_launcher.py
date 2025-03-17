"""
Module principal pour le lancement du trading en direct.
Orchestre tous les composants du système de trading en temps réel.
"""

import os
import json
import logging
import asyncio
import argparse
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import time
import signal

# Import des modules du système de trading
from core.strategy.constructor.constructor import StrategyConstructor
from core.strategy.strategy_manager import StrategyManager
from core.live.live_config import LiveConfig, ExchangeType, LiveTradingMode
from core.live.live_data_manager import LiveDataManager
from core.live.live_backtest import LiveBacktest

# Import des modules d'exchange
from core.live.exchange.exchange_interface import ExchangeInterface
from core.live.exchange.bitget_api import BitgetAPI
from core.live.exchange.binance_api import BinanceAPI
from core.live.live import LiveTrader

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_trading.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("live_launcher")


class LiveLauncher:
    """
    Lanceur principal pour le trading en direct.
    Orchestre l'initialisation, la configuration et l'exécution du trading en temps réel.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialise le lanceur de trading en direct.
        
        Args:
            config_path: Chemin du fichier de configuration (optionnel)
        """
        self.config = None
        self.strategy_manager = StrategyManager()
        self.strategy = None
        self.exchange_api = None
        self.data_manager = None
        self.live_trader = None
        self.live_backtest = None
        self.running = False
        self.last_status_update = None
        
        # Créer les répertoires nécessaires
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data/live", exist_ok=True)
        os.makedirs("results/live", exist_ok=True)
        
        # Charger la configuration si spécifiée
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> bool:
        """
        Charge la configuration depuis un fichier.
        
        Args:
            config_path: Chemin du fichier de configuration
            
        Returns:
            bool: Succès du chargement
        """
        try:
            self.config = LiveConfig.load(config_path)
            logger.info(f"Configuration chargée depuis {config_path}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
            return False
    
    def create_default_config(self, exchange: ExchangeType = ExchangeType.BITGET, symbol: str = "BTCUSDT") -> LiveConfig:
        """
        Crée une configuration par défaut.
        
        Args:
            exchange: Type d'exchange
            symbol: Symbole de la paire
            
        Returns:
            LiveConfig: Configuration par défaut
        """
        from live_config import create_default_config
        
        self.config = create_default_config(exchange, symbol)
        logger.info(f"Configuration par défaut créée pour {symbol} sur {exchange.value}")
        return self.config
    
    def save_config(self, filepath: str = None) -> bool:
        """
        Sauvegarde la configuration dans un fichier.
        
        Args:
            filepath: Chemin du fichier (optionnel)
            
        Returns:
            bool: Succès de la sauvegarde
        """
        if self.config is None:
            logger.error("Aucune configuration à sauvegarder")
            return False
        
        if filepath is None:
            # Créer un nom de fichier par défaut
            filename = f"{self.config.exchange.value}_{self.config.market.symbol}_{datetime.now().strftime('%Y%m%d')}.json"
            filepath = os.path.join("config/live", filename)
        
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Sauvegarder la configuration
        return self.config.save(filepath)
    
    def load_strategy(self, strategy_id: str) -> bool:
        """
        Charge une stratégie depuis son ID.
        
        Args:
            strategy_id: ID de la stratégie
            
        Returns:
            bool: Succès du chargement
        """
        try:
            self.strategy = self.strategy_manager.load_strategy(strategy_id)
            
            if self.strategy is None:
                logger.error(f"Stratégie '{strategy_id}' non trouvée")
                return False
            
            # Mettre à jour la configuration
            if self.config:
                self.config.strategy_id = strategy_id
            
            logger.info(f"Stratégie '{self.strategy.config.name}' chargée")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la stratégie: {str(e)}")
            return False
    
    def initialize_exchange(self) -> bool:
        """
        Initialise l'API de l'exchange selon la configuration.
        
        Returns:
            bool: Succès de l'initialisation
        """
        if self.config is None:
            logger.error("Aucune configuration pour initialiser l'exchange")
            return False
        
        try:
            # Créer l'API de l'exchange approprié
            exchange_type = self.config.exchange
            testnet = self.config.trading_mode != LiveTradingMode.REAL
            
            if exchange_type == ExchangeType.BITGET:
                self.exchange_api = BitgetAPI(
                    api_key=self.config.api_key,
                    api_secret=self.config.api_secret,
                    passphrase=self.config.api_passphrase,
                    testnet=testnet
                )
            elif exchange_type == ExchangeType.BINANCE:
                self.exchange_api = BinanceAPI(
                    api_key=self.config.api_key,
                    api_secret=self.config.api_secret,
                    testnet=testnet
                )
            else:
                logger.error(f"Exchange non supporté: {exchange_type}")
                return False
            
            # Tester la connexion
            account_info = self.exchange_api.get_account_info()
            
            logger.info(f"Connexion à {exchange_type.value} établie")
            logger.info(f"Mode: {'Test' if testnet else 'Production'}")
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'exchange: {str(e)}")
            return False
    
    async def initialize_data_manager(self) -> bool:
        """
        Initialise le gestionnaire de données en utilisant le gestionnaire centralisé.
        Returns:
            bool: Succès de l'initialisation
        """
        if self.config is None or self.exchange_api is None or self.strategy is None:
            logger.error("Configuration, exchange ou stratégie manquante pour initialiser le gestionnaire de données")
            return False
            
        try:
            # Utiliser le gestionnaire centralisé pour les données
            from data.data_manager import get_central_data_manager
            central_manager = get_central_data_manager()
            
            # Récupérer les données nécessaires
            symbol = self.config.market.symbol
            timeframe = self.config.market.timeframe
            exchange = self.config.exchange.value
            
            # Récupérer les données historiques via le gestionnaire centralisé
            df = central_manager.get_or_download_data(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe
            )
            
            if df is None or len(df) < 100:
                logger.warning(f"Données insuffisantes pour {symbol} {timeframe}")
                return False
                
            # Créer le gestionnaire de données en direct avec ces données préchargées
            self.data_manager = LiveDataManager(
                exchange=self.exchange_api,
                config=self.config,
                strategy=self.strategy,
                data_dir="data/live"
            )
            
            # Initialiser avec les données déjà chargées
            await self.data_manager.initialize_with_dataframe(df)
            
            logger.info("Gestionnaire de données initialisé avec données centralisées")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du gestionnaire de données: {str(e)}")
            return False
    
    async def initialize_live_backtest(self) -> bool:
        """
        Initialise le module de backtest en direct.
        
        Returns:
            bool: Succès de l'initialisation
        """
        if self.config is None or self.strategy is None:
            logger.error("Configuration ou stratégie manquante pour initialiser le backtest en direct")
            return False
        
        try:
            # Créer le module de backtest en direct
            self.live_backtest = LiveBacktest(
                strategy=self.strategy,
                config=self.config,
                results_dir="results/live/backtest"
            )
            
            logger.info("Module de backtest en direct initialisé")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du backtest en direct: {str(e)}")
            return False
    
    async def create_live_trader(self) -> bool:
        """
        Crée le trader en direct.
        
        Returns:
            bool: Succès de la création
        """
        if self.config is None or self.strategy is None or self.exchange_api is None:
            logger.error("Configuration, stratégie ou exchange manquante pour créer le trader en direct")
            return False
        
        try:
            # Créer le trader en direct
            self.live_trader = LiveTrader(
                strategy_constructor=self.strategy,
                config=self.config
            )
            
            logger.info("Trader en direct créé")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la création du trader en direct: {str(e)}")
            return False
    
    async def start_trading(self) -> bool:
        """
        Démarre le trading en direct.
        
        Returns:
            bool: Succès du démarrage
        """
        if self.live_trader is None or self.data_manager is None:
            logger.error("Trader ou gestionnaire de données non initialisé")
            return False
        
        try:
            # Démarrer la mise à jour des données
            await self.data_manager.start_updating(self.config.update_interval_seconds)
            
            # Démarrer le trader
            success = await self.live_trader.start()
            
            if success:
                self.running = True
                logger.info(f"Trading en direct démarré pour {self.config.market.symbol} sur {self.config.exchange.value}")
                return True
            else:
                logger.error("Échec du démarrage du trading en direct")
                return False
        except Exception as e:
            logger.error(f"Erreur lors du démarrage du trading: {str(e)}")
            return False
    
    async def stop_trading(self) -> bool:
        """
        Arrête le trading en direct.
        
        Returns:
            bool: Succès de l'arrêt
        """
        try:
            # Arrêter le trader
            if self.live_trader:
                await self.live_trader.stop()
            
            # Arrêter la mise à jour des données
            if self.data_manager:
                await self.data_manager.stop_updating()
            
            self.running = False
            logger.info("Trading en direct arrêté")
            
            # Sauvegarder les résultats
            self.save_trading_results()
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt du trading: {str(e)}")
            return False
    
    async def run_backtest_comparison(self) -> Dict[str, Any]:
        """
        Exécute un backtest de comparaison sur les données récentes.
        
        Returns:
            Dict[str, Any]: Résultats de la comparaison
        """
        if self.live_backtest is None or self.data_manager is None:
            logger.error("Module de backtest ou gestionnaire de données non initialisé")
            return None
        
        try:
            # Préparer les données pour le backtest
            backtest_data = self.data_manager.prepare_backtest_data()
            
            if backtest_data is None or len(backtest_data) < 100:
                logger.warning("Données insuffisantes pour le backtest")
                return None
            
            # Exécuter le backtest
            backtest_results = self.live_backtest.run_backtest(backtest_data)
            
            # Comparer avec les performances en direct
            if self.live_trader and hasattr(self.live_trader, 'get_performance_summary'):
                live_metrics = self.live_trader.get_performance_summary()
                comparison = self.live_backtest.compare_with_live(live_metrics)
                
                logger.info(f"Comparaison backtest/live: Score de cohérence={comparison['consistency_score']:.2f}%")
                return comparison
            
            return backtest_results
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du backtest de comparaison: {str(e)}")
            return None
    
    def save_trading_results(self) -> bool:
        """
        Sauvegarde les résultats du trading en direct.
        
        Returns:
            bool: Succès de la sauvegarde
        """
        if self.live_trader is None:
            logger.warning("Aucun trader pour sauvegarder les résultats")
            return False
        
        try:
            # Créer un répertoire avec horodatage
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("results/live", timestamp)
            
            # Sauvegarder les résultats du trader
            self.live_trader.save_history(output_dir)
            
            # Sauvegarder la configuration
            config_path = os.path.join(output_dir, "live_config.json")
            self.config.save(config_path)
            
            logger.info(f"Résultats du trading sauvegardés dans {output_dir}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats: {str(e)}")
            return False
    
    async def check_status(self) -> Dict[str, Any]:
        """
        Vérifie le statut du trading en direct.
        
        Returns:
            Dict[str, Any]: Informations sur le statut
        """
        status = {
            "timestamp": datetime.now().isoformat(),
            "running": self.running,
            "config": {
                "exchange": self.config.exchange.value if self.config else None,
                "symbol": self.config.market.symbol if self.config else None,
                "strategy_id": self.config.strategy_id if self.config else None,
                "trading_mode": self.config.trading_mode.value if self.config else None
            },
            "data": None,
            "trader": None,
            "performance": None
        }
        
        # Ajouter les informations sur les données
        if self.data_manager:
            status["data"] = self.data_manager.generate_summary()
        
        # Ajouter les informations sur le trader
        if self.live_trader:
            # Récupérer les informations sur les positions
            if hasattr(self.live_trader, 'api') and hasattr(self.live_trader.api, 'get_positions'):
                positions = await self.live_trader.api.get_positions(self.config.market.symbol if self.config else None)
                status["positions"] = positions
            
            # Récupérer les performances
            if hasattr(self.live_trader, 'get_performance_summary'):
                status["performance"] = self.live_trader.get_performance_summary()
        
        self.last_status_update = status
        return status
    
    async def run_periodic_tasks(self) -> None:
        """
        Exécute des tâches périodiques pendant le trading.
        """
        if not self.running:
            return
        
        try:
            # Vérifier le statut toutes les minutes
            await self.check_status()
            
            # Exécuter un backtest de comparaison toutes les heures
            current_time = datetime.now()
            if (self.last_status_update and 
                'timestamp' in self.last_status_update and 
                current_time - datetime.fromisoformat(self.last_status_update['timestamp']) > timedelta(hours=1)):
                await self.run_backtest_comparison()
            
            # Sauvegarder les résultats toutes les heures
            if self.running:
                self.save_trading_results()
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution des tâches périodiques: {str(e)}")
    
    async def main_loop(self) -> None:
        """
        Boucle principale de gestion du trading en direct.
        """
        try:
            # Configurer la gestion des signaux pour l'arrêt propre
            def signal_handler(sig, frame):
                logger.info("Signal d'arrêt reçu")
                asyncio.create_task(self.stop_trading())
            
            # Enregistrer le gestionnaire de signaux
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Boucle principale
            while self.running:
                # Exécuter les tâches périodiques
                await self.run_periodic_tasks()
                
                # Attendre un peu
                await asyncio.sleep(60)  # Vérifier toutes les minutes
                
        except Exception as e:
            logger.error(f"Erreur dans la boucle principale: {str(e)}")
            await self.stop_trading()
    
    async def run(self) -> None:
        """
        Exécute la séquence complète de trading en direct.
        """
        try:
            # 1. Initialiser l'exchange
            if not self.initialize_exchange():
                logger.error("Échec de l'initialisation de l'exchange")
                return
            
            # 2. Initialiser le gestionnaire de données
            if not await self.initialize_data_manager():
                logger.error("Échec de l'initialisation du gestionnaire de données")
                return
            
            # 3. Initialiser le module de backtest en direct
            if not await self.initialize_live_backtest():
                logger.warning("Échec de l'initialisation du module de backtest en direct")
                # Continuer quand même
            
            # 4. Créer le trader en direct
            if not await self.create_live_trader():
                logger.error("Échec de la création du trader en direct")
                return
            
            # 5. Démarrer le trading
            if not await self.start_trading():
                logger.error("Échec du démarrage du trading")
                return
            
            # 6. Exécuter la boucle principale
            await self.main_loop()
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du trading en direct: {str(e)}")
            await self.stop_trading()


async def main():
    """
    Fonction principale pour le lancement du trading en direct.
    """
    # Analyse des arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Lanceur de trading en direct")
    parser.add_argument("--config", type=str, help="Chemin du fichier de configuration")
    parser.add_argument("--strategy", type=str, help="ID de la stratégie")
    parser.add_argument("--exchange", type=str, choices=["bitget", "binance"], default="bitget", help="Type d'exchange")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbole de la paire")
    parser.add_argument("--mode", type=str, choices=["real", "paper", "demo"], default="paper", help="Mode de trading")
    args = parser.parse_args()
    
    # Créer le lanceur
    launcher = LiveLauncher(args.config)
    
    # Si pas de configuration spécifiée, en créer une
    if not launcher.config:
        exchange_type = ExchangeType(args.exchange)
        launcher.create_default_config(exchange_type, args.symbol)
        
        # Définir le mode de trading
        launcher.config.trading_mode = LiveTradingMode(args.mode)
    
    # Charger la stratégie
    if args.strategy:
        if not launcher.load_strategy(args.strategy):
            logger.error(f"Impossible de charger la stratégie {args.strategy}")
            return
    elif launcher.config and launcher.config.strategy_id:
        if not launcher.load_strategy(launcher.config.strategy_id):
            logger.error(f"Impossible de charger la stratégie {launcher.config.strategy_id}")
            return
    else:
        logger.error("Aucune stratégie spécifiée")
        return
    
    # Sauvegarder la configuration
    launcher.save_config()
    
    # Exécuter le trading
    await launcher.run()


if __name__ == "__main__":
    # Exécuter la fonction principale de manière asynchrone
    asyncio.run(main())