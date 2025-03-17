"""
Gestionnaire de données pour le trading en direct.
Récupère et maintient les données historiques et temps réel nécessaires aux stratégies.
S'intègre avec le downloader existant pour obtenir l'historique complet requis.
"""

import os
import pandas as pd
import numpy as np
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import threading
import asyncio

# Import des modules existants
from core.strategy.constructor.constructor import StrategyConstructor
from core.live.exchange.exchange_interface import ExchangeInterface
from core.live.live_config import LiveConfig, ExchangeType

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("live_data_manager")


class LiveDataManager:
    """
    Gestionnaire de données pour le trading en direct.
    Récupère et maintient les données historiques et temps réel.
    """
    
    def __init__(
        self, 
        exchange: ExchangeInterface,
        config: LiveConfig,
        strategy: StrategyConstructor,
        data_dir: str = "data/live"
    ):
        """
        Initialise le gestionnaire de données.
        
        Args:
            exchange: Interface de l'exchange
            config: Configuration du trading en direct
            strategy: Constructeur de stratégie
            data_dir: Répertoire pour stocker les données
        """
        self.exchange = exchange
        self.config = config
        self.strategy = strategy
        self.data_dir = data_dir
        
        # Créer le répertoire de données si nécessaire
        os.makedirs(data_dir, exist_ok=True)
        
        # Données en cache
        self.historical_data = None
        self.latest_kline = None
        self.indicators_data = None
        self.last_update_time = None
        self.required_indicators = self._extract_required_indicators()
        
        # Pour le suivi des mises à jour
        self.update_thread = None
        self.is_running = False
        self._lock = threading.Lock()
    
    def _extract_required_indicators(self) -> Dict[str, int]:
        """
        Extrait les indicateurs requis par la stratégie et leurs périodes.
        
        Returns:
            Dict[str, int]: Dictionnaire des indicateurs et leurs périodes
        """
        required_indicators = {}
        
        # Parcourir les indicateurs de la stratégie
        for name, indicator in self.strategy.config.indicators_manager.list_indicators().items():
            # Récupérer la période de l'indicateur
            period = getattr(indicator.params, 'period', 0)
            
            # Pour MACD, prendre la plus grande période
            if hasattr(indicator.params, 'slow_period'):
                period = max(period, getattr(indicator.params, 'slow_period', 0))
            
            required_indicators[name] = period
        
        return required_indicators
    
    def get_required_lookback(self) -> int:
        """
        Calcule le nombre de points de données nécessaires pour les indicateurs.
        
        Returns:
            int: Nombre de points de données requis
        """
        # Prendre en compte les fenêtres des indicateurs et les décalages
        max_period = 100  # Une valeur par défaut raisonnable
        
        # Si des indicateurs sont définis, calculer le maximum de leurs périodes
        if self.required_indicators:
            max_period = max(self.required_indicators.values(), default=max_period)
        
        # Ajouter une marge de sécurité
        return max(max_period * 3, 500)  # Au moins 500 points de données
    
    def _calculate_start_time(self, lookback_days: int = None) -> int:
        """
        Calcule le timestamp de début pour la récupération des données.
        
        Args:
            lookback_days: Nombre de jours à remonter (si None, utilise la config)
            
        Returns:
            int: Timestamp de début en millisecondes
        """
        if lookback_days is None:
            lookback_days = self.config.data_lookback_days
        
        # Calculer la date de début
        start_time = datetime.now() - timedelta(days=lookback_days)
        
        # Convertir en timestamp
        return int(start_time.timestamp() * 1000)
    
    async def initialize_data(self) -> pd.DataFrame:
        """
        Initialise les données historiques pour la stratégie.
        
        Returns:
            pd.DataFrame: Données historiques avec indicateurs
        """
        logger.info(f"Initialisation des données pour {self.config.market.symbol} sur {self.config.exchange.value}")
        
        # Calculer le nombre de points de données requis
        required_points = self.get_required_lookback()
        
        try:
            # Récupérer les données historiques depuis l'exchange
            historical_data = await self._load_historical_data(required_points)
            
            if historical_data is None or len(historical_data) < 10:
                raise ValueError(f"Données historiques insuffisantes pour {self.config.market.symbol}")
            
            logger.info(f"Données historiques chargées: {len(historical_data)} points")
            
            # Calculer les indicateurs
            data_with_indicators = self.strategy.indicators_calculator.calculate_indicators(historical_data)
            
            # Stocker les données
            self.historical_data = data_with_indicators
            self.last_update_time = datetime.now()
            
            # Enregistrer les données initiales
            self._save_data(data_with_indicators, "initial_data.csv")
            
            return data_with_indicators
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des données: {str(e)}")
            raise
    
    async def initialize_with_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Initialise les données historiques avec un DataFrame préchargé.
        Args:
            dataframe: DataFrame préchargé
        Returns:
            pd.DataFrame: Données avec indicateurs
        """
        logger.info(f"Initialisation des données pour {self.config.market.symbol} avec DataFrame préchargé")
        
        try:
            required_columns = ['timestamp', 'open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in dataframe.columns]
            
            if missing_columns:
                logger.warning(f"Colonnes manquantes dans le DataFrame: {missing_columns}")
                # Tentative d'adaptation
                if 'timestamp' not in dataframe.columns and dataframe.index.name == 'timestamp':
                    dataframe = dataframe.reset_index()
                    
                # Conversion des noms de colonnes en minuscules
                if any(col.lower() in dataframe.columns for col in missing_columns):
                    dataframe.columns = [col.lower() for col in dataframe.columns]
                    missing_columns = [col for col in required_columns if col not in dataframe.columns]
            
            if missing_columns:
                raise ValueError(f"Données incompatibles: colonnes {missing_columns} manquantes")
                
            # Assurer que timestamp est un index datetime
            if 'timestamp' in dataframe.columns:
                if not pd.api.types.is_datetime64_dtype(dataframe['timestamp']):
                    dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])
                dataframe = dataframe.set_index('timestamp').sort_index()
            
            # Calcul des indicateurs
            data_with_indicators = self.strategy.indicators_calculator.calculate_indicators(dataframe)
            self.historical_data = data_with_indicators
            self.last_update_time = datetime.now()
            
            # Sauvegarde pour référence future
            self._save_dataframe(data_with_indicators)
            return data_with_indicators
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation avec DataFrame: {str(e)}")
            raise

    async def _load_historical_data(self, required_points: int) -> pd.DataFrame:
        """
        Charge les données historiques depuis l'exchange ou le downloader.
        
        Args:
            required_points: Nombre de points de données requis
            
        Returns:
            pd.DataFrame: Données historiques
        """
        try:
            # Essayer d'abord avec l'API de l'exchange
            symbol = self.config.market.symbol
            interval = self.config.market.timeframe
            
            # Pour une période plus longue, utiliser le downloader
            if required_points > 1000:
                lookback_days = max(30, self.config.data_lookback_days)
                
                # Utiliser l'exchange pour obtenir les données
                df = self.exchange.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=required_points,
                    lookback_days=lookback_days
                )
                
                # Si les données sont insuffisantes, essayer avec le downloader
                if len(df) < required_points * 0.9:  # Tolérer une petite marge d'erreur
                    logger.warning(f"Données insuffisantes depuis l'API ({len(df)} < {required_points}), " +
                                   f"tentative avec le downloader...")
                    
                    # Essayer avec le downloader du module data
                    from data.data_manager import download_data
                    
                    # Convertir les types d'exchange
                    exchange_name = self.config.exchange.value
                    
                    data = await download_data(
                        exchange=exchange_name,
                        symbol=symbol.replace('USDT', '/USDT'),
                        timeframe=interval,
                        start_date=(datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d'),
                        end_date=datetime.now().strftime('%Y-%m-%d')
                    )
                    
                    if data and hasattr(data, 'dataframe'):
                        df = data.dataframe
                    else:
                        logger.warning("Échec du téléchargement avec le downloader")
            else:
                # Pour une période courte, utiliser directement l'API
                klines = await asyncio.to_thread(
                    self.exchange.get_klines,
                    symbol=symbol,
                    interval=interval,
                    limit=required_points
                )
                
                # Convertir en DataFrame
                if not klines:
                    logger.warning(f"Aucune donnée récupérée pour {symbol} {interval}")
                    return None
                
                df = pd.DataFrame(klines)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Vérifier que les colonnes OHLCV sont présentes
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Colonnes manquantes dans les données: {missing_columns}")
                return None
            
            # Trier par horodatage
            df.sort_values('timestamp', inplace=True)
            
            # Définir l'index
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données historiques: {str(e)}")
            return None
    
    async def update_data(self) -> Optional[pd.DataFrame]:
        """
        Met à jour les données avec les dernières informations du marché.
        
        Returns:
            Optional[pd.DataFrame]: Données mises à jour avec indicateurs
        """
        if self.historical_data is None:
            logger.warning("Les données historiques doivent être initialisées avant de les mettre à jour")
            return None
        
        try:
            # Récupérer la dernière bougie
            latest_klines = await asyncio.to_thread(
                self.exchange.get_klines,
                symbol=self.config.market.symbol,
                interval=self.config.market.timeframe,
                limit=5  # Quelques bougies pour la continuité
            )
            
            if not latest_klines:
                logger.warning("Aucune nouvelle donnée récupérée lors de la mise à jour")
                return self.historical_data
            
            # Convertir en DataFrame
            new_data = pd.DataFrame(latest_klines)
            new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], unit='ms')
            new_data.set_index('timestamp', inplace=True)
            
            # Fusionner avec les données existantes
            merged_data = self._merge_data(self.historical_data, new_data)
            
            # Calculer les indicateurs
            data_with_indicators = self.strategy.indicators_calculator.calculate_indicators(merged_data)
            
            # Mettre à jour les données stockées
            self.historical_data = data_with_indicators
            self.latest_kline = new_data.iloc[-1] if len(new_data) > 0 else None
            self.last_update_time = datetime.now()
            
            # Enregistrer les dernières données
            self._save_data(data_with_indicators.tail(100), "latest_data.csv")
            
            return data_with_indicators
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des données: {str(e)}")
            return self.historical_data
    
    def _merge_data(self, historical_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Fusionne les nouvelles données avec les données historiques.
        
        Args:
            historical_data: Données historiques
            new_data: Nouvelles données
            
        Returns:
            pd.DataFrame: Données fusionnées
        """
        # Combiner les deux DataFrames
        combined_data = pd.concat([historical_data, new_data])
        
        # Supprimer les doublons en gardant la dernière occurrence
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        
        # Trier par horodatage
        combined_data.sort_index(inplace=True)
        
        # Limiter la taille du DataFrame
        max_points = self.get_required_lookback() + 100  # Garder une marge
        if len(combined_data) > max_points:
            combined_data = combined_data.iloc[-max_points:]
        
        return combined_data
    
    def _save_data(self, data: pd.DataFrame, filename: str) -> None:
        """
        Sauvegarde les données dans un fichier CSV.
        
        Args:
            data: Données à sauvegarder
            filename: Nom du fichier
        """
        try:
            # Créer un répertoire spécifique pour les données de ce symbole
            symbol_dir = os.path.join(self.data_dir, self.config.market.symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            
            # Chemin complet du fichier
            filepath = os.path.join(symbol_dir, filename)
            
            # Sauvegarder le DataFrame
            data.to_csv(filepath)
            
            # Sauvegarder aussi les métadonnées
            metadata = {
                "symbol": self.config.market.symbol,
                "timeframe": self.config.market.timeframe,
                "exchange": self.config.exchange.value,
                "last_update": datetime.now().isoformat(),
                "points": len(data),
                "start_date": data.index[0].strftime('%Y-%m-%d %H:%M:%S') if len(data) > 0 else None,
                "end_date": data.index[-1].strftime('%Y-%m-%d %H:%M:%S') if len(data) > 0 else None
            }
            
            metadata_path = os.path.join(symbol_dir, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données: {str(e)}")
    
    def get_current_data(self) -> pd.DataFrame:
        """
        Retourne les données actuelles avec indicateurs.
        
        Returns:
            pd.DataFrame: Données avec indicateurs
        """
        return self.historical_data if self.historical_data is not None else pd.DataFrame()
    
    def get_current_price(self) -> float:
        """
        Retourne le prix actuel du marché.
        
        Returns:
            float: Prix actuel
        """
        try:
            # Si nous avons une dernière bougie, utiliser son prix de clôture
            if self.latest_kline is not None:
                return float(self.latest_kline['close'])
            
            # Sinon, obtenir le prix actuel du ticker
            ticker = self.exchange.get_ticker(self.config.market.symbol)
            return float(ticker['last_price'])
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du prix actuel: {str(e)}")
            
            # En cas d'erreur, utiliser le dernier prix connu
            if self.historical_data is not None and len(self.historical_data) > 0:
                return float(self.historical_data['close'].iloc[-1])
            
            return 0.0
    
    async def start_updating(self, interval_seconds: int = None) -> None:
        """
        Démarre la mise à jour périodique des données.
        
        Args:
            interval_seconds: Intervalle de mise à jour en secondes
        """
        if self.is_running:
            logger.warning("Le processus de mise à jour est déjà en cours")
            return
        
        if interval_seconds is None:
            interval_seconds = self.config.update_interval_seconds
        
        self.is_running = True
        
        logger.info(f"Démarrage des mises à jour de données toutes les {interval_seconds} secondes")
        
        # Créer une tâche asyncio pour la mise à jour périodique
        asyncio.create_task(self._update_loop(interval_seconds))
    
    async def _update_loop(self, interval_seconds: int) -> None:
        """
        Boucle de mise à jour périodique des données.
        
        Args:
            interval_seconds: Intervalle de mise à jour en secondes
        """
        while self.is_running:
            try:
                await self.update_data()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Erreur dans la boucle de mise à jour: {str(e)}")
                await asyncio.sleep(interval_seconds * 2)  # Attendre plus longtemps en cas d'erreur
    
    async def stop_updating(self) -> None:
        """Arrête la mise à jour périodique des données."""
        self.is_running = False
        logger.info("Arrêt des mises à jour de données")
    
    def prepare_backtest_data(self, lookback_periods: int = 1000) -> pd.DataFrame:
        """
        Prépare les données pour un backtest sur l'historique récent.
        
        Args:
            lookback_periods: Nombre de périodes historiques à utiliser
            
        Returns:
            pd.DataFrame: Données pour le backtest
        """
        if self.historical_data is None or len(self.historical_data) < lookback_periods:
            logger.warning("Données insuffisantes pour le backtest")
            return None
        
        # Utiliser les données historiques existantes
        backtest_data = self.historical_data.iloc[-lookback_periods:]
        
        # Sauvegarder ces données pour référence
        self._save_data(backtest_data, "backtest_data.csv")
        
        return backtest_data
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Génère un résumé des données actuelles.
        
        Returns:
            Dict: Résumé des données
        """
        summary = {
            "symbol": self.config.market.symbol,
            "timeframe": self.config.market.timeframe,
            "exchange": self.config.exchange.value,
            "data_points": 0,
            "start_date": None,
            "end_date": None,
            "current_price": 0.0,
            "required_indicators": self.required_indicators,
            "required_lookback": self.get_required_lookback(),
            "last_update": None
        }
        
        if self.historical_data is not None:
            summary.update({
                "data_points": len(self.historical_data),
                "start_date": self.historical_data.index[0].strftime('%Y-%m-%d %H:%M:%S') if len(self.historical_data) > 0 else None,
                "end_date": self.historical_data.index[-1].strftime('%Y-%m-%d %H:%M:%S') if len(self.historical_data) > 0 else None,
                "current_price": self.get_current_price(),
                "last_update": self.last_update_time.isoformat() if self.last_update_time else None
            })
        
        return summary