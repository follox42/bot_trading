"""
Utilitaire pour télécharger les données historiques requises par une stratégie.
S'intègre avec le downloader existant et récupère suffisamment de données
pour satisfaire les besoins en indicateurs de la stratégie.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import asyncio

# Import des modules existants
from core.strategy.constructor import StrategyConstructor
from core.data.downloader import download_data, load_data
from live_config import LiveConfig, ExchangeType
from exchange.exchange_interface import ExchangeInterface

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("history_fetcher")


class HistoryFetcher:
    """
    Télécharge et prépare les données historiques nécessaires pour une stratégie.
    """
    
    def __init__(
        self,
        strategy: StrategyConstructor,
        config: LiveConfig,
        exchange_api: Optional[ExchangeInterface] = None,
        output_dir: str = "data/live"
    ):
        """
        Initialise le récupérateur d'historique.
        
        Args:
            strategy: Constructeur de stratégie
            config: Configuration du trading en direct
            exchange_api: API de l'exchange (optionnel)
            output_dir: Répertoire de sortie pour les données
        """
        self.strategy = strategy
        self.config = config
        self.exchange_api = exchange_api
        self.output_dir = output_dir
        
        # Extraire les besoins en données de la stratégie
        self.required_lookback = self._calculate_required_lookback()
        
        # Créer le répertoire de sortie
        os.makedirs(output_dir, exist_ok=True)
    
    def _calculate_required_lookback(self) -> int:
        """
        Calcule le nombre de points de données nécessaires pour la stratégie.
        
        Returns:
            int: Nombre de points de données requis
        """
        # Récupérer les indicateurs de la stratégie
        indicators = self.strategy.config.indicators_manager.list_indicators()
        
        # Trouver la plus grande période parmi les indicateurs
        max_period = 100  # Valeur par défaut raisonnable
        
        for name, indicator in indicators.items():
            # Récupérer la période de l'indicateur
            period = getattr(indicator.params, 'period', 0)
            
            # Pour MACD, prendre la plus grande période
            if hasattr(indicator.params, 'slow_period'):
                period = max(period, getattr(indicator.params, 'slow_period', 0))
            
            # Mettre à jour la période maximale
            max_period = max(max_period, period)
        
        # Ajouter une marge de sécurité
        return max(max_period * 3, 500)  # Au moins 500 points de données
    
    def _calculate_lookback_days(self, timeframe: str) -> int:
        """
        Calcule le nombre de jours nécessaires pour obtenir suffisamment de données.
        
        Args:
            timeframe: Intervalle de temps
            
        Returns:
            int: Nombre de jours à remonter
        """
        # Convertir le timeframe en minutes
        minutes_map = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "8h": 480,
            "12h": 720,
            "1d": 1440,
            "3d": 4320,
            "1w": 10080
        }
        
        minutes = minutes_map.get(timeframe, 60)  # Par défaut 1h
        
        # Calculer le nombre de jours nécessaires
        required_minutes = self.required_lookback * minutes
        required_days = required_minutes / (60 * 24)
        
        # Ajouter une marge de sécurité (50% supplémentaires)
        return max(int(required_days * 1.5), 30)  # Au moins 30 jours
    
    async def fetch_historical_data(self) -> Optional[pd.DataFrame]:
        """
        Récupère les données historiques nécessaires pour la stratégie.
        
        Returns:
            Optional[pd.DataFrame]: DataFrame avec les données historiques
        """
        try:
            symbol = self.config.market.symbol
            timeframe = self.config.market.timeframe
            exchange = self.config.exchange
            
            # Calculer le nombre de jours à remonter
            lookback_days = self._calculate_lookback_days(timeframe)
            
            logger.info(f"Téléchargement de {self.required_lookback} points de données " +
                       f"(~{lookback_days} jours) pour {symbol} {timeframe} sur {exchange.value}")
            
            # 1. Essayer d'abord avec l'API de l'exchange si disponible
            if self.exchange_api:
                try:
                    df = await asyncio.to_thread(
                        self.exchange_api.get_historical_klines,
                        symbol=symbol,
                        interval=timeframe,
                        limit=self.required_lookback,
                        lookback_days=lookback_days
                    )
                    
                    if df is not None and len(df) >= self.required_lookback * 0.9:  # Tolérer 10% de manque
                        logger.info(f"Données récupérées depuis l'API de l'exchange: {len(df)} points")
                        return self._format_dataframe(df)
                    else:
                        logger.warning(f"Données insuffisantes depuis l'API: {len(df) if df is not None else 0} < {self.required_lookback}")
                except Exception as e:
                    logger.warning(f"Échec de la récupération depuis l'API: {str(e)}")
            
            # 2. Essayer avec le downloader du module data
            try:
                # Préparer les paramètres pour le downloader
                exchange_name = exchange.value
                symbol_format = symbol.replace('USDT', '/USDT')  # Adapter le format du symbole
                
                # Calculer les dates
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days)
                
                # Télécharger les données
                data = await download_data(
                    exchange=exchange_name,
                    symbol=symbol_format,
                    timeframe=timeframe,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                if data and hasattr(data, 'dataframe') and len(data.dataframe) >= self.required_lookback * 0.9:
                    logger.info(f"Données récupérées depuis le downloader: {len(data.dataframe)} points")
                    return self._format_dataframe(data.dataframe)
                else:
                    logger.warning(f"Données insuffisantes depuis le downloader: " +
                                  f"{len(data.dataframe) if data and hasattr(data, 'dataframe') else 0} < {self.required_lookback}")
            except Exception as e:
                logger.warning(f"Échec du téléchargement avec le downloader: {str(e)}")
            
            # 3. Essayer de charger des données existantes
            try:
                df = load_data(
                    exchange=exchange_name,
                    symbol=symbol_format,
                    timeframe=timeframe,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                if df is not None and len(df) >= self.required_lookback * 0.9:
                    logger.info(f"Données chargées depuis les fichiers existants: {len(df)} points")
                    return self._format_dataframe(df)
                else:
                    logger.warning(f"Données insuffisantes depuis les fichiers: {len(df) if df is not None else 0} < {self.required_lookback}")
            except Exception as e:
                logger.warning(f"Échec du chargement depuis les fichiers: {str(e)}")
            
            logger.error(f"Impossible de récupérer suffisamment de données pour {symbol} {timeframe}")
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données historiques: {str(e)}")
            return None
    
    def _format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Formate le DataFrame pour qu'il soit compatible avec la stratégie.
        
        Args:
            df: DataFrame à formater
            
        Returns:
            pd.DataFrame: DataFrame formaté
        """
        # Vérifier que le DataFrame a les colonnes requises
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Convertir les noms de colonnes en minuscules
        df.columns = [col.lower() for col in df.columns]
        
        # Vérifier les colonnes manquantes
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Colonnes manquantes: {missing_columns}")
            
            # Essayer d'adapter le DataFrame
            if 'timestamp' not in df.columns and df.index.name == 'timestamp':
                df.reset_index(inplace=True)
            
            # Créer des colonnes manquantes si nécessaire
            for col in missing_columns:
                if col == 'timestamp':
                    if df.index.dtype == 'datetime64[ns]':
                        df['timestamp'] = df.index
                    else:
                        logger.error("Impossible de créer la colonne timestamp")
                        return None
                elif col in ['open', 'high', 'low'] and 'close' in df.columns:
                    df[col] = df['close']
                elif col == 'volume':
                    df[col] = 0.0
                else:
                    logger.error(f"Impossible de créer la colonne {col}")
                    return None
        
        # Convertir timestamp en datetime si ce n'est pas déjà fait
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Définir timestamp comme index
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        # Trier par index
        df.sort_index(inplace=True)
        
        # Limiter aux colonnes requises
        columns = [col for col in required_columns if col != 'timestamp']
        df = df[columns]
        
        # Sauvegarder les données formatées
        self._save_dataframe(df)
        
        return df
    
    def _save_dataframe(self, df: pd.DataFrame) -> None:
        """
        Sauvegarde le DataFrame dans un fichier CSV.
        
        Args:
            df: DataFrame à sauvegarder
        """
        try:
            # Créer un répertoire spécifique pour les données de ce symbole
            symbol_dir = os.path.join(self.output_dir, self.config.market.symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            
            # Chemin complet du fichier
            timestamp = datetime.now().strftime("%Y%m%d")
            filepath = os.path.join(symbol_dir, f"history_{timestamp}.csv")
            
            # Sauvegarder le DataFrame
            df.to_csv(filepath)
            
            logger.info(f"Données historiques sauvegardées dans {filepath}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données: {str(e)}")


async def fetch_history_for_strategy(
    strategy: StrategyConstructor,
    config: LiveConfig,
    exchange_api: Optional[ExchangeInterface] = None
) -> Optional[pd.DataFrame]:
    """
    Fonction utilitaire pour récupérer les données historiques pour une stratégie.
    
    Args:
        strategy: Constructeur de stratégie
        config: Configuration du trading en direct
        exchange_api: API de l'exchange
        
    Returns:
        Optional[pd.DataFrame]: DataFrame avec les données historiques
    """
    fetcher = HistoryFetcher(strategy, config, exchange_api)
    return await fetcher.fetch_historical_data()