"""
Module de téléchargement de données de marché depuis les exchanges de crypto-monnaies
"""
import os
import time
import traceback
import pandas as pd
import ccxt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

from logger.logger import CentralizedLogger, LogLevel, LoggerType

class MarketDataDownloader:
    def __init__(self, 
                 exchange_name='bitget', 
                 symbol='BTC/USDT', 
                 timeframe='1m', 
                 limit=None,
                 start_date=None,
                 end_date=None,
                 central_logger=None):
        """
        Télécharge des données historiques depuis les exchanges de crypto-monnaies

        Args:
            exchange_name (str): Nom de l'exchange ('bitget' ou 'binance')
            symbol (str): Paire de trading
            timeframe (str): Timeframe des bougies
            limit (int): Nombre de bougies à télécharger
            start_date (str, optional): Date de début (YYYY-MM-DD)
            end_date (str, optional): Date de fin (YYYY-MM-DD)
            central_logger (CentralizedLogger, optional): Logger centralisé
        """
        # Définir le logger
        self.central_logger = central_logger
        if central_logger:
            self.logger = central_logger.get_data_logger(f"downloader_{exchange_name}")
        else:
            # Fallback à un logger standard si pas de logger centralisé
            import logging
            self.logger = logging.getLogger(f"downloader_{exchange_name}")
        
        # Créer le répertoire de données s'il n'existe pas
        self.data_dir = os.path.join(os.getcwd(), 'data', 'historical')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialiser l'exchange
        self._initialize_exchange(exchange_name)
        
        self.exchange_name = exchange_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.start_date = start_date
        self.end_date = end_date
        
        # Générer le nom du fichier de sortie
        filename_parts = [
            self.symbol.replace('/', '_'),
            self.timeframe,
            self.exchange_name
        ]
        
        if self.start_date:
            start_date_str = self.start_date.replace('-', '')
            filename_parts.append(start_date_str)
        elif self.limit:
            filename_parts.append(str(self.limit))

        self.output_path = os.path.join(self.data_dir, f"{'_'.join(filename_parts)}.csv")
        self.log_info(f"Fichier de sortie: {self.output_path}")

    def log_info(self, message):
        """Log un message d'information"""
        if hasattr(self, 'logger'):
            self.logger.info(message)
        else:
            print(f"INFO: {message}")
            
    def log_error(self, message):
        """Log un message d'erreur"""
        if hasattr(self, 'logger'):
            self.logger.error(message)
        else:
            print(f"ERROR: {message}")
            
    def log_warning(self, message):
        """Log un message d'avertissement"""
        if hasattr(self, 'logger'):
            self.logger.warning(message)
        else:
            print(f"WARNING: {message}")

    def _initialize_exchange(self, exchange_name):
        """
        Initialise l'exchange de crypto-monnaies

        Args:
            exchange_name (str): Nom de l'exchange
        """
        exchange_configs = {
            'bitget': {
                'class': ccxt.bitget,
                'options': {'defaultType': 'future'}
            },
            'binance': {
                'class': ccxt.binance,
                'options': {'defaultType': 'future'}
            }
        }
        
        if exchange_name not in exchange_configs:
            error_msg = f"Exchange non supporté: {exchange_name}. Supportés: {list(exchange_configs.keys())}"
            self.log_error(error_msg)
            raise ValueError(error_msg)
        
        config = exchange_configs[exchange_name]
        
        self.exchange = config['class']({
            'enableRateLimit': True,
            'timeout': 30000,
            'options': config['options'] if config['options'] else []
        })
        
        self.log_info(f"Exchange {exchange_name.capitalize()} initialisé")

    def download_historical_data(self):
        """
        Télécharge des données historiques avec des paramètres flexibles

        Returns:
            str: Chemin du fichier de données sauvegardé
        """
        # Vérifier si le fichier existe déjà
        if os.path.exists(self.output_path) and not self.start_date and not self.end_date:
            self.log_info(f"Données existantes trouvées: {self.output_path}")
            return self.output_path

        # Préparer les timestamps
        start_ts = None
        end_ts = None

        if self.start_date:
            start_ts = int(datetime.strptime(self.start_date, "%Y-%m-%d").timestamp() * 1000)
        elif self.limit:
            # Calculer la date de début en fonction du timeframe et de la limite
            timeframe_ms = self._get_timeframe_milliseconds()
            start_ts = int(datetime.now().timestamp() * 1000) - self.limit * timeframe_ms
        else:
            error_msg = "Aucune donnée à télécharger. Spécifiez une date de début ou une limite."
            self.log_error(error_msg)
            raise ValueError(error_msg)
        
        if self.end_date:
            end_ts = int(datetime.strptime(self.end_date, "%Y-%m-%d").timestamp() * 1000)
        else:
            end_ts = int(datetime.now().timestamp() * 1000)

        self.log_info(f"Téléchargement des données pour {self.symbol} en {self.timeframe} sur {self.exchange_name}")
        
        try:
            # Télécharger les données OHLCV
            all_data = self._fetch_data_in_batches(start_ts, end_ts)

            if not all_data or len(all_data) == 0:
                self.log_warning(f"Aucune donnée n'a été récupérée")
                return None

            # Convertir en DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Enregistrer en CSV
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            df.to_csv(self.output_path, index=False)
            
            self.log_info(f"Données enregistrées: {self.output_path}")
            self.log_info(f"Total de lignes: {len(df)}")
            
            # Enregistrer un événement JSON si nous avons un logger centralisé
            if self.central_logger:
                self.central_logger.log_json_event(
                    f"data.downloader_{self.exchange_name}",
                    "data_downloaded",
                    {
                        "symbol": self.symbol,
                        "timeframe": self.timeframe,
                        "start_date": df['timestamp'].min().strftime("%Y-%m-%d %H:%M:%S") if not df.empty else None,
                        "end_date": df['timestamp'].max().strftime("%Y-%m-%d %H:%M:%S") if not df.empty else None,
                        "rows": len(df),
                        "file_path": self.output_path
                    }
                )
            
            return self.output_path

        except Exception as e:
            error_msg = f"Erreur lors du téléchargement des données: {str(e)}"
            self.log_error(error_msg)
            if self.central_logger:
                self.central_logger.log_json_event(
                    f"data.downloader_{self.exchange_name}",
                    "data_download_error",
                    {
                        "symbol": self.symbol,
                        "timeframe": self.timeframe,
                        "error": str(e)
                    },
                    level=LogLevel.ERROR
                )
            traceback.print_exc()
            return None
    
    def _get_timeframe_milliseconds(self):
        """
        Convertit le timeframe en millisecondes
        
        Returns:
            int: Timeframe en millisecondes
        """
        # Mapping timeframe vers millisecondes
        timeframe_multipliers = {
            '1m': 60 * 1000,         # 1 minute
            '5m': 5 * 60 * 1000,     # 5 minutes
            '15m': 15 * 60 * 1000,   # 15 minutes
            '30m': 30 * 60 * 1000,   # 30 minutes
            '1h': 60 * 60 * 1000,    # 1 heure
            '4h': 4 * 60 * 60 * 1000, # 4 heures
            '1d': 24 * 60 * 60 * 1000 # 1 jour
        }
        
        return timeframe_multipliers.get(self.timeframe, 60 * 1000)  # Par défaut 1 minute

    def _fetch_data_in_batches(self, start_ts=None, end_ts=None):
        """
        Télécharge les données OHLCV en lots avec une barre de progression dynamique

        Args:
            start_ts (int, optional): Timestamp de début en millisecondes
            end_ts (int, optional): Timestamp de fin en millisecondes

        Returns:
            list: Liste de données OHLCV
        """
        all_data = []
        current_ts = start_ts or int(self.exchange.milliseconds())
  
        # Calculer le maximum de lots en fonction du timeframe
        delta_time = end_ts - current_ts
        timeframe_ms = self._get_timeframe_milliseconds()
        
        # Calculer le nombre de bougies dans la plage de temps (divisé par 1000 pour la taille du lot)
        max_batches = int((delta_time / timeframe_ms) / 1000) + 1
        
        batch_count = 0
        
        while (batch_count < max_batches) and (not end_ts or current_ts < end_ts):
            try:
                # Télécharger un lot de données
                batch = self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    since=current_ts,
                    limit=1000
                )

                # Sortir s'il n'y a plus de données
                if not batch:
                    break

                # Étendre les données
                all_data.extend(batch)

                # Mettre à jour le timestamp pour le prochain lot
                current_ts = batch[-1][0] + 1
                batch_count += 1

                # Afficher la progression
                progress = min(100, int(batch_count * 100 / max_batches))
                self.log_info(f"Téléchargement {self.symbol} {self.timeframe}: {progress}% ({batch_count}/{max_batches} lots)")

                # Limitation de débit
                time.sleep(self.exchange.rateLimit / 1000)

            except Exception as e:
                error_msg = f"Erreur lors du téléchargement du lot: {str(e)}"
                self.log_error(error_msg)
                break

        return all_data

    @staticmethod
    def list_available_data(data_dir=None):
        """
        Liste les fichiers de données disponibles
        
        Args:
            data_dir (str, optional): Répertoire des données
            
        Returns:
            list: Liste de dictionnaires contenant les informations sur les fichiers
        """
        if not data_dir:
            data_dir = os.path.join(os.getcwd(), 'data', 'historical')
        
        if not os.path.exists(data_dir):
            return []
        
        data_files = []
        
        for filename in os.listdir(data_dir):
            if not filename.endswith('.csv'):
                continue
                
            file_path = os.path.join(data_dir, filename)
            
            # Extraire les informations du nom de fichier
            parts = filename.replace('.csv', '').split('_')
            
            if len(parts) < 3:
                continue
                
            # Analyser le nom du fichier
            symbol = parts[0]
            if len(parts) > 1:
                symbol = f"{parts[0]}/{parts[1]}"
                parts = parts[1:]
                
            timeframe = parts[1] if len(parts) > 1 else "unknown"
            exchange = parts[2] if len(parts) > 2 else "unknown"
            
            # Lire le fichier pour obtenir plus d'informations
            try:
                df = pd.read_csv(file_path)
                
                if 'timestamp' in df.columns and not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    start_date = df['timestamp'].min().strftime("%Y-%m-%d")
                    end_date = df['timestamp'].max().strftime("%Y-%m-%d")
                    row_count = len(df)
                else:
                    start_date = "N/A"
                    end_date = "N/A"
                    row_count = 0
            except Exception as e:
                start_date = "Erreur"
                end_date = "Erreur"
                row_count = 0
            
            # Informations du fichier
            file_info = {
                "filename": filename,
                "symbol": symbol,
                "timeframe": timeframe,
                "exchange": exchange,
                "start_date": start_date,
                "end_date": end_date,
                "rows": row_count,
                "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S"),
                "size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2)
            }
            
            data_files.append(file_info)
        
        return data_files

def download_data(
    exchange='bitget', 
    symbol='BTC/USDT', 
    timeframe='1m', 
    limit=None,
    start_date=None,
    end_date=None,
    central_logger=None
):
    """
    Fonction simplifiée pour télécharger des données de marché

    Args:
        exchange (str): Nom de l'exchange ('bitget' ou 'binance')
        symbol (str): Paire de trading
        timeframe (str): Timeframe des bougies
        limit (int): Nombre de bougies à télécharger
        start_date (str, optional): Date de début (YYYY-MM-DD)
        end_date (str, optional): Date de fin (YYYY-MM-DD)
        central_logger (CentralizedLogger, optional): Logger centralisé

    Returns:
        str: Chemin du fichier de données sauvegardé
    """
    downloader = MarketDataDownloader(
        exchange_name=exchange,
        symbol=symbol, 
        timeframe=timeframe, 
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        central_logger=central_logger
    )
    return downloader.download_historical_data()