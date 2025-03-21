"""
Gestionnaire de données.
Responsable de la gestion des données de marché et de l'interaction avec les exchanges.
"""

import os
import time
import json
import traceback
import pandas as pd
import numpy as np
import ccxt
import sqlite3
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import logging

from data.data_config import MarketDataConfig, IntegrityCheckResult, Timeframe, Exchange, IntegrityCheck
from data.data_config import DataIntegrityError, ProgressInfo

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Données de marché pour une paire et une période"""
    config: MarketDataConfig
    file_path: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    rows: int = 0
    last_modified: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    size_mb: float = 0.0
    integrity_status: Optional[IntegrityCheckResult] = None
    dataframe: Optional[pd.DataFrame] = None
    db_path: Optional[str] = None
    
    def __post_init__(self):
        """Initialisation post-création"""
        if self.dataframe is not None:
            object.__setattr__(self, '_has_dataframe', True)
        else:
            object.__setattr__(self, '_has_dataframe', False)
    
    @classmethod
    def from_file(cls, file_path: str, config: Optional[MarketDataConfig] = None) -> 'MarketData':
        """Crée un objet MarketData à partir d'un fichier existant"""
        filename = os.path.basename(file_path)
        parts = filename.replace('.csv', '').split('_')
        if config is None:
            symbol = parts[0]
            if len(parts) > 1:
                if not symbol.endswith('USDT'):
                    symbol = f"{parts[0]}/{parts[1]}"
                parts = parts[1:]
            timeframe = Timeframe.from_string(parts[1]) if len(parts) > 1 else Timeframe.M1
            exchange = Exchange.from_string(parts[2]) if len(parts) > 2 else Exchange.BITGET
            config = MarketDataConfig(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe
            )
        try:
            df = pd.read_csv(file_path)
            if 'timestamp' in df.columns and not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                start_date = df['timestamp'].min().strftime("%Y-%m-%d")
                end_date = df['timestamp'].max().strftime("%Y-%m-%d")
                row_count = len(df)
            else:
                start_date = None
                end_date = None
                row_count = 0
            if start_date and not config.start_date:
                config.start_date = start_date
            if end_date and not config.end_date:
                config.end_date = end_date
        except Exception:
            start_date = None
            end_date = None
            row_count = 0
            df = None
        return cls(
            config=config,
            file_path=file_path,
            start_date=start_date,
            end_date=end_date,
            rows=row_count,
            last_modified=datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S"),
            size_mb=round(os.path.getsize(file_path) / (1024 * 1024), 2),
            dataframe=df
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit les données en dictionnaire"""
        data_dict = asdict(self)
        data_dict['config'] = self.config.to_dict()
        if 'dataframe' in data_dict:
            del data_dict['dataframe']
        if self.integrity_status:
            data_dict['integrity_status'] = self.integrity_status.to_dict()
        return data_dict
    
    def load_dataframe(self, force_reload: bool = False) -> pd.DataFrame:
        """Charge les données du fichier dans un DataFrame"""
        if self.dataframe is not None and not force_reload:
            return self.dataframe
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Le fichier {self.file_path} n'existe pas")
        
        df = pd.read_csv(self.file_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        self.dataframe = df
        self.rows = len(df)
        
        if not self.start_date and not df.empty:
            self.start_date = df['timestamp'].min().strftime("%Y-%m-%d")
        if not self.end_date and not df.empty:
            self.end_date = df['timestamp'].max().strftime("%Y-%m-%d")
        
        object.__setattr__(self, '_has_dataframe', True)
        return df
    
    def to_db(self, db_path: Optional[str] = None) -> str:
        """Convertit les données en base de données SQLite"""
        if db_path is None:
            db_dir = os.path.join(os.path.dirname(self.file_path), "db")
            os.makedirs(db_dir, exist_ok=True)
            symbol_safe = self.config.symbol.replace('/', '_')
            db_path = os.path.join(db_dir, f"{symbol_safe}_{self.config.timeframe.value}.db")
        
        if not hasattr(self, '_has_dataframe') or not self._has_dataframe:
            self.load_dataframe()
        
        conn = sqlite3.connect(db_path)
        self.dataframe.to_sql('market_data', conn, if_exists='replace', index=False)
        conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON market_data(timestamp)')
        
        metadata = {
            'symbol': self.config.symbol,
            'timeframe': self.config.timeframe.value if isinstance(self.config.timeframe, Enum) else self.config.timeframe,
            'exchange': self.config.exchange.value if isinstance(self.config.exchange, Enum) else self.config.exchange,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'rows': self.rows,
            'created_at': datetime.now().isoformat()
        }
        metadata_df = pd.DataFrame([metadata])
        metadata_df.to_sql('metadata', conn, if_exists='replace', index=False)
        
        conn.close()
        self.db_path = db_path
        logger.info(f"Données converties en base de données: {db_path}")
        return db_path
    
    @classmethod
    def from_db(cls, db_path: str) -> 'MarketData':
        """Charge les données depuis une base de données SQLite"""
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"La base de données {db_path} n'existe pas")
        
        conn = sqlite3.connect(db_path)
        try:
            metadata_df = pd.read_sql('SELECT * FROM metadata LIMIT 1', conn)
            metadata = metadata_df.iloc[0].to_dict()
        except:
            metadata = {}
            
        symbol = metadata.get('symbol', os.path.basename(db_path).split('_')[0])
        timeframe_str = metadata.get('timeframe', os.path.basename(db_path).split('_')[1].replace('.db', ''))
        exchange_str = metadata.get('exchange', 'bitget')
        
        try:
            timeframe = Timeframe.from_string(timeframe_str)
        except:
            timeframe = Timeframe.M1
            
        try:
            exchange = Exchange.from_string(exchange_str)
        except:
            exchange = Exchange.BITGET
            
        config = MarketDataConfig(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=metadata.get('start_date'),
            end_date=metadata.get('end_date')
        )
        
        df = pd.read_sql('SELECT * FROM market_data', conn)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        conn.close()
        
        market_data = cls(
            config=config,
            file_path="",
            start_date=metadata.get('start_date'),
            end_date=metadata.get('end_date'),
            rows=len(df),
            dataframe=df,
            db_path=db_path
        )
        
        return market_data

class DataManager:
    """
    Gestionnaire centralisé des données pour le système de trading.
    Responsable du téléchargement, stockage et récupération des données.
    """
    
    _instance = None
    
    def __new__(cls, data_dir: str = None):
        """Implémentation du Singleton pour assurer une seule instance"""
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, data_dir: str = None):
        """
        Initialise le gestionnaire de données
        Args:
            data_dir: Répertoire des données
        """
        if getattr(self, "_initialized", False):
            return
        
        self.data_dir = data_dir or os.path.join(os.getcwd(), 'data', 'historical')
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.db_dir = os.path.join(self.data_dir, 'db')
        os.makedirs(self.db_dir, exist_ok=True)
        
        self.db_file = os.path.join(self.db_dir, 'market_data_central.db')
        self._init_central_db()
        
        self._data_cache = {}
        self._exchange_instances = {}
        self._initialized = True
        
        logger.info(f"Gestionnaire de données centralisé initialisé: {self.data_dir}")
    
    def _init_central_db(self):
        """Initialise la base de données centrale"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Table pour l'inventaire des données
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            exchange TEXT NOT NULL,
            start_date TEXT,
            end_date TEXT,
            file_path TEXT NOT NULL,
            db_path TEXT,
            rows INTEGER DEFAULT 0,
            last_updated TEXT,
            UNIQUE(symbol, timeframe, exchange)
        );
        ''')
        
        # Table pour les références aux études
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS study_data_references (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_name TEXT NOT NULL,
            data_inventory_id INTEGER NOT NULL,
            created_at TEXT,
            UNIQUE(study_name, data_inventory_id),
            FOREIGN KEY(data_inventory_id) REFERENCES data_inventory(id)
        );
        ''')
        
        conn.commit()
        conn.close()
    
    def get_exchange(self, exchange_type: Union[str, Exchange]) -> ccxt.Exchange:
        """Obtient une instance de l'exchange"""
        if isinstance(exchange_type, str):
            exchange_type = Exchange.from_string(exchange_type)
        
        if exchange_type in self._exchange_instances:
            return self._exchange_instances[exchange_type]
        
        exchange_config = exchange_type.get_config()
        exchange = exchange_config['class']({
            'enableRateLimit': True,
            'timeout': 30000,
            'options': exchange_config['options']
        })
        
        self._exchange_instances[exchange_type] = exchange
        return exchange
    
    def download_data(
        self,
        exchange: Union[str, Exchange],
        symbol: str,
        timeframe: Union[str, Timeframe],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        convert_to_db: bool = True
    ) -> Optional[MarketData]:
        """Télécharge des données historiques"""
        if isinstance(exchange, str):
            exchange = Exchange.from_string(exchange)
        
        if isinstance(timeframe, str):
            timeframe = Timeframe.from_string(timeframe)
        
        config = MarketDataConfig(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        data_id, data_info = self._get_data_inventory(exchange.value, symbol, timeframe.value)
        
        if data_info and not (start_date or end_date) and os.path.exists(data_info['file_path']):
            logger.info(f"Données existantes trouvées: {data_info['file_path']}")
            market_data = MarketData.from_file(data_info['file_path'], config)
            
            if convert_to_db and data_info['db_path'] and os.path.exists(data_info['db_path']):
                market_data.db_path = data_info['db_path']
            elif convert_to_db and not (data_info['db_path'] and os.path.exists(data_info['db_path'])):
                db_path = market_data.to_db()
                self._update_data_inventory(data_id, db_path=db_path)
            
            return market_data
        
        output_path = config.get_output_path()
        
        try:
            exchange_instance = self.get_exchange(exchange)
            
            start_ts = None
            end_ts = None
            
            if start_date:
                start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            elif limit:
                timeframe_ms = timeframe.to_milliseconds()
                start_ts = int(datetime.now().timestamp() * 1000) - limit * timeframe_ms
            else:
                logger.error("Aucune donnée à télécharger. Spécifiez une date de début ou une limite.")
                return None
            
            if end_date:
                end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
            else:
                end_ts = int(datetime.now().timestamp() * 1000)
            
            logger.info(f"Téléchargement des données pour {symbol} en {timeframe.value} sur {exchange.value}")
            
            all_data = self._fetch_data_in_batches(
                exchange_instance,
                symbol,
                timeframe.value,
                start_ts,
                end_ts,
                progress_callback,
                should_cancel
            )
            
            if not all_data or len(all_data) == 0:
                logger.warning(f"Aucune donnée récupérée pour {symbol}")
                return None
            
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            
            market_data = MarketData(
                config=config,
                file_path=output_path,
                start_date=df['timestamp'].min().strftime("%Y-%m-%d") if not df.empty else None,
                end_date=df['timestamp'].max().strftime("%Y-%m-%d") if not df.empty else None,
                rows=len(df),
                last_modified=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                size_mb=round(os.path.getsize(output_path) / (1024 * 1024), 2),
                dataframe=df
            )
            
            logger.info(f"Données enregistrées: {output_path} ({len(df)} points)")
            
            db_path = None
            if convert_to_db:
                db_path = market_data.to_db()
            
            if data_id:
                self._update_data_inventory(
                    data_id,
                    file_path=output_path,
                    db_path=db_path,
                    rows=len(df),
                    start_date=market_data.start_date,
                    end_date=market_data.end_date
                )
            else:
                self._add_data_inventory(
                    exchange.value,
                    symbol,
                    timeframe.value,
                    output_path,
                    db_path,
                    len(df),
                    market_data.start_date,
                    market_data.end_date
                )
            
            return market_data
            
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement des données: {str(e)}")
            traceback.print_exc()
            return None
    
    def _fetch_data_in_batches(
        self,
        exchange: ccxt.Exchange,
        symbol: str,
        timeframe: str,
        start_ts: int,
        end_ts: int,
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None
    ) -> List[List]:
        """Télécharge les données OHLCV en lots"""
        all_data = []
        current_ts = start_ts
        delta_time = end_ts - current_ts
        
        timeframe_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        
        timeframe_ms = timeframe_map.get(timeframe, 60 * 1000)
        max_batches = int((delta_time / timeframe_ms) / 1000) + 1
        batch_count = 0
        start_time = time.time()
        
        while (batch_count < max_batches) and (not end_ts or current_ts < end_ts):
            try:
                if should_cancel and should_cancel():
                    logger.info(f"Téléchargement annulé pour {symbol} {timeframe}")
                    return []
                
                batch = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_ts,
                    limit=1000
                )
                
                if not batch:
                    break
                
                all_data.extend(batch)
                current_ts = batch[-1][0] + 1
                batch_count += 1
                
                progress = min(100, int(batch_count * 100 / max_batches))
                elapsed_time = time.time() - start_time
                
                if batch_count > 0 and progress > 0:
                    estimated_total_time = elapsed_time * 100 / progress
                    remaining_time = estimated_total_time - elapsed_time
                else:
                    remaining_time = None
                
                if progress_callback:
                    progress_info = ProgressInfo(
                        progress=progress,
                        batch_count=batch_count,
                        max_batches=max_batches,
                        elapsed_time=elapsed_time,
                        remaining_time=remaining_time
                    )
                    progress_callback(progress_info)
                
                logger.info(f"Téléchargement {symbol} {timeframe}: {progress}% ({batch_count}/{max_batches} lots)")
                
                time.sleep(exchange.rateLimit / 1000)
                
            except Exception as e:
                logger.error(f"Erreur lors du téléchargement d'un lot: {str(e)}")
                break
        
        return all_data
    
    def load_data(
        self,
        exchange: Union[str, Exchange],
        symbol: str,
        timeframe: Union[str, Timeframe],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        download_if_missing: bool = True,
        use_db: bool = True
    ) -> Optional[pd.DataFrame]:
        """Charge des données historiques depuis le système centralisé"""
        if isinstance(exchange, str):
            exchange = Exchange.from_string(exchange)
        
        if isinstance(timeframe, str):
            timeframe = Timeframe.from_string(timeframe)
        
        cache_key = f"{exchange.value}_{symbol}_{timeframe.value}"
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        data_id, data_info = self._get_data_inventory(exchange.value, symbol, timeframe.value)
        
        if data_info:
            if use_db and data_info['db_path'] and os.path.exists(data_info['db_path']):
                try:
                    market_data = MarketData.from_db(data_info['db_path'])
                    self._data_cache[cache_key] = market_data.dataframe
                    return market_data.dataframe
                except Exception as e:
                    logger.warning(f"Erreur lors du chargement de la base de données, fallback vers CSV: {str(e)}")
            
            if os.path.exists(data_info['file_path']):
                try:
                    market_data = MarketData.from_file(data_info['file_path'])
                    df = market_data.load_dataframe()
                    
                    if use_db and (not data_info['db_path'] or not os.path.exists(data_info['db_path'])):
                        db_path = market_data.to_db()
                        self._update_data_inventory(data_id, db_path=db_path)
                    
                    self._data_cache[cache_key] = df
                    return df
                except Exception as e:
                    logger.warning(f"Erreur lors du chargement du fichier CSV: {str(e)}")
        
        if download_if_missing:
            market_data = self.download_data(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                convert_to_db=use_db
            )
            
            if market_data and market_data.dataframe is not None:
                self._data_cache[cache_key] = market_data.dataframe
                return market_data.dataframe
        
        logger.error(f"Impossible de charger les données pour {symbol} en {timeframe.value}")
        return None
    
    def get_or_download_data(
        self,
        exchange: Union[str, Exchange],
        symbol: str,
        timeframe: Union[str, Timeframe],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_download: bool = False
    ) -> Optional[pd.DataFrame]:
        """Obtient les données depuis la base centralisée ou les télécharge si nécessaire"""
        if force_download:
            market_data = self.download_data(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                convert_to_db=True
            )
            return market_data.dataframe if market_data else None
        else:
            return self.load_data(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                download_if_missing=True,
                use_db=True
            )
    
    def _get_data_inventory(self, exchange: str, symbol: str, timeframe: str) -> Tuple[Optional[int], Optional[Dict]]:
        """Recherche les données dans l'inventaire central"""
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            '''SELECT * FROM data_inventory
            WHERE exchange = ? AND symbol = ? AND timeframe = ?''',
            (exchange, symbol, timeframe)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return row['id'], dict(row)
        
        return None, None
    
    def _add_data_inventory(self, exchange: str, symbol: str, timeframe: str,
                           file_path: str, db_path: str, rows: int,
                           start_date: str = None, end_date: str = None) -> int:
        """Ajoute une entrée à l'inventaire des données"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute(
            '''INSERT OR REPLACE INTO data_inventory
            (exchange, symbol, timeframe, file_path, db_path, rows, start_date, end_date, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (exchange, symbol, timeframe, file_path, db_path, rows,
             start_date, end_date, datetime.now().isoformat())
        )
        
        data_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return data_id
    
    def _update_data_inventory(self, data_id: int, **kwargs) -> bool:
        """Met à jour une entrée de l'inventaire"""
        if not kwargs:
            return False
        
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        set_clause = ", ".join([f"{key} = ?" for key in kwargs.keys()])
        values = list(kwargs.values())
        values.append(datetime.now().isoformat())
        values.append(data_id)
        
        query = f'''UPDATE data_inventory
        SET {set_clause}, last_updated = ?
        WHERE id = ?'''
        
        cursor.execute(query, values)
        conn.commit()
        conn.close()
        
        return True
    
    def associate_study_with_data(self, study_name: str, exchange: str, symbol: str, timeframe: str) -> bool:
        """Associe une étude à des données existantes"""
        data_id, _ = self._get_data_inventory(exchange, symbol, timeframe)
        if not data_id:
            logger.warning(f"Données non trouvées pour {symbol}/{timeframe} sur {exchange}")
            return False
        
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                '''INSERT OR REPLACE INTO study_data_references
                (study_name, data_inventory_id, created_at)
                VALUES (?, ?, ?)''',
                (study_name, data_id, datetime.now().isoformat())
            )
            conn.commit()
            logger.info(f"Étude {study_name} associée aux données {symbol}/{timeframe}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'association de l'étude aux données: {str(e)}")
            return False
        finally:
            conn.close()
    
    def get_data_for_study(self, study_name: str) -> List[Dict]:
        """Récupère les données associées à une étude"""
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            '''SELECT d.* FROM data_inventory d
            INNER JOIN study_data_references r ON d.id = r.data_inventory_id
            WHERE r.study_name = ?''',
            (study_name,)
        )
        
        result = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return result
    
    def list_available_data(self) -> List[Dict]:
        """Liste toutes les données disponibles dans l'inventaire"""
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM data_inventory')
        result = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return result
    
    def clear_cache(self):
        """Vide le cache de données"""
        self._data_cache.clear()
        logger.info("Cache de données vidé")
    
    def load_study_data(self, study_name: str, file_path: str = None) -> Optional[pd.DataFrame]:
        """
        Charge les données associées à une étude.
        Cette méthode est spécifiquement conçue pour fonctionner avec StudyManager
        """
        study_data = self.get_data_for_study(study_name)
        
        if not study_data:
            logger.warning(f"Aucune donnée associée à l'étude '{study_name}'")
            return None
        
        # Utilise le premier jeu de données associé par défaut
        data_info = study_data[0]
        
        if file_path:
            # Si un fichier spécifique est demandé
            for data in study_data:
                if os.path.basename(data['file_path']) == file_path:
                    data_info = data
                    break
        
        # Charge les données
        if data_info['db_path'] and os.path.exists(data_info['db_path']):
            try:
                market_data = MarketData.from_db(data_info['db_path'])
                return market_data.dataframe
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de la base de données, fallback vers CSV: {str(e)}")
        
        if os.path.exists(data_info['file_path']):
            try:
                market_data = MarketData.from_file(data_info['file_path'])
                return market_data.load_dataframe()
            except Exception as e:
                logger.error(f"Erreur lors du chargement du fichier CSV: {str(e)}")
        
        return None
    
    def save_data_for_study(self, study_name: str, data: pd.DataFrame, exchange: str, symbol: str, timeframe: str) -> bool:
        """
        Sauvegarde des données pour une étude spécifique.
        Cette méthode est spécifiquement conçue pour fonctionner avec StudyManager
        """
        try:
            config = MarketDataConfig(
                exchange=Exchange.from_string(exchange) if isinstance(exchange, str) else exchange,
                symbol=symbol,
                timeframe=Timeframe.from_string(timeframe) if isinstance(timeframe, str) else timeframe
            )
            
            output_path = config.get_output_path()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            data.to_csv(output_path, index=False)
            
            # Créer l'objet MarketData
            market_data = MarketData(
                config=config,
                file_path=output_path,
                start_date=data['timestamp'].min().strftime("%Y-%m-%d") if 'timestamp' in data.columns and not data.empty else None,
                end_date=data['timestamp'].max().strftime("%Y-%m-%d") if 'timestamp' in data.columns and not data.empty else None,
                rows=len(data),
                size_mb=round(os.path.getsize(output_path) / (1024 * 1024), 2),
                dataframe=data
            )
            
            # Convertir en DB
            db_path = market_data.to_db()
            
            # Mettre à jour l'inventaire
            data_id, data_info = self._get_data_inventory(exchange.value if isinstance(exchange, Enum) else exchange, 
                                                       symbol, 
                                                       timeframe.value if isinstance(timeframe, Enum) else timeframe)
            
            if data_id:
                self._update_data_inventory(
                    data_id,
                    file_path=output_path,
                    db_path=db_path,
                    rows=len(data),
                    start_date=market_data.start_date,
                    end_date=market_data.end_date
                )
            else:
                data_id = self._add_data_inventory(
                    exchange.value if isinstance(exchange, Enum) else exchange,
                    symbol,
                    timeframe.value if isinstance(timeframe, Enum) else timeframe,
                    output_path,
                    db_path,
                    len(data),
                    market_data.start_date,
                    market_data.end_date
                )
            
            # Associer à l'étude
            self.associate_study_with_data(study_name, 
                                          exchange.value if isinstance(exchange, Enum) else exchange, 
                                          symbol, 
                                          timeframe.value if isinstance(timeframe, Enum) else timeframe)
            
            logger.info(f"Données sauvegardées et associées à l'étude '{study_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données: {str(e)}")
            traceback.print_exc()
            return False

# Fonctions d'aide pour utilisation directe
def get_data_manager() -> DataManager:
    """Obtient une instance du gestionnaire de données"""
    return DataManager()

def download_data(
    exchange: Union[str, Exchange] = 'bitget',
    symbol: str = 'BTC/USDT',
    timeframe: Union[str, Timeframe] = '1m',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
    progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    convert_to_db: bool = True
) -> Optional[MarketData]:
    """Fonction d'aide pour télécharger des données (pour compatibilité)"""
    manager = DataManager()
    return manager.download_data(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        progress_callback=progress_callback,
        should_cancel=should_cancel,
        convert_to_db=convert_to_db
    )