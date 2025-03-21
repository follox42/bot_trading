import os
import logging
import datetime
import json
import threading
from enum import Enum
from typing import Dict, List, Optional, Union, Any

class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

class LoggerType(Enum):
    STRATEGY = "strategy"
    DATA = "data"
    OPTIMIZATION = "optimization"
    SYSTEM = "system"
    UI = "ui"
    API = "api"
    BACKTEST = "backtest"
    LIVE_TRADING = "live_trading"

class CentralizedLogger:
    """
    Système de logging centralisé pour l'application de trading.
    Collecte tous les logs et les assemble en un seul endroit tout en maintenant
    la séparation pour chaque sous-système.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Implémentation Singleton pour s'assurer qu'il n'existe qu'une seule instance du logger"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CentralizedLogger, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, base_dir: str = "logs", console_output: bool = True):
        """
        Initialise le système de logs centralisé
        
        Args:
            base_dir: Répertoire de base pour stocker les fichiers de logs
            console_output: Si True, affiche également les logs dans la console
        """
        if self._initialized:
            return
            
        self.base_dir = base_dir
        self.console_output = console_output
        self.loggers = {}
        
        # Création du répertoire de logs s'il n'existe pas
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
        # Création de sous-répertoires pour chaque type de logger
        for logger_type in LoggerType:
            type_dir = os.path.join(base_dir, logger_type.value)
            if not os.path.exists(type_dir):
                os.makedirs(type_dir)
        
        # Création du logger central qui recueille tous les logs
        self.central_logger = self._create_logger(
            "central", 
            os.path.join(base_dir, "central.log"),
            LogLevel.DEBUG
        )
        
        self.central_logger.info(f"=== INITIALISATION DU LOGGER CENTRAL : {datetime.datetime.now().isoformat()} ===")
        self._initialized = True
    
    def _create_logger(self, name: str, log_file: str, level: LogLevel, 
                      format_str: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s") -> logging.Logger:
        """
        Crée un logger spécifique
        
        Args:
            name: Nom du logger
            log_file: Chemin vers le fichier de log
            level: Niveau de log minimum
            format_str: Format des messages de log
            
        Returns:
            Le logger configuré
        """
        logger = logging.getLogger(name)
        logger.setLevel(level.value)
        
        # Supprime les handlers existants pour éviter les doublons
        if logger.handlers:
            logger.handlers = []
        
        # Gestionnaire de fichier
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level.value)
        
        # Format des logs
        formatter = logging.Formatter(format_str)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Ajouter un gestionnaire de console si demandé
        if self.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level.value)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        return logger
    
    def get_logger(self, name: str, logger_type: LoggerType, level: LogLevel = LogLevel.INFO) -> logging.Logger:
        """
        Récupère ou crée un logger pour un composant spécifique
        
        Args:
            name: Nom du composant (ex: "strategy_ema_cross")
            logger_type: Type de logger (strategy, data, etc.)
            level: Niveau de log minimum
            
        Returns:
            Le logger spécifique au composant
        """
        logger_id = f"{logger_type.value}.{name}"
        
        if logger_id in self.loggers:
            return self.loggers[logger_id]
        
        # Création du fichier de log spécifique à ce composant
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self.base_dir, logger_type.value, f"{name}_{today}.log")
        
        # Création du logger
        component_logger = self._create_logger(logger_id, log_file, level)
        
        # Ajout d'un gestionnaire pour envoyer les logs au logger central également
        class CentralHandler(logging.Handler):
            def __init__(self, central_logger):
                super().__init__()
                self.central_logger = central_logger
                
            def emit(self, record):
                # Ajouter la source au message pour le logger central
                msg = f"[{record.name}] {record.getMessage()}"
                self.central_logger.log(record.levelno, msg)
        
        central_handler = CentralHandler(self.central_logger)
        central_handler.setLevel(level.value)
        component_logger.addHandler(central_handler)
        
        # Enregistrement du logger
        self.loggers[logger_id] = component_logger
        
        return component_logger
    
    def get_strategy_logger(self, strategy_name: str, level: LogLevel = LogLevel.INFO) -> logging.Logger:
        """Raccourci pour créer un logger de stratégie"""
        return self.get_logger(strategy_name, LoggerType.STRATEGY, level)
    
    def get_data_logger(self, name: str = "general", level: LogLevel = LogLevel.INFO) -> logging.Logger:
        """Raccourci pour créer un logger de données"""
        return self.get_logger(name, LoggerType.DATA, level)
    
    def get_ui_logger(self, name: str = "dashboard", level: LogLevel = LogLevel.INFO) -> logging.Logger:
        """Raccourci pour créer un logger d'interface utilisateur"""
        return self.get_logger(name, LoggerType.UI, level)
    
    def get_backtest_logger(self, name: str, level: LogLevel = LogLevel.INFO) -> logging.Logger:
        """Raccourci pour créer un logger de backtest"""
        return self.get_logger(name, LoggerType.BACKTEST, level)
    
    def get_live_trading_logger(self, name: str, level: LogLevel = LogLevel.INFO) -> logging.Logger:
        """Raccourci pour créer un logger de trading en direct"""
        return self.get_logger(name, LoggerType.LIVE_TRADING, level)
    
    def log_json_event(self, source: str, event_type: str, data: Dict[str, Any], 
                      level: LogLevel = LogLevel.INFO) -> None:
        """
        Enregistre un événement formaté en JSON
        
        Args:
            source: Source de l'événement
            event_type: Type d'événement
            data: Données associées à l'événement
            level: Niveau de log
        """
        event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "source": source,
            "event_type": event_type,
            "data": data
        }
        
        event_json = json.dumps(event)
        self.central_logger.log(level.value, event_json)
        
        # Également enregistrer dans le logger spécifique s'il existe
        source_parts = source.split('.')
        if len(source_parts) >= 2:
            logger_type = source_parts[0]
            name = source_parts[1]
            
            try:
                logger_enum_type = LoggerType(logger_type)
                component_logger = self.get_logger(name, logger_enum_type)
                component_logger.log(level.value, event_json)
            except (ValueError, KeyError):
                # Si le logger_type n'est pas un LoggerType valide, ignorez simplement
                pass
    
    def get_recent_logs(self, max_entries=100, logger_type=None, level=None):
        """
        Récupère les logs les plus récents du système - Version améliorée qui gère plusieurs formats
        
        Args:
            max_entries: Nombre maximum d'entrées à récupérer
            logger_type: Type de logger à filtrer (optionnel)
            level: Niveau de log minimum (optionnel)
            
        Returns:
            Liste des entrées de log les plus récentes
        """
        log_entries = []
        log_file = os.path.join(self.base_dir, "central.log")
        
        if not os.path.exists(log_file):
            return []
        
        try:
            with open(log_file, 'r') as f:
                # Lire les dernières lignes
                lines = f.readlines()
                # Prendre les max_entries dernières lignes
                for line in lines[-max_entries:]:
                    # Parser flexible pour différents formats de logs
                    try:
                        # Format 1: "2025-03-10 17:40:50,310 - INFO - central - [ui.splash_screen] Message"
                        if " - INFO - " in line or " - WARNING - " in line or " - ERROR - " in line or " - DEBUG - " in line or " - CRITICAL - " in line:
                            # Nouveau format avec tirets
                            parts = line.split(" - ", 3)
                            if len(parts) >= 4:
                                timestamp = parts[0].strip()
                                level_str = parts[1].strip()
                                # parts[2] contient "central"
                                source_message = parts[3].strip()
                                
                                # Extraire la source et le message
                                if "[" in source_message and "]" in source_message:
                                    source_end = source_message.find("]")
                                    source = source_message[1:source_end].strip()
                                    message = source_message[source_end+1:].strip()
                                else:
                                    source = "unknown"
                                    message = source_message
                        
                        # Format 2: "17:33:52INFO: [central] [ui.dashboard] Message"
                        elif "INFO:" in line or "WARNING:" in line or "ERROR:" in line or "DEBUG:" in line or "CRITICAL:" in line:
                            # Ancien format
                            for level_name in ["INFO:", "WARNING:", "ERROR:", "DEBUG:", "CRITICAL:"]:
                                if level_name in line:
                                    time_level_parts = line.split(level_name)
                                    time_part = time_level_parts[0].strip()
                                    level_str = level_name.replace(":", "").strip()
                                    rest = time_level_parts[1].strip()
                                    
                                    # Extraire [central] et la suite
                                    if "[central]" in rest:
                                        rest = rest.split("[central]", 1)[1].strip()
                                        
                                        # Extraire la source et le message
                                        if "[" in rest and "]" in rest:
                                            source_end = rest.find("]")
                                            source = rest[1:source_end].strip()
                                            message = rest[source_end+1:].strip()
                                        else:
                                            source = "unknown"
                                            message = rest
                                    else:
                                        source = "unknown"
                                        message = rest
                                        
                                    # Ajouter la date si nécessaire pour l'ancien format
                                    if " " not in time_part:  # Pas de date, seulement l'heure
                                        today = datetime.now().strftime("%Y-%m-%d")
                                        timestamp = f"{today} {time_part}"
                                    else:
                                        timestamp = time_part
                                    break
                        else:
                            # Format inconnu, ignorer cette ligne
                            continue
                        
                        # Créer l'entrée de log
                        log_entry = {
                            'timestamp': timestamp,
                            'level': level_str,
                            'source': source,
                            'message': message
                        }
                        
                        # Filtrage par type de logger si spécifié
                        if logger_type and logger_type != 'all':
                            if not source.startswith(logger_type):
                                continue
                        
                        # Filtrage par niveau si spécifié
                        if level and level != 'all':
                            if level_str.upper() != level.upper():
                                continue
                                
                        log_entries.append(log_entry)
                    except Exception as e:
                        print(f"Erreur lors du parsing de la ligne: {line}: {e}")
                        continue
        except Exception as e:
            print(f"Erreur lors de la lecture des logs: {e}")
            
        return log_entries