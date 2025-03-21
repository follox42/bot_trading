"""
Interface abstraite pour tous les exchanges.
Définit les méthodes standards que chaque implémentation d'exchange doit fournir.
"""

import abc
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum
import pandas as pd
from datetime import datetime


class OrderType(Enum):
    """Types d'ordres supportés"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Côtés des ordres"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Statuts possibles des ordres"""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(Enum):
    """Côtés des positions"""
    LONG = "long"
    SHORT = "short"
    BOTH = "both"  # Pour le mode hedge


class ExchangeInterface(abc.ABC):
    """Interface abstraite pour tous les exchanges"""
    
    @abc.abstractmethod
    def __init__(self, api_key: str, api_secret: str, passphrase: Optional[str] = None, testnet: bool = False):
        """
        Initialise l'API de l'exchange.
        
        Args:
            api_key: Clé API
            api_secret: Secret API
            passphrase: Phrase secrète (si nécessaire)
            testnet: Utiliser le réseau de test
        """
        pass
    
    @abc.abstractmethod
    def get_server_time(self) -> int:
        """
        Obtient l'horodatage du serveur.
        
        Returns:
            int: Timestamp en millisecondes
        """
        pass
    
    @abc.abstractmethod
    def get_account_info(self) -> Dict:
        """
        Récupère les informations du compte.
        
        Returns:
            Dict: Informations du compte
        """
        pass
    
    @abc.abstractmethod
    def get_balance(self, asset: str = "USDT") -> float:
        """
        Récupère le solde d'un actif.
        
        Args:
            asset: Symbole de l'actif
            
        Returns:
            float: Solde disponible
        """
        pass
    
    @abc.abstractmethod
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Récupère les positions ouvertes.
        
        Args:
            symbol: Symbole de la paire (optionnel)
            
        Returns:
            List[Dict]: Liste des positions
        """
        pass
    
    @abc.abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Récupère les ordres ouverts.
        
        Args:
            symbol: Symbole de la paire (optionnel)
            
        Returns:
            List[Dict]: Liste des ordres
        """
        pass
    
    @abc.abstractmethod
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Place un ordre sur l'exchange.
        
        Args:
            symbol: Symbole de la paire
            side: Côté de l'ordre
            order_type: Type d'ordre
            quantity: Quantité
            price: Prix (pour les ordres limite)
            stop_price: Prix de déclenchement (pour les ordres stop)
            take_profit: Prix de take profit
            stop_loss: Prix de stop loss
            reduce_only: Réduire uniquement (ne pas ouvrir de nouvelles positions)
            client_order_id: ID d'ordre côté client
            **kwargs: Paramètres additionnels spécifiques à l'exchange
            
        Returns:
            Dict: Informations de l'ordre placé
        """
        pass
    
    @abc.abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """
        Annule un ordre.
        
        Args:
            symbol: Symbole de la paire
            order_id: ID de l'ordre
            
        Returns:
            Dict: Résultat de l'annulation
        """
        pass
    
    @abc.abstractmethod
    def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict:
        """
        Annule tous les ordres ouverts.
        
        Args:
            symbol: Symbole de la paire (optionnel)
            
        Returns:
            Dict: Résultat de l'annulation
        """
        pass
    
    @abc.abstractmethod
    def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """
        Définit l'effet de levier.
        
        Args:
            symbol: Symbole de la paire
            leverage: Niveau de levier
            
        Returns:
            Dict: Résultat de l'opération
        """
        pass
    
    @abc.abstractmethod
    def set_margin_mode(self, symbol: str, margin_mode: str) -> Dict:
        """
        Définit le mode de marge.
        
        Args:
            symbol: Symbole de la paire
            margin_mode: Mode de marge ('isolated' ou 'cross')
            
        Returns:
            Dict: Résultat de l'opération
        """
        pass
    
    @abc.abstractmethod
    def set_position_mode(self, position_mode: str) -> Dict:
        """
        Définit le mode de position.
        
        Args:
            position_mode: Mode de position ('one-way' ou 'hedge')
            
        Returns:
            Dict: Résultat de l'opération
        """
        pass
    
    @abc.abstractmethod
    def get_ticker(self, symbol: str) -> Dict:
        """
        Récupère les informations de ticker.
        
        Args:
            symbol: Symbole de la paire
            
        Returns:
            Dict: Informations du ticker
        """
        pass
    
    @abc.abstractmethod
    def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict]:
        """
        Récupère les chandeliers japonais.
        
        Args:
            symbol: Symbole de la paire
            interval: Intervalle de temps ("1m", "5m", "15m", "1h", etc.)
            limit: Nombre maximum d'éléments
            start_time: Timestamp de début en millisecondes
            end_time: Timestamp de fin en millisecondes
            
        Returns:
            List[Dict]: Liste des chandeliers
        """
        pass
    
    @abc.abstractmethod
    def get_historical_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 1000,
        lookback_days: int = 30
    ) -> pd.DataFrame:
        """
        Récupère des chandeliers historiques, potentiellement plus que la limite standard.
        
        Args:
            symbol: Symbole de la paire
            interval: Intervalle de temps
            limit: Nombre maximum d'éléments à récupérer
            lookback_days: Nombre de jours à remonter
            
        Returns:
            pd.DataFrame: DataFrame avec les données OHLCV
        """
        pass
    
    @abc.abstractmethod
    def place_tp_sl_orders(
        self,
        symbol: str,
        position_side: PositionSide,
        quantity: float,
        take_profit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None
    ) -> Tuple[Dict, Dict]:
        """
        Place des ordres take profit et stop loss pour une position.
        
        Args:
            symbol: Symbole de la paire
            position_side: Côté de la position
            quantity: Quantité
            take_profit_price: Prix du take profit
            stop_loss_price: Prix du stop loss
            
        Returns:
            Tuple[Dict, Dict]: (ordre TP, ordre SL)
        """
        pass
    
    @abc.abstractmethod
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalise le format du symbole pour l'exchange.
        
        Args:
            symbol: Symbole à normaliser
            
        Returns:
            str: Symbole normalisé
        """
        pass
    
    @abc.abstractmethod
    def to_standard_format(self, data: Any, data_type: str) -> Any:
        """
        Convertit les données spécifiques à l'exchange en format standard.
        
        Args:
            data: Données à convertir
            data_type: Type de données ('ticker', 'klines', 'order', etc.)
            
        Returns:
            Any: Données au format standard
        """
        pass