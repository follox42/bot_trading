"""
Module de trading en direct qui permet d'exécuter des stratégies en temps réel
sur différentes plateformes d'échange comme Bitget, en utilisant des comptes réels ou démo.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import time
import hmac
import hashlib
import uuid
from datetime import datetime
import asyncio
import websockets
import requests
from enum import Enum

# Import des modules existants
from core.strategy.constructor.constructor import StrategyConstructor
from core.strategy.strategy_manager import StrategyManager

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExchangeType(Enum):
    """Types d'exchanges supportés"""
    BITGET = "bitget"
    BINANCE = "binance"
    BYBIT = "bybit"
    # Autres exchanges à ajouter


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


class LiveTradingMode(Enum):
    """Modes de trading en direct"""
    REAL = "real"         # Trading réel
    PAPER = "paper"       # Trading papier (simulé)
    DEMO = "demo"         # Compte démo de l'exchange


class LiveTraderConfig:
    """Configuration du trader en direct"""
    
    def __init__(
        self,
        exchange: ExchangeType = ExchangeType.BITGET,
        symbol: str = "BTCUSDT",
        trading_mode: LiveTradingMode = LiveTradingMode.PAPER,
        api_key: str = "",
        api_secret: str = "",
        passphrase: Optional[str] = None,
        base_url: Optional[str] = None,
        ws_url: Optional[str] = None,
        leverage: int = 1,
        margin_mode: str = "isolated",
        position_mode: str = "one-way",
        default_quantity: float = 0.001,
        max_position_value: float = 1000.0,
        enable_tp_sl: bool = True,
        enable_trailing_stop: bool = False,
        update_interval: int = 5  # secondes
    ):
        """
        Initialise la configuration du trader en direct.
        
        Args:
            exchange: Type d'exchange
            symbol: Paire de trading
            trading_mode: Mode de trading (réel, papier, démo)
            api_key: Clé API
            api_secret: Secret API
            passphrase: Phrase secrète (si nécessaire)
            base_url: URL de base de l'API (si différente de celle par défaut)
            ws_url: URL WebSocket (si différente de celle par défaut)
            leverage: Effet de levier
            margin_mode: Mode de marge ('isolated' ou 'cross')
            position_mode: Mode de position ('one-way' ou 'hedge')
            default_quantity: Quantité par défaut
            max_position_value: Valeur maximale de position
            enable_tp_sl: Activer les TP/SL
            enable_trailing_stop: Activer les trailing stops
            update_interval: Intervalle de mise à jour en secondes
        """
        self.exchange = exchange
        self.symbol = symbol
        self.trading_mode = trading_mode
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        
        # URLs par défaut selon l'exchange
        self.base_url = base_url or self._get_default_base_url()
        self.ws_url = ws_url or self._get_default_ws_url()
        
        # Paramètres de trading
        self.leverage = leverage
        self.margin_mode = margin_mode
        self.position_mode = position_mode
        self.default_quantity = default_quantity
        self.max_position_value = max_position_value
        self.enable_tp_sl = enable_tp_sl
        self.enable_trailing_stop = enable_trailing_stop
        self.update_interval = update_interval
        
        # Paramètres additionnels
        self.custom_params = {}
    
    def _get_default_base_url(self) -> str:
        """
        Retourne l'URL de base par défaut selon l'exchange.
        
        Returns:
            str: URL de base
        """
        if self.exchange == ExchangeType.BITGET:
            return "https://api.bitget.com"
        elif self.exchange == ExchangeType.BINANCE:
            return "https://api.binance.com"
        elif self.exchange == ExchangeType.BYBIT:
            return "https://api.bybit.com"
        else:
            return ""
    
    def _get_default_ws_url(self) -> str:
        """
        Retourne l'URL WebSocket par défaut selon l'exchange.
        
        Returns:
            str: URL WebSocket
        """
        if self.exchange == ExchangeType.BITGET:
            return "wss://ws.bitget.com/mix/v1/stream"
        elif self.exchange == ExchangeType.BINANCE:
            return "wss://stream.binance.com:9443/ws"
        elif self.exchange == ExchangeType.BYBIT:
            return "wss://stream.bybit.com/v5/public"
        else:
            return ""
    
    def to_dict(self) -> Dict:
        """
        Convertit la configuration en dictionnaire.
        
        Returns:
            Dict: Représentation dictionnaire de la configuration
        """
        return {
            "exchange": self.exchange.value,
            "symbol": self.symbol,
            "trading_mode": self.trading_mode.value,
            "api_key": "***",  # Ne pas inclure la clé API pour des raisons de sécurité
            "api_secret": "***",  # Ne pas inclure le secret API pour des raisons de sécurité
            "base_url": self.base_url,
            "ws_url": self.ws_url,
            "leverage": self.leverage,
            "margin_mode": self.margin_mode,
            "position_mode": self.position_mode,
            "default_quantity": self.default_quantity,
            "max_position_value": self.max_position_value,
            "enable_tp_sl": self.enable_tp_sl,
            "enable_trailing_stop": self.enable_trailing_stop,
            "update_interval": self.update_interval,
            "custom_params": self.custom_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LiveTraderConfig':
        """
        Crée une configuration à partir d'un dictionnaire.
        
        Args:
            data: Dictionnaire de configuration
            
        Returns:
            LiveTraderConfig: Configuration créée
        """
        exchange = ExchangeType(data.get("exchange", ExchangeType.BITGET.value))
        trading_mode = LiveTradingMode(data.get("trading_mode", LiveTradingMode.PAPER.value))
        
        config = cls(
            exchange=exchange,
            symbol=data.get("symbol", "BTCUSDT"),
            trading_mode=trading_mode,
            api_key=data.get("api_key", ""),
            api_secret=data.get("api_secret", ""),
            passphrase=data.get("passphrase"),
            base_url=data.get("base_url"),
            ws_url=data.get("ws_url"),
            leverage=data.get("leverage", 1),
            margin_mode=data.get("margin_mode", "isolated"),
            position_mode=data.get("position_mode", "one-way"),
            default_quantity=data.get("default_quantity", 0.001),
            max_position_value=data.get("max_position_value", 1000.0),
            enable_tp_sl=data.get("enable_tp_sl", True),
            enable_trailing_stop=data.get("enable_trailing_stop", False),
            update_interval=data.get("update_interval", 5)
        )
        
        if "custom_params" in data:
            config.custom_params = data["custom_params"]
        
        return config


class ExchangeAPI:
    """
    Classe abstraite pour l'interaction avec les APIs des exchanges.
    À implémenter pour chaque exchange supporté.
    """
    
    def __init__(self, config: LiveTraderConfig):
        """
        Initialise l'API de l'exchange.
        
        Args:
            config: Configuration du trader
        """
        self.config = config
        self.session = requests.Session()
        self.ws_connection = None
        self.last_response = None
    
    async def connect(self) -> bool:
        """
        Établit une connexion avec l'API de l'exchange.
        
        Returns:
            bool: Succès de la connexion
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    async def disconnect(self) -> bool:
        """
        Ferme la connexion avec l'API de l'exchange.
        
        Returns:
            bool: Succès de la déconnexion
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    async def get_account_info(self) -> Dict:
        """
        Récupère les informations du compte.
        
        Returns:
            Dict: Informations du compte
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    async def get_positions(self) -> List[Dict]:
        """
        Récupère les positions ouvertes.
        
        Returns:
            List[Dict]: Liste des positions
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    async def get_open_orders(self) -> List[Dict]:
        """
        Récupère les ordres ouverts.
        
        Returns:
            List[Dict]: Liste des ordres
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    async def place_order(
        self,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
        client_order_id: Optional[str] = None
    ) -> Dict:
        """
        Place un ordre sur l'exchange.
        
        Args:
            side: Côté de l'ordre
            order_type: Type d'ordre
            quantity: Quantité
            price: Prix (pour les ordres limite)
            stop_price: Prix de déclenchement (pour les ordres stop)
            take_profit: Prix de take profit
            stop_loss: Prix de stop loss
            reduce_only: Réduire uniquement (ne pas ouvrir de nouvelles positions)
            time_in_force: Durée de vie de l'ordre
            client_order_id: ID d'ordre côté client
            
        Returns:
            Dict: Informations de l'ordre placé
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    async def cancel_order(self, order_id: str) -> Dict:
        """
        Annule un ordre.
        
        Args:
            order_id: ID de l'ordre
            
        Returns:
            Dict: Résultat de l'annulation
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    async def cancel_all_orders(self) -> Dict:
        """
        Annule tous les ordres ouverts.
        
        Returns:
            Dict: Résultat de l'annulation
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    async def set_leverage(self, leverage: int) -> Dict:
        """
        Définit l'effet de levier.
        
        Args:
            leverage: Niveau de levier
            
        Returns:
            Dict: Résultat de l'opération
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    async def set_margin_mode(self, margin_mode: str) -> Dict:
        """
        Définit le mode de marge.
        
        Args:
            margin_mode: Mode de marge ('isolated' ou 'cross')
            
        Returns:
            Dict: Résultat de l'opération
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    async def set_position_mode(self, position_mode: str) -> Dict:
        """
        Définit le mode de position.
        
        Args:
            position_mode: Mode de position ('one-way' ou 'hedge')
            
        Returns:
            Dict: Résultat de l'opération
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    async def get_ticker(self) -> Dict:
        """
        Récupère les informations de ticker.
        
        Returns:
            Dict: Informations du ticker
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    async def get_klines(
        self,
        interval: str = "1m",
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict]:
        """
        Récupère les chandeliers japonais.
        
        Args:
            interval: Intervalle de temps
            limit: Nombre maximum d'éléments
            start_time: Timestamp de début
            end_time: Timestamp de fin
            
        Returns:
            List[Dict]: Liste des chandeliers
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    def _sign_request(self, *args, **kwargs) -> Dict:
        """
        Signe une requête API.
        
        Returns:
            Dict: En-têtes avec signature
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")


class BitgetAPI(ExchangeAPI):
    """
    Implémentation de l'API Bitget.
    """
    
    def __init__(self, config: LiveTraderConfig):
        """
        Initialise l'API Bitget.
        
        Args:
            config: Configuration du trader
        """
        super().__init__(config)
        # Initialisation spécifique à Bitget
    
    # Implémenter toutes les méthodes abstraites pour Bitget


class PaperTradingEngine:
    """
    Moteur de paper trading pour simuler le trading en temps réel.
    """
    
    def __init__(self, config: LiveTraderConfig):
        """
        Initialise le moteur de paper trading.
        
        Args:
            config: Configuration du trader
        """
        self.config = config
        self.balance = 10000.0  # Solde initial
        self.positions = []
        self.orders = []
        self.history = []
        self.last_price = 0.0
    
    # Implémenter les méthodes pour simuler le trading


class LiveTrader:
    """
    Trader en direct qui exécute des stratégies en temps réel.
    """
    
    def __init__(
        self,
        strategy_constructor: StrategyConstructor,
        config: LiveTraderConfig
    ):
        """
        Initialise le trader en direct.
        
        Args:
            strategy_constructor: Constructeur de stratégie
            config: Configuration du trader
        """
        self.strategy = strategy_constructor
        self.config = config
        
        # Sélection de l'API selon l'exchange
        if config.trading_mode == LiveTradingMode.PAPER:
            self.api = PaperTradingEngine(config)
        elif config.exchange == ExchangeType.BITGET:
            self.api = BitgetAPI(config)
        else:
            raise ValueError(f"Exchange non supporté: {config.exchange}")
        
        # État du trader
        self.is_running = False
        self.last_update = None
        self.current_data = None
        self.current_position = None
        
        # Historique et stockage
        self.trade_history = []
        self.signal_history = []
        self.performance_metrics = {}
        
        # Thread/Task pour le traitement asyncnrone
        self.main_task = None
    
    async def start(self) -> bool:
        """
        Démarre le trader en direct.
        
        Returns:
            bool: Succès du démarrage
        """
        if self.is_running:
            logger.warning("Le trader est déjà en cours d'exécution")
            return False
        
        try:
            # Connexion à l'API
            await self.api.connect()
            
            # Configuration initiale
            await self._setup_initial_config()
            
            # Démarrage de la boucle principale
            self.is_running = True
            self.main_task = asyncio.create_task(self._main_loop())
            
            logger.info(f"Trader démarré pour {self.config.symbol} sur {self.config.exchange.value}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du démarrage du trader: {str(e)}")
            return False
    
    async def stop(self) -> bool:
        """
        Arrête le trader en direct.
        
        Returns:
            bool: Succès de l'arrêt
        """
        if not self.is_running:
            logger.warning("Le trader n'est pas en cours d'exécution")
            return False
        
        try:
            # Annulation de la tâche principale
            if self.main_task:
                self.main_task.cancel()
                try:
                    await self.main_task
                except asyncio.CancelledError:
                    pass
            
            # Déconnexion de l'API
            await self.api.disconnect()
            
            self.is_running = False
            logger.info("Trader arrêté")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt du trader: {str(e)}")
            return False
    
    async def _setup_initial_config(self) -> None:
        """
        Configure les paramètres initiaux sur l'exchange.
        """
        # Définir le levier
        await self.api.set_leverage(self.config.leverage)
        
        # Définir le mode de marge
        await self.api.set_margin_mode(self.config.margin_mode)
        
        # Définir le mode de position
        await self.api.set_position_mode(self.config.position_mode)
    
    async def _main_loop(self) -> None:
        """
        Boucle principale du trader en direct.
        """
        while self.is_running:
            try:
                # Récupérer les données actuelles
                await self._update_market_data()
                
                # Mise à jour des positions
                await self._update_positions()
                
                # Évaluation de la stratégie
                await self._evaluate_strategy()
                
                # Exécution des signaux
                await self._execute_signals()
                
                # Gestion des positions ouvertes
                await self._manage_open_positions()
                
                # Délai avant la prochaine mise à jour
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle principale: {str(e)}")
                await asyncio.sleep(self.config.update_interval * 2)  # Attente plus longue en cas d'erreur
    
    async def _update_market_data(self) -> None:
        """
        Met à jour les données de marché.
        """
        # Récupérer les chandeliers pour la stratégie
        klines = await self.api.get_klines(limit=500)  # Adapter selon les besoins de la stratégie
        
        # Conversion en DataFrame
        self.current_data = self._convert_klines_to_dataframe(klines)
        
        # Mise à jour du dernier prix
        ticker = await self.api.get_ticker()
        self.last_price = float(ticker['last_price'])
        
        self.last_update = datetime.now()
    
    async def _update_positions(self) -> None:
        """
        Met à jour les informations sur les positions ouvertes.
        """
        # Récupérer les positions
        positions = await self.api.get_positions()
        
        # Mise à jour de la position courante
        self.current_position = self._find_position_for_symbol(positions)
    
    async def _evaluate_strategy(self) -> None:
        """
        Évalue la stratégie sur les données actuelles.
        """
        if self.current_data is None or len(self.current_data) == 0:
            logger.warning("Pas assez de données pour évaluer la stratégie")
            return
        
        # Générer les signaux
        signals, data_with_signals = self.strategy.generate_signals(self.current_data)
        
        # Stocker le dernier signal
        self.last_signal = signals[-1] if len(signals) > 0 else 0
        
        # Extraire les paramètres de risque
        if 'position_size' in data_with_signals.columns:
            self.position_size = float(data_with_signals['position_size'].iloc[-1])
        
        if 'sl_level' in data_with_signals.columns:
            self.sl_level = float(data_with_signals['sl_level'].iloc[-1])
        
        if 'tp_level' in data_with_signals.columns:
            self.tp_level = float(data_with_signals['tp_level'].iloc[-1])
        
        # Enregistrer le signal dans l'historique
        self.signal_history.append({
            'timestamp': datetime.now().isoformat(),
            'signal': self.last_signal,
            'price': self.last_price,
            'position_size': getattr(self, 'position_size', None),
            'sl_level': getattr(self, 'sl_level', None),
            'tp_level': getattr(self, 'tp_level', None)
        })
        
        logger.debug(f"Signal: {self.last_signal}, Prix: {self.last_price}, SL: {getattr(self, 'sl_level', None)}, TP: {getattr(self, 'tp_level', None)}")
    
    async def _execute_signals(self) -> None:
        """
        Exécute les signaux générés par la stratégie.
        """
        if not hasattr(self, 'last_signal') or self.last_signal == 0:
            return
        
        # État actuel du marché
        has_position = self.current_position is not None and self.current_position['size'] > 0
        current_side = self.current_position['side'] if has_position else None
        
        # Décision d'exécution selon le signal et la position actuelle
        if self.last_signal == 1:  # Signal d'achat
            if not has_position or current_side == PositionSide.SHORT.value:
                await self._open_long_position()
        
        elif self.last_signal == -1:  # Signal de vente
            if not has_position or current_side == PositionSide.LONG.value:
                await self._open_short_position()
    
    async def _manage_open_positions(self) -> None:
        """
        Gère les positions ouvertes (TP/SL, etc.).
        """
        if self.current_position is None or self.current_position['size'] == 0:
            return
        
        # Vérifier les ordres TP/SL existants
        orders = await self.api.get_open_orders()
        has_tp = any(o['type'] == OrderType.TAKE_PROFIT.value for o in orders)
        has_sl = any(o['type'] == OrderType.STOP.value for o in orders)
        
        # Mettre à jour les TP/SL si nécessaire
        if self.config.enable_tp_sl:
            position_side = self.current_position['side']
            
            if not has_tp and hasattr(self, 'tp_level'):
                # Calculer le prix TP
                tp_price = self._calculate_tp_price(position_side)
                await self._place_tp_order(position_side, tp_price)
            
            if not has_sl and hasattr(self, 'sl_level'):
                # Calculer le prix SL
                sl_price = self._calculate_sl_price(position_side)
                await self._place_sl_order(position_side, sl_price)
    
    async def _open_long_position(self) -> None:
        """
        Ouvre une position longue.
        """
        # Vérifier s'il y a déjà une position longue
        if (self.current_position is not None and 
            self.current_position['side'] == PositionSide.LONG.value and 
            self.current_position['size'] > 0):
            logger.debug("Position longue déjà ouverte")
            return
        
        # Fermer la position courte existante si nécessaire
        if (self.current_position is not None and 
            self.current_position['side'] == PositionSide.SHORT.value):
            await self._close_position()
        
        # Calculer la quantité
        quantity = self._calculate_position_size(OrderSide.BUY)
        
        # Placer l'ordre d'achat
        order_result = await self.api.place_order(
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity
        )
        
        # Enregistrer le trade
        self._record_trade(OrderSide.BUY, quantity, self.last_price)
        
        logger.info(f"Position longue ouverte: {quantity} @ {self.last_price}")
    
    async def _open_short_position(self) -> None:
        """
        Ouvre une position courte.
        """
        # Vérifier s'il y a déjà une position courte
        if (self.current_position is not None and 
            self.current_position['side'] == PositionSide.SHORT.value and 
            self.current_position['size'] > 0):
            logger.debug("Position courte déjà ouverte")
            return
        
        # Fermer la position longue existante si nécessaire
        if (self.current_position is not None and 
            self.current_position['side'] == PositionSide.LONG.value):
            await self._close_position()
        
        # Calculer la quantité
        quantity = self._calculate_position_size(OrderSide.SELL)
        
        # Placer l'ordre de vente
        order_result = await self.api.place_order(
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity
        )
        
        # Enregistrer le trade
        self._record_trade(OrderSide.SELL, quantity, self.last_price)
        
        logger.info(f"Position courte ouverte: {quantity} @ {self.last_price}")
    
    async def _close_position(self) -> None:
        """
        Ferme la position actuelle.
        """
        if self.current_position is None or self.current_position['size'] == 0:
            logger.debug("Aucune position à fermer")
            return
        
        position_side = self.current_position['side']
        quantity = self.current_position['size']
        
        # Définir le côté de l'ordre pour fermer la position
        close_side = OrderSide.SELL if position_side == PositionSide.LONG.value else OrderSide.BUY
        
        # Placer l'ordre pour fermer la position
        order_result = await self.api.place_order(
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            reduce_only=True
        )
        
        # Enregistrer le trade
        self._record_trade(close_side, quantity, self.last_price, is_close=True)
        
        logger.info(f"Position fermée: {quantity} @ {self.last_price}")
    
    async def _place_tp_order(self, position_side: str, tp_price: float) -> None:
        """
        Place un ordre Take Profit.
        
        Args:
            position_side: Côté de la position
            tp_price: Prix du Take Profit
        """
        if not self.config.enable_tp_sl:
            return
        
        # Définir le côté pour l'ordre TP
        tp_side = OrderSide.SELL if position_side == PositionSide.LONG.value else OrderSide.BUY
        
        # Quantité de la position
        quantity = self.current_position['size']
        
        # Placer l'ordre TP
        await self.api.place_order(
            side=tp_side,
            order_type=OrderType.TAKE_PROFIT,
            quantity=quantity,
            price=tp_price,
            reduce_only=True
        )
        
        logger.info(f"Ordre TP placé: {quantity} @ {tp_price}")
    
    async def _place_sl_order(self, position_side: str, sl_price: float) -> None:
        """
        Place un ordre Stop Loss.
        
        Args:
            position_side: Côté de la position
            sl_price: Prix du Stop Loss
        """
        if not self.config.enable_tp_sl:
            return
        
        # Définir le côté pour l'ordre SL
        sl_side = OrderSide.SELL if position_side == PositionSide.LONG.value else OrderSide.BUY
        
        # Quantité de la position
        quantity = self.current_position['size']
        
        # Placer l'ordre SL
        await self.api.place_order(
            side=sl_side,
            order_type=OrderType.STOP,
            quantity=quantity,
            stop_price=sl_price,
            reduce_only=True
        )
        
        logger.info(f"Ordre SL placé: {quantity} @ {sl_price}")
    
    def _calculate_position_size(self, side: OrderSide) -> float:
        """
        Calcule la taille de position.
        
        Args:
            side: Côté de l'ordre
            
        Returns:
            float: Quantité à trader
        """
        # Récupérer les paramètres
        account_balance = 10000.0  # À remplacer par la balance réelle
        
        # Utiliser la taille de position de la stratégie si disponible
        if hasattr(self, 'position_size'):
            position_value = account_balance * self.position_size * self.config.leverage
        else:
            position_value = account_balance * 0.1 * self.config.leverage  # 10% par défaut
        
        # Limiter la valeur de la position
        position_value = min(position_value, self.config.max_position_value)
        
        # Calculer la quantité en fonction du prix actuel
        quantity = position_value / self.last_price
        
        # Arrondir à la précision du symbole (à adapter selon l'exchange)
        precision = 5  # Bitget BTC/USDT est généralement 5 décimales
        quantity = round(quantity, precision)
        
        return max(quantity, self.config.default_quantity)
    
    def _calculate_tp_price(self, position_side: str) -> float:
        """
        Calcule le prix du Take Profit.
        
        Args:
            position_side: Côté de la position
            
        Returns:
            float: Prix du Take Profit
        """
        tp_pct = getattr(self, 'tp_level', 0.02)  # 2% par défaut
        
        if position_side == PositionSide.LONG.value:
            tp_price = self.last_price * (1 + tp_pct)
        else:
            tp_price = self.last_price * (1 - tp_pct)
        
        return tp_price
    
    def _calculate_sl_price(self, position_side: str) -> float:
        """
        Calcule le prix du Stop Loss.
        
        Args:
            position_side: Côté de la position
            
        Returns:
            float: Prix du Stop Loss
        """
        sl_pct = getattr(self, 'sl_level', 0.01)  # 1% par défaut
        
        if position_side == PositionSide.LONG.value:
            sl_price = self.last_price * (1 - sl_pct)
        else:
            sl_price = self.last_price * (1 + sl_pct)
        
        return sl_price
    
    def _record_trade(
        self, 
        side: OrderSide, 
        quantity: float, 
        price: float, 
        is_close: bool = False
    ) -> None:
        """
        Enregistre un trade dans l'historique.
        
        Args:
            side: Côté de l'ordre
            quantity: Quantité
            price: Prix
            is_close: True si c'est une clôture de position
        """
        trade = {
            'timestamp': datetime.now().isoformat(),
            'side': side.value,
            'quantity': quantity,
            'price': price,
            'value': quantity * price,
            'is_close': is_close,
            'fees': quantity * price * 0.001  # Frais estimés à 0.1%
        }
        
        self.trade_history.append(trade)
    
    def _find_position_for_symbol(self, positions: List[Dict]) -> Optional[Dict]:
        """
        Trouve la position pour le symbole actuel.
        
        Args:
            positions: Liste des positions
            
        Returns:
            Optional[Dict]: Position trouvée ou None
        """
        for position in positions:
            if position['symbol'] == self.config.symbol:
                return position
        return None
    
    def _convert_klines_to_dataframe(self, klines: List[Dict]) -> pd.DataFrame:
        """
        Convertit les chandeliers en DataFrame.
        
        Args:
            klines: Liste des chandeliers
            
        Returns:
            pd.DataFrame: DataFrame avec les données OHLCV
        """
        # Adapter selon le format de l'exchange
        data = {
            'timestamp': [],
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        }
        
        for kline in klines:
            data['timestamp'].append(kline['timestamp'])
            data['open'].append(float(kline['open']))
            data['high'].append(float(kline['high']))
            data['low'].append(float(kline['low']))
            data['close'].append(float(kline['close']))
            data['volume'].append(float(kline['volume']))
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_performance_summary(self) -> Dict:
        """
        Retourne un résumé des performances du trader.
        
        Returns:
            Dict: Résumé des performances
        """
        # Calculer les métriques de performance à partir de l'historique des trades
        if not self.trade_history:
            return {
                "trades": 0,
                "win_rate": 0,
                "profit": 0,
                "roi": 0
            }
        
        # Nombre de trades
        total_trades = len([t for t in self.trade_history if t['is_close']])
        
        # Profit total
        profit = 0.0
        wins = 0
        
        for i in range(len(self.trade_history)):
            trade = self.trade_history[i]
            
            if trade['is_close']:
                # Trouver le trade d'ouverture correspondant
                for j in range(i-1, -1, -1):
                    open_trade = self.trade_history[j]
                    if not open_trade['is_close'] and open_trade['side'] != trade['side']:
                        # Calculer le P&L
                        if open_trade['side'] == OrderSide.BUY.value:  # Long
                            pnl = (trade['price'] - open_trade['price']) * trade['quantity']
                        else:  # Short
                            pnl = (open_trade['price'] - trade['price']) * trade['quantity']
                        
                        # Soustraire les frais
                        pnl -= trade['fees'] + open_trade['fees']
                        
                        profit += pnl
                        wins += 1 if pnl > 0 else 0
                        break
        
        # Calculer le ROI
        initial_balance = 10000.0  # À remplacer par la balance initiale réelle
        roi = profit / initial_balance if initial_balance > 0 else 0
        
        # Calculer le win rate
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        return {
            "trades": total_trades,
            "win_rate": win_rate,
            "profit": profit,
            "roi": roi,
            "current_position": self.current_position
        }
    
    def save_history(self, output_dir: str) -> bool:
        """
        Sauvegarde l'historique des trades et signaux.
        
        Args:
            output_dir: Répertoire de sortie
            
        Returns:
            bool: Succès de la sauvegarde
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Sauvegarder l'historique des trades
            if self.trade_history:
                trade_df = pd.DataFrame(self.trade_history)
                trade_df.to_csv(os.path.join(output_dir, "trade_history.csv"), index=False)
            
            # Sauvegarder l'historique des signaux
            if self.signal_history:
                signal_df = pd.DataFrame(self.signal_history)
                signal_df.to_csv(os.path.join(output_dir, "signal_history.csv"), index=False)
            
            # Sauvegarder la configuration
            with open(os.path.join(output_dir, "config.json"), 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            
            # Sauvegarder le résumé des performances
            performance = self.get_performance_summary()
            with open(os.path.join(output_dir, "performance.json"), 'w', encoding='utf-8') as f:
                json.dump(performance, f, indent=2)
            
            logger.info(f"Historique sauvegardé dans {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'historique: {str(e)}")
            return False


# Fonction utilitaire pour faciliter la création d'un trader en direct
async def create_live_trader(
    strategy_id: str,
    exchange: ExchangeType = ExchangeType.BITGET,
    symbol: str = "BTCUSDT",
    api_key: str = "",
    api_secret: str = "",
    trading_mode: LiveTradingMode = LiveTradingMode.PAPER
) -> LiveTrader:
    """
    Crée un trader en direct avec une stratégie existante.
    
    Args:
        strategy_id: ID de la stratégie à utiliser
        exchange: Type d'exchange
        symbol: Paire de trading
        api_key: Clé API
        api_secret: Secret API
        trading_mode: Mode de trading
        
    Returns:
        LiveTrader: Instance du trader en direct
    """
    # Charger la stratégie
    manager = StrategyManager()
    constructor = manager.load_strategy(strategy_id)
    
    # Créer la configuration du trader
    config = LiveTraderConfig(
        exchange=exchange,
        symbol=symbol,
        trading_mode=trading_mode,
        api_key=api_key,
        api_secret=api_secret
    )
    
    # Créer le trader
    trader = LiveTrader(constructor, config)
    
    return trader