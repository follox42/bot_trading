"""
Implémentation de l'interface de l'exchange pour Bitget.
"""

import json
import time
import hmac
import hashlib
import base64
import requests
import websocket
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

from .exchange_interface import (
    ExchangeInterface, OrderType, OrderSide, OrderStatus, PositionSide
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bitget_api")


class BitgetAPI(ExchangeInterface):
    """Implémentation de l'API Bitget"""
    
    # URLs de base
    BASE_URL = "https://api.bitget.com"
    TESTNET_URL = "https://api-simulated.bitget.com"
    WS_URL = "wss://ws.bitget.com/mix/v1/stream"
    WS_TESTNET_URL = "wss://ws-simulated.bitget.com/mix/v1/stream"
    
    # Mappings de conversion
    ORDER_TYPE_MAP = {
        OrderType.MARKET: "market",
        OrderType.LIMIT: "limit",
        OrderType.STOP: "stop_market",
        OrderType.STOP_LIMIT: "stop_limit",
        OrderType.TAKE_PROFIT: "take_profit_market",
        OrderType.TRAILING_STOP: "trailing_stop"
    }
    
    ORDER_SIDE_MAP = {
        OrderSide.BUY: "buy",
        OrderSide.SELL: "sell"
    }
    
    INTERVAL_MAP = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "12h": "12h",
        "1d": "1d",
        "3d": "3d",
        "1w": "1w"
    }
    
    def __init__(self, api_key: str, api_secret: str, passphrase: str = "", testnet: bool = False):
        """
        Initialise l'API Bitget.
        
        Args:
            api_key: Clé API
            api_secret: Secret API
            passphrase: Phrase secrète (obligatoire pour Bitget)
            testnet: Utiliser le réseau de test
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.testnet = testnet
        
        # Sélectionner l'URL appropriée
        self.base_url = self.TESTNET_URL if testnet else self.BASE_URL
        self.ws_url = self.WS_TESTNET_URL if testnet else self.WS_URL
        
        # Session HTTP pour les requêtes REST
        self.session = requests.Session()
        
        # Initialiser les connexions WebSocket si nécessaire
        self.ws = None
        self.ws_thread = None
        self.ws_callbacks = {}
        
        # Caches pour les données fréquemment utilisées
        self.symbols_info = {}
        self.last_server_time = None
        
        # Vérifier les identifiants si fournis
        if api_key and api_secret and passphrase:
            try:
                account_info = self.get_account_info()
                logger.info(f"Connexion à Bitget réussie. Mode: {'Testnet' if testnet else 'Production'}")
            except Exception as e:
                logger.error(f"Erreur lors de la connexion à Bitget: {str(e)}")
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """
        Génère la signature pour l'authentification Bitget.
        
        Args:
            timestamp: Horodatage
            method: Méthode HTTP (GET, POST, etc.)
            request_path: Chemin de la requête
            body: Corps de la requête
            
        Returns:
            str: Signature encodée en base64
        """
        message = timestamp + method + request_path + body
        mac = hmac.new(
            bytes(self.api_secret, encoding="utf-8"),
            bytes(message, encoding="utf-8"),
            digestmod=hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode("utf-8")
    
    def _request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """
        Envoie une requête à l'API Bitget.
        
        Args:
            method: Méthode HTTP
            endpoint: Point de terminaison de l'API
            params: Paramètres de la requête (pour GET)
            data: Données de la requête (pour POST)
            
        Returns:
            Dict: Réponse de l'API
        """
        url = f"{self.base_url}{endpoint}"
        
        # Préparer le body si nécessaire
        body = ""
        if data:
            body = json.dumps(data)
        
        # Générer les en-têtes
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, endpoint, body)
        
        headers = {
            "Content-Type": "application/json",
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.passphrase,
            "locale": "en-US"
        }
        
        if self.testnet:
            headers["x-simulated-trading"] = "1"
        
        try:
            if method == "GET":
                response = self.session.get(url, headers=headers, params=params)
            elif method == "POST":
                response = self.session.post(url, headers=headers, data=body)
            elif method == "DELETE":
                response = self.session.delete(url, headers=headers, params=params)
            else:
                raise ValueError(f"Méthode non supportée: {method}")
            
            # Vérifier le code de statut
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                response.raise_for_status()
            
            # Parser la réponse
            result = response.json()
            
            # Vérifier le code de réponse de l'API
            if result.get("code") != "00000":
                logger.error(f"API error: {result.get('code')} - {result.get('msg')}")
                raise Exception(f"API error: {result.get('msg')}")
            
            return result.get("data", {})
            
        except Exception as e:
            logger.error(f"Erreur de requête {method} {endpoint}: {str(e)}")
            raise
    
    def get_server_time(self) -> int:
        """
        Obtient l'horodatage du serveur.
        
        Returns:
            int: Timestamp en millisecondes
        """
        try:
            result = self._request("GET", "/api/spot/v1/public/time")
            self.last_server_time = int(result)
            return self.last_server_time
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du temps serveur: {str(e)}")
            # Fallback sur le temps local
            return int(time.time() * 1000)
    
    def get_account_info(self) -> Dict:
        """
        Récupère les informations du compte.
        
        Returns:
            Dict: Informations du compte
        """
        try:
            result = self._request("GET", "/api/mix/v1/account/account")
            return result
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des informations du compte: {str(e)}")
            raise
    
    def get_balance(self, asset: str = "USDT") -> float:
        """
        Récupère le solde d'un actif.
        
        Args:
            asset: Symbole de l'actif
            
        Returns:
            float: Solde disponible
        """
        try:
            result = self._request("GET", "/api/mix/v1/account/accounts", params={"productType": "umcbl"})
            
            for account in result:
                if account.get("marginCoin") == asset:
                    return float(account.get("available", 0))
            
            return 0.0
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du solde de {asset}: {str(e)}")
            raise
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Récupère les positions ouvertes.
        
        Args:
            symbol: Symbole de la paire (optionnel)
            
        Returns:
            List[Dict]: Liste des positions
        """
        try:
            params = {"productType": "umcbl"}
            if symbol:
                params["symbol"] = self.normalize_symbol(symbol)
                
            result = self._request("GET", "/api/mix/v1/position/allPosition", params=params)
            
            # Convertir les positions au format standard
            positions = []
            for pos in result:
                position = self.to_standard_format(pos, "position")
                positions.append(position)
            
            return positions
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des positions: {str(e)}")
            raise
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Récupère les ordres ouverts.
        
        Args:
            symbol: Symbole de la paire (optionnel)
            
        Returns:
            List[Dict]: Liste des ordres
        """
        try:
            params = {"productType": "umcbl"}
            if symbol:
                params["symbol"] = self.normalize_symbol(symbol)
                
            result = self._request("GET", "/api/mix/v1/order/current", params=params)
            
            # Convertir les ordres au format standard
            orders = []
            for order in result:
                std_order = self.to_standard_format(order, "order")
                orders.append(std_order)
            
            return orders
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des ordres ouverts: {str(e)}")
            raise
    
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
        Place un ordre sur Bitget.
        
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
            **kwargs: Paramètres additionnels
            
        Returns:
            Dict: Informations de l'ordre placé
        """
        try:
            # Normaliser le symbole
            symbol = self.normalize_symbol(symbol)
            
            # Préparer les données de base
            data = {
                "symbol": symbol,
                "marginCoin": "USDT",
                "size": str(quantity),
                "side": self.ORDER_SIDE_MAP[side],
                "orderType": self.ORDER_TYPE_MAP[order_type],
                "timeInForceValue": kwargs.get("time_in_force", "normal"),
                "clientOid": client_order_id or str(int(time.time() * 1000))
            }
            
            # Ajouter le prix si nécessaire
            if price is not None and order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                data["price"] = str(price)
            
            # Ajouter le prix de déclenchement pour les ordres stop
            if stop_price is not None and order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                data["triggerPrice"] = str(stop_price)
            
            # Réduire uniquement
            if reduce_only:
                data["reduceOnly"] = "true"
            
            # Placer l'ordre
            endpoint = "/api/mix/v1/order/placeOrder"
            if take_profit is not None or stop_loss is not None:
                endpoint = "/api/mix/v1/order/placePlanOrder"
                # Ajouter TP/SL pour les ordres plan
                if take_profit is not None:
                    data["presetTakeProfitPrice"] = str(take_profit)
                if stop_loss is not None:
                    data["presetStopLossPrice"] = str(stop_loss)
            
            result = self._request("POST", endpoint, data=data)
            
            # Convertir la réponse au format standard
            return self.to_standard_format(result, "order")
        except Exception as e:
            logger.error(f"Erreur lors du placement de l'ordre: {str(e)}")
            raise
    
    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """
        Annule un ordre.
        
        Args:
            symbol: Symbole de la paire
            order_id: ID de l'ordre
            
        Returns:
            Dict: Résultat de l'annulation
        """
        try:
            symbol = self.normalize_symbol(symbol)
            
            data = {
                "symbol": symbol,
                "marginCoin": "USDT",
                "orderId": order_id
            }
            
            result = self._request("POST", "/api/mix/v1/order/cancel-order", data=data)
            return result
        except Exception as e:
            logger.error(f"Erreur lors de l'annulation de l'ordre: {str(e)}")
            raise
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict:
        """
        Annule tous les ordres ouverts.
        
        Args:
            symbol: Symbole de la paire (optionnel)
            
        Returns:
            Dict: Résultat de l'annulation
        """
        try:
            data = {
                "productType": "umcbl",
                "marginCoin": "USDT"
            }
            
            if symbol:
                data["symbol"] = self.normalize_symbol(symbol)
            
            result = self._request("POST", "/api/mix/v1/order/cancel-all-orders", data=data)
            return result
        except Exception as e:
            logger.error(f"Erreur lors de l'annulation de tous les ordres: {str(e)}")
            raise
    
    def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """
        Définit l'effet de levier.
        
        Args:
            symbol: Symbole de la paire
            leverage: Niveau de levier
            
        Returns:
            Dict: Résultat de l'opération
        """
        try:
            symbol = self.normalize_symbol(symbol)
            
            data = {
                "symbol": symbol,
                "marginCoin": "USDT",
                "leverage": str(leverage),
                "holdSide": "long_short"  # Appliquer aux deux côtés
            }
            
            result = self._request("POST", "/api/mix/v1/account/setLeverage", data=data)
            return result
        except Exception as e:
            logger.error(f"Erreur lors de la définition du levier: {str(e)}")
            raise
    
    def set_margin_mode(self, symbol: str, margin_mode: str) -> Dict:
        """
        Définit le mode de marge.
        
        Args:
            symbol: Symbole de la paire
            margin_mode: Mode de marge ('isolated' ou 'cross')
            
        Returns:
            Dict: Résultat de l'opération
        """
        try:
            symbol = self.normalize_symbol(symbol)
            
            # Bitget utilise "fixed" pour isolated et "crossed" pour cross
            bitget_margin_mode = "fixed" if margin_mode.lower() == "isolated" else "crossed"
            
            data = {
                "symbol": symbol,
                "marginCoin": "USDT",
                "marginMode": bitget_margin_mode
            }
            
            result = self._request("POST", "/api/mix/v1/account/setMarginMode", data=data)
            return result
        except Exception as e:
            logger.error(f"Erreur lors de la définition du mode de marge: {str(e)}")
            raise
    
    def set_position_mode(self, position_mode: str) -> Dict:
        """
        Définit le mode de position.
        
        Args:
            position_mode: Mode de position ('one-way' ou 'hedge')
            
        Returns:
            Dict: Résultat de l'opération
        """
        try:
            # Bitget utilise "single_hold" pour one-way et "double_hold" pour hedge
            bitget_position_mode = "single_hold" if position_mode.lower() == "one-way" else "double_hold"
            
            data = {
                "productType": "umcbl",
                "holdMode": bitget_position_mode
            }
            
            result = self._request("POST", "/api/mix/v1/account/setPositionMode", data=data)
            return result
        except Exception as e:
            logger.error(f"Erreur lors de la définition du mode de position: {str(e)}")
            raise
    
    def get_ticker(self, symbol: str) -> Dict:
        """
        Récupère les informations de ticker.
        
        Args:
            symbol: Symbole de la paire
            
        Returns:
            Dict: Informations du ticker
        """
        try:
            symbol = self.normalize_symbol(symbol)
            
            result = self._request("GET", "/api/mix/v1/market/ticker", params={"symbol": symbol})
            
            # Convertir au format standard
            return self.to_standard_format(result, "ticker")
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du ticker: {str(e)}")
            raise
    
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
            interval: Intervalle de temps
            limit: Nombre maximum d'éléments
            start_time: Timestamp de début en millisecondes
            end_time: Timestamp de fin en millisecondes
            
        Returns:
            List[Dict]: Liste des chandeliers
        """
        try:
            symbol = self.normalize_symbol(symbol)
            
            # Bitget utilise un format différent pour l'intervalle
            bitget_interval = self.INTERVAL_MAP.get(interval, interval)
            
            params = {
                "symbol": symbol,
                "granularity": bitget_interval,
                "limit": min(limit, 1000)  # Bitget limite à 1000 bougies par requête
            }
            
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time
            
            result = self._request("GET", "/api/mix/v1/market/candles", params=params)
            
            # Convertir au format standard
            return self.to_standard_format(result, "klines")
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des klines: {str(e)}")
            raise
    
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
        try:
            symbol = self.normalize_symbol(symbol)
            
            # Calculer la période de temps
            end_time = int(time.time() * 1000)
            start_time = end_time - (lookback_days * 24 * 60 * 60 * 1000)
            
            # Bitget limite à 1000 bougies par requête, donc on doit faire plusieurs requêtes
            all_klines = []
            current_end_time = end_time
            
            while len(all_klines) < limit and current_end_time > start_time:
                # Faire une requête
                klines = self.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=min(limit - len(all_klines), 1000),
                    end_time=current_end_time
                )
                
                if not klines:
                    break
                
                # Ajouter les klines au résultat
                all_klines = klines + all_klines
                
                # Mettre à jour l'horodatage de fin pour la prochaine requête
                current_end_time = int(klines[0]['timestamp']) - 1
                
                # Respecter les limites de l'API
                time.sleep(0.5)
            
            # Convertir en DataFrame
            if not all_klines:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            df = pd.DataFrame(all_klines)
            
            # Trier par horodatage
            df.sort_values('timestamp', inplace=True)
            
            # Limiter au nombre demandé
            if len(df) > limit:
                df = df.tail(limit)
            
            # Convertir l'horodatage en datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des klines historiques: {str(e)}")
            raise
    
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
        try:
            symbol = self.normalize_symbol(symbol)
            
            # Déterminer le côté des ordres
            tp_sl_side = OrderSide.SELL if position_side == PositionSide.LONG else OrderSide.BUY
            
            tp_order = None
            sl_order = None
            
            # Placer l'ordre TP si spécifié
            if take_profit_price is not None:
                tp_order = self.place_order(
                    symbol=symbol,
                    side=tp_sl_side,
                    order_type=OrderType.TAKE_PROFIT,
                    quantity=quantity,
                    price=take_profit_price,
                    reduce_only=True
                )
            
            # Placer l'ordre SL si spécifié
            if stop_loss_price is not None:
                sl_order = self.place_order(
                    symbol=symbol,
                    side=tp_sl_side,
                    order_type=OrderType.STOP,
                    quantity=quantity,
                    stop_price=stop_loss_price,
                    reduce_only=True
                )
            
            return tp_order, sl_order
        except Exception as e:
            logger.error(f"Erreur lors du placement des ordres TP/SL: {str(e)}")
            raise
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalise le format du symbole pour Bitget.
        
        Args:
            symbol: Symbole à normaliser
            
        Returns:
            str: Symbole normalisé
        """
        # Bitget utilise un format spécifique: BTCUSDT_UMCBL pour BTC/USDT
        symbol = symbol.upper().replace('/', '')
        
        if "_UMCBL" not in symbol:
            symbol = f"{symbol}_UMCBL"
        
        return symbol
    
    def to_standard_format(self, data: Any, data_type: str) -> Any:
        """
        Convertit les données spécifiques à Bitget en format standard.
        
        Args:
            data: Données à convertir
            data_type: Type de données ('ticker', 'klines', 'order', etc.)
            
        Returns:
            Any: Données au format standard
        """
        if data_type == "ticker":
            # Format Bitget: {'symbol': 'BTCUSDT_UMCBL', 'last': '28001.07', ...}
            return {
                'symbol': data.get('symbol', '').replace('_UMCBL', ''),
                'last_price': float(data.get('last', 0)),
                'bid_price': float(data.get('bestBid', 0)),
                'ask_price': float(data.get('bestAsk', 0)),
                'high_price': float(data.get('high24h', 0)),
                'low_price': float(data.get('low24h', 0)),
                'volume': float(data.get('volume24h', 0)),
                'timestamp': int(data.get('timestamp', 0))
            }
        
        elif data_type == "klines":
            # Format Bitget: [['1625836800000', '33111.11', '33222.22', '33000', '33111.11', '100', '33111.11'], ...]
            result = []
            
            for kline in data:
                # Kline est une liste avec [timestamp, open, high, low, close, volume, ...]
                # Vérifier si c'est le bon format
                if isinstance(kline, list) and len(kline) >= 6:
                    result.append({
                        'timestamp': int(kline[0]),
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5])
                    })
                elif isinstance(kline, dict):
                    # Format alternatif possible
                    result.append({
                        'timestamp': int(kline.get('timestamp', 0)),
                        'open': float(kline.get('open', 0)),
                        'high': float(kline.get('high', 0)),
                        'low': float(kline.get('low', 0)),
                        'close': float(kline.get('close', 0)),
                        'volume': float(kline.get('volume', 0))
                    })
            
            return result
        
        elif data_type == "order":
            # Convertir les états d'ordre Bitget en standard
            status_map = {
                "new": OrderStatus.NEW,
                "partial-fill": OrderStatus.PARTIALLY_FILLED,
                "full-fill": OrderStatus.FILLED,
                "cancelled": OrderStatus.CANCELED,
                "filled": OrderStatus.FILLED
            }
            
            # Déterminer le statut
            bitget_status = data.get('state', '').lower()
            status = status_map.get(bitget_status, OrderStatus.NEW)
            
            # Déterminer le côté
            side = OrderSide.BUY if data.get('side', '').lower() == 'buy' else OrderSide.SELL
            
            return {
                'order_id': data.get('orderId', ''),
                'client_order_id': data.get('clientOid', ''),
                'symbol': data.get('symbol', '').replace('_UMCBL', ''),
                'side': side,
                'type': data.get('orderType', ''),
                'price': float(data.get('price', 0)),
                'quantity': float(data.get('size', 0)),
                'executed_quantity': float(data.get('filledQty', 0)),
                'status': status,
                'timestamp': int(data.get('cTime', 0)),
                'reduce_only': data.get('reduceOnly', False)
            }
        
        elif data_type == "position":
            # Convertir le côté de position Bitget en standard
            side_map = {
                "long": PositionSide.LONG,
                "short": PositionSide.SHORT
            }
            
            # Déterminer le côté de la position
            bitget_side = data.get('holdSide', '').lower()
            side = side_map.get(bitget_side, PositionSide.LONG)
            
            return {
                'symbol': data.get('symbol', '').replace('_UMCBL', ''),
                'side': side,
                'size': float(data.get('total', 0)),
                'entry_price': float(data.get('averageOpenPrice', 0)),
                'mark_price': float(data.get('marketPrice', 0)),
                'liquidation_price': float(data.get('liquidationPrice', 0)),
                'unrealized_pnl': float(data.get('unrealizedPL', 0)),
                'margin': float(data.get('margin', 0)),
                'leverage': float(data.get('leverage', 1)),
                'timestamp': int(time.time() * 1000)  # Bitget ne fournit pas ceci, donc on utilise le temps actuel
            }
        
        # Pour tout autre type de données, retourner tel quel
        return data