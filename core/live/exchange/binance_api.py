"""
Implémentation de l'interface de l'exchange pour Binance.
"""

import json
import time
import hmac
import hashlib
import requests
import websocket
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import urllib.parse

from .exchange_interface import (
    ExchangeInterface, OrderType, OrderSide, OrderStatus, PositionSide
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("binance_api")


class BinanceAPI(ExchangeInterface):
    """Implémentation de l'API Binance"""
    
    # URLs de base
    BASE_URL = "https://api.binance.com"
    TESTNET_URL = "https://testnet.binance.vision"
    BASE_URL_FUTURES = "https://fapi.binance.com"
    TESTNET_URL_FUTURES = "https://testnet.binancefuture.com"
    WS_URL = "wss://stream.binance.com:9443/ws"
    WS_URL_FUTURES = "wss://fstream.binance.com/ws"
    WS_TESTNET_URL_FUTURES = "wss://stream.binancefuture.com/ws"
    
    # Mappings de conversion
    ORDER_TYPE_MAP = {
        OrderType.MARKET: "MARKET",
        OrderType.LIMIT: "LIMIT",
        OrderType.STOP: "STOP_MARKET",
        OrderType.STOP_LIMIT: "STOP",
        OrderType.TAKE_PROFIT: "TAKE_PROFIT_MARKET",
        OrderType.TRAILING_STOP: "TRAILING_STOP_MARKET"
    }
    
    ORDER_SIDE_MAP = {
        OrderSide.BUY: "BUY",
        OrderSide.SELL: "SELL"
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
        "8h": "8h",
        "12h": "12h",
        "1d": "1d",
        "3d": "3d",
        "1w": "1w",
        "1M": "1M"
    }
    
    def __init__(self, api_key: str, api_secret: str, passphrase: str = "", testnet: bool = False):
        """
        Initialise l'API Binance.
        
        Args:
            api_key: Clé API
            api_secret: Secret API
            passphrase: Non utilisé pour Binance (gardé pour compatibilité avec l'interface)
            testnet: Utiliser le réseau de test
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Sélectionner les URLs appropriées
        self.base_url = self.TESTNET_URL_FUTURES if testnet else self.BASE_URL_FUTURES
        self.ws_url = self.WS_TESTNET_URL_FUTURES if testnet else self.WS_URL_FUTURES
        
        # Session HTTP pour les requêtes REST
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})
        
        # Initialiser les connexions WebSocket si nécessaire
        self.ws = None
        self.ws_thread = None
        self.ws_callbacks = {}
        
        # Caches pour les données fréquemment utilisées
        self.symbols_info = {}
        self.last_server_time = None
        
        # Vérifier les identifiants si fournis
        if api_key and api_secret:
            try:
                self.get_server_time()
                logger.info(f"Connexion à Binance réussie. Mode: {'Testnet' if testnet else 'Production'}")
            except Exception as e:
                logger.error(f"Erreur lors de la connexion à Binance: {str(e)}")
    
    def _generate_signature(self, params: Dict) -> str:
        """
        Génère la signature pour l'authentification Binance.
        
        Args:
            params: Paramètres de la requête
            
        Returns:
            str: Signature HMAC SHA256
        """
        # Convertir les paramètres en chaîne de requête
        query_string = urllib.parse.urlencode(params)
        
        # Générer la signature HMAC SHA256
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _request(self, method: str, endpoint: str, signed: bool = False, params: Dict = None, data: Dict = None) -> Any:
        """
        Envoie une requête à l'API Binance.
        
        Args:
            method: Méthode HTTP
            endpoint: Point de terminaison de l'API
            signed: Requête signée ou non
            params: Paramètres de la requête
            data: Données de la requête (pour POST)
            
        Returns:
            Any: Réponse de l'API
        """
        url = f"{self.base_url}{endpoint}"
        
        # Préparer les paramètres
        params = params or {}
        
        # Ajouter le timestamp pour les requêtes signées
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
        
        try:
            if method == "GET":
                response = self.session.get(url, params=params)
            elif method == "POST":
                response = self.session.post(url, params=params, json=data)
            elif method == "DELETE":
                response = self.session.delete(url, params=params)
            else:
                raise ValueError(f"Méthode non supportée: {method}")
            
            # Vérifier le code de statut
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                response.raise_for_status()
            
            # Parser la réponse
            return response.json()
            
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
            result = self._request("GET", "/fapi/v1/time")
            self.last_server_time = result.get('serverTime', int(time.time() * 1000))
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
            result = self._request("GET", "/fapi/v2/account", signed=True)
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
            result = self._request("GET", "/fapi/v2/account", signed=True)
            
            for balance in result.get('assets', []):
                if balance.get('asset') == asset:
                    return float(balance.get('availableBalance', 0))
            
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
            result = self._request("GET", "/fapi/v2/account", signed=True)
            
            # Extraire les positions
            positions = []
            for pos in result.get('positions', []):
                # Ne prendre que les positions avec une quantité non nulle
                if float(pos.get('positionAmt', 0)) != 0:
                    if symbol and pos.get('symbol') != symbol:
                        continue
                    
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
            params = {}
            if symbol:
                params['symbol'] = symbol
            
            result = self._request("GET", "/fapi/v1/openOrders", signed=True, params=params)
            
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
        Place un ordre sur Binance.
        
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
            # Préparer les paramètres de base
            params = {
                'symbol': symbol,
                'side': self.ORDER_SIDE_MAP[side],
                'type': self.ORDER_TYPE_MAP[order_type],
                'quantity': quantity,
                'timestamp': int(time.time() * 1000)
            }
            
            # Ajouter le prix si nécessaire
            if price is not None and order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                params['price'] = price
            
            # Ajouter le prix de déclenchement pour les ordres stop
            if stop_price is not None and order_type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT]:
                params['stopPrice'] = stop_price
            
            # Réduire uniquement
            if reduce_only:
                params['reduceOnly'] = 'true'
            
            # ID d'ordre client
            if client_order_id:
                params['newClientOrderId'] = client_order_id
            
            # Paramètres additionnels
            if 'time_in_force' in kwargs:
                params['timeInForce'] = kwargs['time_in_force']
            elif order_type == OrderType.LIMIT:
                params['timeInForce'] = 'GTC'  # Good Till Cancel par défaut
            
            # Placer l'ordre
            result = self._request("POST", "/fapi/v1/order", signed=True, params=params)
            
            # Si TP/SL sont spécifiés, placer des ordres supplémentaires
            tp_order = None
            sl_order = None
            
            if take_profit is not None or stop_loss is not None:
                # Récupérer la position ouverte pour déterminer le côté
                positions = self.get_positions(symbol)
                position_side = None
                
                for pos in positions:
                    if pos['symbol'] == symbol:
                        position_side = pos['side']
                        break
                
                if position_side:
                    # Placer TP/SL
                    tp_sl_result = self.place_tp_sl_orders(
                        symbol=symbol,
                        position_side=position_side,
                        quantity=quantity,
                        take_profit_price=take_profit,
                        stop_loss_price=stop_loss
                    )
            
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
            params = {
                'symbol': symbol,
                'orderId': order_id
            }
            
            result = self._request("DELETE", "/fapi/v1/order", signed=True, params=params)
            return result
        except Exception as e:
            logger.error(f"Erreur lors de l'annulation de l'ordre: {str(e)}")
            raise
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict:
        """
        Annule tous les ordres ouverts.
        
        Args:
            symbol: Symbole de la paire (obligatoire pour Binance)
            
        Returns:
            Dict: Résultat de l'annulation
        """
        try:
            if not symbol:
                raise ValueError("Le symbole est obligatoire pour annuler tous les ordres sur Binance")
            
            params = {
                'symbol': symbol
            }
            
            result = self._request("DELETE", "/fapi/v1/allOpenOrders", signed=True, params=params)
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
            params = {
                'symbol': symbol,
                'leverage': leverage
            }
            
            result = self._request("POST", "/fapi/v1/leverage", signed=True, params=params)
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
            # Binance utilise 'ISOLATED' et 'CROSSED'
            binance_margin_mode = "ISOLATED" if margin_mode.lower() == "isolated" else "CROSSED"
            
            params = {
                'symbol': symbol,
                'marginType': binance_margin_mode
            }
            
            result = self._request("POST", "/fapi/v1/marginType", signed=True, params=params)
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
            # Binance utilise 'true' pour hedge mode et 'false' pour one-way
            dual_side_position = position_mode.lower() == "hedge"
            
            params = {
                'dualSidePosition': 'true' if dual_side_position else 'false'
            }
            
            result = self._request("POST", "/fapi/v1/positionSide/dual", signed=True, params=params)
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
            params = {
                'symbol': symbol
            }
            
            result = self._request("GET", "/fapi/v1/ticker/24hr", params=params)
            
            # Si plusieurs résultats, prendre le premier
            if isinstance(result, list):
                result = result[0]
            
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
            # Binance utilise un format différent pour l'intervalle
            binance_interval = self.INTERVAL_MAP.get(interval, interval)
            
            params = {
                'symbol': symbol,
                'interval': binance_interval,
                'limit': min(limit, 1000)  # Binance limite à 1000 bougies par requête
            }
            
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            result = self._request("GET", "/fapi/v1/klines", params=params)
            
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
            # Calculer la période de temps
            end_time = int(time.time() * 1000)
            start_time = end_time - (lookback_days * 24 * 60 * 60 * 1000)
            
            # Binance limite à 1000 bougies par requête, donc on doit faire plusieurs requêtes
            all_klines = []
            current_start_time = start_time
            
            while len(all_klines) < limit and current_start_time < end_time:
                # Faire une requête
                klines = self.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=min(limit - len(all_klines), 1000),
                    start_time=current_start_time
                )
                
                if not klines:
                    break
                
                # Ajouter les klines au résultat
                all_klines.extend(klines)
                
                # Mettre à jour l'horodatage de début pour la prochaine requête
                current_start_time = int(klines[-1]['timestamp']) + 1
                
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
                    stop_price=take_profit_price,
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
        Normalise le format du symbole pour Binance.
        
        Args:
            symbol: Symbole à normaliser
            
        Returns:
            str: Symbole normalisé
        """
        # Binance utilise le format BTCUSDT pour BTC/USDT
        return symbol.upper().replace('/', '')
    
    def to_standard_format(self, data: Any, data_type: str) -> Any:
        """
        Convertit les données spécifiques à Binance en format standard.
        
        Args:
            data: Données à convertir
            data_type: Type de données ('ticker', 'klines', 'order', etc.)
            
        Returns:
            Any: Données au format standard
        """
        if data_type == "ticker":
            # Format Binance: {'symbol': 'BTCUSDT', 'lastPrice': '28001.07', ...}
            return {
                'symbol': data.get('symbol', ''),
                'last_price': float(data.get('lastPrice', 0)),
                'bid_price': float(data.get('bidPrice', 0)),
                'ask_price': float(data.get('askPrice', 0)),
                'high_price': float(data.get('highPrice', 0)),
                'low_price': float(data.get('lowPrice', 0)),
                'volume': float(data.get('volume', 0)),
                'timestamp': int(data.get('time', 0))
            }
        
        elif data_type == "klines":
            # Format Binance: [[1625836800000, '33111.11', '33222.22', '33000', '33111.11', '100', ...], ...]
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
            # Convertir les états d'ordre Binance en standard
            status_map = {
                "NEW": OrderStatus.NEW,
                "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
                "FILLED": OrderStatus.FILLED,
                "CANCELED": OrderStatus.CANCELED,
                "EXPIRED": OrderStatus.EXPIRED,
                "REJECTED": OrderStatus.REJECTED
            }
            
            # Déterminer le statut
            binance_status = data.get('status', '')
            status = status_map.get(binance_status, OrderStatus.NEW)
            
            # Déterminer le côté
            side = OrderSide.BUY if data.get('side', '') == 'BUY' else OrderSide.SELL
            
            return {
                'order_id': str(data.get('orderId', '')),
                'client_order_id': data.get('clientOrderId', ''),
                'symbol': data.get('symbol', ''),
                'side': side,
                'type': data.get('type', ''),
                'price': float(data.get('price', 0)),
                'quantity': float(data.get('origQty', 0)),
                'executed_quantity': float(data.get('executedQty', 0)),
                'status': status,
                'timestamp': int(data.get('time', 0)),
                'reduce_only': data.get('reduceOnly', False)
            }
        
        elif data_type == "position":
            # Convertir le côté de position Binance en standard
            side = PositionSide.LONG if float(data.get('positionAmt', 0)) > 0 else PositionSide.SHORT
            
            # Calculer la taille de la position (absolue)
            size = abs(float(data.get('positionAmt', 0)))
            
            return {
                'symbol': data.get('symbol', ''),
                'side': side,
                'size': size,
                'entry_price': float(data.get('entryPrice', 0)),
                'mark_price': float(data.get('markPrice', 0)),
                'liquidation_price': float(data.get('liquidationPrice', 0)),
                'unrealized_pnl': float(data.get('unrealizedProfit', 0)),
                'margin': float(data.get('isolatedMargin', 0)),
                'leverage': float(data.get('leverage', 1)),
                'timestamp': int(time.time() * 1000)
            }
        
        # Pour tout autre type de données, retourner tel quel
        return data