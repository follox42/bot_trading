"""
Module factory pour créer des instances d'API d'exchange de manière simplifiée.
"""

import logging
from typing import Optional, Dict, Any

from core.live.exchange.exchange_interface import ExchangeInterface
from core.live.exchange.bitget_api import BitgetAPI
from core.live.exchange.binance_api import BinanceAPI
from core.live.live_config import ExchangeType, LiveConfig

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("exchange_factory")


class ExchangeFactory:
    """
    Factory pour créer des instances d'API d'exchange.
    """
    
    @staticmethod
    def create_exchange(
        exchange_type: ExchangeType,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        testnet: bool = False
    ) -> Optional[ExchangeInterface]:
        """
        Crée une instance d'API d'exchange.
        
        Args:
            exchange_type: Type d'exchange
            api_key: Clé API
            api_secret: Secret API
            passphrase: Phrase secrète (requise pour certains exchanges)
            testnet: Utiliser le réseau de test
            
        Returns:
            Optional[ExchangeInterface]: Instance de l'API d'exchange ou None en cas d'erreur
        """
        try:
            if exchange_type == ExchangeType.BITGET:
                return BitgetAPI(
                    api_key=api_key,
                    api_secret=api_secret,
                    passphrase=passphrase,
                    testnet=testnet
                )
            elif exchange_type == ExchangeType.BINANCE:
                return BinanceAPI(
                    api_key=api_key,
                    api_secret=api_secret,
                    passphrase=passphrase,  # Ignoré pour Binance mais gardé pour l'interface uniforme
                    testnet=testnet
                )
            else:
                logger.error(f"Type d'exchange non supporté: {exchange_type}")
                return None
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'API d'exchange {exchange_type.value}: {str(e)}")
            return None
    
    @staticmethod
    def create_from_config(config: LiveConfig) -> Optional[ExchangeInterface]:
        """
        Crée une instance d'API d'exchange à partir d'une configuration.
        
        Args:
            config: Configuration du trading en direct
            
        Returns:
            Optional[ExchangeInterface]: Instance de l'API d'exchange ou None en cas d'erreur
        """
        try:
            # Déterminer si on utilise le testnet en fonction du mode de trading
            from live_config import LiveTradingMode
            testnet = config.trading_mode != LiveTradingMode.REAL
            
            return ExchangeFactory.create_exchange(
                exchange_type=config.exchange,
                api_key=config.api_key,
                api_secret=config.api_secret,
                passphrase=config.api_passphrase,
                testnet=testnet
            )
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'API d'exchange depuis la configuration: {str(e)}")
            return None