"""
Configuration pour le trading en direct sur différents exchanges.
Ce module gère tous les paramètres nécessaires pour le trading en direct
et assure leur persistance et portabilité.
"""

import os
import json
import uuid
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("live_config")


class ExchangeType(Enum):
    """Types d'exchanges supportés"""
    BITGET = "bitget"
    BINANCE = "binance"


class LiveTradingMode(Enum):
    """Modes de trading en direct"""
    REAL = "real"         # Trading réel
    PAPER = "paper"       # Trading papier (simulé)
    DEMO = "demo"         # Compte démo de l'exchange


class MarginMode(Enum):
    """Modes de marge"""
    ISOLATED = "isolated"  # Marge isolée
    CROSS = "cross"        # Marge croisée


class PositionMode(Enum):
    """Modes de position"""
    ONE_WAY = "one-way"    # Une seule direction à la fois
    HEDGE = "hedge"        # Positions longues et courtes simultanées


@dataclass
class MarketConfig:
    """Configuration du marché pour le trading en direct"""
    symbol: str = "BTCUSDT"
    base_asset: str = "BTC"
    quote_asset: str = "USDT"
    price_precision: int = 2     # Nombre de décimales pour le prix
    quantity_precision: int = 5  # Nombre de décimales pour la quantité
    min_order_size: float = 0.001
    tick_size: float = 0.1
    timeframe: str = "1m"       # Intervalle de temps (1m, 5m, 15m, 1h, etc.)
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderConfig:
    """Configuration des ordres pour le trading en direct"""
    default_size_pct: float = 0.1            # Taille par défaut en % du capital
    max_position_size_pct: float = 0.5       # Taille maximale en % du capital
    enable_tp_sl: bool = True                # Activer les TP/SL
    default_tp_pct: float = 0.03             # Take profit par défaut (3%)
    default_sl_pct: float = 0.01             # Stop loss par défaut (1%)
    enable_trailing_stop: bool = False       # Activer les trailing stops
    trailing_stop_callback_pct: float = 0.01 # Callback du trailing stop (1%)
    enable_smart_entry: bool = False         # Entrée intelligente (DCA)
    enable_auto_reduce: bool = False         # Réduction automatique des positions
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskConfig:
    """Configuration de gestion du risque pour le trading en direct"""
    max_risk_per_trade_pct: float = 0.01     # Risque maximum par trade (1%)
    max_daily_loss_pct: float = 0.05         # Perte quotidienne maximale (5%)
    max_weekly_loss_pct: float = 0.15        # Perte hebdomadaire maximale (15%)
    max_open_positions: int = 3              # Nombre maximum de positions ouvertes
    max_leverage: int = 5                    # Levier maximum
    auto_adjust_leverage: bool = True        # Ajuster automatiquement le levier
    auto_reduce_risk: bool = True           # Réduction automatique du risque après pertes
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Métriques de performance pour le trading en direct"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit_loss: float = 0.0
    total_fees: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_consecutive_wins: int = 0
    current_consecutive_losses: int = 0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def update_with_trade(self, profit_loss: float, fees: float = 0.0) -> None:
        """
        Met à jour les métriques avec un nouveau trade.
        
        Args:
            profit_loss: Profit ou perte du trade
            fees: Frais de trading
        """
        self.total_trades += 1
        self.total_profit_loss += profit_loss
        self.total_fees += fees
        
        # Mettre à jour les statistiques de gains/pertes
        is_win = profit_loss > 0
        if is_win:
            self.winning_trades += 1
            self.current_consecutive_wins += 1
            self.current_consecutive_losses = 0
            
            if profit_loss > self.best_trade:
                self.best_trade = profit_loss
                
            if self.current_consecutive_wins > self.max_consecutive_wins:
                self.max_consecutive_wins = self.current_consecutive_wins
        else:
            self.losing_trades += 1
            self.current_consecutive_losses += 1
            self.current_consecutive_wins = 0
            
            if profit_loss < self.worst_trade:
                self.worst_trade = profit_loss
                
            if self.current_consecutive_losses > self.max_consecutive_losses:
                self.max_consecutive_losses = self.current_consecutive_losses
        
        # Calculer les moyennes
        if self.winning_trades > 0:
            # Cette formule est approximative car nous ne stockons pas tous les trades
            self.avg_win = ((self.avg_win * (self.winning_trades - 1)) + (profit_loss if is_win else 0)) / self.winning_trades
        
        if self.losing_trades > 0:
            # Cette formule est approximative car nous ne stockons pas tous les trades
            self.avg_loss = ((self.avg_loss * (self.losing_trades - 1)) + (profit_loss if not is_win else 0)) / self.losing_trades
        
        # Mettre à jour l'horodatage
        self.last_updated = datetime.now().isoformat()
    
    def update_drawdown(self, current_equity: float, peak_equity: float) -> None:
        """
        Met à jour les métriques de drawdown.
        
        Args:
            current_equity: Valeur actuelle du portefeuille
            peak_equity: Valeur maximale historique du portefeuille
        """
        if peak_equity > 0:
            self.current_drawdown = (peak_equity - current_equity) / peak_equity
            
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
            
            # Mettre à jour l'horodatage
            self.last_updated = datetime.now().isoformat()
    
    def reset(self) -> None:
        """Réinitialise les métriques de performance."""
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit_loss = 0.0
        self.total_fees = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.best_trade = 0.0
        self.worst_trade = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        self.current_consecutive_wins = 0
        self.current_consecutive_losses = 0
        self.last_updated = datetime.now().isoformat()


@dataclass
class LiveConfig:
    """Configuration complète pour le trading en direct"""
    # Identifiants et information générales
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Configuration par défaut"
    description: str = "Configuration de trading en direct"
    strategy_id: str = ""  # ID de la stratégie à utiliser
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Configuration de l'exchange
    exchange: ExchangeType = ExchangeType.BITGET
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""  # Pour certains exchanges comme Bitget
    trading_mode: LiveTradingMode = LiveTradingMode.PAPER
    
    # Configuration du marché
    market: MarketConfig = field(default_factory=MarketConfig)
    
    # Configuration de position/marge
    leverage: int = 1
    margin_mode: MarginMode = MarginMode.ISOLATED
    position_mode: PositionMode = PositionMode.ONE_WAY
    
    # Configurations détaillées
    order_config: OrderConfig = field(default_factory=OrderConfig)
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    
    # Métriques de performance
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    
    # Paramètres additionnels
    data_lookback_days: int = 30           # Nombre de jours d'historique à charger
    update_interval_seconds: int = 5       # Intervalle de mise à jour en secondes
    enable_notifications: bool = False     # Activer les notifications
    notification_settings: Dict[str, Any] = field(default_factory=dict)
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialisation post-création"""
        # Vérifier et convertir les enums si nécessaire
        if isinstance(self.exchange, str):
            self.exchange = ExchangeType(self.exchange)
        
        if isinstance(self.trading_mode, str):
            self.trading_mode = LiveTradingMode(self.trading_mode)
        
        if isinstance(self.margin_mode, str):
            self.margin_mode = MarginMode(self.margin_mode)
        
        if isinstance(self.position_mode, str):
            self.position_mode = PositionMode(self.position_mode)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit la configuration en dictionnaire.
        
        Returns:
            Dict: Configuration au format dictionnaire
        """
        config_dict = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "strategy_id": self.strategy_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "exchange": self.exchange.value,
            "trading_mode": self.trading_mode.value,
            "leverage": self.leverage,
            "margin_mode": self.margin_mode.value,
            "position_mode": self.position_mode.value,
            "data_lookback_days": self.data_lookback_days,
            "update_interval_seconds": self.update_interval_seconds,
            "enable_notifications": self.enable_notifications,
            "notification_settings": self.notification_settings,
            "custom_params": self.custom_params
        }
        
        # Masquer les informations sensibles
        config_dict["api_key"] = "***" if self.api_key else ""
        config_dict["api_secret"] = "***" if self.api_secret else ""
        config_dict["api_passphrase"] = "***" if self.api_passphrase else ""
        
        # Ajouter les configurations détaillées
        config_dict["market"] = asdict(self.market)
        config_dict["order_config"] = asdict(self.order_config)
        config_dict["risk_config"] = asdict(self.risk_config)
        config_dict["metrics"] = asdict(self.metrics)
        
        return config_dict
    
    def save(self, filepath: str) -> bool:
        """
        Sauvegarde la configuration dans un fichier JSON.
        
        Args:
            filepath: Chemin du fichier
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        try:
            # Mettre à jour la date de modification
            self.updated_at = datetime.now().isoformat()
            
            # Créer le répertoire si nécessaire
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Sauvegarder en JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)
            
            logger.info(f"Configuration '{self.name}' sauvegardée dans {filepath}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la configuration: {str(e)}")
            return False
    
    @classmethod
    def load(cls, filepath: str) -> 'LiveConfig':
        """
        Charge une configuration depuis un fichier JSON.
        
        Args:
            filepath: Chemin du fichier
            
        Returns:
            LiveConfig: Configuration chargée
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extraire les configurations détaillées
            market_data = data.pop("market", {})
            order_config_data = data.pop("order_config", {})
            risk_config_data = data.pop("risk_config", {})
            metrics_data = data.pop("metrics", {})
            
            # Restaurer les informations sensibles si elles sont masquées
            if data.get("api_key") == "***":
                data["api_key"] = ""
            if data.get("api_secret") == "***":
                data["api_secret"] = ""
            if data.get("api_passphrase") == "***":
                data["api_passphrase"] = ""
            
            # Créer la configuration
            config = cls(**data)
            
            # Ajouter les configurations détaillées
            config.market = MarketConfig(**market_data)
            config.order_config = OrderConfig(**order_config_data)
            config.risk_config = RiskConfig(**risk_config_data)
            config.metrics = PerformanceMetrics(**metrics_data)
            
            logger.info(f"Configuration '{config.name}' chargée depuis {filepath}")
            return config
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
            raise ValueError(f"Impossible de charger la configuration depuis {filepath}: {str(e)}")
    
    def validate(self) -> bool:
        """
        Valide la configuration pour le trading en direct.
        
        Returns:
            bool: True si la configuration est valide
        """
        # Vérifier les paramètres obligatoires pour le trading réel
        if self.trading_mode == LiveTradingMode.REAL:
            if not self.api_key or not self.api_secret:
                logger.error("API key et secret sont obligatoires pour le trading réel")
                return False
            
            if self.exchange == ExchangeType.BITGET and not self.api_passphrase:
                logger.error("API passphrase est obligatoire pour Bitget")
                return False
        
        # Vérifier les limites de risque
        if self.risk_config.max_leverage < self.leverage:
            logger.warning(f"Le levier configuré ({self.leverage}) dépasse le maximum autorisé ({self.risk_config.max_leverage})")
            return False
        
        # Vérifier la cohérence des configurations
        if self.order_config.max_position_size_pct > 1.0:
            logger.warning("La taille maximale de position ne peut pas dépasser 100% du capital")
            return False
        
        # Vérification réussie
        return True
    
    def clone(self) -> 'LiveConfig':
        """
        Clone la configuration actuelle.
        
        Returns:
            LiveConfig: Clone de la configuration
        """
        clone = LiveConfig(
            id=str(uuid.uuid4())[:8],  # Nouvel ID
            name=f"{self.name} (Clone)",
            description=self.description,
            strategy_id=self.strategy_id,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            exchange=self.exchange,
            api_key=self.api_key,
            api_secret=self.api_secret,
            api_passphrase=self.api_passphrase,
            trading_mode=self.trading_mode,
            leverage=self.leverage,
            margin_mode=self.margin_mode,
            position_mode=self.position_mode,
            data_lookback_days=self.data_lookback_days,
            update_interval_seconds=self.update_interval_seconds,
            enable_notifications=self.enable_notifications,
            notification_settings=self.notification_settings.copy(),
            custom_params=self.custom_params.copy()
        )
        
        # Cloner les configurations détaillées
        clone.market = MarketConfig(**asdict(self.market))
        clone.order_config = OrderConfig(**asdict(self.order_config))
        clone.risk_config = RiskConfig(**asdict(self.risk_config))
        clone.metrics = PerformanceMetrics()  # Réinitialiser les métriques
        
        return clone


def create_default_config(exchange: ExchangeType = ExchangeType.BITGET, symbol: str = "BTCUSDT") -> LiveConfig:
    """
    Crée une configuration par défaut pour le trading en direct.
    
    Args:
        exchange: Type d'exchange
        symbol: Symbole de la paire
        
    Returns:
        LiveConfig: Configuration par défaut
    """
    config = LiveConfig(
        name=f"Configuration {exchange.value.capitalize()} {symbol}",
        exchange=exchange,
        trading_mode=LiveTradingMode.PAPER
    )
    
    # Configurer le marché
    base_asset, quote_asset = symbol.split('USDT')[0], 'USDT'
    config.market = MarketConfig(
        symbol=symbol,
        base_asset=base_asset,
        quote_asset=quote_asset
    )
    
    # Paramètres spécifiques à l'exchange
    if exchange == ExchangeType.BITGET:
        config.market.price_precision = 2
        config.market.quantity_precision = 6
    elif exchange == ExchangeType.BINANCE:
        config.market.price_precision = 2
        config.market.quantity_precision = 5
    
    return config


def list_saved_configs(directory: str = "config/live") -> List[Dict[str, Any]]:
    """
    Liste toutes les configurations sauvegardées.
    
    Args:
        directory: Répertoire des configurations
        
    Returns:
        List[Dict]: Liste des métadonnées des configurations
    """
    configs = []
    
    try:
        os.makedirs(directory, exist_ok=True)
        
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                filepath = os.path.join(directory, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    configs.append({
                        "id": data.get("id", ""),
                        "name": data.get("name", ""),
                        "description": data.get("description", ""),
                        "exchange": data.get("exchange", ""),
                        "symbol": data.get("market", {}).get("symbol", ""),
                        "trading_mode": data.get("trading_mode", ""),
                        "updated_at": data.get("updated_at", ""),
                        "filepath": filepath
                    })
                except Exception as e:
                    logger.warning(f"Erreur lors de la lecture de {filepath}: {str(e)}")
        
        # Trier par date de mise à jour
        configs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        return configs
    
    except Exception as e:
        logger.error(f"Erreur lors de la liste des configurations: {str(e)}")
        return []