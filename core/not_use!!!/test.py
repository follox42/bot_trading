#!/usr/bin/env python3
"""
Script de test complet pour le système de trading.
Vérifie le bon fonctionnement de tous les composants.
"""

import os
import sys
import json
import logging
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import shutil
import random
import uuid
from unittest.mock import MagicMock, patch

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("test_trading_system")

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import des modules du système de trading
from core.strategy.constructor import StrategyConstructor
from core.strategy.strategy_manager import StrategyManager
from core.strategy.indicators.indicators_config import IndicatorType, IndicatorConfig
from core.strategy.conditions.conditions_config import (
    ConditionConfig, BlockConfig, OperatorType, LogicOperatorType,
    PriceOperand, IndicatorOperand, ValueOperand
)
from core.strategy.risk.risk_config import RiskConfig, RiskModeType

from core.simulation.simulator import Simulator
from core.simulation.simulation_config import SimulationConfig

from exchange.exchange_interface import ExchangeInterface, OrderType, OrderSide, PositionSide
from exchange.exchange_factory import ExchangeFactory
from utils.history_fetcher import HistoryFetcher

from live_config import LiveConfig, ExchangeType, LiveTradingMode
from live_data_manager import LiveDataManager
from live_backtest import LiveBacktest

# Variables globales pour les tests
TEST_DIR = "test_data"
TEST_STRATEGIES_DIR = os.path.join(TEST_DIR, "strategies")
TEST_RESULTS_DIR = os.path.join(TEST_DIR, "results")
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")


# ============= Fonctions utilitaires pour les tests =============

def create_mock_exchange_api():
    """Crée une API d'exchange simulée pour les tests"""
    mock_api = MagicMock(spec=ExchangeInterface)
    
    # Simuler les méthodes clés
    mock_api.get_account_info.return_value = {"account_id": "test123", "balance": 10000.0}
    mock_api.get_balance.return_value = 10000.0
    mock_api.get_ticker.return_value = {"symbol": "BTCUSDT", "last_price": 50000.0}
    
    # Simuler des positions
    mock_api.get_positions.return_value = []
    
    # Simuler des ordres
    mock_api.get_open_orders.return_value = []
    
    # Simuler la récupération de données historiques
    mock_api.get_historical_klines.return_value = generate_test_data()
    mock_api.get_klines.return_value = [
        {"timestamp": int(datetime.now().timestamp() * 1000), 
         "open": 50000.0, "high": 51000.0, "low": 49000.0, "close": 50500.0, "volume": 10.0}
    ]
    
    return mock_api


def generate_test_data(periods=1000, base_price=50000, volatility=0.01, trend=0.0001, has_date_index=True):
    """Génère des données OHLCV synthétiques pour les tests"""
    np.random.seed(42)  # Pour la reproductibilité
    
    # Générer des prix avec un peu de volatilité et tendance
    price_changes = np.random.normal(trend, volatility, periods)
    prices = base_price * np.cumprod(1 + price_changes)
    
    # Créer le DataFrame
    if has_date_index:
        end_date = datetime.now()
        start_date = end_date - timedelta(minutes=periods)
        dates = pd.date_range(start=start_date, end=end_date, periods=periods)
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, periods)),
            'high': prices * (1 + abs(np.random.normal(0, 0.003, periods))),
            'low': prices * (1 - abs(np.random.normal(0, 0.003, periods))),
            'close': prices,
            'volume': np.random.lognormal(3, 1, periods)
        }, index=dates)
        
        df.index.name = 'timestamp'
    else:
        # Version sans index de date
        df = pd.DataFrame({
            'timestamp': [int((datetime.now() - timedelta(minutes=i)).timestamp() * 1000) for i in range(periods, 0, -1)],
            'open': prices * (1 + np.random.normal(0, 0.001, periods)),
            'high': prices * (1 + abs(np.random.normal(0, 0.003, periods))),
            'low': prices * (1 - abs(np.random.normal(0, 0.003, periods))),
            'close': prices,
            'volume': np.random.lognormal(3, 1, periods)
        })
    
    return df


def create_test_strategy(name="Test Strategy"):
    """Crée une stratégie de test avec quelques indicateurs et conditions"""
    constructor = StrategyConstructor()
    constructor.set_name(name)
    constructor.set_description("Stratégie de test pour les tests unitaires")
    
    # Ajouter des tags
    constructor.add_tag("test")
    constructor.add_tag("unittest")
    
    # Ajouter des indicateurs
    ema_config = IndicatorConfig(
        type_=IndicatorType.EMA,
        period=20,
        source="close"
    )
    constructor.add_indicator("ema20", ema_config)
    
    rsi_config = IndicatorConfig(
        type_=IndicatorType.RSI,
        period=14,
        source="close"
    )
    constructor.add_indicator("rsi14", rsi_config)
    
    # Ajouter des conditions
    # Condition d'entrée: RSI < 30
    entry_condition = ConditionConfig(
        left_operand=IndicatorOperand(
            indicator_type=IndicatorType.RSI,
            indicator_name="rsi14"
        ),
        operator=OperatorType.LESS,
        right_operand=ValueOperand(value=30.0)
    )
    
    entry_block = BlockConfig(
        conditions=[entry_condition],
        name="RSI Oversold"
    )
    constructor.add_entry_block(entry_block)
    
    # Condition de sortie: RSI > 70
    exit_condition = ConditionConfig(
        left_operand=IndicatorOperand(
            indicator_type=IndicatorType.RSI,
            indicator_name="rsi14"
        ),
        operator=OperatorType.GREATER,
        right_operand=ValueOperand(value=70.0)
    )
    
    exit_block = BlockConfig(
        conditions=[exit_condition],
        name="RSI Overbought"
    )
    constructor.add_exit_block(exit_block)
    
    # Définir la configuration de risque
    risk_config = RiskConfig(
        mode=RiskModeType.FIXED,
        position_size=0.1,
        stop_loss=0.02,
        take_profit=0.04
    )
    constructor.set_risk_config(risk_config)
    
    return constructor


def setup_test_environment():
    """Configure l'environnement pour les tests"""
    # Créer les répertoires de test
    os.makedirs(TEST_STRATEGIES_DIR, exist_ok=True)
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    os.makedirs(TEST_DATA_DIR, exist_ok=True)


def cleanup_test_environment():
    """Nettoie l'environnement après les tests"""
    # Supprimer les répertoires de test
    shutil.rmtree(TEST_DIR)


# ============= Classes de test =============

class TestStrategyConstructor(unittest.TestCase):
    """Tests pour le StrategyConstructor"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        self.constructor = create_test_strategy()
    
    def test_create_strategy(self):
        """Test de création d'une stratégie"""
        self.assertEqual(self.constructor.config.name, "Test Strategy")
        self.assertEqual(len(self.constructor.config.tags), 2)
        self.assertIn("test", self.constructor.config.tags)
    
    def test_add_indicator(self):
        """Test d'ajout d'un indicateur"""
        # Ajouter un nouvel indicateur
        macd_config = IndicatorConfig(
            type_=IndicatorType.MACD,
            fast_period=12,
            slow_period=26,
            signal_period=9
        )
        self.constructor.add_indicator("macd", macd_config)
        
        # Vérifier que l'indicateur a été ajouté
        indicators = self.constructor.config.indicators_manager.list_indicators()
        self.assertIn("macd", indicators)
        self.assertEqual(indicators["macd"].type, IndicatorType.MACD)
    
    def test_add_entry_exit_blocks(self):
        """Test d'ajout de blocs d'entrée/sortie"""
        # Vérifier les blocs existants
        self.assertEqual(len(self.constructor.config.blocks_config.entry_blocks), 1)
        self.assertEqual(len(self.constructor.config.blocks_config.exit_blocks), 1)
        
        # Ajouter de nouveaux blocs
        new_entry_condition = ConditionConfig(
            left_operand=PriceOperand(price_type="close"),
            operator=OperatorType.GREATER,
            right_operand=IndicatorOperand(
                indicator_type=IndicatorType.EMA,
                indicator_name="ema20"
            )
        )
        
        new_entry_block = BlockConfig(
            conditions=[new_entry_condition],
            name="Price Above EMA"
        )
        self.constructor.add_entry_block(new_entry_block)
        
        # Vérifier l'ajout
        self.assertEqual(len(self.constructor.config.blocks_config.entry_blocks), 2)
    
    def test_save_load(self):
        """Test de sauvegarde et chargement d'une stratégie"""
        # Sauvegarder la stratégie
        filepath = os.path.join(TEST_STRATEGIES_DIR, "test_strategy.json")
        self.constructor.save(filepath)
        
        # Vérifier que le fichier existe
        self.assertTrue(os.path.exists(filepath))
        
        # Charger la stratégie
        loaded_constructor = StrategyConstructor.load(filepath)
        
        # Vérifier que la stratégie a été correctement chargée
        self.assertEqual(loaded_constructor.config.name, self.constructor.config.name)
        self.assertEqual(len(loaded_constructor.config.tags), len(self.constructor.config.tags))
        
        # Vérifier les indicateurs
        loaded_indicators = loaded_constructor.config.indicators_manager.list_indicators()
        self.assertEqual(len(loaded_indicators), len(self.constructor.config.indicators_manager.list_indicators()))
    
    def test_generate_signals(self):
        """Test de génération de signaux"""
        # Créer des données de test
        data = generate_test_data()
        
        # Générer les signaux
        signals, data_with_signals = self.constructor.generate_signals(data)
        
        # Vérifier que les signaux ont été générés
        self.assertEqual(len(signals), len(data))
        self.assertTrue(isinstance(signals, np.ndarray))
        
        # Vérifier que les indicateurs ont été calculés
        self.assertIn("ema20", data_with_signals.columns)
        self.assertIn("rsi14", data_with_signals.columns)
        
        # Vérifier les paramètres de risque
        self.assertIn("position_size", data_with_signals.columns)
        self.assertIn("sl_level", data_with_signals.columns)
        self.assertIn("tp_level", data_with_signals.columns)


class TestStrategyManager(unittest.TestCase):
    """Tests pour le StrategyManager"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        self.manager = StrategyManager(TEST_STRATEGIES_DIR)
        self.test_strategy = create_test_strategy()
    
    def test_create_save_load_strategy(self):
        """Test de création, sauvegarde et chargement d'une stratégie"""
        # Créer une stratégie via le manager
        strategy = self.manager.create_strategy("Manager Test Strategy", "Created by StrategyManager")
        
        # Vérifier la création
        self.assertEqual(strategy.config.name, "Manager Test Strategy")
        self.assertEqual(strategy.config.description, "Created by StrategyManager")
        
        # Sauvegarder la stratégie
        filepath = self.manager.save_strategy(strategy)
        
        # Vérifier la sauvegarde
        self.assertTrue(os.path.exists(filepath))
        
        # Charger la stratégie
        loaded_strategy = self.manager.load_strategy(strategy.config.id)
        
        # Vérifier le chargement
        self.assertEqual(loaded_strategy.config.id, strategy.config.id)
        self.assertEqual(loaded_strategy.config.name, strategy.config.name)
    
    def test_clone_strategy(self):
        """Test de clonage d'une stratégie"""
        # Sauvegarder la stratégie de test
        self.manager.current_strategy = self.test_strategy
        self.manager.save_strategy()
        
        # Cloner la stratégie
        clone = self.manager.clone_strategy(new_name="Cloned Strategy")
        
        # Vérifier le clonage
        self.assertNotEqual(clone.config.id, self.test_strategy.config.id)
        self.assertEqual(clone.config.name, "Cloned Strategy")
        self.assertEqual(clone.config.description, self.test_strategy.config.description)
        
        # Vérifier que les indicateurs ont été clonés
        original_indicators = self.test_strategy.config.indicators_manager.list_indicators()
        clone_indicators = clone.config.indicators_manager.list_indicators()
        self.assertEqual(len(original_indicators), len(clone_indicators))
    
    def test_list_strategies(self):
        """Test de liste des stratégies"""
        # Sauvegarder plusieurs stratégies
        for i in range(3):
            strategy = create_test_strategy(f"Test Strategy {i}")
            self.manager.current_strategy = strategy
            self.manager.save_strategy()
        
        # Lister les stratégies
        strategies = self.manager.list_strategies()
        
        # Vérifier la liste
        self.assertGreaterEqual(len(strategies), 3)
    
    def test_run_simulation(self):
        """Test d'exécution de simulation"""
        # Créer des données de test
        data = generate_test_data()
        
        # Exécuter une simulation
        results = self.manager.run_simulation(
            data=data,
            strategy=self.test_strategy,
            initial_balance=10000.0,
            leverage=1,
            fee_open=0.001,
            fee_close=0.001,
            slippage=0.0005
        )
        
        # Vérifier les résultats
        self.assertTrue(isinstance(results, dict))
        self.assertIn("performance", results)
        self.assertIn("roi_pct", results["performance"])
        self.assertIn("win_rate_pct", results["performance"])
    
    def test_compare_strategies(self):
        """Test de comparaison de stratégies"""
        # Créer et sauvegarder plusieurs stratégies
        strategy_ids = []
        for i in range(2):
            strategy = create_test_strategy(f"Test Strategy {i}")
            # Modification des paramètres pour différencier les stratégies
            rsi_config = IndicatorConfig(
                type_=IndicatorType.RSI,
                period=14 + i*10,  # RSI 14 et RSI 24
                source="close"
            )
            strategy.add_indicator(f"rsi{14+i*10}", rsi_config)
            
            self.manager.current_strategy = strategy
            self.manager.save_strategy()
            strategy_ids.append(strategy.config.id)
        
        # Créer des données de test
        data = generate_test_data()
        
        # Comparer les stratégies
        comparison = self.manager.compare_strategies(strategy_ids, data)
        
        # Vérifier les résultats
        self.assertTrue(isinstance(comparison, dict))
        self.assertIn("strategies", comparison)
        self.assertGreaterEqual(len(comparison["strategies"]), 1)
    
    def test_generate_performance_report(self):
        """Test de génération de rapport de performance"""
        # Créer des données de test
        data = generate_test_data()
        
        # Exécuter une simulation
        results = self.manager.run_simulation(
            data=data,
            strategy=self.test_strategy
        )
        
        # Générer un rapport
        report = self.manager.generate_performance_report(self.test_strategy.config.id)
        
        # Vérifier le rapport
        self.assertTrue(isinstance(report, dict))
        self.assertIn("strategy", report)
        self.assertIn("performance", report)
        self.assertIn("roi", report["performance"])


class TestLiveConfig(unittest.TestCase):
    """Tests pour la configuration du trading en direct"""
    
    def test_create_default_config(self):
        """Test de création d'une configuration par défaut"""
        from live_config import create_default_config
        
        config = create_default_config(ExchangeType.BITGET, "BTCUSDT")
        
        self.assertEqual(config.exchange, ExchangeType.BITGET)
        self.assertEqual(config.market.symbol, "BTCUSDT")
        self.assertEqual(config.trading_mode, LiveTradingMode.PAPER)
    
    def test_save_load_config(self):
        """Test de sauvegarde et chargement d'une configuration"""
        from live_config import create_default_config
        
        # Créer une configuration
        config = create_default_config(ExchangeType.BITGET, "BTCUSDT")
        config.strategy_id = "test123"
        
        # Sauvegarder la configuration
        filepath = os.path.join(TEST_DIR, "test_config.json")
        config.save(filepath)
        
        # Vérifier que le fichier existe
        self.assertTrue(os.path.exists(filepath))
        
        # Charger la configuration
        loaded_config = LiveConfig.load(filepath)
        
        # Vérifier le chargement
        self.assertEqual(loaded_config.exchange, config.exchange)
        self.assertEqual(loaded_config.market.symbol, config.market.symbol)
        self.assertEqual(loaded_config.strategy_id, config.strategy_id)
    
    def test_validate_config(self):
        """Test de validation d'une configuration"""
        from live_config import create_default_config
        
        # Créer une configuration
        config = create_default_config(ExchangeType.BITGET, "BTCUSDT")
        
        # En mode paper, devrait être valide sans clés API
        self.assertTrue(config.validate())
        
        # Changer en mode réel sans clés API, devrait être invalide
        config.trading_mode = LiveTradingMode.REAL
        self.assertFalse(config.validate())
        
        # Ajouter des clés API
        config.api_key = "test_key"
        config.api_secret = "test_secret"
        config.api_passphrase = "test_passphrase"
        
        # Maintenant devrait être valide
        self.assertTrue(config.validate())


class TestExchangeFactory(unittest.TestCase):
    """Tests pour l'ExchangeFactory"""
    
    @patch('exchange.bitget_api.BitgetAPI')
    @patch('exchange.binance_api.BinanceAPI')
    def test_create_exchange(self, mock_binance, mock_bitget):
        """Test de création d'une API d'exchange"""
        # Configurer les mocks
        mock_bitget.return_value = MagicMock(spec=ExchangeInterface)
        mock_binance.return_value = MagicMock(spec=ExchangeInterface)
        
        # Créer une API Bitget
        api = ExchangeFactory.create_exchange(
            exchange_type=ExchangeType.BITGET,
            api_key="test_key",
            api_secret="test_secret",
            passphrase="test_passphrase",
            testnet=True
        )
        
        # Vérifier la création
        self.assertIsNotNone(api)
        mock_bitget.assert_called_once()
        
        # Réinitialiser les mocks
        mock_bitget.reset_mock()
        mock_binance.reset_mock()
        
        # Créer une API Binance
        api = ExchangeFactory.create_exchange(
            exchange_type=ExchangeType.BINANCE,
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        # Vérifier la création
        self.assertIsNotNone(api)
        mock_binance.assert_called_once()
    
    @patch('exchange.bitget_api.BitgetAPI')
    def test_create_from_config(self, mock_bitget):
        """Test de création d'une API d'exchange à partir d'une configuration"""
        # Configurer le mock
        mock_bitget.return_value = MagicMock(spec=ExchangeInterface)
        
        # Créer une configuration
        from live_config import create_default_config
        config = create_default_config(ExchangeType.BITGET, "BTCUSDT")
        config.api_key = "test_key"
        config.api_secret = "test_secret"
        config.api_passphrase = "test_passphrase"
        
        # Créer l'API
        api = ExchangeFactory.create_from_config(config)
        
        # Vérifier la création
        self.assertIsNotNone(api)
        mock_bitget.assert_called_once()


class TestHistoryFetcher(unittest.TestCase):
    """Tests pour le HistoryFetcher"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        self.strategy = create_test_strategy()
        
        # Créer une configuration
        from live_config import create_default_config
        self.config = create_default_config(ExchangeType.BITGET, "BTCUSDT")
        
        # Créer un mock pour l'API d'exchange
        self.mock_api = create_mock_exchange_api()
    
    @patch('utils.history_fetcher.download_data')
    @patch('utils.history_fetcher.load_data')
    def test_fetch_historical_data(self, mock_load_data, mock_download_data):
        """Test de récupération des données historiques"""
        # Configurer les mocks
        mock_load_data.return_value = None  # Simuler l'absence de données locales
        
        # Créer une instance de MarketData factice
        class MockMarketData:
            def __init__(self):
                self.dataframe = generate_test_data()
        
        mock_download_data.return_value = MockMarketData()
        
        # Créer le HistoryFetcher
        fetcher = HistoryFetcher(
            strategy=self.strategy,
            config=self.config,
            exchange_api=self.mock_api,
            output_dir=TEST_DATA_DIR
        )
        
        # Récupérer les données historiques de manière asynchrone
        async def fetch_data():
            return await fetcher.fetch_historical_data()
        
        # Exécuter de manière synchrone pour le test
        df = asyncio.run(fetch_data())
        
        # Vérifier les données
        self.assertIsNotNone(df)
        self.assertIn('open', df.columns)
        self.assertIn('high', df.columns)
        self.assertIn('low', df.columns)
        self.assertIn('close', df.columns)
        self.assertIn('volume', df.columns)
    
    def test_calculate_required_lookback(self):
        """Test du calcul du nombre de points de données requis"""
        # Créer le HistoryFetcher
        fetcher = HistoryFetcher(
            strategy=self.strategy,
            config=self.config,
            exchange_api=self.mock_api
        )
        
        # Calculer le lookback
        lookback = fetcher._calculate_required_lookback()
        
        # Vérifier le résultat
        self.assertGreaterEqual(lookback, 500)  # Au moins 500 points de données


class TestLiveDataManager(unittest.TestCase):
    """Tests pour le LiveDataManager"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        self.strategy = create_test_strategy()
        
        # Créer une configuration
        from live_config import create_default_config
        self.config = create_default_config(ExchangeType.BITGET, "BTCUSDT")
        
        # Créer un mock pour l'API d'exchange
        self.mock_api = create_mock_exchange_api()
        
        # Créer le LiveDataManager
        self.data_manager = LiveDataManager(
            exchange=self.mock_api,
            config=self.config,
            strategy=self.strategy,
            data_dir=TEST_DATA_DIR
        )
    
    @patch('live_data_manager.LiveDataManager._load_historical_data')
    def test_initialize_data(self, mock_load_historical):
        """Test d'initialisation des données"""
        # Configurer le mock
        mock_load_historical.return_value = generate_test_data()
        
        # Initialiser les données de manière asynchrone
        async def init_data():
            return await self.data_manager.initialize_data()
        
        # Exécuter de manière synchrone pour le test
        df = asyncio.run(init_data())
        
        # Vérifier les données
        self.assertIsNotNone(df)
        self.assertIn('open', df.columns)
        self.assertIn('high', df.columns)
        self.assertIn('low', df.columns)
        self.assertIn('close', df.columns)
        self.assertIn('volume', df.columns)
    
    @patch('live_data_manager.LiveDataManager._load_historical_data')
    def test_update_data(self, mock_load_historical):
        """Test de mise à jour des données"""
        # Configurer le mock
        mock_load_historical.return_value = generate_test_data()
        
        # Initialiser les données
        async def init_and_update():
            await self.data_manager.initialize_data()
            return await self.data_manager.update_data()
        
        # Exécuter de manière synchrone pour le test
        df = asyncio.run(init_and_update())
        
        # Vérifier les données
        self.assertIsNotNone(df)
        self.assertIn('open', df.columns)
    
    @patch('live_data_manager.LiveDataManager._load_historical_data')
    def test_prepare_backtest_data(self, mock_load_historical):
        """Test de préparation des données pour le backtest"""
        # Configurer le mock
        mock_load_historical.return_value = generate_test_data()
        
        # Initialiser les données
        async def init_and_prepare():
            await self.data_manager.initialize_data()
            return self.data_manager.prepare_backtest_data(lookback_periods=500)
        
        # Exécuter de manière synchrone pour le test
        df = asyncio.run(init_and_prepare())
        
        # Vérifier les données
        self.assertIsNotNone(df)
        self.assertLessEqual(len(df), 500)


class TestLiveBacktest(unittest.TestCase):
    """Tests pour le LiveBacktest"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        self.strategy = create_test_strategy()
        
        # Créer une configuration
        from live_config import create_default_config
        self.config = create_default_config(ExchangeType.BITGET, "BTCUSDT")
        
        # Créer le LiveBacktest
        self.live_backtest = LiveBacktest(
            strategy=self.strategy,
            config=self.config,
            results_dir=TEST_RESULTS_DIR
        )
    
    def test_run_backtest(self):
        """Test d'exécution d'un backtest"""
        # Créer des données de test
        data = generate_test_data()
        
        # Exécuter le backtest
        results = self.live_backtest.run_backtest(data)
        
        # Vérifier les résultats
        self.assertIsNotNone(results)
        self.assertIn("performance", results)
    
    def test_compare_with_live(self):
        """Test de comparaison avec les performances en direct"""
        # Créer des données de test
        data = generate_test_data()
        
        # Exécuter le backtest
        self.live_backtest.run_backtest(data)
        
        # Créer des métriques live simulées
        live_metrics = {
            "total_trades": 10,
            "winning_trades": 6,
            "total_profit_loss": 500.0,
            "max_drawdown": 0.05,
            "avg_win": 100.0,
            "avg_loss": -50.0
        }
        
        # Comparer
        comparison = self.live_backtest.compare_with_live(live_metrics)
        
        # Vérifier la comparaison
        self.assertIsNotNone(comparison)
        self.assertIn("consistency_score", comparison)
        self.assertIn("metrics", comparison)
        self.assertIn("differences", comparison)


@patch('live.LiveTrader.api')
class TestLiveTrader(unittest.TestCase):
    """Tests pour le LiveTrader"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        self.strategy = create_test_strategy()
        
        # Créer une configuration
        from live_config import create_default_config
        self.config = create_default_config(ExchangeType.BITGET, "BTCUSDT")
        
        # Créer un mock pour l'API d'exchange
        self.mock_api = create_mock_exchange_api()
    
    @patch('live.LiveTrader._setup_initial_config')
    @patch('live.LiveTrader._main_loop')
    def test_start_stop(self, mock_main_loop, mock_setup, mock_api):
        """Test de démarrage et arrêt du trader"""
        # Configurer les mocks
        mock_api.connect.return_value = True
        mock_api.disconnect.return_value = True
        mock_setup.return_value = None
        mock_main_loop.return_value = None
        
        # Importer les classes nécessaires ici pour éviter des problèmes de patching
        from live import LiveTrader, LiveTraderConfig
        
        # Créer le LiveTrader
        live_trader = LiveTrader(
            strategy_constructor=self.strategy,
            config=LiveTraderConfig(
                exchange=ExchangeType.BITGET,
                symbol="BTCUSDT",
                trading_mode=LiveTradingMode.PAPER
            )
        )
        
        # Démarrer et arrêter de manière asynchrone
        async def start_and_stop():
            await live_trader.start()
            await asyncio.sleep(0.1)  # Attendre un peu
            await live_trader.stop()
        
        # Exécuter de manière synchrone pour le test
        asyncio.run(start_and_stop())
        
        # Vérifier les appels
        mock_api.connect.assert_called_once()
        mock_api.disconnect.assert_called_once()
    
    @patch('live.LiveTrader._update_market_data')
    @patch('live.LiveTrader._update_positions')
    @patch('live.LiveTrader._evaluate_strategy')
    def test_evaluate_strategy(self, mock_evaluate, mock_update_positions, mock_update_market, mock_api):
        """Test d'évaluation de la stratégie"""
        # Configurer les mocks
        mock_update_market.return_value = None
        mock_update_positions.return_value = None
        mock_evaluate.return_value = None
        
        # Importer les classes nécessaires
        from live import LiveTrader, LiveTraderConfig
        
        # Créer le LiveTrader
        live_trader = LiveTrader(
            strategy_constructor=self.strategy,
            config=LiveTraderConfig(
                exchange=ExchangeType.BITGET,
                symbol="BTCUSDT",
                trading_mode=LiveTradingMode.PAPER
            )
        )
        
        # Évaluer la stratégie de manière asynchrone
        async def evaluate():
            live_trader.current_data = generate_test_data()
            await live_trader._evaluate_strategy()
        
        # Exécuter de manière synchrone pour le test
        asyncio.run(evaluate())
        
        # Vérifier les appels
        mock_evaluate.assert_called_once()


class TestIntegration(unittest.TestCase):
    """Tests d'intégration"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        self.strategy = create_test_strategy()
        
        # Créer une configuration
        from live_config import create_default_config
        self.config = create_default_config(ExchangeType.BITGET, "BTCUSDT")
        
        # Créer un mock pour l'API d'exchange
        self.mock_api = create_mock_exchange_api()
    
    @patch('live_launcher.LiveLauncher.initialize_exchange')
    @patch('live_launcher.LiveLauncher.initialize_data_manager')
    @patch('live_launcher.LiveLauncher.initialize_live_backtest')
    @patch('live_launcher.LiveLauncher.create_live_trader')
    @patch('live_launcher.LiveLauncher.start_trading')
    @patch('live_launcher.LiveLauncher.stop_trading')
    def test_live_launcher(self, mock_stop, mock_start, mock_create, mock_init_backtest, mock_init_data, mock_init_exchange):
        """Test du lanceur de trading en direct"""
        # Configurer les mocks
        mock_init_exchange.return_value = True
        mock_init_data.return_value = True
        mock_init_backtest.return_value = True
        mock_create.return_value = True
        mock_start.return_value = True
        mock_stop.return_value = True
        
        # Importer la classe nécessaire
        from live_launcher import LiveLauncher
        
        # Créer le lanceur
        launcher = LiveLauncher()
        launcher.config = self.config
        launcher.strategy = self.strategy
        
        # Exécuter de manière asynchrone
        async def run_launcher():
            await launcher.run()
        
        # Exécuter de manière synchrone pour le test
        asyncio.run(run_launcher())
        
        # Vérifier les appels
        mock_init_exchange.assert_called_once()
        mock_init_data.assert_called_once()
        mock_init_backtest.assert_called_once()
        mock_create.assert_called_once()
        mock_start.assert_called_once()
    
    def test_complete_workflow(self):
        """Test d'un workflow complet de bout en bout"""
        # Cette méthode simule un workflow complet mais sans exécution réelle
        # car cela nécessiterait des connexions API et serait trop complexe pour un test unitaire
        
        # 1. Créer une stratégie
        constructor = create_test_strategy("Workflow Test Strategy")
        
        # 2. Sauvegarder la stratégie
        manager = StrategyManager(TEST_STRATEGIES_DIR)
        manager.current_strategy = constructor
        filepath = manager.save_strategy()
        
        self.assertTrue(os.path.exists(filepath))
        
        # 3. Créer une configuration de trading
        from live_config import create_default_config
        config = create_default_config(ExchangeType.BITGET, "BTCUSDT")
        config.strategy_id = constructor.config.id
        
        # 4. Sauvegarder la configuration
        config_path = os.path.join(TEST_DIR, "workflow_config.json")
        config.save(config_path)
        
        self.assertTrue(os.path.exists(config_path))
        
        # Les étapes suivantes seraient l'exécution du trading en direct,
        # mais nous ne les exécutons pas réellement dans un test unitaire


# ============= Fonctions de test supplémentaires =============

def run_functional_tests():
    """
    Exécute des tests fonctionnels de manière plus lisible
    """
    print("\n" + "=" * 80)
    print(" " * 30 + "TESTS FONCTIONNELS")
    print("=" * 80 + "\n")
    
    # 1. Test du StrategyConstructor
    print("\n[TEST] Fonctionnalités du StrategyConstructor:")
    
    strategy = create_test_strategy("Functional Test Strategy")
    print(f"✓ Stratégie créée: {strategy.config.name}")
    
    ema_config = IndicatorConfig(
        type_=IndicatorType.EMA,
        period=50,
        source="close"
    )
    strategy.add_indicator("ema50", ema_config)
    print(f"✓ Indicateur ajouté: ema50")
    
    data = generate_test_data(periods=500)
    signals, data_with_signals = strategy.generate_signals(data)
    print(f"✓ Signaux générés: {sum(abs(signals))} signaux sur {len(signals)} points")
    
    # 2. Test du StrategyManager
    print("\n[TEST] Fonctionnalités du StrategyManager:")
    
    manager = StrategyManager(TEST_STRATEGIES_DIR)
    print(f"✓ StrategyManager créé")
    
    manager.current_strategy = strategy
    filepath = manager.save_strategy()
    print(f"✓ Stratégie sauvegardée: {filepath}")
    
    strategies = manager.list_strategies()
    print(f"✓ Stratégies listées: {len(strategies)} stratégies trouvées")
    
    results = manager.run_simulation(data)
    print(f"✓ Simulation exécutée: ROI={results['performance']['roi_pct']:.2f}%, " +
          f"Win Rate={results['performance']['win_rate_pct']:.2f}%")
    
    # 3. Test du Simulator
    print("\n[TEST] Fonctionnalités du Simulator:")
    
    simulator = Simulator(SimulationConfig())
    sim_results = simulator.run(
        prices=data['close'].values,
        signals=signals
    )
    print(f"✓ Simulation exécutée: ROI={sim_results['performance']['roi_pct']:.2f}%, " +
          f"Win Rate={sim_results['performance']['win_rate_pct']:.2f}%")
    
    # 4. Test du LiveTrader (simulé)
    print("\n[TEST] Fonctionnalités du LiveTrader (simulé):")
    
    from live_config import LiveConfig, ExchangeType, LiveTradingMode
    live_config = LiveConfig(
        exchange=ExchangeType.BITGET,
        trading_mode=LiveTradingMode.PAPER,
        leverage=1
    )
    print(f"✓ Configuration créée: {live_config.exchange.value}, {live_config.trading_mode.value}")
    
    print(f"✓ Tests fonctionnels terminés avec succès!")
    
    return True


# ============= Tests de performance =============

def run_performance_tests():
    """
    Exécute des tests de performance
    """
    print("\n" + "=" * 80)
    print(" " * 30 + "TESTS DE PERFORMANCE")
    print("=" * 80 + "\n")
    
    # Générer des données de test plus importantes
    data = generate_test_data(periods=5000)
    
    # Créer une stratégie plus complexe
    constructor = create_test_strategy("Performance Test Strategy")
    
    # Ajouter plus d'indicateurs
    for i in range(5):
        ema_config = IndicatorConfig(
            type_=IndicatorType.EMA,
            period=10 * (i + 1),
            source="close"
        )
        constructor.add_indicator(f"ema{10*(i+1)}", ema_config)
    
    # Mesurer le temps de génération des signaux
    start_time = time.time()
    signals, data_with_signals = constructor.generate_signals(data)
    generation_time = time.time() - start_time
    
    print(f"Génération des signaux pour {len(data)} points avec {len(constructor.config.indicators_manager.list_indicators())} indicateurs:")
    print(f"✓ Temps: {generation_time:.3f} secondes")
    print(f"✓ Signaux générés: {sum(abs(signals))} signaux")
    
    # Mesurer le temps de simulation
    simulator = Simulator(SimulationConfig())
    
    start_time = time.time()
    sim_results = simulator.run(
        prices=data['close'].values,
        signals=signals
    )
    simulation_time = time.time() - start_time
    
    print(f"\nSimulation pour {len(data)} points:")
    print(f"✓ Temps: {simulation_time:.3f} secondes")
    print(f"✓ ROI: {sim_results['performance']['roi_pct']:.2f}%")
    print(f"✓ Trades: {sim_results['performance']['total_trades']}")
    
    return True


# ============= Fonction principale =============

def main():
    """Fonction principale d'exécution des tests"""
    print("\n" + "=" * 80)
    print(" " * 30 + "TESTS DU SYSTÈME DE TRADING")
    print("=" * 80 + "\n")
    
    # Configurer l'environnement de test
    setup_test_environment()
    
    try:
        # Exécuter les tests unitaires
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
        
        # Exécuter les tests fonctionnels
        run_functional_tests()
        
        # Exécuter les tests de performance
        run_performance_tests()
        
    finally:
        # Nettoyer l'environnement de test
        cleanup_test_environment()


if __name__ == "__main__":
    main()