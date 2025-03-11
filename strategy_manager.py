"""
Strategy Manager Module

Handles the export, management, and usage of optimized trading strategies
from Optuna studies. This module bridges the gap between the study optimization
and the actual deployment of strategies.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import uuid

from study_manager import StudyManager, StudyMetadata, StudyPerformance, StudyStatus
from simulator.simulator import Simulator, SimulationConfig, prepare_risk_params

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('strategy_manager')


class StrategyConfig:
    """Configuration for a strategy extracted from study results"""
    
    def __init__(self, 
                 strategy_id: str = None, 
                 name: str = "", 
                 description: str = "",
                 study_name: str = "", 
                 params: Dict = None,
                 risk_config: Dict = None,
                 simulation_config: Optional[SimulationConfig] = None):
        """
        Initialize strategy configuration
        
        Args:
            strategy_id: Unique ID for the strategy
            name: User-friendly name for the strategy
            description: Detailed description of the strategy
            study_name: Name of the source study
            params: Strategy parameters from study
            risk_config: Risk management configuration
            simulation_config: Simulation configuration
        """
        self.strategy_id = strategy_id or f"strategy_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.description = description
        self.study_name = study_name
        self.params = params or {}
        self.risk_config = risk_config or {}
        self.simulation_config = simulation_config or SimulationConfig()
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.modified_at = self.created_at
        
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "description": self.description,
            "study_name": self.study_name,
            "params": self.params,
            "risk_config": self.risk_config,
            "simulation_config": self.simulation_config.to_dict() if self.simulation_config else {},
            "created_at": self.created_at,
            "modified_at": self.modified_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyConfig':
        """Create configuration from dictionary"""
        sim_config = None
        if 'simulation_config' in data and data['simulation_config']:
            sim_config = SimulationConfig.from_dict(data['simulation_config'])
            
        return cls(
            strategy_id=data.get('strategy_id'),
            name=data.get('name', ''),
            description=data.get('description', ''),
            study_name=data.get('study_name', ''),
            params=data.get('params', {}),
            risk_config=data.get('risk_config', {}),
            simulation_config=sim_config
        )


class StrategyPerformance:
    """Performance metrics for a strategy"""
    
    def __init__(self, 
                 strategy_id: str,
                 roi: float = 0.0,
                 win_rate: float = 0.0,
                 max_drawdown: float = 0.0,
                 total_trades: int = 0,
                 profit_factor: float = 0.0,
                 sharpe_ratio: float = 0.0,
                 avg_profit_per_trade: float = 0.0,
                 data_info: Dict = None):
        """
        Initialize performance metrics
        
        Args:
            strategy_id: Unique ID of the strategy
            roi: Return on investment as decimal
            win_rate: Win rate as decimal
            max_drawdown: Maximum drawdown as decimal
            total_trades: Total number of trades
            profit_factor: Profit factor
            sharpe_ratio: Sharpe ratio
            avg_profit_per_trade: Average profit per trade
            data_info: Information about the data used for testing
        """
        self.strategy_id = strategy_id
        self.roi = roi
        self.win_rate = win_rate
        self.max_drawdown = max_drawdown
        self.total_trades = total_trades
        self.profit_factor = profit_factor
        self.sharpe_ratio = sharpe_ratio
        self.avg_profit_per_trade = avg_profit_per_trade
        self.data_info = data_info or {}
        self.test_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def to_dict(self) -> Dict:
        """Convert performance metrics to dictionary"""
        return {
            "strategy_id": self.strategy_id,
            "roi": self.roi,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "total_trades": self.total_trades,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "avg_profit_per_trade": self.avg_profit_per_trade,
            "data_info": self.data_info,
            "test_date": self.test_date
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyPerformance':
        """Create performance metrics from dictionary"""
        return cls(
            strategy_id=data.get('strategy_id', ''),
            roi=data.get('roi', 0.0),
            win_rate=data.get('win_rate', 0.0),
            max_drawdown=data.get('max_drawdown', 0.0),
            total_trades=data.get('total_trades', 0),
            profit_factor=data.get('profit_factor', 0.0),
            sharpe_ratio=data.get('sharpe_ratio', 0.0),
            avg_profit_per_trade=data.get('avg_profit_per_trade', 0.0),
            data_info=data.get('data_info', {})
        )


class SignalGenerator:
    """Base class for generating trading signals from strategy parameters"""
    
    def __init__(self, params: Dict):
        """
        Initialize signal generator
        
        Args:
            params: Strategy parameters
        """
        self.params = params
    
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate trading signals from data
        
        Args:
            data: Price data
            
        Returns:
            np.ndarray: Array of signals (1=buy, -1=sell, 0=hold)
        """
        # This is a base class - override in strategy-specific subclasses
        raise NotImplementedError("Subclasses must implement generate_signals")
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for signal generation
        
        Args:
            data: Raw price data
            
        Returns:
            pd.DataFrame: Prepared data
        """
        # Basic data validation and preparation
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_columns if col not in data.columns]
        
        if missing_cols:
            # Try to adapt to different column names
            if 'price' in data.columns:
                # Single price column - create OHLC
                data['open'] = data['close'] = data['high'] = data['low'] = data['price']
            else:
                raise ValueError(f"Missing required columns: {missing_cols}")
        
        if 'volume' not in data.columns:
            # Add dummy volume if missing
            data['volume'] = 0
        
        return data


class MASignalGenerator(SignalGenerator):
    """Moving Average signal generator"""
    
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate signals based on moving average crossovers
        
        Args:
            data: Price data
            
        Returns:
            np.ndarray: Array of signals
        """
        df = self.prepare_data(data)
        
        # Extract parameters
        fast_period = self.params.get('fast_period', 10)
        slow_period = self.params.get('slow_period', 30)
        signal_period = self.params.get('signal_period', 9)
        
        # Calculate moving averages
        df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=slow_period).mean()
        
        # Generate signals
        signals = np.zeros(len(df))
        
        for i in range(slow_period, len(df)):
            # Moving average crossover
            if df['fast_ma'].iloc[i] > df['slow_ma'].iloc[i] and df['fast_ma'].iloc[i-1] <= df['slow_ma'].iloc[i-1]:
                signals[i] = 1  # Buy signal
            elif df['fast_ma'].iloc[i] < df['slow_ma'].iloc[i] and df['fast_ma'].iloc[i-1] >= df['slow_ma'].iloc[i-1]:
                signals[i] = -1  # Sell signal
        
        return signals


class RSISignalGenerator(SignalGenerator):
    """RSI signal generator"""
    
    def calculate_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI indicator"""
        # Calculate price changes
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        
        # Calculate gains and losses
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            # Avoid division by zero
            rs = np.inf
        else:
            rs = up / down
        
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        # Calculate RSI using smoothed moving average
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
                
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            
            if down == 0:
                rs = np.inf
            else:
                rs = up / down
                
            rsi[i] = 100. - 100. / (1. + rs)
            
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate signals based on RSI
        
        Args:
            data: Price data
            
        Returns:
            np.ndarray: Array of signals
        """
        df = self.prepare_data(data)
        
        # Extract parameters
        period = self.params.get('rsi_period', 14)
        oversold = self.params.get('oversold', 30)
        overbought = self.params.get('overbought', 70)
        
        # Calculate RSI
        rsi = self.calculate_rsi(df['close'].values, period)
        
        # Generate signals
        signals = np.zeros(len(df))
        
        for i in range(period+1, len(df)):
            # RSI crosses below oversold -> buy
            if rsi[i] < oversold and rsi[i-1] >= oversold:
                signals[i] = 1
            # RSI crosses above overbought -> sell
            elif rsi[i] > overbought and rsi[i-1] <= overbought:
                signals[i] = -1
        
        return signals


class MACDSignalGenerator(SignalGenerator):
    """MACD signal generator"""
    
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate signals based on MACD
        
        Args:
            data: Price data
            
        Returns:
            np.ndarray: Array of signals
        """
        df = self.prepare_data(data)
        
        # Extract parameters
        fast_period = self.params.get('macd_fast', 12)
        slow_period = self.params.get('macd_slow', 26)
        signal_period = self.params.get('macd_signal', 9)
        
        # Calculate EMA
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD and signal line
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        
        # Generate signals
        signals = np.zeros(len(df))
        
        for i in range(slow_period + signal_period, len(df)):
            # MACD crosses above signal line -> buy
            if macd.iloc[i] > signal_line.iloc[i] and macd.iloc[i-1] <= signal_line.iloc[i-1]:
                signals[i] = 1
            # MACD crosses below signal line -> sell
            elif macd.iloc[i] < signal_line.iloc[i] and macd.iloc[i-1] >= signal_line.iloc[i-1]:
                signals[i] = -1
        
        return signals


class Strategy:
    """Complete trading strategy with signal generation and risk management"""
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize strategy
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.performance = None
        self.simulator = None
        
        # Determine signal generator based on parameters
        self.signal_generator = self._get_signal_generator()
    
    def _get_signal_generator(self) -> SignalGenerator:
        """Get appropriate signal generator based on parameters"""
        # Check for strategy type indicators in parameters
        params = self.config.params
        
        if 'rsi_period' in params:
            return RSISignalGenerator(params)
        elif 'macd_fast' in params or 'macd_slow' in params:
            return MACDSignalGenerator(params)
        else:
            # Default to moving average
            return MASignalGenerator(params)
    
    def backtest(self, data: pd.DataFrame, save_to: Optional[str] = None) -> Dict:
        """
        Backtest the strategy
        
        Args:
            data: Price data
            save_to: Path to save results (optional)
            
        Returns:
            Dict: Backtest results
        """
        if isinstance(data, str):
            # Load data from file path
            if data.endswith('.csv'):
                data = pd.read_csv(data)
            elif data.endswith('.json'):
                data = pd.read_json(data)
            else:
                raise ValueError(f"Unsupported data format: {data}")
        
        # Prepare data
        df = self.signal_generator.prepare_data(data)
        
        # Generate signals
        signals = self.signal_generator.generate_signals(df)
        
        # Prepare risk parameters
        risk_config = self.config.risk_config
        position_sizes, sl_levels, tp_levels = prepare_risk_params(
            risk_type=risk_config.get('risk_type', 'fixed'),
            base_position=risk_config.get('position_size', 0.1),
            base_sl=risk_config.get('stop_loss', 0.01),
            tp_multiplier=risk_config.get('tp_multiplier', 2.0),
            prices=df['close'].values,
            high=df['high'].values,
            low=df['low'].values,
            atr_period=risk_config.get('atr_period', 14),
            atr_multiplier=risk_config.get('atr_multiplier', 1.5),
            vol_period=risk_config.get('vol_period', 20),
            vol_multiplier=risk_config.get('vol_multiplier', 1.0)
        )
        
        # Create simulator
        self.simulator = Simulator(self.config.simulation_config)
        
        # Run simulation
        leverage = np.full_like(signals, self.config.simulation_config.leverage, dtype=np.float64)
        results = self.simulator.run(
            prices=df['close'].values,
            signals=signals,
            position_sizes=position_sizes,
            sl_levels=sl_levels,
            tp_levels=tp_levels,
            leverage_levels=leverage
        )
        
        # Save results if requested
        if save_to:
            save_dir = os.path.dirname(save_to)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                
            self.simulator.save_history_to_csv(save_to)
        
        # Store performance metrics
        self.performance = StrategyPerformance(
            strategy_id=self.config.strategy_id,
            roi=results['performance']['roi'],
            win_rate=results['performance']['win_rate'],
            max_drawdown=results['performance']['max_drawdown'],
            total_trades=results['performance']['total_trades'],
            profit_factor=results['performance']['profit_factor'],
            sharpe_ratio=results['performance'].get('sharpe_ratio', 0.0),
            avg_profit_per_trade=results['performance']['avg_profit_per_trade'],
            data_info={
                'rows': len(df),
                'start_date': df.index[0].strftime('%Y-%m-%d') if hasattr(df.index[0], 'strftime') else str(df.index[0]),
                'end_date': df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else str(df.index[-1]),
                'source': getattr(data, 'name', 'unknown')
            }
        )
        
        return results
    
    def save(self, filepath: str) -> None:
        """
        Save strategy to file
        
        Args:
            filepath: Path to save strategy
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare data
        save_data = {
            'config': self.config.to_dict(),
            'performance': self.performance.to_dict() if self.performance else None
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=4)
    
    @classmethod
    def load(cls, filepath: str) -> 'Strategy':
        """
        Load strategy from file
        
        Args:
            filepath: Path to strategy file
            
        Returns:
            Strategy: Loaded strategy
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = StrategyConfig.from_dict(data['config'])
        strategy = cls(config)
        
        if data.get('performance'):
            strategy.performance = StrategyPerformance.from_dict(data['performance'])
        
        return strategy
    
    def plot_equity_curve(self, save_path: Optional[str] = None):
        """
        Plot equity curve if simulation has been run
        
        Args:
            save_path: Path to save plot image (optional)
        """
        if not self.simulator:
            raise ValueError("No simulation has been run yet")
            
        self.simulator.plot_equity_curve(save_path)
    
    def get_pinescript(self) -> str:
        """
        Generate PineScript code for the strategy
        
        Returns:
            str: PineScript code
        """
        params = self.config.params
        risk = self.config.risk_config
        
        # Determine which strategy type to generate
        if isinstance(self.signal_generator, RSISignalGenerator):
            script = self._generate_rsi_pinescript()
        elif isinstance(self.signal_generator, MACDSignalGenerator):
            script = self._generate_macd_pinescript()
        else:
            script = self._generate_ma_pinescript()
        
        return script
    
    def _generate_ma_pinescript(self) -> str:
        """Generate PineScript for MA strategy"""
        params = self.config.params
        risk = self.config.risk_config
        
        # Extract parameters
        fast_period = params.get('fast_period', 10)
        slow_period = params.get('slow_period', 30)
        position_size = risk.get('position_size', 0.1) * 100
        stop_loss = risk.get('stop_loss', 0.01) * 100
        tp_mult = risk.get('tp_multiplier', 2.0)
        take_profit = stop_loss * tp_mult
        
        script = f"""
//@version=5
strategy("{self.config.name or 'MA Crossover Strategy'}", 
         overlay=true, 
         initial_capital={self.config.simulation_config.initial_balance}, 
         default_qty_type=strategy.percent_of_equity, 
         default_qty_value={position_size},
         commission_type=strategy.commission.percent, 
         commission_value={self.config.simulation_config.fee_open * 100})

// Input parameters
fastLength = input.int({fast_period}, "Fast MA Length", minval=1)
slowLength = input.int({slow_period}, "Slow MA Length", minval=1)
stopLossPercent = input.float({stop_loss}, "Stop Loss %", minval=0.1)
takeProfitPercent = input.float({take_profit}, "Take Profit %", minval=0.1)

// Calculate moving averages
fastMA = ta.sma(close, fastLength)
slowMA = ta.sma(close, slowLength)

// Plot moving averages
plot(fastMA, "Fast MA", color=color.blue)
plot(slowMA, "Slow MA", color=color.red)

// Generate signals
buySignal = ta.crossover(fastMA, slowMA)
sellSignal = ta.crossunder(fastMA, slowMA)

// Execute strategy
if (buySignal)
    strategy.entry("Long", strategy.long)

if (sellSignal)
    strategy.close("Long")

// Set stop loss and take profit
strategy.exit("TP/SL", "Long", 
    profit=takeProfitPercent/100 * close, 
    loss=stopLossPercent/100 * close)
"""
        return script
    
    def _generate_rsi_pinescript(self) -> str:
        """Generate PineScript for RSI strategy"""
        params = self.config.params
        risk = self.config.risk_config
        
        # Extract parameters
        rsi_period = params.get('rsi_period', 14)
        oversold = params.get('oversold', 30)
        overbought = params.get('overbought', 70)
        position_size = risk.get('position_size', 0.1) * 100
        stop_loss = risk.get('stop_loss', 0.01) * 100
        tp_mult = risk.get('tp_multiplier', 2.0)
        take_profit = stop_loss * tp_mult
        
        script = f"""
//@version=5
strategy("{self.config.name or 'RSI Strategy'}", 
         overlay=false, 
         initial_capital={self.config.simulation_config.initial_balance}, 
         default_qty_type=strategy.percent_of_equity, 
         default_qty_value={position_size},
         commission_type=strategy.commission.percent, 
         commission_value={self.config.simulation_config.fee_open * 100})

// Input parameters
rsiLength = input.int({rsi_period}, "RSI Length", minval=1)
oversoldLevel = input.int({oversold}, "Oversold Level", minval=1, maxval=100)
overboughtLevel = input.int({overbought}, "Overbought Level", minval=1, maxval=100)
stopLossPercent = input.float({stop_loss}, "Stop Loss %", minval=0.1)
takeProfitPercent = input.float({take_profit}, "Take Profit %", minval=0.1)

// Calculate RSI
rsiValue = ta.rsi(close, rsiLength)

// Plot RSI and levels
plot(rsiValue, "RSI", color=color.blue)
hline(oversoldLevel, "Oversold", color=color.green)
hline(overboughtLevel, "Overbought", color=color.red)

// Generate signals
buySignal = ta.crossover(rsiValue, oversoldLevel)
sellSignal = ta.crossunder(rsiValue, overboughtLevel)

// Execute strategy
if (buySignal)
    strategy.entry("Long", strategy.long)

if (sellSignal)
    strategy.close("Long")

// Set stop loss and take profit
strategy.exit("TP/SL", "Long", 
    profit=takeProfitPercent/100 * close, 
    loss=stopLossPercent/100 * close)
"""
        return script
    
    def _generate_macd_pinescript(self) -> str:
        """Generate PineScript for MACD strategy"""
        params = self.config.params
        risk = self.config.risk_config
        
        # Extract parameters
        fast_length = params.get('macd_fast', 12)
        slow_length = params.get('macd_slow', 26)
        signal_length = params.get('macd_signal', 9)
        position_size = risk.get('position_size', 0.1) * 100
        stop_loss = risk.get('stop_loss', 0.01) * 100
        tp_mult = risk.get('tp_multiplier', 2.0)
        take_profit = stop_loss * tp_mult
        
        script = f"""
//@version=5
strategy("{self.config.name or 'MACD Strategy'}", 
         overlay=false, 
         initial_capital={self.config.simulation_config.initial_balance}, 
         default_qty_type=strategy.percent_of_equity, 
         default_qty_value={position_size},
         commission_type=strategy.commission.percent, 
         commission_value={self.config.simulation_config.fee_open * 100})

// Input parameters
fastLength = input.int({fast_length}, "Fast Length", minval=1)
slowLength = input.int({slow_length}, "Slow Length", minval=1)
signalLength = input.int({signal_length}, "Signal Length", minval=1)
stopLossPercent = input.float({stop_loss}, "Stop Loss %", minval=0.1)
takeProfitPercent = input.float({take_profit}, "Take Profit %", minval=0.1)

// Calculate MACD
[macdLine, signalLine, histLine] = ta.macd(close, fastLength, slowLength, signalLength)

// Plot MACD
plot(macdLine, "MACD", color=color.blue)
plot(signalLine, "Signal", color=color.red)
plot(histLine, "Histogram", color=color.purple, style=plot.style_histogram)

// Generate signals
buySignal = ta.crossover(macdLine, signalLine)
sellSignal = ta.crossunder(macdLine, signalLine)

// Execute strategy
if (buySignal)
    strategy.entry("Long", strategy.long)

if (sellSignal)
    strategy.close("Long")

// Set stop loss and take profit
strategy.exit("TP/SL", "Long", 
    profit=takeProfitPercent/100 * close, 
    loss=stopLossPercent/100 * close)
"""
        return script


class StrategyManager:
    """Manager for creating, handling, and testing strategies"""
    
    def __init__(self, storage_dir: str = "strategies", study_manager: Optional[StudyManager] = None):
        """
        Initialize strategy manager
        
        Args:
            storage_dir: Directory for storing strategies
            study_manager: StudyManager instance (optional)
        """
        self.storage_dir = storage_dir
        self.study_manager = study_manager
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Cache for loaded strategies
        self.strategy_cache = {}
    
    def create_from_study(self, study_name: str, strategy_name: Optional[str] = None, 
                       description: Optional[str] = None) -> Strategy:
        """
        Create a strategy from a study
        
        Args:
            study_name: Name of the source study
            strategy_name: Custom name for the strategy (optional)
            description: Description of the strategy (optional)
            
        Returns:
            Strategy: Created strategy
        """
        if not self.study_manager:
            raise ValueError("StudyManager required to create strategy from study")
        
        # Get study metadata and performance
        metadata = self.study_manager.get_study_metadata(study_name)
        performance = self.study_manager.get_study_performance(study_name)
        
        if not metadata:
            raise ValueError(f"Study '{study_name}' not found")
        
        if not performance:
            raise ValueError(f"No performance data found for study '{study_name}'")
        
        # Extract parameters from study
        params = performance.best_params
        
        # Extract risk configuration from params
        risk_config = {
            "risk_type": params.get("risk_type", "fixed"),
            "position_size": params.get("position_size", 0.1),
            "stop_loss": params.get("stop_loss", 0.01),
            "tp_multiplier": params.get("tp_multiplier", 2.0),
            "atr_period": params.get("atr_period", 14),
            "atr_multiplier": params.get("atr_multiplier", 1.5),
            "vol_period": params.get("vol_period", 20),
            "vol_multiplier": params.get("vol_multiplier", 1.0)
        }
        
        # Create simulation config
        sim_config = SimulationConfig(
            initial_balance=metadata.config.get("initial_balance", 10000.0),
            fee_open=params.get("fee_open", 0.001),
            fee_close=params.get("fee_close", 0.001),
            slippage=params.get("slippage", 0.001),
            leverage=params.get("leverage", 1),
            margin_mode=params.get("margin_mode", 0),
            trading_mode=params.get("trading_mode", 0)
        )
        
        # Create strategy config
        strategy_config = StrategyConfig(
            name=strategy_name or f"Strategy from {study_name}",
            description=description or f"Strategy created from study {study_name}",
            study_name=study_name,
            params=params,
            risk_config=risk_config,
            simulation_config=sim_config
        )
        
        # Create strategy
        strategy = Strategy(strategy_config)
        
        # Save strategy
        self.save_strategy(strategy)
        
        return strategy
    
    def save_strategy(self, strategy: Strategy) -> str:
        """
        Save strategy to file
        
        Args:
            strategy: Strategy to save
            
        Returns:
            str: Path to saved strategy file
        """
        filepath = os.path.join(self.storage_dir, f"{strategy.config.strategy_id}.json")
        strategy.save(filepath)
        
        # Update cache
        self.strategy_cache[strategy.config.strategy_id] = strategy
        
        return filepath
    
    def load_strategy(self, strategy_id: str) -> Strategy:
        """
        Load strategy from file
        
        Args:
            strategy_id: ID of the strategy to load
            
        Returns:
            Strategy: Loaded strategy
        """
        # Check cache first
        if strategy_id in self.strategy_cache:
            return self.strategy_cache[strategy_id]
        
        filepath = os.path.join(self.storage_dir, f"{strategy_id}.json")
        if not os.path.exists(filepath):
            raise ValueError(f"Strategy '{strategy_id}' not found")
        
        strategy = Strategy.load(filepath)
        
        # Update cache
        self.strategy_cache[strategy_id] = strategy
        
        return strategy
    
    def list_strategies(self) -> List[Dict]:
        """
        List all available strategies
        
        Returns:
            List[Dict]: List of strategy info
        """
        strategies = []
        
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.storage_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    config_data = data.get('config', {})
                    performance_data = data.get('performance', {})
                    
                    strategies.append({
                        'strategy_id': config_data.get('strategy_id'),
                        'name': config_data.get('name', 'Unnamed Strategy'),
                        'study_name': config_data.get('study_name', ''),
                        'created_at': config_data.get('created_at', ''),
                        'modified_at': config_data.get('modified_at', ''),
                        'roi': performance_data.get('roi', 0.0) if performance_data else 0.0,
                        'win_rate': performance_data.get('win_rate', 0.0) if performance_data else 0.0,
                        'total_trades': performance_data.get('total_trades', 0) if performance_data else 0
                    })
                except Exception as e:
                    logger.warning(f"Error loading strategy from {filepath}: {e}")
        
        return strategies
    
    def delete_strategy(self, strategy_id: str) -> bool:
        """
        Delete a strategy
        
        Args:
            strategy_id: ID of the strategy to delete
            
        Returns:
            bool: True if deletion was successful
        """
        filepath = os.path.join(self.storage_dir, f"{strategy_id}.json")
        if not os.path.exists(filepath):
            return False
        
        try:
            os.remove(filepath)
            
            # Remove from cache
            if strategy_id in self.strategy_cache:
                del self.strategy_cache[strategy_id]
                
            return True
        except Exception as e:
            logger.error(f"Error deleting strategy {strategy_id}: {e}")
            return False
    
    def backtest_strategy(self, strategy_id: str, data: Union[pd.DataFrame, str], 
                        save_results: bool = True) -> Dict:
        """
        Backtest a strategy
        
        Args:
            strategy_id: ID of the strategy to backtest
            data: Price data or path to data file
            save_results: Whether to save backtest results
            
        Returns:
            Dict: Backtest results
        """
        strategy = self.load_strategy(strategy_id)
        
        # Generate save path if needed
        save_path = None
        if save_results:
            history_dir = os.path.join(self.storage_dir, "history", strategy_id)
            os.makedirs(history_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(history_dir, f"backtest_{timestamp}")
        
        # Run backtest
        results = strategy.backtest(data, save_path)
        
        # Save updated strategy with performance
        self.save_strategy(strategy)
        
        return results
    
    def compare_strategies(self, strategy_ids: List[str], data: Union[pd.DataFrame, str]) -> Dict:
        """
        Compare multiple strategies on the same data
        
        Args:
            strategy_ids: List of strategy IDs to compare
            data: Price data or path to data file
            
        Returns:
            Dict: Comparison results
        """
        # Load data if needed
        if isinstance(data, str):
            if data.endswith('.csv'):
                data = pd.read_csv(data)
            elif data.endswith('.json'):
                data = pd.read_json(data)
            else:
                raise ValueError(f"Unsupported data format: {data}")
        
        results = {}
        
        for strategy_id in strategy_ids:
            strategy = self.load_strategy(strategy_id)
            backtest_results = strategy.backtest(data, save_to=None)
            
            results[strategy_id] = {
                'name': strategy.config.name,
                'performance': backtest_results['performance']
            }
        
        return results
    
    def plot_comparison(self, strategy_ids: List[str], data: Union[pd.DataFrame, str], 
                       save_path: Optional[str] = None) -> None:
        """
        Plot comparison of multiple strategies
        
        Args:
            strategy_ids: List of strategy IDs to compare
            data: Price data or path to data file
            save_path: Path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            logger.error("matplotlib is required for plotting")
            return
        
        # Run comparison
        comparison = self.compare_strategies(strategy_ids, data)
        
        # Prepare data for plotting
        names = []
        roi_values = []
        drawdown_values = []
        win_rates = []
        trade_counts = []
        
        for strategy_id, result in comparison.items():
            names.append(result['name'])
            perf = result['performance']
            roi_values.append(perf['roi'] * 100)  # As percentage
            drawdown_values.append(perf['max_drawdown'] * 100)  # As percentage
            win_rates.append(perf['win_rate'] * 100)  # As percentage
            trade_counts.append(perf['total_trades'])
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # ROI subplot
        bars = axs[0, 0].bar(names, roi_values)
        axs[0, 0].set_title('Return on Investment (%)')
        axs[0, 0].tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            axs[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.2f}%', ha='center', va='bottom')
        
        # Drawdown subplot
        bars = axs[0, 1].bar(names, drawdown_values, color='red')
        axs[0, 1].set_title('Maximum Drawdown (%)')
        axs[0, 1].tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            axs[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.2f}%', ha='center', va='bottom')
        
        # Win rate subplot
        bars = axs[1, 0].bar(names, win_rates, color='green')
        axs[1, 0].set_title('Win Rate (%)')
        axs[1, 0].tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            axs[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.2f}%', ha='center', va='bottom')
        
        # Trade count subplot
        bars = axs[1, 1].bar(names, trade_counts, color='purple')
        axs[1, 1].set_title('Number of Trades')
        axs[1, 1].tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            axs[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Adjust layout
        fig.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def optimize_strategy_parameters(self, strategy_id: str, data: Union[pd.DataFrame, str], 
                                    n_trials: int = 100, verbose: bool = True) -> Strategy:
        """
        Optimize strategy parameters on given data
        
        Args:
            strategy_id: ID of the strategy to optimize
            data: Price data or path to data file
            n_trials: Number of optimization trials
            verbose: Whether to print optimization progress
            
        Returns:
            Strategy: Optimized strategy
        """
        try:
            import optuna
        except ImportError:
            logger.error("optuna is required for optimization")
            raise ImportError("optuna is required for optimization")
        
        # Load strategy
        strategy = self.load_strategy(strategy_id)
        original_params = strategy.config.params.copy()
        
        # Load data if needed
        if isinstance(data, str):
            if data.endswith('.csv'):
                data = pd.read_csv(data)
            elif data.endswith('.json'):
                data = pd.read_json(data)
            else:
                raise ValueError(f"Unsupported data format: {data}")
        
        # Prepare data for signal generation
        df = strategy.signal_generator.prepare_data(data)
        
        # Determine which parameters to optimize based on signal generator type
        param_ranges = self._get_parameter_ranges(strategy)
        
        # Define objective function
        def objective(trial):
            # Create parameters from trial
            params = {}
            for param_name, param_range in param_ranges.items():
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
            
            # Update strategy with new parameters
            strategy.config.params.update(params)
            strategy.signal_generator = strategy._get_signal_generator()
            
            # Run backtest
            results = strategy.backtest(df, save_to=None)
            
            # Calculate score - prioritize ROI, but consider other factors
            perf = results['performance']
            roi = perf['roi']
            win_rate = perf['win_rate']
            max_drawdown = perf['max_drawdown']
            total_trades = perf['total_trades']
            
            # Score calculation
            if total_trades < 5:  # Require minimum number of trades
                return -100
            
            # Balance ROI with other metrics
            score = (
                2.0 * roi - 
                1.0 * max_drawdown + 
                0.5 * win_rate + 
                0.1 * min(1.0, total_trades / 100)  # Cap at 100 trades
            )
            
            return score
        
        # Create Optuna study
        study = optuna.create_study(direction="maximize")
        
        # Run optimization
        if verbose:
            print(f"Optimizing strategy parameters with {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
        
        # Apply best parameters to strategy
        best_params = study.best_params
        if verbose:
            print(f"Best parameters: {best_params}")
            print(f"Best score: {study.best_value}")
        
        # Update strategy with best parameters
        strategy.config.params.update(best_params)
        strategy.signal_generator = strategy._get_signal_generator()
        
        # Run final backtest
        strategy.backtest(df, save_to=None)
        
        # Save optimized strategy
        optimized_id = f"{strategy_id}_optimized"
        strategy.config.strategy_id = optimized_id
        strategy.config.name = f"{strategy.config.name} (Optimized)"
        strategy.config.description = f"{strategy.config.description}\nOptimized from {strategy_id}"
        
        self.save_strategy(strategy)
        
        return strategy
    
    def _get_parameter_ranges(self, strategy: Strategy) -> Dict:
        """Get parameter ranges based on strategy type"""
        if isinstance(strategy.signal_generator, RSISignalGenerator):
            return {
                'rsi_period': (5, 30),
                'oversold': (20, 40),
                'overbought': (60, 80)
            }
        elif isinstance(strategy.signal_generator, MACDSignalGenerator):
            return {
                'macd_fast': (8, 20),
                'macd_slow': (20, 40),
                'macd_signal': (5, 15)
            }
        else:  # Default to MA
            return {
                'fast_period': (5, 50),
                'slow_period': (20, 200)
            }
    
    def create_combined_strategy(self, strategy_ids: List[str], name: str = "Combined Strategy") -> Strategy:
        """
        Create a strategy combining multiple existing strategies
        
        Args:
            strategy_ids: List of strategy IDs to combine
            name: Name for the combined strategy
            
        Returns:
            Strategy: Combined strategy
        """
        if len(strategy_ids) < 2:
            raise ValueError("At least two strategies are required for combining")
        
        # Load all strategies
        strategies = [self.load_strategy(sid) for sid in strategy_ids]
        
        # Create new combined signal generator
        class CombinedSignalGenerator(SignalGenerator):
            def __init__(self, component_strategies):
                self.component_strategies = component_strategies
                super().__init__({})
            
            def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
                df = self.prepare_data(data)
                
                # Generate signals from each component strategy
                all_signals = []
                for strategy in self.component_strategies:
                    signals = strategy.signal_generator.generate_signals(df)
                    all_signals.append(signals)
                
                # Combine signals (majority vote)
                combined = np.zeros(len(df))
                for i in range(len(df)):
                    votes = [signals[i] for signals in all_signals]
                    buy_votes = sum(1 for v in votes if v > 0)
                    sell_votes = sum(1 for v in votes if v < 0)
                    
                    if buy_votes > sell_votes and buy_votes > len(votes) / 3:
                        combined[i] = 1
                    elif sell_votes > buy_votes and sell_votes > len(votes) / 3:
                        combined[i] = -1
                
                return combined
        
        # Create combined strategy
        # Use configuration from first strategy as base
        base_config = strategies[0].config
        
        # Create new config
        combined_config = StrategyConfig(
            name=name,
            description=f"Combined strategy from {', '.join(strategy_ids)}",
            study_name=None,
            params={},
            risk_config=base_config.risk_config,
            simulation_config=base_config.simulation_config
        )
        
        # Create strategy
        combined = Strategy(combined_config)
        
        # Replace signal generator
        combined.signal_generator = CombinedSignalGenerator(strategies)
        
        # Save strategy
        self.save_strategy(combined)
        
        return combined