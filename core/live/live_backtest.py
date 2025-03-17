"""
Module pour exécuter des backtests sur les données récentes.
Permet de comparer les performances de la stratégie en temps réel avec les backtests.
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

# Import des modules existants
from core.strategy.constructor.constructor import StrategyConstructor
from core.simulation.simulator import Simulator
from core.simulation.simulation_config import SimulationConfig
from core.live.live_config import LiveConfig

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("live_backtest")


class LiveBacktest:
    """
    Exécute des backtests sur les données récentes pour comparer avec les performances en temps réel.
    """
    
    def __init__(
        self,
        strategy: StrategyConstructor,
        config: LiveConfig,
        results_dir: str = "results/live_backtest"
    ):
        """
        Initialise le backtest en direct.
        
        Args:
            strategy: Constructeur de stratégie
            config: Configuration du trading en direct
            results_dir: Répertoire pour les résultats
        """
        self.strategy = strategy
        self.config = config
        self.results_dir = results_dir
        
        # Créer le répertoire des résultats
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialiser le simulateur
        self.simulator = Simulator(self._create_sim_config())
        
        # Stockage des résultats
        self.last_results = None
        self.backtest_history = []
    
    def _create_sim_config(self) -> SimulationConfig:
        """
        Crée une configuration de simulation basée sur la configuration live.
        
        Returns:
            SimulationConfig: Configuration de simulation
        """
        from core.simulation.simulation_config import MarginMode, TradingMode
        
        # Conversion des types d'énumération
        margin_mode = MarginMode.ISOLATED if self.config.margin_mode.value == "isolated" else MarginMode.CROSS
        trading_mode = TradingMode.HEDGE if self.config.position_mode.value == "hedge" else TradingMode.ONE_WAY
        
        # Créer la configuration
        sim_config = SimulationConfig(
            initial_balance=10000.0,  # Balance fictive pour la comparaison
            fee_open=0.001,           # Frais standard
            fee_close=0.001,
            slippage=0.0005,          # Slippage typique
            tick_size=self.config.market.tick_size,
            min_trade_size=self.config.market.min_order_size,
            max_trade_size=10000.0,   # Limite haute pour éviter les erreurs
            leverage=self.config.leverage,
            margin_mode=margin_mode,
            trading_mode=trading_mode
        )
        
        return sim_config
    
    def run_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Exécute un backtest sur les données fournies.
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            Dict: Résultats du backtest
        """
        if data is None or len(data) < 100:
            logger.warning("Données insuffisantes pour le backtest")
            return None
        
        try:
            logger.info(f"Exécution du backtest sur {len(data)} points de données")
            
            # Générer les signaux avec la stratégie
            signals, data_with_signals = self.strategy.generate_signals(data)
            
            # Vérifier que des signaux ont été générés
            if sum(abs(signals)) == 0:
                logger.warning("Aucun signal généré pour le backtest")
                return {
                    "success": False,
                    "message": "Aucun signal généré",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Extraire les paramètres de risque
            position_sizes = data_with_signals['position_size'].values if 'position_size' in data_with_signals.columns else None
            sl_levels = data_with_signals['sl_level'].values if 'sl_level' in data_with_signals.columns else None
            tp_levels = data_with_signals['tp_level'].values if 'tp_level' in data_with_signals.columns else None
            
            # Exécuter la simulation
            results = self.simulator.run(
                prices=data_with_signals['close'].values,
                signals=signals,
                position_sizes=position_sizes,
                sl_levels=sl_levels,
                tp_levels=tp_levels
            )
            
            # Stocker les résultats
            self.last_results = results
            
            # Ajouter un historique du backtest
            backtest_record = {
                "timestamp": datetime.now().isoformat(),
                "data_points": len(data),
                "data_start": data.index[0].strftime('%Y-%m-%d %H:%M:%S') if hasattr(data.index[0], 'strftime') else str(data.index[0]),
                "data_end": data.index[-1].strftime('%Y-%m-%d %H:%M:%S') if hasattr(data.index[-1], 'strftime') else str(data.index[-1]),
                "trades": results['performance']['total_trades'],
                "roi": results['performance']['roi'],
                "roi_pct": results['performance']['roi_pct'],
                "win_rate": results['performance']['win_rate'],
                "win_rate_pct": results['performance']['win_rate_pct'],
                "max_drawdown": results['performance']['max_drawdown'],
                "max_drawdown_pct": results['performance']['max_drawdown_pct']
            }
            
            self.backtest_history.append(backtest_record)
            
            # Sauvegarder les résultats
            self._save_results(results, data_with_signals)
            
            logger.info(f"Backtest terminé: ROI={results['performance']['roi_pct']:.2f}%, " + 
                       f"Trades={results['performance']['total_trades']}, " +
                       f"Win Rate={results['performance']['win_rate_pct']:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors du backtest: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def compare_with_live(self, live_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare les résultats du backtest avec les métriques de trading en direct.
        
        Args:
            live_metrics: Métriques du trading en direct
            
        Returns:
            Dict: Comparaison des performances
        """
        if self.last_results is None:
            logger.warning("Aucun résultat de backtest disponible pour la comparaison")
            return None
        
        try:
            backtest_metrics = self.last_results['performance']
            
            # Calculer les différences
            comparison = {
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "backtest": {
                        "roi_pct": backtest_metrics['roi_pct'],
                        "total_trades": backtest_metrics['total_trades'],
                        "win_rate_pct": backtest_metrics['win_rate_pct'],
                        "max_drawdown_pct": backtest_metrics['max_drawdown_pct'],
                        "profit_factor": backtest_metrics['profit_factor']
                    },
                    "live": {
                        "roi_pct": (live_metrics['total_profit_loss'] / 10000.0) * 100,
                        "total_trades": live_metrics['total_trades'],
                        "win_rate_pct": (live_metrics['winning_trades'] / max(1, live_metrics['total_trades'])) * 100,
                        "max_drawdown_pct": live_metrics['max_drawdown'] * 100,
                        "profit_factor": abs(live_metrics['avg_win'] / max(0.01, abs(live_metrics['avg_loss'])))
                    }
                },
                "differences": {}
            }
            
            # Calculer les différences en pourcentage
            for metric in ['roi_pct', 'win_rate_pct', 'max_drawdown_pct', 'profit_factor']:
                backtest_value = comparison['metrics']['backtest'][metric]
                live_value = comparison['metrics']['live'][metric]
                
                if backtest_value != 0:
                    diff_pct = ((live_value - backtest_value) / abs(backtest_value)) * 100
                else:
                    diff_pct = 0
                
                comparison['differences'][metric] = {
                    "absolute": live_value - backtest_value,
                    "percentage": diff_pct
                }
            
            # Ajouter la différence pour les trades
            backtest_trades = comparison['metrics']['backtest']['total_trades']
            live_trades = comparison['metrics']['live']['total_trades']
            
            if backtest_trades != 0:
                trades_diff_pct = ((live_trades - backtest_trades) / backtest_trades) * 100
            else:
                trades_diff_pct = 0
            
            comparison['differences']['total_trades'] = {
                "absolute": live_trades - backtest_trades,
                "percentage": trades_diff_pct
            }
            
            # Calculer une note globale de cohérence (0-100%)
            consistency_factors = [
                min(100, 100 - min(abs(comparison['differences']['roi_pct']['percentage']), 100)),
                min(100, 100 - min(abs(comparison['differences']['win_rate_pct']['percentage']), 100)),
                min(100, 100 - min(abs(comparison['differences']['max_drawdown_pct']['percentage']), 100)),
                min(100, 100 - min(abs(comparison['differences']['total_trades']['percentage']), 100))
            ]
            
            comparison['consistency_score'] = sum(consistency_factors) / len(consistency_factors)
            
            # Interpréter le score de cohérence
            if comparison['consistency_score'] >= 80:
                comparison['consistency_rating'] = "Excellent"
            elif comparison['consistency_score'] >= 60:
                comparison['consistency_rating'] = "Bon"
            elif comparison['consistency_score'] >= 40:
                comparison['consistency_rating'] = "Moyen"
            else:
                comparison['consistency_rating'] = "Faible"
            
            # Sauvegarder la comparaison
            self._save_comparison(comparison)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Erreur lors de la comparaison des performances: {str(e)}")
            return None
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Génère un rapport de performance du backtest.
        
        Returns:
            Dict: Rapport de performance
        """
        if self.last_results is None:
            logger.warning("Aucun résultat de backtest disponible pour le rapport")
            return None
        
        try:
            # Extraire les informations sur la stratégie
            strategy_info = {
                "name": self.strategy.config.name,
                "id": self.strategy.config.id,
                "description": self.strategy.config.description,
                "tags": self.strategy.config.tags
            }
            
            # Extraire les métriques de performance
            perf = self.last_results['performance']
            
            # Créer le rapport
            report = {
                "strategy_info": strategy_info,
                "market_info": {
                    "symbol": self.config.market.symbol,
                    "timeframe": self.config.market.timeframe,
                    "exchange": self.config.exchange.value
                },
                "backtest_info": {
                    "timestamp": datetime.now().isoformat(),
                    "history": self.backtest_history[-5:] if len(self.backtest_history) > 0 else []
                },
                "performance": {
                    "roi": f"{perf['roi_pct']:.2f}%",
                    "total_trades": perf['total_trades'],
                    "win_rate": f"{perf['win_rate_pct']:.2f}%",
                    "max_drawdown": f"{perf['max_drawdown_pct']:.2f}%",
                    "profit_factor": f"{perf['profit_factor']:.2f}",
                    "sharpe_ratio": perf.get('sharpe_ratio', 'N/A')
                },
                "detailed_metrics": {
                    **perf
                },
                "simulation_config": self.last_results['config']
            }
            
            # Sauvegarder le rapport
            report_path = os.path.join(self.results_dir, "performance_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=4)
            
            return report
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport de performance: {str(e)}")
            return None
    
    def _save_results(self, results: Dict[str, Any], data_with_signals: pd.DataFrame) -> None:
        """
        Sauvegarde les résultats du backtest.
        
        Args:
            results: Résultats du backtest
            data_with_signals: Données avec signaux
        """
        try:
            # Créer un sous-répertoire avec horodatage
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.results_dir, f"backtest_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Sauvegarder les métriques de performance
            performance_path = os.path.join(output_dir, "performance.json")
            with open(performance_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
            
            # Sauvegarder les données avec signaux
            data_path = os.path.join(output_dir, "data_with_signals.csv")
            data_with_signals.to_csv(data_path)
            
            # Sauvegarder la configuration du backtest
            config_path = os.path.join(output_dir, "backtest_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.simulator.config.__dict__, f, indent=4)
            
            logger.info(f"Résultats du backtest sauvegardés dans {output_dir}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats: {str(e)}")
    
    def _save_comparison(self, comparison: Dict[str, Any]) -> None:
        """
        Sauvegarde la comparaison des performances.
        
        Args:
            comparison: Comparaison des performances
        """
        try:
            comparison_path = os.path.join(self.results_dir, "live_comparison.json")
            
            # Charger les comparaisons précédentes si elles existent
            history = []
            if os.path.exists(comparison_path):
                with open(comparison_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    history = data.get('history', [])
            
            # Limiter l'historique aux 10 dernières comparaisons
            history.append(comparison)
            if len(history) > 10:
                history = history[-10:]
            
            # Créer le document complet
            comparison_data = {
                "last_updated": datetime.now().isoformat(),
                "latest": comparison,
                "history": history
            }
            
            # Sauvegarder
            with open(comparison_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_data, f, indent=4)
                
            logger.info(f"Comparaison des performances sauvegardée dans {comparison_path}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la comparaison: {str(e)}")