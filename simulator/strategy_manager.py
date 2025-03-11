"""
Gestionnaire de stratégies avancé qui permet de gérer, comparer et organiser des stratégies de trading.

Ce module simplifie la gestion des stratégies de trading en fournissant des 
fonctionnalités pour:

- Organiser les stratégies en collections (favorites, en test, archivées, etc.)
- Comparer visuellement les performances des stratégies
- Préparer des stratégies pour passer en production
- Gérer les métadonnées et les résultats de backtests

Author: Trading System Developer
Version: 3.0
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from enum import Enum
import datetime
import uuid
import shutil
import logging
import traceback
import time
from pathlib import Path
from copy import deepcopy

# Import des modules de trading
from simulator import Simulator, SimulationConfig
from generate_signals import SignalGenerator
from study_manager import IntegratedStudyManager

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy_manager.log', mode='a'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('strategy_manager')

class StrategyStatus(Enum):
    """Statut des stratégies dans le système"""
    NEW = "new"               # Nouvelle stratégie
    TESTING = "testing"       # En phase de test
    PRODUCTION = "production" # En production
    FAVORITE = "favorite"     # Stratégie favorite
    ARCHIVED = "archived"     # Stratégie archivée
    FAILED = "failed"         # Stratégie échouée aux tests

class StrategyManager:
    """
    Gestionnaire avancé de stratégies de trading.
    
    Cette classe permet de gérer efficacement un portefeuille de stratégies,
    de les organiser, de les comparer et de les préparer pour la production.
    """
    
    def __init__(self, base_dir: str = "strategies"):
        """
        Initialise le gestionnaire de stratégies.
        
        Args:
            base_dir: Répertoire de base pour les stratégies
        """
        self.base_dir = base_dir
        self.collections_dir = os.path.join(base_dir, "collections")
        self.study_manager = IntegratedStudyManager("studies")
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialise la structure de stockage sur disque."""
        # Création du répertoire principal s'il n'existe pas
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            logger.info(f"Répertoire principal créé: {self.base_dir}")
        
        # Création du répertoire des collections
        if not os.path.exists(self.collections_dir):
            os.makedirs(self.collections_dir)
            logger.info(f"Répertoire des collections créé: {self.collections_dir}")
            
            # Création des collections de base
            collections = ["favorites", "testing", "production", "archived"]
            for collection in collections:
                collection_file = os.path.join(self.collections_dir, f"{collection}.json")
                if not os.path.exists(collection_file):
                    with open(collection_file, 'w', encoding='utf-8') as f:
                        json.dump({"strategies": []}, f, indent=4)
        
        # Création des répertoires pour les stratégies
        strategies_dir = os.path.join(self.base_dir, "data")
        if not os.path.exists(strategies_dir):
            os.makedirs(strategies_dir)
            logger.info(f"Répertoire des données de stratégies créé: {strategies_dir}")
        
        # Répertoire pour les comparaisons et visualisations
        visuals_dir = os.path.join(self.base_dir, "visuals")
        if not os.path.exists(visuals_dir):
            os.makedirs(visuals_dir)
            logger.info(f"Répertoire des visualisations créé: {visuals_dir}")
    
    def create_strategy_from_study(self, 
                                  study_name: str, 
                                  strategy_rank: int = 1, 
                                  custom_name: Optional[str] = None,
                                  status: StrategyStatus = StrategyStatus.NEW) -> str:
        """
        Crée une stratégie à partir d'une étude existante.
        
        Args:
            study_name: Nom de l'étude source
            strategy_rank: Rang de la stratégie dans l'étude (1 = meilleure)
            custom_name: Nom personnalisé pour la stratégie (optionnel)
            status: Statut initial de la stratégie
            
        Returns:
            str: ID de la stratégie créée ou None en cas d'erreur
        """
        try:
            # Vérifier si l'étude existe
            if not self.study_manager.study_exists(study_name):
                logger.error(f"L'étude '{study_name}' n'existe pas")
                return None
            
            # Chargement de la stratégie depuis l'étude
            strategy_data = self.study_manager.load_strategy(study_name, strategy_rank)
            if not strategy_data:
                logger.error(f"Impossible de charger la stratégie {strategy_rank} de l'étude '{study_name}'")
                return None
            
            signal_generator, position_calculator, performance = strategy_data
            
            # Récupérer les métadonnées de l'étude
            study_metadata = self.study_manager.get_study_metadata(study_name)
            
            # Récupérer la configuration de trading de l'étude
            trading_config = self.study_manager.get_trading_config(study_name)
            
            # Générer un ID unique pour la stratégie
            strategy_id = str(uuid.uuid4())[:8]
            
            # Déterminer le nom de la stratégie
            if not custom_name:
                if 'name' in performance:
                    strategy_name = performance['name']
                else:
                    strategy_name = f"{study_name}_strategy_{strategy_rank}"
            else:
                strategy_name = custom_name
            
            # Créer le dossier pour la stratégie
            strategy_dir = os.path.join(self.base_dir, "data", strategy_id)
            os.makedirs(strategy_dir)
            
            # Préparer les métadonnées de la stratégie
            metadata = {
                "id": strategy_id,
                "name": strategy_name,
                "source_study": study_name,
                "source_rank": strategy_rank,
                "created_at": datetime.datetime.now().isoformat(),
                "last_modified": datetime.datetime.now().isoformat(),
                "status": status.value,
                "description": f"Stratégie issue de l'étude '{study_name}'",
                "tags": ["auto-generated"],
                "performance": performance if performance else {},
                "market_data": {
                    "asset": study_metadata.get("asset", "unknown"),
                    "timeframe": study_metadata.get("timeframe", "unknown"),
                    "exchange": study_metadata.get("exchange", "unknown"),
                },
                "backtest_results": {},
                "live_results": {},
                "risk_params": {
                    "mode": position_calculator.mode.value,
                    "position_size": position_calculator.base_position,
                    "stop_loss": position_calculator.base_sl,
                    "take_profit_multiplier": position_calculator.tp_multiplier
                }
            }
            
            # Ajouter des paramètres spécifiques selon le mode de risque
            risk_params = metadata["risk_params"]
            if position_calculator.mode.value == "atr_based":
                risk_params.update({
                    "atr_period": position_calculator.atr_period,
                    "atr_multiplier": position_calculator.atr_multiplier
                })
            elif position_calculator.mode.value == "volatility_based":
                risk_params.update({
                    "vol_period": position_calculator.vol_period,
                    "vol_multiplier": position_calculator.vol_multiplier
                })
            
            # Sauvegarder les métadonnées
            with open(os.path.join(strategy_dir, "metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            
            # Convertir les blocs de trading en configuration pour le générateur de signaux
            strategy_config = self._convert_to_signal_generator_config(
                signal_generator, position_calculator
            )
            
            # Sauvegarder la configuration
            with open(os.path.join(strategy_dir, "config.json"), 'w', encoding='utf-8') as f:
                json.dump(strategy_config, f, indent=4, ensure_ascii=False)
            
            # Ajouter la stratégie à la collection correspondante
            self.add_to_collection(strategy_id, status)
            
            logger.info(f"Stratégie '{strategy_name}' (ID: {strategy_id}) créée avec succès")
            return strategy_id
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de la stratégie: {str(e)}")
            traceback.print_exc()
            return None
    
    def _convert_to_signal_generator_config(self, 
                                          signal_generator, 
                                          position_calculator) -> Dict:
        """
        Convertit un SignalGenerator et un PositionCalculator en configuration pour le générateur de signaux.
        
        Args:
            signal_generator: Générateur de signaux
            position_calculator: Calculateur de position
            
        Returns:
            Dict: Configuration pour le générateur de signaux
        """
        # Configuration de base
        config = {
            "risk_mode": position_calculator.mode.value,
            "base_position": position_calculator.base_position,
            "base_sl": position_calculator.base_sl,
            "tp_mult": position_calculator.tp_multiplier,
            "leverage": 1.0
        }
        
        # Paramètres spécifiques au mode de risque
        if position_calculator.mode.value == "atr_based":
            config.update({
                "atr_period": position_calculator.atr_period,
                "atr_multiplier": position_calculator.atr_multiplier
            })
        elif position_calculator.mode.value == "volatility_based":
            config.update({
                "vol_period": position_calculator.vol_period,
                "vol_multiplier": position_calculator.vol_multiplier
            })
        
        # Configuration des blocs d'achat
        config["n_buy_blocks"] = len(signal_generator.buy_blocks)
        for i, block in enumerate(signal_generator.buy_blocks):
            config[f"buy_block_{i}_conditions"] = len(block.conditions)
            
            for j, condition in enumerate(block.conditions):
                prefix = f"buy_b{i}_c{j}"
                
                # Extraction du type d'indicateur et de la période
                ind1_parts = condition.indicator1.split('_')
                ind1_type = ind1_parts[0]
                ind1_period = int(ind1_parts[1]) if len(ind1_parts) > 1 else 14
                
                # Configuration de la condition
                config[f"{prefix}_ind1_type"] = ind1_type
                config[f"{prefix}_ind1_period"] = ind1_period
                config[f"{prefix}_operator"] = condition.operator.value
                
                # Si comparaison avec un autre indicateur
                if condition.indicator2:
                    ind2_parts = condition.indicator2.split('_')
                    ind2_type = ind2_parts[0]
                    ind2_period = int(ind2_parts[1]) if len(ind2_parts) > 1 else 14
                    
                    config[f"{prefix}_ind2_type"] = ind2_type
                    config[f"{prefix}_ind2_period"] = ind2_period
                
                # Si comparaison avec une valeur
                elif condition.value is not None:
                    config[f"{prefix}_value"] = condition.value
                
                # Opérateur logique si nécessaire
                if j < len(block.conditions) - 1:
                    config[f"{prefix}_logic"] = block.logic_operators[j].value
        
        # Configuration des blocs de vente
        config["n_sell_blocks"] = len(signal_generator.sell_blocks)
        for i, block in enumerate(signal_generator.sell_blocks):
            config[f"sell_block_{i}_conditions"] = len(block.conditions)
            
            for j, condition in enumerate(block.conditions):
                prefix = f"sell_b{i}_c{j}"
                
                # Extraction du type d'indicateur et de la période
                ind1_parts = condition.indicator1.split('_')
                ind1_type = ind1_parts[0]
                ind1_period = int(ind1_parts[1]) if len(ind1_parts) > 1 else 14
                
                # Configuration de la condition
                config[f"{prefix}_ind1_type"] = ind1_type
                config[f"{prefix}_ind1_period"] = ind1_period
                config[f"{prefix}_operator"] = condition.operator.value
                
                # Si comparaison avec un autre indicateur
                if condition.indicator2:
                    ind2_parts = condition.indicator2.split('_')
                    ind2_type = ind2_parts[0]
                    ind2_period = int(ind2_parts[1]) if len(ind2_parts) > 1 else 14
                    
                    config[f"{prefix}_ind2_type"] = ind2_type
                    config[f"{prefix}_ind2_period"] = ind2_period
                
                # Si comparaison avec une valeur
                elif condition.value is not None:
                    config[f"{prefix}_value"] = condition.value
                
                # Opérateur logique si nécessaire
                if j < len(block.conditions) - 1:
                    config[f"{prefix}_logic"] = block.logic_operators[j].value
        
        return config
    
    def create_strategy_from_config(self, 
                                  config: Dict, 
                                  name: str,
                                  asset: str,
                                  timeframe: str,
                                  exchange: str,
                                  description: str = "",
                                  tags: List[str] = None,
                                  status: StrategyStatus = StrategyStatus.NEW) -> str:
        """
        Crée une stratégie à partir d'une configuration personnalisée.
        
        Args:
            config: Configuration du générateur de signaux
            name: Nom de la stratégie
            asset: Actif (ex: "BTC/USDT")
            timeframe: Timeframe (ex: "1h")
            exchange: Nom de l'exchange
            description: Description de la stratégie
            tags: Liste de tags pour la stratégie
            status: Statut initial de la stratégie
            
        Returns:
            str: ID de la stratégie créée ou None en cas d'erreur
        """
        try:
            # Générer un ID unique pour la stratégie
            strategy_id = str(uuid.uuid4())[:8]
            
            # Créer le dossier pour la stratégie
            strategy_dir = os.path.join(self.base_dir, "data", strategy_id)
            os.makedirs(strategy_dir)
            
            # Préparer les métadonnées de la stratégie
            metadata = {
                "id": strategy_id,
                "name": name,
                "created_at": datetime.datetime.now().isoformat(),
                "last_modified": datetime.datetime.now().isoformat(),
                "status": status.value,
                "description": description,
                "tags": tags or ["manual-config"],
                "performance": {},
                "market_data": {
                    "asset": asset,
                    "timeframe": timeframe,
                    "exchange": exchange,
                },
                "backtest_results": {},
                "live_results": {},
                "risk_params": {
                    "mode": config.get("risk_mode", "fixed"),
                    "position_size": config.get("base_position", 0.1),
                    "stop_loss": config.get("base_sl", 0.02),
                    "take_profit_multiplier": config.get("tp_mult", 2.0)
                }
            }
            
            # Sauvegarder les métadonnées
            with open(os.path.join(strategy_dir, "metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            
            # Sauvegarder la configuration
            with open(os.path.join(strategy_dir, "config.json"), 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            
            # Ajouter la stratégie à la collection correspondante
            self.add_to_collection(strategy_id, status)
            
            logger.info(f"Stratégie '{name}' (ID: {strategy_id}) créée avec succès")
            return strategy_id
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de la stratégie: {str(e)}")
            traceback.print_exc()
            return None
    
    def list_strategies(self) -> List[Dict]:
        """
        Liste toutes les stratégies avec leurs informations de base.
        
        Returns:
            List[Dict]: Liste des stratégies disponibles
        """
        strategies = []
        
        try:
            data_dir = os.path.join(self.base_dir, "data")
            if not os.path.exists(data_dir):
                return []
            
            for strategy_id in os.listdir(data_dir):
                strategy_dir = os.path.join(data_dir, strategy_id)
                
                if os.path.isdir(strategy_dir):
                    metadata_file = os.path.join(strategy_dir, "metadata.json")
                    
                    if os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            
                            # Récupérer les informations essentielles
                            strategy_info = {
                                "id": metadata.get("id", strategy_id),
                                "name": metadata.get("name", "Unnamed Strategy"),
                                "status": metadata.get("status", StrategyStatus.NEW.value),
                                "created_at": metadata.get("created_at", ""),
                                "last_modified": metadata.get("last_modified", ""),
                                "description": metadata.get("description", ""),
                                "tags": metadata.get("tags", []),
                                "asset": metadata.get("market_data", {}).get("asset", ""),
                                "timeframe": metadata.get("market_data", {}).get("timeframe", ""),
                                "exchange": metadata.get("market_data", {}).get("exchange", "")
                            }
                            
                            # Récupérer les performances si disponibles
                            performance = metadata.get("performance", {})
                            if performance:
                                strategy_info["performance"] = {
                                    "roi": performance.get("roi", 0),
                                    "win_rate": performance.get("win_rate", 0),
                                    "max_drawdown": performance.get("max_drawdown", 0),
                                    "profit_factor": performance.get("profit_factor", 0),
                                    "total_trades": performance.get("total_trades", 0)
                                }
                            
                            strategies.append(strategy_info)
                        except Exception as e:
                            logger.warning(f"Erreur lors de la lecture des métadonnées de '{strategy_id}': {str(e)}")
            
            # Trier par date de dernière modification
            strategies.sort(key=lambda x: x.get("last_modified", ""), reverse=True)
            
            return strategies
            
        except Exception as e:
            logger.error(f"Erreur lors du listage des stratégies: {str(e)}")
            return []
    
    def get_strategy(self, strategy_id: str) -> Optional[Dict]:
        """
        Récupère les détails d'une stratégie spécifique.
        
        Args:
            strategy_id: ID de la stratégie
            
        Returns:
            Optional[Dict]: Détails de la stratégie ou None si non trouvée
        """
        try:
            metadata_file = os.path.join(self.base_dir, "data", strategy_id, "metadata.json")
            config_file = os.path.join(self.base_dir, "data", strategy_id, "config.json")
            
            if not os.path.exists(metadata_file):
                logger.error(f"La stratégie '{strategy_id}' n'existe pas")
                return None
            
            # Charger les métadonnées
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Charger la configuration si disponible
            config = {}
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            
            # Combiner les informations
            strategy_info = {
                "metadata": metadata,
                "config": config
            }
            
            return strategy_info
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la stratégie '{strategy_id}': {str(e)}")
            return None
    
    def update_strategy(self, strategy_id: str, updates: Dict) -> bool:
        """
        Met à jour les métadonnées ou la configuration d'une stratégie.
        
        Args:
            strategy_id: ID de la stratégie
            updates: Dictionnaire des mises à jour
            
        Returns:
            bool: True si la mise à jour a réussi, False sinon
        """
        try:
            strategy_data = self.get_strategy(strategy_id)
            if not strategy_data:
                return False
            
            metadata = strategy_data["metadata"]
            config = strategy_data["config"]
            
            # Mise à jour des métadonnées
            if "metadata" in updates:
                metadata_updates = updates["metadata"]
                
                # Mise à jour des champs simples
                for key, value in metadata_updates.items():
                    if key not in ["id", "created_at"]:  # Champs immuables
                        metadata[key] = value
                
                # Mise à jour de la date de modification
                metadata["last_modified"] = datetime.datetime.now().isoformat()
                
                # Sauvegarde des métadonnées
                metadata_file = os.path.join(self.base_dir, "data", strategy_id, "metadata.json")
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=4, ensure_ascii=False)
            
            # Mise à jour de la configuration
            if "config" in updates:
                config_updates = updates["config"]
                
                # Mise à jour de la configuration
                for key, value in config_updates.items():
                    config[key] = value
                
                # Sauvegarde de la configuration
                config_file = os.path.join(self.base_dir, "data", strategy_id, "config.json")
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
            
            # Mise à jour du statut dans les collections si nécessaire
            if "metadata" in updates and "status" in updates["metadata"]:
                old_status = strategy_data["metadata"].get("status", StrategyStatus.NEW.value)
                new_status = updates["metadata"]["status"]
                
                if old_status != new_status:
                    # Supprimer de l'ancienne collection
                    self.remove_from_collection(strategy_id, StrategyStatus(old_status))
                    
                    # Ajouter à la nouvelle collection
                    self.add_to_collection(strategy_id, StrategyStatus(new_status))
            
            logger.info(f"Stratégie '{strategy_id}' mise à jour avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de la stratégie '{strategy_id}': {str(e)}")
            traceback.print_exc()
            return False
    
    def delete_strategy(self, strategy_id: str) -> bool:
        """
        Supprime une stratégie et ses données associées.
        
        Args:
            strategy_id: ID de la stratégie
            
        Returns:
            bool: True si la suppression a réussi, False sinon
        """
        try:
            strategy_data = self.get_strategy(strategy_id)
            if not strategy_data:
                return False
            
            # Récupérer le statut pour supprimer de la collection
            status = strategy_data["metadata"].get("status", StrategyStatus.NEW.value)
            
            # Supprimer de la collection
            self.remove_from_collection(strategy_id, StrategyStatus(status))
            
            # Supprimer le dossier de la stratégie
            strategy_dir = os.path.join(self.base_dir, "data", strategy_id)
            if os.path.exists(strategy_dir):
                shutil.rmtree(strategy_dir)
            
            logger.info(f"Stratégie '{strategy_id}' supprimée avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de la stratégie '{strategy_id}': {str(e)}")
            return False
    
    def run_backtest(self, 
                   strategy_id: str, 
                   price_data: pd.DataFrame, 
                   initial_balance: float = 10000.0,
                   fee: float = 0.001,
                   slippage: float = 0.001,
                   leverage: float = 1.0) -> Optional[Dict]:
        """
        Exécute un backtest sur une stratégie avec les données fournies.
        
        Args:
            strategy_id: ID de la stratégie
            price_data: DataFrame avec les données OHLCV
            initial_balance: Balance initiale
            fee: Frais de trading
            slippage: Slippage
            leverage: Effet de levier
            
        Returns:
            Optional[Dict]: Résultats du backtest ou None en cas d'erreur
        """
        try:
            strategy_data = self.get_strategy(strategy_id)
            if not strategy_data:
                return None
            
            # Récupérer la configuration
            config = strategy_data["config"]
            
            # Vérifier les données requises
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in price_data.columns for col in required_columns):
                logger.error("Données de prix incomplètes. Colonnes requises: open, high, low, close")
                return None
            
            # Récupérer les tableaux numpy
            prices = price_data['close'].values
            high = price_data['high'].values
            low = price_data['low'].values
            volumes = price_data['volume'].values if 'volume' in price_data.columns else None
            
            # Créer le générateur de signaux
            signal_generator = SignalGenerator(config, verbose=False)
            
            # Générer les signaux et paramètres
            signals, position_sizes, sl_levels, tp_levels = signal_generator.generate_signals_and_parameters(
                prices, high, low, volumes
            )
            
            # Configuration du simulateur
            sim_config = SimulationConfig(
                initial_balance=initial_balance,
                fee_open=fee,
                fee_close=fee,
                slippage=slippage,
                tick_size=0.01,
                leverage=leverage
            )
            
            # Création du simulateur
            simulator = Simulator(config=sim_config)
            
            # Multiplication des paramètres de risque par le levier si nécessaire
            if leverage > 1.0:
                position_sizes = position_sizes * leverage
            
            # Exécution de la simulation
            results = simulator.run(
                prices=prices,
                signals=signals,
                position_sizes=position_sizes,
                sl_levels=sl_levels,
                tp_levels=tp_levels
            )
            
            # Sauvegarder les résultats dans les métadonnées
            backtest_id = f"bt_{int(time.time())}"
            
            # Extraire les métriques principales
            performance = results["performance"]
            key_metrics = {
                "roi": performance["roi"],
                "win_rate": performance["win_rate"],
                "max_drawdown": performance["max_drawdown"],
                "profit_factor": performance.get("profit_factor", 0),
                "total_trades": performance["total_trades"],
                "backtest_id": backtest_id,
                "date": datetime.datetime.now().isoformat(),
                "initial_balance": initial_balance,
                "final_balance": performance["final_balance"],
                "data_range": f"{price_data.index[0]} to {price_data.index[-1]}" if hasattr(price_data.index, '__len__') else ""
            }
            
            # Mettre à jour les métadonnées avec les résultats du backtest
            metadata = strategy_data["metadata"]
            
            if "backtest_results" not in metadata:
                metadata["backtest_results"] = {}
            
            metadata["backtest_results"][backtest_id] = key_metrics
            
            # Mettre à jour la performance générale si c'est le premier backtest ou meilleur ROI
            if not metadata.get("performance") or key_metrics["roi"] > metadata["performance"].get("roi", 0):
                metadata["performance"] = key_metrics
            
            # Sauvegarder les métadonnées mises à jour
            metadata["last_modified"] = datetime.datetime.now().isoformat()
            metadata_file = os.path.join(self.base_dir, "data", strategy_id, "metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            
            # Sauvegarder les résultats détaillés
            backtest_dir = os.path.join(self.base_dir, "data", strategy_id, "backtests")
            os.makedirs(backtest_dir, exist_ok=True)
            
            # Sauvegarder les résultats complets
            results_file = os.path.join(backtest_dir, f"{backtest_id}.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                # Convertir les arrays numpy en listes
                serializable_results = {
                    "performance": results["performance"],
                    "config": results["config"],
                    "settings": {
                        "initial_balance": initial_balance,
                        "fee": fee,
                        "slippage": slippage,
                        "leverage": leverage
                    }
                }
                json.dump(serializable_results, f, indent=4, ensure_ascii=False)
            
            # Sauvegarder l'historique d'équité
            if "equity_curve" in results:
                equity_df = pd.DataFrame(results["equity_curve"], columns=["equity"])
                equity_df.to_csv(os.path.join(backtest_dir, f"{backtest_id}_equity.csv"))
            
            # Sauvegarder l'historique des trades
            if "trade_history" in results:
                trades_df = pd.DataFrame(results["trade_history"])
                trades_df.to_csv(os.path.join(backtest_dir, f"{backtest_id}_trades.csv"), index=False)
            
            # Générer des graphiques
            self._create_backtest_charts(strategy_id, backtest_id, results)
            
            logger.info(f"Backtest '{backtest_id}' exécuté avec succès pour la stratégie '{strategy_id}'")
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du backtest pour la stratégie '{strategy_id}': {str(e)}")
            traceback.print_exc()
            return None
    
    def _create_backtest_charts(self, strategy_id: str, backtest_id: str, results: Dict):
        """
        Crée des graphiques pour visualiser les résultats du backtest.
        
        Args:
            strategy_id: ID de la stratégie
            backtest_id: ID du backtest
            results: Résultats du backtest
        """
        try:
            charts_dir = os.path.join(self.base_dir, "data", strategy_id, "backtests", f"{backtest_id}_charts")
            os.makedirs(charts_dir, exist_ok=True)
            
            # Graphique de la courbe d'équité
            if "account_history" in results and "equity" in results["account_history"]:
                equity = results["account_history"]["equity"]
                
                plt.figure(figsize=(12, 6))
                plt.plot(equity, color='blue')
                plt.title("Évolution du capital")
                plt.xlabel("Barre")
                plt.ylabel("Capital")
                plt.grid(True)
                plt.savefig(os.path.join(charts_dir, "equity_curve.png"))
                plt.close()
            
            # Graphique des trades
            if "trade_history" in results:
                trades = results["trade_history"]
                if trades:
                    trades_df = pd.DataFrame(trades)
                    
                    # Histogramme des profits
                    if 'pnl_pct' in trades_df.columns:
                        plt.figure(figsize=(12, 6))
                        plt.hist(trades_df['pnl_pct'], bins=20, alpha=0.7, color='green')
                        plt.title("Distribution des profits/pertes")
                        plt.xlabel("Profit/Perte (%)")
                        plt.ylabel("Nombre de trades")
                        plt.grid(True)
                        plt.savefig(os.path.join(charts_dir, "pnl_distribution.png"))
                        plt.close()
                    
                    # Graphique des trades cumulés
                    if 'pnl_abs' in trades_df.columns:
                        plt.figure(figsize=(12, 6))
                        trades_df['cumulative_pnl'] = trades_df['pnl_abs'].cumsum()
                        plt.plot(trades_df['cumulative_pnl'])
                        plt.title("PnL cumulé")
                        plt.xlabel("Nombre de trades")
                        plt.ylabel("Profit/Perte")
                        plt.grid(True)
                        plt.savefig(os.path.join(charts_dir, "cumulative_pnl.png"))
                        plt.close()
            
            # Graphique du drawdown
            if "account_history" in results and "drawdown" in results["account_history"]:
                drawdown = results["account_history"]["drawdown"]
                
                plt.figure(figsize=(12, 6))
                plt.plot(drawdown * 100, color='red')
                plt.title("Drawdown")
                plt.xlabel("Barre")
                plt.ylabel("Drawdown (%)")
                plt.grid(True)
                plt.savefig(os.path.join(charts_dir, "drawdown.png"))
                plt.close()
            
        except Exception as e:
            logger.error(f"Erreur lors de la création des graphiques: {str(e)}")
    
    def get_backtest_results(self, strategy_id: str, backtest_id: Optional[str] = None) -> Optional[Dict]:
        """
        Récupère les résultats d'un backtest spécifique ou de tous les backtests d'une stratégie.
        
        Args:
            strategy_id: ID de la stratégie
            backtest_id: ID du backtest (optionnel, si None retourne tous les backtests)
            
        Returns:
            Optional[Dict]: Résultats du backtest ou None en cas d'erreur
        """
        try:
            strategy_data = self.get_strategy(strategy_id)
            if not strategy_data:
                return None
            
            # Récupérer les métadonnées
            metadata = strategy_data["metadata"]
            backtest_results = metadata.get("backtest_results", {})
            
            if backtest_id:
                # Retourner un backtest spécifique
                if backtest_id not in backtest_results:
                    logger.error(f"Backtest '{backtest_id}' introuvable pour la stratégie '{strategy_id}'")
                    return None
                
                # Charger les données détaillées du backtest
                results_file = os.path.join(self.base_dir, "data", strategy_id, "backtests", f"{backtest_id}.json")
                if os.path.exists(results_file):
                    with open(results_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                else:
                    return backtest_results[backtest_id]
            else:
                # Retourner tous les backtests sous forme de tableau
                return list(backtest_results.values())
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des résultats de backtest: {str(e)}")
            return None
    
    def add_to_collection(self, strategy_id: str, collection: Union[str, StrategyStatus]) -> bool:
        """
        Ajoute une stratégie à une collection.
        
        Args:
            strategy_id: ID de la stratégie
            collection: Nom de la collection ou statut StrategyStatus
            
        Returns:
            bool: True si l'ajout a réussi, False sinon
        """
        try:
            # Déterminer le nom de la collection
            collection_name = collection.value if isinstance(collection, StrategyStatus) else collection
            
            # Charger la collection
            collection_file = os.path.join(self.collections_dir, f"{collection_name}.json")
            
            if not os.path.exists(collection_file):
                # Créer la collection si elle n'existe pas
                with open(collection_file, 'w', encoding='utf-8') as f:
                    json.dump({"strategies": []}, f, indent=4)
            
            # Charger la collection existante
            with open(collection_file, 'r', encoding='utf-8') as f:
                collection_data = json.load(f)
            
            # Éviter les doublons
            if strategy_id not in collection_data["strategies"]:
                collection_data["strategies"].append(strategy_id)
                
                # Sauvegarder la collection mise à jour
                with open(collection_file, 'w', encoding='utf-8') as f:
                    json.dump(collection_data, f, indent=4, ensure_ascii=False)
                
                logger.info(f"Stratégie '{strategy_id}' ajoutée à la collection '{collection_name}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout à la collection: {str(e)}")
            return False
    
    def remove_from_collection(self, strategy_id: str, collection: Union[str, StrategyStatus]) -> bool:
        """
        Supprime une stratégie d'une collection.
        
        Args:
            strategy_id: ID de la stratégie
            collection: Nom de la collection ou statut StrategyStatus
            
        Returns:
            bool: True si la suppression a réussi, False sinon
        """
        try:
            # Déterminer le nom de la collection
            collection_name = collection.value if isinstance(collection, StrategyStatus) else collection
            
            # Charger la collection
            collection_file = os.path.join(self.collections_dir, f"{collection_name}.json")
            
            if not os.path.exists(collection_file):
                return False
            
            # Charger la collection existante
            with open(collection_file, 'r', encoding='utf-8') as f:
                collection_data = json.load(f)
            
            # Supprimer la stratégie de la collection
            if strategy_id in collection_data["strategies"]:
                collection_data["strategies"].remove(strategy_id)
                
                # Sauvegarder la collection mise à jour
                with open(collection_file, 'w', encoding='utf-8') as f:
                    json.dump(collection_data, f, indent=4, ensure_ascii=False)
                
                logger.info(f"Stratégie '{strategy_id}' supprimée de la collection '{collection_name}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de la collection: {str(e)}")
            return False
    
    def get_collection(self, collection: Union[str, StrategyStatus]) -> List[Dict]:
        """
        Récupère les stratégies d'une collection.
        
        Args:
            collection: Nom de la collection ou statut StrategyStatus
            
        Returns:
            List[Dict]: Liste des stratégies dans la collection
        """
        try:
            # Déterminer le nom de la collection
            collection_name = collection.value if isinstance(collection, StrategyStatus) else collection
            
            # Charger la collection
            collection_file = os.path.join(self.collections_dir, f"{collection_name}.json")
            
            if not os.path.exists(collection_file):
                return []
            
            # Charger la collection existante
            with open(collection_file, 'r', encoding='utf-8') as f:
                collection_data = json.load(f)
            
            # Récupérer les IDs des stratégies
            strategy_ids = collection_data.get("strategies", [])
            
            # Récupérer les détails des stratégies
            strategies = []
            for strategy_id in strategy_ids:
                strategy_data = self.get_strategy(strategy_id)
                if strategy_data:
                    strategy_info = {
                        "id": strategy_id,
                        "name": strategy_data["metadata"].get("name", "Unnamed Strategy"),
                        "description": strategy_data["metadata"].get("description", ""),
                        "status": strategy_data["metadata"].get("status", StrategyStatus.NEW.value),
                        "performance": strategy_data["metadata"].get("performance", {})
                    }
                    strategies.append(strategy_info)
            
            return strategies
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la collection '{collection_name}': {str(e)}")
            return []
    
    def compare_strategies(self, strategy_ids: List[str], title: str = "Comparaison de stratégies") -> Optional[str]:
        """
        Compare plusieurs stratégies et génère un graphique.
        
        Args:
            strategy_ids: Liste des IDs de stratégies à comparer
            title: Titre du graphique
            
        Returns:
            Optional[str]: Chemin du graphique généré ou None en cas d'erreur
        """
        try:
            if not strategy_ids:
                logger.error("Aucune stratégie spécifiée pour la comparaison")
                return None
            
            # Récupérer les données des stratégies
            strategies_data = []
            for strategy_id in strategy_ids:
                strategy = self.get_strategy(strategy_id)
                if strategy:
                    strategies_data.append({
                        "id": strategy_id,
                        "name": strategy["metadata"].get("name", f"Strategy {strategy_id}"),
                        "performance": strategy["metadata"].get("performance", {}),
                        "backtest_results": strategy["metadata"].get("backtest_results", {})
                    })
            
            if not strategies_data:
                logger.error("Aucune donnée de stratégie valide pour la comparaison")
                return None
            
            # Préparation des données pour la comparaison
            comparison_data = []
            metric_keys = ["roi", "win_rate", "max_drawdown", "profit_factor", "total_trades"]
            
            for strategy in strategies_data:
                data_row = {"name": strategy["name"], "id": strategy["id"]}
                
                # Récupérer les métriques de performance
                performance = strategy["performance"]
                for key in metric_keys:
                    if key in performance:
                        # Conversion en pourcentages pour certaines métriques
                        if key in ["roi", "win_rate", "max_drawdown"]:
                            data_row[key] = performance[key] * 100
                        else:
                            data_row[key] = performance[key]
                
                comparison_data.append(data_row)
            
            # Création d'un DataFrame
            df = pd.DataFrame(comparison_data)
            
            # Génération de la comparaison
            timestamp = int(time.time())
            output_dir = os.path.join(self.base_dir, "visuals")
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, f"comparison_{timestamp}.png")
            
            # Création d'un graphique de comparaison
            self._generate_comparison_chart(df, title, output_path)
            
            logger.info(f"Graphique de comparaison généré: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Erreur lors de la comparaison des stratégies: {str(e)}")
            traceback.print_exc()
            return None
    
    def _generate_comparison_chart(self, df: pd.DataFrame, title: str, output_path: str):
        """
        Génère un graphique de comparaison des performances.
        
        Args:
            df: DataFrame avec les données de comparaison
            title: Titre du graphique
            output_path: Chemin de sortie du graphique
        """
        try:
            # Sélection des métriques principales
            metrics = ["roi", "win_rate", "max_drawdown", "profit_factor"]
            df_metrics = df[["name"] + [m for m in metrics if m in df.columns]]
            
            # Transposition pour faciliter la visualisation
            df_plot = df_metrics.set_index("name").transpose()
            
            # Création du graphique
            plt.figure(figsize=(12, 8))
            ax = df_plot.plot(kind='bar', figsize=(12, 8))
            plt.title(title, fontsize=16)
            plt.ylabel("Valeur", fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(title="Stratégies", fontsize=10)
            
            # Rotation des étiquettes de l'axe X
            plt.xticks(rotation=0, fontsize=12)
            
            # Ajout des valeurs sur les barres
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f', fontsize=9)
            
            # Sauvegarde du graphique
            plt.tight_layout()
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique de comparaison: {str(e)}")
            traceback.print_exc()
    
    def export_strategy_to_production(self, 
                                   strategy_id: str, 
                                   export_path: str,
                                   include_backtests: bool = False) -> bool:
        """
        Exporte une stratégie pour la production.
        
        Args:
            strategy_id: ID de la stratégie
            export_path: Chemin d'exportation
            include_backtests: Inclure les résultats de backtest
            
        Returns:
            bool: True si l'exportation a réussi, False sinon
        """
        try:
            strategy_data = self.get_strategy(strategy_id)
            if not strategy_data:
                return False
            
            # Création du dossier d'exportation
            if not os.path.exists(export_path):
                os.makedirs(export_path)
            
            # Exporter les métadonnées et la configuration
            export_info = {
                "name": strategy_data["metadata"].get("name", f"Strategy {strategy_id}"),
                "description": strategy_data["metadata"].get("description", ""),
                "config": strategy_data["config"],
                "risk_params": strategy_data["metadata"].get("risk_params", {}),
                "market_data": strategy_data["metadata"].get("market_data", {}),
                "performance": strategy_data["metadata"].get("performance", {}),
                "export_date": datetime.datetime.now().isoformat()
            }
            
            # Ajouter les résultats de backtest si demandé
            if include_backtests:
                export_info["backtest_results"] = strategy_data["metadata"].get("backtest_results", {})
            
            # Sauvegarder le fichier de stratégie
            with open(os.path.join(export_path, f"{strategy_id}_strategy.json"), 'w', encoding='utf-8') as f:
                json.dump(export_info, f, indent=4, ensure_ascii=False)
            
            # Copier les graphiques de backtest si disponibles et demandés
            if include_backtests:
                backtest_dir = os.path.join(self.base_dir, "data", strategy_id, "backtests")
                if os.path.exists(backtest_dir):
                    export_backtest_dir = os.path.join(export_path, "backtests")
                    os.makedirs(export_backtest_dir, exist_ok=True)
                    
                    # Copier seulement les graphiques et les fichiers JSON des backtests
                    for item in os.listdir(backtest_dir):
                        if item.endswith(".json") or "_charts" in item:
                            src_path = os.path.join(backtest_dir, item)
                            dst_path = os.path.join(export_backtest_dir, item)
                            
                            if os.path.isdir(src_path):
                                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                            else:
                                shutil.copy2(src_path, dst_path)
            
            # Mettre à jour le statut de la stratégie si elle n'est pas déjà en production
            if strategy_data["metadata"].get("status") != StrategyStatus.PRODUCTION.value:
                self.update_strategy(strategy_id, {
                    "metadata": {"status": StrategyStatus.PRODUCTION.value}
                })
            
            logger.info(f"Stratégie '{strategy_id}' exportée avec succès vers '{export_path}'")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exportation de la stratégie '{strategy_id}': {str(e)}")
            traceback.print_exc()
            return False
    
    def clone_strategy(self, strategy_id: str, new_name: Optional[str] = None) -> Optional[str]:
        """
        Clone une stratégie existante.
        
        Args:
            strategy_id: ID de la stratégie à cloner
            new_name: Nouveau nom pour la stratégie clonée (optionnel)
            
        Returns:
            Optional[str]: ID de la nouvelle stratégie ou None en cas d'erreur
        """
        try:
            strategy_data = self.get_strategy(strategy_id)
            if not strategy_data:
                return None
            
            # Générer un ID unique pour la nouvelle stratégie
            new_strategy_id = str(uuid.uuid4())[:8]
            
            # Déterminer le nom de la nouvelle stratégie
            if not new_name:
                original_name = strategy_data["metadata"].get("name", f"Strategy {strategy_id}")
                new_name = f"{original_name} (Clone)"
            
            # Créer un dossier pour la nouvelle stratégie
            new_strategy_dir = os.path.join(self.base_dir, "data", new_strategy_id)
            os.makedirs(new_strategy_dir)
            
            # Copier et modifier les métadonnées
            metadata = deepcopy(strategy_data["metadata"])
            metadata.update({
                "id": new_strategy_id,
                "name": new_name,
                "created_at": datetime.datetime.now().isoformat(),
                "last_modified": datetime.datetime.now().isoformat(),
                "status": StrategyStatus.NEW.value,
                "description": f"Clone de la stratégie '{metadata.get('name', strategy_id)}'",
                "tags": metadata.get("tags", []) + ["clone"],
                "cloned_from": strategy_id
            })
            
            # Réinitialiser les résultats de backtest
            metadata["backtest_results"] = {}
            
            # Sauvegarder les métadonnées
            with open(os.path.join(new_strategy_dir, "metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            
            # Copier la configuration
            with open(os.path.join(new_strategy_dir, "config.json"), 'w', encoding='utf-8') as f:
                json.dump(strategy_data["config"], f, indent=4, ensure_ascii=False)
            
            # Ajouter la stratégie à la collection des nouvelles stratégies
            self.add_to_collection(new_strategy_id, StrategyStatus.NEW)
            
            logger.info(f"Stratégie '{strategy_id}' clonée avec succès vers '{new_strategy_id}'")
            return new_strategy_id
            
        except Exception as e:
            logger.error(f"Erreur lors du clonage de la stratégie '{strategy_id}': {str(e)}")
            traceback.print_exc()
            return None
    
    def find_strategies_by_tags(self, tags: List[str], match_all: bool = False) -> List[Dict]:
        """
        Recherche des stratégies par tags.
        
        Args:
            tags: Liste des tags à rechercher
            match_all: True pour exiger tous les tags, False pour au moins un
            
        Returns:
            List[Dict]: Liste des stratégies correspondantes
        """
        try:
            # Récupérer toutes les stratégies
            all_strategies = self.list_strategies()
            
            # Filtrer par tags
            matching_strategies = []
            
            for strategy in all_strategies:
                strategy_tags = strategy.get("tags", [])
                
                if match_all:
                    # Tous les tags doivent correspondre
                    if all(tag in strategy_tags for tag in tags):
                        matching_strategies.append(strategy)
                else:
                    # Au moins un tag doit correspondre
                    if any(tag in strategy_tags for tag in tags):
                        matching_strategies.append(strategy)
            
            return matching_strategies
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche de stratégies par tags: {str(e)}")
            return []
    
    def find_strategies_by_performance(self, 
                                     min_roi: float = 0.0, 
                                     min_win_rate: float = 0.0,
                                     max_drawdown: float = 1.0) -> List[Dict]:
        """
        Recherche des stratégies selon des critères de performance.
        
        Args:
            min_roi: ROI minimum (0.0 = 0%)
            min_win_rate: Taux de réussite minimum (0.0 = 0%)
            max_drawdown: Drawdown maximum (1.0 = 100%)
            
        Returns:
            List[Dict]: Liste des stratégies correspondantes
        """
        try:
            # Récupérer toutes les stratégies
            all_strategies = self.list_strategies()
            
            # Filtrer par critères de performance
            matching_strategies = []
            
            for strategy in all_strategies:
                performance = strategy.get("performance", {})
                
                # Vérifier les critères
                if (
                    performance.get("roi", 0) >= min_roi and
                    performance.get("win_rate", 0) >= min_win_rate and
                    performance.get("max_drawdown", 1.0) <= max_drawdown
                ):
                    matching_strategies.append(strategy)
            
            return matching_strategies
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche de stratégies par performance: {str(e)}")
            return []
    
    def get_best_strategies(self, limit: int = 10, metric: str = "roi") -> List[Dict]:
        """
        Récupère les meilleures stratégies selon une métrique.
        
        Args:
            limit: Nombre maximum de stratégies à retourner
            metric: Métrique de tri ('roi', 'win_rate', 'profit_factor', etc.)
            
        Returns:
            List[Dict]: Liste des meilleures stratégies
        """
        try:
            # Récupérer toutes les stratégies
            all_strategies = self.list_strategies()
            
            # Filtrer les stratégies sans performance
            strategies_with_performance = [
                s for s in all_strategies 
                if "performance" in s and metric in s["performance"]
            ]
            
            # Trier par la métrique spécifiée
            reverse = True
            if metric == "max_drawdown":
                reverse = False  # Pour le drawdown, plus petit = meilleur
                
            sorted_strategies = sorted(
                strategies_with_performance,
                key=lambda x: x["performance"][metric],
                reverse=reverse
            )
            
            # Limiter le nombre de résultats
            return sorted_strategies[:limit]
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des meilleures stratégies: {str(e)}")
            return []

# Exemple d'utilisation
if __name__ == "__main__":
    manager = StrategyManager()
    print("Gestionnaire de stratégies initialisé")
    
    # Liste toutes les stratégies
    strategies = manager.list_strategies()
    print(f"Nombre de stratégies: {len(strategies)}")
    
    # Afficher les stratégies favorites
    favorites = manager.get_collection(StrategyStatus.FAVORITE)
    print(f"Stratégies favorites: {len(favorites)}")
    for fav in favorites:
        print(f" - {fav['name']} (ID: {fav['id']})")