"""
Module de gestion des stratégies qui sert d'interface entre le constructeur de stratégie
et d'autres composants comme la simulation et le trading en direct.
"""

import os
import json
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import uuid
import shutil

# Import des modules existants
from core.strategy.constructor.constructor import StrategyConstructor
from core.simulation.simulator import Simulator
from core.simulation.simulation_config import SimulationConfig

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("strategy_manager")


class StrategyManager:
    """
    Gestionnaire de stratégies qui facilite la création, le chargement,
    l'optimisation et la comparaison de stratégies.
    """
    
    def __init__(self, strategies_dir: str = "strategies"):
        """
        Initialise le gestionnaire de stratégies.
        
        Args:
            strategies_dir: Répertoire des stratégies
        """
        self.strategies_dir = strategies_dir
        self.current_strategy = None
        self.simulation_results = {}
        
        # Créer le répertoire des stratégies s'il n'existe pas
        os.makedirs(strategies_dir, exist_ok=True)
    
    def create_strategy(self, name: str, description: str = "") -> StrategyConstructor:
        """
        Crée une nouvelle stratégie.
        
        Args:
            name: Nom de la stratégie
            description: Description de la stratégie
            
        Returns:
            StrategyConstructor: Instance du constructeur de stratégie
        """
        # Créer une nouvelle stratégie
        constructor = StrategyConstructor()
        constructor.set_name(name)
        constructor.set_description(description)
        
        # Mettre à jour la stratégie courante
        self.current_strategy = constructor
        
        logger.info(f"Stratégie '{name}' créée")
        return constructor
    
    def save_strategy(self, strategy: StrategyConstructor = None, directory: str = None) -> str:
        """
        Sauvegarde une stratégie.
        
        Args:
            strategy: Constructeur de stratégie (utilise la stratégie courante si None)
            directory: Répertoire de destination (utilise le répertoire par défaut si None)
            
        Returns:
            str: Chemin du fichier de stratégie sauvegardé
        """
        strategy = strategy or self.current_strategy
        directory = directory or self.strategies_dir
        
        if strategy is None:
            logger.error("Aucune stratégie à sauvegarder")
            return None
        
        # Créer le répertoire si nécessaire
        os.makedirs(directory, exist_ok=True)
        
        # Créer le chemin du fichier
        strategy_id = strategy.config.id
        filename = f"{strategy_id}_{strategy.config.name.replace(' ', '_')}.json"
        filepath = os.path.join(directory, filename)
        
        # Sauvegarder la stratégie
        success = strategy.save(filepath)
        
        if success:
            logger.info(f"Stratégie '{strategy.config.name}' sauvegardée dans {filepath}")
            return filepath
        else:
            logger.error(f"Échec de la sauvegarde de la stratégie '{strategy.config.name}'")
            return None
    
    def load_strategy(self, strategy_id: str) -> Optional[StrategyConstructor]:
        """
        Charge une stratégie depuis son ID.
        
        Args:
            strategy_id: ID de la stratégie
            
        Returns:
            Optional[StrategyConstructor]: Instance du constructeur de stratégie ou None
        """
        # Chercher le fichier de stratégie
        strategy_files = [f for f in os.listdir(self.strategies_dir) if f.startswith(f"{strategy_id}_") and f.endswith(".json")]
        
        if not strategy_files:
            logger.error(f"Aucune stratégie trouvée avec l'ID '{strategy_id}'")
            return None
        
        # Charger la première stratégie trouvée
        filepath = os.path.join(self.strategies_dir, strategy_files[0])
        
        try:
            constructor = StrategyConstructor.load(filepath)
            self.current_strategy = constructor
            logger.info(f"Stratégie '{constructor.config.name}' chargée depuis {filepath}")
            return constructor
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la stratégie depuis {filepath}: {str(e)}")
            return None
    
    def clone_strategy(self, strategy: StrategyConstructor = None, new_name: str = None) -> Optional[StrategyConstructor]:
        """
        Clone une stratégie existante.
        
        Args:
            strategy: Constructeur de stratégie à cloner (utilise la stratégie courante si None)
            new_name: Nouveau nom pour la stratégie clonée
            
        Returns:
            Optional[StrategyConstructor]: Instance du constructeur de stratégie clonée
        """
        strategy = strategy or self.current_strategy
        
        if strategy is None:
            logger.error("Aucune stratégie à cloner")
            return None
        
        try:
            # Cloner la stratégie
            clone = strategy.clone()
            
            # Définir un nouveau nom si spécifié
            if new_name:
                clone.set_name(new_name)
            
            # Mettre à jour la stratégie courante
            self.current_strategy = clone
            
            logger.info(f"Stratégie '{strategy.config.name}' clonée en '{clone.config.name}'")
            return clone
        except Exception as e:
            logger.error(f"Erreur lors du clonage de la stratégie: {str(e)}")
            return None
    
    def list_strategies(self) -> List[Dict[str, Any]]:
        """
        Liste toutes les stratégies disponibles.
        
        Returns:
            List[Dict[str, Any]]: Liste des métadonnées des stratégies
        """
        strategies = []
        
        for filename in os.listdir(self.strategies_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.strategies_dir, filename)
                
                try:
                    # Lire les métadonnées de base
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extraire les informations clés
                    strategy_info = {
                        "id": data.get("id", ""),
                        "name": data.get("name", ""),
                        "description": data.get("description", ""),
                        "version": data.get("version", ""),
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                        "tags": data.get("tags", []),
                        "filepath": filepath
                    }
                    
                    strategies.append(strategy_info)
                except Exception as e:
                    logger.warning(f"Erreur lors de la lecture de {filepath}: {str(e)}")
        
        # Trier par date de mise à jour
        strategies.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        return strategies
    
    def run_simulation(
        self,
        data: pd.DataFrame,
        strategy: StrategyConstructor = None,
        initial_balance: float = 10000.0,
        leverage: int = 1,
        fee_open: float = 0.001,
        fee_close: float = 0.001,
        slippage: float = 0.0005
    ) -> Dict[str, Any]:
        """
        Exécute une simulation avec une stratégie.
        
        Args:
            data: DataFrame avec les données OHLCV
            strategy: Constructeur de stratégie (utilise la stratégie courante si None)
            initial_balance: Balance initiale
            leverage: Effet de levier
            fee_open: Frais d'ouverture de position
            fee_close: Frais de clôture de position
            slippage: Slippage
            
        Returns:
            Dict[str, Any]: Résultats de la simulation
        """
        strategy = strategy or self.current_strategy
        
        if strategy is None:
            logger.error("Aucune stratégie pour la simulation")
            return None
        
        try:
            # Créer une configuration de simulation
            sim_config = SimulationConfig(
                initial_balance=initial_balance,
                leverage=leverage,
                fee_open=fee_open,
                fee_close=fee_close,
                slippage=slippage
            )
            
            # Créer le simulateur
            simulator = Simulator(sim_config)
            
            # Générer les signaux avec la stratégie
            logger.info(f"Génération des signaux pour '{strategy.config.name}'...")
            signals, data_with_signals = strategy.generate_signals(data)
            
            # Vérifier que des signaux ont été générés
            if sum(abs(signals)) == 0:
                logger.warning(f"Aucun signal généré pour '{strategy.config.name}'")
                return {
                    "strategy_id": strategy.config.id,
                    "strategy_name": strategy.config.name,
                    "success": False,
                    "message": "Aucun signal généré",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Extraire les paramètres de risque
            position_sizes = data_with_signals['position_size'].values if 'position_size' in data_with_signals.columns else None
            sl_levels = data_with_signals['sl_level'].values if 'sl_level' in data_with_signals.columns else None
            tp_levels = data_with_signals['tp_level'].values if 'tp_level' in data_with_signals.columns else None
            
            # Exécuter la simulation
            logger.info(f"Exécution de la simulation pour '{strategy.config.name}'...")
            results = simulator.run(
                prices=data_with_signals['close'].values,
                signals=signals,
                position_sizes=position_sizes,
                sl_levels=sl_levels,
                tp_levels=tp_levels
            )
            
            # Ajouter les informations de la stratégie
            results["strategy_id"] = strategy.config.id
            results["strategy_name"] = strategy.config.name
            results["timestamp"] = datetime.now().isoformat()
            
            # Stocker les résultats
            self.simulation_results[strategy.config.id] = results
            
            logger.info(f"Simulation terminée pour '{strategy.config.name}': " +
                       f"ROI={results['performance']['roi_pct']:.2f}%, " +
                       f"Win Rate={results['performance']['win_rate_pct']:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de la simulation: {str(e)}")
            return {
                "strategy_id": strategy.config.id,
                "strategy_name": strategy.config.name,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def save_simulation_results(self, output_dir: str, strategy_id: Optional[str] = None) -> bool:
        """
        Sauvegarde les résultats de simulation.
        
        Args:
            output_dir: Répertoire de sortie
            strategy_id: ID de la stratégie (toutes les stratégies si None)
            
        Returns:
            bool: Succès de la sauvegarde
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Déterminer quels résultats sauvegarder
            if strategy_id:
                if strategy_id not in self.simulation_results:
                    logger.error(f"Aucun résultat de simulation pour la stratégie '{strategy_id}'")
                    return False
                
                results_to_save = {strategy_id: self.simulation_results[strategy_id]}
            else:
                results_to_save = self.simulation_results
            
            # Sauvegarder chaque résultat
            for strategy_id, results in results_to_save.items():
                # Créer un sous-répertoire pour la stratégie
                strategy_dir = os.path.join(output_dir, strategy_id)
                os.makedirs(strategy_dir, exist_ok=True)
                
                # Sauvegarder les résultats en JSON
                results_path = os.path.join(strategy_dir, "simulation_results.json")
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4)
                
                logger.info(f"Résultats de simulation sauvegardés dans {results_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats: {str(e)}")
            return False
    
    def generate_performance_report(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Génère un rapport de performance pour une stratégie.
        
        Args:
            strategy_id: ID de la stratégie (utilise la stratégie courante si None)
            
        Returns:
            Dict[str, Any]: Rapport de performance
        """
        # Déterminer l'ID de la stratégie
        if strategy_id is None and self.current_strategy is not None:
            strategy_id = self.current_strategy.config.id
        
        if not strategy_id or strategy_id not in self.simulation_results:
            logger.error(f"Aucun résultat de simulation pour la stratégie '{strategy_id}'")
            return None
        
        try:
            # Récupérer les résultats de simulation
            results = self.simulation_results[strategy_id]
            
            # Récupérer les informations sur la stratégie
            strategy_info = None
            for strategy_meta in self.list_strategies():
                if strategy_meta["id"] == strategy_id:
                    strategy_info = strategy_meta
                    break
            
            # Extraire les métriques principales
            perf = results["performance"]
            
            # Créer le rapport
            report = {
                "strategy": {
                    "id": strategy_id,
                    "name": results.get("strategy_name", ""),
                    "description": strategy_info.get("description", "") if strategy_info else "",
                    "tags": strategy_info.get("tags", []) if strategy_info else []
                },
                "performance": {
                    "roi": f"{perf['roi_pct']:.2f}%",
                    "total_trades": perf["total_trades"],
                    "win_rate": f"{perf['win_rate_pct']:.2f}%",
                    "max_drawdown": f"{perf['max_drawdown_pct']:.2f}%",
                    "profit_factor": f"{perf['profit_factor']:.2f}",
                    "sharpe_ratio": perf.get("sharpe_ratio", "N/A"),
                    "total_pnl": perf["total_pnl"],
                    "final_balance": perf["final_balance"]
                },
                "risk_metrics": {
                    "max_profit_trade": perf["max_profit_trade"],
                    "max_loss_trade": perf["max_loss_trade"],
                    "avg_profit_per_trade": perf["avg_profit_per_trade"],
                    "avg_profit_per_trade_pct": perf["avg_profit_per_trade_pct"],
                    "liquidation_rate": perf["liquidation_rate"]
                },
                "simulation_params": results["config"],
                "generated_at": datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport: {str(e)}")
            return None
    
    def compare_strategies(self, strategy_ids: List[str], data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare plusieurs stratégies sur les mêmes données.
        
        Args:
            strategy_ids: Liste des IDs de stratégie à comparer
            data: DataFrame avec les données OHLCV
            
        Returns:
            Dict[str, Any]: Résultats de la comparaison
        """
        try:
            comparison = {
                "strategies": [],
                "timestamp": datetime.now().isoformat(),
                "data_info": {
                    "points": len(data),
                    "start_date": data.index[0].strftime('%Y-%m-%d') if isinstance(data.index[0], pd.Timestamp) else str(data.index[0]),
                    "end_date": data.index[-1].strftime('%Y-%m-%d') if isinstance(data.index[-1], pd.Timestamp) else str(data.index[-1])
                }
            }
            
            # Simuler chaque stratégie
            for strategy_id in strategy_ids:
                # Charger la stratégie
                strategy = self.load_strategy(strategy_id)
                
                if strategy is None:
                    logger.warning(f"Stratégie '{strategy_id}' non trouvée, ignorée")
                    continue
                
                # Exécuter la simulation
                results = self.run_simulation(data, strategy)
                
                if results and "performance" in results:
                    # Extraire les métriques principales
                    strategy_results = {
                        "id": strategy_id,
                        "name": strategy.config.name,
                        "roi_pct": results["performance"]["roi_pct"],
                        "win_rate_pct": results["performance"]["win_rate_pct"],
                        "max_drawdown_pct": results["performance"]["max_drawdown_pct"],
                        "total_trades": results["performance"]["total_trades"],
                        "profit_factor": results["performance"]["profit_factor"],
                        "sharpe_ratio": results["performance"].get("sharpe_ratio", None),
                        "final_balance": results["performance"]["final_balance"]
                    }
                    
                    comparison["strategies"].append(strategy_results)
            
            # Trier les stratégies par ROI décroissant
            comparison["strategies"].sort(key=lambda x: x["roi_pct"], reverse=True)
            
            # Ajouter le classement
            for i, strategy in enumerate(comparison["strategies"]):
                strategy["rank"] = i + 1
            
            return comparison
            
        except Exception as e:
            logger.error(f"Erreur lors de la comparaison des stratégies: {str(e)}")
            return None
    
    def optimize_parameters(
        self,
        strategy: StrategyConstructor,
        data: pd.DataFrame,
        param_ranges: Dict[str, List[Any]],
        metric: str = "roi_pct",
        max_combinations: int = 100
    ) -> Dict[str, Any]:
        """
        Optimise les paramètres d'une stratégie.
        
        Args:
            strategy: Constructeur de stratégie
            data: DataFrame avec les données OHLCV
            param_ranges: Plages de paramètres à tester
            metric: Métrique à optimiser
            max_combinations: Nombre maximum de combinaisons à tester
            
        Returns:
            Dict[str, Any]: Résultats de l'optimisation
        """
        try:
            logger.info(f"Début de l'optimisation pour '{strategy.config.name}'...")
            
            # Calculer toutes les combinaisons de paramètres
            from itertools import product
            
            # Extraire les noms et valeurs des paramètres
            param_names = list(param_ranges.keys())
            param_values = list(param_ranges.values())
            
            # Calculer toutes les combinaisons
            all_combinations = list(product(*param_values))
            
            # Limiter le nombre de combinaisons
            if len(all_combinations) > max_combinations:
                logger.warning(f"Nombre de combinaisons ({len(all_combinations)}) > maximum ({max_combinations}), échantillonnage aléatoire")
                import random
                all_combinations = random.sample(all_combinations, max_combinations)
            
            # Préparation des résultats
            results = {
                "strategy_id": strategy.config.id,
                "strategy_name": strategy.config.name,
                "timestamp": datetime.now().isoformat(),
                "param_ranges": param_ranges,
                "metric": metric,
                "total_combinations": len(all_combinations),
                "results": []
            }
            
            # Tester chaque combinaison
            for i, combination in enumerate(all_combinations):
                # Créer un dictionnaire de paramètres
                params = {name: value for name, value in zip(param_names, combination)}
                
                # Cloner la stratégie
                test_strategy = strategy.clone()
                
                # Appliquer les paramètres
                for name, value in params.items():
                    # Déterminer où appliquer le paramètre (indicateur, condition, etc.)
                    if "." in name:
                        parts = name.split(".")
                        if parts[0] == "indicator":
                            # Format: indicator.nom_indicateur.paramètre
                            if len(parts) >= 3:
                                indicator_name = parts[1]
                                param_name = parts[2]
                                # Récupérer la configuration de l'indicateur
                                indicator = test_strategy.config.indicators_manager.get_indicator(indicator_name)
                                if indicator:
                                    # Mettre à jour le paramètre
                                    indicator.update_params(**{param_name: value})
                        elif parts[0] == "risk":
                            # Format: risk.paramètre
                            if len(parts) >= 2:
                                param_name = parts[1]
                                # Mettre à jour la configuration de risque
                                test_strategy.config.risk_config.update_params(**{param_name: value})
                    else:
                        # Paramètre global, définir via set_parameter
                        test_strategy.set_parameter(name, value)
                
                # Exécuter la simulation
                logger.info(f"Test combinaison {i+1}/{len(all_combinations)}: {params}")
                sim_results = self.run_simulation(data, test_strategy)
                
                if sim_results and "performance" in sim_results:
                    # Extraire la métrique
                    metric_value = sim_results["performance"].get(metric, 0)
                    
                    # Enregistrer les résultats
                    combination_result = {
                        "params": params,
                        "metric_value": metric_value,
                        "roi_pct": sim_results["performance"]["roi_pct"],
                        "win_rate_pct": sim_results["performance"]["win_rate_pct"],
                        "max_drawdown_pct": sim_results["performance"]["max_drawdown_pct"],
                        "total_trades": sim_results["performance"]["total_trades"]
                    }
                    
                    results["results"].append(combination_result)
            
            # Trier les résultats par valeur de métrique décroissante
            results["results"].sort(key=lambda x: x["metric_value"], reverse=True)
            
            # Ajouter les meilleurs paramètres
            if results["results"]:
                results["best_params"] = results["results"][0]["params"]
                results["best_metric_value"] = results["results"][0]["metric_value"]
            
            logger.info(f"Optimisation terminée pour '{strategy.config.name}'")
            logger.info(f"Meilleurs paramètres: {results.get('best_params', {})}")
            logger.info(f"Meilleure valeur de {metric}: {results.get('best_metric_value', 0)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation: {str(e)}")
            return None
    
    def apply_best_parameters(self, optimization_results: Dict[str, Any], strategy: StrategyConstructor = None) -> Optional[StrategyConstructor]:
        """
        Applique les meilleurs paramètres d'une optimisation à une stratégie.
        
        Args:
            optimization_results: Résultats de l'optimisation
            strategy: Constructeur de stratégie (utilise la stratégie courante si None)
            
        Returns:
            Optional[StrategyConstructor]: Stratégie avec les meilleurs paramètres
        """
        strategy = strategy or self.current_strategy
        
        if strategy is None:
            logger.error("Aucune stratégie pour appliquer les paramètres")
            return None
        
        if "best_params" not in optimization_results:
            logger.error("Aucun paramètre optimal trouvé dans les résultats d'optimisation")
            return None
        
        try:
            # Cloner la stratégie
            optimized_strategy = strategy.clone()
            
            # Appliquer les meilleurs paramètres
            best_params = optimization_results["best_params"]
            
            for name, value in best_params.items():
                # Déterminer où appliquer le paramètre (indicateur, condition, etc.)
                if "." in name:
                    parts = name.split(".")
                    if parts[0] == "indicator":
                        # Format: indicator.nom_indicateur.paramètre
                        if len(parts) >= 3:
                            indicator_name = parts[1]
                            param_name = parts[2]
                            # Récupérer la configuration de l'indicateur
                            indicator = optimized_strategy.config.indicators_manager.get_indicator(indicator_name)
                            if indicator:
                                # Mettre à jour le paramètre
                                indicator.update_params(**{param_name: value})
                    elif parts[0] == "risk":
                        # Format: risk.paramètre
                        if len(parts) >= 2:
                            param_name = parts[1]
                            # Mettre à jour la configuration de risque
                            optimized_strategy.config.risk_config.update_params(**{param_name: value})
                else:
                    # Paramètre global, définir via set_parameter
                    optimized_strategy.set_parameter(name, value)
            
            # Mettre à jour le nom et la description
            original_name = optimized_strategy.config.name
            optimized_strategy.set_name(f"{original_name} (Optimisé)")
            optimized_strategy.set_description(
                f"{optimized_strategy.config.description}\n" +
                f"Optimisé le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} " +
                f"avec métrique {optimization_results['metric']}"
            )
            
            # Ajouter un tag
            optimized_strategy.add_tag("optimized")
            
            # Mettre à jour la stratégie courante
            self.current_strategy = optimized_strategy
            
            logger.info(f"Paramètres optimaux appliqués à '{original_name}'")
            return optimized_strategy
            
        except Exception as e:
            logger.error(f"Erreur lors de l'application des paramètres optimaux: {str(e)}")
            return None
    
    def get_strategy_info(self, strategy_id: str) -> Dict[str, Any]:
        """
        Récupère les informations détaillées sur une stratégie.
        
        Args:
            strategy_id: ID de la stratégie
            
        Returns:
            Dict[str, Any]: Informations sur la stratégie
        """
        try:
            # Chercher le fichier de stratégie
            strategy_files = [f for f in os.listdir(self.strategies_dir) if f.startswith(f"{strategy_id}_") and f.endswith(".json")]
            
            if not strategy_files:
                logger.error(f"Aucune stratégie trouvée avec l'ID '{strategy_id}'")
                return None
            
            # Lire le fichier JSON
            filepath = os.path.join(self.strategies_dir, strategy_files[0])
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extraire les informations principales
            info = {
                "id": data.get("id", ""),
                "name": data.get("name", ""),
                "description": data.get("description", ""),
                "version": data.get("version", ""),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "tags": data.get("tags", []),
                "filepath": filepath,
                "indicators": [],
                "risk": {}
            }
            
            # Ajouter les informations sur les indicateurs
            if "indicators" in data:
                for name, indicator_data in data["indicators"].items():
                    indicator_info = {
                        "name": name,
                        "type": indicator_data.get("type", ""),
                        "params": indicator_data.get("params", {})
                    }
                    info["indicators"].append(indicator_info)
            
            # Ajouter les informations sur la gestion du risque
            if "risk" in data:
                info["risk"] = data["risk"]
            
            return info
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des informations sur la stratégie: {str(e)}")
            return None
    
    def export_strategy(self, strategy_id: str, output_path: str) -> bool:
        """
        Exporte une stratégie dans un fichier séparé.
        
        Args:
            strategy_id: ID de la stratégie
            output_path: Chemin du fichier de sortie
            
        Returns:
            bool: Succès de l'exportation
        """
        try:
            # Chercher le fichier de stratégie
            strategy_files = [f for f in os.listdir(self.strategies_dir) if f.startswith(f"{strategy_id}_") and f.endswith(".json")]
            
            if not strategy_files:
                logger.error(f"Aucune stratégie trouvée avec l'ID '{strategy_id}'")
                return False
            
            # Copier le fichier
            source_path = os.path.join(self.strategies_dir, strategy_files[0])
            shutil.copy2(source_path, output_path)
            
            logger.info(f"Stratégie '{strategy_id}' exportée vers {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exportation de la stratégie: {str(e)}")
            return False
    
    def import_strategy(self, filepath: str) -> Optional[StrategyConstructor]:
        """
        Importe une stratégie depuis un fichier.
        
        Args:
            filepath: Chemin du fichier de stratégie
            
        Returns:
            Optional[StrategyConstructor]: Instance du constructeur de stratégie
        """
        try:
            # Charger la stratégie
            constructor = StrategyConstructor.load(filepath)
            
            # Générer un nouvel ID pour éviter les conflits
            original_id = constructor.config.id
            constructor.config.id = str(uuid.uuid4())[:8]
            
            # Mettre à jour la date de création et de modification
            constructor.config.created_at = datetime.now().isoformat()
            constructor.config.updated_at = datetime.now().isoformat()
            
            # Ajouter un tag pour indiquer l'importation
            constructor.add_tag(f"imported_{datetime.now().strftime('%Y%m%d')}")
            
            # Sauvegarder la stratégie importée
            self.current_strategy = constructor
            self.save_strategy(constructor)
            
            logger.info(f"Stratégie importée depuis {filepath} avec le nouvel ID {constructor.config.id}")
            return constructor
            
        except Exception as e:
            logger.error(f"Erreur lors de l'importation de la stratégie: {str(e)}")
            return None
    
    def delete_strategy(self, strategy_id: str) -> bool:
        """
        Supprime une stratégie.
        
        Args:
            strategy_id: ID de la stratégie
            
        Returns:
            bool: Succès de la suppression
        """
        try:
            # Chercher le fichier de stratégie
            strategy_files = [f for f in os.listdir(self.strategies_dir) if f.startswith(f"{strategy_id}_") and f.endswith(".json")]
            
            if not strategy_files:
                logger.error(f"Aucune stratégie trouvée avec l'ID '{strategy_id}'")
                return False
            
            # Supprimer le fichier
            filepath = os.path.join(self.strategies_dir, strategy_files[0])
            os.remove(filepath)
            
            # Si c'est la stratégie courante, la réinitialiser
            if self.current_strategy and self.current_strategy.config.id == strategy_id:
                self.current_strategy = None
            
            # Supprimer les résultats de simulation associés
            if strategy_id in self.simulation_results:
                del self.simulation_results[strategy_id]
            
            logger.info(f"Stratégie '{strategy_id}' supprimée")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de la stratégie: {str(e)}")
            return False
    
    def apply_preset(self, preset_name: str) -> Optional[StrategyConstructor]:
        """
        Applique un preset prédéfini à la stratégie courante.
        
        Args:
            preset_name: Nom du preset
            
        Returns:
            Optional[StrategyConstructor]: Stratégie avec le preset appliqué
        """
        if self.current_strategy is None:
            logger.error("Aucune stratégie courante pour appliquer le preset")
            return None
        
        try:
            # Appliquer le preset
            success = self.current_strategy.apply_preset(preset_name)
            
            if success:
                logger.info(f"Preset '{preset_name}' appliqué à '{self.current_strategy.config.name}'")
                return self.current_strategy
            else:
                logger.error(f"Échec de l'application du preset '{preset_name}'")
                return None
                
        except Exception as e:
            logger.error(f"Erreur lors de l'application du preset: {str(e)}")
            return None


# Fonction utilitaire pour créer un gestionnaire de stratégies
def get_strategy_manager(strategies_dir: str = "strategies") -> StrategyManager:
    """
    Crée et retourne un gestionnaire de stratégies.
    
    Args:
        strategies_dir: Répertoire des stratégies
        
    Returns:
        StrategyManager: Instance du gestionnaire de stratégies
    """
    return StrategyManager(strategies_dir)