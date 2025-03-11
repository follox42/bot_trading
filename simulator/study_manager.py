"""
Module de gestion des études pour les stratégies de trading - Version intégrée et améliorée.
Intègre pleinement la configuration flexible et centralise la gestion des études.
Permet la modification des paramètres et des poids des scores avec recalcul automatique.
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import logging
import traceback
import uuid
import matplotlib.pyplot as plt
from pathlib import Path
import optuna
import copy
import shutil

# Import des modules du système de trading
from indicators import SignalGenerator, Block, Condition, Operator, LogicOperator, IndicatorType
from risk import PositionCalculator, RiskMode
from simulator import Simulator, SimulationConfig
from config import (
    FlexibleTradingConfig, create_flexible_default_config, 
    load_flexible_config_from_file, save_flexible_config_to_file,
    convert_to_simulator_config, create_position_calculator_config,
    MarginMode, TradingMode
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('study_manager.log', mode='a'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('study_manager')

class IntegratedStudyManager:
    """
    Gestionnaire d'études amélioré pour les stratégies de trading.
    Intègre pleinement la configuration flexible et centralise la gestion des études.
    Permet la modification des paramètres et le recalcul des scores.
    """
    def __init__(self, base_dir: str = "studies"):
        """
        Initialise le gestionnaire d'études
        
        Args:
            base_dir: Répertoire de base pour stocker les études
        """
        self.base_dir = base_dir
        self._initialize_storage()
        
    def _initialize_storage(self):
        """Initialise la structure de stockage"""
        # Création du répertoire principal s'il n'existe pas
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            logger.info(f"Répertoire des études créé: {self.base_dir}")
            
    def create_study(self, study_name: str, metadata: Dict, trading_config: Optional[FlexibleTradingConfig] = None) -> bool:
        """
        Crée une nouvelle étude avec les métadonnées et la configuration spécifiées
        
        Args:
            study_name: Nom de l'étude
            metadata: Métadonnées de l'étude
            trading_config: Configuration flexible du trading (optionnel, sinon config par défaut)
            
        Returns:
            bool: True si la création a réussi, False sinon
        """
        try:
            # Vérifier si l'étude existe déjà
            if self.study_exists(study_name):
                logger.error(f"L'étude '{study_name}' existe déjà")
                return False
            
            # Créer le dossier de l'étude
            study_dir = os.path.join(self.base_dir, study_name)
            os.makedirs(study_dir)
            
            # Créer la structure des sous-dossiers
            os.makedirs(os.path.join(study_dir, "strategies"), exist_ok=True)
            os.makedirs(os.path.join(study_dir, "results"), exist_ok=True)
            os.makedirs(os.path.join(study_dir, "configs"), exist_ok=True)
            os.makedirs(os.path.join(study_dir, "optimization"), exist_ok=True)
            
            # Ajouter des informations de base
            if "creation_date" not in metadata:
                metadata["creation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if "status" not in metadata:
                metadata["status"] = "created"
                
            # Ajouter les poids de score par défaut si non spécifiés
            if "score_weights" not in metadata:
                metadata["score_weights"] = {
                    "roi": 2.5,
                    "win_rate": 0.5,
                    "max_drawdown": 2.0,
                    "profit_factor": 2.0,
                    "total_trades": 1.0,
                    "avg_profit": 1.0
                }
            
            # Enregistrer les métadonnées
            metadata_path = os.path.join(study_dir, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            
            # Utiliser la configuration fournie ou créer une configuration par défaut
            config = trading_config if trading_config else create_flexible_default_config()
            
            # Sauvegarder la configuration
            config_path = os.path.join(study_dir, "configs", "trading_config.json")
            
            # Sauvegarder au format JSON
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config.to_dict(), f, indent=4, ensure_ascii=False)
            
            logger.info(f"Étude '{study_name}' créée avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'étude '{study_name}': {str(e)}")
            traceback.print_exc()
            return False
    
    def study_exists(self, study_name: str) -> bool:
        """
        Vérifie si une étude existe
        
        Args:
            study_name: Nom de l'étude à vérifier
            
        Returns:
            bool: True si l'étude existe, False sinon
        """
        study_dir = os.path.join(self.base_dir, study_name)
        return os.path.exists(study_dir)
    
    def get_study_metadata(self, study_name: str) -> Optional[Dict]:
        """
        Récupère les métadonnées d'une étude
        
        Args:
            study_name: Nom de l'étude
            
        Returns:
            Optional[Dict]: Métadonnées de l'étude ou None si l'étude n'existe pas
        """
        if not self.study_exists(study_name):
            logger.error(f"L'étude '{study_name}' n'existe pas")
            return None
        
        try:
            metadata_path = os.path.join(self.base_dir, study_name, "metadata.json")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erreur lors de la lecture des métadonnées de l'étude '{study_name}': {str(e)}")
            return None
    
    def update_study_metadata(self, study_name: str, metadata: Dict) -> bool:
        """
        Met à jour les métadonnées d'une étude
        
        Args:
            study_name: Nom de l'étude
            metadata: Nouvelles métadonnées
            
        Returns:
            bool: True si la mise à jour a réussi, False sinon
        """
        if not self.study_exists(study_name):
            logger.error(f"L'étude '{study_name}' n'existe pas")
            return False
        
        try:
            metadata_path = os.path.join(self.base_dir, study_name, "metadata.json")
            
            # Mise à jour de la date de dernière modification
            metadata["last_modified"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Sauvegarde des métadonnées
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
                
            logger.info(f"Métadonnées de l'étude '{study_name}' mises à jour")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des métadonnées: {str(e)}")
            return False
    
    def get_trading_config(self, study_name: str) -> Optional[FlexibleTradingConfig]:
        """
        Récupère la configuration de trading flexible d'une étude
        
        Args:
            study_name: Nom de l'étude
            
        Returns:
            Optional[FlexibleTradingConfig]: Configuration de trading ou None si non trouvée
        """
        if not self.study_exists(study_name):
            logger.error(f"L'étude '{study_name}' n'existe pas")
            return None
        
        try:
            config_path = os.path.join(self.base_dir, study_name, "configs", "trading_config.json")
            if not os.path.exists(config_path):
                logger.warning(f"Configuration de trading non trouvée pour l'étude '{study_name}'")
                return create_flexible_default_config()
            
            # Charger la configuration depuis le fichier JSON
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
                
            # Convertir le dictionnaire en objet FlexibleTradingConfig
            return FlexibleTradingConfig.from_dict(config_dict)
            
        except Exception as e:
            logger.error(f"Erreur lors de la lecture de la configuration de trading: {str(e)}")
            return create_flexible_default_config()
    
    def update_trading_config(self, study_name: str, trading_config: FlexibleTradingConfig) -> bool:
        """
        Met à jour la configuration de trading d'une étude
        
        Args:
            study_name: Nom de l'étude
            trading_config: Nouvelle configuration de trading
            
        Returns:
            bool: True si la mise à jour a réussi, False sinon
        """
        if not self.study_exists(study_name):
            logger.error(f"L'étude '{study_name}' n'existe pas")
            return False
        
        try:
            config_path = os.path.join(self.base_dir, study_name, "configs", "trading_config.json")
            
            # Sauvegarder au format JSON
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(trading_config.to_dict(), f, indent=4, ensure_ascii=False)
            
            # Mettre à jour les métadonnées
            metadata = self.get_study_metadata(study_name)
            if metadata:
                metadata['last_modified'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.update_study_metadata(study_name, metadata)
            
            logger.info(f"Configuration de trading mise à jour pour l'étude '{study_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de la configuration de trading: {str(e)}")
            return False
    
    def update_study_status(self, study_name: str, status: str) -> bool:
        """
        Met à jour le statut d'une étude
        
        Args:
            study_name: Nom de l'étude
            status: Nouveau statut
            
        Returns:
            bool: True si la mise à jour a réussi, False sinon
        """
        if not self.study_exists(study_name):
            logger.error(f"L'étude '{study_name}' n'existe pas")
            return False
        
        try:
            metadata = self.get_study_metadata(study_name)
            if metadata:
                metadata['status'] = status
                metadata['last_modified'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                return self.update_study_metadata(study_name, metadata)
            return False
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du statut de l'étude '{study_name}': {str(e)}")
            return False

    def get_score_weights(self, study_name: str) -> Optional[Dict]:
        """
        Récupère les poids des scores pour une étude
        
        Args:
            study_name: Nom de l'étude
            
        Returns:
            Optional[Dict]: Poids des scores ou None si non trouvés
        """
        metadata = self.get_study_metadata(study_name)
        if not metadata:
            return None
        
        return metadata.get("score_weights", {})
    
    def update_score_weights(self, study_name: str, new_weights: Dict) -> bool:
        """
        Met à jour les poids des scores pour une étude et recalcule les scores si nécessaire
        
        Args:
            study_name: Nom de l'étude
            new_weights: Nouveaux poids des scores
            
        Returns:
            bool: True si la mise à jour a réussi, False sinon
        """
        if not self.study_exists(study_name):
            logger.error(f"L'étude '{study_name}' n'existe pas")
            return False
        
        try:
            # Récupérer les métadonnées et sauvegarder les anciens poids
            metadata = self.get_study_metadata(study_name)
            if not metadata:
                return False
            
            old_weights = copy.deepcopy(metadata.get("score_weights", {}))
            
            # Mettre à jour les poids
            metadata["score_weights"] = new_weights
            
            # Sauvegarder les métadonnées
            if not self.update_study_metadata(study_name, metadata):
                return False
            
            # Vérifier s'il y a des résultats d'optimisation
            optimization_results = self.get_optimization_results(study_name)
            if not optimization_results:
                logger.info(f"Aucun résultat d'optimisation à mettre à jour pour l'étude '{study_name}'")
                return True
            
            # S'il y a des résultats d'optimisation, recalculer les scores
            if old_weights != new_weights:
                logger.info(f"Recalcul des scores avec les nouveaux poids pour l'étude '{study_name}'")
                return self._recalculate_scores(study_name, new_weights)
            
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des poids de score: {str(e)}")
            traceback.print_exc()
            return False
    
    def _recalculate_scores(self, study_name: str, new_weights: Dict) -> bool:
        """
        Recalcule les scores des trials Optuna avec les nouveaux poids
        
        Args:
            study_name: Nom de l'étude
            new_weights: Nouveaux poids de score
            
        Returns:
            bool: True si le recalcul a réussi, False sinon
        """
        try:
            # Récupérer les résultats d'optimisation
            optimization_results = self.get_optimization_results(study_name)
            if not optimization_results:
                logger.warning(f"Aucun résultat d'optimisation trouvé pour l'étude '{study_name}'")
                return False
            
            # Vérifier s'il y a une étude Optuna
            storage_url = f"sqlite:///{study_name}_optimization.db"
            
            try:
                # Charger l'étude Optuna
                study = optuna.load_study(
                    study_name=f"{study_name}_opt",
                    storage=storage_url
                )
            except Exception as e:
                logger.error(f"Impossible de charger l'étude Optuna: {str(e)}")
                return False
            
            # Recalculer les scores pour chaque trial
            best_score = float('-inf')
            best_trial_id = None
            
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE and trial.user_attrs:
                    # Récupérer les métriques de performance
                    metrics = {
                        "roi": trial.user_attrs.get("roi", 0),
                        "win_rate": trial.user_attrs.get("win_rate", 0),
                        "max_drawdown": trial.user_attrs.get("max_drawdown", 1),
                        "profit_factor": trial.user_attrs.get("profit_factor", 0),
                        "total_trades": trial.user_attrs.get("total_trades", 0),
                        "avg_profit": trial.user_attrs.get("avg_profit", 0)
                    }
                    
                    # Calculer le nouveau score
                    new_score = self._calculate_weighted_score(metrics, new_weights)
                    
                    # Mettre à jour le trial avec le nouveau score
                    study._storage.set_trial_value(trial._trial_id, new_score)
                    
                    # Vérifier si c'est le meilleur score
                    if new_score > best_score:
                        best_score = new_score
                        best_trial_id = trial.number
            
            # Forcer la mise à jour du meilleur trial
            study._storage.set_study_user_attr(study._study_id, "best_trial_id", best_trial_id)
            
            # Recharger l'étude pour s'assurer que les modifications sont prises en compte
            study = optuna.load_study(
                study_name=f"{study_name}_opt",
                storage=storage_url
            )
            
            # Mettre à jour les résultats d'optimisation
            best_trial = study.best_trial
            optimization_results["best_trial_id"] = best_trial.number
            optimization_results["best_score"] = best_trial.value
            
            # Mettre à jour les meilleurs trials en fonction des nouveaux scores
            best_trials = sorted(
                [t for t in study.trials if t.value is not None and t.value > float('-inf')],
                key=lambda t: t.value if t.value is not None else float('-inf'),
                reverse=True
            )[:10]  # Top 10
            
            optimization_results["best_trials"] = [{
                'trial_id': t.number,
                'score': t.value,
                'params': t.params,
                'metrics': {
                    k: v for k, v in t.user_attrs.items()
                    if k in ['roi', 'win_rate', 'total_trades', 'max_drawdown', 'profit_factor', 'avg_profit']
                }
            } for t in best_trials]
            
            # Sauvegarder les résultats d'optimisation mis à jour
            self.save_optimization_results(study_name, optimization_results)
            
            # Mise à jour des stratégies avec les nouveaux meilleurs trials
            # (Régénérer les 5 meilleures stratégies)
            self.save_best_strategies(study_name, study)
            
            logger.info(f"Scores recalculés avec succès pour l'étude '{study_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du recalcul des scores: {str(e)}")
            traceback.print_exc()
            return False
    
    def _calculate_weighted_score(self, metrics: Dict[str, float], weights: Dict[str, float]) -> float:
        """
        Calcule un score pondéré à partir des métriques et des poids
        
        Args:
            metrics: Dictionnaire des métriques de performance
            weights: Dictionnaire des poids pour chaque métrique
            
        Returns:
            float: Score final
        """
        # Normalisation des métriques
        normalized_metrics = {}
        
        # ROI - normalisation sigmoïde
        roi = metrics.get("roi", 0)
        normalized_metrics["roi"] = min(1.0, max(0, (1 / (1 + np.exp(-roi * 2)) - 0.5) * 2))
        
        # Win rate - déjà entre 0 et 1
        normalized_metrics["win_rate"] = metrics.get("win_rate", 0)
        
        # Drawdown - transformation pour pénaliser les grands drawdowns
        max_dd = metrics.get("max_drawdown", 1)
        normalized_metrics["max_drawdown"] = max(0.0, 1.0 - max_dd)
        
        # Profit factor - transformation log
        pf = metrics.get("profit_factor", 0)
        normalized_metrics["profit_factor"] = min(1.0, max(0.0, np.log(pf + 0.1) / 2.0)) if pf > 0 else 0
        
        # Nombre de trades - échelle logarithmique
        trades = metrics.get("total_trades", 0)
        normalized_metrics["total_trades"] = min(1.0, np.log(trades + 1) / np.log(1001))
        
        # Profit moyen par trade
        avg_profit = metrics.get("avg_profit", 0)
        normalized_metrics["avg_profit"] = min(1.0, max(0.0, avg_profit * 10 + 0.5))
        
        # Calcul du score final pondéré
        score = 0.0
        total_weight = sum(weights.values())
        
        for metric, weight in weights.items():
            if metric in normalized_metrics:
                score += normalized_metrics[metric] * (weight / total_weight)
        
        # Transformation non-linéaire pour différencier les bonnes stratégies
        return (score ** 1.2) * 10.0  # Échelle finale de 0 à 10

    def update_study_parameters(self, study_name: str, param_updates: Dict) -> bool:
        """
        Met à jour les paramètres d'une étude (plages de valeurs, configurations, etc.)
        
        Args:
            study_name: Nom de l'étude
            param_updates: Dictionnaire des mises à jour de paramètres
            
        Returns:
            bool: True si la mise à jour a réussi, False sinon
        """
        if not self.study_exists(study_name):
            logger.error(f"L'étude '{study_name}' n'existe pas")
            return False
        
        try:
            # Récupérer la configuration actuelle
            trading_config = self.get_trading_config(study_name)
            if not trading_config:
                return False
            
            # Liste des paramètres immuables
            immutable_params = ["study_name", "asset", "timeframe", "exchange", "data_file"]
            
            # Vérifier si des paramètres immuables sont dans les mises à jour
            immutable_updates = [p for p in immutable_params if p in param_updates]
            if immutable_updates:
                logger.warning(f"Les paramètres suivants ne peuvent pas être modifiés: {immutable_updates}")
                # Supprimer les paramètres immuables des mises à jour
                for param in immutable_updates:
                    param_updates.pop(param)
            
            # Appliquer les mises à jour selon leur catégorie
            updated = False
            
            # Mise à jour des paramètres de risque
            if "risk_config" in param_updates:
                risk_updates = param_updates["risk_config"]
                
                # Mettre à jour les plages de position size
                if "position_size_range" in risk_updates:
                    trading_config.risk_config.position_size_range = tuple(risk_updates["position_size_range"])
                    updated = True
                
                # Mettre à jour les plages de stop loss
                if "sl_range" in risk_updates:
                    trading_config.risk_config.sl_range = tuple(risk_updates["sl_range"])
                    updated = True
                
                # Mettre à jour les plages de take profit
                if "tp_multiplier_range" in risk_updates:
                    trading_config.risk_config.tp_multiplier_range = tuple(risk_updates["tp_multiplier_range"])
                    updated = True
                
                # Mise à jour des configurations par mode
                if "mode_configs" in risk_updates:
                    for mode_str, mode_config in risk_updates["mode_configs"].items():
                        mode = RiskMode(mode_str)
                        
                        # Si la configuration de ce mode existe déjà, mettre à jour ses paramètres
                        if mode in trading_config.risk_config.mode_configs:
                            for param, value in mode_config.items():
                                if hasattr(trading_config.risk_config.mode_configs[mode], param):
                                    setattr(trading_config.risk_config.mode_configs[mode], param, tuple(value) if isinstance(value, list) else value)
                                    updated = True
            
            # Mise à jour des paramètres de simulation
            if "sim_config" in param_updates:
                sim_updates = param_updates["sim_config"]
                
                # Mettre à jour la plage de balance initiale
                if "initial_balance_range" in sim_updates:
                    trading_config.sim_config.initial_balance_range = tuple(sim_updates["initial_balance_range"])
                    updated = True
                
                # Mettre à jour les frais
                if "fee" in sim_updates:
                    trading_config.sim_config.fee = sim_updates["fee"]
                    updated = True
                
                # Mettre à jour le slippage
                if "slippage" in sim_updates:
                    trading_config.sim_config.slippage = sim_updates["slippage"]
                    updated = True
                
                # Mettre à jour la plage de levier
                if "leverage_range" in sim_updates:
                    trading_config.sim_config.leverage_range = tuple(sim_updates["leverage_range"])
                    updated = True
            
            # Mise à jour des paramètres des indicateurs
            if "available_indicators" in param_updates:
                for ind_name, ind_config in param_updates["available_indicators"].items():
                    if ind_name in trading_config.available_indicators:
                        # Mettre à jour les plages de période
                        if "min_period" in ind_config:
                            trading_config.available_indicators[ind_name].min_period = ind_config["min_period"]
                            updated = True
                        
                        if "max_period" in ind_config:
                            trading_config.available_indicators[ind_name].max_period = ind_config["max_period"]
                            updated = True
                        
                        if "step" in ind_config:
                            trading_config.available_indicators[ind_name].step = ind_config["step"]
                            updated = True
            
            # Mise à jour des paramètres de structure de stratégie
            if "strategy_structure" in param_updates:
                struct_updates = param_updates["strategy_structure"]
                
                for param, value in struct_updates.items():
                    if hasattr(trading_config.strategy_structure, param):
                        setattr(trading_config.strategy_structure, param, tuple(value) if isinstance(value, list) else value)
                        updated = True
            
            # Si des mises à jour ont été effectuées, sauvegarder la configuration
            if updated:
                if self.update_trading_config(study_name, trading_config):
                    logger.info(f"Paramètres de l'étude '{study_name}' mis à jour avec succès")
                    return True
            else:
                logger.warning(f"Aucun paramètre n'a été mis à jour pour l'étude '{study_name}'")
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des paramètres de l'étude: {str(e)}")
            traceback.print_exc()
            return False

    def save_strategy(self, study_name: str, strategy_rank: int, 
                     signal_generator: SignalGenerator, 
                     position_calculator: PositionCalculator, 
                     performance: Dict = None) -> bool:
        """
        Sauvegarde une stratégie dans une étude
        
        Args:
            study_name: Nom de l'étude
            strategy_rank: Rang de la stratégie (1 = meilleure)
            signal_generator: Générateur de signaux
            position_calculator: Calculateur de position
            performance: Métriques de performance (optionnel)
            
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        if not self.study_exists(study_name):
            logger.error(f"L'étude '{study_name}' n'existe pas")
            return False
        
        try:
            # Chemin du dossier des stratégies
            strategies_dir = os.path.join(self.base_dir, study_name, "strategies")
            os.makedirs(strategies_dir, exist_ok=True)
            
            # Nom du fichier de stratégie
            strategy_file = os.path.join(strategies_dir, f"strategy_{strategy_rank}.json")
            
            # Extraire les blocs d'achat et de vente
            buy_blocks = [block.to_dict() for block in signal_generator.buy_blocks]
            sell_blocks = [block.to_dict() for block in signal_generator.sell_blocks]
            
            # Récupérer la configuration de trading
            trading_config = self.get_trading_config(study_name)
            
            # Extraire la configuration de risque
            risk_config = {
                "risk_mode": position_calculator.mode.value,
                "base_position": position_calculator.base_position,
                "base_sl": position_calculator.base_sl,
                "tp_multiplier": position_calculator.tp_multiplier
            }
            
            # Si configuration spécifique au mode
            if position_calculator.mode == RiskMode.ATR_BASED:
                risk_config.update({
                    "atr_period": position_calculator.atr_period,
                    "atr_multiplier": position_calculator.atr_multiplier
                })
            elif position_calculator.mode == RiskMode.VOLATILITY_BASED:
                risk_config.update({
                    "vol_period": position_calculator.vol_period,
                    "vol_multiplier": position_calculator.vol_multiplier
                })
            
            # Création de l'objet stratégie
            strategy_data = {
                "rank": strategy_rank,
                "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "buy_blocks": buy_blocks,
                "sell_blocks": sell_blocks,
                "risk_config": risk_config,
                "indicator_list": list(signal_generator.indicator_map.keys())
            }
            
            # Ajouter les métriques de performance si fournies
            if performance:
                strategy_data["performance"] = performance
            
            # Sauvegarde de la stratégie
            with open(strategy_file, 'w', encoding='utf-8') as f:
                json.dump(strategy_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Stratégie {strategy_rank} sauvegardée pour l'étude '{study_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la stratégie: {str(e)}")
            traceback.print_exc()
            return False
    
    def load_strategy(self, study_name: str, strategy_rank: int = 1) -> Optional[Tuple[SignalGenerator, PositionCalculator, Dict]]:
        """
        Charge une stratégie depuis une étude
        
        Args:
            study_name: Nom de l'étude
            strategy_rank: Rang de la stratégie (1 = meilleure)
            
        Returns:
            Optional[Tuple[SignalGenerator, PositionCalculator, Dict]]: 
                Générateur de signaux, calculateur de position et performance,
                ou None si la stratégie n'existe pas
        """
        if not self.study_exists(study_name):
            logger.error(f"L'étude '{study_name}' n'existe pas")
            return None
        
        try:
            # Chemin du fichier de stratégie
            strategy_file = os.path.join(self.base_dir, study_name, "strategies", f"strategy_{strategy_rank}.json")
            
            if not os.path.exists(strategy_file):
                logger.error(f"Stratégie {strategy_rank} non trouvée pour l'étude '{study_name}'")
                return None
            
            # Chargement des données de stratégie
            with open(strategy_file, 'r', encoding='utf-8') as f:
                strategy_data = json.load(f)
            
            # Récupération de la configuration de trading
            trading_config = self.get_trading_config(study_name)
            
            # Création du générateur de signaux
            signal_generator = SignalGenerator()
            
            # Ajout des blocs d'achat
            for block_dict in strategy_data.get("buy_blocks", []):
                block = Block.from_dict(block_dict)
                signal_generator.add_block(block, is_buy=True)
            
            # Ajout des blocs de vente
            for block_dict in strategy_data.get("sell_blocks", []):
                block = Block.from_dict(block_dict)
                signal_generator.add_block(block, is_buy=False)
            
            # Création du calculateur de position
            risk_config = strategy_data.get("risk_config", {})
            risk_mode = RiskMode(risk_config.get("risk_mode", "fixed"))
            
            position_calculator = PositionCalculator(
                mode=risk_mode,
                config=risk_config
            )
            
            # Récupération des métriques de performance
            performance = strategy_data.get("performance", {})
            
            logger.info(f"Stratégie {strategy_rank} chargée depuis l'étude '{study_name}'")
            
            return signal_generator, position_calculator, performance
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la stratégie: {str(e)}")
            traceback.print_exc()
            return None
    
    def modify_strategy(self, study_name: str, strategy_rank: int, updates: Dict) -> bool:
        """
        Modifie les paramètres d'une stratégie existante
        
        Args:
            study_name: Nom de l'étude
            strategy_rank: Rang de la stratégie
            updates: Dictionnaire des mises à jour
            
        Returns:
            bool: True si la modification a réussi, False sinon
        """
        if not self.study_exists(study_name):
            logger.error(f"L'étude '{study_name}' n'existe pas")
            return False
        
        try:
            # Charger la stratégie
            strategy = self.load_strategy(study_name, strategy_rank)
            if not strategy:
                return False
            
            signal_generator, position_calculator, performance = strategy
            
            # Appliquer les mises à jour
            updated = False
            
            # Mise à jour des paramètres de risque
            if "risk_config" in updates:
                risk_updates = updates["risk_config"]
                
                # Mise à jour du mode de risque
                if "risk_mode" in risk_updates:
                    new_mode = RiskMode(risk_updates["risk_mode"])
                    if new_mode != position_calculator.mode:
                        position_calculator.set_mode(new_mode)
                        updated = True
                
                # Mise à jour des paramètres de base
                if "base_position" in risk_updates:
                    position_calculator.base_position = risk_updates["base_position"]
                    updated = True
                
                if "base_sl" in risk_updates:
                    position_calculator.base_sl = risk_updates["base_sl"]
                    updated = True
                
                if "tp_multiplier" in risk_updates:
                    position_calculator.tp_multiplier = risk_updates["tp_multiplier"]
                    updated = True
                
                # Mise à jour des paramètres spécifiques au mode
                if position_calculator.mode == RiskMode.ATR_BASED:
                    if "atr_period" in risk_updates:
                        position_calculator.atr_period = risk_updates["atr_period"]
                        updated = True
                    
                    if "atr_multiplier" in risk_updates:
                        position_calculator.atr_multiplier = risk_updates["atr_multiplier"]
                        updated = True
                
                elif position_calculator.mode == RiskMode.VOLATILITY_BASED:
                    if "vol_period" in risk_updates:
                        position_calculator.vol_period = risk_updates["vol_period"]
                        updated = True
                    
                    if "vol_multiplier" in risk_updates:
                        position_calculator.vol_multiplier = risk_updates["vol_multiplier"]
                        updated = True
            
            # Mise à jour des blocs de trading (plus complexe)
            if "blocks" in updates:
                # Pour simplifier, nous ne traitons que le remplacement complet des blocs
                # Une modification plus fine nécessiterait une logique plus complexe
                
                blocks_updates = updates["blocks"]
                
                if "buy_blocks" in blocks_updates:
                    # Supprimer les blocs existants
                    signal_generator.buy_blocks = []
                    
                    # Ajouter les nouveaux blocs
                    for block_dict in blocks_updates["buy_blocks"]:
                        try:
                            block = Block.from_dict(block_dict)
                            signal_generator.add_block(block, is_buy=True)
                            updated = True
                        except Exception as e:
                            logger.error(f"Erreur lors de l'ajout d'un bloc d'achat: {str(e)}")
                
                if "sell_blocks" in blocks_updates:
                    # Supprimer les blocs existants
                    signal_generator.sell_blocks = []
                    
                    # Ajouter les nouveaux blocs
                    for block_dict in blocks_updates["sell_blocks"]:
                        try:
                            block = Block.from_dict(block_dict)
                            signal_generator.add_block(block, is_buy=False)
                            updated = True
                        except Exception as e:
                            logger.error(f"Erreur lors de l'ajout d'un bloc de vente: {str(e)}")
            
            # Si des mises à jour ont été effectuées, sauvegarder la stratégie
            if updated:
                # Si les performances ont été modifiées, les mettre à jour
                if "performance" in updates:
                    performance.update(updates["performance"])
                
                # Sauvegarder la stratégie modifiée
                if self.save_strategy(study_name, strategy_rank, signal_generator, position_calculator, performance):
                    logger.info(f"Stratégie {strategy_rank} de l'étude '{study_name}' modifiée avec succès")
                    return True
            else:
                logger.warning(f"Aucun paramètre n'a été mis à jour pour la stratégie {strategy_rank}")
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur lors de la modification de la stratégie: {str(e)}")
            traceback.print_exc()
            return False
    
    def save_backtest_results(self, study_name: str, strategy_rank: int, 
                            simulation_results: Dict, equity_curve: np.ndarray, 
                            trades_data: List[Dict], simulator: Optional[Simulator] = None) -> bool:
        """
        Sauvegarde les résultats de backtest pour une stratégie
        
        Args:
            study_name: Nom de l'étude
            strategy_rank: Rang de la stratégie
            simulation_results: Résultats de la simulation
            equity_curve: Courbe d'équité
            trades_data: Données des transactions
            simulator: Instance de Simulator (optionnel)
            
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        if not self.study_exists(study_name):
            logger.error(f"L'étude '{study_name}' n'existe pas")
            return False
        
        try:
            # Création du dossier de résultats
            results_dir = os.path.join(self.base_dir, study_name, "results", f"strategy_{strategy_rank}")
            os.makedirs(results_dir, exist_ok=True)
            
            # Sauvegarde des résultats de simulation
            results_file = os.path.join(results_dir, "simulation_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                # Convertir les données numpy en listes pour JSON
                clean_results = json.loads(json.dumps(simulation_results, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))
                json.dump(clean_results, f, indent=4, ensure_ascii=False)
            
            # Utiliser la méthode save_to_csv du Simulator si disponible
            if simulator is not None:
                base_filepath = os.path.join(results_dir, "simulation")
                simulator.save_to_csv(base_filepath)
                logger.info(f"Données de simulation sauvegardées avec Simulator.save_to_csv à {base_filepath}")
            else:
                # Fallback sur la méthode originale si simulator n'est pas disponible
                logger.warning("Instance de Simulator non disponible, utilisation de la méthode de sauvegarde par défaut")
                
                # Sauvegarde de la courbe d'équité en CSV
                equity_file = os.path.join(results_dir, "equity_curve.csv")
                pd.DataFrame(equity_curve, columns=["equity"]).to_csv(equity_file, index=True)
                
                # Sauvegarde des transactions en CSV
                trades_file = os.path.join(results_dir, "trades.csv")
                pd.DataFrame(trades_data).to_csv(trades_file, index=False)
            
            # Création de graphiques
            self._create_backtest_charts(results_dir, equity_curve, trades_data)
            
            logger.info(f"Résultats de backtest sauvegardés pour la stratégie {strategy_rank} de l'étude '{study_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats de backtest: {str(e)}")
            traceback.print_exc()
            return False
        
    def _create_backtest_charts(self, results_dir: str, equity_curve: np.ndarray, trades_data: List[Dict]):
        """
        Crée des graphiques pour les résultats de backtest
        
        Args:
            results_dir: Répertoire des résultats
            equity_curve: Courbe d'équité
            trades_data: Données des transactions
        """
        try:
            # Création du dossier pour les graphiques
            charts_dir = os.path.join(results_dir, "charts")
            os.makedirs(charts_dir, exist_ok=True)
            
            # Graphique de la courbe d'équité
            plt.figure(figsize=(12, 6))
            plt.plot(equity_curve, label="Equity")
            plt.title("Évolution du capital")
            plt.xlabel("Temps")
            plt.ylabel("Capital")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(charts_dir, "equity_curve.png"))
            plt.close()
            
            # Si des transactions sont disponibles
            if trades_data:
                # Convertir en DataFrame
                trades_df = pd.DataFrame(trades_data)
                
                # Distribution des profits/pertes
                if 'pnl' in trades_df.columns:
                    plt.figure(figsize=(12, 6))
                    trades_df['pnl'].hist(bins=20)
                    plt.title("Distribution des profits/pertes")
                    plt.xlabel("Profit/Perte")
                    plt.ylabel("Fréquence")
                    plt.grid(True)
                    plt.savefig(os.path.join(charts_dir, "pnl_distribution.png"))
                    plt.close()
                
                # Graphique de la durée des trades
                if 'duration' in trades_df.columns:
                    plt.figure(figsize=(12, 6))
                    trades_df['duration'].hist(bins=20)
                    plt.title("Distribution des durées de trade")
                    plt.xlabel("Durée (barres)")
                    plt.ylabel("Fréquence")
                    plt.grid(True)
                    plt.savefig(os.path.join(charts_dir, "duration_distribution.png"))
                    plt.close()
                
                # Graphique des rendements cumulés
                if 'pnl' in trades_df.columns:
                    plt.figure(figsize=(12, 6))
                    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
                    plt.plot(trades_df['cumulative_pnl'])
                    plt.title("Rendements cumulés")
                    plt.xlabel("Nombre de trades")
                    plt.ylabel("Profit/Perte cumulé")
                    plt.grid(True)
                    plt.savefig(os.path.join(charts_dir, "cumulative_pnl.png"))
                    plt.close()
            
        except Exception as e:
            logger.error(f"Erreur lors de la création des graphiques de backtest: {str(e)}")

    def run_backtest(self, study_name: str, strategy_rank: int, price_data: pd.DataFrame) -> Optional[Dict]:
        """
        Exécute un backtest sur une stratégie en utilisant la configuration centralisée
        
        Args:
            study_name: Nom de l'étude
            strategy_rank: Rang de la stratégie
            price_data: Données de prix
            
        Returns:
            Optional[Dict]: Résultats du backtest ou None en cas d'erreur
        """
        try:
            # Charger la stratégie
            strategy = self.load_strategy(study_name, strategy_rank)
            if not strategy:
                logger.error(f"Impossible de charger la stratégie {strategy_rank} de l'étude '{study_name}'")
                return None
            
            signal_generator, position_calculator, _ = strategy
            
            # Vérifier si la stratégie a des blocs de trading
            if not signal_generator.buy_blocks and not signal_generator.sell_blocks:
                logger.warning(f"La stratégie {strategy_rank} n'a pas de blocs de trading définis")
                # Créer des blocs vides mais valides pour éviter l'erreur
                empty_condition = Condition(
                    indicator1="EMA_10",
                    operator=Operator.GREATER,
                    indicator2="EMA_20"
                )
                empty_block = Block(conditions=[empty_condition], logic_operators=[])
                signal_generator.add_block(empty_block, is_buy=True)
            
            # Récupérer la configuration de trading
            trading_config = self.get_trading_config(study_name)
            
            # Préparation des données
            prices = price_data['close'].values
            high = price_data['high'].values if 'high' in price_data else None
            low = price_data['low'].values if 'low' in price_data else None
            volumes = price_data['volume'].values if 'volume' in price_data else None
            
            # Génération des signaux
            logger.info(f"Génération des signaux pour la stratégie {strategy_rank} de l'étude '{study_name}'")
            try:
                signals = signal_generator.generate_signals(prices=prices, high=high, low=low, volumes=volumes)
            except ValueError as e:
                if "fingerprint of empty list" in str(e):
                    logger.warning(f"Erreur de liste vide, génération de signaux nuls pour la stratégie {strategy_rank}")
                    signals = np.zeros(len(prices), dtype=np.int32)
                else:
                    raise
            
            # Calcul des paramètres de risque
            logger.info(f"Calcul des paramètres de risque pour la stratégie {strategy_rank}")
            position_sizes, sl_levels, tp_levels = position_calculator.calculate_risk_parameters(
                prices=prices, high=high, low=low
            )
            
            # Configuration de la simulation à partir de la configuration centralisée
            sim_config = convert_to_simulator_config(trading_config.sim_config)
            
            # Création du simulateur
            simulator = Simulator(config=sim_config)
            
            # Exécution de la simulation
            logger.info(f"Exécution de la simulation pour la stratégie {strategy_rank}")
            results = simulator.run(
                prices=prices,
                signals=signals,
                position_sizes=position_sizes,
                sl_levels=sl_levels,
                tp_levels=tp_levels
            )
            
            # Extraction des données pour la sauvegarde
            equity_curve = results.get('account_history', {}).get('equity', np.array([]))
            trades_data = results.get('trade_history', [])
            
            # Sauvegarde des résultats
            self.save_backtest_results(study_name, strategy_rank, results, equity_curve, trades_data, simulator)
            
            logger.info(f"Backtest terminé pour la stratégie {strategy_rank} de l'étude '{study_name}'")
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du backtest: {str(e)}")
            traceback.print_exc()
            return None
    
    def get_optimization_config(self, study_name: str) -> Optional[Dict]:
        """
        Récupère la configuration d'optimisation d'une étude
        
        Args:
            study_name: Nom de l'étude
            
        Returns:
            Optional[Dict]: Configuration d'optimisation ou None si non trouvée
        """
        if not self.study_exists(study_name):
            logger.error(f"L'étude '{study_name}' n'existe pas")
            return None
        
        try:
            # Chemin du fichier de configuration
            config_file = os.path.join(self.base_dir, study_name, "optimization", "optimization_config.json")
            
            if not os.path.exists(config_file):
                logger.warning(f"Configuration d'optimisation non trouvée pour l'étude '{study_name}'")
                return None
            
            # Chargement de la configuration
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Erreur lors de la lecture de la configuration d'optimisation: {str(e)}")
            return None
    
    def save_optimization_config(self, study_name: str, optimization_config: Dict) -> bool:
        """
        Sauvegarde la configuration d'optimisation pour une étude
        
        Args:
            study_name: Nom de l'étude
            optimization_config: Configuration d'optimisation
            
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        if not self.study_exists(study_name):
            logger.error(f"L'étude '{study_name}' n'existe pas")
            return False
        
        try:
            # Chemin du dossier d'optimisation
            optim_dir = os.path.join(self.base_dir, study_name, "optimization")
            os.makedirs(optim_dir, exist_ok=True)
            
            # Chemin du fichier de configuration
            config_file = os.path.join(optim_dir, "optimization_config.json")
            
            # Ajout de la date de modification
            optimization_config['last_modified'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Sauvegarde de la configuration
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(optimization_config, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Configuration d'optimisation sauvegardée pour l'étude '{study_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la configuration d'optimisation: {str(e)}")
            return False
    
    def get_optimization_results(self, study_name: str) -> Optional[Dict]:
        """
        Récupère les résultats d'optimisation d'une étude
        
        Args:
            study_name: Nom de l'étude
            
        Returns:
            Optional[Dict]: Résultats d'optimisation ou None si non trouvés
        """
        if not self.study_exists(study_name):
            logger.error(f"L'étude '{study_name}' n'existe pas")
            return None
        
        try:
            # Chemin du fichier de résultats
            results_file = os.path.join(self.base_dir, study_name, "optimization", "optimization_results.json")
            
            if not os.path.exists(results_file):
                logger.warning(f"Résultats d'optimisation non trouvés pour l'étude '{study_name}'")
                return None
            
            # Chargement des résultats
            with open(results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Erreur lors de la lecture des résultats d'optimisation: {str(e)}")
            return None
    
    def save_optimization_results(self, study_name: str, results: Dict) -> bool:
        """
        Sauvegarde les résultats d'optimisation pour une étude
        
        Args:
            study_name: Nom de l'étude
            results: Résultats d'optimisation
            
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        if not self.study_exists(study_name):
            logger.error(f"L'étude '{study_name}' n'existe pas")
            return False
        
        try:
            # Chemin du dossier d'optimisation
            optim_dir = os.path.join(self.base_dir, study_name, "optimization")
            os.makedirs(optim_dir, exist_ok=True)
            
            # Chemin du fichier de résultats
            results_file = os.path.join(optim_dir, "optimization_results.json")
            
            # Ajout de la date de génération
            results['generation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Sauvegarde des résultats
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Résultats d'optimisation sauvegardés pour l'étude '{study_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats d'optimisation: {str(e)}")
            return False
    
    def save_best_strategies(self, study_name: str, study: optuna.Study) -> bool:
        """
        Sauvegarde les meilleures stratégies d'une étude Optuna.
        
        Args:
            study_name: Nom de l'étude
            study: Étude Optuna
            
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        try:
            # Récupération des meilleurs trials
            best_trials = sorted(
                [t for t in study.trials if t.value is not None and t.value > float('-inf')],
                key=lambda t: t.value if t.value is not None else float('-inf'),
                reverse=True
            )[:10]  # Top 10
            
            if not best_trials:
                logger.error(f"Aucun trial valide trouvé pour l'étude '{study_name}'")
                return False
            
            # Récupération de la configuration de trading
            trading_config = self.get_trading_config(study_name)
            if trading_config is None:
                logger.error(f"Impossible de récupérer la configuration de trading pour l'étude '{study_name}'")
                return False
            
            # Sauvegarde des résultats d'optimisation
            optimization_results = {
                'study_name': study_name,
                'optimization_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'n_trials': len(study.trials),
                'best_trial_id': best_trials[0].number if best_trials else -1,
                'best_score': best_trials[0].value if best_trials else float('-inf'),
                'best_trials': [{
                    'trial_id': t.number,
                    'score': t.value,
                    'params': t.params,
                    'metrics': {
                        k: v for k, v in t.user_attrs.items()
                        if k in ['roi', 'win_rate', 'total_trades', 'max_drawdown', 'profit_factor', 'avg_profit']
                    }
                } for t in best_trials]
            }
            
            self.save_optimization_results(study_name, optimization_results)
            
            # Sauvegarde des meilleures stratégies
            max_strategies = min(5, len(best_trials))
            for i, trial in enumerate(best_trials[:max_strategies]):  # Enregistrer les 5 meilleures
                try:
                    # Création du gestionnaire de blocs
                    block_manager = BlockManager(DummyTrial(trial.params), trading_config)
                    buy_blocks, sell_blocks = block_manager.generate_blocks()
                    
                    # Vérifier si les blocs générés sont valides
                    if not buy_blocks and not sell_blocks:
                        logger.warning(f"Trial {trial.number} n'a pas généré de blocs valides. Ajout d'un bloc par défaut.")
                        # Ajouter un bloc par défaut
                        empty_condition = Condition(
                            indicator1="EMA_10",
                            operator=Operator.GREATER,
                            indicator2="EMA_20"
                        )
                        empty_block = Block(conditions=[empty_condition], logic_operators=[])
                        buy_blocks = [empty_block]
                    
                    # Création du gestionnaire de risque
                    risk_manager = RiskManager(DummyTrial(trial.params), trading_config)
                    
                    # Création du générateur de signaux
                    signal_generator = SignalGenerator()
                    
                    # Ajout des blocs
                    for block in buy_blocks:
                        signal_generator.add_block(block, is_buy=True)
                    
                    for block in sell_blocks:
                        signal_generator.add_block(block, is_buy=False)
                    
                    # Création du calculateur de position
                    position_calculator = PositionCalculator(
                        mode=risk_manager.risk_mode,
                        config=risk_manager.get_config()
                    )
                    
                    # Sauvegarde de la stratégie
                    rank = i + 1
                    
                    performance = {
                        'name': f'Optimized Strategy {rank}',
                        'source': 'Optimization',
                        'trial_id': trial.number,
                        'score': trial.value
                    }
                    
                    # Ajout des métriques
                    for key, value in trial.user_attrs.items():
                        if key in ['roi', 'win_rate', 'total_trades', 'max_drawdown', 'profit_factor', 'avg_profit']:
                            performance[key] = value
                    
                    # Conversion en pourcentages pour certaines métriques
                    for key in ['roi', 'win_rate', 'max_drawdown']:
                        if key in performance:
                            performance[f'{key}_pct'] = performance[key] * 100
                    
                    self.save_strategy(
                        study_name=study_name,
                        strategy_rank=rank,
                        signal_generator=signal_generator,
                        position_calculator=position_calculator,
                        performance=performance
                    )
                    
                    # Nettoyage
                    signal_generator.cleanup()
                except Exception as e:
                    logger.error(f"Erreur lors de la sauvegarde de la stratégie {i+1}: {str(e)}")
                    continue
            
            logger.info(f"Meilleures stratégies sauvegardées pour l'étude '{study_name}'")
            
            # Mise à jour du statut de l'étude
            self.update_study_status(study_name, "optimized")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des meilleures stratégies: {str(e)}")
            traceback.print_exc()
            return False
    
    def list_studies(self) -> List[Dict]:
        """
        Liste toutes les études disponibles
        
        Returns:
            List[Dict]: Liste des études avec leurs métadonnées de base
        """
        studies = []
        
        try:
            # Parcourir les dossiers d'études
            for study_name in os.listdir(self.base_dir):
                study_dir = os.path.join(self.base_dir, study_name)
                
                # Vérifier si c'est un dossier
                if os.path.isdir(study_dir):
                    # Charger les métadonnées
                    metadata_path = os.path.join(study_dir, "metadata.json")
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                                
                                # Compter le nombre de stratégies
                                strategies_dir = os.path.join(study_dir, "strategies")
                                n_strategies = 0
                                if os.path.exists(strategies_dir):
                                    n_strategies = len([f for f in os.listdir(strategies_dir) if f.endswith(".json")])
                                
                                # Vérifier si une optimisation a été effectuée
                                has_optimization = os.path.exists(
                                    os.path.join(study_dir, "optimization", "optimization_results.json")
                                )
                                
                                # Ajouter des informations de base
                                study_info = {
                                    'name': study_name,
                                    'status': metadata.get('status', 'unknown'),
                                    'created': metadata.get('creation_date', ''),
                                    'last_modified': metadata.get('last_modified', ''),
                                    'asset': metadata.get('asset', ''),
                                    'timeframe': metadata.get('timeframe', ''),
                                    'exchange': metadata.get('exchange', ''),
                                    'description': metadata.get('description', ''),
                                    'strategies_count': n_strategies,
                                    'has_optimization': has_optimization
                                }
                                
                                studies.append(study_info)
                        except:
                            # Ignorer les fichiers de métadonnées invalides
                            pass
            
            return sorted(studies, key=lambda x: x.get('last_modified', ''), reverse=True)
            
        except Exception as e:
            logger.error(f"Erreur lors du listage des études: {str(e)}")
            return []
    
    def list_strategies(self, study_name: str) -> List[Dict]:
        """
        Liste toutes les stratégies d'une étude
        
        Args:
            study_name: Nom de l'étude
            
        Returns:
            List[Dict]: Liste des stratégies avec leurs informations de base
        """
        if not self.study_exists(study_name):
            logger.error(f"L'étude '{study_name}' n'existe pas")
            return []
        
        strategies = []
        
        try:
            strategies_dir = os.path.join(self.base_dir, study_name, "strategies")
            if not os.path.exists(strategies_dir):
                return []
            
            # Parcourir les fichiers de stratégies
            for strategy_file in os.listdir(strategies_dir):
                if strategy_file.startswith("strategy_") and strategy_file.endswith(".json"):
                    try:
                        with open(os.path.join(strategies_dir, strategy_file), 'r', encoding='utf-8') as f:
                            strategy_data = json.load(f)
                            
                            # Extraire le rang
                            rank = int(strategy_file.split("_")[1].split(".")[0])
                            
                            # Vérifier si des résultats de backtest existent
                            has_backtest = os.path.exists(
                                os.path.join(self.base_dir, study_name, "results", f"strategy_{rank}")
                            )
                            
                            # Informations de base
                            strategy_info = {
                                'rank': rank,
                                'creation_date': strategy_data.get('creation_date', ''),
                                'name': strategy_data.get('performance', {}).get('name', f'Strategy {rank}'),
                                'performance': {},
                                'has_backtest': has_backtest
                            }
                            
                            # Ajouter les métriques de performance clés
                            perf = strategy_data.get('performance', {})
                            for key in ['roi', 'win_rate', 'max_drawdown', 'profit_factor', 'total_trades']:
                                if key in perf:
                                    strategy_info['performance'][key] = perf[key]
                            
                            strategies.append(strategy_info)
                    except:
                        # Ignorer les fichiers de stratégies invalides
                        pass
            
            return sorted(strategies, key=lambda x: x.get('rank', 0))
            
        except Exception as e:
            logger.error(f"Erreur lors du listage des stratégies: {str(e)}")
            return []
    
    def delete_study(self, study_name: str) -> bool:
        """
        Supprime une étude et tous ses fichiers associés
        
        Args:
            study_name: Nom de l'étude à supprimer
            
        Returns:
            bool: True si la suppression a réussi, False sinon
        """
        if not self.study_exists(study_name):
            logger.error(f"L'étude '{study_name}' n'existe pas")
            return False
        
        try:
            # Supprimer le dossier de l'étude
            study_dir = os.path.join(self.base_dir, study_name)
            shutil.rmtree(study_dir)
            
            # Supprimer le fichier de base de données Optuna si existant
            optuna_db = f"{study_name}_optimization.db"
            if os.path.exists(optuna_db):
                os.remove(optuna_db)
            
            logger.info(f"Étude '{study_name}' supprimée avec succès")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de l'étude '{study_name}': {str(e)}")
            return False
    
    def clone_study(self, study_name: str, new_name: str, clone_config: Dict = None) -> Optional[str]:
        """
        Clone une étude existante avec des modifications optionnelles
        
        Args:
            study_name: Nom de l'étude à cloner
            new_name: Nouveau nom pour l'étude clonée
            clone_config: Options de clonage (par défaut: tout cloner)
            
        Returns:
            Optional[str]: Nom de la nouvelle étude ou None en cas d'erreur
        """
        if not self.study_exists(study_name):
            logger.error(f"L'étude source '{study_name}' n'existe pas")
            return None
        
        if self.study_exists(new_name):
            logger.error(f"Une étude nommée '{new_name}' existe déjà")
            return None
        
        try:
            # Options de clonage par défaut
            default_clone_config = {
                'clone_metadata': True,
                'clone_trading_config': True,
                'clone_strategies': True,
                'clone_optimization_config': True,
                'clone_results': False  # Par défaut, ne pas cloner les résultats
            }
            
            # Fusion avec la configuration fournie
            if clone_config:
                for key, value in clone_config.items():
                    default_clone_config[key] = value
            
            # Création du répertoire de la nouvelle étude
            new_study_dir = os.path.join(self.base_dir, new_name)
            os.makedirs(new_study_dir)
            
            # Création des sous-répertoires
            os.makedirs(os.path.join(new_study_dir, "strategies"), exist_ok=True)
            os.makedirs(os.path.join(new_study_dir, "results"), exist_ok=True)
            os.makedirs(os.path.join(new_study_dir, "configs"), exist_ok=True)
            os.makedirs(os.path.join(new_study_dir, "optimization"), exist_ok=True)
            
            # Clonage des métadonnées
            if default_clone_config['clone_metadata']:
                metadata = self.get_study_metadata(study_name)
                if metadata:
                    # Mise à jour des métadonnées pour la nouvelle étude
                    metadata["creation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    metadata["status"] = "cloned"
                    metadata["cloned_from"] = study_name
                    metadata["name"] = new_name
                    
                    # Sauvegarde des métadonnées
                    metadata_path = os.path.join(new_study_dir, "metadata.json")
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=4, ensure_ascii=False)
            
            # Clonage de la configuration de trading
            if default_clone_config['clone_trading_config']:
                trading_config = self.get_trading_config(study_name)
                if trading_config:
                    # Sauvegarde de la configuration de trading
                    config_path = os.path.join(new_study_dir, "configs", "trading_config.json")
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(trading_config.to_dict(), f, indent=4, ensure_ascii=False)
            
            # Clonage des stratégies
            if default_clone_config['clone_strategies']:
                strategies_src = os.path.join(self.base_dir, study_name, "strategies")
                strategies_dest = os.path.join(new_study_dir, "strategies")
                
                if os.path.exists(strategies_src):
                    for strategy_file in os.listdir(strategies_src):
                        if strategy_file.startswith("strategy_") and strategy_file.endswith(".json"):
                            # Copie du fichier
                            shutil.copy(
                                os.path.join(strategies_src, strategy_file),
                                os.path.join(strategies_dest, strategy_file)
                            )
            
            # Clonage de la configuration d'optimisation
            if default_clone_config['clone_optimization_config']:
                optim_config = self.get_optimization_config(study_name)
                if optim_config:
                    # Mise à jour de la configuration d'optimisation
                    optim_config["creation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    optim_config["cloned_from"] = study_name
                    
                    # Sauvegarde de la configuration d'optimisation
                    self.save_optimization_config(new_name, optim_config)
            
            # Clonage des résultats (optionnel)
            if default_clone_config['clone_results']:
                results_src = os.path.join(self.base_dir, study_name, "results")
                results_dest = os.path.join(new_study_dir, "results")
                
                if os.path.exists(results_src):
                    # Copie récursive des résultats
                    for item in os.listdir(results_src):
                        src_path = os.path.join(results_src, item)
                        dest_path = os.path.join(results_dest, item)
                        
                        if os.path.isdir(src_path):
                            shutil.copytree(src_path, dest_path)
                        else:
                            shutil.copy(src_path, dest_path)
            
            logger.info(f"Étude '{study_name}' clonée avec succès vers '{new_name}'")
            return new_name
            
        except Exception as e:
            logger.error(f"Erreur lors du clonage de l'étude: {str(e)}")
            traceback.print_exc()
            
            # Nettoyage en cas d'erreur
            try:
                if os.path.exists(os.path.join(self.base_dir, new_name)):
                    shutil.rmtree(os.path.join(self.base_dir, new_name))
            except:
                pass
            
            return None
    
    def compare_strategies(self, study_name: str, strategy_ranks: List[int] = None) -> Optional[Dict]:
        """
        Compare plusieurs stratégies d'une étude
        
        Args:
            study_name: Nom de l'étude
            strategy_ranks: Liste des rangs des stratégies à comparer (par défaut, les 3 premières)
            
        Returns:
            Optional[Dict]: Résultats de la comparaison ou None en cas d'erreur
        """
        if not self.study_exists(study_name):
            logger.error(f"L'étude '{study_name}' n'existe pas")
            return None
        
        try:
            # Si aucun rang n'est spécifié, utiliser les 3 premières stratégies
            if strategy_ranks is None:
                strategies = self.list_strategies(study_name)
                strategy_ranks = [s['rank'] for s in strategies[:3]]
            
            # Charger les stratégies
            strategies = []
            for rank in strategy_ranks:
                strategy = self.load_strategy(study_name, rank)
                if strategy:
                    _, _, performance = strategy
                    strategies.append({
                        'rank': rank,
                        'performance': performance
                    })
            
            if not strategies:
                logger.error(f"Aucune stratégie trouvée pour l'étude '{study_name}'")
                return None
            
            # Comparaison des performances
            comparison = {
                'study_name': study_name,
                'strategies': strategies,
                'comparison_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Création d'un graphique de comparaison
            self._create_comparison_chart(study_name, strategies)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Erreur lors de la comparaison des stratégies: {str(e)}")
            traceback.print_exc()
            return None
    
    def _create_comparison_chart(self, study_name: str, strategies: List[Dict]):
        """
        Crée un graphique de comparaison des stratégies
        
        Args:
            study_name: Nom de l'étude
            strategies: Liste des stratégies avec leurs performances
        """
        try:
            # Création du dossier pour les graphiques
            comparison_dir = os.path.join(self.base_dir, study_name, "comparisons")
            os.makedirs(comparison_dir, exist_ok=True)
            
            # Préparation des données pour le graphique
            metrics = ['roi_pct', 'win_rate_pct', 'max_drawdown_pct', 'profit_factor']
            strategy_ranks = [s['rank'] for s in strategies]
            
            # Création d'un graphique pour chaque métrique
            for metric in metrics:
                metric_values = []
                for strategy in strategies:
                    if 'performance' in strategy and metric in strategy['performance']:
                        metric_values.append(strategy['performance'][metric])
                    else:
                        metric_values.append(0)  # Valeur par défaut
                
                # Graphique
                plt.figure(figsize=(10, 6))
                plt.bar(strategy_ranks, metric_values)
                plt.title(f"Comparaison: {metric}")
                plt.xlabel("Rang de la stratégie")
                plt.ylabel(metric)
                plt.grid(True, axis='y')
                plt.savefig(os.path.join(comparison_dir, f"comparison_{metric}.png"))
                plt.close()
            
            # Graphique de comparaison globale
            plt.figure(figsize=(12, 8))
            metrics_display = {
                'roi_pct': 'ROI (%)',
                'win_rate_pct': 'Win Rate (%)',
                'max_drawdown_pct': 'Max Drawdown (%)',
                'profit_factor': 'Profit Factor'
            }
            
            # Création du tableau de comparaison
            data = []
            for strategy in strategies:
                strategy_data = {}
                if 'performance' in strategy:
                    for metric in metrics:
                        if metric in strategy['performance']:
                            strategy_data[metrics_display[metric]] = strategy['performance'][metric]
                        else:
                            strategy_data[metrics_display[metric]] = 0
                data.append(strategy_data)
            
            comparison_df = pd.DataFrame(data, index=[f"Stratégie {rank}" for rank in strategy_ranks])
            
            try:
                # Heatmap pour la comparaison
                import seaborn as sns
                plt.figure(figsize=(12, 8))
                sns.heatmap(comparison_df, annot=True, cmap="YlGnBu", fmt=".2f")
                plt.title("Comparaison des performances")
                plt.tight_layout()
                plt.savefig(os.path.join(comparison_dir, "performance_heatmap.png"))
                plt.close()
            except ImportError:
                logger.warning("Seaborn non disponible, heatmap non générée")
            
            # Sauvegarde du DataFrame en CSV
            comparison_df.to_csv(os.path.join(comparison_dir, "comparison_metrics.csv"))
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du graphique de comparaison: {str(e)}")

# Classe artificielle pour les trials
class BlockManager:
    """Manages the generation of trading blocks based on trial parameters"""
    
    def __init__(self, trial, trading_config):
        self.trial = trial
        self.trading_config = trading_config
        self.structure_config = trading_config.strategy_structure
        self.available_indicators = trading_config.available_indicators
    
    def generate_blocks(self):
        """
        Generates buy and sell blocks based on trial parameters
        
        Returns:
            Tuple[List[Block], List[Block]]: (buy_blocks, sell_blocks)
        """
        # Determine the number of blocks to generate
        min_blocks = self.structure_config.min_blocks
        max_blocks = self.structure_config.max_blocks
        
        n_buy_blocks = self.trial.suggest_int("n_buy_blocks", min_blocks, max_blocks)
        n_sell_blocks = self.trial.suggest_int("n_sell_blocks", min_blocks, max_blocks)
        
        buy_blocks = self._generate_block_set(n_buy_blocks, "buy")
        sell_blocks = self._generate_block_set(n_sell_blocks, "sell")
        
        return buy_blocks, sell_blocks
    
    def _generate_block_set(self, n_blocks, prefix):
        """
        Generates a set of blocks with the given prefix
        
        Args:
            n_blocks: Number of blocks to generate
            prefix: Prefix for parameter names ("buy" or "sell")
            
        Returns:
            List[Block]: List of generated blocks
        """
        blocks = []
        
        for i in range(n_blocks):
            block_prefix = f"{prefix}_block_{i}"
            block = self._generate_single_block(block_prefix)
            blocks.append(block)
        
        return blocks
    
    def _generate_single_block(self, block_prefix):
        """
        Generates a single trading block
        
        Args:
            block_prefix: Prefix for parameter names
            
        Returns:
            Block: Generated block
        """
        min_conditions = self.structure_config.min_conditions_per_block
        max_conditions = self.structure_config.max_conditions_per_block
        
        # Determine number of conditions
        n_conditions = self.trial.suggest_int(f"{block_prefix}_n_conditions", min_conditions, max_conditions)
        
        # Generate conditions
        conditions = []
        logic_operators = []
        
        for j in range(n_conditions):
            cond_prefix = f"{block_prefix}_cond_{j}"
            condition = self._generate_condition(cond_prefix)
            conditions.append(condition)
            
            # Add logic operator if needed
            if j < n_conditions - 1:
                logic_op = self._get_logic_operator(f"{block_prefix}_logic_{j}")
                logic_operators.append(logic_op)
        
        return Block(conditions=conditions, logic_operators=logic_operators)
    
    def _generate_condition(self, cond_prefix):
        """
        Generates a single trading condition
        
        Args:
            cond_prefix: Prefix for parameter names
            
        Returns:
            Condition: Generated condition
        """
        # Choose indicators from available ones
        available_inds = list(self.available_indicators.keys())
        
        # First indicator
        ind1_type = self.trial.suggest_categorical(f"{cond_prefix}_ind1_type", available_inds)
        ind_config = self.available_indicators[ind1_type]
        
        min_period = ind_config.min_period
        max_period = ind_config.max_period
        step = ind_config.step
        
        period1 = self.trial.suggest_int(f"{cond_prefix}_period1", min_period, max_period, step=step)
        ind1 = f"{ind1_type}_{period1}"
        
        # Operator
        operators = [op.value for op in Operator]
        op = Operator(self.trial.suggest_categorical(f"{cond_prefix}_operator", operators))
        
        # Determine if we compare to another indicator or a value
        use_value = self.trial.suggest_float(f"{cond_prefix}_use_value", 0, 1) < self.structure_config.value_comparison_probability
        
        ind2 = None
        value = None
        
        if use_value:
            # Compare to a value
            if ind1_type == "RSI":
                # RSI values are between 0 and 100
                value = self.trial.suggest_float(f"{cond_prefix}_value", 20, 80)
            else:
                # Use a multiplier of current price for other indicators
                multiplier = self.trial.suggest_float(f"{cond_prefix}_multiplier", 0.5, 1.5)
                value = multiplier  # Will be multiplied by price at runtime
        else:
            # Compare to another indicator
            ind2_type = self.trial.suggest_categorical(f"{cond_prefix}_ind2_type", available_inds)
            ind_config2 = self.available_indicators[ind2_type]
            
            min_period2 = ind_config2.min_period
            max_period2 = ind_config2.max_period
            step2 = ind_config2.step
            
            period2 = self.trial.suggest_int(f"{cond_prefix}_period2", min_period2, max_period2, step=step2)
            ind2 = f"{ind2_type}_{period2}"
        
        return Condition(indicator1=ind1, operator=op, indicator2=ind2, value=value)
    
    def _get_logic_operator(self, param_name):
        """
        Gets a logic operator based on the trial parameter
        
        Args:
            param_name: Parameter name
            
        Returns:
            LogicOperator: AND or OR
        """
        use_or = self.trial.suggest_categorical(param_name, [0, 1])
        return LogicOperator.OR if use_or else LogicOperator.AND

class RiskManager:
    """Manages risk parameters based on trial parameters"""
    
    def __init__(self, trial, trading_config):
        self.trial = trial
        self.trading_config = trading_config
        self.risk_config = trading_config.risk_config
        
        # Choose risk mode from available modes
        available_modes = [mode.value for mode in self.risk_config.available_modes]
        chosen_mode = self.trial.suggest_categorical("risk_mode", available_modes)
        self.risk_mode = RiskMode(chosen_mode)
    
    def get_config(self):
        """
        Gets risk configuration based on the chosen risk mode
        
        Returns:
            Dict: Configuration for PositionCalculator
        """
        config = {}
        
        # Base configuration common to all modes
        base_position_range = self.risk_config.position_size_range
        base_sl_range = self.risk_config.sl_range
        tp_mult_range = self.risk_config.tp_multiplier_range
        
        config["base_position"] = self.trial.suggest_float("base_position", base_position_range[0], base_position_range[1], log=True)
        config["base_sl"] = self.trial.suggest_float("base_sl", base_sl_range[0], base_sl_range[1], log=True)
        config["tp_multiplier"] = self.trial.suggest_float("tp_multiplier", tp_mult_range[0], tp_mult_range[1])
        
        # Mode-specific configuration
        if self.risk_mode == RiskMode.FIXED:
            mode_config = self.risk_config.mode_configs.get(RiskMode.FIXED)
            if mode_config:
                fixed_pos_range = mode_config.fixed_position_range
                fixed_sl_range = mode_config.fixed_sl_range
                fixed_tp_range = mode_config.fixed_tp_range
                
                config["base_position"] = self.trial.suggest_float("fixed_position", fixed_pos_range[0], fixed_pos_range[1], log=True)
                config["base_sl"] = self.trial.suggest_float("fixed_sl", fixed_sl_range[0], fixed_sl_range[1], log=True)
                config["tp_multiplier"] = self.trial.suggest_float("fixed_tp_mult", fixed_tp_range[0], fixed_tp_range[1])
        
        elif self.risk_mode == RiskMode.ATR_BASED:
            mode_config = self.risk_config.mode_configs.get(RiskMode.ATR_BASED)
            if mode_config:
                atr_period_range = mode_config.atr_period_range
                atr_mult_range = mode_config.atr_multiplier_range
                
                config["atr_period"] = self.trial.suggest_int("atr_period", atr_period_range[0], atr_period_range[1])
                config["atr_multiplier"] = self.trial.suggest_float("atr_multiplier", atr_mult_range[0], atr_mult_range[1])
        
        elif self.risk_mode == RiskMode.VOLATILITY_BASED:
            mode_config = self.risk_config.mode_configs.get(RiskMode.VOLATILITY_BASED)
            if mode_config:
                vol_period_range = mode_config.vol_period_range
                vol_mult_range = mode_config.vol_multiplier_range
                
                config["vol_period"] = self.trial.suggest_int("vol_period", vol_period_range[0], vol_period_range[1])
                config["vol_multiplier"] = self.trial.suggest_float("vol_multiplier", vol_mult_range[0], vol_mult_range[1])
        
        return config

class DummyTrial:
    """Classe pour simuler un Trial Optuna à partir de paramètres existants"""
    
    def __init__(self, params: Dict):
        """
        Initialise un faux trial avec des paramètres existants.
        
        Args:
            params: Paramètres du trial
        """
        self.params = params
    
    def suggest_categorical(self, name: str, choices: List):
        """Retourne le paramètre existant ou le premier choix"""
        return self.params.get(name, choices[0])
    
    def suggest_int(self, name: str, low: int, high: int, step: int = 1):
        """Retourne le paramètre existant ou la valeur basse"""
        return self.params.get(name, low)
    
    def suggest_float(self, name: str, low: float, high: float, step: float = None, log: bool = False):
        """Retourne le paramètre existant ou la valeur basse"""
        return self.params.get(name, low)