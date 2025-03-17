"""
Module d'optimisation utilisant Optuna pour gérer les études d'optimisation.
Fournit une couche d'abstraction pour la configuration et la gestion des études Optuna.
"""
import os
import json
import logging
import time
import numpy as np
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import MedianPruner, PercentilePruner

from core.optimization.search_config import SearchSpace, SearchParameterRange
from core.optimization.optimizer_task import OptimizerTask

logger = logging.getLogger(__name__)

class OptunaOptimizer:
    """
    Gestionnaire d'optimisation utilisant Optuna pour explorer
    efficacement l'espace des paramètres.
    """
    
    def __init__(
        self,
        study_name: str,
        search_space: Union[Dict, SearchSpace],
        storage: Optional[str] = None,
        optimization_method: str = "tpe",
        method_params: Optional[Dict] = None,
        enable_pruning: bool = True,
        pruner_method: str = "median",
        pruner_params: Optional[Dict] = None,
        early_stopping_n_trials: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Initialise l'optimiseur Optuna.
        
        Args:
            study_name: Nom de l'étude Optuna
            search_space: Espace de recherche (dictionnaire ou objet SearchSpace)
            storage: URL de stockage Optuna (ex: "sqlite:///optuna.db")
            optimization_method: Méthode d'optimisation ("tpe", "random", etc.)
            method_params: Paramètres spécifiques à la méthode d'optimisation
            enable_pruning: Activer la suppression (pruning) des essais peu prometteurs
            pruner_method: Méthode de pruning ("median", "percentile", etc.)
            pruner_params: Paramètres spécifiques à la méthode de pruning
            early_stopping_n_trials: Nombre d'essais sans amélioration avant arrêt
            seed: Graine pour la reproductibilité
        """
        self.study_name = study_name
        self.search_space = search_space if isinstance(search_space, SearchSpace) else SearchSpace.from_dict(search_space)
        self.storage = storage
        self.optimization_method = optimization_method
        self.method_params = method_params or {}
        self.enable_pruning = enable_pruning
        self.pruner_method = pruner_method
        self.pruner_params = pruner_params or {}
        self.early_stopping_n_trials = early_stopping_n_trials
        self.seed = seed
        
        # État interne
        self.study = self._create_study()
        self.trials = []
        self.best_trial = None
    
    def _create_study(self) -> optuna.Study:
        """
        Crée une étude Optuna avec les paramètres spécifiés.
        
        Returns:
            optuna.Study: Étude Optuna configurée
        """
        # Configurer le sampler
        if self.optimization_method == "tpe":
            sampler_params = {
                "seed": self.seed
            }
            sampler_params.update(self.method_params)
            sampler = TPESampler(**sampler_params)
        elif self.optimization_method == "random":
            sampler_params = {
                "seed": self.seed
            }
            sampler_params.update(self.method_params)
            sampler = RandomSampler(**sampler_params)
        else:
            logger.warning(f"Méthode d'optimisation '{self.optimization_method}' non reconnue, utilisation de TPE")
            sampler = TPESampler(seed=self.seed)
        
        # Configurer le pruner
        if self.enable_pruning:
            if self.pruner_method == "median":
                pruner_params = {
                    "n_startup_trials": 10,
                    "n_warmup_steps": 5,
                    "interval_steps": 1
                }
                pruner_params.update(self.pruner_params)
                pruner = MedianPruner(**pruner_params)
            elif self.pruner_method == "percentile":
                pruner_params = {
                    "percentile": 75.0,
                    "n_startup_trials": 10,
                    "n_warmup_steps": 5,
                    "interval_steps": 1
                }
                pruner_params.update(self.pruner_params)
                pruner = PercentilePruner(**pruner_params)
            else:
                logger.warning(f"Méthode de pruning '{self.pruner_method}' non reconnue, utilisation de MedianPruner")
                pruner = MedianPruner()
        else:
            pruner = None
        
        # Créer ou charger l'étude
        try:
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                load_if_exists=True,
                direction="maximize",
                sampler=sampler,
                pruner=pruner
            )
            logger.info(f"Étude Optuna '{self.study_name}' chargée/créée avec succès")
            return study
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'étude Optuna: {str(e)}")
            # Créer une étude en mémoire comme fallback
            logger.info("Création d'une étude en mémoire comme solution de repli")
            return optuna.create_study(
                direction="maximize",
                sampler=sampler,
                pruner=pruner
            )
    
    def generate_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Génère les paramètres pour un trial Optuna à partir de l'espace de recherche.
        
        Args:
            trial: Trial Optuna
        
        Returns:
            Dict[str, Any]: Dictionnaire des paramètres suggérés
        """
        params = {}
        
        # Parcourir tous les paramètres de l'espace de recherche
        for category, param_dict in [
            ("parameters", self.search_space.parameters),
            ("indicators", self.search_space.indicators),
            ("longblock", self.search_space.longblock),
            ("shortblock", self.search_space.shortblock),
            ("risk", self.search_space.risk)
        ]:
            if category == "indicators":
                # Traitement spécial pour les indicateurs qui sont imbriqués
                for ind_name, ind_params in param_dict.items():
                    for param_name, param in ind_params.items():
                        full_name = f"{ind_name}_{param_name}"
                        value = self._suggest_parameter(trial, param, full_name)
                        if value is not None:
                            params[full_name] = value
            else:
                # Traitement standard pour les autres catégories
                for param_name, param in param_dict.items():
                    full_name = param_name
                    if category not in ["parameters", "risk"]:
                        full_name = f"{category}_{param_name}"
                    
                    value = self._suggest_parameter(trial, param, full_name)
                    if value is not None:
                        params[full_name] = value
        
        return params
    
    def _suggest_parameter(self, trial: optuna.Trial, param: SearchParameterRange, name: str) -> Optional[Any]:
        """
        Suggère une valeur pour un paramètre spécifique.
        
        Args:
            trial: Trial Optuna
            param: Configuration du paramètre
            name: Nom du paramètre
        
        Returns:
            Optional[Any]: Valeur suggérée ou None si le paramètre doit être ignoré
        """
        # Vérifier les conditions
        if param.condition:
            for cond_param, cond_value in param.condition.items():
                if not self._check_condition(trial, cond_param, cond_value):
                    return None
        
        # Suggérer la valeur selon le type
        if param.param_type == "int":
            return trial.suggest_int(
                name,
                param.min_value,
                param.max_value,
                step=param.step,
                log=param.log_scale
            )
        elif param.param_type == "float":
            return trial.suggest_float(
                name,
                param.min_value,
                param.max_value,
                step=param.step,
                log=param.log_scale
            )
        elif param.param_type == "categorical":
            return trial.suggest_categorical(name, param.choices)
        else:
            logger.warning(f"Type de paramètre non reconnu: {param.param_type}")
            return None
    
    def _check_condition(self, trial: optuna.Trial, param_name: str, expected_value: Any) -> bool:
        """
        Vérifie si une condition est satisfaite pour un paramètre.
        
        Args:
            trial: Trial Optuna
            param_name: Nom du paramètre conditionnel
            expected_value: Valeur attendue
        
        Returns:
            bool: True si la condition est satisfaite
        """
        # Vérifier si le paramètre existe déjà dans le trial
        param_value = None
        for trial_param in trial.params:
            if trial_param == param_name:
                param_value = trial.params[trial_param]
                break
        
        # Si le paramètre n'existe pas encore, essayer de l'ajouter
        if param_value is None:
            # Chercher le paramètre dans l'espace de recherche
            for category, param_dict in [
                ("parameters", self.search_space.parameters),
                ("indicators", self.search_space.indicators),
                ("longblock", self.search_space.longblock),
                ("shortblock", self.search_space.shortblock),
                ("risk", self.search_space.risk)
            ]:
                if category == "indicators":
                    # Cas spécial pour les indicateurs
                    for ind_name, ind_params in param_dict.items():
                        for param_name_inner, param_obj in ind_params.items():
                            full_name = f"{ind_name}_{param_name_inner}"
                            if full_name == param_name:
                                param_value = self._suggest_parameter(trial, param_obj, full_name)
                                break
                else:
                    # Cas standard
                    if param_name in param_dict:
                        full_name = param_name
                        if category not in ["parameters", "risk"]:
                            full_name = f"{category}_{param_name}"
                        param_value = self._suggest_parameter(trial, param_dict[param_name], full_name)
                        break
        
        # Vérifier la condition
        return param_value == expected_value
    
    def create_tasks(
        self,
        n_trials: int,
        study_path: str,
        data_path: str,
        scoring_formula: str = "standard",
        min_trades: int = 10,
        debug: bool = False
    ) -> List[OptimizerTask]:
        """
        Crée une liste de tâches d'optimisation à partir de l'espace de recherche.
        
        Args:
            n_trials: Nombre de trials à créer
            study_path: Chemin vers le répertoire de l'étude
            data_path: Chemin vers les données pour le backtest
            scoring_formula: Formule de scoring à utiliser
            min_trades: Nombre minimum de trades pour qu'un trial soit valide
            debug: Mode debug
        
        Returns:
            List[OptimizerTask]: Liste des tâches d'optimisation
        """
        tasks = []
        
        # Créer un trial pour chaque tâche
        for i in range(n_trials):
            task = OptimizerTask(
                trial_id=i,
                search_space=self.search_space,
                study_path=study_path,
                data_path=data_path,
                scoring_formula=scoring_formula,
                min_trades=min_trades,
                seed=self.seed + i if self.seed is not None else None,
                debug=debug
            )
            tasks.append(task)
        
        self.trials = tasks
        return tasks
    
    def process_results(self, results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Traite les résultats des tâches d'optimisation et met à jour l'étude Optuna.
        
        Args:
            results: Dictionnaire des résultats {ID de tâche: résultat}
        
        Returns:
            Dict[str, Any]: Résumé des résultats
        """
        # Extraire les résultats valides
        valid_results = {}
        best_score = float('-inf')
        best_trial_id = None
        
        for task_id, result in results.items():
            if result.get('status') == 'error':
                continue
            
            score = result.get('score', float('-inf'))
            if score != float('-inf'):
                valid_results[task_id] = result
                
                # Vérifier si c'est le meilleur score
                if score > best_score:
                    best_score = score
                    best_trial_id = task_id
        
        # Enregistrer les résultats dans Optuna
        for task_id, result in valid_results.items():
            try:
                # Créer un trial à partir de nos résultats
                trial = self.study.ask()
                trial_id = trial._trial_id
                
                # Enregistrer le résultat dans Optuna
                self.study.tell(trial_id, result['score'])
                
                # Enregistrer les paramètres dans Optuna
                for param_name, param_value in result['params'].items():
                    try:
                        self.study._storage.set_trial_param(
                            trial_id, 
                            param_name, 
                            param_value,
                            param_distribution_type="none"
                        )
                    except:
                        logger.warning(f"Impossible d'enregistrer le paramètre {param_name} pour le trial {trial_id}")
                
                # Enregistrer les métriques dans Optuna
                for metric_name, metric_value in result['metrics'].items():
                    try:
                        self.study._storage.set_trial_user_attr(
                            trial_id, 
                            f"metric_{metric_name}", 
                            metric_value
                        )
                    except:
                        logger.warning(f"Impossible d'enregistrer la métrique {metric_name} pour le trial {trial_id}")
                
                # Enregistrer les informations additionnelles
                try:
                    self.study._storage.set_trial_user_attr(
                        trial_id, 
                        "strategy_id", 
                        result.get('strategy_id', '')
                    )
                    self.study._storage.set_trial_user_attr(
                        trial_id, 
                        "backtest_id", 
                        result.get('backtest_id', '')
                    )
                    self.study._storage.set_trial_user_attr(
                        trial_id, 
                        "original_trial_id", 
                        task_id
                    )
                except:
                    logger.warning(f"Impossible d'enregistrer les attributs utilisateur pour le trial {trial_id}")
            
            except Exception as e:
                logger.error(f"Erreur lors de l'enregistrement du trial {task_id} dans Optuna: {str(e)}")
        
        # Préparer le résumé des résultats
        all_trials = self.study.trials
        if all_trials:
            best_trial = self.study.best_trial
            self.best_trial = best_trial
            
            summary = {
                "optuna_study_name": self.study_name,
                "number_of_trials": len(all_trials),
                "valid_trials": len(valid_results),
                "best_trial_id": best_trial_id,
                "best_score": best_score,
                "best_params": best_trial.params if best_trial else None,
                "best_metrics": {},
                "best_trials": []
            }
            
            # Extraire les métriques du meilleur trial
            if best_trial:
                for key, value in best_trial.user_attrs.items():
                    if key.startswith("metric_"):
                        metric_name = key[7:]  # Enlever le préfixe "metric_"
                        summary["best_metrics"][metric_name] = value
            
            # Extraire les meilleurs trials
            top_trials = sorted(
                all_trials, 
                key=lambda t: t.value or float('-inf'), 
                reverse=True
            )[:5]
            
            for t in top_trials:
                original_trial_id = t.user_attrs.get("original_trial_id")
                strategy_id = t.user_attrs.get("strategy_id", "")
                backtest_id = t.user_attrs.get("backtest_id", "")
                
                # Extraire les métriques
                metrics = {}
                for key, value in t.user_attrs.items():
                    if key.startswith("metric_"):
                        metric_name = key[7:]
                        metrics[metric_name] = value
                
                # Créer le résumé du trial
                trial_summary = {
                    "trial_id": original_trial_id,
                    "optuna_trial_id": t._trial_id,
                    "score": t.value,
                    "params": t.params,
                    "metrics": metrics,
                    "strategy_id": strategy_id,
                    "backtest_id": backtest_id
                }
                
                summary["best_trials"].append(trial_summary)
            
            return summary
        else:
            return {
                "optuna_study_name": self.study_name,
                "number_of_trials": 0,
                "valid_trials": 0,
                "best_trial_id": None,
                "best_score": None,
                "best_params": None,
                "best_metrics": {},
                "best_trials": []
            }