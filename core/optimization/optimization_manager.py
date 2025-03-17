"""
Module de gestion de haut niveau pour l'optimisation des stratégies de trading.
Coordonne l'interaction entre les composants d'optimisation et l'infrastructure DB.
"""
import os
import json
import logging
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

from core.db_study.db_operations import create_db_operations, DBOperations
from core.optimization.search_config import SearchSpace
from core.optimization.optuna_optimizer import OptunaOptimizer
from core.optimization.optimization_worker import OptimizationWorker
from core.optimization.optimizer_task import OptimizerTask

logger = logging.getLogger(__name__)

class OptimizationManager:
    """
    Gestionnaire d'optimisation qui coordonne l'interaction entre
    les composants d'optimisation et l'infrastructure DB.
    """
    
    def __init__(self, db_url: Optional[str] = None):
        """
        Initialise le gestionnaire d'optimisation.
        
        Args:
            db_url: URL de connexion à la base de données (SQLAlchemy)
        """
        # Gestionnaire d'opérations DB
        self.db = create_db_operations(db_url)
        
        # État interne
        self.active_optimizations = {}
        self.current_optimizer = None
        self.current_worker = None
    
    def start_optimization(
        self,
        study_name: str,
        study_path: str,
        data_path: str,
        search_space: Union[Dict, SearchSpace],
        n_trials: int = 100,
        n_jobs: int = -1,
        timeout: Optional[int] = None,
        optimization_method: str = "tpe",
        method_params: Optional[Dict] = None,
        enable_pruning: bool = True,
        pruner_method: str = "median",
        pruner_params: Optional[Dict] = None,
        scoring_formula: str = "standard",
        min_trades: int = 10,
        memory_limit: float = 0.8,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict], None]] = None
    ) -> Dict[str, Any]:
        """
        Démarre une optimisation pour une étude donnée.
        
        Args:
            study_name: Nom de l'étude
            study_path: Chemin vers le répertoire de l'étude
            data_path: Chemin vers les données pour le backtest
            search_space: Espace de recherche
            n_trials: Nombre d'essais à effectuer
            n_jobs: Nombre de processus parallèles (-1 pour auto)
            timeout: Timeout en secondes
            optimization_method: Méthode d'optimisation
            method_params: Paramètres spécifiques à la méthode
            enable_pruning: Activer la suppression des essais peu prometteurs
            pruner_method: Méthode de pruning
            pruner_params: Paramètres spécifiques au pruner
            scoring_formula: Formule de scoring à utiliser
            min_trades: Nombre minimum de trades pour qu'un essai soit valide
            memory_limit: Limite de mémoire en pourcentage
            seed: Graine pour la reproductibilité
            progress_callback: Fonction de callback pour les mises à jour
        
        Returns:
            Dict[str, Any]: Informations sur l'optimisation démarrée
        """
        # Vérifier si l'étude existe
        study_info = self.db.get_study(study_name)
        if not study_info:
            logger.error(f"Étude '{study_name}' non trouvée")
            return {
                "status": "error",
                "error": f"Étude '{study_name}' non trouvée"
            }
        
        # Générer un nom unique pour l'étude Optuna
        optuna_study_name = f"study_{study_name}_opt_{int(time.time())}"
        
        # Configurer le stockage Optuna
        optuna_db_path = os.path.join(study_path, "optimizations", "optuna.db")
        os.makedirs(os.path.dirname(optuna_db_path), exist_ok=True)
        storage_url = f"sqlite:///{optuna_db_path}"
        
        try:
            # Créer l'optimiseur Optuna
            self.current_optimizer = OptunaOptimizer(
                study_name=optuna_study_name,
                search_space=search_space,
                storage=storage_url,
                optimization_method=optimization_method,
                method_params=method_params,
                enable_pruning=enable_pruning,
                pruner_method=pruner_method,
                pruner_params=pruner_params,
                seed=seed
            )
            
            # Créer le worker pour l'exécution parallèle
            self.current_worker = OptimizationWorker(
                n_jobs=n_jobs,
                memory_limit=memory_limit,
                timeout=timeout,
                progress_callback=progress_callback
            )
            
            # Créer les tâches d'optimisation
            tasks = self.current_optimizer.create_tasks(
                n_trials=n_trials,
                study_path=study_path,
                data_path=data_path,
                scoring_formula=scoring_formula,
                min_trades=min_trades
            )
            
            # Stocker les informations de l'optimisation active
            self.active_optimizations[optuna_study_name] = {
                "study_name": study_name,
                "optuna_study_name": optuna_study_name,
                "study_path": study_path,
                "start_time": datetime.now().isoformat(),
                "status": "running",
                "n_trials": n_trials,
                "n_jobs": n_jobs,
                "completed_trials": 0,
                "optimizer": self.current_optimizer,
                "worker": self.current_worker
            }
            
            # Exécuter les tâches en arrière-plan (non-bloquant)
            def run_optimization():
                try:
                    # Exécuter les tâches
                    results = self.current_worker.run_tasks(tasks)
                    
                    # Traiter les résultats avec Optuna
                    summary = self.current_optimizer.process_results(results)
                    
                    # Mettre à jour l'état de l'optimisation
                    self.active_optimizations[optuna_study_name].update({
                        "status": "completed",
                        "end_time": datetime.now().isoformat(),
                        "completed_trials": len(results),
                        "best_score": summary.get("best_score"),
                        "best_trial_id": summary.get("best_trial_id"),
                        "summary": summary
                    })
                    
                    # Sauvegarder les résultats dans la DB
                    self._save_optimization_results(
                        study_name=study_name,
                        optuna_study_name=optuna_study_name,
                        summary=summary
                    )
                    
                    logger.info(f"Optimisation '{optuna_study_name}' terminée avec succès")
                except Exception as e:
                    logger.error(f"Erreur lors de l'optimisation '{optuna_study_name}': {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
                    self.active_optimizations[optuna_study_name].update({
                        "status": "error",
                        "end_time": datetime.now().isoformat(),
                        "error": str(e)
                    })
            
            # Démarrer l'optimisation dans un thread séparé
            optimization_thread = threading.Thread(target=run_optimization)
            optimization_thread.daemon = True
            optimization_thread.start()
            
            return {
                "study_name": study_name,
                "optuna_study_name": optuna_study_name,
                "status": "running",
                "start_time": self.active_optimizations[optuna_study_name]["start_time"],
                "n_trials": n_trials,
                "n_jobs": n_jobs
            }
        
        except Exception as e:
            logger.error(f"Erreur lors du démarrage de l'optimisation: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "study_name": study_name,
                "optuna_study_name": optuna_study_name,
                "status": "error",
                "error": str(e)
            }
    
    def _save_optimization_results(
        self,
        study_name: str,
        optuna_study_name: str,
        summary: Dict[str, Any]
    ):
        """
        Sauvegarde les résultats d'une optimisation dans la base de données.
        
        Args:
            study_name: Nom de l'étude
            optuna_study_name: Nom de l'étude Optuna
            summary: Résumé des résultats
        """
        try:
            # Préparer la configuration complète
            optimization_info = self.active_optimizations.get(optuna_study_name, {})
            config = {
                "optuna_study_name": optuna_study_name,
                "start_time": optimization_info.get("start_time", datetime.now().isoformat()),
                "end_time": optimization_info.get("end_time", datetime.now().isoformat()),
                "status": optimization_info.get("status", "completed"),
                "n_trials": summary.get("number_of_trials", 0),
                "valid_trials": summary.get("valid_trials", 0),
                "best_trial_id": summary.get("best_trial_id"),
                "best_score": summary.get("best_score"),
                "best_params": summary.get("best_params"),
                "best_metrics": summary.get("best_metrics", {}),
                "best_trials": summary.get("best_trials", [])
            }
            
            # Sauvegarder dans la base de données
            self.db.save_optimization(study_name, optuna_study_name, config)
            
            # Sauvegarder également sur le disque (pour la rétrocompatibilité)
            study_path = optimization_info.get("study_path")
            if study_path:
                optim_dir = os.path.join(study_path, "optimizations")
                os.makedirs(optim_dir, exist_ok=True)
                results_path = os.path.join(optim_dir, f"{optuna_study_name}_results.json")
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
                
                # Sauvegarder également le dernier résultat
                latest_path = os.path.join(optim_dir, "latest_results.json")
                with open(latest_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Résultats d'optimisation '{optuna_study_name}' sauvegardés")
        
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats d'optimisation: {str(e)}")
    
    def stop_optimization(self, optuna_study_name: str) -> bool:
        """
        Arrête une optimisation en cours.
        
        Args:
            optuna_study_name: Nom de l'étude Optuna à arrêter
        
        Returns:
            bool: True si l'arrêt a réussi
        """
        if optuna_study_name not in self.active_optimizations:
            logger.warning(f"Optimisation '{optuna_study_name}' non trouvée")
            return False
        
        optimization_info = self.active_optimizations[optuna_study_name]
        if optimization_info["status"] in ["completed", "error", "stopped"]:
            logger.info(f"Optimisation '{optuna_study_name}' déjà terminée avec statut {optimization_info['status']}")
            return True
        
        try:
            worker = optimization_info.get("worker")
            if worker:
                worker.stop()
            
            optimization_info.update({
                "status": "stopped",
                "end_time": datetime.now().isoformat()
            })
            
            # Mettre à jour le statut dans la base de données
            study_name = optimization_info["study_name"]
            config = {
                "optuna_study_name": optuna_study_name,
                "status": "stopped",
                "end_time": datetime.now().isoformat()
            }
            self.db.save_optimization(study_name, optuna_study_name, config)
            
            logger.info(f"Optimisation '{optuna_study_name}' arrêtée avec succès")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt de l'optimisation '{optuna_study_name}': {str(e)}")
            return False
    
    def get_optimization_progress(self, optuna_study_name: str) -> Dict[str, Any]:
        """
        Récupère l'état d'avancement d'une optimisation.
        
        Args:
            optuna_study_name: Nom de l'étude Optuna
        
        Returns:
            Dict[str, Any]: État d'avancement
        """
        if optuna_study_name not in self.active_optimizations:
            # Essayer de charger depuis la base de données
            return self._load_optimization_from_db(optuna_study_name)
        
        optimization_info = self.active_optimizations[optuna_study_name]
        worker = optimization_info.get("worker")
        
        if worker:
            state = worker.get_state()
            
            return {
                "study_name": optimization_info["study_name"],
                "optuna_study_name": optuna_study_name,
                "status": optimization_info["status"],
                "start_time": optimization_info["start_time"],
                "end_time": optimization_info.get("end_time"),
                "n_trials": optimization_info["n_trials"],
                "completed_trials": state.get("completed", 0),
                "progress": state.get("progress", 0),
                "best_score": state.get("best_score"),
                "best_trial_id": state.get("best_trial_id"),
                "error": optimization_info.get("error")
            }
        else:
            return {
                "study_name": optimization_info["study_name"],
                "optuna_study_name": optuna_study_name,
                "status": optimization_info["status"],
                "start_time": optimization_info["start_time"],
                "end_time": optimization_info.get("end_time"),
                "n_trials": optimization_info["n_trials"],
                "completed_trials": optimization_info.get("completed_trials", 0),
                "progress": optimization_info.get("completed_trials", 0) / optimization_info["n_trials"],
                "best_score": optimization_info.get("best_score"),
                "best_trial_id": optimization_info.get("best_trial_id"),
                "error": optimization_info.get("error")
            }
    
    def _load_optimization_from_db(self, optuna_study_name: str) -> Dict[str, Any]:
        """
        Charge les informations d'une optimisation depuis la base de données.
        
        Args:
            optuna_study_name: Nom de l'étude Optuna
        
        Returns:
            Dict[str, Any]: Informations sur l'optimisation ou statut d'erreur
        """
        # Trouver l'étude associée à cette optimisation
        all_studies = self.db.list_studies()
        for study in all_studies:
            study_name = study.get("name")
            if study_name:
                # Lister les optimisations de cette étude
                optimizations = self.db.list_optimizations(study_name)
                for optimization in optimizations:
                    if optimization.get("optuna_study_name") == optuna_study_name:
                        # Charger les détails complets
                        config = self.db.get_optimization(study_name, optuna_study_name)
                        if config:
                            return {
                                "study_name": study_name,
                                "optuna_study_name": optuna_study_name,
                                "status": config.get("status", "unknown"),
                                "start_time": config.get("start_time"),
                                "end_time": config.get("end_time"),
                                "n_trials": config.get("n_trials", 0),
                                "completed_trials": config.get("valid_trials", 0),
                                "progress": 1.0,  # Terminé
                                "best_score": config.get("best_score"),
                                "best_trial_id": config.get("best_trial_id"),
                                "loaded_from_db": True
                            }
        
        return {
            "status": "unknown",
            "error": "Optimisation non trouvée",
            "optuna_study_name": optuna_study_name
        }
    
    def get_all_optimizations(self) -> List[Dict[str, Any]]:
        """
        Récupère l'état de toutes les optimisations actives.
        
        Returns:
            List[Dict[str, Any]]: Liste des états des optimisations
        """
        active_states = [
            self.get_optimization_progress(optuna_study_name)
            for optuna_study_name in self.active_optimizations
        ]
        
        # Charger également les optimisations terminées depuis la base de données
        completed_states = []
        all_studies = self.db.list_studies()
        for study in all_studies:
            study_name = study.get("name")
            if study_name:
                optimizations = self.db.list_optimizations(study_name)
                for optimization in optimizations:
                    optuna_study_name = optimization.get("optuna_study_name")
                    if optuna_study_name and optuna_study_name not in self.active_optimizations:
                        completed_states.append({
                            "study_name": study_name,
                            "optuna_study_name": optuna_study_name,
                            "status": optimization.get("status", "completed"),
                            "start_time": optimization.get("start_time"),
                            "end_time": optimization.get("end_time"),
                            "n_trials": optimization.get("n_trials", 0),
                            "completed_trials": optimization.get("completed_trials", 0),
                            "progress": 1.0,  # Terminé
                            "best_score": optimization.get("best_score"),
                            "loaded_from_db": True
                        })
        
        return active_states + completed_states
    
    def get_optimization_results(self, optuna_study_name: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les résultats complets d'une optimisation.
        
        Args:
            optuna_study_name: Nom de l'étude Optuna
        
        Returns:
            Optional[Dict[str, Any]]: Résultats de l'optimisation ou None
        """
        if optuna_study_name in self.active_optimizations:
            optimization_info = self.active_optimizations[optuna_study_name]
            if optimization_info["status"] in ["completed", "stopped"]:
                return optimization_info.get("summary")
            else:
                return self.get_optimization_progress(optuna_study_name)
        else:
            # Essayer de charger depuis la base de données
            return self._load_optimization_from_db(optuna_study_name)
    
    def get_trial_info(self, study_name: str, trial_id: int) -> Optional[Dict]:
        """
        Récupère les informations d'un trial spécifique.
        
        Args:
            study_name: Nom de l'étude
            trial_id: ID du trial
        
        Returns:
            Optional[Dict]: Informations sur le trial ou None
        """
        return self.db.get_trial(study_name, trial_id)

def create_optimization_manager(db_url: Optional[str] = None) -> OptimizationManager:
    """
    Crée une instance du gestionnaire d'optimisation.
    
    Args:
        db_url: URL de connexion à la base de données (facultatif)
    
    Returns:
        OptimizationManager: Instance du gestionnaire
    """
    return OptimizationManager(db_url)