"""
Module de gestion du multiprocessing pour l'optimisation.
Permet d'exécuter plusieurs tâches d'optimisation en parallèle
et de gérer leur état d'avancement.
"""
import os
import json
import logging
import time
import traceback
import multiprocessing as mp
from multiprocessing import Pool, Process, Queue, Manager, Event
import psutil
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

from core.optimization.optimizer_task import OptimizerTask

logger = logging.getLogger(__name__)

class OptimizationWorker:
    """
    Gestionnaire de workers pour l'exécution parallèle des tâches d'optimisation.
    """
    
    def __init__(
        self,
        n_jobs: int = -1,
        memory_limit: float = 0.8,
        timeout: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict], None]] = None
    ):
        """
        Initialise le gestionnaire de workers.
        
        Args:
            n_jobs: Nombre de processus parallèles (-1 pour auto)
            memory_limit: Limite de mémoire en pourcentage
            timeout: Timeout en secondes pour l'optimisation
            progress_callback: Fonction de callback pour les mises à jour de progression
        """
        self.n_jobs = self._get_optimal_jobs(n_jobs)
        self.memory_limit = memory_limit
        self.timeout = timeout
        self.progress_callback = progress_callback
        
        # État interne partagé
        self.manager = Manager()
        self.message_queue = self.manager.Queue()
        self.stop_event = self.manager.Event()
        self.shared_state = self.manager.dict({
            'status': 'idle',
            'progress': 0,
            'completed': 0,
            'total': 0,
            'start_time': 0,
            'end_time': 0,
            'best_score': None,
            'best_trial_id': None,
            'messages': []
        })
        
        self.pool = None
        self.tasks = []
        self.results = {}
    
    def _get_optimal_jobs(self, n_jobs: int) -> int:
        """
        Détermine le nombre optimal de processus à utiliser.
        
        Args:
            n_jobs: Nombre de processus demandé (-1 pour auto)
        
        Returns:
            int: Nombre optimal de processus
        """
        if n_jobs < 1:
            # Utilise le nombre de CPU physiques - 1 (pour laisser un cœur libre)
            cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count()
            return max(1, cpu_count - 1)
        else:
            # Limite au nombre de CPU disponibles
            cpu_count = psutil.cpu_count()
            return min(n_jobs, cpu_count)
    
    def _task_wrapper(self, task_args):
        """
        Wrapper pour exécuter une tâche et gérer les exceptions.
        
        Args:
            task_args: Arguments pour la tâche
        
        Returns:
            Any: Résultat de la tâche
        """
        task, task_id, process_id = task_args
        
        try:
            # Vérifie si l'arrêt a été demandé
            if self.stop_event.is_set():
                return {"status": "stopped", "task_id": task_id}
            
            # Envoie un message de démarrage
            self.message_queue.put({
                "type": "start",
                "task_id": task_id,
                "process_id": process_id,
                "timestamp": time.time()
            })
            
            # Exécute la tâche
            result = task.run()
            
            # Envoie le résultat
            self.message_queue.put({
                "type": "result",
                "task_id": task_id,
                "process_id": process_id,
                "result": result,
                "timestamp": time.time()
            })
            
            return result
        
        except Exception as e:
            # Envoie l'erreur
            error_info = {
                "type": "error",
                "task_id": task_id,
                "process_id": process_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": time.time()
            }
            self.message_queue.put(error_info)
            logger.error(f"Erreur dans le processus {process_id}, tâche {task_id}: {str(e)}")
            
            return error_info
    
    def _message_processor(self):
        """
        Traite les messages de la queue et met à jour l'état partagé.
        """
        while not self.message_queue.empty():
            try:
                message = self.message_queue.get_nowait()
                
                if message["type"] == "start":
                    self._update_state_for_start(message)
                elif message["type"] == "result":
                    self._update_state_for_result(message)
                elif message["type"] == "error":
                    self._update_state_for_error(message)
                
                # Mettre à jour l'historique des messages
                messages = self.shared_state.get('messages', [])
                messages.append(message)
                if len(messages) > 100:  # Limiter le nombre de messages conservés
                    messages = messages[-100:]
                self.shared_state['messages'] = messages
                
                # Appeler le callback de progression si défini
                if self.progress_callback:
                    state_copy = dict(self.shared_state)
                    self.progress_callback(state_copy)
            
            except Exception as e:
                logger.error(f"Erreur lors du traitement des messages: {str(e)}")
    
    def _update_state_for_start(self, message):
        """
        Met à jour l'état partagé pour un message de début de tâche.
        
        Args:
            message: Message de début de tâche
        """
        task_id = message["task_id"]
        logger.debug(f"Tâche {task_id} démarrée sur le processus {message['process_id']}")
    
    def _update_state_for_result(self, message):
        """
        Met à jour l'état partagé pour un message de résultat.
        
        Args:
            message: Message de résultat
        """
        result = message["result"]
        task_id = message["task_id"]
        
        # Stocker le résultat
        self.results[task_id] = result
        
        # Mettre à jour le compteur de tâches terminées
        self.shared_state['completed'] = len(self.results)
        self.shared_state['progress'] = len(self.results) / max(1, self.shared_state['total'])
        
        # Vérifier si c'est le meilleur score
        if 'score' in result and (
            self.shared_state['best_score'] is None or 
            result['score'] > self.shared_state['best_score']
        ):
            self.shared_state['best_score'] = result['score']
            self.shared_state['best_trial_id'] = result.get('trial_id')
        
        logger.info(f"Tâche {task_id} terminée avec score: {result.get('score')}")
    
    def _update_state_for_error(self, message):
        """
        Met à jour l'état partagé pour un message d'erreur.
        
        Args:
            message: Message d'erreur
        """
        task_id = message["task_id"]
        
        # Stocker l'erreur
        self.results[task_id] = {
            "status": "error",
            "task_id": task_id,
            "error": message["error"],
            "traceback": message.get("traceback", "")
        }
        
        # Mettre à jour le compteur de tâches terminées
        self.shared_state['completed'] = len(self.results)
        self.shared_state['progress'] = len(self.results) / max(1, self.shared_state['total'])
        
        logger.error(f"Tâche {task_id} a échoué: {message['error']}")
    
    def run_tasks(self, tasks: List[OptimizerTask]) -> Dict[int, Any]:
        """
        Exécute une liste de tâches en parallèle.
        
        Args:
            tasks: Liste de tâches à exécuter
        
        Returns:
            Dict[int, Any]: Dictionnaire des résultats {ID de tâche: résultat}
        """
        if not tasks:
            logger.warning("Aucune tâche à exécuter")
            return {}
        
        # Réinitialiser l'état
        self.tasks = tasks
        self.results = {}
        self.stop_event.clear()
        
        # Mettre à jour l'état initial
        self.shared_state.update({
            'status': 'running',
            'progress': 0,
            'completed': 0,
            'total': len(tasks),
            'start_time': time.time(),
            'end_time': 0,
            'best_score': None,
            'best_trial_id': None,
            'messages': []
        })
        
        # Vider la queue de messages
        while not self.message_queue.empty():
            self.message_queue.get_nowait()
        
        try:
            # Créer les arguments pour les tâches
            task_args = [(task, i, i % self.n_jobs) for i, task in enumerate(tasks)]
            
            # Créer le pool si nécessaire
            if self.pool is None:
                self.pool = Pool(processes=self.n_jobs)
            
            # Exécuter les tâches en parallèle
            async_results = self.pool.map_async(self._task_wrapper, task_args)
            
            # Surveiller la progression et gérer le timeout
            start_time = time.time()
            while not async_results.ready():
                # Traiter les messages
                self._message_processor()
                
                # Vérifier si le timeout est atteint
                if self.timeout and time.time() - start_time > self.timeout:
                    logger.warning("Timeout atteint, arrêt des tâches")
                    self.stop_event.set()
                    break
                
                # Vérifier si un arrêt a été demandé
                if self.stop_event.is_set():
                    logger.info("Arrêt demandé, annulation des tâches")
                    break
                
                # Attendre un peu pour éviter de surcharger le CPU
                time.sleep(0.1)
            
            # Récupérer les résultats finaux
            if async_results.ready():
                # Traiter les derniers messages
                self._message_processor()
                
                # S'assurer que tous les résultats sont récupérés
                remaining_results = async_results.get(timeout=1)
                for i, result in enumerate(remaining_results):
                    if i not in self.results:
                        self.results[i] = result
            
            # Mettre à jour l'état final
            self.shared_state.update({
                'status': 'completed' if not self.stop_event.is_set() else 'stopped',
                'end_time': time.time(),
                'progress': len(self.results) / max(1, len(tasks))
            })
            
            return self.results
        
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution des tâches: {str(e)}")
            traceback.print_exc()
            
            self.shared_state.update({
                'status': 'error',
                'end_time': time.time()
            })
            
            return {
                -1: {
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            }
        
        finally:
            # Appeler le callback de progression final si défini
            if self.progress_callback:
                state_copy = dict(self.shared_state)
                self.progress_callback(state_copy)
    
    def stop(self):
        """
        Arrête l'exécution des tâches en cours.
        """
        self.stop_event.set()
        
        # Mettre à jour l'état
        self.shared_state.update({
            'status': 'stopped',
            'end_time': time.time()
        })
        
        logger.info("Arrêt des tâches demandé")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Récupère l'état actuel de l'optimisation.
        
        Returns:
            Dict[str, Any]: État actuel
        """
        return dict(self.shared_state)
    
    def __del__(self):
        """
        Libère les ressources lors de la destruction de l'objet.
        """
        if self.pool:
            self.pool.close()
            self.pool.join()