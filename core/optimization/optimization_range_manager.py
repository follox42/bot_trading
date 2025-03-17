"""
Module de gestion des configurations de recherche pour l'optimisation.
Permet de créer, sauvegarder et charger des configurations personnalisées d'espaces de recherche.
"""
import os
import json
import logging
import uuid
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from core.optimization.search_config import SearchSpace, get_predefined_search_space

logger = logging.getLogger(__name__)

class OptimizationRangeManager:
    """
    Gestionnaire des configurations d'espaces de recherche pour l'optimisation.
    Permet de gérer les différentes versions des configurations utilisées dans les optimisations.
    """
    
    def __init__(self, study_path: str):
        """
        Initialise le gestionnaire avec le chemin de l'étude.
        
        Args:
            study_path: Chemin vers le répertoire de l'étude
        """
        self.study_path = study_path
        self.optimization_dir = os.path.join(study_path, "optimizations")
        self.current_config = None
        
    def load_config(self, file_name: Optional[str] = None) -> Optional[SearchSpace]:
        """
        Charge une configuration spécifique ou la configuration par défaut.
        
        Args:
            config_id: Identifiant de la configuration (optionnel)
            
        Returns:
            Optional[SearchSpace]: L'espace de recherche chargé ou None
        """
        if file_name:
            config_path = os.path.join(self.optimization_dir, f"{file_name}.json")
            
            if not os.path.exists(config_path):
                logger.warning(f"Fichier de configuration introuvable: {config_path}")
                return self.get_default_config()
                
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                
                search_space = SearchSpace.from_dict(config_dict)
                self.current_config = search_space
                logger.info(f"Configuration chargée: {search_space.name}")
                return search_space
            except Exception as e:
                logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
                return self.get_default_config()
        else:
            return self.get_default_config()
    
    def save_config(self, search_space: SearchSpace, config_id: Optional[str] = None, 
                    make_default: bool = True) -> str:
        """
        Sauvegarde une configuration.
        
        Args:
            search_space: L'espace de recherche à sauvegarder
            config_id: Identifiant de la configuration (optionnel, généré sinon)
            make_default: Si True, définit cette configuration comme défaut
            
        Returns:
            str: Identifiant de la configuration sauvegardée
        """
        if not config_id:
            config_id = f"search_config_{str(uuid.uuid4())[:8]}"
            
        os.makedirs(self.configs_dir, exist_ok=True)
        config_path = os.path.join(self.configs_dir, f"{config_id}.json")
        
        try:
            config_dict = search_space.to_dict()
            
            # Ajouter des métadonnées
            config_dict["save_date"] = datetime.now().isoformat()
            config_dict["config_id"] = config_id
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
                
            # Sauvegarder également comme configuration par défaut si demandé
            if make_default:
                default_path = os.path.join(self.optimization_dir, "search_config.json")
                with open(default_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                
            self.current_config = search_space
            logger.info(f"Configuration '{search_space.name}' sauvegardée avec ID: {config_id}")
            return config_id
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la configuration: {str(e)}")
            return ""
    
    def create_config(self, name: str, description: str = "", preset_name: Optional[str] = None) -> SearchSpace:
        """
        Crée une nouvelle configuration.
        
        Args:
            name: Nom de la configuration
            description: Description de la configuration
            preset_name: Nom du preset à utiliser (optionnel)
            
        Returns:
            SearchSpace: La configuration créée
        """
        if preset_name:
            search_space = get_predefined_search_space(preset_name)
            search_space.name = name
            search_space.description = description
        else:
            search_space = get_predefined_search_space("default")
            search_space.name = name
            search_space.description = description
            
        return search_space
    
    def list_configs(self) -> List[Dict[str, Any]]:
        """
        Liste toutes les configurations sauvegardées.
        
        Returns:
            List[Dict[str, Any]]: Liste des informations sur les configurations
        """
        configs = []
        if not os.path.exists(self.optimization_dir):
            return configs
            
        for filename in os.listdir(self.optimization_dir):
            if filename.endswith(".json"):
                config_id = filename.replace(".json", "")
                filepath = os.path.join(self.optimization_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        config_dict = json.load(f)
                    
                    configs.append({
                        "id": config_id,
                        "name": config_dict.get("name", "Unnamed"),
                        "description": config_dict.get("description", ""),
                        "save_date": config_dict.get("save_date", "Unknown")
                    })
                except Exception as e:
                    logger.warning(f"Erreur lors de la lecture de {filepath}: {str(e)}")
        
        return configs
    
    def delete_config(self, config_id: str) -> bool:
        """
        Supprime une configuration.
        
        Args:
            config_id: Identifiant de la configuration à supprimer
            
        Returns:
            bool: True si la suppression a réussi
        """
        config_path = os.path.join(self.configs_dir, f"{config_id}.json")
        
        if not os.path.exists(config_path):
            logger.warning(f"Fichier de configuration introuvable: {config_path}")
            return False
            
        try:
            os.remove(config_path)
            logger.info(f"Configuration supprimée: {config_id}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de la configuration: {str(e)}")
            return False
    
    def get_default_config(self) -> Optional[SearchSpace]:
        """
        Charge la configuration par défaut.
        
        Returns:
            Optional[SearchSpace]: La configuration par défaut ou None
        """
        # Chercher d'abord dans le dossier d'optimisation
        default_path = os.path.join(self.optimization_dir, "search_config.json")
        
        if os.path.exists(default_path):
            try:
                with open(default_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                
                search_space = SearchSpace.from_dict(config_dict)
                logger.info(f"Configuration par défaut chargée: {search_space.name}")
                return search_space
            except Exception as e:
                logger.error(f"Erreur lors du chargement de la configuration par défaut: {str(e)}")
        
        # Chercher ensuite dans la configuration de l'étude
        try:
            study_config_path = os.path.join(self.study_path, "config.json")
            
            if os.path.exists(study_config_path):
                with open(study_config_path, 'r', encoding='utf-8') as f:
                    study_config = json.load(f)
                
                if "search_space_config" in study_config and study_config["search_space_config"]:
                    search_space = SearchSpace.from_dict(study_config["search_space_config"])
                    logger.info(f"Configuration de l'étude chargée: {search_space.name}")
                    return search_space
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration de l'étude: {str(e)}")
        
        # Utiliser la configuration par défaut si rien trouvé
        logger.info("Aucune configuration trouvée, utilisation du preset 'default'")
        return get_predefined_search_space("default")

def create_optimization_range_manager(study_path: str) -> OptimizationRangeManager:
    """
    Crée un gestionnaire de configurations d'optimisation pour une étude.
    
    Args:
        study_path: Chemin vers le répertoire de l'étude
        
    Returns:
        OptimizationRangeManager: Gestionnaire de configurations
    """
    return OptimizationRangeManager(study_path)