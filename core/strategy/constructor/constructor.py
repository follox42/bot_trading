"""
Constructeur de stratégies de trading qui permet de créer, configurer, sauvegarder 
et charger des stratégies complètes en combinant les indicateurs, conditions et gestion du risque.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import uuid
from datetime import datetime
import pickle
import warnings

# Import de nos modules de configuration et calcul
from core.strategy.constructor.constructor_config import StrategyConfig
from core.strategy.indicators.indicators_config import IndicatorType, IndicatorConfig, IndicatorsManager
from core.strategy.indicators.indicators import IndicatorCalculator
from core.strategy.conditions.conditions_config import (
    ConditionConfig, BlockConfig, StrategyBlocksConfig, 
    OperatorType, LogicOperatorType,
    PriceOperand, IndicatorOperand, ValueOperand
)
from core.strategy.conditions.conditions import ConditionEvaluator
from core.strategy.risk.risk_config import RiskConfig, RiskModeType
from core.strategy.risk.risk import RiskManager

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyConstructor:
    """
    Constructeur de stratégies qui facilite la création, sauvegarde et chargement.
    Utilise une architecture de configuration intégrée pour une meilleure cohérence.
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        """
        Initialise le constructeur de stratégies.
        
        Args:
            config: Configuration de la stratégie (optionnel)
        """
        # Initialiser la configuration
        self.config = config or StrategyConfig()
        
        # Initialisation des composants d'exécution à partir de la configuration
        self.indicators_calculator = IndicatorCalculator(self.config.indicators_manager)
        self.condition_evaluator = ConditionEvaluator(self.config.blocks_config)
        self.risk_manager = RiskManager(self.config.risk_config)
    
    def set_name(self, name: str):
        """
        Définit le nom de la stratégie.
        
        Args:
            name: Nouveau nom
        """
        self.config.name = name
        self.config.updated_at = datetime.now().isoformat()
    
    def set_description(self, description: str):
        """
        Définit la description de la stratégie.
        
        Args:
            description: Nouvelle description
        """
        self.config.description = description
        self.config.updated_at = datetime.now().isoformat()
    
    def add_tag(self, tag: str):
        """
        Ajoute un tag à la stratégie.
        
        Args:
            tag: Tag à ajouter
        """
        if tag not in self.config.tags:
            self.config.tags.append(tag)
            self.config.updated_at = datetime.now().isoformat()
    
    def remove_tag(self, tag: str):
        """
        Retire un tag de la stratégie.
        
        Args:
            tag: Tag à retirer
        """
        if tag in self.config.tags:
            self.config.tags.remove(tag)
            self.config.updated_at = datetime.now().isoformat()
    
    def set_parameter(self, key: str, value: Any):
        """
        Définit un paramètre personnalisé.
        
        Args:
            key: Clé du paramètre
            value: Valeur du paramètre
        """
        self.config.parameters[key] = value
        self.config.updated_at = datetime.now().isoformat()
    
    def add_indicator(self, name: str, indicator_config: IndicatorConfig):
        """
        Ajoute un indicateur à la stratégie.
        
        Args:
            name: Nom de l'indicateur
            indicator_config: Configuration de l'indicateur
        """
        self.config.indicators_manager.add_indicator(name, indicator_config)
        self.indicators_calculator.update_indicators_manager(self.config.indicators_manager)
        self.config.updated_at = datetime.now().isoformat()
    
    def remove_indicator(self, name: str):
        """
        Retire un indicateur de la stratégie.
        
        Args:
            name: Nom de l'indicateur à retirer
        """
        self.config.indicators_manager.remove_indicator(name)
        self.indicators_calculator.update_indicators_manager(self.config.indicators_manager)
        self.config.updated_at = datetime.now().isoformat()
    
    def set_blocks_config(self, blocks_config: StrategyBlocksConfig):
        """
        Définit la configuration des blocs de conditions.
        
        Args:
            blocks_config: Configuration des blocs
        """
        self.config.blocks_config = blocks_config
        self.condition_evaluator.update_blocks_config(blocks_config)
        self.config.updated_at = datetime.now().isoformat()
    
    def add_entry_block(self, block: BlockConfig):
        """
        Ajoute un bloc d'entrée.
        
        Args:
            block: Bloc de conditions
        """
        self.config.blocks_config.entry_blocks.append(block)
        self.condition_evaluator.update_blocks_config(self.config.blocks_config)
        self.config.updated_at = datetime.now().isoformat()
    
    def add_exit_block(self, block: BlockConfig):
        """
        Ajoute un bloc de sortie.
        
        Args:
            block: Bloc de conditions
        """
        self.config.blocks_config.exit_blocks.append(block)
        self.condition_evaluator.update_blocks_config(self.config.blocks_config)
        self.config.updated_at = datetime.now().isoformat()
    
    def add_filter_block(self, block: BlockConfig):
        """
        Ajoute un bloc de filtre.
        
        Args:
            block: Bloc de conditions
        """
        self.config.blocks_config.filter_blocks.append(block)
        self.condition_evaluator.update_blocks_config(self.config.blocks_config)
        self.config.updated_at = datetime.now().isoformat()
    
    def set_risk_config(self, risk_config: RiskConfig):
        """
        Définit la configuration de gestion du risque.
        
        Args:
            risk_config: Configuration du risque
        """
        self.config.risk_config = risk_config
        self.risk_manager.update_config(risk_config)
        self.config.updated_at = datetime.now().isoformat()
    
    def generate_signals(self, data: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Génère les signaux de trading pour les données fournies.
        
        Args:
            data: DataFrame avec les données OHLC
            
        Returns:
            Tuple[np.ndarray, pd.DataFrame]: (signaux, données avec indicateurs et signaux)
        """
        # Calculer les indicateurs
        data_with_indicators = self.indicators_calculator.calculate_indicators(data)
        
        # Évaluer les conditions et générer les signaux
        signals = self.condition_evaluator.evaluate_strategy(data_with_indicators)
        
        # Calculer les paramètres de risque
        data_with_risk = self.risk_manager.calculate_risk_params(data_with_indicators)
        
        # Ajouter les signaux aux données
        result = data_with_risk.copy()
        result['signal'] = signals
        
        return signals, result
    
    def save(self, filepath: str) -> bool:
        """
        Sauvegarde la stratégie dans un fichier JSON.
        
        Args:
            filepath: Chemin du fichier de sauvegarde
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        try:
            # Mettre à jour la date de modification
            self.config.updated_at = datetime.now().isoformat()
            
            # Utiliser la méthode save du StrategyConfig
            return self.config.save(filepath)
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la stratégie: {str(e)}")
            return False
    
    def save_pickle(self, filepath: str) -> bool:
        """
        Sauvegarde la stratégie complète (avec objets) dans un fichier pickle.
        
        Args:
            filepath: Chemin du fichier de sauvegarde
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        try:
            # Mettre à jour la date de modification
            self.config.updated_at = datetime.now().isoformat()
            
            # Créer le répertoire si nécessaire
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Sauvegarder en pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            
            logger.info(f"Stratégie '{self.config.name}' sauvegardée en pickle dans {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde pickle de la stratégie: {str(e)}")
            return False
    
    @classmethod
    def load(cls, filepath: str) -> 'StrategyConstructor':
        """
        Charge une stratégie depuis un fichier JSON.
        
        Args:
            filepath: Chemin du fichier de stratégie
            
        Returns:
            StrategyConstructor: Instance du constructeur de stratégie
        """
        try:
            # Charger la configuration
            config = StrategyConfig.load(filepath)
            
            # Créer le constructeur
            constructor = cls(config)
            
            logger.info(f"Stratégie '{config.name}' chargée depuis {filepath}")
            return constructor
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la stratégie: {str(e)}")
            raise ValueError(f"Impossible de charger la stratégie depuis {filepath}: {str(e)}")
    
    @classmethod
    def load_pickle(cls, filepath: str) -> 'StrategyConstructor':
        """
        Charge une stratégie complète depuis un fichier pickle.
        
        Args:
            filepath: Chemin du fichier pickle
            
        Returns:
            StrategyConstructor: Instance du constructeur de stratégie
        """
        try:
            # Charger le pickle
            with open(filepath, 'rb') as f:
                constructor = pickle.load(f)
            
            logger.info(f"Stratégie '{constructor.config.name}' chargée depuis pickle {filepath}")
            return constructor
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement pickle: {str(e)}")
            raise ValueError(f"Impossible de charger la stratégie pickle depuis {filepath}: {str(e)}")
    
    def clone(self) -> 'StrategyConstructor':
        """
        Clone la stratégie actuelle.
        
        Returns:
            StrategyConstructor: Clone de la stratégie actuelle
        """
        try:
            # Cloner la configuration
            config_clone = self.config.clone()
            
            # Créer un nouveau constructeur
            clone = StrategyConstructor(config_clone)
            
            return clone
            
        except Exception as e:
            logger.error(f"Erreur lors du clonage de la stratégie: {str(e)}")
            raise ValueError(f"Impossible de cloner la stratégie: {str(e)}")
    
    def apply_preset(self, preset_name: str) -> bool:
        """
        Applique un preset prédéfini à la stratégie.
        
        Args:
            preset_name: Nom du preset à appliquer
            
        Returns:
            bool: True si l'application du preset a réussi
        """
        try:
            from core.strategy.conditions.conditions_config import create_strategy_blocks_from_preset, CONDITION_BLOCK_PRESETS
            
            if preset_name not in CONDITION_BLOCK_PRESETS:
                logger.error(f"Preset '{preset_name}' non trouvé")
                return False
            
            # Appliquer le preset de blocs
            blocks_config = create_strategy_blocks_from_preset(preset_name)
            self.set_blocks_config(blocks_config)
            
            # Ajouter un tag correspondant au preset
            self.add_tag(f"preset:{preset_name}")
            
            logger.info(f"Preset '{preset_name}' appliqué à la stratégie")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'application du preset: {str(e)}")
            return False
    
    def get_summary(self) -> Dict:
        """
        Retourne un résumé de la stratégie.
        
        Returns:
            Dict: Résumé de la stratégie
        """
        return {
            "id": self.config.id,
            "name": self.config.name,
            "description": self.config.description,
            "version": self.config.version,
            "created_at": self.config.created_at,
            "updated_at": self.config.updated_at,
            "tags": self.config.tags,
            "indicators_count": len(self.config.indicators_manager.list_indicators()),
            "entry_blocks_count": len(self.config.blocks_config.entry_blocks),
            "exit_blocks_count": len(self.config.blocks_config.exit_blocks),
            "filter_blocks_count": len(self.config.blocks_config.filter_blocks),
            "risk_mode": self.config.risk_config.mode.value
        }


def create_strategy_from_presets(
    name: str,
    indicators_preset: Optional[str] = None,
    conditions_preset: Optional[str] = None,
    risk_preset: Optional[str] = None
) -> StrategyConstructor:
    """
    Crée une stratégie à partir de presets.
    
    Args:
        name: Nom de la stratégie
        indicators_preset: Preset d'indicateurs (optionnel)
        conditions_preset: Preset de conditions (optionnel)
        risk_preset: Preset de gestion du risque (optionnel)
        
    Returns:
        StrategyConstructor: Constructeur de stratégie configuré
    """
    constructor = StrategyConstructor()
    constructor.set_name(name)
    
    # Appliquer le preset d'indicateurs si spécifié
    if indicators_preset:
        from core.strategy.indicators.indicators_config import INDICATOR_PRESETS
        if indicators_preset in INDICATOR_PRESETS:
            for i, config in enumerate(INDICATOR_PRESETS[indicators_preset]):
                name = f"{config.type.value}_{i}"
                constructor.add_indicator(name, config)
            constructor.add_tag(f"indicators:{indicators_preset}")
        else:
            logger.warning(f"Preset d'indicateurs '{indicators_preset}' non trouvé")
    
    # Appliquer le preset de conditions si spécifié
    if conditions_preset:
        constructor.apply_preset(conditions_preset)
    
    # Appliquer le preset de gestion du risque si spécifié
    if risk_preset:
        from core.strategy.risk.risk_config import create_risk_config, RiskModeType
        
        # Format attendu: "mode:profile" (ex: "fixed:conservative")
        if ":" in risk_preset:
            mode, profile = risk_preset.split(":")
            try:
                risk_config = create_risk_config(mode, profile)
                constructor.set_risk_config(risk_config)
                constructor.add_tag(f"risk:{risk_preset}")
            except Exception as e:
                logger.warning(f"Erreur lors de l'application du preset de risque: {str(e)}")
        else:
            logger.warning(f"Format de preset de risque incorrect: {risk_preset}")
    
    return constructor


def test_strategy(strategy: StrategyConstructor, data: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Teste une stratégie sur des données et affiche les résultats.
    
    Args:
        strategy: Constructeur de stratégie
        data: DataFrame avec les données OHLC
        
    Returns:
        Tuple[np.ndarray, pd.DataFrame]: (signaux, données avec signaux)
    """
    try:
        # Générer les signaux
        signals, result = strategy.generate_signals(data)
        
        # Afficher des statistiques
        n_entries = (signals == 1).sum()
        n_exits = (signals == -1).sum()
        
        print(f"\n===== Test de la stratégie '{strategy.config.name}' =====")
        print(f"Données: {len(data)} points")
        print(f"Signaux d'entrée: {n_entries}")
        print(f"Signaux de sortie: {n_exits}")
        
        # Afficher un graphique des signaux
        try:
            from core.strategy.conditions.conditions import plot_signals
            plot_signals(result, title=f"Stratégie: {strategy.config.name}")
        except:
            print("Impossible d'afficher le graphique des signaux")
        
        return signals, result
        
    except Exception as e:
        logger.error(f"Erreur lors du test de la stratégie: {str(e)}")
        return np.array([]), pd.DataFrame()