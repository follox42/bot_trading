"""
Module centralisant toutes les opérations de base de données.
Responsable de l'interaction avec la base de données pour les études, stratégies,
backtests et optimisations.
"""
import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, Enum, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, scoped_session, Session

from core.study.study_config import StudyStatus

# Configuration du logger
logger = logging.getLogger(__name__)

# Base SQLAlchemy pour les modèles
Base = declarative_base()

# Modèles de base de données

class DBStudy(Base):
    """Modèle de base de données pour une étude de trading"""
    __tablename__ = "studies"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text)
    asset = Column(String(50), nullable=False)
    timeframe = Column(String(20), nullable=False)
    exchange = Column(String(50), nullable=False)
    creation_date = Column(DateTime, default=datetime.now)
    last_modified = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    status = Column(String(20), default=StudyStatus.CREATED.value)
    
    # Associations
    tags = relationship("DBTag", secondary="study_tags", back_populates="studies")
    strategies = relationship("DBStrategy", back_populates="study", cascade="all, delete-orphan")
    backtests = relationship("DBBacktest", back_populates="study", cascade="all, delete-orphan")
    optimizations = relationship("DBOptimization", back_populates="study", cascade="all, delete-orphan")
    
    # Configurations en JSON
    data_config = Column(JSON)
    trading_config = Column(JSON)
    simulation_config = Column(JSON)
    optimization_config = Column(JSON)
    search_space_config = Column(JSON)
    
    # Chemin vers l'étude (pour compatibilité)
    study_path = Column(String(255))

class DBTag(Base):
    """Modèle pour les tags associés aux études"""
    __tablename__ = "tags"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    
    # Associations
    studies = relationship("DBStudy", secondary="study_tags", back_populates="tags")

class DBStudyTag(Base):
    """Table d'association entre études et tags"""
    __tablename__ = "study_tags"
    
    study_id = Column(Integer, ForeignKey("studies.id"), primary_key=True)
    tag_id = Column(Integer, ForeignKey("tags.id"), primary_key=True)

class DBStrategy(Base):
    """Modèle pour les stratégies créées dans une étude"""
    __tablename__ = "strategies"
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(String(50), nullable=False, index=True)
    study_id = Column(Integer, ForeignKey("studies.id"))
    name = Column(String(255))
    description = Column(Text)
    creation_date = Column(DateTime, default=datetime.now)
    config = Column(JSON)
    
    # Associations
    study = relationship("DBStudy", back_populates="strategies")
    backtests = relationship("DBBacktest", back_populates="strategy", cascade="all, delete-orphan")

class DBBacktest(Base):
    """Modèle pour les résultats de backtest"""
    __tablename__ = "backtests"
    
    id = Column(Integer, primary_key=True)
    backtest_id = Column(String(50), nullable=False, index=True)
    study_id = Column(Integer, ForeignKey("studies.id"))
    strategy_id = Column(Integer, ForeignKey("strategies.id"))
    name = Column(String(255))
    date = Column(DateTime, default=datetime.now)
    results = Column(JSON)
    
    # Associations
    study = relationship("DBStudy", back_populates="backtests")
    strategy = relationship("DBStrategy", back_populates="backtests")

class DBOptimization(Base):
    """Modèle pour les résultats d'optimisation"""
    __tablename__ = "optimizations"
    
    id = Column(Integer, primary_key=True)
    study_id = Column(Integer, ForeignKey("studies.id"))
    optuna_study_name = Column(String(255), index=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    status = Column(String(20), default="running")
    n_trials = Column(Integer, default=0)
    completed_trials = Column(Integer, default=0)
    best_score = Column(Float)
    best_params = Column(JSON)
    config = Column(JSON)
    
    # Associations
    study = relationship("DBStudy", back_populates="optimizations")
    trials = relationship("DBTrial", back_populates="optimization", cascade="all, delete-orphan")

class DBTrial(Base):
    """Modèle pour les essais individuels d'optimisation"""
    __tablename__ = "trials"
    
    id = Column(Integer, primary_key=True)
    optimization_id = Column(Integer, ForeignKey("optimizations.id"))
    trial_id = Column(Integer)
    params = Column(JSON)
    score = Column(Float)
    metrics = Column(JSON)
    strategy_id = Column(String(50))
    backtest_id = Column(String(50))
    timestamp = Column(DateTime, default=datetime.now)
    
    # Associations
    optimization = relationship("DBOptimization", back_populates="trials")


class DBOperations:
    """
    Classe centralisant toutes les opérations de base de données.
    Fournit une abstraction pour les opérations CRUD sur toutes les entités
    liées aux études et optimisations.
    """
    
    def __init__(self, db_url: Optional[str] = None):
        """
        Initialise le gestionnaire d'opérations DB.
        
        Args:
            db_url: URL de connexion à la base de données (SQLAlchemy)
        """
        # Connexion à la base de données
        if db_url is None:
            db_path = os.path.join("studies", "trading.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            db_url = f"sqlite:///{db_path}"
        
        self.engine = create_engine(db_url)
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        
        # Création des tables si elles n'existent pas
        Base.metadata.create_all(self.engine)
    
    def get_session(self) -> Session:
        """
        Obtient une session de base de données.
        
        Returns:
            Session: Session SQLAlchemy
        """
        return self.Session()
    
    # ---------- OPÉRATIONS SUR LES ÉTUDES ----------
    
    def create_study(
        self, 
        name: str, 
        description: str, 
        asset: str, 
        timeframe: str, 
        exchange: str, 
        tags: List[str],
        data_config: Dict, 
        search_space_config: Dict,
        study_path: str
    ) -> Optional[Dict]:
        """
        Crée une nouvelle étude dans la base de données.
        
        Args:
            name: Nom de l'étude
            description: Description de l'étude
            asset: Actif étudié
            timeframe: Timeframe de l'étude
            exchange: Exchange utilisé
            tags: Tags associés à l'étude
            data_config: Configuration des données
            search_space_config: Espace de recherche
            study_path: Chemin vers le répertoire de l'étude
        
        Returns:
            Optional[Dict]: Étude créée ou None en cas d'erreur
        """
        session = self.get_session()
        try:
            # Vérifie si l'étude existe déjà
            existing = session.query(DBStudy).filter_by(name=name).first()
            if existing:
                logger.warning(f"L'étude '{name}' existe déjà")
                session.close()
                return None
            
            # Crée une nouvelle étude
            study = DBStudy(
                name=name,
                description=description,
                asset=asset,
                timeframe=timeframe,
                exchange=exchange,
                creation_date=datetime.now(),
                last_modified=datetime.now(),
                status=StudyStatus.CREATED.value,
                data_config=data_config,
                search_space_config=search_space_config,
                study_path=study_path
            )
            
            # Ajoute les tags
            for tag_name in tags:
                tag = session.query(DBTag).filter_by(name=tag_name).first()
                if not tag:
                    tag = DBTag(name=tag_name)
                    session.add(tag)
                study.tags.append(tag)
            
            session.add(study)
            session.commit()
            
            # Convertit l'étude en dictionnaire
            result = {
                "id": study.id,
                "name": study.name,
                "description": study.description,
                "asset": study.asset,
                "timeframe": study.timeframe,
                "exchange": study.exchange,
                "creation_date": study.creation_date.isoformat(),
                "last_modified": study.last_modified.isoformat(),
                "status": study.status,
                "tags": [tag.name for tag in study.tags],
                "data_config": study.data_config,
                "search_space_config": study.search_space_config,
                "study_path": study.study_path
            }
            
            logger.info(f"Étude '{name}' créée avec succès")
            return result
        
        except Exception as e:
            session.rollback()
            logger.error(f"Erreur lors de la création de l'étude '{name}': {str(e)}")
            return None
        
        finally:
            session.close()
    
    def get_study(self, study_name: str) -> Optional[Dict]:
        """
        Récupère une étude par son nom.
        
        Args:
            study_name: Nom de l'étude
        
        Returns:
            Optional[Dict]: Étude trouvée ou None
        """
        session = self.get_session()
        try:
            study = session.query(DBStudy).filter_by(name=study_name).first()
            if not study:
                logger.warning(f"L'étude '{study_name}' n'existe pas")
                return None
            
            # Convertit l'étude en dictionnaire
            result = {
                "id": study.id,
                "name": study.name,
                "description": study.description,
                "asset": study.asset,
                "timeframe": study.timeframe,
                "exchange": study.exchange,
                "creation_date": study.creation_date.isoformat(),
                "last_modified": study.last_modified.isoformat(),
                "status": study.status,
                "tags": [tag.name for tag in study.tags],
                "data_config": study.data_config,
                "trading_config": study.trading_config,
                "simulation_config": study.simulation_config,
                "optimization_config": study.optimization_config,
                "search_space_config": study.search_space_config,
                "study_path": study.study_path
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'étude '{study_name}': {str(e)}")
            return None
        
        finally:
            session.close()
    
    def delete_study(self, study_name: str) -> bool:
        """
        Supprime une étude et toutes ses données associées.
        
        Args:
            study_name: Nom de l'étude à supprimer
        
        Returns:
            bool: True si la suppression a réussi
        """
        session = self.get_session()
        try:
            study = session.query(DBStudy).filter_by(name=study_name).first()
            if not study:
                logger.warning(f"L'étude '{study_name}' n'existe pas")
                return False
            
            # Supprime l'étude (et toutes ses données associées par cascade)
            session.delete(study)
            session.commit()
            
            logger.info(f"Étude '{study_name}' supprimée avec succès")
            return True
        
        except Exception as e:
            session.rollback()
            logger.error(f"Erreur lors de la suppression de l'étude '{study_name}': {str(e)}")
            return False
        
        finally:
            session.close()
    
    def list_studies(self) -> List[Dict]:
        """
        Liste toutes les études disponibles.
        
        Returns:
            List[Dict]: Liste des études
        """
        session = self.get_session()
        try:
            studies = session.query(DBStudy).all()
            
            return [
                {
                    "name": study.name,
                    "description": study.description,
                    "asset": study.asset,
                    "timeframe": study.timeframe,
                    "exchange": study.exchange,
                    "creation_date": study.creation_date.isoformat(),
                    "last_modified": study.last_modified.isoformat(),
                    "status": study.status,
                    "tags": [tag.name for tag in study.tags],
                    "strategy_count": len(study.strategies),
                    "optimization_count": len(study.optimizations),
                    "study_path": study.study_path
                }
                for study in studies
            ]
        
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la liste des études: {str(e)}")
            return []
        
        finally:
            session.close()
    
    def update_study_status(self, study_name: str, status: str) -> bool:
        """
        Met à jour le statut d'une étude.
        
        Args:
            study_name: Nom de l'étude
            status: Nouveau statut
        
        Returns:
            bool: True si la mise à jour a réussi
        """
        session = self.get_session()
        try:
            study = session.query(DBStudy).filter_by(name=study_name).first()
            if not study:
                logger.warning(f"L'étude '{study_name}' n'existe pas")
                return False
            
            # Met à jour le statut
            study.status = status
            study.last_modified = datetime.now()
            session.commit()
            
            logger.info(f"Statut de l'étude '{study_name}' mis à jour: {status}")
            return True
        
        except Exception as e:
            session.rollback()
            logger.error(f"Erreur lors de la mise à jour du statut de l'étude: {str(e)}")
            return False
        
        finally:
            session.close()
    
    def update_study_search_space(self, study_name: str, search_space: Dict) -> bool:
        """
        Met à jour l'espace de recherche d'une étude.
        
        Args:
            study_name: Nom de l'étude
            search_space: Nouvel espace de recherche
        
        Returns:
            bool: True si la mise à jour a réussi
        """
        session = self.get_session()
        try:
            study = session.query(DBStudy).filter_by(name=study_name).first()
            if not study:
                logger.warning(f"L'étude '{study_name}' n'existe pas")
                return False
            
            # Met à jour l'espace de recherche
            study.search_space_config = search_space
            study.last_modified = datetime.now()
            session.commit()
            
            logger.info(f"Espace de recherche de l'étude '{study_name}' mis à jour")
            return True
        
        except Exception as e:
            session.rollback()
            logger.error(f"Erreur lors de la mise à jour de l'espace de recherche: {str(e)}")
            return False
        
        finally:
            session.close()
    
    def get_study_path(self, study_name: str) -> Optional[str]:
        """
        Récupère le chemin du répertoire d'une étude.
        
        Args:
            study_name: Nom de l'étude
        
        Returns:
            Optional[str]: Chemin du répertoire ou None
        """
        session = self.get_session()
        try:
            study = session.query(DBStudy).filter_by(name=study_name).first()
            if not study:
                logger.warning(f"L'étude '{study_name}' n'existe pas")
                return None
            
            return study.study_path
        
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du chemin de l'étude: {str(e)}")
            return None
        
        finally:
            session.close()
    
    # ---------- OPÉRATIONS SUR LES STRATÉGIES ----------
    
    def save_strategy(self, study_name: str, strategy_id: str, config: Dict) -> bool:
        """
        Sauvegarde une stratégie dans la base de données.
        
        Args:
            study_name: Nom de l'étude
            strategy_id: Identifiant de la stratégie
            config: Configuration de la stratégie
        
        Returns:
            bool: True si la sauvegarde a réussi
        """
        session = self.get_session()
        try:
            study = session.query(DBStudy).filter_by(name=study_name).first()
            if not study:
                logger.warning(f"L'étude '{study_name}' n'existe pas")
                return False
            
            # Vérifie si la stratégie existe déjà
            strategy = session.query(DBStrategy).filter_by(
                study_id=study.id, 
                strategy_id=strategy_id
            ).first()
            
            if strategy:
                # Met à jour la stratégie existante
                strategy.name = config.get("name", strategy.name)
                strategy.description = config.get("description", strategy.description)
                strategy.config = config
            else:
                # Crée une nouvelle stratégie
                strategy = DBStrategy(
                    study_id=study.id,
                    strategy_id=strategy_id,
                    name=config.get("name", f"Strategy {strategy_id}"),
                    description=config.get("description", ""),
                    creation_date=datetime.now(),
                    config=config
                )
                session.add(strategy)
            
            # Met à jour la date de dernière modification de l'étude
            study.last_modified = datetime.now()
            session.commit()
            
            logger.info(f"Stratégie '{strategy_id}' sauvegardée pour l'étude '{study_name}'")
            return True
        
        except Exception as e:
            session.rollback()
            logger.error(f"Erreur lors de la sauvegarde de la stratégie: {str(e)}")
            return False
        
        finally:
            session.close()
    
    def list_strategies(self, study_name: str) -> List[Dict]:
        """
        Liste toutes les stratégies d'une étude.
        
        Args:
            study_name: Nom de l'étude
        
        Returns:
            List[Dict]: Liste des stratégies
        """
        session = self.get_session()
        try:
            study = session.query(DBStudy).filter_by(name=study_name).first()
            if not study:
                logger.warning(f"L'étude '{study_name}' n'existe pas")
                return []
            
            return [
                {
                    "id": strategy.strategy_id,
                    "name": strategy.name,
                    "description": strategy.description,
                    "creation_date": strategy.creation_date.isoformat(),
                    "backtest_count": len(strategy.backtests)
                }
                for strategy in study.strategies
            ]
        
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des stratégies: {str(e)}")
            return []
        
        finally:
            session.close()
    
    # ---------- OPÉRATIONS SUR LES BACKTESTS ----------
    
    def save_backtest(
        self, 
        study_name: str, 
        strategy_id: str, 
        backtest_id: str, 
        results: Dict
    ) -> bool:
        """
        Sauvegarde les résultats d'un backtest.
        
        Args:
            study_name: Nom de l'étude
            strategy_id: Identifiant de la stratégie
            backtest_id: Identifiant du backtest
            results: Résultats du backtest
        
        Returns:
            bool: True si la sauvegarde a réussi
        """
        session = self.get_session()
        try:
            # Récupère l'étude
            study = session.query(DBStudy).filter_by(name=study_name).first()
            if not study:
                logger.warning(f"L'étude '{study_name}' n'existe pas")
                return False
            
            # Récupère la stratégie
            strategy = session.query(DBStrategy).filter_by(
                study_id=study.id, 
                strategy_id=strategy_id
            ).first()
            
            if not strategy:
                logger.warning(f"La stratégie '{strategy_id}' n'existe pas dans l'étude '{study_name}'")
                return False
            
            # Vérifie si le backtest existe déjà
            backtest = session.query(DBBacktest).filter_by(
                study_id=study.id,
                strategy_id=strategy.id,
                backtest_id=backtest_id
            ).first()
            
            if backtest:
                # Met à jour le backtest existant
                backtest.results = results
                backtest.date = datetime.now()
            else:
                # Crée un nouveau backtest
                backtest = DBBacktest(
                    study_id=study.id,
                    strategy_id=strategy.id,
                    backtest_id=backtest_id,
                    name=f"Backtest {backtest_id}",
                    date=datetime.now(),
                    results=results
                )
                session.add(backtest)
            
            # Met à jour la date de dernière modification de l'étude
            study.last_modified = datetime.now()
            study.status = StudyStatus.BACKTESTED.value
            session.commit()
            
            logger.info(f"Backtest '{backtest_id}' sauvegardé pour la stratégie '{strategy_id}'")
            return True
        
        except Exception as e:
            session.rollback()
            logger.error(f"Erreur lors de la sauvegarde du backtest: {str(e)}")
            return False
        
        finally:
            session.close()
    
    def list_backtests(self, study_name: str, strategy_id: Optional[str] = None) -> List[Dict]:
        """
        Liste tous les backtests d'une étude ou d'une stratégie.
        
        Args:
            study_name: Nom de l'étude
            strategy_id: Identifiant de la stratégie (optionnel)
        
        Returns:
            List[Dict]: Liste des backtests
        """
        session = self.get_session()
        try:
            study = session.query(DBStudy).filter_by(name=study_name).first()
            if not study:
                logger.warning(f"L'étude '{study_name}' n'existe pas")
                return []
            
            query = session.query(DBBacktest).filter(DBBacktest.study_id == study.id)
            
            if strategy_id:
                # Filtre sur la stratégie spécifique
                strategy = session.query(DBStrategy).filter_by(
                    study_id=study.id, 
                    strategy_id=strategy_id
                ).first()
                
                if not strategy:
                    logger.warning(f"La stratégie '{strategy_id}' n'existe pas dans l'étude '{study_name}'")
                    return []
                
                query = query.filter(DBBacktest.strategy_id == strategy.id)
            
            return [
                {
                    "id": backtest.backtest_id,
                    "name": backtest.name,
                    "date": backtest.date.isoformat(),
                    "strategy_id": session.query(DBStrategy.strategy_id).filter_by(id=backtest.strategy_id).scalar(),
                    "performance": backtest.results.get("performance", {}) if backtest.results else {}
                }
                for backtest in query.all()
            ]
        
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des backtests: {str(e)}")
            return []
        
        finally:
            session.close()
    
    # ---------- OPÉRATIONS SUR LES OPTIMISATIONS ----------
    
    def save_optimization(
        self, 
        study_name: str, 
        optuna_study_name: str, 
        config: Dict
    ) -> bool:
        """
        Sauvegarde une optimisation dans la base de données.
        
        Args:
            study_name: Nom de l'étude
            optuna_study_name: Nom de l'étude Optuna
            config: Configuration et résultats de l'optimisation
        
        Returns:
            bool: True si la sauvegarde a réussi
        """
        session = self.get_session()
        try:
            # Récupère l'étude
            study = session.query(DBStudy).filter_by(name=study_name).first()
            if not study:
                logger.warning(f"L'étude '{study_name}' n'existe pas")
                return False
            
            # Vérifie si l'optimisation existe déjà
            optimization = session.query(DBOptimization).filter_by(
                study_id=study.id,
                optuna_study_name=optuna_study_name
            ).first()
            
            # Extrait les informations de la configuration
            start_time = config.get("start_time", datetime.now().isoformat())
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)
                
            end_time = config.get("end_time", datetime.now().isoformat())
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time)
                
            status = config.get("status", "completed")
            n_trials = config.get("number_of_trials", 0)
            completed_trials = config.get("valid_trials", 0)
            best_score = config.get("best_score")
            best_params = config.get("best_params", {})
            
            if optimization:
                # Met à jour l'optimisation existante
                optimization.start_time = start_time
                optimization.end_time = end_time
                optimization.status = status
                optimization.n_trials = n_trials
                optimization.completed_trials = completed_trials
                optimization.best_score = best_score
                optimization.best_params = best_params
                optimization.config = config
            else:
                # Crée une nouvelle optimisation
                optimization = DBOptimization(
                    study_id=study.id,
                    optuna_study_name=optuna_study_name,
                    start_time=start_time,
                    end_time=end_time,
                    status=status,
                    n_trials=n_trials,
                    completed_trials=completed_trials,
                    best_score=best_score,
                    best_params=best_params,
                    config=config
                )
                session.add(optimization)
                session.flush()  # Pour obtenir l'ID
            
            # Sauvegarde les trials
            if "best_trials" in config:
                for trial_data in config["best_trials"]:
                    trial_id = trial_data.get("trial_id")
                    if trial_id is not None:
                        # Vérifie si le trial existe déjà
                        trial = session.query(DBTrial).filter_by(
                            optimization_id=optimization.id,
                            trial_id=trial_id
                        ).first()
                        
                        if not trial:
                            # Crée un nouveau trial
                            trial = DBTrial(
                                optimization_id=optimization.id,
                                trial_id=trial_id,
                                params=trial_data.get("params", {}),
                                score=trial_data.get("score"),
                                metrics=trial_data.get("metrics", {}),
                                strategy_id=trial_data.get("strategy_id"),
                                backtest_id=trial_data.get("backtest_id"),
                                timestamp=datetime.now()
                            )
                            session.add(trial)
            
            # Met à jour le statut de l'étude
            if status == "completed":
                study.status = StudyStatus.OPTIMIZED.value
            study.last_modified = datetime.now()
            
            session.commit()
            
            logger.info(f"Optimisation '{optuna_study_name}' sauvegardée pour l'étude '{study_name}'")
            return True
        
        except Exception as e:
            session.rollback()
            logger.error(f"Erreur lors de la sauvegarde de l'optimisation: {str(e)}")
            return False
        
        finally:
            session.close()
    
    def get_optimization(self, study_name: str, optuna_study_name: str) -> Optional[Dict]:
        """
        Récupère les résultats d'une optimisation.
        
        Args:
            study_name: Nom de l'étude
            optuna_study_name: Nom de l'étude Optuna
        
        Returns:
            Optional[Dict]: Résultats de l'optimisation ou None
        """
        session = self.get_session()
        try:
            # Récupère l'étude
            study = session.query(DBStudy).filter_by(name=study_name).first()
            if not study:
                logger.warning(f"L'étude '{study_name}' n'existe pas")
                return None
            
            # Récupère l'optimisation
            optimization = session.query(DBOptimization).filter_by(
                study_id=study.id,
                optuna_study_name=optuna_study_name
            ).first()
            
            if not optimization:
                logger.warning(f"Optimisation '{optuna_study_name}' non trouvée")
                return None
            
            return optimization.config
        
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'optimisation: {str(e)}")
            return None
        
        finally:
            session.close()
    
    def list_optimizations(self, study_name: str) -> List[Dict]:
        """
        Liste toutes les optimisations d'une étude.
        
        Args:
            study_name: Nom de l'étude
        
        Returns:
            List[Dict]: Liste des optimisations
        """
        session = self.get_session()
        try:
            # Récupère l'étude
            study = session.query(DBStudy).filter_by(name=study_name).first()
            if not study:
                logger.warning(f"L'étude '{study_name}' n'existe pas")
                return []
            
            return [
                {
                    "optuna_study_name": opt.optuna_study_name,
                    "start_time": opt.start_time.isoformat() if opt.start_time else None,
                    "end_time": opt.end_time.isoformat() if opt.end_time else None,
                    "status": opt.status,
                    "n_trials": opt.n_trials,
                    "completed_trials": opt.completed_trials,
                    "best_score": opt.best_score,
                    "trial_count": session.query(DBTrial).filter_by(optimization_id=opt.id).count()
                }
                for opt in study.optimizations
            ]
        
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des optimisations: {str(e)}")
            return []
        
        finally:
            session.close()
    
    def get_trial(self, study_name: str, trial_id: int) -> Optional[Dict]:
        """
        Récupère les informations d'un trial d'optimisation.
        
        Args:
            study_name: Nom de l'étude
            trial_id: Identifiant du trial
        
        Returns:
            Optional[Dict]: Informations sur le trial ou None
        """
        session = self.get_session()
        try:
            # Récupère l'étude
            study = session.query(DBStudy).filter_by(name=study_name).first()
            if not study:
                logger.warning(f"L'étude '{study_name}' n'existe pas")
                return None
            
            # Récupère le trial
            trial = (
                session.query(DBTrial)
                .join(DBOptimization)
                .filter(DBOptimization.study_id == study.id, DBTrial.trial_id == trial_id)
                .first()
            )
            
            if not trial:
                logger.warning(f"Trial {trial_id} non trouvé pour l'étude '{study_name}'")
                return None
            
            return {
                "trial_id": trial.trial_id,
                "params": trial.params,
                "score": trial.score,
                "metrics": trial.metrics,
                "strategy_id": trial.strategy_id,
                "backtest_id": trial.backtest_id,
                "timestamp": trial.timestamp.isoformat()
            }
        
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du trial: {str(e)}")
            return None
        
        finally:
            session.close()

# Création d'une instance du gestionnaire d'opérations DB
def create_db_operations(db_url: Optional[str] = None) -> DBOperations:
    """
    Crée une instance du gestionnaire d'opérations DB.
    
    Args:
        db_url: URL de connexion à la base de données (SQLAlchemy)
    
    Returns:
        DBOperations: Instance du gestionnaire
    """
    return DBOperations(db_url)