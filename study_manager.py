"""
Study Manager Module

Handles the creation, storage, and management of Optuna studies with metadata
for trading strategy optimization.
"""

import os
import json
import sqlite3
import optuna
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass, field, asdict
import logging
from enum import Enum
import uuid

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('study_manager')


class StudyEvaluationMetric(Enum):
    """Types of evaluation metrics for studies"""
    COMBINED_SCORE = "combined_score"        # Custom combined score
    ROI = "roi"                              # Return on investment
    ROI_TO_DRAWDOWN = "roi_to_drawdown"      # ROI/Drawdown ratio
    SHARPE_RATIO = "sharpe_ratio"            # Sharpe ratio
    SORTINO_RATIO = "sortino_ratio"          # Sortino ratio
    CALMAR_RATIO = "calmar_ratio"            # Calmar ratio
    WIN_RATE = "win_rate"                    # Trade win rate
    PROFIT_FACTOR = "profit_factor"          # Profit factor


class StudyStatus(Enum):
    """Status of an optimization study"""
    IN_PROGRESS = "in_progress"      # Optimization in progress
    COMPLETED = "completed"          # Optimization completed successfully
    FAILED = "failed"                # Optimization failed
    ARCHIVED = "archived"            # Study archived (not active)


@dataclass
class ScoreWeights:
    """Weights for calculating the combined score"""
    roi: float = 2.5                # ROI is crucial
    win_rate: float = 0.5           # Important but not decisive
    drawdown: float = 2.0           # Capital protection
    profit_factor: float = 2.0      # Consistency
    trade_freq: float = 1.0         # Sufficient volume
    avg_profit: float = 1.0         # Trade quality
    
    def to_dict(self) -> Dict:
        """Convert weights to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, weights_dict: Dict) -> 'ScoreWeights':
        """Create weights from dictionary"""
        return cls(**weights_dict)
    
    def recalculate_score(self, metrics: Dict[str, float]) -> float:
        """
        Recalculate score based on current weights and metrics
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            float: Updated score
        """
        # Extract metrics with fallbacks to 0
        roi = metrics.get('roi', 0.0)
        win_rate = metrics.get('win_rate', 0.0) 
        max_drawdown = abs(metrics.get('max_drawdown', 1.0))
        profit_factor = metrics.get('profit_factor', 0.0)
        total_trades = metrics.get('total_trades', 0)
        avg_profit = metrics.get('avg_profit_per_trade', 0.0)
        
        # Normalize ROI (assuming reasonable range is -1 to 2)
        roi_score = np.log1p(max(0, roi * 100)) / np.log1p(100)
        
        # Normalize win rate with sigmoid curve centered at 55%
        win_score = 1 / (1 + np.exp(-0.1 * (win_rate * 100 - 55)))
        
        # Convert drawdown to positive score (lower is better)
        dd_score = max(0, 1 - max_drawdown)
        
        # Normalize profit factor logarithmically
        pf_score = np.log1p(max(0, profit_factor - 1)) / np.log1p(2)
        
        # Normalize trade frequency 
        trade_score = min(1.0, total_trades / 300) if total_trades > 0 else 0
        
        # Normalize average profit per trade
        avg_profit_score = np.tanh(avg_profit * 100)
        
        # Calculate weighted score
        weighted_sum = (
            self.roi * roi_score +
            self.win_rate * win_score +
            self.drawdown * dd_score +
            self.profit_factor * pf_score +
            self.trade_freq * trade_score +
            self.avg_profit * avg_profit_score
        )
        
        # Normalize by sum of weights
        total_weight = (
            self.roi + self.win_rate + self.drawdown + 
            self.profit_factor + self.trade_freq + self.avg_profit
        )
        
        return weighted_sum / total_weight


@dataclass
class StudyMetadata:
    """
    Metadata for an optimization study
    """
    study_name: str
    asset: str
    timeframe: str
    data_path: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    evaluation_metric: StudyEvaluationMetric = StudyEvaluationMetric.COMBINED_SCORE
    description: str = ""
    config: Optional[Dict] = None
    tags: List[str] = field(default_factory=list)
    score_weights: ScoreWeights = field(default_factory=ScoreWeights)
    
    def __post_init__(self):
        """Initialize additional attributes after initialization"""
        self.creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_modified = self.creation_date
        self.status = StudyStatus.IN_PROGRESS
        
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary"""
        return {
            "study_name": self.study_name,
            "asset": self.asset,
            "timeframe": self.timeframe,
            "data_path": self.data_path,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "evaluation_metric": self.evaluation_metric.value,
            "description": self.description,
            "config": self.config,
            "tags": self.tags,
            "creation_date": self.creation_date,
            "last_modified": self.last_modified,
            "status": self.status.value,
            "score_weights": self.score_weights.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StudyMetadata':
        """Create metadata from dictionary"""
        # Extract score weights if present
        score_weights_dict = data.pop('score_weights', {})
        score_weights = ScoreWeights.from_dict(score_weights_dict) if score_weights_dict else ScoreWeights()
        
        # Create instance with basic fields
        metadata = cls(
            study_name=data.get("study_name", ""),
            asset=data.get("asset", ""),
            timeframe=data.get("timeframe", ""),
            data_path=data.get("data_path", ""),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            evaluation_metric=StudyEvaluationMetric(data.get("evaluation_metric", 
                                                 StudyEvaluationMetric.COMBINED_SCORE.value)),
            description=data.get("description", ""),
            config=data.get("config", {}),
            tags=data.get("tags", []),
            score_weights=score_weights
        )
        
        # Set additional attributes
        metadata.creation_date = data.get("creation_date", metadata.creation_date)
        metadata.last_modified = data.get("last_modified", metadata.last_modified)
        metadata.status = StudyStatus(data.get("status", StudyStatus.IN_PROGRESS.value))
        
        return metadata


@dataclass
class StudyPerformance:
    """
    Summary of performance metrics for an optimization study
    """
    study_name: str
    best_value: float = 0.0
    roi: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    avg_trade_duration: float = 0.0
    avg_profit_per_trade: float = 0.0
    best_params: Dict = field(default_factory=dict)
    metrics_history: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize update timestamp"""
        self.update_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def to_dict(self) -> Dict:
        """Convert performance to dictionary"""
        return {
            "study_name": self.study_name,
            "best_value": self.best_value,
            "roi": self.roi,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "total_trades": self.total_trades,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "avg_trade_duration": self.avg_trade_duration,
            "avg_profit_per_trade": self.avg_profit_per_trade,
            "best_params": self.best_params,
            "metrics_history": self.metrics_history,
            "update_timestamp": self.update_timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StudyPerformance':
        """Create performance summary from dictionary"""
        performance = cls(
            study_name=data.get("study_name", ""),
            best_value=data.get("best_value", 0.0),
            roi=data.get("roi", 0.0),
            win_rate=data.get("win_rate", 0.0),
            max_drawdown=data.get("max_drawdown", 0.0),
            total_trades=data.get("total_trades", 0),
            profit_factor=data.get("profit_factor", 0.0),
            sharpe_ratio=data.get("sharpe_ratio", 0.0),
            sortino_ratio=data.get("sortino_ratio", 0.0),
            calmar_ratio=data.get("calmar_ratio", 0.0),
            avg_trade_duration=data.get("avg_trade_duration", 0.0),
            avg_profit_per_trade=data.get("avg_profit_per_trade", 0.0),
            best_params=data.get("best_params", {}),
            metrics_history=data.get("metrics_history", {})
        )
        
        # Set additional attributes
        performance.update_timestamp = data.get("update_timestamp", performance.update_timestamp)
        
        return performance
        
    def update_from_optuna_study(self, study: optuna.Study) -> None:
        """
        Update performance metrics from an Optuna study
        
        Args:
            study: Completed Optuna study
        """
        if not study.trials:
            return
            
        # Best value
        self.best_value = study.best_value
        
        # Best parameters
        self.best_params = study.best_params
        
        # Retrieve user attributes from the best trial
        best_trial = study.best_trial
        
        if hasattr(best_trial, 'user_attrs'):
            # Retrieve performance metrics
            self.roi = best_trial.user_attrs.get('roi', self.roi)
            self.win_rate = best_trial.user_attrs.get('win_rate', self.win_rate)
            self.max_drawdown = best_trial.user_attrs.get('max_drawdown', self.max_drawdown)
            self.total_trades = best_trial.user_attrs.get('total_trades', self.total_trades)
            self.profit_factor = best_trial.user_attrs.get('profit_factor', self.profit_factor)
            self.sharpe_ratio = best_trial.user_attrs.get('sharpe_ratio', self.sharpe_ratio)
            self.sortino_ratio = best_trial.user_attrs.get('sortino_ratio', self.sortino_ratio)
            self.calmar_ratio = best_trial.user_attrs.get('calmar_ratio', self.calmar_ratio)
            self.avg_trade_duration = best_trial.user_attrs.get('avg_trade_duration', self.avg_trade_duration)
            self.avg_profit_per_trade = best_trial.user_attrs.get('avg_profit_per_trade', self.avg_profit_per_trade)
        
        # Update timestamp
        self.update_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Collect metrics history from all trials
        self.collect_metrics_history(study)
    
    def collect_metrics_history(self, study: optuna.Study) -> None:
        """
        Collect metrics history for all trials
        
        Args:
            study: Completed Optuna study
        """
        # Initialize lists for each metric
        metrics = {
            'trial_number': [],
            'value': [],
            'roi': [],
            'win_rate': [],
            'max_drawdown': [],
            'total_trades': [],
            'profit_factor': []
        }
        
        # Process all valid trials
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
                metrics['trial_number'].append(trial.number)
                metrics['value'].append(trial.value)
                
                # Retrieve user attributes
                if hasattr(trial, 'user_attrs'):
                    metrics['roi'].append(trial.user_attrs.get('roi', 0.0))
                    metrics['win_rate'].append(trial.user_attrs.get('win_rate', 0.0))
                    metrics['max_drawdown'].append(trial.user_attrs.get('max_drawdown', 0.0))
                    metrics['total_trades'].append(trial.user_attrs.get('total_trades', 0))
                    metrics['profit_factor'].append(trial.user_attrs.get('profit_factor', 0.0))
        
        # Store history
        self.metrics_history = metrics


class StudyManager:
    """
    Manager for creating, storing, and retrieving optimization studies
    """
    
    def __init__(self, database_path: str = "studies.db"):
        """
        Initialize study manager
        
        Args:
            database_path: Path to SQLite database file
        """
        self.database_path = database_path
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database with required tables"""
        # Create directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(self.database_path)), exist_ok=True)
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Metadata table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS study_metadata (
            study_name TEXT PRIMARY KEY,
            metadata_json TEXT NOT NULL,
            creation_date TEXT NOT NULL,
            last_modified TEXT NOT NULL,
            status TEXT NOT NULL
        )
        ''')
        
        # Performance table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS study_performance (
            study_name TEXT PRIMARY KEY,
            performance_json TEXT NOT NULL,
            update_timestamp TEXT NOT NULL,
            FOREIGN KEY (study_name) REFERENCES study_metadata (study_name)
        )
        ''')
        
        # Tags table for easy search
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS study_tags (
            study_name TEXT,
            tag TEXT,
            PRIMARY KEY (study_name, tag),
            FOREIGN KEY (study_name) REFERENCES study_metadata (study_name)
        )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized at {self.database_path}")
    
    def create_study(self, metadata: StudyMetadata, storage_url: Optional[str] = None) -> Tuple[bool, Optional[optuna.Study]]:
        """
        Create a new study in the database
        
        Args:
            metadata: Study metadata
            storage_url: Optional custom Optuna storage URL
            
        Returns:
            Tuple[bool, Optional[optuna.Study]]: Success status and created study
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # Check if study already exists
            cursor.execute("SELECT study_name FROM study_metadata WHERE study_name = ?", 
                          (metadata.study_name,))
            if cursor.fetchone():
                logger.warning(f"Study '{metadata.study_name}' already exists.")
                return False, None
            
            # Insert metadata
            cursor.execute(
                "INSERT INTO study_metadata (study_name, metadata_json, creation_date, last_modified, status) VALUES (?, ?, ?, ?, ?)",
                (
                    metadata.study_name,
                    json.dumps(metadata.to_dict()),
                    metadata.creation_date,
                    metadata.last_modified,
                    metadata.status.value
                )
            )
            
            # Insert tags
            for tag in metadata.tags:
                cursor.execute(
                    "INSERT INTO study_tags (study_name, tag) VALUES (?, ?)",
                    (metadata.study_name, tag)
                )
            
            conn.commit()
            
            # Create Optuna study
            if storage_url is None:
                # Use SQLite file in the same directory as the database
                db_dir = os.path.dirname(os.path.abspath(self.database_path))
                study_db = os.path.join(db_dir, f"{metadata.study_name}.db")
                storage_url = f"sqlite:///{study_db}"
            
            study = optuna.create_study(
                study_name=metadata.study_name,
                storage=storage_url,
                direction="maximize",
                load_if_exists=True
            )
            
            logger.info(f"Study '{metadata.study_name}' created successfully.")
            return True, study
            
        except Exception as e:
            logger.error(f"Error creating study: {e}")
            conn.rollback()
            return False, None
            
        finally:
            conn.close()
    
    def update_study_metadata(self, metadata: StudyMetadata) -> bool:
        """
        Update metadata for an existing study
        
        Args:
            metadata: Updated metadata
            
        Returns:
            bool: True if update was successful
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # Check if study exists
            cursor.execute("SELECT study_name FROM study_metadata WHERE study_name = ?", 
                          (metadata.study_name,))
            if not cursor.fetchone():
                logger.warning(f"Study '{metadata.study_name}' does not exist.")
                return False
            
            # Update timestamp
            metadata.last_modified = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Update metadata
            cursor.execute(
                "UPDATE study_metadata SET metadata_json = ?, last_modified = ?, status = ? WHERE study_name = ?",
                (
                    json.dumps(metadata.to_dict()),
                    metadata.last_modified,
                    metadata.status.value,
                    metadata.study_name
                )
            )
            
            # Update tags (delete and reinsert)
            cursor.execute("DELETE FROM study_tags WHERE study_name = ?", (metadata.study_name,))
            for tag in metadata.tags:
                cursor.execute(
                    "INSERT INTO study_tags (study_name, tag) VALUES (?, ?)",
                    (metadata.study_name, tag)
                )
            
            conn.commit()
            logger.info(f"Metadata for study '{metadata.study_name}' updated.")
            return True
            
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
            conn.rollback()
            return False
            
        finally:
            conn.close()
    
    def update_study_status(self, study_name: str, status: StudyStatus) -> bool:
        """
        Update the status of an existing study
        
        Args:
            study_name: Name of the study
            status: New status
            
        Returns:
            bool: True if update was successful
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # Retrieve current metadata
            cursor.execute("SELECT metadata_json FROM study_metadata WHERE study_name = ?", 
                          (study_name,))
            metadata_json = cursor.fetchone()
            
            if not metadata_json:
                logger.warning(f"Study '{study_name}' does not exist.")
                return False
            
            # Update metadata
            metadata = StudyMetadata.from_dict(json.loads(metadata_json[0]))
            metadata.status = status
            metadata.last_modified = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save updated metadata
            cursor.execute(
                "UPDATE study_metadata SET metadata_json = ?, last_modified = ?, status = ? WHERE study_name = ?",
                (
                    json.dumps(metadata.to_dict()),
                    metadata.last_modified,
                    status.value,
                    study_name
                )
            )
            
            conn.commit()
            logger.info(f"Status of study '{study_name}' updated to {status.value}.")
            return True
            
        except Exception as e:
            logger.error(f"Error updating study status: {e}")
            conn.rollback()
            return False
            
        finally:
            conn.close()
    
    def update_study_performance(self, performance: StudyPerformance) -> bool:
        """
        Update or create performance data for a study
        
        Args:
            performance: Performance data
            
        Returns:
            bool: True if update was successful
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # Check if study exists
            cursor.execute("SELECT study_name FROM study_metadata WHERE study_name = ?", 
                          (performance.study_name,))
            if not cursor.fetchone():
                logger.warning(f"Study '{performance.study_name}' does not exist.")
                return False
            
            # Check if performance data exists
            cursor.execute("SELECT study_name FROM study_performance WHERE study_name = ?", 
                          (performance.study_name,))
            if cursor.fetchone():
                # Update existing performance
                cursor.execute(
                    "UPDATE study_performance SET performance_json = ?, update_timestamp = ? WHERE study_name = ?",
                    (
                        json.dumps(performance.to_dict()),
                        performance.update_timestamp,
                        performance.study_name
                    )
                )
            else:
                # Insert new performance
                cursor.execute(
                    "INSERT INTO study_performance (study_name, performance_json, update_timestamp) VALUES (?, ?, ?)",
                    (
                        performance.study_name,
                        json.dumps(performance.to_dict()),
                        performance.update_timestamp
                    )
                )
            
            conn.commit()
            logger.info(f"Performance data for study '{performance.study_name}' updated.")
            return True
            
        except Exception as e:
            logger.error(f"Error updating performance data: {e}")
            conn.rollback()
            return False
            
        finally:
            conn.close()
    
    def get_study_metadata(self, study_name: str) -> Optional[StudyMetadata]:
        """
        Retrieve metadata for a study
        
        Args:
            study_name: Name of the study
            
        Returns:
            Optional[StudyMetadata]: Study metadata or None if not found
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT metadata_json FROM study_metadata WHERE study_name = ?", 
                          (study_name,))
            result = cursor.fetchone()
            
            if not result:
                return None
                
            return StudyMetadata.from_dict(json.loads(result[0]))
            
        except Exception as e:
            logger.error(f"Error retrieving metadata: {e}")
            return None
            
        finally:
            conn.close()
    
    def get_study_performance(self, study_name: str) -> Optional[StudyPerformance]:
        """
        Retrieve performance data for a study
        
        Args:
            study_name: Name of the study
            
        Returns:
            Optional[StudyPerformance]: Study performance or None if not found
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT performance_json FROM study_performance WHERE study_name = ?", 
                          (study_name,))
            result = cursor.fetchone()
            
            if not result:
                return None
                
            return StudyPerformance.from_dict(json.loads(result[0]))
            
        except Exception as e:
            logger.error(f"Error retrieving performance data: {e}")
            return None
            
        finally:
            conn.close()
    
    def load_optuna_study(self, study_name: str) -> Optional[optuna.Study]:
        """
        Load an Optuna study by name
        
        Args:
            study_name: Name of the study
            
        Returns:
            Optional[optuna.Study]: Loaded study or None if not found
        """
        try:
            metadata = self.get_study_metadata(study_name)
            if not metadata:
                logger.warning(f"Study '{study_name}' does not exist.")
                return None
            
            # Determine storage URL
            db_dir = os.path.dirname(os.path.abspath(self.database_path))
            study_db = os.path.join(db_dir, f"{study_name}.db")
            storage_url = f"sqlite:///{study_db}"
            
            # Load the study
            study = optuna.load_study(
                study_name=study_name,
                storage=storage_url
            )
            
            return study
            
        except Exception as e:
            logger.error(f"Error loading Optuna study: {e}")
            return None
    
    def delete_study(self, study_name: str) -> bool:
        """
        Delete a study from the database
        
        Args:
            study_name: Name of the study
            
        Returns:
            bool: True if deletion was successful
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # Check if study exists
            cursor.execute("SELECT study_name FROM study_metadata WHERE study_name = ?", 
                          (study_name,))
            if not cursor.fetchone():
                logger.warning(f"Study '{study_name}' does not exist.")
                return False
            
            # Delete tags
            cursor.execute("DELETE FROM study_tags WHERE study_name = ?", (study_name,))
            
            # Delete performance
            cursor.execute("DELETE FROM study_performance WHERE study_name = ?", (study_name,))
            
            # Delete metadata
            cursor.execute("DELETE FROM study_metadata WHERE study_name = ?", (study_name,))
            
            # Try to delete Optuna storage
            try:
                db_dir = os.path.dirname(os.path.abspath(self.database_path))
                study_db = os.path.join(db_dir, f"{study_name}.db")
                if os.path.exists(study_db):
                    os.remove(study_db)
            except Exception as e:
                logger.warning(f"Could not delete Optuna storage: {e}")
            
            conn.commit()
            logger.info(f"Study '{study_name}' deleted.")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting study: {e}")
            conn.rollback()
            return False
            
        finally:
            conn.close()
    
    def list_studies(self, 
                     status: Optional[StudyStatus] = None, 
                     asset: Optional[str] = None,
                     timeframe: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     limit: int = 100) -> List[Dict]:
        """
        List studies matching search criteria
        
        Args:
            status: Filter by status
            asset: Filter by asset
            timeframe: Filter by timeframe
            tags: Filter by tags (all tags must be present)
            limit: Maximum number of studies to return
            
        Returns:
            List[Dict]: List of study metadata
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # Build query
            query = "SELECT metadata_json FROM study_metadata"
            params = []
            
            # Build WHERE clauses
            conditions = []
            
            if status:
                conditions.append("status = ?")
                params.append(status.value)
            
            if asset or timeframe or tags:
                # These filters require parsing the JSON metadata
                # For simplicity, retrieve all metadata and filter
                pass
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            # Limit results
            query += f" LIMIT {limit}"
            
            # Execute query
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            # Process results
            studies = []
            for result in results:
                metadata = json.loads(result[0])
                
                # Additional filtering
                if asset and metadata.get("asset") != asset:
                    continue
                    
                if timeframe and metadata.get("timeframe") != timeframe:
                    continue
                
                # Tag filtering
                if tags:
                    metadata_tags = set(metadata.get("tags", []))
                    if not all(tag in metadata_tags for tag in tags):
                        continue
                
                studies.append(metadata)
            
            return studies
            
        except Exception as e:
            logger.error(f"Error listing studies: {e}")
            return []
            
        finally:
            conn.close()
    
    def search_studies_by_tag(self, tag: str) -> List[str]:
        """
        Search for studies with a specific tag
        
        Args:
            tag: Tag to search for
            
        Returns:
            List[str]: List of study names
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT study_name FROM study_tags WHERE tag = ?", (tag,))
            results = cursor.fetchall()
            
            return [result[0] for result in results]
            
        except Exception as e:
            logger.error(f"Error searching by tag: {e}")
            return []
            
        finally:
            conn.close()
    
    def get_top_performing_studies(self, 
                                  metric: StudyEvaluationMetric = StudyEvaluationMetric.COMBINED_SCORE,
                                  limit: int = 10) -> List[Dict]:
        """
        Get the top performing studies by metric
        
        Args:
            metric: Performance metric to use
            limit: Maximum number of studies to return
            
        Returns:
            List[Dict]: List of studies with performance data
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # Retrieve all performance data
            cursor.execute("""
                SELECT s.metadata_json, p.performance_json
                FROM study_metadata s
                JOIN study_performance p ON s.study_name = p.study_name
                WHERE s.status = ?
            """, (StudyStatus.COMPLETED.value,))
            
            results = cursor.fetchall()
            
            # Process results
            studies = []
            for metadata_json, performance_json in results:
                metadata = json.loads(metadata_json)
                performance = json.loads(performance_json)
                
                # Select appropriate metric
                if metric == StudyEvaluationMetric.COMBINED_SCORE:
                    score = performance.get("best_value", 0.0)
                elif metric == StudyEvaluationMetric.ROI:
                    score = performance.get("roi", 0.0)
                elif metric == StudyEvaluationMetric.ROI_TO_DRAWDOWN:
                    roi = performance.get("roi", 0.0)
                    drawdown = performance.get("max_drawdown", 1.0)
                    score = roi / max(drawdown, 0.01)  # Avoid division by zero
                elif metric == StudyEvaluationMetric.SHARPE_RATIO:
                    score = performance.get("sharpe_ratio", 0.0)
                elif metric == StudyEvaluationMetric.SORTINO_RATIO:
                    score = performance.get("sortino_ratio", 0.0)
                elif metric == StudyEvaluationMetric.CALMAR_RATIO:
                    score = performance.get("calmar_ratio", 0.0)
                elif metric == StudyEvaluationMetric.WIN_RATE:
                    score = performance.get("win_rate", 0.0)
                elif metric == StudyEvaluationMetric.PROFIT_FACTOR:
                    score = performance.get("profit_factor", 0.0)
                else:
                    score = 0.0
                
                studies.append({
                    "metadata": metadata,
                    "performance": performance,
                    "score": score
                })
            
            # Sort by score in descending order
            studies.sort(key=lambda x: x["score"], reverse=True)
            
            # Limit results
            return studies[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving top studies: {e}")
            return []
            
        finally:
            conn.close()
    
    def compare_studies(self, study_names: List[str]) -> Dict:
        """
        Compare multiple studies by their performance metrics
        
        Args:
            study_names: List of study names to compare
            
        Returns:
            Dict: Comparison data formatted for visualization
        """
        comparison_data = {
            "study_names": [],
            "metadata": [],
            "performance": [],
            "metrics": {
                "roi": [],
                "win_rate": [],
                "max_drawdown": [],
                "total_trades": [],
                "profit_factor": [],
                "sharpe_ratio": [],
                "best_value": []
            }
        }
        
        for study_name in study_names:
            # Retrieve metadata and performance
            metadata = self.get_study_metadata(study_name)
            performance = self.get_study_performance(study_name)
            
            if metadata and performance:
                comparison_data["study_names"].append(study_name)
                comparison_data["metadata"].append(metadata.to_dict())
                comparison_data["performance"].append(performance.to_dict())
                
                # Extract metrics for comparison
                comparison_data["metrics"]["roi"].append(performance.roi)
                comparison_data["metrics"]["win_rate"].append(performance.win_rate)
                comparison_data["metrics"]["max_drawdown"].append(performance.max_drawdown)
                comparison_data["metrics"]["total_trades"].append(performance.total_trades)
                comparison_data["metrics"]["profit_factor"].append(performance.profit_factor)
                comparison_data["metrics"]["sharpe_ratio"].append(performance.sharpe_ratio)
                comparison_data["metrics"]["best_value"].append(performance.best_value)
        
        return comparison_data
    
    def update_score_weights(self, study_name: str, weights: ScoreWeights) -> bool:
        """
        Update score weights and recalculate scores for a study
        
        Args:
            study_name: Name of the study
            weights: New score weights
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Get current metadata
            metadata = self.get_study_metadata(study_name)
            if not metadata:
                logger.warning(f"Study '{study_name}' does not exist.")
                return False
            
            # Get performance data
            performance = self.get_study_performance(study_name)
            if not performance:
                logger.warning(f"No performance data for study '{study_name}'.")
                return False
            
            # Update weights
            metadata.score_weights = weights
            
            # Prepare metrics for recalculation
            metrics = {
                'roi': performance.roi,
                'win_rate': performance.win_rate,
                'max_drawdown': performance.max_drawdown,
                'profit_factor': performance.profit_factor,
                'total_trades': performance.total_trades,
                'avg_profit_per_trade': performance.avg_profit_per_trade
            }
            
            # Recalculate score
            new_score = weights.recalculate_score(metrics)
            
            # Update performance with new score
            performance.best_value = new_score
            
            # Save updates
            self.update_study_metadata(metadata)
            self.update_study_performance(performance)
            
            # Try to update Optuna study's best value
            try:
                study = self.load_optuna_study(study_name)
                if study and study.best_trial:
                    # This is a partial update since we can't fully modify the Optuna study
                    logger.info(f"Optuna study has {len(study.trials)} trials.")
                    logger.info(f"Best value update: {study.best_value} -> {new_score}")
                    # Actual modification requires more complex operations on the storage
            except Exception as e:
                logger.warning(f"Could not update Optuna study: {e}")
            
            logger.info(f"Score weights updated for study '{study_name}'.")
            return True
            
        except Exception as e:
            logger.error(f"Error updating score weights: {e}")
            return False
    
    def recalculate_all_scores(self, weights: ScoreWeights) -> int:
        """
        Recalculate scores for all studies using new weights
        
        Args:
            weights: New score weights to apply
            
        Returns:
            int: Number of studies updated
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # Get all studies with performance data
            cursor.execute("""
                SELECT s.study_name 
                FROM study_metadata s
                JOIN study_performance p ON s.study_name = p.study_name
            """)
            
            study_names = [row[0] for row in cursor.fetchall()]
            updated_count = 0
            
            # Update each study
            for study_name in study_names:
                if self.update_score_weights(study_name, weights):
                    updated_count += 1
            
            logger.info(f"Recalculated scores for {updated_count} studies.")
            return updated_count
            
        except Exception as e:
            logger.error(f"Error recalculating scores: {e}")
            return 0
            
        finally:
            conn.close()
    
    def export_study_to_json(self, study_name: str, output_path: Optional[str] = None) -> str:
        """
        Export a study to JSON format
        
        Args:
            study_name: Name of the study
            output_path: Output file path (optional)
            
        Returns:
            str: Path to the exported file
        """
        # Retrieve study data
        metadata = self.get_study_metadata(study_name)
        performance = self.get_study_performance(study_name)
        
        if not metadata:
            raise ValueError(f"Study '{study_name}' not found")
        
        # Prepare export data
        export_data = {
            "metadata": metadata.to_dict(),
            "performance": performance.to_dict() if performance else None
        }
        
        # Determine output path
        if not output_path:
            os.makedirs("exports", exist_ok=True)
            output_path = f"exports/study_{study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Write file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=4)
        
        logger.info(f"Study '{study_name}' exported to {output_path}")
        return output_path
    
    def import_study_from_json(self, input_path: str, overwrite: bool = False) -> bool:
        """
        Import a study from JSON file
        
        Args:
            input_path: Path to JSON file
            overwrite: Whether to overwrite existing study
            
        Returns:
            bool: True if import was successful
        """
        try:
            # Read file
            with open(input_path, 'r') as f:
                import_data = json.load(f)
            
            # Validate data
            if not isinstance(import_data, dict) or "metadata" not in import_data:
                raise ValueError("Invalid file format")
            
            # Create objects
            metadata = StudyMetadata.from_dict(import_data["metadata"])
            
            # Check if study already exists
            existing_metadata = self.get_study_metadata(metadata.study_name)
            if existing_metadata and not overwrite:
                raise ValueError(f"Study '{metadata.study_name}' already exists. Use overwrite=True to replace.")
            
            # Import metadata
            if existing_metadata and overwrite:
                self.update_study_metadata(metadata)
            else:
                # For new study, create Optuna study as well
                success, _ = self.create_study(metadata)
                if not success:
                    return False
            
            # Import performance
            if "performance" in import_data and import_data["performance"]:
                performance = StudyPerformance.from_dict(import_data["performance"])
                self.update_study_performance(performance)
            
            logger.info(f"Study '{metadata.study_name}' imported from {input_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing study: {e}")
            return False
    
    def clone_study(self, source_study_name: str, target_study_name: str) -> bool:
        """
        Clone an existing study to a new one
        
        Args:
            source_study_name: Name of the source study
            target_study_name: Name of the target study
            
        Returns:
            bool: True if cloning was successful
        """
        # Retrieve source study data
        source_metadata = self.get_study_metadata(source_study_name)
        source_performance = self.get_study_performance(source_study_name)
        
        if not source_metadata:
            logger.warning(f"Source study '{source_study_name}' not found")
            return False
        
        # Check if target study already exists
        target_metadata = self.get_study_metadata(target_study_name)
        if target_metadata:
            logger.warning(f"Target study '{target_study_name}' already exists")
            return False
        
        # Create target study with modified metadata
        target_metadata_dict = source_metadata.to_dict()
        target_metadata_dict["study_name"] = target_study_name
        target_metadata_dict["creation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        target_metadata_dict["last_modified"] = target_metadata_dict["creation_date"]
        target_metadata_dict["status"] = StudyStatus.IN_PROGRESS.value
        
        target_metadata = StudyMetadata.from_dict(target_metadata_dict)
        
        success, _ = self.create_study(target_metadata)
        if not success:
            return False
        
        # Copy performance if available
        if source_performance:
            target_performance_dict = source_performance.to_dict()
            target_performance_dict["study_name"] = target_study_name
            target_performance_dict["update_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            target_performance = StudyPerformance.from_dict(target_performance_dict)
            
            if not self.update_study_performance(target_performance):
                return False
        
        logger.info(f"Study '{source_study_name}' cloned to '{target_study_name}'")
        return True


class OptunaObjectiveHandler:
    """
    Handler for Optuna objective functions with scoring
    """
    
    def __init__(self, 
                 study_manager: StudyManager,
                 study_name: str,
                 simulator_cls,  # Type of simulator to use
                 config_cls,     # Type of config to use
                 data_path: str):
        """
        Initialize handler
        
        Args:
            study_manager: StudyManager instance
            study_name: Name of the study to handle
            simulator_cls: Simulator class to use
            config_cls: Configuration class for simulator
            data_path: Path to data file
        """
        self.study_manager = study_manager
        self.study_name = study_name
        self.simulator_cls = simulator_cls
        self.config_cls = config_cls
        self.data_path = data_path
        
        # Load study metadata
        self.metadata = study_manager.get_study_metadata(study_name)
        if not self.metadata:
            raise ValueError(f"Study '{study_name}' not found")
        
        # Cache for loaded data
        self.data_cache = None
        
    def load_data(self):
        """Load and cache the data"""
        if self.data_cache is None:
            # Load data from file
            try:
                if self.data_path.endswith('.csv'):
                    self.data_cache = pd.read_csv(self.data_path)
                elif self.data_path.endswith('.json'):
                    self.data_cache = pd.read_json(self.data_path)
                else:
                    raise ValueError(f"Unsupported data format: {self.data_path}")
                    
                logger.info(f"Data loaded: {self.data_path}, shape: {self.data_cache.shape}")
                
            except Exception as e:
                logger.error(f"Error loading data: {e}")
                raise
                
        return self.data_cache
    
    def __call__(self, trial):
        """
        Objective function for Optuna optimization
        
        Args:
            trial: Optuna trial
            
        Returns:
            float: Score for the trial
        """
        # Load data
        data = self.load_data()
        
        # Generate configuration from trial
        config = self._generate_config(trial)
        
        # Run simulation
        simulator = self.simulator_cls(config)
        results = simulator.run(data)
        
        # Extract metrics
        metrics = self._extract_metrics(results)
        
        # Store metrics in trial
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
            
        # Calculate score
        score = self.metadata.score_weights.recalculate_score(metrics)
        
        return score
    
    def _generate_config(self, trial):
        """
        Generate configuration from trial
        This method should be implemented by subclasses
        
        Args:
            trial: Optuna trial
            
        Returns:
            Config: Configuration for the simulator
        """
        raise NotImplementedError("Subclasses must implement _generate_config")
    
    def _extract_metrics(self, results: Dict) -> Dict:
        """
        Extract metrics from simulation results
        
        Args:
            results: Simulation results
            
        Returns:
            Dict: Extracted performance metrics
        """
        perf = results.get('performance', {})
        
        return {
            'roi': perf.get('roi', 0.0),
            'win_rate': perf.get('win_rate', 0.0),
            'max_drawdown': perf.get('max_drawdown', 0.0),
            'total_trades': perf.get('total_trades', 0),
            'profit_factor': perf.get('profit_factor', 0.0),
            'avg_profit_per_trade': perf.get('avg_profit_per_trade', 0.0),
            'sharpe_ratio': perf.get('sharpe_ratio', 0.0)
        }