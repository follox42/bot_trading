"""
Module de la page de gestion et de comparaison des études de trading.
Cette page permet de créer, comparer, analyser et gérer les études d'optimisation de stratégies.
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from dash import html, dcc, Input, Output, State, callback, dash_table, no_update, ALL, ctx
import dash_bootstrap_components as dbc
import dash
import traceback

from study_manager import (
    StudyManager,
    StudyMetadata,
    StudyPerformance,
    StudyStatus,
    StudyEvaluationMetric
)

from logger.logger import LoggerType, CentralizedLogger


def create_studies_page(central_logger=None):
    """
    Crée la page de gestion des études de trading
    
    Args:
        central_logger: Instance du logger centralisé
    
    Returns:
        Layout de la page des études
    """
    # Initialiser le logger
    ui_logger = None
    if central_logger:
        ui_logger = central_logger.get_logger("studies_page", LoggerType.UI)
        ui_logger.info("Création de la page des études")
    
    # Initialiser le gestionnaire d'études
    try:
        study_manager = StudyManager("data/studies.db")
        if ui_logger:
            ui_logger.info("Gestionnaire d'études initialisé avec succès")
    except Exception as e:
        if ui_logger:
            ui_logger.error(f"Erreur lors de l'initialisation du gestionnaire d'études: {str(e)}")
            ui_logger.error(traceback.format_exc())
        study_manager = None
    
    # Récupération des données initiales
    initial_studies = []
    asset_options = [{"label": "Tous", "value": "all"}]
    timeframe_options = [{"label": "Tous", "value": "all"}]
    tag_options = []
    
    try:
        if study_manager:
            # Récupérer toutes les études
            studies = study_manager.list_studies(limit=1000)
            
            # Extraire les actifs et timeframes uniques
            assets = sorted(set(study.get('asset', '') for study in studies))
            timeframes = sorted(set(study.get('timeframe', '') for study in studies))
            
            # Mettre à jour les options
            asset_options.extend([{"label": a, "value": a} for a in assets if a])
            timeframe_options.extend([{"label": t, "value": t} for t in timeframes if t])
            
            # Collecter les tags uniques
            all_tags = set()
            for study in studies:
                all_tags.update(study.get('tags', []))
            tag_options = [{"label": tag, "value": tag} for tag in sorted(all_tags) if tag]
            
            if ui_logger:
                ui_logger.info(f"Données initiales chargées: {len(studies)} études, {len(assets)} actifs, {len(timeframes)} timeframes")
    except Exception as e:
        if ui_logger:
            ui_logger.error(f"Erreur lors du chargement des données initiales: {str(e)}")
            ui_logger.error(traceback.format_exc())
    
    # Création des composants de la page
    return html.Div([
        html.H2("ÉTUDES ET STRATÉGIES DE TRADING", className="text-xl text-cyan-300 font-bold mb-6 border-b border-gray-700 pb-2"),
        
        # Conteneur à onglets principal
        dbc.Tabs([
            # Onglet Tableau de bord
            dbc.Tab([
                # Statistiques des études
                dbc.Row([
                    dbc.Col([
                        create_studies_stats_card()
                    ], width=12, className="mb-4")
                ]),
                
                # Meilleures études
                dbc.Row([
                    dbc.Col([
                        create_top_studies_card()
                    ], width=12, md=6, className="mb-4"),
                    dbc.Col([
                        create_recent_studies_card()
                    ], width=12, md=6, className="mb-4")
                ]),
                
                # Graphiques de performance
                dbc.Row([
                    dbc.Col([
                        create_performance_overview_card()
                    ], width=12, className="mb-4")
                ])
            ], 
            label="Tableau de Bord", 
            tab_id="dashboard",
            labelClassName="retro-tab-label",
            activeLabelClassName="retro-tab-active"),
            
            # Onglet Recherche et comparaison
            dbc.Tab([
                # Filtres de recherche
                dbc.Row([
                    dbc.Col([
                        create_search_filters_card(asset_options, timeframe_options, tag_options)
                    ], width=12, className="mb-4")
                ]),
                
                # Tableau des études
                dbc.Row([
                    dbc.Col([
                        html.Div(id="studies-table-container")
                    ], width=12, className="mb-4")
                ]),
                
                # Vue de comparaison
                dbc.Row([
                    dbc.Col([
                        html.Div(id="studies-comparison-container")
                    ], width=12, className="mb-4")
                ])
            ], 
            label="Recherche et Comparaison", 
            tab_id="search",
            labelClassName="retro-tab-label",
            activeLabelClassName="retro-tab-active"),
            
            # Onglet Détails de l'étude
            dbc.Tab([
                # Détails de l'étude sélectionnée
                dbc.Row([
                    dbc.Col([
                        html.Div(id="study-details-container")
                    ], width=12, className="mb-4")
                ])
            ], 
            label="Détails de l'Étude", 
            tab_id="details",
            labelClassName="retro-tab-label",
            activeLabelClassName="retro-tab-active"),
            
            # Onglet Création et importation
            dbc.Tab([
                # Formulaire de création d'étude
                dbc.Row([
                    dbc.Col([
                        create_study_creation_card(asset_options, timeframe_options)
                    ], width=12, md=6, className="mb-4"),
                    dbc.Col([
                        create_study_import_card()
                    ], width=12, md=6, className="mb-4")
                ])
            ], 
            label="Créer / Importer", 
            tab_id="create",
            labelClassName="retro-tab-label",
            activeLabelClassName="retro-tab-active")
        ], 
        id="studies-main-tabs", 
        active_tab="dashboard"),
        
        # Stockage des données
        dcc.Store(id="studies-data-store", data=initial_studies),
        dcc.Store(id="selected-studies-store", data=[]),
        dcc.Store(id="active-study-store", data=None),
        
        # Intervalles d'actualisation
        dcc.Interval(
            id="studies-refresh-interval",
            interval=30 * 1000,  # 30 secondes
            n_intervals=0
        ),
        
        # Modals
        create_confirmation_modal(),
        create_notification_modal(),
        
        # Container caché pour les études par tag
        html.Div(id="studies-by-tag-container", style={"display": "none"})
    ])

def create_studies_stats_card():
    """
    Crée une carte avec les statistiques sur les études
    
    Returns:
        dbc.Card: Carte avec les statistiques
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H3("STATISTIQUES DES ÉTUDES", className="card-title"),
            html.Button([
                html.I(className="bi bi-arrow-repeat me-2"), 
                "Actualiser"
            ], id="refresh-stats-button", className="retro-button secondary ms-auto")
        ], className="d-flex justify-content-between align-items-center"),
        dbc.CardBody([
            dbc.Row([
                # Total des études
                dbc.Col([
                    html.Div([
                        html.H2(id="total-studies-count", className="text-center text-info mb-0"),
                        html.P("Études Totales", className="text-center text-muted mb-0")
                    ], className="border-start border-info border-4 ps-3")
                ], width=6, md=3, className="mb-3"),
                
                # Études complétées
                dbc.Col([
                    html.Div([
                        html.H2(id="completed-studies-count", className="text-center text-success mb-0"),
                        html.P("Études Complétées", className="text-center text-muted mb-0")
                    ], className="border-start border-success border-4 ps-3")
                ], width=6, md=3, className="mb-3"),
                
                # Études en cours
                dbc.Col([
                    html.Div([
                        html.H2(id="in-progress-studies-count", className="text-center text-warning mb-0"),
                        html.P("Études En Cours", className="text-center text-muted mb-0")
                    ], className="border-start border-warning border-4 ps-3")
                ], width=6, md=3, className="mb-3"),
                
                # Actifs étudiés
                dbc.Col([
                    html.Div([
                        html.H2(id="assets-count", className="text-center text-primary mb-0"),
                        html.P("Actifs Étudiés", className="text-center text-muted mb-0")
                    ], className="border-start border-primary border-4 ps-3")
                ], width=6, md=3, className="mb-3")
            ])
        ])
    ], className="retro-card")

def create_top_studies_card():
    """
    Crée une carte avec les meilleures études
    
    Returns:
        dbc.Card: Carte avec le top des études
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H3("MEILLEURES ÉTUDES", className="card-title"),
            dbc.Select(
                id="top-studies-metric-select",
                options=[
                    {"label": "ROI", "value": "roi"},
                    {"label": "Win Rate", "value": "win_rate"},
                    {"label": "Profit Factor", "value": "profit_factor"},
                    {"label": "Score Combiné", "value": "combined_score"}
                ],
                value="roi",
                className="ms-auto",
                style={"width": "200px"}
            )
        ], className="d-flex justify-content-between align-items-center"),
        dbc.CardBody([
            html.Div(id="top-studies-container")
        ])
    ], className="retro-card")

def create_recent_studies_card():
    """
    Crée une carte avec les études récentes
    
    Returns:
        dbc.Card: Carte avec les études récentes
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H3("ÉTUDES RÉCENTES", className="card-title"),
        ]),
        dbc.CardBody([
            html.Div(id="recent-studies-container")
        ])
    ], className="retro-card")

def create_performance_overview_card():
    """
    Crée une carte avec l'aperçu des performances
    
    Returns:
        dbc.Card: Carte avec l'aperçu des performances
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H3("APERÇU DES PERFORMANCES", className="card-title"),
            dbc.Select(
                id="performance-chart-select",
                options=[
                    {"label": "ROI par Actif", "value": "roi_by_asset"},
                    {"label": "Win Rate par Timeframe", "value": "win_rate_by_timeframe"},
                    {"label": "Distribution des Trades", "value": "trades_distribution"},
                    {"label": "Évolution Temporelle", "value": "time_evolution"}
                ],
                value="roi_by_asset",
                className="ms-auto",
                style={"width": "250px"}
            )
        ], className="d-flex justify-content-between align-items-center"),
        dbc.CardBody([
            html.Div(id="performance-chart-container", style={"height": "500px"})
        ])
    ], className="retro-card")

def create_search_filters_card(asset_options, timeframe_options, tag_options):
    """
    Crée une carte avec les filtres de recherche
    
    Args:
        asset_options: Options pour le filtre d'actifs
        timeframe_options: Options pour le filtre de timeframes
        tag_options: Options pour le filtre de tags
        
    Returns:
        dbc.Card: Carte avec les filtres de recherche
    """
    # Statuts possibles
    status_options = [
        {"label": "Tous", "value": "all"},
        {"label": "Terminés", "value": StudyStatus.COMPLETED.value},
        {"label": "En cours", "value": StudyStatus.IN_PROGRESS.value},
        {"label": "Échoués", "value": StudyStatus.FAILED.value},
        {"label": "Archivés", "value": StudyStatus.ARCHIVED.value}
    ]
    
    # Métriques d'évaluation
    metric_options = [
        {"label": "Score combiné", "value": StudyEvaluationMetric.COMBINED_SCORE.value},
        {"label": "ROI", "value": StudyEvaluationMetric.ROI.value},
        {"label": "ROI/Drawdown", "value": StudyEvaluationMetric.ROI_TO_DRAWDOWN.value},
        {"label": "Ratio de Sharpe", "value": StudyEvaluationMetric.SHARPE_RATIO.value},
        {"label": "Taux de réussite", "value": StudyEvaluationMetric.WIN_RATE.value},
        {"label": "Facteur de profit", "value": StudyEvaluationMetric.PROFIT_FACTOR.value}
    ]
    
    return dbc.Card([
        dbc.CardHeader([
            html.H3("FILTRES ET RECHERCHE", className="card-title"),
            html.Button([
                html.I(className="bi bi-arrow-repeat me-2"), 
                "Actualiser"
            ], id="refresh-search-button", className="retro-button secondary ms-auto")
        ], className="d-flex justify-content-between align-items-center"),
        dbc.CardBody([
            dbc.Row([
                # Première colonne - Actif et Timeframe
                dbc.Col([
                    html.Div([
                        html.Label("Actif", className="mb-2"),
                        dbc.Select(
                            id="asset-filter-select",
                            options=asset_options,
                            value="all"
                        )
                    ], className="mb-3"),
                    html.Div([
                        html.Label("Timeframe", className="mb-2"),
                        dbc.Select(
                            id="timeframe-filter-select",
                            options=timeframe_options,
                            value="all"
                        )
                    ], className="mb-3")
                ], width=12, md=3),
                
                # Deuxième colonne - Statut et Métrique
                dbc.Col([
                    html.Div([
                        html.Label("Statut", className="mb-2"),
                        dbc.Select(
                            id="status-filter-select",
                            options=status_options,
                            value="all"
                        )
                    ], className="mb-3"),
                    html.Div([
                        html.Label("Métrique d'évaluation", className="mb-2"),
                        dbc.Select(
                            id="metric-filter-select",
                            options=metric_options,
                            value=StudyEvaluationMetric.COMBINED_SCORE.value
                        )
                    ], className="mb-3")
                ], width=12, md=3),
                
                # Troisième colonne - Tags et Recherche
                dbc.Col([
                    html.Div([
                        html.Label("Tags", className="mb-2"),
                        dcc.Dropdown(
                            id="tags-filter-dropdown",
                            options=tag_options,
                            multi=True,
                            placeholder="Sélectionner des tags...",
                            className="text-dark"
                        )
                    ], className="mb-3"),
                    html.Div([
                        html.Label("Recherche par nom", className="mb-2"),
                        dbc.Input(
                            id="search-filter-input",
                            type="text",
                            placeholder="Rechercher..."
                        )
                    ], className="mb-3")
                ], width=12, md=3),
                
                # Quatrième colonne - Boutons d'action
                dbc.Col([
                    html.Div([
                        html.Label("Actions", className="mb-2"),
                        dbc.Button(
                            [html.I(className="bi bi-search me-2"), "Appliquer les filtres"],
                            id="apply-filters-button",
                            color="primary",
                            className="w-100 mb-2"
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-bar-chart-line me-2"), "Comparer sélection"],
                            id="compare-selected-button",
                            color="success",
                            className="w-100"
                        )
                    ], className="d-grid gap-2")
                ], width=12, md=3)
            ])
        ])
    ], className="retro-card")

def create_study_creation_card(asset_options, timeframe_options):
    """
    Crée une carte améliorée pour la création d'études avec configuration des paramètres
    
    Returns:
        dbc.Card: Carte pour la création d'études
    """
    # Métriques d'évaluation
    metric_options = [
        {"label": "Score combiné", "value": StudyEvaluationMetric.COMBINED_SCORE.value},
        {"label": "ROI", "value": StudyEvaluationMetric.ROI.value},
        {"label": "ROI/Drawdown", "value": StudyEvaluationMetric.ROI_TO_DRAWDOWN.value},
        {"label": "Ratio de Sharpe", "value": StudyEvaluationMetric.SHARPE_RATIO.value},
        {"label": "Taux de réussite", "value": StudyEvaluationMetric.WIN_RATE.value},
        {"label": "Facteur de profit", "value": StudyEvaluationMetric.PROFIT_FACTOR.value}
    ]
    
    # Types de paramètres
    param_types = [
        {"label": "Nombre", "value": "number"},
        {"label": "Entier", "value": "integer"},
        {"label": "Booléen", "value": "boolean"},
        {"label": "Sélection", "value": "categorical"}
    ]
    
    return dbc.Card([
        dbc.CardHeader([
            html.H3("CRÉER UNE NOUVELLE ÉTUDE", className="card-title")
        ]),
        dbc.CardBody([
            dbc.Tabs([
                # Onglet Informations Générales
                dbc.Tab([
                    html.Div([
                        dbc.Form([
                            # Nom de l'étude
                            dbc.Row([
                                dbc.Label("Nom de l'étude", width=12, html_for="study-name-input"),
                                dbc.Col([
                                    dbc.Input(
                                        id="study-name-input",
                                        type="text",
                                        placeholder="ex: btc_usdt_1h_ema_cross"
                                    ),
                                    dbc.FormText("Un nom unique pour identifier l'étude")
                                ])
                            ], className="mb-3"),
                            
                            # Actif et Timeframe
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Actif", className="mb-2", html_for="study-asset-select"),
                                    dbc.Select(
                                        id="study-asset-select",
                                        options=asset_options,
                                        value=""
                                    )
                                ], width=12, md=6),
                                dbc.Col([
                                    dbc.Label("Timeframe", className="mb-2", html_for="study-timeframe-select"),
                                    dbc.Select(
                                        id="study-timeframe-select",
                                        options=timeframe_options,
                                        value=""
                                    )
                                ], width=12, md=6)
                            ], className="mb-3"),
                            
                            # Dates
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Date de début", className="mb-2", html_for="study-start-date-input"),
                                    dbc.Input(
                                        id="study-start-date-input",
                                        type="date",
                                        value=datetime.now().strftime("%Y-%m-%d")
                                    )
                                ], width=12, md=6),
                                dbc.Col([
                                    dbc.Label("Date de fin (optionnel)", className="mb-2", html_for="study-end-date-input"),
                                    dbc.Input(
                                        id="study-end-date-input",
                                        type="date"
                                    )
                                ], width=12, md=6)
                            ], className="mb-3"),
                            
                            # Métrique d'évaluation
                            dbc.Row([
                                dbc.Label("Métrique d'évaluation", width=12, html_for="study-metric-select"),
                                dbc.Col([
                                    dbc.Select(
                                        id="study-metric-select",
                                        options=metric_options,
                                        value=StudyEvaluationMetric.COMBINED_SCORE.value
                                    ),
                                    dbc.FormText("Métrique utilisée pour évaluer la performance")
                                ])
                            ], className="mb-3"),
                            
                            # Description
                            dbc.Row([
                                dbc.Label("Description", width=12, html_for="study-description-textarea"),
                                dbc.Col([
                                    dbc.Textarea(
                                        id="study-description-textarea",
                                        placeholder="Description détaillée de l'étude...",
                                        style={"height": "120px"}
                                    )
                                ])
                            ], className="mb-3"),
                            
                            # Tags
                            dbc.Row([
                                dbc.Label("Tags (séparés par des virgules)", width=12, html_for="study-tags-input"),
                                dbc.Col([
                                    dbc.Input(
                                        id="study-tags-input",
                                        type="text",
                                        placeholder="ex: bitcoin, tendance, ema"
                                    ),
                                    dbc.FormText("Tags pour catégoriser l'étude")
                                ])
                            ], className="mb-3")
                        ])
                    ])
                ], label="Informations générales", tab_id="general"),
                
                # Onglet Paramètres d'optimisation
                dbc.Tab([
                    html.Div([
                        html.H4("PARAMÈTRES DE LA STRATÉGIE", className="mb-3"),
                        html.P("Définissez les paramètres à optimiser avec leurs valeurs initiales et leurs plages.", className="mb-3"),
                        
                        # Paramètres dynamiques
                        html.Div(id="parameters-container", children=[
                            # Premier paramètre
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        # Nom du paramètre
                                        dbc.Col([
                                            dbc.Label("Nom", className="mb-1"),
                                            dbc.Input(
                                                id={"type": "param-name", "index": 0},
                                                type="text",
                                                placeholder="ex: ema_fast"
                                            )
                                        ], width=12, md=3),
                                        
                                        # Type du paramètre
                                        dbc.Col([
                                            dbc.Label("Type", className="mb-1"),
                                            dbc.Select(
                                                id={"type": "param-type", "index": 0},
                                                options=param_types,
                                                value="number"
                                            )
                                        ], width=12, md=2),
                                        
                                        # Valeur initiale
                                        dbc.Col([
                                            dbc.Label("Valeur", className="mb-1"),
                                            dbc.Input(
                                                id={"type": "param-value", "index": 0},
                                                type="number",
                                                value=0
                                            )
                                        ], width=12, md=2),
                                        
                                        # Valeur minimale
                                        dbc.Col([
                                            dbc.Label("Min", className="mb-1"),
                                            dbc.Input(
                                                id={"type": "param-min", "index": 0},
                                                type="number",
                                                value=0
                                            )
                                        ], width=12, md=2),
                                        
                                        # Valeur maximale
                                        dbc.Col([
                                            dbc.Label("Max", className="mb-1"),
                                            dbc.Input(
                                                id={"type": "param-max", "index": 0},
                                                type="number",
                                                value=100
                                            )
                                        ], width=12, md=2),
                                        
                                        # Bouton de suppression
                                        dbc.Col([
                                            html.Div([
                                                dbc.Button(
                                                    html.I(className="bi bi-trash"),
                                                    color="danger",
                                                    size="sm",
                                                    id={"type": "remove-param", "index": 0},
                                                    className="mt-4"
                                                )
                                            ], className="d-flex align-items-center justify-content-center h-100")
                                        ], width=12, md=1)
                                    ])
                                ])
                            ], className="mb-2")
                        ]),
                        
                        # Bouton pour ajouter un paramètre
                        dbc.Button(
                            [html.I(className="bi bi-plus-circle me-2"), "Ajouter un paramètre"],
                            id="add-parameter-button",
                            color="primary",
                            outline=True,
                            className="mb-4 mt-2"
                        ),
                        
                        # Paramètres spécifiques de la stratégie
                        html.H4("PRÉRÉGLAGES COMMUNS", className="mb-3"),
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Type de stratégie", className="mb-2"),
                                        dbc.Select(
                                            id="strategy-type-select",
                                            options=[
                                                {"label": "Croisement de moyennes mobiles", "value": "ma_cross"},
                                                {"label": "Suivi de tendance", "value": "trend_following"},
                                                {"label": "Oscillateur RSI", "value": "rsi_oscillator"},
                                                {"label": "Bandes de Bollinger", "value": "bollinger_bands"},
                                                {"label": "Divergence MACD", "value": "macd_divergence"},
                                                {"label": "Stratégie personnalisée", "value": "custom"}
                                            ],
                                            value="ma_cross"
                                        )
                                    ], width=12, md=6),
                                    dbc.Col([
                                        dbc.Label("Appliquer un préréglage", className="mb-2"),
                                        dbc.Button(
                                            "Charger les paramètres par défaut",
                                            id="load-preset-button",
                                            color="secondary",
                                            className="w-100 mt-2"
                                        )
                                    ], width=12, md=6)
                                ])
                            ])
                        ], className="mb-4")
                    ])
                ], label="Paramètres", tab_id="parameters"),
                
                # Onglet Configuration avancée
                dbc.Tab([
                    html.Div([
                        html.H4("CONFIGURATION AVANCÉE", className="mb-3"),
                        
                        # Configuration du take profit et stop loss
                        dbc.Card([
                            dbc.CardHeader("Take Profit et Stop Loss"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Take Profit (%)", className="mb-2"),
                                        dbc.Input(
                                            id="take-profit-input",
                                            type="number",
                                            value=2.0,
                                            min=0.1,
                                            max=100,
                                            step=0.1
                                        )
                                    ], width=12, md=6),
                                    dbc.Col([
                                        dbc.Label("Stop Loss (%)", className="mb-2"),
                                        dbc.Input(
                                            id="stop-loss-input",
                                            type="number",
                                            value=1.0,
                                            min=0.1,
                                            max=100,
                                            step=0.1
                                        )
                                    ], width=12, md=6)
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Checkbox(
                                            id="trailing-stop-checkbox",
                                            label="Utiliser un trailing stop",
                                            value=False,
                                            className="mt-3"
                                        )
                                    ], width=12)
                                ])
                            ])
                        ], className="mb-4"),
                        
                        # Configuration du money management
                        dbc.Card([
                            dbc.CardHeader("Money Management"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Taille de position (%)", className="mb-2"),
                                        dbc.Input(
                                            id="position-size-input",
                                            type="number",
                                            value=10.0,
                                            min=1,
                                            max=100,
                                            step=1
                                        )
                                    ], width=12, md=6),
                                    dbc.Col([
                                        dbc.Label("Capital initial ($)", className="mb-2"),
                                        dbc.Input(
                                            id="initial-capital-input",
                                            type="number",
                                            value=10000,
                                            min=100,
                                            step=100
                                        )
                                    ], width=12, md=6)
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Frais de trading (%)", className="mb-2"),
                                        dbc.Input(
                                            id="trading-fees-input",
                                            type="number",
                                            value=0.1,
                                            min=0,
                                            max=10,
                                            step=0.01
                                        )
                                    ], width=12, md=6),
                                    dbc.Col([
                                        dbc.Label("Slippage (%)", className="mb-2"),
                                        dbc.Input(
                                            id="slippage-input",
                                            type="number",
                                            value=0.05,
                                            min=0,
                                            max=5,
                                            step=0.01
                                        )
                                    ], width=12, md=6)
                                ])
                            ])
                        ], className="mb-4"),
                        
                        # Configuration de l'optimisation
                        dbc.Card([
                            dbc.CardHeader("Paramètres d'optimisation"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Nombre d'itérations", className="mb-2"),
                                        dbc.Input(
                                            id="iterations-input",
                                            type="number",
                                            value=100,
                                            min=10,
                                            max=1000,
                                            step=10
                                        )
                                    ], width=12, md=6),
                                    dbc.Col([
                                        dbc.Label("Algorithme", className="mb-2"),
                                        dbc.Select(
                                            id="algorithm-select",
                                            options=[
                                                {"label": "TPE", "value": "tpe"},
                                                {"label": "Recherche aléatoire", "value": "random"},
                                                {"label": "Bayésien", "value": "bayesian"},
                                                {"label": "Grid Search", "value": "grid"}
                                            ],
                                            value="tpe"
                                        )
                                    ], width=12, md=6)
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Checkbox(
                                            id="cross-validation-checkbox",
                                            label="Utiliser la validation croisée (walk-forward)",
                                            value=True,
                                            className="mt-3"
                                        )
                                    ], width=12)
                                ])
                            ])
                        ])
                    ])
                ], label="Configuration avancée", tab_id="advanced")
            ], id="study-creation-tabs"),
            
            # Bouton de création
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Créer et Optimiser",
                        id="create-study-button",
                        color="primary",
                        className="w-100 mt-4"
                    )
                ])
            ])
        ])
    ], className="retro-card")

# Callbacks pour la gestion dynamique des paramètres
def register_parameter_callbacks(app, central_logger=None):
    """
    Enregistre les callbacks pour la gestion dynamique des paramètres
    
    Args:
        app: L'instance de l'application Dash
        central_logger: Instance du logger centralisé
    """
    # Initialiser le logger
    ui_logger = None
    if central_logger:
        ui_logger = central_logger.get_logger("study_parameters", LoggerType.UI)
    
    # Callback pour ajouter un paramètre
    @app.callback(
        Output("parameters-container", "children"),
        [Input("add-parameter-button", "n_clicks"),
         Input({"type": "remove-param", "index": ALL}, "n_clicks")],
        [State("parameters-container", "children")],
        prevent_initial_call=True
    )
    def manage_parameters(add_clicks, remove_clicks, current_parameters):
        # Déterminer quel bouton a été cliqué
        ctx_triggered = dash.callback_context.triggered_id
        
        if not ctx_triggered:
            return dash.no_update
        
        # Cas 1: Ajouter un paramètre
        if ctx_triggered == "add-parameter-button":
            # Déterminer le nouvel index
            new_index = len(current_parameters)
            
            # Créer un nouveau paramètre
            new_parameter = dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        # Nom du paramètre
                        dbc.Col([
                            dbc.Label("Nom", className="mb-1"),
                            dbc.Input(
                                id={"type": "param-name", "index": new_index},
                                type="text",
                                placeholder=f"paramètre_{new_index + 1}"
                            )
                        ], width=12, md=3),
                        
                        # Type du paramètre
                        dbc.Col([
                            dbc.Label("Type", className="mb-1"),
                            dbc.Select(
                                id={"type": "param-type", "index": new_index},
                                options=[
                                    {"label": "Nombre", "value": "number"},
                                    {"label": "Entier", "value": "integer"},
                                    {"label": "Booléen", "value": "boolean"},
                                    {"label": "Sélection", "value": "categorical"}
                                ],
                                value="number"
                            )
                        ], width=12, md=2),
                        
                        # Valeur initiale
                        dbc.Col([
                            dbc.Label("Valeur", className="mb-1"),
                            dbc.Input(
                                id={"type": "param-value", "index": new_index},
                                type="number",
                                value=0
                            )
                        ], width=12, md=2),
                        
                        # Valeur minimale
                        dbc.Col([
                            dbc.Label("Min", className="mb-1"),
                            dbc.Input(
                                id={"type": "param-min", "index": new_index},
                                type="number",
                                value=0
                            )
                        ], width=12, md=2),
                        
                        # Valeur maximale
                        dbc.Col([
                            dbc.Label("Max", className="mb-1"),
                            dbc.Input(
                                id={"type": "param-max", "index": new_index},
                                type="number",
                                value=100
                            )
                        ], width=12, md=2),
                        
                        # Bouton de suppression
                        dbc.Col([
                            html.Div([
                                dbc.Button(
                                    html.I(className="bi bi-trash"),
                                    color="danger",
                                    size="sm",
                                    id={"type": "remove-param", "index": new_index},
                                    className="mt-4"
                                )
                            ], className="d-flex align-items-center justify-content-center h-100")
                        ], width=12, md=1)
                    ])
                ])
            ], className="mb-2")
            
            # Ajouter le nouveau paramètre à la liste
            return current_parameters + [new_parameter]
            
        # Cas 2: Supprimer un paramètre
        elif isinstance(ctx_triggered, dict) and ctx_triggered.get("type") == "remove-param":
            # Récupérer l'index du paramètre à supprimer
            index_to_remove = ctx_triggered.get("index")
            
            # Créer une nouvelle liste sans le paramètre à supprimer
            new_parameters = []
            for i, param in enumerate(current_parameters):
                if i != index_to_remove:
                    new_parameters.append(param)
            
            # S'il n'y a plus de paramètres, en ajouter un par défaut
            if not new_parameters:
                new_parameters = [
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                # Nom du paramètre
                                dbc.Col([
                                    dbc.Label("Nom", className="mb-1"),
                                    dbc.Input(
                                        id={"type": "param-name", "index": 0},
                                        type="text",
                                        placeholder="ex: ema_fast"
                                    )
                                ], width=12, md=3),
                                
                                # Type du paramètre
                                dbc.Col([
                                    dbc.Label("Type", className="mb-1"),
                                    dbc.Select(
                                        id={"type": "param-type", "index": 0},
                                        options=[
                                            {"label": "Nombre", "value": "number"},
                                            {"label": "Entier", "value": "integer"},
                                            {"label": "Booléen", "value": "boolean"},
                                            {"label": "Sélection", "value": "categorical"}
                                        ],
                                        value="number"
                                    )
                                ], width=12, md=2),
                                
                                # Valeur initiale
                                dbc.Col([
                                    dbc.Label("Valeur", className="mb-1"),
                                    dbc.Input(
                                        id={"type": "param-value", "index": 0},
                                        type="number",
                                        value=0
                                    )
                                ], width=12, md=2),
                                
                                # Valeur minimale
                                dbc.Col([
                                    dbc.Label("Min", className="mb-1"),
                                    dbc.Input(
                                        id={"type": "param-min", "index": 0},
                                        type="number",
                                        value=0
                                    )
                                ], width=12, md=2),
                                
                                # Valeur maximale
                                dbc.Col([
                                    dbc.Label("Max", className="mb-1"),
                                    dbc.Input(
                                        id={"type": "param-max", "index": 0},
                                        type="number",
                                        value=100
                                    )
                                ], width=12, md=2),
                                
                                # Bouton de suppression
                                dbc.Col([
                                    html.Div([
                                        dbc.Button(
                                            html.I(className="bi bi-trash"),
                                            color="danger",
                                            size="sm",
                                            id={"type": "remove-param", "index": 0},
                                            className="mt-4"
                                        )
                                    ], className="d-flex align-items-center justify-content-center h-100")
                                ], width=12, md=1)
                            ])
                        ])
                    ], className="mb-2")
                ]
            
            return new_parameters
            
        return dash.no_update
    
    # Callback pour charger les préréglages
    @app.callback(
        Output("parameters-container", "children", allow_duplicate=True),
        Input("load-preset-button", "n_clicks"),
        State("strategy-type-select", "value"),
        prevent_initial_call=True
    )
    def load_preset_parameters(n_clicks, strategy_type):
        if not n_clicks:
            return dash.no_update
        
        # Préréglages pour différents types de stratégies
        presets = {
            "ma_cross": [
                {"name": "ema_fast", "type": "integer", "value": 12, "min": 5, "max": 50},
                {"name": "ema_slow", "type": "integer", "value": 26, "min": 10, "max": 200},
                {"name": "exit_bars", "type": "integer", "value": 5, "min": 1, "max": 20}
            ],
            "trend_following": [
                {"name": "atr_period", "type": "integer", "value": 14, "min": 5, "max": 30},
                {"name": "atr_multiplier", "type": "number", "value": 2.0, "min": 0.5, "max": 5.0},
                {"name": "trailing_stop", "type": "number", "value": 1.5, "min": 0.5, "max": 3.0}
            ],
            "rsi_oscillator": [
                {"name": "rsi_period", "type": "integer", "value": 14, "min": 5, "max": 30},
                {"name": "overbought", "type": "integer", "value": 70, "min": 60, "max": 90},
                {"name": "oversold", "type": "integer", "value": 30, "min": 10, "max": 40},
                {"name": "exit_bars", "type": "integer", "value": 5, "min": 1, "max": 20}
            ],
            "bollinger_bands": [
                {"name": "bb_period", "type": "integer", "value": 20, "min": 10, "max": 50},
                {"name": "bb_std", "type": "number", "value": 2.0, "min": 1.0, "max": 4.0},
                {"name": "exit_threshold", "type": "number", "value": 0.5, "min": 0.1, "max": 1.0}
            ],
            "macd_divergence": [
                {"name": "macd_fast", "type": "integer", "value": 12, "min": 5, "max": 20},
                {"name": "macd_slow", "type": "integer", "value": 26, "min": 15, "max": 50},
                {"name": "macd_signal", "type": "integer", "value": 9, "min": 5, "max": 15},
                {"name": "divergence_bars", "type": "integer", "value": 5, "min": 3, "max": 10}
            ],
            "custom": [
                {"name": "parameter_1", "type": "number", "value": 10, "min": 0, "max": 100},
                {"name": "parameter_2", "type": "number", "value": 20, "min": 0, "max": 100}
            ]
        }
        
        # Obtenir les préréglages pour le type de stratégie sélectionné
        preset_params = presets.get(strategy_type, [])
        
        # Créer les cartes de paramètres
        parameter_cards = []
        for i, param in enumerate(preset_params):
            parameter_cards.append(
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            # Nom du paramètre
                            dbc.Col([
                                dbc.Label("Nom", className="mb-1"),
                                dbc.Input(
                                    id={"type": "param-name", "index": i},
                                    type="text",
                                    value=param["name"],
                                    placeholder=f"paramètre_{i + 1}"
                                )
                            ], width=12, md=3),
                            
                            # Type du paramètre
                            dbc.Col([
                                dbc.Label("Type", className="mb-1"),
                                dbc.Select(
                                    id={"type": "param-type", "index": i},
                                    options=[
                                        {"label": "Nombre", "value": "number"},
                                        {"label": "Entier", "value": "integer"},
                                        {"label": "Booléen", "value": "boolean"},
                                        {"label": "Sélection", "value": "categorical"}
                                    ],
                                    value=param["type"]
                                )
                            ], width=12, md=2),
                            
                            # Valeur initiale
                            dbc.Col([
                                dbc.Label("Valeur", className="mb-1"),
                                dbc.Input(
                                    id={"type": "param-value", "index": i},
                                    type="number",
                                    value=param["value"]
                                )
                            ], width=12, md=2),
                            
                            # Valeur minimale
                            dbc.Col([
                                dbc.Label("Min", className="mb-1"),
                                dbc.Input(
                                    id={"type": "param-min", "index": i},
                                    type="number",
                                    value=param["min"]
                                )
                            ], width=12, md=2),
                            
                            # Valeur maximale
                            dbc.Col([
                                dbc.Label("Max", className="mb-1"),
                                dbc.Input(
                                    id={"type": "param-max", "index": i},
                                    type="number",
                                    value=param["max"]
                                )
                            ], width=12, md=2),
                            
                            # Bouton de suppression
                            dbc.Col([
                                html.Div([
                                    dbc.Button(
                                        html.I(className="bi bi-trash"),
                                        color="danger",
                                        size="sm",
                                        id={"type": "remove-param", "index": i},
                                        className="mt-4"
                                    )
                                ], className="d-flex align-items-center justify-content-center h-100")
                            ], width=12, md=1)
                        ])
                    ])
                ], className="mb-2")
            )
        
        return parameter_cards
    
    # Callback pour collecter tous les paramètres lors de la création d'une étude
    @app.callback(
        [Output("notification-modal", "is_open", allow_duplicate=True),
         Output("notification-title", "children", allow_duplicate=True),
         Output("notification-message", "children", allow_duplicate=True)],
        [Input("create-study-button", "n_clicks")],
        [State("study-name-input", "value"),
         State("study-asset-select", "value"),
         State("study-timeframe-select", "value"),
         State("study-start-date-input", "value"),
         State("study-end-date-input", "value"),
         State("study-metric-select", "value"),
         State("study-description-textarea", "value"),
         State("study-tags-input", "value"),
         State({"type": "param-name", "index": ALL}, "value"),
         State({"type": "param-type", "index": ALL}, "value"),
         State({"type": "param-value", "index": ALL}, "value"),
         State({"type": "param-min", "index": ALL}, "value"),
         State({"type": "param-max", "index": ALL}, "value"),
         State("take-profit-input", "value"),
         State("stop-loss-input", "value"),
         State("trailing-stop-checkbox", "value"),
         State("position-size-input", "value"),
         State("initial-capital-input", "value"),
         State("trading-fees-input", "value"),
         State("slippage-input", "value"),
         State("iterations-input", "value"),
         State("algorithm-select", "value"),
         State("cross-validation-checkbox", "value")],
        prevent_initial_call=True
    )
    def create_study_with_parameters(n_clicks, study_name, asset, timeframe, start_date, end_date, metric, 
                                   description, tags, param_names, param_types, param_values, param_mins, 
                                   param_maxs, take_profit, stop_loss, trailing_stop, position_size, 
                                   initial_capital, trading_fees, slippage, iterations, algorithm, 
                                   cross_validation):
        if not n_clicks:
            return False, "", ""
        
        # Initialiser le gestionnaire d'études
        try:
            study_manager = StudyManager("data/studies.db")
        except Exception as e:
            if ui_logger:
                ui_logger.error(f"Erreur lors de l'initialisation du gestionnaire d'études: {str(e)}")
            return True, "Erreur", "Gestionnaire d'études non disponible"
        
        try:
            # Validation des champs requis
            if not study_name:
                return True, "Erreur", "Le nom de l'étude est requis"
                
            if not asset:
                return True, "Erreur", "L'actif est requis"
                
            if not timeframe:
                return True, "Erreur", "Le timeframe est requis"
            
            # Vérifier si l'étude existe déjà
            existing_metadata = study_manager.get_study_metadata(study_name)
            if existing_metadata:
                return True, "Erreur", f"L'étude '{study_name}' existe déjà"
            
            # Préparation des tags
            tag_list = []
            if tags:
                tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
            
            # Préparation des paramètres
            parameters = {}
            parameter_ranges = {}
            for i in range(len(param_names)):
                if param_names[i]:  # Ignorer les paramètres sans nom
                    param_name = param_names[i]
                    param_type = param_types[i]
                    param_value = param_values[i]
                    param_min = param_mins[i]
                    param_max = param_maxs[i]
                    
                    # Conversion selon le type
                    if param_type == "integer":
                        param_value = int(param_value)
                        param_min = int(param_min)
                        param_max = int(param_max)
                    elif param_type == "number":
                        param_value = float(param_value)
                        param_min = float(param_min)
                        param_max = float(param_max)
                    elif param_type == "boolean":
                        param_value = bool(param_value)
                        param_min = False
                        param_max = True
                    
                    parameters[param_name] = param_value
                    parameter_ranges[param_name] = {
                        "type": param_type,
                        "min": param_min,
                        "max": param_max
                    }
            
            # Configuration avancée
            config = {
                "parameters": parameters,
                "parameter_ranges": parameter_ranges,
                "money_management": {
                    "take_profit": float(take_profit),
                    "stop_loss": float(stop_loss),
                    "trailing_stop": bool(trailing_stop),
                    "position_size": float(position_size),
                    "initial_capital": float(initial_capital),
                    "trading_fees": float(trading_fees),
                    "slippage": float(slippage)
                },
                "optimization": {
                    "iterations": int(iterations),
                    "algorithm": algorithm,
                    "cross_validation": bool(cross_validation)
                }
            }
            
            # Création de l'étude
            metadata = StudyMetadata(
                study_name=study_name,
                asset=asset,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date if end_date else None,
                evaluation_metric=StudyEvaluationMetric(metric),
                description=description if description else "",
                tags=tag_list,
                config=config
            )
            
            # Enregistrement de l'étude
            success = study_manager.create_study(metadata)
            
            if success:
                if ui_logger:
                    ui_logger.info(f"Nouvelle étude créée avec paramètres: {study_name}")
                return True, "Succès", f"L'étude '{study_name}' a été créée avec succès avec {len(parameters)} paramètres"
            else:
                return True, "Erreur", f"Erreur lors de la création de l'étude"
                
        except Exception as e:
            if ui_logger:
                ui_logger.error(f"Erreur lors de la création de l'étude avec paramètres: {str(e)}")
            return True, "Erreur", f"Erreur: {str(e)}"
        
def create_study_import_card():
    """
    Crée une carte pour l'importation d'études
    
    Returns:
        dbc.Card: Carte pour l'importation d'études
    """
    return dbc.Card([
        dbc.CardHeader([
            html.H3("IMPORTER UNE ÉTUDE", className="card-title")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.P("Importez une étude existante à partir d'un fichier JSON.", className="mb-4"),
                        
                        # Section de téléchargement
                        html.Div([
                            html.I(className="bi bi-cloud-upload display-4 d-block text-center mb-3"),
                            dcc.Upload(
                                id="upload-study-json",
                                children=html.Div([
                                    "Glissez-déposez ou ",
                                    html.A("sélectionnez un fichier JSON")
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px 0'
                                },
                                multiple=False
                            ),
                            html.Div(id="upload-status")
                        ], className="mb-4"),
                        
                        # Options d'importation
                        dbc.Checkbox(
                            id="overwrite-existing-checkbox",
                            label="Écraser l'étude existante si elle existe déjà",
                            className="mb-3"
                        ),
                        
                        # Bouton d'importation
                        dbc.Button(
                            [html.I(className="bi bi-file-earmark-arrow-down me-2"), "Importer Étude"],
                            id="import-study-button",
                            color="success",
                            className="w-100"
                        )
                    ])
                ])
            ])
        ])
    ], className="retro-card")

def create_confirmation_modal():
    """
    Crée une modal de confirmation
    
    Returns:
        dbc.Modal: Modal de confirmation
    """
    return dbc.Modal([
        dbc.ModalHeader([
            html.H4("CONFIRMATION", className="text-warning")
        ]),
        dbc.ModalBody([
            html.P(id="confirmation-message")
        ]),
        dbc.ModalFooter([
            dbc.Button(
                "Annuler",
                id="cancel-confirmation-button",
                color="secondary",
                className="me-2"
            ),
            dbc.Button(
                "Confirmer",
                id="confirm-action-button",
                color="warning"
            )
        ])
    ], id="confirmation-modal", centered=True, className="retro-modal")

def create_notification_modal():
    """
    Crée une modal de notification
    
    Returns:
        dbc.Modal: Modal de notification
    """
    return dbc.Modal([
        dbc.ModalHeader([
            html.H4(id="notification-title")
        ]),
        dbc.ModalBody([
            html.P(id="notification-message")
        ]),
        dbc.ModalFooter([
            dbc.Button(
                "Fermer",
                id="close-notification-button",
                color="primary"
            )
        ])
    ], id="notification-modal", centered=True, className="retro-modal")

def create_formatted_table(studies_data):
    """
    Crée un tableau formaté à partir des données d'études
    
    Args:
        studies_data: Liste des données d'études
        
    Returns:
        dash_table.DataTable: Tableau formaté
    """
    if not studies_data:
        return html.Div("Aucune étude trouvée correspondant aux critères.", className="text-center text-muted my-5")
    
    # Préparation des données pour le tableau
    table_data = []
    for study in studies_data:
        # Extraire les métadonnées
        metadata = study.get('metadata', {})
        study_name = metadata.get('study_name', '')
        
        # Extraire les performances
        performance = study.get('performance', {})
        roi = performance.get('roi', 0) * 100 if performance else 0
        win_rate = performance.get('win_rate', 0) * 100 if performance else 0
        max_drawdown = performance.get('max_drawdown', 0) * 100 if performance else 0
        total_trades = performance.get('total_trades', 0) if performance else 0
        profit_factor = performance.get('profit_factor', 0) if performance else 0
        
        # Formatage pour le tableau
        table_data.append({
            "selected": False,
            "study_name": study_name,
            "asset": metadata.get('asset', ''),
            "timeframe": metadata.get('timeframe', ''),
            "status": metadata.get('status', ''),
            "roi": f"{roi:.2f}%",
            "roi_value": roi,  # Pour le tri
            "win_rate": f"{win_rate:.2f}%",
            "win_rate_value": win_rate,  # Pour le tri
            "max_drawdown": f"{max_drawdown:.2f}%",
            "max_drawdown_value": max_drawdown,  # Pour le tri
            "total_trades": total_trades,
            "profit_factor": f"{profit_factor:.2f}",
            "profit_factor_value": profit_factor,  # Pour le tri
            "creation_date": metadata.get('creation_date', ''),
            "last_modified": metadata.get('last_modified', '')
        })
    
    return dash_table.DataTable(
        id='studies-table',
        columns=[
            {"name": "", "id": "selected", "type": "checkbox", "presentation": "checkbox"},
            {"name": "Nom de l'étude", "id": "study_name"},
            {"name": "Actif", "id": "asset"},
            {"name": "Timeframe", "id": "timeframe"},
            {"name": "Statut", "id": "status"},
            {"name": "ROI", "id": "roi", "sort_by": "roi_value"},
            {"name": "Win Rate", "id": "win_rate", "sort_by": "win_rate_value"},
            {"name": "Max DD", "id": "max_drawdown", "sort_by": "max_drawdown_value"},
            {"name": "Trades", "id": "total_trades"},
            {"name": "Profit Factor", "id": "profit_factor", "sort_by": "profit_factor_value"},
            {"name": "Créé le", "id": "creation_date"}
        ],
        data=table_data,
        sort_action="native",
        sort_mode="single",
        row_selectable="multi",
        selected_rows=[],
        page_action="native",
        page_current=0,
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'backgroundColor': '#1F2937',
            'color': '#D1D5DB',
            'fontFamily': '"Share Tech Mono", monospace',
            'border': '1px solid #374151'
        },
        style_header={
            'backgroundColor': '#111827',
            'fontWeight': 'bold',
            'border': '1px solid #374151'
        },
        style_data_conditional=[
            {
                'if': {'filter_query': '{selected} eq true'},
                'backgroundColor': 'rgba(34, 211, 238, 0.2)',
                'border': '1px solid #22D3EE'
            },
            {
                'if': {'column_id': 'roi', 'filter_query': '{roi_value} < 0'},
                'color': '#F87171'
            },
            {
                'if': {'column_id': 'roi', 'filter_query': '{roi_value} >= 0'},
                'color': '#4ADE80'
            },
            {
                'if': {'column_id': 'status', 'filter_query': '{status} eq "completed"'},
                'color': '#4ADE80'
            },
            {
                'if': {'column_id': 'status', 'filter_query': '{status} eq "in_progress"'},
                'color': '#22D3EE'
            },
            {
                'if': {'column_id': 'status', 'filter_query': '{status} eq "failed"'},
                'color': '#F87171'
            }
        ]
    )

def create_enhanced_comparison_view(selected_studies_data):
    """
    Crée une vue de comparaison améliorée pour les études sélectionnées avec contrôles des paramètres
    
    Args:
        selected_studies_data: Données des études sélectionnées
        
    Returns:
        html.Div: Vue de comparaison améliorée
    """
    if not selected_studies_data:
        return html.Div()
    
    # Préparation des données pour la comparaison
    study_names = []
    roi_values = []
    win_rate_values = []
    drawdown_values = []
    trade_counts = []
    profit_factors = []
    
    # Collecte des paramètres uniques
    all_params = set()
    param_values = {}
    
    for study in selected_studies_data:
        # Extraction des métadonnées
        metadata = study.get('metadata', {})
        study_name = metadata.get('study_name', '')
        study_names.append(study_name)
        
        # Extraction des performances
        performance = study.get('performance', {})
        roi_values.append(performance.get('roi', 0) * 100 if performance else 0)
        win_rate_values.append(performance.get('win_rate', 0) * 100 if performance else 0)
        drawdown_values.append(performance.get('max_drawdown', 0) * 100 if performance else 0)
        trade_counts.append(performance.get('total_trades', 0) if performance else 0)
        profit_factors.append(performance.get('profit_factor', 0) if performance else 0)
        
        # Collecte des paramètres
        best_params = performance.get('best_params', {}) if performance else {}
        for param_name, param_value in best_params.items():
            all_params.add(param_name)
            if param_name not in param_values:
                param_values[param_name] = []
            param_values[param_name].append((study_name, param_value))
    
    # Sortir les paramètres par ordre alphabétique
    all_params_sorted = sorted(all_params)
    
    # Création du layout principal
    return dbc.Card([
        dbc.CardHeader([
            html.H3("COMPARAISON DES ÉTUDES SÉLECTIONNÉES", className="card-title"),
            html.Span(f"{len(selected_studies_data)} études comparées", className="badge bg-info text-dark ms-2")
        ], className="d-flex align-items-center"),
        dbc.CardBody([
            # Onglets pour différents types de comparaisons
            dbc.Tabs([
                # Onglet Métriques de Performance
                dbc.Tab([
                    dbc.Row([
                        # Contrôles des facteurs de pondération
                        dbc.Col([
                            html.H4("PONDÉRATION DES FACTEURS", className="mb-3"),
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("ROI", className="mb-2"),
                                            dcc.Slider(
                                                id="roi-weight-slider",
                                                min=0,
                                                max=10,
                                                step=1,
                                                value=5,
                                                marks={i: str(i) for i in range(0, 11, 2)},
                                                className="mb-3"
                                            )
                                        ], width=12)
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Win Rate", className="mb-2"),
                                            dcc.Slider(
                                                id="winrate-weight-slider",
                                                min=0,
                                                max=10,
                                                step=1,
                                                value=5,
                                                marks={i: str(i) for i in range(0, 11, 2)},
                                                className="mb-3"
                                            )
                                        ], width=12)
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Max Drawdown", className="mb-2"),
                                            dcc.Slider(
                                                id="drawdown-weight-slider",
                                                min=0,
                                                max=10,
                                                step=1,
                                                value=5,
                                                marks={i: str(i) for i in range(0, 11, 2)},
                                                className="mb-3"
                                            )
                                        ], width=12)
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Profit Factor", className="mb-2"),
                                            dcc.Slider(
                                                id="profitfactor-weight-slider",
                                                min=0,
                                                max=10,
                                                step=1,
                                                value=5,
                                                marks={i: str(i) for i in range(0, 11, 2)},
                                                className="mb-3"
                                            )
                                        ], width=12)
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Button(
                                                "Appliquer les pondérations",
                                                id="apply-weights-button",
                                                color="primary",
                                                className="w-100 mt-2"
                                            )
                                        ], width=12)
                                    ])
                                ])
                            ]),
                            
                            # Métriques agrégées
                            html.H4("SCORE AGRÉGÉ", className="mb-3 mt-4"),
                            dbc.Card([
                                dbc.CardBody([
                                    html.Div(id="aggregate-scores-container", children=[
                                        # Sera rempli par un callback
                                        html.P("Utilisez les sliders ci-dessus pour ajuster la pondération des facteurs, puis cliquez sur Appliquer.", className="text-muted")
                                    ])
                                ])
                            ])
                        ], width=12, lg=3),
                        
                        # Graphiques de comparaison
                        dbc.Col([
                            dcc.Graph(
                                id="performance-comparison-graph",
                                figure=create_performance_comparison_figure(
                                    study_names, roi_values, win_rate_values, 
                                    drawdown_values, trade_counts, profit_factors
                                ),
                                config={'displayModeBar': False},
                                style={"height": "600px"}
                            )
                        ], width=12, lg=9)
                    ])
                ], label="Métriques de Performance", tab_id="performance"),
                
                # Onglet Distribution des Paramètres
                dbc.Tab([
                    dbc.Row([
                        # Sélection du paramètre
                        dbc.Col([
                            html.H4("SÉLECTION DU PARAMÈTRE", className="mb-3"),
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Select(
                                        id="param-select",
                                        options=[{"label": param, "value": param} for param in all_params_sorted],
                                        value=all_params_sorted[0] if all_params_sorted else None,
                                        className="mb-3"
                                    ),
                                    html.Div(id="param-stats-container", children=[
                                        # Statistiques sur le paramètre sélectionné
                                        create_param_stats_card(all_params_sorted[0] if all_params_sorted else None, param_values)
                                    ])
                                ])
                            ])
                        ], width=12, lg=3),
                        
                        # Visualisation des paramètres
                        dbc.Col([
                            dcc.Graph(
                                id="param-distribution-graph",
                                figure=create_param_distribution_figure(all_params_sorted[0] if all_params_sorted else None, param_values, study_names),
                                config={'displayModeBar': False},
                                style={"height": "600px"}
                            )
                        ], width=12, lg=9)
                    ])
                ], label="Distribution des Paramètres", tab_id="parameters"),
                
                # Onglet Tableau Comparatif
                dbc.Tab([
                    create_detailed_comparison_table(selected_studies_data)
                ], label="Tableau Comparatif", tab_id="table")
            ], id="comparison-tabs", active_tab="performance")
        ])
    ], className="retro-card")

def create_performance_comparison_figure(study_names, roi_values, win_rate_values, drawdown_values, trade_counts, profit_factors):
    """
    Crée une figure de comparaison des performances
    
    Args:
        study_names: Noms des études
        roi_values: Valeurs de ROI (%)
        win_rate_values: Valeurs de Win Rate (%)
        drawdown_values: Valeurs de Max Drawdown (%)
        trade_counts: Nombre de trades
        profit_factors: Facteurs de profit
        
    Returns:
        go.Figure: Figure de comparaison
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Création de la figure avec sous-graphiques
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "ROI (%)", 
            "Win Rate (%)", 
            "Max Drawdown (%)",
            "Nombre de Trades", 
            "Facteur de Profit",
            "Comparaison des Métriques"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}]
        ]
    )
    
    # Couleurs des barres
    colors = ['#00b894', '#0984e3', '#6c5ce7', '#fd79a8', '#fdcb6e', '#e17055']
    
    # ROI
    fig.add_trace(
        go.Bar(
            x=study_names,
            y=roi_values,
            text=[f"{v:.2f}%" for v in roi_values],
            textposition='auto',
            marker_color=['#4ADE80' if v >= 0 else '#F87171' for v in roi_values],
            name="ROI"
        ),
        row=1, col=1
    )
    
    # Win Rate
    fig.add_trace(
        go.Bar(
            x=study_names,
            y=win_rate_values,
            text=[f"{v:.2f}%" for v in win_rate_values],
            textposition='auto',
            marker_color='#22D3EE',
            name="Win Rate"
        ),
        row=1, col=2
    )
    
    # Max Drawdown
    fig.add_trace(
        go.Bar(
            x=study_names,
            y=drawdown_values,
            text=[f"{v:.2f}%" for v in drawdown_values],
            textposition='auto',
            marker_color='#F87171',
            name="Max Drawdown"
        ),
        row=1, col=3
    )
    
    # Nombre de Trades
    fig.add_trace(
        go.Bar(
            x=study_names,
            y=trade_counts,
            text=trade_counts,
            textposition='auto',
            marker_color='#A78BFA',
            name="Trades"
        ),
        row=2, col=1
    )
    
    # Facteur de Profit
    fig.add_trace(
        go.Bar(
            x=study_names,
            y=profit_factors,
            text=[f"{v:.2f}" for v in profit_factors],
            textposition='auto',
            marker_color='#FBBF24',
            name="Profit Factor"
        ),
        row=2, col=2
    )
    
    # Graphique radar pour la comparaison globale
    # Normalisation des valeurs pour le graphique radar
    max_roi = max(abs(r) for r in roi_values) if roi_values else 1
    max_win_rate = max(win_rate_values) if win_rate_values else 1
    max_drawdown = max(drawdown_values) if drawdown_values else 1
    max_trades = max(trade_counts) if trade_counts else 1
    max_profit_factor = max(profit_factors) if profit_factors else 1
    
    # Inversion des valeurs de drawdown pour qu'une valeur inférieure soit meilleure
    norm_drawdown_values = [1 - (d / max_drawdown) for d in drawdown_values]
    
    for i, study in enumerate(study_names):
        # Calculer les valeurs normalisées
        try:
            normalized_values = [
                roi_values[i] / max_roi if max_roi > 0 else 0,
                win_rate_values[i] / max_win_rate if max_win_rate > 0 else 0,
                norm_drawdown_values[i],
                trade_counts[i] / max_trades if max_trades > 0 else 0,
                profit_factors[i] / max_profit_factor if max_profit_factor > 0 else 0
            ]
            # Ajouter la première valeur à la fin pour fermer le polygone
            radar_values = normalized_values + [normalized_values[0]]
            
            # Ajouter les catégories
            categories = ["ROI", "Win Rate", "Low Drawdown", "Trade Count", "Profit Factor", "ROI"]
            
            fig.add_trace(
                go.Scatterpolar(
                    r=radar_values,
                    theta=categories,
                    fill='toself',
                    name=study,
                    opacity=0.7
                ),
                row=2, col=3
            )
        except:
            # En cas d'erreur, ignorer cette série
            pass
    
    # Mise en page
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5
        ),
        template="plotly_dark",
        paper_bgcolor='#111827',
        plot_bgcolor='#1F2937',
        margin=dict(t=50, b=30, l=30, r=30),
        font=dict(family='"Share Tech Mono", monospace')
    )
    
    # Configuration du graphique radar
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        )
    )
    
    # Mise en forme des axes
    axis_config = dict(
        showgrid=True,
        gridcolor='#374151',
        linecolor='#4B5563',
        tickfont=dict(family='"Share Tech Mono", monospace')
    )
    
    fig.update_xaxes(axis_config)
    fig.update_yaxes(axis_config)
    
    return fig

def create_param_distribution_figure(param_name, param_values, study_names):
    """
    Crée une figure montrant la distribution des valeurs d'un paramètre
    
    Args:
        param_name: Nom du paramètre à visualiser
        param_values: Dictionnaire des valeurs de paramètres par étude
        study_names: Noms des études à comparer
        
    Returns:
        go.Figure: Figure de distribution
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    if not param_name or param_name not in param_values:
        # Figure vide
        fig = go.Figure()
        fig.update_layout(
            title="Sélectionnez un paramètre pour voir sa distribution",
            template="plotly_dark",
            paper_bgcolor='#111827',
            plot_bgcolor='#1F2937'
        )
        return fig
    
    # Récupérer les valeurs du paramètre
    values = []
    labels = []
    for study_name, value in param_values[param_name]:
        if study_name in study_names:
            values.append(value)
            labels.append(study_name)
    
    if not values:
        # Figure vide
        fig = go.Figure()
        fig.update_layout(
            title=f"Pas de données pour le paramètre {param_name}",
            template="plotly_dark",
            paper_bgcolor='#111827',
            plot_bgcolor='#1F2937'
        )
        return fig
    
    # Création de la figure avec sous-graphiques
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Valeurs de '{param_name}' par étude", 
            f"Distribution de '{param_name}'",
            f"Impact de '{param_name}' sur le ROI",
            f"Corrélation avec le Win Rate"
        ),
        specs=[
            [{"type": "bar"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ]
    )
    
    # Graphique des valeurs par étude
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            text=[f"{v:.4g}" if isinstance(v, (int, float)) else str(v) for v in values],
            textposition='auto',
            marker_color='#22D3EE',
            name=param_name
        ),
        row=1, col=1
    )
    
    # Histogramme de distribution
    if all(isinstance(v, (int, float)) for v in values):
        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=min(10, len(values)),
                marker_color='#4ADE80',
                name="Distribution"
            ),
            row=1, col=2
        )
    
    # Graphiques de corrélation fictifs (à remplacer par des données réelles)
    # Ici on crée des données fictives juste pour l'exemple
    x = np.array(values)
    try:
        if all(isinstance(v, (int, float)) for v in values):
            y_roi = 10 * x + np.random.normal(0, 5, size=len(x))
            y_winrate = 50 + 5 * x + np.random.normal(0, 10, size=len(x))
            
            # Graphique de l'impact sur le ROI
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_roi,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='#F59E0B',
                        symbol='circle'
                    ),
                    name="ROI"
                ),
                row=2, col=1
            )
            
            # Ligne de tendance
            if len(x) > 1:
                z = np.polyfit(x, y_roi, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(x), max(x), 100)
                y_line = p(x_line)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode='lines',
                        line=dict(color='#F59E0B', dash='dash'),
                        name="Tendance ROI"
                    ),
                    row=2, col=1
                )
            
            # Graphique de corrélation avec le win rate
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_winrate,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='#A78BFA',
                        symbol='circle'
                    ),
                    name="Win Rate"
                ),
                row=2, col=2
            )
            
            # Ligne de tendance
            if len(x) > 1:
                z = np.polyfit(x, y_winrate, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(x), max(x), 100)
                y_line = p(x_line)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode='lines',
                        line=dict(color='#A78BFA', dash='dash'),
                        name="Tendance Win Rate"
                    ),
                    row=2, col=2
                )
    except:
        # En cas d'erreur, ignorer les graphiques de corrélation
        pass
    
    # Mise en page
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5
        ),
        template="plotly_dark",
        paper_bgcolor='#111827',
        plot_bgcolor='#1F2937',
        margin=dict(t=50, b=30, l=30, r=30),
        font=dict(family='"Share Tech Mono", monospace')
    )
    
    # Mise en forme des axes
    axis_config = dict(
        showgrid=True,
        gridcolor='#374151',
        linecolor='#4B5563',
        tickfont=dict(family='"Share Tech Mono", monospace')
    )
    
    fig.update_xaxes(axis_config)
    fig.update_yaxes(axis_config)
    
    return fig

def create_param_stats_card(param_name, param_values):
    """
    Crée une carte avec les statistiques sur un paramètre
    
    Args:
        param_name: Nom du paramètre
        param_values: Dictionnaire des valeurs de paramètres par étude
        
    Returns:
        html.Div: Carte de statistiques
    """
    import numpy as np
    
    if not param_name or param_name not in param_values:
        return html.Div("Sélectionnez un paramètre pour voir ses statistiques", className="text-muted")
    
    # Récupérer les valeurs du paramètre
    values = [value for _, value in param_values[param_name]]
    
    # Vérifier si les valeurs sont numériques
    if not values:
        return html.Div("Pas de données pour ce paramètre", className="text-muted")
    
    if all(isinstance(v, (int, float)) for v in values):
        # Statistiques numériques
        try:
            min_value = min(values)
            max_value = max(values)
            mean_value = np.mean(values)
            median_value = np.median(values)
            std_value = np.std(values)
            
            return html.Div([
                html.H5("Statistiques", className="mb-3"),
                html.Table([
                    html.Tbody([
                        html.Tr([
                            html.Td("Minimum:", className="font-weight-bold"),
                            html.Td(f"{min_value:.4g}" if abs(min_value) < 0.01 else f"{min_value:.4f}")
                        ]),
                        html.Tr([
                            html.Td("Maximum:", className="font-weight-bold"),
                            html.Td(f"{max_value:.4g}" if abs(max_value) < 0.01 else f"{max_value:.4f}")
                        ]),
                        html.Tr([
                            html.Td("Moyenne:", className="font-weight-bold"),
                            html.Td(f"{mean_value:.4g}" if abs(mean_value) < 0.01 else f"{mean_value:.4f}")
                        ]),
                        html.Tr([
                            html.Td("Médiane:", className="font-weight-bold"),
                            html.Td(f"{median_value:.4g}" if abs(median_value) < 0.01 else f"{median_value:.4f}")
                        ]),
                        html.Tr([
                            html.Td("Écart-type:", className="font-weight-bold"),
                            html.Td(f"{std_value:.4g}" if abs(std_value) < 0.01 else f"{std_value:.4f}")
                        ]),
                        html.Tr([
                            html.Td("Nombre:", className="font-weight-bold"),
                            html.Td(f"{len(values)}")
                        ])
                    ])
                ], className="retro-table w-100 mb-3"),
                
                html.H5("Valeur optimale estimée", className="mb-2"),
                html.Div([
                    html.Span(f"{median_value:.4g}" if abs(median_value) < 0.01 else f"{median_value:.4f}", 
                             className="badge bg-info p-2 fs-6")
                ], className="text-center mb-3"),
                
                html.Div([
                    html.Span("Plage recommandée:", className="d-block mb-2"),
                    html.Div([
                        html.Span(f"{(mean_value - std_value):.4g}" if abs(mean_value - std_value) < 0.01 else f"{(mean_value - std_value):.4f}",
                                 className="badge bg-secondary p-2"),
                        html.Span(" à ", className="mx-2"),
                        html.Span(f"{(mean_value + std_value):.4g}" if abs(mean_value + std_value) < 0.01 else f"{(mean_value + std_value):.4f}",
                                 className="badge bg-secondary p-2")
                    ], className="text-center")
                ])
            ])
        except:
            return html.Div("Erreur lors du calcul des statistiques", className="text-danger")
    else:
        # Statistiques non numériques
        value_counts = {}
        for value in values:
            value_str = str(value)
            if value_str not in value_counts:
                value_counts[value_str] = 0
            value_counts[value_str] += 1
        
        # Trier par fréquence
        sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        
        rows = []
        for value, count in sorted_values:
            rows.append(html.Tr([
                html.Td(value),
                html.Td(count),
                html.Td(f"{count/len(values)*100:.1f}%")
            ]))
        
        return html.Div([
            html.H5("Répartition des valeurs", className="mb-3"),
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Valeur"),
                        html.Th("Nombre"),
                        html.Th("Pourcentage")
                    ])
                ]),
                html.Tbody(rows)
            ], className="retro-table w-100 mb-3"),
            
            html.H5("Valeur la plus fréquente", className="mb-2"),
            html.Div([
                html.Span(sorted_values[0][0] if sorted_values else "N/A", 
                         className="badge bg-info p-2 fs-6")
            ], className="text-center")
        ])

def create_detailed_comparison_table(selected_studies_data):
    """
    Crée un tableau détaillé comparant toutes les études sélectionnées
    
    Args:
        selected_studies_data: Données des études sélectionnées
        
    Returns:
        html.Div: Tableau de comparaison détaillé
    """
    import html
    
    if not selected_studies_data:
        return html.Div("Aucune étude sélectionnée", className="text-muted text-center py-4")
    
    # Extraction des métriques pour toutes les études
    metrics_data = []
    params_data = []
    
    for study in selected_studies_data:
        metadata = study.get('metadata', {})
        performance = study.get('performance', {})
        
        # Données des métriques
        metrics_data.append({
            "study_name": metadata.get('study_name', ''),
            "asset": metadata.get('asset', ''),
            "timeframe": metadata.get('timeframe', ''),
            "status": metadata.get('status', ''),
            "roi": performance.get('roi', 0) * 100 if performance else 0,
            "win_rate": performance.get('win_rate', 0) * 100 if performance else 0,
            "max_drawdown": performance.get('max_drawdown', 0) * 100 if performance else 0,
            "total_trades": performance.get('total_trades', 0) if performance else 0,
            "profit_factor": performance.get('profit_factor', 0) if performance else 0,
            "sharpe_ratio": performance.get('sharpe_ratio', 0) if performance else 0,
            "sortino_ratio": performance.get('sortino_ratio', 0) if performance else 0,
            "calmar_ratio": performance.get('calmar_ratio', 0) if performance else 0,
        })
        
        # Données des paramètres
        best_params = performance.get('best_params', {}) if performance else {}
        params_data.append({
            "study_name": metadata.get('study_name', ''),
            **best_params
        })
    
    # Création des tableaux
    return html.Div([
        # Tableau des métriques
        html.H4("MÉTRIQUES DE PERFORMANCE", className="mb-3"),
        html.Div([
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Étude"),
                        html.Th("Actif"),
                        html.Th("Timeframe"),
                        html.Th("ROI (%)"),
                        html.Th("Win Rate (%)"),
                        html.Th("Max DD (%)"),
                        html.Th("Trades"),
                        html.Th("Profit Factor"),
                        html.Th("Sharpe"),
                        html.Th("Sortino"),
                        html.Th("Calmar")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(metrics["study_name"]),
                        html.Td(metrics["asset"]),
                        html.Td(metrics["timeframe"]),
                        html.Td(f"{metrics['roi']:.2f}%", className=f"{'text-success' if metrics['roi'] >= 0 else 'text-danger'}"),
                        html.Td(f"{metrics['win_rate']:.2f}%"),
                        html.Td(f"{metrics['max_drawdown']:.2f}%"),
                        html.Td(f"{metrics['total_trades']}"),
                        html.Td(f"{metrics['profit_factor']:.2f}"),
                        html.Td(f"{metrics['sharpe_ratio']:.2f}"),
                        html.Td(f"{metrics['sortino_ratio']:.2f}"),
                        html.Td(f"{metrics['calmar_ratio']:.2f}")
                    ]) for metrics in metrics_data
                ])
            ], className="retro-table w-100")
        ], className="table-responsive mb-4"),
        
        # Tableau des paramètres
        html.H4("PARAMÈTRES OPTIMAUX", className="mb-3"),
        html.Div([
            create_params_comparison_table(params_data)
        ], className="table-responsive")
    ])

def create_params_comparison_table(params_data):
    """
    Crée un tableau comparatif des paramètres optimaux
    
    Args:
        params_data: Données des paramètres par étude
        
    Returns:
        html.Table: Tableau de comparaison des paramètres
    """
    import html
    
    if not params_data:
        return html.Div("Aucune donnée de paramètres disponible", className="text-muted text-center py-4")
    
    # Collecter tous les paramètres uniques
    all_params = set()
    for study_params in params_data:
        for param in study_params.keys():
            if param != "study_name":
                all_params.add(param)
    
    # Trier les paramètres
    all_params = sorted(all_params)
    
    # Créer les en-têtes
    headers = [html.Th("Étude")]
    for param in all_params:
        headers.append(html.Th(param))
    
    # Créer les lignes
    rows = []
    for study_params in params_data:
        study_name = study_params.get("study_name", "")
        row = [html.Td(study_name)]
        
        for param in all_params:
            value = study_params.get(param, "N/A")
            if isinstance(value, float):
                value = f"{value:.4g}" if abs(value) < 0.01 else f"{value:.4f}"
            row.append(html.Td(str(value)))
        
        rows.append(html.Tr(row))
    
    # Créer le tableau
    return html.Table([
        html.Thead([html.Tr(headers)]),
        html.Tbody(rows)
    ], className="retro-table w-100")

def create_study_details_view(metadata, performance):
    """
    Crée une vue détaillée d'une étude
    
    Args:
        metadata: Métadonnées de l'étude
        performance: Performances de l'étude
        
    Returns:
        dbc.Card: Carte de détails de l'étude
    """
    if not metadata:
        return html.Div("Métadonnées non disponibles", className="text-danger")
    
    # Extraction des données
    study_name = metadata.study_name
    asset = metadata.asset
    timeframe = metadata.timeframe
    status = metadata.status.value
    description = metadata.description
    tags = metadata.tags
    start_date = metadata.start_date
    end_date = metadata.end_date
    metric = metadata.evaluation_metric.value
    creation_date = metadata.creation_date
    last_modified = metadata.last_modified
    
    # Détermination de la couleur de statut
    status_color = {
        StudyStatus.COMPLETED.value: "success",
        StudyStatus.IN_PROGRESS.value: "info",
        StudyStatus.FAILED.value: "danger",
        StudyStatus.ARCHIVED.value: "secondary"
    }.get(status, "secondary")
    
    # Onglets de détails
    tabs = dbc.Tabs([
        # Onglet Résumé
        dbc.Tab([
            dbc.Row([
                # Colonne Gauche - Métriques principales
                dbc.Col([
                    html.Div([
                        html.H4("MÉTRIQUES DE PERFORMANCE", className="mb-3"),
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.H3(f"{performance.roi * 100:.2f}%" if performance else "N/A", 
                                                   className=f"text-{'success' if performance and performance.roi >= 0 else 'danger'} text-center"),
                                            html.P("ROI", className="text-center text-muted mb-0")
                                        ], className="border-bottom pb-3 mb-3")
                                    ], width=6),
                                    dbc.Col([
                                        html.Div([
                                            html.H3(f"{performance.win_rate * 100:.2f}%" if performance else "N/A", 
                                                   className="text-info text-center"),
                                            html.P("Win Rate", className="text-center text-muted mb-0")
                                        ], className="border-bottom pb-3 mb-3")
                                    ], width=6)
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.H3(f"{performance.max_drawdown * 100:.2f}%" if performance else "N/A", 
                                                   className="text-danger text-center"),
                                            html.P("Max Drawdown", className="text-center text-muted mb-0")
                                        ], className="border-bottom pb-3 mb-3")
                                    ], width=6),
                                    dbc.Col([
                                        html.Div([
                                            html.H3(f"{performance.total_trades}" if performance else "N/A", 
                                                   className="text-warning text-center"),
                                            html.P("Trades", className="text-center text-muted mb-0")
                                        ], className="border-bottom pb-3 mb-3")
                                    ], width=6)
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.H3(f"{performance.profit_factor:.2f}" if performance else "N/A", 
                                                   className="text-purple text-center"),
                                            html.P("Profit Factor", className="text-center text-muted mb-0")
                                        ])
                                    ], width=6),
                                    dbc.Col([
                                        html.Div([
                                            html.H3(f"{performance.best_value:.2f}" if performance else "N/A", 
                                                   className="text-cyan text-center"),
                                            html.P("Meilleur Score", className="text-center text-muted mb-0")
                                        ])
                                    ], width=6)
                                ])
                            ])
                        ], className="mb-4"),
                        
                        # Graphique d'évolution de l'optimisation
                        html.H4("ÉVOLUTION DE L'OPTIMISATION", className="mb-3"),
                        dbc.Card([
                            dbc.CardBody([
                                create_optimization_graph(performance) if performance else 
                                html.P("Données d'optimisation non disponibles", className="text-muted text-center")
                            ])
                        ])
                    ])
                ], width=12, lg=6),
                
                # Colonne Droite - Informations et détails
                dbc.Col([
                    html.Div([
                        html.H4("INFORMATIONS GÉNÉRALES", className="mb-3"),
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Strong("Statut:", className="d-block mb-2"),
                                        html.Strong("Actif:", className="d-block mb-2"),
                                        html.Strong("Timeframe:", className="d-block mb-2"),
                                        html.Strong("Période d'étude:", className="d-block mb-2"),
                                        html.Strong("Métrique d'évaluation:", className="d-block mb-2"),
                                        html.Strong("Créé le:", className="d-block mb-2"),
                                        html.Strong("Dernière mise à jour:", className="d-block mb-2")
                                    ], width=4),
                                    dbc.Col([
                                        html.Span(status, className=f"badge bg-{status_color}"),
                                        html.Span(asset, className="d-block mb-2"),
                                        html.Span(timeframe, className="d-block mb-2"),
                                        html.Span(f"{start_date} {' - ' + end_date if end_date else ''}", className="d-block mb-2"),
                                        html.Span(metric, className="d-block mb-2"),
                                        html.Span(creation_date, className="d-block mb-2"),
                                        html.Span(last_modified, className="d-block mb-2")
                                    ], width=8)
                                ])
                            ])
                        ], className="mb-4"),
                        
                        # Description
                        html.H4("DESCRIPTION", className="mb-3"),
                        dbc.Card([
                            dbc.CardBody([
                                html.P(description if description else "Pas de description disponible.", 
                                      style={"white-space": "pre-wrap"})
                            ])
                        ], className="mb-4"),
                        
                        # Tags
                        html.H4("TAGS", className="mb-3"),
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.Span(tag, className="badge bg-secondary me-2 mb-2")
                                    for tag in tags
                                ]) if tags else html.P("Pas de tags")
                            ])
                        ])
                    ])
                ], width=12, lg=6)
            ])
        ], label="Résumé", tab_id="summary", labelClassName="retro-tab-label", activeLabelClassName="retro-tab-active"),
        
        # Onglet Paramètres
        dbc.Tab([
            html.Div([
                html.H4("PARAMÈTRES OPTIMAUX", className="mb-3"),
                
                # Tableau des paramètres
                dbc.Card([
                    dbc.CardBody([
                        create_parameters_table(performance.best_params if performance else {})
                    ])
                ], className="mb-4"),
                
                # Configuration
                html.H4("CONFIGURATION", className="mb-3"),
                dbc.Card([
                    dbc.CardBody([
                        html.Pre(
                            json.dumps(metadata.config, indent=4),
                            style={
                                'backgroundColor': '#111827',
                                'padding': '15px',
                                'borderRadius': '5px',
                                'color': '#D1D5DB',
                                'fontFamily': '"Share Tech Mono", monospace',
                                'fontSize': '14px',
                                'overflowX': 'auto'
                            }
                        )
                    ])
                ])
            ])
        ], label="Paramètres", tab_id="parameters", labelClassName="retro-tab-label", activeLabelClassName="retro-tab-active"),
        
        # Onglet Actions
        dbc.Tab([
            html.Div([
                html.H4("ACTIONS", className="mb-3"),
                
                # Modification de l'étude
                dbc.Card([
                    dbc.CardHeader("Modification et clonage"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-pencil-square me-2"), "Modifier métadonnées"],
                                    id={"type": "edit-study-button", "index": study_name},
                                    color="primary",
                                    className="w-100 mb-2"
                                )
                            ], width=12, md=4),
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-stars me-2"), "Cloner étude"],
                                    id={"type": "clone-study-button", "index": study_name},
                                    color="info",
                                    className="w-100 mb-2"
                                )
                            ], width=12, md=4),
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-archive me-2"), "Archiver"],
                                    id={"type": "archive-study-button", "index": study_name},
                                    color="secondary",
                                    className="w-100 mb-2"
                                )
                            ], width=12, md=4)
                        ])
                    ])
                ], className="mb-4"),
                
                # Exportation et importation
                dbc.Card([
                    dbc.CardHeader("Exportation et génération"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-file-earmark-text me-2"), "Exporter JSON"],
                                    id={"type": "export-study-button", "index": study_name},
                                    color="success",
                                    className="w-100 mb-2"
                                )
                            ], width=12, md=4),
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-file-earmark-code me-2"), "Générer code"],
                                    id={"type": "generate-code-button", "index": study_name},
                                    color="warning",
                                    className="w-100 mb-2"
                                )
                            ], width=12, md=4),
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-clipboard me-2"), "Copier config"],
                                    id={"type": "copy-config-button", "index": study_name},
                                    color="info",
                                    className="w-100 mb-2"
                                )
                            ], width=12, md=4)
                        ])
                    ])
                ], className="mb-4"),
                
                # Suppression
                dbc.Card([
                    dbc.CardHeader("Suppression"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-trash me-2"), "Supprimer l'étude"],
                                    id={"type": "delete-study-button", "index": study_name},
                                    color="danger",
                                    className="w-100"
                                )
                            ], width=12)
                        ])
                    ])
                ])
            ])
        ], label="Actions", tab_id="actions", labelClassName="retro-tab-label", activeLabelClassName="retro-tab-active")
    ], id="study-details-tabs", active_tab="summary")
    
    return dbc.Card([
        dbc.CardHeader([
            html.H3(study_name, className="card-title"),
            html.Span(f"{asset} {timeframe}", className="badge bg-info text-dark ms-2")
        ], className="d-flex align-items-center"),
        dbc.CardBody([tabs])
    ], className="retro-card")

def create_parameters_table(params):
    """
    Crée un tableau des paramètres
    
    Args:
        params: Dictionnaire des paramètres
        
    Returns:
        html.Table: Tableau des paramètres
    """
    if not params:
        return html.P("Pas de paramètres disponibles", className="text-muted")
    
    rows = []
    for key, value in params.items():
        # Formatage de la valeur
        formatted_value = value
        if isinstance(value, float):
            formatted_value = f"{value:.4g}" if abs(value) < 0.01 else f"{value:.4f}"
        
        rows.append(
            html.Tr([
                html.Td(key),
                html.Td(str(formatted_value))
            ])
        )
    
    return html.Table([
        html.Thead([
            html.Tr([
                html.Th("Paramètre"),
                html.Th("Valeur")
            ])
        ]),
        html.Tbody(rows)
    ], className="retro-table w-100")

def create_optimization_graph(performance):
    """
    Crée un graphique d'évolution des métriques pendant l'optimisation
    
    Args:
        performance: Performance de l'étude
        
    Returns:
        dcc.Graph: Graphique d'évolution
    """
    if not performance or not performance.metrics_history:
        return html.P("Données d'optimisation non disponibles", className="text-muted text-center")
    
    metrics_history = performance.metrics_history
    
    # Vérifier les métriques disponibles
    available_metrics = [metric for metric in ['trial_number', 'value', 'roi', 'win_rate', 'max_drawdown'] 
                      if metric in metrics_history and metrics_history[metric]]
    
    if len(available_metrics) < 2:  # Besoin d'au moins trial_number et une métrique
        return html.P("Données d'optimisation insuffisantes", className="text-muted text-center")
    
    # Création du graphique
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Valeur d'Optimisation", 
            "ROI (%)",
            "Win Rate (%)", 
            "Max Drawdown (%)"
        )
    )
    
    trial_numbers = metrics_history.get("trial_number", [])
    
    # Valeur d'optimisation
    if "value" in metrics_history and metrics_history["value"]:
        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=metrics_history["value"],
                mode='markers+lines',
                name='Valeur',
                line=dict(color='#22D3EE')
            ),
            row=1, col=1
        )
    
    # ROI
    if "roi" in metrics_history and metrics_history["roi"]:
        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=[roi * 100 for roi in metrics_history["roi"]],  # Conversion en pourcentage
                mode='markers+lines',
                name='ROI',
                line=dict(color='#4ADE80')
            ),
            row=1, col=2
        )
    
    # Win Rate
    if "win_rate" in metrics_history and metrics_history["win_rate"]:
        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=[wr * 100 for wr in metrics_history["win_rate"]],  # Conversion en pourcentage
                mode='markers+lines',
                name='Win Rate',
                line=dict(color='#3B82F6')
            ),
            row=2, col=1
        )
    
    # Max Drawdown
    if "max_drawdown" in metrics_history and metrics_history["max_drawdown"]:
        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=[dd * 100 for dd in metrics_history["max_drawdown"]],  # Conversion en pourcentage
                mode='markers+lines',
                name='Max DD',
                line=dict(color='#F87171')
            ),
            row=2, col=2
        )
    
    # Mise en page
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="right",
            x=1
        ),
        template="plotly_dark",
        paper_bgcolor='#111827',
        plot_bgcolor='#1F2937',
        margin=dict(t=50, b=30, l=30, r=30),
        font=dict(family='"Share Tech Mono", monospace')
    )
    
    # Mise en forme des axes
    axis_config = dict(
        showgrid=True,
        gridcolor='#374151',
        linecolor='#4B5563',
        tickfont=dict(family='"Share Tech Mono", monospace')
    )
    
    fig.update_xaxes(axis_config)
    fig.update_yaxes(axis_config)
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

# Callbacks supplémentaires pour la page des études
def register_additional_callbacks(app, study_manager, central_logger=None):
    """
    Enregistre des callbacks supplémentaires pour la page des études
    
    Args:
        app: L'instance de l'application Dash
        study_manager: Instance du gestionnaire d'études
        central_logger: Instance du logger centralisé
    """
    # Initialiser le logger
    ui_logger = None
    if central_logger:
        ui_logger = central_logger.get_logger("studies_additional", LoggerType.UI)

    # Callback pour fermer la modal de notification
    @app.callback(
        Output("notification-modal", "is_open", allow_duplicate=True),
        Input("close-notification-button", "n_clicks"),
        prevent_initial_call=True
    )
    def close_notification_modal(n_clicks):
        if n_clicks:
            return False
        return dash.no_update
    
    # Callback pour le traitement des fichiers importés
    @app.callback(
        [Output("upload-status", "children"),
         Output("notification-modal", "is_open", allow_duplicate=True),
         Output("notification-title", "children", allow_duplicate=True),
         Output("notification-message", "children", allow_duplicate=True)],
        Input("upload-study-json", "contents"),
        State("upload-study-json", "filename"),
        State("overwrite-existing-checkbox", "checked"),
        prevent_initial_call=True
    )
    def process_uploaded_file(contents, filename, overwrite):
        if not contents:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        if not study_manager:
            return html.Div("Gestionnaire d'études non disponible", className="text-danger"), True, "Erreur", "Gestionnaire d'études non disponible"
        
        try:
            # Décoder le contenu du fichier
            import base64
            import tempfile
            import os
            
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            # Sauvegarder dans un fichier temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
                temp_file.write(decoded)
                temp_path = temp_file.name
            
            # Importer l'étude
            success = study_manager.import_study_from_json(temp_path, overwrite=overwrite)
            
            # Nettoyer le fichier temporaire
            os.unlink(temp_path)
            
            if success:
                if ui_logger:
                    ui_logger.info(f"Étude importée avec succès depuis {filename}")
                return html.Div([
                    html.I(className="bi bi-check-circle-fill me-2 text-success"),
                    f"Fichier {filename} importé avec succès"
                ]), True, "Succès", f"L'étude a été importée avec succès depuis {filename}"
            else:
                return html.Div([
                    html.I(className="bi bi-exclamation-triangle-fill me-2 text-warning"),
                    f"Échec de l'importation de {filename}"
                ]), True, "Erreur", "Échec de l'importation de l'étude. Vérifiez le format du fichier."
                
        except Exception as e:
            if ui_logger:
                ui_logger.error(f"Erreur lors de l'importation du fichier: {str(e)}")
            return html.Div([
                html.I(className="bi bi-exclamation-triangle-fill me-2 text-danger"),
                f"Erreur: {str(e)}"
            ]), True, "Erreur", f"Erreur lors de l'importation: {str(e)}"
    
    # Callback pour le bouton d'importation
    @app.callback(
        [Output("notification-modal", "is_open", allow_duplicate=True),
         Output("notification-title", "children", allow_duplicate=True),
         Output("notification-message", "children", allow_duplicate=True)],
        Input("import-study-button", "n_clicks"),
        State("upload-study-json", "contents"),
        prevent_initial_call=True
    )
    def import_study_from_upload(n_clicks, contents):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update
        
        if not contents:
            return True, "Attention", "Aucun fichier n'a été téléchargé. Veuillez d'abord sélectionner un fichier JSON."
        
        # L'importation proprement dite est gérée par le callback process_uploaded_file
        return dash.no_update, dash.no_update, dash.no_update
    
    # Callback pour les boutons d'action sur les études
    @app.callback(
        [Output("confirmation-modal", "is_open"),
         Output("confirmation-message", "children"),
         Output("confirm-action-button", "n_clicks"),
         Output("confirm-action-button", "id")],
        [Input({"type": "delete-study-button", "index": ALL}, "n_clicks"),
         Input({"type": "archive-study-button", "index": ALL}, "n_clicks"),
         Input({"type": "clone-study-button", "index": ALL}, "n_clicks")],
        prevent_initial_call=True
    )
    def show_confirmation_for_study_action(delete_clicks, archive_clicks, clone_clicks):
        # Déterminer quel bouton a été cliqué
        ctx_triggered = dash.callback_context.triggered_id
        
        if not ctx_triggered:
            return False, "", 0, {"type": "confirm-action-button", "action": "none", "study": ""}
        
        study_name = ctx_triggered.get("index", "")
        action_type = ctx_triggered.get("type", "")
        
        if not study_name:
            return False, "", 0, {"type": "confirm-action-button", "action": "none", "study": ""}
        
        if action_type == "delete-study-button":
            message = f"Êtes-vous sûr de vouloir supprimer l'étude '{study_name}' ? Cette action est irréversible."
            action = "delete"
        elif action_type == "archive-study-button":
            message = f"Êtes-vous sûr de vouloir archiver l'étude '{study_name}' ?"
            action = "archive"
        elif action_type == "clone-study-button":
            message = f"Voulez-vous créer une copie de l'étude '{study_name}' ?"
            action = "clone"
        else:
            return False, "", 0, {"type": "confirm-action-button", "action": "none", "study": ""}
        
        return True, message, 0, {"type": "confirm-action-button", "action": action, "study": study_name}
    
    # Callback pour gérer la confirmation des actions sur les études
    @app.callback(
        [Output("notification-modal", "is_open", allow_duplicate=True),
         Output("notification-title", "children", allow_duplicate=True),
         Output("notification-message", "children", allow_duplicate=True),
         Output("confirmation-modal", "is_open", allow_duplicate=True)],
        Input({"type": "confirm-action-button", "action": ALL, "study": ALL}, "n_clicks"),
        State({"type": "confirm-action-button", "action": ALL, "study": ALL}, "id"),
        prevent_initial_call=True
    )
    def handle_study_action_confirmation(n_clicks, button_id):
        if not n_clicks or not button_id or n_clicks[0] is None or n_clicks[0] == 0:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        if not study_manager:
            return True, "Erreur", "Gestionnaire d'études non disponible", False
        
        # Extraire l'action et le nom de l'étude
        action = button_id[0].get("action")
        study_name = button_id[0].get("study")
        
        if action == "none" or not study_name:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        try:
            if action == "delete":
                # Supprimer l'étude
                success = study_manager.delete_study(study_name)
                
                if success:
                    if ui_logger:
                        ui_logger.info(f"Étude supprimée: {study_name}")
                    return True, "Succès", f"L'étude '{study_name}' a été supprimée avec succès", False
                else:
                    return True, "Erreur", f"Erreur lors de la suppression de l'étude '{study_name}'", False
                    
            elif action == "archive":
                # Archiver l'étude
                success = study_manager.update_study_status(study_name, StudyStatus.ARCHIVED)
                
                if success:
                    if ui_logger:
                        ui_logger.info(f"Étude archivée: {study_name}")
                    return True, "Succès", f"L'étude '{study_name}' a été archivée avec succès", False
                else:
                    return True, "Erreur", f"Erreur lors de l'archivage de l'étude '{study_name}'", False
                    
            elif action == "clone":
                # Cloner l'étude
                new_study_name = f"{study_name}_copy_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                success = study_manager.clone_study(study_name, new_study_name)
                
                if success:
                    if ui_logger:
                        ui_logger.info(f"Étude clonée: {study_name} -> {new_study_name}")
                    return True, "Succès", f"L'étude '{study_name}' a été clonée avec succès sous le nom '{new_study_name}'", False
                else:
                    return True, "Erreur", f"Erreur lors du clonage de l'étude '{study_name}'", False
            
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
                
        except Exception as e:
            if ui_logger:
                ui_logger.error(f"Erreur lors de l'action '{action}' sur l'étude '{study_name}': {str(e)}")
            return True, "Erreur", f"Erreur: {str(e)}", False
    
    # Callback pour annuler la confirmation
    @app.callback(
        Output("confirmation-modal", "is_open", allow_duplicate=True),
        Input("cancel-confirmation-button", "n_clicks"),
        prevent_initial_call=True
    )
    def cancel_confirmation(n_clicks):
        if n_clicks:
            return False
        return dash.no_update
    
    # Callback pour les autres boutons d'action sur les études
    @app.callback(
        [Output("notification-modal", "is_open", allow_duplicate=True),
         Output("notification-title", "children", allow_duplicate=True),
         Output("notification-message", "children", allow_duplicate=True)],
        [Input({"type": "export-study-button", "index": ALL}, "n_clicks"),
         Input({"type": "generate-code-button", "index": ALL}, "n_clicks"),
         Input({"type": "copy-config-button", "index": ALL}, "n_clicks"),
         Input({"type": "edit-study-button", "index": ALL}, "n_clicks")],
        prevent_initial_call=True
    )
    def handle_study_export_actions(export_clicks, generate_clicks, copy_clicks, edit_clicks):
        # Déterminer quel bouton a été cliqué
        ctx_triggered = dash.callback_context.triggered_id
        
        if not ctx_triggered:
            return dash.no_update, dash.no_update, dash.no_update
        
        study_name = ctx_triggered.get("index", "")
        action_type = ctx_triggered.get("type", "")
        
        if not study_name or not study_manager:
            return dash.no_update, dash.no_update, dash.no_update
        
        try:
            if action_type == "export-study-button":
                # Exporter l'étude au format JSON
                os.makedirs("exports", exist_ok=True)
                export_path = os.path.join("exports", f"{study_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
                
                # Exporter l'étude
                export_path = study_manager.export_study_to_json(study_name, export_path)
                
                if export_path:
                    if ui_logger:
                        ui_logger.info(f"Étude exportée: {study_name} -> {export_path}")
                    return True, "Succès", f"L'étude '{study_name}' a été exportée avec succès vers '{export_path}'"
                else:
                    return True, "Erreur", f"Erreur lors de l'exportation de l'étude '{study_name}'"
                    
            elif action_type == "generate-code-button":
                # Génération de code (implémentation simplifiée)
                return True, "Information", "La génération de code n'est pas encore implémentée"
                    
            elif action_type == "copy-config-button":
                # Copier la configuration dans le presse-papier
                return True, "Information", f"Configuration de l'étude '{study_name}' prête à être copiée (non implémenté côté serveur)"
                    
            elif action_type == "edit-study-button":
                # Éditer l'étude (ouvrirait normalement un formulaire)
                return True, "Information", "L'édition d'étude n'est pas encore implémentée"
            
            return dash.no_update, dash.no_update, dash.no_update
                
        except Exception as e:
            if ui_logger:
                ui_logger.error(f"Erreur lors de l'action '{action_type}' sur l'étude '{study_name}': {str(e)}")
            return True, "Erreur", f"Erreur: {str(e)}"

    # Callback pour le changement de paramètre à visualiser
    @app.callback(
        [Output("param-distribution-graph", "figure"),
         Output("param-stats-container", "children")],
        [Input("param-select", "value")],
        [State("selected-studies-store", "data")]
    )
    def update_param_visualization(param_name, selected_studies_data):
        if not param_name or not selected_studies_data:
            return dash.no_update, dash.no_update
        
        try:
            # Collecter les noms d'études et les valeurs de paramètres
            study_names = []
            param_values = {}
            
            for study in selected_studies_data:
                metadata = study.get('metadata', {})
                performance = study.get('performance', {})
                
                study_name = metadata.get('study_name', '')
                study_names.append(study_name)
                
                # Collecter les paramètres
                best_params = performance.get('best_params', {}) if performance else {}
                for p_name, p_value in best_params.items():
                    if p_name not in param_values:
                        param_values[p_name] = []
                    param_values[p_name].append((study_name, p_value))
            
            # Mettre à jour la figure et les statistiques
            fig = create_param_distribution_figure(param_name, param_values, study_names)
            stats = create_param_stats_card(param_name, param_values)
            
            return fig, stats
            
        except Exception as e:
            if ui_logger:
                ui_logger.error(f"Erreur lors de la mise à jour de la visualisation des paramètres: {str(e)}")
            return dash.no_update, dash.no_update
    
    # Callback pour le calcul du score agrégé avec les pondérations
    @app.callback(
        Output("aggregate-scores-container", "children"),
        [Input("apply-weights-button", "n_clicks")],
        [State("roi-weight-slider", "value"),
         State("winrate-weight-slider", "value"),
         State("drawdown-weight-slider", "value"),
         State("profitfactor-weight-slider", "value"),
         State("selected-studies-store", "data")]
    )
    def calculate_aggregate_scores(n_clicks, roi_weight, winrate_weight, drawdown_weight, profitfactor_weight, selected_studies_data):
        if not n_clicks or not selected_studies_data:
            return dash.no_update
        
        try:
            import numpy as np
            
            # Normalisation des pondérations
            total_weight = roi_weight + winrate_weight + drawdown_weight + profitfactor_weight
            if total_weight == 0:
                return html.Div("La somme des pondérations est nulle. Veuillez ajuster les valeurs.", className="text-warning")
            
            # Calcul des scores pour chaque étude
            scores = []
            
            for study in selected_studies_data:
                metadata = study.get('metadata', {})
                performance = study.get('performance', {})
                
                study_name = metadata.get('study_name', '')
                
                if performance:
                    # Récupération des métriques
                    roi = performance.get('roi', 0)
                    win_rate = performance.get('win_rate', 0)
                    # Inverser le drawdown pour qu'une valeur plus faible soit meilleure
                    max_drawdown = 1 / (performance.get('max_drawdown', 0.01) + 0.01)  # Éviter la division par zéro
                    profit_factor = performance.get('profit_factor', 0)
                    
                    # Normalisation des métriques
                    roi_norm = np.tanh(roi * 5) if roi_weight > 0 else 0  # Utiliser tanh pour limiter l'influence des valeurs extrêmes
                    winrate_norm = win_rate if winrate_weight > 0 else 0
                    drawdown_norm = np.tanh(max_drawdown) if drawdown_weight > 0 else 0
                    profitfactor_norm = np.tanh(profit_factor - 1) if profitfactor_weight > 0 else 0  # -1 car PF=1 est neutre
                    
                    # Calcul du score agrégé
                    score = (
                        roi_weight * roi_norm +
                        winrate_weight * winrate_norm +
                        drawdown_weight * drawdown_norm +
                        profitfactor_weight * profitfactor_norm
                    ) / total_weight
                    
                    scores.append({
                        "study_name": study_name,
                        "score": score,
                        "roi": roi,
                        "win_rate": win_rate,
                        "max_drawdown": performance.get('max_drawdown', 0),
                        "profit_factor": profit_factor
                    })
            
            # Tri des scores
            scores.sort(key=lambda x: x["score"], reverse=True)
            
            # Création du tableau des scores
            rows = []
            for i, score_data in enumerate(scores):
                rows.append(html.Tr([
                    html.Td(f"{i+1}"),
                    html.Td(score_data["study_name"]),
                    html.Td(f"{score_data['score']*100:.2f}", className="font-weight-bold text-info"),
                    html.Td(f"{score_data['roi']*100:.2f}%", className=f"{'text-success' if score_data['roi'] >= 0 else 'text-danger'}"),
                    html.Td(f"{score_data['win_rate']*100:.2f}%"),
                    html.Td(f"{score_data['max_drawdown']*100:.2f}%"),
                    html.Td(f"{score_data['profit_factor']:.2f}")
                ]))
            
            return html.Div([
                html.P(f"Pondérations appliquées: ROI ({roi_weight}), Win Rate ({winrate_weight}), Max DD ({drawdown_weight}), Profit Factor ({profitfactor_weight})", 
                      className="text-muted mb-2"),
                html.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("#"),
                            html.Th("Étude"),
                            html.Th("Score"),
                            html.Th("ROI"),
                            html.Th("Win Rate"),
                            html.Th("Max DD"),
                            html.Th("PF")
                        ])
                    ]),
                    html.Tbody(rows)
                ], className="retro-table w-100")
            ])
            
        except Exception as e:
            if ui_logger:
                ui_logger.error(f"Erreur lors du calcul des scores agrégés: {str(e)}")
            return html.Div(f"Erreur lors du calcul: {str(e)}", className="text-danger")

def register_studies_callbacks(app, central_logger=None):
    """
    Enregistre les callbacks pour la page des études
    
    Args:
        app: L'instance de l'application Dash
        central_logger: Instance du logger centralisé
    """
    # Initialiser le logger
    ui_logger = None
    if central_logger:
        ui_logger = central_logger.get_logger("studies_page", LoggerType.UI)
    
    # Initialiser le gestionnaire d'études
    try:
        study_manager = StudyManager("data/studies.db")
        if ui_logger:
            ui_logger.info("Gestionnaire d'études initialisé avec succès pour les callbacks")
    except Exception as e:
        if ui_logger:
            ui_logger.error(f"Erreur lors de l'initialisation du gestionnaire d'études: {str(e)}")
            ui_logger.error(traceback.format_exc())
        study_manager = None
    
    # Callback pour la mise à jour des statistiques
    @app.callback(
        [Output("total-studies-count", "children"),
         Output("completed-studies-count", "children"),
         Output("in-progress-studies-count", "children"),
         Output("assets-count", "children")],
        [Input("studies-refresh-interval", "n_intervals"),
         Input("refresh-stats-button", "n_clicks")]
    )
    def update_studies_stats(n_intervals, n_clicks):
        if not study_manager:
            return "N/A", "N/A", "N/A", "N/A"
        
        try:
            # Récupérer toutes les études
            all_studies = study_manager.list_studies(limit=1000)
            
            # Total des études
            total_studies = len(all_studies)
            
            # Études par statut
            completed_studies = sum(1 for study in all_studies if study.get('status') == StudyStatus.COMPLETED.value)
            in_progress_studies = sum(1 for study in all_studies if study.get('status') == StudyStatus.IN_PROGRESS.value)
            
            # Actifs uniques
            unique_assets = set(study.get('asset', '') for study in all_studies)
            assets_count = len([asset for asset in unique_assets if asset])
            
            return total_studies, completed_studies, in_progress_studies, assets_count
            
        except Exception as e:
            if ui_logger:
                ui_logger.error(f"Erreur lors de la mise à jour des statistiques: {str(e)}")
                ui_logger.error(traceback.format_exc())
            return "N/A", "N/A", "N/A", "N/A"
    
    # Callback pour les meilleures études
    @app.callback(
        Output("top-studies-container", "children"),
        [Input("top-studies-metric-select", "value"),
         Input("studies-refresh-interval", "n_intervals"),
         Input("refresh-stats-button", "n_clicks")]
    )
    def update_top_studies(metric, n_intervals, n_clicks):
        if not study_manager:
            return html.Div("Gestionnaire d'études non disponible", className="text-danger")
        
        try:
            # Déterminer la métrique d'évaluation
            eval_metric = None
            if metric == "roi":
                eval_metric = StudyEvaluationMetric.ROI
            elif metric == "win_rate":
                eval_metric = StudyEvaluationMetric.WIN_RATE
            elif metric == "profit_factor":
                eval_metric = StudyEvaluationMetric.PROFIT_FACTOR
            else:
                eval_metric = StudyEvaluationMetric.COMBINED_SCORE
            
            # Récupérer les meilleures études
            top_studies = study_manager.get_top_performing_studies(metric=eval_metric, limit=5)
            
            if not top_studies:
                return html.Div("Aucune étude disponible", className="text-muted text-center py-4")
            
            # Création de la liste des études
            studies_list = []
            for study in top_studies:
                metadata = study.get('metadata', {})
                performance = study.get('performance', {})
                score = study.get('score', 0)
                
                # Calcul de la valeur à afficher selon la métrique
                if metric == "roi":
                    value = f"{performance.get('roi', 0) * 100:.2f}%"
                    color = "success" if performance.get('roi', 0) >= 0 else "danger"
                elif metric == "win_rate":
                    value = f"{performance.get('win_rate', 0) * 100:.2f}%"
                    color = "info"
                elif metric == "profit_factor":
                    value = f"{performance.get('profit_factor', 0):.2f}"
                    color = "warning"
                else:
                    value = f"{performance.get('best_value', 0):.2f}"
                    color = "primary"
                
                # Création de l'élément de liste
                studies_list.append(
                    dbc.ListGroupItem([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Strong(metadata.get('study_name', '')),
                                    html.Div([
                                        html.Small(f"{metadata.get('asset', '')} {metadata.get('timeframe', '')}", 
                                               className="text-muted")
                                    ])
                                ])
                            ], width=8),
                            dbc.Col([
                                html.Div([
                                    html.Strong(value, className=f"text-{color}"),
                                    html.Div([
                                        html.Small(f"{performance.get('total_trades', 0)} trades", 
                                               className="text-muted")
                                    ])
                                ], className="text-end")
                            ], width=4)
                        ])
                    ], action=True, href="#", id={"type": "top-study-item", "index": metadata.get('study_name', '')})
                )
            
            return dbc.ListGroup(studies_list)
            
        except Exception as e:
            if ui_logger:
                ui_logger.error(f"Erreur lors de la mise à jour des meilleures études: {str(e)}")
                ui_logger.error(traceback.format_exc())
            return html.Div(f"Erreur: {str(e)}", className="text-danger")
    
    # Callback pour les études récentes
    @app.callback(
        Output("recent-studies-container", "children"),
        [Input("studies-refresh-interval", "n_intervals"),
         Input("refresh-stats-button", "n_clicks")]
    )
    def update_recent_studies(n_intervals, n_clicks):
        if not study_manager:
            return html.Div("Gestionnaire d'études non disponible", className="text-danger")
        
        try:
            # Récupérer toutes les études
            all_studies = study_manager.list_studies(limit=1000)
            
            # Trier par date de dernière modification
            recent_studies = sorted(
                all_studies, 
                key=lambda s: s.get('last_modified', ''),
                reverse=True
            )[:5]
            
            if not recent_studies:
                return html.Div("Aucune étude disponible", className="text-muted text-center py-4")
            
            # Création de la liste des études
            studies_list = []
            for study in recent_studies:
                status = study.get('status', '')
                status_color = {
                    StudyStatus.COMPLETED.value: "success",
                    StudyStatus.IN_PROGRESS.value: "info",
                    StudyStatus.FAILED.value: "danger",
                    StudyStatus.ARCHIVED.value: "secondary"
                }.get(status, "secondary")
                
                # Création de l'élément de liste
                studies_list.append(
                    dbc.ListGroupItem([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Strong(study.get('study_name', '')),
                                    html.Div([
                                        html.Small(f"{study.get('asset', '')} {study.get('timeframe', '')}", 
                                               className="text-muted")
                                    ])
                                ])
                            ], width=8),
                            dbc.Col([
                                html.Div([
                                    html.Span(status, className=f"badge bg-{status_color}"),
                                    html.Div([
                                        html.Small(study.get('last_modified', '').split(' ')[0], 
                                               className="text-muted")
                                    ])
                                ], className="text-end")
                            ], width=4)
                        ])
                    ], action=True, href="#", id={"type": "recent-study-item", "index": study.get('study_name', '')})
                )
            
            return dbc.ListGroup(studies_list)
            
        except Exception as e:
            if ui_logger:
                ui_logger.error(f"Erreur lors de la mise à jour des études récentes: {str(e)}")
                ui_logger.error(traceback.format_exc())
            return html.Div(f"Erreur: {str(e)}", className="text-danger")
    
    # Callback pour les graphiques de performance
    @app.callback(
        Output("performance-chart-container", "children"),
        [Input("performance-chart-select", "value"),
         Input("studies-refresh-interval", "n_intervals"),
         Input("refresh-stats-button", "n_clicks")]
    )
    def update_performance_chart(chart_type, n_intervals, n_clicks):
        if not study_manager:
            return html.Div("Gestionnaire d'études non disponible", className="text-danger")
        
        try:
            # Récupérer les études avec statut COMPLETED
            all_studies = study_manager.list_studies(status=StudyStatus.COMPLETED, limit=1000)
            
            if not all_studies:
                return html.Div("Aucune étude complétée disponible", className="text-muted text-center py-4")
            
            # Collecter les données de performance
            study_data = []
            for study in all_studies:
                # Récupérer les détails de la performance
                performance = study_manager.get_study_performance(study.get('study_name', ''))
                if performance:
                    study_data.append({
                        'study_name': study.get('study_name', ''),
                        'asset': study.get('asset', ''),
                        'timeframe': study.get('timeframe', ''),
                        'creation_date': study.get('creation_date', '').split(' ')[0],
                        'roi': performance.roi * 100,
                        'win_rate': performance.win_rate * 100,
                        'max_drawdown': performance.max_drawdown * 100,
                        'total_trades': performance.total_trades,
                        'profit_factor': performance.profit_factor
                    })
            
            if not study_data:
                return html.Div("Aucune donnée de performance disponible", className="text-muted text-center py-4")
            
            # Création du graphique selon le type sélectionné
            fig = None
            
            if chart_type == "roi_by_asset":
                # Regroupement des données par actif
                asset_data = {}
                for study in study_data:
                    asset = study['asset']
                    if asset not in asset_data:
                        asset_data[asset] = []
                    asset_data[asset].append(study['roi'])
                
                # Calcul des moyennes et médianes
                asset_means = {asset: np.mean(values) for asset, values in asset_data.items()}
                asset_medians = {asset: np.median(values) for asset, values in asset_data.items()}
                
                # Création du graphique
                fig = go.Figure()
                
                # Barres pour les moyennes
                fig.add_trace(go.Bar(
                    x=list(asset_means.keys()),
                    y=list(asset_means.values()),
                    name='ROI Moyen',
                    marker_color='#4ADE80'
                ))
                
                # Ligne pour les médianes
                fig.add_trace(go.Scatter(
                    x=list(asset_medians.keys()),
                    y=list(asset_medians.values()),
                    name='ROI Médian',
                    mode='markers+lines',
                    marker=dict(size=10),
                    line=dict(width=2, dash='dot'),
                    marker_color='#F59E0B'
                ))
                
                # Mise en page
                fig.update_layout(
                    title="ROI Moyen et Médian par Actif",
                    xaxis_title="Actif",
                    yaxis_title="ROI (%)",
                    height=500
                )
                
            elif chart_type == "win_rate_by_timeframe":
                # Regroupement des données par timeframe
                timeframe_data = {}
                for study in study_data:
                    timeframe = study['timeframe']
                    if timeframe not in timeframe_data:
                        timeframe_data[timeframe] = []
                    timeframe_data[timeframe].append(study['win_rate'])
                
                # Calcul des moyennes
                timeframe_means = {tf: np.mean(values) for tf, values in timeframe_data.items()}
                
                # Tri des timeframes (conversion en minutes pour le tri)
                def tf_to_minutes(tf):
                    if 'm' in tf:
                        return int(tf.replace('m', ''))
                    elif 'h' in tf:
                        return int(tf.replace('h', '')) * 60
                    elif 'd' in tf:
                        return int(tf.replace('d', '')) * 1440
                    else:
                        return 0
                
                sorted_timeframes = sorted(timeframe_means.keys(), key=tf_to_minutes)
                
                # Création du graphique
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=sorted_timeframes,
                    y=[timeframe_means[tf] for tf in sorted_timeframes],
                    marker_color='#22D3EE',
                    text=[f"{timeframe_means[tf]:.1f}%" for tf in sorted_timeframes],
                    textposition='auto'
                ))
                
                # Mise en page
                fig.update_layout(
                    title="Win Rate Moyen par Timeframe",
                    xaxis_title="Timeframe",
                    yaxis_title="Win Rate (%)",
                    height=500
                )
                
            elif chart_type == "trades_distribution":
                # Préparation des données
                trade_counts = [study['total_trades'] for study in study_data]
                
                # Création des bins pour l'histogramme
                max_trades = max(trade_counts) if trade_counts else 100
                bin_size = max(1, max_trades // 20)  # 20 bins maximum
                
                # Création du graphique
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=trade_counts,
                    nbinsx=20,
                    marker_color='#A78BFA',
                    opacity=0.7
                ))
                
                # Moyenne et médiane
                mean_trades = np.mean(trade_counts)
                median_trades = np.median(trade_counts)
                
                # Ajout des lignes verticales pour la moyenne et la médiane
                fig.add_vline(x=mean_trades, line_dash="dash", line_color="#F59E0B",
                             annotation_text=f"Moyenne: {mean_trades:.1f}", annotation_position="top right")
                fig.add_vline(x=median_trades, line_dash="dash", line_color="#22D3EE",
                             annotation_text=f"Médiane: {median_trades:.1f}", annotation_position="top left")
                
                # Mise en page
                fig.update_layout(
                    title="Distribution du Nombre de Trades par Étude",
                    xaxis_title="Nombre de Trades",
                    yaxis_title="Nombre d'Études",
                    bargap=0.1,
                    height=500
                )
                
            elif chart_type == "time_evolution":
                # Création d'un DataFrame pour faciliter l'analyse
                import pandas as pd
                
                # Convertir les dates de création en objets datetime
                df = pd.DataFrame(study_data)
                df['creation_date'] = pd.to_datetime(df['creation_date'])
                df = df.sort_values('creation_date')
                
                # Calcul des moyennes mobiles
                window_size = max(1, len(df) // 10)  # 10% des données
                df['roi_ma'] = df['roi'].rolling(window=window_size, min_periods=1).mean()
                df['win_rate_ma'] = df['win_rate'].rolling(window=window_size, min_periods=1).mean()
                
                # Création du graphique
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Points de ROI individuels
                fig.add_trace(
                    go.Scatter(
                        x=df['creation_date'],
                        y=df['roi'],
                        mode='markers',
                        name='ROI',
                        marker=dict(
                            size=8,
                            color=df['roi'],
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title="ROI (%)"),
                            cmin=-50,
                            cmax=50
                        )
                    ),
                    secondary_y=False
                )
                
                # Moyenne mobile du ROI
                fig.add_trace(
                    go.Scatter(
                        x=df['creation_date'],
                        y=df['roi_ma'],
                        mode='lines',
                        name='ROI (Moyenne Mobile)',
                        line=dict(width=3, color='#4ADE80')
                    ),
                    secondary_y=False
                )
                
                # Moyenne mobile du Win Rate
                fig.add_trace(
                    go.Scatter(
                        x=df['creation_date'],
                        y=df['win_rate_ma'],
                        mode='lines',
                        name='Win Rate (Moyenne Mobile)',
                        line=dict(width=3, color='#22D3EE')
                    ),
                    secondary_y=True
                )
                
                # Mise en page
                fig.update_layout(
                    title="Évolution Temporelle des Performances",
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    hovermode="x unified"
                )
                
                # Update des axes
                fig.update_xaxes(title_text="Date de création")
                fig.update_yaxes(title_text="ROI (%)", secondary_y=False)
                fig.update_yaxes(title_text="Win Rate (%)", secondary_y=True)
            
            # Configuration commune
            if fig:
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='#111827',
                    plot_bgcolor='#1F2937',
                    font=dict(family='"Share Tech Mono", monospace'),
                    margin=dict(t=50, b=40, l=40, r=40)
                )
                
                return dcc.Graph(figure=fig, config={'displayModeBar': False}, style={"height": "100%"})
            else:
                return html.Div("Type de graphique non reconnu", className="text-danger")
            
        except Exception as e:
            if ui_logger:
                ui_logger.error(f"Erreur lors de la mise à jour du graphique de performance: {str(e)}")
                ui_logger.error(traceback.format_exc())
            return html.Div(f"Erreur: {str(e)}", className="text-danger")
    
    # Callback pour filtrer les études
    @app.callback(
        [Output("studies-table-container", "children"),
         Output("studies-data-store", "data")],
        [Input("apply-filters-button", "n_clicks"),
         Input("refresh-search-button", "n_clicks")],
        [State("asset-filter-select", "value"),
         State("timeframe-filter-select", "value"),
         State("status-filter-select", "value"),
         State("metric-filter-select", "value"),
         State("tags-filter-dropdown", "value"),
         State("search-filter-input", "value")]
    )
    def filter_studies(n_clicks, refresh_clicks, asset, timeframe, status, metric, tags, search):
        if not study_manager:
            return html.Div("Gestionnaire d'études non disponible", className="text-danger"), []
        
        # Vérifier si le callback a été déclenché
        if not ctx.triggered_id:
            # Charge toutes les études au chargement initial
            all_studies = study_manager.list_studies(limit=1000)
            studies_with_performance = []
            
            for study in all_studies:
                # Récupérer la performance
                performance = study_manager.get_study_performance(study.get('study_name', ''))
                studies_with_performance.append({
                    'metadata': study,
                    'performance': performance.to_dict() if performance else None
                })
            
            # Création du tableau
            return dbc.Card([
                dbc.CardHeader([
                    html.H3("RÉSULTATS DE RECHERCHE", className="card-title"),
                    html.Span(f"{len(studies_with_performance)} études trouvées", className="badge bg-info text-dark ms-2")
                ], className="d-flex align-items-center"),
                dbc.CardBody([
                    create_formatted_table(studies_with_performance)
                ])
            ], className="retro-card"), studies_with_performance
        
        try:
            # Déterminer le statut à filtrer
            status_filter = None if status == "all" else status
            
            # Récupérer toutes les études
            all_studies = study_manager.list_studies(
                status=StudyStatus(status_filter) if status_filter else None,
                limit=1000
            )
            
            # Filtrage côté client
            filtered_studies = []
            for study in all_studies:
                # Filtrage par actif
                if asset != "all" and study.get('asset', '') != asset:
                    continue
                
                # Filtrage par timeframe
                if timeframe != "all" and study.get('timeframe', '') != timeframe:
                    continue
                
                # Filtrage par tags
                if tags:
                    study_tags = set(study.get('tags', []))
                    if not all(tag in study_tags for tag in tags):
                        continue
                
                # Filtrage par recherche de texte
                if search:
                    study_name = study.get('study_name', '').lower()
                    if search.lower() not in study_name:
                        continue
                
                # Récupérer la performance
                performance = study_manager.get_study_performance(study.get('study_name', ''))
                filtered_studies.append({
                    'metadata': study,
                    'performance': performance.to_dict() if performance else None
                })
            
            # Tri par métrique d'évaluation
            if metric:
                if metric == StudyEvaluationMetric.COMBINED_SCORE.value:
                    # Tri par score combiné (best_value)
                    filtered_studies.sort(
                        key=lambda s: s.get('performance', {}).get('best_value', 0) 
                        if s.get('performance') else 0,
                        reverse=True
                    )
                elif metric == StudyEvaluationMetric.ROI.value:
                    # Tri par ROI
                    filtered_studies.sort(
                        key=lambda s: s.get('performance', {}).get('roi', 0) 
                        if s.get('performance') else 0,
                        reverse=True
                    )
                elif metric == StudyEvaluationMetric.WIN_RATE.value:
                    # Tri par Win Rate
                    filtered_studies.sort(
                        key=lambda s: s.get('performance', {}).get('win_rate', 0) 
                        if s.get('performance') else 0,
                        reverse=True
                    )
                elif metric == StudyEvaluationMetric.PROFIT_FACTOR.value:
                    # Tri par Profit Factor
                    filtered_studies.sort(
                        key=lambda s: s.get('performance', {}).get('profit_factor', 0) 
                        if s.get('performance') else 0,
                        reverse=True
                    )
            
            # Création du tableau
            return dbc.Card([
                dbc.CardHeader([
                    html.H3("RÉSULTATS DE RECHERCHE", className="card-title"),
                    html.Span(f"{len(filtered_studies)} études trouvées", className="badge bg-info text-dark ms-2")
                ], className="d-flex align-items-center"),
                dbc.CardBody([
                    create_formatted_table(filtered_studies)
                ])
            ], className="retro-card"), filtered_studies
            
        except Exception as e:
            if ui_logger:
                ui_logger.error(f"Erreur lors du filtrage des études: {str(e)}")
                ui_logger.error(traceback.format_exc())
            return html.Div(f"Erreur: {str(e)}", className="text-danger"), []
    
    # Callback pour la sélection d'études dans le tableau
    @app.callback(
        Output("selected-studies-store", "data"),
        [Input("studies-table", "selected_rows")],
        [State("studies-data-store", "data")]
    )
    def update_selected_studies(selected_rows, studies_data):
        if not selected_rows or not studies_data:
            return []
        
        selected_studies = []
        for idx in selected_rows:
            if idx < len(studies_data):
                selected_studies.append(studies_data[idx])
        
        return selected_studies
    
    # Callback pour la comparaison des études sélectionnées
    @app.callback(
        Output("studies-comparison-container", "children"),
        [Input("compare-selected-button", "n_clicks")],
        [State("selected-studies-store", "data")]
    )
    def compare_selected_studies(n_clicks, selected_studies):
        if not n_clicks or not selected_studies:
            return html.Div()
        
        return create_enhanced_comparison_view(selected_studies)
    
    # Callback pour voir les détails d'une étude depuis le tableau
    @app.callback(
        [Output("study-details-container", "children"),
         Output("active-study-store", "data"),
         Output("studies-main-tabs", "active_tab")],
        [Input("studies-table", "active_cell"),
         Input({"type": "top-study-item", "index": ALL}, "n_clicks"),
         Input({"type": "recent-study-item", "index": ALL}, "n_clicks")],
        [State("studies-data-store", "data"),
         State("studies-main-tabs", "active_tab")]
    )
    def view_study_details(active_cell, top_studies_clicks, recent_studies_clicks, studies_data, current_tab):
        # Déterminer ce qui a déclenché le callback
        trigger = ctx.triggered_id
        
        if not trigger:
            return html.Div(), None, current_tab
        
        study_name = None
        
        # Cas 1: Clic sur une cellule du tableau
        if trigger == "studies-table" and active_cell:
            row = active_cell['row']
            if studies_data and row < len(studies_data):
                study_name = studies_data[row].get('metadata', {}).get('study_name', '')
        
        # Cas 2: Clic sur une étude dans la liste des meilleures
        elif isinstance(trigger, dict) and trigger.get('type') == 'top-study-item':
            study_name = trigger.get('index')
        
        # Cas 3: Clic sur une étude dans la liste des récentes
        elif isinstance(trigger, dict) and trigger.get('type') == 'recent-study-item':
            study_name = trigger.get('index')
        
        # Si on a trouvé un nom d'étude
        if study_name and study_manager:
            try:
                # Récupérer les métadonnées et performances
                metadata = study_manager.get_study_metadata(study_name)
                performance = study_manager.get_study_performance(study_name)
                
                if metadata:
                    # Créer la vue détaillée
                    details_content = create_study_details_view(metadata, performance)
                    return details_content, study_name, "details"
                
            except Exception as e:
                if ui_logger:
                    ui_logger.error(f"Erreur lors de la récupération des détails de l'étude: {str(e)}")
                    ui_logger.error(traceback.format_exc())
                return html.Div(f"Erreur: {str(e)}", className="text-danger"), None, current_tab
        
        return html.Div(), None, current_tab
    
    # Callback pour créer une nouvelle étude
    @app.callback(
        [Output("notification-modal", "is_open"),
         Output("notification-title", "children"),
         Output("notification-message", "children")],
        [Input("create-study-button", "n_clicks")],
        [State("study-name-input", "value"),
         State("study-asset-select", "value"),
         State("study-timeframe-select", "value"),
         State("study-start-date-input", "value"),
         State("study-end-date-input", "value"),
         State("study-metric-select", "value"),
         State("study-description-textarea", "value"),
         State("study-tags-input", "value")]
    )
    def create_new_study(n_clicks, study_name, asset, timeframe, start_date, end_date, metric, description, tags):
        if not n_clicks:
            return False, "", ""
        
        if not study_manager:
            return True, "Erreur", "Gestionnaire d'études non disponible"
        
        try:
            # Validation des champs requis
            if not study_name:
                return True, "Erreur", "Le nom de l'étude est requis"
                
            if not asset:
                return True, "Erreur", "L'actif est requis"
                
            if not timeframe:
                return True, "Erreur", "Le timeframe est requis"
            
            # Vérifier si l'étude existe déjà
            existing_metadata = study_manager.get_study_metadata(study_name)
            if existing_metadata:
                return True, "Erreur", f"L'étude '{study_name}' existe déjà"
            
            # Préparation des tags
            tag_list = []
            if tags:
                tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
            
            # Création de l'étude
            metadata = StudyMetadata(
                study_name=study_name,
                asset=asset,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date if end_date else None,
                evaluation_metric=StudyEvaluationMetric(metric),
                description=description if description else "",
                tags=tag_list
            )
            
            # Enregistrement de l'étude
            success = study_manager.create_study(metadata)
            
            if success:
                if ui_logger:
                    ui_logger.info(f"Nouvelle étude créée: {study_name}")
                return True, "Succès", f"L'étude '{study_name}' a été créée avec succès"
            else:
                return True, "Erreur", f"Erreur lors de la création de l'étude"
                
        except Exception as e:
            if ui_logger:
                ui_logger.error(f"Erreur lors de la création de l'étude: {str(e)}")
                ui_logger.error(traceback.format_exc())
            return True, "Erreur", f"Erreur: {str(e)}"

# Point d'entrée principal pour enregistrer tous les callbacks
def register_all_studies_callbacks(app, central_logger=None):
    """
    Enregistre tous les callbacks pour la page des études
    
    Args:
        app: L'instance de l'application Dash
        central_logger: Instance du logger centralisé
    """
    # Initialiser le gestionnaire d'études
    try:
        study_manager = StudyManager("data/studies.db")
        if central_logger:
            ui_logger = central_logger.get_logger("studies_page", LoggerType.UI)
            ui_logger.info("Gestionnaire d'études initialisé avec succès pour les callbacks")
    except Exception as e:
        if central_logger:
            ui_logger = central_logger.get_logger("studies_page", LoggerType.UI)
            ui_logger.error(f"Erreur lors de l'initialisation du gestionnaire d'études: {str(e)}")
        study_manager = None
    
    # Enregistrer les callbacks principaux
    register_studies_callbacks(app, central_logger)
    
    # Enregistrer les callbacks supplémentaires
    if study_manager:
        register_additional_callbacks(app, study_manager, central_logger)

# Fonction pour intégrer la page des études dans l'application principale
def add_studies_page_to_app(app, central_logger=None):
    """
    Intègre la page des études dans l'application principale
    
    Args:
        app: L'instance de l'application Dash
        central_logger: Instance du logger centralisé
    """
    # Enregistrer tous les callbacks
    register_all_studies_callbacks(app, central_logger)
    
    # Retourner le layout de la page
    return create_studies_page(central_logger)

# Exécution en mode standalone pour les tests
if __name__ == "__main__":
    import dash
    
    # Créer une application Dash pour les tests
    app = dash.Dash(__name__, external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css",
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css"
    ])
    
    # Initialiser le logger
    from logger.logger import CentralizedLogger
    central_logger = CentralizedLogger()
    
    # Définir le layout
    app.layout = html.Div([
        create_studies_page(central_logger)
    ])
    
    # Enregistrer les callbacks
    register_all_studies_callbacks(app, central_logger)
    
    # Lancer l'application
    app.run_server(debug=True)