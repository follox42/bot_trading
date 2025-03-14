"""
Composant pour la gestion des optimisations.
Version refactorisée avec interface utilisateur améliorée et flux de travail optimisé.
"""
from dash import html, dcc, Input, Output, State, callback, callback_context
import dash_bootstrap_components as dbc
import dash
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import os
import threading
import time

from logger.logger import LoggerType

# Import de la configuration et des modules d'optimisation
from simulator.study_manager import IntegratedStudyManager
from simulator.config import OptimizationConfig, ScoringFormula, OptimizationMethod, PrunerMethod
from simulator.study_config_definitions import (
    OPTIMIZATION_METHODS, 
    PRUNER_METHODS, 
    SCORING_FORMULAS, 
    AVAILABLE_METRICS
)

# Import du module de détails d'optimisation
from ui.components.studies.optimization_details import create_optimization_details

# Données partagées pour les optimisations en cours
running_optimizations = {}
optimization_progress = {}

def create_optimization_panel(central_logger=None):
    """
    Crée le panneau de gestion des optimisations avec une interface améliorée.
    
    Args:
        central_logger: Instance du logger centralisé
    
    Returns:
        Composant du panneau d'optimisation
    """
    # Initialiser le logger
    if central_logger:
        ui_logger = central_logger.get_logger("optimization_panel", LoggerType.UI)
        ui_logger.info("Création du panneau d'optimisation")
    
    # Récupérer la liste des études
    try:
        study_manager = IntegratedStudyManager("studies")
        studies = study_manager.get_studies_list()
    except Exception as e:
        if central_logger:
            ui_logger.error(f"Erreur lors de la récupération des études: {str(e)}")
        studies = []
    
    # Récupérer l'étude sélectionnée (depuis le store)
    selected_study = dash.callback_context.inputs.get("current-study-id.data", None)
    
    # Récupérer les optimisations en cours
    current_optimizations = list(running_optimizations.values())
    
    # Style pour les inputs et dropdowns
    dropdown_style = {"width": "100%"}
    
    # Options pour l'échantillonnage
    sampling_options = [
        {"label": method_info["name"], "value": method_name}
        for method_name, method_info in OPTIMIZATION_METHODS.items()
    ]
    
    # Options pour les formules de scoring
    scoring_options = [
        {"label": formula_info["name"], "value": formula_name}
        for formula_name, formula_info in SCORING_FORMULAS.items()
    ]
    
    # Options pour les méthodes de pruning
    pruner_options = [
        {"label": "Aucun pruning", "value": "none"}
    ] + [
        {"label": pruner_info["name"], "value": pruner_name}
        for pruner_name, pruner_info in PRUNER_METHODS.items()
        if pruner_name != "none"
    ]
    
    # Options pour le nombre d'essais
    trials_options = [
        {"label": "Rapide (100 essais)", "value": 100},
        {"label": "Standard (500 essais)", "value": 500},
        {"label": "Approfondi (1000 essais)", "value": 1000},
        {"label": "Exhaustif (2000 essais)", "value": 2000},
    ]
    
    # Options pour le nombre de processus
    cores_options = [
        {"label": "Monocœur", "value": 1},
        {"label": "2 cœurs", "value": 2},
        {"label": "4 cœurs", "value": 4},
        {"label": "Automatique (tous les cœurs)", "value": -1},
    ]
    
    # Création du layout avec une structure en onglets
    return html.Div([
        # En-tête avec titre et actions
        html.Div([
            html.H2(
                "OPTIMISATIONS DE STRATÉGIES", 
                className="text-xl text-cyan-300 font-bold mb-4 d-flex align-items-center"
            ),
            html.Div([
                html.Button(
                    [html.I(className="bi bi-arrow-repeat me-2"), "ACTUALISER"],
                    id="btn-refresh-optimizations",
                    className="retro-button secondary ms-2"
                )
            ], className="ms-auto")
        ], className="d-flex justify-content-between align-items-center mb-4"),
        
        # Onglets principaux
        dbc.Tabs([
            # Onglet "Exécuter une optimisation"
            dbc.Tab([
                dbc.Row([
                    # Colonne gauche - Configuration
                    dbc.Col([
                        # Carte de paramètres de base
                        html.Div([
                            html.Div(
                                className="retro-card-header d-flex justify-content-between align-items-center",
                                children=[
                                    html.H3("PARAMÈTRES DE BASE", className="retro-card-title mb-0"),
                                    html.I(className="bi bi-sliders text-cyan-300")
                                ]
                            ),
                            html.Div(
                                className="retro-card-body",
                                children=[
                                    # Sélection de l'étude
                                    html.Div([
                                        dbc.Label("Étude à optimiser", html_for="select-study-to-optimize", className="text-cyan-100 mb-1"),
                                        dbc.Tooltip(
                                            "Sélectionnez l'étude pour laquelle vous souhaitez optimiser les paramètres de stratégie",
                                            target="select-study-to-optimize",
                                            placement="top"
                                        ),
                                        html.Div(
                                            dcc.Dropdown(
                                                id="select-study-to-optimize",
                                                options=[{"label": s.get('name', ''), "value": s.get('name', '')} for s in studies],
                                                value=selected_study,
                                                persistence=True,
                                                persistence_type="session",
                                                style=dropdown_style,
                                                placeholder="Sélectionner une étude"
                                            ),
                                            className="retro-dropdown mb-3"
                                        ),
                                    ]),
                                    
                                    # Paramètres d'optimisation
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Méthode d'optimisation", html_for="select-sampling-method", className="text-cyan-100 mb-1"),
                                            dbc.Tooltip(
                                                "TPE apprend des résultats précédents pour améliorer les suggestions de paramètres",
                                                target="select-sampling-method-container",
                                                placement="top"
                                            ),
                                            html.Div(
                                                id="select-sampling-method-container",
                                                children=dcc.Dropdown(
                                                    id="select-sampling-method",
                                                    persistence=True,
                                                    persistence_type="session",
                                                    options=sampling_options,
                                                    value="tpe",
                                                    style=dropdown_style
                                                ),
                                                className="retro-dropdown mb-3"
                                            ),
                                        ], width=12, md=6),
                                        
                                        dbc.Col([
                                            dbc.Label("Nombre d'essais", html_for="select-n-trials", className="text-cyan-100 mb-1"),
                                            dbc.Tooltip(
                                                "Plus d'essais = meilleurs résultats mais temps d'exécution plus long",
                                                target="select-n-trials-container",
                                                placement="top"
                                            ),
                                            html.Div(
                                                id="select-n-trials-container",
                                                children=dcc.Dropdown(
                                                    id="select-n-trials",
                                                    persistence=True,
                                                    persistence_type="session",
                                                    options=trials_options,
                                                    value=500,
                                                    style=dropdown_style
                                                ),
                                                className="retro-dropdown mb-3"
                                            ),
                                        ], width=12, md=6),
                                    ]),
                                    
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Formule de scoring", html_for="select-scoring-formula", className="text-cyan-100 mb-1"),
                                            dbc.Tooltip(
                                                "Détermine comment les stratégies sont évaluées et classées",
                                                target="select-scoring-formula-container",
                                                placement="top"
                                            ),
                                            html.Div(
                                                id="select-scoring-formula-container",
                                                children=dcc.Dropdown(
                                                    id="select-scoring-formula",
                                                    persistence=True,
                                                    persistence_type="session",
                                                    options=scoring_options,
                                                    value="standard",
                                                    style=dropdown_style
                                                ),
                                                className="retro-dropdown mb-3"
                                            ),
                                        ], width=12, md=6),
                                        
                                        dbc.Col([
                                            dbc.Label("Nombre de processus", html_for="select-n-jobs", className="text-cyan-100 mb-1"),
                                            dbc.Tooltip(
                                                "Utiliser plus de processeurs accélère l'optimisation",
                                                target="select-n-jobs-container",
                                                placement="top"
                                            ),
                                            html.Div(
                                                id="select-n-jobs-container",
                                                children=dcc.Dropdown(
                                                    id="select-n-jobs",
                                                    persistence=True,
                                                    persistence_type="session",
                                                    options=cores_options,
                                                    value=-1,
                                                    style=dropdown_style
                                                ),
                                                className="retro-dropdown mb-3"
                                            ),
                                        ], width=12, md=6),
                                    ]),
                                    
                                    # Description de la formule de scoring
                                    html.Div(
                                        id="scoring-formula-description",
                                        className="bg-dark p-3 rounded mb-3 text-sm border border-secondary",
                                        children=[
                                            html.I(className="bi bi-info-circle me-2 text-info"),
                                            SCORING_FORMULAS["standard"]["description"]
                                        ]
                                    ),
                                    
                                    # Bouton de personnalisation des poids
                                    html.Div([
                                        html.Button(
                                            [html.I(className="bi bi-sliders me-2"), "PERSONNALISER LES POIDS DE MÉTRIQUES"],
                                            id="btn-customize-scoring",
                                            className="retro-button secondary w-100 mb-3",
                                            n_clicks=0
                                        ),
                                    ]),
                                ]
                            )
                        ], className="retro-card mb-4"),
                        
                        # Carte des paramètres avancés
                        html.Div([
                            html.Div(
                                className="retro-card-header d-flex justify-content-between align-items-center",
                                children=[
                                    html.H3("PARAMÈTRES AVANCÉS", className="retro-card-title mb-0"),
                                    html.I(className="bi bi-gear-fill text-cyan-300")
                                ]
                            ),
                            html.Div(
                                className="retro-card-body",
                                children=[
                                    # Minimum de trades
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Nombre minimum de trades", className="text-cyan-100 mb-1"),
                                            dbc.Tooltip(
                                                "Les stratégies avec moins de trades que ce seuil seront ignorées",
                                                target="min-trades",
                                                placement="top"
                                            ),
                                            dbc.Input(
                                                id="min-trades",
                                                type="number",
                                                min=1,
                                                max=1000,
                                                step=1,
                                                value=10,
                                                className="highlighted-input mb-3"
                                            ),
                                        ], width=12, md=6),
                                        
                                        # Pruning
                                        dbc.Col([
                                            html.Div([
                                                dbc.Checkbox(
                                                    id="enable-pruning",
                                                    className="retro-checkbox mb-2",
                                                    label="Activer le pruning",
                                                    value=False
                                                ),
                                                dbc.Tooltip(
                                                    "Le pruning arrête les essais non prometteurs pour économiser du temps",
                                                    target="enable-pruning",
                                                    placement="top"
                                                ),
                                            ]),
                                            
                                            html.Div(
                                                id="pruner-method-container",
                                                children=dcc.Dropdown(
                                                    id="select-pruner-method",
                                                    options=pruner_options,
                                                    value="median",
                                                    persistence=True,
                                                    persistence_type="session",
                                                    style=dropdown_style,
                                                    disabled=True
                                                ),
                                                className="retro-dropdown mb-3"
                                            ),
                                        ], width=12, md=6),
                                    ]),
                                    
                                    # Arrêt anticipé
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Arrêt anticipé (nombre d'essais sans amélioration)", className="text-cyan-100 mb-1"),
                                            dbc.Tooltip(
                                                "L'optimisation s'arrêtera après ce nombre d'essais sans amélioration (0 = désactivé)",
                                                target="early-stopping-n-trials",
                                                placement="top"
                                            ),
                                            dbc.Input(
                                                id="early-stopping-n-trials",
                                                type="number",
                                                min=0,
                                                max=1000,
                                                step=10,
                                                value=0,
                                                className="highlighted-input mb-3"
                                            ),
                                        ], width=12),
                                    ]),
                                    
                                    # Données à utiliser
                                    html.Div([
                                        html.H5("Données à utiliser", className="text-cyan-200 mb-2"),
                                        
                                        # Affichage des informations du fichier de données
                                        html.Div([
                                            dbc.Label("Fichier de données", html_for="data-file-info", className="text-cyan-100 mb-1"),
                                            dbc.Tooltip(
                                                "Informations sur le fichier de données associé à l'étude",
                                                target="data-file-info-container",
                                                placement="top"
                                            ),
                                            html.Div(
                                                id="data-file-info-container",
                                                className="p-3 rounded border border-secondary mb-3",
                                                children=[
                                                    # Le contenu sera mis à jour par le callback
                                                    html.Div(id="selected-data-file-display",
                                                        children=[
                                                            html.Span("Sélectionnez une étude pour voir les informations du fichier de données", 
                                                                    className="text-muted")
                                                        ]
                                                    )
                                                ]
                                            ),
                                            
                                            # Alerte pour les données manquantes (masquée par défaut)
                                            html.Div(
                                                id="study-data-file-container",
                                                className="d-none",
                                                children=[
                                                    dbc.Alert(
                                                        [
                                                            html.I(className="bi bi-exclamation-triangle me-2"),
                                                            "Aucun fichier de données associé à cette étude. L'optimisation ne pourra pas être exécutée sans données."
                                                        ],
                                                        color="warning",
                                                        className="mb-3 py-2"
                                                    )
                                                ]
                                            ),
                                        ]),
                                    ]), 
                                    ]
                            )
                        ], className="retro-card mb-4"),
                        
                        # Bouton de lancement et zone d'état
                        html.Div([
                            html.Button(
                                [html.I(className="bi bi-play-fill me-2"), "LANCER L'OPTIMISATION"],
                                id="btn-start-optimization",
                                className="retro-button w-100 start-optimization-btn mb-4"
                            ),
                            
                            html.Div(
                                id="optimization-status",
                                className="mt-3"
                            ),
                        ]),
                    ], width=12, lg=5),
                    
                    # Colonne droite - Visualisation et information
                    dbc.Col([
                        # Optimisations en cours
                        html.Div([
                            html.Div(
                                className="retro-card-header d-flex justify-content-between align-items-center",
                                children=[
                                    html.H3("OPTIMISATIONS EN COURS", className="retro-card-title mb-0"),
                                    html.I(className="bi bi-hourglass-split text-cyan-300")
                                ]
                            ),
                            html.Div(
                                className="retro-card-body",
                                id="running-optimizations-container",
                                children=create_running_optimizations_list(current_optimizations)
                            )
                        ], className="retro-card mb-4"),
                        
                        # Carte d'aide et d'information
                        html.Div([
                            html.Div(
                                className="retro-card-header d-flex justify-content-between align-items-center",
                                children=[
                                    html.H3("GUIDE D'OPTIMISATION", className="retro-card-title mb-0"),
                                    html.I(className="bi bi-lightbulb text-cyan-300")
                                ]
                            ),
                            html.Div(
                                className="retro-card-body",
                                children=[
                                    dbc.Tabs([
                                        dbc.Tab([
                                            html.H5("Qu'est-ce que l'optimisation ?", className="text-cyan-200 mt-3 mb-2"),
                                            html.P(
                                                "L'optimisation de stratégie utilise des algorithmes avancés pour trouver les meilleurs paramètres de trading en testant automatiquement des milliers de combinaisons.",
                                                className="mb-3"
                                            ),
                                            
                                            html.H5("Méthodes d'optimisation", className="text-cyan-200 mb-2"),
                                            html.Ul([
                                                html.Li([html.Strong("TPE"), " (Tree-structured Parzen Estimator) : Méthode bayésienne qui apprend des résultats précédents pour cibler les zones prometteuses"], className="mb-1"),
                                                html.Li([html.Strong("Random Search"), " : Exploration aléatoire efficace pour les grands espaces de paramètres"], className="mb-1"),
                                                html.Li([html.Strong("CMA-ES"), " : Algorithme évolutionnaire puissant pour les problèmes complexes et non linéaires"], className="mb-1")
                                            ], className="mb-3"),
                                            
                                            html.H5("Conseils", className="text-cyan-200 mb-2"),
                                            html.Ul([
                                                html.Li("Commencez par une optimisation rapide (100 essais) pour identifier les zones prometteuses", className="mb-1"),
                                                html.Li("Utilisez ensuite une optimisation standard ou approfondie sur ces zones", className="mb-1"),
                                                html.Li("Le pruning permet d'économiser du temps en arrêtant tôt les essais non prometteurs", className="mb-1"),
                                                html.Li("L'optimisation s'exécute en arrière-plan, vous pouvez continuer à utiliser l'application", className="mb-1")
                                            ]),
                                        ], label="Bases", tab_id="tab-basics"),
                                        
                                        dbc.Tab([
                                            html.H5("Formules de scoring", className="text-cyan-200 mt-3 mb-2"),
                                            html.Ul([
                                                html.Li([html.Strong("Standard"), " : Équilibre entre ROI, win rate et drawdown. Bon point de départ pour la plupart des stratégies."], className="mb-1"),
                                                html.Li([html.Strong("Consistency"), " : Favorise la stabilité des résultats et minimise les risques. Idéal pour le trading à long terme."], className="mb-1"), 
                                                html.Li([html.Strong("Aggressive"), " : Priorise le ROI et profit factor. Adapté pour la recherche de hauts rendements."], className="mb-1"),
                                                html.Li([html.Strong("Conservative"), " : Minimise les drawdowns et les pertes consécutives. Pour les approches à faible risque."], className="mb-1"),
                                                html.Li([html.Strong("Volume"), " : Favorise les stratégies qui génèrent beaucoup de trades. Pour le trading à haute fréquence."], className="mb-1"),
                                                html.Li([html.Strong("Custom"), " : Personnalisez les poids des métriques selon vos objectifs spécifiques."], className="mb-1")
                                            ], className="mb-3"),
                                            
                                            html.H5("Temps d'exécution", className="text-cyan-200 mb-2"),
                                            html.Div([
                                                html.Div([
                                                    html.Strong("100 essais"), 
                                                    html.Span(" : ~5-15 minutes", className="text-muted")
                                                ], className="d-flex justify-content-between mb-1"),
                                                html.Div([
                                                    html.Strong("500 essais"), 
                                                    html.Span(" : ~30-60 minutes", className="text-muted")
                                                ], className="d-flex justify-content-between mb-1"),
                                                html.Div([
                                                    html.Strong("1000+ essais"), 
                                                    html.Span(" : plusieurs heures", className="text-muted")
                                                ], className="d-flex justify-content-between mb-1")
                                            ], className="bg-dark p-2 rounded mb-3 small"),
                                        ], label="Avancé", tab_id="tab-advanced"),
                                    ], id="help-tabs", active_tab="tab-basics", className="mt-2"),
                                    
                                    html.Div(
                                        html.Div([
                                            html.I(className="bi bi-cpu text-info me-2"),
                                            "Utiliser plus de cœurs de processeur accélère l'optimisation mais consomme plus de ressources système."
                                        ], className="p-3 rounded border border-info small"),
                                        className="mt-3"
                                    )
                                ]
                            )
                        ], className="retro-card")
                    ], width=12, lg=7),
                ]),
            ], label="CONFIGURER ET EXÉCUTER", tab_id="tab-configure"),
            
            # Onglet "Optimisations récentes"
            dbc.Tab([
                # Résultats des optimisations récentes
                html.Div([
                    html.Div(
                        className="retro-card-header d-flex justify-content-between align-items-center",
                        children=[
                            html.H3("RÉSULTATS D'OPTIMISATIONS", className="retro-card-title mb-0"),
                            html.I(className="bi bi-trophy text-cyan-300")
                        ]
                    ),
                    html.Div(
                        className="retro-card-body",
                        id="recent-optimizations-container",
                        children=create_recent_optimizations_list(studies)
                    )
                ], className="retro-card mb-4"),
                
                # Graphiques de comparaison des optimisations
                html.Div([
                    html.Div(
                        className="retro-card-header d-flex justify-content-between align-items-center",
                        children=[
                            html.H3("ANALYSE COMPARATIVE", className="retro-card-title mb-0"),
                            html.I(className="bi bi-graph-up text-cyan-300")
                        ]
                    ),
                    html.Div(
                        className="retro-card-body",
                        id="optimization-comparison-container",
                        children=html.Div([
                            html.I(className="bi bi-info-circle me-2 text-cyan-300 opacity-50", style={"fontSize": "24px"}),
                            html.Span("Sélectionnez plusieurs optimisations pour afficher une comparaison", className="text-muted"),
                        ], className="text-center py-5")
                    )
                ], className="retro-card"),
            ], label="RÉSULTATS ET ANALYSE", tab_id="tab-results"),
        ], id="optimization-tabs", active_tab="tab-configure"),
        
        # Stores and intervals
        dcc.Store(id="optimization-data", data=None),
        dcc.Store(id="custom-weights-data", data=None),
        dcc.Store(id="expanded-optimizations-store", data={}),  # Nouveau store pour l'état d'expansion
        
        # Store pour l'ouverture du modal des poids
        dcc.Store(id="open-modal-weights-trigger", data=None),
        
        # Liens pour la navigation (toujours inclus mais masqués si nécessaire)
        html.Div([
            dcc.Link(id="link-to-config-tab-empty", href="#"),
            dcc.Link(id="link-to-config-tab-recent", href="#")
        ], style={"display": "none"}),
        
        # Add this line for the refresh interval
        dcc.Interval(id="optimization-refresh-interval", interval=5000, n_intervals=0), # 5000 ms = 5 seconds

        # Modal de détails d'optimisation
        dbc.Modal(
            [
                dbc.ModalHeader(
                    html.H4("DÉTAILS DE L'OPTIMISATION", className="retro-title"),
                    close_button=True
                ),
                dbc.ModalBody(
                    id="optimization-details-content",
                    children=[]
                ),
                dbc.ModalFooter(
                    dbc.Button(
                        "Fermer",
                        id="btn-close-optimization-details",
                        className="retro-button"
                    )
                ),
            ],
            id="optimization-details-modal",
            size="xl",
            is_open=False,
            centered=True,
            scrollable=True,
            className="retro-modal"
        ),
        
        # Modal pour les poids personnalisés
        dbc.Modal(
            [
                dbc.ModalHeader(
                    html.H4("PERSONNALISER LES POIDS DE SCORING", className="retro-title"),
                    close_button=True
                ),
                dbc.ModalBody(
                    id="custom-weights-content",
                    children=create_custom_weights_editor()
                ),
                dbc.ModalFooter([
                    dbc.Button(
                        "Annuler",
                        id="btn-cancel-custom-weights",
                        className="retro-button secondary me-2", 
                        n_clicks=0
                    ),
                    dbc.Button(
                        "Appliquer",
                        id="btn-apply-custom-weights",
                        className="retro-button",
                        n_clicks=0
                    )
                ]),
            ],
            id="custom-weights-modal",
            size="lg",
            is_open=False,
            centered=True,
            scrollable=True,
            className="retro-modal"
        ),
    ])

def create_running_optimizations_list(optimizations, expanded_state=None):
    """
    Crée la liste des optimisations en cours avec style amélioré et meilleur affichage des erreurs.
    
    Args:
        optimizations: Liste des optimisations en cours
        expanded_state: État d'expansion pour chaque étude
    
    Returns:
        Composant d'affichage des optimisations en cours
    """
    if expanded_state is None:
        expanded_state = {}
        
    if not optimizations:
        return html.Div([
            html.I(className="bi bi-info-circle me-2 text-cyan-300 opacity-50", style={"fontSize": "24px"}),
            html.Span("Aucune optimisation en cours", className="text-muted"),
            html.Div([
                dcc.Link(
                    html.Button(
                        [html.I(className="bi bi-play-fill me-2"), "LANCER UNE OPTIMISATION"],
                        className="retro-button mt-3"
                    ),
                    id="link-to-config-tab-empty",
                    href="#"
                ),
            ], className="text-center mt-3")
        ], className="text-center py-5")
    
    # Trier par date de début (plus récent en premier)
    optimizations = sorted(optimizations, key=lambda x: x.get('start_time', 0), reverse=True)
    
    # Créer une carte pour chaque optimisation
    optimization_cards = []
    for opt in optimizations:
        # Récupérer les informations
        study_name = opt.get('study_name', 'Inconnue')
        start_time = datetime.fromtimestamp(opt.get('start_time', 0)).strftime("%H:%M:%S")
        elapsed_time = format_elapsed_time(time.time() - opt.get('start_time', time.time()))
        n_trials = opt.get('n_trials', 0)
        n_completed = optimization_progress.get(study_name, {}).get('completed', 0)
        best_value = optimization_progress.get(study_name, {}).get('best_value', None)
        best_metrics = optimization_progress.get(study_name, {}).get('best_metrics', {})
        
        # Vérifier s'il y a une erreur
        status = opt.get('status', 'running')
        error = opt.get('error', None)
        error_details = opt.get('error_details', None)
        
        # Si le statut est erreur mais sans message, vérifie dans optimization_progress
        if status == 'error' and not error:
            error = optimization_progress.get(study_name, {}).get('error', "Erreur inconnue")
            error_details = optimization_progress.get(study_name, {}).get('error_details', None)
        
        # Calculer la progression
        progress = (n_completed / n_trials) * 100 if n_trials > 0 else 0
        estimated_remaining = format_elapsed_time((time.time() - opt.get('start_time', time.time())) / max(0.01, progress) * (100 - progress)) if progress > 0 else "Calcul en cours..."
        
        # Métriques du meilleur essai
        roi = best_metrics.get('roi', 0) * 100 if best_metrics else None
        win_rate = best_metrics.get('win_rate', 0) * 100 if best_metrics else None
        
        # Style de la carte selon statut
        card_style = {"backgroundColor": "rgba(34, 211, 238, 0.05)"}
        border_class = "border border-info"
        
        if status == 'error':
            card_style = {"backgroundColor": "rgba(248, 113, 113, 0.1)"}
            border_class = "border border-danger"
        elif status == 'completed':
            card_style = {"backgroundColor": "rgba(74, 222, 128, 0.1)"}
            border_class = "border border-success"
        elif status == 'initializing':
            card_style = {"backgroundColor": "rgba(251, 191, 36, 0.1)"}
            border_class = "border border-warning"
        
        # Vérifier l'état d'expansion pour cette étude
        is_expanded = expanded_state.get(study_name, False) or status == 'error'
        
        # Créer une carte plus élaborée avec détails collapsables et informations d'erreur
        card = html.Div([
            # En-tête avec le nom de l'étude
            html.Div([
                html.Div([
                    html.H5(study_name, className="mb-0 text-cyan-200"),
                    html.Div([
                        html.Span(f"Démarré: {start_time}", className="me-2 text-muted small"),
                        html.Span(f"Durée: {elapsed_time}", className="me-2 text-muted small"),
                        html.Span(f"État: ", className="me-1 text-muted small"),
                        html.Span(
                            {"initializing": "Initialisation", "running": "En cours", 
                             "completed": "Terminé", "failed": "Échec", "error": "Erreur"}.get(status, status),
                            className={
                                "initializing": "text-warning",
                                "running": "text-info",
                                "completed": "text-success",
                                "failed": "text-danger",
                                "error": "text-danger"
                            }.get(status, "text-muted"),
                        )
                    ])
                ]),
                html.Button(
                    [html.I(className="bi bi-chevron-down toggle-icon me-1" if not is_expanded else "bi bi-chevron-up toggle-icon me-1"), 
                    "DÉTAILS" if not is_expanded else "MASQUER"],
                    id={"type": "btn-toggle-optimization-details", "study": study_name},
                    className="retro-button secondary btn-sm"
                )
            ], className="d-flex justify-content-between align-items-center mb-2"),
            
            # Section d'erreur (si applicable)
            html.Div([
                html.Div([
                    html.I(className="bi bi-exclamation-triangle text-danger me-2"),
                    html.Strong("ERREUR: ", className="text-danger"),
                    html.Span(error, className="text-danger")
                ], className="bg-dark p-2 rounded mb-3 small"),
                
                # Détails de l'erreur dépliables
                html.Details([
                    html.Summary("Détails techniques", className="text-danger small"),
                    html.Pre(
                        error_details or "Aucun détail disponible", 
                        className="bg-dark p-2 mt-2 text-white-50",
                        style={"white-space": "pre-wrap", "font-size": "0.7rem", "max-height": "200px", "overflow": "auto"}
                    )
                ]) if error_details else None
            ]) if status == 'error' else None,
            
            # Progression (seulement si pas d'erreur ou initialisation)
            html.Div([
                html.Div([
                    html.Span(f"Progression: {progress:.1f}% ({n_completed}/{n_trials} essais)"),
                    html.Span(f"Reste estimé: {estimated_remaining}")
                ], className="d-flex justify-content-between small mb-2"),
                
                # Barre de progression
                html.Div([
                    html.Div(
                        className="progress-bar-fill",
                        style={
                            "width": f"{progress}%",
                            "height": "10px",
                            "borderRadius": "5px",
                            "transition": "width 0.5s ease",
                            "background": {
                                "running": "linear-gradient(90deg, rgba(34, 211, 238, 0.7), rgba(34, 211, 238, 1))",
                                "initializing": "linear-gradient(90deg, rgba(251, 191, 36, 0.7), rgba(251, 191, 36, 1))",
                                "completed": "linear-gradient(90deg, rgba(74, 222, 128, 0.7), rgba(74, 222, 128, 1))",
                                "failed": "linear-gradient(90deg, rgba(248, 113, 113, 0.7), rgba(248, 113, 113, 1))"
                            }.get(status, "linear-gradient(90deg, rgba(34, 211, 238, 0.7), rgba(34, 211, 238, 1))")
                        }
                    )
                ], className="progress-bar-container mb-2"),
            ]) if status != 'error' and status != 'initializing' else None,
            
            # Section de détails collapsable
            dbc.Collapse([
                # Métriques actuelles du meilleur essai
                html.Div([
                    html.Div([
                        html.Strong("Meilleur score: "),
                        html.Span(f"{best_value:.4f}" if best_value is not None else "Aucun résultat"),
                    ], className="text-info"),
                    
                    html.Div([
                        html.Div([
                            html.Strong("ROI: "),
                            html.Span(f"{roi:.2f}%" if roi is not None else "N/A", 
                                      className="text-success" if roi is not None and roi > 0 else "text-danger"),
                        ], className="me-3"),
                        html.Div([
                            html.Strong("Win Rate: "),
                            html.Span(f"{win_rate:.2f}%" if win_rate is not None else "N/A"),
                        ]),
                    ], className="d-flex"),
                ], className="bg-dark p-2 rounded mb-3 small") if status != 'error' and best_value is not None else None,
                
                # CPUs / Process info
                html.Div([
                    html.Div([
                        html.I(className="bi bi-cpu me-2 text-muted"),
                        html.Span("Processeurs: "),
                        html.Span(f"{opt.get('n_jobs', -1)} cœurs" if opt.get('n_jobs', -1) > 0 else "Automatique"),
                    ], className="me-3"),
                    html.Div([
                        html.I(className="bi bi-gear me-2 text-muted"),
                        html.Span("Méthode: "),
                        html.Span(opt.get('optimization_method', 'TPE')),
                    ]),
                ], className="d-flex justify-content-between small mb-3"),
                
                # Boutons d'action
                html.Div([
                    html.Button(
                        [html.I(className="bi bi-eye me-1"), "VOIR DÉTAILS"],
                        id={"type": "btn-view-running-optimization", "study": study_name},
                        className="retro-button btn-sm me-2"
                    ),
                    html.Button(
                        [html.I(className="bi bi-x-circle me-1"), "ARRÊTER"],
                        id={"type": "btn-stop-optimization", "study": study_name},
                        className="retro-button danger btn-sm"
                    ),
                ], className="d-flex justify-content-end"),
            ], id={"type": "optimization-details-collapse", "study": study_name}, is_open=is_expanded)
        ], className=f"p-3 mb-3 rounded {border_class}", style=card_style)
        
        optimization_cards.append(card)
    
    return html.Div(optimization_cards)

def format_elapsed_time(seconds):
    """
    Formate un temps en secondes en format lisible.
    
    Args:
        seconds: Nombre de secondes
    
    Returns:
        Chaîne formatée HH:MM:SS ou MM:SS
    """
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    else:
        return f"{minutes:02d}m {seconds:02d}s"

def create_recent_optimizations_list(studies):
    """
    Crée la liste des optimisations récentes avec style amélioré.
    
    Args:
        studies: Liste des études disponibles
    
    Returns:
        Composant d'affichage des optimisations récentes
    """
    # Filtrer les études qui ont des résultats d'optimisation
    optimized_studies = [s for s in studies if s.get('has_optimization', False)]
    
    if not optimized_studies:
        return html.Div([
            html.I(className="bi bi-info-circle me-2 text-cyan-300 opacity-50", style={"fontSize": "24px"}),
            html.Span("Aucune optimisation récente trouvée", className="text-muted"),
            html.Div([
                dcc.Link(
                    html.Button(
                        [html.I(className="bi bi-play-fill me-2"), "COMMENCER À OPTIMISER"],
                        className="retro-button mt-3"
                    ),
                    id="link-to-config-tab-recent",
                    href="#"
                ),
            ], className="text-center mt-3")
        ], className="text-center py-5")
    
    # Trier par date de dernière modification
    optimized_studies = sorted(optimized_studies, key=lambda x: x.get('last_modified', ''), reverse=True)
    
    # Créer les cartes d'optimisation récentes
    optimization_cards = []
    for study in optimized_studies[:8]:  # Limiter aux 8 plus récentes
        study_name = study.get('name', 'Inconnue')
        asset = study.get('asset', 'N/A')
        timeframe = study.get('timeframe', 'N/A')
        exchange = study.get('exchange', 'N/A')
        last_modified = study.get('last_modified', 'N/A')
        strategies_count = study.get('strategies_count', 0)
        
        # Créer une carte pour chaque optimisation
        card = html.Div([
            html.Div([
                html.H5([
                    html.I(className="bi bi-file-earmark-text me-2 text-cyan-200"),
                    study_name
                ], className="mb-1 text-cyan-200"),
                html.Div([
                    html.Span(f"{asset} • {timeframe} • {exchange}", className="text-muted small"),
                ], className="mb-2"),
                html.Div([
                    html.Div([
                        html.I(className="bi bi-calendar me-2 text-muted"),
                        html.Span("Dernière modification: "),
                        html.Span(last_modified),
                    ], className="small mb-1"),
                    html.Div([
                        html.I(className="bi bi-code-square me-2 text-muted"),
                        html.Span("Stratégies: "),
                        html.Span(strategies_count),
                    ], className="small"),
                ], className="mb-3"),
                
                html.Div([
                    html.Button(
                        [html.I(className="bi bi-eye me-1"), "DÉTAILS"],
                        id={"type": "btn-view-optimization-results", "study": study_name},
                        className="retro-button btn-sm me-2"
                    ),
                    html.Button(
                        [html.I(className="bi bi-play-fill me-1"), "RELANCER"],
                        id={"type": "btn-restart-optimization-from-list", "study": study_name},
                        className="retro-button secondary btn-sm me-2"
                    ),
                    html.Div(
                        html.I(className="bi bi-check-square", style={"cursor": "pointer"}),
                        id={"type": "btn-select-for-compare", "study": study_name},
                        className="retro-badge retro-badge-blue d-flex align-items-center justify-content-center",
                        style={"width": "32px", "height": "32px"},
                        n_clicks=0
                    )
                ], className="d-flex justify-content-start"),
            ], className="p-3")
        ], className="retro-card mb-3")
        
        optimization_cards.append(card)
    
    return html.Div([
        html.Div(optimization_cards, className="row row-cols-1 row-cols-md-2 row-cols-xl-4 g-3"),
        
        html.Div([
            html.Button(
                [html.I(className="bi bi-arrow-right me-2"), "VOIR TOUTES LES OPTIMISATIONS"],
                id="btn-view-all-optimizations",
                className="retro-button secondary mt-3"
            ),
        ], className="text-center mt-3")
    ])

def create_custom_weights_editor():
    """
    Crée un éditeur de poids personnalisés pour la formule de scoring.
    
    Returns:
        Composant pour l'édition des poids personnalisés
    """
    # Récupérer les métriques disponibles
    metrics = AVAILABLE_METRICS
    
    # Créer un contrôle pour chaque métrique
    metric_controls = []
    for metric_name, metric_info in metrics.items():
        if metric_name in ['roi', 'win_rate', 'max_drawdown', 'profit_factor', 'total_trades', 'avg_profit']:
            # Définir la couleur en fonction de si higher_is_better
            color_class = "text-success" if metric_info.get('higher_is_better', True) else "text-danger"
            trend_icon = "bi bi-graph-up" if metric_info.get('higher_is_better', True) else "bi bi-graph-down"
            
            control = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.I(className=f"{trend_icon} me-2 {color_class}"),
                            html.Strong(metric_info['name'], className="text-cyan-200"),
                        ], className="d-flex align-items-center mb-1"),
                        html.Div(metric_info['description'], className="text-white-50 small mb-2")
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        # Un seul contrôle pour gérer la valeur du poids
                        dcc.Slider(
                            id={"type": "weight-slider", "metric": metric_name},
                            min=0,
                            max=5,
                            step=0.1,
                            value=metric_info.get('default_weight', 1.0),
                            marks={i: {"label": str(i), "style": {"color": "white"}} for i in range(6)},
                            className="retro-slider mb-1"
                        ),
                    ], width=9),
                    dbc.Col([
                        # Affichage uniquement de la valeur courante
                        html.Div(
                            id={"type": "weight-display", "metric": metric_name},
                            children=f"{metric_info.get('default_weight', 1.0):.1f}×",
                            className="weight-value-display text-white fw-bold"
                        )
                    ], width=3)
                ]),
            ], className="mb-4 p-3 rounded", style={"backgroundColor": "rgba(34, 211, 238, 0.05)"})
            
            metric_controls.append(control)
    
    return html.Div([
        # En-tête explicatif
        html.Div([
            html.I(className="bi bi-info-circle me-2 text-info"),
            html.Span("Configurez l'importance relative de chaque métrique pour l'évaluation des stratégies"),
        ], className="bg-dark p-3 rounded mb-4 border border-info"),
        
        # Contrôles des poids
        html.Div(metric_controls),
        
        # Store pour stocker toutes les valeurs
        dcc.Store(id="weights-values-store", data={}),
        
        # Note informative
        html.Div([
            html.I(className="bi bi-lightbulb me-2 text-warning"),
            "La somme des poids est automatiquement normalisée. Les valeurs relatives sont plus importantes que les valeurs absolues."
        ], className="text-white bg-dark p-2 rounded small border border-warning")
    ])

def register_optimization_callbacks(app, central_logger=None):
    """
    Enregistre les callbacks pour la gestion des optimisations.
    
    Args:
        app: L'instance de l'application Dash
        central_logger: Logger centralisé
    """
    # Initialiser le logger
    if central_logger:
        ui_logger = central_logger.get_logger("optimization_callbacks", LoggerType.UI)
    
    # ================== CALLBACK POUR LE PRUNING ==================
    @app.callback(
        Output("select-pruner-method", "disabled"),
        [Input("enable-pruning", "value")]
    )
    def toggle_pruning_options(enable_pruning):
        """Active ou désactive les options de pruning"""
        return not enable_pruning
    
    # ================== CALLBACK POUR LA DESCRIPTION DE LA FORMULE DE SCORING ==================
    # Modification du callback pour la description de la formule de scoring
    @app.callback(
        Output("scoring-formula-description", "children"),
        [Input("select-scoring-formula", "value"),
        Input("custom-weights-data", "data")],  # Ajouter les poids personnalisés comme entrée
        prevent_initial_call=False
    )
    def update_scoring_description(formula, custom_weights_data):
        """Met à jour la description de la formule de scoring en tenant compte des poids personnalisés"""
        
        if not formula:
            return [
                html.I(className="bi bi-info-circle me-2 text-info"),
                "Sélectionnez une formule de scoring"
            ]
        
        # Récupérer la description de base de la formule
        if formula in SCORING_FORMULAS:
            description = SCORING_FORMULAS[formula]["description"]
            
            # Initialiser les poids à partir de la formule par défaut
            weights = SCORING_FORMULAS[formula].get("weights", {})
            
            # Si des poids personnalisés sont disponibles, les utiliser à la place
            if custom_weights_data:
                try:
                    data = json.loads(custom_weights_data) if isinstance(custom_weights_data, str) else custom_weights_data
                    if isinstance(data, dict) and "weights" in data:
                        # Nouvelle structure avec formule et poids
                        custom_weights = data.get("weights", {})
                        if data.get("formula") == formula:  # Utiliser les poids uniquement s'ils correspondent à la formule actuelle
                            weights = custom_weights
                    elif isinstance(data, dict):
                        # Ancienne structure (juste les poids)
                        weights = data
                except:
                    pass
            
            # Tableau des poids si disponible
            weights_display = []
            if weights and formula != "custom":
                for metric, weight in weights.items():
                    if weight > 0:
                        metric_name = AVAILABLE_METRICS.get(metric, {}).get("name", metric)
                        weights_display.append(html.Div([
                            html.Span(metric_name),
                            html.Span(f"{weight:.1f}×", className="badge bg-info ms-2")
                        ], className="me-3"))
            
            return [
                html.I(className="bi bi-info-circle me-2 text-info"),
                html.Span(description, className="mb-2 d-block"),
                html.Div(weights_display, className="d-flex flex-wrap mt-2") if weights_display else None
            ]
        
        return [
            html.I(className="bi bi-info-circle me-2 text-info"),
            "Sélectionnez une formule de scoring"
        ]

    # ================== CALLBACK POUR LA MISE À JOUR DES VALEURS DE POIDS ==================
    @app.callback(
        Output({"type": "weight-display", "metric": dash.MATCH}, "children"),
        [Input({"type": "weight-slider", "metric": dash.MATCH}, "value")],
        prevent_initial_call=True
    )
    def update_weight_display(slider_value):
        """Met à jour l'affichage de la valeur du poids"""
        return f"{slider_value:.1f}×"
    
    # ================== CALLBACK POUR LES OPTIMISATIONS EN COURS ==================
    @app.callback(
        [Output("running-optimizations-container", "children"),
        Output("recent-optimizations-container", "children")],
        [Input("btn-refresh-optimizations", "n_clicks"),
        Input("optimization-refresh-interval", "n_intervals")],
        [State("expanded-optimizations-store", "data")]
    )
    def update_optimizations(n_clicks, n_intervals, expanded_state):
        """Met à jour la liste des optimisations en préservant l'état d'expansion"""
        current_optimizations = list(running_optimizations.values())
        
        # Vérification que expanded_state est valide
        if not expanded_state or not isinstance(expanded_state, dict):
            expanded_state = {}
        
        try:
            # Récupérer la liste des études pour les optimisations récentes
            study_manager = IntegratedStudyManager("studies")
            studies = study_manager.get_studies_list()
        except Exception as e:
            if central_logger:
                ui_logger.error(f"Erreur lors de la récupération des études: {str(e)}")
            studies = []
        
        # Passer l'état d'expansion à la fonction de création
        return create_running_optimizations_list(current_optimizations, expanded_state), create_recent_optimizations_list(studies)

    @app.callback(
        Output("optimization-refresh-interval", "disabled"),
        [Input("custom-weights-modal", "is_open"),
        Input("optimization-details-modal", "is_open")],
        prevent_initial_call=True
    )
    def disable_refresh_when_modal_open(is_weights_modal_open, is_details_modal_open):
        """Désactive l'intervalle de rafraîchissement lorsqu'un modal est ouvert"""
        return is_weights_modal_open or is_details_modal_open
    
    # ================== CALLBACK POUR L'AFFICHAGE/MASQUAGE DES DÉTAILS D'OPTIMISATION EN COURS ==================
    @app.callback(
        [Output({"type": "optimization-details-collapse", "study": dash.MATCH}, "is_open"),
        Output({"type": "btn-toggle-optimization-details", "study": dash.MATCH}, "children")],
        [Input({"type": "btn-toggle-optimization-details", "study": dash.MATCH}, "n_clicks")],
        [State({"type": "optimization-details-collapse", "study": dash.MATCH}, "is_open"),
        State({"type": "btn-toggle-optimization-details", "study": dash.MATCH}, "id")],
        prevent_initial_call=True
    )
    def toggle_optimization_details(n_clicks, is_open, toggle_id):
        """Affiche ou masque les détails d'une optimisation en cours"""
        if not n_clicks:
            return is_open, [html.I(className="bi bi-chevron-down toggle-icon me-1"), "DÉTAILS"]
        
        # Récupérer le nom de l'étude
        study_name = toggle_id["study"]
        
        # Inverser l'état
        new_open = not is_open
        
        # Mettre à jour le texte du bouton
        button_text = [html.I(className="bi bi-chevron-up toggle-icon me-1"), "MASQUER"] if new_open else [html.I(className="bi bi-chevron-down toggle-icon me-1"), "DÉTAILS"]
        
        return new_open, button_text

    # ================== CALLBACKS POUR LES DETAILS D'OPTIMISATION ==================
    @app.callback(
        [Output("optimization-details-modal", "is_open"),
         Output("optimization-details-content", "children")],
        [Input({"type": "btn-view-running-optimization", "study": dash.ALL}, "n_clicks"),
         Input({"type": "btn-view-optimization-results", "study": dash.ALL}, "n_clicks"),
         Input("btn-close-optimization-details", "n_clicks")],
        [State("optimization-details-modal", "is_open")]
    )
    def show_optimization_details(view_running_clicks, view_results_clicks, close_clicks, is_open):
        """Affiche ou masque le modal des détails d'optimisation"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return is_open, dash.no_update
        
        # Récupérer l'ID du déclencheur
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Fermeture du modal
        if button_id == "btn-close-optimization-details" and close_clicks:
            return False, dash.no_update
        
        # Ouverture du modal pour optimisation en cours ou résultats d'optimisation
        if ("btn-view-running-optimization" in button_id and any(view_running_clicks)) or \
           ("btn-view-optimization-results" in button_id and any(view_results_clicks)):
            # Déterminer le nom de l'étude
            study_name = json.loads(button_id)["study"]
            
            # Utiliser la fonction importée de optimization_details.py pour créer le contenu du modal
            return True, create_optimization_details(study_name)
        
        return is_open, dash.no_update
    
    # ================== CALLBACK POUR LE LANCEMENT D'OPTIMISATION ==================
    @app.callback(
        Output("optimization-status", "children"),
        [Input("btn-start-optimization", "n_clicks")],
        [State("select-study-to-optimize", "value"),
        State("select-sampling-method", "value"),
        State("select-scoring-formula", "value"),
        State("select-n-trials", "value"),
        State("select-n-jobs", "value"),
        State("enable-pruning", "value"),
        State("select-pruner-method", "value"),
        State("min-trades", "value"),
        State("early-stopping-n-trials", "value"),
        State("custom-weights-data", "data")]
    )
    def start_optimization(n_clicks, study_name, sampling_method, scoring_formula, 
                        n_trials, n_jobs, enable_pruning, pruner_method, 
                        min_trades, early_stopping_trials, custom_weights_data):
        """Lance une optimisation de stratégie"""
        if not n_clicks:
            return dash.no_update
        
        if not study_name:
            return dbc.Alert(
                [html.I(className="bi bi-exclamation-triangle me-2"),
                "Veuillez sélectionner une étude à optimiser"],
                color="danger",
                dismissable=True
            )
        
        try:
            import os  # Import explicite
            study_manager = IntegratedStudyManager("studies")
            
            # Vérifier la configuration de trading
            trading_config = study_manager.get_trading_config(study_name)
            
            if not trading_config:
                return dbc.Alert(
                    [html.I(className="bi bi-exclamation-triangle me-2"),
                    "ERREUR : Configuration de trading non trouvée pour cette étude."],
                    color="danger",
                    dismissable=True
                )
            
            # Vérifier si une optimisation est déjà en cours pour cette étude
            if study_name in running_optimizations:
                return dbc.Alert(
                    [html.I(className="bi bi-exclamation-circle me-2"),
                    f"Une optimisation est déjà en cours pour l'étude '{study_name}'"],
                    color="warning",
                    dismissable=True
                )
            
            # Préparer les poids personnalisés si nécessaire
            custom_weights = None
            if scoring_formula == "custom" and custom_weights_data:
                custom_weights = json.loads(custom_weights_data) if isinstance(custom_weights_data, str) else custom_weights_data
            
            # Préparation de la configuration d'optimisation
            optimization_config = {
                'n_trials': n_trials,
                'n_jobs': n_jobs,
                'optimization_method': sampling_method,
                'scoring_formula': scoring_formula
            }
            
            # Test de validation préliminaire avec strategyOptimizer
            from simulator.strategy_optimizer import StrategyOptimizer
            test_optimizer = StrategyOptimizer(study_manager)
            
            # Vérifie que la configuration peut être appliquée
            try:
                test_optimizer.configure(optimization_config)
            except Exception as config_error:
                if central_logger:
                    ui_logger.error(f"Erreur de configuration: {str(config_error)}")
                
                return dbc.Alert(
                    [html.I(className="bi bi-exclamation-triangle me-2"),
                    "Erreur de configuration de l'optimiseur: ",
                    html.Br(),
                    html.Code(str(config_error))],
                    color="danger",
                    dismissable=True
                )
            
            # Vérifie que les données sont disponibles
            data_file = study_manager.get_study_data_file(study_name)
            if not data_file:
                return dbc.Alert(
                    [html.I(className="bi bi-exclamation-triangle me-2"),
                    "Aucune donnée disponible pour l'optimisation. Veuillez importer des données ou activer le téléchargement automatique."],
                    color="danger",
                    dismissable=True
                )
            
            # Lancement de l'optimisation en arrière-plan avec gestion des erreurs améliorée
            def run_optimization_task():
                try:
                    # Mise à jour du statut - En cours de configuration
                    running_optimizations[study_name] = {
                        'study_name': study_name,
                        'start_time': time.time(),
                        'n_trials': n_trials,
                        'n_jobs': n_jobs,
                        'optimization_method': OPTIMIZATION_METHODS.get(sampling_method, {}).get('name', sampling_method),
                        'status': 'initializing'
                    }

                    # Initialiser le suivi de progression - RÉINITIALISER LE COMPTEUR
                    optimization_progress[study_name] = {
                        'completed': 0,  # Réinitialisation à 0
                        'best_value': None,
                        'best_metrics': None,
                        'error': None
                    }
                    
                    optimizer = StrategyOptimizer(study_manager)
                    
                    # Configurer et s'assurer que ça fonctionne
                    try:
                        optimizer.configure(optimization_config)
                    except Exception as config_error:
                        if central_logger:
                            ui_logger.error(f"Erreur de configuration: {str(config_error)}")
                        
                        running_optimizations[study_name]['status'] = 'error'
                        running_optimizations[study_name]['error'] = f"Erreur de configuration: {str(config_error)}"
                        optimization_progress[study_name]['error'] = str(config_error)
                        return
                    
                    # Mise à jour de l'état - Exécution
                    running_optimizations[study_name]['status'] = 'running'
                    
                    # Exécuter l'optimisation
                    success = optimizer.run_optimization(study_name, data_file)
                    
                    # Mettre à jour le statut
                    if success:
                        running_optimizations[study_name]['status'] = 'completed'
                    else:
                        running_optimizations[study_name]['status'] = 'failed'
                    
                    # Supprimer du dictionnaire après un certain temps
                    time.sleep(60)  # Garder l'optimisation dans la liste pendant 1 minute après la fin
                    if study_name in running_optimizations:
                        del running_optimizations[study_name]
                    if study_name in optimization_progress:
                        del optimization_progress[study_name]
                    
                except Exception as e:
                    if central_logger:
                        ui_logger.error(f"Erreur lors de l'optimisation: {str(e)}")
                    
                    # Détail complet de l'erreur avec traceback
                    import traceback
                    error_details = traceback.format_exc()
                    
                    # Mettre à jour le statut en cas d'erreur
                    if study_name in running_optimizations:
                        running_optimizations[study_name]['status'] = 'error'
                        running_optimizations[study_name]['error'] = str(e)
                        running_optimizations[study_name]['error_details'] = error_details
                    
                    # Mettre à jour le progress pour l'affichage
                    if study_name in optimization_progress:
                        optimization_progress[study_name]['error'] = str(e)
                        optimization_progress[study_name]['error_details'] = error_details
            
            # Démarrer l'optimisation dans un thread séparé
            import threading
            optimization_thread = threading.Thread(target=run_optimization_task)
            optimization_thread.daemon = True
            optimization_thread.start()
            
            # Retourner un message de confirmation
            return dbc.Alert(
                [html.I(className="bi bi-info-circle me-2"),
                f"Lancement de l'optimisation pour l'étude '{study_name}'...",
                html.Br(),
                html.Small("Vérifiez le panneau 'Optimisations en cours' pour suivre l'état et voir les erreurs éventuelles.")],
                color="info", 
                dismissable=True
            )
            
        except Exception as e:
            # Capturer toutes les erreurs et les afficher
            import traceback
            error_details = traceback.format_exc()
            
            if central_logger:
                ui_logger.error(f"Erreur lors du lancement de l'optimisation: {str(e)}")
                ui_logger.error(f"Détails: {error_details}")
            
            return dbc.Alert(
                [html.I(className="bi bi-exclamation-triangle me-2"),
                html.Strong("Erreur lors du lancement de l'optimisation:"),
                html.Br(),
                html.Code(str(e)),
                html.Br(),
                html.Br(),
                html.Details([
                    html.Summary("Détails techniques"),
                    html.Pre(error_details, style={"white-space": "pre-wrap", "font-size": "0.8rem", "max-height": "200px", "overflow": "auto"})
                ])],
                color="danger",
                dismissable=True
            )
    
    # ================== CALLBACKS POUR LA MODAL DES POIDS ==================
    # Séparation des callbacks pour éviter les erreurs de fermeture prématurée

    # 1. Callback pour capturer le clic sur le bouton de personnalisation
    @app.callback(
        Output("open-modal-weights-trigger", "data"),
        [Input("btn-customize-scoring", "n_clicks")],
        prevent_initial_call=True
    )
    def capture_weights_modal_buttons(btn_clicks):
        """
        Capture les clics sur le bouton de personnalisation des poids
        """
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update
                
        # Vérifier si le bouton a été cliqué
        if btn_clicks:
            if central_logger:
                ui_logger.info(f"Bouton de configuration des scores cliqué")
            return {"timestamp": datetime.now().isoformat()}
                
        return dash.no_update

    # 2. Callback pour l'ouverture/fermeture du modal
    @app.callback(
        Output("custom-weights-modal", "is_open"),
        [Input("open-modal-weights-trigger", "data"),
         Input("btn-cancel-custom-weights", "n_clicks"),
         Input("btn-apply-custom-weights", "n_clicks")],
        [State("custom-weights-modal", "is_open")],
        prevent_initial_call=True
    )
    def toggle_weights_modal(trigger_data, cancel_clicks, apply_clicks, is_open):
        """
        Gère l'ouverture/fermeture du modal de personnalisation des poids.
        """
        ctx = callback_context
        if not ctx.triggered:
            return is_open

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Si le déclencheur est activé, ouvrir le modal
        if button_id == "open-modal-weights-trigger" and trigger_data:
            if central_logger:
                ui_logger.info("Ouverture du modal des poids personnalisés")
            return True
            
        # Sur annuler => ferme
        if button_id == "btn-cancel-custom-weights" and cancel_clicks:
            if central_logger:
                ui_logger.info("Fermeture du modal des poids (annulation)")
            return False
            
        # Sur appliquer => ferme aussi
        if button_id == "btn-apply-custom-weights" and apply_clicks:
            if central_logger:
                ui_logger.info("Fermeture du modal des poids (application)")
            return False

        return is_open

    # 3. Callback pour charger les poids initiaux
    @app.callback(
        Output("weights-values-store", "data"),
        [Input("open-modal-weights-trigger", "data")],
        [State("select-scoring-formula", "value"),
         State("custom-weights-data", "data")],
        prevent_initial_call=True
    )
    def load_weights_from_formula(trigger_data, formula, custom_weights_data):
        """Charge les poids de la formule sélectionnée ou les poids personnalisés existants"""
        if not trigger_data:
            return dash.no_update
        
        # Priorité aux poids personnalisés existants
        if custom_weights_data:
            try:
                data = json.loads(custom_weights_data) if isinstance(custom_weights_data, str) else custom_weights_data
                if isinstance(data, dict):
                    if "weights" in data and "formula" in data:
                        # Nouvelle structure avec formule et poids
                        stored_formula = data.get("formula")
                        weights = data.get("weights", {})
                        
                        # Si la formule sélectionnée correspond à la formule stockée, utiliser ces poids
                        if stored_formula == formula:
                            if central_logger:
                                ui_logger.info(f"Utilisation des poids personnalisés existants pour {formula}")
                            return weights
                    else:
                        # Ancienne structure: directement les poids
                        if central_logger:
                            ui_logger.info("Utilisation des poids personnalisés avec ancienne structure")
                        return data
            except Exception as e:
                if central_logger:
                    ui_logger.error(f"Erreur lors du chargement des poids personnalisés: {str(e)}")
        
        # Charger les poids par défaut de la formule sélectionnée
        if formula in SCORING_FORMULAS:
            weights = SCORING_FORMULAS[formula].get("weights", {})
            if central_logger:
                ui_logger.info(f"Chargement des poids par défaut pour {formula}")
            return weights
        
        # Si aucune formule valide, retourner des poids par défaut
        default_weights = {
            "roi": 2.5,
            "win_rate": 0.5,
            "max_drawdown": 2.0,
            "profit_factor": 2.0,
            "total_trades": 1.0,
            "avg_profit": 1.0
        }
        
        if central_logger:
            ui_logger.info("Utilisation des poids par défaut")
        
        return default_weights

    # 4. Callback pour mettre à jour chaque slider avec les valeurs du store
    @app.callback(
        Output({"type": "weight-slider", "metric": dash.MATCH}, "value"),
        [Input("weights-values-store", "data")],
        [State({"type": "weight-slider", "metric": dash.MATCH}, "id")],
        prevent_initial_call=True
    )
    def update_weight_sliders(weights_data, slider_id):
        """Met à jour les sliders avec les valeurs du store"""
        if not weights_data:
            return dash.no_update
        
        # Récupérer le nom de la métrique
        metric_name = slider_id["metric"]
        
        # Récupérer la valeur du poids pour cette métrique
        value = weights_data.get(metric_name, AVAILABLE_METRICS.get(metric_name, {}).get("default_weight", 1.0))
        
        return value

    # 5. Callback pour mettre à jour le store quand les sliders changent
    @app.callback(
        Output("weights-values-store", "data", allow_duplicate=True),
        [Input({"type": "weight-slider", "metric": dash.ALL}, "value")],
        [State({"type": "weight-slider", "metric": dash.ALL}, "id"),
         State("weights-values-store", "data")],
        prevent_initial_call=True
    )
    def update_weights_store(slider_values, slider_ids, current_weights):
        """Met à jour le store avec les valeurs des sliders"""
        if not slider_values or not slider_ids:
            return dash.no_update
        
        # Initialiser le dictionnaire des poids
        weights = current_weights or {}
        
        # Mettre à jour les valeurs
        for i, slider_id in enumerate(slider_ids):
            metric_name = slider_id["metric"]
            weights[metric_name] = slider_values[i]
        
        return weights

    # 6. Callback pour appliquer les poids personnalisés
    @app.callback(
        Output("custom-weights-data", "data"),
        [Input("btn-apply-custom-weights", "n_clicks")],
        [State("weights-values-store", "data"),
         State("select-scoring-formula", "value")],
        prevent_initial_call=True
    )
    def apply_custom_weights(n_clicks, weights_data, formula):
        """Applique les poids personnalisés"""
        if not n_clicks or not weights_data:
            return dash.no_update
        
        # Structure de données avec formule et poids
        weights_with_formula = {
            "formula": formula,
            "weights": weights_data
        }
        
        if central_logger:
            ui_logger.info(f"Application des poids personnalisés pour {formula}")
        
        return weights_with_formula

    # ================== CALLBACK POUR ARRÊTER UNE OPTIMISATION ==================
    @app.callback(
        Output("optimization-status", "children", allow_duplicate=True),
        [Input({"type": "btn-stop-optimization", "study": dash.ALL}, "n_clicks")],
        [State({"type": "btn-stop-optimization", "study": dash.ALL}, "id")],
        prevent_initial_call=True
    )
    def stop_optimization(stop_clicks, stop_ids):
        """Arrête une optimisation en cours"""
        ctx = dash.callback_context
        if not ctx.triggered or not any(stop_clicks):
            return dash.no_update
        
        # Déterminer quelle optimisation arrêter
        button_idx = None
        for i, clicks in enumerate(stop_clicks):
            if clicks:
                button_idx = i
                break
        
        if button_idx is None:
            return dash.no_update
        
        # Récupérer le nom de l'étude
        study_name = stop_ids[button_idx]["study"]
        
        if central_logger:
            ui_logger.info(f"Demande d'arrêt de l'optimisation pour {study_name}")
        
        # Demander l'arrêt de l'optimisation
        try:
            from simulator.strategy_optimizer import StrategyOptimizer
            StrategyOptimizer.stop_optimization(study_name)
            
            # Mettre à jour le statut
            if study_name in running_optimizations:
                running_optimizations[study_name]['status'] = 'stopped'
            
            return dbc.Alert(
                [html.I(className="bi bi-check-circle me-2"),
                f"Demande d'arrêt de l'optimisation pour '{study_name}' envoyée. L'optimisation s'arrêtera après l'essai en cours."],
                color="success",
                dismissable=True
            )
            
        except Exception as e:
            if central_logger:
                ui_logger.error(f"Erreur lors de l'arrêt de l'optimisation: {str(e)}")
            
            return dbc.Alert(
                [html.I(className="bi bi-exclamation-triangle me-2"),
                f"Erreur lors de l'arrêt de l'optimisation: {str(e)}"],
                color="danger",
                dismissable=True
            )

    # ================== CALLBACK POUR REDÉMARRER UNE OPTIMISATION DEPUIS LA LISTE ==================
    @app.callback(
        Output("select-study-to-optimize", "value"),
        Output("optimization-tabs", "active_tab"),
        [Input({"type": "btn-restart-optimization-from-list", "study": dash.ALL}, "n_clicks")],
        [State({"type": "btn-restart-optimization-from-list", "study": dash.ALL}, "id")],
        prevent_initial_call=True
    )
    def restart_optimization_from_list(restart_clicks, restart_ids):
        """Redémarre une optimisation depuis la liste"""
        ctx = dash.callback_context
        if not ctx.triggered or not any(restart_clicks):
            return dash.no_update, dash.no_update
        
        # Déterminer quelle optimisation redémarrer
        button_idx = None
        for i, clicks in enumerate(restart_clicks):
            if clicks:
                button_idx = i
                break
        
        if button_idx is None:
            return dash.no_update, dash.no_update
        
        # Récupérer le nom de l'étude
        study_name = restart_ids[button_idx]["study"]
        
        if central_logger:
            ui_logger.info(f"Redémarrage de l'optimisation pour {study_name}")
        
        # Sélectionner l'étude et passer à l'onglet de configuration
        return study_name, "tab-configure"

    # ================== CALLBACK POUR LA NAVIGATION ENTRE ONGLETS ==================
    @app.callback(
        Output("optimization-tabs", "active_tab", allow_duplicate=True),
        [Input("link-to-config-tab-empty", "n_clicks"),
         Input("link-to-config-tab-recent", "n_clicks")],
        prevent_initial_call=True
    )
    def navigate_to_config_tab(empty_clicks, recent_clicks):
        """Navigue vers l'onglet de configuration"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update
        
        # Peu importe quel lien a été cliqué, on va à l'onglet de configuration
        return "tab-configure"

    # ================== CALLBACK POUR AFFICHER LES INFO DU FICHIER DE DONNÉES DE L'ÉTUDE ==================
    @app.callback(
        [Output("selected-data-file-display", "children"),
         Output("study-data-file-container", "className")],
        [Input("select-study-to-optimize", "value")]
    )
    def update_study_data_file_info(study_name):
        """Met à jour les informations du fichier de données de l'étude"""
        if not study_name:
            return [html.Span("Sélectionnez une étude pour voir les informations du fichier de données", className="text-muted")], "d-none"
        
        try:
            # Récupérer les informations du fichier de données
            study_manager = IntegratedStudyManager("studies")
            data_file = study_manager.get_study_data_file(study_name)
            
            if not data_file:
                # Aucun fichier associé
                return [
                    html.I(className="bi bi-exclamation-triangle text-warning me-2"),
                    html.Span("Aucun fichier de données associé à cette étude", className="text-warning")
                ], ""
            
            # Vérifier que le fichier existe
            if not os.path.exists(data_file):
                return [
                    html.I(className="bi bi-exclamation-triangle text-warning me-2"),
                    html.Span(f"Le fichier '{os.path.basename(data_file)}' n'existe pas", className="text-warning")
                ], ""
            
            # Obtenir des informations sur le fichier
            file_size = os.path.getsize(data_file) / (1024 * 1024)  # En Mo
            file_date = datetime.fromtimestamp(os.path.getmtime(data_file)).strftime("%Y-%m-%d %H:%M")
            
            # Obtenir le nombre de lignes (limité pour performance)
            line_count = 0
            with open(data_file, 'r') as f:
                for _ in range(10):  # Juste un échantillon pour estimer
                    if f.readline():
                        line_count += 1
                
                # Vérifier s'il y a plus de lignes
                if f.readline():
                    line_count = "> 10"
            
            return [
                html.Div([
                    html.I(className="bi bi-file-earmark-binary me-2 text-cyan-300"),
                    html.Strong(os.path.basename(data_file), className="text-cyan-200"),
                ], className="mb-1"),
                html.Div([
                    html.Span(f"Taille: {file_size:.2f} Mo", className="me-3 text-muted small"),
                    html.Span(f"Date: {file_date}", className="me-3 text-muted small"),
                    html.Span(f"Lignes: {line_count}", className="text-muted small"),
                ], className="d-flex")
            ], "d-none"
            
        except Exception as e:
            if central_logger:
                ui_logger.error(f"Erreur lors de la récupération des informations du fichier de données: {str(e)}")
            
            return [
                html.I(className="bi bi-exclamation-triangle text-warning me-2"),
                html.Span(f"Erreur: {str(e)}", className="text-warning")
            ], ""

    # ================== CALLBACKS POUR LA SÉLECTION ET LA COMPARAISON DES OPTIMISATIONS ==================
    # Store pour les études sélectionnées pour comparaison
    @app.callback(
        Output("expanded-optimizations-store", "data", allow_duplicate=True),
        [Input({"type": "btn-toggle-optimization-details", "study": dash.ALL}, "n_clicks")],
        [State({"type": "btn-toggle-optimization-details", "study": dash.ALL}, "id"),
         State({"type": "optimization-details-collapse", "study": dash.ALL}, "is_open"),
         State("expanded-optimizations-store", "data")],
        prevent_initial_call=True
    )
    def update_expanded_state(toggle_clicks, toggle_ids, collapse_states, current_state):
        """Met à jour l'état d'expansion des optimisations"""
        ctx = dash.callback_context
        if not ctx.triggered or not any(toggle_clicks):
            # Retourner l'état courant, pas dash.no_update
            return current_state or {}
        
        # Initialiser l'état s'il n'existe pas
        expanded_state = current_state or {}
        
        # Mettre à jour l'état pour chaque optimisation
        for i, (clicks, toggle_id, is_open) in enumerate(zip(toggle_clicks, toggle_ids, collapse_states)):
            if clicks:
                study_name = toggle_id["study"]
                expanded_state[study_name] = is_open
        
        return expanded_state

    # Callback pour gérer la sélection des optimisations à comparer
    @app.callback(
        [Output("optimization-comparison-container", "children"),
         Output({"type": "btn-select-for-compare", "study": dash.ALL}, "className")],
        [Input({"type": "btn-select-for-compare", "study": dash.ALL}, "n_clicks")],
        [State({"type": "btn-select-for-compare", "study": dash.ALL}, "id"),
         State({"type": "btn-select-for-compare", "study": dash.ALL}, "className")]
    )
    def handle_optimization_selection(select_clicks, select_ids, current_classes):
        """Gère la sélection des optimisations à comparer"""
        ctx = dash.callback_context
        
        # Même si aucun clic, on doit toujours retourner les classes actuelles et non dash.no_update
        # pour éviter l'erreur avec les sorties de type wildcard
        if not ctx.triggered or not any(select_clicks):
            default_content = html.Div([
                html.I(className="bi bi-info-circle me-2 text-cyan-300 opacity-50", style={"fontSize": "24px"}),
                html.Span("Sélectionnez plusieurs optimisations pour afficher une comparaison", className="text-muted"),
            ], className="text-center py-5")
            return default_content, current_classes
        
        # Initialiser la liste des études sélectionnées
        selected_studies = []
        updated_classes = []
        
        # Déterminer les études sélectionnées
        for i, (clicks, select_id, class_name) in enumerate(zip(select_clicks, select_ids, current_classes)):
            study_name = select_id["study"]
            
            # Vérifier si la classe indique que l'étude est déjà sélectionnée
            is_selected = "retro-badge-green" in class_name
            
            # Inverser l'état si le bouton a été cliqué
            if clicks and clicks % 2 == 1:
                is_selected = not is_selected
            
            # Mettre à jour la classe
            if is_selected:
                selected_studies.append(study_name)
                updated_classes.append("retro-badge retro-badge-green d-flex align-items-center justify-content-center")
            else:
                updated_classes.append("retro-badge retro-badge-blue d-flex align-items-center justify-content-center")
        
        # Si aucune étude sélectionnée, afficher un message
        if not selected_studies:
            return html.Div([
                html.I(className="bi bi-info-circle me-2 text-cyan-300 opacity-50", style={"fontSize": "24px"}),
                html.Span("Sélectionnez plusieurs optimisations pour afficher une comparaison", className="text-muted"),
            ], className="text-center py-5"), updated_classes
        
        # Si une seule étude sélectionnée, afficher une instruction
        if len(selected_studies) == 1:
            return html.Div([
                html.I(className="bi bi-info-circle me-2 text-cyan-300 opacity-50", style={"fontSize": "24px"}),
                html.Span("Sélectionnez au moins une autre optimisation pour afficher une comparaison", className="text-muted"),
                html.Div(f"Sélection actuelle: {selected_studies[0]}", className="mt-2 text-cyan-300")
            ], className="text-center py-5"), updated_classes
        
        # Créer une visualisation de comparaison pour les études sélectionnées
        try:
            study_manager = IntegratedStudyManager("studies")
            comparison_data = []
            
            # Récupérer les données d'optimisation pour chaque étude
            for study_name in selected_studies:
                optimization_results = study_manager.get_optimization_results(study_name)
                if optimization_results:
                    best_trial = None
                    for trial in optimization_results.get('best_trials', []):
                        if trial.get('trial_id') == optimization_results.get('best_trial_id'):
                            best_trial = trial
                            break
                    
                    if best_trial:
                        metrics = best_trial.get('metrics', {})
                        comparison_data.append({
                            'study': study_name,
                            'score': best_trial.get('score', 0),
                            'roi': metrics.get('roi', 0) * 100,
                            'win_rate': metrics.get('win_rate', 0) * 100,
                            'max_drawdown': metrics.get('max_drawdown', 0) * 100,
                            'profit_factor': metrics.get('profit_factor', 0),
                            'total_trades': metrics.get('total_trades', 0)
                        })
            
            # Si pas assez de données pour la comparaison
            if len(comparison_data) < 2:
                return html.Div([
                    html.I(className="bi bi-exclamation-triangle me-2 text-warning"),
                    html.Span("Données insuffisantes pour la comparaison", className="text-warning"),
                ], className="text-center py-5"), updated_classes
            
            # Créer un DataFrame pour la visualisation
            comparison_df = pd.DataFrame(comparison_data)
            
            # Créer des graphiques de comparaison
            
            # 1. Graphique radar pour comparer les métriques clés
            from plotly.subplots import make_subplots
            
            # Normaliser les métriques pour le radar
            radar_metrics = ['roi', 'win_rate', 'max_drawdown', 'profit_factor', 'total_trades']
            radar_data = []
            
            # Normalisation simple pour chaque métrique
            max_values = {
                'roi': max(comparison_df['roi'].max(), 1),
                'win_rate': 100,
                'max_drawdown': 100,
                'profit_factor': max(comparison_df['profit_factor'].max(), 1),
                'total_trades': max(comparison_df['total_trades'].max(), 1)
            }
            
            for _, row in comparison_df.iterrows():
                values = [
                    max(0, min(1, row['roi'] / max_values['roi'])),  # ROI
                    row['win_rate'] / max_values['win_rate'],        # Win rate
                    1 - row['max_drawdown'] / max_values['max_drawdown'],  # Drawdown (inversé)
                    min(1, row['profit_factor'] / max_values['profit_factor']),  # Profit factor
                    min(1, row['total_trades'] / max_values['total_trades'])   # Trades
                ]
                radar_data.append(values)
            
            # Créer un graphique radar
            radar_fig = go.Figure()
            
            # Ajouter chaque étude au radar
            category_names = ['ROI', 'Win Rate', 'Min Drawdown', 'Profit Factor', 'Volume']
            colors = ['rgb(34, 211, 238)', 'rgb(74, 222, 128)', 'rgb(139, 92, 246)', 'rgb(251, 191, 36)']
            
            for i, (study_name, values) in enumerate(zip(comparison_df['study'], radar_data)):
                radar_fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=category_names,
                    fill='toself',
                    name=study_name,
                    line_color=colors[i % len(colors)],
                    fillcolor=colors[i % len(colors)].replace('rgb', 'rgba').replace(')', ', 0.2)')
                ))
            
            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=40, b=40),
                height=400,
                title={
                    'text': 'Comparaison des métriques clés',
                    'font': {'size': 18, 'color': '#22d3ee'}
                }
            )
            
            # 2. Graphique à barres pour comparer les métriques
            bar_fig = go.Figure()
            
            # ROI
            bar_fig.add_trace(go.Bar(
                x=comparison_df['study'],
                y=comparison_df['roi'],
                name='ROI (%)',
                marker_color='rgb(34, 211, 238)'
            ))
            
            # Win Rate
            bar_fig.add_trace(go.Bar(
                x=comparison_df['study'],
                y=comparison_df['win_rate'],
                name='Win Rate (%)',
                marker_color='rgb(74, 222, 128)'
            ))
            
            # Drawdown
            bar_fig.add_trace(go.Bar(
                x=comparison_df['study'],
                y=comparison_df['max_drawdown'],
                name='Max Drawdown (%)',
                marker_color='rgb(248, 113, 113)'
            ))
            
            bar_fig.update_layout(
                barmode='group',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30,41,59,0.5)',
                margin=dict(l=40, r=40, t=60, b=80),
                height=400,
                title={
                    'text': 'Comparaison des performances',
                    'font': {'size': 18, 'color': '#22d3ee'}
                },
                xaxis=dict(
                    title="Études",
                    titlefont=dict(size=14, color='#d1d5db')
                ),
                yaxis=dict(
                    title="Valeur (%)",
                    titlefont=dict(size=14, color='#d1d5db')
                ),
                legend=dict(
                    orientation="h",
                    y=1.02,
                    x=0.5,
                    xanchor="center"
                )
            )
            
            # Assembler les graphiques dans un conteneur
            return html.Div([
                # En-tête
                html.Div([
                    html.H4("Comparaison des optimisations sélectionnées", className="text-cyan-300 mb-3"),
                    html.Div([
                        html.Span("Études sélectionnées: ", className="text-light me-2"),
                        html.Span(", ".join(selected_studies), className="text-cyan-200")
                    ], className="mb-3"),
                ], className="mb-4"),
                
                # Graphiques
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Profil des performances", className="mb-0 text-cyan-300")),
                            dbc.CardBody(
                                dcc.Graph(
                                    figure=radar_fig,
                                    config={'displayModeBar': False},
                                    className="retro-graph"
                                )
                            )
                        ], className="retro-card mb-4")
                    ], md=6),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Métriques principales", className="mb-0 text-cyan-300")),
                            dbc.CardBody(
                                dcc.Graph(
                                    figure=bar_fig,
                                    config={'displayModeBar': False},
                                    className="retro-graph"
                                )
                            )
                        ], className="retro-card mb-4")
                    ], md=6),
                ]),
                
                # Tableau de comparaison
                dbc.Card([
                    dbc.CardHeader(html.H5("Tableau comparatif", className="mb-0 text-cyan-300")),
                    dbc.CardBody(
                        html.Div([
                            html.Table([
                                html.Thead([
                                    html.Tr([
                                        html.Th("Étude", className="text-center"),
                                        html.Th("Score", className="text-center"),
                                        html.Th("ROI", className="text-center"),
                                        html.Th("Win Rate", className="text-center"),
                                        html.Th("Max DD", className="text-center"),
                                        html.Th("Profit Factor", className="text-center"),
                                        html.Th("Trades", className="text-center")
                                    ], className="retro-table-header")
                                ]),
                                html.Tbody([
                                    html.Tr([
                                        html.Td(row['study'], className="text-center"),
                                        html.Td(f"{row['score']:.4f}", className="text-center"),
                                        html.Td(f"{row['roi']:.2f}%", className=f"text-{'success' if row['roi'] > 0 else 'danger'} text-center"),
                                        html.Td(f"{row['win_rate']:.2f}%", className="text-center"),
                                        html.Td(f"{row['max_drawdown']:.2f}%", className="text-danger text-center"),
                                        html.Td(f"{row['profit_factor']:.2f}", className="text-center"),
                                        html.Td(f"{int(row['total_trades'])}", className="text-center")
                                    ]) for _, row in comparison_df.iterrows()
                                ])
                            ], className="retro-table w-100")
                        ], className="table-responsive")
                    )
                ], className="retro-card")
            ]), updated_classes
        
        except Exception as e:
            if central_logger:
                ui_logger.error(f"Erreur lors de la création de la comparaison: {str(e)}")
            
            return html.Div([
                html.I(className="bi bi-exclamation-triangle me-2 text-warning"),
                html.Span(f"Erreur lors de la création de la comparaison: {str(e)}", className="text-warning"),
            ], className="text-center py-5"), updated_classes