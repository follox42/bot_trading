"""
Composant amélioré pour l'affichage de la liste des études.
Utilise un design rétro attrayant avec des animations et des effets visuels.
"""
from dash import html, dcc, Input, Output, State, callback, callback_context
import dash_bootstrap_components as dbc
import dash
import json
from datetime import datetime

from logger.logger import LoggerType

def create_study_list(central_logger=None):
    """
    Crée le composant d'affichage amélioré de la liste des études.
    
    Args:
        central_logger: Instance du logger centralisé
    
    Returns:
        Composant d'affichage amélioré de la liste des études
    """
    # Initialiser le logger
    if central_logger:
        ui_logger = central_logger.get_logger("study_list", LoggerType.UI)
        ui_logger.info("Chargement de la liste des études améliorée")
    
    # Récupérer la liste des études
    try:
        from simulator.study_manager import IntegratedStudyManager
        study_manager = IntegratedStudyManager("studies")
        studies = study_manager.get_studies_list()
    except Exception as e:
        if central_logger:
            ui_logger.error(f"Erreur lors de la récupération des études: {str(e)}")
        studies = []
    
    # Création du layout complet
    return html.Div([
        # En-tête avec effet néon et boutons d'action
        html.Div(
            className="retro-header mb-4",
            children=[
                html.Div(
                    className="d-flex justify-content-between align-items-center",
                    children=[
                        html.H3(
                            "MES ÉTUDES", 
                            className="retro-title neon-text mb-0"
                        ),
                        html.Div(
                            className="retro-button-group",
                            children=[
                                html.Button(
                                    [html.I(className="bi bi-plus-circle me-2"), "NOUVELLE ÉTUDE"],
                                    id={"type": "btn-new-study-header", "index": 0},  # Pattern-matching ID
                                    n_clicks=0,
                                    className="retro-button retro-glow me-2"
                                ),
                                html.Button(
                                    [html.I(className="bi bi-arrow-repeat me-2"), "ACTUALISER"],
                                    id="btn-refresh-studies",
                                    className="retro-button retro-glow secondary"
                                ),
                            ]
                        )
                    ]
                )
            ]
        ),
        
        # Barre de recherche améliorée avec filtres
        html.Div(
            className="retro-search-bar mb-4",
            children=[
                dbc.Row([
                    dbc.Col(
                        html.Div(
                            className="search-input-container",
                            children=[
                                html.I(className="bi bi-search search-icon"),
                                dbc.Input(
                                    id="filter-studies",
                                    type="text",
                                    placeholder="Rechercher une étude...",
                                    className="retro-search-input"
                                )
                            ]
                        ),
                        width=12, lg=7
                    ),
                    dbc.Col(
                        html.Div(
                            className="filter-dropdown-container",
                            children=[
                                html.I(className="bi bi-funnel filter-icon"),
                                dcc.Dropdown(
                                    id="filter-studies-status",
                                    options=[
                                        {"label": "Tous les statuts", "value": "all"},
                                        {"label": "Nouveaux", "value": "created"},
                                        {"label": "Optimisés", "value": "optimized"},
                                        {"label": "Clonés", "value": "cloned"},
                                        {"label": "Archivés", "value": "archived"},
                                    ],
                                    value="all",
                                    className="retro-dropdown"
                                )
                            ]
                        ),
                        width=12, lg=5
                    ),
                ])
            ]
        ),
        
        # Grille des études avec effet d'apparition
        html.Div(
            id="studies-container",
            className="fade-in-animation",
            children=create_studies_grid(studies) if studies else create_empty_state()
        ),
    ])

def create_studies_grid(studies):
    """
    Crée une grille de cartes d'études.
    
    Args:
        studies: Liste des études à afficher
        
    Returns:
        Composant de grille d'études
    """
    return dbc.Row(
        [
            dbc.Col(
                create_retro_study_card(study),
                width=12, md=6, xl=4,
                className="study-card-container mb-4"  # Ajout de mb-4 pour marge en bas
            )
            for study in studies
        ],
        className="studies-grid g-3"  # Ajout de g-3 pour gutter entre colonnes
    )

def create_empty_state():
    """
    Crée un état vide pour quand aucune étude n'est disponible.
    
    Returns:
        Composant d'état vide
    """
    return html.Div(
        className="retro-empty-state p-5 text-center",
        children=[
            html.I(className="bi bi-file-earmark-plus display-4 mb-3 text-cyan-300"),
            html.H4("Aucune étude disponible", className="mb-3 retro-text"),
            html.P(
                "Commencez par créer une nouvelle étude pour explorer des stratégies de trading.",
                className="text-muted mb-4"
            ),
            html.Button(
                [html.I(className="bi bi-plus-circle me-2"), "CRÉER UNE ÉTUDE"],
                id={"type": "btn-new-study-empty", "index": 0},  # Pattern-matching ID
                className="retro-button retro-glow"
            )
        ]
    )

def create_retro_study_card(study):
    """
    Crée une carte d'étude améliorée avec design rétro.
    
    Args:
        study: Les données de l'étude
        
    Returns:
        Composant Dash pour une carte d'étude
    """
    # Déterminer les badges selon le statut
    status = study.get('status', 'unknown')
    status_badge = None
    
    if status == 'created':
        status_badge = html.Span("NOUVEAU", className="retro-badge retro-badge-blue")
    elif status == 'optimized':
        status_badge = html.Span("OPTIMISÉ", className="retro-badge retro-badge-green")
    elif status == 'cloned':
        status_badge = html.Span("CLONÉ", className="retro-badge retro-badge-purple")
    elif status == 'archived':
        status_badge = html.Span("ARCHIVÉ", className="retro-badge retro-badge-yellow")
    
    # Déterminer les icônes supplémentaires
    icons = []
    if study.get('has_optimization', False):
        icons.append(html.I(className="bi bi-speedometer2 ms-2 text-cyan-300", title="Optimisé"))
    
    if study.get('strategies_count', 0) > 0:
        icons.append(html.I(className="bi bi-graph-up ms-2 text-green-300", title=f"{study.get('strategies_count', 0)} stratégies"))
    
    strategies_count = study.get('strategies_count', 0)
    
    # Créer la carte d'étude avec design amélioré
    # Rendre la carte entière cliquable en l'enveloppant dans un div avec ID pour l'ouverture
    return html.Div(
        id={"type": "clickable-study-card", "index": study.get('name')},
        className="retro-study-card h-100 cursor-pointer", # Ajout de h-100 pour hauteur complète et cursor-pointer
        style={"cursor": "pointer"},  # Curseur en forme de main pour indiquer que c'est cliquable
        children=[
            # En-tête avec effet de brillance
            html.Div(
                className="retro-study-header",
                children=[
                    html.Div(
                        className="header-content d-flex justify-content-between align-items-center",
                        children=[
                            html.H5(
                                study.get('name', 'Étude sans nom'), 
                                className="card-title mb-0 text-white"
                            ),
                            html.Div([status_badge, *icons] if status_badge else icons, className="ms-2 d-flex align-items-center")
                        ]
                    ),
                    # Effet de brillance animé
                    html.Div(className="header-glow-effect")
                ]
            ),
            
            # Corps de la carte avec effet de profondeur
            html.Div(
                className="retro-study-body",
                children=[
                    # Informations principales sur deux colonnes avec style rétro
                    html.Div(
                        className="info-grid",
                        children=[
                            # Colonne gauche
                            html.Div(
                                className="info-column",
                                children=[
                                    html.Div(
                                        className="info-item",
                                        children=[
                                            html.Div("ASSET", className="info-label"),
                                            html.Div(
                                                study.get('asset', 'N/A'), 
                                                className="info-value"
                                            )
                                        ]
                                    ),
                                    html.Div(
                                        className="info-item",
                                        children=[
                                            html.Div("TIMEFRAME", className="info-label"),
                                            html.Div(
                                                study.get('timeframe', 'N/A'), 
                                                className="info-value"
                                            )
                                        ]
                                    ),
                                ]
                            ),
                            
                            # Colonne droite
                            html.Div(
                                className="info-column",
                                children=[
                                    html.Div(
                                        className="info-item",
                                        children=[
                                            html.Div("DATE", className="info-label"),
                                            html.Div(
                                                study.get('creation_date', 'N/A'), 
                                                className="info-value"
                                            )
                                        ]
                                    ),
                                    html.Div(
                                        className="info-item",
                                        children=[
                                            html.Div("STRATÉGIES", className="info-label"),
                                            html.Div(
                                                f"{strategies_count}", 
                                                className=f"info-value {'text-green-300' if strategies_count > 0 else ''}"
                                            )
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                    
                    # Description avec contour stylisé
                    html.Div(
                        className="description-container",
                        children=[
                            html.P(
                                study.get('description', 'Aucune description disponible.'),
                                className="study-description"
                            )
                        ]
                    ),
                    
                    # Bouton d'optimisation uniquement (sans le bouton Ouvrir car la carte est cliquable)
                    html.Div(
                        className="action-buttons mt-3",
                        children=[
                            html.Button(
                                html.Span([
                                    html.I(className="bi bi-lightning-charge me-2"),
                                    "OPTIMISER"
                                ]),
                                id={"type": "btn-optimize-study", "index": study.get('name')},
                                className="retro-button w-100",
                                # Pour éviter que le clic sur le bouton n'ouvre aussi l'étude
                                **{"data-stop-propagation": "true"}
                            )
                        ]
                    )
                ]
            )
        ]
    )

def register_study_list_callbacks(app, central_logger=None):
    """
    Enregistre les callbacks pour la liste des études améliorée
    
    Args:
        app: L'instance de l'application Dash
        central_logger: Instance du logger centralisé
    """
    # Initialiser le logger
    if central_logger:
        ui_logger = central_logger.get_logger("retro_study_list_callbacks", LoggerType.UI)
    
    # Callback pour les cartes d'études cliquables
    @app.callback(
        Output("current-study-id", "data"),
        [Input({"type": "clickable-study-card", "index": dash.ALL}, "n_clicks")],
        prevent_initial_call=True
    )
    def open_study_from_card(n_clicks_list):
        """Ouvre l'étude lorsqu'on clique sur la carte"""
        ctx = dash.callback_context
        if not ctx.triggered or not any(n_clicks_list):
            return dash.no_update
            
        # Récupérer l'ID du composant qui a déclenché le callback
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        card_data = json.loads(trigger_id)
        study_name = card_data.get("index")
        
        if central_logger:
            ui_logger.info(f"Ouverture de l'étude par clic sur carte: {study_name}")
        
        return study_name
    
    # Callback pour éviter la propagation du clic sur le bouton d'optimisation
    app.clientside_callback(
        """
        function(n_clicks) {
            if (n_clicks) {
                // Empêche la propagation de l'événement de clic
                var buttons = document.querySelectorAll('[data-stop-propagation="true"]');
                buttons.forEach(function(button) {
                    button.addEventListener('click', function(e) {
                        e.stopPropagation();
                    });
                });
            }
            return '';
        }
        """,
        Output("_", "children"),
        [Input("studies-container", "children")],
        prevent_initial_call=True
    )
    
    # Callback pour le bouton d'optimisation
    @app.callback(
        [Output("studies-tabs", "active_tab"),
         Output("current-study-id", "data", allow_duplicate=True)],
        [Input({"type": "btn-optimize-study", "index": dash.ALL}, "n_clicks")],
        prevent_initial_call=True
    )
    def optimize_study(n_clicks_list):
        """Redirige vers l'onglet d'optimisation avec l'étude sélectionnée"""
        ctx = dash.callback_context
        if not ctx.triggered or not any(n_clicks_list):
            return dash.no_update, dash.no_update
        
        # Récupérer l'ID de l'étude à partir du bouton cliqué
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        button_data = json.loads(trigger_id)
        study_name = button_data["index"]
        
        if central_logger:
            ui_logger.info(f"Redirection vers l'optimisation pour l'étude: {study_name}")
        
        # Changer d'onglet et définir l'étude courante
        return "tab-optimizations", study_name

    # Callback pour filtrer les études
    @app.callback(
        Output("studies-container", "children"),
        [Input("filter-studies", "value"),
         Input("filter-studies-status", "value"),
         Input("btn-refresh-studies", "n_clicks"),
         Input("study-action", "data")],  # Pour mettre à jour après la création/suppression
        prevent_initial_call=True
    )
    def filter_studies(search_term, status_filter, n_refresh, study_action):
        """Filtre les études selon les critères de recherche et de statut"""
        # Récupérer la liste des études
        try:
            from simulator.study_manager import IntegratedStudyManager
            study_manager = IntegratedStudyManager("studies")
            studies = study_manager.get_studies_list()
        except Exception as e:
            if central_logger:
                ui_logger.error(f"Erreur lors de la récupération des études: {str(e)}")
            studies = []
        
        # Filtrer par texte de recherche
        if search_term:
            search_term = search_term.lower()
            studies = [
                s for s in studies if (
                    search_term in s.get('name', '').lower() or
                    search_term in s.get('description', '').lower() or
                    search_term in s.get('asset', '').lower() or
                    search_term in s.get('timeframe', '').lower() or
                    search_term in s.get('exchange', '').lower()
                )
            ]
        
        # Filtrer par statut
        if status_filter and status_filter != "all":
            studies = [s for s in studies if s.get('status') == status_filter]
        
        # Si aucune étude ne correspond aux filtres, afficher un message
        if not studies:
            return create_empty_state()
            
        # Créer la grille d'études
        return create_studies_grid(studies)