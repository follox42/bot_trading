from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import dash
import json
from datetime import datetime
import os

from logger.logger import LoggerType
from ui.components.optimization_dashboard import create_optimization_dashboard, register_optimization_dashboard_callbacks
from ui.components.optimization_creator import create_optimization_creator, register_optimization_creator_callbacks
from ui.components.optimization_details import create_optimization_details, register_optimization_details_callbacks
import config

def create_optimization_page(central_logger=None):
    """
    Crée la page de gestion des optimisations
    
    Args:
        central_logger: Instance du logger centralisé
    
    Returns:
        Layout de la page des optimisations
    """
    # Initialiser le logger
    if central_logger:
        ui_logger = central_logger.get_logger("optimization_page", LoggerType.UI)
        ui_logger.info("Création de la page des optimisations")
    
    return html.Div([
        html.H2("OPTIMISATIONS", className="text-xl text-cyan-300 font-bold mb-6 border-b border-gray-700 pb-2"),
        
        # Deux onglets principaux: Liste des optimisations et Création
        dbc.Tabs([
            dbc.Tab(
                create_optimization_dashboard(central_logger),
                label="OPTIMISATIONS EN COURS",
                tab_id="tab-optimization-list",
                activeTabClassName="text-cyan-300 border-cyan-300",
                tabClassName="text-gray-400"
            ),
            dbc.Tab(
                create_optimization_creator(central_logger),
                label="NOUVELLE OPTIMISATION",
                tab_id="tab-optimization-create",
                activeTabClassName="text-cyan-300 border-cyan-300",
                tabClassName="text-gray-400"
            ),
        ], id="optimization-tabs", active_tab="tab-optimization-list", className="mb-4"),
        
        # Zone pour afficher les détails d'une optimisation sélectionnée
        html.Div(id="optimization-details-container", style={"display": "none"}),
        
        # Zone de stockage des états
        dcc.Store(id="selected-optimization", data=""),
        dcc.Store(id="optimizations-list-data", data=json.dumps([])),
        dcc.Store(id="optimization-page-loaded", data="true"),
        
        # Intervalle pour rafraîchir automatiquement les optimisations en cours
        dcc.Interval(
            id="optimization-refresh-interval",
            interval=5000,  # 5 secondes
            n_intervals=0
        )
    ])

def register_optimization_callbacks(app, central_logger=None):
    """
    Enregistre les callbacks pour la page des optimisations
    
    Args:
        app: L'instance de l'application Dash
        central_logger: Instance du logger centralisé
    """
    # Initialiser le logger
    if central_logger:
        ui_logger = central_logger.get_logger("optimization_page", LoggerType.UI)
    
    # Callback pour gérer la sélection d'une optimisation
    @app.callback(
        [Output("optimization-details-container", "children"),
         Output("optimization-details-container", "style"),
         Output("selected-optimization", "data")],
        [Input("optimization-view-btn", "n_clicks"),
         Input("optimization-back-btn", "n_clicks")],
        [State("optimization-view-btn", "value"),
         State("selected-optimization", "data")]
    )
    def handle_optimization_selection(view_clicks, back_clicks, selected_value, current_selection):
        """Gère l'affichage des détails d'une optimisation sélectionnée"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update
            
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == "optimization-view-btn" and view_clicks:
            if selected_value:
                # Afficher les détails de l'optimisation sélectionnée
                if central_logger:
                    ui_logger.info(f"Sélection de l'optimisation: {selected_value}")
                
                return create_optimization_details(selected_value, central_logger), {"display": "block"}, selected_value
                
        elif button_id == "optimization-back-btn" and back_clicks:
            # Retour à la liste des optimisations
            if central_logger:
                ui_logger.info("Retour à la liste des optimisations")
                
            return dash.no_update, {"display": "none"}, ""
            
        return dash.no_update, dash.no_update, current_selection
    
    # Enregistrer les callbacks spécifiques aux composants
    register_optimization_dashboard_callbacks(app, central_logger)
    register_optimization_creator_callbacks(app, central_logger)
    register_optimization_details_callbacks(app, central_logger)