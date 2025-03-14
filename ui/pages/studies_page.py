"""
Page "Études & Optimisations" de l'application de trading.
Permet de gérer, visualiser et optimiser les études de trading.
"""
from dash import html, dcc, Input, Output, State, callback, callback_context
import dash_bootstrap_components as dbc
import dash
import json
from datetime import datetime

# Logger et config
from logger.logger import LoggerType
from simulator.study_manager import IntegratedStudyManager
from simulator.config import TradingConfig, RiskMode, MarginMode, TradingMode

# Composants
from ui.components.studies.study_list import create_study_list, register_study_list_callbacks
from ui.components.studies.study_detail import create_study_detail
from ui.components.studies.optimization_panel import create_optimization_panel, register_optimization_callbacks
from ui.components.studies.study_creator import create_study_creator, register_study_creator_callbacks

def create_studies_page(central_logger=None):
    """
    Page "Études & Optimisations"
    - Bouton NEW STUDY qui ouvre le modal
    - Tabs Mes Études / Optimisations
    - Callback pour charger le détail d'une étude, etc.
    """
    if central_logger:
        ui_logger = central_logger.get_logger("studies_page", LoggerType.UI)
        ui_logger.info("Création de la page des études")

    # On peut récupérer la liste des études pour l'afficher si besoin
    try:
        study_manager = IntegratedStudyManager("studies")
        studies = study_manager.list_studies()
    except Exception as e:
        if central_logger:
            ui_logger.error(f"Erreur lors de la récupération des études: {str(e)}")
        studies = []

    return html.Div([
        html.H2(
            "ÉTUDES & OPTIMISATIONS",
            className="text-xl text-cyan-300 font-bold mb-6 border-b border-gray-700 pb-2"
        ),

        dbc.Tabs([
            dbc.Tab(label="Mes Études", tab_id="tab-studies-list"),
            dbc.Tab(label="Optimisations", tab_id="tab-optimizations"),
        ], id="studies-tabs", active_tab="tab-studies-list"),

        html.Div(id="studies-content", className="mt-4"),

        # Modal de création d'étude (avancée)
        create_study_creator(central_logger),

        # Stores
        dcc.Store(id="current-study-id", data=None),
        dcc.Store(id="study-action", data=None),
        dcc.Store(id="open-study-creator-trigger", data=None),  # Pour déclencher l'ouverture du modal

    ])


def register_studies_callbacks(app, central_logger=None):
    """
    Enregistre les callbacks pour la page des études
    
    Args:
        app: L'instance de l'application Dash
        central_logger: Instance du logger centralisé
    """
    ui_logger = central_logger.get_logger("studies_callbacks", LoggerType.UI) if central_logger else None

    # Enregistrement des callbacks
    register_study_creator_callbacks(app, central_logger)
    register_study_list_callbacks(app, central_logger)
    register_optimization_callbacks(app, central_logger)

    # ================== AFFICHAGE SELON ONGLET / ACTION ==================
    @app.callback(
        Output("studies-content", "children"),
        [
            Input("studies-tabs", "active_tab"),
            Input("current-study-id", "data"),
            Input("study-action", "data")
        ]
    )
    def render_studies_content(active_tab, study_id, action):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        if trigger_id in ["studies-tabs", None]:
            if active_tab == "tab-studies-list":
                return create_study_list(central_logger)
            elif active_tab == "tab-optimizations":
                return create_optimization_panel(central_logger)

        elif trigger_id == "current-study-id" and study_id:
            return create_study_detail(study_id, central_logger)

        elif trigger_id == "study-action" and action:
            action_data = json.loads(action) if isinstance(action, str) else action
            action_type = action_data.get('type')

            if action_type == "create-success":
                if ui_logger:
                    ui_logger.info(f"Étude créée avec succès: {action_data.get('study_name')}")
                return create_study_list(central_logger)

            elif action_type == "back-to-list":
                return create_study_list(central_logger)

            elif action_type == "refresh-detail" and action_data.get('study_id'):
                return create_study_detail(action_data.get('study_id'), central_logger)

        return create_study_list(central_logger)

    # Callback pour capturer les événements des boutons et les stocker dans le trigger
    @app.callback(
        Output("open-study-creator-trigger", "data"),
        [
            Input({"type": "btn-new-study-header", "index": dash.ALL}, "n_clicks"),
            Input({"type": "btn-new-study-empty", "index": dash.ALL}, "n_clicks")
        ],
        prevent_initial_call=True
    )
    def capture_study_creator_buttons(header_clicks, empty_clicks):
        """
        Capture les clics sur n'importe quel bouton de création d'étude
        """
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update
            
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Vérifier si un des boutons a été cliqué
        if (header_clicks and any(header_clicks)) or (empty_clicks and any(empty_clicks)):
            if ui_logger:
                ui_logger.info(f"Bouton de création d'étude cliqué: {trigger_id}")
            return {"timestamp": datetime.now().isoformat()}
            
        return dash.no_update

    # CALLBACK CENTRALISÉ pour l'ouverture/fermeture du modal
    @app.callback(
        Output("study-creator-modal", "is_open"),
        [
            Input("open-study-creator-trigger", "data"),
            Input("btn-close-study-creator", "n_clicks"),
            Input("study-action", "data")
        ],
        [State("study-creator-modal", "is_open")],
        prevent_initial_call=True
    )
    def toggle_study_creator_modal(trigger_data, n_close, action_data, is_open):
        """
        Gère l'ouverture/fermeture du modal de création d'étude.
        """
        ctx = callback_context
        if not ctx.triggered:
            return is_open

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Si le déclencheur est activé, ouvrir le modal
        if button_id == "open-study-creator-trigger" and trigger_data:
            if ui_logger:
                ui_logger.info("Ouverture du modal de création d'étude via déclencheur")
            return True
        
        # Sur annuler => ferme
        if button_id == "btn-close-study-creator" and n_close:
            if ui_logger:
                ui_logger.info("Fermeture du modal de création d'étude (annulation)")
            return False
            
        # Sur action réussie => ferme
        if button_id == "study-action" and action_data:
            try:
                action = json.loads(action_data) if isinstance(action_data, str) else action_data
                if action.get("type") == "create-success":
                    if ui_logger:
                        ui_logger.info("Fermeture du modal de création d'étude (création réussie)")
                    return False
            except:
                pass
        
        return is_open