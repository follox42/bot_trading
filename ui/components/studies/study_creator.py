"""
Module principal de création d'études pour les stratégies de trading.
Version améliorée avec une meilleure intégration de la configuration et du gestionnaire d'études.
"""
import dash
from dash import html, dcc, Input, Output, State, ALL, callback_context
import dash_bootstrap_components as dbc
import json
import os
import traceback
from datetime import datetime
import pandas as pd
import time

from logger.logger import CentralizedLogger, LogLevel, LoggerType
from simulator.risk import RiskMode
from simulator.config import (
    MarginMode, TradingMode, TradingConfig, IndicatorConfig, 
    create_trading_default_config
)
from simulator.study_manager import IntegratedStudyManager
from simulator.study_config_definitions import INDICATOR_DEFAULTS
from simulator.indicators import IndicatorType

# Importations des modules UI
from ui.components.retro_ui import register_retro_ui_callbacks
from ui.components.studies.tabs.basic_info_tab import create_basic_info_tab, register_basic_info_callbacks
from ui.components.studies.tabs.simulation_tab import create_simulation_tab, register_simulation_callbacks
from ui.components.studies.tabs.risk_tab import create_risk_tab, register_risk_callbacks
from ui.components.studies.tabs.structure_tab import create_structure_tab, register_structure_callbacks


def create_study_creator(central_logger=None):
    """
    Crée le composant modal de création d'étude avec son contenu.
    
    Args:
        central_logger: Instance du logger centralisé
        
    Returns:
        Un composant modal Dash
    """
    # Initialiser le logger
    ui_logger = None
    if central_logger:
        ui_logger = central_logger.get_logger("study_creator", LoggerType.UI)
        ui_logger.info("Initialisation du créateur d'étude amélioré")
    
    # Création du modal
    return dbc.Modal(
        [
            dbc.ModalHeader(
                html.H4("CRÉATION D'ÉTUDE", className="retro-title text-cyan-300"),
                close_button=True
            ),
            dbc.ModalBody([
                # Zone d'alerte pour les messages
                html.Div(id="study-create-alert"),
                
                # Panneau à onglets pour organiser les paramètres
                dbc.Tabs([
                    # Onglet Informations de Base
                    dbc.Tab(
                        create_basic_info_tab(),
                        label="Informations",
                        tab_id="tab-study-info",
                        className="retro-tabs"
                    ),
                    
                    # Onglet Simulation
                    dbc.Tab(
                        create_simulation_tab(),
                        label="Simulation",
                        tab_id="tab-simulation",
                        className="retro-tabs"
                    ),
                    
                    # Onglet Gestion du Risque
                    dbc.Tab(
                        create_risk_tab(),
                        label="Risque",
                        tab_id="tab-risk",
                        className="retro-tabs"
                    ),
                    
                    # Onglet Structure de Stratégie
                    dbc.Tab(
                        create_structure_tab(),
                        label="Structure",
                        tab_id="tab-structure",
                        className="retro-tabs"
                    ),
                ], id="study-creator-tabs", active_tab="tab-study-info"),
            ]),
            dbc.ModalFooter([
                html.Button(
                    "Annuler",
                    id="btn-close-study-creator",
                    className="retro-button secondary me-2",
                    n_clicks=0
                ),
                html.Button(
                    [html.I(className="bi bi-plus-lg me-2"), "Créer l'étude"],
                    id="btn-create-study",
                    className="retro-button",
                    n_clicks=0
                ),
            ]),
        ],
        id="study-creator-modal",
        size="xl",
        is_open=False,
        centered=True,
        scrollable=True,
        className="retro-modal"
    )


def register_study_creator_callbacks(app, central_logger=None):
    """
    Enregistre tous les callbacks nécessaires pour le créateur d'étude
    
    Args:
        app: L'instance de l'application Dash
        central_logger: Instance du logger centralisé
    """
    ui_logger = None
    if central_logger:
        ui_logger = central_logger.get_logger("study_creator_callbacks", LoggerType.UI)
    
    # Enregistrement des callbacks des composants UI de base
    register_retro_ui_callbacks(app)
    
    # Enregistrement des callbacks spécifiques aux onglets
    register_basic_info_callbacks(app)
    register_simulation_callbacks(app)
    register_risk_callbacks(app)
    register_structure_callbacks(app)
    
    # Fonction pour traiter les valeurs de plage
    def process_range_value(range_str, default_min, default_max):
        """Traite une chaîne de plage pour extraire min et max"""
        if not range_str:
            return (default_min, default_max)
        
        parts = range_str.split('-')
        if len(parts) == 2:
            try:
                return (float(parts[0]), float(parts[1]))
            except:
                pass
                
        return (default_min, default_max)
    
    # Callback pour la création d'étude
    @app.callback(
        [
            Output("study-action", "data"), 
            Output("study-create-alert", "children"),
            Output("input-study-name", "value"),
            Output("input-study-description", "value")
        ],
        [
            Input("btn-create-study", "n_clicks")
        ],
        [
            # Informations de base
            State("input-study-name", "value"),
            State("input-study-asset", "value"),
            State("input-study-timeframe", "value"),
            State("input-study-exchange", "value"),
            State("input-study-description", "value"),
            # Données sélectionnées
            State("selected-data-file-path", "data"),
            State("selected-data-metadata", "data"),
            
            # Valeurs des inputs fixes pour la simulation
            State({"type": "fixed-input", "id": "initial-balance"}, "value"),
            State({"type": "fixed-input", "id": "fee"}, "value"),
            State({"type": "fixed-input", "id": "slippage"}, "value"),
            # Seul leverage est un range slider
            State({"type": "range-input", "id": "leverage"}, "value"),
            
            # Modes actifs
            State({"type": "trading-mode-toggle", "index": ALL}, "data-active"),
            State({"type": "trading-mode-toggle", "index": ALL}, "data-value"),
            State({"type": "margin-mode-toggle", "index": ALL}, "data-active"),
            State({"type": "margin-mode-toggle", "index": ALL}, "data-value"),
            
            # Risque
            State({"type": "risk-mode-toggle", "index": ALL}, "data-active"),
            State({"type": "risk-mode-toggle", "index": ALL}, "data-value"),
            State({"type": "range-input", "id": "position-size"}, "value"),
            State({"type": "range-input", "id": "stop-loss"}, "value"),
            State({"type": "range-input", "id": "take-profit"}, "value"),
            
            # Structure
            State({"type": "range-input", "id": "blocks-count"}, "value"),
            State({"type": "range-input", "id": "conditions-per-block"}, "value"),
            State({"type": "range-input", "id": "cross-probability"}, "value"),
            State({"type": "range-input", "id": "value-comparison-probability"}, "value"),

            # Indicateurs - Switches d'activation
            State({"type": "indicator-switch", "name": ALL}, "value"),
            State({"type": "indicator-switch", "name": ALL}, "id"),
            
            # Indicateurs - Paramètres de période
            State({"type": "indicator-min-period", "name": ALL}, "value"),
            State({"type": "indicator-max-period", "name": ALL}, "value"),
            State({"type": "indicator-step", "name": ALL}, "value"),
            State({"type": "indicator-price-type", "name": ALL}, "value")
        ],
        prevent_initial_call=True
    )
    def handle_study_creation(
        create_clicks,
        # Infos de base
        study_name, asset, timeframe, exchange, description,
        data_file_path, data_file_metadata,
        # Simulation
        initial_balance, fee, slippage, leverage_range,
        trading_mode_active, trading_mode_values, margin_mode_active, margin_mode_values,
        # Risque
        risk_mode_active, risk_mode_values, position_size_range, sl_range, tp_range,
        # Structure
        blocks_range, conditions_range, cross_probability_range, value_comparison_probability_range,
        # Indicateurs
        indicators_enabled, indicators_ids, 
        indicator_min_periods, indicator_max_periods, indicator_steps, indicator_price_types
    ):
        """Gère la création d'une étude et la réinitialisation du formulaire"""
        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
        
        # Valeurs par défaut pour le retour
        study_action_data = None
        alert_children = None
        new_study_name = dash.no_update
        new_description = dash.no_update
        
        # Bouton de création d'étude cliqué
        if trigger_id == "btn-create-study" and create_clicks:
            if ui_logger:
                ui_logger.info(f"Tentative de création d'étude: {study_name}")
            
            # Validation des entrées
            if not study_name:
                return None, dbc.Alert("Veuillez saisir un nom pour l'étude.", color="danger"), dash.no_update, dash.no_update
            
            if not asset or not timeframe or not exchange:
                return None, dbc.Alert("Veuillez remplir les informations d'asset, timeframe et exchange.", color="danger"), dash.no_update, dash.no_update

            # Vérifier qu'au moins un indicateur est activé
            enabled_indicators = [ind_id["name"] for i, ind_id in enumerate(indicators_ids) if indicators_enabled[i]]
            
            if not enabled_indicators:
                return None, dbc.Alert(
                    [
                        html.I(className="bi bi-exclamation-circle me-2"),
                        "Vous devez activer au moins un indicateur technique pour créer une étude."
                    ],
                    color="danger",
                    dismissable=True
                ), dash.no_update, dash.no_update
        
            if not data_file_path:
                return None, dbc.Alert("Veuillez sélectionner un fichier de données pour cette étude.", color="danger"), dash.no_update, dash.no_update
            
            try:
                # Création d'une instance du gestionnaire d'études
                study_manager = IntegratedStudyManager("studies")
                
                # Vérifier si l'étude existe déjà
                if study_manager.study_exists(study_name):
                    return None, dbc.Alert(f"Une étude nommée '{study_name}' existe déjà.", color="danger"), dash.no_update, dash.no_update
                
                # Traitement des valeurs de configuration
                # Pour les inputs fixes, utiliser directement la valeur
                bal_value = float(initial_balance) if initial_balance is not None else 10000.0
                fee_value = float(fee) if fee is not None else 0.01
                slip_value = float(slippage) if slippage is not None else 0.01
                
                # Pour leverage qui est un range slider
                lev_min, lev_max = process_range_value(leverage_range, 1, 10)
                
                # Récupérer les modes actifs
                active_trading_modes = [
                    int(val) for i, (is_active, val) in enumerate(zip(trading_mode_active, trading_mode_values))
                    if is_active == "true"
                ]
                
                active_margin_modes = [
                    int(val) for i, (is_active, val) in enumerate(zip(margin_mode_active, margin_mode_values))
                    if is_active == "true"
                ]
                
                active_risk_modes = [
                    val for i, (is_active, val) in enumerate(zip(risk_mode_active, risk_mode_values))
                    if is_active == "true"
                ]
                
                # Traitement des plages pour le risque
                pos_min, pos_max = process_range_value(position_size_range, 0.01, 1.0)
                sl_min, sl_max = process_range_value(sl_range, 0.001, 0.1)
                tp_min, tp_max = process_range_value(tp_range, 1.0, 5.0)
                
                # Traitement des plages pour la structure
                min_blocks, max_blocks = process_range_value(blocks_range, 1, 3)
                min_cond, max_cond = process_range_value(conditions_range, 1, 3)
                cross_prob_min, cross_prob_max = process_range_value(cross_probability_range, 0, 0.3)
                value_comp_prob_min, value_comp_prob_max = process_range_value(value_comparison_probability_range, 0, 0.4)
                
                # Construction de la configuration flexible
                config = TradingConfig()
                
                # -- SIM CONFIG
                config.sim_config.initial_balance_range = (bal_value, bal_value)  # Même valeur min et max
                config.sim_config.fee = fee_value
                config.sim_config.slippage = slip_value
                config.sim_config.leverage_range = (int(lev_min), int(lev_max))
                
                if active_margin_modes:
                    config.sim_config.margin_modes = [MarginMode(int(x)) for x in active_margin_modes]
                    
                if active_trading_modes:
                    config.sim_config.trading_modes = [TradingMode(int(x)) for x in active_trading_modes]
                
                # -- RISK
                if active_risk_modes:
                    config.risk_config.available_modes = [RiskMode(x) for x in active_risk_modes]
                    
                config.risk_config.position_size_range = (pos_min/100, pos_max/100)  # Convertir en décimal
                config.risk_config.sl_range = (sl_min/100, sl_max/100)  # Convertir en décimal
                config.risk_config.tp_multiplier_range = (tp_min, tp_max)
                
                # -- STRATEGY STRUCTURE
                config.strategy_structure.max_blocks = int(max_blocks)
                config.strategy_structure.min_blocks = int(min_blocks)
                config.strategy_structure.max_conditions_per_block = int(max_cond)
                config.strategy_structure.min_conditions_per_block = int(min_cond)
                config.strategy_structure.cross_signals_probability = float(cross_prob_max)
                config.strategy_structure.value_comparison_probability = float(value_comp_prob_max)
                
                # -- INDICATORS CONFIGURATION
                # On crée un dictionnaire des indicateurs activés avec leurs paramètres
                indicators_dict = {}
                
                for i, ind_id in enumerate(indicators_ids):
                    indicator_name = ind_id["name"]
                    
                    # Vérifier si l'indicateur est activé
                    if indicators_enabled[i]:
                        if ui_logger:
                            ui_logger.info(f"Ajout de l'indicateur {indicator_name} à la configuration")
                        
                        # Récupérer les paramètres de l'indicateur
                        min_period = int(indicator_min_periods[i]) if i < len(indicator_min_periods) else INDICATOR_DEFAULTS[IndicatorType(indicator_name)]["min_period"]
                        max_period = int(indicator_max_periods[i]) if i < len(indicator_max_periods) else INDICATOR_DEFAULTS[IndicatorType(indicator_name)]["max_period"]
                        step = int(indicator_steps[i]) if i < len(indicator_steps) else INDICATOR_DEFAULTS[IndicatorType(indicator_name)]["step"]
                        price_type = indicator_price_types[i] if i < len(indicator_price_types) else INDICATOR_DEFAULTS[IndicatorType(indicator_name)]["price_type"]
                        
                        # Créer la configuration de l'indicateur
                        ind_config = IndicatorConfig(
                            type=IndicatorType(indicator_name),
                            min_period=min_period,
                            max_period=max_period,
                            step=step,
                            price_type=price_type
                        )
                        
                        # Ajouter à notre dictionnaire
                        indicators_dict[indicator_name] = ind_config
                
                # Assigner le dictionnaire à la configuration
                config.available_indicators = indicators_dict
                
                if ui_logger:
                    ui_logger.info(f"Configuration créée avec {len(indicators_dict)} indicateurs")
                
                # Préparation des métadonnées
                metadata = {
                    "name": study_name,
                    "asset": asset,
                    "timeframe": timeframe,
                    "exchange": exchange,
                    "description": description or "Étude de trading",
                    "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "created",
                    "data_file_path": data_file_path,
                    "data_metadata": data_file_metadata
                }
                
                # Création de l'étude
                success = study_manager.create_study(study_name, metadata, config)
                
                if success:
                    if ui_logger:
                        ui_logger.info(f"Étude '{study_name}' créée avec succès avec {len(indicators_dict)} indicateurs")
                    # Notification de succès
                    study_action_data = json.dumps({"type": "create-success", "study_name": study_name})
                    alert_children = dbc.Alert(
                        [
                            html.I(className="bi bi-check-circle me-2"),
                            f"Étude '{study_name}' créée avec succès !"
                        ],
                        color="success",
                        dismissable=True
                    )
                    # Réinitialisation des champs
                    new_study_name = ""
                    new_description = ""
                else:
                    if ui_logger:
                        ui_logger.error(f"Échec de création de l'étude '{study_name}'.")
                    alert_children = dbc.Alert(
                        [
                            html.I(className="bi bi-exclamation-circle me-2"),
                            "Erreur lors de la création de l'étude."
                        ],
                        color="danger",
                        dismissable=True
                    )
                    
            except Exception as e:
                if ui_logger:
                    ui_logger.error(f"Exception lors de la création: {str(e)}")
                    ui_logger.error(traceback.format_exc())
                alert_children = dbc.Alert(
                    [
                        html.I(className="bi bi-exclamation-triangle me-2"),
                        f"Exception lors de la création: {str(e)}"
                    ],
                    color="danger",
                    dismissable=True
                )
                
        return study_action_data, alert_children, new_study_name, new_description

    # Callback pour l'affichage des informations du fichier de données
    @app.callback(
        Output("data-file-info", "children", allow_duplicate=True),
        [Input("selected-data-file-path", "data"),
        Input("selected-data-metadata", "data")],
        prevent_initial_call=True  # Prevent firing on initial load
    )
    def update_data_file_info(file_path, file_metadata):
        """Affiche les informations du fichier de données sélectionné"""
        if not file_path:
            return html.Div([
                html.I(className="bi bi-info-circle me-2 text-warning"),
                "Aucun fichier de données sélectionné"
            ], className="mt-2 text-muted")
                
        try:
            filename = os.path.basename(file_path)
            
            # Essayer de lire le début du fichier pour l'aperçu
            try:
                df = pd.read_csv(file_path, nrows=5)
                rows_count = len(pd.read_csv(file_path))
                columns_count = len(df.columns)
                
                # Determine date range if timestamp column exists
                date_range = ""
                if 'timestamp' in df.columns:
                    df_full = pd.read_csv(file_path)
                    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
                    min_date = df_full['timestamp'].min().strftime("%Y-%m-%d")
                    max_date = df_full['timestamp'].max().strftime("%Y-%m-%d")
                    date_range = f"Période: {min_date} à {max_date}"
            except Exception as e:
                rows_count = "Erreur"
                columns_count = "Erreur"
                date_range = ""
                
            return html.Div([
                html.Div([
                    html.I(className="bi bi-file-earmark-text me-2 text-success"),
                    html.Strong(filename),
                ], className="mb-1"),
                html.Div([
                    html.Span(f"Lignes: {rows_count}", className="me-3"),
                    html.Span(f"Colonnes: {columns_count}")
                ], className="small text-muted"),
                html.Div(date_range, className="small text-info") if date_range else None
            ], className="mt-2")
            
        except Exception as e:
            return html.Div([
                html.I(className="bi bi-exclamation-triangle me-2 text-warning"),
                f"Erreur lors de la lecture du fichier: {str(e)}"
            ], className="mt-2 text-warning")