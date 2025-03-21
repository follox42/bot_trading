"""
Composant d'édition des paramètres d'une étude.
Permet de modifier les configurations d'indicateurs, de gestion du risque
et les paramètres de simulation de manière visuelle.
"""
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import dash
import json
from datetime import datetime

from logger.logger import LoggerType

def create_parameter_editor(study_name, metadata, central_logger=None):
    """
    Crée l'éditeur de paramètres pour une étude.
    
    Args:
        study_name: Nom de l'étude
        metadata: Métadonnées de l'étude (instance de StudyMetadata)
        central_logger: Instance du logger centralisé
    
    Returns:
        Modal d'édition des paramètres
    """
    # Initialiser le logger
    if central_logger:
        ui_logger = central_logger.get_logger("parameter_editor", LoggerType.UI)
        ui_logger.info(f"Initialisation de l'éditeur de paramètres pour l'étude '{study_name}'")
    
    # Récupérer la configuration de trading pour cette étude
    trading_config = None
    try:
        from simulator.study_manager import IntegratedStudyManager
        study_manager = IntegratedStudyManager("studies")
        
        trading_config = study_manager.get_trading_config(study_name)
    except Exception as e:
        if central_logger:
            ui_logger.error(f"Erreur lors de la récupération de la configuration: {str(e)}")
    
    # Si la configuration n'est pas disponible, créer une interface minimale
    if not trading_config:
        return dbc.Modal(
            [
                dbc.ModalHeader(
                    html.H4(f"PARAMÈTRES DE L'ÉTUDE: {study_name}", className="retro-title"),
                    close_button=True
                ),
                dbc.ModalBody(
                    [
                        html.Div("Configuration non disponible. Veuillez réessayer ultérieurement."),
                        
                        # Champs de base
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Nom de l'étude", html_for="param-study-name"),
                                dbc.Input(
                                    id="param-study-name",
                                    type="text",
                                    value=study_name,
                                    disabled=True
                                ),
                            ]),
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Asset", html_for="param-asset"),
                                dbc.Input(
                                    id="param-asset",
                                    type="text",
                                    value=getattr(metadata, "asset", ""),
                                    disabled=True
                                ),
                            ]),
                        ], className="mb-3"),
                    ]
                ),
                dbc.ModalFooter(
                    dbc.Button(
                        "Fermer",
                        id="btn-close-parameter-editor",
                        className="retro-button"
                    )
                ),
            ],
            id="parameter-editor-modal",
            size="xl",
            is_open=False,
            centered=True,
            scrollable=True,
            className="retro-modal"
        )
    
    # Préparer les onglets pour les différentes catégories de paramètres
    tabs = dbc.Tabs(
        [
            dbc.Tab(
                label="Information",
                tab_id="tab-info",
                children=[
                    html.Div([html.I(className="bi bi-info-circle me-2"), "Information"], className="d-flex align-items-center mb-3"),
                    create_info_tab(study_name, metadata)
                ],
                className="pt-3"
            ),
            dbc.Tab(
                label="Indicateurs",
                tab_id="tab-indicators",
                children=[
                    html.Div([html.I(className="bi bi-graph-up me-2"), "Indicateurs"], className="d-flex align-items-center mb-3"),
                    create_indicators_tab(trading_config)
                ],
                className="pt-3"
            ),
            dbc.Tab(
                label="Gestion du risque",
                tab_id="tab-risk",
                children=[
                    html.Div([html.I(className="bi bi-shield me-2"), "Gestion du risque"], className="d-flex align-items-center mb-3"),
                    create_risk_tab(trading_config)
                ],
                className="pt-3"
            ),
            dbc.Tab(
                label="Simulation",
                tab_id="tab-simulation",
                children=[
                    html.Div([html.I(className="bi bi-gear me-2"), "Simulation"], className="d-flex align-items-center mb-3"),
                    create_simulation_tab(trading_config)
                ],
                className="pt-3"
            ),
            dbc.Tab(
                label="Structure",
                tab_id="tab-structure",
                children=[
                    html.Div([html.I(className="bi bi-sliders me-2"), "Structure"], className="d-flex align-items-center mb-3"),
                    create_structure_tab(trading_config)
                ],
                className="pt-3"
            ),
        ],
        id="parameter-tabs",
        active_tab="tab-info"
    )
    
    # Création du modal complet
    return dbc.Modal(
        [
            dbc.ModalHeader(
                html.H4(f"PARAMÈTRES DE L'ÉTUDE: {study_name}", className="retro-title"),
                close_button=True
            ),
            dbc.ModalBody(
                [
                    # Alerte pour afficher les messages
                    html.Div(id="parameter-edit-alert"),
                    
                    # Onglets de paramètres
                    tabs,
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Annuler",
                        id="btn-close-parameter-editor",
                        className="retro-button secondary"
                    ),
                    dbc.Button(
                        "Sauvegarder",
                        id="btn-save-parameters",
                        className="retro-button ms-auto"
                    ),
                ]
            ),
        ],
        id="parameter-editor-modal",
        size="xl",
        is_open=False,
        centered=True,
        scrollable=True,
        className="retro-modal"
    )

def create_info_tab(study_name, metadata):
    """
    Crée l'onglet d'informations générales sur l'étude.
    
    Args:
        study_name: Nom de l'étude
        metadata: Métadonnées de l'étude
        
    Returns:
        Contenu de l'onglet d'informations
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Nom de l'étude", html_for="param-study-name"),
                        dbc.Input(
                            id="param-study-name",
                            type="text",
                            value=study_name,
                            readonly=True  # Le nom de l'étude ne peut pas être modifié
                        ),
                    ]),
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Description", html_for="param-description"),
                        dbc.Textarea(
                            id="param-description",
                            value=getattr(metadata, "description", ""),
                            style={"height": "100px"}
                        ),
                    ]),
                ], className="mb-3"),
            ], width=6),
            
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Asset", html_for="param-asset"),
                        dbc.Input(
                            id="param-asset",
                            type="text",
                            value=getattr(metadata, "asset", ""),
                            readonly=True  # L'asset ne peut pas être modifié
                        ),
                    ]),
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Timeframe", html_for="param-timeframe"),
                        dbc.Input(
                            id="param-timeframe",
                            type="text",
                            value=getattr(metadata, "timeframe", ""),
                            readonly=True  # Le timeframe ne peut pas être modifié
                        ),
                    ]),
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Exchange", html_for="param-exchange"),
                        dbc.Input(
                            id="param-exchange",
                            type="text",
                            value=getattr(metadata, "exchange", ""),
                            readonly=True  # L'exchange ne peut pas être modifié
                        ),
                    ]),
                ], className="mb-3"),
            ], width=6),
        ]),
        
        # Informations supplémentaires
        html.Div([
            html.H5("Informations", className="mb-3"),
            html.P([
                html.Strong("Date de création: "),
                html.Span(getattr(metadata, "creation_date", "Inconnue"))
            ], className="mb-2"),
            html.P([
                html.Strong("Dernière modification: "),
                html.Span(getattr(metadata, "last_modified", "Jamais"))
            ], className="mb-2"),
            html.P([
                html.Strong("Statut: "),
                # Fix: Convert enum to string value
                html.Span(getattr(metadata, "status", "Inconnu").value if hasattr(getattr(metadata, "status", None), "value") else "Inconnu")
            ], className="mb-2"),
            
            html.P("Les paramètres immuables (nom, asset, timeframe, exchange) ne peuvent pas être modifiés après la création. Pour changer ces paramètres, vous devez cloner l'étude.", 
                   className="text-muted small mt-4")
        ], className="mt-4"),
    ])

def create_indicators_tab(trading_config):
    """
    Crée l'onglet des paramètres d'indicateurs.
    
    Args:
        trading_config: Configuration de trading
        
    Returns:
        Contenu de l'onglet des indicateurs
    """
    # Récupérer les indicateurs disponibles
    available_indicators = trading_config.available_indicators
    
    # Créer un formulaire pour chaque indicateur
    indicator_forms = []
    
    for name, config in available_indicators.items():
        indicator_form = dbc.Card([
            dbc.CardHeader(
                html.H5(f"Indicateur {name}", className="mb-0")
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Période minimale"),
                        dbc.Input(
                            id={"type": "indicator-min-period", "name": name},
                            type="number",
                            min=1,
                            max=1000,
                            step=1,
                            value=config.min_period
                        ),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Période maximale"),
                        dbc.Input(
                            id={"type": "indicator-max-period", "name": name},
                            type="number",
                            min=1,
                            max=1000,
                            step=1,
                            value=config.max_period
                        ),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Pas"),
                        dbc.Input(
                            id={"type": "indicator-step", "name": name},
                            type="number",
                            min=1,
                            max=50,
                            step=1,
                            value=config.step
                        ),
                    ], width=4),
                ]),
            ])
        ], className="mb-3")
        
        indicator_forms.append(indicator_form)
    
    # Si aucun indicateur n'est disponible
    if not indicator_forms:
        return html.Div([
            html.H5("Configuration des indicateurs", className="mb-3"),
            html.P("Aucun indicateur disponible dans cette configuration.")
        ])
    
    return html.Div([
        html.H5("Configuration des indicateurs", className="mb-3"),
        html.P("Définissez les plages de périodes pour chaque indicateur technique utilisé dans l'optimisation.", className="mb-4"),
        
        html.Div(indicator_forms),
        
        html.P([
            "Ces paramètres définissent les plages explorées lors de l'optimisation. ",
            "Des plages plus larges permettent plus d'exploration mais augmentent le temps d'optimisation."
        ], className="text-muted small mt-2")
    ])

def create_risk_tab(trading_config):
    """
    Crée l'onglet des paramètres de gestion du risque.
    
    Args:
        trading_config: Configuration de trading
        
    Returns:
        Contenu de l'onglet de gestion du risque
    """
    # Récupérer la configuration de risque
    risk_config = trading_config.risk_config
    
    # Récupérer les plages de paramètres
    position_range = risk_config.position_size_range
    sl_range = risk_config.sl_range
    tp_range = risk_config.tp_multiplier_range
    
    # Récupérer les modes disponibles
    available_modes = [mode.value for mode in risk_config.available_modes]
    
    # Créer des sections pour chaque mode
    mode_configs = []
    
    # Section pour le mode Fixed
    if 'fixed' in available_modes:
        fixed_config = risk_config.mode_configs.get(risk_config.available_modes[0])  # RiskMode.FIXED
        if fixed_config:
            fixed_form = dbc.Card([
                dbc.CardHeader(
                    html.H5("Mode Fixed", className="mb-0")
                ),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Position Size (%)"),
                            dbc.InputGroup([
                                dbc.Input(
                                    id="fixed-position-min",
                                    type="number",
                                    min=0.1,
                                    max=100,
                                    step=0.1,
                                    value=fixed_config.fixed_position_range[0] * 100
                                ),
                                dbc.InputGroupText("à"),
                                dbc.Input(
                                    id="fixed-position-max",
                                    type="number",
                                    min=0.1,
                                    max=100,
                                    step=0.1,
                                    value=fixed_config.fixed_position_range[1] * 100
                                ),
                                dbc.InputGroupText("%")
                            ]),
                        ], width=12, className="mb-3"),
                        
                        dbc.Col([
                            dbc.Label("Stop Loss (%)"),
                            dbc.InputGroup([
                                dbc.Input(
                                    id="fixed-sl-min",
                                    type="number",
                                    min=0.1,
                                    max=20,
                                    step=0.1,
                                    value=fixed_config.fixed_sl_range[0] * 100
                                ),
                                dbc.InputGroupText("à"),
                                dbc.Input(
                                    id="fixed-sl-max",
                                    type="number",
                                    min=0.1,
                                    max=20,
                                    step=0.1,
                                    value=fixed_config.fixed_sl_range[1] * 100
                                ),
                                dbc.InputGroupText("%")
                            ]),
                        ], width=6),
                        
                        dbc.Col([
                            dbc.Label("Take Profit (×SL)"),
                            dbc.InputGroup([
                                dbc.Input(
                                    id="fixed-tp-min",
                                    type="number",
                                    min=1,
                                    max=10,
                                    step=0.1,
                                    value=fixed_config.fixed_tp_range[0] / fixed_config.fixed_sl_range[0]
                                ),
                                dbc.InputGroupText("à"),
                                dbc.Input(
                                    id="fixed-tp-max",
                                    type="number",
                                    min=1,
                                    max=10,
                                    step=0.1,
                                    value=fixed_config.fixed_tp_range[1] / fixed_config.fixed_sl_range[0]
                                ),
                                dbc.InputGroupText("×")
                            ]),
                        ], width=6),
                    ]),
                ])
            ], className="mb-3")
            
            mode_configs.append(fixed_form)
    
    # Section pour le mode ATR
    if 'atr_based' in available_modes:
        atr_config = risk_config.mode_configs.get(risk_config.available_modes[1])  # RiskMode.ATR_BASED
        if atr_config:
            atr_form = dbc.Card([
                dbc.CardHeader(
                    html.H5("Mode ATR", className="mb-0")
                ),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Période ATR"),
                            dbc.InputGroup([
                                dbc.Input(
                                    id="atr-period-min",
                                    type="number",
                                    min=1,
                                    max=100,
                                    step=1,
                                    value=atr_config.atr_period_range[0]
                                ),
                                dbc.InputGroupText("à"),
                                dbc.Input(
                                    id="atr-period-max",
                                    type="number",
                                    min=1,
                                    max=100,
                                    step=1,
                                    value=atr_config.atr_period_range[1]
                                ),
                            ]),
                        ], width=6),
                        
                        dbc.Col([
                            dbc.Label("Multiplicateur ATR"),
                            dbc.InputGroup([
                                dbc.Input(
                                    id="atr-mult-min",
                                    type="number",
                                    min=0.1,
                                    max=10,
                                    step=0.1,
                                    value=atr_config.atr_multiplier_range[0]
                                ),
                                dbc.InputGroupText("à"),
                                dbc.Input(
                                    id="atr-mult-max",
                                    type="number",
                                    min=0.1,
                                    max=10,
                                    step=0.1,
                                    value=atr_config.atr_multiplier_range[1]
                                ),
                                dbc.InputGroupText("×")
                            ]),
                        ], width=6),
                    ]),
                ])
            ], className="mb-3")
            
            mode_configs.append(atr_form)
    
    # Section pour le mode Volatility
    if 'volatility_based' in available_modes:
        vol_config = risk_config.mode_configs.get(risk_config.available_modes[2])  # RiskMode.VOLATILITY_BASED
        if vol_config:
            vol_form = dbc.Card([
                dbc.CardHeader(
                    html.H5("Mode Volatility", className="mb-0")
                ),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Période de volatilité"),
                            dbc.InputGroup([
                                dbc.Input(
                                    id="vol-period-min",
                                    type="number",
                                    min=1,
                                    max=100,
                                    step=1,
                                    value=vol_config.vol_period_range[0]
                                ),
                                dbc.InputGroupText("à"),
                                dbc.Input(
                                    id="vol-period-max",
                                    type="number",
                                    min=1,
                                    max=100,
                                    step=1,
                                    value=vol_config.vol_period_range[1]
                                ),
                            ]),
                        ], width=6),
                        
                        dbc.Col([
                            dbc.Label("Multiplicateur de volatilité"),
                            dbc.InputGroup([
                                dbc.Input(
                                    id="vol-mult-min",
                                    type="number",
                                    min=0.1,
                                    max=10,
                                    step=0.1,
                                    value=vol_config.vol_multiplier_range[0]
                                ),
                                dbc.InputGroupText("à"),
                                dbc.Input(
                                    id="vol-mult-max",
                                    type="number",
                                    min=0.1,
                                    max=10,
                                    step=0.1,
                                    value=vol_config.vol_multiplier_range[1]
                                ),
                                dbc.InputGroupText("×")
                            ]),
                        ], width=6),
                    ]),
                ])
            ], className="mb-3")
            
            mode_configs.append(vol_form)
    
    # Section des paramètres généraux
    general_form = dbc.Card([
        dbc.CardHeader(
            html.H5("Paramètres généraux", className="mb-0")
        ),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Position Size (%)"),
                    dbc.InputGroup([
                        dbc.Input(
                            id="position-size-min",
                            type="number",
                            min=0.1,
                            max=100,
                            step=0.1,
                            value=position_range[0] * 100
                        ),
                        dbc.InputGroupText("à"),
                        dbc.Input(
                            id="position-size-max",
                            type="number",
                            min=0.1,
                            max=100,
                            step=0.1,
                            value=position_range[1] * 100
                        ),
                        dbc.InputGroupText("%")
                    ]),
                ], width=12, className="mb-3"),
                
                dbc.Col([
                    dbc.Label("Stop Loss (%)"),
                    dbc.InputGroup([
                        dbc.Input(
                            id="sl-min",
                            type="number",
                            min=0.1,
                            max=20,
                            step=0.1,
                            value=sl_range[0] * 100
                        ),
                        dbc.InputGroupText("à"),
                        dbc.Input(
                            id="sl-max",
                            type="number",
                            min=0.1,
                            max=20,
                            step=0.1,
                            value=sl_range[1] * 100
                        ),
                        dbc.InputGroupText("%")
                    ]),
                ], width=12, className="mb-3"),
                
                dbc.Col([
                    dbc.Label("Take Profit Multiplier"),
                    dbc.InputGroup([
                        dbc.Input(
                            id="tp-mult-min",
                            type="number",
                            min=1,
                            max=10,
                            step=0.1,
                            value=tp_range[0]
                        ),
                        dbc.InputGroupText("à"),
                        dbc.Input(
                            id="tp-mult-max",
                            type="number",
                            min=1,
                            max=10,
                            step=0.1,
                            value=tp_range[1]
                        ),
                        dbc.InputGroupText("×")
                    ]),
                ], width=12),
            ]),
        ])
    ], className="mb-3")
    
    return html.Div([
        html.H5("Gestion du risque", className="mb-3"),
        html.P("Configurez les paramètres de gestion du risque pour cette étude.", className="mb-4"),
        
        general_form,
        
        html.H5("Modes de risque spécifiques", className="mb-3"),
        
        html.Div(mode_configs if mode_configs else 
                 html.P("Aucun mode de risque spécifique configuré.")),
        
        html.P([
            "Ces paramètres définissent comment le système gère le risque et calcule les tailles de position, ",
            "les niveaux de stop loss et de take profit."
        ], className="text-muted small mt-2")
    ])

def create_simulation_tab(trading_config):
    """
    Crée l'onglet des paramètres de simulation.
    
    Args:
        trading_config: Configuration de trading
        
    Returns:
        Contenu de l'onglet de simulation
    """
    # Récupérer la configuration de simulation
    sim_config = trading_config.sim_config
    
    # Récupérer les plages
    balance_range = sim_config.initial_balance_range
    leverage_range = sim_config.leverage_range
    
    # Créer le formulaire
    return html.Div([
        html.H5("Paramètres de simulation", className="mb-3"),
        html.P("Configurez les paramètres utilisés pour simuler le trading lors des backtests et optimisations.", className="mb-4"),
        
        dbc.Card([
            dbc.CardHeader(
                html.H5("Paramètres de compte", className="mb-0")
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Balance initiale"),
                        dbc.InputGroup([
                            dbc.Input(
                                id="balance-min",
                                type="number",
                                min=100,
                                max=1000000,
                                step=100,
                                value=balance_range[0]
                            ),
                            dbc.InputGroupText("à"),
                            dbc.Input(
                                id="balance-max",
                                type="number",
                                min=100,
                                max=1000000,
                                step=100,
                                value=balance_range[1]
                            ),
                            dbc.InputGroupText("$")
                        ]),
                    ], width=6),
                    
                    dbc.Col([
                        dbc.Label("Levier"),
                        dbc.InputGroup([
                            dbc.Input(
                                id="leverage-min",
                                type="number",
                                min=1,
                                max=125,
                                step=1,
                                value=leverage_range[0]
                            ),
                            dbc.InputGroupText("à"),
                            dbc.Input(
                                id="leverage-max",
                                type="number",
                                min=1,
                                max=125,
                                step=1,
                                value=leverage_range[1]
                            ),
                            dbc.InputGroupText("×")
                        ]),
                    ], width=6),
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Frais de trading (%)"),
                        dbc.Input(
                            id="trading-fee",
                            type="number",
                            min=0,
                            max=1,
                            step=0.001,
                            value=sim_config.fee * 100
                        ),
                    ], width=6),
                    
                    dbc.Col([
                        dbc.Label("Slippage (%)"),
                        dbc.Input(
                            id="slippage",
                            type="number",
                            min=0,
                            max=1,
                            step=0.001,
                            value=sim_config.slippage * 100
                        ),
                    ], width=6),
                ]),
            ])
        ], className="mb-3"),
        
        dbc.Card([
            dbc.CardHeader(
                html.H5("Types de trading", className="mb-0")
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Mode de marge"),
                        dbc.Checklist(
                            id="margin-modes",
                            options=[
                                {"label": "Isolated", "value": 0},
                                {"label": "Cross", "value": 1}
                            ],
                            value=[mode.value for mode in sim_config.margin_modes],
                            inline=True
                        ),
                    ], width=6),
                    
                    dbc.Col([
                        dbc.Label("Mode de trading"),
                        dbc.Checklist(
                            id="trading-modes",
                            options=[
                                {"label": "One-way", "value": 0},
                                {"label": "Hedge", "value": 1}
                            ],
                            value=[mode.value for mode in sim_config.trading_modes],
                            inline=True
                        ),
                    ], width=6),
                ]),
            ])
        ], className="mb-3"),
        
        dbc.Card([
            dbc.CardHeader(
                html.H5("Limites de trading", className="mb-0")
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Taille minimale de trade"),
                        dbc.InputGroup([
                            dbc.Input(
                                id="min-trade-size",
                                type="number",
                                min=0.001,
                                max=10,
                                step=0.001,
                                value=sim_config.min_trade_size
                            ),
                            dbc.InputGroupText("BTC")
                        ]),
                    ], width=6),
                    
                    dbc.Col([
                        dbc.Label("Taille maximale de trade"),
                        dbc.InputGroup([
                            dbc.Input(
                                id="max-trade-size",
                                type="number",
                                min=0.1,
                                max=1000000,
                                step=0.1,
                                value=sim_config.max_trade_size
                            ),
                            dbc.InputGroupText("USDT")
                        ]),
                    ], width=6),
                ]),
            ])
        ], className="mb-3"),
        
        html.P([
            "Ces paramètres définissent comment le système simule le trading lors des backtests et optimisations. ",
            "Ils affectent les résultats des backtests mais pas les décisions de trading."
        ], className="text-muted small mt-2")
    ])

def create_structure_tab(trading_config):
    """
    Crée l'onglet des paramètres de structure de stratégie.
    
    Args:
        trading_config: Configuration de trading
        
    Returns:
        Contenu de l'onglet de structure
    """
    # Récupérer la configuration de structure
    structure_config = trading_config.strategy_structure
    
    # Récupérer les plages
    rsi_range = structure_config.rsi_value_range
    price_range = structure_config.price_value_range
    general_range = structure_config.general_value_range
    
    return html.Div([
        html.H5("Structure des stratégies", className="mb-3"),
        html.P("Configurez les paramètres de structure utilisés lors de la génération des stratégies.", className="mb-4"),
        
        dbc.Card([
            dbc.CardHeader(
                html.H5("Complexité des stratégies", className="mb-0")
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Nombre de blocs"),
                        dbc.InputGroup([
                            dbc.Input(
                                id="min-blocks",
                                type="number",
                                min=1,
                                max=10,
                                step=1,
                                value=structure_config.min_blocks
                            ),
                            dbc.InputGroupText("à"),
                            dbc.Input(
                                id="max-blocks",
                                type="number",
                                min=1,
                                max=10,
                                step=1,
                                value=structure_config.max_blocks
                            ),
                        ]),
                    ], width=6),
                    
                    dbc.Col([
                        dbc.Label("Conditions par bloc"),
                        dbc.InputGroup([
                            dbc.Input(
                                id="min-conditions",
                                type="number",
                                min=1,
                                max=10,
                                step=1,
                                value=structure_config.min_conditions_per_block
                            ),
                            dbc.InputGroupText("à"),
                            dbc.Input(
                                id="max-conditions",
                                type="number",
                                min=1,
                                max=10,
                                step=1,
                                value=structure_config.max_conditions_per_block
                            ),
                        ]),
                    ], width=6),
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Probabilité de croisement de signaux"),
                        dbc.Input(
                            id="cross-probability",
                            type="number",
                            min=0,
                            max=1,
                            step=0.01,
                            value=structure_config.cross_signals_probability
                        ),
                    ], width=6),
                    
                    dbc.Col([
                        dbc.Label("Probabilité de comparaison de valeur"),
                        dbc.Input(
                            id="value-comparison-probability",
                            type="number",
                            min=0,
                            max=1,
                            step=0.01,
                            value=structure_config.value_comparison_probability
                        ),
                    ], width=6),
                ]),
            ])
        ], className="mb-3"),
        
        dbc.Card([
            dbc.CardHeader(
                html.H5("Plages de valeurs", className="mb-0")
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Valeurs RSI"),
                        dbc.InputGroup([
                            dbc.Input(
                                id="rsi-min",
                                type="number",
                                min=0,
                                max=100,
                                step=1,
                                value=rsi_range[0]
                            ),
                            dbc.InputGroupText("à"),
                            dbc.Input(
                                id="rsi-max",
                                type="number",
                                min=0,
                                max=100,
                                step=1,
                                value=rsi_range[1]
                            ),
                        ]),
                    ], width=12, className="mb-3"),
                    
                    dbc.Col([
                        dbc.Label("Valeurs de prix (multiplicateurs)"),
                        dbc.InputGroup([
                            dbc.Input(
                                id="price-min",
                                type="number",
                                min=0,
                                max=10000,
                                step=1,
                                value=price_range[0]
                            ),
                            dbc.InputGroupText("à"),
                            dbc.Input(
                                id="price-max",
                                type="number",
                                min=0,
                                max=10000,
                                step=1,
                                value=price_range[1]
                            ),
                        ]),
                    ], width=12, className="mb-3"),
                    
                    dbc.Col([
                        dbc.Label("Valeurs générales"),
                        dbc.InputGroup([
                            dbc.Input(
                                id="general-min",
                                type="number",
                                min=-1000,
                                max=1000,
                                step=1,
                                value=general_range[0]
                            ),
                            dbc.InputGroupText("à"),
                            dbc.Input(
                                id="general-max",
                                type="number",
                                min=-1000,
                                max=1000,
                                step=1,
                                value=general_range[1]
                            ),
                        ]),
                    ], width=12),
                ]),
            ])
        ], className="mb-3"),
        
        html.P([
            "Ces paramètres définissent comment le système génère et structure les stratégies lors de l'optimisation. ",
            "Ils affectent la complexité et la diversité des stratégies générées."
        ], className="text-muted small mt-2")
    ])

def register_parameter_editor_callbacks(app, central_logger=None):
    """
    Enregistre les callbacks pour l'éditeur de paramètres
    
    Args:
        app: L'instance de l'application Dash
        central_logger: Instance du logger centralisé
    """
    # Initialiser le logger
    if central_logger:
        ui_logger = central_logger.get_logger("parameter_editor_callbacks", LoggerType.UI)
    
    # Callback pour la sauvegarde des paramètres
    @app.callback(
        [Output("parameter-edit-alert", "children"),
         Output("study-action", "data")],
        [Input("btn-save-parameters", "n_clicks")],
        [State("parameter-editor-modal", "id"),
         State("param-description", "value"),
         # ... Ajouter ici les states pour tous les autres champs ...
         # Indicateurs
         State({"type": "indicator-min-period", "name": dash.ALL}, "value"),
         State({"type": "indicator-max-period", "name": dash.ALL}, "value"),
         State({"type": "indicator-step", "name": dash.ALL}, "value"),
         State({"type": "indicator-min-period", "name": dash.ALL}, "id"),
         # Risque
         State("position-size-min", "value"),
         State("position-size-max", "value"),
         State("sl-min", "value"),
         State("sl-max", "value"),
         State("tp-mult-min", "value"),
         State("tp-mult-max", "value"),
         # Fixed
         State("fixed-position-min", "value"),
         State("fixed-position-max", "value"),
         State("fixed-sl-min", "value"),
         State("fixed-sl-max", "value"),
         State("fixed-tp-min", "value"),
         State("fixed-tp-max", "value"),
         # ATR
         State("atr-period-min", "value"),
         State("atr-period-max", "value"),
         State("atr-mult-min", "value"),
         State("atr-mult-max", "value"),
         # Vol
         State("vol-period-min", "value"),
         State("vol-period-max", "value"),
         State("vol-mult-min", "value"),
         State("vol-mult-max", "value"),
         # Simulation
         State("balance-min", "value"),
         State("balance-max", "value"),
         State("leverage-min", "value"),
         State("leverage-max", "value"),
         State("trading-fee", "value"),
         State("slippage", "value"),
         State("margin-modes", "value"),
         State("trading-modes", "value"),
         State("min-trade-size", "value"),
         State("max-trade-size", "value"),
         # Structure
         State("min-blocks", "value"),
         State("max-blocks", "value"),
         State("min-conditions", "value"),
         State("max-conditions", "value"),
         State("cross-probability", "value"),
         State("value-comparison-probability", "value"),
         State("rsi-min", "value"),
         State("rsi-max", "value"),
         State("price-min", "value"),
         State("price-max", "value"),
         State("general-min", "value"),
         State("general-max", "value"),
         # Nom de l'étude
         State("param-study-name", "value")],
        prevent_initial_call=True
    )
    def save_parameters(n_clicks, modal_id, description, 
                       ind_min_periods, ind_max_periods, ind_steps, ind_ids,
                       pos_min, pos_max, sl_min, sl_max, tp_min, tp_max,
                       fixed_pos_min, fixed_pos_max, fixed_sl_min, fixed_sl_max, fixed_tp_min, fixed_tp_max,
                       atr_period_min, atr_period_max, atr_mult_min, atr_mult_max,
                       vol_period_min, vol_period_max, vol_mult_min, vol_mult_max,
                       balance_min, balance_max, leverage_min, leverage_max,
                       trading_fee, slippage, margin_modes, trading_modes,
                       min_trade_size, max_trade_size,
                       min_blocks, max_blocks, min_conditions, max_conditions,
                       cross_probability, value_comparison_probability,
                       rsi_min, rsi_max, price_min, price_max, general_min, general_max,
                       study_name):
        """Sauvegarde les paramètres de l'étude"""
        if not n_clicks:
            return dash.no_update, dash.no_update
        
        if not study_name:
            return dbc.Alert(
                "Erreur: Nom d'étude manquant",
                color="danger",
                dismissable=True
            ), dash.no_update
        
        try:
            from simulator.study_manager import IntegratedStudyManager
            study_manager = IntegratedStudyManager("studies")
            
            # Vérifier si l'étude existe
            if not study_manager.study_exists(study_name):
                return dbc.Alert(
                    f"L'étude '{study_name}' n'existe pas",
                    color="danger",
                    dismissable=True
                ), dash.no_update
            
            # Récupérer la configuration de trading
            trading_config = study_manager.get_trading_config(study_name)
            if not trading_config:
                return dbc.Alert(
                    "Impossible de récupérer la configuration de trading",
                    color="danger",
                    dismissable=True
                ), dash.no_update
            
            # Mettre à jour les métadonnées
            metadata = study_manager.get_study_metadata(study_name)
            if metadata:
                # Utiliser setattr pour mettre à jour la description
                setattr(metadata, "description", description)
                setattr(metadata, "last_modified", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                # Transformer le metadata en dictionnaire avant de le passer à update_study_metadata
                metadata_dict = {"description": description, "last_modified": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                study_manager.update_study_metadata(study_name, metadata_dict)
            
            # Mise à jour des indicateurs
            if ind_min_periods and ind_max_periods and ind_steps and ind_ids:
                for i, ind_id in enumerate(ind_ids):
                    ind_name = ind_id["name"]
                    if ind_name in trading_config.available_indicators:
                        trading_config.available_indicators[ind_name].min_period = ind_min_periods[i]
                        trading_config.available_indicators[ind_name].max_period = ind_max_periods[i]
                        trading_config.available_indicators[ind_name].step = ind_steps[i]
            
            # Mise à jour des paramètres de risque généraux
            if pos_min is not None and pos_max is not None:
                trading_config.risk_config.position_size_range = (pos_min / 100, pos_max / 100)
            
            if sl_min is not None and sl_max is not None:
                trading_config.risk_config.sl_range = (sl_min / 100, sl_max / 100)
            
            if tp_min is not None and tp_max is not None:
                trading_config.risk_config.tp_multiplier_range = (tp_min, tp_max)
            
            # Mise à jour des paramètres de mode Fixed
            if "fixed" in [mode.value for mode in trading_config.risk_config.available_modes]:
                fixed_mode = [mode for mode in trading_config.risk_config.available_modes if mode.value == "fixed"][0]
                if fixed_pos_min is not None and fixed_pos_max is not None:
                    trading_config.risk_config.mode_configs[fixed_mode].fixed_position_range = (fixed_pos_min / 100, fixed_pos_max / 100)
                
                if fixed_sl_min is not None and fixed_sl_max is not None:
                    trading_config.risk_config.mode_configs[fixed_mode].fixed_sl_range = (fixed_sl_min / 100, fixed_sl_max / 100)
                
                if fixed_tp_min is not None and fixed_tp_max is not None and fixed_sl_min is not None:
                    # Convertir les multiplicateurs en valeurs absolues
                    tp_min_abs = (fixed_tp_min * fixed_sl_min / 100)
                    tp_max_abs = (fixed_tp_max * fixed_sl_max / 100)
                    trading_config.risk_config.mode_configs[fixed_mode].fixed_tp_range = (tp_min_abs, tp_max_abs)
            
            # Mise à jour des paramètres de mode ATR
            if "atr_based" in [mode.value for mode in trading_config.risk_config.available_modes]:
                atr_mode = [mode for mode in trading_config.risk_config.available_modes if mode.value == "atr_based"][0]
                if atr_period_min is not None and atr_period_max is not None:
                    trading_config.risk_config.mode_configs[atr_mode].atr_period_range = (atr_period_min, atr_period_max)
                
                if atr_mult_min is not None and atr_mult_max is not None:
                    trading_config.risk_config.mode_configs[atr_mode].atr_multiplier_range = (atr_mult_min, atr_mult_max)
            
            # Mise à jour des paramètres de mode Volatility
            if "volatility_based" in [mode.value for mode in trading_config.risk_config.available_modes]:
                vol_mode = [mode for mode in trading_config.risk_config.available_modes if mode.value == "volatility_based"][0]
                if vol_period_min is not None and vol_period_max is not None:
                    trading_config.risk_config.mode_configs[vol_mode].vol_period_range = (vol_period_min, vol_period_max)
                
                if vol_mult_min is not None and vol_mult_max is not None:
                    trading_config.risk_config.mode_configs[vol_mode].vol_multiplier_range = (vol_mult_min, vol_mult_max)
            
            # Mise à jour des paramètres de simulation
            if balance_min is not None and balance_max is not None:
                trading_config.sim_config.initial_balance_range = (balance_min, balance_max)
            
            if leverage_min is not None and leverage_max is not None:
                trading_config.sim_config.leverage_range = (leverage_min, leverage_max)
            
            if trading_fee is not None:
                trading_config.sim_config.fee = trading_fee / 100
            
            if slippage is not None:
                trading_config.sim_config.slippage = slippage / 100
            
            if margin_modes:
                trading_config.sim_config.margin_modes = []
                for mode_value in margin_modes:
                    trading_config.sim_config.margin_modes.append(MarginMode(mode_value))
            
            if trading_modes:
                trading_config.sim_config.trading_modes = []
                for mode_value in trading_modes:
                    trading_config.sim_config.trading_modes.append(TradingMode(mode_value))
            
            if min_trade_size is not None:
                trading_config.sim_config.min_trade_size = min_trade_size
            
            if max_trade_size is not None:
                trading_config.sim_config.max_trade_size = max_trade_size
            
            # Mise à jour des paramètres de structure
            if min_blocks is not None and max_blocks is not None:
                trading_config.strategy_structure.min_blocks = min_blocks
                trading_config.strategy_structure.max_blocks = max_blocks
            
            if min_conditions is not None and max_conditions is not None:
                trading_config.strategy_structure.min_conditions_per_block = min_conditions
                trading_config.strategy_structure.max_conditions_per_block = max_conditions
            
            if cross_probability is not None:
                trading_config.strategy_structure.cross_signals_probability = cross_probability
            
            if value_comparison_probability is not None:
                trading_config.strategy_structure.value_comparison_probability = value_comparison_probability
            
            if rsi_min is not None and rsi_max is not None:
                trading_config.strategy_structure.rsi_value_range = (rsi_min, rsi_max)
            
            if price_min is not None and price_max is not None:
                trading_config.strategy_structure.price_value_range = (price_min, price_max)
            
            if general_min is not None and general_max is not None:
                trading_config.strategy_structure.general_value_range = (general_min, general_max)
            
            # Sauvegarder la configuration mise à jour
            study_manager.update_trading_config(study_name, trading_config)
            
            if central_logger:
                ui_logger.info(f"Paramètres de l'étude '{study_name}' sauvegardés avec succès")
            
            # Notification de succès et demande de rafraîchissement
            return dbc.Alert(
                "Paramètres sauvegardés avec succès",
                color="success",
                dismissable=True
            ), json.dumps({
                "type": "refresh-detail",
                "study_id": study_name
            })
            
        except Exception as e:
            if central_logger:
                ui_logger.error(f"Erreur lors de la sauvegarde des paramètres: {str(e)}")
            
            # Notification d'erreur
            return dbc.Alert(
                f"Erreur lors de la sauvegarde des paramètres: {str(e)}",
                color="danger",
                dismissable=True
            ), dash.no_update