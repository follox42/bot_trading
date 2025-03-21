"""
Onglet Simulation du créateur d'étude avancé avec valeurs fixes pour certains paramètres.
"""
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import dash

from simulator.study_config_definitions import SIMULATION_PARAMS
from ui.components.retro_ui import create_retro_range_slider, create_retro_toggle_button, create_collapsible_card

def create_retro_number_input(id_prefix, label, min_val, max_val, step, value, unit=""):
    """
    Crée un input numérique simple avec style rétro.
    
    Args:
        id_prefix: Préfixe pour les IDs des composants
        label: Label de l'input
        min_val: Valeur minimale autorisée
        max_val: Valeur maximale autorisée
        step: Pas de l'input
        value: Valeur actuelle
        unit: Unité à afficher (%, $, etc.)
    
    Returns:
        Composant Dash pour l'input numérique
    """
    # Formater la valeur pour l'affichage
    if isinstance(value, int):
        display_value = f"{value}"
    else:
        display_value = f"{value:.2f}" if value % 1 else f"{int(value)}"
    
    return html.Div(
        className="retro-input-container mb-4",
        children=[
            # En-tête avec label et valeur actuelle
            html.Div(
                className="retro-input-header",
                children=[
                    html.Div(label, className="retro-range-label"),
                ]
            ),
            
            # L'input numérique avec style rétro
            html.Div(
                className="retro-input-wrapper",
                children=[
                    dbc.Input(
                        id={"type": "fixed-input", "id": id_prefix},
                        type="number",
                        min=min_val,
                        max=max_val,
                        step=step,
                        value=value,
                        className="retro-number-input"
                    ),
                    html.Span(unit, className="retro-input-unit") if unit else None
                ]
            )
        ]
    )

def create_simulation_tab():
    """
    Crée l'onglet Simulation avec la nouvelle interface retro.
    
    Returns:
        Contenu de l'onglet simulation
    """
    return html.Div([
        html.P("Configurez les paramètres de simulation et de backtest", className="text-muted mb-4"),
        
        # Paramètres de simulation
        create_simulation_params_section()
    ])

def create_simulation_params_section():
    """
    Crée la section des paramètres de simulation.
    
    Returns:
        Composant pour la section des paramètres de simulation
    """
    # Création d'une grille avec les inputs
    fixed_params_content = html.Div(
        className="retro-grid",
        children=[
            # Balance initiale - valeur fixe
            create_retro_number_input(
                id_prefix="initial-balance",
                label="Balance Initiale",
                min_val=SIMULATION_PARAMS["balance"]["min"],
                max_val=SIMULATION_PARAMS["balance"]["max"],
                step=SIMULATION_PARAMS["balance"]["step"],
                value=10000,
                unit=SIMULATION_PARAMS["balance"]["unit"]
            ),
            
            # Frais de trading - valeur fixe
            create_retro_number_input(
                id_prefix="fee",
                label="Frais de Trading",
                min_val=0,
                max_val=0.5,
                step=0.01,
                value=SIMULATION_PARAMS["fee"]["value"],
                unit=SIMULATION_PARAMS["fee"]["unit"]
            ),
            
            # Slippage - valeur fixe
            create_retro_number_input(
                id_prefix="slippage",
                label="Slippage",
                min_val=0,
                max_val=0.5,
                step=0.01,
                value=SIMULATION_PARAMS["slippage"]["value"],
                unit=SIMULATION_PARAMS["slippage"]["unit"]
            ),
        ]
    )
    
    # Paramètres à tester (plages)
    test_params_content = html.Div(
        className="mt-4",
        children=[
            html.Div("Paramètres à Optimiser", className="text-cyan-300 font-bold mb-3"),
            html.Div(
                className="retro-grid",
                children=[
                    # Plage de levier - plage à tester
                    create_retro_range_slider(
                        id_prefix="leverage",
                        label="Levier",
                        min_val=SIMULATION_PARAMS["leverage"]["min"],
                        max_val=SIMULATION_PARAMS["leverage"]["max"],
                        step=SIMULATION_PARAMS["leverage"]["step"],
                        current_min=1,
                        current_max=10,
                        unit=SIMULATION_PARAMS["leverage"]["unit"]
                    ),
                ]
            )
        ]
    )
    
    # Modes de trading et de marge
    trading_modes = html.Div([
        html.Div("Modes de Trading", className="mt-4 mb-2 text-cyan-300 font-bold"),
        html.Div(
            className="retro-toggle-group",
            children=[
                create_retro_toggle_button(
                    id_prefix="trading-mode",
                    label=mode["label"],
                    value=mode["value"],
                    index=i,
                    is_active=True
                ) for i, mode in enumerate(SIMULATION_PARAMS["trading_modes"])
            ]
        ),
        
        html.Div("Modes de Marge", className="mt-4 mb-2 text-cyan-300 font-bold"),
        html.Div(
            className="retro-toggle-group",
            children=[
                create_retro_toggle_button(
                    id_prefix="margin-mode",
                    label=mode["label"],
                    value=mode["value"],
                    index=i,
                    is_active=True
                ) for i, mode in enumerate(SIMULATION_PARAMS["margin_modes"])
            ]
        )
    ])
    
    # Regrouper le contenu
    all_content = html.Div([
        html.Div("Paramètres Fixes", className="text-cyan-300 font-bold mb-3"),
        fixed_params_content,
        test_params_content,
        trading_modes
    ])
    
    return create_collapsible_card(
        title="Paramètres de Simulation",
        content=all_content,
        id_prefix="sim-params-card",
        is_open=True
    )

def register_simulation_callbacks(app):
    """
    Enregistre les callbacks spécifiques à l'onglet Simulation
    
    Args:
        app: L'instance de l'application Dash
    """
    # Callback pour valider les entrées numériques
    @app.callback(
        Output({"type": "fixed-input", "id": dash.MATCH}, "className"),
        [Input({"type": "fixed-input", "id": dash.MATCH}, "value")],
        [State({"type": "fixed-input", "id": dash.MATCH}, "id")]
    )
    def validate_fixed_input(value, input_id):
        """Valide les entrées numériques et applique un style en conséquence"""
        # Si la valeur est nulle ou vide, appliquer un style d'erreur
        if value is None or value == "":
            return "retro-number-input is-invalid"
        
        id_name = input_id.get('id', '')
        
        # Vérifications spécifiques selon le type d'input
        if id_name == "initial-balance":
            # La balance initiale doit être positive
            if value <= 0:
                return "retro-number-input is-invalid"
        elif id_name in ["fee", "slippage"]:
            # Les frais et le slippage doivent être entre 0 et 100%
            if value < 0 or value > 100:
                return "retro-number-input is-invalid"
        
        # Style normal si la valeur est valide
        return "retro-number-input"