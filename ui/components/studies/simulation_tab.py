"""
Onglet Simulation du créateur d'étude avancé.
"""
from dash import html, dcc
import dash_bootstrap_components as dbc

from simulator.study_config_definitions import SIMULATION_PARAMS
from ui.components.retro_ui import create_retro_range_slider, create_retro_toggle_button, create_collapsible_card

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
    # Création d'une grille avec les sliders
    grid_content = html.Div(
        className="retro-grid",
        children=[
            # Plage de balance initiale
            create_retro_range_slider(
                id_prefix="initial-balance",
                label="Balance Initiale",
                min_val=SIMULATION_PARAMS["balance"]["min"],
                max_val=SIMULATION_PARAMS["balance"]["max"],
                step=SIMULATION_PARAMS["balance"]["step"],
                current_min=1000,
                current_max=10000,
                unit=SIMULATION_PARAMS["balance"]["unit"]
            ),
            
            # Plage de levier
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
            
            # Frais de trading
            create_retro_range_slider(
                id_prefix="fee",
                label="Frais de Trading",
                min_val=0,
                max_val=0.5,
                step=0.01,
                current_min=0,
                current_max=SIMULATION_PARAMS["fee"]["value"],
                unit=SIMULATION_PARAMS["fee"]["unit"]
            ),
            
            # Slippage
            create_retro_range_slider(
                id_prefix="slippage",
                label="Slippage",
                min_val=0,
                max_val=0.5,
                step=0.01,
                current_min=0,
                current_max=SIMULATION_PARAMS["slippage"]["value"],
                unit=SIMULATION_PARAMS["slippage"]["unit"]
            ),
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
                    is_active=True
                ) for mode in SIMULATION_PARAMS["trading_modes"]
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
                    is_active=True
                ) for mode in SIMULATION_PARAMS["margin_modes"]
            ]
        )
    ])
    
    return create_collapsible_card(
        title="Paramètres de Simulation",
        content=html.Div([grid_content, trading_modes]),
        id_prefix="sim-params-card",
        is_open=True
    )

def register_simulation_callbacks(app):
    """
    Enregistre les callbacks spécifiques à l'onglet Simulation
    
    Args:
        app: L'instance de l'application Dash
    """
    # Pas de callbacks spécifiques à cet onglet actuellement, 
    # les callbacks pour les composants retro sont gérés globalement
    pass