"""
Onglet Gestion du Risque du créateur d'étude avancé.
"""
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import dash

from simulator.study_config_definitions import RISK_MODES, GENERAL_RISK_PARAMS
from ui.components.retro_ui import create_retro_range_slider, create_retro_toggle_button, create_collapsible_card

def create_risk_tab():
    """
    Crée l'onglet Gestion du Risque avec la nouvelle interface retro.
    
    Returns:
        Contenu de l'onglet gestion du risque
    """
    return html.Div([
        html.P("Configurez les paramètres de gestion du risque", className="text-muted mb-4"),
        
        # Modes de risque
        create_risk_modes_section(),
        
        # Paramètres généraux de risque
        create_general_risk_section(),
        
        # Sections spécifiques à chaque mode
        *create_risk_modes_params_sections()
    ])

def create_risk_modes_section(active_modes=None):
    """
    Crée la section de sélection des modes de risque.
    
    Args:
        active_modes: Liste des modes actifs
        
    Returns:
        Composant pour la section des modes de risque
    """
    if active_modes is None:
        active_modes = ["fixed", "atr_based"]
        
    risk_modes_data = [
        {"id": "fixed", "label": "FIXED", "index": 1},
        {"id": "atr_based", "label": "ATR BASED", "index": 2},
        {"id": "volatility_based", "label": "VOLATILITY", "index": 3}
    ]
    
    toggle_buttons = html.Div(
        className="retro-toggle-group",
        children=[
            create_retro_toggle_button(
                id_prefix="risk-mode",
                label=mode["label"],
                value=mode["id"],
                index=mode["index"],
                is_active=mode["id"] in active_modes
            ) for mode in risk_modes_data
        ]
    )
    
    return create_collapsible_card(
        title="Modes de Risque",
        content=toggle_buttons,
        id_prefix="risk-modes-card",
        is_open=True
    )

def create_general_risk_section():
    """
    Crée la section des paramètres généraux de risque.
    
    Returns:
        Composant pour la section des paramètres de risque
    """
    # Création d'une grille avec les sliders
    grid_content = html.Div(
        className="retro-grid",
        children=[
            # Plage de taille de position
            create_retro_range_slider(
                id_prefix="position-size",
                label="Taille de Position",
                min_val=GENERAL_RISK_PARAMS["position_size"]["min"],
                max_val=GENERAL_RISK_PARAMS["position_size"]["max"],
                step=GENERAL_RISK_PARAMS["position_size"]["step"],
                current_min=1.0,
                current_max=10.0,
                unit=GENERAL_RISK_PARAMS["position_size"]["unit"]
            ),
            
            # Plage de Stop Loss
            create_retro_range_slider(
                id_prefix="stop-loss",
                label="Stop Loss",
                min_val=GENERAL_RISK_PARAMS["stop_loss"]["min"],
                max_val=GENERAL_RISK_PARAMS["stop_loss"]["max"],
                step=GENERAL_RISK_PARAMS["stop_loss"]["step"],
                current_min=0.5,
                current_max=3.0,
                unit=GENERAL_RISK_PARAMS["stop_loss"]["unit"]
            ),
            
            # Plage de Take Profit
            create_retro_range_slider(
                id_prefix="take-profit",
                label="Take Profit",
                min_val=GENERAL_RISK_PARAMS["take_profit"]["min"],
                max_val=GENERAL_RISK_PARAMS["take_profit"]["max"],
                step=GENERAL_RISK_PARAMS["take_profit"]["step"],
                current_min=1.5,
                current_max=3.0,
                unit=GENERAL_RISK_PARAMS["take_profit"]["unit"]
            ),
        ]
    )
    
    return create_collapsible_card(
        title="Paramètres de Risque Généraux",
        content=grid_content,
        id_prefix="general-risk-card",
        is_open=True
    )

def create_risk_modes_params_sections():
    """
    Crée les sections de paramètres pour chaque mode de risque.
    
    Returns:
        Liste de composants pour les sections de paramètres de mode de risque
    """
    sections = []
    
    # Section Mode Fixed
    fixed_content = html.Div(
        className="retro-grid",
        children=[
            # Taille de position fixe
            create_retro_range_slider(
                id_prefix="fixed-position",
                label="Taille de Position (Fixed)",
                min_val=1,
                max_val=20,
                step=0.1,
                current_min=5,
                current_max=15,
                unit="%"
            ),
            
            # Stop Loss fixe
            create_retro_range_slider(
                id_prefix="fixed-sl",
                label="Stop Loss (Fixed)",
                min_val=0.1,
                max_val=10,
                step=0.1,
                current_min=1,
                current_max=5,
                unit="%"
            ),
            
            # Take Profit fixe
            create_retro_range_slider(
                id_prefix="fixed-tp",
                label="Take Profit (Fixed)",
                min_val=0.5,
                max_val=10,
                step=0.1,
                current_min=1.5,
                current_max=5,
                unit="x"
            ),
        ]
    )
    
    sections.append(create_collapsible_card(
        title="Mode Fixed",
        content=fixed_content,
        id_prefix="fixed-mode-card",
        is_open=True
    ))
    
    # Section Mode ATR
    atr_content = html.Div(
        className="retro-grid",
        children=[
            # Période ATR
            create_retro_range_slider(
                id_prefix="atr-period",
                label="Période ATR",
                min_val=5,
                max_val=30,
                step=1,
                current_min=7,
                current_max=20
            ),
            
            # Multiplicateur ATR
            create_retro_range_slider(
                id_prefix="atr-multiplier",
                label="Multiplicateur ATR",
                min_val=0.5,
                max_val=5.0,
                step=0.1,
                current_min=1.0,
                current_max=3.0
            ),
        ]
    )
    
    sections.append(create_collapsible_card(
        title="Mode ATR",
        content=atr_content,
        id_prefix="atr-mode-card",
        is_open=True
    ))
    
    # Section Mode Volatilité
    vol_content = html.Div(
        className="retro-grid",
        children=[
            # Période Volatilité
            create_retro_range_slider(
                id_prefix="vol-period",
                label="Période Volatilité",
                min_val=10,
                max_val=50,
                step=1,
                current_min=20,
                current_max=30
            ),
            
            # Multiplicateur Volatilité
            create_retro_range_slider(
                id_prefix="vol-multiplier",
                label="Multiplicateur Volatilité",
                min_val=0.5,
                max_val=5.0,
                step=0.1,
                current_min=1.0,
                current_max=2.0
            ),
        ]
    )
    
    sections.append(create_collapsible_card(
        title="Mode Volatilité",
        content=vol_content,
        id_prefix="vol-mode-card",
        is_open=True
    ))
    
    return sections

def register_risk_callbacks(app):
    """
    Enregistre les callbacks spécifiques à l'onglet Gestion du Risque
    
    Args:
        app: L'instance de l'application Dash
    """
    # Callback pour gérer la visibilité des sections de mode selon les choix de l'utilisateur
    @app.callback(
        [
            Output("fixed-mode-card", "style"),
            Output("atr-mode-card", "style"),
            Output("vol-mode-card", "style")
        ],
        [
            Input({"type": "risk-mode-toggle", "index": dash.ALL}, "data-active"),
            Input({"type": "risk-mode-toggle", "index": dash.ALL}, "data-value")
        ]
    )
    def toggle_risk_mode_sections(active_states, values):
        """Affiche ou masque les sections de mode selon les choix de l'utilisateur"""
        # Style par défaut (visible)
        visible_style = {"display": "block"}
        # Style masqué
        hidden_style = {"display": "none"}
        
        # État initial
        fixed_style = hidden_style
        atr_style = hidden_style
        vol_style = hidden_style
        
        # Si aucun mode actif, garder tout masqué
        if not active_states:
            return fixed_style, atr_style, vol_style
        
        # Parcourir les états actifs et valeurs pour déterminer quelles sections afficher
        for i, (is_active, value) in enumerate(zip(active_states, values)):
            if is_active == "true":  # La valeur est sous forme de chaîne
                if value == "fixed":
                    fixed_style = visible_style
                elif value == "atr_based":
                    atr_style = visible_style
                elif value == "volatility_based":
                    vol_style = visible_style
        
        return fixed_style, atr_style, vol_style