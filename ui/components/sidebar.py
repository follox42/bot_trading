from dash import html, dcc
import dash_bootstrap_components as dbc
from ui.components.icons import (
    icon_trading,
    icon_strategy,
    icon_data,
    icon_backtest,
    icon_logs,
    icon_settings,
    icon_studies
)

def create_sidebar(active_tab="dashboard"):
    """
    Crée la barre latérale de navigation
    
    Args:
        active_tab: Onglet actuellement actif
    
    Returns:
        Composant de barre latérale
    """
    # Éléments de navigation
    navigation_items = [
        {"id": "nav-dashboard", "text": "Dashboard", "icon": icon_trading, "value": "dashboard"},
        {"id": "nav-strategies", "text": "Stratégies", "icon": icon_strategy, "value": "strategies"},
        {"id": "nav-data", "text": "Données", "icon": icon_data, "value": "data"},
        {"id": "nav-backtest", "text": "Backtest", "icon": icon_backtest, "value": "backtest"},
        {"id": "nav-studies", "text": "Études", "icon": icon_studies, "value": "studies"},  # Nouvelle entrée
        {"id": "nav-logs", "text": "Logs", "icon": icon_logs, "value": "logs"},
        {"id": "nav-settings", "text": "Paramètres", "icon": icon_settings, "value": "settings"}
    ]
    
    # Créer les éléments de navigation
    nav_elements = []
    for item in navigation_items:
        is_active = item["value"] == active_tab
        
        nav_elements.append(
            html.Div(
                [
                    item["icon"](),
                    html.Span(item["text"])
                ],
                id=item["id"], 
                className=f"retro-nav-item {'active' if is_active else ''}", 
                n_clicks=0
            )
        )
    
    return html.Div(
        className="retro-card mb-4",
        children=[
            html.Div(nav_elements, className="p-2"),
            # Store pour suivre l'onglet actif
            dcc.Store(id='active-tab-store', data=active_tab)
        ]
    )