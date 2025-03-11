from dash import html
import dash_bootstrap_components as dbc

def create_stats_card(stats):
    """
    Crée une carte de statistiques
    
    Args:
        stats: Dictionnaire contenant les statistiques
    
    Returns:
        Composant de carte de statistiques
    """
    return html.Div([
        html.Div(
            className="retro-card-header",
            children=[
                html.H3("STATISTIQUES", className="retro-card-title")
            ]
        ),
        html.Div(
            className="retro-card-body",
            children=[
                dbc.Row([
                    dbc.Col([
                        html.Div("Stratégies actives", style={"color": "#9CA3AF", "fontSize": "14px"}),
                        html.Div(f"{stats['activeStrategies']}", style={"color": "#4ADE80", "fontSize": "22px", "fontWeight": "bold"})
                    ], width=6, className="mb-3"),
                    dbc.Col([
                        html.Div("Backtests totaux", style={"color": "#9CA3AF", "fontSize": "14px"}),
                        html.Div(f"{stats['totalBacktests']}", style={"color": "#22D3EE", "fontSize": "22px", "fontWeight": "bold"})
                    ], width=6, className="mb-3"),
                    dbc.Col([
                        html.Div("Optimisations", style={"color": "#9CA3AF", "fontSize": "14px"}),
                        html.Div(f"{stats['runningOptimizations']}", style={"color": "#A78BFA", "fontSize": "22px", "fontWeight": "bold"})
                    ], width=6, className="mb-3"),
                    dbc.Col([
                        html.Div("Dernière MàJ", style={"color": "#9CA3AF", "fontSize": "14px"}),
                        html.Div(f"{stats['dataLastUpdate']}", style={"color": "#FBBF24", "fontSize": "22px", "fontWeight": "bold"})
                    ], width=6, className="mb-3")
                ])
            ]
        )
    ], className="retro-card mb-4")