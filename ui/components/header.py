from dash import html
import dash_bootstrap_components as dbc
import config

def create_header():
    """
    Crée l'en-tête de l'application
    
    Returns:
        Composant d'en-tête
    """
    return html.Header(
        className="retro-header",
        children=[
            dbc.Container([
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H1(
                                    "TRADING NEXUS", 
                                    className="retro-title glitch-text", 
                                    style={"fontSize": "28px", "marginBottom": "0"}
                                ),
                                html.Span(
                                    config.APP_VERSION.split()[0], 
                                    className="retro-badge retro-badge-blue", 
                                    style={"marginLeft": "10px"}
                                )
                            ], className="d-flex align-items-center")
                        ], width=6),
                        dbc.Col([
                            html.Div([
                                html.Div(id="current-time", className="text-right", style={"color": "#9CA3AF", "fontSize": "14px"}),
                                html.Div([
                                    html.Span("SYSTÈME: ", style={"color": "#9CA3AF"}),
                                    html.Span("OPÉRATIONNEL", style={"color": "#4ADE80"})
                                ], className="text-right", style={"fontSize": "12px"})
                            ])
                        ], width=6, className="text-right")
                    ])
                ], className="retro-header-content")
            ])
        ]
    )