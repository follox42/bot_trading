from dash import html, dcc
import dash_bootstrap_components as dbc

def create_log_viewer(log_entries):
    """
    Crée un composant pour afficher les logs
    
    Args:
        log_entries: Liste des entrées de log à afficher
    
    Returns:
        Composant de visualisation des logs
    """
    logs_content = html.Div([
        html.Div([
            html.Span(log["timestamp"], className="log-timestamp"),
            html.Span(f"{log['level']}: ", className=f"log-{log['level'].lower()}"),
            html.Span(log["message"])
        ], style={"marginBottom": "5px"}) for log in log_entries
    ], className="retro-log-container", id="log-container")
    
    return html.Div([
        html.Div(
            className="retro-card-header",
            children=[
                html.H3("LOGS SYSTÈME", className="retro-card-title")
            ]
        ),
        html.Div(
            className="retro-card-body",
            children=[
                logs_content,
                html.Div([
                    # Lien direct vers la page des logs sans utiliser de callback
                    html.Button(
                        "Voir tous les logs >>",
                        id="btn-view-all-logs",
                        className="retro-button secondary mt-3",
                        style={"float": "right"}
                    )
                ], style={"textAlign": "right"})
            ]
        )
    ], className="retro-card")