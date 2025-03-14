import dash
import dash_bootstrap_components as dbc
from dash import html
import json
import os

import config
from logger.logger import LoggerType
from ui.splash_screen import create_intro_page
from ui.dashboard import create_dashboard
from ui.pages.data_page import register_data_callbacks
from ui.pages.studies_page import create_studies_page

def create_app(central_logger=None):
    """
    Crée et configure l'application Dash
    
    Args:
        central_logger: Instance du logger centralisé
    
    Returns:
        L'application Dash configurée
    """
    # Initialiser le logger UI
    if central_logger:
        ui_logger = central_logger.get_logger("app", LoggerType.UI)
        ui_logger.info("Initialisation de l'application Dash")
    
    # Obtenir le chemin pour servir les assets statiques
    assets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets')
    
    # Initialiser l'app Dash avec thème sombre et icônes Bootstrap
    app = dash.Dash(
        __name__, 
        external_stylesheets=[
            getattr(dbc.themes, config.UI_THEME),
            dbc.icons.BOOTSTRAP # Ajout des icônes Bootstrap
        ],
        suppress_callback_exceptions=True,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        assets_folder=assets_path
    )
    
    # Injecter les styles dans l'app
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>Trading Nexus - Système de Trading</title>
            {%favicon%}
            {%css%}
            <link href="https://fonts.googleapis.com/css2?family=VT323&family=Share+Tech+Mono&display=swap" rel="stylesheet">
            <link rel="stylesheet" href="/assets/styles.css">
        </head>
        <body>
            <div class="crt-screen">
                <div class="scanline"></div>
                {%app_entry%}
            </div>
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Layout principal de l'application
    app.layout = html.Div([
        # Placeholder pour le contenu de la page actuelle
        html.Div(id="page-content", children=[
            create_intro_page(),
            create_dashboard(central_logger)
        ]),
        
        # Store l'état actuel de l'animation
        dash.dcc.Store(id='animation-state', data=json.dumps({'current_line': 0, 'animation_done': False})),
        
        # Store pour les tâches actives (au niveau global)
        dash.dcc.Store(id='active-tasks-store', data=json.dumps([])),
        
        # Interval pour l'animation d'intro
        dash.dcc.Interval(
            id='interval-animation',
            interval=150,  # en millisecondes
            n_intervals=0,
            max_intervals=-1  # Continue indéfiniment
        ),
        
        # Interval pour rafraîchir les logs
        dash.dcc.Interval(
            id='interval-log-refresh',
            interval=5000,  # toutes les 5 secondes
            n_intervals=0
        )
    ])
    
    # Enregistrer tous les callbacks
    if central_logger:
        from ui.splash_screen import register_intro_callbacks
        register_intro_callbacks(app, central_logger)
        from ui.dashboard import register_dashboard_callbacks
        register_dashboard_callbacks(app, central_logger)
        register_data_callbacks(app, central_logger)
        from ui.pages.logs_page import register_logs_callbacks
        register_logs_callbacks(app, central_logger)
        from ui.pages.data_page_actions import register_data_action_callbacks
        register_data_action_callbacks(app, central_logger)
        from ui.pages.studies_page import register_studies_callbacks
        register_studies_callbacks(app, central_logger)
    
    return app