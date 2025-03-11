from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import dash
from datetime import datetime
import json

import config
from logger.logger import LoggerType
from data.mock_data import (
    generate_performance_data,
    generate_strategy_data,
    generate_system_stats
)

from ui.components.header import create_header
from ui.components.footer import create_footer, update_footer_stats
from ui.components.sidebar import create_sidebar
from ui.components.stats_card import create_stats_card
from ui.components.log_viewer import create_log_viewer
from ui.components.charts import create_performance_chart, create_strategies_table

def create_dashboard_content(central_logger=None):
    """
    Crée le contenu du tableau de bord principal
    
    Args:
        central_logger: Instance du logger centralisé
    
    Returns:
        Éléments HTML du contenu du tableau de bord
    """
    # Initialiser le logger UI
    if central_logger:
        ui_logger = central_logger.get_logger("dashboard_content", LoggerType.UI)
        ui_logger.info("Création du contenu du tableau de bord")
    
    # Génération des données de démonstration
    performance_data = generate_performance_data()
    strategy_data = generate_strategy_data(3)
    
    # Récupérer les logs réels du système
    log_entries = central_logger.get_recent_logs(max_entries=7) if central_logger else []
    
    formatted_logs = []
    for log in log_entries:
        # Extraire l'heure de la timestamp si présente pour les logs réels
        if isinstance(log, dict) and 'timestamp' in log:
            # Pour les logs réels du CentralizedLogger
            if isinstance(log['timestamp'], str) and ' ' in log['timestamp']:
                log_time = log['timestamp'].split(' ')[1]
            else:
                log_time = log.get('timestamp', 'N/A')
            
            formatted_logs.append({
                "timestamp": log_time,
                "level": log.get('level', 'INFO'),
                "message": log.get('message', '')
            })
    
    # Création du contenu du tableau de bord
    return [
        # En-tête du contenu principal
        html.H2("TABLEAU DE BORD", className="text-xl text-cyan-300 font-bold mb-6 border-b border-gray-700 pb-2"),
        
        # Graphique de performance
        create_performance_chart(performance_data),
        
        # Stratégies et logs en 2 colonnes
        dbc.Row([
            dbc.Col([
                create_strategies_table(strategy_data)
            ], width=6),
            dbc.Col([
                create_log_viewer(formatted_logs)
            ], width=6)
        ], className="mb-4"),
        
        # Actions rapides
        html.Div([
            html.Div(
                className="retro-card-header",
                children=[
                    html.H3("ACTIONS RAPIDES", className="retro-card-title")
                ]
            ),
            html.Div(
                className="retro-card-body",
                children=[
                    dbc.Row([
                        dbc.Col([
                            html.Button("Nouvelle stratégie", id="btn-new-strategy", className="retro-button w-100")
                        ], width=3),
                        dbc.Col([
                            html.Button("Télécharger données", id="btn-dashboard-download-data", className="retro-button w-100")
                        ], width=3),
                        dbc.Col([
                            html.Button("Lancer backtest", id="btn-run-backtest", className="retro-button w-100")
                        ], width=3),
                        dbc.Col([
                            html.Button("Arrêter tout", id="btn-stop-all", className="retro-button danger w-100")
                        ], width=3)
                    ])
                ]
            )
        ], className="retro-card")
    ]

def create_dashboard(central_logger=None):
    """
    Crée le tableau de bord principal
    
    Args:
        central_logger: Instance du logger centralisé
    
    Returns:
        Éléments HTML du tableau de bord
    """
    # Initialiser le logger UI
    if central_logger:
        ui_logger = central_logger.get_logger("dashboard", LoggerType.UI)
        ui_logger.info("Création du tableau de bord principal")
    
    # Création du tableau de bord
    return html.Div(
        id="dashboard-layout",
        style={"display": "none"},  # Initialement caché, sera affiché après l'intro
        children=[
            # En-tête
            create_header(),
            
            # Corps du dashboard
            dbc.Container([
                dbc.Row([
                    # Barre latérale
                    dbc.Col([
                        # Menu de navigation - accessible via l'ID pour les mises à jour
                        html.Div(id="sidebar-container", children=[
                            create_sidebar(active_tab="dashboard")
                        ]),
                        
                        # Carte des statistiques
                        create_stats_card(generate_system_stats())
                    ], width=3),
                    
                    # Contenu principal - Sera mis à jour dynamiquement
                    dbc.Col([
                        html.Div(
                            id='page-content-container', 
                            children=create_dashboard_content(central_logger)
                        )
                    ], width=8)
                ])
            ], style={"max-width": "90%"}),
            
            # Pied de page
            create_footer(central_logger),
            
            # Modale d'information pour les actions
            dbc.Modal(
                id="modal-info",
                centered=True,
                className="retro-modal",
                children=[
                    dbc.ModalHeader(html.H4("INFORMATION", className="retro-title")),
                    dbc.ModalBody([
                        html.Div([
                            html.P("Cette action est en cours de développement.", className="text-center mb-4")
                        ])
                    ]),
                    dbc.ModalFooter([
                        html.Button(
                            "Fermer",
                            id="btn-close-modal",
                            className="retro-button"
                        )
                    ])
                ]
            ),
            
            # Intervalle pour mettre à jour l'heure et les logs
            dcc.Interval(
                id='interval-time-update',
                interval=config.UI_REFRESH_INTERVAL,  # en millisecondes
                n_intervals=0
            )
        ]
    )

def register_dashboard_callbacks(app, central_logger):
    """
    Enregistre les callbacks pour le tableau de bord
    
    Args:
        app: L'instance de l'application Dash
        central_logger: Instance du logger centralisé
    """
    ui_logger = central_logger.get_logger("dashboard", LoggerType.UI)
    
    # Callback pour mettre à jour l'heure actuelle
    @app.callback(
        Output('current-time', 'children'),
        Input('interval-time-update', 'n_intervals')
    )
    def update_time(n_intervals):
        current_time = datetime.now().strftime("%H:%M:%S")
        ui_logger.debug(f"Mise à jour de l'heure: {current_time}")
        return current_time
        
    # Callback pour mettre à jour le footer
    @app.callback(
        [Output('footer-time', 'children'),
         Output('footer-stats', 'children')],
        [Input('footer-update-interval', 'n_intervals')]
    )
    def update_footer(n_intervals):
        return update_footer_stats(n_intervals)
    
    # Callback pour mettre à jour les tâches actives dans le footer
    @app.callback(
        Output('active-tasks', 'children'),
        [Input('active-tasks-store', 'data'),
         Input('footer-update-interval', 'n_intervals')],
        prevent_initial_call=True
    )
    def update_active_tasks(tasks_json, n_intervals):
        if not tasks_json:
            return html.Div("Aucune tâche en cours", className="text-muted small")
            
        tasks = json.loads(tasks_json) if tasks_json else []
        if not tasks:
            return html.Div("Aucune tâche en cours", className="text-muted small")
        
        task_elements = []
        for task in tasks:
            task_element = html.Div([
                html.I(className=f"bi {task.get('icon', 'bi-gear')} me-2"),
                html.Span(task.get('description', 'Tâche en cours'))
            ], className="me-3 badge bg-info text-dark")
            task_elements.append(task_element)
        
        return task_elements

    # Callback pour les boutons de navigation
    @app.callback(
        [Output('page-content-container', 'children'),
        Output('sidebar-container', 'children')],
        [Input('nav-dashboard', 'n_clicks'),
        Input('nav-strategies', 'n_clicks'),
        Input('nav-data', 'n_clicks'),
        Input('nav-backtest', 'n_clicks'),
        Input('nav-studies', 'n_clicks'),
        Input('nav-logs', 'n_clicks'),
        Input('nav-settings', 'n_clicks')],
        prevent_initial_call=True
    )
    def navigate_to_page(dash_clicks, strat_clicks, data_clicks, backtest_clicks, studies_clicks, logs_clicks, settings_clicks):
        ctx = dash.callback_context
        if not ctx.triggered:
            # Par défaut, afficher le dashboard
            return html.Div(id="dashboard-content", children=create_dashboard_content(central_logger)), create_sidebar("dashboard")
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        content = None
        active_tab = "dashboard"  # Par défaut
        
        if button_id == 'nav-dashboard':
            ui_logger.info("Navigation: Dashboard")
            content = html.Div(id="dashboard-content", children=create_dashboard_content(central_logger))
            active_tab = "dashboard"
        elif button_id == 'nav-strategies':
            ui_logger.info("Navigation: Stratégies")
            from ui.pages.placeholder import create_placeholder_page
            content = create_placeholder_page("Stratégies")
            active_tab = "strategies"
        elif button_id == 'nav-data':
            ui_logger.info("Navigation: Données")
            from ui.pages.data_page import create_data_page
            content = create_data_page(central_logger)
            active_tab = "data"
        elif button_id == 'nav-backtest':
            ui_logger.info("Navigation: Backtest")
            from ui.pages.placeholder import create_placeholder_page
            content = create_placeholder_page("Backtest")
            active_tab = "backtest"
        elif button_id == 'nav-studies':
            ui_logger.info("Navigation: Études")
            from ui.pages.studies_page import create_studies_page
            content = create_studies_page(central_logger)
            active_tab = "studies"
        elif button_id == 'nav-logs':
            ui_logger.info("Navigation: Logs")
            from ui.pages.logs_page import create_logs_page
            content = create_logs_page(central_logger)
            active_tab = "logs"
        elif button_id == 'nav-settings':
            ui_logger.info("Navigation: Paramètres")
            from ui.pages.placeholder import create_placeholder_page
            content = create_placeholder_page("Paramètres")
            active_tab = "settings"
        
        # Fallback
        if content is None:
            content = html.Div(id="dashboard-content", children=create_dashboard_content(central_logger))
            active_tab = "dashboard"
            
        # Créer une nouvelle sidebar avec l'onglet actif mis à jour
        sidebar = create_sidebar(active_tab=active_tab)
            
        return content, sidebar

    # Callback pour le bouton "Voir tous les logs"
    @app.callback(
        [Output('page-content-container', 'children', allow_duplicate=True),
        Output('sidebar-container', 'children', allow_duplicate=True)],
        [Input('btn-view-all-logs', 'n_clicks')],
        prevent_initial_call=True
    )
    def navigate_to_logs_page(n_clicks):
        if n_clicks and n_clicks > 0:
            ui_logger.info("Navigation vers la page des logs via le bouton du dashboard")
            from ui.pages.logs_page import create_logs_page
            return create_logs_page(central_logger), create_sidebar("logs")
        return dash.no_update, dash.no_update

    # Callback pour les boutons d'action
    @app.callback(
        Output('modal-info', 'is_open'),
        [Input('btn-new-strategy', 'n_clicks'),
         Input('btn-dashboard-download-data', 'n_clicks'),
         Input('btn-run-backtest', 'n_clicks'),
         Input('btn-stop-all', 'n_clicks'),
         Input('btn-close-modal', 'n_clicks')],
        [State('modal-info', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_action_modal(new_strat, download, backtest, stop_all, close, is_open):
        ctx = dash.callback_context
        if not ctx.triggered:
            return False
        
        # Vérifier que n_clicks est défini et supérieur à 0 pour être sûr qu'il s'agit d'un vrai clic
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        trigger_value = ctx.triggered[0]['value']
        
        # Si la valeur du trigger n'est pas un nombre ou est 0 ou None, ignorer
        if trigger_value is None or (isinstance(trigger_value, (int, float)) and trigger_value <= 0):
            return is_open
        
        # Journaliser le déclencheur pour déboguer
        ui_logger.info(f"Action modale déclenchée par: {button_id} avec valeur: {trigger_value}")
        
        if button_id == 'btn-close-modal':
            return False
        elif button_id == 'btn-stop-all' and stop_all:
            ui_logger.warning("Demande d'arrêt de toutes les stratégies")
            return True
        elif button_id == 'btn-dashboard-download-data' and download:
            ui_logger.info("Tentative de téléchargement de données depuis le dashboard")
            return True
        elif button_id == 'btn-run-backtest' and backtest:
            ui_logger.info("Tentative de lancement d'un backtest")
            return True
        elif button_id == 'btn-new-strategy' and new_strat:
            ui_logger.info("Création d'une nouvelle stratégie")
            return True
        
        return is_open

    # Rafraîchir le contenu des logs
    @app.callback(
        Output('log-container', 'children'),
        [Input('interval-log-refresh', 'n_intervals')]
    )
    def refresh_logs(n_intervals):
        ui_logger.debug("Rafraîchissement des logs")
        
        # Récupérer les logs réels du système
        log_data = central_logger.get_recent_logs(max_entries=20)
        formatted_logs = []
        
        for log in log_data:
            # Extraire l'heure de la timestamp si présente
            log_time = log.get('timestamp', '').split(' ')[1] if ' ' in log.get('timestamp', '') else ''
            formatted_logs.append({
                "timestamp": log_time,
                "level": log.get('level', 'INFO'),
                "message": log.get('message', '')
            })
        
        # S'il n'y a pas de logs, afficher un message plutôt que des logs fictifs
        if not formatted_logs:
            return [html.Div("Aucun log disponible pour le moment...", className="text-muted text-center py-5")]
        
        return [
            html.Div([
                html.Span(log["timestamp"], className="log-timestamp"),
                html.Span(f"{log['level']}: ", className=f"log-{log['level'].lower()}"),
                html.Span(log["message"])
            ], style={"marginBottom": "5px"}) for log in formatted_logs
        ]