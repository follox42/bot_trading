from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import dash
import json
from datetime import datetime
import os

from logger.logger import LoggerType
import config

def create_logs_page(central_logger=None):
    """
    Crée la page de gestion des logs avec filtrage
    
    Args:
        central_logger: Instance du logger centralisé
    
    Returns:
        Layout de la page des logs
    """
    # Initialiser le logger
    if central_logger:
        ui_logger = central_logger.get_logger("logs_page", LoggerType.UI)
        ui_logger.info("Création de la page des logs")
    
    # Options de filtrage des logs
    log_types = [
        {"label": "Tous les logs", "value": "all"},
        {"label": "Système", "value": LoggerType.SYSTEM.value},
        {"label": "Stratégie", "value": LoggerType.STRATEGY.value},
        {"label": "Données", "value": LoggerType.DATA.value},
        {"label": "Interface", "value": LoggerType.UI.value},
        {"label": "API", "value": LoggerType.API.value},
        {"label": "Backtest", "value": LoggerType.BACKTEST.value},
        {"label": "Trading en direct", "value": LoggerType.LIVE_TRADING.value},
        {"label": "Optimisation", "value": LoggerType.OPTIMIZATION.value}
    ]
    
    # Options de filtrage par niveau de log
    log_levels = [
        {"label": "Tous les niveaux", "value": "all"},
        {"label": "DEBUG", "value": "DEBUG"},
        {"label": "INFO", "value": "INFO"},
        {"label": "WARNING", "value": "WARNING"},
        {"label": "ERROR", "value": "ERROR"},
        {"label": "CRITICAL", "value": "CRITICAL"}
    ]
    
    # Récupérer les logs récents pour avoir une idée des entrées
    logs_recent = central_logger.get_recent_logs(max_entries=100) if central_logger else []
    
    return html.Div([
        html.H2("LOGS DU SYSTÈME", className="text-xl text-cyan-300 font-bold mb-6 border-b border-gray-700 pb-2"),
        
        # Filtre et options
        html.Div([
            html.Div(
                className="retro-card-header",
                children=[
                    html.H3("FILTRES & ACTIONS", className="retro-card-title")
                ]
            ),
            html.Div(
                className="retro-card-body",
                style={"overflow": "visible"},
                children=[
                    dbc.Row([
                        # Filtres
                        dbc.Col([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Type de logs", className="mb-2"),
                                    dcc.Dropdown(
                                        id="log-type-filter",
                                        options=log_types,
                                        value="all",
                                        className="text-dark"
                                    )
                                ], width=6, style={"overflow": "visible"}),
                                dbc.Col([
                                    html.Label("Niveau", className="mb-2"),
                                    dcc.Dropdown(
                                        id="log-level-filter",
                                        options=log_levels,
                                        value="all",
                                        className="text-dark"
                                    )
                                ], width=6, style={"overflow": "visible"}),
                            ], className="mb-3", style={"overflow": "visible"}),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Nombre de logs à afficher", className="mb-2"),
                                    dcc.Slider(
                                        id="log-count-slider",
                                        min=10,
                                        max=500,
                                        step=10,
                                        value=100,
                                        marks={
                                            10: {'label': '10', 'style': {'color': '#9CA3AF'}},
                                            100: {'label': '100', 'style': {'color': '#9CA3AF'}},
                                            200: {'label': '200', 'style': {'color': '#9CA3AF'}},
                                            500: {'label': '500', 'style': {'color': '#9CA3AF'}}
                                        },
                                        className="mb-2"
                                    ),
                                ], width=9),
                                dbc.Col([
                                    html.Div(id="log-count-display", className="text-center pt-4"),
                                ], width=3),
                            ]),
                        ], width=8, style={"overflow": "visible"}),
                        
                        # Actions
                        dbc.Col([
                            html.Button(
                                [html.I(className="bi bi-arrow-repeat me-2"), "Actualiser"],
                                id="btn-refresh-logs",
                                className="retro-button w-100 mb-3"
                            ),
                            html.Button(
                                [html.I(className="bi bi-download me-2"), "Exporter"],
                                id="btn-export-logs",
                                className="retro-button secondary w-100 mb-3"
                            ),
                            html.Button(
                                [html.I(className="bi bi-trash me-2"), "Effacer"],
                                id="btn-clear-logs",
                                className="retro-button danger w-100"
                            )
                        ], width=4, className="d-flex flex-column justify-content-center")
                    ])
                ]
            )
        ], className="retro-card mb-4", style={"overflow": "visible"}),
        
        # Visionneuse de logs principale
        html.Div([
            html.Div(
                className="retro-card-header d-flex justify-content-between align-items-center",
                children=[
                    html.H3("JOURNAUX DU SYSTÈME", className="retro-card-title mb-0"),
                    html.Div([
                        html.Span("Auto-refresh", className="me-2"),
                        dbc.Switch(id="log-auto-refresh", value=True, className="d-inline-block")
                    ])
                ]
            ),
            html.Div(
                className="retro-card-body p-0",
                children=[
                    html.Div(
                        id="logs-container",
                        className="logs-terminal",
                        style={
                            "height": "500px", 
                            "overflowY": "auto",
                            "padding": "15px",
                            "backgroundColor": "#0F172A",
                            "fontFamily": "'Share Tech Mono', monospace",
                            "fontSize": "13px",
                            "color": "#D1D5DB"
                        },
                        children=[
                            # Logs seront ajoutés ici par le callback
                        ]
                    )
                ]
            ),
            html.Div(
                id="logs-status",
                className="retro-card-footer text-center py-2",
                children=[
                    html.Span(f"Affichage de {len(logs_recent)} logs")
                ]
            )
        ], className="retro-card mb-4"),
        
        # Zone de stockage et intervalle
        dcc.Store(id="logs-cache", data=json.dumps([])),
        dcc.Store(id="logs-page-loaded", data="true"),
        dcc.Store(id="simulation-active", data=True),
        dcc.Interval(
            id="logs-refresh-interval",
            interval=5000,  # 5 secondes
            n_intervals=0,
            disabled=False
        ),
        
        # Éléments pour l'interaction
        html.Div(id="scroll-trigger", style={"display": "none"}),
        html.Div(id="simulation-status", style={"display": "none"}),
        html.Div(id="logs-refresh-status", style={"display": "none"}),
        
        # Composant de téléchargement
        dcc.Download(id="logs-download")
    ])

def format_log_entry(log, show_source=True):
    """
    Formate une entrée de log pour l'affichage
    
    Args:
        log: Dictionnaire contenant les infos de log
        show_source: Afficher ou non la source du log
    
    Returns:
        Élément HTML pour l'entrée de log
    """
    # Déterminer la classe de style en fonction du niveau
    level = log.get('level', 'INFO').upper()
    level_class = 'log-info'  # Valeur par défaut
    
    if level == 'ERROR' or level == 'CRITICAL':
        level_class = 'log-error'
    elif level == 'WARNING':
        level_class = 'log-warning'
    elif level == 'DEBUG':
        level_class = 'log-debug'
    
    # Traitement du timestamp
    timestamp = log.get('timestamp', '')
    if isinstance(timestamp, str):
        # Différentes possibilités de format pour le timestamp
        if ' ' in timestamp:
            # Format "YYYY-MM-DD HH:MM:SS"
            time_part = timestamp.split(' ')[1].split(',')[0]  # Enlever les millisecondes s'il y en a
        elif 'T' in timestamp:
            # Format ISO "YYYY-MM-DDTHH:MM:SS"
            time_part = timestamp.split('T')[1].split('.')[0]
        else:
            # Si format inconnu, utiliser tel quel
            time_part = timestamp
    else:
        # Si pas une chaîne, utiliser une chaîne vide
        time_part = ''
    
    # Traitement de la source du log
    source_text = ""
    if show_source:
        source = log.get('source', '')
        if source:
            # Simplifier l'affichage de la source (enlever les préfixes communs si trop long)
            if len(source) > 25:
                # Prendre juste la dernière partie du nom si trop long
                source_parts = source.split('.')
                if len(source_parts) > 1:
                    source = source_parts[-1]
            source_text = f"[{source}] "
    
    # Message principal
    message = log.get('message', '')
    
    # Créer l'élément log avec une classe pour stylisation
    return html.Div(
        className="log-entry",
        children=[
            html.Span(time_part, className="log-timestamp me-2"),
            html.Span(f"{level}: ", className=level_class),
            html.Span(source_text, className="log-source me-1") if source_text else None,
            html.Span(message)
        ]
    )

def register_logs_callbacks(app, central_logger=None):
    """
    Enregistre les callbacks pour la page des logs
    
    Args:
        app: L'instance de l'application Dash
        central_logger: Instance du logger centralisé
    """
    # Initialiser le logger
    if central_logger:
        ui_logger = central_logger.get_logger("logs_page", LoggerType.UI)
    
    # Callback pour afficher le nombre de logs sélectionné
    @app.callback(
        Output("log-count-display", "children"),
        Input("log-count-slider", "value")
    )
    def update_log_count_display(count):
        return f"{count} logs"
    
    # Callback pour activer/désactiver l'auto-refresh
    @app.callback(
        Output("logs-refresh-interval", "disabled"),
        Input("log-auto-refresh", "value")
    )
    def toggle_refresh(auto_refresh):
        return not auto_refresh

    # Callback principal pour charger et filtrer les logs
    @app.callback(
        [Output("logs-container", "children"),
         Output("logs-status", "children"),
         Output("logs-cache", "data")],
        [Input("logs-refresh-interval", "n_intervals"),
         Input("btn-refresh-logs", "n_clicks"),
         Input("log-type-filter", "value"),
         Input("log-level-filter", "value"),
         Input("log-count-slider", "value")]
    )
    def update_logs(n_intervals, n_clicks, log_type, log_level, max_logs):
        """Fonction complète pour la mise à jour et le filtrage des logs"""
        if not central_logger:
            return [html.Div("Système de logging non disponible", className="text-danger")], "", json.dumps([])
        
        # Détecter ce qui a déclenché le callback
        ctx = dash.callback_context
        trigger_id = None
        
        if ctx.triggered:
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger_id == "btn-refresh-logs" and n_clicks:
                if central_logger:
                    ui_logger.info("Actualisation manuelle des logs")
        
        # Force la récupération de tous les logs lors d'une actualisation manuelle
        forced_refresh = trigger_id == "btn-refresh-logs"
        
        # Récupérer tous les logs avec une limite très élevée
        try:
            # Utilisez la méthode avec les nouveaux paramètres de filtrage
            if log_type != "all":
                all_logs = central_logger.get_recent_logs(max_entries=100000, logger_type=log_type)
            else:
                all_logs = central_logger.get_recent_logs(max_entries=100000)
                
            # Filtrage supplémentaire par niveau si nécessaire
            if log_level != "all":
                all_logs = [log for log in all_logs if log.get('level', '').upper() == log_level.upper()]
                
        except Exception as e:
            ui_logger.error(f"Erreur lors de la récupération des logs: {str(e)}")
            return [html.Div(f"Erreur lors de la récupération des logs: {str(e)}", className="text-danger")], "", json.dumps([])
        
        # Vérifier si des logs ont été récupérés
        if not all_logs:
            return [html.Div("Aucun log disponible", className="text-muted")], "Aucun log trouvé", json.dumps([])
        
        # Tri des logs par timestamp (plus récent en premier)
        # Conversion des timestamps en objets datetime pour le tri
        for log in all_logs:
            if 'timestamp' in log and isinstance(log['timestamp'], str):
                try:
                    # Format attendu peut varier
                    if ',' in log['timestamp']:
                        # Format avec millisecondes: "2025-03-10 17:40:50,310"
                        timestamp_str = log['timestamp'].split(',')[0]
                    else:
                        # Format sans millisecondes
                        timestamp_str = log['timestamp']
                    log['_timestamp_obj'] = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                except (ValueError, IndexError) as e:
                    ui_logger.debug(f"Erreur de parsing timestamp: {log['timestamp']}: {e}")
                    # En cas d'échec, utiliser l'heure actuelle
                    log['_timestamp_obj'] = datetime.now()
        
        # Tri par timestamp
        all_logs = sorted(all_logs, key=lambda x: x.get('_timestamp_obj', datetime.now()), reverse=True)
        
        # Supprimer l'attribut temporaire
        for log in all_logs:
            if '_timestamp_obj' in log:
                del log['_timestamp_obj']
        
        # Compter le nombre total de logs
        total_logs_count = len(all_logs)
        
        # Limiter au nombre demandé
        displayed_logs = all_logs[:max_logs]
        
        # Formater les logs pour l'affichage
        if not displayed_logs:
            log_elements = [html.Div("Aucun log correspondant aux critères de filtrage", className="text-muted")]
        else:
            # Créer une NOUVELLE liste avec les éléments formatés
            log_elements = []
            for log in displayed_logs:
                log_element = format_log_entry(log)
                
                # Ajouter un attribut data-new pour les entrées récentes lors d'une actualisation manuelle
                if forced_refresh and trigger_id == "btn-refresh-logs":
                    # Ajouter un style de surbrillance pour les nouveaux logs
                    log_element.style = {"marginBottom": "2px", "animation": "highlight 2s ease-in-out"}
                
                log_elements.append(log_element)
        
        # Mettre à jour le statut avec des informations détaillées
        status_text = f"Affichage de {len(displayed_logs)} logs sur {total_logs_count} filtrés"
        
        # Ajouter une indication spécifique si c'est une actualisation manuelle
        if forced_refresh:
            status_text += " - Actualisation effectuée"
            
        status = html.Span(status_text)
        
        # Stocker les logs dans le cache pour l'export
        logs_cache = json.dumps(displayed_logs)
        
        return log_elements, status, logs_cache

    # Callback pour effacer les logs
    @app.callback(
        Output("logs-container", "children", allow_duplicate=True),
        [Input("btn-clear-logs", "n_clicks")],
        prevent_initial_call=True
    )
    def clear_logs(n_clicks):
        if not n_clicks:
            return dash.no_update
            
        if central_logger:
            ui_logger.warning("Effacement des logs demandé")
            # Cette fonction ne fait que vider l'affichage, pas les fichiers de log
            
        return [html.Div("Logs effacés de l'affichage", className="text-warning")]
    
    # Callback pour exporter les logs
    @app.callback(
        Output("logs-download", "data"),
        [Input("btn-export-logs", "n_clicks")],
        [State("logs-cache", "data"),
         State("log-type-filter", "value")],
        prevent_initial_call=True
    )
    def export_logs(n_clicks, logs_cache, log_type):
        if not n_clicks or not logs_cache:
            return dash.no_update
            
        # Convertir le cache en données CSV
        logs = json.loads(logs_cache)
        if not logs:
            return dash.no_update
            
        # Créer le contenu CSV
        csv_content = "timestamp,level,source,message\n"
        for log in logs:
            timestamp = log.get('timestamp', '')
            level = log.get('level', '')
            source = log.get('source', '')
            message = log.get('message', '').replace(',', ';')  # Éviter les problèmes avec les virgules
            
            csv_content += f"{timestamp},{level},{source},{message}\n"
        
        # Générer un nom de fichier basé sur la date et le filtre
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filter_name = log_type if log_type != "all" else "all_logs"
        filename = f"trading_nexus_logs_{filter_name}_{now}.csv"
        
        if central_logger:
            ui_logger.info(f"Export des logs vers {filename}")
            
        return dict(content=csv_content, filename=filename)
    
    # Callback pour montrer visuellement que l'actualisation se produit
    @app.callback(
        Output("btn-refresh-logs", "className"),
        [Input("btn-refresh-logs", "n_clicks")],
        prevent_initial_call=True
    )
    def show_refresh_feedback(n_clicks):
        """Ajoute une classe temporaire pour montrer que l'actualisation est en cours"""
        # Cette classe sera modifiée par le code JS ci-dessus
        return "retro-button w-100 mb-3 pulse-animation"
    
    # Callback client pour scroller automatiquement vers le haut lors des actualisations
    app.clientside_callback(
        """
        function(n_clicks, n_intervals) {
            // Solution plus robuste pour détecter l'input déclencheur
            // Utilise une variable d'état locale pour suivre les changements
            
            // Variables statiques pour suivre les valeurs précédentes
            if (typeof this.prevClicks === 'undefined') {
                this.prevClicks = 0;
            }
            if (typeof this.prevIntervals === 'undefined') {
                this.prevIntervals = 0;
            }
            
            // Détecter quel input a changé
            let triggeredByButton = false;
            if (n_clicks > this.prevClicks) {
                triggeredByButton = true;
                this.prevClicks = n_clicks;
            }
            this.prevIntervals = n_intervals;
            
            // Appliquer les effets si le bouton a été cliqué
            if (triggeredByButton) {
                var container = document.getElementById('logs-container');
                if (container) {
                    // Scroll vers le haut
                    container.scrollTop = 0;
                    
                    // Effet visuel subtil
                    container.style.boxShadow = "0 0 15px #22D3EE";
                    setTimeout(function() {
                        container.style.boxShadow = "none";
                    }, 500);
                    
                    // Restaurer la classe normale du bouton après l'animation
                    setTimeout(function() {
                        var button = document.getElementById('btn-refresh-logs');
                        if (button) {
                            button.className = "retro-button w-100 mb-3";
                        }
                    }, 600);
                }
            }
            
            return null;
        }
        """,
        Output("scroll-trigger", "children"),
        [Input("btn-refresh-logs", "n_clicks"),
        Input("logs-refresh-interval", "n_intervals")]
    )

def debug_log_filter(app, central_logger):
    """
    Ajoute un callback de débogage pour voir les sources de logs disponibles
    
    Args:
        app: L'instance de l'application Dash
        central_logger: Instance du logger centralisé
    """
    # Callback pour afficher les informations de débogage
    @app.callback(
        Output("debug-info", "children"),
        Input("btn-debug-logs", "n_clicks"),
        prevent_initial_call=True
    )
    def show_debug_info(n_clicks):
        if not n_clicks:
            return dash.no_update
            
        # Récupérer tous les logs
        all_logs = central_logger.get_recent_logs(max_entries=1000)
        
        # Collecter toutes les sources uniques
        unique_sources = set()
        source_counts = {}
        
        for log in all_logs:
            source = log.get('source', '')
            unique_sources.add(source)
            
            # Compter les occurrences
            if source in source_counts:
                source_counts[source] += 1
            else:
                source_counts[source] = 1
        
        # Créer un tableau HTML avec les informations
        debug_info = [
            html.H4("Informations de débogage des logs"),
            html.P(f"Nombre total de logs: {len(all_logs)}"),
            html.P(f"Nombre de sources uniques: {len(unique_sources)}"),
            html.H5("Sources disponibles:"),
            html.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Source"),
                        html.Th("Nombre de logs")
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(source),
                        html.Td(source_counts[source])
                    ]) for source in sorted(unique_sources)
                ])
            ], className="retro-table w-100")
        ]
        
        return debug_info