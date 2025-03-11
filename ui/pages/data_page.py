from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import dash
from datetime import datetime, timedelta
import pandas as pd
import os
import threading
import json

from logger.logger import LoggerType
from data.downloader import MarketDataDownloader, download_data
import config

def create_data_page(central_logger=None):
    """
    Crée la page de gestion des données
    
    Args:
        central_logger: Instance du logger centralisé
    
    Returns:
        Layout de la page de données
    """
    # Initialiser le logger
    if central_logger:
        ui_logger = central_logger.get_logger("data_page", LoggerType.UI)
        ui_logger.info("Création de la page de données")
    
    # Liste des symboles disponibles
    symbols = [
        {"label": "BTC/USDT", "value": "BTC/USDT"},
        {"label": "ETH/USDT", "value": "ETH/USDT"},
        {"label": "SOL/USDT", "value": "SOL/USDT"},
        {"label": "BNB/USDT", "value": "BNB/USDT"},
        {"label": "XRP/USDT", "value": "XRP/USDT"},
        {"label": "ADA/USDT", "value": "ADA/USDT"},
        {"label": "DOGE/USDT", "value": "DOGE/USDT"},
        {"label": "DOT/USDT", "value": "DOT/USDT"}
    ]
    
    # Timeframes disponibles
    timeframes = [
        {"label": "1 minute", "value": "1m"},
        {"label": "5 minutes", "value": "5m"},
        {"label": "15 minutes", "value": "15m"},
        {"label": "30 minutes", "value": "30m"},
        {"label": "1 heure", "value": "1h"},
        {"label": "4 heures", "value": "4h"},
        {"label": "1 jour", "value": "1d"}
    ]
    
    # Exchanges disponibles
    exchanges = [
        {"label": "Bitget", "value": "bitget"},
        {"label": "Binance", "value": "binance"}
    ]
    
    # Date par défaut (30 jours en arrière)
    default_start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    default_end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Données disponibles
    available_data = MarketDataDownloader.list_available_data()
    
    # Formatage du tableau des données disponibles
    data_table = format_data_table(available_data)
    
    return html.Div([
        html.H2("GESTION DES DONNÉES", className="text-xl text-cyan-300 font-bold mb-6 border-b border-gray-700 pb-2"),
        
        dbc.Row([
            # Colonne gauche - Formulaire de téléchargement
            dbc.Col([
                html.Div([
                    html.Div(
                        className="retro-card-header",
                        children=[
                            html.H3("TÉLÉCHARGER DES DONNÉES", className="retro-card-title")
                        ]
                    ),
                    html.Div(
                        className="retro-card-body",
                        children=[
                            dbc.Form([
                                # Sélection du symbole
                                dbc.Row([
                                    dbc.Label("Symbole", width=12, html_for="input-symbol"),
                                    dbc.Col([
                                        dcc.Dropdown(
                                            id="input-symbol",
                                            options=symbols,
                                            value="BTC/USDT",
                                            className="text-dark"
                                        ),
                                    ]),
                                ], className="mb-3"),
                                
                                # Sélection du timeframe
                                dbc.Row([
                                    dbc.Label("Timeframe", width=12, html_for="input-timeframe"),
                                    dbc.Col([
                                        dcc.Dropdown(
                                            id="input-timeframe",
                                            options=timeframes,
                                            value="15m",
                                            className="text-dark"
                                        ),
                                    ]),
                                ], className="mb-3"),
                                
                                # Sélection de l'exchange
                                dbc.Row([
                                    dbc.Label("Exchange", width=12, html_for="input-exchange"),
                                    dbc.Col([
                                        dcc.Dropdown(
                                            id="input-exchange",
                                            options=exchanges,
                                            value="bitget",
                                            className="text-dark"
                                        ),
                                    ]),
                                ], className="mb-3"),
                                
                                # Date de début
                                dbc.Row([
                                    dbc.Label("Date de début", width=12, html_for="input-start-date"),
                                    dbc.Col([
                                        dbc.Input(
                                            id="input-start-date",
                                            type="date",
                                            value=default_start_date,
                                        ),
                                    ]),
                                ], className="mb-3"),
                                
                                # Date de fin
                                dbc.Row([
                                    dbc.Label("Date de fin", width=12, html_for="input-end-date"),
                                    dbc.Col([
                                        dbc.Input(
                                            id="input-end-date",
                                            type="date",
                                            value=default_end_date,
                                        ),
                                    ]),
                                ], className="mb-3"),
                                
                                # Bouton de téléchargement
                                dbc.Button(
                                    "Télécharger les données", 
                                    id="btn-download-data-page", 
                                    className="retro-button w-100",
                                    color="primary"
                                ),
                                
                                # Indicateurs de statut
                                html.Div(
                                    id="download-status",
                                    className="mt-3"
                                )
                            ]),
                        ]
                    )
                ], className="retro-card mb-4"),
                
                # Informations supplémentaires
                html.Div([
                    html.Div(
                        className="retro-card-header",
                        children=[
                            html.H3("INFORMATION", className="retro-card-title")
                        ]
                    ),
                    html.Div(
                        className="retro-card-body",
                        children=[
                            html.P([
                                "Les données sont téléchargées depuis l'API des exchanges et stockées localement en format CSV.",
                                html.Br(),
                                "Statistiques système :",
                            ]),
                            html.Ul([
                                html.Li(f"Nombre de fichiers : {len(available_data)}"),
                                html.Li(f"Espace total : {sum(d.get('size_mb', 0) for d in available_data):.2f} Mo"),
                                html.Li(f"Répertoire : {os.path.join(os.getcwd(), 'data', 'historical')}")
                            ], className="list-unstyled")
                        ]
                    )
                ], className="retro-card")
            ], width=12, lg=4),
            
            # Colonne droite - Données disponibles
            dbc.Col([
                html.Div([
                    html.Div(
                        className="retro-card-header d-flex justify-content-between align-items-center",
                        children=[
                            html.H3("DONNÉES DISPONIBLES", className="retro-card-title mb-0"),
                            html.Button(
                                [html.I(className="bi bi-arrow-repeat me-2"), "Actualiser"],
                                id="btn-refresh-data",
                                className="retro-button secondary btn-sm"
                            )
                        ]
                    ),
                    html.Div(
                        className="retro-card-body",
                        style={"overflowX": "auto"},
                        children=[
                            html.Div(
                                id="data-table-container",
                                children=data_table
                            )
                        ]
                    )
                ], className="retro-card")
            ], width=12, lg=8)
        ]),
        
        # Intervalles pour les mises à jour
        dcc.Interval(
            id='data-refresh-interval',
            interval=10000,  # 10 secondes
            n_intervals=0
        ),
        
        # Zone de stockage des données
        dcc.Store(id='download-in-progress', data=False)
    ])

def format_data_table(data_files):
    """
    Formate le tableau des données disponibles
    
    Args:
        data_files: Liste des fichiers de données
        
    Returns:
        Tableau HTML des données
    """
    if not data_files:
        return html.Div("Aucune donnée disponible.", className="text-center p-4 text-muted")
    
    # Trier par date de dernière modification (plus récente en premier)
    data_files = sorted(data_files, key=lambda x: x.get('last_modified', ''), reverse=True)
    
    # Créer le tableau
    rows = []
    for i, data in enumerate(data_files):
        row = html.Tr([
            html.Td(data['symbol']),
            html.Td(data['timeframe']),
            html.Td(data['exchange']),
            html.Td(f"{data['start_date']} à {data['end_date']}"),
            html.Td(f"{data['rows']:,}"),
            html.Td(f"{data['size_mb']:.2f} Mo"),
            html.Td(data['last_modified']),
            html.Td([
                html.Button(
                    html.I(className="bi bi-arrow-repeat"),
                    id={"type": "btn-update-data", "index": i, "filename": data['filename']},
                    className="retro-button secondary btn-sm me-1",
                    title="Mettre à jour"
                ),
                html.Button(
                    html.I(className="bi bi-trash"),
                    id={"type": "btn-delete-data", "index": i, "filename": data['filename']},
                    className="retro-button danger btn-sm",
                    title="Supprimer"
                )
            ])
        ])
        rows.append(row)
    
    table = html.Table([
        html.Thead(
            html.Tr([
                html.Th("Symbole", style={"width": "15%"}),
                html.Th("Timeframe", style={"width": "10%"}),
                html.Th("Exchange", style={"width": "10%"}),
                html.Th("Période", style={"width": "20%"}),
                html.Th("Lignes", style={"width": "10%"}),
                html.Th("Taille", style={"width": "10%"}),
                html.Th("Dernière MàJ", style={"width": "15%"}),
                html.Th("Actions", style={"width": "10%"})
            ])
        ),
        html.Tbody(rows)
    ], className="retro-table w-100")
    
    return table

def register_data_callbacks(app, central_logger=None):
    """
    Enregistre les callbacks pour la page de données
    
    Args:
        app: L'instance de l'application Dash
        central_logger: Instance du logger centralisé
    """
    # Initialiser le logger
    if central_logger:
        ui_logger = central_logger.get_logger("data_page", LoggerType.UI)
    
    # Store pour suivre l'état global des téléchargements
    global_download_state = {
        "in_progress": False,
        "thread_name": None,
        "start_time": None,
        "task_id": None
    }
    
    # Callback pour le téléchargement des données
    @app.callback(
        [Output("download-status", "children"),
         Output("download-in-progress", "data"),
         Output("data-table-container", "children"),
         Output("active-tasks-store", "data", allow_duplicate=True)],
        [Input("btn-download-data-page", "n_clicks"),
         Input("data-refresh-interval", "n_intervals"),
         Input("btn-refresh-data", "n_clicks")],
        [State("input-symbol", "value"),
         State("input-timeframe", "value"),
         State("input-exchange", "value"),
         State("input-start-date", "value"),
         State("input-end-date", "value"),
         State("download-in-progress", "data"),
         State("active-tasks-store", "data")],
        prevent_initial_call=True
    )
    def handle_data_operations(download_clicks, interval, refresh_clicks, 
                               symbol, timeframe, exchange, start_date, end_date, 
                               download_in_progress, active_tasks_json):
        """Gère les opérations de données (téléchargement et rafraîchissement)"""
        ctx = dash.callback_context
        
        # Si aucun déclencheur n'est détecté, ne rien faire
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Si le déclencheur est l'intervalle initial, ne rien faire
        if trigger_id == "data-refresh-interval" and ctx.triggered[0]['value'] == 0:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        # Charger les tâches actives
        active_tasks = json.loads(active_tasks_json) if active_tasks_json else []
        
        # Logique pour le téléchargement des données
        if trigger_id == "btn-download-data-page" and download_clicks and not download_in_progress:
            if central_logger:
                ui_logger.info(f"Début du téléchargement de {symbol} {timeframe} depuis {exchange}")
            
            # Créer un identifiant unique pour ce téléchargement
            thread_name = f"download_thread_{symbol.replace('/', '_')}_{timeframe}_{exchange}_{datetime.now().strftime('%H%M%S')}"
            
            # Enregistrer l'état global
            global_download_state["in_progress"] = True
            global_download_state["thread_name"] = thread_name
            global_download_state["start_time"] = datetime.now()
            
            # Créer un thread pour télécharger les données en arrière-plan
            def download_thread():
                try:
                    if central_logger:
                        ui_logger.info(f"Thread de téléchargement {thread_name} démarré")
                    
                    download_data(
                        exchange=exchange,
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                        central_logger=central_logger
                    )
                    
                    if central_logger:
                        ui_logger.info(f"Thread de téléchargement {thread_name} terminé avec succès")
                except Exception as e:
                    if central_logger:
                        ui_logger.error(f"Erreur dans le thread {thread_name}: {str(e)}")
            
            thread = threading.Thread(target=download_thread, name=thread_name)
            thread.daemon = True
            thread.start()
            
            # Ajouter la tâche à la liste des tâches actives
            task_id = f"download_{symbol.replace('/', '_')}_{timeframe}_{exchange}"
            global_download_state["task_id"] = task_id
            
            new_task = {
                "id": task_id,
                "type": "download",
                "description": f"Téléchargement {symbol} {timeframe}",
                "icon": "bi-cloud-download",
                "start_time": datetime.now().strftime("%H:%M:%S"),
                "details": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "exchange": exchange,
                    "thread_name": thread_name
                }
            }
            
            # Vérifier si la tâche existe déjà
            task_exists = False
            for i, task in enumerate(active_tasks):
                if task.get("id") == task_id:
                    task_exists = True
                    active_tasks[i] = new_task  # Mettre à jour la tâche existante
                    break
                    
            if not task_exists:
                active_tasks.append(new_task)
            
            return (
                dbc.Alert("Téléchargement en cours...", color="info"),
                True,  # Marquer le téléchargement comme en cours
                dash.no_update,
                json.dumps(active_tasks)
            )
        
        # Logique pour actualiser la liste des données et vérifier l'état des téléchargements
        elif ((trigger_id == "data-refresh-interval" and ctx.triggered[0]['value'] > 0) or 
              trigger_id == "btn-refresh-data" or 
              download_in_progress):
            
            # Vérifier si un téléchargement est en cours
            if download_in_progress or global_download_state["in_progress"]:
                active_threads = [t.name for t in threading.enumerate()]
                
                thread_name = global_download_state.get("thread_name")
                if thread_name and thread_name not in active_threads:
                    # Le téléchargement est terminé
                    if central_logger:
                        ui_logger.info(f"Téléchargement {thread_name} terminé (thread non trouvé)")
                    
                    # Mettre à jour l'état global
                    global_download_state["in_progress"] = False
                    global_download_state["thread_name"] = None
                    
                    # Mettre à jour le tableau des données
                    available_data = MarketDataDownloader.list_available_data()
                    data_table = format_data_table(available_data)
                    
                    # Supprimer la tâche de téléchargement de la liste des tâches actives
                    task_id = global_download_state.get("task_id")
                    updated_tasks = []
                    for task in active_tasks:
                        if task.get("id") != task_id:  # Garder les autres tâches
                            updated_tasks.append(task)
                    
                    if central_logger:
                        ui_logger.info(f"Tâche {task_id} supprimée des tâches actives")
                    
                    global_download_state["task_id"] = None
                    
                    return (
                        dbc.Alert("Téléchargement terminé avec succès !", color="success"),
                        False,  # Marquer le téléchargement comme terminé
                        data_table,
                        json.dumps(updated_tasks)
                    )
            
            # Sinon, simplement rafraîchir le tableau des données
            if trigger_id == "btn-refresh-data" and central_logger:
                ui_logger.info("Actualisation manuelle des données")
                
            available_data = MarketDataDownloader.list_available_data()
            data_table = format_data_table(available_data)
            
            return dash.no_update, download_in_progress, data_table, dash.no_update
        
        return dash.no_update, download_in_progress, dash.no_update, dash.no_update
    
    # Intervalle plus court pour vérifier l'état des téléchargements dans le footer
    @app.callback(
        Output("active-tasks-store", "data", allow_duplicate=True),
        Input("footer-update-interval", "n_intervals"),
        State("active-tasks-store", "data"),
        prevent_initial_call=True
    )
    def check_download_status(n_intervals, active_tasks_json):
        """Vérifie régulièrement l'état des téléchargements en cours"""
        if not global_download_state["in_progress"]:
            return dash.no_update
            
        active_tasks = json.loads(active_tasks_json) if active_tasks_json else []
        if not active_tasks:
            return dash.no_update
            
        active_threads = [t.name for t in threading.enumerate()]
        thread_name = global_download_state.get("thread_name")
        
        # Vérifier si le thread de téléchargement est toujours actif
        if thread_name and thread_name not in active_threads:
            # Le téléchargement est terminé
            if central_logger:
                ui_logger.info(f"Téléchargement {thread_name} terminé (vérification périodique)")
            
            # Mettre à jour l'état global
            global_download_state["in_progress"] = False
            global_download_state["thread_name"] = None
            
            # Supprimer la tâche de téléchargement de la liste des tâches actives
            task_id = global_download_state.get("task_id")
            updated_tasks = []
            for task in active_tasks:
                if task.get("id") != task_id:  # Garder les autres tâches
                    updated_tasks.append(task)
            
            if central_logger:
                ui_logger.info(f"Tâche {task_id} supprimée des tâches actives (vérification périodique)")
            
            global_download_state["task_id"] = None
            
            return json.dumps(updated_tasks)
        
        return dash.no_update