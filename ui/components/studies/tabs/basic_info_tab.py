"""
Onglet Informations de Base du créateur d'étude amélioré avec sélection de données.
"""
from dash import html, dcc, Input, Output, State, callback, callback_context
import dash_bootstrap_components as dbc
import dash
import os
import pandas as pd
import threading
import time
from datetime import datetime
import json
from queue import Queue

from simulator.study_config_definitions import (
    DATA_SOURCES, DATA_PERIODS, STUDY_TYPES
)
from data.downloader import MarketDataDownloader

# Liste des paires de trading courantes organisées par catégorie
TRADING_PAIRS = [
    {
        "label": "Paires USDT",
        "options": [
            {"label": "BTC/USDT", "value": "BTC/USDT"},
            {"label": "ETH/USDT", "value": "ETH/USDT"},
            {"label": "BNB/USDT", "value": "BNB/USDT"},
            {"label": "XRP/USDT", "value": "XRP/USDT"},
            {"label": "ADA/USDT", "value": "ADA/USDT"},
            {"label": "SOL/USDT", "value": "SOL/USDT"},
            {"label": "DOT/USDT", "value": "DOT/USDT"},
            {"label": "DOGE/USDT", "value": "DOGE/USDT"},
            {"label": "AVAX/USDT", "value": "AVAX/USDT"},
            {"label": "MATIC/USDT", "value": "MATIC/USDT"},
            {"label": "LINK/USDT", "value": "LINK/USDT"},
            {"label": "LTC/USDT", "value": "LTC/USDT"},
            {"label": "UNI/USDT", "value": "UNI/USDT"}
        ]
    },
    {
        "label": "Paires USD",
        "options": [
            {"label": "BTC/USD", "value": "BTC/USD"},
            {"label": "ETH/USD", "value": "ETH/USD"},
            {"label": "XRP/USD", "value": "XRP/USD"},
            {"label": "BCH/USD", "value": "BCH/USD"}
        ]
    },
    {
        "label": "Paires BTC",
        "options": [
            {"label": "ETH/BTC", "value": "ETH/BTC"},
            {"label": "XRP/BTC", "value": "XRP/BTC"},
            {"label": "ADA/BTC", "value": "ADA/BTC"},
            {"label": "DOT/BTC", "value": "DOT/BTC"},
            {"label": "LTC/BTC", "value": "LTC/BTC"}
        ]
    },
    {
        "label": "Paires EUR",
        "options": [
            {"label": "BTC/EUR", "value": "BTC/EUR"},
            {"label": "ETH/EUR", "value": "ETH/EUR"},
            {"label": "XRP/EUR", "value": "XRP/EUR"}
        ]
    },
    {
        "label": "Autres",
        "options": [
            {"label": "Autre (personnalisé)", "value": "other"}
        ]
    }
]

# File partagée pour la communication entre threads
progress_queue = Queue()

# Structure partagée avec lock pour le suivi de la progression
class DownloadState:
    def __init__(self):
        self.data = {}
        self.lock = threading.Lock()
        
    def update(self, new_data):
        with self.lock:
            self.data.update(new_data)
            
    def get(self):
        with self.lock:
            return self.data.copy()

# Instance globale de l'état du téléchargement
download_state = DownloadState()

def create_basic_info_tab():
    """
    Crée l'onglet Informations de Base avec sélection de données.
    
    Returns:
        Contenu de l'onglet des informations de base
    """
    return html.Div([
        html.P("Configurez les informations de base de votre étude", className="text-muted mb-4"),
        
        # Informations essentielles
        dbc.Row([
            # Nom de l'étude
            dbc.Col([
                dbc.Label("Nom de l'étude", html_for="input-study-name"),
                dbc.Input(
                    id="input-study-name", 
                    type="text", 
                    placeholder="Ex: BTC_EMA_Cross_Strategy",
                    className="retro-date-input mb-3"
                ),
            ], width=12),
            
            # Asset/Paire avec liste déroulante
            dbc.Col([
                dbc.Label("Asset/Paire", html_for="input-study-asset-dropdown"),
                dcc.Dropdown(
                    id="input-study-asset-dropdown",
                    options=[option for group in TRADING_PAIRS for option in group['options']],
                    value="BTC/USDT",
                    className="retro-dropdown mb-2"
                ),
                # Champ texte pour paire personnalisée (initialement caché)
                dbc.Collapse(
                    dbc.Input(
                        id="input-study-asset-custom",
                        type="text",
                        placeholder="Entrez une paire personnalisée (ex: LINK/USDT)",
                        className="retro-date-input"
                    ),
                    id="custom-asset-collapse",
                    is_open=False,
                    className="mb-3"
                ),
                # Input caché pour stocker la valeur finale
                dcc.Input(
                    id="input-study-asset",
                    type="hidden",
                    value="BTC/USDT"
                )
            ], width=6),
            
            # Timeframe
            dbc.Col([
                dbc.Label("Timeframe", html_for="input-study-timeframe"),
                dcc.Dropdown(
                    id="input-study-timeframe",
                    options=[
                        {"label": "1 minute", "value": "1m"},
                        {"label": "5 minutes", "value": "5m"},
                        {"label": "15 minutes", "value": "15m"},
                        {"label": "30 minutes", "value": "30m"},
                        {"label": "1 heure", "value": "1h"},
                        {"label": "4 heures", "value": "4h"},
                        {"label": "1 jour", "value": "1d"},
                        {"label": "1 semaine", "value": "1w"}
                    ],
                    value="1h",
                    className="retro-dropdown mb-3"
                ),
            ], width=6),
            
            # Exchange
            dbc.Col([
                dbc.Label("Exchange", html_for="input-study-exchange"),
                dcc.Dropdown(
                    id="input-study-exchange",
                    options=[
                        {"label": "Binance", "value": "binance"},
                        {"label": "Bybit", "value": "bybit"},
                        {"label": "Bitget", "value": "bitget"},
                        {"label": "Kraken", "value": "kraken"},
                        {"label": "Kucoin", "value": "kucoin"},
                        {"label": "Coinbase", "value": "coinbase"}
                    ],
                    value="binance",
                    className="retro-dropdown mb-3"
                ),
            ], width=6),
            
            # Type d'étude
            dbc.Col([
                dbc.Label("Type d'étude", html_for="input-study-type"),
                dcc.Dropdown(
                    id="input-study-type",
                    options=STUDY_TYPES,
                    value="standard",
                    className="retro-dropdown mb-3"
                ),
            ], width=6),
            
            # Description
            dbc.Col([
                dbc.Label("Description", html_for="input-study-description"),
                dbc.Textarea(
                    id="input-study-description",
                    placeholder="Description de l'étude et de ses objectifs...",
                    style={"height": "100px"},
                    className="retro-date-input mb-3"
                ),
            ], width=12),
        ]),
        
        # Section données historiques améliorée
        dbc.Card([
            dbc.CardHeader(html.H5("Données historiques", className="mb-0 text-cyan-300")),
            dbc.CardBody([
                # Zone d'alerte pour les messages concernant les données
                html.Div(id="data-selection-alert", className="mb-3"),
                
                # Sélection de fichier de données
                html.Div([
                    dbc.Label("Sélection des données", html_for="data-file-selector"),
                    dcc.Dropdown(
                        id="data-file-selector",
                        options=[],  # Sera rempli par le callback
                        placeholder="Sélectionnez un fichier de données",
                        className="retro-dropdown mb-3"
                    ),
                    html.Div(id="data-file-info", className="mt-2 small")
                ], id="data-selection-container", className="mb-3"),
                
                # Bouton pour télécharger des données
                html.Div([
                    html.Button(
                        [html.I(className="bi bi-download me-2"), "TÉLÉCHARGER DES DONNÉES"],
                        id="btn-download-data-trigger",
                        className="retro-button me-3"
                    ),
                    html.Button(
                        [html.I(className="bi bi-arrow-repeat me-2"), "RAFRAÎCHIR LA LISTE"],
                        id="btn-refresh-data-list",
                        className="retro-button secondary ms-2"
                    )
                ], className="mb-3"),
                
                # Modal pour le téléchargement des données
                dbc.Modal([
                    dbc.ModalHeader([
                        html.H4("TÉLÉCHARGEMENT DE DONNÉES", className="retro-title text-cyan-300")
                    ], close_button=True),
                    dbc.ModalBody([
                        # Alert pour les erreurs ou messages
                        html.Div(id="download-data-alert", className="mb-3"),
                        
                        # Options de téléchargement
                        dbc.Row([
                            # Exchange
                            dbc.Col([
                                dbc.Label("Exchange", html_for="download-exchange"),
                                dcc.Dropdown(
                                    id="download-exchange",
                                    options=[
                                        {"label": "Binance", "value": "binance"},
                                        {"label": "Bitget", "value": "bitget"},
                                        {"label": "Bybit", "value": "bybit"}
                                    ],
                                    placeholder="Sélectionner un exchange",
                                    className="retro-dropdown mb-3"
                                ),
                            ], width=12),
                            
                            # Asset/Paire
                            dbc.Col([
                                dbc.Label("Asset/Paire", html_for="download-symbol"),
                                dbc.Input(
                                    id="download-symbol",
                                    type="text",
                                    placeholder="Ex: BTC/USDT",
                                    className="retro-date-input mb-3"
                                ),
                            ], width=12),
                            
                            # Timeframe
                            dbc.Col([
                                dbc.Label("Timeframe", html_for="download-timeframe"),
                                dcc.Dropdown(
                                    id="download-timeframe",
                                    options=[
                                        {"label": "1 minute", "value": "1m"},
                                        {"label": "5 minutes", "value": "5m"},
                                        {"label": "15 minutes", "value": "15m"},
                                        {"label": "30 minutes", "value": "30m"},
                                        {"label": "1 heure", "value": "1h"},
                                        {"label": "4 heures", "value": "4h"},
                                        {"label": "1 jour", "value": "1d"}
                                    ],
                                    placeholder="Sélectionner un timeframe",
                                    className="retro-dropdown mb-3"
                                ),
                            ], width=12),
                            
                            # Période prédéfinie ou dates personnalisées
                            dbc.Col([
                                dbc.Label("Période", html_for="download-period"),
                                dcc.Dropdown(
                                    id="download-period",
                                    options=[
                                        {"label": "1 mois", "value": "1m"},
                                        {"label": "3 mois", "value": "3m"},
                                        {"label": "6 mois", "value": "6m"},
                                        {"label": "1 an", "value": "1y"},
                                        {"label": "2 ans", "value": "2y"},
                                        {"label": "5 ans", "value": "5y"},
                                        {"label": "Tout l'historique", "value": "all"},
                                        {"label": "Dates personnalisées", "value": "custom"}
                                    ],
                                    value="1y",
                                    className="retro-dropdown mb-3"
                                ),
                            ], width=12),
                            
                            # Dates personnalisées (initialement masquées)
                            dbc.Collapse([
                                dbc.Label("Dates personnalisées"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Date de début"),
                                        dbc.Input(
                                            id="download-start-date",
                                            type="date",
                                            className="retro-date-input mb-3"
                                        ),
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Label("Date de fin"),
                                        dbc.Input(
                                            id="download-end-date",
                                            type="date",
                                            className="retro-date-input mb-3"
                                        ),
                                    ], width=6),
                                ]),
                            ], id="custom-dates-collapse", is_open=False),
                            
                            # Zone de progression (initialement masquée)
                            dbc.Collapse([
                                html.Div([
                                    html.H5("Téléchargement en cours...", className="text-center mb-3"),
                                    dbc.Progress(
                                        id="download-progress-bar",
                                        value=0,
                                        striped=True,
                                        animated=True,
                                        className="mb-3"
                                    ),
                                    html.P(
                                        id="download-progress-text",
                                        className="text-center small"
                                    )
                                ])
                            ], id="download-progress-collapse", is_open=False, className="mt-3"),
                        ]),
                    ]),
                    dbc.ModalFooter([
                        dbc.Button(
                            "Annuler", 
                            id="btn-cancel-download", 
                            className="retro-button secondary me-2"
                        ),
                        dbc.Button(
                            "Télécharger", 
                            id="btn-start-download", 
                            className="retro-button"
                        ),
                    ]),
                ], id="download-data-modal", size="lg", centered=True, className="retro-modal"),
                
                # Stockage du chemin du fichier de données
                dcc.Store(id="selected-data-file-path", data=None),
                
                # Stockage des métadonnées du fichier sélectionné
                dcc.Store(id="selected-data-metadata", data=None),
                
                # Stockage de l'état du téléchargement
                dcc.Store(id="download-state", data=None),
                
                # Store pour garder trace du téléchargement en cours
                dcc.Store(id="current-download-info", data=None),
                
                # Store pour le drapeau d'annulation
                dcc.Store(id="download-cancel-flag", data={"cancel": False}),
                
                # Debug output (invisible)
                html.Div(id="progress-debug", style={"display": "none"}),
            ]),
        ], className="mb-4 retro-subcard"),
        
        # Interval pour vérifier le statut du téléchargement et rafraîchir la liste
        dcc.Interval(
            id="download-check-interval",
            interval=500,  # 500ms pour une mise à jour plus fréquente
            n_intervals=0,
            disabled=True
        ),
    ])

def get_available_data_files(symbol=None, timeframe=None, exchange=None):
    """
    Récupère la liste des fichiers de données disponibles correspondant aux critères.
    
    Args:
        symbol (str, optional): Symbole/paire à filtrer
        timeframe (str, optional): Timeframe à filtrer
        exchange (str, optional): Exchange à filtrer
        
    Returns:
        list: Liste des fichiers de données correspondants
    """
    # Récupérer tous les fichiers de données disponibles
    all_data_files = MarketDataDownloader.list_available_data()
    
    # Filtrer selon les critères
    filtered_files = []
    
    for file_info in all_data_files:
        # Convertir les formats de symbole pour la comparaison
        file_symbol = file_info["symbol"].replace('_', '/')
        
        match = True
        
        if symbol and symbol.lower() not in file_symbol.lower():
            match = False
        
        if timeframe and file_info["timeframe"] != timeframe:
            match = False
            
        if exchange and file_info["exchange"] != exchange:
            match = False
            
        if match:
            filtered_files.append(file_info)
            
    return filtered_files

# Fonction pour le thread de téléchargement
def download_data_thread(params):
    """
    Fonction exécutée dans un thread pour télécharger les données avec support d'annulation
    """
    try:
        # Importation ici pour éviter les problèmes de dépendances circulaires
        from data.downloader import download_data
        
        # Marquer comme en cours de téléchargement
        params['status'] = 'downloading'
        params['progress'] = 0
        
        # Initialiser les variables selon la période
        start_date = None
        end_date = None
        limit = None
        
        if params['period'] == 'custom':
            # Utiliser les dates personnalisées
            start_date = params['start_date']
            end_date = params['end_date']
        else:
            # Convertir la période en limite
            if params['period'] == "1m":
                limit = 30 * 24 * 60  # ~30 jours en minutes
            elif params['period'] == "3m":
                limit = 90 * 24 * 60  # ~90 jours en minutes
            elif params['period'] == "6m":
                limit = 180 * 24 * 60  # ~180 jours en minutes
            elif params['period'] == "1y":
                limit = 365 * 24 * 60  # ~365 jours en minutes
            elif params['period'] == "2y":
                limit = 2 * 365 * 24 * 60  # ~2 ans en minutes
            elif params['period'] == "5y":
                limit = 5 * 365 * 24 * 60  # ~5 ans en minutes
            else:
                # Pour "all" ou autres valeurs non reconnues, utiliser 1 an par défaut
                limit = 365 * 24 * 60
        
        # Check if file already exists before starting download
        # Generate the filename to check
        import os
        data_dir = os.path.join(os.getcwd(), 'data', 'historical')
        filename_parts = [
            params['symbol'].replace('/', '_'),
            params['timeframe'],
            params['exchange']
        ]
        if limit:
            filename_parts.append(str(limit))
        file_path = os.path.join(data_dir, f"{'_'.join(filename_parts)}.csv")
        
        if os.path.exists(file_path) and not (start_date and end_date):
            # File already exists, skip download
            params['status'] = 'completed'
            params['progress'] = 100
            params['result'] = file_path
            params['message'] = "Fichier déjà existant, pas besoin de télécharger"
            
            # Mettre à jour l'état partagé
            download_state.update(params)
            # Mettre à jour la file
            progress_queue.put(params.copy())
            return
        
        # Fonction de callback pour suivre la progression réelle
        def progress_callback(progress, batch_count, max_batches, elapsed_time, remaining_time):
            # Calculate the percentage more explicitly
            percentage = min(100, int((batch_count / max_batches) * 100)) if max_batches > 0 else 0
            
            # Mise à jour des informations de progression
            params['progress'] = percentage  # Use our calculated percentage
            params['batch_count'] = batch_count
            params['max_batches'] = max_batches
            params['elapsed_time'] = elapsed_time
            params['remaining_time'] = remaining_time
            
            # Mettre à jour l'état partagé
            download_state.update(params)
            
            # Mettre à jour la file pour synchronisation avec l'UI
            progress_queue.put(params.copy())
            
            # Debug info
            print(f"Progress updated: {percentage}% ({batch_count}/{max_batches})")
        
        # Fonction pour vérifier si l'annulation a été demandée
        def should_cancel():
            return params.get('cancel', False)
        
        # Téléchargement des données avec le callback et support d'annulation
        result = download_data(
            exchange=params['exchange'],
            symbol=params['symbol'],
            timeframe=params['timeframe'],
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            progress_callback=progress_callback,
            should_cancel=should_cancel
        )
        
        # Vérifier si le téléchargement a été annulé
        if params.get('cancel', False):
            params['status'] = 'cancelled'
            params['progress'] = 0
            
            # Mettre à jour l'état partagé
            download_state.update(params)
            # Mettre à jour la file
            progress_queue.put(params.copy())
            return
        
        if result:
            # Téléchargement réussi
            params['status'] = 'completed'
            params['progress'] = 100
            params['result'] = result
        else:
            # Échec du téléchargement
            params['status'] = 'error'
            params['error'] = 'Échec du téléchargement des données'
            
        # Mettre à jour l'état partagé
        download_state.update(params)
        # Mettre à jour la file
        progress_queue.put(params.copy())
            
    except Exception as e:
        # Erreur lors du téléchargement
        params['status'] = 'error'
        params['error'] = str(e)
        
        # Mettre à jour l'état partagé
        download_state.update(params)
        # Mettre à jour la file
        progress_queue.put(params.copy())

def register_basic_info_callbacks(app):
    """
    Enregistre les callbacks spécifiques à l'onglet Informations de Base
    
    Args:
        app: L'instance de l'application Dash
    """
    # Debug progress
    @app.callback(
        Output("progress-debug", "children"),
        [Input("current-download-info", "data")]
    )
    def debug_progress(download_info):
        if download_info:
            print(f"Debug - Progress value: {download_info.get('progress', 0)}")
        return ""
    
    # Callback pour gérer l'affichage du champ personnalisé et mettre à jour la valeur finale
    @app.callback(
        [
            Output("custom-asset-collapse", "is_open"),
            Output("input-study-asset", "value")
        ],
        [
            Input("input-study-asset-dropdown", "value"),
            Input("input-study-asset-custom", "value")
        ]
    )
    def toggle_custom_asset(dropdown_value, custom_value):
        """Gère l'affichage du champ personnalisé et la valeur finale"""
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
        
        if dropdown_value == "other":
            # Afficher le champ personnalisé
            if trigger_id == "input-study-asset-custom" and custom_value:
                # Mise à jour depuis le champ personnalisé
                return True, custom_value
            else:
                # Juste afficher le champ personnalisé
                return True, dash.no_update
        else:
            # Utiliser la valeur du dropdown
            return False, dropdown_value
    
    # Callback pour mettre à jour la liste des fichiers de données disponibles
    @app.callback(
        [
            Output("data-file-selector", "options"),
            Output("data-file-selector", "value"),
            Output("data-selection-alert", "children")
        ],
        [
            Input("input-study-asset", "value"),
            Input("input-study-timeframe", "value"),
            Input("input-study-exchange", "value"),
            Input("btn-refresh-data-list", "n_clicks"),
            Input("download-check-interval", "n_intervals")
        ]
    )
    def update_data_files_list(symbol, timeframe, exchange, refresh_clicks, n_intervals):
        """Met à jour la liste des fichiers de données disponibles"""
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
        
        # Vérifier si tous les champs nécessaires sont remplis
        if not symbol or not timeframe or not exchange:
            # Message d'instructions
            alert = dbc.Alert(
                "Complétez les informations d'asset, timeframe et exchange pour voir les données disponibles",
                color="info",
                dismissable=True
            )
            return [], None, alert
        
        # Récupérer les fichiers disponibles
        available_files = get_available_data_files(symbol, timeframe, exchange)
        
        # Options pour le dropdown
        options = []
        for file_info in available_files:
            # Formatage de l'affichage
            label = f"{file_info['symbol']} | {file_info['timeframe']} | {file_info['rows']} bougies | {file_info['start_date']} à {file_info['end_date']}"
            
            options.append({
                "label": label,
                "value": file_info["filename"]
            })
        
        # Si aucun fichier n'est disponible
        if not options:
            # Message indiquant qu'aucune donnée n'est disponible
            alert = dbc.Alert(
                "Aucune donnée correspondant à ces critères n'est disponible. Utilisez le bouton pour télécharger des données.",
                color="warning",
                dismissable=True
            )
            return [], None, alert
        
        # Message de succès
        alert = dbc.Alert(
            f"{len(options)} fichiers de données disponibles pour ces critères.",
            color="success",
            dismissable=True
        ) if trigger_id == "btn-refresh-data-list" or trigger_id == "download-check-interval" else dash.no_update
        
        # Si des fichiers sont disponibles, sélectionner le plus récent
        selected_value = options[0]["value"] if options else None
        
        return options, selected_value, alert
    
    # Callback pour mettre à jour les informations du fichier sélectionné
    @app.callback(
        [
            Output("data-file-info", "children"),
            Output("selected-data-file-path", "data"),
            Output("selected-data-metadata", "data")
        ],
        [Input("data-file-selector", "value")]
    )
    def update_data_file_info(filename):
        """Met à jour les informations du fichier sélectionné"""
        if not filename:
            return None, None, None
            
        # Récupérer les informations du fichier
        all_files = MarketDataDownloader.list_available_data()
        file_info = next((f for f in all_files if f["filename"] == filename), None)
            
        if not file_info:
            return "Fichier non trouvé", None, None
            
        # Créer l'affichage des informations
        info_display = html.Div([
            html.Div([
                html.Strong("Période: "),
                html.Span(f"{file_info['start_date']} à {file_info['end_date']}")
            ], className="mb-1"),
            html.Div([
                html.Strong("Nombre de bougies: "),
                html.Span(f"{file_info['rows']}")
            ], className="mb-1"),
            html.Div([
                html.Strong("Taille: "),
                html.Span(f"{file_info['size_mb']} MB")
            ], className="mb-1"),
            html.Div([
                html.Strong("Dernière modification: "),
                html.Span(f"{file_info['last_modified']}")
            ])
        ])
        
        # Chemin complet du fichier
        data_dir = os.path.join(os.getcwd(), 'data', 'historical')
        file_path = os.path.join(data_dir, filename)
        
        return info_display, file_path, file_info
    
    # Callback pour afficher/masquer le modal de téléchargement
    @app.callback(
        Output("download-data-modal", "is_open"),
        [
            Input("btn-download-data-trigger", "n_clicks"),
            Input("btn-cancel-download", "n_clicks"),
            Input("download-state", "data")
        ],
        [State("download-data-modal", "is_open")]
    )
    def toggle_download_modal(n_trigger, n_cancel, download_state, is_open):
        """Ouvre ou ferme le modal de téléchargement"""
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
        
        # Si c'est le bouton de déclenchement, ouvrir le modal
        if trigger_id == "btn-download-data-trigger" and n_trigger:
            return True
        
        # Si c'est le bouton d'annulation, fermer le modal
        if trigger_id == "btn-cancel-download" and n_cancel:
            return False
        
        # Si le téléchargement est terminé avec succès, fermer le modal
        if trigger_id == "download-state" and download_state:
            download_status = download_state.get('status')
            if download_status in ['completed', 'cancelled', 'error']:
                return False
        
        return is_open

    # Callback pour afficher/masquer les dates personnalisées
    @app.callback(
        Output("custom-dates-collapse", "is_open"),
        [Input("download-period", "value")]
    )
    def toggle_custom_dates(period):
        """Affiche ou masque les dates personnalisées"""
        return period == "custom"
        
    # Callback pour gérer le drapeau d'annulation - MODIFIÉ pour éviter le cycle de dépendance
    @app.callback(
        Output("download-cancel-flag", "data"),
        [
            Input("btn-cancel-download", "n_clicks"),
            Input("btn-start-download", "n_clicks")
        ],
        [State("download-cancel-flag", "data")]
    )
    def handle_cancel_button(n_cancel, n_start, current_flag):
        """Gère le drapeau d'annulation du téléchargement"""
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
        
        # Initialisation
        flag = current_flag or {"cancel": False}
        
        # Si c'est le bouton d'annulation
        if trigger_id == "btn-cancel-download" and n_cancel:
            flag["cancel"] = True
            return flag
        
        # Si c'est le bouton de démarrage du téléchargement, réinitialiser le drapeau
        if trigger_id == "btn-start-download" and n_start:
            flag["cancel"] = False
            return flag
        
        return current_flag
    
    # Callback pour gérer le processus de téléchargement de données - CORRIGÉ
    @app.callback(
        [
            Output("download-progress-collapse", "is_open"),
            Output("btn-start-download", "disabled"),
            Output("btn-cancel-download", "disabled"),
            Output("download-data-alert", "children"),
            Output("download-state", "data"),
            Output("current-download-info", "data"),
            Output("download-check-interval", "disabled"),
            Output("download-cancel-flag", "data", allow_duplicate=True)
        ],
        [
            Input("btn-start-download", "n_clicks"),
            Input("btn-cancel-download", "n_clicks"),
            Input("download-check-interval", "n_intervals"),
            Input("download-cancel-flag", "data")
        ],
        [
            State("download-exchange", "value"),
            State("download-symbol", "value"),
            State("download-timeframe", "value"),
            State("download-period", "value"),
            State("download-start-date", "value"),
            State("download-end-date", "value"),
            State("current-download-info", "data")
        ],
        prevent_initial_call=True
    )
    def handle_download_process(
        n_clicks, n_cancel, n_intervals, cancel_flag,
        exchange, symbol, timeframe, period, 
        start_date, end_date, current_download
    ):
        """Gère le processus de téléchargement des données"""
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
        
        # Valeurs par défaut
        progress_visible = False
        start_btn_disabled = False
        cancel_btn_disabled = False
        alert = None
        download_state_data = None
        download_info = current_download
        interval_disabled = True
        
        # Si c'est le bouton de démarrage du téléchargement
        if trigger_id == "btn-start-download" and n_clicks:
            # Validation des entrées
            if not exchange or not symbol or not timeframe:
                alert = dbc.Alert(
                    "Veuillez remplir tous les champs obligatoires (exchange, symbol, timeframe).",
                    color="danger",
                    dismissable=True
                )
                return False, False, False, alert, None, None, True, dash.no_update
            
            # Si dates personnalisées, vérifier qu'elles sont bien renseignées
            if period == "custom" and (not start_date or not end_date):
                alert = dbc.Alert(
                    "Veuillez renseigner les dates de début et de fin pour un téléchargement personnalisé.",
                    color="danger",
                    dismissable=True
                )
                return False, False, False, alert, None, None, True, dash.no_update
            
            # Préparation des paramètres de téléchargement
            download_params = {
                'exchange': exchange,
                'symbol': symbol,
                'timeframe': timeframe,
                'period': period,
                'status': 'starting',
                'progress': 0,
                'start_time': time.time(),
                'cancel': False  # Initialiser le drapeau d'annulation
            }
            
            # Si dates personnalisées, les ajouter aux paramètres
            if period == "custom":
                download_params['start_date'] = start_date
                download_params['end_date'] = end_date
            
            # Afficher la progression et désactiver les boutons
            progress_visible = True
            start_btn_disabled = True
            
            # Activer l'intervalle de vérification
            interval_disabled = False
            
            # Réinitialiser l'état partagé
            global download_state
            download_state = DownloadState()
            download_state.update(download_params)
            
            # Vider la file de progression
            while not progress_queue.empty():
                progress_queue.get()
            
            # Lancer le téléchargement en arrière-plan
            thread = threading.Thread(
                target=download_data_thread,
                args=(download_params,)
            )
            thread.daemon = True
            thread.start()
            
            # Retourner les informations de téléchargement
            download_info = download_params
            
            alert = dbc.Alert(
                "Téléchargement démarré. Veuillez patienter...",
                color="info",
                dismissable=True
            )
            
            return progress_visible, start_btn_disabled, cancel_btn_disabled, alert, download_state_data, download_info, interval_disabled, {"cancel": False}
        
        # Si c'est le bouton d'annulation
        elif trigger_id == "btn-cancel-download" and n_cancel and current_download:
            # Marquer le téléchargement comme devant être annulé
            current_download['cancel'] = True
            
            # Mettre à jour l'état partagé
            download_state.update({'cancel': True})
            
            # Mettre à jour l'interface
            alert = dbc.Alert(
                "Annulation du téléchargement en cours...",
                color="warning",
                dismissable=True
            )
            
            # Désactiver le bouton de démarrage et d'annulation pour éviter les clics multiples
            progress_visible = True
            start_btn_disabled = True
            cancel_btn_disabled = True
            
            # On laisse le download-check-interval activé pour surveiller l'annulation
            return progress_visible, start_btn_disabled, cancel_btn_disabled, alert, None, current_download, interval_disabled, {"cancel": True}

        # Si c'est la mise à jour du drapeau d'annulation
        elif trigger_id == "download-cancel-flag" and cancel_flag and current_download:
            # Mettre à jour le drapeau d'annulation dans les infos du téléchargement
            current_download['cancel'] = cancel_flag.get('cancel', False)
            
            # Mettre à jour l'état partagé
            download_state.update({'cancel': cancel_flag.get('cancel', False)})
            
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, current_download, dash.no_update, dash.no_update
        
        # Si c'est l'intervalle de vérification
        elif trigger_id == "download-check-interval":
            # Récupérer les informations de progression depuis la structure partagée
            shared_info = download_state.get()
            
            # Vérifier s'il y a de nouvelles informations dans la file
            updated_info = None
            while not progress_queue.empty():
                updated_info = progress_queue.get()
            
            # Utiliser les informations les plus récentes
            if updated_info:
                download_info = updated_info
            elif shared_info:
                download_info = shared_info
            
            # Vérifier si un téléchargement est en cours
            if download_info and (download_info.get('status') == 'starting' or download_info.get('status') == 'downloading'):
                # Mise à jour de l'état du téléchargement
                progress_visible = True
                start_btn_disabled = True
                cancel_btn_disabled = False
                
                # Téléchargement toujours en cours
                interval_disabled = False
                
                # Pas de nouvelle mise à jour nécessaire
                return progress_visible, start_btn_disabled, cancel_btn_disabled, dash.no_update, dash.no_update, download_info, interval_disabled, dash.no_update
            
            # Si le téléchargement a été annulé
            elif download_info and download_info.get('status') == 'cancelled':
                progress_visible = False
                start_btn_disabled = False
                cancel_btn_disabled = False
                
                # Désactiver l'intervalle
                interval_disabled = True
                
                alert = dbc.Alert(
                    "Téléchargement annulé par l'utilisateur.",
                    color="warning",
                    dismissable=True
                )
                
                # Créer l'état pour fermer le modal
                download_state_data = {'status': 'cancelled'}
                
                return progress_visible, start_btn_disabled, cancel_btn_disabled, alert, download_state_data, None, interval_disabled, {"cancel": False}
            
            # Si le téléchargement est terminé
            elif download_info and download_info.get('status') == 'completed':
                progress_visible = False
                start_btn_disabled = False
                cancel_btn_disabled = False
                
                # Désactiver l'intervalle
                interval_disabled = True
                
                # Créer l'état de téléchargement pour fermer le modal
                download_state_data = {'status': 'completed'}
                
                # Vérifier s'il y a un message personnalisé
                if 'message' in download_info:
                    alert = dbc.Alert(
                        download_info['message'],
                        color="success",
                        dismissable=True
                    )
                else:
                    alert = dbc.Alert(
                        "Téléchargement terminé avec succès !",
                        color="success",
                        dismissable=True
                    )
                
                return progress_visible, start_btn_disabled, cancel_btn_disabled, alert, download_state_data, None, interval_disabled, {"cancel": False}
            
            # Si le téléchargement a échoué
            elif download_info and download_info.get('status') == 'error':
                progress_visible = False
                start_btn_disabled = False
                cancel_btn_disabled = False
                
                # Désactiver l'intervalle
                interval_disabled = True
                
                alert = dbc.Alert(
                    f"Erreur lors du téléchargement: {download_info.get('error', 'Erreur inconnue')}",
                    color="danger",
                    dismissable=True
                )
                
                # Créer l'état pour fermer le modal
                download_state_data = {'status': 'error'}
                
                return progress_visible, start_btn_disabled, cancel_btn_disabled, alert, download_state_data, None, interval_disabled, {"cancel": False}
        
        # Par défaut
        return progress_visible, start_btn_disabled, cancel_btn_disabled, alert, download_state_data, download_info, interval_disabled, dash.no_update
        
    # CALLBACK CORRIGÉ pour la mise à jour de la barre de progression
    @app.callback(
        [
            Output("download-progress-bar", "value"),
            Output("download-progress-text", "children")
        ],
        [Input("download-check-interval", "n_intervals")],
        prevent_initial_call=True
    )
    def update_progress_bar(n_intervals):
        """Met à jour la barre de progression du téléchargement avec estimation précise"""
        # Obtenir les informations de téléchargement depuis la structure partagée
        download_info = download_state.get()
        
        if not download_info:
            return 0, "Initialisation..."
            
        progress = download_info.get('progress', 0)
        print(f"UI updating progress bar: {progress}")
        
        # Texte de progression
        if download_info.get('status') == 'starting':
            progress_text = "Initialisation du téléchargement..."
        elif download_info.get('status') == 'downloading':
            # Si l'annulation a été demandée
            if download_info.get('cancel', False):
                progress_text = "Annulation du téléchargement en cours..."
            else:
                # Obtenir le temps écoulé
                elapsed_time = download_info.get('elapsed_time')
                if not elapsed_time:
                    elapsed_time = time.time() - download_info.get('start_time', time.time())
                
                # Formatage du temps écoulé
                elapsed_minutes = int(elapsed_time) // 60
                elapsed_seconds = int(elapsed_time) % 60
                
                # Formatage du temps restant si disponible
                remaining_time = download_info.get('remaining_time')
                if remaining_time is not None and remaining_time > 0:
                    remaining_minutes = int(remaining_time) // 60
                    remaining_seconds = int(remaining_time) % 60
                    
                    # Texte complet avec estimation de temps restant
                    progress_text = f"Téléchargement en cours... {progress}% - {elapsed_minutes}m {elapsed_seconds}s écoulées - environ {remaining_minutes}m {remaining_seconds}s restantes"
                    
                    # Ajouter des informations sur les lots si disponibles
                    if 'batch_count' in download_info and 'max_batches' in download_info:
                        progress_text += f" ({download_info['batch_count']}/{download_info['max_batches']} lots)"
                else:
                    # Sans estimation de temps restant
                    progress_text = f"Téléchargement en cours... {progress}% ({elapsed_minutes}m {elapsed_seconds}s écoulées)"
        elif download_info.get('status') == 'cancelled':
            progress_text = "Téléchargement annulé par l'utilisateur."
        elif download_info.get('status') == 'completed':
            progress_text = "Téléchargement terminé avec succès !"
        else:
            progress_text = f"Statut: {download_info.get('status', 'inconnu')}"
                
        return progress, progress_text

    # Initialisation du téléchargement modal
    @app.callback(
        [
            Output("download-exchange", "value"),
            Output("download-symbol", "value"),
            Output("download-timeframe", "value")
        ],
        [Input("btn-download-data-trigger", "n_clicks")],
        [
            State("input-study-exchange", "value"),
            State("input-study-asset", "value"),
            State("input-study-timeframe", "value")
        ]
    )
    def init_download_modal(n_clicks, exchange, asset, timeframe):
        """Initialise le modal de téléchargement avec les valeurs de l'étude"""
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update
            
        return exchange, asset, timeframe