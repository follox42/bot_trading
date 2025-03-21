from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import dash
from datetime import datetime, timedelta
import pandas as pd
import os
import json
import threading

from logger.logger import LoggerType
from data.downloader import MarketDataDownloader, download_data
import config

def register_data_action_callbacks(app, central_logger=None):
    """
    Enregistre les callbacks pour les actions sur les données (suppression, mise à jour)
    
    Args:
        app: L'instance de l'application Dash
        central_logger: Instance du logger centralisé
    """
    # Initialiser le logger
    if central_logger:
        ui_logger = central_logger.get_logger("data_actions", LoggerType.UI)
        ui_logger.info("Enregistrement des callbacks pour actions sur les données")
    
    # Callback simplifié pour la suppression de fichiers
    @app.callback(
        [Output("download-status", "children", allow_duplicate=True)],
        [Input({"type": "btn-delete-data", "index": dash.ALL, "filename": dash.ALL}, "n_clicks")],
        prevent_initial_call=True
    )
    def delete_data_file(n_clicks_list):
        """Supprime un fichier de données lorsque le bouton est cliqué"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return [dash.no_update]
            
        # Identifier quel bouton a été cliqué
        trigger = ctx.triggered[0]
        if trigger['value'] is None or trigger['value'] == 0:
            return [dash.no_update]
            
        button_id_str = trigger['prop_id'].split('.n_clicks')[0]
        try:
            # Remplacer les simples quotes par des doubles quotes pour JSON valide
            button_id = json.loads(button_id_str)
            filename = button_id.get("filename")
            
            if central_logger:
                ui_logger.info(f"Tentative de suppression du fichier: {filename}")
                
            # Chemin du fichier à supprimer
            data_dir = os.path.join(os.getcwd(), 'data', 'historical')
            file_path = os.path.join(data_dir, filename)
            
            # Supprimer le fichier
            if os.path.exists(file_path):
                os.remove(file_path)
                if central_logger:
                    ui_logger.info(f"Fichier supprimé avec succès: {filename}")
                return [dbc.Alert(f"Fichier supprimé: {filename}", color="success")]
            else:
                if central_logger:
                    ui_logger.warning(f"Fichier introuvable: {filename}")
                return [dbc.Alert(f"Fichier introuvable: {filename}", color="warning")]
                
        except Exception as e:
            if central_logger:
                ui_logger.error(f"Erreur lors de la suppression: {str(e)}")
            return [dbc.Alert(f"Erreur: {str(e)}", color="danger")]
    
    # Callback pour actualiser la table après une action
    @app.callback(
        Output("data-table-container", "children", allow_duplicate=True),
        [Input("download-status", "children")],
        prevent_initial_call=True
    )
    def refresh_data_table_after_action(status):
        """Actualise le tableau de données après une action (suppression ou mise à jour)"""
        # Récupérer les données disponibles
        available_data = MarketDataDownloader.list_available_data()
        from ui.pages.data_page import format_data_table
        data_table = format_data_table(available_data)
        
        return data_table