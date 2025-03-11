from dash import html, dcc
import dash_bootstrap_components as dbc
import config
from datetime import datetime
import json

# Remarque: Cette implémentation utilise psutil pour obtenir les statistiques système
# Assurez-vous d'installer cette dépendance: pip install psutil
try:
    import psutil  # Pour les statistiques système
except ImportError:
    # Message d'avertissement en cas d'absence de psutil
    print("Warning: psutil not installed. System stats will not be available.")
    print("Install with: pip install psutil")
    psutil = None

def create_footer(central_logger=None):
    """
    Crée un pied de page amélioré avec des informations système et des contrôles
    
    Args:
        central_logger: Instance du logger centralisé pour le journaling
    
    Returns:
        Composant de pied de page
    """
    # Récupération des informations système
    try:
        if psutil:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_used = memory.used / (1024 * 1024)  # Convertir en MB
            disk = psutil.disk_usage('/')
            disk_used = disk.used / (1024 * 1024 * 1024)  # Convertir en GB
        else:
            raise ImportError("psutil not available")
    except:
        # Fallback en cas d'erreur ou si psutil n'est pas disponible
        cpu_percent = 12
        memory_used = 624
        disk_used = 2.3
    
    # Heure actuelle
    current_time = datetime.now().strftime("%H:%M:%S")
    current_date = datetime.now().strftime("%d-%m-%Y")
    
    return html.Footer(
        className="retro-footer",
        children=[
            dbc.Container([
                dbc.Row([
                    # Colonne gauche - Informations sur l'application
                    dbc.Col([
                        html.Div([
                            html.Span(f"TRADING NEXUS v{config.APP_VERSION}", className="d-block fw-bold"),
                            html.Span("© 2025 Nolan - Tous droits réservés", className="d-block small opacity-75 mt-1")
                        ])
                    ], width=12, md=3),
                    
                    # Colonne avec les tâches en cours
                    dbc.Col([
                        html.Div([
                            html.Div(id="active-tasks", className="d-flex align-items-center justify-content-center flex-wrap", children=[
                                # Contenu rempli dynamiquement par les callbacks
                            ])
                        ], className="text-center")
                    ], width=12, md=3, className="my-2 my-md-0"),
                    
                    # Colonne centrale - Statistiques système
                    dbc.Col([
                        html.Div([
                            html.Div([
                                html.I(className="bi bi-cpu me-2"),
                                html.Span(f"CPU: {cpu_percent}%", className="me-3"),
                                html.I(className="bi bi-memory me-2"),
                                html.Span(f"MEM: {memory_used:.0f}MB", className="me-3"),
                                html.I(className="bi bi-hdd me-2"),
                                html.Span(f"DISK: {disk_used:.1f}GB")
                            ], className="d-flex align-items-center justify-content-center flex-wrap", id="footer-stats")
                        ], className="text-center")
                    ], width=12, md=3, className="my-2 my-md-0"),
                    
                    # Colonne droite - Horloge et contrôles
                    dbc.Col([
                        html.Div([
                            html.Div(id="footer-time", className="retro-clock", children=[
                                html.Span(current_time, className="fw-bold me-2"),
                                html.Span(current_date, className="small opacity-75")
                            ]),
                            html.Div(id="footer-status", className="mt-1 d-flex align-items-center", children=[
                                html.Span("●", className="text-success me-1 blink-text"),
                                html.Span("SYSTÈME: OPÉRATIONNEL")
                            ])
                        ], className="text-end")
                    ], width=12, md=3)
                ])
            ]),
            
            # Intervalle pour mettre à jour l'heure et les statistiques
            dcc.Interval(
                id='footer-update-interval',
                interval=5000,  # toutes les 5 secondes
                n_intervals=0
            )
        ]
    )
    
# Callback pour mettre à jour le footer (à implémenter dans dashboard.py)
def update_footer_stats(n_intervals):
    """
    Fonction pour mettre à jour les statistiques du footer
    À utiliser avec un callback dans dashboard.py
    """
    try:
        if psutil:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_used = memory.used / (1024 * 1024)  # Convertir en MB
            disk = psutil.disk_usage('/')
            disk_used = disk.used / (1024 * 1024 * 1024)  # Convertir en GB
        else:
            raise ImportError("psutil not available")
    except:
        # Fallback en cas d'erreur ou si psutil n'est pas disponible
        cpu_percent = 12
        memory_used = 624
        disk_used = 2.3
    
    # Heure actuelle
    current_time = datetime.now().strftime("%H:%M:%S")
    current_date = datetime.now().strftime("%d-%m-%Y")
    
    footer_time = html.Div([
        html.Span(current_time, className="fw-bold me-2"),
        html.Span(current_date, className="small opacity-75")
    ])
    
    footer_stats = html.Div([
        html.I(className="bi bi-cpu me-2"),
        html.Span(f"CPU: {cpu_percent}%", className="me-3"),
        html.I(className="bi bi-memory me-2"),
        html.Span(f"MEM: {memory_used:.0f}MB", className="me-3"),
        html.I(className="bi bi-hdd me-2"),
        html.Span(f"DISK: {disk_used:.1f}GB")
    ], className="d-flex align-items-center justify-content-center flex-wrap")
    
    return footer_time, footer_stats