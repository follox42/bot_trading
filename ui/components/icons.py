"""
Composants d'icônes pour l'interface utilisateur
Utilise Bootstrap Icons (incluses dans le thème Bootstrap)
"""
from dash import html

def icon_trading():
    """Icône de graphique/trading"""
    return html.I(className="bi bi-graph-up me-2")

def icon_strategy():
    """Icône de stratégie"""
    return html.I(className="bi bi-sliders me-2")

def icon_data():
    """Icône de données"""
    return html.I(className="bi bi-database me-2")

def icon_backtest():
    """Icône de backtest"""
    return html.I(className="bi bi-clipboard-data me-2")

def icon_logs():
    """Icône de logs"""
    return html.I(className="bi bi-journal-text me-2")

def icon_settings():
    """Icône de paramètres"""
    return html.I(className="bi bi-gear me-2")
    
def icon_studies():
    """Icône des études et comparaisons"""
    return html.I(className="bi bi-bar-chart-line me-2")