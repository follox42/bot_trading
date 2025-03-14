from dash import html
import dash_bootstrap_components as dbc

def create_placeholder_page(title):
    """
    Crée une page placeholder pour les fonctionnalités non implémentées
    
    Args:
        title: Titre de la page
    
    Returns:
        Layout de la page placeholder
    """
    return html.Div([
        # En-tête de la page
        html.H2(title.upper(), className="text-xl text-cyan-300 font-bold mb-6 border-b border-gray-700 pb-2"),
        
        # Contenu principal
        html.Div([
            html.Div(
                className="retro-card-header",
                children=[
                    html.H3("FONCTIONNALITÉ EN DÉVELOPPEMENT", className="retro-card-title")
                ]
            ),
            html.Div(
                className="retro-card-body",
                children=[
                    # Icône de construction
                    html.Div([
                        html.I(className="bi bi-gear-wide-connected", style={"fontSize": "80px", "color": "#22D3EE"})
                    ], className="text-center mb-5 mt-4"),
                    
                    # Message
                    html.Div([
                        html.H4("Cette section est en cours de développement", className="text-center mb-3 glitch-text"),
                        html.P("La fonctionnalité sera disponible dans une prochaine mise à jour.", className="text-center mb-4"),
                        html.P(f"Module: {title}", className="text-center text-muted mb-5")
                    ]),
                    
                    # Pied de page de la carte
                    html.Div([
                        html.Pre('''
    ──────▄▌▐▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▌
    ───▄▄██▌█ CONSTRUCTION EN COURS
    ▄▄▄▌▐██▌█ MODULE EN DÉVELOPPEMENT
    ███████▌█▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▌
    ▀(⊙)▀▀▀▀▀▀▀(⊙)(⊙)▀▀▀▀▀▀▀▀▀▀(⊙)▀▀
                        ''', className="text-center small", style={"color": "#FBBF24"})
                    ], className="text-center")
                ]
            )
        ], className="retro-card")
    ])

def create_error_page(error_message="Une erreur s'est produite"):
    """
    Crée une page d'erreur
    
    Args:
        error_message: Message d'erreur à afficher
    
    Returns:
        Layout de la page d'erreur
    """
    return html.Div([
        html.H2("ERREUR", className="text-xl text-red-500 font-bold mb-6 border-b border-gray-700 pb-2"),
        
        html.Div([
            html.Div(
                className="retro-card-header",
                children=[
                    html.H3("ERREUR SYSTÈME", className="retro-card-title", style={"color": "#F87171"})
                ]
            ),
            html.Div(
                className="retro-card-body",
                children=[
                    html.Div([
                        html.I(className="bi bi-exclamation-triangle", style={"fontSize": "80px", "color": "#F87171"})
                    ], className="text-center mb-5 mt-4"),
                    
                    html.Div([
                        html.H4("Erreur détectée", className="text-center mb-3 text-danger"),
                        html.P(error_message, className="text-center mb-4"),
                        html.Button(
                            "Retour au Dashboard", 
                            id="btn-back-to-dashboard", 
                            className="retro-button mt-3",
                            n_clicks=0
                        )
                    ], className="text-center")
                ]
            )
        ], className="retro-card")
    ])