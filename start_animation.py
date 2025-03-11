import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import time
import random
import os
import base64
from datetime import datetime
import json

# ASCII Art pour l'animation d'intro
def matrix_rain():
    rain = []
    for i in range(100):  # Nombre de gouttes
        x = random.randint(0, 98)  # Position X (colonnes)
        y = random.randint(0, 30)  # Position Y (lignes)
        char = random.choice('01')  # Caractère (0 ou 1)
        opacity = random.uniform(0.1, 1.0)  # Opacité aléatoire
        speed = random.uniform(0.5, 2.0)  # Vitesse de chute
        
        rain.append({
            'x': x,
            'y': y,
            'char': char,
            'opacity': opacity,
            'speed': speed
        })
    
    return rain

def create_splash_screen():
    # Styles pour l'écran d'introduction
    splash_styles = '''
    @font-face {
        font-family: 'VT323';
        src: url('https://fonts.googleapis.com/css2?family=VT323&display=swap');
    }
    
    @font-face {
        font-family: 'Share Tech Mono';
        src: url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    }
    
    body, html {
        margin: 0;
        padding: 0;
        height: 100%;
        width: 100%;
        background-color: #000;
        overflow: hidden;
        font-family: 'Share Tech Mono', monospace;
    }
    
    .splash-screen {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color: #000;
        color: #00FF9C;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 9999;
    }
    
    /* Effet scanline */
    .scanline {
        position: absolute;
        width: 100%;
        height: 4px;
        background: rgba(0, 255, 156, 0.2);
        z-index: 8;
        opacity: 0.75;
        animation: scanlines 3s linear infinite;
    }
    
    @keyframes scanlines {
        0% { top: -5%; }
        100% { top: 105%; }
    }
    
    /* Effet CRT */
    .crt-effect {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(ellipse at center, rgba(0,0,0,0) 0%,rgba(0,0,0,0.4) 100%);
        pointer-events: none;
        z-index: 7;
    }
    
    /* Effet de bruit */
    .noise {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("data:image/png;base64,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");
        opacity: 0.05;
        z-index: 6;
        pointer-events: none;
        animation: noise 0.5s linear infinite;
    }
    
    @keyframes noise {
        0% { background-position: 0 0; }
        100% { background-position: 100% 100%; }
    }
    
    /* Animation de flicker */
    .flicker {
        animation: flicker 0.3s infinite alternate;
    }
    
    @keyframes flicker {
        0%, 100% { opacity: 1.0; }
        30% { opacity: 0.95; }
        50% { opacity: 0.97; }
        70% { opacity: 0.93; }
    }
    
    /* Animation de blink */
    .blink {
        animation: blink 1s infinite step-end;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }
    
    /* Logo container */
    .logo-container {
        position: relative;
        width: 80%;
        max-width: 800px;
        margin-bottom: 30px;
        text-align: center;
    }
    
    /* Terminal container */
    .terminal-container {
        position: relative;
        width: 80%;
        max-width: 800px;
        height: 400px;
        border: 2px solid #00FF9C;
        box-shadow: 0 0 20px rgba(0, 255, 156, 0.7), inset 0 0 10px rgba(0, 255, 156, 0.5);
        padding: 20px;
        font-family: 'Share Tech Mono', monospace;
        overflow: hidden;
        background-color: rgba(0, 0, 0, 0.8);
    }
    
    .terminal-header {
        position: absolute;
        top: 0;
        right: 0;
        padding: 10px 20px;
        text-align: right;
    }
    
    .terminal-content {
        height: 100%;
        overflow-y: auto;
        padding-top: 40px;
        font-size: 16px;
        line-height: 1.4;
    }
    
    .terminal-line {
        margin-bottom: 8px;
    }
    
    .terminal-prompt {
        margin-top: 10px;
    }
    
    .press-key {
        position: absolute;
        bottom: 20px;
        left: 0;
        width: 100%;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    /* Matrix effect */
    .matrix-container {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 5;
        pointer-events: none;
        overflow: hidden;
    }
    
    .matrix-char {
        position: absolute;
        color: #00FF9C;
        font-family: monospace;
        font-size: 16px;
        text-shadow: 0 0 5px #00FF9C;
    }
    
    /* Glitch text effect */
    .glitch {
        position: relative;
        text-shadow: 0.05em 0 0 rgba(255,0,0,0.75), -0.025em -0.05em 0 rgba(0,255,0,0.75), 0.025em 0.05em 0 rgba(0,0,255,0.75);
        animation: glitch 500ms infinite;
    }
    
    .glitch span {
        position: absolute;
        top: 0;
        left: 0;
    }
    
    .glitch span:first-child {
        animation: glitch 650ms infinite;
        clip-path: polygon(0 0, 100% 0, 100% 45%, 0 45%);
        transform: translate(-0.025em, -0.0125em);
        opacity: 0.8;
    }
    
    .glitch span:last-child {
        animation: glitch 375ms infinite;
        clip-path: polygon(0 80%, 100% 20%, 100% 100%, 0 100%);
        transform: translate(0.0125em, 0.025em);
        opacity: 0.8;
    }
    
    @keyframes glitch {
        0% {
            text-shadow: 0.05em 0 0 rgba(255,0,0,0.75), -0.025em -0.05em 0 rgba(0,255,0,0.75), 0.025em 0.05em 0 rgba(0,0,255,0.75);
        }
        14% {
            text-shadow: 0.05em 0 0 rgba(255,0,0,0.75), -0.025em -0.05em 0 rgba(0,255,0,0.75), 0.025em 0.05em 0 rgba(0,0,255,0.75);
        }
        15% {
            text-shadow: -0.05em -0.025em 0 rgba(255,0,0,0.75), 0.025em 0.025em 0 rgba(0,255,0,0.75), -0.05em -0.05em 0 rgba(0,0,255,0.75);
        }
        49% {
            text-shadow: -0.05em -0.025em 0 rgba(255,0,0,0.75), 0.025em 0.025em 0 rgba(0,255,0,0.75), -0.05em -0.05em 0 rgba(0,0,255,0.75);
        }
        50% {
            text-shadow: 0.025em 0.05em 0 rgba(255,0,0,0.75), 0.05em 0 0 rgba(0,255,0,0.75), 0 -0.05em 0 rgba(0,0,255,0.75);
        }
        99% {
            text-shadow: 0.025em 0.05em 0 rgba(255,0,0,0.75), 0.05em 0 0 rgba(0,255,0,0.75), 0 -0.05em 0 rgba(0,0,255,0.75);
        }
        100% {
            text-shadow: -0.025em 0 0 rgba(255,0,0,0.75), -0.025em -0.025em 0 rgba(0,255,0,0.75), -0.025em -0.05em 0 rgba(0,0,255,0.75);
        }
    }
    
    /* Bouton pour accéder au dashboard */
    .enter-button {
        display: none;
        margin-top: 20px;
        padding: 10px 20px;
        background-color: transparent;
        color: #00FF9C;
        border: 2px solid #00FF9C;
        text-transform: uppercase;
        font-family: 'Share Tech Mono', monospace;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px rgba(0, 255, 156, 0.3);
    }
    
    .enter-button:hover {
        background-color: rgba(0, 255, 156, 0.2);
        box-shadow: 0 0 15px rgba(0, 255, 156, 0.5);
    }
    '''
    
    # Logo ASCII art
    logo_art = """
    ████████╗██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗ 
    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝ 
       ██║   ██████╔╝███████║██║  ██║██║██╔██╗ ██║██║  ███╗
       ██║   ██╔══██╗██╔══██║██║  ██║██║██║╚██╗██║██║   ██║
       ██║   ██║  ██║██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝
       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝ 
                                                            
    ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗             
    ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝             
    ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗             
    ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║             
    ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║             
    ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝             
    """
    
    # Séquence de démarrage pour l'animation
    boot_sequence = [
        "SYSTEM BOOTUP INITIATED...",
        "LOADING KERNEL...",
        "INITIALIZING HARDWARE...",
        "MEMORY CHECK: 16GB OK",
        "STORAGE CHECK: 1TB SSD OK",
        "NETWORK CHECK: CONNECTED",
        "GPU CHECK: NVIDIA RTX 4090 OK",
        "-----------------------------------------",
        "INITIALIZING TRADING SYSTEM...",
        "LOADING CORE MODULES...",
        "[ OK ] Data module loaded",
        "[ OK ] Optimization engine loaded",
        "[ OK ] Strategy framework loaded",
        "[ OK ] Backtest engine loaded",
        "CHECKING MARKET CONNECTIONS...",
        "[ OK ] Binance API connected",
        "[ OK ] Bitget API connected",
        "[ OK ] Database connection established",
        "INITIALIZING TRADING ALGORITHMS...",
        "[ OK ] All systems operational",
        "TRADING NEXUS v1.0.0 READY",
    ]
    
    # Création du HTML pour l'écran d'intro
    return html.Div(
        id="splash-container",
        className="splash-screen",
        children=[
            # Effets visuels CRT
            html.Div(className="scanline"),
            html.Div(className="crt-effect"),
            html.Div(className="noise"),
            
            # Effet Matrix en arrière-plan
            html.Div(
                id="matrix-container",
                className="matrix-container"
            ),
            
            # Logo
            html.Div(
                className="logo-container",
                children=[
                    html.Pre(
                        logo_art,
                        className="flicker",
                        style={
                            "fontSize": "10px",
                            "lineHeight": "1",
                            "margin": "0 auto",
                            "color": "#00FF9C",
                            "textShadow": "0 0 10px #00FF9C"
                        }
                    ),
                    html.Div(
                        "QUANTUM ALPHA RELEASE",
                        className="glitch",
                        style={
                            "marginTop": "10px",
                            "fontSize": "18px",
                            "color": "#00FFFF"
                        }
                    )
                ]
            ),
            
            # Container du terminal
            html.Div(
                className="terminal-container",
                children=[
                    # En-tête du terminal
                    html.Div(
                        className="terminal-header",
                        children=[
                            html.H3(
                                "NEXUS OS v1.0.0",
                                className="flicker",
                                style={
                                    "fontSize": "18px",
                                    "margin": "0",
                                    "color": "#00FF9C",
                                    "textShadow": "0 0 10px #00FF9C"
                                }
                            ),
                            html.P(
                                "SECURE TERMINAL",
                                style={
                                    "fontSize": "12px",
                                    "margin": "0",
                                    "color": "#00AAAA"
                                }
                            )
                        ]
                    ),
                    
                    # Contenu du terminal (sera mis à jour par le callback)
                    html.Div(
                        id="terminal-output",
                        className="terminal-content",
                        children=[]
                    ),
                    
                    # Message "Appuyez sur une touche"
                    html.Div(
                        id="press-key-message",
                        className="press-key blink",
                        style={"display": "none"},
                        children="APPUYEZ SUR UNE TOUCHE POUR CONTINUER"
                    )
                ]
            ),
            
            # Bouton pour accéder au dashboard (initialement caché)
            html.Button(
                "ENTRER DANS LE SYSTÈME",
                id="enter-button",
                className="enter-button",
                n_clicks=0
            ),
            
            # Store pour les données d'animation
            dcc.Store(id='boot-sequence-data', data=json.dumps({
                'sequence': boot_sequence,
                'current_index': 0,
                'completed': False
            })),
            
            # Interval pour l'animation
            dcc.Interval(
                id='animation-interval',
                interval=150,  # ms
                n_intervals=0,
                max_intervals=-1  # Continue indéfiniment
            ),
            
            # Interval pour l'animation Matrix
            dcc.Interval(
                id='matrix-interval',
                interval=50,  # ms
                n_intervals=0,
                max_intervals=-1
            ),
            
            # Style pour l'écran d'intro
            html.Style(splash_styles)
        ]
    )


# Callback pour l'animation de démarrage
def register_splash_callbacks(app):
    @app.callback(
        [Output('terminal-output', 'children'),
         Output('boot-sequence-data', 'data'),
         Output('press-key-message', 'style')],
        [Input('animation-interval', 'n_intervals')],
        [State('boot-sequence-data', 'data')]
    )
    def update_terminal_output(n_intervals, data_json):
        data = json.loads(data_json)
        boot_sequence = data['sequence']
        current_index = data['current_index']
        completed = data['completed']
        
        # Si l'animation est terminée, ne rien faire
        if completed:
            return dash.no_update, dash.no_update, dash.no_update
        
        # Si pas encore à la fin de la séquence
        if current_index < len(boot_sequence):
            # Ajouter la ligne actuelle
            terminal_lines = []
            
            for i in range(current_index + 1):
                line = boot_sequence[i]
                line_style = {}
                
                # Styler différemment selon le contenu
                if "[ OK ]" in line:
                    line_style = {"color": "#00FF00"}
                elif "INITIALIZING" in line or "LOADING" in line or "CHECKING" in line:
                    line_style = {"color": "#00FFFF"}
                elif "READY" in line:
                    line_style = {"color": "#FFFF00"}
                elif "ERROR" in line:
                    line_style = {"color": "#FF0000"}
                
                terminal_lines.append(html.Div(line, className="terminal-line", style=line_style))
            
            # Ajouter le curseur si c'est la dernière ligne
            if current_index == len(boot_sequence) - 1:
                terminal_lines.append(
                    html.Div(
                        className="terminal-prompt",
                        children=[
                            "$ ",
                            html.Span("█", className="blink")
                        ]
                    )
                )
                
                # Animation terminée, montrer le message "Appuyez sur une touche"
                updated_data = json.dumps({
                    'sequence': boot_sequence,
                    'current_index': current_index + 1,
                    'completed': True
                })
                
                # Afficher le message
                return terminal_lines, updated_data, {"display": "block"}
            
            # Mettre à jour l'index
            updated_data = json.dumps({
                'sequence': boot_sequence,
                'current_index': current_index + 1,
                'completed': False
            })
            
            return terminal_lines, updated_data, {"display": "none"}
        
        # Ne devrait pas arriver, mais au cas où
        return dash.no_update, dash.no_update, dash.no_update

    @app.callback(
        Output('matrix-container', 'children'),
        [Input('matrix-interval', 'n_intervals')]
    )
    def update_matrix_effect(n_intervals):
        # Génère des caractères Matrix aléatoires
        matrix_chars = []
        
        for i in range(50):  # Limiter le nombre pour les performances
            x = random.randint(0, 100)
            y = (n_intervals * random.uniform(0.2, 1.0)) % 100
            char = random.choice("01")
            opacity = random.uniform(0.1, 0.5)
            
            matrix_chars.append(
                html.Span(
                    char,
                    className="matrix-char",
                    style={
                        "left": f"{x}%",
                        "top": f"{y}%",
                        "opacity": opacity
                    }
                )
            )
        
        return matrix_chars

    @app.callback(
        Output('splash-container', 'style'),
        [Input('enter-button', 'n_clicks'),
         Input('press-key-message', 'n_clicks')],
        [State('boot-sequence-data', 'data')]
    )
    def hide_splash_screen(btn_clicks, msg_clicks, data_json):
        data = json.loads(data_json)
        completed = data['completed']
        
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update
            
        if completed and (btn_clicks > 0 or msg_clicks > 0):
            # Masquer l'écran d'intro
            return {"display": "none"}
            
        return dash.no_update

    # Code JavaScript pour détecter l'appui sur une touche
    app.clientside_callback(
        """
        function(completed) {
            if (completed === "true") {
                // Ajouter un événement pour détecter l'appui sur une touche
                document.addEventListener('keydown', function() {
                    document.getElementById('enter-button').click();
                }, {once: true});
                
                // Ajouter un événement pour détecter les clics
                document.getElementById('splash-container').addEventListener('click', function() {
                    document.getElementById('enter-button').click();
                }, {once: true});
                
                // Afficher le bouton après un délai
                setTimeout(function() {
                    document.getElementById('enter-button').style.display = 'block';
                }, 2000);
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('enter-button', 'data-completed'),
        [Input('boot-sequence-data', 'data')]
    )

# Fonction pour intégrer l'écran de démarrage dans n'importe quelle application Dash
def add_splash_screen_to_app(app):
    # Obtenir le layout original
    original_layout = app.layout
    
    # Créer l'écran de démarrage
    splash_screen = create_splash_screen()
    
    # Créer un nouveau layout contenant l'écran de démarrage et le layout original
    app.layout = html.Div([
        splash_screen,
        html.Div(id="app-content", children=original_layout)
    ])
    
    # Enregistrer les callbacks
    register_splash_callbacks(app)
    
    # Gérer la visibilité du contenu de l'application
    @app.callback(
        Output('app-content', 'style'),
        [Input('splash-container', 'style')]
    )
    def show_app_content(splash_style):
        if splash_style and splash_style.get('display') == 'none':
            return {"display": "block"}
        return {"display": "none"}

# Exemple d'utilisation
if __name__ == "__main__":
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    
    # Layout de base de l'application
    app.layout = html.Div("Contenu principal de l'application")
    
    # Ajouter l'écran de démarrage
    add_splash_screen_to_app(app)
    
    app.run_server(debug=True)