from dash import html, dcc, Input, Output, State, callback
import dash
import config
import json
from logger.logger import LoggerType
from logger.logger import LoggerType

def create_intro_page():
    """
    Crée l'écran d'introduction animé avec l'effet de terminal
    
    Returns:
        L'élément HTML de l'écran d'introduction
    """
    return html.Div(
        id="intro-container",
        className="crt",
        style={
            "height": "100vh",
            "width": "100vw",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "backgroundColor": "#000",
            "position": "fixed",
            "top": 0,
            "left": 0,
            "zIndex": 1000
        },
        children=[
            html.Div(
                className="terminal-window",
                style={
                    "width": "80%",
                    "height": "80%",
                    "display": "flex",
                    "flexDirection": "column",
                    "position": "relative"
                },
                children=[
                    # Logo d'en-tête
                    html.Div(
                        style={
                            "position": "absolute",
                            "top": "10px",
                            "right": "15px",
                            "textAlign": "right",
                            "zIndex": 10
                        },
                        children=[
                            html.H2("TRADING NEXUS", className="retro-title", style={"marginBottom": "5px"}),
                            html.P(f"V{config.APP_VERSION}", style={"color": "#666", "fontSize": "14px", "margin": 0})
                        ]
                    ),
                    # Contenu du terminal
                    html.Div(
                        id="terminal-content",
                        style={
                            "flex": 1,
                            "overflow": "hidden",
                            "paddingTop": "40px"
                        },
                        children=[
                            # Les lignes seront ajoutées dynamiquement
                        ]
                    ),
                    # Message "Appuyez sur une touche"
                    html.Div(
                        id="press-key-message",
                        style={"textAlign": "center", "marginTop": "auto", "marginBottom": "20px", "display": "none"},
                        children=[
                            html.Span("APPUYEZ SUR UNE TOUCHE POUR CONTINUER", className="blink-text", style={"fontSize": "24px", "fontWeight": "bold"})
                        ]
                    )
                ]
            ),
            # Bouton invisible pour capturer le clic
            html.Button(
                id="intro-click-target",
                style={"display": "none"},
                n_clicks=0
            ),
            # Div pour les logs ASCII art
            html.Div(id="ascii-art-container", style={"display": "none"})
        ]
    )

def register_intro_callbacks(app, central_logger):
    """
    Enregistre les callbacks pour l'écran d'introduction
    
    Args:
        app: L'instance de l'application Dash
        central_logger: Instance du logger centralisé
    """
    ui_logger = central_logger.get_logger("splash_screen", LoggerType.UI)
    
    # Callback pour l'animation au démarrage
    @app.callback(
        [Output('terminal-content', 'children'),
         Output('animation-state', 'data'),
         Output('press-key-message', 'style'),
         Output('interval-animation', 'disabled')],
        [Input('interval-animation', 'n_intervals')],
        [State('animation-state', 'data')]
    )
    def update_animation(n_intervals, animation_state_json):
        ui_logger.debug(f"Animation update: interval {n_intervals}, state: {animation_state_json}")
        
        # Convertir la chaîne JSON en dictionnaire
        if isinstance(animation_state_json, str):
            try:
                animation_state = json.loads(animation_state_json)
            except json.JSONDecodeError:
                animation_state = {'current_line': 0, 'animation_done': False}
        else:
            animation_state = animation_state_json or {'current_line': 0, 'animation_done': False}
        
        if animation_state.get('animation_done', False):
            return dash.no_update, animation_state_json, dash.no_update, True
        
        current_line = animation_state.get('current_line', 0)
        
        if current_line < len(config.BOOT_SEQUENCE):
            # Ajouter la ligne suivante
            line_content = config.BOOT_SEQUENCE[current_line]
            
            # Log l'action
            ui_logger.info(f"Animation: {line_content}")
            
            # Déterminer la couleur en fonction du contenu
            style = {}
            if "[ OK ]" in line_content:
                style = {"color": "#0F0"}
            elif "INITIALIZING" in line_content or "LOADING" in line_content or "CHECKING" in line_content:
                style = {"color": "#0FF"}
            elif "READY" in line_content:
                style = {"color": "#FF0"}
            
            # Mettre à jour la liste des lignes
            terminal_lines = []
            
            # Obtenir les lignes existantes du terminal
            if n_intervals > 0:
                # Dans une application réelle, il faudrait récupérer le state actuel du terminal
                terminal_lines = [
                    html.Div(config.BOOT_SEQUENCE[i], style={"marginBottom": "8px"}) 
                    for i in range(current_line)
                ]
            
            # Ajouter la nouvelle ligne
            terminal_lines.append(html.Div(line_content, style={"marginBottom": "8px", **style}))
            
            # Ajouter le prompt de terminal à la fin
            if current_line == len(config.BOOT_SEQUENCE) - 1:
                terminal_lines.append(
                    html.Div([
                        "$ ",
                        html.Span("█", className="cursor")
                    ], style={"marginTop": "10px"})
                )
                
                # Montrer le message "Appuyez sur une touche"
                press_key_style = {"textAlign": "center", "marginTop": "auto", "marginBottom": "20px"}
                # Mettre à jour l'état d'animation
                new_animation_state = {
                    'current_line': current_line + 1,
                    'animation_done': True
                }
                new_animation_state_json = json.dumps(new_animation_state)
                
                # Désactiver l'intervalle
                disable_interval = True
                
                ui_logger.info("Animation terminée, affichage du message d'appui sur une touche")
                return terminal_lines, new_animation_state_json, press_key_style, disable_interval
            
            # Mise à jour de l'état pour la prochaine ligne
            new_animation_state = {
                'current_line': current_line + 1,
                'animation_done': False
            }
            new_animation_state_json = json.dumps(new_animation_state)
            
            return terminal_lines, new_animation_state_json, {"display": "none"}, False
        
        # Si toutes les lignes ont été affichées
        return dash.no_update, animation_state, dash.no_update, True

    # Callback pour passer de l'intro au dashboard
    @app.callback(
        [Output('intro-container', 'style'),
         Output('dashboard-layout', 'style')],
        [Input('intro-click-target', 'n_clicks')],
        [State('animation-state', 'data')]
    )
    def transition_to_dashboard(n_clicks, animation_state_json):
        if n_clicks > 0 and animation_state_json:
            try:
                # Si c'est une chaîne JSON, la parser
                if isinstance(animation_state_json, str):
                    animation_state = json.loads(animation_state_json)
                else:
                    animation_state = animation_state_json
                
                if animation_state.get('animation_done', False):
                    ui_logger.info("Transition vers le tableau de bord")
                    return {"display": "none"}, {"display": "block"}
            except (json.JSONDecodeError, TypeError) as e:
                ui_logger.error(f"Erreur lors du décodage de l'état d'animation: {e}")
        
        return dash.no_update, dash.no_update

    # Capture des clics sur l'interface d'introduction
    app.clientside_callback(
        """
        function(n_intervals, animation_state_json) {
            if (animation_state_json) {
                try {
                    // Parser la chaîne JSON
                    var animation_state = JSON.parse(animation_state_json);
                    
                    if (animation_state && animation_state.animation_done === true) {
                        // Listen for key press or click
                        document.addEventListener('keydown', function() {
                            var target = document.getElementById('intro-click-target');
                            if (target) {
                                target.click();
                            }
                        }, {once: true});
                        
                        document.addEventListener('click', function() {
                            var target = document.getElementById('intro-click-target');
                            if (target) {
                                target.click();
                            }
                        }, {once: true});
                    }
                } catch (e) {
                    console.error("Erreur lors du parsing de l'état d'animation:", e);
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('ascii-art-container', 'children'),
        [Input('interval-animation', 'n_intervals'),
         Input('animation-state', 'data')]
    )