"""
Composants d'interface utilisateur rétro réutilisables pour l'application de trading.
Inclut les range sliders avec champs éditables et synchronisation bidirectionnelle.
"""
from dash import html, dcc, Input, Output, State, callback, ALL, MATCH
import dash
import json

def create_retro_range_slider(id_prefix, label, min_val, max_val, step,
                            current_min, current_max, unit="", default_value=None):
    """
    Crée un slider de plage à double poignée avec style retro et champs éditables.
    
    Args:
        id_prefix: Préfixe pour les IDs des composants
        label: Label du slider
        min_val: Valeur minimale autorisée
        max_val: Valeur maximale autorisée
        step: Pas du slider
        current_min: Valeur minimum actuelle
        current_max: Valeur maximum actuelle
        unit: Unité à afficher (%, $, etc.)
        default_value: Valeur par défaut sous forme de texte (ex: "1.0-10.0") 
    
    Returns:
        Composant Dash pour le slider de plage
    """
    if default_value:
        try:
            parts = default_value.split('-')
            if len(parts) == 2:
                current_min = float(parts[0])
                current_max = float(parts[1])
        except:
            pass
            
    # Générer des repères équidistants (5 points)
    marks = {}
    for i in range(5):
        value = min_val + i * (max_val - min_val) / 4
        if isinstance(value, int) or value.is_integer():
            marks[value] = f"{int(value)}"
        else:
            marks[value] = f"{value:.1f}"
    
    # Formater les valeurs pour l'affichage
    if isinstance(current_min, int):
        display_min = f"{current_min}"
        display_max = f"{current_max}"
    else:
        display_min = f"{current_min:.1f}" if current_min % 1 else f"{int(current_min)}"
        display_max = f"{current_max:.1f}" if current_max % 1 else f"{int(current_max)}"
    
    return html.Div(
        className="retro-range-container",
        children=[
            # En-tête avec label
            html.Div(
                className="retro-range-header",
                children=[
                    html.Div(label, className="retro-range-label"),
                ]
            ),
            
            # Champs de saisie numérique pour min/max avec min et max définis
            html.Div(
                className="retro-range-values mb-2",
                children=[
                    html.Div([
                        dcc.Input(
                            id={"type": "range-min-input", "id": id_prefix},
                            type="number",
                            min=min_val,
                            max=max_val,
                            step=step,
                            value=current_min,
                            className="highlighted-input no-spinner",
                            style={"width": "80px", "textAlign": "center"}
                        ),
                        html.Span(" " + unit if unit else "", className="ms-1 me-2"),
                        html.Span("à", className="mx-2"),
                        dcc.Input(
                            id={"type": "range-max-input", "id": id_prefix},
                            type="number",
                            min=min_val,
                            max=max_val,
                            step=step,
                            value=current_max,
                            className="highlighted-input no-spinner",
                            style={"width": "80px", "textAlign": "center"}
                        ),
                        html.Span(" " + unit if unit else "", className="ms-1")
                    ], className="d-flex align-items-center mx-auto", style={"width": "fit-content"})
                ]
            ),
            
            # Input caché pour stocker la valeur sous forme de chaîne
            dcc.Input(
                id={"type": "range-input", "id": id_prefix},
                type="hidden",
                value=f"{current_min}-{current_max}"
            ),
            
            # Le RangeSlider avec style retro
            html.Div(
                className="retro-range-wrapper",
                children=[
                    dcc.RangeSlider(
                        id={"type": "range-slider", "id": id_prefix},
                        min=min_val,
                        max=max_val,
                        step=step,
                        value=[current_min, current_max],
                        marks=marks,
                        className="retro-range-slider",
                        tooltip={"always_visible": False, "placement": "bottom"}
                    )
                ]
            ),
            
            # Stocker les limites min/max dans des attributs data
            html.Div(
                id={"type": "range-limits", "id": id_prefix},
                style={"display": "none"},
                **{
                    "data-min": str(min_val),
                    "data-max": str(max_val),
                    "data-step": str(step)
                }
            )
        ]
    )

def create_retro_toggle_button(id_prefix, label, value, index=None, is_active=False):
    """
    Crée un bouton toggle avec style retro.
    
    Args:
        id_prefix: Préfixe pour l'ID du composant
        label: Label affiché sur le bouton
        value: Valeur associée au bouton
        index: Index optionnel pour le bouton (pour l'affichage visuel)
        is_active: Si le bouton est actif par défaut
        
    Returns:
        Composant Dash pour un bouton toggle
    """
    return html.Div(
        id={"type": f"{id_prefix}-toggle", "index": index or value},
        className=f"retro-toggle{'  active' if is_active else ''}",
        children=[
            html.Span(f"{index or '•'}", className="retro-badge retro-badge-blue"),
            f" {label}"
        ],
        n_clicks=0,
        # Attribut personnalisé pour stocker l'état actif
        **{'data-active': 'true' if is_active else 'false', 'data-value': value}
    )

def create_collapsible_card(title, content, id_prefix, is_open=True):
    """
    Crée une carte rétractable/dépliable avec style retro.
    
    Args:
        title: Titre de la carte
        content: Contenu à afficher dans la carte
        id_prefix: Préfixe pour les IDs des composants
        is_open: Si la carte est dépliée par défaut
        
    Returns:
        Composant Dash représentant une carte dépliable
    """
    return html.Div(
        className="retro-card mb-4",
        children=[
            # En-tête de carte avec bouton de bascule (l'ensemble de l'en-tête est cliquable)
            html.Div(
                id={"type": "card-header-toggle", "id": id_prefix},
                className="retro-card-header",
                style={"cursor": "pointer"},  # Curseur pointeur pour indiquer que c'est cliquable
                children=[
                    html.H3(title, className="retro-card-title"),
                    html.Span(
                        "▼" if is_open else "▶", 
                        id={"type": "card-toggle-icon", "id": id_prefix},
                        className="section-toggle"
                    )
                ]
            ),
            # Corps de la carte qui peut se replier
            html.Div(
                id={"type": "card-body", "id": id_prefix},
                className=f"retro-card-body collapsible-section{' open' if is_open else ''}",
                children=content
            )
        ],
        style={"--card-delay": f"{hash(id_prefix) % 10 / 10}"}  # Décalage d'animation aléatoire
    )

def register_retro_ui_callbacks(app):
    """
    Enregistre les callbacks pour les composants rétro
    
    Args:
        app: L'instance de l'application Dash
    """
    # Remplacer le callback client-side par des callbacks serveur pour plus de fiabilité
    
    # Callback pour synchroniser les inputs et le slider (slider -> inputs)
    @app.callback(
        [
            Output({"type": "range-min-input", "id": MATCH}, "value"),
            Output({"type": "range-max-input", "id": MATCH}, "value"),
            Output({"type": "range-input", "id": MATCH}, "value")
        ],
        [Input({"type": "range-slider", "id": MATCH}, "value")],
        prevent_initial_call=True
    )
    def update_inputs_from_slider(slider_values):
        if not slider_values:
            return dash.no_update, dash.no_update, dash.no_update
        
        min_val = slider_values[0]
        max_val = slider_values[1]
        range_string = f"{min_val}-{max_val}"
        
        return min_val, max_val, range_string
    
    # Callback pour synchroniser les inputs et le slider (min input -> slider)
    @app.callback(
        [
            Output({"type": "range-slider", "id": MATCH}, "value", allow_duplicate=True),
            Output({"type": "range-input", "id": MATCH}, "value", allow_duplicate=True)
        ],
        [Input({"type": "range-min-input", "id": MATCH}, "value")],
        [
            State({"type": "range-slider", "id": MATCH}, "value"),
            State({"type": "range-limits", "id": MATCH}, "data-min"),
            State({"type": "range-limits", "id": MATCH}, "data-max")
        ],
        prevent_initial_call=True
    )
    def update_slider_from_min_input(min_input, current_values, min_limit, max_limit):
        if min_input is None or current_values is None:
            return dash.no_update, dash.no_update
        
        # Valeurs actuelles du slider
        current_min = current_values[0]
        current_max = current_values[1]
        
        # Limites du slider
        min_limit = float(min_limit) if min_limit else 0
        max_limit = float(max_limit) if max_limit else 100
        
        # Limiter la valeur min_input aux bornes
        min_input = max(min_limit, min(max_limit, min_input))
        
        # S'assurer que min <= max
        if min_input > current_max:
            min_input = current_max
        
        # Mettre à jour le slider avec la nouvelle valeur min
        new_values = [min_input, current_max]
        
        # Mettre à jour la chaîne de range
        range_string = f"{min_input}-{current_max}"
        
        return new_values, range_string
    
    # Callback pour synchroniser les inputs et le slider (max input -> slider)
    @app.callback(
        [
            Output({"type": "range-slider", "id": MATCH}, "value", allow_duplicate=True),
            Output({"type": "range-input", "id": MATCH}, "value", allow_duplicate=True)
        ],
        [Input({"type": "range-max-input", "id": MATCH}, "value")],
        [
            State({"type": "range-slider", "id": MATCH}, "value"),
            State({"type": "range-limits", "id": MATCH}, "data-min"),
            State({"type": "range-limits", "id": MATCH}, "data-max")
        ],
        prevent_initial_call=True
    )
    def update_slider_from_max_input(max_input, current_values, min_limit, max_limit):
        if max_input is None or current_values is None:
            return dash.no_update, dash.no_update
        
        # Valeurs actuelles du slider
        current_min = current_values[0]
        current_max = current_values[1]
        
        # Limites du slider
        min_limit = float(min_limit) if min_limit else 0
        max_limit = float(max_limit) if max_limit else 100
        
        # Limiter la valeur max_input aux bornes
        max_input = max(min_limit, min(max_limit, max_input))
        
        # S'assurer que max >= min
        if max_input < current_min:
            max_input = current_min
        
        # Mettre à jour le slider avec la nouvelle valeur max
        new_values = [current_min, max_input]
        
        # Mettre à jour la chaîne de range
        range_string = f"{current_min}-{max_input}"
        
        return new_values, range_string
    
    # Callback pour basculer les sections dépliables quand on clique sur l'en-tête
    @app.callback(
        [
            Output({"type": "card-body", "id": MATCH}, "className"),
            Output({"type": "card-toggle-icon", "id": MATCH}, "children")
        ],
        [Input({"type": "card-header-toggle", "id": MATCH}, "n_clicks")],
        [State({"type": "card-body", "id": MATCH}, "className")],
        prevent_initial_call=True
    )
    def toggle_card_section(n_clicks, current_class):
        """Bascule l'état ouvert/fermé d'une section"""
        if not n_clicks:
            return dash.no_update, dash.no_update
            
        is_open = "open" in current_class
        new_class = "retro-card-body collapsible-section" + ("" if is_open else " open")
        new_icon = "▼" if not is_open else "▶"
        
        return new_class, new_icon
    
    # Callback pour basculer les boutons de mode
    @app.callback(
        [
            Output({"type": MATCH, "index": MATCH}, "className"),
            Output({"type": MATCH, "index": MATCH}, "data-active")
        ],
        [Input({"type": MATCH, "index": MATCH}, "n_clicks")],
        [State({"type": MATCH, "index": MATCH}, "className"),
         State({"type": MATCH, "index": MATCH}, "data-active")],
        prevent_initial_call=True
    )
    def toggle_button(n_clicks, current_class, is_active):
        """Bascule l'état actif/inactif d'un bouton toggle"""
        if not n_clicks:
            return dash.no_update, dash.no_update
            
        is_active_bool = is_active == "true"
        new_active = not is_active_bool
        new_class = "retro-toggle" + (" active" if new_active else "")
        
        return new_class, str(new_active).lower()