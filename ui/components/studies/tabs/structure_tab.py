"""
Onglet Structure de Stratégie du créateur d'étude avancé avec améliorations d'UX.
Utilise les composants avec les mêmes noms qu'avant pour la compatibilité.
"""
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import dash

from simulator.study_config_definitions import (
    STRATEGY_STRUCTURE_PARAMS, INDICATOR_CATEGORIES, INDICATOR_DEFAULTS
)
from ui.components.retro_ui import create_retro_range_slider, create_retro_toggle_button, create_collapsible_card

def create_structure_tab():
    """
    Crée l'onglet Structure de Stratégie avec la nouvelle interface retro améliorée.
    
    Returns:
        Contenu de l'onglet structure de stratégie
    """
    return html.Div([
        html.P("Configurez les paramètres de structure de stratégie pour l'optimisation", className="text-muted mb-4"),
        
        # Structure de stratégie
        create_strategy_structure_section(),
        
        # Configuration des indicateurs - directement visible sans switch
        create_indicators_section()
    ])

def create_strategy_structure_section():
    """
    Crée la section de structure de stratégie.
    
    Returns:
        Composant pour la section de structure de stratégie
    """
    # Création d'une grille avec les sliders
    grid_content = html.Div(
        className="retro-grid",
        children=[
            # Nombre de blocs
            create_retro_range_slider(
                id_prefix="blocks-count",
                label="Nombre de Blocs",
                min_val=1,
                max_val=10,
                step=1,
                current_min=STRATEGY_STRUCTURE_PARAMS["blocks"]["min_blocks"]["value"],
                current_max=STRATEGY_STRUCTURE_PARAMS["blocks"]["max_blocks"]["value"]
            ),
            
            # Conditions par bloc
            create_retro_range_slider(
                id_prefix="conditions-per-block",
                label="Conditions par Bloc",
                min_val=1,
                max_val=10,
                step=1,
                current_min=STRATEGY_STRUCTURE_PARAMS["conditions"]["min_conditions"]["value"],
                current_max=STRATEGY_STRUCTURE_PARAMS["conditions"]["max_conditions"]["value"]
            ),
            
            # Probabilité de croisement
            create_retro_range_slider(
                id_prefix="cross-probability",
                label="Probabilité de Croisement",
                min_val=0,
                max_val=1,
                step=0.05,
                current_min=0,
                current_max=STRATEGY_STRUCTURE_PARAMS["probabilities"]["cross_probability"]["value"]
            ),
            
            # Probabilité de comparaison de valeur
            create_retro_range_slider(
                id_prefix="value-comparison-probability",
                label="Probabilité Comparaison Valeur",
                min_val=0,
                max_val=1,
                step=0.05,
                current_min=0,
                current_max=STRATEGY_STRUCTURE_PARAMS["probabilities"]["value_comparison_probability"]["value"]
            ),
        ]
    )
    
    return create_collapsible_card(
        title="Structure de Stratégie",
        content=grid_content,
        id_prefix="strategy-structure-card",
        is_open=True
    )

def create_indicators_section():
    """
    Crée la section des indicateurs avec switches individuels.
    
    Returns:
        Composant pour la section des indicateurs
    """
    # Générer les contrôles pour chaque catégorie d'indicateurs
    indicator_categories = []
    
    for category_key, category_data in INDICATOR_CATEGORIES.items():
        category_indicators = []
        
        for ind_name in category_data['indicators']:
            if ind_name in INDICATOR_DEFAULTS:
                ind_config = INDICATOR_DEFAULTS[ind_name]
                
                # Convert enum to string for component ID
                indicator_id_name = ind_name.value if hasattr(ind_name, 'value') else str(ind_name)
                
                # Créer un switch pour activer/désactiver l'indicateur
                indicator_header = dbc.Row([
                    dbc.Col(html.H6(ind_config['description'], className="mb-0"), width=9),
                    dbc.Col(
                        dbc.Switch(
                            id={"type": "indicator-switch", "name": indicator_id_name},
                            value=True,  # Par défaut, l'indicateur est activé
                            className="float-end"
                        ),
                        width=3,
                        className="text-end"
                    )
                ])
                
                # Créer les contrôles pour cet indicateur avec champs éditables
                indicator_card = dbc.Card([
                    dbc.CardHeader(indicator_header),
                    dbc.CardBody([
                        # Collapse qui sera contrôlé par le switch
                        dbc.Collapse(
                            # Paramètres de l'indicateur
                            html.Div([
                                # Période - maintenant avec champs d'édition manuels
                                html.Div([
                                    html.Label("Périodes (min-max)", className="mb-2"),
                                    dbc.Row([
                                        dbc.Col(
                                            dbc.Input(
                                                id={"type": "indicator-min-period", "name": indicator_id_name},
                                                type="number",
                                                min=1,
                                                max=500,
                                                step=1,
                                                value=ind_config['min_period'],
                                                className="highlighted-input"
                                            ),
                                            width=5
                                        ),
                                        dbc.Col(
                                            html.Span("à", className="text-center d-block pt-2"),
                                            width=2,
                                            className="text-center"
                                        ),
                                        dbc.Col(
                                            dbc.Input(
                                                id={"type": "indicator-max-period", "name": indicator_id_name},
                                                type="number",
                                                min=1,
                                                max=500,
                                                step=1,
                                                value=ind_config['max_period'],
                                                className="highlighted-input"
                                            ),
                                            width=5
                                        )
                                    ], className="mb-3")
                                ]),
                                
                                # Options supplémentaires
                                dbc.Row([
                                    # Pas pour l'optimisation
                                    dbc.Col([
                                        html.Label("Pas d'optimisation", className="mb-2"),
                                        dbc.Input(
                                            id={"type": "indicator-step", "name": indicator_id_name},
                                            type="number",
                                            min=1,
                                            max=50,
                                            step=1,
                                            value=ind_config['step'],
                                            className="highlighted-input"
                                        )
                                    ], width=6),
                                    
                                    # Type de prix
                                    dbc.Col([
                                        html.Label("Type de prix", className="mb-2"),
                                        dcc.Dropdown(
                                            id={"type": "indicator-price-type", "name": indicator_id_name},
                                            options=[
                                                {"label": "Clôture", "value": "close"},
                                                {"label": "Ouverture", "value": "open"},
                                                {"label": "Haut", "value": "high"},
                                                {"label": "Bas", "value": "low"},
                                                {"label": "Médian", "value": "median"},
                                                {"label": "Typique", "value": "typical"}
                                            ],
                                            value=ind_config['price_type'],
                                            clearable=False,
                                            className="retro-dropdown"
                                        )
                                    ], width=6)
                                ]),
                            ]),
                            id={"type": "indicator-collapse", "name": indicator_id_name},
                            is_open=True  # Par défaut, les paramètres sont visibles
                        )
                    ]),
                ], className="mb-3")
                
                category_indicators.append(indicator_card)
        
        if category_indicators:
            # Créer une carte pour la catégorie
            category_card = create_collapsible_card(
                title=category_data['label'],
                content=html.Div(category_indicators),
                id_prefix=f"indicator-category-{category_key}",
                is_open=category_key == "trend"  # Par défaut, seule la première catégorie est ouverte
            )
            
            indicator_categories.append(category_card)
    
    return html.Div([
        html.H5("Configuration des Indicateurs", className="mb-3 text-cyan-300"),
        html.Div(indicator_categories, className="mt-4")
    ])

def register_structure_callbacks(app):
    """
    Enregistre les callbacks spécifiques à l'onglet Structure de Stratégie
    
    Args:
        app: L'instance de l'application Dash
    """
    # Callback pour contrôler chaque indicateur individuellement
    @app.callback(
        Output({"type": "indicator-collapse", "name": dash.MATCH}, "is_open"),
        [Input({"type": "indicator-switch", "name": dash.MATCH}, "value")]
    )
    def toggle_indicator_params(is_active):
        """Affiche ou masque les paramètres d'un indicateur spécifique"""
        return is_active

    # Callback pour valider les périodes des indicateurs
    @app.callback(
        [
            Output({"type": "indicator-min-period", "name": dash.MATCH}, "className"),
            Output({"type": "indicator-max-period", "name": dash.MATCH}, "className")
        ],
        [
            Input({"type": "indicator-min-period", "name": dash.MATCH}, "value"),
            Input({"type": "indicator-max-period", "name": dash.MATCH}, "value")
        ]
    )
    def validate_indicator_periods(min_period, max_period):
        """Valide que la période minimale est inférieure à la période maximale"""
        ctx = callback_context
        
        # Style par défaut
        min_style = "highlighted-input"
        max_style = "highlighted-input"
        
        # Si les deux valeurs sont définies, vérifier que min < max
        if min_period is not None and max_period is not None:
            if min_period > max_period:
                # Déterminer quel champ a été modifié
                if ctx.triggered and ctx.triggered[0]["prop_id"].split(".")[0].endswith("min-period"):
                    min_style = "highlighted-input is-invalid"
                else:
                    max_style = "highlighted-input is-invalid"
        
        return min_style, max_style

    # Callback pour mettre à jour les classes des dropdowns pour améliorer l'UX
    app.clientside_callback(
        """
        function() {
            // Amélioration des dropdowns: rendre la zone cliquable plus large
            setTimeout(function() {
                const dropdowns = document.querySelectorAll('.retro-dropdown .Select');
                dropdowns.forEach(dropdown => {
                    // Ajouter une classe pour la surcharge CSS
                    dropdown.classList.add('retro-dropdown-improved');
                    
                    // Rendre toute la zone cliquable
                    const dropdownControl = dropdown.querySelector('.Select-control');
                    if (dropdownControl) {
                        dropdownControl.style.cursor = 'pointer';
                        dropdownControl.addEventListener('click', function(e) {
                            // Forcer l'ouverture/fermeture du menu
                            const arrow = this.querySelector('.Select-arrow');
                            if (arrow) {
                                arrow.click();
                            }
                        });
                    }
                });
            }, 100);
            
            return null;
        }
        """,
        Output("_", "children", allow_duplicate=True),
        [Input("study-creator-tabs", "active_tab")],
        prevent_initial_call=True
    )