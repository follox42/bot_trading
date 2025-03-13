"""
Module d'affichage des détails d'optimisation avec un design amélioré.
Fournit une interface claire et fonctionnelle pour visualiser les résultats d'optimisation.
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
from datetime import datetime

from logger.logger import LoggerType
from simulator.study_manager import IntegratedStudyManager
from simulator.study_config_definitions import (
    OPTIMIZATION_METHODS, 
    PRUNER_METHODS, 
    SCORING_FORMULAS, 
    AVAILABLE_METRICS
)

def create_optimization_details(study_name):
    """
    Crée le contenu des détails d'optimisation avec une interface améliorée.
    
    Args:
        study_name: Nom de l'étude
    
    Returns:
        Composant d'affichage des détails d'optimisation
    """
    try:
        # Initialiser le gestionnaire d'études
        study_manager = IntegratedStudyManager("studies")
        
        # Vérifier si l'étude existe
        if not study_manager.study_exists(study_name):
            return create_error_message(f"L'étude '{study_name}' n'existe pas.")
        
        # Récupérer les résultats d'optimisation
        optimization_results = study_manager.get_optimization_results(study_name)
        
        if not optimization_results:
            return create_error_message("Aucun résultat d'optimisation disponible pour cette étude.")
        
        # Extraire les informations
        best_trials = optimization_results.get('best_trials', [])
        best_trial_id = optimization_results.get('best_trial_id', -1)
        n_trials = optimization_results.get('n_trials', 0)
        optimization_date = optimization_results.get('optimization_date', 'Non disponible')
        optimization_config = optimization_results.get('optimization_config', {})
        
        # Vérifier si les données sont suffisantes
        if not best_trials:
            return create_error_message("Aucun essai disponible dans les résultats d'optimisation.")
        
        # Créer la mise en page
        return html.Div([
            # En-tête
            create_header_section(study_name, optimization_date, n_trials),
            
            # Conteneur principal avec deux colonnes
            dbc.Row([
                # Colonne 1: Information générale et meilleures stratégies
                dbc.Col([
                    # Résumé d'optimisation
                    create_optimization_summary(optimization_config, best_trial_id, n_trials),
                    
                    # Meilleurs essais
                    create_best_trials_table(best_trials, best_trial_id),
                ], lg=5, md=12, className="mb-4"),
                
                # Colonne 2: Visualisations et détail du meilleur essai
                dbc.Col([
                    # Visualisations
                    create_performance_visualizations(best_trials, best_trial_id),
                    
                    # Détails du meilleur essai
                    create_best_trial_details(best_trials, best_trial_id),
                ], lg=7, md=12),
            ]),
            
            # Section inférieure: Détails avancés avec onglets
            html.Div([
                dbc.Tabs([
                    # Onglet Paramètres
                    dbc.Tab(
                        create_parameters_tab(best_trials, best_trial_id),
                        label="Paramètres",
                        tab_id="tab-parameters",
                        className="mt-3 p-3"
                    ),
                    
                    # Onglet Analyse
                    dbc.Tab(
                        create_analysis_tab(best_trials),
                        label="Analyse",
                        tab_id="tab-analysis",
                        className="mt-3 p-3"
                    ),
                    
                    # Onglet Actions
                    dbc.Tab(
                        create_actions_tab(study_name),
                        label="Actions",
                        tab_id="tab-actions",
                        className="mt-3 p-3"
                    ),
                ], className="retro-tabs mt-4"),
            ]),
            
            # Pied de page avec informations techniques
            create_footer_section(study_name, optimization_results),
        ], className="optimization-details-container")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return create_error_message(f"Erreur lors du chargement des détails d'optimisation: {str(e)}")

def create_error_message(message):
    """
    Crée un message d'erreur stylisé.
    
    Args:
        message: Message d'erreur
    
    Returns:
        Composant d'affichage du message d'erreur
    """
    return html.Div([
        html.Div([
            html.I(className="bi bi-exclamation-triangle-fill text-warning me-3", style={"fontSize": "2rem"}),
            html.Div([
                html.H4("Impossible d'afficher les détails", className="text-warning mb-2"),
                html.P(message, className="mb-0 text-light")
            ])
        ], className="d-flex align-items-center"),
        
        html.Div([
            dbc.Button(
                [html.I(className="bi bi-arrow-left me-2"), "Retour aux études"],
                id="btn-back-to-studies",
                className="retro-button mt-3"
            )
        ], className="mt-4")
    ], className="retro-card p-4 mt-3 border border-warning bg-dark")

def create_header_section(study_name, optimization_date, n_trials):
    """
    Crée la section d'en-tête avec le titre et les informations de base.
    
    Args:
        study_name: Nom de l'étude
        optimization_date: Date d'optimisation
        n_trials: Nombre d'essais
    
    Returns:
        Composant d'en-tête
    """
    return html.Div([
        dbc.Row([
            # Informations de l'étude
            dbc.Col([
                html.H3([
                    html.I(className="bi bi-bar-chart-line-fill text-cyan-300 me-3"),
                    "Résultats d'optimisation"
                ], className="text-cyan-300 mb-3"),
                
                html.H4([
                    f"Étude: ",
                    html.Span(study_name, className="text-white fw-bold")
                ], className="text-light mb-4"),
            ], md=8),
            
            # Statistiques
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("Date: ", className="text-cyan-100"),
                        html.Span(optimization_date, className="text-white")
                    ], className="mb-2"),
                    
                    html.Div([
                        html.Span("Essais: ", className="text-cyan-100"),
                        html.Span(f"{n_trials}", className="text-white")
                    ], className="mb-2"),
                ], className="d-flex flex-column justify-content-center h-100 text-end")
            ], md=4),
        ]),
        
        # Barre de séparation
        html.Hr(className="border-secondary mb-4"),
    ])

def create_optimization_summary(optimization_config, best_trial_id, n_trials):
    """
    Crée un résumé de la configuration d'optimisation.
    
    Args:
        optimization_config: Configuration d'optimisation
        best_trial_id: ID du meilleur essai
        n_trials: Nombre d'essais
    
    Returns:
        Carte résumant la configuration
    """
    # Récupérer les informations sur la méthode d'optimisation
    method_name = optimization_config.get('method', 'tpe')
    method_info = OPTIMIZATION_METHODS.get(method_name, OPTIMIZATION_METHODS['tpe'])
    
    # Récupérer les informations sur la formule de scoring
    scoring_name = optimization_config.get('scoring_formula', 'standard')
    scoring_info = SCORING_FORMULAS.get(scoring_name, SCORING_FORMULAS['standard'])
    
    # Récupérer les informations sur le pruning
    pruner_name = optimization_config.get('pruner_method', 'none')
    if pruner_name and pruner_name != 'none':
        pruner_info = PRUNER_METHODS.get(pruner_name, {'name': 'Non spécifié'})
        pruner_display = pruner_info.get('name', pruner_name)
    else:
        pruner_display = "Non activé"
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-gear-fill me-2"),
                "Configuration de l'optimisation"
            ], className="text-cyan-300 mb-0")
        ], className="bg-dark border-secondary"),
        
        dbc.CardBody([
            # Informations principales
            html.Div([
                html.Div([
                    html.I(className="bi bi-trophy-fill text-warning me-2"),
                    html.Span("Meilleur essai: ", className="text-cyan-100"),
                    html.Span(f"#{best_trial_id}", className="text-white fw-bold")
                ], className="mb-2"),
                
                html.Div([
                    html.I(className="bi bi-search me-2 text-cyan-300"),
                    html.Span("Méthode: ", className="text-cyan-100"),
                    html.Span(method_info['name'], className="text-white")
                ], className="mb-2"),
                
                html.Div([
                    html.I(className="bi bi-calculator me-2 text-cyan-300"),
                    html.Span("Scoring: ", className="text-cyan-100"),
                    html.Span(scoring_info['name'], className="text-white")
                ], className="mb-2"),
                
                html.Div([
                    html.I(className="bi bi-scissors me-2 text-cyan-300"),
                    html.Span("Pruning: ", className="text-cyan-100"),
                    html.Span(pruner_display, className="text-white")
                ], className="mb-0"),
            ], className="px-2 py-1"),
            
            # Description du scoring
            html.Div([
                html.P(
                    scoring_info['description'],
                    className="mt-3 mb-0 text-muted small"
                )
            ], className="bg-dark rounded p-2 mt-3"),
        ], className="bg-dark border-secondary"),
    ], className="mb-4 retro-card shadow")

def create_best_trials_table(best_trials, best_trial_id):
    """
    Crée un tableau des meilleurs essais.
    
    Args:
        best_trials: Liste des meilleurs essais
        best_trial_id: ID du meilleur essai
    
    Returns:
        Composant tableau des meilleurs essais
    """
    if not best_trials:
        return html.Div("Aucun résultat disponible", className="text-center text-muted p-3")
    
    rows = []
    for i, trial in enumerate(best_trials[:5]):  # Limiter à 5 meilleurs essais pour la lisibilité
        trial_id = trial.get('trial_id', 0)
        score = trial.get('score', 0)
        metrics = trial.get('metrics', {})
        
        roi = metrics.get('roi', 0) * 100
        win_rate = metrics.get('win_rate', 0) * 100
        max_dd = metrics.get('max_drawdown', 0) * 100
        profit_factor = metrics.get('profit_factor', 0)
        total_trades = metrics.get('total_trades', 0)
        
        # Style pour le meilleur essai
        row_class = "best-trial fw-bold" if trial_id == best_trial_id else ""
        
        row = html.Tr([
            html.Td(f"#{trial_id}", className="text-center"),
            html.Td(f"{score:.4f}", className="text-center"),
            html.Td(f"{roi:.2f}%", className=f"text-{'success' if roi > 0 else 'danger'}"),
            html.Td(f"{win_rate:.2f}%"),
            html.Td(f"{max_dd:.2f}%", className="text-danger"),
            html.Td(f"{int(total_trades)}"),
            html.Td([
                html.Button(
                    html.I(className="bi bi-eye"),
                    id={"type": "btn-view-trial", "trial_id": trial_id},
                    className="retro-button btn-sm",
                    title="Voir les détails"
                )
            ], className="text-center")
        ], className=f"trial-row {row_class}")
        
        rows.append(row)
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-stars me-2"),
                "Meilleurs essais"
            ], className="text-cyan-300 mb-0")
        ], className="bg-dark border-secondary"),
        
        dbc.CardBody([
            html.Div([
                html.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("ID", className="text-center"),
                            html.Th("Score", className="text-center"),
                            html.Th("ROI"),
                            html.Th("Win Rate"),
                            html.Th("Max DD"),
                            html.Th("Trades"),
                            html.Th("", className="text-center")
                        ], className="retro-table-header")
                    ]),
                    html.Tbody(rows)
                ], className="retro-table w-100")
            ], className="table-responsive")
        ], className="p-0 bg-dark"),
    ], className="retro-card shadow")

def create_performance_visualizations(best_trials, best_trial_id):
    """
    Crée des visualisations des performances des meilleurs essais.
    
    Args:
        best_trials: Liste des meilleurs essais
        best_trial_id: ID du meilleur essai
    
    Returns:
        Composant avec graphiques de performance
    """
    if not best_trials:
        return html.Div("Données insuffisantes pour les visualisations", className="text-center text-muted p-3")
    
    # Préparer les données
    trial_ids = []
    scores = []
    rois = []
    win_rates = []
    drawdowns = []
    
    for trial in best_trials[:8]:  # Limiter à 8 pour la lisibilité
        trial_ids.append(trial.get('trial_id', 0))
        scores.append(trial.get('score', 0))
        metrics = trial.get('metrics', {})
        rois.append(metrics.get('roi', 0) * 100)
        win_rates.append(metrics.get('win_rate', 0) * 100)
        drawdowns.append(metrics.get('max_drawdown', 0) * 100)
    
    # Créer le graphique de comparaison
    comparison_fig = go.Figure()
    
    # Barres pour les scores
    comparison_fig.add_trace(go.Bar(
        x=trial_ids,
        y=scores,
        name='Score',
        marker_color='rgb(34, 211, 238)',
        opacity=0.8
    ))
    
    # Ligne pour le ROI
    comparison_fig.add_trace(go.Scatter(
        x=trial_ids,
        y=rois,
        name='ROI (%)',
        marker_color='rgb(74, 222, 128)',
        mode='lines+markers',
        line=dict(width=2)
    ))
    
    # Mettre en évidence le meilleur essai
    if best_trial_id in trial_ids:
        best_idx = trial_ids.index(best_trial_id)
        comparison_fig.add_trace(go.Scatter(
            x=[best_trial_id],
            y=[scores[best_idx]],
            mode='markers',
            marker=dict(
                size=12,
                color='rgb(255, 255, 255)',
                symbol='star',
                line=dict(color='rgb(34, 211, 238)', width=2)
            ),
            name='Meilleur essai',
            hoverinfo='skip'
        ))
    
    # Mise en page
    comparison_fig.update_layout(
        title='Comparaison des meilleurs essais',
        xaxis_title='Trial ID',
        yaxis_title='Valeur',
        legend=dict(
            orientation="h",
            y=1.02,
            x=0.5,
            xanchor="center"
        ),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,41,59,0.5)',
        margin=dict(l=40, r=30, t=50, b=40),
        height=350
    )
    
    # Graphiques en radar pour le meilleur essai
    best_trial = None
    for trial in best_trials:
        if trial.get('trial_id') == best_trial_id:
            best_trial = trial
            break
    
    radar_fig = None
    if best_trial:
        metrics = best_trial.get('metrics', {})
        
        # Normaliser les métriques pour le radar
        radar_data = [
            min(1.0, max(0, metrics.get('roi', 0) * 5)),  # ROI normalisé (jusqu'à 20%)
            metrics.get('win_rate', 0),                   # Win rate déjà entre 0 et 1
            1 - min(1.0, metrics.get('max_drawdown', 0) * 5),  # Drawdown inversé
            min(1.0, metrics.get('profit_factor', 0) / 5),      # Profit factor normalisé
            min(1.0, metrics.get('total_trades', 0) / 100)      # Trades normalisés
        ]
        
        category_names = ['ROI', 'Win Rate', 'Min Drawdown', 'Profit Factor', 'Volume de Trades']
        
        radar_fig = go.Figure()
        
        radar_fig.add_trace(go.Scatterpolar(
            r=radar_data,
            theta=category_names,
            fill='toself',
            name='Meilleur essai',
            line_color='rgb(34, 211, 238)',
            fillcolor='rgba(34, 211, 238, 0.2)'
        ))
        
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=20, b=40),
            height=350
        )
    
    return html.Div([
        # Graphique de comparaison
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="bi bi-bar-chart-fill me-2"),
                    "Comparaison des performances"
                ], className="text-cyan-300 mb-0")
            ], className="bg-dark border-secondary"),
            
            dbc.CardBody([
                dcc.Graph(
                    figure=comparison_fig,
                    config={'displayModeBar': False},
                    className="retro-graph"
                )
            ], className="bg-dark p-0 pt-2"),
        ], className="retro-card shadow mb-4"),
        
        # Graphique radar pour le meilleur essai
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="bi bi-bullseye me-2"),
                    f"Profil du meilleur essai (#{best_trial_id})"
                ], className="text-cyan-300 mb-0")
            ], className="bg-dark border-secondary"),
            
            dbc.CardBody(
                dcc.Graph(
                    figure=radar_fig,
                    config={'displayModeBar': False},
                    className="retro-graph"
                ) if radar_fig else html.Div("Données insuffisantes", className="text-center text-muted p-3"),
                className="bg-dark p-0 pt-2"
            ),
        ], className="retro-card shadow"),
    ])

def create_best_trial_details(best_trials, best_trial_id):
    """
    Crée une section présentant les détails du meilleur essai.
    
    Args:
        best_trials: Liste des meilleurs essais
        best_trial_id: ID du meilleur essai
    
    Returns:
        Composant avec les détails du meilleur essai
    """
    best_trial = None
    for trial in best_trials:
        if trial.get('trial_id') == best_trial_id:
            best_trial = trial
            break
    
    if not best_trial:
        return html.Div("Détails du meilleur essai non disponibles", className="text-center text-muted p-3")
    
    metrics = best_trial.get('metrics', {})
    
    # Métriques à afficher
    metrics_display = [
        {"name": "ROI", "value": f"{metrics.get('roi', 0)*100:.2f}%", "color": "success" if metrics.get('roi', 0) > 0 else "danger"},
        {"name": "Win Rate", "value": f"{metrics.get('win_rate', 0)*100:.2f}%", "color": "cyan-300"},
        {"name": "Max Drawdown", "value": f"{metrics.get('max_drawdown', 0)*100:.2f}%", "color": "danger"},
        {"name": "Profit Factor", "value": f"{metrics.get('profit_factor', 0):.2f}", "color": "success" if metrics.get('profit_factor', 0) > 1 else "warning"},
        {"name": "Trades", "value": f"{int(metrics.get('total_trades', 0))}", "color": "white"},
        {"name": "Avg. Profit", "value": f"{metrics.get('avg_profit', 0)*100:.2f}%", "color": "success" if metrics.get('avg_profit', 0) > 0 else "danger"},
    ]
    
    # Créer des cartes de métrique
    metric_cards = []
    for metric in metrics_display:
        metric_cards.append(
            dbc.Col(
                html.Div([
                    html.Div(metric["name"], className="small text-muted mb-1"),
                    html.Div(metric["value"], className=f"h4 mb-0 text-{metric['color']}")
                ], className="p-3 bg-dark rounded border border-secondary text-center"),
                xs=6, md=4, className="mb-3"
            )
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-trophy-fill me-2 text-warning"),
                f"Meilleur essai (#{best_trial_id})"
            ], className="text-cyan-300 mb-0")
        ], className="bg-dark border-secondary"),
        
        dbc.CardBody([
            # Métriques
            dbc.Row(metric_cards),
            
            # Boutons d'action
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        [html.I(className="bi bi-backpack me-2"), "Tester la stratégie"],
                        id={"type": "btn-backtest-trial", "trial_id": best_trial_id},
                        className="retro-button me-2 mb-2 w-100"
                    ),
                ], xs=12, md=6),
                
                dbc.Col([
                    dbc.Button(
                        [html.I(className="bi bi-save me-2"), "Utiliser cette stratégie"],
                        id={"type": "btn-use-trial", "trial_id": best_trial_id},
                        className="retro-button btn-sm secondary w-100 mb-2"
                    ),
                ], xs=12, md=6),
            ], className="mt-2"),
        ], className="bg-dark p-3"),
    ], className="retro-card shadow mt-4")

def create_parameters_tab(best_trials, best_trial_id):
    """
    Crée l'onglet des paramètres pour le meilleur essai.
    
    Args:
        best_trials: Liste des meilleurs essais
        best_trial_id: ID du meilleur essai
    
    Returns:
        Contenu de l'onglet des paramètres
    """
    best_trial = None
    for trial in best_trials:
        if trial.get('trial_id') == best_trial_id:
            best_trial = trial
            break
    
    if not best_trial:
        return html.Div("Paramètres non disponibles", className="text-center text-muted p-3")
    
    params = best_trial.get('params', {})
    
    if not params:
        return html.Div("Aucun paramètre disponible", className="text-center text-muted p-3")
    
    # Regrouper les paramètres par catégorie
    param_categories = {
        "Blocs d'achat": [],
        "Blocs de vente": [],
        "Gestion du risque": [],
        "Simulation": [],
        "Autres": []
    }
    
    for param_name, param_value in params.items():
        if param_name.startswith("buy_"):
            param_categories["Blocs d'achat"].append((param_name, param_value))
        elif param_name.startswith("sell_"):
            param_categories["Blocs de vente"].append((param_name, param_value))
        elif param_name in ["risk_mode", "base_position", "base_sl", "tp_multiplier", 
                         "atr_period", "atr_multiplier", "vol_period", "vol_multiplier"]:
            param_categories["Gestion du risque"].append((param_name, param_value))
        elif param_name in ["leverage", "margin_mode", "trading_mode", "initial_balance"]:
            param_categories["Simulation"].append((param_name, param_value))
        else:
            param_categories["Autres"].append((param_name, param_value))
    
    # Créer les cartes de paramètres par catégorie
    param_cards = []
    
    for category, params_list in param_categories.items():
        if not params_list:
            continue
            
        param_items = []
        
        # Trier les paramètres par nom
        params_list.sort(key=lambda x: x[0])
        
        for param_name, param_value in params_list:
            # Formatage spécial pour certains paramètres
            if param_name == "risk_mode":
                value_display = f"{param_value}"
            elif param_name == "base_position" or param_name == "base_sl":
                value_display = f"{param_value*100:.2f}%"
            elif param_name == "tp_multiplier":
                value_display = f"{param_value:.2f}×"
            elif param_name == "leverage":
                value_display = f"{param_value}×"
            elif isinstance(param_value, float):
                value_display = f"{param_value:.4f}"
            else:
                value_display = str(param_value)
            
            param_items.append(
                html.Div([
                    html.Span(param_name, className="small text-light"),
                    html.Span(value_display, className="text-cyan-300 float-end")
                ], className="param-item py-1 border-bottom border-dark")
            )
        
        # Créer la carte pour cette catégorie
        param_cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H6(category, className="mb-0 text-cyan-300"),
                        className="bg-dark border-secondary"
                    ),
                    dbc.CardBody(
                        html.Div(param_items),
                        className="bg-dark p-3"
                    )
                ], className="param-category-card mb-3")
            ], md=6)
        )
    
    return html.Div([
        html.P([
            html.I(className="bi bi-info-circle me-2"),
            "Ces paramètres représentent la configuration exacte de la meilleure stratégie trouvée pendant l'optimisation."
        ], className="text-info small mb-4"),
        
        dbc.Row(param_cards)
    ])

def create_analysis_tab(best_trials):
    """
    Crée l'onglet d'analyse comparative des essais.
    
    Args:
        best_trials: Liste des meilleurs essais
    
    Returns:
        Contenu de l'onglet d'analyse
    """
    if len(best_trials) < 3:
        return html.Div("Pas assez de données pour l'analyse comparative", className="text-center text-muted p-3")
    
    # Extraction des métriques
    metrics_data = []
    for trial in best_trials:
        metrics = trial.get('metrics', {})
        if metrics:
            row = {
                'trial_id': trial.get('trial_id', 0),
                'score': trial.get('score', 0),
                'roi': metrics.get('roi', 0) * 100,
                'win_rate': metrics.get('win_rate', 0) * 100,
                'max_drawdown': metrics.get('max_drawdown', 0) * 100,
                'profit_factor': metrics.get('profit_factor', 0),
                'total_trades': metrics.get('total_trades', 0)
            }
            metrics_data.append(row)
    
    df = pd.DataFrame(metrics_data)
    
    # Matrice de corrélation
    correlation_matrix = df[['score', 'roi', 'win_rate', 'max_drawdown', 'profit_factor', 'total_trades']].corr()
    
    # Heatmap de corrélation
    corr_fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='Viridis',
        colorbar=dict(title='Correlation'),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    corr_fig.update_layout(
        title='Corrélation entre métriques',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,41,59,0.5)',
        height=400,
        margin=dict(t=40, r=20, b=30, l=40),
    )
    
    # Graphique de distribution des scores
    score_fig = go.Figure()
    
    score_fig.add_trace(go.Histogram(
        x=df['score'],
        nbinsx=10,
        marker_color='rgb(34, 211, 238)',
        opacity=0.7,
        name='Distribution'
    ))
    
    score_fig.update_layout(
        title='Distribution des scores',
        xaxis_title='Score',
        yaxis_title='Fréquence',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,41,59,0.5)',
        height=350,
        margin=dict(t=40, r=20, b=30, l=40),
    )
    
    # Diagramme de dispersion ROI vs Win Rate
    scatter_fig = go.Figure()
    
    scatter_fig.add_trace(go.Scatter(
        x=df['roi'],
        y=df['win_rate'],
        mode='markers',
        marker=dict(
            size=12,
            color=df['score'],
            colorscale='Viridis',
            colorbar=dict(title='Score'),
            showscale=True
        ),
        text=df['trial_id'],
        hovertemplate='Trial #%{text}<br>ROI: %{x:.2f}%<br>Win Rate: %{y:.2f}%<extra></extra>'
    ))
    
    scatter_fig.update_layout(
        title='ROI vs Win Rate',
        xaxis_title='ROI (%)',
        yaxis_title='Win Rate (%)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,41,59,0.5)',
        height=350,
        margin=dict(t=40, r=20, b=30, l=40),
    )
    
    return html.Div([
        # Information
        html.P([
            html.I(className="bi bi-info-circle me-2"),
            "Cette analyse montre les relations entre les différentes métriques des meilleurs essais."
        ], className="text-info small mb-4"),
        
        # Graphiques d'analyse
        dbc.Row([
            # Matrice de corrélation
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H6("Corrélation entre métriques", className="mb-0 text-cyan-300"),
                        className="bg-dark border-secondary"
                    ),
                    dbc.CardBody(
                        dcc.Graph(
                            figure=corr_fig,
                            config={'displayModeBar': False},
                            className="retro-graph"
                        ),
                        className="bg-dark p-0 pt-2"
                    )
                ], className="retro-card mb-4")
            ], md=6),
            
            # Graphique de dispersion
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H6("ROI vs Win Rate", className="mb-0 text-cyan-300"),
                        className="bg-dark border-secondary"
                    ),
                    dbc.CardBody(
                        dcc.Graph(
                            figure=scatter_fig,
                            config={'displayModeBar': False},
                            className="retro-graph"
                        ),
                        className="bg-dark p-0 pt-2"
                    )
                ], className="retro-card mb-4")
            ], md=6),
        ]),
        
        # Distribution des scores
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H6("Distribution des scores", className="mb-0 text-cyan-300"),
                        className="bg-dark border-secondary"
                    ),
                    dbc.CardBody(
                        dcc.Graph(
                            figure=score_fig,
                            config={'displayModeBar': False},
                            className="retro-graph"
                        ),
                        className="bg-dark p-0 pt-2"
                    )
                ], className="retro-card")
            ], md=12),
        ]),
    ])

def create_actions_tab(study_name):
    """
    Crée l'onglet des actions possibles.
    
    Args:
        study_name: Nom de l'étude
    
    Returns:
        Contenu de l'onglet des actions
    """
    return html.Div([
        html.P([
            html.I(className="bi bi-info-circle me-2"),
            "Ces actions vous permettent d'interagir avec les résultats d'optimisation."
        ], className="text-info small mb-4"),
        
        dbc.Row([
            # Actions principales
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H6("Actions principales", className="mb-0 text-cyan-300"),
                        className="bg-dark border-secondary"
                    ),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-backpack-fill me-2"), "Tester les meilleures stratégies"],
                                    id={"type": "btn-backtest-best", "study": study_name},
                                    className="retro-button w-100 mb-3"
                                ),
                            ], md=6),
                            
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-arrow-repeat me-2"), "Relancer l'optimisation"],
                                    id={"type": "btn-restart-optimization", "study": study_name},
                                    className="retro-button secondary w-100 mb-3"
                                ),
                            ], md=6),
                            
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-save me-2"), "Exporter les résultats"],
                                    id={"type": "btn-export-results", "study": study_name},
                                    className="retro-button secondary w-100 mb-3"
                                ),
                            ], md=6),
                            
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-clipboard-check me-2"), "Comparer les stratégies"],
                                    id={"type": "btn-compare-strategies", "study": study_name},
                                    className="retro-button secondary w-100 mb-3"
                                ),
                            ], md=6),
                        ])
                    ], className="bg-dark")
                ], className="retro-card mb-4")
            ], md=6),
            
            # Paramètres d'optimisation
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H6("Modifier les paramètres", className="mb-0 text-cyan-300"),
                        className="bg-dark border-secondary"
                    ),
                    dbc.CardBody([
                        html.P("Ajustez les poids de scoring pour la prochaine optimisation:", className="mb-3 small"),
                        
                        html.Div([
                            html.Div([
                                html.Span("ROI:", className="text-light me-2"),
                                dcc.Slider(
                                    id={"type": "weight-slider", "metric": "roi"},
                                    min=0,
                                    max=5,
                                    step=0.1,
                                    value=2.5,
                                    marks={i: str(i) for i in range(6)},
                                    className="retro-slider"
                                ),
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Span("Win Rate:", className="text-light me-2"),
                                dcc.Slider(
                                    id={"type": "weight-slider", "metric": "win_rate"},
                                    min=0,
                                    max=5,
                                    step=0.1,
                                    value=0.5,
                                    marks={i: str(i) for i in range(6)},
                                    className="retro-slider"
                                ),
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Span("Max Drawdown:", className="text-light me-2"),
                                dcc.Slider(
                                    id={"type": "weight-slider", "metric": "max_drawdown"},
                                    min=0,
                                    max=5,
                                    step=0.1,
                                    value=2.0,
                                    marks={i: str(i) for i in range(6)},
                                    className="retro-slider"
                                ),
                            ], className="mb-3"),
                        ]),
                        
                        dbc.Button(
                            [html.I(className="bi bi-gear-fill me-2"), "Modifier la configuration"],
                            id={"type": "btn-modify-config", "study": study_name},
                            className="retro-button secondary w-100 mt-2"
                        ),
                    ], className="bg-dark")
                ], className="retro-card mb-4")
            ], md=6),
        ]),
        
        # Retour aux études
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    [html.I(className="bi bi-arrow-left me-2"), "Retour à la liste des études"],
                    id="btn-back-to-studies",
                    className="retro-button secondary w-100"
                ),
            ], md=12),
        ]),
    ])

def create_footer_section(study_name, optimization_results):
    """
    Crée un pied de page avec des informations techniques.
    
    Args:
        study_name: Nom de l'étude
        optimization_results: Résultats d'optimisation
    
    Returns:
        Composant de pied de page
    """
    n_trials = optimization_results.get('n_trials', 0)
    optimization_date = optimization_results.get('optimization_date', 'Non disponible')
    
    return html.Div([
        html.Hr(className="border-secondary mt-4 mb-3"),
        
        html.Div([
            html.Span(f"Étude: {study_name}", className="me-3 text-muted small"),
            html.Span(f"Essais: {n_trials}", className="me-3 text-muted small"),
            html.Span(f"Date: {optimization_date}", className="me-3 text-muted small"),
            html.Span(f"ID: {optimization_results.get('best_trial_id', 'N/A')}", className="text-muted small"),
        ], className="d-flex flex-wrap justify-content-center"),
        
    ], className="mt-4 mb-3")

def register_optimization_details_callbacks(app, central_logger=None):
    """
    Enregistre les callbacks nécessaires pour la page de détails d'optimisation.
    
    Args:
        app: L'instance de l'application Dash
        central_logger: Instance du logger centralisé
    """
    # Callback pour revenir à la liste des études
    @app.callback(
        Output("study-action", "data", allow_duplicate=True),
        [Input("btn-back-to-studies", "n_clicks")],
        prevent_initial_call=True
    )
    def back_to_studies(n_clicks):
        """Retourne à la liste des études"""
        if n_clicks:
            return json.dumps({"type": "back-to-list"})
        return dash.no_update
    
    # Callbacks pour les actions sur les essais et stratégies
    @app.callback(
        Output("optimization-status", "children", allow_duplicate=True),
        [
            Input({"type": "btn-view-trial", "trial_id": dash.ALL}, "n_clicks"),
            Input({"type": "btn-backtest-trial", "trial_id": dash.ALL}, "n_clicks"),
            Input({"type": "btn-use-trial", "trial_id": dash.ALL}, "n_clicks"),
            Input({"type": "btn-backtest-best", "study": dash.ALL}, "n_clicks"),
            Input({"type": "btn-restart-optimization", "study": dash.ALL}, "n_clicks"),
            Input({"type": "btn-export-results", "study": dash.ALL}, "n_clicks"),
            Input({"type": "btn-compare-strategies", "study": dash.ALL}, "n_clicks"),
            Input({"type": "btn-modify-config", "study": dash.ALL}, "n_clicks"),
        ],
        prevent_initial_call=True
    )
    def handle_trial_actions(view_clicks, backtest_clicks, use_clicks, 
                           backtest_best_clicks, restart_opt_clicks,
                           export_clicks, compare_clicks, modify_clicks):
        """Gère les actions sur les essais"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update
            
        # Récupérer le bouton cliqué
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if "btn-view-trial" in button_id:
            # Voir les détails d'un essai
            trial_id = json.loads(button_id)["trial_id"]
            return dbc.Alert(f"Affichage des détails de l'essai #{trial_id}", color="info", dismissable=True)
        
        elif "btn-backtest-trial" in button_id:
            # Tester un essai
            trial_id = json.loads(button_id)["trial_id"]
            return dbc.Alert(f"Démarrage du backtest pour l'essai #{trial_id}", color="success", dismissable=True)
        
        elif "btn-use-trial" in button_id:
            # Utiliser un essai
            trial_id = json.loads(button_id)["trial_id"]
            return dbc.Alert(f"Stratégie de l'essai #{trial_id} sélectionnée", color="success", dismissable=True)
        
        elif "btn-backtest-best" in button_id:
            # Tester les meilleures stratégies
            study = json.loads(button_id)["study"]
            return dbc.Alert(f"Démarrage du backtest des meilleures stratégies pour l'étude '{study}'", color="success", dismissable=True)
        
        elif "btn-restart-optimization" in button_id:
            # Relancer l'optimisation
            study = json.loads(button_id)["study"]
            return dbc.Alert(f"Préparation du redémarrage de l'optimisation pour l'étude '{study}'", color="info", dismissable=True)
        
        elif "btn-export-results" in button_id:
            # Exporter les résultats
            study = json.loads(button_id)["study"]
            return dbc.Alert(f"Export des résultats pour l'étude '{study}'", color="info", dismissable=True)
        
        elif "btn-compare-strategies" in button_id:
            # Comparer les stratégies
            study = json.loads(button_id)["study"]
            return dbc.Alert(f"Préparation de la comparaison des stratégies pour l'étude '{study}'", color="info", dismissable=True)
        
        elif "btn-modify-config" in button_id:
            # Modifier la configuration
            study = json.loads(button_id)["study"]
            return dbc.Alert(f"Ouverture de l'éditeur de configuration pour l'étude '{study}'", color="info", dismissable=True)
        
        return dash.no_update