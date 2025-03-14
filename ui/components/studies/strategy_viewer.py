"""
Composant modulaire pour l'affichage détaillé d'une stratégie avec design rétro.
Divise l'affichage en sections distinctes pour une meilleure organisation.
"""
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import dash
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime

from logger.logger import LoggerType

class StrategyViewer:
    """
    Classe modulaire pour l'affichage d'une stratégie avec design rétro.
    """
    
    def __init__(self, study_name, strategy_rank, central_logger=None):
        """
        Initialise le visualiseur de stratégie.
        
        Args:
            study_name: Nom de l'étude
            strategy_rank: Rang de la stratégie
            central_logger: Instance du logger centralisé
        """
        self.study_name = study_name
        self.strategy_rank = strategy_rank
        self.central_logger = central_logger
        
        # Initialiser le logger
        if central_logger:
            self.ui_logger = central_logger.get_logger("strategy_viewer", LoggerType.UI)
            self.ui_logger.info(f"Initialisation du visualiseur pour la stratégie {strategy_rank} de l'étude {study_name}")
        
        # Charger les données de la stratégie
        self._load_strategy_data()
    
    def _load_strategy_data(self):
        """Charge les données de la stratégie depuis le gestionnaire d'études."""
        try:
            from simulator.study_manager import IntegratedStudyManager
            study_manager = IntegratedStudyManager("studies")
            
            # Charger la stratégie
            strategy_data = study_manager.load_strategy(self.study_name, self.strategy_rank)
            
            if not strategy_data:
                self.signal_generator = None
                self.position_calculator = None
                self.performance = None
                self.strategy_exists = False
                self.has_backtest = False
                return
            
            self.signal_generator, self.position_calculator, self.performance = strategy_data
            self.strategy_exists = True
            
            # Vérifier si des résultats de backtest existent
            self.has_backtest = study_manager.get_backtest_results(self.study_name, self.strategy_rank) is not None
            
            # Extraire des informations supplémentaires
            self.strategy_name = self.performance.get('name', f'Stratégie {self.strategy_rank}')
            
        except Exception as e:
            if self.central_logger:
                self.ui_logger.error(f"Erreur lors du chargement de la stratégie: {str(e)}")
            
            self.signal_generator = None
            self.position_calculator = None
            self.performance = None
            self.strategy_exists = False
            self.has_backtest = False
    
    def create_layout(self):
        """
        Crée le layout complet pour la visualisation de la stratégie.
        
        Returns:
            Composant Dash pour la visualisation de la stratégie
        """
        if not self.strategy_exists:
            return self._create_error_layout()
        
        return html.Div([
            # En-tête avec le nom de la stratégie et les actions
            self._create_header(),
            
            # Contenu principal en 2 colonnes
            dbc.Row([
                # Colonne gauche - Blocs de trading et paramètres
                dbc.Col([
                    # Blocs de trading
                    self._create_trading_blocks_card(),
                    
                    # Paramètres de risque
                    self._create_risk_params_card(),
                ], width=12, lg=5, className="strategy-left-column"),
                
                # Colonne droite - Performance et actions
                dbc.Col([
                    # Performances
                    self._create_performance_card(),
                    
                    # Actions disponibles
                    self._create_actions_card(),
                    
                    # Résultats de backtest (si disponibles)
                    html.Div(
                        id={"type": "strategy-backtest-results", "study": self.study_name, "rank": self.strategy_rank},
                        children=self._create_backtest_results() if self.has_backtest else []
                    )
                ], width=12, lg=7, className="strategy-right-column"),
            ], className="strategy-content-row"),
            
            # Modal pour le backtest
            self._create_backtest_modal(),
            
            # Store pour le statut du backtest
            dcc.Store(
                id={"type": "backtest-status", "study": self.study_name, "rank": self.strategy_rank},
                data=None
            ),
        ], id="strategy-viewer-container", className="strategy-viewer fade-in")
    
    def _create_error_layout(self):
        """Crée un layout d'erreur si la stratégie n'existe pas."""
        return html.Div([
            html.Div(
                className="retro-error-container text-center p-5",
                children=[
                    html.I(className="bi bi-exclamation-triangle text-warning display-4 mb-3"),
                    html.H3("Stratégie non trouvée", className="text-warning mb-3"),
                    html.P(f"La stratégie {self.strategy_rank} de l'étude '{self.study_name}' n'existe pas ou n'a pas pu être chargée.", className="text-muted mb-4"),
                    html.Button(
                        [html.I(className="bi bi-arrow-left me-2"), "RETOUR"],
                        id="btn-close-strategy-view",
                        className="retro-button"
                    )
                ]
            )
        ], className="fade-in")
    
    def _create_header(self):
        """Crée l'en-tête de la stratégie avec le titre et les actions."""
        return dbc.Row([
            dbc.Col(
                html.Div(
                    className="d-flex align-items-center strategy-title",
                    children=[
                        html.H3(self.strategy_name, className="text-cyan-300 mb-0 me-3"),
                        html.Span(f"#{self.strategy_rank}", className="strategy-rank")
                    ]
                ),
                width=8
            ),
            dbc.Col(
                html.Div([
                    html.Button(
                        [html.I(className="bi bi-x-lg")],
                        id="btn-close-strategy-view",
                        className="retro-button secondary btn-sm ms-auto",
                        title="Fermer"
                    )
                ], className="d-flex"),
                width=4
            ),
        ], className="mb-4 strategy-header")
    
    def _create_trading_blocks_card(self):
        """Crée la carte des blocs de trading."""
        return html.Div([
            html.Div(
                className="retro-card-header",
                children=[
                    html.H4("BLOCS DE TRADING", className="retro-card-title")
                ]
            ),
            html.Div(
                className="retro-card-body",
                children=[
                    self._create_trading_blocks_content()
                ]
            )
        ], className="retro-card trading-blocks-card mb-4")
    
    def _create_trading_blocks_content(self):
        """Crée le contenu des blocs de trading."""
        # Récupérer les blocs d'achat et de vente
        buy_blocks = self.signal_generator.buy_blocks
        sell_blocks = self.signal_generator.sell_blocks
        
        if not buy_blocks and not sell_blocks:
            return html.Div(
                "Aucun bloc de trading défini pour cette stratégie.", 
                className="empty-blocks-message"
            )
        
        # Créer l'affichage des blocs d'achat
        buy_blocks_display = []
        for i, block in enumerate(buy_blocks):
            # Créer l'affichage des conditions pour ce bloc
            conditions_display = []
            for j, condition in enumerate(block.conditions):
                # Formatage de la condition
                if condition.indicator2 is not None:
                    cond_text = f"{condition.indicator1} {condition.operator.value} {condition.indicator2}"
                else:
                    cond_text = f"{condition.indicator1} {condition.operator.value} {condition.value}"
                
                # Ajouter l'opérateur logique si nécessaire
                logic_op = ""
                if j < len(block.logic_operators):
                    logic_op = f" {block.logic_operators[j].value.upper()} "
                
                # Créer l'élément de condition
                condition_element = html.Div([
                    html.Span(cond_text, className="condition-text"),
                    html.Span(logic_op, className="logic-operator")
                ], className="condition-item")
                
                conditions_display.append(condition_element)
            
            # Créer le bloc
            buy_block = html.Div([
                html.Div(
                    [html.I(className="bi bi-arrow-up-circle me-2"), f"Bloc d'achat #{i+1}"], 
                    className="block-header mb-2"
                ),
                html.Div(conditions_display, className="block-conditions")
            ], className="trading-block buy-block mb-3")
            
            buy_blocks_display.append(buy_block)
        
        # Créer l'affichage des blocs de vente
        sell_blocks_display = []
        for i, block in enumerate(sell_blocks):
            # Créer l'affichage des conditions pour ce bloc
            conditions_display = []
            for j, condition in enumerate(block.conditions):
                # Formatage de la condition
                if condition.indicator2 is not None:
                    cond_text = f"{condition.indicator1} {condition.operator.value} {condition.indicator2}"
                else:
                    cond_text = f"{condition.indicator1} {condition.operator.value} {condition.value}"
                
                # Ajouter l'opérateur logique si nécessaire
                logic_op = ""
                if j < len(block.logic_operators):
                    logic_op = f" {block.logic_operators[j].value.upper()} "
                
                # Créer l'élément de condition
                condition_element = html.Div([
                    html.Span(cond_text, className="condition-text"),
                    html.Span(logic_op, className="logic-operator")
                ], className="condition-item")
                
                conditions_display.append(condition_element)
            
            # Créer le bloc
            sell_block = html.Div([
                html.Div(
                    [html.I(className="bi bi-arrow-down-circle me-2"), f"Bloc de vente #{i+1}"], 
                    className="block-header mb-2"
                ),
                html.Div(conditions_display, className="block-conditions")
            ], className="trading-block sell-block mb-3")
            
            sell_blocks_display.append(sell_block)
        
        # Assembler l'affichage complet
        return html.Div([
            # Section des blocs d'achat
            html.Div([
                html.H5("Blocs d'achat", className="section-title mb-3"),
                html.Div(buy_blocks_display) if buy_blocks_display else html.P("Aucun bloc d'achat défini.")
            ], className="mb-4"),
            
            # Section des blocs de vente
            html.Div([
                html.H5("Blocs de vente", className="section-title mb-3"),
                html.Div(sell_blocks_display) if sell_blocks_display else html.P("Aucun bloc de vente défini.")
            ]),
        ])
    
    def _create_risk_params_card(self):
        """Crée la carte des paramètres de risque."""
        return html.Div([
            html.Div(
                className="retro-card-header",
                children=[
                    html.H4("PARAMÈTRES DE RISQUE", className="retro-card-title")
                ]
            ),
            html.Div(
                className="retro-card-body",
                children=[
                    self._create_risk_params_content()
                ]
            )
        ], className="retro-card risk-params-card")
    
    def _create_risk_params_content(self):
        """Crée le contenu des paramètres de risque."""
        # Récupérer le mode de risque
        risk_mode = self.position_calculator.mode.value
        
        # Paramètres de base
        base_position = self.position_calculator.base_position * 100  # Convertir en pourcentage
        base_sl = self.position_calculator.base_sl * 100  # Convertir en pourcentage
        tp_multiplier = self.position_calculator.tp_multiplier
        
        # Définir l'icône du mode de risque
        risk_icon = None
        risk_mode_display = None
        
        if risk_mode == "fixed":
            risk_icon = "bi-lock"
            risk_mode_display = "Fixe"
        elif risk_mode == "atr_based":
            risk_icon = "bi-bar-chart"
            risk_mode_display = "Basé sur l'ATR"
        elif risk_mode == "volatility_based":
            risk_icon = "bi-activity"
            risk_mode_display = "Basé sur la volatilité"
        
        # Créer l'affichage des paramètres
        params_items = [
            # Mode de risque
            html.Div([
                html.I(className=f"bi {risk_icon} risk-icon"),
                html.Div([
                    html.Div("Mode de Risque", className="param-label"),
                    html.Div(risk_mode_display, className="param-value")
                ])
            ], className="risk-param-item"),
            
            # Paramètres de base
            html.Div([
                html.I(className="bi bi-percent risk-icon"),
                html.Div([
                    html.Div("Taille de Position", className="param-label"),
                    html.Div(f"{base_position:.2f}%", className="param-value")
                ])
            ], className="risk-param-item"),
            
            html.Div([
                html.I(className="bi bi-shield-exclamation risk-icon"),
                html.Div([
                    html.Div("Stop Loss", className="param-label"),
                    html.Div(f"{base_sl:.2f}%", className="param-value")
                ])
            ], className="risk-param-item"),
            
            html.Div([
                html.I(className="bi bi-gem risk-icon"),
                html.Div([
                    html.Div("Multiplicateur TP", className="param-label"),
                    html.Div(f"{tp_multiplier:.2f}x", className="param-value")
                ])
            ], className="risk-param-item"),
        ]
        
        # Paramètres spécifiques au mode
        if risk_mode == "atr_based":
            params_items.extend([
                html.Div([
                    html.I(className="bi bi-calendar3 risk-icon"),
                    html.Div([
                        html.Div("Période ATR", className="param-label"),
                        html.Div(f"{self.position_calculator.atr_period}", className="param-value")
                    ])
                ], className="risk-param-item"),
                
                html.Div([
                    html.I(className="bi bi-asterisk risk-icon"),
                    html.Div([
                        html.Div("Multiplicateur ATR", className="param-label"),
                        html.Div(f"{self.position_calculator.atr_multiplier:.2f}", className="param-value")
                    ])
                ], className="risk-param-item"),
            ])
        elif risk_mode == "volatility_based":
            params_items.extend([
                html.Div([
                    html.I(className="bi bi-calendar3 risk-icon"),
                    html.Div([
                        html.Div("Période de volatilité", className="param-label"),
                        html.Div(f"{self.position_calculator.vol_period}", className="param-value")
                    ])
                ], className="risk-param-item"),
                
                html.Div([
                    html.I(className="bi bi-asterisk risk-icon"),
                    html.Div([
                        html.Div("Multiplicateur de volatilité", className="param-label"),
                        html.Div(f"{self.position_calculator.vol_multiplier:.2f}", className="param-value")
                    ])
                ], className="risk-param-item"),
            ])
        
        # Valeurs calculées
        tp_value = base_sl * tp_multiplier
        
        params_items.append(
            html.Div([
                html.I(className="bi bi-calculator risk-icon"),
                html.Div([
                    html.Div("Take Profit calculé", className="param-label"),
                    html.Div(f"{tp_value:.2f}%", className="param-value highlight-value")
                ])
            ], className="risk-param-item")
        )
        
        return html.Div(params_items, className="risk-params-container")
    
    def _create_performance_card(self):
        """Crée la carte des performances."""
        return html.Div([
            html.Div(
                className="retro-card-header",
                children=[
                    html.H4("PERFORMANCE", className="retro-card-title")
                ]
            ),
            html.Div(
                className="retro-card-body",
                children=[
                    self._create_performance_content()
                ]
            )
        ], className="retro-card performance-card mb-4")
    
    def _create_performance_content(self):
        """Crée le contenu des performances."""
        if not self.performance:
            return html.Div("Aucune donnée de performance disponible.",
                           className="empty-performance-message")
        
        # Extraire les métriques clés
        roi = self.performance.get('roi_pct', self.performance.get('roi', 0) * 100 if 'roi' in self.performance else 0)
        win_rate = self.performance.get('win_rate_pct', self.performance.get('win_rate', 0) * 100 if 'win_rate' in self.performance else 0)
        max_drawdown = self.performance.get('max_drawdown_pct', self.performance.get('max_drawdown', 0) * 100 if 'max_drawdown' in self.performance else 0)
        profit_factor = self.performance.get('profit_factor', 0)
        total_trades = self.performance.get('total_trades', 0)
        
        # Style pour le ROI
        roi_style = {"color": "#4ADE80"} if roi > 0 else {"color": "#F87171"}
        
        # Créer une jauge pour le win rate
        win_rate_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = win_rate,
            title = {'text': "Win Rate", 'font': {'color': 'white'}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            number = {'suffix': "%", 'font': {'color': 'white'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#4ADE80" if win_rate >= 50 else "#FBBF24"},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(248, 113, 113, 0.3)"},
                    {'range': [40, 50], 'color': "rgba(251, 191, 36, 0.3)"},
                    {'range': [50, 100], 'color': "rgba(74, 222, 128, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 2},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        win_rate_gauge.update_layout(
            height=200,
            margin=dict(l=30, r=30, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "white"}
        )
        
        # Créer l'affichage des métriques
        return html.Div([
            # ROI et profit factor
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("ROI", className="metric-label"),
                        html.Div(f"{roi:.2f}%", className="metric-value", style=roi_style)
                    ], className="performance-metric")
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Div("Profit Factor", className="metric-label"),
                        html.Div(f"{profit_factor:.2f}", className="metric-value")
                    ], className="performance-metric")
                ], width=6),
            ], className="mb-3"),
            
            # Jauge de win rate
            dcc.Graph(figure=win_rate_gauge, config={'displayModeBar': False}, className="win-rate-gauge"),
            
            # Autres métriques
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("Drawdown Max", className="metric-label"),
                        html.Div(f"{max_drawdown:.2f}%", className="metric-value", style={"color": "#F87171"})
                    ], className="performance-metric")
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Div("Trades", className="metric-label"),
                        html.Div(f"{total_trades}", className="metric-value")
                    ], className="performance-metric")
                ], width=6),
            ], className="mt-2"),
        ], className="performance-metrics-container")
    
    def _create_actions_card(self):
        """Crée la carte des actions disponibles."""
        return html.Div([
            html.Div(
                className="retro-card-header",
                children=[
                    html.H4("ACTIONS", className="retro-card-title")
                ]
            ),
            html.Div(
                className="retro-card-body",
                children=[
                    # Backtest
                    html.Button(
                        [html.I(className="bi bi-bar-chart-line me-2"), "LANCER BACKTEST"],
                        id={"type": "btn-run-strategy-backtest", "study": self.study_name, "rank": self.strategy_rank},
                        className="retro-button action-button w-100 mb-3"
                    ),
                    
                    # Export
                    html.Button(
                        [html.I(className="bi bi-download me-2"), "EXPORTER LA STRATÉGIE"],
                        id={"type": "btn-export-strategy", "study": self.study_name, "rank": self.strategy_rank},
                        className="retro-button action-button secondary w-100 mb-3"
                    ),
                    
                    # Modification
                    html.Button(
                        [html.I(className="bi bi-pencil me-2"), "MODIFIER"],
                        id={"type": "btn-edit-strategy", "study": self.study_name, "rank": self.strategy_rank},
                        className="retro-button action-button secondary w-100 mb-3",
                        disabled=True  # Fonctionnalité à implémenter plus tard
                    ),
                    
                    # Trading en direct
                    html.Button(
                        [html.I(className="bi bi-lightning me-2"), "TRADING EN DIRECT"],
                        id={"type": "btn-live-trading", "study": self.study_name, "rank": self.strategy_rank},
                        className="retro-button action-button w-100",
                        disabled=True  # Fonctionnalité à implémenter plus tard
                    ),
                ]
            )
        ], className="retro-card actions-card mb-4")
    
    def _create_backtest_results(self):
        """Crée l'affichage des résultats de backtest."""
        try:
            from simulator.study_manager import IntegratedStudyManager
            study_manager = IntegratedStudyManager("studies")
            
            backtest_results = study_manager.get_backtest_results(self.study_name, self.strategy_rank)
            
            if not backtest_results:
                return []
            
            # Créer un graphique d'equity
            equity_curve = backtest_results.get('equity_curve', [])
            trades = backtest_results.get('trade_history', [])
            
            # Préparer les figures si des données sont disponibles
            figures = []
            
            if equity_curve:
                # Graphique de l'équité
                eq_fig = go.Figure()
                
                eq_fig.add_trace(go.Scatter(
                    y=equity_curve,
                    mode='lines',
                    name='Equity',
                    line=dict(color='#22D3EE', width=2)
                ))
                
                eq_fig.update_layout(
                    title='Courbe d\'équité',
                    xaxis_title='Barre',
                    yaxis_title='Équité',
                    template='plotly_dark',
                    height=300,
                    margin=dict(l=40, r=40, t=40, b=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'}
                )
                
                figures.append(dcc.Graph(figure=eq_fig, className="mb-4 backtest-graph"))
            
            if trades:
                # Histogramme des profits
                trades_df = pd.DataFrame(trades)
                
                if 'pnl_pct' in trades_df.columns:
                    pnl_fig = px.histogram(
                        trades_df,
                        x='pnl_pct',
                        color='type',
                        title='Distribution des profits/pertes',
                        template='plotly_dark',
                        color_discrete_map={'long': '#4ADE80', 'short': '#F87171'}
                    )
                    
                    pnl_fig.update_layout(
                        xaxis_title='Profit/Perte (%)',
                        yaxis_title='Nombre de trades',
                        height=300,
                        margin=dict(l=40, r=40, t=40, b=40),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': 'white'}
                    )
                    
                    figures.append(dcc.Graph(figure=pnl_fig, className="backtest-graph"))
            
            if figures:
                return html.Div([
                    html.Div(
                        className="retro-card-header",
                        children=[
                            html.H4("RÉSULTATS DE BACKTEST", className="retro-card-title")
                        ]
                    ),
                    html.Div(
                        className="retro-card-body",
                        children=figures
                    )
                ], className="retro-card backtest-results-card")
            
        except Exception as e:
            if self.central_logger:
                self.ui_logger.error(f"Erreur lors de la récupération des résultats de backtest: {str(e)}")
        
        return []
    
    def _create_backtest_modal(self):
        """Crée le modal pour le backtest."""
        return dbc.Modal(
            [
                dbc.ModalHeader(
                    html.H4("BACKTEST EN COURS", className="retro-title"),
                    close_button=False
                ),
                dbc.ModalBody([
                    html.Div([
                        html.P([
                            "Exécution du backtest pour la stratégie ",
                            html.Strong(self.strategy_name),
                            " de l'étude ",
                            html.Strong(self.study_name),
                            "..."
                        ], className="mb-3 text-center"),
                        
                        # Animation de progression
                        html.Div(
                            className="retro-loader-container",
                            children=[
                                html.Div(className="retro-loader"),
                                html.Div(className="loading-text", children="Backtest en cours...")
                            ]
                        ),
                        
                        html.P(
                            "Veuillez patienter pendant l'exécution du backtest. Cette opération peut prendre quelques minutes.", 
                            className="mt-4 text-muted text-center"
                        )
                    ])
                ]),
            ],
            id={"type": "backtest-modal", "study": self.study_name, "rank": self.strategy_rank},
            is_open=False,
            centered=True,
            backdrop="static",
            keyboard=False,
            className="retro-modal"
        )

def create_strategy_viewer(study_name, strategy_rank, central_logger=None):
    """
    Fonction principale pour créer le visualiseur de stratégie.
    
    Args:
        study_name: Nom de l'étude
        strategy_rank: Rang de la stratégie
        central_logger: Instance du logger centralisé
        
    Returns:
        Composant Dash pour la visualisation de la stratégie
    """
    viewer = StrategyViewer(study_name, strategy_rank, central_logger)
    return viewer.create_layout()

def register_strategy_viewer_callbacks(app, central_logger=None):
    """
    Enregistre les callbacks pour la visualisation des stratégies
    
    Args:
        app: L'instance de l'application Dash
        central_logger: Instance du logger centralisé
    """
    # Initialiser le logger
    if central_logger:
        ui_logger = central_logger.get_logger("strategy_viewer_callbacks", LoggerType.UI)
    
    # Callback pour fermer la vue de stratégie
    @app.callback(
        [Output("selected-strategy-content", "children"),
         Output("selected-strategy-content", "style")],
        [Input("btn-close-strategy-view", "n_clicks")],
        prevent_initial_call=True
    )
    def close_strategy_view(n_clicks):
        """Ferme la vue de stratégie"""
        if not n_clicks:
            return dash.no_update, dash.no_update
        
        if central_logger:
            ui_logger.info("Fermeture de la vue de stratégie")
        
        return [], {"display": "none"}
    
    # Callback pour lancer un backtest
    @app.callback(
        [Output({"type": "backtest-modal", "study": dash.ALL, "rank": dash.ALL}, "is_open"),
         Output({"type": "backtest-status", "study": dash.ALL, "rank": dash.ALL}, "data")],
        [Input({"type": "btn-run-strategy-backtest", "study": dash.ALL, "rank": dash.ALL}, "n_clicks")],
        prevent_initial_call=True
    )
    def run_strategy_backtest(n_clicks_list):
        """Lance un backtest pour la stratégie sélectionnée"""
        ctx = dash.callback_context
        if not ctx.triggered or not any(n_clicks_list):
            return dash.no_update, dash.no_update
        
        # Récupérer les informations du bouton cliqué
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        button_data = json.loads(button_id)
        study_name = button_data["study"]
        strategy_rank = button_data["rank"]
        
        if central_logger:
            ui_logger.info(f"Lancement du backtest pour la stratégie {strategy_rank} de l'étude {study_name}")
        
        # Logique pour lancer le backtest
        # Note: Dans une implémentation réelle, il faudrait lancer le backtest en arrière-plan
        # et mettre à jour les résultats une fois terminé
        
        # Simulation d'un backtest pour la démonstration
        import threading
        import time
        
        def run_backtest_thread():
            try:
                if central_logger:
                    ui_logger.info(f"Exécution du backtest pour la stratégie {strategy_rank} de l'étude {study_name}")
                
                # Simuler un temps d'exécution
                time.sleep(3)
                
                # Mise à jour des résultats
                # Dans une implémentation réelle, il faudrait exécuter le backtest et mettre à jour les résultats
                
                if central_logger:
                    ui_logger.info(f"Backtest terminé pour la stratégie {strategy_rank} de l'étude {study_name}")
            except Exception as e:
                if central_logger:
                    ui_logger.error(f"Erreur lors de l'exécution du backtest: {str(e)}")
        
        # Démarrer le thread
        thread = threading.Thread(target=run_backtest_thread)
        thread.daemon = True
        thread.start()
        
        # Retourner les mises à jour pour tous les modals pattern-matching
        return [True], [json.dumps({"status": "running", "study": study_name, "rank": strategy_rank})]
    
    # Callback pour mettre à jour les résultats de backtest
    @app.callback(
        [Output({"type": "strategy-backtest-results", "study": dash.ALL, "rank": dash.ALL}, "children"),
         Output({"type": "backtest-modal", "study": dash.ALL, "rank": dash.ALL}, "is_open", allow_duplicate=True)],
        [Input({"type": "backtest-status", "study": dash.ALL, "rank": dash.ALL}, "data")],
        prevent_initial_call=True
    )
    def update_backtest_results(status_list):
        """Met à jour les résultats de backtest une fois terminé"""
        if not status_list or not any(status_list):
            return dash.no_update, dash.no_update
        
        # Vérifier l'état du backtest
        updates = []
        modal_updates = []
        
        for status_json in status_list:
            if not status_json:
                updates.append(dash.no_update)
                modal_updates.append(dash.no_update)
                continue
            
            status = json.loads(status_json)
            
            if status.get("status") == "running":
                # Le backtest est en cours, ne rien faire pour l'instant
                updates.append(dash.no_update)
                modal_updates.append(True)  # Garder le modal ouvert
            elif status.get("status") == "completed":
                # Le backtest est terminé, mettre à jour les résultats
                study_name = status.get("study")
                strategy_rank = status.get("rank")
                
                # Créer l'affichage des résultats
                viewer = StrategyViewer(study_name, strategy_rank, central_logger)
                results_display = viewer._create_backtest_results()
                updates.append(results_display)
                modal_updates.append(False)  # Fermer le modal
            else:
                updates.append(dash.no_update)
                modal_updates.append(dash.no_update)
        
        return updates, modal_updates
    
    # Callback pour l'exportation de stratégie
    @app.callback(
        Output("strategy-action", "data"),
        [Input({"type": "btn-export-strategy", "study": dash.ALL, "rank": dash.ALL}, "n_clicks")],
        prevent_initial_call=True
    )
    def export_strategy(n_clicks_list):
        """Exporte la stratégie sélectionnée"""
        ctx = dash.callback_context
        if not ctx.triggered or not any(n_clicks_list):
            return dash.no_update
        
        # Récupérer les informations du bouton cliqué
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        button_data = json.loads(button_id)
        study_name = button_data["study"]
        strategy_rank = button_data["rank"]
        
        if central_logger:
            ui_logger.info(f"Exportation de la stratégie {strategy_rank} de l'étude {study_name}")
        
        # Logique pour exporter la stratégie
        # Note: Dans une implémentation réelle, il faudrait exporter la stratégie vers un format spécifique
        
        # Retourner une notification de succès
        return json.dumps({
            "type": "export-strategy",
            "study": study_name,
            "rank": strategy_rank,
            "status": "success"
        })
    
    # Intervalles pour vérifier l'état des backtests
    app.clientside_callback(
        """
        function(n_intervals, status_list) {
            if (!status_list || status_list.length === 0) {
                return status_list;
            }
            
            // Vérifier l'état de chaque backtest
            const updatedStatus = status_list.map(statusJson => {
                if (!statusJson) {
                    return statusJson;
                }
                
                const status = JSON.parse(statusJson);
                
                // Si le backtest est en cours, le marquer comme terminé après quelques secondes
                if (status.status === "running") {
                    // Simuler la fin du backtest après quelques intervalles
                    return JSON.stringify({
                        ...status,
                        status: "completed"
                    });
                }
                
                return statusJson;
            });
            
            return updatedStatus;
        }
        """,
        Output({"type": "backtest-status", "study": dash.ALL, "rank": dash.ALL}, "data", allow_duplicate=True),
        [Input("optimization-refresh-interval", "n_intervals")],
        [State({"type": "backtest-status", "study": dash.ALL, "rank": dash.ALL}, "data")],
        prevent_initial_call=True
    )
