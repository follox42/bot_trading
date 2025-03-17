"""
Main layout for the optimization details view.
This is the entry point for displaying optimization results.
"""
import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import json
import os
from datetime import datetime

from ui.components.studies.optimizations.optimization_summary import create_optimization_summary
from ui.components.studies.optimizations.optimization_best_trials import create_best_trials_section
from ui.components.studies.optimizations.optimization_parameters import create_parameters_section
from ui.components.studies.optimizations.optimization_strategy_viz import create_strategy_visualization
from ui.components.studies.optimizations.optimization_advanced_analytics import create_advanced_analytics
from ui.components.studies.optimizations.optimization_actions import create_actions_panel

def create_optimization_details(study_name, central_logger=None):
    """
    Creates the main layout for optimization details view.
    
    Args:
        study_name: Name of the optimization study
        central_logger: Instance of the centralized logger
        
    Returns:
        Complete optimization details panel
    """
    try:
        from simulator.study_manager import IntegratedStudyManager
        study_manager = IntegratedStudyManager("studies")
        
        if not study_manager.study_exists(study_name):
            return create_error_message(f"Study '{study_name}' does not exist")
            
        optimization_results = study_manager.get_optimization_results(study_name)
        if not optimization_results:
            return create_error_message("No optimization results available for this study")
            
        best_trials = optimization_results.get('best_trials', [])
        best_trial_id = optimization_results.get('best_trial_id', -1)
        n_trials = optimization_results.get('n_trials', 0)
        optimization_date = optimization_results.get('optimization_date', 'Not available')
        optimization_config = optimization_results.get('optimization_config', {})
        
        if not best_trials:
            return create_error_message("No trials available in optimization results")
            
        return html.Div([
            create_header_section(study_name, optimization_date, n_trials),
            
            dbc.Container([
                dbc.Row([
                    # Left Column: Summary & Best Trials
                    dbc.Col([
                        create_optimization_summary(optimization_config, best_trial_id, n_trials),
                        create_best_trials_section(best_trials, best_trial_id),
                    ], lg=5, md=12, className="mb-4"),
                    
                    # Right Column: Strategy Visualization and Performance
                    dbc.Col([
                        create_strategy_visualization(best_trials, best_trial_id),
                    ], lg=7, md=12),
                ]),
                
                # Parameter Analysis Section
                dbc.Row([
                    dbc.Col([
                        create_parameters_section(best_trials, best_trial_id),
                    ], className="mt-4"),
                ]),
                
                # Advanced Analytics Tabs
                dbc.Row([
                    dbc.Col([
                        create_advanced_analytics(best_trials, study_name),
                    ], className="mt-4"),
                ]),
                
                # Actions Panel
                dbc.Row([
                    dbc.Col([
                        create_actions_panel(study_name, best_trial_id),
                    ], className="mt-4 mb-5"),
                ]),
            ], fluid=True, className="fade-in-animation"),
            
            # Footer with meta information
            html.Div([
                html.Hr(className="border-secondary mt-4 mb-3"),
                html.Div([
                    html.Span(f"Study: {study_name}", className="me-3 text-muted small"),
                    html.Span(f"Trials: {n_trials}", className="me-3 text-muted small"),
                    html.Span(f"Date: {optimization_date}", className="me-3 text-muted small"),
                    html.Span(f"ID: {best_trial_id}", className="text-muted small"),
                ], className="d-flex flex-wrap justify-content-center"),
            ], className="mt-4 mb-3"),
            
            # Hidden storage for state
            dcc.Store(id="optimization-details-data", data=json.dumps({
                "study_name": study_name,
                "best_trial_id": best_trial_id,
                "n_trials": n_trials
            })),
        ], className="optimization-details-container")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return create_error_message(f"Error loading optimization details: {str(e)}")


def create_header_section(study_name, optimization_date, n_trials):
    """Creates the header section with study information."""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H3([
                    html.I(className="bi bi-cpu me-2"),
                    f"Optimization Results: ",
                    html.Span(study_name, className="text-cyan-300")
                ], className="header-content glitch-text"),
                
                html.Div([
                    html.Span("OPTIMIZED", className="retro-badge retro-badge-green me-2"),
                    html.Span(f"TRIALS: {n_trials}", className="retro-badge retro-badge-blue me-2"),
                    html.Span(f"DATE: {optimization_date}", className="retro-badge retro-badge-yellow"),
                ], className="mt-2"),
            ], md=8),
            
            dbc.Col([
                html.Button([
                    html.I(className="bi bi-arrow-left me-2"),
                    "Back to Optimizations"
                ], id="optimization-back-btn", className="retro-button float-end"),
            ], md=4, className="d-flex align-items-center justify-content-end"),
        ]),
        
        # Retro Scanline effect
        html.Div(className="scanline"),
        
    ], className="retro-card-header p-3 mb-4 d-flex align-items-center")


def create_error_message(message):
    """Creates an error message display."""
    return html.Div([
        html.Div([
            html.I(className="bi bi-exclamation-triangle-fill text-warning me-2 fs-3"),
            html.H4("Error Loading Optimization", className="text-warning"),
            html.P(message, className="text-muted"),
            html.Button([
                html.I(className="bi bi-arrow-left me-2"),
                "Return to Optimizations List"
            ], id="optimization-back-btn", className="retro-button mt-3"),
        ], className="text-center p-5")
    ], className="retro-card bg-dark my-5")


def register_optimization_details_callbacks(app, central_logger=None):
    """Registers callbacks for the optimization details page."""
    
    # Register callbacks from all subcomponents
    from components.optimization_summary import register_summary_callbacks
    from components.optimization_best_trials import register_best_trials_callbacks
    from components.optimization_parameters import register_parameters_callbacks
    from components.optimization_strategy_viz import register_strategy_viz_callbacks
    from components.optimization_advanced_analytics import register_advanced_analytics_callbacks
    from components.optimization_actions import register_actions_callbacks
    
    register_summary_callbacks(app, central_logger)
    register_best_trials_callbacks(app, central_logger)
    register_parameters_callbacks(app, central_logger)
    register_strategy_viz_callbacks(app, central_logger)
    register_advanced_analytics_callbacks(app, central_logger)
    register_actions_callbacks(app, central_logger)