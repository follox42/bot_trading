"""
Main panel for the optimization details view.
Integrates all components into a cohesive layout.
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

from ui.components.studies.optimizations.header import create_header_section, create_error_message
from ui.components.studies.optimizations.summary import create_optimization_summary, create_best_trials_table
from ui.components.studies.optimizations.performance import create_performance_visualizations, create_best_trial_details
from ui.components.studies.optimizations.parameters import create_parameters_tab
from ui.components.studies.optimizations.actions import create_actions_tab
from ui.components.studies.optimizations.analysis import (
    create_analysis_tab,
    create_comparative_metrics_table,
    create_parameter_space_analysis
)


def create_optimization_details(study_name):
    """
    Creates the optimization details panel for a given study.
    
    Args:
        study_name: Name of the study
    
    Returns:
        Complete optimization details panel
    """
    try:
        # Initialize the study manager
        from simulator.study_manager import IntegratedStudyManager
        
        study_manager = IntegratedStudyManager("studies")
        
        # Check if the study exists
        if not study_manager.study_exists(study_name):
            return create_error_message(f"Study '{study_name}' does not exist.")
        
        # Get optimization results
        optimization_results = study_manager.get_optimization_results(study_name)
        
        if not optimization_results:
            return create_error_message("No optimization results available for this study.")
        
        # Extract information
        best_trials = optimization_results.get('best_trials', [])
        best_trial_id = optimization_results.get('best_trial_id', -1)
        n_trials = optimization_results.get('n_trials', 0)
        optimization_date = optimization_results.get('optimization_date', 'Not available')
        optimization_config = optimization_results.get('optimization_config', {})
        
        # Check if data is sufficient
        if not best_trials:
            return create_error_message("No trials available in optimization results.")
        
        # Create the layout
        return html.Div([
            # Header
            create_header_section(study_name, optimization_date, n_trials),
            
            # Main container with two columns
            dbc.Row([
                # Column 1: General information and best trials
                dbc.Col([
                    # Optimization summary
                    create_optimization_summary(optimization_config, best_trial_id, n_trials),
                    
                    # Best trials table
                    create_best_trials_table(best_trials, best_trial_id),
                ], lg=5, md=12, className="mb-4"),
                
                # Column 2: Visualizations and best trial details
                dbc.Col([
                    # Performance visualizations
                    create_performance_visualizations(best_trials, best_trial_id),
                    
                    # Best trial details
                    create_best_trial_details(best_trials, best_trial_id),
                ], lg=7, md=12),
            ]),
            
            # Tabs section for detailed analysis
            html.Div([
                dbc.Tabs([
                    # Parameters tab
                    dbc.Tab(
                        create_parameters_tab(best_trials, best_trial_id),
                        label="Parameters",
                        tab_id="tab-parameters",
                        className="mt-3 p-3"
                    ),
                    
                    # Analysis tab
                    dbc.Tab(
                        create_analysis_tab(best_trials),
                        label="Analysis",
                        tab_id="tab-analysis",
                        className="mt-3 p-3"
                    ),
                    
                    # Parameter space analysis
                    dbc.Tab(
                        create_parameter_space_analysis(best_trials),
                        label="Parameter Space",
                        tab_id="tab-parameter-space",
                        className="mt-3 p-3"
                    ),
                    
                    # Actions tab
                    dbc.Tab(
                        create_actions_tab(study_name),
                        label="Actions",
                        tab_id="tab-actions",
                        className="mt-3 p-3"
                    ),
                ], className="retro-tabs mt-4", id="optimization-details-tabs"),
            ]),
            
            # Footer with technical information
            html.Div([
                html.Hr(className="border-secondary mt-4 mb-3"),
                
                html.Div([
                    html.Span(f"Study: {study_name}", className="me-3 text-muted small"),
                    html.Span(f"Trials: {n_trials}", className="me-3 text-muted small"),
                    html.Span(f"Date: {optimization_date}", className="me-3 text-muted small"),
                    html.Span(f"ID: {best_trial_id}", className="text-muted small"),
                ], className="d-flex flex-wrap justify-content-center"),
                
            ], className="mt-4 mb-3")
        ], className="optimization-details-container")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return create_error_message(f"Error loading optimization details: {str(e)}")