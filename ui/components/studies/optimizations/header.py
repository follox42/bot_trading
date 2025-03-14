"""
Header components for the optimization details panel.
Includes error messages and the main header.
"""
import dash
from dash import html
import dash_bootstrap_components as dbc


def create_error_message(message):
    """
    Creates a styled error message.
    
    Args:
        message: Error message to display
    
    Returns:
        Error message component
    """
    return html.Div([
        html.Div([
            html.I(className="bi bi-exclamation-triangle-fill text-warning me-3", style={"fontSize": "2rem"}),
            html.Div([
                html.H4("Unable to display details", className="text-warning mb-2"),
                html.P(message, className="mb-0 text-light")
            ])
        ], className="d-flex align-items-center"),
        
        html.Div([
            dbc.Button(
                [html.I(className="bi bi-arrow-left me-2"), "Back to studies"],
                id="btn-back-to-studies",
                className="retro-button mt-3"
            )
        ], className="mt-4")
    ], className="retro-card p-4 mt-3 border border-warning bg-dark")


def create_header_section(study_name, optimization_date, n_trials):
    """
    Creates the header section with title and basic information.
    
    Args:
        study_name: Name of the study
        optimization_date: Date of optimization
        n_trials: Number of trials
    
    Returns:
        Header component
    """
    return html.Div([
        dbc.Row([
            # Study information
            dbc.Col([
                html.H3([
                    html.I(className="bi bi-bar-chart-line-fill text-cyan-300 me-3"),
                    "Optimization Results"
                ], className="text-cyan-300 mb-3 neon-text"),
                
                html.H4([
                    "Study: ",
                    html.Span(study_name, className="text-white fw-bold")
                ], className="text-light mb-4"),
            ], md=8),
            
            # Statistics
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("Date: ", className="text-cyan-100"),
                        html.Span(optimization_date, className="text-white")
                    ], className="mb-2"),
                    
                    html.Div([
                        html.Span("Trials: ", className="text-cyan-100"),
                        html.Span(f"{n_trials}", className="text-white")
                    ], className="mb-2"),
                    
                    # Status indicator
                    html.Div([
                        html.Span("Status: ", className="text-cyan-100 me-2"),
                        html.Span([
                            html.I(className="bi bi-check-circle-fill text-success me-1"),
                            "Completed"
                        ], className="retro-badge retro-badge-green")
                    ], className="mb-2"),
                ], className="d-flex flex-column justify-content-center h-100 text-end")
            ], md=4),
        ]),
        
        # Separator bar
        html.Hr(className="border-secondary mb-4"),
    ])