"""
Analysis tab components for the optimization details panel.
Contains comparative analysis of trials and advanced visualizations.
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

from advanced_visualizations import (
    create_multidimensional_exploration,
    create_correlation_heatmap,
    create_parameter_importance,
    create_metric_distributions,
    create_optimization_progress,
    create_strategy_composition
)


def create_analysis_tab(best_trials):
    """
    Creates the analysis tab with comparative visualizations.
    
    Args:
        best_trials: List of the best trials
    
    Returns:
        Analysis tab content
    """
    if len(best_trials) < 3:
        return html.Div("Not enough data for comparative analysis", className="text-center text-muted p-3")
    
    return html.Div([
        html.P([
            html.I(className="bi bi-info-circle me-2"),
            "This analysis shows the relationships between different parameters and metrics across the best trials."
        ], className="text-info small mb-4"),
        
        # Top section: Optimization Progress and Parameter Importance
        dbc.Row([
            dbc.Col([
                create_optimization_progress(best_trials)
            ], lg=12)
        ]),
        
        # Middle section: Multidimensional exploration and correlation heatmap
        dbc.Row([
            dbc.Col([
                create_multidimensional_exploration(best_trials)
            ], lg=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                create_correlation_heatmap(best_trials)
            ], lg=12)
        ]),
        
        # Add parameter importance section
        dbc.Row([
            dbc.Col([
                create_parameter_importance(best_trials)
            ], lg=12)
        ]),
        
        # Add metric distributions
        dbc.Row([
            dbc.Col([
                create_metric_distributions(best_trials)
            ], lg=12)
        ]),
        
        # Add strategy composition analysis
        dbc.Row([
            dbc.Col([
                create_strategy_composition(best_trials)
            ], lg=12)
        ])
    ])


def create_comparative_metrics_table(best_trials):
    """
    Creates a comparative table of key metrics across top trials.
    
    Args:
        best_trials: List of best trials
    
    Returns:
        Comparative metrics table component
    """
    if not best_trials:
        return html.Div("No trials available for comparison", className="text-center text-muted p-3")
    
    # Top 5 trials
    top_trials = sorted(
        best_trials[:5], 
        key=lambda t: t.get('score', 0), 
        reverse=True
    )
    
    # Create table header
    header = html.Tr([
        html.Th("Metric", className="text-start"),
        *[html.Th(f"Trial #{t.get('trial_id', i)}", className="text-center") 
          for i, t in enumerate(top_trials)]
    ], className="retro-table-header")
    
    # Define metrics to display
    metrics = [
        {"name": "Score", "key": "score", "format": lambda x: f"{x:.4f}"},
        {"name": "ROI", "key": "roi", "format": lambda x: f"{x*100:.2f}%"},
        {"name": "Win Rate", "key": "win_rate", "format": lambda x: f"{x*100:.2f}%"},
        {"name": "Max Drawdown", "key": "max_drawdown", "format": lambda x: f"{x*100:.2f}%"},
        {"name": "Profit Factor", "key": "profit_factor", "format": lambda x: f"{x:.2f}"},
        {"name": "Total Trades", "key": "total_trades", "format": lambda x: f"{int(x)}"},
        {"name": "Avg Profit", "key": "avg_profit", "format": lambda x: f"{x*100:.2f}%"},
    ]
    
    # Create table rows
    rows = []
    for metric in metrics:
        row_cells = [html.Td(metric["name"], className="fw-bold")]
        
        for trial in top_trials:
            value = trial.get('metrics', {}).get(metric["key"], 0)
            
            # For score, get it directly from trial
            if metric["key"] == "score":
                value = trial.get("score", 0)
            
            formatted_value = metric["format"](value)
            
            # Add color classes for certain metrics
            css_class = ""
            if metric["key"] == "roi" or metric["key"] == "avg_profit":
                css_class = "text-success" if value > 0 else "text-danger"
            elif metric["key"] == "max_drawdown":
                css_class = "text-danger"
            elif metric["key"] == "profit_factor":
                css_class = "text-success" if value > 1 else "text-warning"
            
            row_cells.append(html.Td(formatted_value, className=f"text-center {css_class}"))
        
        rows.append(html.Tr(row_cells))
    
    # Create table
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-table me-2"),
                "Comparative Performance Metrics"
            ], className="text-cyan-300 mb-0")
        ], className="bg-dark border-secondary"),
        
        dbc.CardBody([
            html.Div([
                html.Table([
                    html.Thead(header),
                    html.Tbody(rows)
                ], className="retro-table w-100")
            ], className="table-responsive")
        ], className="bg-dark p-0")
    ], className="retro-card shadow mb-4")


def create_analysis_selector(best_trials):
    """
    Creates a selector component for different analysis visualizations.
    
    Args:
        best_trials: List of best trials
    
    Returns:
        Analysis selector component
    """
    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="bi bi-sliders me-2"),
                    "Analysis Tools"
                ], className="text-cyan-300 mb-0")
            ], className="bg-dark border-secondary"),
            
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Div("Select Visualization", className="fw-bold mb-2 text-light"),
                            dbc.RadioItems(
                                id="analysis-selector",
                                options=[
                                    {"label": "3D Parameter Exploration", "value": "3d-exploration"},
                                    {"label": "Correlation Heatmap", "value": "correlation"},
                                    {"label": "Parameter Importance", "value": "importance"},
                                    {"label": "Metric Distributions", "value": "distributions"},
                                    {"label": "Optimization Progress", "value": "progress"},
                                    {"label": "Strategy Composition", "value": "composition"},
                                ],
                                value="3d-exploration",
                                className="analysis-selector"
                            )
                        ], className="mb-3")
                    ], md=6),
                    
                    dbc.Col([
                        html.Div([
                            html.Div("Analysis Options", className="fw-bold mb-2 text-light"),
                            dbc.Checklist(
                                id="analysis-options",
                                options=[
                                    {"label": "Show all trials", "value": "all-trials"},
                                    {"label": "Highlight best trial", "value": "highlight-best"},
                                    {"label": "Show trend lines", "value": "trend-lines"}
                                ],
                                value=["highlight-best", "trend-lines"],
                                className="analysis-options"
                            )
                        ], className="mb-3")
                    ], md=6)
                ]),
                
                dbc.Button(
                    [html.I(className="bi bi-arrow-clockwise me-2"), "Update Analysis"],
                    id="btn-update-analysis",
                    className="retro-button w-100"
                )
            ], className="bg-dark p-3")
        ], className="retro-card shadow mb-4"),
        
        # Placeholder for selected analysis
        html.Div(id="selected-analysis-container")
    ])