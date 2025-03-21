"""
Component for displaying the optimization summary metrics.
Shows key performance indicators and optimization settings.
"""
import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import json

def create_optimization_summary(optimization_config, best_trial_id, n_trials):
    """
    Creates the optimization summary panel.
    
    Args:
        optimization_config: Configuration used for the optimization
        best_trial_id: ID of the best trial
        n_trials: Number of trials executed
        
    Returns:
        Summary component with key metrics
    """
    return html.Div([
        html.Div([
            html.H4([
                html.I(className="bi bi-bar-chart-fill me-2"),
                "Optimization Summary"
            ], className="retro-card-title d-flex align-items-center"),
        ], className="retro-card-header p-3"),
        
        html.Div([
            # Overall metrics
            html.Div([
                create_metric_card("Total Trials", n_trials, "bi-cpu", "cyan"),
                create_metric_card("Best Trial", best_trial_id, "bi-trophy", "green"),
                create_metric_card("Success Rate", f"{(n_trials - optimization_config.get('failed_trials', 0)) / max(1, n_trials):.1%}", "bi-check-circle", "yellow"),
            ], className="d-flex flex-wrap justify-content-between mb-4"),
            
            # Performance indicators
            html.H5("Best Strategy Performance", className="text-cyan-300 mb-3"),
            
            dbc.Row([
                dbc.Col([
                    create_performance_metric("ROI", optimization_config.get("best_roi", 0) * 100, "%", threshold=0, prefix="+")
                ], md=6, className="mb-3"),
                dbc.Col([
                    create_performance_metric("Win Rate", optimization_config.get("best_win_rate", 0) * 100, "%", threshold=50)
                ], md=6, className="mb-3"),
                dbc.Col([
                    create_performance_metric("Max Drawdown", optimization_config.get("best_max_drawdown", 0) * 100, "%", threshold=20, reverse=True)
                ], md=6, className="mb-3"),
                dbc.Col([
                    create_performance_metric("Profit Factor", optimization_config.get("best_profit_factor", 1), "", threshold=1.5)
                ], md=6, className="mb-3"),
                dbc.Col([
                    create_performance_metric("Total Trades", optimization_config.get("best_total_trades", 0), "", threshold=50)
                ], md=6, className="mb-3"),
                dbc.Col([
                    create_performance_metric("Avg Profit/Trade", optimization_config.get("best_avg_profit", 0) * 100, "%", threshold=0)
                ], md=6, className="mb-3"),
            ]),
            
            # Optimization settings
            html.H5("Optimization Settings", className="text-cyan-300 mt-4 mb-3"),
            html.Div([
                html.Div(className="params-grid", children=[
                    create_param_item("Optimization Method", optimization_config.get("method", "TPE")),
                    create_param_item("Pruning", optimization_config.get("pruning", "Disabled")),
                    create_param_item("Time Limit", f"{optimization_config.get('timeout', 'None')} sec"),
                    create_param_item("Parallel Jobs", optimization_config.get("n_jobs", 1)),
                    create_param_item("Min Trades", optimization_config.get("min_trades", 10)),
                    create_param_item("Early Stopping", "Enabled" if optimization_config.get("early_stopping", False) else "Disabled"),
                ]),
            ], className="retro-subcard p-3 mt-2"),
            
        ], className="retro-card-body p-3"),
    ], className="retro-card mb-4")


def create_metric_card(title, value, icon, color):
    """Create a small metric card for the summary section."""
    return html.Div([
        html.Div([
            html.I(className=f"bi {icon}"),
        ], className=f"text-{color}-300 fs-4"),
        html.Div([
            html.Div(title, className="text-muted small"),
            html.Div(str(value), className=f"fs-5 text-{color}-300 fw-bold"),
        ]),
    ], className="d-flex align-items-center gap-2 p-2 retro-subcard")


def create_performance_metric(label, value, unit, threshold=0, reverse=False, prefix=""):
    """Creates a performance metric display with value and label."""
    value_float = float(value) if isinstance(value, (int, float)) else 0
    
    # Determine color based on threshold and whether higher is better (reverse=False) or lower is better (reverse=True)
    color = "text-yellow"
    if (not reverse and value_float > threshold) or (reverse and value_float < threshold):
        color = "text-green"
    elif (not reverse and value_float < threshold) or (reverse and value_float > threshold):
        color = "text-red"
        
    # Format the value with appropriate precision
    if isinstance(value, float):
        value_display = f"{prefix}{value:.2f}{unit}"
    else:
        value_display = f"{prefix}{value}{unit}"
    
    return html.Div([
        html.Div(label, className="param-label mb-1"),
        html.Div(value_display, className=f"fs-5 fw-bold {color}"),
    ], className="param-item")


def create_param_item(label, value):
    """Creates a parameter display item."""
    return html.Div([
        html.Div(label, className="param-label"),
        html.Div(value, className="param-value text-cyan-300"),
    ], className="param-item")


def register_summary_callbacks(app, central_logger=None):
    """
    Register callbacks for the optimization summary component.
    
    Args:
        app: Dash app instance
        central_logger: Optional logger instance
    """
    # No callbacks needed for this component yet
    pass