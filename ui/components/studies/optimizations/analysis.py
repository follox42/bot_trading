"""
Analysis tab components for the optimization details panel.
Contains comparative analysis of trials and advanced visualizations.
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ui.components.studies.optimizations.advanced_visualizations import (
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


def create_parameter_space_analysis(best_trials):
    """
    Creates a component for analyzing the parameter search space and optimization trajectory.
    
    Args:
        best_trials: List of best trials
    
    Returns:
        Parameter space analysis component
    """
    if len(best_trials) < 5:
        return html.Div("Not enough trials for parameter space analysis (minimum 5 required)", 
                         className="text-center text-muted p-3")
    
    # Extract parameters and scores for analysis
    params_data = []
    key_params = set()
    
    for trial in best_trials:
        trial_params = trial.get('params', {})
        score = trial.get('score', 0)
        trial_id = trial.get('trial_id', 0)
        
        # Only consider numeric parameters
        numeric_params = {}
        for param_name, param_value in trial_params.items():
            if isinstance(param_value, (int, float)) and not param_name.endswith('_conditions'):
                numeric_params[param_name] = param_value
                key_params.add(param_name)
        
        numeric_params['score'] = score
        numeric_params['trial_id'] = trial_id
        params_data.append(numeric_params)
    
    if not params_data:
        return html.Div("No numeric parameters found for space analysis", 
                         className="text-center text-muted p-3")
    
    # Convert to DataFrame
    df = pd.DataFrame(params_data)
    
    # Create parallel coordinates plot
    # Select top 8 most important parameters based on variance
    if len(key_params) > 8:
        param_var = {param: df[param].var() for param in key_params if param in df.columns}
        sorted_params = sorted(param_var.items(), key=lambda x: x[1], reverse=True)
        selected_params = [p[0] for p in sorted_params[:8]]
    else:
        selected_params = list(key_params)
    
    # Create the plot dimensions
    dimensions = [
        {"label": "Score", "values": df['score'].tolist(), "range": [min(df['score']), max(df['score'])]}
    ]
    
    for param in selected_params:
        if param in df.columns:
            dimensions.append({
                "label": param,
                "values": df[param].tolist(),
                "range": [min(df[param]), max(df[param])]
            })
    
    # Create figure
    parallel_fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df['score'],
                colorscale='Viridis',
                showscale=True,
                colorbar={"title": "Score"}
            ),
            dimensions=dimensions
        )
    )
    
    parallel_fig.update_layout(
        title="Parameter Space Parallel Coordinates",
        height=600,
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.9)",
        plot_bgcolor="rgba(15, 23, 42, 0.9)",
        font=dict(family="'Share Tech Mono', monospace", color="#D1D5DB"),
        margin=dict(l=100, r=30, t=50, b=30)
    )
    
    # Create parameter pairplot matrix for the top parameters
    # Choose top 4 parameters for readability
    if len(selected_params) > 4:
        matrix_params = selected_params[:4]
    else:
        matrix_params = selected_params
    
    # Add score to create pairs with each parameter
    matrix_params.append('score')
    
    # Create pairplot figure
    pairplot_fig = go.Figure()
    
    if len(matrix_params) > 1:
        # Create scatter plot matrix
        for i, param1 in enumerate(matrix_params[:-1]):  # Exclude score for rows
            for j, param2 in enumerate([p for p in matrix_params if p != param1]):
                # Create scatter plot for each parameter pair
                pairplot_fig.add_trace(
                    go.Scatter(
                        x=df[param1],
                        y=df[param2],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=df['score'],
                            colorscale='Viridis',
                            showscale=j == 0,  # Only show colorbar once per row
                            colorbar=dict(title="Score") if j == 0 else None,
                            opacity=0.7
                        ),
                        name=f"{param1} vs {param2}",
                        xaxis=f'x{i+1}',
                        yaxis=f'y{j+1}'
                    )
                )
    
        # Layout with grid of subplots
        pairplot_fig.update_layout(
            title="Parameter Pairs Analysis",
            height=800,
            grid=dict(
                rows=len(matrix_params) - 1,
                columns=len(matrix_params) - 1,
                pattern='independent'
            ),
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.9)",
            plot_bgcolor="rgba(15, 23, 42, 0.9)",
            font=dict(family="'Share Tech Mono', monospace", color="#D1D5DB"),
            margin=dict(l=50, r=30, t=50, b=50)
        )
    
    # Create optimization trajectory visualization
    # Sort trials by ID to get chronological order
    sorted_trials = sorted(best_trials, key=lambda t: t.get('trial_id', 0))
    
    # Extract trial IDs and scores
    trial_ids = [t.get('trial_id', 0) for t in sorted_trials]
    scores = [t.get('score', 0) for t in sorted_trials]
    
    # Calculate cumulative best score
    best_scores = []
    current_best = float('-inf')
    for score in scores:
        current_best = max(current_best, score)
        best_scores.append(current_best)
    
    trajectory_fig = go.Figure()
    
    # Add score points
    trajectory_fig.add_trace(
        go.Scatter(
            x=trial_ids,
            y=scores,
            mode='markers',
            name='Trial Scores',
            marker=dict(
                size=8,
                color=scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Score")
            ),
            hovertemplate="Trial #%{x}<br>Score: %{y:.4f}<extra></extra>"
        )
    )
    
    # Add best score line
    trajectory_fig.add_trace(
        go.Scatter(
            x=trial_ids,
            y=best_scores,
            mode='lines',
            name='Best Score',
            line=dict(color='#FF6B6B', width=2),
            hovertemplate="Trial #%{x}<br>Best Score: %{y:.4f}<extra></extra>"
        )
    )
    
    trajectory_fig.update_layout(
        title="Optimization Trajectory",
        xaxis_title="Trial ID",
        yaxis_title="Score",
        height=500,
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.9)",
        plot_bgcolor="rgba(15, 23, 42, 0.9)",
        font=dict(family="'Share Tech Mono', monospace", color="#D1D5DB"),
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    # Return the complete component
    return html.Div([
        html.P([
            html.I(className="bi bi-info-circle me-2"),
            "This analysis explores the parameter space and optimization trajectory of the search process."
        ], className="text-info small mb-4"),
        
        # Parallel coordinates plot
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="bi bi-layers me-2"),
                    "Parameter Space Exploration"
                ], className="text-cyan-300 mb-0")
            ], className="bg-dark border-secondary"),
            
            dbc.CardBody([
                html.P([
                    "This visualization shows the relationship between different parameters and the optimization score. "
                    "Each line represents a trial, with parameters normalized to a common scale. "
                    "Line color indicates the score value, with brighter colors representing higher scores."
                ], className="text-info small mb-3"),
                
                dcc.Graph(
                    figure=parallel_fig,
                    config={
                        'displayModeBar': True,
                        'displaylogo': False
                    },
                    className="retro-graph"
                )
            ], className="bg-dark p-3")
        ], className="retro-card shadow mb-4"),
        
        # Optimization trajectory
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="bi bi-graph-up-arrow me-2"),
                    "Optimization Trajectory"
                ], className="text-cyan-300 mb-0")
            ], className="bg-dark border-secondary"),
            
            dbc.CardBody([
                html.P([
                    "This chart shows the progression of the optimization process. "
                    "Each point represents a trial, while the red line shows the best score found so far. "
                    "The steepness of this line indicates how quickly the optimizer found good solutions."
                ], className="text-info small mb-3"),
                
                dcc.Graph(
                    figure=trajectory_fig,
                    config={
                        'displayModeBar': True,
                        'displaylogo': False
                    },
                    className="retro-graph"
                )
            ], className="bg-dark p-3")
        ], className="retro-card shadow mb-4"),
        
        # Parameter pairplot
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="bi bi-grid-3x3 me-2"),
                    "Parameter Relationships"
                ], className="text-cyan-300 mb-0")
            ], className="bg-dark border-secondary"),
            
            dbc.CardBody([
                html.P([
                    "This matrix shows relationships between the most important parameters and the score. "
                    "Each scatter plot reveals how pairs of parameters interact to affect the final score. "
                    "Clusters of bright points indicate parameter combinations that lead to high scores."
                ], className="text-info small mb-3"),
                
                dcc.Graph(
                    figure=pairplot_fig,
                    config={
                        'displayModeBar': True,
                        'displaylogo': False
                    },
                    className="retro-graph"
                ) if len(matrix_params) > 1 else html.Div(
                    "Not enough parameters for pair analysis",
                    className="text-center text-muted p-3"
                )
            ], className="bg-dark p-3")
        ], className="retro-card shadow")
    ])


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