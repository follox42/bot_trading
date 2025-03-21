"""
Component for analyzing parameter distributions and importance.
Shows parameter importance, distributions, and correlations.
"""
import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json

def create_parameters_section(best_trials, best_trial_id):
    """
    Creates the parameters analysis section.
    
    Args:
        best_trials: List of best trial data
        best_trial_id: ID of the best trial
        
    Returns:
        Component with parameter analysis visualizations
    """
    return html.Div([
        html.Div([
            html.H4([
                html.I(className="bi bi-sliders me-2"),
                "Parameter Analysis"
            ], className="retro-card-title d-flex align-items-center"),
        ], className="retro-card-header p-3"),
        
        html.Div([
            # Parameter tabs for different visualizations
            dbc.Tabs([
                dbc.Tab(
                    create_param_importance_tab(best_trials),
                    label="PARAMETER IMPORTANCE",
                    tab_id="tab-param-importance",
                    className="retro-tabs",
                    activeLabelClassName="text-cyan-300"
                ),
                dbc.Tab(
                    create_param_distribution_tab(best_trials),
                    label="DISTRIBUTIONS",
                    tab_id="tab-param-distribution",
                    className="retro-tabs",
                    activeLabelClassName="text-cyan-300"
                ),
                dbc.Tab(
                    create_param_correlation_tab(best_trials),
                    label="CORRELATIONS",
                    tab_id="tab-param-correlation",
                    className="retro-tabs",
                    activeLabelClassName="text-cyan-300"
                ),
                dbc.Tab(
                    create_best_params_tab(best_trials, best_trial_id),
                    label="BEST PARAMETERS",
                    tab_id="tab-best-params",
                    className="retro-tabs",
                    activeLabelClassName="text-cyan-300"
                ),
            ], id="params-tabs", className="retro-tabs"),
            
        ], className="retro-card-body p-3"),
    ], className="retro-card mb-4")


def create_param_importance_tab(best_trials):
    """Creates the parameter importance visualization tab."""
    
    # Extract parameter importance data (mock for now, would be calculated from trials)
    param_importance = calculate_param_importance(best_trials)
    
    if not param_importance:
        return html.Div("Not enough data to calculate parameter importance", 
                        className="text-center text-muted p-5")
    
    # Create importance bar chart
    param_names = list(param_importance.keys())
    param_values = list(param_importance.values())
    
    # Sort by importance value
    sorted_indices = np.argsort(param_values)[::-1]
    param_names = [param_names[i] for i in sorted_indices]
    param_values = [param_values[i] for i in sorted_indices]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=param_names,
        x=param_values,
        orientation='h',
        marker=dict(
            color=param_values,
            colorscale='Viridis',
            colorbar=dict(title="Importance")
        )
    ))
    
    fig.update_layout(
        title="Parameter Importance",
        xaxis_title="Relative Importance",
        yaxis_title="Parameter",
        height=max(400, len(param_names) * 30),
        margin=dict(l=10, r=10, t=50, b=10),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.3)',
    )
    
    return html.Div([
        html.P([
            html.I(className="bi bi-info-circle me-2"),
            "This visualization shows which parameters had the most impact on optimization results."
        ], className="text-muted mb-4"),
        
        dcc.Graph(
            figure=fig,
            id='param-importance-graph',
            className='retro-graph'
        )
    ], className="p-3")


def create_param_distribution_tab(best_trials):
    """Creates the parameter distribution visualization tab."""
    
    # Extract all numerical parameters
    all_params = {}
    for trial in best_trials:
        params = trial.get('params', {})
        for param, value in params.items():
            if isinstance(value, (int, float)):
                if param not in all_params:
                    all_params[param] = []
                all_params[param].append(value)
    
    if not all_params:
        return html.Div("No numerical parameters found", className="text-center text-muted p-5")
    
    # Create parameter selector
    param_options = [{'label': param, 'value': param} for param in all_params.keys()]
    
    return html.Div([
        html.P([
            html.I(className="bi bi-info-circle me-2"),
            "Select a parameter to see its distribution across the top trials."
        ], className="text-muted mb-4"),
        
        dbc.Row([
            dbc.Col([
                html.Label("Select Parameter:", className="mb-2 text-cyan-300"),
                dcc.Dropdown(
                    options=param_options,
                    value=param_options[0]['value'] if param_options else None,
                    id='param-distribution-selector',
                    className='retro-dropdown'
                )
            ], md=6, className="mb-3")
        ]),
        
        html.Div(id='param-distribution-graph-container')
    ], className="p-3")


def create_param_correlation_tab(best_trials):
    """Creates the parameter correlation visualization tab."""
    
    # Extract all parameters and performance metrics
    param_data = []
    for trial in best_trials:
        trial_data = {}
        
        # Add parameters
        params = trial.get('params', {})
        for param, value in params.items():
            if isinstance(value, (int, float)):
                trial_data[param] = value
        
        # Add metrics
        trial_data['roi'] = trial.get('roi', 0)
        trial_data['win_rate'] = trial.get('win_rate', 0)
        trial_data['max_drawdown'] = trial.get('max_drawdown', 0)
        trial_data['total_trades'] = trial.get('total_trades', 0)
        
        if trial_data:
            param_data.append(trial_data)
    
    if not param_data:
        return html.Div("Not enough data to calculate correlations", 
                        className="text-center text-muted p-5")
    
    # Create correlation matrix
    df = pd.DataFrame(param_data)
    correlation = df.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation.values,
        x=correlation.columns,
        y=correlation.index,
        colorscale='RdBu_r',
        zmid=0
    ))
    
    fig.update_layout(
        title='Parameter Correlation Matrix',
        height=700,
        margin=dict(l=10, r=10, t=50, b=10),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.3)',
    )
    
    return html.Div([
        html.P([
            html.I(className="bi bi-info-circle me-2"),
            "This heatmap shows correlations between parameters and performance metrics. " +
            "Blue indicates positive correlation, red indicates negative correlation."
        ], className="text-muted mb-4"),
        
        dcc.Graph(
            figure=fig,
            id='param-correlation-graph',
            className='retro-graph'
        )
    ], className="p-3")


def create_best_params_tab(best_trials, best_trial_id):
    """Creates the best parameters display tab."""
    
    # Find the best trial
    best_trial = next((t for t in best_trials if t.get('trial_id') == best_trial_id), 
                      best_trials[0] if best_trials else None)
    
    if not best_trial:
        return html.Div("Best trial data not available", className="text-center text-muted p-5")
    
    params = best_trial.get('params', {})
    
    # Group parameters by type
    param_groups = {
        "Trading Structure": [],
        "Indicator Parameters": [],
        "Risk Management": [],
        "Simulation Settings": []
    }
    
    for param, value in params.items():
        if param.startswith(('buy_', 'sell_', 'n_buy', 'n_sell')):
            param_groups["Trading Structure"].append((param, value))
        elif any(ind in param for ind in ['ema', 'sma', 'rsi', 'macd', 'atr', 'ind']):
            param_groups["Indicator Parameters"].append((param, value))
        elif any(p in param for p in ['risk', 'sl_', 'tp_', 'position', 'leverage']):
            param_groups["Risk Management"].append((param, value))
        else:
            param_groups["Simulation Settings"].append((param, value))
    
    # Sort parameters within each group
    for group in param_groups:
        param_groups[group] = sorted(param_groups[group], key=lambda x: x[0])
    
    return html.Div([
        html.P([
            html.I(className="bi bi-info-circle me-2"),
            "These are the parameters of the best performing strategy."
        ], className="text-muted mb-4"),
        
        # Parameter groups
        html.Div([
            html.Div([
                html.H5(group, className="text-cyan-300 mb-3"),
                html.Div([
                    html.Div([
                        html.Div(format_param_name(param), className="param-label"),
                        html.Div(format_param_value(value), className="param-value highlight-value")
                    ], className="param-item")
                    for param, value in params
                ], className="params-grid")
            ], className="mb-4 param-category-card p-3")
            for group, params in param_groups.items() if params
        ])
    ], className="p-3")


def calculate_param_importance(best_trials, top_n=30):
    """
    Calculate parameter importance from trial data.
    
    This is a simplified version - a more advanced implementation would use
    methods like feature importance from decision trees.
    """
    if len(best_trials) < 5:
        return {}
        
    # Extract parameters and scores
    param_data = []
    scores = []
    
    for trial in best_trials:
        params = trial.get('params', {})
        if not params:
            continue
            
        param_dict = {}
        for param, value in params.items():
            if isinstance(value, (int, float)):
                param_dict[param] = value
                
        if param_dict:
            param_data.append(param_dict)
            scores.append(trial.get('score', 0))
    
    if not param_data:
        return {}
        
    # Create a DataFrame
    df = pd.DataFrame(param_data)
    
    # Calculate correlation with score
    importance = {}
    for column in df.columns:
        importance[column] = abs(np.corrcoef(df[column].values, scores)[0, 1])
    
    # Handle NaN values
    importance = {k: v if not np.isnan(v) else 0 for k, v in importance.items()}
    
    # Normalize to sum to 1
    total = sum(importance.values())
    if total > 0:
        importance = {k: v / total for k, v in importance.items()}
    
    # Sort and return top N
    sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n])
    return sorted_importance


def format_param_name(param_name):
    """Format parameter name for display."""
    # Replace underscores with spaces
    name = param_name.replace('_', ' ')
    
    # Capitalize first letter of each word
    name = ' '.join(word.capitalize() for word in name.split())
    
    return name


def format_param_value(value):
    """Format parameter value for display."""
    if isinstance(value, float):
        # Format percentage values
        if 0 <= value <= 1:
            return f"{value:.2%}"
        return f"{value:.4f}"
    return str(value)


def register_parameters_callbacks(app, central_logger=None):
    """
    Register callbacks for the parameters analysis component.
    
    Args:
        app: Dash app instance
        central_logger: Optional logger instance
    """
    @app.callback(
        Output('param-distribution-graph-container', 'children'),
        Input('param-distribution-selector', 'value'),
        State('optimization-details-data', 'data'),
        prevent_initial_call=True
    )
    def update_param_distribution(selected_param, details_data):
        """Update the parameter distribution graph based on selection."""
        if not selected_param or not details_data:
            return html.Div("Please select a parameter", className="text-center text-muted p-4")
            
        details = {}
        try:
            details = json.loads(details_data)
        except:
            return html.Div("Error loading optimization data", className="text-center text-muted p-4")
            
        study_name = details.get("study_name", "unknown")
        
        try:
            from core.study.study_manager import IntegratedStudyManager
            study_manager = IntegratedStudyManager("studies")
            optimization_results = study_manager.get_optimization_results(study_name)
            
            if not optimization_results or 'best_trials' not in optimization_results:
                return html.Div("No parameter data available", className="text-center text-muted p-4")
                
            best_trials = optimization_results['best_trials']
            
            # Extract parameter values
            param_values = []
            scores = []
            
            for trial in best_trials:
                params = trial.get('params', {})
                if selected_param in params and isinstance(params[selected_param], (int, float)):
                    param_values.append(params[selected_param])
                    scores.append(trial.get('score', 0))
            
            if not param_values:
                return html.Div(f"No data available for parameter '{selected_param}'", 
                               className="text-center text-muted p-4")
            
            # Create visualization
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=param_values,
                nbinsx=30,
                name="Distribution",
                marker=dict(
                    color='rgba(34, 211, 238, 0.7)',
                    line=dict(color='rgba(34, 211, 238, 1)', width=1)
                )
            ))
            
            # Add scatter plot of parameter values vs scores
            fig.add_trace(go.Scatter(
                x=param_values,
                y=scores,
                mode='markers',
                name='Scores',
                yaxis='y2',
                marker=dict(
                    size=8,
                    color='rgba(74, 222, 128, 0.9)',
                    line=dict(width=1, color='rgba(74, 222, 128, 1)')
                )
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Distribution of '{format_param_name(selected_param)}'",
                xaxis_title=format_param_name(selected_param),
                yaxis_title="Frequency",
                yaxis2=dict(
                    title="Score",
                    overlaying="y",
                    side="right",
                    showgrid=False
                ),
                barmode='overlay',
                height=500,
                margin=dict(l=10, r=10, t=50, b=10),
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,23,42,0.3)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return dcc.Graph(
                figure=fig,
                id='param-distribution-graph',
                className='retro-graph'
            )
            
        except Exception as e:
            if central_logger:
                central_logger.error(f"Error updating parameter distribution: {str(e)}")
            return html.Div("Error creating parameter distribution visualization", 
                           className="text-center text-muted p-4")