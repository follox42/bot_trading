"""
Component for advanced analytics of optimization results.
Includes exploration visualization, metric distributions, and trade analysis.
"""
import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def create_advanced_analytics(best_trials, study_name):
    """
    Creates the advanced analytics section with multiple tabs.
    
    Args:
        best_trials: List of best trial data
        study_name: Name of the study
        
    Returns:
        Component with advanced analytics
    """
    return html.Div([
        html.Div([
            html.H4([
                html.I(className="bi bi-graph-up me-2"),
                "Advanced Analytics"
            ], className="retro-card-title d-flex align-items-center"),
        ], className="retro-card-header p-3"),
        
        html.Div([
            # Tabs for different advanced analytics
            dbc.Tabs([
                dbc.Tab(
                    create_exploration_viz_tab(best_trials),
                    label="EXPLORATION SPACE",
                    tab_id="tab-exploration-viz",
                    className="retro-tabs",
                    activeLabelClassName="text-cyan-300"
                ),
                dbc.Tab(
                    create_metric_distribution_tab(best_trials),
                    label="METRIC DISTRIBUTIONS",
                    tab_id="tab-metric-distribution",
                    className="retro-tabs",
                    activeLabelClassName="text-cyan-300"
                ),
                dbc.Tab(
                    create_trade_analysis_tab(best_trials),
                    label="TRADE ANALYSIS",
                    tab_id="tab-trade-analysis",
                    className="retro-tabs",
                    activeLabelClassName="text-cyan-300"
                ),
                dbc.Tab(
                    create_optimization_history_tab(study_name),
                    label="OPTIMIZATION HISTORY",
                    tab_id="tab-optimization-history",
                    className="retro-tabs",
                    activeLabelClassName="text-cyan-300"
                ),
            ], id="advanced-tabs", className="retro-tabs"),
            
        ], className="retro-card-body p-3"),
    ], className="retro-card mb-4")


def create_exploration_viz_tab(best_trials):
    """Creates the exploration visualization tab."""
    
    if len(best_trials) < 5:
        return html.Div("Not enough trials for exploration visualization", 
                        className="text-center text-muted p-5")
    
    return html.Div([
        html.P([
            html.I(className="bi bi-info-circle me-2"),
            "This visualization shows how the parameter space was explored during optimization."
        ], className="text-muted mb-4"),
        
        html.Div([
            html.Button([
                html.I(className="bi bi-arrow-repeat me-2"),
                "Generate Exploration Visualization"
            ], id="generate-exploration-viz-btn", className="retro-button"),
        ], className="text-center mb-4"),
        
        html.Div(id="exploration-viz-container", className="mt-4")
    ], className="p-3")


def create_metric_distribution_tab(best_trials):
    """Creates the metric distribution visualization tab."""
    
    if len(best_trials) < 3:
        return html.Div("Not enough trials for metric distribution analysis", 
                        className="text-center text-muted p-5")
    
    # Create distribution histogram for metrics
    metrics = ['roi', 'win_rate', 'max_drawdown', 'profit_factor', 'total_trades']
    metric_labels = {
        'roi': 'ROI',
        'win_rate': 'Win Rate',
        'max_drawdown': 'Max Drawdown',
        'profit_factor': 'Profit Factor',
        'total_trades': 'Total Trades'
    }
    
    # Extract metrics data
    metrics_data = {metric: [] for metric in metrics}
    for trial in best_trials:
        for metric in metrics:
            if metric in trial:
                # Convert percentages for display
                if metric in ['roi', 'win_rate', 'max_drawdown']:
                    metrics_data[metric].append(trial[metric] * 100)
                else:
                    metrics_data[metric].append(trial[metric])
    
    fig = make_subplots(rows=3, cols=2, subplot_titles=[metric_labels.get(m, m) for m in metrics] + [''])
    
    # Add histograms for each metric
    row, col = 1, 1
    colors = ['#22D3EE', '#4ADE80', '#F87171', '#A78BFA', '#FBBF24']
    
    for i, (metric, values) in enumerate(metrics_data.items()):
        if values:
            fig.add_trace(
                go.Histogram(
                    x=values,
                    nbinsx=20,
                    marker_color=colors[i % len(colors)],
                    name=metric_labels.get(metric, metric)
                ),
                row=row, col=col
            )
            
            # Update position for next plot
            col += 1
            if col > 2:
                col = 1
                row += 1
    
    # Update layout
    fig.update_layout(
        height=700,
        margin=dict(l=10, r=10, t=50, b=10),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.3)',
        bargap=0.2,
        showlegend=False
    )
    
    # Update axes labels
    for i, metric in enumerate(metrics):
        row = i // 2 + 1
        col = i % 2 + 1
        if metric in ['roi', 'win_rate', 'max_drawdown']:
            fig.update_xaxes(title_text="%", row=row, col=col)
        else:
            fig.update_xaxes(title_text="Value", row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)
    
    return html.Div([
        html.P([
            html.I(className="bi bi-info-circle me-2"),
            "These histograms show the distribution of key performance metrics across all trials."
        ], className="text-muted mb-4"),
        
        dcc.Graph(
            figure=fig,
            id='metric-distribution-graph',
            className='retro-graph'
        )
    ], className="p-3")


def create_trade_analysis_tab(best_trials):
    """Creates the trade analysis visualization tab."""
    
    # Find the best trial with trade history
    best_trial = next(
        (t for t in best_trials if 'trade_history' in t and t['trade_history']), 
        None
    )
    
    if not best_trial:
        return html.Div("No trade history data available", className="text-center text-muted p-5")
    
    trade_history = best_trial.get('trade_history', [])
    
    # Create trade analysis visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "P&L Distribution", 
            "Cumulative P&L", 
            "Trade Duration",
            "Win/Loss by Type"
        ],
        specs=[
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "bar"}]
        ]
    )
    
    # Extract trade data
    pnl_values = [trade.get('pnl_pct', 0) for trade in trade_history]
    pnl_abs = [trade.get('pnl_abs', 0) for trade in trade_history]
    durations = [trade.get('duration', 0) for trade in trade_history]
    
    # Count trades by type and result
    trade_types = {}
    for trade in trade_history:
        trade_type = trade.get('type', 'unknown')
        success = trade.get('success', False)
        
        if trade_type not in trade_types:
            trade_types[trade_type] = {'win': 0, 'loss': 0}
            
        if success:
            trade_types[trade_type]['win'] += 1
        else:
            trade_types[trade_type]['loss'] += 1
    
    # 1. P&L Distribution
    fig.add_trace(
        go.Histogram(
            x=pnl_values,
            nbinsx=20,
            marker_color='#22D3EE',
            name="P&L Distribution"
        ),
        row=1, col=1
    )
    
    # 2. Cumulative P&L
    cum_pnl = np.cumsum(pnl_abs)
    fig.add_trace(
        go.Scatter(
            x=list(range(len(cum_pnl))),
            y=cum_pnl,
            mode='lines',
            line=dict(color='#4ADE80', width=2),
            name="Cumulative P&L"
        ),
        row=1, col=2
    )
    
    # 3. Trade Duration
    fig.add_trace(
        go.Histogram(
            x=durations,
            nbinsx=20,
            marker_color='#A78BFA',
            name="Trade Duration"
        ),
        row=2, col=1
    )
    
    # 4. Win/Loss by Type
    types = list(trade_types.keys())
    wins = [trade_types[t]['win'] for t in types]
    losses = [trade_types[t]['loss'] for t in types]
    
    fig.add_trace(
        go.Bar(
            x=types,
            y=wins,
            name="Wins",
            marker_color='#4ADE80'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=types,
            y=losses,
            name="Losses",
            marker_color='#F87171'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=700,
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
        ),
        barmode='group'
    )
    
    return html.Div([
        html.P([
            html.I(className="bi bi-info-circle me-2"),
            "This analysis shows detailed trade statistics from the best strategy's backtest."
        ], className="text-muted mb-4"),
        
        dcc.Graph(
            figure=fig,
            id='trade-analysis-graph',
            className='retro-graph'
        )
    ], className="p-3")


def create_optimization_history_tab(study_name):
    """Creates the optimization history visualization tab."""
    
    return html.Div([
        html.P([
            html.I(className="bi bi-info-circle me-2"),
            "This visualization shows how optimization scores evolved over time."
        ], className="text-muted mb-4"),
        
        html.Div(id="optimization-history-container", className="mt-4"),
        
        # Hidden div to trigger callback
        html.Div(study_name, id="optimization-history-study-name", style={"display": "none"})
    ], className="p-3")


def register_advanced_analytics_callbacks(app, central_logger=None):
    """
    Register callbacks for the advanced analytics component.
    
    Args:
        app: Dash app instance
        central_logger: Optional logger instance
    """
    @app.callback(
        Output('exploration-viz-container', 'children'),
        Input('generate-exploration-viz-btn', 'n_clicks'),
        State('optimization-details-data', 'data'),
        prevent_initial_call=True
    )
    def generate_exploration_visualization(n_clicks, details_data):
        """Generate exploration visualization when button is clicked."""
        if not n_clicks or not details_data:
            return html.Div("Click the button to generate visualization", 
                           className="text-center text-muted p-4")
            
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
            
            if not optimization_results or 'all_trials' not in optimization_results:
                return html.Div("Not enough data for visualization", className="text-center text-muted p-4")
                
            all_trials = optimization_results.get('all_trials', [])
            
            if len(all_trials) < 10:
                return html.Div("Not enough trials for visualization (minimum 10 required)",
                               className="text-center text-muted p-4")
            
            # Extract parameters and scores
            param_data = []
            scores = []
            trial_ids = []
            
            for trial in all_trials:
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
                    trial_ids.append(trial.get('trial_id', 0))
            
            if len(param_data) < 10:
                return html.Div("Not enough numerical parameters for visualization",
                               className="text-center text-muted p-4")
            
            # Create DataFrame
            df = pd.DataFrame(param_data)
            
            # Standardize the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(df)
            
            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=3)
            data_pca = pca.fit_transform(data_scaled)
            
            # Apply t-SNE for 2D visualization
            tsne = TSNE(n_components=2, random_state=42)
            data_tsne = tsne.fit_transform(data_scaled)
            
            # Create multi-view visualization
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "scatter3d"}, {"type": "scatter"}]],
                subplot_titles=["3D Parameter Space (PCA)", "2D Parameter Space (t-SNE)"]
            )
            
            # Normalize scores for color scale
            norm_scores = np.array(scores)
            score_range = np.max(norm_scores) - np.min(norm_scores)
            if score_range > 0:
                norm_scores = (norm_scores - np.min(norm_scores)) / score_range
            
            # 3D PCA plot
            fig.add_trace(
                go.Scatter3d(
                    x=data_pca[:, 0],
                    y=data_pca[:, 1],
                    z=data_pca[:, 2],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=norm_scores,
                        colorscale='Viridis',
                        opacity=0.8,
                        colorbar=dict(
                            title="Score",
                            x=0.45
                        )
                    ),
                    text=[f"Trial {id}<br>Score: {score:.4f}" for id, score in zip(trial_ids, scores)],
                    hoverinfo="text"
                ),
                row=1, col=1
            )
            
            # 2D t-SNE plot
            fig.add_trace(
                go.Scatter(
                    x=data_tsne[:, 0],
                    y=data_tsne[:, 1],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=norm_scores,
                        colorscale='Viridis',
                        opacity=0.8,
                        colorbar=dict(
                            title="Score",
                            x=1.0
                        )
                    ),
                    text=[f"Trial {id}<br>Score: {score:.4f}" for id, score in zip(trial_ids, scores)],
                    hoverinfo="text"
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=700,
                margin=dict(l=10, r=10, t=50, b=10),
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,23,42,0.3)',
                scene=dict(
                    xaxis_title="PC1",
                    yaxis_title="PC2",
                    zaxis_title="PC3",
                    aspectmode='cube'
                ),
                xaxis=dict(
                    title="t-SNE Dimension 1"
                ),
                yaxis=dict(
                    title="t-SNE Dimension 2"
                ),
                showlegend=False
            )
            
            return dcc.Graph(
                figure=fig,
                id='exploration-space-graph',
                className='retro-graph'
            )
            
        except Exception as e:
            if central_logger:
                central_logger.error(f"Error generating exploration visualization: {str(e)}")
            return html.Div(f"Error generating visualization: {str(e)}", 
                           className="text-center text-muted p-4")
    
    
    @app.callback(
        Output('optimization-history-container', 'children'),
        Input('optimization-history-study-name', 'children'),
        prevent_initial_call=True
    )
    def generate_optimization_history(study_name):
        """Generate optimization history visualization."""
        if not study_name:
            return html.Div("Study name not provided", className="text-center text-muted p-4")
            
        try:
            from core.study.study_manager import IntegratedStudyManager
            study_manager = IntegratedStudyManager("studies")
            optimization_results = study_manager.get_optimization_results(study_name)
            
            if not optimization_results or 'all_trials' not in optimization_results:
                return html.Div("No optimization history data available", 
                               className="text-center text-muted p-4")
                
            all_trials = optimization_results.get('all_trials', [])
            
            if len(all_trials) < 2:
                return html.Div("Not enough trials for history visualization",
                               className="text-center text-muted p-4")
            
            # Extract trial data
            trial_ids = []
            scores = []
            best_scores = []
            current_best = float('-inf')
            
            for i, trial in enumerate(all_trials):
                trial_id = trial.get('trial_id', i)
                score = trial.get('score', float('-inf'))
                
                trial_ids.append(trial_id)
                scores.append(score)
                
                if score > current_best:
                    current_best = score
                best_scores.append(current_best)
            
            # Create visualization
            fig = go.Figure()
            
            # Add individual trial scores
            fig.add_trace(go.Scatter(
                x=list(range(len(scores))),
                y=scores,
                mode='markers',
                name='Trial Score',
                marker=dict(
                    size=6,
                    color=scores,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Score")
                ),
                text=[f"Trial {id}<br>Score: {score:.4f}" for id, score in zip(trial_ids, scores)],
                hoverinfo="text"
            ))
            
            # Add best score line
            fig.add_trace(go.Scatter(
                x=list(range(len(best_scores))),
                y=best_scores,
                mode='lines',
                name='Best Score',
                line=dict(color='#F87171', width=2)
            ))
            
            # Update layout
            fig.update_layout(
                title="Optimization History",
                xaxis_title="Trial Number",
                yaxis_title="Score",
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
                id='optimization-history-graph',
                className='retro-graph'
            )
            
        except Exception as e:
            if central_logger:
                central_logger.error(f"Error generating optimization history: {str(e)}")
            return html.Div(f"Error generating optimization history: {str(e)}",
                           className="text-center text-muted p-4")