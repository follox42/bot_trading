"""
Basic visualization components for the optimization details panel.
Includes performance visualizations and result charts.
"""
import dash
from dash import dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from dash import html


def create_performance_visualizations(best_trials, best_trial_id):
    """
    Creates basic performance visualizations.
    
    Args:
        best_trials: List of the best trials
        best_trial_id: ID of the best trial
    
    Returns:
        Performance visualizations component
    """
    if not best_trials:
        return html.Div("Insufficient data for visualizations", className="text-center text-muted p-3")
    
    # Prepare data
    trial_ids = []
    scores = []
    rois = []
    win_rates = []
    drawdowns = []
    profit_factors = []
    
    for trial in best_trials[:8]:  # Limit to 8 for readability
        trial_ids.append(trial.get('trial_id', 0))
        scores.append(trial.get('score', 0))
        metrics = trial.get('metrics', {})
        rois.append(metrics.get('roi', 0) * 100)
        win_rates.append(metrics.get('win_rate', 0) * 100)
        drawdowns.append(metrics.get('max_drawdown', 0) * 100)
        profit_factors.append(metrics.get('profit_factor', 0))
    
    # Create comparison figure
    comparison_fig = go.Figure()
    
    # Bars for scores
    comparison_fig.add_trace(go.Bar(
        x=trial_ids,
        y=scores,
        name='Score',
        marker_color='rgb(34, 211, 238)',
        opacity=0.8
    ))
    
    # Line for ROI
    comparison_fig.add_trace(go.Scatter(
        x=trial_ids,
        y=rois,
        name='ROI (%)',
        marker_color='rgb(74, 222, 128)',
        mode='lines+markers',
        line=dict(width=2)
    ))
    
    # Highlight the best trial
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
            name='Best trial',
            hoverinfo='skip'
        ))
    
    # Layout
    comparison_fig.update_layout(
        title='Comparison of Best Trials',
        xaxis_title='Trial ID',
        yaxis_title='Value',
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
    
    # Radar chart for the best trial
    best_trial = None
    for trial in best_trials:
        if trial.get('trial_id') == best_trial_id:
            best_trial = trial
            break
    
    radar_fig = None
    if best_trial:
        metrics = best_trial.get('metrics', {})
        
        # Normalize metrics for the radar
        radar_data = [
            min(1.0, max(0, metrics.get('roi', 0) * 5)),  # ROI normalized (up to 20%)
            metrics.get('win_rate', 0),                   # Win rate already between 0 and 1
            1 - min(1.0, metrics.get('max_drawdown', 0) * 5),  # Reversed drawdown
            min(1.0, metrics.get('profit_factor', 0) / 5),      # Profit factor normalized
            min(1.0, metrics.get('total_trades', 0) / 100)      # Trades normalized
        ]
        
        category_names = ['ROI', 'Win Rate', 'Min Drawdown', 'Profit Factor', 'Trade Volume']
        
        radar_fig = go.Figure()
        
        radar_fig.add_trace(go.Scatterpolar(
            r=radar_data,
            theta=category_names,
            fill='toself',
            name='Best trial',
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
        # Comparison chart
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="bi bi-bar-chart-fill me-2"),
                    "Performance Comparison"
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
        
        # Radar chart for the best trial
        dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="bi bi-bullseye me-2"),
                    f"Best Trial Profile (#{best_trial_id})"
                ], className="text-cyan-300 mb-0")
            ], className="bg-dark border-secondary"),
            
            dbc.CardBody(
                dcc.Graph(
                    figure=radar_fig,
                    config={'displayModeBar': False},
                    className="retro-graph"
                ) if radar_fig else html.Div("Insufficient data", className="text-center text-muted p-3"),
                className="bg-dark p-0 pt-2"
            ),
        ], className="retro-card shadow"),
    ])


def create_results_chart(best_trials, best_trial_id):
    """
    Creates a more detailed results chart.
    
    Args:
        best_trials: List of the best trials
        best_trial_id: ID of the best trial
    
    Returns:
        Results chart component
    """
    # If no trials, return a message
    if not best_trials:
        return html.Div([
            html.I(className="bi bi-exclamation-circle text-warning me-2"),
            "No trials available to display a chart."
        ], className="text-center p-5")
    
    # Create a DataFrame for the chart
    data = []
    for trial in best_trials:
        metrics = trial.get('metrics', {})
        data.append({
            'trial_id': trial.get('trial_id', 0),
            'score': trial.get('score', 0),
            'roi': metrics.get('roi', 0) * 100 if metrics.get('roi') is not None else 0,
            'win_rate': metrics.get('win_rate', 0) * 100 if metrics.get('win_rate') is not None else 0,
            'total_trades': metrics.get('total_trades', 0),
            'max_drawdown': metrics.get('max_drawdown', 0) * 100 if metrics.get('max_drawdown') is not None else 0,
            'profit_factor': metrics.get('profit_factor', 0),
            'is_best': trial.get('trial_id', 0) == best_trial_id
        })
    
    df = pd.DataFrame(data)
    
    # Create scatter plot with color mapping
    fig = px.scatter(
        df, 
        x='roi', 
        y='win_rate', 
        size='total_trades',
        color='score',
        hover_name='trial_id',
        color_continuous_scale='Viridis',
        size_max=30,
        labels={
            'roi': 'ROI (%)',
            'win_rate': 'Win Rate (%)',
            'total_trades': 'Number of trades',
            'score': 'Score'
        },
        height=400
    )
    
    # Highlight the best trial
    best_trial = df[df['is_best']]
    if not best_trial.empty:
        fig.add_trace(
            go.Scatter(
                x=best_trial['roi'],
                y=best_trial['win_rate'],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=20,
                    color='gold',
                    line=dict(width=2, color='black')
                ),
                name='Best trial',
                hoverinfo='name+text',
                text=[f"Trial #{best_trial_id}<br>Score: {best_trial['score'].values[0]:.4f}"]
            )
        )
    
    # Enhance the appearance for retro look
    fig.update_layout(
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        paper_bgcolor="rgba(15, 23, 42, 0.9)",
        font=dict(
            color="#e2e8f0",
            family="'Share Tech Mono', monospace"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor="rgba(34, 211, 238, 0.3)"
        ),
        margin=dict(l=40, r=20, t=40, b=40),
        title={
            'text': "Analysis of Best Trials",
            'font': {'color': '#22d3ee', 'size': 18, 'family': "'VT323', 'Share Tech Mono', monospace"}
        },
        xaxis=dict(
            gridcolor="rgba(51, 65, 85, 0.6)",
            zerolinecolor="rgba(34, 211, 238, 0.4)",
            title_font={"color": "#22d3ee"},
            tickfont={"color": "#d1d5db"}
        ),
        yaxis=dict(
            gridcolor="rgba(51, 65, 85, 0.6)",
            zerolinecolor="rgba(34, 211, 238, 0.4)",
            title_font={"color": "#22d3ee"},
            tickfont={"color": "#d1d5db"}
        ),
        coloraxis_colorbar=dict(
            title=dict(
                text="Score",
                side="right",
                font={"color": "#22d3ee", "family": "'Share Tech Mono', monospace"}
            ),
            tickcolor="#d1d5db",
            tickfont=dict(color="#d1d5db", family="'Share Tech Mono', monospace"),
            outlinecolor="rgba(34, 211, 238, 0.3)"
        ),
        shapes=[
            # Add retro border with glow effect
            dict(
                type="rect",
                xref="paper", yref="paper",
                x0=0, y0=0.99, x1=1, y1=1,
                line=dict(color="rgba(34, 211, 238, 0.5)", width=2),
                fillcolor="rgba(34, 211, 238, 0.1)",
                layer="below"
            )
        ]
    )
    
    # Create a second figure for the distribution of scores
    histogram_fig = px.histogram(
        df, 
        x='score', 
        nbins=20,
        labels={'score': 'Score', 'count': 'Number of trials'},
        title="Distribution of Scores",
        color_discrete_sequence=['#22d3ee'],
        template="plotly_dark",
        height=250
    ).update_layout(
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        paper_bgcolor="rgba(15, 23, 42, 0.9)",
        font=dict(color="#e2e8f0", family="'Share Tech Mono', monospace"),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(
            gridcolor="rgba(51, 65, 85, 0.6)",
            title_font={"color": "#22d3ee"},
            tickfont={"color": "#d1d5db"}
        ),
        yaxis=dict(
            gridcolor="rgba(51, 65, 85, 0.6)",
            title_font={"color": "#22d3ee"},
            tickfont={"color": "#d1d5db"}
        ),
        bargap=0.05,
        title={
            'text': "Distribution of Scores",
            'font': {'color': '#22d3ee', 'size': 16, 'family': "'VT323', 'Share Tech Mono', monospace"}
        }
    )
    
    return dbc.Card([
        dbc.CardBody([
            dcc.Graph(
                figure=fig, 
                className="mb-4 retro-graph", 
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': [
                        'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d'
                    ]
                }
            ),
            
            # Retro separator with scan line
            html.Div(className="retro-graph-separator mb-4"),
            
            # Second graph with score histogram
            html.Div([
                dcc.Graph(
                    figure=histogram_fig,
                    className="retro-graph",
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': [
                            'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d'
                        ]
                    }
                )
            ])
        ], className="p-0")
    ], className="retro-card retro-graph-card bg-dark border border-cyan shadow-lg")