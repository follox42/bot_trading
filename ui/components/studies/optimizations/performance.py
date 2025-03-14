"""
Performance visualization components for the optimization details panel.
Includes performance charts and best trial metrics.
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def create_performance_visualizations(best_trials, best_trial_id):
    """
    Creates visualizations of performance metrics for the best trials.
    
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
    
    for trial in best_trials[:8]:  # Limit to 8 for readability
        trial_ids.append(trial.get('trial_id', 0))
        scores.append(trial.get('score', 0))
        metrics = trial.get('metrics', {})
        rois.append(metrics.get('roi', 0) * 100)
        win_rates.append(metrics.get('win_rate', 0) * 100)
        drawdowns.append(metrics.get('max_drawdown', 0) * 100)
    
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
    
    # Create radar chart for the best trial
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


def create_best_trial_details(best_trials, best_trial_id):
    """
    Creates a detailed view of the best trial's performance metrics.
    
    Args:
        best_trials: List of the best trials
        best_trial_id: ID of the best trial
    
    Returns:
        Best trial details component
    """
    best_trial = None
    for trial in best_trials:
        if trial.get('trial_id', 0) == best_trial_id:
            best_trial = trial
            break
    
    if not best_trial:
        return html.Div("Details of the best trial not available", className="text-center text-muted p-3")
    
    metrics = best_trial.get('metrics', {})
    
    # Format metrics for display
    roi = metrics.get('roi', 0) * 100
    win_rate = metrics.get('win_rate', 0) * 100
    max_drawdown = metrics.get('max_drawdown', 0) * 100
    profit_factor = metrics.get('profit_factor', 0)
    total_trades = int(metrics.get('total_trades', 0))
    avg_profit = metrics.get('avg_profit', 0) * 100
    
    # Additional metrics if available
    sharpe = metrics.get('sharpe', None)
    sortino = metrics.get('sortino', None)
    calmar = metrics.get('calmar', None)
    max_consecutive_losses = metrics.get('max_consecutive_losses', None)
    trades_per_day = metrics.get('trades_per_day', None)
    
    # Metrics to display in cards
    primary_metrics = [
        {"name": "ROI", "value": f"{roi:.2f}%", "color": "success" if roi > 0 else "danger", "icon": "bi-graph-up-arrow"},
        {"name": "Win Rate", "value": f"{win_rate:.2f}%", "color": "cyan-300", "icon": "bi-trophy"},
        {"name": "Max Drawdown", "value": f"{max_drawdown:.2f}%", "color": "danger", "icon": "bi-arrow-down-right"},
        {"name": "Profit Factor", "value": f"{profit_factor:.2f}", "color": "success" if profit_factor > 1 else "warning", "icon": "bi-calculator"},
        {"name": "Total Trades", "value": f"{total_trades}", "color": "white", "icon": "bi-shuffle"},
        {"name": "Avg. Profit", "value": f"{avg_profit:.2f}%", "color": "success" if avg_profit > 0 else "danger", "icon": "bi-cash-stack"},
    ]
    
    # Add additional metrics if available
    if sharpe is not None:
        primary_metrics.append({"name": "Sharpe Ratio", "value": f"{sharpe:.2f}", "color": "cyan-300", "icon": "bi-bar-chart-line"})
    
    if sortino is not None:
        primary_metrics.append({"name": "Sortino Ratio", "value": f"{sortino:.2f}", "color": "cyan-300", "icon": "bi-bar-chart-line"})
    
    if calmar is not None:
        primary_metrics.append({"name": "Calmar Ratio", "value": f"{calmar:.2f}", "color": "cyan-300", "icon": "bi-bar-chart-line"})
    
    if max_consecutive_losses is not None:
        primary_metrics.append({"name": "Max Cons. Losses", "value": f"{int(max_consecutive_losses)}", "color": "warning", "icon": "bi-exclamation-triangle"})
    
    if trades_per_day is not None:
        primary_metrics.append({"name": "Trades/Day", "value": f"{trades_per_day:.2f}", "color": "white", "icon": "bi-calendar-check"})
        
    # Create metric cards
    metric_cards = []
    for i, metric in enumerate(primary_metrics):
        metric_cards.append(
            dbc.Col(
                dbc.Card([
                    html.Div([
                        html.I(className=f"bi {metric['icon']} me-2"),
                        metric["name"]
                    ], className="text-muted small mb-1"),
                    html.Div(metric["value"], className=f"h4 mb-0 text-{metric['color']}")
                ], className="bg-dark p-3 h-100 text-center retro-card border-secondary"),
                xs=6, md=4, xl=3, className="mb-3"
            )
        )
    
    # Create gauge charts for key metrics
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=("Win Rate", "ROI", "Profit Factor")
    )
    
    # Win Rate gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=win_rate,
            title={'text': "Win Rate (%)"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(34, 211, 238, 0.8)"},
                'bgcolor': "rgba(0, 0, 0, 0)",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(255, 0, 0, 0.3)'},
                    {'range': [30, 50], 'color': 'rgba(255, 255, 0, 0.3)'},
                    {'range': [50, 100], 'color': 'rgba(0, 255, 0, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            },
            number={'suffix': "%", 'font': {'color': '#22D3EE'}}
        ),
        row=1, col=1
    )
    
    # ROI gauge
    max_roi = max(100, roi * 1.5) if roi > 0 else 100
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=roi,
            title={'text': "ROI (%)"},
            gauge={
                'axis': {'range': [0, max_roi], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(74, 222, 128, 0.8)" if roi > 0 else "rgba(248, 113, 113, 0.8)"},
                'bgcolor': "rgba(0, 0, 0, 0)",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, max_roi/3], 'color': 'rgba(255, 255, 0, 0.3)'},
                    {'range': [max_roi/3, max_roi], 'color': 'rgba(0, 255, 0, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            },
            number={'suffix': "%", 'font': {'color': '#4ADE80' if roi > 0 else '#F87171'}}
        ),
        row=1, col=2
    )
    
    # Profit Factor gauge
    max_pf = max(3, profit_factor * 1.2)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=profit_factor,
            title={'text': "Profit Factor"},
            gauge={
                'axis': {'range': [0, max_pf], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(167, 139, 250, 0.8)"},
                'bgcolor': "rgba(0, 0, 0, 0)",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 1], 'color': 'rgba(255, 0, 0, 0.3)'},
                    {'range': [1, 2], 'color': 'rgba(255, 255, 0, 0.3)'},
                    {'range': [2, max_pf], 'color': 'rgba(0, 255, 0, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 1
                }
            },
            number={'font': {'color': '#A78BFA'}}
        ),
        row=1, col=3
    )
    
    # Update gauge layout
    fig.update_layout(
        height=300,
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.9)",
        plot_bgcolor="rgba(15, 23, 42, 0.9)",
        font=dict(family="'Share Tech Mono', monospace", color="#D1D5DB"),
        margin=dict(l=30, r=30, t=30, b=0)
    )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-trophy-fill me-2 text-warning"),
                f"Best Trial (#{best_trial_id})"
            ], className="text-cyan-300 mb-0")
        ], className="bg-dark border-secondary"),
        
        dbc.CardBody([
            # Metric cards
            dbc.Row(metric_cards),
            
            # Gauges
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=fig,
                        config={'displayModeBar': False},
                        className="mt-3"
                    )
                ])
            ]),
            
            # Action buttons
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        [html.I(className="bi bi-backpack me-2"), "Backtest This Strategy"],
                        id={"type": "btn-backtest-trial", "trial_id": best_trial_id},
                        className="retro-button me-2 mb-2 w-100"
                    ),
                ], xs=12, md=6),
                
                dbc.Col([
                    dbc.Button(
                        [html.I(className="bi bi-save me-2"), "Use This Strategy"],
                        id={"type": "btn-use-trial", "trial_id": best_trial_id},
                        className="retro-button secondary w-100 mb-2"
                    ),
                ], xs=12, md=6),
            ], className="mt-3"),
        ], className="bg-dark p-3"),
    ], className="retro-card shadow mt-4")


def create_simulated_equity_curve(roi, max_drawdown, total_trades):
    """
    Creates a simulated equity curve based on the trial metrics.
    
    Args:
        roi: Return on investment (%)
        max_drawdown: Maximum drawdown (%)
        total_trades: Total number of trades
    
    Returns:
        Plotly figure with simulated equity curve
    """
    # Generate simulated equity curve
    n_points = min(max(50, total_trades * 2), 200)  # Reasonable number of points
    
    # Create a random walk with drift that achieves the target ROI and max drawdown
    np.random.seed(42)  # For reproducibility
    
    # Target final value
    target = 1 + (roi / 100)
    
    # Generate random increments
    if roi > 0:
        drift = roi / 100 / n_points
        volatility = max_drawdown / 100 / 2.5  # Scale volatility based on drawdown
    else:
        drift = roi / 100 / n_points
        volatility = max(abs(roi), max_drawdown) / 100 / 2.5
    
    # Generate random walk
    increments = np.random.normal(drift, volatility, n_points)
    equity_curve = np.cumprod(1 + increments)
    
    # Scale to hit target ROI
    equity_curve = equity_curve * (target / equity_curve[-1])
    
    # Ensure max drawdown is respected
    peak = 1.0
    drawdowns = []
    for i in range(len(equity_curve)):
        if equity_curve[i] > peak:
            peak = equity_curve[i]
        drawdown = (peak - equity_curve[i]) / peak * 100
        drawdowns.append(drawdown)
    
    actual_max_dd = max(drawdowns)
    if actual_max_dd < max_drawdown:
        # Add an artificial drawdown
        dd_point = np.random.randint(n_points // 3, 2 * n_points // 3)
        dd_depth = equity_curve[dd_point] * (max_drawdown / 100)
        equity_curve[dd_point] -= dd_depth
        
        # Smooth the curve after the drawdown
        recovery_rate = np.linspace(0, 1, n_points - dd_point)
        equity_curve[dd_point:] = equity_curve[dd_point:] + dd_depth * recovery_rate
    
    # Create timestamps
    timestamps = [f"Day {i+1}" for i in range(n_points)]
    
    # Create drawdown curve
    drawdown_curve = np.zeros(n_points)
    peak = equity_curve[0]
    for i in range(n_points):
        if equity_curve[i] > peak:
            peak = equity_curve[i]
        drawdown_curve[i] = (peak - equity_curve[i]) / peak * 100
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve", "Drawdown (%)")
    )
    
    # Add equity curve
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=equity_curve,
            mode='lines',
            name='Equity',
            line=dict(color='#22D3EE', width=2),
            fill='tozeroy',
            fillcolor='rgba(34, 211, 238, 0.1)'
        ),
        row=1, col=1
    )
    
    # Add horizontal line at initial value
    fig.add_trace(
        go.Scatter(
            x=[timestamps[0], timestamps[-1]],
            y=[1, 1],
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add drawdown curve
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=drawdown_curve,
            mode='lines',
            name='Drawdown',
            line=dict(color='#F87171', width=2),
            fill='tozeroy',
            fillcolor='rgba(248, 113, 113, 0.1)'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.9)",
        plot_bgcolor="rgba(15, 23, 42, 0.9)",
        font=dict(family="'Share Tech Mono', monospace", color="#D1D5DB"),
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        xaxis2=dict(title="Time")
    )
    
    # Update y-axes
    fig.update_yaxes(
        title="Equity Multiplier",
        gridcolor="rgba(51, 65, 85, 0.6)",
        row=1, col=1
    )
    
    fig.update_yaxes(
        title="Drawdown (%)",
        gridcolor="rgba(51, 65, 85, 0.6)",
        row=2, col=1
    )
    
    return fig