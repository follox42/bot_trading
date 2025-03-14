"""
Best trial detail components for the optimization details panel.
Displays performance metrics and details of the best trial.
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def create_best_trial_details(best_trials, best_trial_id):
    """
    Creates a component with detailed information about the best trial.
    
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


def create_performance_summary(best_trial):
    """
    Creates a detailed performance summary for the best trial.
    
    Args:
        best_trial: The best trial
        
    Returns:
        Performance summary component
    """
    if not best_trial:
        return html.Div("Performance data not available", className="text-center text-muted p-3")
    
    metrics = best_trial.get('metrics', {})
    
    # Extract key metrics
    roi = metrics.get('roi', 0) * 100
    win_rate = metrics.get('win_rate', 0) * 100
    max_drawdown = metrics.get('max_drawdown', 0) * 100
    profit_factor = metrics.get('profit_factor', 0)
    total_trades = int(metrics.get('total_trades', 0))
    avg_profit = metrics.get('avg_profit', 0) * 100
    
    # Additional metrics
    winning_trades = int(total_trades * win_rate / 100) if total_trades > 0 else 0
    losing_trades = total_trades - winning_trades
    
    # Calculate additional derived metrics
    if 'max_profit' in metrics and 'max_loss' in metrics:
        max_profit = metrics.get('max_profit', 0) * 100
        max_loss = metrics.get('max_loss', 0) * 100
    else:
        max_profit = avg_profit * 3  # Estimate
        max_loss = avg_profit * -2 if avg_profit > 0 else avg_profit * 3  # Estimate
    
    # Create summary card
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-clipboard-data me-2"),
                "Performance Summary"
            ], className="text-cyan-300 mb-0")
        ], className="bg-dark border-secondary"),
        
        dbc.CardBody([
            dbc.Row([
                # General stats
                dbc.Col([
                    html.H6("General Statistics", className="mb-3 text-cyan-300"),
                    
                    html.Div([
                        html.Div("Total Return:", className="param-label"),
                        html.Div(f"{roi:.2f}%", className=f"param-value text-{'success' if roi > 0 else 'danger'}")
                    ], className="d-flex justify-content-between mb-2"),
                    
                    html.Div([
                        html.Div("Total Trades:", className="param-label"),
                        html.Div(f"{total_trades}", className="param-value")
                    ], className="d-flex justify-content-between mb-2"),
                    
                    html.Div([
                        html.Div("Winning Trades:", className="param-label"),
                        html.Div(f"{winning_trades} ({win_rate:.1f}%)", className="param-value text-success")
                    ], className="d-flex justify-content-between mb-2"),
                    
                    html.Div([
                        html.Div("Losing Trades:", className="param-label"),
                        html.Div(f"{losing_trades} ({100-win_rate:.1f}%)", className="param-value text-danger")
                    ], className="d-flex justify-content-between mb-2"),
                    
                    html.Div([
                        html.Div("Average Profit:", className="param-label"),
                        html.Div(f"{avg_profit:.2f}%", className=f"param-value text-{'success' if avg_profit > 0 else 'danger'}")
                    ], className="d-flex justify-content-between mb-2"),
                ], md=6),
                
                # Risk metrics
                dbc.Col([
                    html.H6("Risk Metrics", className="mb-3 text-cyan-300"),
                    
                    html.Div([
                        html.Div("Max Drawdown:", className="param-label"),
                        html.Div(f"{max_drawdown:.2f}%", className="param-value text-danger")
                    ], className="d-flex justify-content-between mb-2"),
                    
                    html.Div([
                        html.Div("Profit Factor:", className="param-label"),
                        html.Div(f"{profit_factor:.2f}", className=f"param-value text-{'success' if profit_factor > 1 else 'danger'}")
                    ], className="d-flex justify-content-between mb-2"),
                    
                    html.Div([
                        html.Div("Max Profit Trade:", className="param-label"),
                        html.Div(f"{max_profit:.2f}%", className="param-value text-success")
                    ], className="d-flex justify-content-between mb-2"),
                    
                    html.Div([
                        html.Div("Max Loss Trade:", className="param-label"),
                        html.Div(f"{max_loss:.2f}%", className="param-value text-danger")
                    ], className="d-flex justify-content-between mb-2"),
                    
                    html.Div([
                        html.Div("Payoff Ratio:", className="param-label"),
                        html.Div(
                            f"{abs(max_profit/max_loss):.2f}" if max_loss != 0 else "N/A", 
                            className="param-value text-cyan-300"
                        )
                    ], className="d-flex justify-content-between mb-2"),
                ], md=6),
            ]),
            
            # Create equity curve visualization
            html.H6("Example Equity Curve (Simulated)", className="mt-4 mb-3 text-cyan-300"),
            
            dcc.Graph(
                figure=create_simulated_equity_curve(roi, max_drawdown, total_trades),
                config={'displayModeBar': False},
                className="mb-3"
            )
        ], className="bg-dark p-3")
    ], className="retro-card shadow mb-4")


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