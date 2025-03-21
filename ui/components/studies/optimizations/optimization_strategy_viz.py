"""
Component for visualizing the best strategy structure and performance.
Shows trading rules, performance metrics, and equity curve.
"""
import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json

def create_strategy_visualization(best_trials, best_trial_id):
    """
    Creates the strategy visualization section.
    
    Args:
        best_trials: List of best trials
        best_trial_id: ID of the best trial
        
    Returns:
        Component with strategy visualization
    """
    # Find the best trial data
    best_trial = next((t for t in best_trials if t.get('trial_id') == best_trial_id), 
                      best_trials[0] if best_trials else None)
    
    if not best_trial:
        return html.Div("No strategy data available", className="text-center text-muted p-4")
    
    return html.Div([
        html.Div([
            html.H4([
                html.I(className="bi bi-gear-wide-connected me-2"),
                "Best Strategy Analysis"
            ], className="retro-card-title d-flex align-items-center"),
        ], className="retro-card-header p-3"),
        
        html.Div([
            # Trading Strategy Structure
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H5("Trading Rules", className="section-title"),
                        create_trading_blocks(best_trial)
                    ], className="trading-blocks-card retro-card mb-3 p-3"),
                ], md=7),
                
                dbc.Col([
                    html.Div([
                        html.H5("Risk Parameters", className="section-title"),
                        create_risk_parameters(best_trial)
                    ], className="risk-params-container retro-card mb-3 p-3"),
                ], md=5),
            ]),
            
            # Performance Visualization
            dbc.Row([
                dbc.Col([
                    html.H5("Performance Metrics", className="section-title"),
                    create_performance_metrics(best_trial)
                ], md=5),
                
                dbc.Col([
                    html.H5("Equity Curve", className="section-title"),
                    create_equity_curve(best_trial)
                ], md=7),
            ]),
            
        ], className="retro-card-body p-3"),
    ], className="retro-card mb-4")


def create_trading_blocks(trial_data):
    """Creates a visualization of the trading strategy blocks/rules."""
    buy_blocks = trial_data.get('buy_blocks', [])
    sell_blocks = trial_data.get('sell_blocks', [])
    
    if not buy_blocks and not sell_blocks:
        return html.Div("No trading blocks available", className="empty-blocks-message")
    
    blocks_html = []
    
    # Buy blocks
    if buy_blocks:
        blocks_html.append(html.H6("Buy Conditions:", className="text-green my-3"))
        
        for i, block in enumerate(buy_blocks):
            blocks_html.append(
                html.Div([
                    html.Div(f"Buy Block {i+1}", className="block-header"),
                    html.Div([
                        html.Div(condition, className="condition-item")
                        for condition in block.get('conditions', [])
                    ], className="block-conditions mt-2"),
                ], className="trading-block buy-block mb-3")
            )
    
    # Sell blocks
    if sell_blocks:
        blocks_html.append(html.H6("Sell Conditions:", className="text-red my-3"))
        
        for i, block in enumerate(sell_blocks):
            blocks_html.append(
                html.Div([
                    html.Div(f"Sell Block {i+1}", className="block-header"),
                    html.Div([
                        html.Div(condition, className="condition-item")
                        for condition in block.get('conditions', [])
                    ], className="block-conditions mt-2"),
                ], className="trading-block sell-block mb-3")
            )
    
    return html.Div(blocks_html)


def create_risk_parameters(trial_data):
    """Creates a display of risk management parameters."""
    params = trial_data.get('params', {})
    
    risk_params = [
        {"icon": "bi-coin", "label": "Position Size", "value": f"{params.get('position_size', 0) * 100:.2f}%"},
        {"icon": "bi-shield", "label": "Stop Loss", "value": f"{params.get('sl_pct', 0) * 100:.2f}%"},
        {"icon": "bi-graph-up-arrow", "label": "Take Profit", "value": f"{params.get('tp_multiplier', 1) * params.get('sl_pct', 0) * 100:.2f}%"},
        {"icon": "bi-speedometer", "label": "Leverage", "value": f"{params.get('leverage', 1)}x"},
        {"icon": "bi-gear", "label": "Risk Mode", "value": params.get('risk_mode', 'fixed').capitalize()},
    ]
    
    # Add risk mode specific parameters
    if params.get('risk_mode') == 'atr_based':
        risk_params.append({"icon": "bi-activity", "label": "ATR Period", "value": params.get('atr_period', 14)})
        risk_params.append({"icon": "bi-arrows-expand", "label": "ATR Multiplier", "value": params.get('atr_multiplier', 1.5)})
    elif params.get('risk_mode') == 'volatility_based':
        risk_params.append({"icon": "bi-activity", "label": "Volatility Period", "value": params.get('vol_period', 20)})
        risk_params.append({"icon": "bi-arrows-expand", "label": "Volatility Multiplier", "value": params.get('vol_multiplier', 1.0)})
    
    return html.Div([
        html.Div([
            html.Div(className=f"risk-icon bi {param['icon']}"),
            html.Div([
                html.Div(param['label'], className="param-label"),
                html.Div(param['value'], className="param-value highlight-value")
            ])
        ], className="risk-param-item")
        for param in risk_params
    ])


def create_performance_metrics(trial_data):
    """Creates a performance metrics display."""
    metrics = [
        {"name": "ROI", "value": f"{trial_data.get('roi', 0) * 100:.2f}%"},
        {"name": "Win Rate", "value": f"{trial_data.get('win_rate', 0) * 100:.2f}%"},
        {"name": "Total Trades", "value": trial_data.get('total_trades', 0)},
        {"name": "Max Drawdown", "value": f"{trial_data.get('max_drawdown', 0) * 100:.2f}%"},
        {"name": "Profit Factor", "value": f"{trial_data.get('profit_factor', 0):.2f}"},
        {"name": "Avg Profit/Trade", "value": f"{trial_data.get('avg_profit_per_trade', 0) * 100:.2f}%"},
    ]
    
    return dbc.Row([
        dbc.Col([
            html.Div([
                html.Div(metric["name"], className="metric-label"),
                html.Div(metric["value"], className="metric-value")
            ], className="performance-metric mb-3")
        ], md=6)
        for metric in metrics
    ])


def create_equity_curve(trial_data):
    """Creates an equity curve visualization."""
    equity_data = trial_data.get('equity_curve', [])
    
    if not equity_data:
        return html.Div("No equity curve data available", className="empty-performance-message")
    
    # Create equity curve figure
    fig = go.Figure()
    
    # Add equity curve line
    fig.add_trace(go.Scatter(
        x=list(range(len(equity_data))),
        y=equity_data,
        mode='lines',
        name='Equity',
        line=dict(color='#22D3EE', width=2)
    ))
    
    # Add drawdown area
    max_equity = pd.Series(equity_data).cummax()
    drawdown = (pd.Series(equity_data) / max_equity - 1) * 100
    
    fig.add_trace(go.Scatter(
        x=list(range(len(equity_data))),
        y=drawdown,
        mode='lines',
        name='Drawdown',
        yaxis='y2',
        line=dict(color='#F87171', width=1.5)
    ))
    
    # Configure the layout
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
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
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(34,211,238,0.1)',
            tickfont=dict(size=10),
            title="Trade Time"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(34,211,238,0.1)',
            tickfont=dict(size=10),
            title="Equity"
        ),
        yaxis2=dict(
            title="Drawdown %",
            titlefont=dict(color="#F87171"),
            tickfont=dict(color="#F87171"),
            anchor="x",
            overlaying="y",
            side="right",
            rangemode="nonpositive"
        ),
        hovermode="x unified"
    )
    
    return dcc.Graph(
        figure=fig,
        id='equity-curve-graph',
        className='retro-graph'
    )


def register_strategy_viz_callbacks(app, central_logger=None):
    """
    Register callbacks for the strategy visualization component.
    
    Args:
        app: Dash app instance
        central_logger: Optional logger instance
    """
    # No callbacks needed for this component yet
    pass