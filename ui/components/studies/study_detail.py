"""
Enhanced study details component with a comprehensive retro UI.
Provides detailed information, visualizations, and management capabilities for trading studies.
"""
from dash import html, dcc, Input, Output, State, callback, ALL, MATCH, callback_context
import dash_bootstrap_components as dbc
import dash
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback
import os

from logger.logger import LoggerType, CentralizedLogger
from ui.components.studies.strategy_viewer import create_strategy_viewer
from ui.components.studies.parameter_editor import create_parameter_editor
from ui.components.retro_ui import create_retro_toggle_button, create_collapsible_card

def create_study_detail(study_name, central_logger=None):
    """
    Creates an enhanced detailed view for a specific study with retro UI styling.
    
    Args:
        study_name: Name of the study to display
        central_logger: Centralized logger instance
    
    Returns:
        Component with the detailed study view
    """
    # Initialize logger
    ui_logger = None
    if central_logger:
        ui_logger = central_logger.get_logger("study_detail", LoggerType.UI)
        ui_logger.info(f"Displaying enhanced details for study: {study_name}")
    
    # Retrieve study details
    try:
        from simulator.study_manager import IntegratedStudyManager
        study_manager = IntegratedStudyManager("studies")
        
        # Check if study exists
        if not study_manager.study_exists(study_name):
            return html.Div([
                html.Div(
                    className="retro-error-container p-4 text-center",
                    children=[
                        html.I(className="bi bi-exclamation-triangle display-4 mb-3 text-warning"),
                        html.H3("Study Not Found", className="text-warning mb-3"),
                        html.P(f"The study '{study_name}' doesn't exist or has been deleted.", className="mb-4"),
                        html.Button(
                            [html.I(className="bi bi-arrow-left me-2"), "BACK TO LIST"],
                            id="btn-back-to-studies",
                            className="retro-button"
                        )
                    ]
                )
            ])
        
        # Retrieve metadata
        metadata = study_manager.get_study_metadata(study_name)
        
        # Get trading configuration
        trading_config = study_manager.get_trading_config(study_name)
        
        # Get strategies list
        strategies = study_manager.list_strategies(study_name)
        
        # Get optimization results
        optimization_results = study_manager.get_optimization_results(study_name)
        
        # Get data file info
        data_file_path = study_manager.get_study_data_file(study_name)
        data_info = get_data_file_info(data_file_path) if data_file_path else None
        
    except Exception as e:
        error_message = str(e)
        if ui_logger:
            ui_logger.error(f"Error retrieving details for study '{study_name}': {error_message}")
            ui_logger.error(traceback.format_exc())
        
        return html.Div([
            html.Div(
                className="retro-error-container p-4 text-center",
                children=[
                    html.I(className="bi bi-exclamation-circle display-4 mb-3 text-danger"),
                    html.H3("Error", className="text-danger mb-3"),
                    html.P(f"An error occurred while loading the study: {error_message}", className="mb-3"),
                    html.Pre(
                        traceback.format_exc(), 
                        className="bg-dark text-light p-3 text-start",
                        style={"maxHeight": "200px", "overflow": "auto", "fontSize": "0.8rem"}
                    ),
                    html.Button(
                        [html.I(className="bi bi-arrow-left me-2"), "BACK TO LIST"],
                        id="btn-back-to-studies",
                        className="retro-button mt-3"
                    )
                ]
            )
        ])
    
    # Format creation date
    creation_date = metadata.creation_date if hasattr(metadata, 'creation_date') else 'Unknown'
    last_modified = metadata.last_modified if hasattr(metadata, 'last_modified') else 'Never'
    
    # Get status and create badge
    status = metadata.status.value if hasattr(metadata, 'status') else 'unknown'
    status_badge = create_status_badge(status)
    
    # Get best strategy if available
    best_strategy_id = study_manager.get_best_strategy_id(study_name)
    best_strategy = None
    
    if best_strategy_id and strategies:
        for strategy in strategies:
            if strategy.get('id') == best_strategy_id:
                best_strategy = strategy
                break
    
    # Create main layout
    return html.Div([
        # Scanner line effect
        html.Div(className="scanline"),
        
        # Action bar with glowing buttons
        html.Div(
            className="d-flex justify-content-between align-items-center mb-4",
            children=[
                # Back button
                html.Button(
                    [html.I(className="bi bi-arrow-left me-2"), "BACK"],
                    id="btn-back-to-studies",
                    className="retro-button"
                ),
                
                # Action buttons
                html.Div(
                    className="d-flex",
                    children=[
                        html.Button(
                            [html.I(className="bi bi-gear me-2"), "SETTINGS"],
                            id="btn-study-settings",
                            className="retro-button secondary me-2"
                        ),
                        html.Button(
                            [html.I(className="bi bi-lightning-charge me-2"), "OPTIMIZE"],
                            id="btn-optimize-from-detail",
                            className="retro-button me-2"
                        ),
                        html.Button(
                            [html.I(className="bi bi-files me-2"), "CLONE"],
                            id="btn-clone-study",
                            className="retro-button secondary me-2"
                        ),
                        html.Button(
                            [html.I(className="bi bi-trash me-2"), "DELETE"],
                            id="btn-delete-study",
                            className="retro-button danger"
                        ),
                    ]
                )
            ]
        ),
        
        # Study header with glowing title and status badge
        html.Div(
            className="strategy-header mb-4",
            children=[
                html.Div(
                    className="d-flex justify-content-between align-items-center",
                    children=[
                        html.Div([
                            html.H2(
                                metadata.name if hasattr(metadata, 'name') else 'Unnamed Study',
                                className="text-cyan-300 mb-2 glitch-text"
                            ),
                            html.P(
                                metadata.description if hasattr(metadata, 'description') else 'No description available.',
                                className="mb-0 text-light"
                            )
                        ]),
                        html.Div([
                            status_badge,
                            html.Div(
                                f"Last modified: {last_modified}", 
                                className="text-secondary mt-2 small"
                            )
                        ])
                    ]
                )
            ]
        ),
        
        # Main content in a 3-column layout
        dbc.Row([
            # Left column - Study info and configuration
            dbc.Col([
                # Study information card
                create_info_card(metadata, data_info),
                
                # Study configuration card
                create_config_card(trading_config),
                
            ], width=12, lg=4, className="mb-4"),
            
            # Middle column - Performance and strategies
            dbc.Col([
                # Performance summary card
                create_performance_card(best_strategy),
                
                # Strategies table card
                create_strategies_card(strategies, study_name, best_strategy_id),
                
            ], width=12, lg=4, className="mb-4"),
            
            # Right column - Optimization results and actions
            dbc.Col([
                # Optimization status and results
                create_optimization_card(optimization_results, study_name),
                
                # Recent activity card
                create_activity_card(metadata),
                
            ], width=12, lg=4, className="mb-4"),
        ]),
        
        # Strategy visualization area (initially hidden)
        html.Div(
            id="selected-strategy-content",
            style={"display": "none"},
            className="mt-4 fade-in-animation"
        ),
        
        # Parameter editor modal
        create_parameter_editor(study_name, metadata, central_logger),
        
        # Delete confirmation modal
        create_delete_confirmation_modal(study_name),
        
        # Stores for actions
        dcc.Store(id="strategy-action", data=None),
        dcc.Store(id={"type": "study-detail-store", "id": study_name}, data=json.dumps({
            "name": study_name,
            "status": status
        })),
    ], className="strategy-viewer fade-in")

def create_status_badge(status):
    """
    Creates a status badge with appropriate styling.
    
    Args:
        status: Study status
    
    Returns:
        HTML component for the status badge
    """
    badge_class = "retro-badge "
    badge_text = status.upper()
    
    if status == 'created':
        badge_class += "retro-badge-blue"
    elif status == 'optimized':
        badge_class += "retro-badge-green"
    elif status == 'backtested':
        badge_class += "retro-badge-purple"
    elif status == 'archived':
        badge_class += "retro-badge-yellow"
    elif status == 'error':
        badge_class += "retro-badge-red"
    else:
        badge_class += "retro-badge-blue"
        badge_text = "UNKNOWN"
    
    return html.Span(badge_text, className=badge_class)

def get_data_file_info(file_path):
    """
    Gets information about the data file.
    
    Args:
        file_path: Path to the data file
    
    Returns:
        Dictionary with file information
    """
    if not file_path or not os.path.exists(file_path):
        return None
    
    try:
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
        
        # Format file size
        if file_size < 1024:
            size_str = f"{file_size} bytes"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        
        # Try to read file for preview
        df = pd.read_csv(file_path, nrows=5)
        rows_count = len(pd.read_csv(file_path))
        columns_count = len(df.columns)
        columns = ", ".join(df.columns.tolist())
        
        # Get date range if timestamp column exists
        date_range = None
        if 'timestamp' in df.columns:
            df_full = pd.read_csv(file_path)
            df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
            min_date = df_full['timestamp'].min()
            max_date = df_full['timestamp'].max()
            date_range = {
                "start": min_date.strftime("%Y-%m-%d"),
                "end": max_date.strftime("%Y-%m-%d")
            }
        
        return {
            "filename": filename,
            "size": size_str,
            "last_modified": last_modified.strftime("%Y-%m-%d %H:%M:%S"),
            "rows": rows_count,
            "columns": columns_count,
            "column_names": columns,
            "date_range": date_range
        }
    except Exception as e:
        return {
            "filename": os.path.basename(file_path),
            "error": str(e)
        }

def create_info_card(metadata, data_info):
    """
    Creates a card with study information.
    
    Args:
        metadata: Study metadata
        data_info: Data file information
    
    Returns:
        Card component with study information
    """
    # Extract metadata attributes safely
    get_attr = lambda obj, attr, default: getattr(obj, attr, default) if hasattr(obj, attr) else default
    
    asset = get_attr(metadata, 'asset', 'N/A')
    timeframe = get_attr(metadata, 'timeframe', 'N/A')
    exchange = get_attr(metadata, 'exchange', 'N/A')
    creation_date = get_attr(metadata, 'creation_date', 'Unknown')
    
    # Data section
    data_section = []
    if data_info:
        # File information
        data_section = [
            html.Div(
                className="mb-3",
                children=[
                    html.H6("DATA SOURCE", className="text-secondary mb-2"),
                    html.Div(
                        className="d-flex align-items-center mb-2",
                        children=[
                            html.I(className="bi bi-file-earmark-text me-2 text-cyan-300"),
                            html.Strong(data_info.get("filename", "Unknown file"))
                        ]
                    ),
                    html.Div(
                        className="d-flex flex-wrap",
                        children=[
                            html.Div(
                                className="me-3 mb-1",
                                children=[
                                    html.Span("Size: ", className="text-secondary"),
                                    html.Span(data_info.get("size", "N/A"))
                                ]
                            ),
                            html.Div(
                                className="me-3 mb-1",
                                children=[
                                    html.Span("Rows: ", className="text-secondary"),
                                    html.Span(str(data_info.get("rows", "N/A")))
                                ]
                            ),
                            html.Div(
                                className="mb-1",
                                children=[
                                    html.Span("Columns: ", className="text-secondary"),
                                    html.Span(str(data_info.get("columns", "N/A")))
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
        
        # Add date range if available
        if data_info.get("date_range"):
            date_range = data_info["date_range"]
            data_section.append(
                html.Div(
                    className="mb-3",
                    children=[
                        html.Div(
                            className="d-flex align-items-center",
                            children=[
                                html.I(className="bi bi-calendar-range me-2 text-cyan-300"),
                                html.Span("Data period:", className="text-secondary me-2"),
                                html.Span(f"{date_range['start']} to {date_range['end']}")
                            ]
                        )
                    ]
                )
            )
    else:
        data_section = [
            html.Div(
                className="text-center py-3",
                children=[
                    html.I(className="bi bi-exclamation-circle text-warning mb-2", style={"fontSize": "2rem"}),
                    html.P("No data file information available.", className="mb-0 text-muted")
                ]
            )
        ]
    
    # Create the card
    return html.Div(
        className="retro-card mb-4",
        children=[
            html.Div(
                className="retro-card-header",
                children=[
                    html.H3("STUDY INFORMATION", className="retro-card-title")
                ]
            ),
            html.Div(
                className="retro-card-body",
                children=[
                    # Study basics
                    html.Div(
                        className="mb-4",
                        children=[
                            html.H6("SETTINGS", className="text-secondary mb-2"),
                            html.Div(
                                className="d-flex flex-wrap",
                                children=[
                                                                            html.Div(
                                        className="col-6 mb-2",
                                        children=[
                                            html.Div("Asset", className="info-label"),
                                            html.Div(asset, className="info-value text-cyan-300")
                                        ]
                                    ),
                                    html.Div(
                                        className="col-6 mb-2",
                                        children=[
                                            html.Div("Timeframe", className="info-label"),
                                            html.Div(timeframe, className="info-value text-cyan-300")
                                        ]
                                    ),
                                    html.Div(
                                        className="col-6 mb-2",
                                        children=[
                                            html.Div("Exchange", className="info-label"),
                                            html.Div(exchange, className="info-value text-cyan-300")
                                        ]
                                    ),
                                    html.Div(
                                        className="col-6 mb-2",
                                        children=[
                                            html.Div("Created", className="info-label"),
                                            html.Div(creation_date, className="info-value")
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    
                    # Data file info
                    *data_section,
                    
                    # Tags (if available)
                    html.Div(
                        className="mt-3",
                        children=[
                            html.Div(
                                className="d-flex flex-wrap gap-1",
                                children=[
                                    html.Span("TAGS:", className="text-secondary me-2 mt-1"),
                                    html.Span("None", className="text-muted fst-italic")
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )

def create_config_card(trading_config):
    """
    Creates a card with study configuration details.
    
    Args:
        trading_config: Trading configuration object
    
    Returns:
        Card component with configuration information
    """
    # Handle case when config is not available
    if not trading_config:
        return html.Div(
            className="retro-card mb-4",
            children=[
                html.Div(
                    className="retro-card-header",
                    children=[
                        html.H3("CONFIGURATION", className="retro-card-title")
                    ]
                ),
                html.Div(
                    className="retro-card-body text-center py-3",
                    children=[
                        html.I(className="bi bi-gear text-secondary mb-2", style={"fontSize": "2rem"}),
                        html.P("No configuration information available.", className="mb-0 text-muted")
                    ]
                )
            ]
        )
    
    # Extract configuration details
    # Get config properties safely
    indicators = []
    if hasattr(trading_config, 'available_indicators'):
        for ind_name, ind_config in trading_config.available_indicators.items():
            indicators.append({
                "name": ind_name,
                "min_period": ind_config.min_period,
                "max_period": ind_config.max_period
            })
    
    # Get risk modes
    risk_modes = []
    if hasattr(trading_config, 'risk_config') and hasattr(trading_config.risk_config, 'available_modes'):
        risk_modes = [mode.value for mode in trading_config.risk_config.available_modes]
    
    # Get other configs
    sim_config = getattr(trading_config, 'sim_config', None)
    
    # Create the card
    return html.Div(
        className="retro-card mb-4",
        children=[
            html.Div(
                className="retro-card-header d-flex justify-content-between align-items-center",
                children=[
                    html.H3("CONFIGURATION", className="retro-card-title mb-0"),
                    html.Button(
                        html.I(className="bi bi-pencil"),
                        id="btn-edit-config",
                        className="retro-button btn-sm",
                        title="Edit configuration"
                    )
                ]
            ),
            html.Div(
                className="retro-card-body",
                children=[
                    # Collapsible section for indicators
                    create_collapsible_card(
                        "INDICATORS",
                        create_indicators_section(indicators),
                        "indicators-section",
                        True
                    ),
                    
                    # Collapsible section for risk parameters
                    create_collapsible_card(
                        "RISK PARAMETERS",
                        create_risk_section(trading_config),
                        "risk-section",
                        True
                    ),
                    
                    # Collapsible section for simulation parameters
                    create_collapsible_card(
                        "SIMULATION PARAMETERS",
                        create_simulation_section(sim_config),
                        "simulation-section",
                        False
                    )
                ]
            )
        ]
    )

def create_indicators_section(indicators):
    """
    Creates the indicators section of the configuration card.
    
    Args:
        indicators: List of indicator configurations
    
    Returns:
        Component with indicator information
    """
    if not indicators:
        return html.Div(
            "No indicators configured for this study.",
            className="text-muted fst-italic p-2"
        )
    
    # Create indicator items
    indicator_items = []
    for ind in indicators:
        indicator_items.append(
            html.Div(
                className="param-item mb-2",
                children=[
                    html.Div(
                        className="d-flex justify-content-between",
                        children=[
                            html.Div(
                                className="d-flex align-items-center",
                                children=[
                                    html.I(className="bi bi-graph-up me-2 text-cyan-300"),
                                    html.Span(ind["name"], className="fw-bold")
                                ]
                            ),
                            html.Div(
                                f"Periods: {ind['min_period']}-{ind['max_period']}",
                                className="text-secondary"
                            )
                        ]
                    )
                ]
            )
        )
    
    return html.Div(
        className="params-grid",
        children=indicator_items
    )

def create_risk_section(trading_config):
    """
    Creates the risk parameters section of the configuration card.
    
    Args:
        trading_config: Trading configuration object
    
    Returns:
        Component with risk parameter information
    """
    # Safely extract risk config
    risk_config = getattr(trading_config, 'risk_config', None)
    if not risk_config:
        return html.Div(
            "No risk parameters configured.",
            className="text-muted fst-italic p-2"
        )
    
    # Extract risk parameters
    risk_modes = []
    position_size_range = (0, 0)
    sl_range = (0, 0)
    tp_multiplier_range = (0, 0)
    
    if hasattr(risk_config, 'available_modes'):
        risk_modes = [mode.value for mode in risk_config.available_modes]
    
    if hasattr(risk_config, 'position_size_range'):
        position_size_range = risk_config.position_size_range
    
    if hasattr(risk_config, 'sl_range'):
        sl_range = risk_config.sl_range
    
    if hasattr(risk_config, 'tp_multiplier_range'):
        tp_multiplier_range = risk_config.tp_multiplier_range
    
    # Create content
    return html.Div([
        # Risk modes
        html.Div(
            className="mb-3",
            children=[
                html.Div("Risk Modes:", className="param-label mb-2"),
                html.Div(
                    className="d-flex flex-wrap gap-2",
                    children=[
                        html.Span(
                            mode,
                            className="retro-badge retro-badge-blue"
                        ) for mode in risk_modes
                    ] if risk_modes else html.Span("None", className="text-muted fst-italic")
                )
            ]
        ),
        
        # Parameters table
        html.Table(
            className="w-100",
            children=[
                html.Thead(
                    html.Tr([
                        html.Th("Parameter", className="text-secondary"),
                        html.Th("Min", className="text-secondary text-center"),
                        html.Th("Max", className="text-secondary text-center")
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td("Position Size"),
                        html.Td(f"{position_size_range[0] * 100:.1f}%", className="text-center"),
                        html.Td(f"{position_size_range[1] * 100:.1f}%", className="text-center")
                    ]),
                    html.Tr([
                        html.Td("Stop Loss"),
                        html.Td(f"{sl_range[0] * 100:.2f}%", className="text-center"),
                        html.Td(f"{sl_range[1] * 100:.2f}%", className="text-center")
                    ]),
                    html.Tr([
                        html.Td("Take Profit Multiplier"),
                        html.Td(f"{tp_multiplier_range[0]:.1f}x", className="text-center"),
                        html.Td(f"{tp_multiplier_range[1]:.1f}x", className="text-center")
                    ])
                ])
            ]
        )
    ])

def create_simulation_section(sim_config):
    """
    Creates the simulation parameters section of the configuration card.
    
    Args:
        sim_config: Simulation configuration object
    
    Returns:
        Component with simulation parameter information
    """
    if not sim_config:
        return html.Div(
            "No simulation parameters configured.",
            className="text-muted fst-italic p-2"
        )
    
    # Extract simulation parameters
    initial_balance_range = getattr(sim_config, 'initial_balance_range', (0, 0))
    fee = getattr(sim_config, 'fee', 0)
    slippage = getattr(sim_config, 'slippage', 0)
    leverage_range = getattr(sim_config, 'leverage_range', (0, 0))
    
    # Format parameters
    parameters = [
        {"name": "Initial Balance", "value": f"${initial_balance_range[0]:,.2f} - ${initial_balance_range[1]:,.2f}"},
        {"name": "Fee", "value": f"{fee * 100:.2f}%"},
        {"name": "Slippage", "value": f"{slippage * 100:.2f}%"},
        {"name": "Leverage", "value": f"{leverage_range[0]}x - {leverage_range[1]}x"}
    ]
    
    # Create content
    return html.Div([
        html.Table(
            className="w-100",
            children=[
                html.Tbody([
                    html.Tr([
                        html.Td(param["name"], className="text-secondary"),
                        html.Td(param["value"], className="text-end")
                    ]) for param in parameters
                ])
            ]
        )
    ])

def create_performance_card(best_strategy):
    """
    Creates a card with performance metrics of the best strategy.
    
    Args:
        best_strategy: Best strategy data
    
    Returns:
        Card component with performance metrics
    """
    if not best_strategy or 'performance' not in best_strategy:
        return html.Div(
            className="retro-card mb-4",
            children=[
                html.Div(
                    className="retro-card-header",
                    children=[
                        html.H3("PERFORMANCE", className="retro-card-title")
                    ]
                ),
                html.Div(
                    className="retro-card-body text-center py-3",
                    children=[
                        html.I(className="bi bi-bar-chart text-secondary mb-2", style={"fontSize": "2rem"}),
                        html.P("No performance data available yet.", className="mb-2 text-muted"),
                        html.P([
                            "Run an optimization to generate strategies and performance metrics.",
                        ], className="text-muted small")
                    ]
                )
            ]
        )
    
    # Extract performance metrics
    perf = best_strategy.get('performance', {})
    roi = perf.get('roi_pct', 0)
    win_rate = perf.get('win_rate_pct', 0)
    max_drawdown = perf.get('max_drawdown_pct', 0)
    total_trades = perf.get('total_trades', 0)
    profit_factor = perf.get('profit_factor', 0)
    
    # Create gauge figure for win rate
    win_rate_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=win_rate,
        title={"text": "Win Rate", "font": {"size": 12, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar": {"color": "#22D3EE"},
            "bgcolor": "rgba(0, 0, 0, 0.2)",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 40], "color": "rgba(248, 113, 113, 0.3)"},
                {"range": [40, 60], "color": "rgba(251, 191, 36, 0.3)"},
                {"range": [60, 100], "color": "rgba(74, 222, 128, 0.3)"}
            ]
        },
        number={"suffix": "%", "font": {"color": "white"}}
    ))
    
    win_rate_fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=30, r=30, t=30, b=0),
        height=150
    )
    
    # Create the card
    return html.Div(
        className="retro-card mb-4",
        children=[
            html.Div(
                className="retro-card-header",
                children=[
                    html.H3("BEST STRATEGY PERFORMANCE", className="retro-card-title")
                ]
            ),
            html.Div(
                className="retro-card-body",
                children=[
                    # Strategy identification
                    html.Div(
                        className="mb-3 text-center",
                        children=[
                            html.Div(
                                className="d-flex justify-content-center align-items-center",
                                children=[
                                    html.I(className="bi bi-trophy text-warning me-2"),
                                    html.Strong(
                                        best_strategy.get('name', f"Strategy #{best_strategy.get('rank', 'N/A')}"),
                                        className="text-light"
                                    )
                                ]
                            ),
                            html.Div(
                                f"Trial #{best_strategy.get('trial_id', 'N/A')}",
                                className="small text-secondary"
                            )
                        ]
                    ),
                    
                    # Win rate gauge
                    html.Div(
                        className="win-rate-gauge mb-3",
                        children=dcc.Graph(
                            figure=win_rate_fig,
                            config={'displayModeBar': False}
                        )
                    ),
                    
                    # Key metrics grid
                    html.Div(
                        className="performance-metrics-container",
                        children=[
                            dbc.Row([
                                dbc.Col(
                                    html.Div(
                                        className="performance-metric",
                                        children=[
                                            html.Div("ROI", className="metric-label"),
                                            html.Div(
                                                f"{roi:.2f}%", 
                                                className="metric-value",
                                                style={"color": "#4ADE80" if roi > 0 else "#F87171"}
                                            )
                                        ]
                                    ),
                                    width=6, className="mb-3"
                                ),
                                dbc.Col(
                                    html.Div(
                                        className="performance-metric",
                                        children=[
                                            html.Div("Max Drawdown", className="metric-label"),
                                            html.Div(
                                                f"{max_drawdown:.2f}%", 
                                                className="metric-value",
                                                style={"color": "#F87171"}
                                            )
                                        ]
                                    ),
                                    width=6, className="mb-3"
                                ),
                                dbc.Col(
                                    html.Div(
                                        className="performance-metric",
                                        children=[
                                            html.Div("Profit Factor", className="metric-label"),
                                            html.Div(
                                                f"{profit_factor:.2f}", 
                                                className="metric-value",
                                                style={"color": "#4ADE80" if profit_factor > 1 else "#F87171"}
                                            )
                                        ]
                                    ),
                                    width=6, className="mb-3"
                                ),
                                dbc.Col(
                                    html.Div(
                                        className="performance-metric",
                                        children=[
                                            html.Div("Total Trades", className="metric-label"),
                                            html.Div(
                                                f"{total_trades}", 
                                                className="metric-value"
                                            )
                                        ]
                                    ),
                                    width=6, className="mb-3"
                                ),
                            ]),
                        ]
                    )
                ]
            )
        ]
    )

def create_strategies_card(strategies, study_name, best_strategy_id):
    """
    Creates a card with the strategies table.
    
    Args:
        strategies: List of strategies
        study_name: Name of the study
        best_strategy_id: ID of the best strategy
    
    Returns:
        Card component with strategies table
    """
    return html.Div(
        className="retro-card trading-blocks-card",
        children=[
            html.Div(
                className="retro-card-header d-flex justify-content-between align-items-center",
                children=[
                    html.H3("STRATEGIES", className="retro-card-title mb-0"),
                    html.Button(
                        [html.I(className="bi bi-plus-lg me-2"), "ADD"],
                        id="btn-add-strategy",
                        className="retro-button secondary btn-sm"
                    )
                ]
            ),
            html.Div(
                className="retro-card-body p-0",
                children=[
                    create_strategies_table(strategies, study_name, best_strategy_id)
                ]
            )
        ]
    )

def create_strategies_table(strategies, study_name, best_strategy_id):
    """
    Creates a table of strategies for a study.
    
    Args:
        strategies: List of strategies
        study_name: Name of the study
        best_strategy_id: ID of the best strategy
    
    Returns:
        Table component with strategies
    """
    if not strategies:
        return html.Div(
            className="empty-blocks-message",
            children=[
                html.I(className="bi bi-exclamation-circle text-warning mb-3", style={"fontSize": "2rem"}),
                html.P("No strategies available yet.", className="mb-2"),
                html.P([
                    "Run an optimization to generate strategies, or ",
                    html.Button(
                        "add one manually",
                        id="btn-add-strategy-empty",
                        className="p-0 border-0 bg-transparent text-cyan-300",
                        style={"textDecoration": "underline"}
                    ),
                    "."
                ])
            ]
        )
    
    # Sort strategies by rank
    strategies = sorted(strategies, key=lambda s: s.get('rank', 0))
    
    # Create table rows
    rows = []
    for strategy in strategies:
        strategy_id = strategy.get('id', '')
        rank = strategy.get('rank', 0)
        name = strategy.get('name', f'Strategy {rank}')
        
        # Check if this is the best strategy
        is_best = (strategy_id == best_strategy_id)
        
        # Get performance metrics
        perf = strategy.get('performance', {})
        roi = perf.get('roi_pct', 0)
        win_rate = perf.get('win_rate_pct', 0)
        trades = perf.get('total_trades', 0)
        
        # Format ROI with appropriate color
        roi_style = {"color": "#4ADE80"} if roi > 0 else {"color": "#F87171"}
        
        # Add best strategy indicator
        best_badge = html.Span(
            "BEST", 
            className="retro-badge retro-badge-green ms-2"
        ) if is_best else None
        
        row = html.Tr([
            html.Td(
                html.Div(
                    className="d-flex align-items-center",
                    children=[
                        html.Span(f"#{rank}"),
                        best_badge
                    ]
                )
            ),
            html.Td(name),
            html.Td(f"{roi:.2f}%", style=roi_style),
            html.Td(f"{win_rate:.2f}%"),
            html.Td(f"{trades}"),
            html.Td(
                html.Div(
                    className="d-flex",
                    children=[
                        html.Button(
                            html.I(className="bi bi-eye"),
                            id={"type": "btn-view-strategy", "index": strategy_id, "study": study_name},
                            className="retro-button btn-sm me-1",
                            title="View this strategy"
                        ),
                        html.Button(
                            html.I(className="bi bi-graph-up"),
                            id={"type": "btn-backtest-strategy", "index": strategy_id, "study": study_name},
                            className="retro-button secondary btn-sm me-1",
                            title="Backtest"
                        ),
                        html.Button(
                            html.I(className="bi bi-trash"),
                            id={"type": "btn-delete-strategy", "index": strategy_id, "study": study_name},
                            className="retro-button danger btn-sm",
                            title="Delete"
                        )
                    ]
                )
            )
        ], className=("best-strategy" if is_best else ""))
        rows.append(row)
    
    # Create table
    return html.Table(
        [
            html.Thead(
                html.Tr([
                    html.Th("Rank"),
                    html.Th("Name"),
                    html.Th("ROI"),
                    html.Th("Win Rate"),
                    html.Th("Trades"),
                    html.Th("Actions")
                ], className="retro-table-header")
            ),
            html.Tbody(rows)
        ],
        className="retro-table w-100"
    )

def create_optimization_card(optimization_results, study_name):
    """
    Creates a card with optimization information and results.
    
    Args:
        optimization_results: Optimization results data
        study_name: Name of the study
    
    Returns:
        Card component with optimization information
    """
    card_content = []
    
    if not optimization_results:
        card_content = [
            html.Div(
                className="text-center py-4",
                children=[
                    html.I(className="bi bi-lightning-charge text-warning mb-3", style={"fontSize": "2.5rem"}),
                    html.P("No optimization has been performed yet.", className="mb-3"),
                    html.Button(
                        [html.I(className="bi bi-play-fill me-2"), "START OPTIMIZATION"],
                        id={"type": "btn-start-optimization", "study": study_name},
                        className="retro-button"
                    )
                ]
            )
        ]
    else:
        # Extract optimization information
        best_trials = optimization_results.get('best_trials', [])
        best_trial_id = optimization_results.get('best_trial_id', -1)
        n_trials = optimization_results.get('n_trials', 0)
        optimization_date = optimization_results.get('optimization_date', 'Unknown')
        duration = optimization_results.get('duration', 0)
        
        # Format duration
        if duration:
            duration_str = f"{duration:.1f} seconds"
            if duration > 60:
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                duration_str = f"{minutes} min {seconds} sec"
        else:
            duration_str = "Unknown"
        
        # Create optimization info
        opt_info = html.Div([
            html.Div(
                className="d-flex justify-content-between mb-3",
                children=[
                    html.Div([
                        html.Div("Last optimization:", className="text-secondary small"),
                        html.Div(optimization_date, className="fw-bold")
                    ]),
                    html.Div([
                        html.Div("Trials:", className="text-secondary small text-end"),
                        html.Div(n_trials, className="fw-bold text-end")
                    ])
                ]
            ),
            html.Div(
                className="d-flex justify-content-between mb-4",
                children=[
                    html.Div([
                        html.Div("Duration:", className="text-secondary small"),
                        html.Div(duration_str, className="")
                    ]),
                    html.Div([
                        html.Div("Best trial:", className="text-secondary small text-end"),
                        html.Div(f"#{best_trial_id}", className="fw-bold text-end text-success")
                    ])
                ]
            ),
        ])
        
        # Create top trials table
        trials_table = None
        if best_trials:
            # Sort trials by score
            best_trials = sorted(best_trials, key=lambda t: t.get('score', 0), reverse=True)[:5]
            
            # Create table rows
            trial_rows = []
            for trial in best_trials:
                trial_id = trial.get('trial_id', 0)
                score = trial.get('score', 0)
                metrics = trial.get('metrics', {})
                
                roi = metrics.get('roi', 0) * 100
                win_rate = metrics.get('win_rate', 0) * 100
                
                # Highlight best trial
                is_best = (trial_id == best_trial_id)
                row_class = "trial-row best-trial" if is_best else "trial-row"
                
                row = html.Tr([
                    html.Td(f"#{trial_id}"),
                    html.Td(f"{score:.2f}", className="fw-bold"),
                    html.Td(
                        f"{roi:.2f}%", 
                        style={"color": "#4ADE80" if roi > 0 else "#F87171"}
                    ),
                    html.Td(f"{win_rate:.2f}%"),
                    html.Td(
                        html.Button(
                            "DETAILS",
                            id={"type": "btn-view-trial", "trial": trial_id, "study": study_name},
                            className="retro-button btn-sm"
                        )
                    )
                ], className=row_class)
                trial_rows.append(row)
            
            trials_table = html.Div([
                html.H6("TOP TRIALS", className="text-secondary mb-2"),
                html.Table(
                    [
                        html.Thead(
                            html.Tr([
                                html.Th("ID"),
                                html.Th("Score"),
                                html.Th("ROI"),
                                html.Th("Win Rate"),
                                html.Th("Action")
                            ], className="retro-table-header")
                        ),
                        html.Tbody(trial_rows)
                    ],
                    className="retro-table w-100 mb-3"
                )
            ])
            
        # Buttons
        action_buttons = html.Div([
            html.Button(
                [html.I(className="bi bi-play-fill me-2"), "RE-OPTIMIZE"],
                id={"type": "btn-restart-optimization", "study": study_name},
                className="retro-button me-2"
            ),
            html.Button(
                [html.I(className="bi bi-eye me-2"), "FULL RESULTS"],
                id={"type": "btn-view-optimization", "study": study_name},
                className="retro-button secondary"
            ),
        ], className="d-flex mt-3")
        
        # Combine all elements
        card_content = [
            opt_info,
            trials_table,
            action_buttons
        ]
    
    # Create the card
    return html.Div(
        className="retro-card mb-4",
        children=[
            html.Div(
                className="retro-card-header",
                children=[
                    html.H3("OPTIMIZATION", className="retro-card-title")
                ]
            ),
            html.Div(
                className="retro-card-body",
                children=card_content
            )
        ]
    )

def create_activity_card(metadata):
    """
    Creates a card showing recent activity for the study.
    
    Args:
        metadata: Study metadata
    
    Returns:
        Card component with activity information
    """
    # Sample activity (this would come from actual study history in a real implementation)
    creation_date = metadata.creation_date if hasattr(metadata, 'creation_date') else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last_modified = metadata.last_modified if hasattr(metadata, 'last_modified') else creation_date
    
    # Create sample activities (in a real implementation, these would come from actual logs)
    try:
        creation_datetime = datetime.strptime(creation_date, "%Y-%m-%d %H:%M:%S")
        last_modified_datetime = datetime.strptime(last_modified, "%Y-%m-%d %H:%M:%S")
        
        # Add some sample activities
        activities = [
            {
                "type": "creation",
                "date": creation_datetime,
                "message": "Study created"
            },
            {
                "type": "modification",
                "date": last_modified_datetime,
                "message": "Study modified"
            }
        ]
        
        # Add some optimization activities if the study is optimized
        if hasattr(metadata, 'status') and metadata.status.value == 'optimized':
            opt_date = last_modified_datetime - timedelta(minutes=30)
            activities.append({
                "type": "optimization",
                "date": opt_date,
                "message": "Optimization completed"
            })
        
        # Sort activities by date (newest first)
        activities.sort(key=lambda x: x["date"], reverse=True)
        
    except Exception:
        # Fallback if dates can't be parsed
        activities = [
            {
                "type": "creation",
                "date": creation_date,
                "message": "Study created"
            }
        ]
    
    # Create activity items
    activity_items = []
    for activity in activities[:5]:  # Show only the 5 most recent activities
        # Format date
        if isinstance(activity["date"], datetime):
            date_str = activity["date"].strftime("%Y-%m-%d %H:%M")
        else:
            date_str = str(activity["date"])
        
        # Select icon based on activity type
        icon_class = "bi bi-info-circle"
        if activity["type"] == "creation":
            icon_class = "bi bi-plus-circle"
        elif activity["type"] == "optimization":
            icon_class = "bi bi-lightning-charge"
        elif activity["type"] == "backtest":
            icon_class = "bi bi-graph-up"
        elif activity["type"] == "modification":
            icon_class = "bi bi-pencil"
        
        activity_items.append(
            html.Div(
                className="d-flex mb-3",
                children=[
                    html.Div(
                        className="me-3",
                        children=html.I(className=f"{icon_class} text-cyan-300")
                    ),
                    html.Div(
                        className="flex-grow-1",
                        children=[
                            html.Div(activity["message"], className="mb-1"),
                            html.Div(date_str, className="text-secondary small")
                        ]
                    )
                ]
            )
        )
    
    # Create the card
    return html.Div(
        className="retro-card",
        children=[
            html.Div(
                className="retro-card-header",
                children=[
                    html.H3("RECENT ACTIVITY", className="retro-card-title")
                ]
            ),
            html.Div(
                className="retro-card-body",
                children=activity_items if activity_items else [
                    html.Div(
                        className="text-center py-3",
                        children=[
                            html.I(className="bi bi-activity text-secondary mb-2", style={"fontSize": "2rem"}),
                            html.P("No activity recorded.", className="mb-0 text-muted")
                        ]
                    )
                ]
            )
        ]
    )

def create_delete_confirmation_modal(study_name):
    """
    Creates a modal for confirming study deletion.
    
    Args:
        study_name: Name of the study
    
    Returns:
        Modal component for confirming deletion
    """
    return dbc.Modal(
        [
            dbc.ModalHeader(
                html.H4("CONFIRM DELETION", className="text-danger"),
                close_button=True
            ),
            dbc.ModalBody([
                html.P([
                    "Are you sure you want to delete the study ",
                    html.Strong(study_name),
                    "?"
                ]),
                html.P("This action cannot be undone.", className="text-danger"),
                html.Div(
                    className="d-flex align-items-center mt-3",
                    children=[
                        dbc.Checkbox(
                            id="delete-study-confirm",
                            className="me-2"
                        ),
                        html.Label(
                            "I understand that all data related to this study will be permanently deleted.",
                            htmlFor="delete-study-confirm"
                        )
                    ]
                )
            ]),
            dbc.ModalFooter([
                dbc.Button(
                    "Cancel",
                    id="btn-cancel-delete-study",
                    className="me-2",
                    color="secondary"
                ),
                dbc.Button(
                    "Delete Study",
                    id="btn-confirm-delete-study",
                    color="danger",
                    disabled=True
                )
            ])
        ],
        id="delete-study-modal",
        centered=True,
        is_open=False
    )

def register_study_detail_callbacks(app, central_logger=None):
    """
    Registers callbacks for the study detail page.
    
    Args:
        app: Dash app instance
        central_logger: Centralized logger instance
    """
    # Initialize logger
    ui_logger = None
    if central_logger:
        ui_logger = central_logger.get_logger("study_detail_callbacks", LoggerType.UI)
    
    # Callback for back button
    @app.callback(
        Output("study-action", "data"),
        [Input("btn-back-to-studies", "n_clicks")],
        prevent_initial_call=True
    )
    def back_to_studies(n_clicks):
        """Returns to the list of studies"""
        if not n_clicks:
            return dash.no_update
        
        if ui_logger:
            ui_logger.info("Returning to studies list")
        
        return json.dumps({"type": "back-to-list"})
    
    # Callback for viewing strategy details
    @app.callback(
        [Output("selected-strategy-content", "children"),
         Output("selected-strategy-content", "style")],
        [Input({"type": "btn-view-strategy", "index": ALL, "study": ALL}, "n_clicks")],
        prevent_initial_call=True
    )
    def view_strategy(n_clicks_list):
        """Displays details of a selected strategy"""
        ctx = callback_context
        if not ctx.triggered or not any(filter(None, n_clicks_list)):
            return dash.no_update, dash.no_update
        
        # Get button ID that triggered the callback
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        button_data = json.loads(button_id)
        strategy_id = button_data["index"]
        study_name = button_data["study"]
        
        if ui_logger:
            ui_logger.info(f"Viewing strategy {strategy_id} of study {study_name}")
        
        # Create strategy viewer content
        strategy_content = create_strategy_viewer(study_name, strategy_id, central_logger)
        
        # Return content and make it visible
        return strategy_content, {"display": "block"}
    
    # Callback for navigating to optimization
    @app.callback(
        [Output("studies-tabs", "active_tab", allow_duplicate=True),
         Output("current-study-id", "data", allow_duplicate=True)],
        [Input("btn-optimize-from-detail", "n_clicks"),
         Input({"type": "btn-start-optimization", "study": ALL}, "n_clicks"),
         Input({"type": "btn-restart-optimization", "study": ALL}, "n_clicks")],
        prevent_initial_call=True
    )
    def go_to_optimization(n_clicks, start_clicks, restart_clicks):
        """Redirects to the optimization tab with the selected study"""
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Determine which study to optimize
        study_name = None
        
        if button_id == "btn-optimize-from-detail":
            # Get study from store
            study_name = callback_context.inputs.get("current-study-id.data")
        else:
            # Get study from pattern-matching button
            button_data = json.loads(button_id)
            study_name = button_data["study"]
        
        if not study_name:
            return dash.no_update, dash.no_update
        
        if ui_logger:
            ui_logger.info(f"Redirecting to optimization for study: {study_name}")
        
        # Change tab and set current study
        return "tab-optimizations", study_name
    
    # Callback for parameter editor
    @app.callback(
        Output("parameter-editor-modal", "is_open"),
        [Input("btn-study-settings", "n_clicks"),
         Input("btn-edit-config", "n_clicks"),
         Input("btn-close-parameter-editor", "n_clicks")],
        [State("parameter-editor-modal", "is_open")]
    )
    def toggle_parameter_editor(n_settings, n_edit, n_close, is_open):
        """Opens or closes the parameter editor"""
        ctx = callback_context
        if not ctx.triggered:
            return is_open
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if button_id in ["btn-study-settings", "btn-edit-config"] and (n_settings or n_edit):
            if ui_logger:
                ui_logger.info("Opening parameter editor")
            return True
        elif button_id == "btn-close-parameter-editor":
            return False
        
        return is_open
    
    # Callback for delete confirmation modal
    @app.callback(
        Output("delete-study-modal", "is_open"),
        [Input("btn-delete-study", "n_clicks"),
         Input("btn-cancel-delete-study", "n_clicks"),
         Input("btn-confirm-delete-study", "n_clicks")],
        [State("delete-study-modal", "is_open")]
    )
    def toggle_delete_modal(n_delete, n_cancel, n_confirm, is_open):
        """Opens or closes the delete confirmation modal"""
        ctx = callback_context
        if not ctx.triggered:
            return is_open
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if button_id == "btn-delete-study" and n_delete:
            if ui_logger:
                ui_logger.info("Opening delete confirmation modal")
            return True
        elif button_id in ["btn-cancel-delete-study", "btn-confirm-delete-study"]:
            return False
        
        return is_open
    
    # Callback to enable/disable the delete confirmation button
    @app.callback(
        Output("btn-confirm-delete-study", "disabled"),
        [Input("delete-study-confirm", "value")]
    )
    def toggle_delete_button(confirmed):
        """Enables or disables the delete confirmation button"""
        return not confirmed
    
    # Callback for deleting a study
    @app.callback(
        Output("study-action", "data", allow_duplicate=True),
        [Input("btn-confirm-delete-study", "n_clicks")],
        [State("current-study-id", "data")],
        prevent_initial_call=True
    )
    def delete_study(n_clicks, study_name):
        """Deletes the current study and returns to the list"""
        if not n_clicks or not study_name:
            return dash.no_update
        
        try:
            from simulator.study_manager import IntegratedStudyManager
            study_manager = IntegratedStudyManager("studies")
            
            # Delete the study
            success = study_manager.delete_study(study_name)
            
            if ui_logger:
                if success:
                    ui_logger.info(f"Study '{study_name}' deleted successfully")
                else:
                    ui_logger.error(f"Failed to delete study '{study_name}'")
            
            # Return to list regardless of success (will show error on UI if needed)
            return json.dumps({"type": "back-to-list", "deleted": success, "study_name": study_name})
            
        except Exception as e:
            if ui_logger:
                ui_logger.error(f"Error deleting study '{study_name}': {str(e)}")
            
            # Return to the list with error info
            return json.dumps({
                "type": "back-to-list", 
                "deleted": False, 
                "study_name": study_name,
                "error": str(e)
            })
    
    # Callback for clone button
    @app.callback(
        Output("study-action", "data", allow_duplicate=True),
        [Input("btn-clone-study", "n_clicks")],
        [State("current-study-id", "data")],
        prevent_initial_call=True
    )
    def clone_study(n_clicks, study_name):
        """Clones the current study"""
        if not n_clicks or not study_name:
            return dash.no_update
        
        try:
            from simulator.study_manager import IntegratedStudyManager
            study_manager = IntegratedStudyManager("studies")
            
            # Generate a new name
            new_name = f"{study_name}_clone_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Clone the study (implementation depends on study_manager capabilities)
            # This is a placeholder - real implementation would depend on actual methods
            success = False
            if hasattr(study_manager, 'clone_study'):
                success = study_manager.clone_study(study_name, new_name)
            
            if ui_logger:
                if success:
                    ui_logger.info(f"Study '{study_name}' cloned successfully to '{new_name}'")
                else:
                    ui_logger.warning(f"Study cloning not implemented or failed for '{study_name}'")
            
            # Return to list to show the new study
            return json.dumps({
                "type": "back-to-list", 
                "cloned": success, 
                "study_name": study_name,
                "new_name": new_name if success else None
            })
            
        except Exception as e:
            if ui_logger:
                ui_logger.error(f"Error cloning study '{study_name}': {str(e)}")
            
            # Stay on same page
            return dash.no_update