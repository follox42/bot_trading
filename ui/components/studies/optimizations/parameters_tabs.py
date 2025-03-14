"""
Parameter detail components for the optimization details panel.
Displays the parameters of the best trials in an organized manner.
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc


def create_parameters_tab(best_trials, best_trial_id):
    """
    Creates a tab displaying the parameters of the best trial.
    
    Args:
        best_trials: List of the best trials
        best_trial_id: ID of the best trial
    
    Returns:
        Parameters tab content
    """
    # Find the best trial
    best_trial = None
    for trial in best_trials:
        if trial.get('trial_id', 0) == best_trial_id:
            best_trial = trial
            break
    
    if not best_trial:
        return html.Div("Parameters not available", className="text-center text-muted p-3")
    
    params = best_trial.get('params', {})
    
    if not params:
        return html.Div("No parameters available", className="text-center text-muted p-3")
    
    # Group parameters by category
    param_categories = {
        "Buy Blocks": [],
        "Sell Blocks": [],
        "Risk Management": [],
        "Simulation": [],
        "Other": []
    }
    
    for param_name, param_value in params.items():
        # Determine category based on parameter name
        if param_name.startswith("buy_"):
            param_categories["Buy Blocks"].append((param_name, param_value))
        elif param_name.startswith("sell_"):
            param_categories["Sell Blocks"].append((param_name, param_value))
        elif param_name in ["risk_mode", "base_position", "base_sl", "tp_multiplier", 
                         "atr_period", "atr_multiplier", "vol_period", "vol_multiplier",
                         "fixed_position", "fixed_sl", "fixed_tp_mult"]:
            param_categories["Risk Management"].append((param_name, param_value))
        elif param_name in ["leverage", "margin_mode", "trading_mode", "initial_balance"]:
            param_categories["Simulation"].append((param_name, param_value))
        else:
            param_categories["Other"].append((param_name, param_value))
    
    # Create parameter cards for each category
    param_cards = []
    
    for category, params_list in param_categories.items():
        if not params_list:
            continue
            
        # Sort parameters by name
        params_list.sort(key=lambda x: x[0])
        
        param_items = []
        
        for param_name, param_value in params_list:
            # Special formatting for certain parameters
            if param_name == "risk_mode":
                value_display = f"{param_value}"
            elif param_name == "base_position" or param_name == "base_sl" or param_name == "fixed_position" or param_name == "fixed_sl":
                value_display = f"{param_value*100:.2f}%"
            elif param_name == "tp_multiplier" or param_name == "fixed_tp_mult" or param_name == "atr_multiplier" or param_name == "vol_multiplier":
                value_display = f"{param_value:.2f}×"
            elif param_name == "leverage":
                value_display = f"{param_value}×"
            elif param_name == "margin_mode":
                value_display = "Isolated" if param_value == 0 else "Cross"
            elif param_name == "trading_mode":
                value_display = "One-Way" if param_value == 0 else "Hedge"
            elif isinstance(param_value, float):
                value_display = f"{param_value:.4f}"
            else:
                value_display = str(param_value)
            
            param_items.append(
                html.Div([
                    html.Div(param_name, className="small text-light"),
                    html.Div(value_display, className="text-cyan-300 float-end fw-bold")
                ], className="param-item py-1 border-bottom border-dark")
            )
        
        # Create card for this category
        param_cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H6(category, className="mb-0 text-cyan-300"),
                        className="bg-dark border-secondary"
                    ),
                    dbc.CardBody(
                        html.Div(param_items, className="params-container"),
                        className="bg-dark p-3"
                    )
                ], className="param-category-card mb-3 retro-card shadow")
            ], md=6)
        )
    
    # If no parameters found
    if not param_cards:
        return html.Div("No parameters available", className="text-center text-muted p-3")
    
    return html.Div([
        html.P([
            html.I(className="bi bi-info-circle me-2"),
            "These parameters represent the exact configuration of the best strategy found during optimization."
        ], className="text-info small mb-4"),
        
        dbc.Row(param_cards),
        
        html.Div([
            dbc.Button(
                [html.I(className="bi bi-file-earmark-code me-2"), "Export Parameters"],
                id={"type": "btn-export-params", "trial_id": best_trial_id},
                className="retro-button mt-3"
            ),
        ], className="text-center mt-3")
    ])


def create_structured_parameters_view(best_trial):
    """
    Creates a more structured view of the parameters, organized by strategy components.
    
    Args:
        best_trial: The best trial to display parameters for
        
    Returns:
        Structured parameters view component
    """
    if not best_trial:
        return html.Div("Parameters not available", className="text-center text-muted p-3")
    
    params = best_trial.get('params', {})
    
    if not params:
        return html.Div("No parameters available", className="text-center text-muted p-3")
    
    # Extract blocks structure
    n_buy_blocks = params.get('n_buy_blocks', 0)
    n_sell_blocks = params.get('n_sell_blocks', 0)
    
    # Build buy blocks
    buy_blocks = []
    for b in range(n_buy_blocks):
        conditions = []
        n_conditions = params.get(f'buy_block_{b}_conditions', 0)
        
        for c in range(n_conditions):
            ind1_type = params.get(f'buy_b{b}_c{c}_ind1_type', '')
            ind1_period = params.get(f'buy_b{b}_c{c}_ind1_period', '')
            operator = params.get(f'buy_b{b}_c{c}_operator', '')
            
            # Check if it's a comparison with another indicator or a value
            if f'buy_b{b}_c{c}_ind2_type' in params:
                ind2_type = params.get(f'buy_b{b}_c{c}_ind2_type', '')
                ind2_period = params.get(f'buy_b{b}_c{c}_ind2_period', '')
                condition_text = f"{ind1_type}_{ind1_period} {operator} {ind2_type}_{ind2_period}"
            else:
                value = params.get(f'buy_b{b}_c{c}_value', '')
                condition_text = f"{ind1_type}_{ind1_period} {operator} {value}"
            
            conditions.append(condition_text)
            
            # Add logic operator if not the last condition
            if c < n_conditions - 1:
                logic_op = params.get(f'buy_b{b}_c{c}_logic', 'AND')
                conditions.append(logic_op)
        
        if conditions:
            buy_blocks.append(
                html.Div([
                    html.Div(f"Buy Block {b+1}", className="block-header"),
                    html.Div(
                        [html.Span(cond, className=f"{'logic-operator' if i % 2 else 'condition-text'}")
                         for i, cond in enumerate(conditions)],
                        className="condition-item my-2"
                    )
                ], className="trading-block buy-block mb-3")
            )
    
    # Build sell blocks
    sell_blocks = []
    for b in range(n_sell_blocks):
        conditions = []
        n_conditions = params.get(f'sell_block_{b}_conditions', 0)
        
        for c in range(n_conditions):
            ind1_type = params.get(f'sell_b{b}_c{c}_ind1_type', '')
            ind1_period = params.get(f'sell_b{b}_c{c}_ind1_period', '')
            operator = params.get(f'sell_b{b}_c{c}_operator', '')
            
            # Check if it's a comparison with another indicator or a value
            if f'sell_b{b}_c{c}_ind2_type' in params:
                ind2_type = params.get(f'sell_b{b}_c{c}_ind2_type', '')
                ind2_period = params.get(f'sell_b{b}_c{c}_ind2_period', '')
                condition_text = f"{ind1_type}_{ind1_period} {operator} {ind2_type}_{ind2_period}"
            else:
                value = params.get(f'sell_b{b}_c{c}_value', '')
                condition_text = f"{ind1_type}_{ind1_period} {operator} {value}"
            
            conditions.append(condition_text)
            
            # Add logic operator if not the last condition
            if c < n_conditions - 1:
                logic_op = params.get(f'sell_b{b}_c{c}_logic', 'AND')
                conditions.append(logic_op)
        
        if conditions:
            sell_blocks.append(
                html.Div([
                    html.Div(f"Sell Block {b+1}", className="block-header"),
                    html.Div(
                        [html.Span(cond, className=f"{'logic-operator' if i % 2 else 'condition-text'}")
                         for i, cond in enumerate(conditions)],
                        className="condition-item my-2"
                    )
                ], className="trading-block sell-block mb-3")
            )
    
    # Risk parameters
    risk_params = []
    risk_fields = [
        ("risk_mode", "Risk Mode"),
        ("base_position", "Position Size"),
        ("base_sl", "Stop Loss"),
        ("tp_multiplier", "TP Multiplier"),
        ("atr_period", "ATR Period"),
        ("atr_multiplier", "ATR Multiplier"),
        ("vol_period", "Volatility Period"),
        ("vol_multiplier", "Volatility Multiplier"),
        ("leverage", "Leverage"),
        ("margin_mode", "Margin Mode"),
        ("trading_mode", "Trading Mode")
    ]
    
    for param_name, label in risk_fields:
        if param_name in params:
            value = params[param_name]
            
            # Format values
            if param_name == "base_position" or param_name == "base_sl":
                value_display = f"{value*100:.2f}%"
            elif param_name == "tp_multiplier" or param_name == "atr_multiplier" or param_name == "vol_multiplier":
                value_display = f"{value:.2f}×"
            elif param_name == "leverage":
                value_display = f"{value}×"
            elif param_name == "margin_mode":
                value_display = "Isolated" if value == 0 else "Cross"
            elif param_name == "trading_mode":
                value_display = "One-Way" if value == 0 else "Hedge"
            else:
                value_display = str(value)
            
            risk_params.append(
                html.Div([
                    html.I(className=f"bi bi-gear me-2 risk-icon"),
                    html.Div([
                        html.Div(label, className="param-label"),
                        html.Div(value_display, className="param-value highlight-value")
                    ])
                ], className="risk-param-item")
            )
    
    return html.Div([
        dbc.Row([
            # Trading blocks
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="bi bi-code-square me-2"),
                            "Trading Blocks"
                        ], className="text-cyan-300 mb-0")
                    ], className="bg-dark border-secondary"),
                    
                    dbc.CardBody([
                        html.H6("Buy Conditions", className="section-title text-success"),
                        html.Div(
                            buy_blocks if buy_blocks else html.Div("No buy blocks defined", className="empty-blocks-message"),
                            className="buy-blocks-container"
                        ),
                        
                        html.H6("Sell Conditions", className="section-title text-danger mt-4"),
                        html.Div(
                            sell_blocks if sell_blocks else html.Div("No sell blocks defined", className="empty-blocks-message"),
                            className="sell-blocks-container"
                        )
                    ], className="bg-dark p-3 trading-blocks-card")
                ], className="retro-card shadow mb-4")
            ], md=7),
            
            # Risk parameters
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="bi bi-shield-lock me-2"),
                            "Risk Parameters"
                        ], className="text-cyan-300 mb-0")
                    ], className="bg-dark border-secondary"),
                    
                    dbc.CardBody([
                        html.Div(
                            risk_params if risk_params else html.Div("No risk parameters defined", className="empty-blocks-message"),
                            className="risk-params-container"
                        )
                    ], className="bg-dark p-3")
                ], className="retro-card shadow mb-4")
            ], md=5)
        ]),
        
        html.Div([
            dbc.Button(
                [html.I(className="bi bi-file-earmark-code me-2"), "Export Strategy Configuration"],
                id={"type": "btn-export-strategy", "trial_id": best_trial.get('trial_id', 0)},
                className="retro-button"
            ),
        ], className="text-center")
    ])