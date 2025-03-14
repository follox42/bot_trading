"""
Summary components for the optimization details panel.
Includes optimization summary and best trials table.
"""
import dash
from dash import html
import dash_bootstrap_components as dbc


def create_optimization_summary(optimization_config, best_trial_id, n_trials):
    """
    Creates a summary of the optimization configuration.
    
    Args:
        optimization_config: Optimization configuration
        best_trial_id: ID of the best trial
        n_trials: Number of trials
    
    Returns:
        Optimization summary component
    """
    # Get information about the optimization method
    method_name = optimization_config.get('method', 'tpe')
    
    # Import here to avoid circular import issues
    from simulator.study_config_definitions import (
        OPTIMIZATION_METHODS, 
        PRUNER_METHODS, 
        SCORING_FORMULAS
    )
    
    method_info = OPTIMIZATION_METHODS.get(method_name, OPTIMIZATION_METHODS['tpe'])
    
    # Get information about the scoring formula
    scoring_name = optimization_config.get('scoring_formula', 'standard')
    scoring_info = SCORING_FORMULAS.get(scoring_name, SCORING_FORMULAS['standard'])
    
    # Get information about pruning
    pruner_name = optimization_config.get('pruner_method', 'none')
    if pruner_name and pruner_name != 'none':
        pruner_info = PRUNER_METHODS.get(pruner_name, {'name': 'Not specified'})
        pruner_display = pruner_info.get('name', pruner_name)
    else:
        pruner_display = "Not enabled"
    
    # Create the card
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-gear-fill me-2"),
                "Optimization Configuration"
            ], className="text-cyan-300 mb-0")
        ], className="bg-dark border-secondary"),
        
        dbc.CardBody([
            # Main information
            html.Div([
                html.Div([
                    html.I(className="bi bi-trophy-fill text-warning me-2"),
                    html.Span("Best trial: ", className="text-cyan-100"),
                    html.Span(f"#{best_trial_id}", className="text-white fw-bold")
                ], className="mb-2"),
                
                html.Div([
                    html.I(className="bi bi-search me-2 text-cyan-300"),
                    html.Span("Method: ", className="text-cyan-100"),
                    html.Span(method_info['name'], className="text-white")
                ], className="mb-2"),
                
                html.Div([
                    html.I(className="bi bi-calculator me-2 text-cyan-300"),
                    html.Span("Scoring: ", className="text-cyan-100"),
                    html.Span(scoring_info['name'], className="text-white")
                ], className="mb-2"),
                
                html.Div([
                    html.I(className="bi bi-scissors me-2 text-cyan-300"),
                    html.Span("Pruning: ", className="text-cyan-100"),
                    html.Span(pruner_display, className="text-white")
                ], className="mb-0"),
            ], className="px-2 py-1"),
            
            # Advanced parameters expandable section
            html.Div([
                dbc.Button(
                    [
                        html.I(className="bi bi-plus-circle me-2"),
                        "Advanced Parameters"
                    ],
                    id="btn-toggle-advanced-params",
                    className="retro-button secondary mt-3 w-100",
                    size="sm"
                ),
                
                dbc.Collapse(
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-braces me-2 text-cyan-300"),
                            html.Span("Early stopping: ", className="text-cyan-100"),
                            html.Span(
                                str(optimization_config.get('early_stopping_n_trials', 'Disabled')), 
                                className="text-white"
                            )
                        ], className="mb-2 mt-3"),
                        
                        html.Div([
                            html.I(className="bi bi-cpu me-2 text-cyan-300"),
                            html.Span("Threads: ", className="text-cyan-100"),
                            html.Span(
                                str(optimization_config.get('n_jobs', 'Auto')), 
                                className="text-white"
                            )
                        ], className="mb-2"),
                        
                        html.Div([
                            html.I(className="bi bi-memory me-2 text-cyan-300"),
                            html.Span("Memory limit: ", className="text-cyan-100"),
                            html.Span(
                                f"{optimization_config.get('memory_limit', 0.8)*100:.0f}%", 
                                className="text-white"
                            )
                        ], className="mb-2"),
                        
                        html.Div([
                            html.I(className="bi bi-save me-2 text-cyan-300"),
                            html.Span("Checkpoints: ", className="text-cyan-100"),
                            html.Span(
                                "Enabled" if optimization_config.get('save_checkpoints', False) else "Disabled", 
                                className="text-white"
                            )
                        ], className="mb-0"),
                    ], className="px-2 py-1"),
                    id="collapse-advanced-params",
                    is_open=False
                )
            ]),
            
            # Scoring description
            html.Div([
                html.P(
                    scoring_info['description'],
                    className="mt-3 mb-0 text-muted small"
                )
            ], className="bg-dark rounded p-2 mt-3"),
        ], className="bg-dark border-secondary"),
    ], className="mb-4 retro-card shadow")


def create_best_trials_table(best_trials, best_trial_id):
    """
    Creates a table of the best trials.
    
    Args:
        best_trials: List of the best trials
        best_trial_id: ID of the best trial
    
    Returns:
        Best trials table component
    """
    if not best_trials:
        return html.Div("No results available", className="text-center text-muted p-3")
    
    rows = []
    for i, trial in enumerate(best_trials[:5]):  # Limit to 5 best trials for readability
        trial_id = trial.get('trial_id', 0)
        score = trial.get('score', 0)
        metrics = trial.get('metrics', {})
        
        roi = metrics.get('roi', 0) * 100
        win_rate = metrics.get('win_rate', 0) * 100
        max_dd = metrics.get('max_drawdown', 0) * 100
        profit_factor = metrics.get('profit_factor', 0)
        total_trades = metrics.get('total_trades', 0)
        
        # Style for the best trial
        row_class = "best-trial fw-bold" if trial_id == best_trial_id else ""
        
        row = html.Tr([
            html.Td(f"#{trial_id}", className="text-center"),
            html.Td(f"{score:.4f}", className="text-center"),
            html.Td(f"{roi:.2f}%", className=f"text-{'success' if roi > 0 else 'danger'}"),
            html.Td(f"{win_rate:.2f}%"),
            html.Td(f"{max_dd:.2f}%", className="text-danger"),
            html.Td(f"{profit_factor:.2f}", className=f"text-{'success' if profit_factor > 1 else 'warning'}"),
            html.Td(f"{int(total_trades)}"),
            html.Td([
                html.Button(
                    html.I(className="bi bi-eye"),
                    id={"type": "btn-view-trial", "trial_id": trial_id},
                    className="retro-button btn-sm",
                    title="View details"
                )
            ], className="text-center")
        ], className=f"trial-row {row_class}")
        
        rows.append(row)
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-stars me-2"),
                "Best Trials"
            ], className="text-cyan-300 mb-0")
        ], className="bg-dark border-secondary"),
        
        dbc.CardBody([
            html.Div([
                html.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("ID", className="text-center"),
                            html.Th("Score", className="text-center"),
                            html.Th("ROI"),
                            html.Th("Win Rate"),
                            html.Th("Max DD"),
                            html.Th("PF"),
                            html.Th("Trades"),
                            html.Th("", className="text-center")
                        ], className="retro-table-header")
                    ]),
                    html.Tbody(rows)
                ], className="retro-table w-100")
            ], className="table-responsive")
        ], className="p-0 bg-dark"),
    ], className="retro-card shadow")