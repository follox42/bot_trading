"""
Component for displaying the best trials from optimization results.
Shows a table of top performing strategies with metrics.
"""
import dash
from dash import html, dcc, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json

def create_best_trials_section(best_trials, best_trial_id):
    """
    Creates the best trials table section.
    
    Args:
        best_trials: List of best trial data
        best_trial_id: ID of the best trial
        
    Returns:
        Component with table of best trials
    """
    return html.Div([
        html.Div([
            html.H4([
                html.I(className="bi bi-trophy me-2"),
                "Top Performing Strategies"
            ], className="retro-card-title d-flex align-items-center"),
        ], className="retro-card-header p-3"),
        
        html.Div([
            html.Div([
                create_best_trials_table(best_trials, best_trial_id)
            ], className="trading-blocks-card"),
            
            # Download button
            html.Div([
                html.Button([
                    html.I(className="bi bi-download me-2"),
                    "Export Top Strategies"
                ], id="export-top-strategies-btn", className="retro-button mt-3 w-100"),
                
                # Hidden download component
                dcc.Download(id="download-top-strategies"),
            ], className="text-center mt-3"),
            
        ], className="retro-card-body p-3"),
    ], className="retro-card mb-4")


def create_best_trials_table(best_trials, best_trial_id):
    """Creates a table of best trials with highlighting for the best one."""
    if not best_trials:
        return html.Div("No trials available", className="text-center text-muted p-4")
    
    # Prepare data for the table
    table_data = []
    for i, trial in enumerate(best_trials[:10]):  # Show top 10 trials
        trial_id = trial.get('trial_id', i)
        is_best = trial_id == best_trial_id
        
        # Format metrics for display
        roi = trial.get('roi', 0) * 100
        win_rate = trial.get('win_rate', 0) * 100
        max_dd = trial.get('max_drawdown', 0) * 100
        
        table_data.append({
            'rank': i + 1,
            'trial_id': trial_id,
            'roi': f"{roi:.2f}%",
            'win_rate': f"{win_rate:.2f}%",
            'trades': trial.get('total_trades', 0),
            'max_drawdown': f"{max_dd:.2f}%",
            'profit_factor': f"{trial.get('profit_factor', 0):.2f}",
            'score': trial.get('score', 0),
            'is_best': is_best
        })
    
    # Create the table
    return dash_table.DataTable(
        id='best-trials-table',
        columns=[
            {'name': 'Rank', 'id': 'rank', 'type': 'numeric'},
            {'name': 'Trial ID', 'id': 'trial_id', 'type': 'numeric'},
            {'name': 'ROI', 'id': 'roi'},
            {'name': 'Win Rate', 'id': 'win_rate'},
            {'name': 'Trades', 'id': 'trades', 'type': 'numeric'},
            {'name': 'Max Drawdown', 'id': 'max_drawdown'},
            {'name': 'Profit Factor', 'id': 'profit_factor'},
            {'name': 'Score', 'id': 'score', 'type': 'numeric', 'format': {'specifier': '.4f'}}
        ],
        data=table_data,
        style_header={
            'backgroundColor': 'rgb(25, 32, 49)',
            'color': 'rgb(209, 213, 219)',
            'fontWeight': 'bold',
            'border': '1px solid rgb(55, 65, 81)',
            'textAlign': 'center'
        },
        style_cell={
            'backgroundColor': 'rgb(17, 24, 39)',
            'color': 'rgb(209, 213, 219)',
            'border': '1px solid rgb(55, 65, 81)',
            'font-family': '"Share Tech Mono", monospace',
            'padding': '8px',
            'textAlign': 'center'
        },
        style_data_conditional=[
            # Highlight the best trial
            {
                'if': {'filter_query': '{is_best} eq true'},
                'backgroundColor': 'rgba(74, 222, 128, 0.1)',
                'border-left': '4px solid rgba(74, 222, 128, 0.7)'
            },
            # Highlight positive ROI
            {
                'if': {
                    'filter_query': '{roi} contains "+"',
                    'column_id': 'roi'
                },
                'color': 'rgb(74, 222, 128)'
            },
            # Highlight negative ROI
            {
                'if': {
                    'filter_query': '{roi} contains "-"',
                    'column_id': 'roi'
                },
                'color': 'rgb(248, 113, 113)'
            },
            # Highlight high win rate
            {
                'if': {
                    'filter_query': '{win_rate} > "50%"',
                    'column_id': 'win_rate'
                },
                'color': 'rgb(74, 222, 128)'
            },
            # Highlight top 3 ranks
            {
                'if': {'filter_query': '{rank} <= 3', 'column_id': 'rank'},
                'color': 'rgb(251, 191, 36)',
                'fontWeight': 'bold'
            }
        ],
        sort_action='native',
        filter_action='native',
        page_size=10,
        style_as_list_view=False
    )


def register_best_trials_callbacks(app, central_logger=None):
    """
    Register callbacks for the best trials component.
    
    Args:
        app: Dash app instance
        central_logger: Optional logger instance
    """
    @app.callback(
        Output("download-top-strategies", "data"),
        Input("export-top-strategies-btn", "n_clicks"),
        State("optimization-details-data", "data"),
        prevent_initial_call=True
    )
    def export_top_strategies(n_clicks, details_data):
        """Export top strategies to CSV when button is clicked."""
        if not n_clicks or not details_data:
            return dash.no_update
            
        details = {}
        try:
            details = json.loads(details_data)
        except:
            return dash.no_update
            
        study_name = details.get("study_name", "unknown")
        
        try:
            from simulator.study_manager import IntegratedStudyManager
            study_manager = IntegratedStudyManager("studies")
            optimization_results = study_manager.get_optimization_results(study_name)
            
            if not optimization_results or 'best_trials' not in optimization_results:
                return dash.no_update
                
            best_trials = optimization_results['best_trials']
            
            # Convert to DataFrame
            data = []
            for i, trial in enumerate(best_trials):
                trial_data = {
                    'rank': i + 1,
                    'trial_id': trial.get('trial_id', i),
                    'roi': trial.get('roi', 0),
                    'win_rate': trial.get('win_rate', 0),
                    'total_trades': trial.get('total_trades', 0),
                    'max_drawdown': trial.get('max_drawdown', 0),
                    'profit_factor': trial.get('profit_factor', 0),
                    'score': trial.get('score', 0)
                }
                
                # Add parameters
                for param_key, param_value in trial.get('params', {}).items():
                    trial_data[f"param_{param_key}"] = param_value
                    
                data.append(trial_data)
                
            df = pd.DataFrame(data)
            
            # Return the CSV for download
            return dcc.send_data_frame(
                df.to_csv, 
                f"{study_name}_top_strategies.csv", 
                index=False
            )
            
        except Exception as e:
            if central_logger:
                central_logger.error(f"Error exporting top strategies: {str(e)}")
            return dash.no_update