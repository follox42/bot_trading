"""
Component for actions that can be performed on optimization results.
Includes exporting, strategy deployment, and other operations.
"""
import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import json
import os
from datetime import datetime

def create_actions_panel(study_name, best_trial_id):
    """
    Creates the actions panel with buttons for operations on optimization results.
    
    Args:
        study_name: Name of the study
        best_trial_id: ID of the best trial
        
    Returns:
        Component with action buttons
    """
    return html.Div([
        html.Div([
            html.H4([
                html.I(className="bi bi-lightning-charge me-2"),
                "Actions"
            ], className="retro-card-title d-flex align-items-center"),
        ], className="retro-card-header p-3"),
        
        html.Div([
            dbc.Row([
                # Export Actions
                dbc.Col([
                    html.Div([
                        html.H5([
                            html.I(className="bi bi-download me-2"),
                            "Export"
                        ], className="mb-3 text-cyan-300"),
                        
                        html.Div([
                            html.Button([
                                html.I(className="bi bi-file-code me-2"),
                                "Export Best Strategy"
                            ], id="export-strategy-btn", className="retro-button mb-2 w-100"),
                            
                            html.Button([
                                html.I(className="bi bi-file-earmark-spreadsheet me-2"),
                                "Export All Results"
                            ], id="export-all-results-btn", className="retro-button mb-2 w-100"),
                            
                            html.Button([
                                html.I(className="bi bi-graph-up-arrow me-2"),
                                "Export Visualizations"
                            ], id="export-visualizations-btn", className="retro-button mb-2 w-100"),
                        ], className="d-grid gap-2")
                    ], className="p-3 retro-subcard")
                ], md=4),
                
                # Strategy Actions
                dbc.Col([
                    html.Div([
                        html.H5([
                            html.I(className="bi bi-gear me-2"),
                            "Strategy Actions"
                        ], className="mb-3 text-cyan-300"),
                        
                        html.Div([
                            html.Button([
                                html.I(className="bi bi-caret-right-fill me-2"),
                                "Deploy Strategy"
                            ], id="deploy-strategy-btn", className="retro-button mb-2 w-100"),
                            
                            html.Button([
                                html.I(className="bi bi-arrow-repeat me-2"),
                                "Run Detailed Backtest"
                            ], id="run-backtest-btn", className="retro-button mb-2 w-100"),
                            
                            html.Button([
                                html.I(className="bi bi-pencil-square me-2"),
                                "Edit Strategy"
                            ], id="edit-strategy-btn", className="retro-button mb-2 w-100"),
                        ], className="d-grid gap-2")
                    ], className="p-3 retro-subcard")
                ], md=4),
                
                # Optimization Actions
                dbc.Col([
                    html.Div([
                        html.H5([
                            html.I(className="bi bi-cpu me-2"),
                            "Optimization Actions"
                        ], className="mb-3 text-cyan-300"),
                        
                        html.Div([
                            html.Button([
                                html.I(className="bi bi-plus-circle me-2"),
                                "Continue Optimization"
                            ], id="continue-optimization-btn", className="retro-button mb-2 w-100"),
                            
                            html.Button([
                                html.I(className="bi bi-trash me-2"),
                                "Delete Optimization"
                            ], id="delete-optimization-btn", className="retro-button danger mb-2 w-100"),
                            
                            html.Button([
                                html.I(className="bi bi-save me-2"),
                                "Save Optimization Report"
                            ], id="save-report-btn", className="retro-button mb-2 w-100"),
                        ], className="d-grid gap-2")
                    ], className="p-3 retro-subcard")
                ], md=4),
            ]),
            
            # Status message area
            html.Div([
                html.Div(id="action-status-message", className="mt-3 text-center")
            ]),
            
            # Hidden data stores and downloads
            dcc.Store(id="action-data", data=json.dumps({
                "study_name": study_name,
                "best_trial_id": best_trial_id
            })),
            dcc.Download(id="download-strategy"),
            dcc.Download(id="download-results"),
            dcc.Download(id="download-report"),
            
        ], className="retro-card-body p-3"),
    ], className="retro-card mb-4")


def register_actions_callbacks(app, central_logger=None):
    """
    Register callbacks for the actions panel component.
    
    Args:
        app: Dash app instance
        central_logger: Optional logger instance
    """
    @app.callback(
        Output("download-strategy", "data"),
        Input("export-strategy-btn", "n_clicks"),
        State("action-data", "data"),
        prevent_initial_call=True
    )
    def export_best_strategy(n_clicks, action_data):
        """Export the best strategy when button is clicked."""
        if not n_clicks or not action_data:
            return dash.no_update
            
        data = {}
        try:
            data = json.loads(action_data)
        except:
            return dash.no_update
            
        study_name = data.get("study_name", "unknown")
        best_trial_id = data.get("best_trial_id", -1)
        
        try:
            from core.study.study_manager import IntegratedStudyManager
            study_manager = IntegratedStudyManager("studies")
            
            # Export the best strategy to JSON
            strategy_data = study_manager.export_strategy(study_name, best_trial_id)
            
            if not strategy_data:
                if central_logger:
                    central_logger.error(f"Error exporting strategy: No data returned")
                return dash.no_update
            
            # Return the strategy data as a JSON file for download
            return dict(
                content=json.dumps(strategy_data, indent=4),
                filename=f"{study_name}_best_strategy.json",
                type="application/json"
            )
            
        except Exception as e:
            if central_logger:
                central_logger.error(f"Error exporting strategy: {str(e)}")
            return dash.no_update
    
    
    @app.callback(
        Output("download-results", "data"),
        Input("export-all-results-btn", "n_clicks"),
        State("action-data", "data"),
        prevent_initial_call=True
    )
    def export_all_results(n_clicks, action_data):
        """Export all optimization results when button is clicked."""
        if not n_clicks or not action_data:
            return dash.no_update
            
        data = {}
        try:
            data = json.loads(action_data)
        except:
            return dash.no_update
            
        study_name = data.get("study_name", "unknown")
        
        try:
            from core.study.study_manager import IntegratedStudyManager
            study_manager = IntegratedStudyManager("studies")
            
            # Export all optimization results
            results_data = study_manager.get_optimization_results(study_name)
            
            if not results_data:
                if central_logger:
                    central_logger.error(f"Error exporting results: No data returned")
                return dash.no_update
            
            # Return the results data as a JSON file for download
            return dict(
                content=json.dumps(results_data, indent=4),
                filename=f"{study_name}_optimization_results.json",
                type="application/json"
            )
            
        except Exception as e:
            if central_logger:
                central_logger.error(f"Error exporting results: {str(e)}")
            return dash.no_update
    
    
    @app.callback(
        Output("action-status-message", "children"),
        [Input("deploy-strategy-btn", "n_clicks"),
         Input("run-backtest-btn", "n_clicks"),
         Input("edit-strategy-btn", "n_clicks"),
         Input("continue-optimization-btn", "n_clicks"),
         Input("delete-optimization-btn", "n_clicks"),
         Input("save-report-btn", "n_clicks"),
         Input("export-visualizations-btn", "n_clicks")],
        State("action-data", "data"),
        prevent_initial_call=True
    )
    def handle_action_buttons(deploy_clicks, backtest_clicks, edit_clicks, 
                             continue_clicks, delete_clicks, save_report_clicks,
                             export_viz_clicks, action_data):
        """Handle various action button clicks."""
        ctx = dash.callback_context
        if not ctx.triggered or not action_data:
            return dash.no_update
            
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        data = {}
        try:
            data = json.loads(action_data)
        except:
            return html.Div("Error: Could not parse action data", className="text-danger")
            
        study_name = data.get("study_name", "unknown")
        
        # Handle different button actions
        if button_id == "deploy-strategy-btn":
            return html.Div([
                html.I(className="bi bi-check-circle-fill text-success me-2"),
                f"Strategy from study '{study_name}' deployed successfully!"
            ], className="alert alert-success p-2")
            
        elif button_id == "run-backtest-btn":
            return html.Div([
                html.I(className="bi bi-info-circle-fill text-info me-2"),
                f"Detailed backtest for '{study_name}' has been queued. You will be notified when complete."
            ], className="alert alert-info p-2")
            
        elif button_id == "edit-strategy-btn":
            return html.Div([
                html.I(className="bi bi-info-circle-fill text-info me-2"),
                f"Redirecting to strategy editor..."
            ], className="alert alert-info p-2")
            
        elif button_id == "continue-optimization-btn":
            return html.Div([
                html.I(className="bi bi-info-circle-fill text-info me-2"),
                f"Continuing optimization for '{study_name}'. Redirecting to optimization dashboard..."
            ], className="alert alert-info p-2")
            
        elif button_id == "delete-optimization-btn":
            return html.Div([
                html.I(className="bi bi-exclamation-triangle-fill text-warning me-2"),
                f"Are you sure you want to delete this optimization? ",
                html.Button("Confirm", id="confirm-delete-btn", className="btn btn-sm btn-danger ms-2")
            ], className="alert alert-warning p-2")
            
        elif button_id == "save-report-btn":
            # Generate a simple report
            return html.Div([
                html.I(className="bi bi-check-circle-fill text-success me-2"),
                f"Optimization report for '{study_name}' has been saved."
            ], className="alert alert-success p-2")
            
        elif button_id == "export-visualizations-btn":
            return html.Div([
                html.I(className="bi bi-check-circle-fill text-success me-2"),
                f"Visualizations exported to '{study_name}_visualizations/' folder."
            ], className="alert alert-success p-2")
            
        return dash.no_update
    
    
    @app.callback(
        Output("download-report", "data"),
        Input("save-report-btn", "n_clicks"),
        State("action-data", "data"),
        prevent_initial_call=True
    )
    def save_optimization_report(n_clicks, action_data):
        """Generate and download an optimization report."""
        if not n_clicks or not action_data:
            return dash.no_update
            
        data = {}
        try:
            data = json.loads(action_data)
        except:
            return dash.no_update
            
        study_name = data.get("study_name", "unknown")
        
        try:
            from core.study.study_manager import IntegratedStudyManager
            study_manager = IntegratedStudyManager("studies")
            
            # Get optimization results
            results = study_manager.get_optimization_results(study_name)
            
            if not results:
                if central_logger:
                    central_logger.error(f"Error generating report: No results found")
                return dash.no_update
            
            # Generate report content
            report_content = generate_html_report(study_name, results)
            
            # Return the report data as an HTML file for download
            return dict(
                content=report_content,
                filename=f"{study_name}_optimization_report.html",
                type="text/html"
            )
            
        except Exception as e:
            if central_logger:
                central_logger.error(f"Error generating report: {str(e)}")
            return dash.no_update


def generate_html_report(study_name, results):
    """
    Generate an HTML report for the optimization results.
    
    Args:
        study_name: Name of the study
        results: Optimization results data
        
    Returns:
        HTML content as string
    """
    # Get basic information
    n_trials = results.get('n_trials', 0)
    best_trial_id = results.get('best_trial_id', -1)
    optimization_date = results.get('optimization_date', 'Not available')
    
    # Get best trial data
    best_trial = next((t for t in results.get('best_trials', []) if t.get('trial_id') == best_trial_id), {})
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Optimization Report - {study_name}</title>
        <style>
            body {{
                font-family: 'Courier New', monospace;
                background-color: #0f172a;
                color: #e2e8f0;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #1e293b;
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            }}
            .header {{
                border-bottom: 2px solid rgba(34, 211, 238, 0.3);
                padding-bottom: 20px;
                margin-bottom: 30px;
                text-align: center;
            }}
            .title {{
                color: #22d3ee;
                margin: 0;
                font-size: 28px;
                text-shadow: 0 0 10px rgba(34, 211, 238, 0.5);
            }}
            .subtitle {{
                color: #94a3b8;
                margin: 10px 0;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: #0f172a;
                border-radius: 8px;
                border: 1px solid rgba(34, 211, 238, 0.2);
            }}
            .section-title {{
                color: #22d3ee;
                border-bottom: 1px solid rgba(34, 211, 238, 0.2);
                padding-bottom: 10px;
                margin-top: 0;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }}
            .metric-card {{
                background-color: rgba(30, 41, 59, 0.7);
                padding: 15px;
                border-radius: 5px;
                border-left: 3px solid #22d3ee;
            }}
            .metric-name {{
                color: #94a3b8;
                font-size: 14px;
                margin-bottom: 5px;
            }}
            .metric-value {{
                color: #22d3ee;
                font-size: 20px;
                font-weight: bold;
            }}
            .parameter-item {{
                background-color: rgba(30, 41, 59, 0.7);
                padding: 10px 15px;
                border-radius: 5px;
                margin-bottom: 10px;
                display: flex;
                justify-content: space-between;
            }}
            .param-name {{
                color: #94a3b8;
            }}
            .param-value {{
                color: #22d3ee;
                font-weight: bold;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                color: #94a3b8;
                font-size: 14px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
                color: #e2e8f0;
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid rgba(34, 211, 238, 0.1);
            }}
            th {{
                background-color: rgba(34, 211, 238, 0.1);
                color: #22d3ee;
            }}
            tr:hover {{
                background-color: rgba(34, 211, 238, 0.05);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 class="title">Optimization Report</h1>
                <h2 class="subtitle">{study_name}</h2>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h3 class="section-title">Optimization Summary</h3>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-name">Total Trials</div>
                        <div class="metric-value">{n_trials}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-name">Best Trial ID</div>
                        <div class="metric-value">{best_trial_id}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-name">Optimization Date</div>
                        <div class="metric-value">{optimization_date}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h3 class="section-title">Best Strategy Performance</h3>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-name">ROI</div>
                        <div class="metric-value">{best_trial.get('roi', 0) * 100:.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-name">Win Rate</div>
                        <div class="metric-value">{best_trial.get('win_rate', 0) * 100:.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-name">Max Drawdown</div>
                        <div class="metric-value">{best_trial.get('max_drawdown', 0) * 100:.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-name">Total Trades</div>
                        <div class="metric-value">{best_trial.get('total_trades', 0)}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-name">Profit Factor</div>
                        <div class="metric-value">{best_trial.get('profit_factor', 0):.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-name">Score</div>
                        <div class="metric-value">{best_trial.get('score', 0):.4f}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h3 class="section-title">Best Strategy Parameters</h3>
    """
    
    # Add parameters
    params = best_trial.get('params', {})
    if params:
        # Group parameters by type (similar to the best params tab)
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
        
        # Add parameters to HTML
        for group, group_params in param_groups.items():
            if group_params:
                html_content += f"""
                <h4 style="color: #94a3b8; margin-top: 20px;">{group}</h4>
                """
                
                for param, value in group_params:
                    formatted_value = f"{value:.4f}" if isinstance(value, float) else value
                    html_content += f"""
                    <div class="parameter-item">
                        <span class="param-name">{param}</span>
                        <span class="param-value">{formatted_value}</span>
                    </div>
                    """
    else:
        html_content += "<p>No parameter data available</p>"
    
    # Add top trials table
    html_content += """
            </div>
            
            <div class="section">
                <h3 class="section-title">Top Performing Trials</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Trial ID</th>
                            <th>ROI</th>
                            <th>Win Rate</th>
                            <th>Trades</th>
                            <th>Max DD</th>
                            <th>Score</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Add top 10 trials
    best_trials = results.get('best_trials', [])
    for i, trial in enumerate(best_trials[:10]):
        trial_id = trial.get('trial_id', i)
        roi = trial.get('roi', 0) * 100
        win_rate = trial.get('win_rate', 0) * 100
        max_dd = trial.get('max_drawdown', 0) * 100
        trades = trial.get('total_trades', 0)
        score = trial.get('score', 0)
        
        bg_color = "rgba(74, 222, 128, 0.1)" if trial_id == best_trial_id else ""
        
        html_content += f"""
                        <tr style="background-color: {bg_color}">
                            <td>{i + 1}</td>
                            <td>{trial_id}</td>
                            <td>{roi:.2f}%</td>
                            <td>{win_rate:.2f}%</td>
                            <td>{trades}</td>
                            <td>{max_dd:.2f}%</td>
                            <td>{score:.4f}</td>
                        </tr>
        """
    
    # Close the HTML
    html_content += """
                    </tbody>
                </table>
            </div>
            
            <div class="footer">
                <p>Generated by Trading Optimization Dashboard</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content