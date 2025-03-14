"""
Actions tab components for the optimization details panel.
Contains action buttons and controls for interacting with optimization results.
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc


def create_actions_tab(study_name):
    """
    Creates the actions tab with various control buttons.
    
    Args:
        study_name: Name of the study
    
    Returns:
        Actions tab content
    """
    return html.Div([
        html.P([
            html.I(className="bi bi-info-circle me-2"),
            "These actions allow you to interact with the optimization results."
        ], className="text-info small mb-4"),
        
        dbc.Row([
            # Main actions
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="bi bi-lightning-charge me-2"),
                            "Main Actions"
                        ], className="text-cyan-300 mb-0")
                    ], className="bg-dark border-secondary"),
                    
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-backpack-fill me-2"), "Test Best Strategies"],
                                    id={"type": "btn-backtest-best", "study": study_name},
                                    className="retro-button w-100 mb-3"
                                ),
                            ], md=6),
                            
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-arrow-repeat me-2"), "Restart Optimization"],
                                    id={"type": "btn-restart-optimization", "study": study_name},
                                    className="retro-button secondary w-100 mb-3"
                                ),
                            ], md=6),
                            
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-save me-2"), "Export Results"],
                                    id={"type": "btn-export-results", "study": study_name},
                                    className="retro-button secondary w-100 mb-3"
                                ),
                            ], md=6),
                            
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-clipboard-check me-2"), "Compare Strategies"],
                                    id={"type": "btn-compare-strategies", "study": study_name},
                                    className="retro-button secondary w-100 mb-3"
                                ),
                            ], md=6),
                        ])
                    ], className="bg-dark")
                ], className="retro-card shadow mb-4")
            ], md=6),
            
            # Optimization parameters
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="bi bi-sliders me-2"),
                            "Modify Parameters"
                        ], className="text-cyan-300 mb-0")
                    ], className="bg-dark border-secondary"),
                    
                    dbc.CardBody([
                        html.P("Adjust scoring weights for the next optimization:", className="mb-3 small"),
                        
                        create_weight_sliders(),
                        
                        dbc.Button(
                            [html.I(className="bi bi-gear-fill me-2"), "Modify Configuration"],
                            id={"type": "btn-modify-config", "study": study_name},
                            className="retro-button secondary w-100 mt-3"
                        ),
                    ], className="bg-dark")
                ], className="retro-card shadow mb-4")
            ], md=6),
        ]),
        
        # Advanced actions
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="bi bi-tools me-2"),
                            "Advanced Actions"
                        ], className="text-cyan-300 mb-0")
                    ], className="bg-dark border-secondary"),
                    
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-layers-half me-2"), "Parameter Space Analysis"],
                                    id={"type": "btn-param-space", "study": study_name},
                                    className="retro-button secondary w-100 mb-3"
                                ),
                            ], md=6),
                            
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-bezier2 me-2"), "Hyperparameter Fine-Tuning"],
                                    id={"type": "btn-fine-tune", "study": study_name},
                                    className="retro-button secondary w-100 mb-3"
                                ),
                            ], md=6),
                            
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-code-square me-2"), "Generate Strategy Code"],
                                    id={"type": "btn-generate-code", "study": study_name},
                                    className="retro-button secondary w-100 mb-3"
                                ),
                            ], md=6),
                            
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-shield-check me-2"), "Robustness Testing"],
                                    id={"type": "btn-robustness", "study": study_name},
                                    className="retro-button secondary w-100 mb-3"
                                ),
                            ], md=6),
                        ])
                    ], className="bg-dark")
                ], className="retro-card shadow mb-4")
            ], md=12),
        ]),
        
        # Return to studies
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    [html.I(className="bi bi-arrow-left me-2"), "Back to Studies List"],
                    id="btn-back-to-studies",
                    className="retro-button secondary w-100"
                ),
            ], md=12),
        ]),
    ])


def create_weight_sliders():
    """
    Creates sliders for adjusting optimization scoring weights.
    
    Returns:
        Weight sliders component
    """
    weights = [
        {"name": "ROI", "id": "roi", "value": 2.5},
        {"name": "Win Rate", "id": "win_rate", "value": 0.5},
        {"name": "Max Drawdown", "id": "max_drawdown", "value": 2.0},
    ]
    
    sliders = []
    
    for weight in weights:
        sliders.append(
            html.Div([
                html.Div([
                    html.Span(f"{weight['name']}:", className="text-light me-2"),
                    html.Span(weight['value'], id=f"weight-value-{weight['id']}", className="text-cyan-300 fw-bold")
                ], className="d-flex justify-content-between mb-1"),
                
                dcc.Slider(
                    id={"type": "weight-slider", "metric": weight['id']},
                    min=0,
                    max=5,
                    step=0.1,
                    value=weight['value'],
                    marks={i: str(i) for i in range(6)},
                    className="retro-slider"
                ),
            ], className="mb-3")
        )
    
    return html.Div(sliders)


def create_optimization_config_modal(study_name):
    """
    Creates a modal for modifying optimization configuration.
    
    Args:
        study_name: Name of the study
    
    Returns:
        Optimization configuration modal
    """
    return dbc.Modal([
        dbc.ModalHeader(
            dbc.ModalTitle([
                html.I(className="bi bi-gear-fill me-2"),
                f"Optimization Configuration: {study_name}"
            ]),
            close_button=True
        ),
        
        dbc.ModalBody([
            # Tabs for different configuration sections
            dbc.Tabs([
                # Basic tab
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Label("Number of Trials", className="form-label text-light"),
                                dbc.Input(
                                    id="input-n-trials",
                                    type="number",
                                    value=500,
                                    min=10,
                                    max=10000,
                                    step=10,
                                    className="bg-dark text-light border-secondary"
                                )
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Label("Timeout (seconds)", className="form-label text-light"),
                                dbc.Input(
                                    id="input-timeout",
                                    type="number",
                                    value=3600,
                                    min=60,
                                    max=86400,
                                    step=60,
                                    className="bg-dark text-light border-secondary"
                                )
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Label("Number of Jobs", className="form-label text-light"),
                                dbc.Input(
                                    id="input-n-jobs",
                                    type="number",
                                    value=-1,
                                    min=-1,
                                    max=32,
                                    step=1,
                                    className="bg-dark text-light border-secondary"
                                ),
                                html.Small("Use -1 for all cores", className="text-muted")
                            ], className="mb-3"),
                        ], md=6),
                        
                        dbc.Col([
                            html.Div([
                                html.Label("Optimization Method", className="form-label text-light"),
                                dbc.Select(
                                    id="select-method",
                                    options=[
                                        {"label": "Tree-structured Parzen Estimator (TPE)", "value": "tpe"},
                                        {"label": "Random Search", "value": "random"},
                                        {"label": "CMA-ES", "value": "cmaes"},
                                        {"label": "NSGA-II", "value": "nsgaii"}
                                    ],
                                    value="tpe",
                                    className="bg-dark text-light border-secondary"
                                )
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Label("Scoring Formula", className="form-label text-light"),
                                dbc.Select(
                                    id="select-scoring",
                                    options=[
                                        {"label": "Standard Balanced", "value": "standard"},
                                        {"label": "Consistency Focus", "value": "consistency"},
                                        {"label": "Aggressive Growth", "value": "aggressive"},
                                        {"label": "Conservative", "value": "conservative"},
                                        {"label": "High Volume", "value": "volume"},
                                        {"label": "Custom Weights", "value": "custom"}
                                    ],
                                    value="standard",
                                    className="bg-dark text-light border-secondary"
                                )
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Label("Minimum Trades", className="form-label text-light"),
                                dbc.Input(
                                    id="input-min-trades",
                                    type="number",
                                    value=10,
                                    min=1,
                                    max=1000,
                                    step=1,
                                    className="bg-dark text-light border-secondary"
                                )
                            ], className="mb-3"),
                        ], md=6),
                    ]),
                    
                    html.Div([
                        dbc.Checkbox(
                            id="checkbox-enable-pruning",
                            label="Enable pruning",
                            value=False,
                            className="retro-checkbox"
                        ),
                    ], className="mb-3"),
                    
                    html.Div(
                        id="pruning-options",
                        style={"display": "none"},
                        children=[
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.Label("Pruner Method", className="form-label text-light"),
                                        dbc.Select(
                                            id="select-pruner",
                                            options=[
                                                {"label": "Median Pruner", "value": "median"},
                                                {"label": "Percentile Pruner", "value": "percentile"},
                                                {"label": "Hyperband Pruner", "value": "hyperband"}
                                            ],
                                            value="median",
                                            className="bg-dark text-light border-secondary"
                                        )
                                    ], className="mb-3"),
                                ], md=6),
                                
                                dbc.Col([
                                    html.Div([
                                        html.Label("Early Stopping Trials", className="form-label text-light"),
                                        dbc.Input(
                                            id="input-early-stopping",
                                            type="number",
                                            value=50,
                                            min=10,
                                            max=1000,
                                            step=10,
                                            className="bg-dark text-light border-secondary"
                                        )
                                    ], className="mb-3"),
                                ], md=6),
                            ]),
                        ]
                    )
                ], label="Basic", tab_id="tab-basic"),
                
                # Advanced tab
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Label("Startup Trials", className="form-label text-light"),
                                dbc.Input(
                                    id="input-startup-trials",
                                    type="number",
                                    value=10,
                                    min=5,
                                    max=100,
                                    step=5,
                                    className="bg-dark text-light border-secondary"
                                ),
                                html.Small("Number of random trials before optimization starts", className="text-muted")
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Label("EI Candidates", className="form-label text-light"),
                                dbc.Input(
                                    id="input-ei-candidates",
                                    type="number",
                                    value=24,
                                    min=10,
                                    max=100,
                                    step=1,
                                    className="bg-dark text-light border-secondary"
                                ),
                                html.Small("Number of EI candidates for TPE (advanced)", className="text-muted")
                            ], className="mb-3"),
                            
                            html.Div([
                                dbc.Checkbox(
                                    id="checkbox-multivariate",
                                    label="Use multivariate TPE",
                                    value=True,
                                    className="retro-checkbox"
                                ),
                            ], className="mb-3"),
                            
                            html.Div([
                                dbc.Checkbox(
                                    id="checkbox-group-params",
                                    label="Group related parameters",
                                    value=True,
                                    className="retro-checkbox"
                                ),
                            ], className="mb-3"),
                        ], md=6),
                        
                        dbc.Col([
                            html.Div([
                                html.Label("Memory Limit (%)", className="form-label text-light"),
                                dbc.Input(
                                    id="input-memory-limit",
                                    type="number",
                                    value=80,
                                    min=10,
                                    max=95,
                                    step=5,
                                    className="bg-dark text-light border-secondary"
                                ),
                                html.Small("Maximum memory usage as percentage", className="text-muted")
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Label("Checkpoint Frequency", className="form-label text-light"),
                                dbc.Input(
                                    id="input-checkpoint-every",
                                    type="number",
                                    value=10,
                                    min=1,
                                    max=100,
                                    step=1,
                                    className="bg-dark text-light border-secondary"
                                ),
                                html.Small("Save checkpoint every N trials", className="text-muted")
                            ], className="mb-3"),
                            
                            html.Div([
                                dbc.Checkbox(
                                    id="checkbox-save-checkpoints",
                                    label="Save checkpoints",
                                    value=True,
                                    className="retro-checkbox"
                                ),
                            ], className="mb-3"),
                            
                            html.Div([
                                dbc.Checkbox(
                                    id="checkbox-debug",
                                    label="Debug mode",
                                    value=False,
                                    className="retro-checkbox"
                                ),
                            ], className="mb-3"),
                        ], md=6),
                    ]),
                ], label="Advanced", tab_id="tab-advanced"),
                
                # Custom Weights tab
                dbc.Tab([
                    html.P([
                        html.I(className="bi bi-info-circle me-2"),
                        "Customize the weights used for scoring optimization trials."
                    ], className="text-info small mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Label("ROI Weight", className="form-label text-light"),
                                html.Div([
                                    dcc.Slider(
                                        id="weight-slider-roi",
                                        min=0,
                                        max=5,
                                        step=0.1,
                                        value=2.5,
                                        marks={i: str(i) for i in range(6)},
                                        className="retro-slider"
                                    ),
                                    html.Div(
                                        id="weight-value-roi",
                                        className="weight-value-display"
                                    )
                                ], className="d-flex align-items-center gap-3")
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Label("Win Rate Weight", className="form-label text-light"),
                                html.Div([
                                    dcc.Slider(
                                        id="weight-slider-win-rate",
                                        min=0,
                                        max=5,
                                        step=0.1,
                                        value=0.5,
                                        marks={i: str(i) for i in range(6)},
                                        className="retro-slider"
                                    ),
                                    html.Div(
                                        id="weight-value-win-rate",
                                        className="weight-value-display"
                                    )
                                ], className="d-flex align-items-center gap-3")
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Label("Max Drawdown Weight", className="form-label text-light"),
                                html.Div([
                                    dcc.Slider(
                                        id="weight-slider-drawdown",
                                        min=0,
                                        max=5,
                                        step=0.1,
                                        value=2.0,
                                        marks={i: str(i) for i in range(6)},
                                        className="retro-slider"
                                    ),
                                    html.Div(
                                        id="weight-value-drawdown",
                                        className="weight-value-display"
                                    )
                                ], className="d-flex align-items-center gap-3")
                            ], className="mb-3"),
                        ], md=6),
                        
                        dbc.Col([
                            html.Div([
                                html.Label("Profit Factor Weight", className="form-label text-light"),
                                html.Div([
                                    dcc.Slider(
                                        id="weight-slider-profit-factor",
                                        min=0,
                                        max=5,
                                        step=0.1,
                                        value=2.0,
                                        marks={i: str(i) for i in range(6)},
                                        className="retro-slider"
                                    ),
                                    html.Div(
                                        id="weight-value-profit-factor",
                                        className="weight-value-display"
                                    )
                                ], className="d-flex align-items-center gap-3")
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Label("Total Trades Weight", className="form-label text-light"),
                                html.Div([
                                    dcc.Slider(
                                        id="weight-slider-trades",
                                        min=0,
                                        max=5,
                                        step=0.1,
                                        value=1.0,
                                        marks={i: str(i) for i in range(6)},
                                        className="retro-slider"
                                    ),
                                    html.Div(
                                        id="weight-value-trades",
                                        className="weight-value-display"
                                    )
                                ], className="d-flex align-items-center gap-3")
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Label("Average Profit Weight", className="form-label text-light"),
                                html.Div([
                                    dcc.Slider(
                                        id="weight-slider-avg-profit",
                                        min=0,
                                        max=5,
                                        step=0.1,
                                        value=1.0,
                                        marks={i: str(i) for i in range(6)},
                                        className="retro-slider"
                                    ),
                                    html.Div(
                                        id="weight-value-avg-profit",
                                        className="weight-value-display"
                                    )
                                ], className="d-flex align-items-center gap-3")
                            ], className="mb-3"),
                        ], md=6),
                    ]),
                    
                    dbc.Button(
                        "Apply Custom Weights",
                        id="btn-apply-custom-weights",
                        color="primary",
                        className="mt-3 w-100 retro-button"
                    ),
                ], label="Custom Weights", tab_id="tab-weights"),
            ], id="config-tabs", active_tab="tab-basic"),
        ]),
        
        dbc.ModalFooter([
            dbc.Button(
                "Cancel",
                id="btn-cancel-config",
                className="retro-button secondary me-2"
            ),
            dbc.Button(
                "Save Configuration",
                id="btn-save-config",
                className="retro-button"
            ),
        ])
    ], id="modal-optimization-config", size="lg", is_open=False)