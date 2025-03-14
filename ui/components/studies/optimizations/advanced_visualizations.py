"""
Advanced visualization components for the optimization details panel.
Includes 3D exploration plots, correlation heatmaps, and distribution analysis.
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def create_multidimensional_exploration(best_trials):
    """
    Creates a 3D visualization of parameter space exploration with PCA and t-SNE.
    
    Args:
        best_trials: List of best optimization trials
        
    Returns:
        Multidimensional exploration component
    """
    if len(best_trials) < 10:
        return html.Div(
            "Not enough trials for multidimensional analysis (minimum 10 required)",
            className="text-center text-muted p-3"
        )
    
    # Extract parameters and scores
    all_params = set()
    for trial in best_trials:
        params = trial.get('params', {})
        all_params.update(params.keys())
    
    param_names = sorted(list(all_params))
    
    # Create feature matrix
    X = []
    for trial in best_trials:
        params = trial.get('params', {})
        param_values = []
        for param in param_names:
            value = params.get(param, 0)
            # Convert categorical parameters to numeric
            if isinstance(value, str):
                try:
                    value = float(value)
                except:
                    value = hash(value) % 100 / 100.0
            param_values.append(value)
        X.append(param_values)
    
    X = np.array(X)
    
    # Handle NaN values
    X = np.nan_to_num(X)
    
    scores = np.array([t.get('score', 0) for t in best_trials])
    trials = np.array([t.get('trial_id', 0) for t in best_trials])
    
    # Apply scaling and dimensionality reduction
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA for 3D visualization
    pca = PCA(n_components=min(3, X.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    
    # t-SNE for 2D visualization
    tsne = TSNE(n_components=2, perplexity=min(30, len(X_scaled) - 1))
    X_tsne = tsne.fit_transform(X_scaled)
    
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{'type': 'scene'}, {'type': 'xy'}],
            [{'colspan': 2}, None]
        ],
        subplot_titles=(
            '3D Exploration (PCA)',
            '2D Projection (t-SNE)',
            'Principal Component Evolution'
        )
    )
    
    # 3D scatter plot
    fig.add_trace(
        go.Scatter3d(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            z=X_pca[:, 2] if X_pca.shape[1] > 2 else np.zeros(X_pca.shape[0]),
            mode='markers',
            marker=dict(
                size=8,
                color=scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Score",
                    x=0.45
                )
            ),
            text=[f"Trial #{t}<br>Score: {s:.4f}" for t, s in zip(trials, scores)],
            hoverinfo='text',
            name='PCA Projection'
        ),
        row=1, col=1
    )
    
    # Add connecting lines between consecutive trials
    indices = np.argsort(trials)
    fig.add_trace(
        go.Scatter3d(
            x=X_pca[indices, 0],
            y=X_pca[indices, 1],
            z=X_pca[indices, 2] if X_pca.shape[1] > 2 else np.zeros(X_pca.shape[0]),
            mode='lines',
            line=dict(color='rgba(255, 0, 0, 0.3)', width=2),
            name='Optimization Path',
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    
    # t-SNE 2D scatter plot
    fig.add_trace(
        go.Scatter(
            x=X_tsne[:, 0],
            y=X_tsne[:, 1],
            mode='markers',
            marker=dict(
                size=10,
                color=scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Score")
            ),
            text=[f"Trial #{t}<br>Score: {s:.4f}" for t, s in zip(trials, scores)],
            hoverinfo='text',
            name='t-SNE Projection'
        ),
        row=1, col=2
    )
    
    # Evolution of principal components
    for i in range(min(3, X_pca.shape[1])):
        fig.add_trace(
            go.Scatter(
                x=trials[indices],
                y=X_pca[indices, i],
                mode='lines+markers',
                name=f'PC{i+1}',
                marker=dict(size=5),
                line=dict(width=2)
            ),
            row=2, col=1
        )
    
    # Add variance explained text
    explained_variance = pca.explained_variance_ratio_
    variance_text = (
        f"Variance explained:<br>"
        f"PC1: {explained_variance[0]:.2%}<br>"
        f"PC2: {explained_variance[1]:.2%}<br>"
        f"PC3: {explained_variance[2]:.2%}" if len(explained_variance) > 2 else ""
    )
    
    fig.add_annotation(
        text=variance_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=12, color="#22D3EE"),
        bgcolor="rgba(15, 23, 42, 0.7)",
        bordercolor="#22D3EE",
        borderwidth=1,
        borderpad=4
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.9)",
        plot_bgcolor="rgba(15, 23, 42, 0.9)",
        font=dict(family="'Share Tech Mono', monospace", color="#D1D5DB"),
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            xaxis=dict(gridcolor="rgba(51, 65, 85, 0.6)"),
            yaxis=dict(gridcolor="rgba(51, 65, 85, 0.6)"),
            zaxis=dict(gridcolor="rgba(51, 65, 85, 0.6)")
        ),
        xaxis=dict(
            gridcolor="rgba(51, 65, 85, 0.6)",
            title="t-SNE Dimension 1"
        ),
        yaxis=dict(
            gridcolor="rgba(51, 65, 85, 0.6)",
            title="t-SNE Dimension 2"
        ),
        xaxis2=dict(
            gridcolor="rgba(51, 65, 85, 0.6)",
            title="Trial ID"
        ),
        yaxis2=dict(
            gridcolor="rgba(51, 65, 85, 0.6)",
            title="Component Value"
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-diagram-3-fill me-2"),
                "Multidimensional Parameter Exploration"
            ], className="text-cyan-300 mb-0")
        ], className="bg-dark border-secondary"),
        
        dbc.CardBody([
            html.P([
                html.I(className="bi bi-info-circle me-2"),
                """This visualization shows how the optimization explored the parameter space. 
                The 3D plot uses PCA to reduce dimensions, while the 2D plot uses t-SNE. 
                Colors represent the score value of each trial."""
            ], className="text-info small mb-3"),
            
            dcc.Graph(
                figure=fig,
                config={
                    'displayModeBar': True,
                    'displaylogo': False
                },
                className="retro-graph"
            )
        ], className="bg-dark p-3")
    ], className="retro-card shadow mb-4")

def create_correlation_heatmap(best_trials):
    """
    Creates a correlation heatmap between parameters and metrics.
    
    Args:
        best_trials: List of best optimization trials
        
    Returns:
        Correlation heatmap component
    """
    if len(best_trials) < 10:
        return html.Div(
            "Not enough trials for correlation analysis (minimum 10 required)",
            className="text-center text-muted p-3"
        )
    
    # Extract parameters and metrics
    data = []
    for trial in best_trials:
        trial_data = {}
        
        # Add parameters
        params = trial.get('params', {})
        for param_name, param_value in params.items():
            if isinstance(param_value, (int, float)):
                trial_data[f"param_{param_name}"] = param_value
        
        # Add metrics
        metrics = trial.get('metrics', {})
        trial_data['roi'] = metrics.get('roi', 0)
        trial_data['win_rate'] = metrics.get('win_rate', 0)
        trial_data['max_drawdown'] = metrics.get('max_drawdown', 0)
        trial_data['profit_factor'] = metrics.get('profit_factor', 0)
        trial_data['total_trades'] = metrics.get('total_trades', 0)
        trial_data['score'] = trial.get('score', 0)
        
        data.append(trial_data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Remove columns with all NaN or all same values
    df = df.loc[:, df.nunique() > 1]
    
    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix.values, 3),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    # Move score to the beginning of both axes
    if 'score' in corr_matrix.columns:
        score_idx = list(corr_matrix.columns).index('score')
        fig.update_layout(
            xaxis=dict(categoryarray=list(corr_matrix.columns)[score_idx:] + list(corr_matrix.columns)[:score_idx]),
            yaxis=dict(categoryarray=list(corr_matrix.columns)[score_idx:] + list(corr_matrix.columns)[:score_idx])
        )
    
    fig.update_layout(
        title='Correlation Between Parameters and Metrics',
        template="plotly_dark",
        height=700,
        paper_bgcolor="rgba(15, 23, 42, 0.9)",
        plot_bgcolor="rgba(15, 23, 42, 0.9)",
        font=dict(family="'Share Tech Mono', monospace", color="#D1D5DB"),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            tickfont=dict(size=10)
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-grid-3x3 me-2"),
                "Parameter-Metric Correlation Matrix"
            ], className="text-cyan-300 mb-0")
        ], className="bg-dark border-secondary"),
        
        dbc.CardBody([
            html.P([
                html.I(className="bi bi-info-circle me-2"),
                """This heatmap shows correlations between optimization parameters and performance metrics. 
                Strong positive correlations are shown in blue, negative in red."""
            ], className="text-info small mb-3"),
            
            dcc.Graph(
                figure=fig,
                config={
                    'displayModeBar': True,
                    'displaylogo': False
                },
                className="retro-graph"
            )
        ], className="bg-dark p-3")
    ], className="retro-card shadow mb-4")

def create_parameter_importance(best_trials):
    """
    Creates a bar chart showing the relative importance of different parameters.
    
    Args:
        best_trials: List of best optimization trials
        
    Returns:
        Parameter importance component
    """
    if len(best_trials) < 10:
        return html.Div(
            "Not enough trials for parameter importance analysis (minimum 10 required)",
            className="text-center text-muted p-3"
        )
    
    # Extract parameters and scores
    X = []
    y = []
    param_names = set()
    
    for trial in best_trials:
        params = trial.get('params', {})
        score = trial.get('score', 0)
        
        # Only consider numeric parameters
        numeric_params = {}
        for param_name, param_value in params.items():
            if isinstance(param_value, (int, float)):
                numeric_params[param_name] = param_value
                param_names.add(param_name)
        
        if numeric_params:
            X.append(numeric_params)
            y.append(score)
    
    if not X or not param_names:
        return html.Div(
            "No numeric parameters found for importance analysis",
            className="text-center text-muted p-3"
        )
    
    # Convert to DataFrame
    param_names = sorted(list(param_names))
    param_data = []
    
    for params, score in zip(X, y):
        row = {param: params.get(param, np.nan) for param in param_names}
        row['score'] = score
        param_data.append(row)
    
    df = pd.DataFrame(param_data)
    
    # Calculate feature importance using correlation
    importance = {}
    for param in param_names:
        if df[param].nunique() > 1:  # Skip constant parameters
            correlation = abs(df[param].corr(df['score']))
            if not np.isnan(correlation):
                importance[param] = correlation
    
    # Sort by importance
    importance = {k: v for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}
    
    # Create the importance chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(importance.values()),
        y=list(importance.keys()),
        orientation='h',
        marker=dict(
            color=list(importance.values()),
            colorscale='Viridis',
            colorbar=dict(title="Importance"),
            cmid=0.5
        ),
        hovertemplate="Parameter: %{y}<br>Importance: %{x:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        title='Relative Parameter Importance',
        xaxis_title='Importance (absolute correlation with score)',
        yaxis_title='Parameter',
        template="plotly_dark",
        height=min(800, max(400, len(importance) * 30)),  # Dynamic height based on number of parameters
        paper_bgcolor="rgba(15, 23, 42, 0.9)",
        plot_bgcolor="rgba(15, 23, 42, 0.9)",
        font=dict(family="'Share Tech Mono', monospace", color="#D1D5DB"),
        margin=dict(l=200, r=50, t=40, b=40)  # Added left margin for parameter names
    )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-bar-chart-steps me-2"),
                "Parameter Importance Analysis"
            ], className="text-cyan-300 mb-0")
        ], className="bg-dark border-secondary"),
        
        dbc.CardBody([
            html.P([
                html.I(className="bi bi-info-circle me-2"),
                """This chart shows which parameters had the most impact on the optimization score. 
                Higher values indicate stronger correlation with the final score."""
            ], className="text-info small mb-3"),
            
            dcc.Graph(
                figure=fig,
                config={
                    'displayModeBar': True,
                    'displaylogo': False
                },
                className="retro-graph"
            )
        ], className="bg-dark p-3")
    ], className="retro-card shadow mb-4")

def create_metric_distributions(best_trials):
    """
    Creates violin plots showing the distribution of performance metrics.
    
    Args:
        best_trials: List of best optimization trials
        
    Returns:
        Metric distributions component
    """
    if len(best_trials) < 10:
        return html.Div(
            "Not enough trials for metric distribution analysis (minimum 10 required)",
            className="text-center text-muted p-3"
        )
    
    # Extract metrics
    data = []
    for trial in best_trials:
        metrics = trial.get('metrics', {})
        
        data.append({
            'ROI (%)': metrics.get('roi', 0) * 100,
            'Win Rate (%)': metrics.get('win_rate', 0) * 100,
            'Max Drawdown (%)': metrics.get('max_drawdown', 0) * 100,
            'Profit Factor': metrics.get('profit_factor', 0),
            'Total Trades': metrics.get('total_trades', 0),
            'Score': trial.get('score', 0)
        })
    
    df = pd.DataFrame(data)
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'ROI Distribution', 'Win Rate Distribution', 'Max Drawdown Distribution',
            'Profit Factor Distribution', 'Total Trades Distribution', 'Score Distribution'
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # Add violin plots for each metric
    metrics = ['ROI (%)', 'Win Rate (%)', 'Max Drawdown (%)', 'Profit Factor', 'Total Trades', 'Score']
    colors = ['#22D3EE', '#4ADE80', '#F87171', '#A78BFA', '#FBBF24', '#FB923C']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        row = i // 3 + 1
        col = i % 3 + 1
        
        # Calculate statistics
        mean_val = df[metric].mean()
        median_val = df[metric].median()
        std_val = df[metric].std()
        
        # Add violin plot
        fig.add_trace(
            go.Violin(
                y=df[metric],
                box_visible=True,
                line_color=color,
                fillcolor=f'rgba({",".join(str(int(c)) for c in px.colors.hex_to_rgb(color))},0.5)',
                opacity=0.6,
                meanline_visible=True,
                name=metric,
                hoverinfo='y',
                hovertemplate=f"{metric}: %{{y:.2f}}<extra></extra>"
            ),
            row=row, col=col
        )
        
        # Add mean and median annotations
        fig.add_annotation(
            x=0, y=mean_val,
            text=f"Mean: {mean_val:.2f}",
            showarrow=False,
            font=dict(size=10, color="#FFFFFF"),
            bgcolor=f'rgba({",".join(str(int(c)) for c in px.colors.hex_to_rgb(color))},0.7)',
            xanchor="right",
            xshift=-10,
            row=row, col=col
        )
        
        fig.add_annotation(
            x=0, y=median_val,
            text=f"Median: {median_val:.2f}",
            showarrow=False,
            font=dict(size=10, color="#FFFFFF"),
            bgcolor="rgba(0,0,0,0.5)",
            xanchor="left",
            xshift=10,
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        height=800,
        paper_bgcolor="rgba(15, 23, 42, 0.9)",
        plot_bgcolor="rgba(15, 23, 42, 0.9)",
        font=dict(family="'Share Tech Mono', monospace", color="#D1D5DB"),
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Update axes
    for i in range(1, 7):
        fig.update_xaxes(
            showticklabels=False,
            row=i//3 + 1, col=i%3 + 1,
            gridcolor="rgba(51, 65, 85, 0.6)"
        )
        fig.update_yaxes(
            gridcolor="rgba(51, 65, 85, 0.6)",
            row=i//3 + 1, col=i%3 + 1
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-reception-4 me-2"),
                "Performance Metric Distributions"
            ], className="text-cyan-300 mb-0")
        ], className="bg-dark border-secondary"),
        
        dbc.CardBody([
            html.P([
                html.I(className="bi bi-info-circle me-2"),
                """These violin plots show the distribution of performance metrics across trials. 
                The width indicates the density of trials at each value level."""
            ], className="text-info small mb-3"),
            
            dcc.Graph(
                figure=fig,
                config={
                    'displayModeBar': True,
                    'displaylogo': False
                },
                className="retro-graph"
            )
        ], className="bg-dark p-3")
    ], className="retro-card shadow mb-4")

def create_optimization_progress(best_trials):
    """
    Creates a visualization of the optimization progress over trials.
    
    Args:
        best_trials: List of best optimization trials
        
    Returns:
        Optimization progress component
    """
    if len(best_trials) < 5:
        return html.Div(
            "Not enough trials for optimization progress analysis",
            className="text-center text-muted p-3"
        )
    
    # Sort trials by ID
    sorted_trials = sorted(best_trials, key=lambda t: t.get('trial_id', 0))
    
    trial_ids = [t.get('trial_id', 0) for t in sorted_trials]
    scores = [t.get('score', 0) for t in sorted_trials]
    
    # Calculate best score so far
    best_scores = []
    current_best = float('-inf')
    for score in scores:
        current_best = max(current_best, score)
        best_scores.append(current_best)
    
    # Get ROI values
    rois = [t.get('metrics', {}).get('roi', 0) * 100 for t in sorted_trials]
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Score Evolution", "ROI Evolution")
    )
    
    # Add score trace
    fig.add_trace(
        go.Scatter(
            x=trial_ids,
            y=scores,
            mode='markers',
            name='Trial Score',
            marker=dict(
                size=8,
                color=scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Score", y=0.85, len=0.3)
            ),
            hovertemplate="Trial #%{x}<br>Score: %{y:.4f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Add best score trace
    fig.add_trace(
        go.Scatter(
            x=trial_ids,
            y=best_scores,
            mode='lines',
            name='Best Score',
            line=dict(color='red', width=2),
            hovertemplate="Trial #%{x}<br>Best Score: %{y:.4f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Add ROI trace
    fig.add_trace(
        go.Scatter(
            x=trial_ids,
            y=rois,
            mode='markers+lines',
            name='ROI (%)',
            marker=dict(size=8),
            line=dict(color='#4ADE80', width=1),
            hovertemplate="Trial #%{x}<br>ROI: %{y:.2f}%<extra></extra>"
        ),
        row=2, col=1
    )
    
    # Add a reference line at ROI = 0
    fig.add_trace(
        go.Scatter(
            x=[min(trial_ids), max(trial_ids)],
            y=[0, 0],
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        height=600,
        paper_bgcolor="rgba(15, 23, 42, 0.9)",
        plot_bgcolor="rgba(15, 23, 42, 0.9)",
        font=dict(family="'Share Tech Mono', monospace", color="#D1D5DB"),
        legend=dict(orientation="h", y=1.1),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    fig.update_xaxes(
        title="Trial ID",
        gridcolor="rgba(51, 65, 85, 0.6)",
        row=2, col=1
    )
    
    fig.update_yaxes(
        gridcolor="rgba(51, 65, 85, 0.6)",
        row=1, col=1
    )
    
    fig.update_yaxes(
        title="ROI (%)",
        gridcolor="rgba(51, 65, 85, 0.6)",
        row=2, col=1
    )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-graph-up-arrow me-2"),
                "Optimization Progress"
            ], className="text-cyan-300 mb-0")
        ], className="bg-dark border-secondary"),
        
        dbc.CardBody([
            html.P([
                html.I(className="bi bi-info-circle me-2"),
                """This visualization shows how the optimization score and ROI evolved across trials.
                The red line shows the best score found so far."""
            ], className="text-info small mb-3"),
            
            dcc.Graph(
                figure=fig,
                config={
                    'displayModeBar': True,
                    'displaylogo': False
                },
                className="retro-graph"
            )
        ], className="bg-dark p-3")
    ], className="retro-card shadow mb-4")

def create_strategy_composition(best_trials):
    """
    Creates a visualization of the composition of strategies across best trials.
    
    Args:
        best_trials: List of best optimization trials
        
    Returns:
        Strategy composition component
    """
    if len(best_trials) < 5:
        return html.Div(
            "Not enough trials for strategy composition analysis",
            className="text-center text-muted p-3"
        )
    
    # Extract strategy composition information
    buy_indicators = {}
    sell_indicators = {}
    buy_operators = {}
    sell_operators = {}
    buy_periods = []
    sell_periods = []
    buy_blocks = []
    sell_blocks = []
    position_sizes = []
    stop_losses = []
    take_profits = []
    leverages = []
    risk_modes = {}
    
    for trial in best_trials:
        params = trial.get('params', {})
        
        # Extract number of blocks
        n_buy_blocks = params.get('n_buy_blocks', 0)
        n_sell_blocks = params.get('n_sell_blocks', 0)
        buy_blocks.append(n_buy_blocks)
        sell_blocks.append(n_sell_blocks)
        
        # Extract indicators, operators, and periods
        for b in range(n_buy_blocks):
            for c in range(params.get(f'buy_block_{b}_conditions', 0)):
                # Extract indicators
                ind_type = params.get(f'buy_b{b}_c{c}_ind1_type', '')
                if ind_type:
                    buy_indicators[ind_type] = buy_indicators.get(ind_type, 0) + 1
                
                # Extract operators
                op = params.get(f'buy_b{b}_c{c}_operator', '')
                if op:
                    buy_operators[op] = buy_operators.get(op, 0) + 1
                
                # Extract periods
                period = params.get(f'buy_b{b}_c{c}_ind1_period', 0)
                if period:
                    buy_periods.append(period)
        
        for b in range(n_sell_blocks):
            for c in range(params.get(f'sell_block_{b}_conditions', 0)):
                # Extract indicators
                ind_type = params.get(f'sell_b{b}_c{c}_ind1_type', '')
                if ind_type:
                    sell_indicators[ind_type] = sell_indicators.get(ind_type, 0) + 1
                
                # Extract operators
                op = params.get(f'sell_b{b}_c{c}_operator', '')
                if op:
                    sell_operators[op] = sell_operators.get(op, 0) + 1
                
                # Extract periods
                period = params.get(f'sell_b{b}_c{c}_ind1_period', 0)
                if period:
                    sell_periods.append(period)
        
        # Extract risk parameters
        risk_mode = params.get('risk_mode', '')
        if risk_mode:
            risk_modes[risk_mode] = risk_modes.get(risk_mode, 0) + 1
        
        position_size = params.get('base_position', 0)
        if position_size:
            position_sizes.append(position_size * 100)  # Convert to percentage
        
        sl = params.get('base_sl', 0)
        if sl:
            stop_losses.append(sl * 100)  # Convert to percentage
        
        tp_mult = params.get('tp_multiplier', 0)
        if tp_mult and sl:
            take_profits.append(tp_mult * sl * 100)  # Convert to percentage
        
        leverage = params.get('leverage', 0)
        if leverage:
            leverages.append(leverage)
    
    # Create figure with 3x3 subplots
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'Buy Indicators', 'Sell Indicators', 'Blocks',
            'Buy Operators', 'Sell Operators', 'Indicator Periods',
            'Risk Parameters', 'Position Sizing', 'Risk Mode Distribution'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'bar'}, {'type': 'box'}],
            [{'type': 'bar'}, {'type': 'bar'}, {'type': 'box'}],
            [{'type': 'box'}, {'type': 'box'}, {'type': 'pie'}]
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.05
    )
    
    # Add traces
    # 1. Buy Indicators
    if buy_indicators:
        fig.add_trace(
            go.Bar(
                x=list(buy_indicators.keys()),
                y=list(buy_indicators.values()),
                marker_color='#22D3EE',
                name='Buy Indicators'
            ),
            row=1, col=1
        )
    
    # 2. Sell Indicators
    if sell_indicators:
        fig.add_trace(
            go.Bar(
                x=list(sell_indicators.keys()),
                y=list(sell_indicators.values()),
                marker_color='#F87171',
                name='Sell Indicators'
            ),
            row=1, col=2
        )
    
    # 3. Blocks
    fig.add_trace(
        go.Box(
            y=buy_blocks,
            name='Buy Blocks',
            marker_color='#22D3EE',
            boxmean=True
        ),
        row=1, col=3
    )
    
    fig.add_trace(
        go.Box(
            y=sell_blocks,
            name='Sell Blocks',
            marker_color='#F87171',
            boxmean=True
        ),
        row=1, col=3
    )
    
    # 4. Buy Operators
    if buy_operators:
        fig.add_trace(
            go.Bar(
                x=list(buy_operators.keys()),
                y=list(buy_operators.values()),
                marker_color='#22D3EE',
                name='Buy Operators'
            ),
            row=2, col=1
        )
    
    # 5. Sell Operators
    if sell_operators:
        fig.add_trace(
            go.Bar(
                x=list(sell_operators.keys()),
                y=list(sell_operators.values()),
                marker_color='#F87171',
                name='Sell Operators'
            ),
            row=2, col=2
        )
    
    # 6. Periods
    fig.add_trace(
        go.Box(
            y=buy_periods,
            name='Buy Periods',
            marker_color='#22D3EE',
            boxmean=True
        ),
        row=2, col=3
    )
    
    fig.add_trace(
        go.Box(
            y=sell_periods,
            name='Sell Periods',
            marker_color='#F87171',
            boxmean=True
        ),
        row=2, col=3
    )
    
    # 7. Position Sizing, Stop Loss, Take Profit
    fig.add_trace(
        go.Box(
            y=position_sizes,
            name='Position Size (%)',
            marker_color='#22D3EE',
            boxmean=True
        ),
        row=3, col=1
    )
    
    # 8. Risk Parameters
    fig.add_trace(
        go.Box(
            y=stop_losses,
            name='Stop Loss (%)',
            marker_color='#F87171',
            boxmean=True
        ),
        row=3, col=2
    )
    
    fig.add_trace(
        go.Box(
            y=take_profits,
            name='Take Profit (%)',
            marker_color='#4ADE80',
            boxmean=True
        ),
        row=3, col=2
    )
    
    fig.add_trace(
        go.Box(
            y=leverages,
            name='Leverage',
            marker_color='#A78BFA',
            boxmean=True
        ),
        row=3, col=2
    )
    
    # 9. Risk Mode Distribution
    if risk_modes:
        fig.add_trace(
            go.Pie(
                labels=list(risk_modes.keys()),
                values=list(risk_modes.values()),
                textinfo='percent',
                marker=dict(
                    colors=['#22D3EE', '#4ADE80', '#F87171', '#A78BFA', '#FBBF24']
                ),
                name='Risk Modes'
            ),
            row=3, col=3
        )
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        height=900,
        paper_bgcolor="rgba(15, 23, 42, 0.9)",
        plot_bgcolor="rgba(15, 23, 42, 0.9)",
        font=dict(family="'Share Tech Mono', monospace", color="#D1D5DB"),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        margin=dict(l=0, r=0, t=40, b=50)
    )
    
    # Update all axes
    for i in range(1, 4):
        for j in range(1, 4):
            fig.update_xaxes(
                gridcolor="rgba(51, 65, 85, 0.6)",
                row=i, col=j
            )
            fig.update_yaxes(
                gridcolor="rgba(51, 65, 85, 0.6)",
                row=i, col=j
            )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="bi bi-clipboard-data-fill me-2"),
                "Strategy Composition Analysis"
            ], className="text-cyan-300 mb-0")
        ], className="bg-dark border-secondary"),
        
        dbc.CardBody([
            html.P([
                html.I(className="bi bi-info-circle me-2"),
                """This analysis shows the composition of successful strategies across trials, 
                including indicators, operators, and risk parameters."""
            ], className="text-info small mb-3"),
            
            dcc.Graph(
                figure=fig,
                config={
                    'displayModeBar': True,
                    'displaylogo': False
                },
                className="retro-graph"
            )
        ], className="bg-dark p-3")
    ], className="retro-card shadow mb-4")