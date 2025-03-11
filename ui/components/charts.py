import plotly.graph_objects as go
from dash import dcc, html

def create_performance_chart(performance_data):
    """
    Crée un graphique de performance avec deux axes
    
    Args:
        performance_data: DataFrame contenant les données de performance
    
    Returns:
        Composant de graphique
    """
    # Créer la figure Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=performance_data['date'],
        y=performance_data['value'],
        mode='lines+markers',
        name='Performance (%)',
        line=dict(color='#22D3EE', width=2),
        marker=dict(size=8, color='#22D3EE')
    ))
    
    fig.add_trace(go.Scatter(
        x=performance_data['date'],
        y=performance_data['equity'],
        mode='lines+markers',
        name='Équité ($)',
        yaxis='y2',
        line=dict(color='#4ADE80', width=2),
        marker=dict(size=8, color='#4ADE80')
    ))
    
    # Noter que 'titlefont' a été remplacé par 'title_font' pour corriger l'erreur
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1F2937',
        plot_bgcolor='#1F2937',
        margin=dict(l=20, r=20, t=20, b=20),
        height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#D1D5DB')
        ),
        yaxis=dict(
            title='Performance (%)',
            title_font=dict(color='#22D3EE'),  # Corrigé : title_font au lieu de titlefont
            tickfont=dict(color='#9CA3AF'),
            gridcolor='#374151',
            zerolinecolor='#4B5563'
        ),
        yaxis2=dict(
            title='Équité ($)',
            title_font=dict(color='#4ADE80'),  # Corrigé : title_font au lieu de titlefont
            tickfont=dict(color='#9CA3AF'),
            overlaying='y',
            side='right'
        ),
        xaxis=dict(
            gridcolor='#374151',
            zerolinecolor='#4B5563',
            tickfont=dict(color='#9CA3AF')
        )
    )
    
    return html.Div([
        html.Div(
            className="retro-card-header",
            children=[
                html.H3("PERFORMANCE GLOBALE", className="retro-card-title")
            ]
        ),
        html.Div(
            className="retro-card-body",
            children=[
                dcc.Graph(figure=fig, config={'displayModeBar': False})
            ]
        )
    ], className="retro-card mb-4")

def create_strategies_table(strategy_data):
    """
    Crée un tableau des stratégies
    
    Args:
        strategy_data: Liste de dictionnaires contenant les données des stratégies
    
    Returns:
        Composant de tableau HTML
    """
    from dash import html
    
    strategies_table = html.Table([
        html.Thead(
            html.Tr([
                html.Th("NOM", className="text-left"),
                html.Th("PAIR", className="text-left"),
                html.Th("TYPE", className="text-center"),
                html.Th("PERF", className="text-right"),
                html.Th("STATUT", className="text-right")
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td(strategy["name"], className="font-weight-bold"),
                html.Td(strategy["symbol"]),
                html.Td(
                    html.Span(
                        strategy["type"], 
                        className=f"retro-badge retro-badge-{'blue' if strategy['type']=='trend' else 'yellow' if strategy['type']=='volatility' else 'purple'}"
                    ),
                    className="text-center"
                ),
                html.Td(
                    html.Span(
                        f"{'+' if strategy['performance'] > 0 else ''}{strategy['performance']}%", 
                        style={"color": "#4ADE80" if strategy["performance"] > 0 else "#F87171"},
                        className="text-right d-block"
                    )
                ),
                html.Td(
                    html.Span(
                        strategy["status"], 
                        className=f"retro-badge retro-badge-{'green' if strategy['status']=='active' else 'yellow'}",
                        style={"float": "right"}
                    )
                )
            ]) for strategy in strategy_data
        ])
    ], className="retro-table w-100")
    
    return html.Div([
        html.Div(
            className="retro-card-header",
            children=[
                html.H3("STRATÉGIES ACTIVES", className="retro-card-title")
            ]
        ),
        html.Div(
            className="retro-card-body",
            children=[
                strategies_table,
                html.Div([
                    html.Button(
                        "Toutes les stratégies >>", 
                        id="btn-view-all-strategies", 
                        className="retro-button secondary mt-3", 
                        style={"float": "right"}
                    )
                ], style={"textAlign": "right"})
            ]
        )
    ], className="retro-card")