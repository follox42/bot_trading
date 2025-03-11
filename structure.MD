trading_nexus/
│
├── main.py                      # Point d'entrée principal
├── config.py                    # Configuration globale de l'application
│
├── logger/
│   ├── __init__.py
│   └── logger.py                # Système de logging centralisé
│
├── ui/
│   ├── __init__.py
│   ├── app.py                   # Configuration de l'application Dash
│   ├── splash_screen.py         # Écran d'introduction
│   ├── dashboard.py             # Tableau de bord principal
│   ├── components/              # Composants UI réutilisables
│   │   ├── __init__.py
│   │   ├── header.py            # En-tête de page
│   │   ├── footer.py            # Pied de page
│   │   ├── sidebar.py           # Barre latérale de navigation
│   │   ├── stats_card.py        # Carte de statistiques
│   │   ├── log_viewer.py        # Visualiseur de logs
│   │   └── charts.py            # Composants de graphiques
│   │
│   └── pages/                   # Pages de l'application
│       ├── __init__.py
│       ├── strategies.py        # Page de gestion des stratégies
│       ├── data.py              # Page de gestion des données
│       ├── backtest.py          # Page de backtest
│       ├── logs.py              # Page de logs complète
│       └── settings.py          # Page de paramètres
│
├── data/
│   ├── __init__.py
│   └── mock_data.py             # Données de démonstration
│
└── assets/                      # Fichiers statiques (CSS, images)
    └── styles.css               # Styles CSS pour l'interface