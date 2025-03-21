"""
Interface de trading simplifiée.
Cette interface conserve les fonctionnalités essentielles du système de trading
tout en offrant une expérience utilisateur plus simple et intuitive.
"""
import sys
import os
# Ajouter le répertoire parent au chemin d'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import logging
from datetime import datetime

# Importation des modules du système de trading
from core.study.study_manager import StudyManager
from data.data_manager import get_data_manager
from core.strategy.strategy_manager import create_strategy_manager_for_study
from core.optimization.parallel_optimizer import OptimizationConfig
from core.optimization.search_config import get_predefined_search_space
from core.strategy.indicators.indicators_config import IndicatorType, IndicatorConfig
from core.strategy.conditions.conditions_config import (
    ConditionConfig, BlockConfig, OperatorType,
    IndicatorOperand, ValueOperand
)
from core.strategy.risk.risk_config import RiskConfig, RiskModeType

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading.log')
    ]
)
logger = logging.getLogger('trading_interface')

class SimpleTradingApp:
    """Interface simplifiée pour le système de trading."""
    
    def __init__(self, root):
        """Initialise l'interface principale."""
        self.root = root
        self.root.title("Système de Trading - Interface Simplifiée")
        self.root.geometry("1000x700")
        
        # Initialisation des managers
        self.study_manager = StudyManager()
        self.data_manager = get_data_manager()
        self.strategy_manager = None
        
        # Variables pour stocker les données actuelles
        self.current_study = None
        self.current_data = None
        
        # Création de l'interface
        self.create_ui()
        
        # Rafraîchissement initial des études
        self.refresh_studies()
    
    def create_ui(self):
        """Crée l'interface utilisateur avec des onglets simplifiés."""
        # Création du notebook principal
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Création des onglets principaux
        self.create_studies_tab()
        self.create_strategies_tab()
        self.create_optimization_tab()
        
        # Barre de statut
        self.status_var = tk.StringVar(value="Prêt")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_studies_tab(self):
        """Crée l'onglet de gestion des études de trading."""
        studies_frame = ttk.Frame(self.notebook)
        self.notebook.add(studies_frame, text="Études & Données")
        
        # Panneau de gauche: liste des études
        left_frame = ttk.LabelFrame(studies_frame, text="Études disponibles")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Liste des études avec un Treeview
        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ("name", "asset", "timeframe", "data")
        self.studies_tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        self.studies_tree.heading("name", text="Nom")
        self.studies_tree.heading("asset", text="Asset")
        self.studies_tree.heading("timeframe", text="Timeframe")
        self.studies_tree.heading("data", text="Données")
        
        self.studies_tree.column("name", width=150)
        self.studies_tree.column("asset", width=100)
        self.studies_tree.column("timeframe", width=80)
        self.studies_tree.column("data", width=80)
        
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.studies_tree.yview)
        self.studies_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.studies_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Liaisons d'événements
        self.studies_tree.bind('<<TreeviewSelect>>', self.on_study_selected)
        
        # Boutons d'action
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Créer étude", command=self.create_study).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Rafraîchir", command=self.refresh_studies).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Supprimer", command=self.delete_study).pack(side=tk.LEFT, padx=2)
        
        # Panneau de droite: détails et données
        right_frame = ttk.Frame(studies_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Sous-panneau pour les détails de l'étude
        details_frame = ttk.LabelFrame(right_frame, text="Détails de l'étude")
        details_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        self.study_details = scrolledtext.ScrolledText(details_frame, height=8, wrap=tk.WORD)
        self.study_details.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Sous-panneau pour les données
        data_frame = ttk.LabelFrame(right_frame, text="Données")
        data_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        data_control_frame = ttk.Frame(data_frame)
        data_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(data_control_frame, text="Télécharger données", command=self.download_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(data_control_frame, text="Importer CSV", command=self.import_csv_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(data_control_frame, text="Afficher graphique", command=self.show_data_chart).pack(side=tk.LEFT, padx=5)
        
        # Zone d'information sur les données
        self.data_info = scrolledtext.ScrolledText(data_frame, height=4, wrap=tk.WORD)
        self.data_info.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        # Zone de visualisation des données
        self.chart_frame = ttk.Frame(data_frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_strategies_tab(self):
        """Crée l'onglet de gestion des stratégies de trading."""
        strategies_frame = ttk.Frame(self.notebook)
        self.notebook.add(strategies_frame, text="Stratégies")
        
        # Panneau supérieur: sélection de l'étude
        top_frame = ttk.Frame(strategies_frame)
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(top_frame, text="Étude:").pack(side=tk.LEFT, padx=5)
        self.strategy_study_var = tk.StringVar()
        self.strategy_study_combo = ttk.Combobox(top_frame, textvariable=self.strategy_study_var, state="readonly", width=25)
        self.strategy_study_combo.pack(side=tk.LEFT, padx=5)
        self.strategy_study_combo.bind("<<ComboboxSelected>>", self.on_strategy_study_selected)
        
        ttk.Button(top_frame, text="Créer stratégie", command=self.create_strategy).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Tester stratégie", command=self.test_strategy).pack(side=tk.LEFT, padx=5)
        
        # Panneau principal divisé en deux
        main_frame = ttk.Frame(strategies_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panneau de gauche: liste des stratégies
        left_frame = ttk.LabelFrame(main_frame, text="Stratégies disponibles")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5, ipadx=5, ipady=5)
        left_frame.pack_propagate(False)
        left_frame.config(width=250)
        
        # Liste des stratégies
        self.strategies_listbox = tk.Listbox(left_frame)
        self.strategies_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.strategies_listbox.bind('<<ListboxSelect>>', self.on_strategy_selected)
        
        # Panneau de droite: détails et résultats
        right_frame = ttk.LabelFrame(main_frame, text="Détails et résultats")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Notebook pour les détails, résultats et graphiques
        strat_notebook = ttk.Notebook(right_frame)
        strat_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Onglet des détails
        details_tab = ttk.Frame(strat_notebook)
        strat_notebook.add(details_tab, text="Configuration")
        
        self.strategy_details = scrolledtext.ScrolledText(details_tab, wrap=tk.WORD)
        self.strategy_details.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Onglet des résultats
        results_tab = ttk.Frame(strat_notebook)
        strat_notebook.add(results_tab, text="Résultats")
        
        self.strategy_results = scrolledtext.ScrolledText(results_tab, wrap=tk.WORD)
        self.strategy_results.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Onglet du graphique
        chart_tab = ttk.Frame(strat_notebook)
        strat_notebook.add(chart_tab, text="Graphique")
        
        self.strategy_chart_frame = ttk.Frame(chart_tab)
        self.strategy_chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_optimization_tab(self):
        """Crée l'onglet d'optimisation des stratégies."""
        optimization_frame = ttk.Frame(self.notebook)
        self.notebook.add(optimization_frame, text="Optimisation")
        
        # Panneau supérieur: sélection de l'étude
        top_frame = ttk.Frame(optimization_frame)
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(top_frame, text="Étude:").pack(side=tk.LEFT, padx=5)
        self.optim_study_var = tk.StringVar()
        self.optim_study_combo = ttk.Combobox(top_frame, textvariable=self.optim_study_var, state="readonly", width=25)
        self.optim_study_combo.pack(side=tk.LEFT, padx=5)
        self.optim_study_combo.bind("<<ComboboxSelected>>", self.on_optim_study_selected)
        
        # Panneau principal divisé
        main_frame = ttk.Frame(optimization_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panneau de gauche: configuration
        left_frame = ttk.LabelFrame(main_frame, text="Configuration de l'optimisation")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        left_frame.pack_propagate(False)
        left_frame.config(width=300)
        
        # Espace de recherche
        search_frame = ttk.LabelFrame(left_frame, text="Espace de recherche")
        search_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(search_frame, text="Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.search_type_var = tk.StringVar(value="default")
        self.search_type_combo = ttk.Combobox(search_frame, textvariable=self.search_type_var, width=15,
                                              values=["default", "trend_following", "mean_reversion"])
        self.search_type_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # Configuration des essais
        trials_frame = ttk.LabelFrame(left_frame, text="Configuration des essais")
        trials_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(trials_frame, text="Nombre d'essais:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.trials_var = tk.IntVar(value=20)
        ttk.Spinbox(trials_frame, from_=5, to=100, textvariable=self.trials_var, width=5).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(trials_frame, text="Processus parallèles:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.jobs_var = tk.IntVar(value=2)
        ttk.Spinbox(trials_frame, from_=1, to=8, textvariable=self.jobs_var, width=5).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(trials_frame, text="Trades minimum:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.min_trades_var = tk.IntVar(value=10)
        ttk.Spinbox(trials_frame, from_=1, to=50, textvariable=self.min_trades_var, width=5).grid(row=2, column=1, padx=5, pady=5)
        
        # Boutons d'action
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(btn_frame, text="Lancer l'optimisation", command=self.start_optimization).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Arrêter", command=self.stop_optimization).pack(side=tk.LEFT, padx=5)
        
        # Panneau de droite: résultats
        right_frame = ttk.LabelFrame(main_frame, text="Résultats de l'optimisation")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Barre de progression
        progress_frame = ttk.Frame(right_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(progress_frame, text="Progression:").pack(side=tk.LEFT, padx=5)
        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=300,
                                           mode='determinate', variable=self.progress_var)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.optim_status_var = tk.StringVar(value="Prêt")
        ttk.Label(progress_frame, textvariable=self.optim_status_var).pack(side=tk.RIGHT, padx=5)
        
        # Notebook pour les résultats
        results_notebook = ttk.Notebook(right_frame)
        results_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Onglet des meilleurs essais
        trials_tab = ttk.Frame(results_notebook)
        results_notebook.add(trials_tab, text="Essais")
        
        trials_list_frame = ttk.Frame(trials_tab)
        trials_list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        trials_list_frame.pack_propagate(False)
        trials_list_frame.config(width=200)
        
        ttk.Label(trials_list_frame, text="Meilleurs essais:").pack(fill=tk.X, padx=5, pady=2)
        
        self.trials_listbox = tk.Listbox(trials_list_frame)
        self.trials_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.trials_listbox.bind('<<ListboxSelect>>', self.on_trial_selected)
        
        trial_detail_frame = ttk.Frame(trials_tab)
        trial_detail_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.trial_details = scrolledtext.ScrolledText(trial_detail_frame, wrap=tk.WORD)
        self.trial_details.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Onglet du journal
        logs_tab = ttk.Frame(results_notebook)
        results_notebook.add(logs_tab, text="Journal")
        
        self.optim_logs = scrolledtext.ScrolledText(logs_tab, wrap=tk.WORD)
        self.optim_logs.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def refresh_studies(self):
        """Rafraîchit la liste des études disponibles."""
        studies = self.study_manager.list_studies()
        
        # Mise à jour du TreeView
        self.studies_tree.delete(*self.studies_tree.get_children())
        
        study_names = []
        for study in studies:
            name = study.get('name')
            asset = study.get('asset', 'N/A')
            timeframe = study.get('timeframe', 'N/A')
            data_count = study.get('data_count', 0)
            self.studies_tree.insert("", tk.END, values=(name, asset, timeframe, f"{data_count} fichiers"))
            study_names.append(name)
        
        # Mise à jour des combobox
        self.strategy_study_combo['values'] = study_names
        self.optim_study_combo['values'] = study_names
        
        self.status_var.set(f"Études chargées: {len(studies)}")
    
    def on_study_selected(self, event):
        """Gère la sélection d'une étude dans le TreeView."""
        selection = self.studies_tree.selection()
        if not selection:
            return
        
        selected_item = self.studies_tree.item(selection[0])
        study_name = selected_item['values'][0]
        self.current_study = study_name
        
        # Affichage des détails de l'étude
        study = self.study_manager.get_study(study_name)
        if study:
            details = f"Nom: {study.get('name')}\n"
            details += f"Description: {study.get('description', 'N/A')}\n"
            details += f"Asset: {study.get('asset')}\n"
            details += f"Timeframe: {study.get('timeframe')}\n"
            details += f"Exchange: {study.get('exchange')}\n"
            details += f"Statut: {study.get('status')}\n"
            details += f"Tags: {', '.join(study.get('tags', []))}\n"
            
            self.study_details.delete(1.0, tk.END)
            self.study_details.insert(tk.END, details)
            
            # Chargement des données si disponibles
            self.load_study_data(study_name)
    
    def load_study_data(self, study_name):
        """Charge les données d'une étude."""
        try:
            data = self.data_manager.load_study_data(study_name)
            self.current_data = data
            
            if data is not None:
                self.show_data_info()
            else:
                self.data_info.delete(1.0, tk.END)
                self.data_info.insert(tk.END, "Aucune donnée disponible pour cette étude.\n")
                self.data_info.insert(tk.END, "Utilisez 'Télécharger données' ou 'Importer CSV' pour ajouter des données.")
                
                # Effacer le graphique
                for widget in self.chart_frame.winfo_children():
                    widget.destroy()
        except Exception as e:
            self.data_info.delete(1.0, tk.END)
            self.data_info.insert(tk.END, f"Erreur lors du chargement des données: {str(e)}")
            logger.error(f"Erreur lors du chargement des données pour {study_name}: {str(e)}")
    
    def show_data_info(self):
        """Affiche les informations sur les données chargées."""
        if self.current_data is None:
            return
        
        rows = len(self.current_data)
        cols = len(self.current_data.columns)
        
        info = f"Données: {rows} lignes × {cols} colonnes\n"
        info += f"Colonnes: {', '.join(self.current_data.columns[:5])}...\n"
        
        if 'timestamp' in self.current_data.columns:
            start_date = self.current_data['timestamp'].min()
            end_date = self.current_data['timestamp'].max()
            info += f"Période: du {start_date} au {end_date}\n"
        
        if 'close' in self.current_data.columns:
            min_price = self.current_data['close'].min()
            max_price = self.current_data['close'].max()
            info += f"Prix: min={min_price:.2f}, max={max_price:.2f}\n"
        
        self.data_info.delete(1.0, tk.END)
        self.data_info.insert(tk.END, info)
    
    def show_data_chart(self):
        """Affiche un graphique des données pour l'étude courante."""
        if self.current_data is None:
            messagebox.showwarning("Avertissement", "Aucune donnée à afficher")
            return
        
        # Nettoyer le frame du graphique
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        if 'timestamp' not in self.current_data.columns or 'close' not in self.current_data.columns:
            messagebox.showwarning("Avertissement", "Données incorrectes pour le graphique")
            return
        
        try:
            # Création de la figure matplotlib
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Échantillonnage si beaucoup de données
            data = self.current_data
            if len(data) > 1000:
                data = data.iloc[::len(data)//1000].copy()
            
            # Tracer le prix de clôture
            ax.plot(data['timestamp'], data['close'], label='Prix', color='blue')
            
            # Ajouter le volume si disponible
            if 'volume' in data.columns and data['volume'].sum() > 0:
                ax2 = ax.twinx()
                ax2.bar(data['timestamp'], data['volume'], alpha=0.3, color='gray', label='Volume')
                ax2.set_ylabel('Volume')
            
            # Configuration du graphique
            ax.set_title(f"Prix de {self.current_study}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Prix")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            
            # Rotation des dates sur l'axe x
            fig.autofmt_xdate()
            
            # Affichage dans l'interface
            canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la création du graphique: {str(e)}")
            logger.error(f"Erreur lors de la création du graphique: {str(e)}")
    
    def create_study(self):
        """Crée une nouvelle étude de trading."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Créer une nouvelle étude")
        dialog.geometry("400x300")
        dialog.grab_set()
        
        ttk.Label(dialog, text="Nom:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=name_var, width=30).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Description:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        description_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=description_var, width=30).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Asset:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        asset_var = tk.StringVar(value="BTC/USDT")
        ttk.Entry(dialog, textvariable=asset_var, width=30).grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Exchange:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        exchange_var = tk.StringVar(value="binance")
        ttk.Combobox(dialog, textvariable=exchange_var, values=["binance", "bitget"], width=28).grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Timeframe:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        timeframe_var = tk.StringVar(value="1h")
        ttk.Combobox(dialog, textvariable=timeframe_var, values=["1m", "5m", "15m", "30m", "1h", "4h", "1d"], width=28).grid(row=4, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Tags (séparés par virgules):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        tags_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=tags_var, width=30).grid(row=5, column=1, padx=5, pady=5)
        
        def save_study():
            name = name_var.get().strip()
            if not name:
                messagebox.showerror("Erreur", "Le nom est obligatoire")
                return
            
            tags = []
            if tags_var.get():
                tags = [tag.strip() for tag in tags_var.get().split(',')]
            
            try:
                study_name = self.study_manager.create_study(
                    name=name,
                    description=description_var.get(),
                    asset=asset_var.get(),
                    exchange=exchange_var.get(),
                    timeframe=timeframe_var.get(),
                    tags=tags
                )
                
                if study_name:
                    messagebox.showinfo("Succès", f"Étude '{study_name}' créée avec succès")
                    self.refresh_studies()
                    dialog.destroy()
                else:
                    messagebox.showerror("Erreur", "Échec de la création de l'étude")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur: {str(e)}")
                logger.error(f"Erreur lors de la création de l'étude: {str(e)}")
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.grid(row=6, column=0, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="Créer", command=save_study).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Annuler", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def delete_study(self):
        """Supprime l'étude sélectionnée."""
        selection = self.studies_tree.selection()
        if not selection:
            messagebox.showwarning("Avertissement", "Veuillez sélectionner une étude")
            return
        
        selected_item = self.studies_tree.item(selection[0])
        study_name = selected_item['values'][0]
        
        if messagebox.askyesno("Confirmation", f"Êtes-vous sûr de vouloir supprimer l'étude '{study_name}' ?"):
            try:
                success = self.study_manager.delete_study(study_name)
                if success:
                    messagebox.showinfo("Succès", f"Étude '{study_name}' supprimée")
                    self.refresh_studies()
                    
                    # Nettoyer les zones d'information
                    self.study_details.delete(1.0, tk.END)
                    self.data_info.delete(1.0, tk.END)
                    
                    # Nettoyer le graphique
                    for widget in self.chart_frame.winfo_children():
                        widget.destroy()
                else:
                    messagebox.showerror("Erreur", "Échec de la suppression")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur: {str(e)}")
                logger.error(f"Erreur lors de la suppression de l'étude: {str(e)}")
    
    def download_data(self):
        """Télécharge des données pour l'étude sélectionnée."""
        if not self.current_study:
            messagebox.showwarning("Avertissement", "Veuillez d'abord sélectionner une étude")
            return
        
        study = self.study_manager.get_study(self.current_study)
        if not study:
            messagebox.showerror("Erreur", "Étude introuvable")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Télécharger des données")
        dialog.geometry("400x250")
        dialog.grab_set()
        
        ttk.Label(dialog, text="Exchange:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        exchange_var = tk.StringVar(value=study.get('exchange', 'binance'))
        ttk.Combobox(dialog, textvariable=exchange_var, values=["binance", "bitget"]).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Symbole:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        symbol_var = tk.StringVar(value=study.get('asset', 'BTC/USDT'))
        ttk.Entry(dialog, textvariable=symbol_var).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Timeframe:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        timeframe_var = tk.StringVar(value=study.get('timeframe', '1h'))
        ttk.Combobox(dialog, textvariable=timeframe_var, values=["1m", "5m", "15m", "30m", "1h", "4h", "1d"]).grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Date de début:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        start_date_var = tk.StringVar(value="2023-01-01")
        ttk.Entry(dialog, textvariable=start_date_var).grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Date de fin:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        end_date_var = tk.StringVar(value="2023-12-31")
        ttk.Entry(dialog, textvariable=end_date_var).grid(row=4, column=1, padx=5, pady=5)
        
        progress_var = tk.IntVar(value=0)
        progress_bar = ttk.Progressbar(dialog, orient=tk.HORIZONTAL, length=300, mode='determinate', variable=progress_var)
        progress_bar.grid(row=5, column=0, columnspan=2, padx=5, pady=10, sticky=tk.EW)
        
        status_var = tk.StringVar(value="Prêt")
        status_label = ttk.Label(dialog, textvariable=status_var)
        status_label.grid(row=6, column=0, columnspan=2, pady=5)
        
        def start_download():
            status_var.set("Téléchargement en cours...")
            dialog.update_idletasks()
            
            def download_thread():
                try:
                    def progress_callback(progress_info):
                        progress = progress_info.progress
                        self.root.after(0, lambda: progress_var.set(progress))
                        status_text = f"Téléchargement: {progress}% - "
                        status_text += f"Lot {progress_info.batch_count}/{progress_info.max_batches}"
                        self.root.after(0, lambda: status_var.set(status_text))
                    
                    def should_cancel():
                        return False
                    
                    market_data = self.data_manager.download_data(
                        exchange=exchange_var.get(),
                        symbol=symbol_var.get(),
                        timeframe=timeframe_var.get(),
                        start_date=start_date_var.get(),
                        end_date=end_date_var.get(),
                        progress_callback=progress_callback,
                        should_cancel=should_cancel
                    )
                    
                    if market_data:
                        self.data_manager.associate_study_with_data(
                            study_name=self.current_study,
                            exchange=exchange_var.get(),
                            symbol=symbol_var.get(),
                            timeframe=timeframe_var.get()
                        )
                        
                        self.root.after(0, lambda: status_var.set(f"Téléchargement réussi: {market_data.rows} points"))
                        self.load_study_data(self.current_study)
                        self.refresh_studies()
                    else:
                        self.root.after(0, lambda: status_var.set("Échec du téléchargement"))
                
                except Exception as e:
                    self.root.after(0, lambda: status_var.set(f"Erreur: {str(e)}"))
                    logger.error(f"Erreur lors du téléchargement: {str(e)}")
            
            threading.Thread(target=download_thread, daemon=True).start()
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.grid(row=7, column=0, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="Télécharger", command=start_download).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Fermer", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def import_csv_data(self):
        """Importe des données depuis un fichier CSV."""
        if not self.current_study:
            messagebox.showwarning("Avertissement", "Veuillez d'abord sélectionner une étude")
            return
        
        file_path = filedialog.askopenfilename(
            title="Importer un fichier CSV",
            filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            data = pd.read_csv(file_path)
            
            # Vérification des colonnes requises
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            # Tentative de correction pour les colonnes majuscules
            if missing_columns:
                if 'Open' in data.columns and 'open' not in data.columns:
                    data['open'] = data['Open']
                if 'High' in data.columns and 'high' not in data.columns:
                    data['high'] = data['High']
                if 'Low' in data.columns and 'low' not in data.columns:
                    data['low'] = data['Low']
                if 'Close' in data.columns and 'close' not in data.columns:
                    data['close'] = data['Close']
                if 'Volume' in data.columns and 'volume' not in data.columns:
                    data['volume'] = data['Volume']
                
                missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                messagebox.showerror("Erreur", f"Colonnes manquantes: {', '.join(missing_columns)}")
                return
            
            # Ajout d'un timestamp si manquant
            if 'timestamp' not in data.columns:
                data['timestamp'] = pd.date_range(start='2023-01-01', periods=len(data), freq='H')
            
            study = self.study_manager.get_study(self.current_study)
            exchange = study.get('exchange', 'binance')
            symbol = study.get('asset', 'BTC/USDT')
            timeframe = study.get('timeframe', '1h')
            
            success = self.data_manager.save_data_for_study(
                study_name=self.current_study,
                data=data,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe
            )
            
            if success:
                messagebox.showinfo("Succès", f"Données importées avec succès: {len(data)} points")
                self.load_study_data(self.current_study)
                self.refresh_studies()
            else:
                messagebox.showerror("Erreur", "Échec de l'importation des données")
        
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'importation: {str(e)}")
            logger.error(f"Erreur lors de l'importation CSV: {str(e)}")
    
    def on_strategy_study_selected(self, event):
        """Gère la sélection d'une étude dans l'onglet Stratégies."""
        study_name = self.strategy_study_var.get()
        if not study_name:
            return
        
        study_path = self.study_manager.get_study_path(study_name)
        if not study_path:
            messagebox.showerror("Erreur", "Chemin de l'étude introuvable")
            return
        
        # Initialiser le gestionnaire de stratégies
        self.strategy_manager = create_strategy_manager_for_study(study_path)
        
        # Rafraîchir la liste des stratégies
        self.refresh_strategies()
    
    def refresh_strategies(self):
        """Rafraîchit la liste des stratégies disponibles."""
        if not self.strategy_manager:
            return
        
        try:
            strategies = self.strategy_manager.list_strategies()
            
            # Mise à jour de la liste
            self.strategies_listbox.delete(0, tk.END)
            
            for strategy in strategies:
                name = strategy.get("name", "Sans nom")
                strategy_id = strategy.get("id", "")
                trial_id = strategy.get("trial_id", None)
                
                if trial_id is not None:
                    self.strategies_listbox.insert(tk.END, f"{name} (Trial {trial_id})")
                else:
                    self.strategies_listbox.insert(tk.END, f"{name}")
            
            # Nettoyer les zones d'information
            self.strategy_details.delete(1.0, tk.END)
            self.strategy_results.delete(1.0, tk.END)
            
            # Nettoyer le graphique
            for widget in self.strategy_chart_frame.winfo_children():
                widget.destroy()
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement des stratégies: {str(e)}")
            logger.error(f"Erreur lors du chargement des stratégies: {str(e)}")
    
    def on_strategy_selected(self, event):
        """Gère la sélection d'une stratégie dans la liste."""
        selection = self.strategies_listbox.curselection()
        if not selection or not self.strategy_manager:
            return
        
        strategy_text = self.strategies_listbox.get(selection[0])
        strategy_id = None
        
        # Déterminer l'ID de la stratégie
        if "Trial" in strategy_text:
            trial_id = strategy_text.split("Trial ")[1].strip().rstrip(")")
            strategy_id = f"trial_{trial_id}"
        else:
            strategies = self.strategy_manager.list_strategies()
            for strategy in strategies:
                if strategy.get("name") == strategy_text:
                    strategy_id = strategy.get("id")
                    break
        
        if not strategy_id:
            messagebox.showerror("Erreur", "Impossible de déterminer l'ID de la stratégie")
            return
        
        try:
            # Charger la stratégie
            constructor = self.strategy_manager.load_strategy(strategy_id)
            if not constructor:
                messagebox.showerror("Erreur", f"Stratégie '{strategy_id}' introuvable")
                return
            
            # Afficher les détails de la stratégie
            self.strategy_details.delete(1.0, tk.END)
            details = f"=== Stratégie {constructor.config.name} ===\n\n"
            details += f"ID: {constructor.config.id}\n"
            details += f"Description: {constructor.config.description}\n"
            details += f"Tags: {', '.join(constructor.config.tags)}\n\n"
            
            # Afficher les indicateurs
            indicators = constructor.config.indicators_manager.list_indicators()
            if indicators:
                details += "Indicateurs:\n"
                for name, config in indicators.items():
                    details += f"- {name} ({config.type.value})\n"
                    indicator_params = config.params.__dict__
                    for param, value in indicator_params.items():
                        if param != 'offset' and not param.startswith('_'):
                            details += f"  • {param}: {value}\n"
            
            # Afficher les blocs d'entrée/sortie
            entry_blocks = constructor.config.blocks_config.entry_blocks
            exit_blocks = constructor.config.blocks_config.exit_blocks
            
            if entry_blocks:
                details += "\nBlocs d'entrée:\n"
                for i, block in enumerate(entry_blocks):
                    details += f"- Bloc {i+1}: {block.name}\n"
                    for j, condition in enumerate(block.conditions):
                        details += f"  • Condition {j+1}: {condition}\n"
            
            if exit_blocks:
                details += "\nBlocs de sortie:\n"
                for i, block in enumerate(exit_blocks):
                    details += f"- Bloc {i+1}: {block.name}\n"
                    for j, condition in enumerate(block.conditions):
                        details += f"  • Condition {j+1}: {condition}\n"
            
            # Afficher la gestion des risques
            risk_config = constructor.config.risk_config
            if risk_config:
                details += f"\nGestion du risque: {risk_config.mode.value}\n"
                risk_params = risk_config.params.__dict__
                for param, value in risk_params.items():
                    if not param.startswith('_'):
                        details += f"- {param}: {value}\n"
            
            self.strategy_details.insert(tk.END, details)
            
            # Afficher les résultats de backtest
            self.strategy_results.delete(1.0, tk.END)
            backtests = self.strategy_manager.list_backtests(strategy_id)
            
            if backtests:
                self.strategy_results.insert(tk.END, f"=== Backtests ({len(backtests)}) ===\n\n")
                
                for i, backtest in enumerate(backtests):
                    backtest_id = backtest.get('id', 'N/A')
                    date = backtest.get('date', 'N/A')
                    self.strategy_results.insert(tk.END, f"Backtest {i+1}: {backtest_id}\n")
                    self.strategy_results.insert(tk.END, f"Date: {date}\n\n")
                    
                    performance = backtest.get("performance", {})
                    self.strategy_results.insert(tk.END, "Performance:\n")
                    
                    for key, value in performance.items():
                        if key in ['roi_pct', 'win_rate_pct', 'max_drawdown_pct']:
                            self.strategy_results.insert(tk.END, f"- {key}: {value:.2f}%\n")
                        elif isinstance(value, (int, float)):
                            self.strategy_results.insert(tk.END, f"- {key}: {value:.4f}\n")
                    
                    self.strategy_results.insert(tk.END, "\n")
                
                # Afficher le graphique du premier backtest
                self.show_strategy_chart(strategy_id, backtests[0]['id'])
            else:
                self.strategy_results.insert(tk.END, "Aucun résultat de backtest disponible")
                
                # Nettoyer le graphique
                for widget in self.strategy_chart_frame.winfo_children():
                    widget.destroy()
        
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement de la stratégie: {str(e)}")
            logger.error(f"Erreur lors du chargement de la stratégie: {str(e)}")
    
    def show_strategy_chart(self, strategy_id, backtest_id):
        """Affiche le graphique des résultats d'un backtest."""
        study_name = self.strategy_study_var.get()
        if not study_name:
            return
        
        study_path = self.study_manager.get_study_path(study_name)
        if not study_path:
            return
        
        # Nettoyer le graphique
        for widget in self.strategy_chart_frame.winfo_children():
            widget.destroy()
        
        # Rechercher le fichier CSV du backtest
        account_csv = None
        backtest_dir = os.path.join(study_path, "strategies", strategy_id, "backtests")
        
        if os.path.exists(backtest_dir):
            for file in os.listdir(backtest_dir):
                if file.startswith(f"{backtest_id}_") and file.endswith("_account.csv"):
                    account_csv = os.path.join(backtest_dir, file)
                    break
        
        if not account_csv:
            label = ttk.Label(self.strategy_chart_frame, text="Données de graphique non disponibles")
            label.pack(expand=True)
            return
        
        try:
            # Charger les données du backtest
            account_data = pd.read_csv(account_csv)
            
            # Créer la figure
            fig, ax = plt.subplots(figsize=(8, 4))
            
            if 'equity' in account_data.columns:
                # Tracer l'equity
                equity = account_data['equity']
                ax.plot(equity, label='Equity', color='blue')
                
                # Ajouter le drawdown si disponible
                if 'drawdown' in account_data.columns:
                    drawdown = account_data['drawdown']
                    ax2 = ax.twinx()
                    ax2.fill_between(range(len(drawdown)), 0, drawdown * 100, 
                                    alpha=0.3, color='red', label='Drawdown %')
                    ax2.set_ylabel('Drawdown %')
                    ax2.legend(loc='upper right')
                
                # Configuration du graphique
                ax.set_title(f"Courbe d'Equity - {strategy_id}")
                ax.set_xlabel("Bougies")
                ax.set_ylabel("Valeur")
                ax.legend(loc='upper left')
                
                # Ajouter les entrées/sorties si disponibles
                if 'long_active' in account_data.columns:
                    long_entries = account_data[account_data['long_active'] > 0].index
                    long_exits = []
                    
                    for i in range(1, len(account_data)):
                        if account_data.iloc[i-1]['long_active'] > 0 and account_data.iloc[i]['long_active'] == 0:
                            long_exits.append(i)
                    
                    if len(long_entries) > 0:
                        ax.scatter(long_entries, account_data.loc[long_entries, 'equity'],
                                marker='^', color='green', s=50, label='Entrée Long')
                    
                    if long_exits:
                        ax.scatter(long_exits, [account_data.iloc[i]['equity'] for i in long_exits],
                                marker='v', color='red', s=50, label='Sortie Long')
            else:
                ax.text(0.5, 0.5, "Données d'equity non disponibles",
                       horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            
            # Afficher le graphique
            canvas = FigureCanvasTkAgg(fig, master=self.strategy_chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du graphique: {str(e)}")
            label = ttk.Label(self.strategy_chart_frame, text=f"Erreur: {str(e)}")
            label.pack(expand=True)
    
    def create_strategy(self):
        """Crée une nouvelle stratégie de trading."""
        if not self.strategy_manager:
            messagebox.showwarning("Avertissement", "Veuillez d'abord sélectionner une étude")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Créer une stratégie")
        dialog.geometry("450x500")
        dialog.grab_set()
        
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Onglet Général
        general_tab = ttk.Frame(notebook)
        notebook.add(general_tab, text="Général")
        
        ttk.Label(general_tab, text="Nom:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        name_var = tk.StringVar(value="Ma Stratégie")
        ttk.Entry(general_tab, textvariable=name_var, width=30).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(general_tab, text="Description:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        desc_var = tk.StringVar(value="Stratégie de test")
        ttk.Entry(general_tab, textvariable=desc_var, width=30).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(general_tab, text="Type:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        type_var = tk.StringVar(value="trend_following")
        ttk.Combobox(general_tab, textvariable=type_var,
                   values=["trend_following", "mean_reversion", "breakout"]).grid(row=2, column=1, padx=5, pady=5)
        
        # Onglet Indicateurs
        indicator_tab = ttk.Frame(notebook)
        notebook.add(indicator_tab, text="Indicateurs")
        
        ttk.Label(indicator_tab, text="EMA rapide:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        fast_var = tk.IntVar(value=12)
        ttk.Spinbox(indicator_tab, from_=2, to=50, textvariable=fast_var, width=5).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(indicator_tab, text="EMA lente:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        slow_var = tk.IntVar(value=26)
        ttk.Spinbox(indicator_tab, from_=5, to=200, textvariable=slow_var, width=5).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(indicator_tab, text="RSI période:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        rsi_period_var = tk.IntVar(value=14)
        ttk.Spinbox(indicator_tab, from_=2, to=50, textvariable=rsi_period_var, width=5).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(indicator_tab, text="RSI surachat:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        rsi_ob_var = tk.DoubleVar(value=70.0)
        ttk.Spinbox(indicator_tab, from_=50.0, to=90.0, increment=1.0, textvariable=rsi_ob_var, width=5).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(indicator_tab, text="RSI survente:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        rsi_os_var = tk.DoubleVar(value=30.0)
        ttk.Spinbox(indicator_tab, from_=10.0, to=50.0, increment=1.0, textvariable=rsi_os_var, width=5).grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Onglet Gestion du risque
        risk_tab = ttk.Frame(notebook)
        notebook.add(risk_tab, text="Gestion du risque")
        
        ttk.Label(risk_tab, text="Mode:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        risk_mode_var = tk.StringVar(value="fixed")
        ttk.Combobox(risk_tab, textvariable=risk_mode_var,
                   values=["fixed", "atr_based", "vol_based"]).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(risk_tab, text="Taille (%):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        size_var = tk.DoubleVar(value=2.0)
        ttk.Spinbox(risk_tab, from_=0.1, to=10.0, increment=0.1, textvariable=size_var, width=5).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(risk_tab, text="Stop loss (%):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        sl_var = tk.DoubleVar(value=2.0)
        ttk.Spinbox(risk_tab, from_=0.1, to=10.0, increment=0.1, textvariable=sl_var, width=5).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(risk_tab, text="Take profit (%):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        tp_var = tk.DoubleVar(value=4.0)
        ttk.Spinbox(risk_tab, from_=0.1, to=20.0, increment=0.1, textvariable=tp_var, width=5).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        def save_strategy():
            try:
                # Créer le constructeur de stratégie
                constructor = self.strategy_manager.create_strategy(
                    name=name_var.get(),
                    description=desc_var.get(),
                    preset=type_var.get()
                )
                
                # Ajouter les indicateurs
                fast_ema_config = IndicatorConfig(IndicatorType.EMA, period=fast_var.get())
                slow_ema_config = IndicatorConfig(IndicatorType.EMA, period=slow_var.get())
                rsi_config = IndicatorConfig(IndicatorType.RSI, period=rsi_period_var.get(),
                                          overbought=rsi_ob_var.get(), oversold=rsi_os_var.get())
                
                constructor.add_indicator("fast_ema", fast_ema_config)
                constructor.add_indicator("slow_ema", slow_ema_config)
                constructor.add_indicator("rsi", rsi_config)
                
                # Créer les opérandes pour les conditions
                fast_ema_operand = IndicatorOperand(
                    indicator_type=IndicatorType.EMA,
                    indicator_name="fast_ema",
                    indicator_params={"period": fast_var.get()}
                )
                
                slow_ema_operand = IndicatorOperand(
                    indicator_type=IndicatorType.EMA,
                    indicator_name="slow_ema",
                    indicator_params={"period": slow_var.get()}
                )
                
                rsi_operand = IndicatorOperand(
                    indicator_type=IndicatorType.RSI,
                    indicator_name="rsi",
                    indicator_params={"period": rsi_period_var.get()}
                )
                
                # Créer les conditions d'entrée
                entry_conditions = []
                
                ema_cross_condition = ConditionConfig(
                    left_operand=fast_ema_operand,
                    operator=OperatorType.CROSS_ABOVE,
                    right_operand=slow_ema_operand
                )
                entry_conditions.append(ema_cross_condition)
                
                rsi_condition = ConditionConfig(
                    left_operand=rsi_operand,
                    operator=OperatorType.GREATER,
                    right_operand=ValueOperand(value=rsi_os_var.get())
                )
                entry_conditions.append(rsi_condition)
                
                # Créer le bloc d'entrée
                entry_block = BlockConfig(
                    conditions=entry_conditions,
                    name="EMA & RSI Entry"
                )
                constructor.add_entry_block(entry_block)
                
                # Créer les conditions de sortie
                exit_conditions = []
                
                ema_cross_exit_condition = ConditionConfig(
                    left_operand=fast_ema_operand,
                    operator=OperatorType.CROSS_BELOW,
                    right_operand=slow_ema_operand
                )
                exit_conditions.append(ema_cross_exit_condition)
                
                rsi_exit_condition = ConditionConfig(
                    left_operand=rsi_operand,
                    operator=OperatorType.GREATER,
                    right_operand=ValueOperand(value=rsi_ob_var.get())
                )
                exit_conditions.append(rsi_exit_condition)
                
                # Créer le bloc de sortie
                exit_block = BlockConfig(
                    conditions=exit_conditions,
                    name="EMA & RSI Exit"
                )
                constructor.add_exit_block(exit_block)
                
                # Configurer la gestion du risque
                risk_config = RiskConfig(
                    mode=RiskModeType(risk_mode_var.get()),
                    position_size=size_var.get() / 100.0,
                    stop_loss=sl_var.get() / 100.0,
                    take_profit=tp_var.get() / 100.0
                )
                constructor.set_risk_config(risk_config)
                
                # Sauvegarder la stratégie
                strategy_id = self.strategy_manager.save_strategy()
                
                messagebox.showinfo("Succès", f"Stratégie créée avec succès (ID: {strategy_id})")
                self.refresh_strategies()
                dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur: {str(e)}")
                logger.error(f"Erreur lors de la création de la stratégie: {str(e)}")
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(btn_frame, text="Créer", command=save_strategy).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Annuler", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def test_strategy(self):
        """Teste la stratégie sélectionnée sur les données actuelles."""
        if not self.strategy_manager:
            messagebox.showwarning("Avertissement", "Veuillez d'abord sélectionner une étude")
            return
        
        selection = self.strategies_listbox.curselection()
        if not selection:
            messagebox.showwarning("Avertissement", "Veuillez sélectionner une stratégie")
            return
        
        strategy_text = self.strategies_listbox.get(selection[0])
        strategy_id = None
        
        # Déterminer l'ID de la stratégie
        if "Trial" in strategy_text:
            trial_id = strategy_text.split("Trial ")[1].strip().rstrip(")")
            strategy_id = f"trial_{trial_id}"
        else:
            strategies = self.strategy_manager.list_strategies()
            for strategy in strategies:
                if strategy.get("name") == strategy_text:
                    strategy_id = strategy.get("id")
                    break
        
        if not strategy_id:
            messagebox.showerror("Erreur", "Impossible de déterminer l'ID de la stratégie")
            return
        
        # Charger la stratégie
        constructor = self.strategy_manager.load_strategy(strategy_id)
        if not constructor:
            messagebox.showerror("Erreur", "Impossible de charger la stratégie")
            return
        
        # Charger les données
        study_name = self.strategy_study_var.get()
        data = self.data_manager.load_study_data(study_name)
        
        if data is None:
            messagebox.showerror("Erreur", "Aucune donnée disponible pour cette étude")
            return
        
        try:
            self.status_var.set(f"Simulation de la stratégie '{constructor.config.name}' en cours...")
            
            def run_simulation():
                try:
                    results = self.strategy_manager.run_simulation(data)
                    
                    if results:
                        backtest_id = self.strategy_manager.save_backtest_results()
                        
                        self.root.after(0, lambda: [
                            self.status_var.set(f"Simulation terminée: {backtest_id}"),
                            self.on_strategy_selected(None),
                            messagebox.showinfo("Succès", f"Simulation terminée: {backtest_id}")
                        ])
                    else:
                        self.root.after(0, lambda: [
                            self.status_var.set("La simulation n'a pas produit de résultats"),
                            messagebox.showerror("Erreur", "La simulation n'a pas produit de résultats")
                        ])
                
                except Exception as e:
                    self.root.after(0, lambda: [
                        self.status_var.set(f"Erreur: {str(e)}"),
                        messagebox.showerror("Erreur", f"Erreur lors de la simulation: {str(e)}")
                    ])
                    logger.error(f"Erreur lors de la simulation: {str(e)}")
            
            threading.Thread(target=run_simulation, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la préparation de la simulation: {str(e)}")
            logger.error(f"Erreur lors de la préparation de la simulation: {str(e)}")
    
    def on_optim_study_selected(self, event):
        """Gère la sélection d'une étude dans l'onglet Optimisation."""
        study_name = self.optim_study_var.get()
        if not study_name:
            return
        
        # Réinitialiser l'interface d'optimisation
        self.trials_listbox.delete(0, tk.END)
        self.trial_details.delete(1.0, tk.END)
        self.optim_logs.delete(1.0, tk.END)
        self.progress_var.set(0)
        self.optim_status_var.set("Prêt")
    
    def start_optimization(self):
        """Lance l'optimisation de la stratégie pour l'étude sélectionnée."""
        study_name = self.optim_study_var.get()
        if not study_name:
            messagebox.showwarning("Avertissement", "Veuillez sélectionner une étude")
            return
        
        study_path = self.study_manager.get_study_path(study_name)
        if not study_path:
            messagebox.showerror("Erreur", "Chemin de l'étude introuvable")
            return
        
        data = self.data_manager.load_study_data(study_name)
        if data is None:
            messagebox.showerror("Erreur", "Aucune donnée disponible pour cette étude")
            return
        
        # Créer le gestionnaire de stratégies si nécessaire
        strategy_manager = create_strategy_manager_for_study(study_path)
        
        # Obtenir l'espace de recherche
        search_space_type = self.search_type_var.get()
        search_space = get_predefined_search_space(search_space_type)
        
        # Créer la configuration d'optimisation
        config = OptimizationConfig(
            n_trials=self.trials_var.get(),
            search_space=search_space,
            optimization_method="tpe",
            scoring_formula="standard",
            min_trades=self.min_trades_var.get(),
            n_jobs=self.jobs_var.get()
        )
        
        # Créer l'optimiseur
        from core.optimization.parallel_optimizer import create_optimizer
        optimizer = create_optimizer(config)
        
        # Réinitialiser l'interface
        self.progress_var.set(0)
        self.optim_status_var.set("Initialisation de l'optimisation...")
        self.trials_listbox.delete(0, tk.END)
        self.trial_details.delete(1.0, tk.END)
        self.optim_logs.delete(1.0, tk.END)
        
        def log_message(message):
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.optim_logs.insert(tk.END, f"[{timestamp}] {message}\n")
            self.optim_logs.see(tk.END)
        
        log_message(f"Démarrage de l'optimisation pour '{study_name}' avec {self.trials_var.get()} essais")
        
        def optimization_thread():
            try:
                success, results = optimizer.run_optimization(study_path, data)
                
                if success:
                    self.root.after(0, lambda: self.optimization_completed(results))
                else:
                    error_message = "Échec de l'optimisation"
                    
                    if hasattr(optimizer, 'optimization_progress') and study_name in optimizer.optimization_progress:
                        error_details = optimizer.optimization_progress[study_name].get('error_message', '')
                        if error_details:
                            error_message += f": {error_details}"
                    
                    self.root.after(0, lambda: [
                        self.optim_status_var.set(error_message),
                        messagebox.showerror("Erreur", error_message),
                        log_message(error_message)
                    ])
            
            except Exception as e:
                self.root.after(0, lambda: [
                    self.optim_status_var.set(f"Erreur: {str(e)}"),
                    messagebox.showerror("Erreur", f"Erreur: {str(e)}"),
                    log_message(f"Erreur: {str(e)}")
                ])
                logger.error(f"Erreur lors de l'optimisation: {str(e)}")
        
        def monitor_progress():
            """Fonction pour suivre en temps réel l'optimisation."""
            if not hasattr(optimizer, 'optimization_progress') or study_name not in optimizer.optimization_progress:
                self.root.after(500, monitor_progress)
                return
            
            progress = optimizer.get_optimization_progress(study_name)
            
            if not progress:
                self.root.after(500, monitor_progress)
                return
            
            status = progress.get('status', '')
            
            if status in ['running', 'created']:
                completed = progress.get('completed', 0)
                total = progress.get('total', 1)
                progress_pct = int(100 * completed / total) if total > 0 else 0
                
                status_text = f"Optimisation en cours: {completed}/{total} essais"
                
                best_value = progress.get('best_value')
                if best_value is not None:
                    status_text += f" - Meilleur score: {best_value:.4f}"
                
                self.progress_var.set(progress_pct)
                self.optim_status_var.set(status_text)
                
                messages = progress.get('messages', [])
                if messages:
                    latest_messages = messages[-5:]
                    for msg in latest_messages:
                        log_message(str(msg))
                
                self.update_trials_list(progress.get('trial_results', {}))
                
                self.root.after(500, monitor_progress)
            else:
                self.progress_var.set(100)
                self.optim_status_var.set(f"Optimisation terminée: {status}")
        
        threading.Thread(target=optimization_thread, daemon=True).start()
        self.root.after(100, monitor_progress)
    
    def stop_optimization(self):
        """Arrête l'optimisation en cours."""
        study_name = self.optim_study_var.get()
        if not study_name:
            messagebox.showinfo("Information", "Aucune étude sélectionnée")
            return
        
        if messagebox.askyesno("Confirmation", "Voulez-vous vraiment arrêter l'optimisation en cours?"):
            self.optim_status_var.set("Arrêt de l'optimisation en cours...")
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.optim_logs.insert(tk.END, f"[{timestamp}] Demande d'arrêt de l'optimisation envoyée\n")
            self.optim_logs.see(tk.END)
            
            # Note: L'arrêt effectif dépend de l'implémentation de l'optimiseur
    
    def update_trials_list(self, trial_results):
        """Met à jour la liste des essais avec les résultats."""
        self.trials_listbox.delete(0, tk.END)
        
        sorted_trials = []
        for trial_id, result in trial_results.items():
            try:
                score = result.get('score', 0)
                strategy_id = result.get('strategy_id', '')
                roi = result.get('metrics', {}).get('roi', 0) * 100
                sorted_trials.append((trial_id, score, roi, strategy_id))
            except:
                pass
        
        # Trier par score décroissant
        sorted_trials.sort(key=lambda x: x[1], reverse=True)
        
        for trial_id, score, roi, strategy_id in sorted_trials:
            self.trials_listbox.insert(tk.END, f"Trial {trial_id}: {score:.3f} (ROI: {roi:.1f}%)")
    
    def on_trial_selected(self, event):
        """Affiche les détails d'un essai sélectionné."""
        selection = self.trials_listbox.curselection()
        if not selection:
            return
        
        trial_text = self.trials_listbox.get(selection[0])
        trial_id = int(trial_text.split(':')[0].replace('Trial ', '').strip())
        
        # Charger les informations sur l'essai
        # Note: Cette implémentation est simplifiée
        self.trial_details.delete(1.0, tk.END)
        self.trial_details.insert(tk.END, f"Chargement des détails de l'essai {trial_id}...\n")
        
        # Dans une implémentation complète, il faudrait charger les détails réels de l'essai
        # depuis l'optimiseur ou la base de données
    
    def optimization_completed(self, results):
        """Traite la fin de l'optimisation."""
        self.progress_var.set(100)
        self.optim_status_var.set("Optimisation terminée")
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.optim_logs.insert(tk.END, f"[{timestamp}] Optimisation terminée!\n")
        
        best_score = results.get('best_score', 0)
        best_trial_id = results.get('best_trial_id', 'none')
        best_strategies = results.get('best_trials', [])
        
        summary = (
            f"Optimisation terminée avec succès!\n\n"
            f"Meilleur score: {best_score:.4f}\n"
            f"ID de l'essai: {best_trial_id}\n"
            f"Stratégies sauvegardées: {len(best_strategies)}\n\n"
            f"Les meilleures stratégies sont disponibles dans l'onglet 'Stratégies'."
        )
        
        self.trial_details.delete(1.0, tk.END)
        self.trial_details.insert(tk.END, summary)
        
        messagebox.showinfo("Succès", f"Optimisation terminée. {len(best_strategies)} stratégies sauvegardées.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleTradingApp(root)
    root.mainloop()