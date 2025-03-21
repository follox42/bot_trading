"""
Interface d'optimisation améliorée pour les stratégies de trading.
Offre une meilleure visibilité et traçabilité des essais et leurs espaces de recherche.
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import json
import logging
from datetime import datetime
from core.study.study_manager import StudyManager
from data.data_manager import get_data_manager
from core.strategy.strategy_manager import create_strategy_manager_for_study
from core.optimization.parallel_optimizer import create_optimizer, OptimizationConfig
from core.optimization.search_config import SearchSpace, get_predefined_search_space

logger = logging.getLogger(__name__)

class OptimizationUI(ttk.Frame):
    """Interface utilisateur améliorée pour l'optimisation des stratégies"""
    def __init__(self, parent, study_manager=None):
        """
        Initialise l'interface d'optimisation.
        Args:
            parent: Widget parent
            study_manager: Gestionnaire d'études
        """
        super().__init__(parent)
        self.parent = parent
        self.study_manager = study_manager or StudyManager()
        self.data_manager = get_data_manager()
        self.strategy_manager = None
        self.current_study = None
        self.current_optimizer = None
        self.create_ui()

    def create_ui(self):
        """Crée l'interface utilisateur"""
        left_frame = ttk.LabelFrame(self, text="Configuration de l'optimisation")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)

        ttk.Label(left_frame, text="Étude:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.study_var = tk.StringVar()
        self.study_combo = ttk.Combobox(left_frame, textvariable=self.study_var, state="readonly", width=25)
        self.study_combo.grid(row=0, column=1, padx=5, pady=5)
        self.study_combo.bind("<<ComboboxSelected>>", self.on_study_selected)

        search_frame = ttk.LabelFrame(left_frame, text="Espace de recherche")
        search_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)

        ttk.Label(search_frame, text="Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.search_type_var = tk.StringVar(value="default")
        self.search_type_combo = ttk.Combobox(search_frame, textvariable=self.search_type_var, width=20,
                                             values=["default", "trend_following", "mean_reversion", "custom"])
        self.search_type_combo.grid(row=0, column=1, padx=5, pady=5)
        self.search_type_combo.bind("<<ComboboxSelected>>", self.on_search_type_selected)

        ttk.Button(search_frame, text="Configurer", command=self.configure_search_space).grid(row=0, column=2, padx=5, pady=5)

        trials_frame = ttk.LabelFrame(left_frame, text="Configuration des essais")
        trials_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)

        ttk.Label(trials_frame, text="Nombre d'essais:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.trials_var = tk.IntVar(value=20)
        ttk.Spinbox(trials_frame, from_=5, to=1000, textvariable=self.trials_var, width=8).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(trials_frame, text="Processus parallèles:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.jobs_var = tk.IntVar(value=2)
        ttk.Spinbox(trials_frame, from_=1, to=8, textvariable=self.jobs_var, width=5).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(trials_frame, text="Formule de scoring:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.formula_var = tk.StringVar(value="standard")
        ttk.Combobox(trials_frame, textvariable=self.formula_var,
                    values=["standard", "performance", "conservative", "comprehensive"]).grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(trials_frame, text="Trades minimum:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.min_trades_var = tk.IntVar(value=10)
        ttk.Spinbox(trials_frame, from_=1, to=100, textvariable=self.min_trades_var, width=5).grid(row=3, column=1, padx=5, pady=5)

        control_frame = ttk.Frame(left_frame)
        control_frame.grid(row=3, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=10)
        ttk.Button(control_frame, text="Lancer l'optimisation", command=self.start_optimization).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Arrêter", command=self.stop_optimization).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Rafraîchir", command=self.refresh_studies).pack(side=tk.LEFT, padx=5)

        center_frame = ttk.LabelFrame(self, text="Résultats de l'optimisation")
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        progress_frame = ttk.Frame(center_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(progress_frame, text="Progression:").pack(side=tk.LEFT, padx=5)
        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=300,
                                           mode='determinate', variable=self.progress_var)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.status_var = tk.StringVar(value="Prêt")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.pack(side=tk.RIGHT, padx=5)

        self.results_notebook = ttk.Notebook(center_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.trials_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.trials_tab, text="Essais")

        # Correction ici : créer le frame avec une largeur fixe
        trials_list_frame = ttk.Frame(self.trials_tab)
        trials_list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        # Empêcher la propagation pour maintenir la largeur
        trials_list_frame.pack_propagate(False)
        # Définir la largeur après le pack
        trials_list_frame.config(width=200)

        ttk.Label(trials_list_frame, text="Liste des essais:").pack(fill=tk.X, padx=5, pady=2)
        self.trials_listbox = tk.Listbox(trials_list_frame, width=30)
        self.trials_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.trials_listbox.bind('<<ListboxSelect>>', self.on_trial_selected)

        trial_detail_frame = ttk.Frame(self.trials_tab)
        trial_detail_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.trial_details = scrolledtext.ScrolledText(trial_detail_frame, wrap=tk.WORD)
        self.trial_details.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.strategies_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.strategies_tab, text="Stratégies")

        # Même correction pour strategies_list_frame
        strategies_list_frame = ttk.Frame(self.strategies_tab)
        strategies_list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        strategies_list_frame.pack_propagate(False)
        strategies_list_frame.config(width=200)

        ttk.Label(strategies_list_frame, text="Stratégies optimisées:").pack(fill=tk.X, padx=5, pady=2)
        self.strategies_listbox = tk.Listbox(strategies_list_frame, width=30)
        self.strategies_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.strategies_listbox.bind('<<ListboxSelect>>', self.on_strategy_selected)

        strategy_detail_frame = ttk.Frame(self.strategies_tab)
        strategy_detail_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.strategy_details = scrolledtext.ScrolledText(strategy_detail_frame, wrap=tk.WORD)
        self.strategy_details.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.logs_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.logs_tab, text="Logs")
        self.logs_text = scrolledtext.ScrolledText(self.logs_tab, wrap=tk.WORD)
        self.logs_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        right_frame = ttk.LabelFrame(self, text="Visualisation")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.plot_frame = ttk.Frame(right_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.refresh_studies()

    def refresh_studies(self):
        """Rafraîchit la liste des études"""
        studies = self.study_manager.list_studies()
        self.study_combo['values'] = [study['name'] for study in studies]
        if studies and self.study_var.get() not in [study['name'] for study in studies]:
            self.study_var.set(studies[0]['name'])
        self.on_study_selected(None)

    def on_study_selected(self, event):
        """Met à jour l'interface quand une étude est sélectionnée"""
        study_name = self.study_var.get()
        if not study_name:
            return

        self.current_study = study_name
        study_config = self.study_manager.get_study(study_name)
        if study_config:
            if hasattr(study_config, 'search_space_config') and study_config.search_space_config:
                try:
                    search_space = study_config.search_space_config
                    if isinstance(search_space, dict) and "name" in search_space:
                        self.search_type_var.set(search_space["name"])
                    else:
                        self.search_type_var.set("custom")
                except:
                    self.search_type_var.set("default")
            else:
                self.search_type_var.set("default")

        study_path = self.study_manager.get_study_path(study_name)
        if study_path:
            self.strategy_manager = create_strategy_manager_for_study(study_path)
            self.refresh_optimization_results()
            self.refresh_strategies()

    def on_search_type_selected(self, event):
        """Met à jour quand le type d'espace de recherche est sélectionné"""
        search_type = self.search_type_var.get()
        if search_type == "custom":
            self.configure_search_space()

    def configure_search_space(self):
        """Configure l'espace de recherche pour l'étude sélectionnée"""
        study_name = self.study_var.get()
        if not study_name:
            messagebox.showwarning("Avertissement", "Veuillez sélectionner une étude")
            return

        study_path = self.study_manager.get_study_path(study_name)
        if not study_path:
            messagebox.showerror("Erreur", "Chemin de l'étude introuvable")
            return

        # Utilisation du gestionnaire de configurations d'optimisation
        from core.optimization.optimization_range_manager import create_optimization_range_manager
        range_manager = create_optimization_range_manager(study_path)
        
        # Récupération de la configuration actuelle
        current_search_space = range_manager.get_default_config()
        
        dialog = tk.Toplevel(self.parent)
        dialog.title(f"Configuration de l'espace de recherche - {study_name}")
        dialog.geometry("650x550")
        dialog.grab_set()

        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        general_tab = ttk.Frame(notebook)
        notebook.add(general_tab, text="Général")

        ttk.Label(general_tab, text="Nom:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        name_var = tk.StringVar(value=current_search_space.name)
        ttk.Entry(general_tab, textvariable=name_var, width=30).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(general_tab, text="Description:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        desc_var = tk.StringVar(value=current_search_space.description)
        ttk.Entry(general_tab, textvariable=desc_var, width=30).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(general_tab, text="Modèle:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        template_var = tk.StringVar(value=self.search_type_var.get())
        template_combo = ttk.Combobox(general_tab, textvariable=template_var,
                                     values=["default", "trend_following", "mean_reversion"])
        template_combo.grid(row=2, column=1, padx=5, pady=5)

        # Liste des configurations existantes
        ttk.Label(general_tab, text="Configurations sauvegardées:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        saved_configs = range_manager.list_configs()
        config_names = [f"{cfg['name']} ({cfg['id']})" for cfg in saved_configs]
        config_var = tk.StringVar()
        config_combo = ttk.Combobox(general_tab, textvariable=config_var, values=config_names, width=30)
        config_combo.grid(row=3, column=1, padx=5, pady=5)
        
        def load_saved_config():
            selected = config_var.get()
            if not selected:
                return
                
            config_id = selected.split("(")[-1].strip(")")
            loaded_space = range_manager.load_config(config_id)
            if loaded_space:
                name_var.set(loaded_space.name)
                desc_var.set(loaded_space.description)
                space_editor.delete(1.0, tk.END)
                space_editor.insert(tk.END, json.dumps(loaded_space.to_dict(), indent=2))
                messagebox.showinfo("Succès", f"Configuration '{loaded_space.name}' chargée")
        
        ttk.Button(general_tab, text="Charger cette configuration", command=load_saved_config).grid(row=3, column=2, padx=5, pady=5)

        def load_template():
            template = template_var.get()
            try:
                new_space = get_predefined_search_space(template)
                new_space.name = name_var.get()
                new_space.description = desc_var.get()
                space_editor.delete(1.0, tk.END)
                space_editor.insert(tk.END, json.dumps(new_space.to_dict(), indent=2))
                messagebox.showinfo("Succès", f"Modèle '{template}' chargé")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur: {str(e)}")

        ttk.Button(general_tab, text="Charger ce modèle", command=load_template).grid(row=2, column=2, padx=5, pady=5)

        editor_tab = ttk.Frame(notebook)
        notebook.add(editor_tab, text="Éditeur JSON")
        space_editor = scrolledtext.ScrolledText(editor_tab)
        space_editor.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        try:
            space_editor.insert(tk.END, json.dumps(current_search_space.to_dict(), indent=2))
        except Exception as e:
            space_editor.insert(tk.END, f"Erreur lors du formatage: {str(e)}")

        def save_search_space():
            try:
                space_text = space_editor.get(1.0, tk.END)
                new_space_dict = json.loads(space_text)
                
                # Mettre à jour les métadonnées
                new_space_dict["name"] = name_var.get()
                new_space_dict["description"] = desc_var.get()
                
                # Créer l'objet SearchSpace
                new_space = SearchSpace.from_dict(new_space_dict)
                
                # Mettre à jour le type d'espace de recherche dans l'interface
                self.search_type_var.set(new_space.name)
                
                # Sauvegarder avec notre gestionnaire de configurations
                config_id = range_manager.save_config(new_space)
                
                # Mettre à jour l'étude dans le gestionnaire d'études
                result = self.study_manager.update_study_search_space(study_name, new_space)
                
                if result:
                    messagebox.showinfo("Succès", f"Espace de recherche '{new_space.name}' sauvegardé avec succès (ID: {config_id})")
                    dialog.destroy()
                else:
                    messagebox.showerror("Erreur", "Échec de la sauvegarde de l'espace de recherche dans l'étude")
                    
            except json.JSONDecodeError as e:
                messagebox.showerror("Erreur JSON", f"Format JSON invalide: {str(e)}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur: {str(e)}")

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="Sauvegarder", command=save_search_space).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Annuler", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def start_optimization(self):
        """Lance l'optimisation pour l'étude sélectionnée"""
        study_name = self.study_var.get()
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

        # Demander la configuration à utiliser
        from core.optimization.optimization_range_manager import create_optimization_range_manager
        range_manager = create_optimization_range_manager(study_path)
        saved_configs = range_manager.list_configs()
        
        if saved_configs:
            dialog = tk.Toplevel(self.parent)
            dialog.title("Sélection de la configuration d'optimisation")
            dialog.geometry("500x300")
            dialog.grab_set()
            
            ttk.Label(dialog, text="Choisissez une configuration d'optimisation:").pack(padx=10, pady=10)
            
            config_frame = ttk.Frame(dialog)
            config_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # En-têtes
            ttk.Label(config_frame, text="Nom", width=20).grid(row=0, column=0, sticky=tk.W)
            ttk.Label(config_frame, text="Description", width=30).grid(row=0, column=1, sticky=tk.W)
            ttk.Label(config_frame, text="Date", width=20).grid(row=0, column=2, sticky=tk.W)
            
            # Liste des configurations
            config_listbox = tk.Listbox(config_frame, height=10, width=50)
            config_listbox.grid(row=1, column=0, columnspan=3, sticky=tk.EW)
            
            # Mapping pour retrouver l'ID à partir de l'index
            config_map = {}
            
            for i, config in enumerate(saved_configs):
                display_text = f"{config['name']} - {config['description'][:30]}"
                config_listbox.insert(tk.END, display_text)
                config_map[i] = config['id']
            
            # Option utiliser la config par défaut
            use_default_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(dialog, text="Utiliser la configuration par défaut", variable=use_default_var).pack(padx=10, pady=5)
            
            config_id_var = tk.StringVar()
            
            def on_select(evt):
                selection = config_listbox.curselection()
                if selection:
                    index = selection[0]
                    config_id_var.set(config_map[index])
                    use_default_var.set(False)
            
            config_listbox.bind('<<ListboxSelect>>', on_select)
            
            def start_with_config():
                config_id = None if use_default_var.get() else config_id_var.get()
                dialog.destroy()
                self._run_optimization(study_path, data)
            
            def cancel():
                dialog.destroy()
            
            button_frame = ttk.Frame(dialog)
            button_frame.pack(fill=tk.X, padx=10, pady=10)
            
            ttk.Button(button_frame, text="Lancer l'optimisation", command=start_with_config).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Annuler", command=cancel).pack(side=tk.LEFT, padx=5)
        else:
            # Aucune configuration sauvegardée, on utilise la configuration par défaut
            self._run_optimization(study_path, data)

    def _run_optimization(self, study_path, data, config_id=None):
        """Exécute l'optimisation avec la configuration spécifiée"""
        study_name = os.path.basename(study_path)
        search_space = None
        
        if config_id:
            from core.optimization.optimization_range_manager import create_optimization_range_manager
            range_manager = create_optimization_range_manager(study_path)
            search_space = range_manager.load_config(config_id)
            self.log(f"Utilisation de la configuration '{search_space.name}' (ID: {config_id})")
        
        search_space_type = self.search_type_var.get()
        if not search_space:
            from core.optimization.search_config import get_predefined_search_space
            search_space = get_predefined_search_space(search_space_type)
            self.log(f"Utilisation de l'espace de recherche prédéfini: {search_space_type}")

        config = OptimizationConfig(
            n_trials=self.trials_var.get(),
            search_space=search_space,
            optimization_method="tpe",
            scoring_formula=self.formula_var.get(),
            min_trades=self.min_trades_var.get(),
            n_jobs=self.jobs_var.get()
        )

        self.current_optimizer = create_optimizer(config)
        self.progress_var.set(0)
        self.status_var.set("Initialisation de l'optimisation...")
        self.trials_listbox.delete(0, tk.END)
        self.trial_details.delete(1.0, tk.END)
        self.log(f"Démarrage de l'optimisation pour '{study_name}' avec {self.trials_var.get()} essais")

        def optimization_thread():
            try:
                success, results = self.current_optimizer.run_optimization(study_path, data, config_id)
                if success:
                    self.parent.after(0, lambda: self.optimization_completed(results))
                else:
                    error_message = "Échec de l'optimisation"
                    if study_name in self.current_optimizer.optimization_progress:
                        error_details = self.current_optimizer.optimization_progress[study_name].get('error_message', '')
                        if error_details:
                            error_message += f": {error_details}"
                    self.parent.after(0, lambda: [
                        self.status_var.set(error_message),
                        messagebox.showerror("Erreur", error_message)
                    ])
            except Exception as e:
                self.parent.after(0, lambda: [
                    self.status_var.set(f"Erreur: {str(e)}"),
                    messagebox.showerror("Erreur", f"Erreur: {str(e)}")
                ])

        def monitor_progress():
            """Fonction de suivi en temps réel de l'optimisation"""
            if not self.current_optimizer or not study_name in self.current_optimizer.optimization_progress:
                self.parent.after(500, monitor_progress)
                return

            progress = self.current_optimizer.get_optimization_progress(study_name)
            if not progress:
                self.parent.after(500, monitor_progress)
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
                self.status_var.set(status_text)

                messages = progress.get('messages', [])
                if messages:
                    latest_messages = messages[-5:]
                    for msg in latest_messages:
                        self.log(msg)

                self.update_trials_list(progress.get('trial_results', {}))
                self.parent.after(500, monitor_progress)
            else:
                self.progress_var.set(100)
                self.status_var.set(f"Optimisation terminée: {status}")

        threading.Thread(target=optimization_thread, daemon=True).start()
        self.parent.after(100, monitor_progress)

    def stop_optimization(self):
        """Arrête l'optimisation en cours"""
        if not self.current_optimizer:
            messagebox.showinfo("Information", "Aucune optimisation en cours")
            return

        study_name = self.study_var.get()
        if not study_name:
            return

        if messagebox.askyesno("Confirmation", "Voulez-vous vraiment arrêter l'optimisation en cours?"):
            self.current_optimizer.stop_optimization(study_name)
            self.status_var.set("Arrêt de l'optimisation en cours...")
            self.log("Demande d'arrêt de l'optimisation envoyée")

    def update_trials_list(self, trial_results):
        """Met à jour la liste des essais avec les résultats"""
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
        
        sorted_trials.sort(key=lambda x: x[1], reverse=True)
        for trial_id, score, roi, strategy_id in sorted_trials:
            self.trials_listbox.insert(tk.END, f"Trial {trial_id}: {score:.3f} (ROI: {roi:.1f}%)")

    def refresh_optimization_results(self):
        """Récupère et affiche les résultats d'optimisation précédents"""
        study_name = self.study_var.get()
        if not study_name:
            return

        try:
            optimization_results = self.study_manager.get_optimization_results(study_name)
            if not optimization_results:
                self.log(f"Aucun résultat d'optimisation trouvé pour '{study_name}'")
                return

            latest_result = optimization_results[-1]
            trial_results = latest_result.get('trial_results', {})
            self.update_trials_list(trial_results)
        except Exception as e:
            self.log(f"Erreur lors du chargement des résultats: {str(e)}")

    def on_trial_selected(self, event):
        """Affiche les détails d'un essai sélectionné"""
        selection = self.trials_listbox.curselection()
        if not selection:
            return

        trial_text = self.trials_listbox.get(selection[0])
        trial_id = int(trial_text.split(':')[0].replace('Trial ', '').strip())
        study_name = self.study_var.get()
        trial_info = None

        if self.current_optimizer:
            trial_info = self.current_optimizer.get_trial_info(study_name, trial_id)

        if not trial_info:
            trial_info = self.study_manager.get_trial_info(study_name, trial_id)

        if not trial_info:
            self.trial_details.delete(1.0, tk.END)
            self.trial_details.insert(tk.END, f"Aucune information disponible pour l'essai {trial_id}")
            return

        details = f"=== Essai {trial_id} ===\n\n"
        metrics = trial_info.get('metrics', {})
        details += "Performance:\n"
        details += f"- Score: {trial_info.get('score', 0):.4f}\n"
        details += f"- ROI: {metrics.get('roi', 0) * 100:.2f}%\n"
        details += f"- Win Rate: {metrics.get('win_rate', 0) * 100:.2f}%\n"
        details += f"- Trades: {metrics.get('total_trades', 0)}\n"
        details += f"- Drawdown Max: {metrics.get('max_drawdown', 0) * 100:.2f}%\n"
        details += f"- Profit Factor: {metrics.get('profit_factor', 0):.2f}\n\n"

        strategy_id = trial_info.get('strategy_id', '')
        details += f"ID de la stratégie: {strategy_id}\n"
        backtest_id = trial_info.get('backtest_id', '')
        if backtest_id:
            details += f"ID du backtest: {backtest_id}\n\n"

        params = trial_info.get('params', {})
        if params:
            details += "Paramètres importants:\n"
            important_params = {}
            for key, value in params.items():
                if any(key.startswith(prefix) for prefix in ['ema_', 'rsi_', 'macd_', 'atr_', 'risk_']):
                    category = key.split('_')[0]
                    if category not in important_params:
                        important_params[category] = []
                    important_params[category].append((key, value))

            for category, param_list in important_params.items():
                details += f"\n{category.upper()}:\n"
                for key, value in param_list:
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    details += f"- {key}: {formatted_value}\n"

        self.trial_details.delete(1.0, tk.END)
        self.trial_details.insert(tk.END, details)
        self.plot_trial_results(trial_id, strategy_id)

    def plot_trial_results(self, trial_id, strategy_id):
        """Affiche un graphique des résultats de l'essai"""
        study_name = self.study_var.get()
        study_path = self.study_manager.get_study_path(study_name)
        if not study_path or not strategy_id:
            return

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        backtest_dir = os.path.join(study_path, "strategies", strategy_id, "backtests")
        account_csv = None
        
        if os.path.exists(backtest_dir):
            for file in os.listdir(backtest_dir):
                if file.endswith("_account.csv"):
                    account_csv = os.path.join(backtest_dir, file)
                    break

        if not account_csv:
            label = ttk.Label(self.plot_frame, text="Données de graphique non disponibles")
            label.pack(expand=True)
            return

        try:
            account_data = pd.read_csv(account_csv)
            fig, ax = plt.subplots(figsize=(10, 6))

            if 'equity' in account_data.columns:
                equity = account_data['equity']
                ax.plot(equity, label='Equity', color='blue')

                if 'drawdown' in account_data.columns:
                    drawdown = account_data['drawdown']
                    ax2 = ax.twinx()
                    ax2.fill_between(range(len(drawdown)), 0, drawdown * 100, alpha=0.3, color='red', label='Drawdown %')
                    ax2.set_ylabel('Drawdown %')
                    ax2.legend(loc='upper right')

                ax.set_title(f"Equity Curve - Trial {trial_id}")
                ax.set_xlabel("Candles")
                ax.set_ylabel("Value")
                ax.legend(loc='upper left')

                if 'long_active' in account_data.columns:
                    long_entries = account_data[account_data['long_active'] > 0].index
                    long_exits = []
                    for i in range(1, len(account_data)):
                        if account_data.iloc[i-1]['long_active'] > 0 and account_data.iloc[i]['long_active'] == 0:
                            long_exits.append(i)

                    ax.scatter(long_entries, account_data.loc[long_entries, 'equity'], 
                              marker='^', color='green', s=50, label='Long Entry')
                    ax.scatter(long_exits, account_data.loc[long_exits, 'equity'], 
                              marker='v', color='red', s=50, label='Long Exit')
            else:
                ax.text(0.5, 0.5, "Données d'equity non disponibles", 
                       horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            self.log(f"Erreur lors de la création du graphique: {str(e)}")
            label = ttk.Label(self.plot_frame, text=f"Erreur: {str(e)}")
            label.pack(expand=True)

    def refresh_strategies(self):
        """Récupère et affiche les stratégies existantes"""
        if not self.strategy_manager:
            return

        try:
            strategies = self.strategy_manager.list_strategies()
            self.strategies_listbox.delete(0, tk.END)
            for strategy in strategies:
                name = strategy.get("name", "Sans nom")
                strategy_id = strategy.get("id", "")
                trial_id = strategy.get("trial_id", None)
                if trial_id is not None:
                    self.strategies_listbox.insert(tk.END, f"{name} (Trial {trial_id})")
                else:
                    self.strategies_listbox.insert(tk.END, f"{name} (ID: {strategy_id})")
        except Exception as e:
            self.log(f"Erreur lors du chargement des stratégies: {str(e)}")

    def on_strategy_selected(self, event):
        """Affiche les détails d'une stratégie sélectionnée"""
        selection = self.strategies_listbox.curselection()
        if not selection or not self.strategy_manager:
            return

        strategy_text = self.strategies_listbox.get(selection[0])
        strategy_id = None

        if "Trial" in strategy_text:
            trial_id = strategy_text.split("Trial ")[1].strip().rstrip(")")
            strategy_id = f"trial_{trial_id}"
        elif "ID:" in strategy_text:
            strategy_id = strategy_text.split("ID: ")[1].rstrip(")")

        if not strategy_id:
            return

        try:
            constructor = self.strategy_manager.load_strategy(strategy_id)
            if not constructor:
                self.strategy_details.delete(1.0, tk.END)
                self.strategy_details.insert(tk.END, f"Stratégie '{strategy_id}' introuvable")
                return

            details = f"=== Stratégie {constructor.config.name} ===\n\n"
            details += f"Description: {constructor.config.description}\n"
            details += f"ID: {constructor.config.id}\n"
            details += f"Tags: {', '.join(constructor.config.tags)}\n\n"

            indicators = constructor.config.indicators_manager.list_indicators()
            if indicators:
                details += "Indicateurs:\n"
                for name, config in indicators.items():
                    details += f"- {name} ({config.type.value})\n"
                    indicator_params = config.params.__dict__
                    for param, value in indicator_params.items():
                        if param != 'offset' and not param.startswith('_'):
                            details += f" • {param}: {value}\n"

            entry_blocks = constructor.config.blocks_config.entry_blocks
            exit_blocks = constructor.config.blocks_config.exit_blocks
            
            if entry_blocks:
                details += "\nBlocs d'entrée:\n"
                for i, block in enumerate(entry_blocks):
                    details += f"- Bloc {i+1}: {block.name}\n"
                    for j, condition in enumerate(block.conditions):
                        details += f" • Condition {j+1}: {condition}\n"

            if exit_blocks:
                details += "\nBlocs de sortie:\n"
                for i, block in enumerate(exit_blocks):
                    details += f"- Bloc {i+1}: {block.name}\n"
                    for j, condition in enumerate(block.conditions):
                        details += f" • Condition {j+1}: {condition}\n"

            risk_config = constructor.config.risk_config
            if risk_config:
                details += f"\nGestion du risque: {risk_config.mode.value}\n"
                risk_params = risk_config.params.__dict__
                for param, value in risk_params.items():
                    if not param.startswith('_'):
                        details += f"- {param}: {value}\n"

            self.strategy_details.delete(1.0, tk.END)
            self.strategy_details.insert(tk.END, details)

            backtests = self.strategy_manager.list_backtests(strategy_id)
            if backtests:
                latest_backtest = backtests[0]
                self.strategy_details.insert(tk.END, f"\nDernier backtest: {latest_backtest['id']}\n")
                performance = latest_backtest.get("performance", {})
                self.strategy_details.insert(tk.END, "\nPerformance:\n")
                for key, value in performance.items():
                    if key in ['roi_pct', 'win_rate_pct', 'max_drawdown_pct']:
                        self.strategy_details.insert(tk.END, f"- {key}: {value:.2f}%\n")
                    elif isinstance(value, (int, float)):
                        self.strategy_details.insert(tk.END, f"- {key}: {value:.4f}\n")

        except Exception as e:
            self.log(f"Erreur lors du chargement de la stratégie: {str(e)}")
            self.strategy_details.delete(1.0, tk.END)
            self.strategy_details.insert(tk.END, f"Erreur: {str(e)}")

    def log(self, message):
        """Ajoute un message au log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.logs_text.see(tk.END)
        
    def optimization_completed(self, results):
        """Traitement à la fin de l'optimisation"""
        self.log("Optimisation terminée!")
        self.progress_var.set(100)
        self.status_var.set("Optimisation terminée")

        best_score = results.get('best_score', 0)
        best_trial_id = results.get('best_trial_id', 'none')
        best_strategies = results.get('best_trials', [])
        
        summary = (
            f"Optimisation terminée avec succès!\n\n"
            f"Meilleur score: {best_score:.4f}\n"
            f"ID de l'essai: {best_trial_id}\n"
            f"Stratégies sauvegardées: {len(best_strategies)}\n\n"
            f"Les résultats sont disponibles dans les onglets 'Essais' et 'Stratégies'."
        )
        
        self.trial_details.delete(1.0, tk.END)
        self.trial_details.insert(tk.END, summary)
        
        self.refresh_optimization_results()
        self.refresh_strategies()
        
        messagebox.showinfo("Succès", f"Optimisation terminée. {len(best_strategies)} stratégies sauvegardées.")