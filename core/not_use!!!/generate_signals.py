"""
Générateur de signaux de trading modulaire et réutilisable.

Ce module permet de créer et d'utiliser facilement des stratégies de trading
à partir de blocs de conditions configurables. Il offre une interface propre
pour générer des signaux et paramètres de risque utilisables dans n'importe
quel environnement de trading.

Author: Trading System Developer
Version: 2.0
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union
import logging
import traceback

# Import des modules du système de trading
from indicators import (
    IndicatorType, Operator, LogicOperator, Condition, Block, SignalGenerator as BaseSignalGenerator
)
from risk import PositionCalculator, RiskMode

class SignalGenerator:
    """
    Générateur de signaux pour stratégies de trading algorithmiques.
    
    Cette classe encapsule toute la logique nécessaire pour charger une
    configuration de stratégie, générer des signaux de trading et calculer
    les paramètres de gestion du risque. Elle utilise les composants de base
    du système de trading (SignalGenerator, PositionCalculator) pour créer
    une interface unifiée et facile à utiliser.
    """
    
    def __init__(self, config: Dict[str, Any], verbose: bool = False):
        """
        Initialise le générateur de signaux.
        
        Args:
            config: Dictionnaire de configuration de la stratégie
            verbose: Activer le mode verbeux pour plus d'informations
        """
        self.config = config
        self.strategy = BaseSignalGenerator()
        self.verbose = verbose
        self.logger = self._setup_logger()
        
        # Configuration des blocs de trading
        n_buy_blocks = self._setup_buy_blocks()
        n_sell_blocks = self._setup_sell_blocks()
        
        # Création du calculateur de position
        self.position_calculator = self._setup_position_calculator()
        
        if verbose:
            print(f"✅ Stratégie initialisée avec {n_buy_blocks} blocs d'achat et {n_sell_blocks} blocs de vente")
    
    def _setup_logger(self) -> logging.Logger:
        """
        Configure un logger pour le générateur de signaux.
        
        Returns:
            Logger configuré
        """
        logger = logging.getLogger("SignalGenerator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # Niveau INFO si verbose, sinon WARNING
            logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        return logger
    
    def _setup_position_calculator(self) -> PositionCalculator:
        """
        Configure le calculateur de position basé sur les paramètres de risque.
        
        Returns:
            PositionCalculator configuré
        """
        # Déterminer le mode de risque
        risk_mode_str = self.config.get('risk_mode', 'fixed')
        risk_mode = RiskMode.FIXED
        
        if risk_mode_str == 'atr_based' or risk_mode_str == 'dynamic_atr':
            risk_mode = RiskMode.ATR_BASED
        elif risk_mode_str == 'volatility_based' or risk_mode_str == 'dynamic_vol':
            risk_mode = RiskMode.VOLATILITY_BASED
        
        # Construire la configuration
        risk_config = {
            'base_position': self.config.get('base_position', 0.1),
            'base_sl': self.config.get('base_sl', 0.02),
            'tp_multiplier': self.config.get('tp_mult', 2.0)
        }
        
        # Ajouter des paramètres spécifiques selon le mode
        if risk_mode == RiskMode.ATR_BASED:
            risk_config.update({
                'atr_period': self.config.get('atr_period', 14),
                'atr_multiplier': self.config.get('atr_multiplier', 1.5)
            })
        elif risk_mode == RiskMode.VOLATILITY_BASED:
            risk_config.update({
                'vol_period': self.config.get('vol_period', 20),
                'vol_multiplier': self.config.get('vol_multiplier', 1.0)
            })
        
        return PositionCalculator(mode=risk_mode, config=risk_config)
    
    def _setup_buy_blocks(self) -> int:
        """
        Configure les blocs d'achat et retourne leur nombre.
        
        Returns:
            Nombre de blocs d'achat configurés
        """
        n_buy_blocks = self.config.get('n_buy_blocks', 0)
        
        for b_idx in range(n_buy_blocks):
            block = self._create_block('buy', b_idx)
            if block and len(block.conditions) > 0:
                self.strategy.add_block(block, is_buy=True)
        
        return len(self.strategy.buy_blocks)
    
    def _setup_sell_blocks(self) -> int:
        """
        Configure les blocs de vente et retourne leur nombre.
        
        Returns:
            Nombre de blocs de vente configurés
        """
        n_sell_blocks = self.config.get('n_sell_blocks', 0)
        
        for b_idx in range(n_sell_blocks):
            block = self._create_block('sell', b_idx)
            if block and len(block.conditions) > 0:
                self.strategy.add_block(block, is_buy=False)
        
        return len(self.strategy.sell_blocks)
    
    def _create_block(self, block_type: str, block_idx: int) -> Optional[Block]:
        """
        Crée un bloc de conditions de trading à partir de la configuration.
        
        Args:
            block_type: Type de bloc ('buy' ou 'sell')
            block_idx: Index du bloc
            
        Returns:
            Bloc créé ou None si échec
        """
        # Nombre de conditions dans le bloc
        n_conditions = self.config.get(f'{block_type}_block_{block_idx}_conditions', 0)
        
        if n_conditions <= 0:
            return None
        
        conditions = []
        logic_operators = []
        
        for c_idx in range(n_conditions):
            try:
                # Préfixe de base pour les paramètres de condition
                prefix = f"{block_type}_b{block_idx}_c{c_idx}"
                
                # Récupération du premier indicateur
                ind1_type = self.config.get(f'{prefix}_ind1_type')
                if not ind1_type:
                    self.logger.warning(f"Type d'indicateur manquant pour {prefix}")
                    continue
                
                ind1_period = self.config.get(f'{prefix}_ind1_period', 14)
                indicator1 = f"{ind1_type}_{ind1_period}"
                
                # Récupération de l'opérateur
                op_str = self.config.get(f'{prefix}_operator', '>' if block_type == 'buy' else '<')
                operator = Operator(op_str)
                
                # Vérification si c'est une comparaison avec valeur ou indicateur
                if f'{prefix}_value' in self.config:
                    # Comparaison avec valeur
                    value = float(self.config[f'{prefix}_value'])
                    condition = Condition(
                        indicator1=indicator1,
                        operator=operator,
                        value=value
                    )
                elif f'{prefix}_ind2_type' in self.config:
                    # Comparaison entre indicateurs
                    ind2_type = self.config[f'{prefix}_ind2_type']
                    ind2_period = self.config.get(f'{prefix}_ind2_period', 50)
                    indicator2 = f"{ind2_type}_{ind2_period}"
                    
                    condition = Condition(
                        indicator1=indicator1,
                        operator=operator,
                        indicator2=indicator2
                    )
                else:
                    self.logger.warning(f"Ni valeur ni second indicateur trouvé pour {prefix}")
                    continue
                
                conditions.append(condition)
                
                # Ajout de l'opérateur logique si nécessaire
                logic_key = f'{prefix}_logic'
                if c_idx < n_conditions - 1 and logic_key in self.config:
                    logic_operators.append(LogicOperator(self.config[logic_key]))
                
            except Exception as e:
                self.logger.warning(f"Erreur lors de la création de la condition: {str(e)}")
                traceback.print_exc()
        
        # Création du bloc si nous avons des conditions
        if conditions:
            # S'assurer que nous avons le bon nombre d'opérateurs logiques
            while len(logic_operators) < len(conditions) - 1:
                logic_operators.append(LogicOperator.AND)
            
            return Block(conditions=conditions, logic_operators=logic_operators)
        
        return None
    
    def generate_signals_and_parameters(
        self, 
        prices: np.ndarray, 
        high: Optional[np.ndarray] = None, 
        low: Optional[np.ndarray] = None,
        volumes: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Génère les signaux de trading et les paramètres de risque associés.
        
        Args:
            prices: Array des prix de clôture
            high: Array des prix hauts (optionnel)
            low: Array des prix bas (optionnel)
            volumes: Array des volumes (optionnel)
            
        Returns:
            Tuple (signals, position_sizes, sl_levels, tp_levels)
            - signals: 1 pour achat, -1 pour vente, 0 pour neutre
            - position_sizes: Taille de position en % du capital
            - sl_levels: Niveaux de stop loss en % du prix
            - tp_levels: Niveaux de take profit en % du prix
        """
        if self.verbose:
            print(f"🔄 Génération de signaux pour {len(prices)} points de données...")
            
        # Vérification des données
        if prices is None or len(prices) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        try:
            # Conversion en arrays numpy
            prices_np = np.asarray(prices, dtype=np.float64)
            high_np = np.asarray(high, dtype=np.float64) if high is not None else None
            low_np = np.asarray(low, dtype=np.float64) if low is not None else None
            volumes_np = np.asarray(volumes, dtype=np.float64) if volumes is not None else None
            
            # Génération des signaux
            signals = self.strategy.generate_signals(
                prices=prices_np, 
                high=high_np, 
                low=low_np, 
                volumes=volumes_np
            )
            
            # Calcul des paramètres de risque
            position_sizes, sl_levels, tp_levels = self.position_calculator.calculate_risk_parameters(
                prices=prices_np, 
                high=high_np, 
                low=low_np
            )
            
            # Affichage des statistiques si en mode verbeux
            if self.verbose:
                buy_count = np.sum(signals == 1)
                sell_count = np.sum(signals == -1)
                neutral_count = np.sum(signals == 0)
                
                print("\n📊 Statistiques des signaux:")
                print(f"  • Total: {len(signals)} points")
                print(f"  • Achat: {buy_count} ({buy_count/len(signals)*100:.2f}%)")
                print(f"  • Vente: {sell_count} ({sell_count/len(signals)*100:.2f}%)")
                print(f"  • Neutres: {neutral_count} ({neutral_count/len(signals)*100:.2f}%)")
            
            return signals, position_sizes, sl_levels, tp_levels
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des signaux: {str(e)}")
            traceback.print_exc()
            return np.array([0]), np.array([0]), np.array([0]), np.array([0])
    
    def convert_to_actions(
        self, 
        prices: np.ndarray, 
        high: Optional[np.ndarray] = None, 
        low: Optional[np.ndarray] = None,
        volumes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convertit les signaux en format d'actions pour environnement de trading.
        
        Args:
            prices: Array des prix de clôture
            high: Array des prix hauts (optionnel)
            low: Array des prix bas (optionnel)
            volumes: Array des volumes (optionnel)
            
        Returns:
            Array d'actions [direction, taille, sl, tp, levier]
        """
        # Afficher la stratégie pour information
        if self.verbose:
            self.display_strategy()
            
        # Vérification des données
        if prices is None or len(prices) == 0:
            return np.array([[0, 0, 0, 0, 1.0]])
        
        try:
            # Génération des signaux et paramètres
            signals, pos_sizes, sl_levels, tp_levels = self.generate_signals_and_parameters(
                prices, high, low, volumes
            )
            
            # Configuration du levier
            leverage = float(self.config.get('leverage', 1.0))
            
            # Création du tableau d'actions
            n_candles = len(signals)
            actions = np.zeros((n_candles, 5), dtype=np.float32)
            
            # Remplissage du tableau d'actions
            for i in range(n_candles):
                if signals[i] != 0:
                    actions[i] = [
                        float(signals[i]),        # Direction
                        float(pos_sizes[i]),      # Taille de position
                        float(sl_levels[i]),      # Stop loss
                        float(tp_levels[i]),      # Take profit
                        float(leverage)           # Levier
                    ]
                else:
                    actions[i] = [0.0, 0.0, 0.0, 0.0, leverage]
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la conversion en actions: {str(e)}")
            traceback.print_exc()
            return np.array([[0, 0, 0, 0, 1.0]])
    
    def display_strategy(self) -> None:
        """Affiche un résumé propre de la stratégie configurée."""
        print("\n")
        print("┌─────────────────────────────────────┐")
        print("│         RÉSUMÉ DE LA STRATÉGIE      │")
        print("└─────────────────────────────────────┘")
        
        # Blocs d'achat
        if self.strategy.buy_blocks:
            print(f"\n🟢 SIGNAUX D'ACHAT ({len(self.strategy.buy_blocks)} blocs):")
            for i, block in enumerate(self.strategy.buy_blocks):
                print(f"  • Bloc {i+1}: {block}")
        else:
            print("\n🟢 SIGNAUX D'ACHAT: Aucun configuré")
        
        # Blocs de vente
        if self.strategy.sell_blocks:
            print(f"\n🔴 SIGNAUX DE VENTE ({len(self.strategy.sell_blocks)} blocs):")
            for i, block in enumerate(self.strategy.sell_blocks):
                print(f"  • Bloc {i+1}: {block}")
        else:
            print("\n🔴 SIGNAUX DE VENTE: Aucun configuré")
        
        # Gestion du risque
        print("\n⚠️ GESTION DU RISQUE:")
        print(f"  • Mode: {self.config.get('risk_mode', 'fixed')}")
        print(f"  • Taille de position: {self.config.get('base_position', 0.1)*100:.1f}%")
        print(f"  • Stop loss: {self.config.get('base_sl', 0.02)*100:.2f}%")
        print(f"  • Take profit: {self.config.get('base_sl', 0.02)*self.config.get('tp_mult', 2.0)*100:.2f}%")
        print(f"  • Levier: {self.config.get('leverage', 1.0):.1f}x")
        
        # Paramètres spécifiques au mode de risque
        if self.config.get('risk_mode') == 'dynamic_atr' or self.config.get('risk_mode') == 'atr_based':
            print(f"  • Période ATR: {self.config.get('atr_period', 14)}")
            print(f"  • Multiplicateur ATR: {self.config.get('atr_multiplier', 1.5):.2f}")
        elif self.config.get('risk_mode') == 'dynamic_vol' or self.config.get('risk_mode') == 'volatility_based':
            print(f"  • Période volatilité: {self.config.get('vol_period', 20)}")
            print(f"  • Multiplicateur volatilité: {self.config.get('vol_multiplier', 1.0):.2f}")
        
        # Paramètres de position
        if 'margin_mode' in self.config:
            margin_mode = "Croisée" if self.config['margin_mode'] == 1 else "Isolée"
            print(f"  • Mode de marge: {margin_mode}")
        
        if 'trading_mode' in self.config:
            trading_mode = "Hedge" if self.config['trading_mode'] == 1 else "One-way"
            print(f"  • Mode de trading: {trading_mode}")
        
        print("\n")
    
    def save_to_json(self, filepath: str) -> bool:
        """
        Sauvegarde la configuration de la stratégie au format JSON.
        
        Args:
            filepath: Chemin du fichier de sortie
            
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        import json
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            
            if self.verbose:
                print(f"✅ Configuration sauvegardée dans {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de la configuration: {str(e)}")
            return False

    @classmethod
    def load_from_json(cls, filepath: str, verbose: bool = False) -> 'SignalGenerator':
        """
        Charge une configuration de stratégie depuis un fichier JSON.
        
        Args:
            filepath: Chemin du fichier de configuration
            verbose: Activer le mode verbeux
            
        Returns:
            Instance de SignalGenerator configurée
        """
        import json
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if verbose:
                print(f"✅ Configuration chargée depuis {filepath}")
            
            return cls(config, verbose)
        except Exception as e:
            logging.error(f"Erreur lors du chargement de la configuration: {str(e)}")
            raise ValueError(f"Impossible de charger la configuration depuis {filepath}: {str(e)}")


# Fonction utilitaire pour tester le générateur de signaux
def test_signal_generator(config: Dict[str, Any], data_path: str) -> None:
    """
    Teste le générateur de signaux sur des données historiques.
    
    Args:
        config: Configuration de la stratégie
        data_path: Chemin vers le fichier de données CSV
    """
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Chargement des données
        print(f"📊 Chargement des données depuis {data_path}...")
        df = pd.read_csv(data_path)
        
        # Vérification des colonnes requises
        required_columns = ['close']
        if not all(col in df.columns for col in required_columns):
            print(f"❌ Colonnes requises manquantes. Le CSV doit contenir: {required_columns}")
            return
        
        # Colonnes optionnelles
        high = df['high'].values if 'high' in df.columns else None
        low = df['low'].values if 'low' in df.columns else None
        volumes = df['volume'].values if 'volume' in df.columns else None
        
        # Création du générateur de signaux
        print("🔧 Initialisation du générateur de signaux...")
        generator = SignalGenerator(config, verbose=True)
        
        # Génération des signaux
        prices = df['close'].values
        print("⚙️ Génération des signaux...")
        signals, position_sizes, sl_levels, tp_levels = generator.generate_signals_and_parameters(
            prices, high, low, volumes
        )
        
        # Création d'un DataFrame des résultats
        result_df = pd.DataFrame({
            'timestamp': df.index,
            'close': prices,
            'signal': signals,
            'position_size': position_sizes,
            'sl_level': sl_levels,
            'tp_level': tp_levels
        })
        
        # Analyse des signaux
        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()
        
        print("✅ Test terminé!")
        print(f"📈 Signaux d'achat: {buy_signals}")
        print(f"📉 Signaux de vente: {sell_signals}")
        
        # Visualisation simple des signaux
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, prices, label='Prix')
        plt.scatter(df.index[signals == 1], prices[signals == 1], color='green', label='Achat', marker='^')
        plt.scatter(df.index[signals == -1], prices[signals == -1], color='red', label='Vente', marker='v')
        plt.legend()
        plt.title('Signaux de Trading')
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        traceback.print_exc()

# Exemple d'utilisation
if __name__ == "__main__":
    print("🚀 Test du générateur de signaux")
    
    # Exemple de configuration
    example_config = {
        # Configuration des blocs d'achat
        'n_buy_blocks': 1,
        'buy_block_0_conditions': 2,
        'buy_b0_c0_ind1_type': 'EMA',
        'buy_b0_c0_ind1_period': 10,
        'buy_b0_c0_operator': '>',
        'buy_b0_c0_ind2_type': 'EMA',
        'buy_b0_c0_ind2_period': 20,
        'buy_b0_c0_logic': 'and',
        'buy_b0_c1_ind1_type': 'RSI',
        'buy_b0_c1_ind1_period': 14,
        'buy_b0_c1_operator': '>',
        'buy_b0_c1_value': 50,
        
        # Configuration des blocs de vente
        'n_sell_blocks': 1,
        'sell_block_0_conditions': 1,
        'sell_b0_c0_ind1_type': 'RSI',
        'sell_b0_c0_ind1_period': 14,
        'sell_b0_c0_operator': '<',
        'sell_b0_c0_value': 30,
        
        # Configuration de gestion du risque
        'risk_mode': 'atr_based',
        'base_position': 0.1,
        'base_sl': 0.02,
        'tp_mult': 2.0,
        'atr_period': 14,
        'atr_multiplier': 1.5,
        'leverage': 1.0
    }
    
    # Test avec un fichier CSV
    # test_signal_generator(example_config, "data/BTC_USDT_1h.csv")