"""
Module de simulation de trading optimisé pour le backtesting de stratégies.
Ce module fournit des méthodes hautes performances pour simuler des transactions
avec différentes conditions de marché et configurations de risque.
"""

import numpy as np
from numba import njit, prange, float64, int64, boolean
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import time
import pandas as pd
from dataclasses import dataclass
import os

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('simulator.log', mode='a'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('simulator')

# ===== Structures de données optimisées pour Numba =====
@dataclass
class SimulationConfig:
    """Configuration des paramètres de simulation"""
    initial_balance: float = 10000.0
    fee_open: float = 0.001  # 0.1% par trade
    fee_close: float = 0.001
    slippage: float = 0.001
    tick_size: float = 0.01
    min_trade_size: float = 0.001
    max_trade_size: float = 100000.0
    leverage: int = 1
    margin_mode: int = 0  # 0=Isolated, 1=Cross
    trading_mode: int = 0  # 0=One-way, 1=Hedge

@njit(cache=True)
def create_position_state() -> np.ndarray:
    """
    Structure de données pour une position:
    [0] direction (1 long, -1 short, 0 inactive)
    [1] size_contracts (taille en contrats)
    [2] size_usd (valeur en USD)
    [3] entry_price (prix moyen d'entrée)
    [4] leverage (levier utilisé)
    [5] margin (marge)
    [6] unrealized_pnl (PnL non réalisé)
    [7] liquidation_price (prix de liquidation)
    [8] margin_ratio (ratio de marge actuel)
    [9] tp_price (prix du Take Profit)
    [10] sl_price (prix du Stop Loss)
    [11] entry_time (index temporel d'entrée)
    """
    return np.zeros(12, dtype=np.float64)

@njit(cache=True)
def create_account_state() -> np.ndarray:
    """
    Structure du compte:
    [0] wallet_balance: Balance réelle du compte
    [1] unrealized_pnl: PnL non réalisé total
    [2] equity: wallet_balance + unrealized_pnl
    [3] margin_ratio: Ratio de marge
    """
    return np.zeros(4, dtype=np.float64)

@njit(cache=True)
def create_history_arrays(n_candles: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crée les arrays pour stocker l'historique complet
    Returns:
        account_history: Métriques du compte [n_candles, 4]
            [0] wallet_balance
            [1] equity
            [2] unrealized_pnl
            [3] drawdown

        long_history: Données position longue [n_candles, 9]
            [0] active (0/1)
            [1] contracts
            [2] entry_price
            [3] leverage
            [4] margin
            [5] unrealized_pnl
            [6] liquidation_price
            [7] tp_price
            [8] sl_price

        short_history: Données position courte [n_candles, 9]
            [structure identique à long_history]
    """
    account_history = np.zeros((n_candles, 4), dtype=np.float64)
    long_history = np.zeros((n_candles, 9), dtype=np.float64)
    short_history = np.zeros((n_candles, 9), dtype=np.float64)
    
    return account_history, long_history, short_history

@njit(cache=True)
def create_trade_stats() -> np.ndarray:
    """
    Crée un tableau pour stocker les statistiques de trading.
    [0] total_trades
    [1] winning_trades
    [2] liquidated_trades
    [3] total_fees  (nouveau)
    [4] max_profit_trade
    [5] max_loss_trade
    [6] consecutive_losses (nouveau)
    [7] max_consecutive_losses (nouveau)
    """
    return np.zeros(8, dtype=np.float64)

# ===== Fonctions de mise à jour et de calculs =====
@njit(cache=True)
def update_account_state(
    account: np.ndarray,
    long_pos: np.ndarray,
    short_pos: np.ndarray,
    margin_mode: int64
) -> None:
    """
    Met à jour l'état du compte 
    """
    # Calcul PnL total
    total_unrealized = long_pos[6] + short_pos[6]
    margin = long_pos[5] + short_pos[5]
    account[1] = total_unrealized
    
    # Mise à jour equity
    account[2] = account[0] + total_unrealized + margin
    
    # Calcul margin ratio
    total_margin = long_pos[5] + short_pos[5]
    if total_margin > 0:
        account[3] = account[2] / total_margin
    else:
        account[3] = 0.0

@njit(cache=True)
def get_tier_info(position_value: float64) -> Tuple[float64, float64]:
    """
    Retourne les informations de niveau de position sur un exchange.
    Args:
        position_value (float64): La valeur de la position en USDT.
    Returns:
        Tuple[float64, float64]: (levier_max, taux_marge_maintenance)
    """
    # Tableau simplifié des niveaux de position (similaire à Binance/Bitget)
    position_tiers = np.array([
        [50_000, 125.0, 0.004],
        [200_000, 100.0, 0.005],
        [1_000_000, 50.0, 0.01],
        [5_000_000, 20.0, 0.025],
        [20_000_000, 10.0, 0.05],
        [50_000_000, 5.0, 0.10],
        [100_000_000, 4.0, 0.125],
        [200_000_000, 3.0, 0.15],
        [300_000_000, 2.0, 0.25],
        [500_000_000, 1.0, 0.50]
    ])
    
    for i in range(len(position_tiers)):
        if position_value < position_tiers[i, 0]:
            return position_tiers[i, 1], position_tiers[i, 2]
    
    # Si plus grand que le dernier niveau
    return 1.0, 0.50

@njit(cache=True)
def round_to_tick(value: float64, tick_size: float64) -> float64:
    """
    Arrondit une valeur au tick le plus proche
    """
    return np.round(value / tick_size) * tick_size

@njit(cache=True)
def update_position_metrics(
    position: np.ndarray,
    current_price: float64,
    slippage: float64 = 0.0
) -> None:
    """
    Met à jour les métriques d'une position (PnL, valeur, etc.)
    """
    if position[0] == 0:  # Position inactive
        return
        
    direction = position[0]  # 1 pour long, -1 pour short
    contracts = position[1]
    entry_price = position[3]
    
    # Prix effectif avec slippage
    effective_price = current_price * (1.0 - direction * slippage)
    
    # Mise à jour de la valeur de la position en USD
    position[2] = contracts * effective_price
    
    # Calcul du PnL non réalisé
    if direction > 0:  # Long
        position[6] = (effective_price - entry_price) * contracts
    else:  # Short
        position[6] = (entry_price - effective_price) * contracts
    
    # Mise à jour du ratio de marge (si nécessaire)
    if position[5] > 0:  # Si marge > 0
        _, mmr = get_tier_info(position[2])
        position[8] = position[5] / (position[2] * mmr)

@njit(cache=True)
def calculate_liquidation_price(
    position: np.ndarray,
    account: np.ndarray,
    long_pos: np.ndarray,
    short_pos: np.ndarray,
    current_price: float64,
    margin_mode: int64,
    trading_mode: int64,
    taker_fee: float64,
    safety_buffer: float64 = 0.0  # Buffer de sécurité en pourcentage
) -> float64:
    """
    Calcule le prix de liquidation d'une position
    
    Args:
        position: Position à calculer
        account: État du compte
        long_pos: Position longue
        short_pos: Position courte
        current_price: Prix actuel
        margin_mode: Mode de marge (0=Isolé, 1=Cross)
        trading_mode: Mode de trading (0=Unidirectionnel, 1=Hedge)
        taker_fee: Frais de liquidation
        safety_buffer: Marge de sécurité
        
    Returns:
        float64: Prix de liquidation
    """
    # Si position inactive
    if position[0] == 0:
        return 0.0
        
    direction = position[0]  # 1 pour long, -1 pour short
    size = position[1]       # Taille en contrats
    entry_price = position[3]  # Prix d'entrée
    position_value = position[2]  # Valeur en USD
    margin = position[5]     # Marge
    leverage = position[4]   # Levier
    
    # Paramètres exchange
    _, mmr = get_tier_info(position_value)  # Taux de maintenance
    
    # Formule simplifiée pour le mode Isolated
    if margin_mode == 0:  # Marge isolée
        # Pour long: prix = entry_price * (1 - 1/leverage + mmr)
        # Pour short: prix = entry_price * (1 + 1/leverage - mmr)
        if direction > 0:  # Long
            liq_price = entry_price * (1.0 - 1.0/leverage + mmr + taker_fee)
        else:  # Short
            liq_price = entry_price * (1.0 + 1.0/leverage - mmr - taker_fee)
    else:  # Marge croisée
        # En mode croisé, utiliser l'equity totale
        equity = account[2]
        
        if trading_mode == 0:  # Unidirectionnel
            # Position unique - calcul simplifié
            if direction > 0:  # Long
                liq_price = entry_price - (equity - margin * mmr) / size
            else:  # Short
                liq_price = entry_price + (equity - margin * mmr) / size
        else:  # Hedge mode
            # Positions long et short possibles
            if direction > 0:  # Long
                # Calcul pour position longue
                liq_price = entry_price - (equity - long_pos[5] * mmr - short_pos[5] * mmr) / size
            else:  # Short
                # Calcul pour position courte
                liq_price = entry_price + (equity - long_pos[5] * mmr - short_pos[5] * mmr) / size
    
    # Ajout d'un buffer de sécurité
    if direction > 0:  # Long
        liq_price = liq_price * (1.0 + safety_buffer)
    else:  # Short
        liq_price = liq_price * (1.0 - safety_buffer)
    
    # Vérification des valeurs aberrantes
    if liq_price <= 0:
        liq_price = entry_price * 0.1  # 90% de perte pour long
    
    # Protection contre les prix de liquidation trop proches
    min_distance = 0.003  # 0.3% minimum
    if direction > 0 and liq_price > entry_price * (1.0 - min_distance):
        liq_price = entry_price * (1.0 - min_distance)
    elif direction < 0 and liq_price < entry_price * (1.0 + min_distance):
        liq_price = entry_price * (1.0 + min_distance)
    
    return liq_price

@njit(cache=True)
def process_trade_signal(
    signal: int64,
    current_time: int64,
    current_price: float64,
    size_pct: float64,
    leverage: float64,
    sl_pct: float64,
    tp_pct: float64,
    account: np.ndarray,
    long_pos: np.ndarray,
    short_pos: np.ndarray,
    trading_mode: int64,
    margin_mode: int64,
    fees: float64,
    tick_size: float64,
    min_size: float64,
    max_size: float64,
    safety_buffer: float64,
    stats: np.ndarray
) -> Tuple[boolean, float64]:
    """
    Traite un signal de trading pour ouvrir une position.
    
    Args:
        signal: Direction du signal (1=long, -1=short, 0=neutre)
        current_time: Index temporel actuel
        current_price: Prix actuel
        size_pct: Taille de position en % du capital
        leverage: Multiplicateur de levier
        sl_pct: Niveau du stop loss en %
        tp_pct: Niveau du take profit en %
        account: État du compte
        long_pos: Position longue
        short_pos: Position courte
        trading_mode: Mode de trading (0=Unidirectionnel, 1=Hedge)
        margin_mode: Mode de marge (0=Isolé, 1=Cross)
        fees: Frais d'ouverture
        tick_size: Taille minimale de tick
        min_size: Taille minimum
        max_size: Taille maximum
        safety_buffer: Marge de sécurité pour le prix de liquidation
        stats: Statistiques de trading
    
    Returns:
        Tuple[boolean, float64]: (succès, prix_liquidation)
    """
    if signal == 0:
        return False, 0.0

    # Mode One-Way: Fermer position opposée si nécessaire
    if trading_mode == 0:
        if signal == 1 and short_pos[0] != 0:
            close_position(short_pos, account, current_price, margin_mode, fees, 0.0, stats)
        elif signal == -1 and long_pos[0] != 0:
            close_position(long_pos, account, current_price, margin_mode, fees, 0.0, stats)
    
    # Target position selon signal
    target_pos = long_pos if signal == 1 else short_pos
    
    # Si déjà une position dans la même direction, ne pas en ouvrir une nouvelle
    if target_pos[0] == signal:
        return False, 0.0
    
    # Calcul des paramètres de position
    available_balance = account[0]  # Utiliser wallet balance
    
    # Calculer la valeur de la position
    position_value = available_balance * size_pct * leverage
    
    # Vérification des limites de taille
    # MODIFIÉ: Utilisez min/max au lieu de np.clip pour les scalaires
    position_value = min(max_size, max(min_size, position_value))
    
    # Obtention limites de levier exchange
    max_lev, mmr = get_tier_info(position_value)
    actual_leverage = min(leverage, max_lev)
    
    # Recalcul de la valeur avec le levier ajusté
    position_value = available_balance * size_pct * actual_leverage
    position_value = min(max_size, max(min_size, position_value))
    
    # Calcul de la taille en contrats avec tick size
    size_btc = round_to_tick(position_value / current_price, tick_size)
    
    # Recalcul de la valeur exacte après arrondi
    position_value = size_btc * current_price
    
    # Calcul de la marge requise
    margin_required = position_value / actual_leverage
    
    # Calcul des frais
    open_fee = position_value * fees
    
    # Vérification de la taille minimum
    if size_btc < tick_size or position_value < min_size:
        return False, 0.0
    
    # Vérification des fonds disponibles
    if margin_required + open_fee > available_balance * 0.99:  # 99% max de la balance
        return False, 0.0
    
    # Déduction des frais
    account[0] -= open_fee
    stats[3] += open_fee  # Frais totaux

    # Création de la position
    target_pos[0] = signal  # direction
    target_pos[1] = size_btc  # size_contracts
    target_pos[2] = position_value  # size_usd
    target_pos[3] = current_price  # entry_price
    target_pos[4] = actual_leverage  # leverage
    target_pos[5] = margin_required  # margin
    target_pos[11] = current_time  # entry_time
    
    # Définition des niveaux de TP/SL
    if signal > 0:  # Long
        target_pos[10] = current_price * (1.0 - sl_pct)  # SL price
        target_pos[9] = current_price * (1.0 + tp_pct)  # TP price
    else:  # Short
        target_pos[10] = current_price * (1.0 + sl_pct)  # SL price
        target_pos[9] = current_price * (1.0 - tp_pct)  # TP price
    
    # Déduction de la marge du wallet
    account[0] -= margin_required
    
    # Calcul du prix de liquidation
    liq_price = calculate_liquidation_price(
        target_pos, account, long_pos, short_pos,
        current_price, margin_mode, trading_mode, fees, safety_buffer
    )
    target_pos[7] = liq_price
    
    # Mise à jour des métriques
    update_position_metrics(target_pos, current_price)
    update_account_state(account, long_pos, short_pos, margin_mode)
    
    return True, liq_price

@njit(cache=True)
def close_position(
    position: np.ndarray,
    account: np.ndarray,
    current_price: float64,
    margin_mode: int64,
    fees: float64,
    slippage: float64 = 0.001,
    stats: np.ndarray = None
) -> Tuple[float64, boolean]:
    """
    Ferme une position et retourne le PnL réalisé
    
    Args:
        position: Position à fermer
        account: État du compte
        current_price: Prix actuel
        margin_mode: Mode de marge (0=Isolé, 1=Cross)
        fees: Frais de clôture
        slippage: Slippage à la clôture
        stats: Statistiques de trading à mettre à jour
        
    Returns:
        Tuple[float64, boolean]: (pnl, is_win)
    """
    if position[0] == 0:  # Position inactive
        return 0.0, False
    
    direction = position[0]
    contracts = position[1]
    entry_price = position[3]
    margin = position[5]
    
    # Prix de sortie avec slippage
    exit_price = current_price * (1.0 - direction * slippage)
    
    # Calcul du PnL
    raw_pnl = (exit_price - entry_price) * contracts * direction
    
    # Calcul des frais de sortie
    close_fee = exit_price * contracts * fees
    
    # PnL final après frais
    final_pnl = raw_pnl - close_fee
    
    # Mise à jour du compte
    account[0] += margin + final_pnl
    
    # Mise à jour des statistiques
    is_win = final_pnl > 0
    if stats is not None:
        stats[0] += 1  # Nombre total de trades
        stats[1] += 1 if is_win else 0  # Trades gagnants
        stats[3] += close_fee  # Frais totaux
        
        # Mise à jour des statistiques de P&L
        if is_win and final_pnl > stats[4]:
            stats[4] = final_pnl  # Mise à jour du trade le plus profitable
        elif not is_win and final_pnl < stats[5]:
            stats[5] = final_pnl  # Mise à jour de la perte maximale
        
        # Suivi des pertes consécutives
        if not is_win:
            stats[6] += 1  # Incrémentation des pertes consécutives
            if stats[6] > stats[7]:
                stats[7] = stats[6]  # Mise à jour max pertes consécutives
        else:
            stats[6] = 0  # Réinitialisation des pertes consécutives
    
    # Réinitialisation de la position
    position.fill(0.0)
    
    return final_pnl, is_win

@njit(cache=True)
def check_and_handle_tp_sl(
    position: np.ndarray,
    account: np.ndarray,
    current_price: float64,
    margin_mode: int64,
    fee_close: float64,
    slippage: float64,
    stats: np.ndarray
) -> Tuple[boolean, float64]:
    """
    Vérifie si le prix atteint TP ou SL et ferme la position si nécessaire
    
    Args:
        position: Position à vérifier
        account: État du compte
        current_price: Prix actuel
        margin_mode: Mode de marge
        fee_close: Frais de clôture
        slippage: Slippage de clôture
        stats: Statistiques de trading
        
    Returns:
        Tuple[boolean, float64]: (position_fermée, pnl)
    """
    if position[0] == 0:  # Position inactive
        return False, 0.0
    
    direction = position[0]
    tp_price = position[9]
    sl_price = position[10]
    
    # Si TP/SL ne sont pas définis
    if tp_price == 0.0 and sl_price == 0.0:
        return False, 0.0
    
    hit_tp = False
    hit_sl = False
    
    if direction > 0:  # Long
        if tp_price > 0 and current_price >= tp_price:
            hit_tp = True
        if sl_price > 0 and current_price <= sl_price:
            hit_sl = True
    else:  # Short
        if tp_price > 0 and current_price <= tp_price:
            hit_tp = True
        if sl_price > 0 and current_price >= sl_price:
            hit_sl = True
    
    if hit_tp or hit_sl:
        pnl, is_win = close_position(
            position, account, current_price, 
            margin_mode, fee_close, slippage, stats
        )
        return True, pnl
    
    return False, 0.0

@njit(cache=True)
def update_history(
    account_history: np.ndarray,
    long_history: np.ndarray,
    short_history: np.ndarray,
    index: int64,
    account: np.ndarray,
    long_pos: np.ndarray,
    short_pos: np.ndarray,
    current_drawdown: float64
) -> None:
    """
    Met à jour les arrays d'historique avec l'état actuel
    
    Args:
        account_history: Array pour l'historique du compte
        long_history: Array pour l'historique de la position longue
        short_history: Array pour l'historique de la position courte
        index: Index temporel actuel
        account: État du compte
        long_pos: Position longue
        short_pos: Position courte
        current_drawdown: Drawdown actuel
    """
    # Métriques du compte
    account_history[index, 0] = account[0]  # wallet_balance
    account_history[index, 1] = account[2]  # equity
    account_history[index, 2] = account[1]  # unrealized_pnl
    account_history[index, 3] = current_drawdown  # drawdown
    
    # Position longue
    if index < len(long_history):
        long_history[index, 0] = 1.0 if long_pos[0] != 0 else 0.0  # active
        long_history[index, 1] = long_pos[1]  # contracts
        long_history[index, 2] = long_pos[3]  # entry_price
        long_history[index, 3] = long_pos[4]  # leverage
        long_history[index, 4] = long_pos[5]  # margin
        long_history[index, 5] = long_pos[6]  # unrealized_pnl
        long_history[index, 6] = long_pos[7]  # liquidation_price
        long_history[index, 7] = long_pos[9]  # tp_price
        long_history[index, 8] = long_pos[10]  # sl_price
    
    # Position courte
    if index < len(short_history):
        short_history[index, 0] = 1.0 if short_pos[0] != 0 else 0.0  # active
        short_history[index, 1] = short_pos[1]  # contracts
        short_history[index, 2] = short_pos[3]  # entry_price
        short_history[index, 3] = short_pos[4]  # leverage
        short_history[index, 4] = short_pos[5]  # margin
        short_history[index, 5] = short_pos[6]  # unrealized_pnl
        short_history[index, 6] = short_pos[7]  # liquidation_price
        short_history[index, 7] = short_pos[9]  # tp_price
        short_history[index, 8] = short_pos[10]  # sl_price

@njit(cache=True)
def check_and_handle_liquidation(
    position: np.ndarray,
    account: np.ndarray,
    current_price: float64,
    margin_mode: int64,
    fee_close: float64,
    slippage: float64,
    stats: np.ndarray
) -> Tuple[boolean, float64]:
    """
    Vérifie et gère la liquidation d'une position
    
    Args:
        position: Position à vérifier
        account: État du compte
        current_price: Prix actuel
        margin_mode: Mode de marge
        fee_close: Frais de clôture
        slippage: Slippage à la liquidation
        stats: Statistiques de trading
        
    Returns:
        Tuple[boolean, float64]: (position_liquidée, pnl)
    """
    if position[0] == 0:  # Position inactive
        return False, 0.0
    
    direction = position[0]
    liq_price = position[7]
    
    # Si prix de liquidation non défini ou invalide
    if liq_price <= 0:
        return False, 0.0
    
    # Vérification de la liquidation
    is_liquidated = False
    
    if direction > 0:  # Long
        is_liquidated = current_price <= liq_price
    else:  # Short
        is_liquidated = current_price >= liq_price
    
    if is_liquidated:
        # Dans le cas d'une liquidation, la marge est perdue
        margin = position[5]
        pnl = -margin
        
        # Mise à jour des statistiques
        if stats is not None:
            stats[0] += 1  # Nombre total de trades
            stats[2] += 1  # Nombre de liquidations
            stats[5] = min(stats[5], pnl)  # Mise à jour de la perte maximale
            stats[6] += 1  # Incrémentation des pertes consécutives
            if stats[6] > stats[7]:
                stats[7] = stats[6]  # Mise à jour max pertes consécutives
        
        # Réinitialisation de la position
        position.fill(0.0)
        
        return True, pnl
    
    return False, 0.0

@njit(cache=True)
def simulate_realistic_trading(
    prices: np.ndarray,           
    signals: np.ndarray,          
    position_size: np.ndarray,    
    sl_pct: np.ndarray,          
    tp_pct: np.ndarray,          
    leverage: np.ndarray,             
    initial_balance: float64,      
    slippage: float64,            
    fee_open: float64,            
    fee_close: float64,           
    margin_mode: int64,           
    trading_mode: int64,            
    tick_size: float64,
    min_size: float64 = 0.001,
    max_size: float64 = 100000.0,
    safety_buffer: float64 = 0.01        
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Simule le trading avec gestion des stop loss, take profit et liquidations
    
    Args:
        prices: Array des prix
        signals: Array des signaux (1=long, -1=short, 0=neutre)
        position_size: Array des tailles de position en % du capital
        sl_pct: Array des niveaux de stop loss en %
        tp_pct: Array des niveaux de take profit en %
        leverage: Array des niveaux de levier
        initial_balance: Balance initiale du compte
        slippage: Slippage à l'exécution
        fee_open: Frais d'ouverture de position
        fee_close: Frais de clôture de position
        margin_mode: Mode de marge (0=Isolé, 1=Cross)
        trading_mode: Mode de trading (0=Unidirectionnel, 1=Hedge)
        tick_size: Taille minimale de tick
        min_size: Taille minimum
        max_size: Taille maximum
        safety_buffer: Marge de sécurité pour le prix de liquidation
        
    Returns:
        Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
            Métriques de performance, (account_history, long_history, short_history)
    """
    # Nombre de candles
    n_candles = len(prices)
    
    # Initialisation des structures
    account = create_account_state()
    account[0] = initial_balance  # wallet_balance
    account[2] = initial_balance  # equity
    
    long_pos = create_position_state()
    short_pos = create_position_state()
    
    # Création de l'historique
    account_history, long_history, short_history = create_history_arrays(n_candles)
    
    # Statistiques de trading
    stats = create_trade_stats()
    
    # Variables de tracking
    peak_equity = initial_balance
    max_drawdown = 0.0
    is_bankrupt = False
    
    # Boucle principale de simulation
    for i in range(n_candles):
        current_price = prices[i]
        current_leverage = leverage[i]
        
        # Mise à jour des métriques des positions existantes
        update_position_metrics(long_pos, current_price)
        update_position_metrics(short_pos, current_price)
        update_account_state(account, long_pos, short_pos, margin_mode)
        
        # Mise à jour du drawdown
        if account[2] > peak_equity:
            peak_equity = account[2]
        
        current_drawdown = (peak_equity - account[2]) / peak_equity if peak_equity > 0 else 0.0
        max_drawdown = max(max_drawdown, current_drawdown)
        
        # Vérification de faillite (equity proche de zéro)
        if account[2] <= initial_balance * 0.001 or current_drawdown >= 0.999:
            account[0] = 0.0  # Mettre la balance à 0
            account[2] = 0.0  # Mettre l'equity à 0
            is_bankrupt = True
            
            # Fermer les positions existantes avec perte totale
            if long_pos[0] != 0:
                long_pos.fill(0.0)
            if short_pos[0] != 0:
                short_pos.fill(0.0)
            
            # Enregistrer l'état actuel
            update_history(
                account_history, long_history, short_history,
                i, account, long_pos, short_pos, 1.0  # 100% drawdown
            )
            
            # Propager l'état de faillite pour le reste de l'historique
            for j in range(i+1, n_candles):
                account_history[j, 0] = 0.0  # wallet_balance = 0
                account_history[j, 1] = 0.0  # equity = 0
                account_history[j, 2] = 0.0  # unrealized_pnl = 0
                account_history[j, 3] = 1.0  # drawdown = 100%
                
                if j < len(long_history):
                    long_history[j].fill(0.0)
                if j < len(short_history):
                    short_history[j].fill(0.0)
            
            break  # Sortir de la boucle, la simulation est terminée
        
        # Si déjà en faillite, passer au prochain pas de temps
        if is_bankrupt:
            continue
        
        # Vérification des liquidations
        long_liquidated, long_liq_pnl = check_and_handle_liquidation(
            long_pos, account, current_price, margin_mode, fee_close, slippage, stats
        )
        
        short_liquidated, short_liq_pnl = check_and_handle_liquidation(
            short_pos, account, current_price, margin_mode, fee_close, slippage, stats
        )
        
        # Si une position a été liquidée, mettre à jour l'état du compte
        if long_liquidated or short_liquidated:
            update_account_state(account, long_pos, short_pos, margin_mode)
        
        # Vérification des TP/SL (si pas de liquidation)
        long_closed, long_pnl = False, 0.0
        short_closed, short_pnl = False, 0.0
        
        if not long_liquidated:
            long_closed, long_pnl = check_and_handle_tp_sl(
                long_pos, account, current_price, margin_mode, fee_close, slippage, stats
            )
        
        if not short_liquidated:
            short_closed, short_pnl = check_and_handle_tp_sl(
                short_pos, account, current_price, margin_mode, fee_close, slippage, stats
            )
        
        # Si une position a été fermée via TP/SL, mettre à jour l'état du compte
        if long_closed or short_closed:
            update_account_state(account, long_pos, short_pos, margin_mode)
        
        # Traitement des signaux de trading (si pas de liquidation ou TP/SL)
        if signals[i] != 0 and not (long_liquidated or short_liquidated or long_closed or short_closed):
            success, _ = process_trade_signal(
                signals[i], i, current_price, 
                position_size[i], current_leverage,
                sl_pct[i], tp_pct[i],
                account, long_pos, short_pos,
                trading_mode, margin_mode, fee_open,
                tick_size, min_size, max_size,
                safety_buffer, stats
            )
            
            # Si une position a été ouverte, mettre à jour l'état du compte
            if success:
                update_account_state(account, long_pos, short_pos, margin_mode)
        
        # Mise à jour de l'historique
        update_history(
            account_history, long_history, short_history,
            i, account, long_pos, short_pos, current_drawdown
        )
    
    # Fermeture des positions à la fin de la simulation (seulement si pas de faillite)
    if not is_bankrupt:
        if long_pos[0] != 0:
            pnl, is_win = close_position(long_pos, account, prices[-1], margin_mode, fee_close, slippage, stats)
        
        if short_pos[0] != 0:
            pnl, is_win = close_position(short_pos, account, prices[-1], margin_mode, fee_close, slippage, stats)
    
    # Extraction et compilation des résultats
    total_trades = stats[0]
    winning_trades = stats[1]
    liquidated_trades = stats[2]
    total_fees = stats[3]
    max_profit = stats[4]
    max_loss = stats[5]
    max_consecutive_losses = stats[7]
    
    # Protection contre les divisions par zéro
    if total_trades < 1:
        total_trades = 1
    
    # Calcul des métriques de performance
    roi = (account[0] / initial_balance) - 1.0
    win_rate = winning_trades / total_trades
    liquidation_rate = liquidated_trades / total_trades
    avg_profit_per_trade = (account[0] - initial_balance) / total_trades
    
    # Calcul du profit factor
    if max_loss != 0 and max_profit > 0:
        profit_factor = abs(max_profit / max_loss)
    else:
        profit_factor = 0.0
    
    # Création de l'array des métriques
    metrics = np.array([
        roi,                    # ROI
        win_rate,               # Win rate
        total_trades,           # Nombre total de trades
        max_drawdown,           # Max drawdown
        avg_profit_per_trade,   # Profit moyen par trade
        liquidation_rate,       # Taux de liquidation
        max_profit,             # Trade le plus profitable
        max_loss,               # Perte maximale sur un trade
        profit_factor           # Profit factor
    ], dtype=np.float64)
    
    return metrics, (account_history, long_history, short_history)

class Simulator:
    """
    Simulateur de trading avancé avec gestion de l'historique et des rapports
    """
    
    def __init__(self, config: SimulationConfig = None):
        """
        Initialise le simulateur de trading.
        
        Args:
            config: Configuration de la simulation (optionnel)
        """
        self.config = config or SimulationConfig()
        self.metrics = None
        self.history = None
        self.trade_history = []
        self.execution_time = 0
    
    def run(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        position_sizes: np.ndarray = None,
        sl_levels: np.ndarray = None,
        tp_levels: np.ndarray = None,
        leverage_levels: np.ndarray = None
    ) -> Dict:
        """
        Exécute la simulation avec les données fournies.
        
        Args:
            prices: Array des prix
            signals: Array des signaux (1=long, -1=short, 0=neutre)
            position_sizes: Array des tailles de position (optionnel)
            sl_levels: Array des niveaux de stop loss (optionnel)
            tp_levels: Array des niveaux de take profit (optionnel)
            leverage_levels: Array des niveaux de levier (optionnel)
            
        Returns:
            Dict: Résultats de la simulation
        """
        start_time = time.time()
        
        # Vérification des données d'entrée
        n_candles = len(prices)
        if n_candles == 0:
            raise ValueError("L'array des prix est vide")
        
        if len(signals) != n_candles:
            raise ValueError(f"La longueur des signaux ({len(signals)}) ne correspond pas à la longueur des prix ({n_candles})")
        
        # Configuration des arrays par défaut si non fournis
        if position_sizes is None:
            position_sizes = np.full(n_candles, 0.1, dtype=np.float64)  # 10% du capital par défaut
        
        if sl_levels is None:
            sl_levels = np.full(n_candles, 0.01, dtype=np.float64)  # 1% par défaut
        
        if tp_levels is None:
            tp_levels = np.full(n_candles, 0.02, dtype=np.float64)  # 2% par défaut
        
        if leverage_levels is None:
            leverage_levels = np.full(n_candles, self.config.leverage, dtype=np.float64)
        
        # Exécution de la simulation
        self.metrics, self.history = simulate_realistic_trading(
            prices=prices,
            signals=signals,
            position_size=position_sizes,
            sl_pct=sl_levels,
            tp_pct=tp_levels,
            leverage=leverage_levels,
            initial_balance=self.config.initial_balance,
            slippage=self.config.slippage,
            fee_open=self.config.fee_open,
            fee_close=self.config.fee_close,
            margin_mode=self.config.margin_mode,
            trading_mode=self.config.trading_mode,
            tick_size=self.config.tick_size,
            min_size=self.config.min_trade_size,
            max_size=self.config.max_trade_size,
            safety_buffer=0.01
        )
        
        # Enregistrement du temps d'exécution
        self.execution_time = time.time() - start_time
        
        # Extraction de l'historique des trades
        self.trade_history = self._extract_trade_history(prices, signals)
        
        # Formatage des résultats
        return self.get_results()
    
    def _extract_trade_history(self, prices: np.ndarray, signals: np.ndarray) -> List[Dict]:
        """
        Extrait l'historique détaillé des trades à partir des résultats de simulation.
        
        Args:
            prices: Array des prix
            signals: Array des signaux
            
        Returns:
            List[Dict]: Liste des trades avec leurs détails
        """
        trades = []
        account_history, long_history, short_history = self.history
        
        # Tracking des positions ouvertes
        active_long = False
        active_short = False
        long_entry_idx = 0
        short_entry_idx = 0
        
        for i in range(1, len(prices)):
            # Vérification de l'entrée en position longue
            if not active_long and long_history[i, 0] > 0 and long_history[i-1, 0] == 0:
                active_long = True
                long_entry_idx = i
            
            # Vérification de la sortie de position longue
            if active_long and long_history[i, 0] == 0 and long_history[i-1, 0] > 0:
                active_long = False
                
                # Calcul des métriques du trade
                entry_price = long_history[long_entry_idx, 2]
                exit_price = prices[i]
                contracts = long_history[long_entry_idx, 1]
                leverage = long_history[long_entry_idx, 3]
                duration = i - long_entry_idx
                pnl_pct = (exit_price / entry_price - 1.0) * 100.0 * leverage
                pnl_abs = (exit_price - entry_price) * contracts
                
                # Enregistrement du trade
                trades.append({
                    'type': 'long',
                    'entry_time': long_entry_idx,
                    'exit_time': i,
                    'entry_price': float(entry_price),
                    'exit_price': float(exit_price),
                    'contracts': float(contracts),
                    'leverage': float(leverage),
                    'duration': int(duration),
                    'pnl_pct': float(pnl_pct),
                    'pnl_abs': float(pnl_abs),
                    'success': bool(pnl_pct > 0)
                })
            
            # Vérification de l'entrée en position courte
            if not active_short and short_history[i, 0] > 0 and short_history[i-1, 0] == 0:
                active_short = True
                short_entry_idx = i
            
            # Vérification de la sortie de position courte
            if active_short and short_history[i, 0] == 0 and short_history[i-1, 0] > 0:
                active_short = False
                
                # Calcul des métriques du trade
                entry_price = short_history[short_entry_idx, 2]
                exit_price = prices[i]
                contracts = short_history[short_entry_idx, 1]
                leverage = short_history[short_entry_idx, 3]
                duration = i - short_entry_idx
                pnl_pct = (entry_price / exit_price - 1.0) * 100.0 * leverage
                pnl_abs = (entry_price - exit_price) * contracts
                
                # Enregistrement du trade
                trades.append({
                    'type': 'short',
                    'entry_time': short_entry_idx,
                    'exit_time': i,
                    'entry_price': float(entry_price),
                    'exit_price': float(exit_price),
                    'contracts': float(contracts),
                    'leverage': float(leverage),
                    'duration': int(duration),
                    'pnl_pct': float(pnl_pct),
                    'pnl_abs': float(pnl_abs),
                    'success': bool(pnl_pct > 0)
                })
        
        return trades
    
    def get_results(self) -> Dict:
        """
        Récupère les résultats formatés de la simulation.
        
        Returns:
            Dict: Résultats complets de la simulation
        """
        if self.metrics is None:
            return {"error": "Aucune simulation n'a été exécutée"}
        
        roi, win_rate, total_trades, max_drawdown, avg_profit_per_trade, liquidation_rate, max_profit, max_loss, profit_factor = self.metrics
        
        # Formatage des résultats
        results = {
            "performance": {
                "roi": float(roi),
                "roi_pct": float(roi * 100),
                "win_rate": float(win_rate),
                "win_rate_pct": float(win_rate * 100),
                "total_trades": int(total_trades),
                "max_drawdown": float(max_drawdown),
                "max_drawdown_pct": float(max_drawdown * 100),
                "avg_profit_per_trade": float(avg_profit_per_trade),
                "avg_profit_per_trade_pct": float(avg_profit_per_trade * 100),
                "liquidation_rate": float(liquidation_rate),
                "max_profit_trade": float(max_profit),
                "max_loss_trade": float(max_loss),
                "profit_factor": float(profit_factor),
                "total_pnl": float(roi * self.config.initial_balance),
                "final_balance": float(self.config.initial_balance * (1 + roi))
            },
            "config": {
                "initial_balance": self.config.initial_balance,
                "fee_open": self.config.fee_open,
                "fee_close": self.config.fee_close,
                "slippage": self.config.slippage,
                "tick_size": self.config.tick_size,
                "min_trade_size": self.config.min_trade_size,
                "max_trade_size": self.config.max_trade_size,
                "leverage": self.config.leverage,
                "margin_mode": self.config.margin_mode,
                "trading_mode": self.config.trading_mode
            },
            "execution_time": self.execution_time,
            "trade_count": len(self.trade_history)
        }
        
        # Calcul de métriques additionnelles si des trades existent
        if total_trades > 0:
            # Ratio de Sharpe approximatif
            if self.trade_history:
                pnl_values = [trade["pnl_pct"] for trade in self.trade_history]
                if pnl_values and len(pnl_values) > 1:
                    mean_return = np.mean(pnl_values)
                    std_return = max(np.std(pnl_values), 1e-6)  # Éviter division par zéro
                    sharpe_ratio = mean_return / std_return
                    results["performance"]["sharpe_ratio"] = float(sharpe_ratio)
                
                # Durée moyenne des trades
                durations = [trade["duration"] for trade in self.trade_history]
                if durations:
                    results["performance"]["avg_trade_duration"] = float(np.mean(durations))
        
        return results
    
    def save_to_csv(self, base_filepath: str) -> None:
        """
        Sauvegarde les résultats de la simulation au format CSV.
        
        Args:
            base_filepath: Chemin de base pour les fichiers (sans extension)
        """
        if self.history is None:
            raise ValueError("Aucun historique de simulation à sauvegarder")
        
        # Création du répertoire si nécessaire
        os.makedirs(os.path.dirname(base_filepath), exist_ok=True)
        
        # Décomposition de l'historique
        account_history, long_history, short_history = self.history
        
        # Sauvegarde de l'historique du compte
        account_df = pd.DataFrame(
            account_history,
            columns=["wallet_balance", "equity", "unrealized_pnl", "drawdown"]
        )
        account_df.to_csv(f"{base_filepath}_account.csv", index=True)
        
        # Sauvegarde des positions longues
        long_df = pd.DataFrame(
            long_history,
            columns=["active", "contracts", "entry_price", "leverage", "margin",
                    "unrealized_pnl", "liquidation_price", "tp_price", "sl_price"]
        )
        long_df.to_csv(f"{base_filepath}_long.csv", index=True)
        
        # Sauvegarde des positions courtes
        short_df = pd.DataFrame(
            short_history,
            columns=["active", "contracts", "entry_price", "leverage", "margin",
                    "unrealized_pnl", "liquidation_price", "tp_price", "sl_price"]
        )
        short_df.to_csv(f"{base_filepath}_short.csv", index=True)
        
        # Sauvegarde de l'historique des trades
        if self.trade_history:
            trade_df = pd.DataFrame(self.trade_history)
            trade_df.to_csv(f"{base_filepath}_trades.csv", index=False)
        
        # Sauvegarde du résumé
        summary_df = pd.DataFrame([self.get_results()])
        summary_df.to_csv(f"{base_filepath}_summary.csv", index=False)
        
        logger.info(f"Historique de simulation sauvegardé dans {base_filepath}_*.csv")