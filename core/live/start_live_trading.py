#!/usr/bin/env python3
"""
Script de lancement rapide pour le trading en direct.
Fournit une interface simple pour démarrer, configurer et surveiller le trading en direct.
"""

import os
import sys
import json
import argparse
import logging
import asyncio
from datetime import datetime

from live_launcher import LiveLauncher
from core.strategy.strategy_manager import StrategyManager
from live_config import ExchangeType, LiveTradingMode, create_default_config

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("start_live_trading")


def print_header():
    """Affiche l'en-tête du script"""
    print("\n" + "=" * 80)
    print(" " * 30 + "LANCEUR DE TRADING EN DIRECT")
    print("=" * 80 + "\n")


def list_available_strategies():
    """
    Liste toutes les stratégies disponibles.
    
    Returns:
        List: Liste des stratégies disponibles
    """
    manager = StrategyManager()
    strategies = manager.list_strategies()
    
    if not strategies:
        print("\nAucune stratégie disponible. Créez des stratégies avec le constructeur.")
        return []
    
    print("\nSTRATÉGIES DISPONIBLES:")
    print("-" * 80)
    print(f"{'ID':<10} {'NOM':<30} {'DERNIÈRE MODIFICATION':<25} {'TAGS'}")
    print("-" * 80)
    
    for strategy in strategies:
        # Formater la date
        updated_at = strategy.get("updated_at", "")
        if updated_at:
            try:
                dt = datetime.fromisoformat(updated_at)
                updated_at = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass
        
        # Formater les tags
        tags = ", ".join(strategy.get("tags", []))
        
        print(f"{strategy['id']:<10} {strategy['name'][:30]:<30} {updated_at:<25} {tags}")
    
    return strategies


def list_available_configs():
    """
    Liste toutes les configurations disponibles.
    
    Returns:
        List: Liste des configurations disponibles
    """
    config_dir = "config/live"
    
    if not os.path.exists(config_dir):
        print("\nAucune configuration disponible.")
        return []
    
    configs = []
    
    for filename in os.listdir(config_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(config_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extraire les informations clés
                config_info = {
                    "name": data.get("name", ""),
                    "exchange": data.get("exchange", ""),
                    "symbol": data.get("market", {}).get("symbol", ""),
                    "strategy_id": data.get("strategy_id", ""),
                    "trading_mode": data.get("trading_mode", ""),
                    "filepath": filepath
                }
                
                configs.append(config_info)
            except Exception as e:
                logger.warning(f"Erreur lors de la lecture de {filepath}: {str(e)}")
    
    if not configs:
        print("\nAucune configuration disponible.")
        return []
    
    print("\nCONFIGURATIONS DISPONIBLES:")
    print("-" * 80)
    print(f"{'NOM':<30} {'EXCHANGE':<10} {'SYMBOL':<10} {'MODE':<8} {'STRATÉGIE':<10}")
    print("-" * 80)
    
    for config in configs:
        print(f"{config['name'][:30]:<30} {config['exchange']:<10} {config['symbol']:<10} "
              f"{config['trading_mode']:<8} {config['strategy_id']:<10}")
    
    return configs


def create_new_config():
    """
    Crée une nouvelle configuration de trading.
    
    Returns:
        str: Chemin du fichier de configuration créé
    """
    print("\nCRÉATION D'UNE NOUVELLE CONFIGURATION")
    print("-" * 80)
    
    # Sélection de l'exchange
    print("\nExchanges disponibles:")
    for i, exchange in enumerate([ExchangeType.BITGET, ExchangeType.BINANCE]):
        print(f"{i+1}. {exchange.value}")
    
    exchange_choice = int(input("\nChoisissez un exchange (numéro): ") or "1")
    exchange = [ExchangeType.BITGET, ExchangeType.BINANCE][exchange_choice - 1]
    
    # Saisie du symbole
    symbol = input("\nEntrez le symbole (par exemple BTCUSDT): ") or "BTCUSDT"
    
    # Sélection du mode de trading
    print("\nModes de trading disponibles:")
    for i, mode in enumerate([LiveTradingMode.PAPER, LiveTradingMode.DEMO, LiveTradingMode.REAL]):
        print(f"{i+1}. {mode.value}")
    
    mode_choice = int(input("\nChoisissez un mode de trading (numéro): ") or "1")
    trading_mode = [LiveTradingMode.PAPER, LiveTradingMode.DEMO, LiveTradingMode.REAL][mode_choice - 1]
    
    # Sélection de la stratégie
    print("\nSélection de la stratégie:")
    strategies = list_available_strategies()
    
    if not strategies:
        print("Vous devez créer une stratégie avant de pouvoir configurer le trading en direct.")
        return None
    
    strategy_id = input("\nEntrez l'ID de la stratégie: ")
    
    if not any(s['id'] == strategy_id for s in strategies):
        print(f"Stratégie '{strategy_id}' non trouvée.")
        return None
    
    # Création de la configuration
    config = create_default_config(exchange, symbol)
    config.name = f"Config {exchange.value} {symbol} {trading_mode.value}"
    config.trading_mode = trading_mode
    config.strategy_id = strategy_id
    
    # Saisie des clés API si mode réel
    if trading_mode == LiveTradingMode.REAL:
        print("\nConfiguration des clés API pour le trading réel:")
        config.api_key = input("API Key: ")
        config.api_secret = input("API Secret: ")
        
        if exchange == ExchangeType.BITGET:
            config.api_passphrase = input("API Passphrase: ")
    
    # Sauvegarder la configuration
    os.makedirs("config/live", exist_ok=True)
    filename = f"{exchange.value}_{symbol}_{trading_mode.value}.json"
    filepath = os.path.join("config/live", filename)
    
    config.save(filepath)
    print(f"\nConfiguration sauvegardée dans {filepath}")
    
    return filepath


async def start_trading_with_config(config_path):
    """
    Démarre le trading avec une configuration existante.
    
    Args:
        config_path: Chemin du fichier de configuration
    """
    try:
        # Créer le lanceur
        launcher = LiveLauncher(config_path)
        
        # Vérifier que la configuration a été chargée
        if not launcher.config:
            print(f"Erreur: Impossible de charger la configuration depuis {config_path}")
            return
        
        # Charger la stratégie
        if not launcher.load_strategy(launcher.config.strategy_id):
            print(f"Erreur: Impossible de charger la stratégie {launcher.config.strategy_id}")
            return
        
        # Démarrer le trading
        print(f"\nDémarrage du trading pour {launcher.config.market.symbol} sur {launcher.config.exchange.value}")
        print(f"Mode: {launcher.config.trading_mode.value}")
        print(f"Stratégie: {launcher.strategy.config.name}")
        print("\nAppuyez sur Ctrl+C pour arrêter le trading.\n")
        
        # Exécuter
        await launcher.run()
        
    except KeyboardInterrupt:
        print("\nArrêt du trading...")
        if launcher and hasattr(launcher, 'stop_trading'):
            await launcher.stop_trading()
    except Exception as e:
        logger.error(f"Erreur lors du démarrage du trading: {str(e)}")


async def interactive_menu():
    """Menu interactif pour le lancement du trading"""
    while True:
        print_header()
        print("OPTIONS:")
        print("1. Lister les stratégies disponibles")
        print("2. Lister les configurations disponibles")
        print("3. Créer une nouvelle configuration")
        print("4. Démarrer le trading avec une configuration existante")
        print("0. Quitter")
        
        choice = input("\nChoisissez une option: ")
        
        if choice == "1":
            list_available_strategies()
        elif choice == "2":
            configs = list_available_configs()
        elif choice == "3":
            create_new_config()
        elif choice == "4":
            configs = list_available_configs()
            if configs:
                config_path = input("\nEntrez le chemin complet du fichier de configuration: ")
                if os.path.exists(config_path):
                    await start_trading_with_config(config_path)
                else:
                    print(f"Fichier {config_path} introuvable.")
        elif choice == "0":
            print("\nAu revoir!\n")
            break
        else:
            print("\nOption invalide. Veuillez réessayer.")
        
        input("\nAppuyez sur Entrée pour continuer...")


async def main():
    """
    Fonction principale du script.
    """
    parser = argparse.ArgumentParser(description="Lanceur de trading en direct simplifié")
    parser.add_argument("--config", type=str, help="Chemin du fichier de configuration")
    parser.add_argument("--interactive", action="store_true", help="Mode interactif")
    
    args = parser.parse_args()
    
    if args.interactive or not args.config:
        await interactive_menu()
    elif args.config:
        if os.path.exists(args.config):
            await start_trading_with_config(args.config)
        else:
            print(f"Fichier de configuration {args.config} introuvable.")
            sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nArrêt du programme.")
    except Exception as e:
        logger.error(f"Erreur non gérée: {str(e)}")
        sys.exit(1)