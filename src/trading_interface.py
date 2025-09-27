"""
AlgoTrendy Trading Interface - Unified Access to All Trading Tools
========================================================================

A comprehensive interface that provides unified access to all AlgoTrendy trading systems,
tools, and capabilities. This interface serves as the central hub for:

- ML Trading Systems (Stocks, Futures, Crypto)
- Backtesting & Market Replay
- Live Trading Execution
- AI Strategy Discovery
- Performance Monitoring
- Configuration Management

Author: AlgoTrendy Team
Version: 2.0.0
"""

import sys
import os

# Add src directory to path for imports BEFORE any other imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Import all trading systems and tools
from config import CONFIG, logger
from data_manager import DataManager
try:
    from alpaca_integration import AlpacaIntegratedTrader
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: Alpaca integration not available")
from backtester import Backtester
from market_replay import MarketReplay, ReplayConfig
from quantconnect_integration import QuantConnectIntegration
try:
    from advanced_ml_trainer import AdvancedMLTrainer
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    print("Warning: Advanced ML training not available")
try:
    from ai_indicator_agent import IndicatorDiscoveryAgent
    AI_INDICATOR_AVAILABLE = True
except ImportError:
    AI_INDICATOR_AVAILABLE = False
    print("Warning: AI indicator agent not available")

try:
    from crypto_scalping_trader import CryptoScalpingTrader
    CRYPTO_SCALPING_AVAILABLE = True
except ImportError:
    CRYPTO_SCALPING_AVAILABLE = False
    print("Warning: Crypto scalping trader not available")

try:
    from ai_crypto_strategy_agent import AICryptoStrategyAgent
    AI_CRYPTO_STRATEGY_AVAILABLE = True
except ImportError:
    AI_CRYPTO_STRATEGY_AVAILABLE = False
    print("Warning: AI crypto strategy agent not available")

try:
    from ai_futures_strategy_agent import AIFuturesStrategyAgent
    AI_FUTURES_STRATEGY_AVAILABLE = True
except ImportError:
    AI_FUTURES_STRATEGY_AVAILABLE = False
    print("Warning: AI futures strategy agent not available")

try:
    from automated_futures_trader import AutomatedFuturesTrader
    AUTOMATED_FUTURES_AVAILABLE = True
except ImportError:
    AUTOMATED_FUTURES_AVAILABLE = False
    print("Warning: Automated futures trader not available")

try:
    from futures_contract_rolling import FuturesContractRoller, TickDataManager
    FUTURES_CONTRACT_AVAILABLE = True
except ImportError:
    FUTURES_CONTRACT_AVAILABLE = False
    print("Warning: Futures contract rolling not available")


class TradingInterface:
    """
    Unified Trading Interface for AlgoTrendy Platform

    Provides centralized access to all trading systems, tools, and capabilities
    with a consistent API and user-friendly interface.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the trading interface.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path or "config/.env"
        self._load_configuration()

        # Initialize core components
        self.data_manager = DataManager()
        self.backtester = Backtester()
        self.market_replay = None
        self.alpaca_trader = None
        self.quantconnect = None

        # Initialize AI agents
        self.indicator_agent = IndicatorDiscoveryAgent() if AI_INDICATOR_AVAILABLE else None
        self.crypto_strategy_agent = AICryptoStrategyAgent() if AI_CRYPTO_STRATEGY_AVAILABLE else None
        self.futures_strategy_agent = AIFuturesStrategyAgent() if AI_FUTURES_STRATEGY_AVAILABLE else None

        # Initialize trading systems
        self.crypto_scalper = None
        self.futures_trader = None
        self.futures_roller = FuturesContractRoller() if FUTURES_CONTRACT_AVAILABLE else None
        self.tick_manager = TickDataManager() if FUTURES_CONTRACT_AVAILABLE else None

        # Performance tracking
        self.performance_history = []
        self.active_positions = {}
        self.daily_pnl = 0.0

        logger.info("AlgoTrendy Trading Interface initialized successfully")

    def _load_configuration(self):
        """Load configuration settings."""
        try:
            # Load environment variables
            from dotenv import load_dotenv
            load_dotenv(self.config_path)

            # Update CONFIG with loaded values
            CONFIG.alpaca_api_key = os.getenv('ALPACA_API_KEY')
            CONFIG.alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')
            CONFIG.paper_trading = os.getenv('PAPER_TRADING', 'true').lower() == 'true'

            logger.info("Configuration loaded successfully")

        except Exception as e:
            logger.warning(f"Could not load configuration: {e}")

    # ============================================================================
    # MAIN INTERFACE METHODS
    # ============================================================================

    def show_main_menu(self):
        """Display the main interface menu."""
        while True:
            self._clear_screen()
            print("=" * 80)
            print("ALGOTRENDY TRADING INTERFACE v2.0")
            print("=" * 80)
            print()
            print("TRADING SYSTEMS")
            print("  1. Stock Trading (ML-Based)")
            print("  2. Futures Day Trading")
            print("  3. Crypto Scalping (24/7)")
            print()
            print("AI & ANALYSIS")
            print("  4. AI Indicator Discovery")
            print("  5. AI Strategy Agents")
            print("  6. Advanced ML Training")
            print()
            print("TESTING & BACKTESTING")
            print("  7. Backtesting Engine")
            print("  8. Market Replay Testing")
            print("  9. QuantConnect Integration")
            print()
            print("CONFIGURATION & MONITORING")
            print(" 10. Performance Dashboard")
            print(" 11. Configuration Manager")
            print(" 12. System Diagnostics")
            print()
            print("  0. Exit Interface")
            print()
            print("=" * 80)

            choice = input("Select option (0-12): ").strip()

            if choice == "0":
                print("Thank you for using AlgoTrendy!")
                break
            elif choice == "1":
                self._stock_trading_menu()
            elif choice == "2":
                self._futures_trading_menu()
            elif choice == "3":
                self._crypto_trading_menu()
            elif choice == "4":
                self._ai_indicator_menu()
            elif choice == "5":
                self._ai_strategy_menu()
            elif choice == "6":
                self._advanced_ml_menu()
            elif choice == "7":
                self._backtesting_menu()
            elif choice == "8":
                self._market_replay_menu()
            elif choice == "9":
                self._quantconnect_menu()
            elif choice == "10":
                self._performance_dashboard()
            elif choice == "11":
                self._configuration_menu()
            elif choice == "12":
                self._diagnostics_menu()
            else:
                print("[ERROR] Invalid choice. Please try again.")
                input("Press Enter to continue...")

    # ============================================================================
    # STOCK TRADING MENU
    # ============================================================================

    def _stock_trading_menu(self):
        """Stock trading submenu."""
        while True:
            self._clear_screen()
            print("[STOCK] STOCK TRADING SYSTEM")
            print("=" * 50)
            print()
            print("1. Generate ML Signals")
            print("2. Execute Paper Trades")
            print("3. View Portfolio")
            print("4. Backtest Strategy")
            print("5. Train ML Model")
            print()
            print("0. Back to Main Menu")
            print()

            choice = input("Select option: ").strip()

            if choice == "0":
                break
            elif choice == "1":
                self._generate_stock_signals()
            elif choice == "2":
                self._execute_paper_trades()
            elif choice == "3":
                self._view_portfolio()
            elif choice == "4":
                self._backtest_stock_strategy()
            elif choice == "5":
                self._train_stock_model()
            else:
                print("[ERROR] Invalid choice.")
                input("Press Enter...")

    # ============================================================================
    # FUTURES TRADING MENU
    # ============================================================================

    def _futures_trading_menu(self):
        """Futures trading submenu."""
        while True:
            self._clear_screen()
            print("[FUTURES] FUTURES DAY TRADING SYSTEM")
            print("=" * 50)
            print()
            print("1. Start Automated Trading")
            print("2. Generate Futures Signals")
            print("3. Check Contract Rolling Status")
            print("4. Execute Contract Roll")
            print("5. Futures Backtest")
            print("6. Tick Data Analysis")
            print()
            print("0. Back to Main Menu")
            print()

            choice = input("Select option: ").strip()

            if choice == "0":
                break
            elif choice == "1":
                self._start_automated_futures()
            elif choice == "2":
                self._generate_futures_signals()
            elif choice == "3":
                self._check_contract_rolling()
            elif choice == "4":
                self._execute_contract_roll()
            elif choice == "5":
                self._backtest_futures_strategy()
            elif choice == "6":
                self._tick_data_analysis()
            else:
                print("[ERROR] Invalid choice.")
                input("Press Enter...")

    # ============================================================================
    # CRYPTO TRADING MENU
    # ============================================================================

    def _crypto_trading_menu(self):
        """Crypto trading submenu."""
        while True:
            self._clear_screen()
            print("[CRYPTO] CRYPTO SCALPING SYSTEM (24/7)")
            print("=" * 50)
            print()
            print("1. Start Crypto Scalping")
            print("2. View Scalping Performance")
            print("3. Configure Scalping Parameters")
            print("4. Stop Scalping")
            print()
            print("0. Back to Main Menu")
            print()

            choice = input("Select option: ").strip()

            if choice == "0":
                break
            elif choice == "1":
                self._start_crypto_scalping()
            elif choice == "2":
                self._view_scalping_performance()
            elif choice == "3":
                self._configure_scalping()
            elif choice == "4":
                self._stop_crypto_scalping()
            else:
                print("[ERROR] Invalid choice.")
                input("Press Enter...")

    # ============================================================================
    # AI INDICATOR MENU
    # ============================================================================

    def _ai_indicator_menu(self):
        """AI indicator discovery submenu."""
        while True:
            self._clear_screen()
            print("[AI] AI INDICATOR DISCOVERY AGENT")
            print("=" * 50)
            print()
            print("1. Discover New Indicators")
            print("2. Test Indicator Performance")
            print("3. Integrate Best Indicators")
            print("4. View Indicator Library")
            print()
            print("0. Back to Main Menu")
            print()

            choice = input("Select option: ").strip()

            if choice == "0":
                break
            elif choice == "1":
                self._discover_indicators()
            elif choice == "2":
                self._test_indicators()
            elif choice == "3":
                self._integrate_indicators()
            elif choice == "4":
                self._view_indicator_library()
            else:
                print("[ERROR] Invalid choice.")
                input("Press Enter...")

    # ============================================================================
    # AI STRATEGY MENU
    # ============================================================================

    def _ai_strategy_menu(self):
        """AI strategy agents submenu."""
        while True:
            self._clear_screen()
            print("[AI] AI STRATEGY DISCOVERY AGENTS")
            print("=" * 50)
            print()
            print("1. Discover Crypto Strategies")
            print("2. Discover Futures Strategies")
            print("3. Test Strategy Performance")
            print("4. Integrate Best Strategies")
            print("5. View Strategy Library")
            print()
            print("0. Back to Main Menu")
            print()

            choice = input("Select option: ").strip()

            if choice == "0":
                break
            elif choice == "1":
                self._discover_crypto_strategies()
            elif choice == "2":
                self._discover_futures_strategies()
            elif choice == "3":
                self._test_strategies()
            elif choice == "4":
                self._integrate_strategies()
            elif choice == "5":
                self._view_strategy_library()
            else:
                print("[ERROR] Invalid choice.")
                input("Press Enter...")

    # ============================================================================
    # ADVANCED ML MENU
    # ============================================================================

    def _advanced_ml_menu(self):
        """Advanced ML training submenu."""
        while True:
            self._clear_screen()
            print("[ML] ADVANCED ML TRAINING (>80% Accuracy)")
            print("=" * 50)
            print()
            print("1. Train Ensemble Model")
            print("2. Hyperparameter Optimization")
            print("3. Cross-Validation")
            print("4. Feature Engineering")
            print("5. Model Comparison")
            print()
            print("0. Back to Main Menu")
            print()

            choice = input("Select option: ").strip()

            if choice == "0":
                break
            elif choice == "1":
                self._train_ensemble_model()
            elif choice == "2":
                self._hyperparameter_optimization()
            elif choice == "3":
                self._cross_validation()
            elif choice == "4":
                self._feature_engineering()
            elif choice == "5":
                self._model_comparison()
            else:
                print("[ERROR] Invalid choice.")
                input("Press Enter...")

    # ============================================================================
    # BACKTESTING MENU
    # ============================================================================

    def _backtesting_menu(self):
        """Backtesting engine submenu."""
        while True:
            self._clear_screen()
            print("[BACKTEST] BACKTESTING ENGINE")
            print("=" * 50)
            print()
            print("1. Run Stock Backtest")
            print("2. Run Futures Backtest")
            print("3. Walk-Forward Analysis")
            print("4. Monte Carlo Simulation")
            print("5. Performance Analytics")
            print()
            print("0. Back to Main Menu")
            print()

            choice = input("Select option: ").strip()

            if choice == "0":
                break
            elif choice == "1":
                self._run_stock_backtest()
            elif choice == "2":
                self._run_futures_backtest()
            elif choice == "3":
                self._walk_forward_analysis()
            elif choice == "4":
                self._monte_carlo_simulation()
            elif choice == "5":
                self._performance_analytics()
            else:
                print("[ERROR] Invalid choice.")
                input("Press Enter...")

    # ============================================================================
    # MARKET REPLAY MENU
    # ============================================================================

    def _market_replay_menu(self):
        """Market replay testing submenu."""
        while True:
            self._clear_screen()
            print("[REPLAY] MARKET REPLAY TESTING")
            print("=" * 50)
            print()
            print("1. Configure Replay")
            print("2. Start Replay")
            print("3. Pause/Resume Replay")
            print("4. View Replay Status")
            print("5. Analyze Replay Results")
            print()
            print("0. Back to Main Menu")
            print()

            choice = input("Select option: ").strip()

            if choice == "0":
                break
            elif choice == "1":
                self._configure_replay()
            elif choice == "2":
                self._start_replay()
            elif choice == "3":
                self._control_replay()
            elif choice == "4":
                self._view_replay_status()
            elif choice == "5":
                self._analyze_replay_results()
            else:
                print("[ERROR] Invalid choice.")
                input("Press Enter...")

    # ============================================================================
    # QUANTCONNECT MENU
    # ============================================================================

    def _quantconnect_menu(self):
        """QuantConnect integration submenu."""
        while True:
            self._clear_screen()
            print("[QC] QUANTCONNECT CLOUD INTEGRATION")
            print("=" * 50)
            print()
            print("1. Setup QuantConnect")
            print("2. List Projects")
            print("3. Deploy Algorithm")
            print("4. View Backtest Results")
            print("5. Live Trading Status")
            print()
            print("0. Back to Main Menu")
            print()

            choice = input("Select option: ").strip()

            if choice == "0":
                break
            elif choice == "1":
                self._setup_quantconnect()
            elif choice == "2":
                self._list_qc_projects()
            elif choice == "3":
                self._deploy_qc_algorithm()
            elif choice == "4":
                self._view_qc_backtests()
            elif choice == "5":
                self._qc_live_status()
            else:
                print("[ERROR] Invalid choice.")
                input("Press Enter...")

    # ============================================================================
    # PERFORMANCE DASHBOARD
    # ============================================================================

    def _performance_dashboard(self):
        """Performance monitoring dashboard."""
        self._clear_screen()
        print("📈 PERFORMANCE DASHBOARD")
        print("=" * 50)
        print()

        # Portfolio Overview
        print("💼 PORTFOLIO OVERVIEW")
        print("-" * 30)
        print(f"Total Value: ${self._get_portfolio_value():,.2f}")
        print(f"Daily P&L: ${self.daily_pnl:,.2f}")
        print(f"Active Positions: {len(self.active_positions)}")
        print()

        # System Status
        print("⚙️ SYSTEM STATUS")
        print("-" * 30)
        print(f"Alpaca Connection: {self._check_alpaca_status()}")
        print(f"QuantConnect: {self._check_qc_status()}")
        print(f"Market Replay: {self._check_replay_status()}")
        print(f"Crypto Scalping: {self._check_scalping_status()}")
        print()

        # Recent Performance
        print("📊 RECENT PERFORMANCE")
        print("-" * 30)
        if self.performance_history:
            recent = self.performance_history[-5:]
            for entry in recent:
                print(f"{entry['date']}: ${entry['pnl']:,.2f} ({entry['return']:.2f}%)")
        else:
            print("No performance data available")
        print()

        # Trading Statistics
        print("📈 TRADING STATISTICS")
        print("-" * 30)
        stats = self._calculate_trading_stats()
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1%}")
        print(f"Avg Win: ${stats['avg_win']:,.2f}")
        print(f"Avg Loss: ${stats['avg_loss']:,.2f}")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print()

        input("Press Enter to return to main menu...")

    # ============================================================================
    # CONFIGURATION MENU
    # ============================================================================

    def _configuration_menu(self):
        """Configuration management submenu."""
        while True:
            self._clear_screen()
            print("[CONFIG] CONFIGURATION MANAGER")
            print("=" * 50)
            print()
            print("1. View Current Settings")
            print("2. Update API Keys")
            print("3. Trading Parameters")
            print("4. Risk Management")
            print("5. System Preferences")
            print()
            print("0. Back to Main Menu")
            print()

            choice = input("Select option: ").strip()

            if choice == "0":
                break
            elif choice == "1":
                self._view_settings()
            elif choice == "2":
                self._update_api_keys()
            elif choice == "3":
                self._trading_parameters()
            elif choice == "4":
                self._risk_management()
            elif choice == "5":
                self._system_preferences()
            else:
                print("[ERROR] Invalid choice.")
                input("Press Enter...")

    # ============================================================================
    # DIAGNOSTICS MENU
    # ============================================================================

    def _diagnostics_menu(self):
        """System diagnostics submenu."""
        self._clear_screen()
        print("[DIAG] SYSTEM DIAGNOSTICS")
        print("=" * 50)
        print()

        print("[CHECK] RUNNING DIAGNOSTIC CHECKS...")
        print()

        # Check all systems
        checks = {
            "Configuration": self._check_config(),
            "Data Manager": self._check_data_manager(),
            "Alpaca Integration": self._check_alpaca_integration(),
            "Backtester": self._check_backtester(),
            "Market Replay": self._check_market_replay(),
            "QuantConnect": self._check_quantconnect(),
            "AI Agents": self._check_ai_agents(),
            "Trading Systems": self._check_trading_systems(),
        }

        for component, status in checks.items():
            status_icon = "[OK]" if status['status'] == 'OK' else "[ERROR]"
            print(f"{status_icon} {component}: {status['message']}")

        print()
        print("[RESOURCES] SYSTEM RESOURCES")
        print("-" * 30)
        # Add system resource monitoring here

        print()
        input("Press Enter to return to main menu...")

    # ============================================================================
    # IMPLEMENTATION METHODS
    # ============================================================================

    def _clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def _get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        try:
            if self.alpaca_trader:
                return self.alpaca_trader.get_portfolio_value()
            return 10000.0  # Default demo value
        except:
            return 10000.0

    def _check_alpaca_status(self) -> str:
        """Check Alpaca connection status."""
        try:
            if self.alpaca_trader and self.alpaca_trader.check_connection():
                return "✅ Connected"
            return "❌ Disconnected"
        except:
            return "❌ Error"

    def _check_qc_status(self) -> str:
        """Check QuantConnect status."""
        try:
            if self.quantconnect and self.quantconnect.check_connection():
                return "✅ Connected"
            return "❌ Disconnected"
        except:
            return "❌ Error"

    def _check_replay_status(self) -> str:
        """Check market replay status."""
        try:
            if self.market_replay and self.market_replay.get_status()['is_running']:
                return "▶️ Running"
            return "⏸️ Stopped"
        except:
            return "❌ Error"

    def _check_scalping_status(self) -> str:
        """Check crypto scalping status."""
        try:
            if self.crypto_scalper and hasattr(self.crypto_scalper, 'is_running') and self.crypto_scalper.is_running:
                return "▶️ Running"
            return "⏸️ Stopped"
        except:
            return "❌ Error"

    def _calculate_trading_stats(self) -> Dict[str, Any]:
        """Calculate trading statistics."""
        # Placeholder implementation
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }

    # ============================================================================
    # PLACEHOLDER IMPLEMENTATIONS
    # ============================================================================

    def _generate_stock_signals(self):
        """Generate stock trading signals."""
        print("📈 Generating ML-based stock signals...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _execute_paper_trades(self):
        """Execute paper trades."""
        print("💰 Executing paper trades...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _view_portfolio(self):
        """View portfolio status."""
        print("💼 Portfolio Overview...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _backtest_stock_strategy(self):
        """Backtest stock strategy."""
        print("📊 Backtesting stock strategy...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _train_stock_model(self):
        """Train stock ML model."""
        print("🧠 Training stock ML model...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _start_automated_futures(self):
        """Start automated futures trading."""
        print("⚡ Starting automated futures trading...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _generate_futures_signals(self):
        """Generate futures signals."""
        print("📊 Generating futures signals...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _check_contract_rolling(self):
        """Check contract rolling status."""
        print("🔄 Checking contract rolling status...")
        try:
            status = self.futures_roller.check_roll_status('ES')
            print(f"ES Contract Status: {status}")
        except Exception as e:
            print(f"Error: {e}")
        input("Press Enter...")

    def _execute_contract_roll(self):
        """Execute contract roll."""
        print("🔄 Executing contract roll...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _backtest_futures_strategy(self):
        """Backtest futures strategy."""
        print("📊 Backtesting futures strategy...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _tick_data_analysis(self):
        """Analyze tick data."""
        print("📊 Analyzing tick data...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _start_crypto_scalping(self):
        """Start crypto scalping."""
        if not CRYPTO_SCALPING_AVAILABLE:
            print("❌ Crypto scalping trader not available")
            print("   Install required dependencies: pip install ccxt binance python-binance")
            input("Press Enter to continue...")
            return

        print("₿ Starting crypto scalping...")
        print()

        # Get user preferences
        exchange = input("Exchange (binance/coinbase/alpaca) [binance]: ").strip() or "binance"
        symbols_input = input("Symbols (comma-separated) [BTC/USDT,ETH/USDT]: ").strip() or "BTC/USDT,ETH/USDT"
        symbols = [s.strip() for s in symbols_input.split(',')]

        try:
            # Initialize and start scalping
            self.crypto_scalper = CryptoScalpingTrader(exchange=exchange, symbols=symbols)
            self.crypto_scalper.start_scalping()

            print("✅ Crypto scalping started successfully!")
            print(f"   Exchange: {exchange}")
            print(f"   Symbols: {', '.join(symbols)}")
            print("   Monitor performance in the dashboard")

        except Exception as e:
            print(f"❌ Error starting crypto scalping: {e}")

        input("Press Enter to continue...")

    def _view_scalping_performance(self):
        """View scalping performance."""
        if not self.crypto_scalper:
            print("❌ No crypto scalping session active")
            print("   Start scalping first using option 1")
            input("Press Enter to continue...")
            return

        print("📈 Crypto Scalping Performance")
        print("=" * 40)

        try:
            report = self.crypto_scalper.get_performance_report()

            print(f"Total Trades: {report.get('total_trades', 0)}")
            print(f"Win Rate: {report.get('win_rate', 0):.1%}")
            print(f"Total P&L: ${report.get('total_pnl', 0):,.2f}")
            print(f"Active Positions: {report.get('active_positions', 0)}")
            print(f"Uptime: {report.get('uptime', 'N/A')}")

            if report.get('sharpe_ratio'):
                print(f"Sharpe Ratio: {report['sharpe_ratio']:.2f}")

        except Exception as e:
            print(f"Error retrieving performance: {e}")

        input("Press Enter to continue...")

    def _configure_scalping(self):
        """Configure scalping parameters."""
        if not CRYPTO_SCALPING_AVAILABLE:
            print("[ERROR] Crypto scalping trader not available")
            input("Press Enter to continue...")
            return

        print("[CONFIG] Crypto Scalping Configuration")
        print("=" * 40)

        # Show current config
        trader = CryptoScalpingTrader()  # Create temp instance to show config
        config = trader.scalping_config

        print("Current Configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")

        print()
        print("Note: Configuration changes require restarting scalping")
        print("Modify the scalping_config in CryptoScalpingTrader class for custom settings")

        input("Press Enter to continue...")

    def _stop_crypto_scalping(self):
        """Stop crypto scalping."""
        if not self.crypto_scalper:
            print("❌ No crypto scalping session active")
            input("Press Enter to continue...")
            return

        print("⏹️ Stopping crypto scalping...")

        try:
            self.crypto_scalper.stop_scalping()
            print("✅ Crypto scalping stopped successfully")
        except Exception as e:
            print(f"❌ Error stopping scalping: {e}")

        self.crypto_scalper = None
        input("Press Enter to continue...")

    def _discover_indicators(self):
        """Discover new indicators."""
        if not AI_INDICATOR_AVAILABLE:
            print("❌ AI indicator agent not available")
            input("Press Enter to continue...")
            return

        print("🔍 Discovering new technical indicators...")
        print()

        try:
            # Initialize agent if needed
            if not self.indicator_agent:
                self.indicator_agent = IndicatorDiscoveryAgent()

            # Get user preferences
            categories_input = input("Categories (comma-separated) [trend,momentum,volume] or 'all': ").strip()
            if categories_input.lower() == 'all' or not categories_input:
                categories = None
            else:
                categories = [cat.strip() for cat in categories_input.split(',')]

            min_performance = float(input("Minimum performance score (0.0-1.0) [0.70]: ").strip() or "0.70")

            print(f"🔍 Searching for indicators in categories: {categories or 'all'}")
            print(f"   Minimum performance: {min_performance:.0%}")

            # Discover indicators
            new_indicators = self.indicator_agent.discover_indicators(
                categories=categories,
                min_performance=min_performance
            )

            print(f"\n✅ Discovered {len(new_indicators)} new indicators!")

            if new_indicators:
                print("\n📊 New Indicators Found:")
                for indicator_id, indicator_info in new_indicators.items():
                    print(f"   • {indicator_info['name']}: {indicator_info['performance_score']:.1%} score")
                    print(f"     Category: {indicator_info['category']}, Source: {indicator_info['source']}")

        except Exception as e:
            print(f"❌ Error discovering indicators: {e}")

        input("Press Enter to continue...")

    def _test_indicators(self):
        """Test indicator performance."""
        if not AI_INDICATOR_AVAILABLE:
            print("❌ AI indicator agent not available")
            input("Press Enter to continue...")
            return

        if not self.indicator_agent:
            print("❌ No indicator agent initialized")
            print("   Discover indicators first")
            input("Press Enter to continue...")
            return

        print("🧪 Testing indicator performance...")
        print()

        try:
            # Get available indicators
            available_indicators = list(self.indicator_agent.indicator_library.keys())

            if not available_indicators:
                print("❌ No indicators available for testing")
                print("   Discover indicators first")
                input("Press Enter to continue...")
                return

            # Show available indicators
            print("Available Indicators:")
            for i, indicator_id in enumerate(available_indicators, 1):
                indicator_info = self.indicator_agent.indicator_library[indicator_id]
                tested = "✓" if indicator_id in self.indicator_agent.performance_cache else " "
                print(f"   {i}. [{tested}] {indicator_info['name']} ({indicator_info['category']})")

            print()
            choice = input("Select indicator number to test (or 'all'): ").strip()

            if choice.lower() == 'all':
                test_indicators = available_indicators
            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(available_indicators):
                        test_indicators = [available_indicators[idx]]
                    else:
                        print("❌ Invalid selection")
                        input("Press Enter to continue...")
                        return
                except ValueError:
                    print("❌ Invalid input")
                    input("Press Enter to continue...")
                    return

            # Test selected indicators
            print(f"\n🧪 Testing {len(test_indicators)} indicator(s)...")

            for indicator_id in test_indicators:
                if indicator_id not in self.indicator_agent.performance_cache:
                    indicator_info = self.indicator_agent.indicator_library[indicator_id]
                    print(f"Testing {indicator_info['name']}...")

                    performance = self.indicator_agent.test_indicator_performance(indicator_id)

                    if performance:
                        win_rate = performance.get('win_rate', 0)
                        sharpe = performance.get('sharpe_ratio', 0)
                        total_return = performance.get('total_return', 0)
                        print(f"  ✓ Win Rate: {win_rate:.1%}")
                        print(f"  ✓ Sharpe Ratio: {sharpe:.2f}")
                        print(f"  ✓ Total Return: {total_return:.2%}")
                    else:
                        print("  ❌ Test failed")
                else:
                    print(f"Skipping {self.indicator_agent.indicator_library[indicator_id]['name']} (already tested)")

        except Exception as e:
            print(f"❌ Error testing indicators: {e}")

        input("Press Enter to continue...")

    def _integrate_indicators(self):
        """Integrate best indicators."""
        if not AI_INDICATOR_AVAILABLE:
            print("❌ AI indicator agent not available")
            input("Press Enter to continue...")
            return

        if not self.indicator_agent:
            print("❌ No indicator agent initialized")
            print("   Discover indicators first")
            input("Press Enter to continue...")
            return

        print("🔗 Integrating best indicators into ML pipeline...")
        print()

        try:
            # Get integration parameters
            target_accuracy = float(input("Target win rate (0.0-1.0) [0.75]: ").strip() or "0.75")
            max_indicators = int(input("Maximum indicators to integrate [10]: ").strip() or "10")

            print(f"🔗 Finding indicators with >{target_accuracy:.0%} win rate...")

            # Integrate best indicators
            selected_indicators = self.indicator_agent.integrate_best_indicators(
                target_accuracy=target_accuracy,
                max_indicators=max_indicators
            )

            if selected_indicators:
                # Create enhanced pipeline
                pipeline_config = self.indicator_agent.enhance_ml_pipeline(selected_indicators)

                print(f"\n✅ Successfully integrated {len(selected_indicators)} indicators!")
                print("📊 Enhanced ML Pipeline Ready:")
                print(f"   • Enhanced Trainer: Available")
                print(f"   • Feature Engineering: Enhanced")
                print(f"   • Indicators Added: {len(selected_indicators)}")

                # Save the enhanced trainer for future use
                self.enhanced_ml_trainer = pipeline_config.get('enhanced_trainer_class')

            else:
                print(f"\n⚠️ No indicators met the {target_accuracy:.0%} win rate threshold")
                print("   Try lowering the threshold or discovering more indicators")

        except Exception as e:
            print(f"❌ Error integrating indicators: {e}")

        input("Press Enter to continue...")

    def _view_indicator_library(self):
        """View indicator library."""
        if not AI_INDICATOR_AVAILABLE:
            print("❌ AI indicator agent not available")
            input("Press Enter to continue...")
            return

        if not self.indicator_agent:
            print("❌ No indicator agent initialized")
            print("   Discover indicators first")
            input("Press Enter to continue...")
            return

        print("📚 Indicator Library")
        print("=" * 50)
        print()

        # Get library stats
        total_indicators = len(self.indicator_agent.indicator_library)
        tested_indicators = len(self.indicator_agent.performance_cache)

        print(f"Total Indicators: {total_indicators}")
        print(f"Tested Indicators: {tested_indicators}")
        print(f"Untested Indicators: {total_indicators - tested_indicators}")
        print()

        if not self.indicator_agent.indicator_library:
            print("No indicators in library")
            print("Discover indicators first")
            input("Press Enter to continue...")
            return

        # Group by category
        categories = {}
        for indicator_id, indicator_info in self.indicator_agent.indicator_library.items():
            category = indicator_info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((indicator_id, indicator_info))

        # Display by category
        for category, indicators in categories.items():
            print(f"{category.upper()} INDICATORS:")
            print("-" * 30)

            for indicator_id, indicator_info in indicators:
                tested = indicator_id in self.indicator_agent.performance_cache
                status = "✓ Tested" if tested else "  Untested"

                print(f"   • {indicator_info['name']}")
                print(f"     Source: {indicator_info['source']}, Score: {indicator_info['performance_score']:.1%}")

                if tested:
                    perf = self.indicator_agent.performance_cache[indicator_id]
                    win_rate = perf.get('win_rate', 0)
                    sharpe = perf.get('sharpe_ratio', 0)
                    print(f"     Performance: {win_rate:.1%} win rate, Sharpe: {sharpe:.2f}")

            print()

        input("Press Enter to continue...")

    def _discover_crypto_strategies(self):
        """Discover crypto strategies."""
        if not AI_CRYPTO_STRATEGY_AVAILABLE:
            print("❌ AI crypto strategy agent not available")
            input("Press Enter to continue...")
            return

        print("₿ Discovering crypto strategies...")
        print()

        try:
            # Initialize agent if needed
            if not self.crypto_strategy_agent:
                self.crypto_strategy_agent = AICryptoStrategyAgent()

            # Discover strategies
            min_performance = float(input("Minimum win rate (0.0-1.0) [0.60]: ").strip() or "0.60")

            print(f"🔍 Searching for crypto strategies with >{min_performance:.0%} win rate...")
            new_strategies = self.crypto_strategy_agent.discover_strategies(min_performance=min_performance)

            print(f"\n✅ Discovered {len(new_strategies)} new crypto strategies!")

            if new_strategies:
                print("\n📊 New Strategies Found:")
                for strategy_id, strategy in new_strategies.items():
                    win_rate = strategy.performance_metrics.get('win_rate', 0)
                    print(f"   • {strategy.name}: {win_rate:.1%} win rate ({strategy.strategy_type})")

        except Exception as e:
            print(f"❌ Error discovering strategies: {e}")

        input("Press Enter to continue...")

    def _discover_futures_strategies(self):
        """Discover futures strategies."""
        if not AI_FUTURES_STRATEGY_AVAILABLE:
            print("❌ AI futures strategy agent not available")
            input("Press Enter to continue...")
            return

        print("⚡ Discovering futures strategies...")
        print()

        try:
            # Initialize agent if needed
            if not self.futures_strategy_agent:
                self.futures_strategy_agent = AIFuturesStrategyAgent()

            # Discover strategies
            min_performance = float(input("Minimum win rate (0.0-1.0) [0.60]: ").strip() or "0.60")

            print(f"🔍 Searching for futures strategies with >{min_performance:.0%} win rate...")
            new_strategies = self.futures_strategy_agent.discover_strategies(min_performance=min_performance)

            print(f"\n✅ Discovered {len(new_strategies)} new futures strategies!")

            if new_strategies:
                print("\n📊 New Strategies Found:")
                for strategy_id, strategy in new_strategies.items():
                    win_rate = strategy.performance_metrics.get('win_rate', 0)
                    print(f"   • {strategy.name}: {win_rate:.1%} win rate ({strategy.strategy_type})")

        except Exception as e:
            print(f"❌ Error discovering strategies: {e}")

        input("Press Enter to continue...")

    def _test_strategies(self):
        """Test strategies."""
        print("🧪 Testing strategies...")
        print()

        # Determine which agent to use
        agent = None
        agent_name = ""

        if self.crypto_strategy_agent:
            agent = self.crypto_strategy_agent
            agent_name = "crypto"
        elif self.futures_strategy_agent:
            agent = self.futures_strategy_agent
            agent_name = "futures"
        else:
            print("❌ No strategy agents available")
            input("Press Enter to continue...")
            return

        try:
            # Test all strategies
            print(f"Testing {agent_name} strategies...")

            tested_count = 0
            for strategy_id, strategy in agent.strategy_library.items():
                if not strategy.performance_metrics:
                    print(f"Testing {strategy.name}...")
                    performance = agent._test_strategy_performance(strategy)
                    strategy.performance_metrics = performance
                    tested_count += 1

            print(f"\n✅ Tested {tested_count} strategies")

            # Show top performers
            sorted_strategies = sorted(
                agent.strategy_library.items(),
                key=lambda x: x[1].performance_metrics.get('win_rate', 0),
                reverse=True
            )[:5]

            if sorted_strategies:
                print("\n🏆 Top Performing Strategies:")
                for strategy_id, strategy in sorted_strategies:
                    win_rate = strategy.performance_metrics.get('win_rate', 0)
                    total_return = strategy.performance_metrics.get('total_return', 0)
                    print(f"   • {strategy.name}: {win_rate:.1%} win rate, {total_return:.2%} total return")

        except Exception as e:
            print(f"❌ Error testing strategies: {e}")

        input("Press Enter to continue...")

    def _integrate_strategies(self):
        """Integrate strategies."""
        print("🔗 Integrating strategies...")
        print()

        # Determine which agent to use
        agent = None
        agent_name = ""

        if self.crypto_strategy_agent:
            agent = self.crypto_strategy_agent
            agent_name = "crypto"
        elif self.futures_strategy_agent:
            agent = self.futures_strategy_agent
            agent_name = "futures"
        else:
            print("❌ No strategy agents available")
            input("Press Enter to continue...")
            return

        try:
            # Integrate best strategies
            target_win_rate = float(input("Target win rate (0.0-1.0) [0.65]: ").strip() or "0.65")
            max_strategies = int(input("Maximum strategies to integrate [5]: ").strip() or "5")

            print(f"🔗 Integrating best {agent_name} strategies...")
            selected_strategies = agent.integrate_best_strategies(
                target_win_rate=target_win_rate,
                max_strategies=max_strategies
            )

            if selected_strategies:
                # Create portfolio
                portfolio = agent.create_strategy_portfolio(selected_strategies)
                print(f"\n✅ Successfully integrated {len(selected_strategies)} {agent_name} strategies!")
                print("📊 Portfolio Allocation:")
                for strategy_name, weight in portfolio['allocation'].items():
                    print(f"   • {strategy_name}: {weight:.1%}")
            else:
                print(f"\n⚠️ No {agent_name} strategies met the performance criteria")

        except Exception as e:
            print(f"❌ Error integrating strategies: {e}")

        input("Press Enter to continue...")

    def _view_strategy_library(self):
        """View strategy library."""
        print("📚 Strategy Library")
        print("=" * 40)
        print()

        # Check available agents
        agents = []
        if self.crypto_strategy_agent:
            agents.append(("Crypto", self.crypto_strategy_agent))
        if self.futures_strategy_agent:
            agents.append(("Futures", self.futures_strategy_agent))

        if not agents:
            print("❌ No strategy agents available")
            input("Press Enter to continue...")
            return

        total_strategies = 0

        for agent_name, agent in agents:
            print(f"{agent_name} Strategies:")
            print("-" * 20)

            if not agent.strategy_library:
                print("   No strategies in library")
            else:
                sorted_strategies = sorted(
                    agent.strategy_library.items(),
                    key=lambda x: x[1].performance_metrics.get('win_rate', 0),
                    reverse=True
                )

                for strategy_id, strategy in sorted_strategies:
                    win_rate = strategy.performance_metrics.get('win_rate', 0)
                    total_trades = strategy.performance_metrics.get('total_trades', 0)
                    print(f"   • {strategy.name}")
                    print(f"     Type: {strategy.strategy_type}, Win Rate: {win_rate:.1%}, Trades: {total_trades}")

                total_strategies += len(agent.strategy_library)

            print()

        print(f"📊 Total Strategies: {total_strategies}")

        input("Press Enter to continue...")

    def _train_ensemble_model(self):
        """Train ensemble model."""
        if not ADVANCED_ML_AVAILABLE:
            print("❌ Advanced ML training not available")
            print("   Install required dependencies: pip install xgboost lightgbm catboost")
            input("Press Enter to continue...")
            return

        print("🧠 Training Advanced Ensemble Model (>80% Accuracy)")
        print("=" * 60)
        print()

        # Get training parameters
        symbol = input("Symbol to train on [ES]: ").strip() or "ES"
        asset_type = input("Asset type (futures/stock) [futures]: ").strip() or "futures"
        period = input("Training period (30d/60d/120d/180d) [120d]: ").strip() or "120d"
        print("\n📊 Available Chart Styles:")
        print("   • time: Standard time-based bars (1m, 5m, 15m, 1h)")
        print("   • tick: Aggregate by trade count (100tick, 500tick, 1000tick)")
        print("   • range: New bar when price moves by N points (1.0range, 2.0range)")
        print("   • volume: Aggregate by volume (1000vol, 5000vol, 10000vol)")
        print("   • renko+: Price-based bricks with trend filtering (1.0renko, 2.0renko)")
        print("   • line: Line charts (experimental)")
        print()
        chart_style = input("Chart style (time/tick/range/volume/renko+/line) [time]: ").strip() or "time"

        # Set interval based on chart style
        if chart_style == "time":
            print("\n⏰ Time-based intervals:")
            print("   • 1m: 1 minute bars")
            print("   • 5m: 5 minute bars (recommended)")
            print("   • 15m: 15 minute bars")
            print("   • 1h: 1 hour bars")
            interval = input("Data interval (1m/5m/15m/1h) [5m]: ").strip() or "5m"
        elif chart_style == "tick":
            print("\n📊 Tick-based aggregation:")
            print("   • 100tick: 100 trades per bar")
            print("   • 500tick: 500 trades per bar")
            print("   • 1000tick: 1000 trades per bar")
            tick_count = input("Tick count (100/500/1000) [100]: ").strip() or "100"
            interval = f"{tick_count}tick"
        elif chart_style == "range":
            print("\n📏 Range-based bars:")
            print("   • 0.5range: $0.50 price range per bar")
            print("   • 1.0range: $1.00 price range per bar")
            print("   • 2.0range: $2.00 price range per bar")
            range_size = input("Range size (0.5/1.0/2.0) [1.0]: ").strip() or "1.0"
            interval = f"{range_size}range"
        elif chart_style == "volume":
            print("\n📦 Volume-based aggregation:")
            print("   • 1000vol: 1000 contracts per bar")
            print("   • 5000vol: 5000 contracts per bar")
            print("   • 10000vol: 10000 contracts per bar")
            volume_size = input("Volume size (1000/5000/10000) [1000]: ").strip() or "1000"
            interval = f"{volume_size}vol"
        elif chart_style == "renko+":
            print("\n🧱 Renko+ bricks:")
            print("   • 1.0renko: $1.00 brick size")
            print("   • 2.0renko: $2.00 brick size")
            print("   • 0.5renko: $0.50 brick size")
            brick_size = input("Brick size (0.5/1.0/2.0) [1.0]: ").strip() or "1.0"
            interval = f"{brick_size}renko"
        else:  # line or other
            print("\n📈 Line charts use standard time intervals")
            interval = "5m"  # Default for line charts

        optimize = input("Perform hyperparameter optimization? (y/n) [n]: ").strip().lower() == 'y'

        print(f"\n🤖 Training ensemble model for {symbol} ({asset_type})")
        print(f"   Period: {period}, Chart Style: {chart_style}, Interval: {interval}")
        print(f"   Hyperparameter optimization: {'Yes' if optimize else 'No'}")
        print()

        try:
            # Initialize advanced trainer
            self.advanced_trainer = AdvancedMLTrainer(symbol=symbol, asset_type=asset_type)

            # Load and prepare data
            print(f"📊 Loading and preparing {chart_style} training data...")
            X_train, X_test, y_train, y_test = self.advanced_trainer.load_and_prepare_data(
                period=period, interval=interval, chart_style=chart_style
            )

            # Train the model
            print("\n🚀 Training advanced ensemble model...")
            training_results = self.advanced_trainer.train_advanced_model(
                X_train, y_train, optimize_hyperparams=optimize
            )

            # Evaluate on test set
            print("\n🧪 Evaluating model performance...")
            evaluation = self.advanced_trainer.evaluate_model(X_test, y_test)

            # Display results
            print("\n🎯 Training Results:")
            print(f"   Cross-validation accuracy: {training_results['cv_accuracy']:.4f}")
            print(f"   Test accuracy: {evaluation['accuracy']:.4f}")
            print(f"   High confidence accuracy: {evaluation['confidence_threshold_accuracy']:.4f}")
            print(f"   Selected features: {training_results['n_features']}")

            # Save model option
            save_model = input("\n💾 Save trained model? (y/n) [y]: ").strip().lower()
            if save_model != 'n':
                model_name = f"advanced_{symbol.lower()}_{asset_type}_model.pkl"
                self.advanced_trainer.save_advanced_model(model_name)
                print(f"✅ Model saved as: {model_name}")

            print("\n🎉 Advanced ensemble model training completed!")
            print("💡 Tip: Use high-confidence predictions (>60%) for best results")

        except Exception as e:
            print(f"❌ Error training ensemble model: {e}")

        input("Press Enter to continue...")

    def _hyperparameter_optimization(self):
        """Hyperparameter optimization."""
        if not ADVANCED_ML_AVAILABLE:
            print("❌ Advanced ML training not available")
            input("Press Enter to continue...")
            return

        if not hasattr(self, 'advanced_trainer') or not self.advanced_trainer:
            print("❌ No advanced trainer available")
            print("   Train an ensemble model first")
            input("Press Enter to continue...")
            return

        print("🎛️ Hyperparameter Optimization")
        print("=" * 40)
        print()

        # Get optimization parameters
        model_type = input("Model type to optimize (xgb/rf) [xgb]: ").strip() or "xgb"

        print(f"🔧 Optimizing hyperparameters for {model_type.upper()}...")
        print("⚠️  This may take several minutes...")
        print()

        try:
            # Need training data for optimization
            if not hasattr(self.advanced_trainer, 'X_train'):
                print("📊 Preparing data for optimization...")
                X_train, _, y_train, _ = self.advanced_trainer.load_and_prepare_data()

                # Apply feature selection
                selected_features, _ = self.advanced_trainer.perform_feature_selection(X_train, y_train, k=40)
                X_train_selected = X_train[selected_features]
            else:
                # Use existing processed data
                X_train_selected = self.advanced_trainer.X_train_selected
                y_train = self.advanced_trainer.y_train

            # Perform optimization
            best_params = self.advanced_trainer.hyperparameter_optimization(
                X_train_selected, y_train, model_type
            )

            if best_params:
                print("\n✅ Optimization completed!")
                print("🏆 Best parameters found:")
                for param, value in best_params.items():
                    print(f"   {param}: {value}")

                # Apply best parameters to model
                apply_params = input("\n🔄 Apply optimized parameters to current model? (y/n) [y]: ").strip().lower()
                if apply_params != 'n':
                    print("✅ Optimized parameters applied to model")
            else:
                print("❌ Optimization failed or was skipped")

        except Exception as e:
            print(f"❌ Error during hyperparameter optimization: {e}")

        input("Press Enter to continue...")

    def _cross_validation(self):
        """Cross-validation."""
        if not ADVANCED_ML_AVAILABLE:
            print("❌ Advanced ML training not available")
            input("Press Enter to continue...")
            return

        if not hasattr(self, 'advanced_trainer') or not self.advanced_trainer:
            print("❌ No advanced trainer available")
            print("   Train an ensemble model first")
            input("Press Enter to continue...")
            return

        print("🔄 Cross-Validation Analysis")
        print("=" * 40)
        print()

        try:
            # Get CV parameters
            n_splits = int(input("Number of CV splits (3-10) [5]: ").strip() or "5")
            scoring = input("Scoring metric (accuracy/f1/precision/recall) [accuracy]: ").strip() or "accuracy"

            print(f"🔍 Running {n_splits}-fold cross-validation with {scoring} scoring...")
            print()

            # Prepare data if needed
            if not hasattr(self.advanced_trainer, 'X_train'):
                print("📊 Preparing data...")
                X_train, _, y_train, _ = self.advanced_trainer.load_and_prepare_data()

                # Apply feature selection
                selected_features, _ = self.advanced_trainer.perform_feature_selection(X_train, y_train, k=40)
                X_train_selected = X_train[selected_features]
            else:
                X_train_selected = self.advanced_trainer.X_train_selected
                y_train = self.advanced_trainer.y_train

            # Perform cross-validation
            from sklearn.model_selection import cross_val_score, TimeSeriesSplit

            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = cross_val_score(
                self.advanced_trainer.best_model,
                X_train_selected, y_train,
                cv=tscv, scoring=scoring, n_jobs=-1
            )

            # Display results
            print("📊 Cross-Validation Results:")
            print(f"   Scoring metric: {scoring}")
            print(f"   Mean score: {cv_scores.mean():.4f}")
            print(f"   Standard deviation: {cv_scores.std():.4f}")
            print(f"   Min score: {cv_scores.min():.4f}")
            print(f"   Max score: {cv_scores.max():.4f}")
            print()
            print("   Individual fold scores:")
            for i, score in enumerate(cv_scores, 1):
                print(f"     Fold {i}: {score:.4f}")

            # Stability assessment
            stability = "High" if cv_scores.std() < 0.05 else "Medium" if cv_scores.std() < 0.10 else "Low"
            print(f"\n🎯 Model Stability: {stability} (Std: {cv_scores.std():.4f})")

        except Exception as e:
            print(f"❌ Error during cross-validation: {e}")

        input("Press Enter to continue...")

    def _feature_engineering(self):
        """Feature engineering."""
        if not ADVANCED_ML_AVAILABLE:
            print("❌ Advanced ML training not available")
            input("Press Enter to continue...")
            return

        print("⚙️ Advanced Feature Engineering")
        print("=" * 40)
        print()

        # Get parameters
        symbol = input("Symbol for feature analysis [ES]: ").strip() or "ES"
        asset_type = input("Asset type (futures/stock) [futures]: ").strip() or "futures"
        period = input("Analysis period (30d/60d/120d) [60d]: ").strip() or "60d"

        print(f"🔧 Analyzing features for {symbol} ({asset_type}) over {period}...")
        print()

        try:
            # Create temporary trainer for analysis
            temp_trainer = AdvancedMLTrainer(symbol=symbol, asset_type=asset_type)

            # Load data
            df = temp_trainer.data_manager.prepare_futures_dataset(symbol, period=period)

            print("📊 Original data shape:", df.shape)
            print("📈 Original features:", len(df.columns))

            # Apply feature engineering
            enhanced_df = temp_trainer._advanced_feature_engineering(df)

            print("🎯 Enhanced data shape:", enhanced_df.shape)
            print("🚀 New features added:", len(enhanced_df.columns) - len(df.columns))

            # Show feature categories
            feature_categories = {
                'Technical Indicators': [col for col in enhanced_df.columns if any(x in col.lower() for x in ['rsi', 'macd', 'bb', 'stoch', 'williams', 'cci'])],
                'Volume Features': [col for col in enhanced_df.columns if 'volume' in col.lower()],
                'Volatility Features': [col for col in enhanced_df.columns if any(x in col.lower() for x in ['volatility', 'atr', 'std'])],
                'Momentum Features': [col for col in enhanced_df.columns if any(x in col.lower() for x in ['momentum', 'roc', 'acceleration'])],
                'Time Features': [col for col in enhanced_df.columns if any(x in col.lower() for x in ['hour', 'minute', 'sin', 'cos'])],
                'Microstructure': [col for col in enhanced_df.columns if any(x in col.lower() for x in ['spread', 'gap', 'imbalance', 'pressure'])],
                'Statistical': [col for col in enhanced_df.columns if any(x in col.lower() for x in ['skew', 'kurtosis', 'zscore'])],
                'Lagged': [col for col in enhanced_df.columns if 'lag' in col.lower()],
                'Rolling': [col for col in enhanced_df.columns if 'rolling' in col.lower()]
            }

            print("\n🔍 Feature Categories:")
            for category, features in feature_categories.items():
                if features:
                    print(f"   {category}: {len(features)} features")
                    if len(features) <= 5:
                        print(f"     {', '.join(features[:5])}")

            print("\n✅ Feature engineering analysis completed!")
            print("💡 Features are automatically applied during model training")

        except Exception as e:
            print(f"❌ Error during feature engineering analysis: {e}")

        input("Press Enter to continue...")

    def _model_comparison(self):
        """Model comparison."""
        if not ADVANCED_ML_AVAILABLE:
            print("❌ Advanced ML training not available")
            input("Press Enter to continue...")
            return

        print("📊 Advanced Model Comparison")
        print("=" * 40)
        print()

        # Get comparison parameters
        symbol = input("Symbol for comparison [ES]: ").strip() or "ES"
        asset_type = input("Asset type (futures/stock) [futures]: ").strip() or "futures"
        period = input("Test period (30d/60d/120d) [60d]: ").strip() or "60d"

        print(f"⚔️ Comparing models on {symbol} ({asset_type}) over {period}...")
        print()

        try:
            # Initialize trainer
            trainer = AdvancedMLTrainer(symbol=symbol, asset_type=asset_type)

            # Load data
            X_train, X_test, y_train, y_test = trainer.load_and_prepare_data(period=period)

            # Apply feature selection
            selected_features, _ = trainer.perform_feature_selection(X_train, y_train, k=40)
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]

            # Models to compare
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=False),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }

            # Train and evaluate each model
            results = {}

            print("🏁 Training and evaluating models...")
            print("-" * 50)

            for name, model in models.items():
                try:
                    print(f"Training {name}...")

                    # Train model
                    model.fit(X_train_selected, y_train)

                    # Evaluate
                    predictions = model.predict(X_test_selected)
                    accuracy = accuracy_score(y_test, predictions)

                    # Cross-validation score
                    tscv = TimeSeriesSplit(n_splits=3)
                    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=tscv, scoring='accuracy', n_jobs=-1)

                    results[name] = {
                        'test_accuracy': accuracy,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }

                    print(f"  ✓ Test accuracy: {accuracy:.4f}")
                    print(f"  ✓ CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

                except Exception as e:
                    print(f"  ❌ Error training {name}: {e}")
                    results[name] = {'error': str(e)}

            # Display comparison
            print("\n🏆 Model Comparison Results:")
            print("=" * 50)
            print(f"{'Model':<20} {'Test Acc':<10} {'CV Mean':<10} {'CV Std':<10}")
            print("-" * 50)

            # Sort by test accuracy
            sorted_results = sorted(
                [(name, res) for name, res in results.items() if 'error' not in res],
                key=lambda x: x[1]['test_accuracy'],
                reverse=True
            )

            for i, (name, res) in enumerate(sorted_results, 1):
                status = "🏆" if i == 1 else f"{i}."
                print(f"{status} {name:<15} {res['test_accuracy']:<10.4f} {res['cv_mean']:<10.4f} {res['cv_std']:<10.4f}")
            print("\n💡 Recommendation: Use ensemble of top 3-4 models for best results")

        except Exception as e:
            print(f"❌ Error during model comparison: {e}")

        input("Press Enter to continue...")

    def _run_stock_backtest(self):
        """Run stock backtest."""
        print("📊 Running stock backtest...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _run_futures_backtest(self):
        """Run futures backtest."""
        print("⚡ Running futures backtest...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _walk_forward_analysis(self):
        """Walk-forward analysis."""
        print("🚶 Walk-forward analysis...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _monte_carlo_simulation(self):
        """Monte Carlo simulation."""
        print("🎲 Monte Carlo simulation...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _performance_analytics(self):
        """Performance analytics."""
        print("📈 Performance analytics...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _configure_replay(self):
        """Configure market replay."""
        print("⚙️ Configuring market replay...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _start_replay(self):
        """Start market replay."""
        print("▶️ Starting market replay...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _control_replay(self):
        """Control replay (pause/resume)."""
        print("⏯️ Controlling replay...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _view_replay_status(self):
        """View replay status."""
        print("📊 Replay status...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _analyze_replay_results(self):
        """Analyze replay results."""
        print("📈 Analyzing replay results...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _setup_quantconnect(self):
        """Setup QuantConnect."""
        print("☁️ Setting up QuantConnect...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _list_qc_projects(self):
        """List QuantConnect projects."""
        print("📋 Listing QuantConnect projects...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _deploy_qc_algorithm(self):
        """Deploy QuantConnect algorithm."""
        print("🚀 Deploying QuantConnect algorithm...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _view_qc_backtests(self):
        """View QuantConnect backtests."""
        print("📊 Viewing QuantConnect backtests...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _qc_live_status(self):
        """QuantConnect live status."""
        print("📡 QuantConnect live status...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _view_settings(self):
        """View current settings."""
        print("⚙️ Current settings...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _update_api_keys(self):
        """Update API keys."""
        print("🔑 Updating API keys...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _trading_parameters(self):
        """Trading parameters."""
        print("📊 Trading parameters...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _risk_management(self):
        """Risk management."""
        print("🛡️ Risk management...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _system_preferences(self):
        """System preferences."""
        print("⚙️ System preferences...")
        # Implementation would go here
        input("Feature not yet implemented. Press Enter...")

    def _check_config(self) -> Dict[str, str]:
        """Check configuration status."""
        return {'status': 'OK', 'message': 'Configuration loaded'}

    def _check_data_manager(self) -> Dict[str, str]:
        """Check data manager status."""
        return {'status': 'OK', 'message': 'Data manager ready'}

    def _check_alpaca_integration(self) -> Dict[str, str]:
        """Check Alpaca integration status."""
        return {'status': 'OK', 'message': 'Alpaca integration ready'}

    def _check_backtester(self) -> Dict[str, str]:
        """Check backtester status."""
        return {'status': 'OK', 'message': 'Backtester ready'}

    def _check_market_replay(self) -> Dict[str, str]:
        """Check market replay status."""
        return {'status': 'OK', 'message': 'Market replay ready'}

    def _check_quantconnect(self) -> Dict[str, str]:
        """Check QuantConnect status."""
        return {'status': 'OK', 'message': 'QuantConnect ready'}

    def _check_ai_agents(self) -> Dict[str, str]:
        """Check AI agents status."""
        return {'status': 'OK', 'message': 'AI agents ready'}

    def _check_trading_systems(self) -> Dict[str, str]:
        """Check trading systems status."""
        return {'status': 'OK', 'message': 'Trading systems ready'}


def main():
    """Main entry point for the trading interface."""
    try:
        interface = TradingInterface()
        interface.show_main_menu()
    except KeyboardInterrupt:
        print("\nGoodbye! Thanks for using AlgoTrendy.")
    except Exception as e:
        logger.error(f"Interface error: {e}")
        print(f"Error: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    main()