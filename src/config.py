"""
AlgoTrendy - XGBoost Trading Strategy Discovery
Main configuration and setup for the trading system.
"""

import os
from pathlib import Path
import logging
from typing import Dict, Any
from dataclasses import dataclass

# Project root directory
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

@dataclass
class TradingConfig:
    """Configuration class for trading parameters"""

    # Asset type
    asset_type: str = "stock"  # "stock" or "futures"

    # Data settings
    symbols: list = None
    futures_symbols: list = None
    timeframes: list = None
    futures_timeframes: list = None
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"

    # Model settings
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42

    # XGBoost parameters
    xgb_params: Dict[str, Any] = None
    futures_xgb_params: Dict[str, Any] = None

    # Trading parameters
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1% for stocks
    futures_commission: float = 0.0005  # 0.05% for futures (lower due to volume)
    slippage: float = 0.0005   # 0.05%
    max_positions: int = 5

    # Futures-specific parameters
    futures_leverage: float = 5.0  # Default leverage for futures
    contract_multiplier: int = 50   # Default contract multiplier (ES = 50)
    margin_initial: float = 1320    # Initial margin per contract (ES example)
    margin_maintenance: float = 1200  # Maintenance margin per contract

    # Risk management
    max_position_size: float = 0.2  # 20% of portfolio for stocks
    futures_max_position_size: float = 0.1  # 10% of portfolio for futures (due to leverage)
    stop_loss: float = 0.02         # 2% for stocks
    futures_stop_loss: float = 0.01  # 1% for futures (tighter stops)
    take_profit: float = 0.06       # 6% for stocks
    futures_take_profit: float = 0.02  # 2% for futures (quicker profits)

    # Day trading parameters
    day_trading_enabled: bool = False
    max_daily_trades: int = 10
    max_daily_loss: float = 0.05  # 5% max daily loss
    daily_profit_target: float = 0.03  # 3% daily profit target

    # QuantConnect integration
    quantconnect_enabled: bool = False
    qc_user_id: str = None
    qc_api_token: str = None
    qc_project_name: str = "AlgoTrendy Futures"
    qc_server_type: str = "LIVE"  # LIVE or PAPER

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

        if self.futures_symbols is None:
            self.futures_symbols = ["ES", "NQ", "RTY", "CL", "GC"]  # Popular futures contracts

        if self.timeframes is None:
            self.timeframes = ["1h", "4h", "1d"]

        if self.futures_timeframes is None:
            self.futures_timeframes = ["5m", "15m", "30m", "1h"]  # Intraday for day trading

        if self.xgb_params is None:
            self.xgb_params = {
                'objective': 'binary:logistic',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': 0
            }

        if self.futures_xgb_params is None:
            # Optimized parameters for futures (more complex due to higher frequency data)
            self.futures_xgb_params = {
                'objective': 'binary:logistic',
                'max_depth': 8,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': 0,
                'early_stopping_rounds': 20
            }

# Global configuration instance
CONFIG = TradingConfig()

# Logging setup
def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Setup file handler
    file_handler = logging.FileHandler(LOGS_DIR / "algotrendy.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logging
logger = setup_logging()
logger.info("AlgoTrendy initialized successfully")