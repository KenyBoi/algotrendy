"""
Backtesting engine for evaluating trading strategies.
Includes portfolio management, risk controls, and performance analytics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG, RESULTS_DIR
from data_manager import FUTURES_CONTRACTS

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Single trade record"""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int  # For futures, this is number of contracts
    side: str  # 'long' or 'short'
    asset_type: str = "stock"  # "stock" or "futures"
    contract_multiplier: int = 1  # Futures contract multiplier (50 for ES, etc.)
    pnl: Optional[float] = None
    return_pct: Optional[float] = None

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float

class Backtester:
    """Comprehensive backtesting engine for stocks and futures"""

    def __init__(self, initial_capital: float = 100000.0,
                 commission: float = 0.001, slippage: float = 0.0005,
                 asset_type: str = "stock"):
        """
        Initialize backtester

        Args:
            initial_capital: Starting capital
            commission: Commission rate (e.g., 0.001 = 0.1% for stocks, different for futures)
            slippage: Slippage rate (e.g., 0.0005 = 0.05%)
            asset_type: "stock" or "futures"
        """
        self.initial_capital = initial_capital
        self.asset_type = asset_type

        # Commission rates differ for stocks vs futures
        if asset_type == "futures":
            self.commission = 0.0005  # 0.05% for futures (lower than stocks)
            self.futures_leverage = CONFIG.futures_leverage
        else:
            self.commission = commission

        self.slippage = slippage

        # Portfolio tracking
        self.portfolio_value = []
        self.cash = initial_capital
        self.positions = {}  # symbol -> {'quantity': int, 'contract_multiplier': int, 'entry_price': float}
        self.trades = []
        self.daily_returns = []

        # Performance tracking
        self.equity_curve = pd.Series(dtype=float)
        self.drawdown_curve = pd.Series(dtype=float)

        # Futures-specific tracking
        self.margin_used = 0.0  # Track margin requirements for futures
        
    def reset(self):
        """Reset backtester state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_value = []
        self.daily_returns = []
        self.equity_curve = pd.Series(dtype=float)
        self.drawdown_curve = pd.Series(dtype=float)
        self.margin_used = 0.0
    
    def calculate_position_size(self, price: float, signal_strength: float = 1.0,
                               contract_multiplier: int = 1) -> int:
        """
        Calculate position size based on available capital and risk management

        Args:
            price: Current price
            signal_strength: Signal strength (0-1)
            contract_multiplier: Futures contract multiplier (1 for stocks)

        Returns:
            Number of shares/contracts to trade
        """
        if self.asset_type == "futures":
            # For futures, position sizing based on margin requirements
            # Get contract info (simplified - using ES as example)
            contract_symbol = "ES"  # This should be passed in or derived from symbol
            margin_per_contract = CONFIG.margin_initial  # $1,320 for ES

            # Calculate how many contracts we can afford
            available_margin = self.cash * CONFIG.futures_max_position_size * signal_strength
            max_contracts = int(available_margin / margin_per_contract)

            # Limit to reasonable number per position
            return min(max_contracts, 5)  # Max 5 contracts per position
        else:
            # Stock position sizing
            max_position_value = self.cash * CONFIG.max_position_size * signal_strength
            total_cost_per_share = price * (1 + self.commission + self.slippage)
            shares = int(max_position_value / total_cost_per_share)

            return max(0, shares)
    
    def enter_position(self, symbol: str, date: datetime, price: float,
                      signal: int, signal_strength: float = 1.0) -> bool:
        """
        Enter a new position for stocks or futures

        Args:
            symbol: Trading symbol
            date: Entry date
            price: Entry price
            signal: 1 for long, -1 for short, 0 for no action
            signal_strength: Signal confidence (0-1)

        Returns:
            True if position entered successfully
        """
        try:
            if signal == 0:
                return False

            # Get contract multiplier for futures
            contract_multiplier = 1
            if self.asset_type == "futures":
                # Extract contract symbol (e.g., 'ES' from 'ES=F')
                contract_symbol = symbol.replace('=F', '')[:2]
                if contract_symbol in FUTURES_CONTRACTS:
                    contract_multiplier = FUTURES_CONTRACTS[contract_symbol]['multiplier']

            # Calculate position size
            quantity = self.calculate_position_size(price, signal_strength, contract_multiplier)

            if quantity == 0:
                return False

            if self.asset_type == "futures":
                # For futures, only margin is required, not full contract value
                margin_per_contract = CONFIG.margin_initial
                total_margin_required = quantity * margin_per_contract

                if total_margin_required > self.cash:
                    return False

                # Deduct margin from cash
                self.cash -= total_margin_required
                self.margin_used += total_margin_required
            else:
                # For stocks, pay full amount
                cost = quantity * price
                total_cost = cost * (1 + self.commission + self.slippage)

                if total_cost > self.cash:
                    return False

                self.cash -= total_cost

            # Update positions
            if signal == 1:  # Long
                side = 'long'
            else:  # Short
                side = 'short'

            # Store position with additional info
            self.positions[symbol] = {
                'quantity': quantity if signal == 1 else -quantity,
                'contract_multiplier': contract_multiplier,
                'entry_price': price,
                'entry_date': date
            }

            # Record trade
            trade = Trade(
                symbol=symbol,
                entry_date=date,
                exit_date=None,
                entry_price=price,
                exit_price=None,
                quantity=abs(quantity),
                side=side,
                asset_type=self.asset_type,
                contract_multiplier=contract_multiplier
            )
            self.trades.append(trade)

            logger.debug(f"Entered {self.asset_type} {side} position: {symbol} x{quantity} @ ${price:.2f}")

            return True

        except Exception as e:
            logger.error(f"Error entering {self.asset_type} position: {e}")
            return False
    
    def exit_position(self, symbol: str, date: datetime, price: float,
                     exit_reason: str = "signal") -> bool:
        """
        Exit existing position for stocks or futures

        Args:
            symbol: Trading symbol
            date: Exit date
            price: Exit price
            exit_reason: Reason for exit

        Returns:
            True if position exited successfully
        """
        try:
            if symbol not in self.positions or self.positions[symbol]['quantity'] == 0:
                return False

            position_info = self.positions[symbol]
            quantity = abs(position_info['quantity'])
            contract_multiplier = position_info['contract_multiplier']
            entry_price = position_info['entry_price']
            side = 'long' if position_info['quantity'] > 0 else 'short'

            if self.asset_type == "futures":
                # For futures, P&L is based on contract value change
                price_change = price - entry_price if side == 'long' else entry_price - price
                contract_pnl = price_change * quantity * contract_multiplier

                # Subtract commissions (futures commissions are per contract)
                commission_cost = quantity * 2.5  # $2.50 per contract round trip (approximate)
                total_pnl = contract_pnl - commission_cost

                # Return margin to cash
                margin_returned = quantity * CONFIG.margin_initial
                self.cash += margin_returned + total_pnl
                self.margin_used -= margin_returned
            else:
                # Stock position exit
                proceeds = quantity * price
                net_proceeds = proceeds * (1 - self.commission - self.slippage)

                if side == 'long':
                    self.cash += net_proceeds
                else:  # short
                    self.cash += 2 * (quantity * entry_price) - net_proceeds  # Cover short

                # Calculate P&L for stocks
                if side == 'long':
                    total_pnl = (price - entry_price) * quantity
                else:
                    total_pnl = (entry_price - price) * quantity

                # Account for costs
                total_costs = (entry_price + price) * quantity * (self.commission + self.slippage)
                total_pnl -= total_costs

            # Close position
            self.positions[symbol]['quantity'] = 0

            # Update trade record
            for trade in reversed(self.trades):
                if (trade.symbol == symbol and trade.exit_date is None and
                    trade.side == side):
                    trade.exit_date = date
                    trade.exit_price = price

                    # Calculate P&L and return %
                    if self.asset_type == "futures":
                        # For futures, use the calculated P&L above
                        trade.pnl = total_pnl
                        trade.return_pct = total_pnl / (quantity * CONFIG.margin_initial)
                    else:
                        # Stock calculations
                        if side == 'long':
                            trade.pnl = (price - trade.entry_price) * trade.quantity
                            trade.return_pct = (price - trade.entry_price) / trade.entry_price
                        else:  # short
                            trade.pnl = (trade.entry_price - price) * trade.quantity
                            trade.return_pct = (trade.entry_price - price) / trade.entry_price

                        # Account for costs
                        total_costs = (trade.entry_price + price) * trade.quantity * (self.commission + self.slippage)
                        trade.pnl -= total_costs

                    break

            logger.debug(f"Exited {self.asset_type} {side} position: {symbol} x{quantity} @ ${price:.2f}")

            return True

        except Exception as e:
            logger.error(f"Error exiting {self.asset_type} position: {e}")
            return False
    
    def update_portfolio_value(self, date: datetime, prices: Dict[str, float]):
        """
        Update portfolio value based on current prices for stocks or futures

        Args:
            date: Current date
            prices: Dictionary of symbol -> current price
        """
        try:
            position_value = 0

            if self.asset_type == "futures":
                # For futures, calculate unrealized P&L
                for symbol, position_info in self.positions.items():
                    if symbol in prices and position_info['quantity'] != 0:
                        quantity = position_info['quantity']
                        entry_price = position_info['entry_price']
                        current_price = prices[symbol]
                        contract_multiplier = position_info['contract_multiplier']

                        # Calculate P&L per contract
                        if quantity > 0:  # Long position
                            pnl_per_contract = (current_price - entry_price) * contract_multiplier
                        else:  # Short position
                            pnl_per_contract = (entry_price - current_price) * contract_multiplier

                        position_value += pnl_per_contract * abs(quantity)
            else:
                # For stocks, calculate position values
                for symbol, quantity in self.positions.items():
                    if symbol in prices and quantity != 0:
                        position_value += abs(quantity) * prices[symbol]

            # Total portfolio value (cash + margin used back + unrealized P&L)
            if self.asset_type == "futures":
                total_value = self.cash + self.margin_used + position_value
            else:
                total_value = self.cash + position_value

            self.portfolio_value.append(total_value)

            # Calculate daily return
            if len(self.portfolio_value) > 1:
                daily_return = (total_value - self.portfolio_value[-2]) / self.portfolio_value[-2]
                self.daily_returns.append(daily_return)

            # Update equity curve
            self.equity_curve[date] = total_value

            # Calculate drawdown
            if len(self.equity_curve) > 0:
                peak = self.equity_curve.expanding().max()[date]
                drawdown = (total_value - peak) / peak
                self.drawdown_curve[date] = drawdown

        except Exception as e:
            logger.error(f"Error updating {self.asset_type} portfolio value: {e}")
    
    def run_backtest(self, data: pd.DataFrame, signals: pd.Series,
                    symbol: str = "STOCK") -> Dict:
        """
        Run complete backtest for stocks or futures

        Args:
            data: DataFrame with OHLCV data
            signals: Series with trading signals (1=buy, -1=sell, 0=hold)
            symbol: Trading symbol

        Returns:
            Dictionary with backtest results
        """
        try:
            logger.info(f"Running {self.asset_type} backtest for {symbol}...")

            self.reset()

            # Align data and signals
            data = data.copy()
            signals = signals.reindex(data.index, fill_value=0)

            current_position = 0

            for date, row in data.iterrows():
                current_price = row['close']
                signal = signals.get(date, 0)

                # Position management
                if current_position == 0 and signal != 0:
                    # Enter new position
                    if self.enter_position(symbol, date, current_price, signal):
                        current_position = signal

                elif current_position != 0 and signal == -current_position:
                    # Exit current position
                    if self.exit_position(symbol, date, current_price):
                        current_position = 0

                # Update portfolio value
                prices = {symbol: current_price}
                self.update_portfolio_value(date, prices)

            # Close any remaining positions
            if current_position != 0:
                last_date = data.index[-1]
                last_price = data['close'].iloc[-1]
                self.exit_position(symbol, last_date, last_price, "end_of_backtest")

            # Calculate performance metrics
            metrics = self.calculate_performance_metrics()

            results = {
                'metrics': metrics,
                'equity_curve': self.equity_curve,
                'drawdown_curve': self.drawdown_curve,
                'trades': self.trades,
                'final_value': self.portfolio_value[-1] if self.portfolio_value else self.initial_capital,
                'asset_type': self.asset_type
            }

            logger.info(f"{self.asset_type.title()} backtest completed. Final value: ${results['final_value']:,.2f}")

            return results

        except Exception as e:
            logger.error(f"Error running {self.asset_type} backtest: {e}")
            raise
    
    def calculate_performance_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            if not self.portfolio_value:
                return PortfolioMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            # Basic returns
            total_return = (self.portfolio_value[-1] - self.initial_capital) / self.initial_capital
            
            # Annualized return (assuming daily data)
            days = len(self.portfolio_value)
            annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
            
            # Volatility
            if len(self.daily_returns) > 1:
                volatility = np.std(self.daily_returns) * np.sqrt(252)
            else:
                volatility = 0
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            max_drawdown = self.drawdown_curve.min() if len(self.drawdown_curve) > 0 else 0
            
            # Trade statistics
            completed_trades = [t for t in self.trades if t.exit_date is not None]
            
            if completed_trades:
                winning_trades = [t for t in completed_trades if t.pnl > 0]
                losing_trades = [t for t in completed_trades if t.pnl <= 0]
                
                win_rate = len(winning_trades) / len(completed_trades)
                
                avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([abs(t.pnl) for t in losing_trades]) if losing_trades else 0
                
                profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
                avg_trade_return = np.mean([t.return_pct for t in completed_trades])
            else:
                win_rate = 0
                profit_factor = 0
                avg_trade_return = 0
            
            return PortfolioMetrics(
                total_return=total_return,
                annual_return=annual_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=len(completed_trades),
                avg_trade_return=avg_trade_return
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return PortfolioMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def plot_results(self, title: str = "Backtest Results"):
        """Plot backtest results"""
        try:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # Equity curve
            self.equity_curve.plot(ax=axes[0], title="Portfolio Value")
            axes[0].axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.7, label='Initial Capital')
            axes[0].set_ylabel('Portfolio Value ($)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Drawdown
            (self.drawdown_curve * 100).plot(ax=axes[1], color='red', title="Drawdown")
            axes[1].fill_between(self.drawdown_curve.index, 0, self.drawdown_curve * 100, color='red', alpha=0.3)
            axes[1].set_ylabel('Drawdown (%)')
            axes[1].grid(True, alpha=0.3)
            
            # Daily returns distribution
            if self.daily_returns:
                axes[2].hist(np.array(self.daily_returns) * 100, bins=50, alpha=0.7, edgecolor='black')
                axes[2].set_title("Daily Returns Distribution")
                axes[2].set_xlabel('Daily Return (%)')
                axes[2].set_ylabel('Frequency')
                axes[2].grid(True, alpha=0.3)
            
            plt.suptitle(title)
            plt.tight_layout()
            
            # Save plot
            plot_path = RESULTS_DIR / f"backtest_results.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Backtest plot saved to: {plot_path}")
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")

if __name__ == "__main__":
    # Example usage for both stocks and futures
    from data_manager import DataManager
    from simple_trader import SimpleXGBoostTrader

    print("🚀 AlgoTrendy Backtesting Examples")
    print("=" * 50)

    # Stock trading example
    print("\n📈 Stock Trading Backtest (AAPL)")
    print("-" * 30)

    dm = DataManager()
    df_stock = dm.prepare_dataset("AAPL", period="1y", interval="1d")

    # Simple trader for demo
    trader_stock = SimpleXGBoostTrader()
    X_stock, y_stock = trader_stock.prepare_features(df_stock)
    metrics_stock = trader_stock.train(X_stock, y_stock)

    signals_stock = trader_stock.predict(X_stock)
    signals_series_stock = pd.Series(signals_stock, index=df_stock.index)

    backtester_stock = Backtester(asset_type="stock")
    backtest_results_stock = backtester_stock.run_backtest(df_stock, signals_series_stock, "AAPL")

    metrics = backtest_results_stock['metrics']
    print(f"Total Return: {metrics.total_return:.2%}")
    print(f"Annual Return: {metrics.annual_return:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Win Rate: {metrics.win_rate:.2%}")

    # Futures trading example (simulated with stock data for demo)
    print("\n🔥 Futures Trading Backtest (ES Simulation)")
    print("-" * 30)

    # For demo, we'll use stock data but treat it as futures
    df_futures = dm.prepare_dataset("SPY", period="60d", interval="1h")  # Use SPY as proxy for ES

    trader_futures = SimpleXGBoostTrader()
    X_futures, y_futures = trader_futures.prepare_features(df_futures)
    metrics_futures = trader_futures.train(X_futures, y_futures)

    signals_futures = trader_futures.predict(X_futures)
    signals_series_futures = pd.Series(signals_futures, index=df_futures.index)

    backtester_futures = Backtester(initial_capital=100000, asset_type="futures")
    backtest_results_futures = backtester_futures.run_backtest(df_futures, signals_series_futures, "ES=F")

    metrics_f = backtest_results_futures['metrics']
    print(f"Total Return: {metrics_f.total_return:.2%}")
    print(f"Annual Return: {metrics_f.annual_return:.2%}")
    print(f"Sharpe Ratio: {metrics_f.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics_f.max_drawdown:.2%}")
    print(f"Win Rate: {metrics_f.win_rate:.2%}")

    print("\n✅ Backtesting examples completed!")
    print("💡 Note: Futures backtest uses SPY data as proxy. Use real futures data for production.")