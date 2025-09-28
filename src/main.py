"""
Main application script for AlgoTrendy XGBoost Trading System.
Orchestrates data collection, model training, backtesting, and live trading.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from algotrendy.config import CONFIG, MODELS_DIR, logger, setup_logging
from algotrendy.data_adapter import DataAdapter
from algotrendy.backtester import Backtester
from algotrendy.simple_trader import SimpleMLTrader
from algotrendy.model_monitor import ModelMonitor
from algotrendy.feature_analysis import FeatureImportanceReporter



class AlgoTrendyApp:
    """Main application class for AlgoTrendy"""

    def __init__(self):
        self.data_manager = DataAdapter()
        self.monitor = ModelMonitor()
        self.feature_reporter = FeatureImportanceReporter()
        self.models: Dict[str, Dict[str, Any]] = {}
        self.default_period = '2y'
        self.default_interval = '1d'
        self.futures_period = '90d'
        self.futures_interval = '15m'

    # ------------------------------------------------------------------
    def _resolve_asset_profile(self, model_type: str) -> Dict[str, str]:
        asset_type = 'futures' if (model_type or '').lower().startswith('futures') else 'stock'
        period = self.futures_period if asset_type == 'futures' else self.default_period
        interval = self.futures_interval if asset_type == 'futures' else self.default_interval
        return {'asset_type': asset_type, 'period': period, 'interval': interval}

    # ------------------------------------------------------------------
    def _fetch_price_frame(
        self,
        symbol: str,
        profile: Dict[str, str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        import pandas as pd

        df = self.data_manager.fetch_data(
            symbol,
            period=profile['period'],
            interval=profile['interval'],
            asset_type=profile['asset_type'],
        )
        if start_date:
            df = df.loc[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df.loc[df.index <= pd.to_datetime(end_date)]
        return df.dropna()

    # ------------------------------------------------------------------
    @staticmethod
    def _to_price_dict(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        import numpy as np

        return {
            'open': df['open'].to_numpy(),
            'high': df['high'].to_numpy(),
            'low': df['low'].to_numpy(),
            'close': df['close'].to_numpy(),
            'volume': df['volume'].to_numpy(),
            'index': df.index.to_numpy(),
        }

    # ------------------------------------------------------------------
    def _store_model(
        self,
        symbol: str,
        trader: SimpleMLTrader,
        asset_type: str,
        model_type: str,
        metrics: Dict[str, Any],
        save_model: bool,
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            'model': trader,
            'asset_type': asset_type,
            'model_type': model_type,
            'metrics': metrics,
            'trained_at': metrics.get('trained_at'),
        }
        if save_model:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model_path = MODELS_DIR / f"{symbol}_{asset_type}_{model_type}.pkl"
            trader.save_model(str(model_path))
            entry['model_path'] = str(model_path)
        self.models[symbol] = entry
        return entry

    # ------------------------------------------------------------------
    def _ensure_model(self, symbol: str, model_type: str) -> Dict[str, Any]:
        entry = self.models.get(symbol)
        if entry and entry.get('model') is not None:
            return entry
        self.train_single_symbol(symbol, model_type=model_type, save_model=False)
        return self.models[symbol]

    # ------------------------------------------------------------------
    @staticmethod
    def _signal_label(signal: int) -> str:
        mapping = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}
        return mapping.get(int(signal), 'HOLD')

    # ------------------------------------------------------------------
    def train_single_symbol(self, symbol: str, model_type: str = 'binary', save_model: bool = True) -> Dict[str, Any]:
        """Train ML model for a single symbol using SimpleMLTrader."""
        try:
            profile = self._resolve_asset_profile(model_type)
            logger.info("Training %s model for %s (%s)", model_type, symbol, profile['asset_type'])

            price_df = self._fetch_price_frame(symbol, profile)
            price_dict = self._to_price_dict(price_df)

            trader = SimpleMLTrader()
            X, y = trader.prepare_features(price_dict)
            if len(X) == 0:
                raise ValueError('Insufficient feature rows after preprocessing')

            metrics = trader.train(X, y)
            trained_at = datetime.now(timezone.utc).replace(microsecond=0)
            metrics['trained_at'] = trained_at.isoformat() + 'Z'

            feature_report_df = self.feature_reporter.build_report(
                symbol,
                profile['asset_type'],
                trader.get_feature_importance(),
                metadata={'model_type': model_type, 'trained_at': metrics['trained_at']},
            )
            summary = self.feature_reporter.summarize(feature_report_df)
            summary_path = self.feature_reporter.export_summary_markdown(summary)

            entry = self._store_model(symbol, trader, profile['asset_type'], model_type, metrics, save_model)
            entry['feature_report_path'] = str(self.feature_reporter.output_dir / f"{symbol.lower()}_feature_importance.csv")
            entry['feature_summary_path'] = str(summary_path)
            entry['feature_summary'] = summary

            self.monitor.record_training(symbol, metrics, asset_type=profile['asset_type'])
            drift_info = self.monitor.check_accuracy_drift(symbol, metrics.get('test_accuracy', 0.0))
            entry['drift'] = drift_info

            result = {
                'symbol': symbol,
                'asset_type': profile['asset_type'],
                'model_type': model_type,
                'metrics': metrics,
                'n_samples': len(X),
                'model_path': entry.get('model_path'),
                'feature_report_path': entry['feature_report_path'],
                'feature_summary_path': entry['feature_summary_path'],
                'drift': drift_info,
            }
            logger.info("Training completed for %s: accuracy %.3f", symbol, metrics.get('test_accuracy', 0.0))
            return result

        except Exception as exc:
            logger.error("Error training model for %s: %s", symbol, exc)
            raise

    # ------------------------------------------------------------------
    def train_multiple_symbols(self, symbols: List[str], model_type: str = 'binary') -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for symbol in symbols:
            try:
                results[symbol] = self.train_single_symbol(symbol, model_type=model_type)
            except Exception as exc:
                results[symbol] = {'error': str(exc)}
        return results

    # ------------------------------------------------------------------
    def backtest_strategy(
        self,
        symbol: str,
        model_type: str = 'binary',
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Dict[str, Any]:
        """Run a realistic backtest using the stored model."""
        try:
            import pandas as pd
            import numpy as np

            profile = self._resolve_asset_profile(model_type)
            entry = self._ensure_model(symbol, model_type)
            trader: SimpleMLTrader = entry['model']

            price_df = self._fetch_price_frame(symbol, profile, start_date=start_date, end_date=end_date)
            price_dict = self._to_price_dict(price_df)
            X_all, _, feature_index = trader.prepare_features(price_dict, return_index=True)
            if len(X_all) == 0:
                raise ValueError('Not enough feature rows to backtest')

            evaluation_window = max(int(len(X_all) * 0.2), 30)
            evaluation_window = min(evaluation_window, len(X_all))
            X_eval = X_all[-evaluation_window:]
            eval_index = pd.to_datetime(feature_index[-evaluation_window:])

            signals_array = trader.predict(X_eval)
            probabilities = trader.predict_proba(X_eval)
            signal_series = pd.Series(signals_array, index=eval_index)

            aligned_prices = price_df.reindex(signal_series.index, method='pad').dropna()
            signal_series = signal_series.loc[aligned_prices.index]

            backtester = Backtester(initial_capital=CONFIG.initial_capital, asset_type=profile['asset_type'])
            results = backtester.run_backtest(aligned_prices, signal_series, symbol)

            signal_distribution = {
                'buy': int((signal_series == 1).sum()),
                'sell': int((signal_series == -1).sum()),
                'hold': int((signal_series == 0).sum()),
            }

            metrics_obj = results['metrics']
            if hasattr(metrics_obj, '__dataclass_fields__'):
                metrics_dict = asdict(metrics_obj)
            elif hasattr(metrics_obj, '_asdict'):
                metrics_dict = metrics_obj._asdict()
            else:
                metrics_dict = dict(metrics_obj)

            self.monitor.record_backtest(
                symbol,
                {**metrics_dict, **signal_distribution},
                asset_type=profile['asset_type'],
            )

            probability_columns = ["prob_" + self._signal_label(cls).lower() for cls in trader.classes_]
            prob_frame = pd.DataFrame(probabilities, index=signal_series.index, columns=probability_columns)

            return {
                'symbol': symbol,
                'asset_type': profile['asset_type'],
                'model_type': model_type,
                'metrics': metrics_dict,
                'prediction_count': len(signal_series),
                'signal_distribution': signal_distribution,
                'probabilities': prob_frame,
            }

        except Exception as exc:
            logger.error("Error backtesting %s: %s", symbol, exc)
            raise

    # ------------------------------------------------------------------
    def generate_signals(self, symbols: List[str], model_type: str = 'binary', window: int = 5) -> pd.DataFrame:
        """Generate current signals using the latest model state."""
        import pandas as pd
        import numpy as np

        records = []
        for symbol in symbols:
            try:
                entry = self._ensure_model(symbol, model_type)
                profile = self._resolve_asset_profile(entry.get('model_type', model_type))

                price_df = self._fetch_price_frame(symbol, profile)
                price_dict = self._to_price_dict(price_df)

                trader: SimpleMLTrader = entry['model']
                X_all, _, feature_index = trader.prepare_features(price_dict, return_index=True)
                if len(X_all) == 0:
                    continue

                window_size = min(window, len(X_all))
                X_latest = X_all[-window_size:]
                latest_index = pd.to_datetime(feature_index[-window_size:])

                preds = trader.predict(X_latest)
                prob = trader.predict_proba(X_latest)
                class_labels = trader.classes_

                for idx, date in enumerate(latest_index):
                    probabilities = prob[idx]
                    prob_map = {
                        f"prob_{self._signal_label(cls).lower()}": float(probabilities[pos])
                        for pos, cls in enumerate(class_labels)
                    }
                    confidence = float(probabilities[np.argmax(probabilities)])
                    records.append({
                        'symbol': symbol,
                        'timestamp': pd.to_datetime(date),
                        'signal': int(preds[idx]),
                        'signal_name': self._signal_label(preds[idx]),
                        'confidence': confidence,
                        'current_price': float(price_df['close'].iloc[-1]),
                        'asset_type': profile['asset_type'],
                        'model_accuracy': entry.get('metrics', {}).get('test_accuracy'),
                        'drift_detected': entry.get('drift', {}).get('drift_detected'),
                        **prob_map,
                    })
            except Exception as exc:
                logger.error("Signal generation failed for %s: %s", symbol, exc)

        signals_df = pd.DataFrame(records)
        if not signals_df.empty:
            signals_df = signals_df.sort_values('confidence', ascending=False).reset_index(drop=True)
        return signals_df

    # ------------------------------------------------------------------
    def run_full_analysis(self, symbols: List[str] | None = None, model_type: str = 'binary') -> Dict[str, Any]:
        """Train, backtest, and generate signals for the provided symbols."""
        symbols = symbols or CONFIG.symbols
        logger.info("Starting full analysis for %d symbols", len(symbols))

        training_results = self.train_multiple_symbols(symbols, model_type=model_type)
        backtest_results: Dict[str, Any] = {}

        for symbol, result in training_results.items():
            if 'error' in result:
                continue
            try:
                backtest_results[symbol] = self.backtest_strategy(symbol, model_type=model_type)
            except Exception as exc:
                logger.error("Backtest failed for %s: %s", symbol, exc)

        signals_df = self.generate_signals(symbols, model_type=model_type)

        logger.info("Full analysis completed")
        return {
            'training_results': training_results,
            'backtest_results': backtest_results,
            'current_signals': signals_df,
        }



def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='AlgoTrendy XGBoost Trading System')
    parser.add_argument('command', choices=['train', 'backtest', 'signals', 'full', 'retrain',
                                           'futures-train', 'futures-backtest', 'futures-signals',
                                           'futures-auto', 'replay-demo',
                                           'qc-setup', 'qc-projects', 'qc-deploy',
                                           'advanced-train', 'discover-indicators',
                                           'crypto-scalp', 'crypto-strategies', 'futures-strategies',
                                           'interface', 'ai'], help='Command to execute')
    parser.add_argument('--symbols', nargs='+', default=CONFIG.symbols,
                        help='Trading symbols to analyze')
    parser.add_argument('--futures-symbols', nargs='+', default=CONFIG.futures_symbols,
                        help='Futures symbols to analyze')
    parser.add_argument('--model-type', choices=['binary', 'multiclass', 'regression'], default='binary', help='Model type to use')
    parser.add_argument('--force', action='store_true', help='Force retraining cycle even if thresholds are not met')
    parser.add_argument('--symbol', help='Single symbol for train/backtest commands')
    parser.add_argument('--asset-type', choices=['stock', 'futures'], default='stock',
                        help='Asset type (stock or futures)')
    parser.add_argument('--interval', default='1d', help='Data interval (1m, 5m, 1h, 1d) or chart parameters (100tick, 1.0range, 1000vol, 1.0renko)')
    parser.add_argument('--period', default='2y', help='Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y)')
    parser.add_argument('--chart-style', choices=['time', 'tick', 'range', 'volume', 'renko+', 'line'], default='time', help='Chart style for data aggregation')

    args = parser.parse_args()

    # Initialize app
    app = AlgoTrendyApp()

    try:
        if args.command == 'train':
            symbol = args.symbol or args.symbols[0]
            result = app.train_single_symbol(symbol, args.model_type)
            print(f"Training completed for {symbol}")
            print(f"Test accuracy: {result.get('test_accuracy', result.get('test_r2')):.4f}")


        elif args.command == 'retrain':
            from algotrendy.retraining import ModelRetrainer

            symbols = args.symbols or CONFIG.symbols
            retrainer = ModelRetrainer(app)
            outcomes = retrainer.run_cycle(symbols, model_type=args.model_type, force=getattr(args, 'force', False))

            print(f"Running retraining cycle for {len(outcomes)} symbols...")
            for outcome in outcomes:
                status = outcome.get('status', 'unknown').upper()
                symbol = outcome.get('symbol')
                print(f"  - {symbol}: {status}")
                result = outcome.get('result') or {}
                metrics = result.get('metrics') or {}
                accuracy = metrics.get('test_accuracy')
                if accuracy is not None:
                    print(f"      Test accuracy: {accuracy:.3f}")
                if status == 'RETRAINED':
                    model_path = result.get('model_path')
                    if model_path:
                        print(f"      Model saved to: {model_path}")
                    report_path = result.get('feature_report_path')
                    if report_path:
                        print(f"      Feature report: {report_path}")
        elif args.command == 'backtest':
            symbol = args.symbol or args.symbols[0]
            result = app.backtest_strategy(symbol, args.model_type)
            metrics = result['metrics']
            print(f"Backtest Results for {symbol}:")
            print(f"Total Return: {metrics.total_return:.2%}")
            print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            print(f"Max Drawdown: {metrics.max_drawdown:.2%}")

        elif args.command == 'signals':
            signals = app.generate_signals(args.symbols, args.model_type)
            print("Current Trading Signals:")
            print(signals.to_string(index=False))

        elif args.command == 'full':
            app.run_full_analysis(args.symbols, args.model_type)

        # Handle futures commands
        elif args.command == 'futures-train':
            from algotrendy.simple_trader import SimpleMLTrader

            symbols = args.futures_symbols
            print(f"Training futures models for: {symbols}")

            for symbol in symbols:
                try:
                    # Prepare futures data
                    df = app.data_manager.prepare_futures_dataset(symbol, period=args.period, interval=args.interval)

                    # Train model
                    trader = SimpleMLTrader()
                    X, y = trader.prepare_features(df)
                    metrics = trader.train(X, y)

                    print(f"✅ {symbol}: {metrics['test_accuracy']:.3f} accuracy, {len(X)} samples")

                    # Save model
                    model_filename = f"futures_{symbol}_model.pkl"
                    trader.save_model(model_filename)
                    print(f"   Model saved: {model_filename}")

                except Exception as e:
                    print(f"❌ {symbol}: Error - {e}")

        elif args.command == 'futures-backtest':
            from algotrendy.backtester import Backtester
            from algotrendy.simple_trader import SimpleMLTrader

            symbols = args.futures_symbols
            print(f"Backtesting futures strategies for: {symbols}")

            for symbol in symbols:
                try:
                    # Load or train model
                    model_filename = f"futures_{symbol}_model.pkl"
                    trader = SimpleMLTrader()

                    try:
                        trader.load_model(model_filename)
                        print(f"Loaded existing model for {symbol}")
                    except:
                        # Train new model
                        df = app.data_manager.prepare_futures_dataset(symbol, period="60d", interval="5m")
                        X, y = trader.prepare_features(df)
                        trader.train(X, y)
                        trader.save_model(model_filename)

                    # Prepare data for backtesting
                    df = app.data_manager.prepare_futures_dataset(symbol, period="60d", interval="5m")
                    X, _ = trader.prepare_features(df)
                    signals = trader.predict(X)
                    signals_series = pd.Series(signals, index=df.index)

                    # Run backtest
                    backtester = Backtester(initial_capital=100000, asset_type="futures")
                    results = backtester.run_backtest(df, signals_series, f"{symbol}=F")

                    metrics = results['metrics']
                    print(f"📊 {symbol} Backtest Results:")
                    print(f"   Total Return: {metrics.total_return:.2%}")
                    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
                    print(f"   Max Drawdown: {metrics.max_drawdown:.2%}")
                    print(f"   Win Rate: {metrics.win_rate:.2%}")

                except Exception as e:
                    print(f"❌ {symbol}: Backtest error - {e}")

        elif args.command == 'futures-signals':
            from algotrendy.simple_trader import SimpleMLTrader

            symbols = args.futures_symbols
            print(f"Generating futures signals for: {symbols}")

            for symbol in symbols:
                try:
                    # Load model
                    model_filename = f"futures_{symbol}_model.pkl"
                    trader = SimpleMLTrader()
                    trader.load_model(model_filename)

                    # Get latest data
                    df = app.data_manager.prepare_futures_dataset(symbol, period="5d", interval="5m")
                    X, _ = trader.prepare_features(df)

                    # Generate signal
                    latest_X = X[-1:]
                    signal = trader.predict(latest_X)[0]
                    confidence = np.max(trader.predict_proba(latest_X)[0])

                    signal_text = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
                    print(f"📈 {symbol}: {signal_text} (confidence: {confidence:.2f})")

                except Exception as e:
                    print(f"❌ {symbol}: Signal generation error - {e}")

        elif args.command == 'futures-auto':
            from algotrendy.automated_futures_trader import AutomatedFuturesTrader

            symbols = args.futures_symbols
            print(f"🚀 Starting automated futures trading for: {symbols}")
            print("⚠️  This will run continuous automated trading. Press Ctrl+C to stop.")

            # Get API credentials
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')

            if not api_key or not secret_key:
                print("❌ ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables required")
                return 1

            auto_trader = AutomatedFuturesTrader(api_key, secret_key, paper=True)

            try:
                # Start automated trading
                result = auto_trader.start_trading(
                    symbols=symbols,
                    max_daily_trades=10,
                    daily_profit_target=0.02,
                    daily_loss_limit=0.03
                )

                print(f"✅ Automated trading started for: {result['symbols']}")

                # Keep running until interrupted
                import time
                while True:
                    time.sleep(60)
                    status = auto_trader.get_status()
                    print(f"📊 Status: {status['daily_trades']} trades, P&L: {status['daily_pnl']:.2%}")

            except KeyboardInterrupt:
                print("\n🛑 Stopping automated trading...")
                auto_trader.stop_trading()
            except Exception as e:
                print(f"❌ Automated trading error: {e}")
                auto_trader.stop_trading()

        elif args.command == 'replay-demo':
            from algotrendy.market_replay import run_market_replay_demo

            print("🎬 Running Market Replay Demo")
            print("This will replay historical market data to test algorithms")
            run_market_replay_demo()

        elif args.command == 'qc-setup':
            from algotrendy.quantconnect_integration import setup_quantconnect_connection

            print("Setting up QuantConnect connection...")
            qc = setup_quantconnect_connection()
            if qc:
                print("QuantConnect setup complete!")
            else:
                print("QuantConnect setup failed")

        elif args.command == 'qc-projects':
            from algotrendy.quantconnect_integration import QuantConnectIntegration

            print("Getting QuantConnect projects...")
            qc = QuantConnectIntegration()
            if qc.authenticate():
                projects = qc.get_projects()
                print(f"Found {len(projects)} projects:")
                for project in projects[:10]:  # Show first 10
                    print(f"  - {project['name']} (ID: {project['projectId']})")
            else:
                print("Failed to authenticate with QuantConnect")

        elif args.command == 'qc-deploy':
            from algotrendy.quantconnect_integration import QuantConnectIntegration, QuantConnectAlgorithmManager, generate_qc_futures_algorithm

            symbols = args.futures_symbols or ['ES']
            print(f"Deploying futures algorithm to QuantConnect for symbols: {symbols}")

            # Generate algorithm code
            algorithm_code = generate_qc_futures_algorithm(symbols, {
                'max_position_size': 0.1,
                'stop_loss': 0.01,
                'take_profit': 0.02
            })

            # Initialize QuantConnect
            qc = QuantConnectIntegration()
            if not qc.authenticate():
                print("QuantConnect authentication failed")
                return 1

            # Deploy algorithm
            manager = QuantConnectAlgorithmManager(qc)
            result = manager.deploy_futures_algorithm(algorithm_code, f"AlgoTrendy Futures {symbols}")

            if result['success']:
                print("Algorithm deployed successfully!")
                print(f"   Project ID: {result['project_id']}")
                print(f"   Algorithm: {result['algorithm_name']}")
            else:
                print(f"Deployment failed: {result['error']}")

        elif args.command == 'advanced-train':
            from algotrendy.advanced_ml_trainer import run_advanced_training_demo

            print(f"Training advanced ML model for >80% accuracy ({args.chart_style} charts)...")
            run_advanced_training_demo(chart_style=args.chart_style)

        elif args.command == 'discover-indicators':
            from algotrendy.ai_indicator_agent import run_indicator_discovery_demo

            print("Discovering and integrating open-source technical indicators...")
            run_indicator_discovery_demo()

        elif args.command == 'crypto-scalp':
            from algotrendy.crypto_scalping_trader import run_crypto_scalping_demo

            print("Initializing crypto scalping system...")
            run_crypto_scalping_demo()

        elif args.command == 'crypto-strategies':
            from algotrendy.ai_crypto_strategy_agent import run_crypto_strategy_discovery_demo

            print("Discovering and integrating crypto trading strategies...")
            run_crypto_strategy_discovery_demo()

        elif args.command == 'futures-strategies':
            from algotrendy.ai_futures_strategy_agent import run_futures_strategy_discovery_demo

            print("Discovering and integrating futures trading strategies...")
            run_futures_strategy_discovery_demo()

        elif args.command == 'interface':
            from algotrendy.trading_interface import TradingInterface

            print("Launching AlgoTrendy Trading Interface...")
            print("This provides unified access to all trading tools and systems.")
            print()

            try:
                interface = TradingInterface()
                interface.show_main_menu()
            except KeyboardInterrupt:
                print("\nThanks for using AlgoTrendy!")
            except Exception as e:
                print(f"Interface error: {e}")
                logger.error(f"Trading interface error: {e}")

        elif args.command == 'ai':
            from algotrendy.ai_interface import AIInterface

            print("[AI] Launching AlgoTrendy AI Assistant...")
            print("Natural language control for all trading systems.")
            print()

            try:
                ai = AIInterface()
                ai.start_chat()
            except KeyboardInterrupt:
                print("\n[AI] AI Assistant stopped. Goodbye!")
            except Exception as e:
                print(f"AI Interface error: {e}")
                logger.error(f"AI interface error: {e}")

    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
