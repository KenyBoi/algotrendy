"""
Example usage script demonstrating AlgoTrendy functionality.
This script shows how to use the system step by step.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from main import AlgoTrendyApp
from algotrendy.config import CONFIG, logger

def main():
    """Run example analysis"""
    
    print("🚀 AlgoTrendy XGBoost Trading System - Example Usage")
    print("=" * 60)
    
    # Initialize the application
    app = AlgoTrendyApp()
    
    # Example 1: Train a single model
    print("\n📈 Example 1: Training XGBoost model for AAPL")
    print("-" * 40)
    
    try:
        result = app.train_single_symbol("AAPL", model_type="binary", save_model=True)
        
        print(f"✅ Model trained successfully!")
        print(f"   - Test Accuracy: {result['test_accuracy']:.3f}")
        print(f"   - Data Shape: {result['data_shape']}")
        print(f"   - Model saved as: {result['model_file']}")
        
        # Show top features
        top_features = result['feature_importance'].head(10)
        print(f"\n🔍 Top 10 Most Important Features:")
        for _, row in top_features.iterrows():
            print(f"   {row['feature']:<20}: {row['importance']:.4f}")
            
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return
    
    # Example 2: Run backtest
    print(f"\n📊 Example 2: Backtesting AAPL strategy")
    print("-" * 40)
    
    try:
        backtest_result = app.backtest_strategy("AAPL", model_type="binary")
        metrics = backtest_result['metrics']
        
        print(f"✅ Backtest completed!")
        print(f"   - Total Return: {metrics.total_return:.2%}")
        print(f"   - Annual Return: {metrics.annual_return:.2%}")
        print(f"   - Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"   - Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"   - Win Rate: {metrics.win_rate:.2%}")
        print(f"   - Total Trades: {metrics.total_trades}")
        
        final_value = backtest_result['final_value']
        print(f"   - Final Portfolio Value: ${final_value:,.2f}")
        
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        return
    
    # Example 3: Generate current signals
    print(f"\n📡 Example 3: Generating current trading signals")
    print("-" * 40)
    
    try:
        symbols = ["AAPL", "MSFT", "GOOGL"]  # Smaller list for example
        signals_df = app.generate_signals(symbols, model_type="binary")
        
        if not signals_df.empty:
            print("✅ Signals generated!")
            print("\nCurrent Trading Signals:")
            
            # Display formatted signals
            for _, row in signals_df.iterrows():
                signal_text = {1: "🟢 BUY", -1: "🔴 SELL", 0: "⚪ HOLD"}[row['signal']]
                print(f"   {row['symbol']:<6}: {signal_text} (confidence: {row['confidence']:.2f})")
                print(f"           Price: ${row['current_price']:.2f}, RSI: {row['rsi']:.1f}")
        else:
            print("⚪ No signals generated")
            
    except Exception as e:
        print(f"❌ Error generating signals: {e}")
        return
    
    # Example 4: Quick multi-symbol analysis
    print(f"\n🎯 Example 4: Quick multi-symbol analysis")
    print("-" * 40)
    
    try:
        symbols = ["AAPL", "MSFT"]  # Small list for demo
        results = app.run_full_analysis(symbols, model_type="binary")
        
        print("✅ Multi-symbol analysis completed!")
        print("Check the detailed output above for comprehensive results.")
        
    except Exception as e:
        print(f"❌ Error in multi-symbol analysis: {e}")
        return
    
    print(f"\n🎉 Example completed successfully!")
    print("=" * 60)
    print("Next steps:")
    print("1. Modify CONFIG in config.py to customize parameters")
    print("2. Add more symbols to analyze")
    print("3. Experiment with different model types (binary/multiclass/regression)")
    print("4. Tune hyperparameters for better performance")
    print("5. Implement live trading with your broker's API")

if __name__ == "__main__":
    main()
