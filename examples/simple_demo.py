"""
Simplified demo of XGBoost trading system using available packages.
This demonstrates the core concepts without requiring pandas/yfinance.
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_mock_trading_data(n_samples=10000, n_features=50):
    """
    Generate synthetic trading data for demonstration
    
    Args:
        n_samples: Number of trading samples
        n_features: Number of technical indicators/features
        
    Returns:
        X: Feature matrix (technical indicators)
        y: Target (1 for profitable trade, 0 for unprofitable)
        feature_names: Names of the features
    """
    logger.info(f"Generating {n_samples} samples with {n_features} features...")
    
    # Generate base classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=30,
        n_redundant=10,
        n_clusters_per_class=2,
        random_state=42
    )
    
    # Create realistic feature names for trading indicators
    feature_names = [
        'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
        'rsi_14', 'macd', 'macd_signal', 'macd_diff',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'stoch_k', 'stoch_d', 'atr', 'cci', 'williams_r',
        'volume_sma', 'vwap', 'price_change', 'volatility_10',
        'volatility_20', 'support', 'resistance',
        'price_above_sma20', 'price_above_sma50', 'sma20_above_sma50',
        'return_1d', 'return_3d', 'return_5d', 'return_10d',
        'high_5d', 'low_5d', 'rsi_overbought', 'rsi_oversold',
        'bb_squeeze', 'macd_bullish', 'volume_spike', 'volume_ratio',
        'gap_up', 'gap_down', 'hour', 'day_of_week', 'month',
        'momentum_5d', 'trend_strength', 'market_regime',
        'liquidity_score', 'news_sentiment', 'options_flow'
    ]
    
    # Ensure we have the right number of feature names
    while len(feature_names) < n_features:
        feature_names.append(f'feature_{len(feature_names)}')
    
    feature_names = feature_names[:n_features]
    
    logger.info(f"Generated trading dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    return X, y, feature_names

def train_xgboost_model(X, y, feature_names):
    """
    Train XGBoost model for trading signal prediction
    
    Args:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
        
    Returns:
        Dictionary with model and results
    """
    logger.info("Training XGBoost trading model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Configure XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Train model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    logger.info(f"Training accuracy: {train_accuracy:.4f}")
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    
    # Feature importance
    feature_importance = np.array(model.feature_importances_)
    importance_df = list(zip(feature_names, feature_importance))
    importance_df = sorted(importance_df, key=lambda x: x[1], reverse=True)
    
    results = {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'y_test': y_test,
        'y_pred': y_pred_test,
        'y_pred_proba': y_pred_proba,
        'feature_importance': importance_df,
    'classification_report': classification_report(y_test, y_pred_test, zero_division=0)
    }
    
    return results

def simulate_backtest(signals, returns=None):
    """
    Simple backtest simulation
    
    Args:
        signals: Array of trading signals (1=buy, 0=hold)
        returns: Array of returns (generated if None)
        
    Returns:
        Dictionary with backtest results
    """
    logger.info("Running backtest simulation...")
    
    n_periods = len(signals)
    
    # Generate random returns if not provided
    if returns is None:
        # Create realistic return distribution
        base_return = 0.0005  # 0.05% daily base return
        volatility = 0.02     # 2% daily volatility
        returns = np.random.normal(base_return, volatility, n_periods)
    
    # Simple backtest logic
    portfolio_returns = []
    position = 0
    
    for i in range(n_periods):
        # Update position based on signal
        if signals[i] == 1 and position == 0:  # Buy signal
            position = 1
        elif signals[i] == 0 and position == 1:  # Sell signal (simplified)
            position = 0
        
        # Calculate return
        if position == 1:
            portfolio_return = returns[i]  # Long position
        else:
            portfolio_return = 0  # No position
        
        portfolio_returns.append(portfolio_return)
    
    # Calculate performance metrics
    portfolio_returns = np.array(portfolio_returns)
    
    total_return = np.sum(portfolio_returns)
    sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252) if np.std(portfolio_returns) > 0 else 0
    
    # Win rate
    winning_trades = portfolio_returns[portfolio_returns > 0]
    losing_trades = portfolio_returns[portfolio_returns < 0]
    win_rate = len(winning_trades) / max(1, len(winning_trades) + len(losing_trades))
    
    # Maximum drawdown (simplified)
    cumulative_returns = np.cumsum(portfolio_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = cumulative_returns - running_max
    max_drawdown = np.min(drawdowns)
    
    results = {
        'total_return': total_return,
        'annual_return': total_return * 252 / n_periods,  # Annualized
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'num_trades': np.sum(np.diff(np.concatenate([[0], signals])) == 1),
        'portfolio_returns': portfolio_returns,
        'cumulative_returns': np.cumsum(portfolio_returns)
    }
    
    logger.info(f"Backtest completed:")
    logger.info(f"  Total return: {results['total_return']:.2%}")
    logger.info(f"  Annual return: {results['annual_return']:.2%}")
    logger.info(f"  Sharpe ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"  Win rate: {results['win_rate']:.2%}")
    logger.info(f"  Max drawdown: {results['max_drawdown']:.2%}")
    logger.info(f"  Number of trades: {results['num_trades']}")
    
    return results

def plot_results(model_results, backtest_results):
    """Plot model and backtest results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Feature importance
    top_features = model_results['feature_importance'][:15]
    feature_names, importances = zip(*top_features)
    
    axes[0, 0].barh(range(len(feature_names)), importances)
    axes[0, 0].set_yticks(range(len(feature_names)))
    axes[0, 0].set_yticklabels(feature_names)
    axes[0, 0].set_title('Top 15 Feature Importance')
    axes[0, 0].set_xlabel('Importance')
    
    # Prediction probabilities
    proba_positive = model_results['y_pred_proba'][:, 1]
    axes[0, 1].hist(proba_positive, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Prediction Probability Distribution')
    axes[0, 1].set_xlabel('Probability of Profitable Trade')
    axes[0, 1].set_ylabel('Frequency')
    
    # Portfolio performance
    cumulative_returns = backtest_results['cumulative_returns']
    axes[1, 0].plot(cumulative_returns)
    axes[1, 0].set_title('Cumulative Portfolio Returns')
    axes[1, 0].set_xlabel('Time Period')
    axes[1, 0].set_ylabel('Cumulative Return')
    axes[1, 0].grid(True)
    
    # Return distribution
    portfolio_returns = backtest_results['portfolio_returns']
    axes[1, 1].hist(portfolio_returns, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Portfolio Return Distribution')
    axes[1, 1].set_xlabel('Daily Return')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    logger.info("Results plotted successfully")

def main():
    """Main demo function"""
    print("🚀 AlgoTrendy XGBoost Trading System - Demo")
    print("=" * 50)
    
    try:
        # Generate synthetic trading data
        X, y, feature_names = generate_mock_trading_data(n_samples=10000, n_features=50)
        
        # Train XGBoost model
        model_results = train_xgboost_model(X, y, feature_names)
        
        print(f"\n📊 Model Training Results:")
        print(f"Training Accuracy: {model_results['train_accuracy']:.4f}")
        print(f"Test Accuracy: {model_results['test_accuracy']:.4f}")
        
        print(f"\n🔍 Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(model_results['feature_importance'][:10], 1):
            print(f"{i:2d}. {feature:<20}: {importance:.4f}")
        
        # Generate trading signals using the model
        test_size = len(model_results['y_test'])
        signals = model_results['y_pred']
        
        # Run backtest
        backtest_results = simulate_backtest(signals)
        
        print(f"\n📈 Backtest Results:")
        print(f"Total Return: {backtest_results['total_return']:+.2%}")
        print(f"Annualized Return: {backtest_results['annual_return']:+.2%}")
        print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:+.2f}")
        print(f"Win Rate: {backtest_results['win_rate']:.2%}")
        print(f"Max Drawdown: {backtest_results['max_drawdown']:+.2%}")
        print(f"Number of Trades: {backtest_results['num_trades']}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(model_results['classification_report'])
        
        # Plot results
        print(f"\n📊 Generating plots...")
        plot_results(model_results, backtest_results)
        
        print(f"\n✅ Demo completed successfully!")
        print(f"\nThis demo shows the core concepts of the AlgoTrendy system:")
        print(f"1. ✅ XGBoost model training with technical indicators")
        print(f"2. ✅ Feature importance analysis")
        print(f"3. ✅ Trading signal generation")
        print(f"4. ✅ Backtesting and performance evaluation")
        print(f"5. ✅ Visualization of results")
        
        print(f"\n🚀 Next Steps:")
        print(f"- Install pandas and yfinance to work with real market data")
        print(f"- Use the full AlgoTrendy system with: python main.py full")
        print(f"- Customize the XGBoost parameters for better performance")
        print(f"- Add more sophisticated risk management rules")
        
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        print(f"❌ Demo failed with error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)