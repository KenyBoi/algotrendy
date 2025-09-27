#!/usr/bin/env python3
"""
Simple standalone backend API for AlgoTrendy frontend
Provides mock trading data without complex dependencies
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, date
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AlgoTrendy Simple API",
    description="Simple backend API for AlgoTrendy trading platform frontend",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "AlgoTrendy Simple API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Trading System API Endpoints
@app.get("/trading/models")
async def get_ml_models():
    """Get available ML models and their performance metrics"""
    try:
        models = [
            {
                "id": "futures_es_advanced_v1",
                "name": "Advanced ES Futures Model",
                "symbol": "ES",
                "asset_type": "futures",
                "accuracy": 0.742,
                "precision": 0.738,
                "recall": 0.745,
                "sharpe_ratio": 2.34,
                "status": "active",
                "last_trained": "2024-09-25T14:30:00"
            },
            {
                "id": "crypto_btc_predictor_v2",
                "name": "BTC Price Predictor",
                "symbol": "BTC-USD",
                "asset_type": "crypto",
                "accuracy": 0.689,
                "precision": 0.692,
                "recall": 0.685,
                "sharpe_ratio": 1.87,
                "status": "training",
                "last_trained": "2024-09-24T09:15:00"
            },
            {
                "id": "equity_spy_momentum",
                "name": "SPY Momentum Model",
                "symbol": "SPY",
                "asset_type": "equity",
                "accuracy": 0.705,
                "precision": 0.698,
                "recall": 0.712,
                "sharpe_ratio": 2.01,
                "status": "active",
                "last_trained": "2024-09-23T16:45:00"
            },
            {
                "id": "gold_trend_follower",
                "name": "Gold Trend Following Model",
                "symbol": "GC",
                "asset_type": "futures",
                "accuracy": 0.663,
                "precision": 0.659,
                "recall": 0.667,
                "sharpe_ratio": 1.54,
                "status": "active",
                "last_trained": "2024-09-22T11:20:00"
            }
        ]
        
        return {"models": models, "total_models": len(models)}
        
    except Exception as e:
        logger.error(f"Failed to get ML models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trading/strategies")
async def get_trading_strategies():
    """Get available trading strategies"""
    try:
        strategies = [
            {
                "id": "futures_momentum_breakout",
                "name": "Futures Momentum Breakout",
                "description": "Captures explosive moves following consolidation periods in futures markets",
                "strategy_type": "day_trading",
                "asset_type": "futures",
                "parameters": {
                    "consolidation_period": 20,
                    "breakout_threshold": 0.8,
                    "momentum_period": 5,
                    "profit_target": 0.025,
                    "stop_loss": 0.015
                },
                "performance_metrics": {
                    "win_rate": 0.65,
                    "avg_return": 0.024,
                    "max_drawdown": 0.08,
                    "sharpe_ratio": 2.1
                },
                "status": "active"
            },
            {
                "id": "es_intraday_reversal",
                "name": "ES Intraday Mean Reversion",
                "description": "Mean reversion strategy optimized for E-mini S&P 500 futures intraday patterns",
                "strategy_type": "mean_reversion",
                "asset_type": "futures",
                "parameters": {
                    "lookback_period": 20,
                    "deviation_threshold": 2.0,
                    "profit_target": 0.015,
                    "stop_loss": 0.01
                },
                "performance_metrics": {
                    "win_rate": 0.68,
                    "avg_return": 0.019,
                    "max_drawdown": 0.06,
                    "sharpe_ratio": 2.3
                },
                "status": "active"
            },
            {
                "id": "mean_reversion_scalp",
                "name": "Crypto Mean Reversion Scalp",
                "description": "Identifies short-term deviations from moving averages for quick scalping opportunities",
                "strategy_type": "scalping",
                "asset_type": "crypto",
                "parameters": {
                    "lookback_period": 20,
                    "deviation_threshold": 0.5,
                    "profit_target": 0.003,
                    "stop_loss": 0.001,
                    "max_hold_time": 300
                },
                "performance_metrics": {
                    "win_rate": 0.58,
                    "avg_return": 0.018,
                    "max_drawdown": 0.12,
                    "sharpe_ratio": 1.7
                },
                "status": "active"
            },
            {
                "id": "btc_momentum_follow",
                "name": "BTC Momentum Following",
                "description": "Follows strong momentum moves in Bitcoin with multi-timeframe trend confirmation",
                "strategy_type": "momentum",
                "asset_type": "crypto",
                "parameters": {
                    "momentum_period": 14,
                    "trend_confirmation": True,
                    "profit_target": 0.025,
                    "stop_loss": 0.015
                },
                "performance_metrics": {
                    "win_rate": 0.61,
                    "avg_return": 0.022,
                    "max_drawdown": 0.15,
                    "sharpe_ratio": 1.9
                },
                "status": "active"
            },
            {
                "id": "spy_swing_trader",
                "name": "SPY Swing Trading Strategy",
                "description": "Swing trading strategy for SPY using technical indicators and market sentiment",
                "strategy_type": "swing_trading",
                "asset_type": "equity",
                "parameters": {
                    "holding_period": 5,
                    "rsi_threshold": 70,
                    "profit_target": 0.04,
                    "stop_loss": 0.02
                },
                "performance_metrics": {
                    "win_rate": 0.72,
                    "avg_return": 0.031,
                    "max_drawdown": 0.09,
                    "sharpe_ratio": 2.5
                },
                "status": "active"
            }
        ]
        
        return {"strategies": strategies, "total_strategies": len(strategies)}
        
    except Exception as e:
        logger.error(f"Failed to get trading strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trading/backtests")
async def get_backtest_results():
    """Get backtest results"""
    try:
        backtests = [
            {
                "id": "bt_001",
                "strategy_name": "ES Momentum Breakout",
                "symbol": "ES",
                "start_date": "2024-01-01",
                "end_date": "2024-09-27",
                "initial_value": 100000.0,
                "final_value": 142350.0,
                "total_return": 0.4235,
                "sharpe_ratio": 2.34,
                "max_drawdown": 0.087,
                "win_rate": 0.653,
                "total_trades": 147,
                "status": "completed"
            },
            {
                "id": "bt_002", 
                "strategy_name": "BTC Mean Reversion",
                "symbol": "BTC-USD",
                "start_date": "2024-03-01",
                "end_date": "2024-09-27",
                "initial_value": 50000.0,
                "final_value": 67890.0,
                "total_return": 0.3578,
                "sharpe_ratio": 1.89,
                "max_drawdown": 0.156,
                "win_rate": 0.592,
                "total_trades": 203,
                "status": "completed"
            },
            {
                "id": "bt_003",
                "strategy_name": "SPY Swing Trading",
                "symbol": "SPY",
                "start_date": "2023-10-01", 
                "end_date": "2024-09-27",
                "initial_value": 75000.0,
                "final_value": 98750.0,
                "total_return": 0.3167,
                "sharpe_ratio": 2.12,
                "max_drawdown": 0.094,
                "win_rate": 0.708,
                "total_trades": 89,
                "status": "completed"
            },
            {
                "id": "bt_004",
                "strategy_name": "Gold Trend Following",
                "symbol": "GC",
                "start_date": "2024-06-01",
                "end_date": "2024-09-27",
                "initial_value": 80000.0,
                "final_value": 93200.0,
                "total_return": 0.165,
                "sharpe_ratio": 1.67,
                "max_drawdown": 0.078,
                "win_rate": 0.614,
                "total_trades": 76,
                "status": "completed"
            }
        ]
        
        return {"backtests": backtests, "total_backtests": len(backtests)}
        
    except Exception as e:
        logger.error(f"Failed to get backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trading/backtests/run")
async def run_backtest(backtest_config: dict):
    """Run a new backtest"""
    try:
        # Mock response for running a new backtest
        result = {
            "id": f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "running",
            "message": "Backtest started successfully",
            "estimated_completion": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to run backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the simple backend server on port 8000
    uvicorn.run(
        "simple_backend:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info"
    )