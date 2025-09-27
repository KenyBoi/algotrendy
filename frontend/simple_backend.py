#!/usr/bin/env python3
"""
Simple standalone backend API for AlgoTrendy frontend
Provides mock trading data without complex dependencies
"""

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timedelta
import uvicorn
import logging
import random
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models for Request/Response
class OrderRequest(BaseModel):
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    order_type: str  # "market", "limit", "stop"
    price: Optional[float] = None
    stop_price: Optional[float] = None

class WatchlistRequest(BaseModel):
    symbol: str
    name: Optional[str] = None

class StrategyRequest(BaseModel):
    name: str
    description: str
    strategy_type: str
    asset_type: str
    parameters: Dict[str, Any]

class BacktestRequest(BaseModel):
    strategy_id: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float

class ModelRequest(BaseModel):
    name: str
    symbol: str
    asset_type: str
    model_type: str
    parameters: Dict[str, Any]

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

@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    return {
        "status": "online",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "trading": "active",
            "market_data": "active",
            "backtesting": "active",
            "ml_models": "active"
        }
    }

# =============================================================================
# PORTFOLIO MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio overview with balance, positions, and P&L"""
    try:
        return {
            "account_balance": 142750.85,
            "cash_balance": 34250.00,
            "invested_amount": 108500.85,
            "total_value": 142750.85,
            "unrealized_pnl": 8750.25,
            "realized_pnl": 12450.50,
            "day_change": 2150.35,
            "day_change_percent": 0.0153,
            "portfolio_performance": {
                "1d": 0.0153,
                "1w": 0.0287,
                "1m": 0.0645,
                "3m": 0.1234,
                "ytd": 0.1875,
                "1y": 0.2456
            },
            "risk_metrics": {
                "beta": 1.12,
                "sharpe_ratio": 1.87,
                "max_drawdown": 0.089,
                "volatility": 0.145
            },
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/positions")
async def get_positions():
    """Get current positions list"""
    try:
        positions = [
            {
                "symbol": "ES",
                "name": "E-mini S&P 500 Future",
                "asset_type": "futures",
                "quantity": 5.0,
                "average_price": 4425.75,
                "current_price": 4468.25,
                "market_value": 111706.25,
                "unrealized_pnl": 2128.50,
                "unrealized_pnl_percent": 0.0195,
                "day_change": 425.00,
                "day_change_percent": 0.0096,
                "position_type": "long"
            },
            {
                "symbol": "BTC-USD",
                "name": "Bitcoin",
                "asset_type": "crypto",
                "quantity": 1.25,
                "average_price": 42850.00,
                "current_price": 43920.00,
                "market_value": 54900.00,
                "unrealized_pnl": 1337.50,
                "unrealized_pnl_percent": 0.025,
                "day_change": 890.00,
                "day_change_percent": 0.0206,
                "position_type": "long"
            },
            {
                "symbol": "SPY",
                "name": "SPDR S&P 500 ETF Trust",
                "asset_type": "equity",
                "quantity": 150.0,
                "average_price": 435.20,
                "current_price": 442.85,
                "market_value": 66427.50,
                "unrealized_pnl": 1147.50,
                "unrealized_pnl_percent": 0.0176,
                "day_change": 337.50,
                "day_change_percent": 0.0051,
                "position_type": "long"
            },
            {
                "symbol": "GC",
                "name": "Gold Future",
                "asset_type": "futures",
                "quantity": -2.0,
                "average_price": 1985.50,
                "current_price": 1978.25,
                "market_value": -39565.00,
                "unrealized_pnl": 145.00,
                "unrealized_pnl_percent": 0.0037,
                "day_change": -85.00,
                "day_change_percent": -0.0021,
                "position_type": "short"
            }
        ]
        
        return {
            "positions": positions,
            "total_positions": len(positions),
            "total_market_value": sum(pos["market_value"] for pos in positions),
            "total_unrealized_pnl": sum(pos["unrealized_pnl"] for pos in positions),
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio/performance")
async def get_portfolio_performance():
    """Get historical portfolio performance data"""
    try:
        # Generate realistic performance data for the last 30 days
        base_date = datetime.now() - timedelta(days=30)
        base_value = 125000.0
        performance_data = []
        
        for i in range(31):
            current_date = base_date + timedelta(days=i)
            # Add some realistic market volatility
            daily_return = random.uniform(-0.03, 0.04)
            base_value *= (1 + daily_return)
            
            performance_data.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "portfolio_value": round(base_value, 2),
                "daily_return": round(daily_return * 100, 3),
                "cumulative_return": round((base_value - 125000) / 125000 * 100, 3)
            })
        
        return {
            "performance_data": performance_data,
            "summary": {
                "start_value": performance_data[0]["portfolio_value"],
                "end_value": performance_data[-1]["portfolio_value"],
                "total_return": performance_data[-1]["cumulative_return"],
                "best_day": max(performance_data, key=lambda x: x["daily_return"])["daily_return"],
                "worst_day": min(performance_data, key=lambda x: x["daily_return"])["daily_return"],
                "avg_daily_return": round(sum(d["daily_return"] for d in performance_data) / len(performance_data), 3)
            },
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get portfolio performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# TRADING ENDPOINTS
# =============================================================================

@app.post("/api/orders")
async def submit_order(order: OrderRequest):
    """Submit a new buy/sell order"""
    try:
        order_id = f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        # Mock order processing
        order_data = {
            "id": order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "order_type": order.order_type,
            "price": order.price,
            "stop_price": order.stop_price,
            "status": "submitted",
            "submitted_at": datetime.now().isoformat(),
            "filled_quantity": 0.0,
            "remaining_quantity": order.quantity,
            "avg_fill_price": None,
            "commission": 0.0,
            "message": "Order submitted successfully"
        }
        
        return order_data
    except Exception as e:
        logger.error(f"Failed to submit order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/orders")
async def get_orders(status: Optional[str] = Query(None), limit: int = Query(50, ge=1, le=100)):
    """Get order history and status"""
    try:
        # Mock order history data
        orders = [
            {
                "id": "order_20240927_143522_1234",
                "symbol": "ES",
                "side": "buy",
                "quantity": 2.0,
                "order_type": "market",
                "price": None,
                "status": "filled",
                "submitted_at": "2024-09-27T14:35:22",
                "filled_at": "2024-09-27T14:35:25",
                "filled_quantity": 2.0,
                "remaining_quantity": 0.0,
                "avg_fill_price": 4462.25,
                "commission": 4.50
            },
            {
                "id": "order_20240927_112845_5678",
                "symbol": "BTC-USD",
                "side": "buy",
                "quantity": 0.5,
                "order_type": "limit",
                "price": 43800.00,
                "status": "filled",
                "submitted_at": "2024-09-27T11:28:45",
                "filled_at": "2024-09-27T11:29:12",
                "filled_quantity": 0.5,
                "remaining_quantity": 0.0,
                "avg_fill_price": 43785.00,
                "commission": 12.75
            },
            {
                "id": "order_20240927_095234_9012",
                "symbol": "SPY",
                "side": "buy",
                "quantity": 50.0,
                "order_type": "limit",
                "price": 440.00,
                "status": "open",
                "submitted_at": "2024-09-27T09:52:34",
                "filled_at": None,
                "filled_quantity": 0.0,
                "remaining_quantity": 50.0,
                "avg_fill_price": None,
                "commission": 0.0
            },
            {
                "id": "order_20240926_163412_3456",
                "symbol": "GC",
                "side": "sell",
                "quantity": 1.0,
                "order_type": "market",
                "price": None,
                "status": "filled",
                "submitted_at": "2024-09-26T16:34:12",
                "filled_at": "2024-09-26T16:34:15",
                "filled_quantity": 1.0,
                "remaining_quantity": 0.0,
                "avg_fill_price": 1982.50,
                "commission": 3.25
            },
            {
                "id": "order_20240926_140923_7890",
                "symbol": "BTC-USD",
                "side": "sell",
                "quantity": 0.25,
                "order_type": "stop",
                "price": 42500.00,
                "stop_price": 42800.00,
                "status": "cancelled",
                "submitted_at": "2024-09-26T14:09:23",
                "cancelled_at": "2024-09-26T15:45:18",
                "filled_quantity": 0.0,
                "remaining_quantity": 0.25,
                "avg_fill_price": None,
                "commission": 0.0
            }
        ]
        
        # Filter by status if provided
        if status:
            orders = [order for order in orders if order["status"] == status]
        
        # Apply limit
        orders = orders[:limit]
        
        return {
            "orders": orders,
            "total_orders": len(orders),
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/orders/{order_id}")
async def get_order_details(order_id: str = Path(...)):
    """Get specific order details"""
    try:
        # Mock order details
        order_details = {
            "id": order_id,
            "symbol": "ES",
            "side": "buy",
            "quantity": 2.0,
            "order_type": "market",
            "price": None,
            "status": "filled",
            "submitted_at": "2024-09-27T14:35:22",
            "filled_at": "2024-09-27T14:35:25",
            "filled_quantity": 2.0,
            "remaining_quantity": 0.0,
            "avg_fill_price": 4462.25,
            "commission": 4.50,
            "fills": [
                {
                    "fill_id": "fill_001",
                    "quantity": 1.0,
                    "price": 4461.75,
                    "timestamp": "2024-09-27T14:35:24",
                    "commission": 2.25
                },
                {
                    "fill_id": "fill_002",
                    "quantity": 1.0,
                    "price": 4462.75,
                    "timestamp": "2024-09-27T14:35:25",
                    "commission": 2.25
                }
            ]
        }
        
        return order_details
    except Exception as e:
        logger.error(f"Failed to get order details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/orders/{order_id}")
async def cancel_order(order_id: str = Path(...)):
    """Cancel an order"""
    try:
        # Mock order cancellation
        return {
            "id": order_id,
            "status": "cancelled",
            "cancelled_at": datetime.now().isoformat(),
            "message": "Order cancelled successfully"
        }
    except Exception as e:
        logger.error(f"Failed to cancel order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# MARKET DATA ENDPOINTS
# =============================================================================

@app.get("/api/market/prices")
async def get_market_prices():
    """Get current market prices"""
    try:
        prices = [
            {
                "symbol": "ES",
                "name": "E-mini S&P 500 Future",
                "price": 4468.25,
                "change": 21.75,
                "change_percent": 0.0049,
                "volume": 1234567,
                "bid": 4467.75,
                "ask": 4468.75,
                "last_updated": datetime.now().isoformat()
            },
            {
                "symbol": "BTC-USD",
                "name": "Bitcoin",
                "price": 43920.00,
                "change": 1120.00,
                "change_percent": 0.0262,
                "volume": 28456,
                "bid": 43915.00,
                "ask": 43925.00,
                "last_updated": datetime.now().isoformat()
            },
            {
                "symbol": "SPY",
                "name": "SPDR S&P 500 ETF",
                "price": 442.85,
                "change": 2.15,
                "change_percent": 0.0049,
                "volume": 45678921,
                "bid": 442.84,
                "ask": 442.86,
                "last_updated": datetime.now().isoformat()
            },
            {
                "symbol": "GC",
                "name": "Gold Future",
                "price": 1978.25,
                "change": -7.25,
                "change_percent": -0.0037,
                "volume": 234567,
                "bid": 1978.00,
                "ask": 1978.50,
                "last_updated": datetime.now().isoformat()
            },
            {
                "symbol": "ETH-USD",
                "name": "Ethereum",
                "price": 2635.50,
                "change": 45.30,
                "change_percent": 0.0175,
                "volume": 18765,
                "bid": 2635.00,
                "ask": 2636.00,
                "last_updated": datetime.now().isoformat()
            }
        ]
        
        return {
            "prices": prices,
            "total_symbols": len(prices),
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get market prices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/watchlist")
async def get_watchlist():
    """Get user watchlist"""
    try:
        watchlist = [
            {
                "symbol": "ES",
                "name": "E-mini S&P 500 Future",
                "price": 4468.25,
                "change": 21.75,
                "change_percent": 0.0049,
                "added_at": "2024-09-20T10:30:00"
            },
            {
                "symbol": "BTC-USD", 
                "name": "Bitcoin",
                "price": 43920.00,
                "change": 1120.00,
                "change_percent": 0.0262,
                "added_at": "2024-09-18T14:15:00"
            },
            {
                "symbol": "SPY",
                "name": "SPDR S&P 500 ETF",
                "price": 442.85,
                "change": 2.15,
                "change_percent": 0.0049,
                "added_at": "2024-09-15T09:45:00"
            },
            {
                "symbol": "AAPL",
                "name": "Apple Inc",
                "price": 178.25,
                "change": -1.45,
                "change_percent": -0.0081,
                "added_at": "2024-09-10T16:20:00"
            }
        ]
        
        return {
            "watchlist": watchlist,
            "total_symbols": len(watchlist),
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/market/watchlist")
async def add_to_watchlist(request: WatchlistRequest):
    """Add symbol to watchlist"""
    try:
        return {
            "symbol": request.symbol,
            "name": request.name or f"Symbol {request.symbol}",
            "added_at": datetime.now().isoformat(),
            "message": f"Successfully added {request.symbol} to watchlist"
        }
    except Exception as e:
        logger.error(f"Failed to add to watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/quotes/{symbol}")
async def get_quote(symbol: str = Path(...)):
    """Get real-time quote for a symbol"""
    try:
        # Mock quote data
        quote = {
            "symbol": symbol,
            "price": round(random.uniform(100, 5000), 2),
            "bid": round(random.uniform(100, 5000), 2),
            "ask": round(random.uniform(100, 5000), 2),
            "change": round(random.uniform(-50, 50), 2),
            "change_percent": round(random.uniform(-0.05, 0.05), 4),
            "volume": random.randint(10000, 1000000),
            "high": round(random.uniform(100, 5000), 2),
            "low": round(random.uniform(100, 5000), 2),
            "open": round(random.uniform(100, 5000), 2),
            "previous_close": round(random.uniform(100, 5000), 2),
            "last_updated": datetime.now().isoformat()
        }
        
        return quote
    except Exception as e:
        logger.error(f"Failed to get quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# STRATEGY MANAGEMENT ENDPOINTS  
# =============================================================================

@app.get("/api/strategies")
async def get_strategies():
    """List all trading strategies with full CRUD operations"""
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
                "status": "active",
                "created_at": "2024-09-15T10:30:00",
                "last_updated": "2024-09-25T14:22:00"
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
                "status": "active",
                "created_at": "2024-09-10T16:45:00",
                "last_updated": "2024-09-24T11:15:00"
            },
            {
                "id": "crypto_scalping_v2",
                "name": "Advanced Crypto Scalping",
                "description": "High-frequency scalping strategy for cryptocurrency markets with advanced risk management",
                "strategy_type": "scalping",
                "asset_type": "crypto",
                "parameters": {
                    "timeframe": "1m",
                    "max_positions": 3,
                    "profit_target": 0.005,
                    "stop_loss": 0.002,
                    "max_hold_time": 180
                },
                "performance_metrics": {
                    "win_rate": 0.62,
                    "avg_return": 0.021,
                    "max_drawdown": 0.094,
                    "sharpe_ratio": 1.85
                },
                "status": "inactive",
                "created_at": "2024-08-22T09:20:00",
                "last_updated": "2024-09-20T13:45:00"
            }
        ]
        
        return {
            "strategies": strategies,
            "total_strategies": len(strategies),
            "active_strategies": len([s for s in strategies if s["status"] == "active"]),
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategies")
async def create_strategy(strategy: StrategyRequest):
    """Create a new trading strategy"""
    try:
        strategy_id = f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        new_strategy = {
            "id": strategy_id,
            "name": strategy.name,
            "description": strategy.description,
            "strategy_type": strategy.strategy_type,
            "asset_type": strategy.asset_type,
            "parameters": strategy.parameters,
            "status": "inactive",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "message": "Strategy created successfully"
        }
        
        return new_strategy
    except Exception as e:
        logger.error(f"Failed to create strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/strategies/{strategy_id}")
async def update_strategy(strategy: StrategyRequest, strategy_id: str = Path(...)):
    """Update an existing strategy"""
    try:
        updated_strategy = {
            "id": strategy_id,
            "name": strategy.name,
            "description": strategy.description,
            "strategy_type": strategy.strategy_type,
            "asset_type": strategy.asset_type,
            "parameters": strategy.parameters,
            "last_updated": datetime.now().isoformat(),
            "message": "Strategy updated successfully"
        }
        
        return updated_strategy
    except Exception as e:
        logger.error(f"Failed to update strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/strategies/{strategy_id}")
async def delete_strategy(strategy_id: str = Path(...)):
    """Delete a strategy"""
    try:
        return {
            "id": strategy_id,
            "deleted_at": datetime.now().isoformat(),
            "message": "Strategy deleted successfully"
        }
    except Exception as e:
        logger.error(f"Failed to delete strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategies/{strategy_id}/activate")
async def activate_strategy(strategy_id: str = Path(...)):
    """Activate a strategy"""
    try:
        return {
            "id": strategy_id,
            "status": "active",
            "activated_at": datetime.now().isoformat(),
            "message": "Strategy activated successfully"
        }
    except Exception as e:
        logger.error(f"Failed to activate strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategies/{strategy_id}/deactivate")
async def deactivate_strategy(strategy_id: str = Path(...)):
    """Deactivate a strategy"""
    try:
        return {
            "id": strategy_id,
            "status": "inactive",
            "deactivated_at": datetime.now().isoformat(),
            "message": "Strategy deactivated successfully"
        }
    except Exception as e:
        logger.error(f"Failed to deactivate strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# BACKTESTING ENDPOINTS
# =============================================================================

@app.get("/api/backtests")
async def get_backtests():
    """List all backtest results with comprehensive operations"""
    try:
        backtests = [
            {
                "id": "bt_20240927_001",
                "strategy_name": "Advanced ES Futures Model",
                "symbol": "ES",
                "start_date": "2024-01-01",
                "end_date": "2024-09-27",
                "initial_capital": 100000.0,
                "final_value": 142350.0,
                "total_return": 0.4235,
                "annualized_return": 0.3187,
                "sharpe_ratio": 2.34,
                "max_drawdown": 0.087,
                "win_rate": 0.653,
                "total_trades": 147,
                "profitable_trades": 96,
                "avg_trade_duration": "2.3 hours",
                "status": "completed",
                "created_at": "2024-09-25T14:30:00",
                "completed_at": "2024-09-25T14:45:00"
            },
            {
                "id": "bt_20240926_002",
                "strategy_name": "BTC Mean Reversion",
                "symbol": "BTC-USD",
                "start_date": "2024-03-01",
                "end_date": "2024-09-26",
                "initial_capital": 50000.0,
                "final_value": 67890.0,
                "total_return": 0.3578,
                "annualized_return": 0.2834,
                "sharpe_ratio": 1.89,
                "max_drawdown": 0.156,
                "win_rate": 0.592,
                "total_trades": 203,
                "profitable_trades": 120,
                "avg_trade_duration": "4.7 hours",
                "status": "completed",
                "created_at": "2024-09-24T09:15:00",
                "completed_at": "2024-09-24T09:42:00"
            },
            {
                "id": "bt_20240925_003",
                "strategy_name": "SPY Swing Trading",
                "symbol": "SPY",
                "start_date": "2023-10-01",
                "end_date": "2024-09-25",
                "initial_capital": 75000.0,
                "final_value": 98750.0,
                "total_return": 0.3167,
                "annualized_return": 0.2892,
                "sharpe_ratio": 2.12,
                "max_drawdown": 0.094,
                "win_rate": 0.708,
                "total_trades": 89,
                "profitable_trades": 63,
                "avg_trade_duration": "3.2 days",
                "status": "completed",
                "created_at": "2024-09-23T16:45:00",
                "completed_at": "2024-09-23T17:12:00"
            },
            {
                "id": "bt_20240927_004",
                "strategy_name": "Crypto Scalping Advanced",
                "symbol": "ETH-USD",
                "start_date": "2024-08-01",
                "end_date": "2024-09-27",
                "initial_capital": 25000.0,
                "final_value": 25000.0,
                "total_return": 0.0,
                "annualized_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "profitable_trades": 0,
                "avg_trade_duration": "0 minutes",
                "status": "running",
                "created_at": "2024-09-27T14:30:00",
                "estimated_completion": "2024-09-27T15:15:00"
            }
        ]
        
        return {
            "backtests": backtests,
            "total_backtests": len(backtests),
            "completed_backtests": len([bt for bt in backtests if bt["status"] == "completed"]),
            "running_backtests": len([bt for bt in backtests if bt["status"] == "running"]),
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get backtests: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backtests")
async def run_backtest(backtest: BacktestRequest):
    """Run a new backtest"""
    try:
        backtest_id = f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(100, 999)}"
        
        result = {
            "id": backtest_id,
            "strategy_id": backtest.strategy_id,
            "symbol": backtest.symbol,
            "start_date": backtest.start_date,
            "end_date": backtest.end_date,
            "initial_capital": backtest.initial_capital,
            "status": "running",
            "created_at": datetime.now().isoformat(),
            "estimated_completion": (datetime.now() + timedelta(minutes=15)).isoformat(),
            "message": "Backtest started successfully"
        }
        
        return result
    except Exception as e:
        logger.error(f"Failed to run backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/backtests/{backtest_id}")
async def get_backtest_results(backtest_id: str = Path(...)):
    """Get specific backtest results"""
    try:
        backtest_results = {
            "id": backtest_id,
            "strategy_name": "Advanced ES Futures Model",
            "symbol": "ES",
            "start_date": "2024-01-01",
            "end_date": "2024-09-27",
            "initial_capital": 100000.0,
            "final_value": 142350.0,
            "total_return": 0.4235,
            "annualized_return": 0.3187,
            "sharpe_ratio": 2.34,
            "sortino_ratio": 3.12,
            "max_drawdown": 0.087,
            "win_rate": 0.653,
            "profit_factor": 1.89,
            "total_trades": 147,
            "profitable_trades": 96,
            "losing_trades": 51,
            "avg_winning_trade": 1245.50,
            "avg_losing_trade": -673.25,
            "largest_winning_trade": 4250.00,
            "largest_losing_trade": -2150.00,
            "avg_trade_duration": "2.3 hours",
            "status": "completed",
            "created_at": "2024-09-25T14:30:00",
            "completed_at": "2024-09-25T14:45:00",
            "equity_curve": [
                {"date": "2024-01-01", "value": 100000.0, "drawdown": 0.0},
                {"date": "2024-01-15", "value": 102340.0, "drawdown": 0.0},
                {"date": "2024-02-01", "value": 105670.0, "drawdown": 0.0},
                {"date": "2024-03-01", "value": 108950.0, "drawdown": 0.0},
                {"date": "2024-04-01", "value": 112340.0, "drawdown": 0.0},
                {"date": "2024-05-01", "value": 109850.0, "drawdown": 0.022},
                {"date": "2024-06-01", "value": 116720.0, "drawdown": 0.0},
                {"date": "2024-07-01", "value": 121450.0, "drawdown": 0.0},
                {"date": "2024-08-01", "value": 125890.0, "drawdown": 0.0},
                {"date": "2024-09-01", "value": 138920.0, "drawdown": 0.0},
                {"date": "2024-09-27", "value": 142350.0, "drawdown": 0.0}
            ]
        }
        
        return backtest_results
    except Exception as e:
        logger.error(f"Failed to get backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/backtests/{backtest_id}/metrics")
async def get_backtest_metrics(backtest_id: str = Path(...)):
    """Get backtest performance metrics"""
    try:
        metrics = {
            "id": backtest_id,
            "performance_metrics": {
                "total_return": 0.4235,
                "annualized_return": 0.3187,
                "sharpe_ratio": 2.34,
                "sortino_ratio": 3.12,
                "calmar_ratio": 3.67,
                "max_drawdown": 0.087,
                "avg_drawdown": 0.023,
                "volatility": 0.185,
                "downside_deviation": 0.098
            },
            "trade_metrics": {
                "total_trades": 147,
                "winning_trades": 96,
                "losing_trades": 51,
                "win_rate": 0.653,
                "profit_factor": 1.89,
                "avg_trade": 287.76,
                "avg_winning_trade": 1245.50,
                "avg_losing_trade": -673.25,
                "largest_winning_trade": 4250.00,
                "largest_losing_trade": -2150.00,
                "consecutive_wins": 8,
                "consecutive_losses": 4
            },
            "risk_metrics": {
                "var_95": -1250.0,
                "cvar_95": -1875.0,
                "beta": 1.12,
                "alpha": 0.087,
                "tracking_error": 0.045,
                "information_ratio": 1.93
            },
            "last_updated": datetime.now().isoformat()
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Failed to get backtest metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# ML MODEL MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/api/models")
async def get_models():
    """List all ML models with full lifecycle management"""
    try:
        models = [
            {
                "id": "futures_es_advanced_v1",
                "name": "Advanced ES Futures Model",
                "symbol": "ES",
                "asset_type": "futures",
                "model_type": "xgboost",
                "accuracy": 0.742,
                "precision": 0.738,
                "recall": 0.745,
                "f1_score": 0.741,
                "sharpe_ratio": 2.34,
                "status": "active",
                "training_data_size": 50000,
                "features": ["price", "volume", "volatility", "rsi", "macd", "bollinger_bands"],
                "last_trained": "2024-09-25T14:30:00",
                "created_at": "2024-09-20T10:15:00",
                "version": "1.2.3"
            },
            {
                "id": "crypto_btc_predictor_v2",
                "name": "BTC Price Predictor",
                "symbol": "BTC-USD",
                "asset_type": "crypto",
                "model_type": "neural_network",
                "accuracy": 0.689,
                "precision": 0.692,
                "recall": 0.685,
                "f1_score": 0.688,
                "sharpe_ratio": 1.87,
                "status": "training",
                "training_data_size": 75000,
                "features": ["price", "volume", "sentiment", "on_chain_metrics", "technical_indicators"],
                "last_trained": "2024-09-24T09:15:00",
                "created_at": "2024-08-15T16:45:00",
                "version": "2.1.0",
                "training_progress": 0.65,
                "estimated_completion": "2024-09-27T16:30:00"
            },
            {
                "id": "equity_spy_momentum",
                "name": "SPY Momentum Model",
                "symbol": "SPY",
                "asset_type": "equity",
                "model_type": "lightgbm",
                "accuracy": 0.705,
                "precision": 0.698,
                "recall": 0.712,
                "f1_score": 0.705,
                "sharpe_ratio": 2.01,
                "status": "active",
                "training_data_size": 60000,
                "features": ["price", "volume", "momentum", "mean_reversion", "sector_rotation"],
                "last_trained": "2024-09-23T16:45:00",
                "created_at": "2024-07-10T11:20:00",
                "version": "1.5.2"
            },
            {
                "id": "multi_asset_ensemble_v1",
                "name": "Multi-Asset Ensemble Model", 
                "symbol": "MULTI",
                "asset_type": "mixed",
                "model_type": "ensemble",
                "accuracy": 0.721,
                "precision": 0.718,
                "recall": 0.724,
                "f1_score": 0.721,
                "sharpe_ratio": 2.18,
                "status": "inactive",
                "training_data_size": 100000,
                "features": ["cross_asset_momentum", "volatility_regime", "correlation_matrix", "macro_indicators"],
                "last_trained": "2024-09-18T13:22:00",
                "created_at": "2024-08-01T09:30:00",
                "version": "1.0.0"
            }
        ]
        
        return {
            "models": models,
            "total_models": len(models),
            "active_models": len([m for m in models if m["status"] == "active"]),
            "training_models": len([m for m in models if m["status"] == "training"]),
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models")
async def create_model(model: ModelRequest):
    """Create/train a new ML model"""
    try:
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        new_model = {
            "id": model_id,
            "name": model.name,
            "symbol": model.symbol,
            "asset_type": model.asset_type,
            "model_type": model.model_type,
            "parameters": model.parameters,
            "status": "training",
            "created_at": datetime.now().isoformat(),
            "estimated_completion": (datetime.now() + timedelta(hours=2)).isoformat(),
            "training_progress": 0.0,
            "message": "Model training started successfully"
        }
        
        return new_model
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/{model_id}")
async def get_model_details(model_id: str = Path(...)):
    """Get model details and status"""
    try:
        model_details = {
            "id": model_id,
            "name": "Advanced ES Futures Model",
            "symbol": "ES",
            "asset_type": "futures",
            "model_type": "xgboost",
            "status": "active",
            "version": "1.2.3",
            "performance_metrics": {
                "accuracy": 0.742,
                "precision": 0.738,
                "recall": 0.745,
                "f1_score": 0.741,
                "auc_roc": 0.834,
                "sharpe_ratio": 2.34,
                "max_drawdown": 0.087
            },
            "training_info": {
                "training_data_size": 50000,
                "validation_data_size": 12500,
                "test_data_size": 12500,
                "training_duration": "45 minutes",
                "last_trained": "2024-09-25T14:30:00",
                "epochs": 100,
                "early_stopping": True
            },
            "features": {
                "total_features": 15,
                "feature_list": ["price", "volume", "volatility", "rsi", "macd", "bollinger_bands", "stochastic", "momentum", "price_change", "volume_change", "volatility_change", "hour_of_day", "day_of_week", "month_of_year", "quarter"],
                "feature_importance": {
                    "price": 0.18,
                    "volume": 0.15,
                    "volatility": 0.12,
                    "rsi": 0.11,
                    "macd": 0.10,
                    "bollinger_bands": 0.08,
                    "other": 0.26
                }
            },
            "hyperparameters": {
                "n_estimators": 500,
                "max_depth": 8,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8
            },
            "deployment_info": {
                "deployed_at": "2024-09-25T15:00:00",
                "prediction_endpoint": f"/api/models/{model_id}/predict",
                "avg_prediction_time": "2.3ms",
                "total_predictions": 12547
            }
        }
        
        return model_details
    except Exception as e:
        logger.error(f"Failed to get model details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/{model_id}/predict")
async def run_prediction(prediction_data: Dict[str, Any], model_id: str = Path(...)):
    """Run predictions using the model"""
    try:
        # Mock prediction result
        prediction = {
            "model_id": model_id,
            "prediction": {
                "signal": random.choice(["buy", "sell", "hold"]),
                "confidence": round(random.uniform(0.6, 0.95), 3),
                "probability": {
                    "buy": round(random.uniform(0.2, 0.4), 3),
                    "sell": round(random.uniform(0.2, 0.4), 3),
                    "hold": round(random.uniform(0.3, 0.6), 3)
                },
                "target_price": round(random.uniform(4400, 4500), 2),
                "stop_loss": round(random.uniform(4350, 4400), 2),
                "risk_score": round(random.uniform(0.1, 0.8), 3),
                "predicted_at": datetime.now().isoformat()
            },
            "input_features": prediction_data,
            "model_version": "1.2.3",
            "prediction_time_ms": round(random.uniform(1.5, 3.5), 1)
        }
        
        return prediction
    except Exception as e:
        logger.error(f"Failed to run prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str = Path(...)):
    """Delete an ML model"""
    try:
        return {
            "id": model_id,
            "deleted_at": datetime.now().isoformat(),
            "message": "Model deleted successfully"
        }
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# SYSTEM CONTROL ENDPOINTS
# =============================================================================

@app.get("/api/system/status")
async def get_system_status():
    """Get system health and status"""
    try:
        return {
            "system_status": "online",
            "version": "1.0.0",
            "uptime": "3 days, 14 hours, 22 minutes",
            "services": {
                "trading_engine": {
                    "status": "active",
                    "last_heartbeat": datetime.now().isoformat(),
                    "active_strategies": 3,
                    "total_positions": 4
                },
                "market_data": {
                    "status": "active", 
                    "last_update": datetime.now().isoformat(),
                    "data_sources": ["binance", "alpaca", "polygon"],
                    "symbols_tracked": 15
                },
                "ml_models": {
                    "status": "active",
                    "active_models": 3,
                    "training_models": 1,
                    "total_predictions_today": 2847
                },
                "backtesting": {
                    "status": "active",
                    "running_backtests": 1,
                    "completed_backtests_today": 5
                },
                "database": {
                    "status": "healthy",
                    "connections": 12,
                    "query_response_time": "1.2ms"
                },
                "risk_management": {
                    "status": "active",
                    "daily_loss_limit": 0.02,
                    "current_drawdown": 0.003,
                    "position_limit_used": 0.75
                }
            },
            "performance": {
                "cpu_usage": 0.45,
                "memory_usage": 0.68,
                "disk_usage": 0.32,
                "network_latency": "12ms"
            },
            "today_stats": {
                "total_trades": 47,
                "profitable_trades": 29,
                "total_pnl": 2847.50,
                "win_rate": 0.617
            },
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/system/start")
async def start_trading_systems():
    """Start trading systems"""
    try:
        return {
            "action": "start_systems",
            "status": "success",
            "services_started": [
                "trading_engine",
                "market_data_feed",
                "risk_management",
                "strategy_executor"
            ],
            "started_at": datetime.now().isoformat(),
            "message": "All trading systems started successfully"
        }
    except Exception as e:
        logger.error(f"Failed to start trading systems: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/system/stop")
async def stop_trading_systems():
    """Stop trading systems"""
    try:
        return {
            "action": "stop_systems",
            "status": "success",
            "services_stopped": [
                "trading_engine",
                "strategy_executor",
                "automated_trading"
            ],
            "stopped_at": datetime.now().isoformat(),
            "message": "Trading systems stopped successfully. Market data and monitoring continue running."
        }
    except Exception as e:
        logger.error(f"Failed to stop trading systems: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )