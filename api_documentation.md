# AlgoTrendy Microservices API Documentation

## Overview

This document provides comprehensive API documentation for the AlgoTrendy microservices architecture. All APIs follow RESTful principles with JSON payloads and use standard HTTP status codes.

## Authentication

All API endpoints require JWT authentication via the `Authorization: Bearer <token>` header.

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## Common Response Formats

### Success Response
```json
{
  "success": true,
  "data": {...},
  "message": "Operation completed successfully"
}
```

### Error Response
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {...}
  }
}
```

### Pagination Response
```json
{
  "success": true,
  "data": [...],
  "pagination": {
    "page": 1,
    "limit": 50,
    "total": 150,
    "total_pages": 3,
    "has_next": true,
    "has_prev": false
  }
}
```

## 1. API Gateway Endpoints

### Authentication
```http
POST /auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "password123"
}

Response:
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
    "expires_in": 3600,
    "token_type": "Bearer"
  }
}
```

```http
POST /auth/refresh
Authorization: Bearer <refresh_token>

Response: Same as login
```

### Health Check
```http
GET /health

Response:
{
  "status": "healthy",
  "services": {
    "market-data": "healthy",
    "trading-engine": "healthy",
    "ml-model": "healthy"
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 2. Market Data Service API

### Get Historical Data
```http
GET /api/v1/market-data/{symbol}?period=1y&interval=1d&chart_style=time&asset_type=stock

Response:
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "period": "1y",
    "interval": "1d",
    "chart_style": "time",
    "data": [
      {
        "timestamp": "2023-01-01T00:00:00Z",
        "open": 150.0,
        "high": 155.0,
        "low": 148.0,
        "close": 152.0,
        "volume": 1000000,
        "vwap": 151.5,
        "rsi": 65.0,
        "macd": 1.2,
        "bb_upper": 160.0,
        "bb_middle": 150.0,
        "bb_lower": 140.0
      }
    ]
  }
}
```

### Batch Historical Data
```http
POST /api/v1/market-data/batch
Content-Type: application/json

{
  "symbols": ["AAPL", "GOOGL", "MSFT"],
  "period": "6mo",
  "interval": "1d",
  "include_indicators": true
}

Response:
{
  "success": true,
  "data": {
    "AAPL": [...],
    "GOOGL": [...],
    "MSFT": [...]
  }
}
```

### Real-time Data Subscription
```http
POST /api/v1/market-data/realtime/subscribe
Content-Type: application/json

{
  "symbols": ["AAPL", "SPY"],
  "fields": ["price", "volume", "vwap"],
  "update_interval": "1s"
}

Response:
{
  "success": true,
  "data": {
    "subscription_id": "sub_123456",
    "status": "active",
    "websocket_url": "ws://api.algotrendy.com/realtime/sub_123456"
  }
}
```

### Chart Style Transformation
```http
POST /api/v1/market-data/transform
Content-Type: application/json

{
  "symbol": "ES",
  "data": [...], // OHLCV data
  "chart_style": "renko",
  "params": {
    "brick_size": 1.0,
    "tick_based": true
  }
}

Response:
{
  "success": true,
  "data": {
    "chart_style": "renko",
    "data": [
      {
        "timestamp": "2024-01-01T10:00:00Z",
        "open": 4500.0,
        "high": 4505.0,
        "low": 4500.0,
        "close": 4505.0,
        "direction": 1,
        "brick_size": 5.0
      }
    ]
  }
}
```

### Blockchain Data (CovalentHQ Integration)
```http
GET /api/v1/market-data/blockchain/{chain_id}/token/{address}
    Query: ?start_date=2024-01-01&end_date=2024-01-31&interval=1d

Response:
{
  "success": true,
  "data": {
    "chain_id": 1,
    "token_address": "0xa0b86a33e6c0c5e4c8e4f8f0f8f8f8f8f8f8f8f8",
    "symbol": "UNI",
    "name": "Uniswap",
    "price_data": [
      {
        "timestamp": "2024-01-01T00:00:00Z",
        "price": 5.25,
        "volume_24h": 125000000,
        "market_cap": 3250000000,
        "price_change_24h": 2.15
      }
    ]
  }
}
```

```http
GET /api/v1/market-data/blockchain/{chain_id}/wallet/{address}/balances

Response:
{
  "success": true,
  "data": {
    "chain_id": 1,
    "wallet_address": "0x1234567890123456789012345678901234567890",
    "balances": [
      {
        "token_address": "0xa0b86a33e6c0c5e4c8e4f8f0f8f8f8f8f8f8f8f8",
        "symbol": "UNI",
        "balance": "1250.5",
        "quote": 6566.31,
        "price": 5.25
      }
    ],
    "total_quote": 125000.00
  }
}
```

```http
GET /api/v1/market-data/blockchain/{chain_id}/transactions
    Query: ?address=0x123...&start_block=18000000&end_block=18100000

Response:
{
  "success": true,
  "data": {
    "chain_id": 1,
    "transactions": [
      {
        "tx_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        "block_height": 18050000,
        "timestamp": "2024-01-01T12:30:00Z",
        "from": "0x1234567890123456789012345678901234567890",
        "to": "0xabcdef1234567890abcdef1234567890abcdef12",
        "value": "1.5",
        "gas_used": 21000,
        "gas_price": "20000000000",
        "token_transfers": [
          {
            "token_address": "0xa0b86a33e6c0c5e4c8e4f8f0f8f8f8f8f8f8f8f8",
            "from": "0x1234567890123456789012345678901234567890",
            "to": "0xabcdef1234567890abcdef1234567890abcdef12",
            "value": "500.0"
          }
        ]
      }
    ]
  }
}
```

## 3. ML Model Service API

### Train Model
```http
POST /api/v1/ml/models/train
Content-Type: application/json

{
  "name": "ES_Futures_Model",
  "symbol": "ES",
  "asset_type": "futures",
  "model_type": "ensemble",
  "training_config": {
    "period": "180d",
    "interval": "5m",
    "hyperparams": {
      "n_estimators": 100,
      "max_depth": 6,
      "learning_rate": 0.1
    },
    "feature_selection": {
      "method": "rfecv",
      "k": 40
    }
  }
}

Response:
{
  "success": true,
  "data": {
    "model_id": "model_123456",
    "status": "training",
    "estimated_completion": "2024-01-01T13:00:00Z",
    "progress_url": "/api/v1/ml/models/model_123456/progress"
  }
}
```

### Get Model Status
```http
GET /api/v1/ml/models/{model_id}/status

Response:
{
  "success": true,
  "data": {
    "model_id": "model_123456",
    "status": "completed",
    "progress": 1.0,
    "metrics": {
      "train_accuracy": 0.92,
      "validation_accuracy": 0.85,
      "test_accuracy": 0.82,
      "sharpe_ratio": 1.45,
      "max_drawdown": 0.12
    }
  }
}
```

### Model Inference
```http
POST /api/v1/ml/models/{model_id}/predict
Content-Type: application/json

{
  "features": {
    "close": 4500.0,
    "rsi": 65.0,
    "macd": 1.2,
    "bb_position": 0.8,
    "volume_zscore": 1.5
  },
  "prediction_horizon": 5
}

Response:
{
  "success": true,
  "data": {
    "prediction": 0.75,
    "confidence": 0.82,
    "probabilities": [0.18, 0.15, 0.12, 0.20, 0.35],
    "signal": "BUY",
    "expected_return": 0.008,
    "risk_score": 0.15
  }
}
```

### Batch Inference
```http
POST /api/v1/ml/models/{model_id}/predict/batch
Content-Type: application/json

{
  "features_batch": [
    {"close": 4500.0, "rsi": 65.0, ...},
    {"close": 4495.0, "rsi": 62.0, ...}
  ]
}

Response:
{
  "success": true,
  "data": {
    "predictions": [
      {"prediction": 0.75, "confidence": 0.82},
      {"prediction": 0.45, "confidence": 0.78}
    ]
  }
}
```

### Model Management
```http
GET /api/v1/ml/models?symbol=ES&status=active&limit=20

Response:
{
  "success": true,
  "data": [
    {
      "model_id": "model_123",
      "name": "ES_Ensemble_v1",
      "symbol": "ES",
      "accuracy": 0.82,
      "created_at": "2024-01-01T10:00:00Z",
      "status": "active"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 45,
    "total_pages": 3
  }
}
```

```http
DELETE /api/v1/ml/models/{model_id}

Response:
{
  "success": true,
  "message": "Model archived successfully"
}
```

## 4. Trading Engine Service API

### Submit Order
```http
POST /api/v1/trading/orders
Content-Type: application/json

{
  "portfolio_id": "port_123456",
  "symbol": "AAPL",
  "side": "BUY",
  "quantity": 100,
  "order_type": "MARKET",
  "time_in_force": "DAY",
  "broker": "alpaca"
}

Response:
{
  "success": true,
  "data": {
    "order_id": "ord_123456",
    "status": "submitted",
    "broker_order_id": "alpaca_ord_789",
    "estimated_fill_price": 150.25,
    "commission": 0.5
  }
}
```

### Get Order Status
```http
GET /api/v1/trading/orders/{order_id}

Response:
{
  "success": true,
  "data": {
    "order_id": "ord_123456",
    "portfolio_id": "port_123456",
    "symbol": "AAPL",
    "side": "BUY",
    "quantity": 100,
    "filled_quantity": 100,
    "avg_fill_price": 150.30,
    "status": "filled",
    "created_at": "2024-01-01T10:00:00Z",
    "filled_at": "2024-01-01T10:00:05Z"
  }
}
```

### Cancel Order
```http
PUT /api/v1/trading/orders/{order_id}/cancel
Content-Type: application/json

{
  "reason": "market_conditions_changed"
}

Response:
{
  "success": true,
  "data": {
    "order_id": "ord_123456",
    "status": "cancelled",
    "cancelled_at": "2024-01-01T10:01:00Z"
  }
}
```

### Position Management
```http
GET /api/v1/trading/positions/{portfolio_id}

Response:
{
  "success": true,
  "data": [
    {
      "symbol": "AAPL",
      "quantity": 150,
      "avg_price": 148.50,
      "current_price": 152.25,
      "unrealized_pnl": 570.00,
      "unrealized_pnl_percent": 2.56,
      "market_value": 22837.50
    }
  ]
}
```

```http
POST /api/v1/trading/positions/{position_id}/close
Content-Type: application/json

{
  "quantity": 50,
  "reason": "take_profit"
}

Response:
{
  "success": true,
  "data": {
    "closing_order_id": "ord_123457",
    "estimated_proceeds": 7612.50,
    "commission": 0.25
  }
}
```

### Portfolio Rebalancing
```http
POST /api/v1/trading/portfolio/{portfolio_id}/rebalance
Content-Type: application/json

{
  "target_allocations": {
    "AAPL": 0.30,
    "GOOGL": 0.25,
    "MSFT": 0.20,
    "BIL": 0.25  // Cash equivalent
  },
  "rebalance_method": "full_rebalance",
  "tax_optimization": true,
  "max_deviation": 0.05
}

Response:
{
  "success": true,
  "data": {
    "rebalance_id": "reb_123456",
    "orders": [
      {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 25,
        "estimated_price": 152.00
      },
      {
        "symbol": "GOOGL",
        "side": "SELL",
        "quantity": 10,
        "estimated_price": 2800.00
      }
    ],
    "estimated_cost": 125.50,
    "expected_completion": "2024-01-01T16:00:00Z"
  }
}
```

## 5. Strategy Engine Service API

### Create Strategy
```http
POST /api/v1/strategies
Content-Type: application/json

{
  "name": "Momentum RSI Strategy",
  "description": "Mean reversion strategy using RSI",
  "asset_class": "stocks",
  "parameters": {
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "holding_period": 5
  },
  "entry_conditions": [
    "rsi < rsi_oversold",
    "price > sma_20"
  ],
  "exit_conditions": [
    "rsi > rsi_overbought",
    "holding_period > 5"
  ]
}

Response:
{
  "success": true,
  "data": {
    "strategy_id": "strat_123456",
    "status": "created",
    "validation_status": "pending"
  }
}
```

### AI Strategy Discovery
```http
POST /api/v1/strategies/discover
Content-Type: application/json

{
  "asset_class": "futures",
  "time_horizon": "intraday",
  "risk_profile": "moderate",
  "performance_targets": {
    "min_sharpe": 1.5,
    "max_drawdown": 0.15
  },
  "discovery_method": "ai_agent",
  "search_space": {
    "indicators": ["rsi", "macd", "bb", "stoch"],
    "max_conditions": 3,
    "timeframes": ["5m", "15m", "1h"]
  }
}

Response:
{
  "success": true,
  "data": {
    "discovery_id": "disc_123456",
    "status": "running",
    "estimated_completion": "2024-01-01T12:30:00Z",
    "progress_url": "/api/v1/strategies/discovery/disc_123456/progress"
  }
}
```

### Backtest Strategy
```http
POST /api/v1/strategies/{strategy_id}/backtest
Content-Type: application/json

{
  "start_date": "2020-01-01",
  "end_date": "2024-01-01",
  "initial_capital": 100000,
  "commission": 0.0005,
  "slippage": 0.0001,
  "benchmark": "SPY",
  "walk_forward": {
    "enabled": true,
    "training_window": 252,
    "testing_window": 21
  }
}

Response:
{
  "success": true,
  "data": {
    "backtest_id": "bt_123456",
    "status": "running",
    "estimated_completion": "2024-01-01T14:00:00Z"
  }
}
```

### Get Backtest Results
```http
GET /api/v1/strategies/backtest/{backtest_id}/results

Response:
{
  "success": true,
  "data": {
    "backtest_id": "bt_123456",
    "strategy_id": "strat_123456",
    "period": {
      "start": "2020-01-01",
      "end": "2024-01-01"
    },
    "performance": {
      "total_return": 0.456,
      "annualized_return": 0.089,
      "sharpe_ratio": 1.67,
      "max_drawdown": 0.123,
      "win_rate": 0.58,
      "profit_factor": 1.45,
      "total_trades": 245,
      "avg_trade": 0.012
    },
    "risk_metrics": {
      "var_95": 0.023,
      "expected_shortfall": 0.035,
      "beta": 0.78,
      "alpha": 0.045
    },
    "benchmark_comparison": {
      "benchmark_return": 0.312,
      "outperformance": 0.144,
      "tracking_error": 0.089
    },
    "charts": {
      "equity_curve": "https://...",
      "drawdown_chart": "https://...",
      "monthly_returns": "https://..."
    }
  }
}
```

### Strategy Optimization
```http
POST /api/v1/strategies/{strategy_id}/optimize
Content-Type: application/json

{
  "parameters": [
    {
      "name": "rsi_period",
      "type": "int",
      "min": 10,
      "max": 20
    },
    {
      "name": "rsi_oversold",
      "type": "int",
      "min": 20,
      "max": 35
    }
  ],
  "optimization_method": "grid_search",
  "objective": "max_sharpe",
  "constraints": [
    "max_drawdown < 0.20",
    "win_rate > 0.50"
  ]
}

Response:
{
  "success": true,
  "data": {
    "optimization_id": "opt_123456",
    "status": "running",
    "estimated_completion": "2024-01-01T15:00:00Z"
  }
}
```

## 6. Risk Engine Service API

### Portfolio Risk Assessment
```http
GET /api/v1/risk/portfolio/{portfolio_id}?period=1y

Response:
{
  "success": true,
  "data": {
    "portfolio_id": "port_123456",
    "assessment_date": "2024-01-01T12:00:00Z",
    "risk_metrics": {
      "var_95": 0.023,
      "var_99": 0.045,
      "expected_shortfall": 0.035,
      "max_drawdown": 0.123,
      "sharpe_ratio": 1.67,
      "sortino_ratio": 1.45,
      "beta": 0.78,
      "alpha": 0.045
    },
    "position_risks": [
      {
        "symbol": "AAPL",
        "quantity": 150,
        "exposure": 22837.50,
        "contribution_to_risk": 0.35,
        "stress_test_impact": -0.08
      }
    ],
    "risk_limits": {
      "max_var_95": 0.05,
      "max_drawdown": 0.20,
      "max_concentration": 0.25,
      "min_liquidity": 0.10
    },
    "limit_breaches": []
  }
}
```

### Pre-trade Risk Check
```http
POST /api/v1/risk/pre-trade-check
Content-Type: application/json

{
  "portfolio_id": "port_123456",
  "order": {
    "symbol": "AAPL",
    "side": "BUY",
    "quantity": 100,
    "price": 152.00
  }
}

Response:
{
  "success": true,
  "data": {
    "approved": true,
    "risk_score": 0.15,
    "checks": [
      {
        "check_type": "position_limit",
        "status": "passed",
        "current_exposure": 0.18,
        "proposed_exposure": 0.22,
        "limit": 0.25
      },
      {
        "check_type": "portfolio_var",
        "status": "passed",
        "current_var": 0.023,
        "proposed_var": 0.025,
        "limit": 0.05
      },
      {
        "check_type": "liquidity",
        "status": "passed",
        "required_liquidity": 15200.00,
        "available_liquidity": 25000.00
      }
    ],
    "warnings": [
      "Position concentration increasing to 22% of portfolio"
    ]
  }
}
```

### Stress Testing
```http
POST /api/v1/risk/stress-test
Content-Type: application/json

{
  "portfolio_id": "port_123456",
  "scenarios": [
    {
      "name": "market_crash",
      "description": "2008-style market crash",
      "returns": {
        "SPY": -0.50,
        "QQQ": -0.55,
        "IWM": -0.60
      }
    },
    {
      "name": "tech_sector_drop",
      "description": "Technology sector -30%",
      "returns": {
        "AAPL": -0.30,
        "GOOGL": -0.30,
        "MSFT": -0.30
      }
    },
    {
      "name": "interest_rate_hike",
      "description": "Fed rate hike +100bps",
      "macro_variables": {
        "fed_funds_rate": 1.00
      }
    }
  ]
}

Response:
{
  "success": true,
  "data": {
    "stress_test_id": "stress_123456",
    "portfolio_value": 150000.00,
    "scenarios": [
      {
        "scenario_name": "market_crash",
        "portfolio_loss": -45000.00,
        "portfolio_loss_percent": -30.0,
        "var_breach": true,
        "liquidity_impact": -0.15,
        "worst_positions": [
          {"symbol": "AAPL", "loss": -8000.00, "loss_percent": -35.0}
        ]
      }
    ],
    "recommendations": [
      "Reduce technology sector exposure",
      "Increase cash position to 15%",
      "Consider hedging with put options"
    ]
  }
}
```

## 7. Portfolio Service API

### Get Portfolio Holdings
```http
GET /api/v1/portfolio/{portfolio_id}/holdings?include_performance=true

Response:
{
  "success": true,
  "data": {
    "portfolio_id": "port_123456",
    "total_value": 150000.00,
    "cash": 15000.00,
    "holdings": [
      {
        "symbol": "AAPL",
        "quantity": 150,
        "avg_cost": 148.50,
        "current_price": 152.25,
        "market_value": 22837.50,
        "unrealized_pnl": 570.00,
        "unrealized_pnl_percent": 2.56,
        "weight": 0.152,
        "sector": "Technology",
        "country": "USA"
      }
    ],
    "performance": {
      "daily_return": 0.015,
      "weekly_return": 0.032,
      "monthly_return": 0.089,
      "ytd_return": 0.156,
      "total_return": 0.456
    }
  }
}
```

### Portfolio Performance
```http
GET /api/v1/portfolio/{portfolio_id}/performance?period=1y&benchmark=SPY&frequency=daily

Response:
{
  "success": true,
  "data": {
    "period": "1y",
    "frequency": "daily",
    "portfolio_returns": [
      {"date": "2023-01-01", "return": 0.012},
      {"date": "2023-01-02", "return": -0.008}
    ],
    "benchmark_returns": [
      {"date": "2023-01-01", "return": 0.010},
      {"date": "2023-01-02", "return": -0.005}
    ],
    "metrics": {
      "total_return": 0.156,
      "annualized_return": 0.156,
      "volatility": 0.123,
      "sharpe_ratio": 1.27,
      "max_drawdown": 0.089,
      "beta": 0.95,
      "alpha": 0.023,
      "win_rate": 0.56,
      "profit_factor": 1.34
    },
    "charts": {
      "equity_curve": "https://charts.algotrendy.com/port_123456/equity_curve.png",
      "rolling_sharpe": "https://charts.algotrendy.com/port_123456/rolling_sharpe.png",
      "drawdown_chart": "https://charts.algotrendy.com/port_123456/drawdown.png"
    }
  }
}
```

### Performance Attribution
```http
GET /api/v1/portfolio/{portfolio_id}/attribution?period=1y&method=brinson

Response:
{
  "success": true,
  "data": {
    "method": "brinson",
    "period": "1y",
    "total_attribution": 0.023,
    "allocation_effect": 0.015,
    "selection_effect": 0.008,
    "interaction_effect": 0.000,
    "sector_attribution": [
      {
        "sector": "Technology",
        "weight": 0.35,
        "return": 0.189,
        "benchmark_return": 0.156,
        "attribution": 0.011
      }
    ],
    "security_attribution": [
      {
        "symbol": "AAPL",
        "weight": 0.152,
        "return": 0.234,
        "benchmark_return": 0.156,
        "attribution": 0.015
      }
    ]
  }
}
```

## 8. AI Agent Service API

### Conversational Interface
```http
POST /api/v1/ai/chat
Content-Type: application/json

{
  "message": "What should I do with my AAPL position? It's up 5% this week.",
  "context": {
    "portfolio_id": "port_123456",
    "user_id": "user_789",
    "conversation_id": "conv_123"
  },
  "preferences": {
    "risk_tolerance": "moderate",
    "investment_horizon": "long_term"
  }
}

Response:
{
  "success": true,
  "data": {
    "response": "Based on your moderate risk tolerance and the current market conditions, I recommend holding your AAPL position. The stock has strong fundamentals with consistent earnings growth, and the recent 5% gain is within normal volatility for this holding period.",
    "actions": [
      {
        "type": "analysis",
        "description": "Technical analysis shows AAPL is in an uptrend with support at $145",
        "confidence": 0.85
      },
      {
        "type": "recommendation",
        "action": "HOLD",
        "reasoning": "Position is performing well within your risk parameters",
        "time_horizon": "3-6 months"
      }
    ],
    "follow_up_questions": [
      "Are you concerned about any specific market risks?",
      "Do you have a target price for AAPL?"
    ],
    "conversation_id": "conv_123"
  }
}
```

### Strategy Recommendations
```http
GET /api/v1/ai/recommendations/{portfolio_id}?type=strategy&horizon=short_term

Response:
{
  "success": true,
  "data": {
    "portfolio_id": "port_123456",
    "recommendations": [
      {
        "type": "strategy_change",
        "strategy_name": "Defensive Momentum",
        "reasoning": "Market volatility increasing, suggesting more defensive positioning",
        "confidence": 0.78,
        "expected_impact": {
          "risk_reduction": 0.15,
          "return_impact": -0.05
        },
        "implementation": {
          "reduce_tech_exposure": 0.10,
          "increase_defensive_sectors": 0.10
        }
      },
      {
        "type": "rebalancing",
        "reasoning": "Portfolio drift from target allocations",
        "confidence": 0.92,
        "actions": [
          {
            "symbol": "AAPL",
            "action": "SELL",
            "quantity": 25,
            "reason": "Overweight position"
          }
        ]
      }
    ],
    "market_context": {
      "volatility_regime": "moderate",
      "trend": "sideways",
      "key_risks": ["interest_rates", "geopolitical"]
    }
  }
}
```

## 9. Backtesting Service API

### Run Backtest
```http
POST /api/v1/backtest
Content-Type: application/json

{
  "strategy_id": "strat_123456",
  "config": {
    "start_date": "2020-01-01",
    "end_date": "2024-01-01",
    "initial_capital": 100000,
    "commission": 0.0005,
    "slippage": 0.0001,
    "benchmark": "SPY",
    "universe": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
    "rebalance_frequency": "monthly",
    "max_position_size": 0.10,
    "risk_management": {
      "stop_loss": 0.05,
      "take_profit": 0.10,
      "max_drawdown": 0.20
    }
  }
}

Response:
{
  "success": true,
  "data": {
    "backtest_id": "bt_123456",
    "status": "queued",
    "estimated_duration": "45 minutes",
    "progress_url": "/api/v1/backtest/bt_123456/progress"
  }
}
```

### Backtest Results
```http
GET /api/v1/backtest/{backtest_id}/results

Response:
{
  "success": true,
  "data": {
    "backtest_id": "bt_123456",
    "status": "completed",
    "execution_time": "42 minutes",
    "config": {...},
    "summary": {
      "total_return": 0.456,
      "annualized_return": 0.089,
      "volatility": 0.123,
      "sharpe_ratio": 1.67,
      "max_drawdown": 0.123,
      "win_rate": 0.58,
      "profit_factor": 1.45,
      "total_trades": 245,
      "avg_trade_duration": "12 days"
    },
    "returns": {
      "daily": [...],
      "monthly": [...],
      "yearly": [...]
    },
    "risk_metrics": {
      "var_95": 0.023,
      "expected_shortfall": 0.035,
      "beta": 0.78,
      "tracking_error": 0.045
    },
    "trade_analysis": {
      "winning_trades": 142,
      "losing_trades": 103,
      "avg_win": 0.025,
      "avg_loss": -0.015,
      "largest_win": 0.089,
      "largest_loss": -0.034,
      "consecutive_wins": 8,
      "consecutive_losses": 5
    },
    "charts": {
      "equity_curve": "https://...",
      "drawdown": "https://...",
      "monthly_returns": "https://...",
      "trade_analysis": "https://..."
    }
  }
}
```

## 10. Notification Service API

### Send Notification
```http
POST /api/v1/notifications/send
Content-Type: application/json

{
  "type": "email",
  "recipient": "user@example.com",
  "subject": "Trade Executed Successfully",
  "template": "trade_confirmation",
  "data": {
    "portfolio_name": "Growth Portfolio",
    "symbol": "AAPL",
    "side": "BUY",
    "quantity": 100,
    "price": 152.30,
    "timestamp": "2024-01-01T10:05:00Z"
  },
  "channels": ["email", "push"],
  "priority": "normal"
}

Response:
{
  "success": true,
  "data": {
    "notification_id": "notif_123456",
    "status": "sent",
    "channels_used": ["email"],
    "estimated_delivery": "2024-01-01T10:05:30Z"
  }
}
```

### Create Alert Rule
```http
POST /api/v1/notifications/alerts
Content-Type: application/json

{
  "name": "Portfolio Risk Alert",
  "description": "Alert when portfolio VaR exceeds threshold",
  "portfolio_id": "port_123456",
  "condition": "portfolio_var_95 > 0.05",
  "channels": ["email", "sms"],
  "recipients": ["user@example.com", "+1234567890"],
  "frequency": "immediate",
  "cooldown_period": "1h",
  "enabled": true
}

Response:
{
  "success": true,
  "data": {
    "alert_id": "alert_123456",
    "status": "active",
    "next_check": "2024-01-01T11:00:00Z"
  }
}
```

## Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `VALIDATION_ERROR` | Invalid input parameters | 400 |
| `AUTHENTICATION_ERROR` | Invalid or missing authentication | 401 |
| `AUTHORIZATION_ERROR` | Insufficient permissions | 403 |
| `NOT_FOUND` | Resource not found | 404 |
| `CONFLICT` | Resource conflict | 409 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `INTERNAL_ERROR` | Server error | 500 |
| `SERVICE_UNAVAILABLE` | Service temporarily unavailable | 503 |

## Rate Limits

- **Market Data**: 1000 requests/minute per user
- **Trading Operations**: 100 requests/minute per user
- **ML Inference**: 500 requests/minute per user
- **Backtesting**: 10 concurrent backtests per user
- **AI Chat**: 50 messages/minute per user

## WebSocket APIs

### Real-time Market Data
```javascript
const ws = new WebSocket('ws://api.algotrendy.com/realtime/sub_123456');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Handle real-time price updates
  console.log(data);
};

// Message format
{
  "type": "price_update",
  "symbol": "AAPL",
  "data": {
    "price": 152.30,
    "change": 0.45,
    "change_percent": 0.30,
    "volume": 1250000,
    "timestamp": "2024-01-01T10:05:00Z"
  }
}
```

### Trading Notifications
```javascript
const ws = new WebSocket('ws://api.algotrendy.com/trading/notifications');

ws.onmessage = (event) => {
  const notification = JSON.parse(event.data);
  // Handle trading notifications
  console.log(notification);
};

// Message format
{
  "type": "order_filled",
  "order_id": "ord_123456",
  "data": {
    "symbol": "AAPL",
    "side": "BUY",
    "quantity": 100,
    "price": 152.30,
    "timestamp": "2024-01-01T10:05:00Z"
  }
}
```

This comprehensive API documentation provides all the necessary information for integrating with the AlgoTrendy microservices platform. Each endpoint includes detailed request/response formats, error handling, and usage examples.