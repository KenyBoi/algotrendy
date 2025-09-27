"""
AI Orchestrator REST API

FastAPI-based REST API for the AI Orchestrator Module.
Provides endpoints for monitoring, querying, and managing AI providers.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uvicorn
import logging

from ai_orchestrator import (
    get_ai_orchestrator, AIQuery, QueryType, ProviderStatus,
    AIResponse, ProviderMetrics
)

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import trading system components conditionally to handle missing dependencies
try:
    from advanced_ml_trainer import AdvancedMLTrainer
    ADVANCED_ML_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Advanced ML trainer not available: {e}")
    ADVANCED_ML_AVAILABLE = False

try:
    from backtester import Backtester  
    BACKTESTER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Backtester not available: {e}")
    BACKTESTER_AVAILABLE = False

try:
    from ai_futures_strategy_agent import AIFuturesStrategyAgent
    FUTURES_STRATEGIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Futures strategies not available: {e}")
    FUTURES_STRATEGIES_AVAILABLE = False

try:
    from ai_crypto_strategy_agent import AICryptoStrategyAgent
    CRYPTO_STRATEGIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Crypto strategies not available: {e}")
    CRYPTO_STRATEGIES_AVAILABLE = False

# Create FastAPI app
app = FastAPI(
    title="AI Orchestrator API",
    description="REST API for monitoring and managing AI providers in AlgoTrendy",
    version="1.0.0"
)

# Add CORS middleware for Retool
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Retool domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator = get_ai_orchestrator()


# Pydantic models for API
class QueryRequest(BaseModel):
    query: str = Field(..., description="The AI query text")
    query_type: str = Field(..., description="Type of query (analysis, strategy, conversation, etc.)")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    user_id: Optional[str] = Field(None, description="User identifier")
    max_cost: Optional[float] = Field(None, description="Maximum cost limit")
    speed_priority: Optional[str] = Field("balanced", description="Speed priority (fast, balanced, quality)")


class QueryResponse(BaseModel):
    content: str
    provider: str
    confidence: float
    cost: float
    processing_time: float
    tokens_used: int
    timestamp: datetime
    metadata: Dict[str, Any]


class ProviderStatusResponse(BaseModel):
    provider: str
    status: str
    response_time: float
    error_rate: float
    cost_per_query: float
    success_rate: float
    consecutive_failures: int
    last_health_check: datetime


class OrchestratorMetrics(BaseModel):
    total_queries: int
    total_cost: float
    provider_usage: Dict[str, int]
    query_types: Dict[str, int]
    errors: int


class ComparisonRequest(BaseModel):
    query: str
    query_type: str
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    providers: Optional[List[str]] = Field(None, description="Specific providers to compare")


class ComparisonResponse(BaseModel):
    query: str
    responses: Dict[str, QueryResponse]
    consensus_score: Optional[float] = None
    best_provider: Optional[str] = None


# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "AI Orchestrator API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Process an AI query through the orchestrator"""
    try:
        # Convert string query_type to enum
        try:
            query_type_enum = QueryType(request.query_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid query_type. Must be one of: {[t.value for t in QueryType]}"
            )

        # Create AIQuery object
        ai_query = AIQuery(
            query=request.query,
            query_type=query_type_enum,
            context=request.context or {},
            user_id=request.user_id,
            max_cost=request.max_cost,
            speed_priority=request.speed_priority or "balanced"
        )

        # Start orchestrator if not already started
        if not hasattr(orchestrator, '_health_monitor_task') or orchestrator._health_monitor_task.done():
            background_tasks.add_task(orchestrator.start)

        # Process query
        response = await orchestrator.process_query(ai_query)

        return QueryResponse(
            content=response.content,
            provider=response.provider,
            confidence=response.confidence,
            cost=response.cost,
            processing_time=response.processing_time,
            tokens_used=response.tokens_used,
            timestamp=response.timestamp,
            metadata=response.metadata
        )

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=ComparisonResponse)
async def compare_providers(request: ComparisonRequest, background_tasks: BackgroundTasks):
    """Compare responses from multiple AI providers"""
    try:
        # Convert string query_type to enum
        try:
            query_type_enum = QueryType(request.query_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid query_type. Must be one of: {[t.value for t in QueryType]}"
            )

        # Create AIQuery object
        ai_query = AIQuery(
            query=request.query,
            query_type=query_type_enum,
            context=request.context or {}
        )

        # Start orchestrator if needed
        if not hasattr(orchestrator, '_health_monitor_task') or orchestrator._health_monitor_task.done():
            background_tasks.add_task(orchestrator.start)

        # Get comparison results
        responses = await orchestrator.compare_providers(ai_query, request.providers)

        # Convert responses to API format
        api_responses = {}
        for provider_name, response in responses.items():
            if response:
                api_responses[provider_name] = QueryResponse(
                    content=response.content,
                    provider=response.provider,
                    confidence=response.confidence,
                    cost=response.cost,
                    processing_time=response.processing_time,
                    tokens_used=response.tokens_used,
                    timestamp=response.timestamp,
                    metadata=response.metadata
                )
            else:
                api_responses[provider_name] = None

        # Calculate simple consensus (highest confidence)
        valid_responses = {k: v for k, v in api_responses.items() if v is not None}
        if valid_responses:
            best_provider = max(valid_responses.keys(),
                              key=lambda x: valid_responses[x].confidence)
            consensus_score = sum(r.confidence for r in valid_responses.values()) / len(valid_responses)
        else:
            best_provider = None
            consensus_score = 0.0

        return ComparisonResponse(
            query=request.query,
            responses=api_responses,
            consensus_score=consensus_score,
            best_provider=best_provider
        )

    except Exception as e:
        logger.error(f"Provider comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/providers", response_model=List[ProviderStatusResponse])
async def get_provider_status():
    """Get status of all AI providers"""
    try:
        status = orchestrator.get_provider_status()

        return [
            ProviderStatusResponse(
                provider=name,
                status=metrics.status.value,
                response_time=metrics.response_time,
                error_rate=metrics.error_rate,
                cost_per_query=metrics.cost_per_query,
                success_rate=metrics.success_rate,
                consecutive_failures=metrics.consecutive_failures,
                last_health_check=metrics.last_health_check
            )
            for name, metrics in status.items()
        ]

    except Exception as e:
        logger.error(f"Failed to get provider status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=OrchestratorMetrics)
async def get_metrics():
    """Get orchestrator metrics"""
    try:
        metrics = orchestrator.get_metrics()

        return OrchestratorMetrics(
            total_queries=metrics.get('total_queries', 0),
            total_cost=metrics.get('total_cost', 0.0),
            provider_usage=metrics.get('provider_usage', {}),
            query_types=metrics.get('query_types', {}),
            errors=metrics.get('errors', 0)
        )

    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/providers/{provider_name}/health-check")
async def trigger_health_check(provider_name: str):
    """Manually trigger health check for a specific provider"""
    try:
        if provider_name not in orchestrator.providers:
            raise HTTPException(status_code=404, detail=f"Provider {provider_name} not found")

        provider = orchestrator.providers[provider_name]
        status = await provider.health_check()
        provider.metrics.status = status
        provider.metrics.last_health_check = datetime.utcnow()

        return {"provider": provider_name, "status": status.value, "timestamp": datetime.utcnow()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed for {provider_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query-types")
async def get_query_types():
    """Get available query types"""
    return {"query_types": [t.value for t in QueryType]}


@app.get("/providers/list")
async def list_providers():
    """List all available providers"""
    return {"providers": list(orchestrator.providers.keys())}


# Trading System API Endpoints
@app.get("/trading/models")
async def get_ml_models():
    """Get available ML models and their performance metrics"""
    try:
        # Return mock data for now (will be replaced with real data when components are available)
        models = [
            {
                "id": "futures_es_advanced_v1",
                "name": "Advanced ES Futures Model",
                "symbol": "ES",
                "asset_type": "futures",
                "accuracy": 0.847,
                "precision": 0.823,
                "recall": 0.856,
                "f1_score": 0.839,
                "sharpe_ratio": 2.34,
                "last_trained": "2024-09-25T14:30:00Z",
                "status": "active"
            },
            {
                "id": "crypto_btc_scalp_v2",
                "name": "BTC Scalping Model",
                "symbol": "BTCUSD",
                "asset_type": "crypto",
                "accuracy": 0.781,
                "precision": 0.768,
                "recall": 0.795,
                "f1_score": 0.781,
                "sharpe_ratio": 1.87,
                "last_trained": "2024-09-24T09:15:00Z",
                "status": "active"
            },
            {
                "id": "stocks_aapl_swing_v1",
                "name": "AAPL Swing Trading Model",
                "symbol": "AAPL",
                "asset_type": "stock",
                "accuracy": 0.723,
                "precision": 0.715,
                "recall": 0.738,
                "f1_score": 0.726,
                "sharpe_ratio": 1.45,
                "last_trained": "2024-09-23T16:45:00Z",
                "status": "training"
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
        strategies = []
        
        # Mock futures strategies data
        futures_strategies = [
            {
                "id": "futures_momentum_breakout",
                "name": "Futures Momentum Breakout",
                "description": "Captures explosive moves following consolidation periods",
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
                "description": "Mean reversion strategy for E-mini S&P 500 futures",
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
            }
        ]
        
        # Mock crypto strategies data
        crypto_strategies = [
            {
                "id": "mean_reversion_scalp",
                "name": "Mean Reversion Scalp",
                "description": "Identifies short-term deviations from moving averages for quick scalps",
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
                "description": "Follows strong momentum moves in Bitcoin with trend confirmation",
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
            }
        ]
        
        strategies.extend(futures_strategies)
        strategies.extend(crypto_strategies)
        
        return {"strategies": strategies, "total_strategies": len(strategies)}
        
    except Exception as e:
        logger.error(f"Failed to get trading strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trading/backtests")
async def get_backtest_results():
    """Get recent backtest results"""
    try:
        # Mock backtest results (in real implementation, this would read from saved results)
        backtests = [
            {
                "id": "bt_20240925_es_momentum",
                "strategy_name": "Futures Momentum Breakout",
                "symbol": "ES",
                "start_date": "2024-01-01",
                "end_date": "2024-09-25",
                "initial_capital": 100000,
                "final_value": 142350,
                "total_return": 0.4235,
                "sharpe_ratio": 2.34,
                "max_drawdown": 0.087,
                "win_rate": 0.67,
                "total_trades": 156,
                "avg_trade_duration": "4.2 hours",
                "status": "completed",
                "created_at": "2024-09-25T10:30:00Z"
            },
            {
                "id": "bt_20240924_btc_scalp",
                "strategy_name": "BTC Mean Reversion Scalp",
                "symbol": "BTCUSD",
                "start_date": "2024-08-01",
                "end_date": "2024-09-24",
                "initial_capital": 50000,
                "final_value": 68750,
                "total_return": 0.375,
                "sharpe_ratio": 1.87,
                "max_drawdown": 0.124,
                "win_rate": 0.61,
                "total_trades": 342,
                "avg_trade_duration": "8.5 minutes",
                "status": "completed",
                "created_at": "2024-09-24T15:45:00Z"
            },
            {
                "id": "bt_20240923_aapl_swing",
                "strategy_name": "AAPL ML Swing Trading",
                "symbol": "AAPL",
                "start_date": "2024-06-01",
                "end_date": "2024-09-23",
                "initial_capital": 100000,
                "final_value": 118500,
                "total_return": 0.185,
                "sharpe_ratio": 1.45,
                "max_drawdown": 0.065,
                "win_rate": 0.58,
                "total_trades": 28,
                "avg_trade_duration": "3.2 days",
                "status": "completed",
                "created_at": "2024-09-23T12:20:00Z"
            }
        ]
        
        return {"backtests": backtests, "total_backtests": len(backtests)}
        
    except Exception as e:
        logger.error(f"Failed to get backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trading/backtests/run")
async def run_backtest(request: dict):
    """Run a new backtest"""
    try:
        # Extract parameters
        strategy_id = request.get("strategy_id")
        symbol = request.get("symbol", "ES")
        start_date = request.get("start_date", "2024-01-01")
        end_date = request.get("end_date", "2024-09-27")
        initial_capital = request.get("initial_capital", 100000)
        
        # Mock backtest execution (in real implementation, this would run actual backtest)
        backtest_id = f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol.lower()}"
        
        result = {
            "backtest_id": backtest_id,
            "status": "running",
            "message": f"Backtest started for {strategy_id} on {symbol}",
            "estimated_completion": "2-3 minutes",
            "parameters": {
                "strategy_id": strategy_id,
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": initial_capital
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to start backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator on startup"""
    logger.info("Starting AI Orchestrator API...")
    await orchestrator.start()


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up orchestrator on shutdown"""
    logger.info("Shutting down AI Orchestrator API...")
    await orchestrator.stop()


if __name__ == "__main__":
    uvicorn.run(
        "ai_orchestrator_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )