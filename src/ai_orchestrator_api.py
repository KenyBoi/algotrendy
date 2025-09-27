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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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