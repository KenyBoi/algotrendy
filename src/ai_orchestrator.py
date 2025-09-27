"""
AI Orchestrator Module for AlgoTrendy

This module provides intelligent orchestration of multiple AI providers (Copilot, ChatGPT, Claude)
with load balancing, failover, and response optimization capabilities.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import hashlib
import json

import aiohttp
import redis.asyncio as redis
from pydantic import BaseModel, Field
import openai
import anthropic
from github import Github

# Simple config class for AI orchestrator
class Config:
    """Configuration for AI Orchestrator"""
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """AI provider types"""
    COPILOT = "copilot"
    CHATGPT = "chatgpt"
    CLAUDE = "claude"


class QueryType(Enum):
    """Types of AI queries for intelligent routing"""
    ANALYSIS = "analysis"
    STRATEGY = "strategy"
    CONVERSATION = "conversation"
    CODE_GENERATION = "code_generation"
    RISK_ASSESSMENT = "risk_assessment"
    MARKET_INSIGHT = "market_insight"


class ProviderStatus(Enum):
    """Provider health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class AIQuery:
    """Represents an AI query with context"""
    query: str
    query_type: QueryType
    context: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    max_cost: Optional[float] = None
    speed_priority: str = "balanced"  # "fast", "balanced", "quality"
    allowed_providers: List[str] = field(default_factory=list)


@dataclass
class AIResponse:
    """Standardized AI response format"""
    content: str
    provider: str
    confidence: float = 0.0
    cost: float = 0.0
    processing_time: float = 0.0
    tokens_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ProviderMetrics:
    """Provider performance metrics"""
    provider: str
    status: ProviderStatus
    response_time: float = 0.0
    error_rate: float = 0.0
    cost_per_query: float = 0.0
    success_rate: float = 1.0
    last_health_check: datetime = field(default_factory=datetime.utcnow)
    consecutive_failures: int = 0


class AIProviderAdapter(ABC):
    """Abstract base class for AI provider adapters"""

    def __init__(self, provider_name: str, config: Config):
        self.provider_name = provider_name
        self.config = config
        self.metrics = ProviderMetrics(provider=provider_name, status=ProviderStatus.HEALTHY)

    @abstractmethod
    async def query(self, ai_query: AIQuery) -> AIResponse:
        """Execute query against the AI provider"""
        pass

    @abstractmethod
    async def health_check(self) -> ProviderStatus:
        """Check provider health and availability"""
        pass

    @abstractmethod
    def estimate_cost(self, query: str) -> float:
        """Estimate cost for a given query"""
        pass

    @abstractmethod
    def get_rate_limits(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        pass


class CopilotAdapter(AIProviderAdapter):
    """GitHub Copilot provider adapter"""

    def __init__(self, config: Config):
        super().__init__("copilot", config)
        self.github_token = config.github_token
        self.github = Github(self.github_token) if self.github_token else None

    async def query(self, ai_query: AIQuery) -> AIResponse:
        """Query GitHub Copilot (simplified implementation)"""
        start_time = time.time()

        try:
            # Note: This is a simplified implementation
            # GitHub Copilot API is not publicly available
            # In practice, this would integrate with GitHub's Copilot API

            # For now, simulate a response
            response_content = f"Copilot analysis for: {ai_query.query[:100]}..."

            processing_time = time.time() - start_time

            return AIResponse(
                content=response_content,
                provider="copilot",
                confidence=0.85,
                cost=0.02,
                processing_time=processing_time,
                tokens_used=len(ai_query.query.split()) * 2
            )

        except Exception as e:
            logger.error(f"Copilot query failed: {e}")
            self.metrics.consecutive_failures += 1
            raise

    async def health_check(self) -> ProviderStatus:
        """Check Copilot API health"""
        try:
            if not self.github_token:
                return ProviderStatus.OFFLINE

            # Simple health check
            rate_limit = self.github.get_rate_limit()
            if rate_limit.core.remaining > 100:
                return ProviderStatus.HEALTHY
            elif rate_limit.core.remaining > 10:
                return ProviderStatus.DEGRADED
            else:
                return ProviderStatus.UNHEALTHY

        except Exception as e:
            logger.error(f"Copilot health check failed: {e}")
            return ProviderStatus.OFFLINE

    def estimate_cost(self, query: str) -> float:
        """Estimate Copilot query cost"""
        # Copilot pricing (example)
        return 0.02  # Fixed cost per query

    def get_rate_limits(self) -> Dict[str, Any]:
        """Get Copilot rate limit status"""
        try:
            if self.github:
                rate_limit = self.github.get_rate_limit()
                return {
                    "remaining": rate_limit.core.remaining,
                    "limit": rate_limit.core.limit,
                    "reset_time": rate_limit.core.reset.timestamp()
                }
        except Exception as e:
            logger.error(f"Failed to get Copilot rate limits: {e}")

        return {"remaining": 0, "limit": 5000, "reset_time": None}


class ChatGPTAdapter(AIProviderAdapter):
    """OpenAI ChatGPT provider adapter"""

    def __init__(self, config: Config):
        super().__init__("chatgpt", config)
        self.api_key = config.openai_api_key
        self.client = openai.AsyncOpenAI(api_key=self.api_key) if self.api_key else None

    async def query(self, ai_query: AIQuery) -> AIResponse:
        """Query OpenAI ChatGPT"""
        start_time = time.time()

        try:
            messages = [{"role": "user", "content": ai_query.query}]

            # Add context if available
            if ai_query.context:
                system_message = f"You are an expert trading assistant. Context: {json.dumps(ai_query.context)}"
                messages.insert(0, {"role": "system", "content": system_message})

            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                max_tokens=2000,
                temperature=0.7
            )

            processing_time = time.time() - start_time

            return AIResponse(
                content=response.choices[0].message.content,
                provider="chatgpt",
                confidence=0.88,
                cost=self._calculate_cost(response.usage),
                processing_time=processing_time,
                tokens_used=response.usage.total_tokens,
                metadata={
                    "model": response.model,
                    "finish_reason": response.choices[0].finish_reason
                }
            )

        except Exception as e:
            logger.error(f"ChatGPT query failed: {e}")
            self.metrics.consecutive_failures += 1
            raise

    async def health_check(self) -> ProviderStatus:
        """Check OpenAI API health"""
        try:
            if not self.api_key:
                return ProviderStatus.OFFLINE

            # Simple health check via models endpoint
            await self.client.models.list()
            return ProviderStatus.HEALTHY

        except Exception as e:
            logger.error(f"ChatGPT health check failed: {e}")
            return ProviderStatus.OFFLINE

    def estimate_cost(self, query: str) -> float:
        """Estimate ChatGPT query cost"""
        # GPT-4 Turbo pricing: $0.01/1K input tokens, $0.03/1K output tokens
        estimated_tokens = len(query.split()) * 1.5  # Rough estimate
        return (estimated_tokens / 1000) * 0.01 + (estimated_tokens / 1000) * 0.03

    def _calculate_cost(self, usage) -> float:
        """Calculate actual cost from token usage"""
        input_cost = (usage.prompt_tokens / 1000) * 0.01
        output_cost = (usage.completion_tokens / 1000) * 0.03
        return input_cost + output_cost

    def get_rate_limits(self) -> Dict[str, Any]:
        """Get ChatGPT rate limit status"""
        # OpenAI doesn't provide real-time rate limit info in API
        return {"remaining": 1000, "limit": 10000, "reset_time": None}


class ClaudeAdapter(AIProviderAdapter):
    """Anthropic Claude provider adapter"""

    def __init__(self, config: Config):
        super().__init__("claude", config)
        self.api_key = config.anthropic_api_key
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key) if self.api_key else None

    async def query(self, ai_query: AIQuery) -> AIResponse:
        """Query Anthropic Claude"""
        start_time = time.time()

        try:
            system_prompt = "You are an expert trading assistant with deep knowledge of financial markets, technical analysis, and risk management."

            # Add context to system prompt
            if ai_query.context:
                system_prompt += f" Additional context: {json.dumps(ai_query.context)}"

            response = await self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                temperature=0.7,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": ai_query.query}
                ]
            )

            processing_time = time.time() - start_time

            return AIResponse(
                content=response.content[0].text,
                provider="claude",
                confidence=0.92,
                cost=self._calculate_cost(response.usage),
                processing_time=processing_time,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                metadata={
                    "model": response.model,
                    "stop_reason": response.stop_reason
                }
            )

        except Exception as e:
            logger.error(f"Claude query failed: {e}")
            self.metrics.consecutive_failures += 1
            raise

    async def health_check(self) -> ProviderStatus:
        """Check Claude API health"""
        try:
            if not self.api_key:
                return ProviderStatus.OFFLINE

            # Simple health check
            # Note: Anthropic doesn't have a direct health check endpoint
            # This is a basic connectivity test
            return ProviderStatus.HEALTHY

        except Exception as e:
            logger.error(f"Claude health check failed: {e}")
            return ProviderStatus.OFFLINE

    def estimate_cost(self, query: str) -> float:
        """Estimate Claude query cost"""
        # Claude pricing: $15/1M input tokens, $75/1M output tokens
        estimated_tokens = len(query.split()) * 1.5
        return (estimated_tokens / 1000000) * 15 + (estimated_tokens / 1000000) * 75

    def _calculate_cost(self, usage) -> float:
        """Calculate actual cost from token usage"""
        input_cost = (usage.input_tokens / 1000000) * 15
        output_cost = (usage.output_tokens / 1000000) * 75
        return input_cost + output_cost

    def get_rate_limits(self) -> Dict[str, Any]:
        """Get Claude rate limit status"""
        # Anthropic provides rate limit headers in responses
        return {"remaining": 1000, "limit": 1000, "reset_time": None}


class AILoadBalancer:
    """Load balancer for AI provider distribution"""

    def __init__(self):
        self.provider_usage = {}
        self.provider_limits = {
            'copilot': 50,    # requests per minute
            'chatgpt': 100,
            'claude': 50
        }

    def select_provider(self, available_providers: List[str], query_type: QueryType) -> str:
        """Select best provider based on load and query type"""
        # Task-based routing
        if query_type == QueryType.CODE_GENERATION:
            return 'copilot'
        elif query_type == QueryType.CONVERSATION:
            return 'chatgpt'
        elif query_type in [QueryType.RISK_ASSESSMENT, QueryType.ANALYSIS]:
            return 'claude'

        # Load-based routing for other queries
        return self._round_robin_select(available_providers)

    def _round_robin_select(self, available_providers: List[str]) -> str:
        """Simple round-robin selection"""
        if not available_providers:
            raise ValueError("No providers available")

        # Find provider with lowest usage
        min_usage = float('inf')
        selected_provider = available_providers[0]

        for provider in available_providers:
            usage = self.provider_usage.get(provider, 0)
            if usage < min_usage:
                min_usage = usage
                selected_provider = provider

        self.provider_usage[selected_provider] = self.provider_usage.get(selected_provider, 0) + 1
        return selected_provider


class AICache:
    """Intelligent caching for AI responses"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.cache_ttl = 3600  # 1 hour

    def _generate_cache_key(self, ai_query: AIQuery) -> str:
        """Generate cache key from query"""
        key_data = {
            'query': ai_query.query,
            'query_type': ai_query.query_type.value,
            'context': ai_query.context
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return f"ai_cache:{hashlib.md5(key_string.encode()).hexdigest()}"

    async def get(self, ai_query: AIQuery) -> Optional[AIResponse]:
        """Get cached response if available"""
        try:
            cache_key = self._generate_cache_key(ai_query)
            cached_data = await self.redis.get(cache_key)

            if cached_data:
                response_data = json.loads(cached_data)
                response_data['timestamp'] = datetime.fromisoformat(response_data['timestamp'])
                return AIResponse(**response_data)

        except Exception as e:
            logger.error(f"Cache get failed: {e}")

        return None

    async def set(self, ai_query: AIQuery, response: AIResponse) -> None:
        """Cache AI response"""
        try:
            cache_key = self._generate_cache_key(ai_query)
            response_data = {
                'content': response.content,
                'provider': response.provider,
                'confidence': response.confidence,
                'cost': response.cost,
                'processing_time': response.processing_time,
                'tokens_used': response.tokens_used,
                'metadata': response.metadata,
                'timestamp': response.timestamp.isoformat()
            }

            await self.redis.setex(cache_key, self.cache_ttl, json.dumps(response_data))

        except Exception as e:
            logger.error(f"Cache set failed: {e}")


class AIMetrics:
    """AI usage metrics and analytics"""

    def __init__(self):
        self.metrics = {
            'total_queries': 0,
            'total_cost': 0.0,
            'provider_usage': {},
            'query_types': {},
            'errors': 0
        }

    async def record_query(self, ai_query: AIQuery, response: AIResponse) -> None:
        """Record query metrics"""
        self.metrics['total_queries'] += 1
        self.metrics['total_cost'] += response.cost

        # Provider usage
        provider = response.provider
        if provider not in self.metrics['provider_usage']:
            self.metrics['provider_usage'][provider] = 0
        self.metrics['provider_usage'][provider] += 1

        # Query type distribution
        query_type = ai_query.query_type.value
        if query_type not in self.metrics['query_types']:
            self.metrics['query_types'][query_type] = 0
        self.metrics['query_types'][query_type] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()


class AIOrchestrator:
    """Main AI Orchestrator class"""

    def __init__(self, config: Config):
        self.config = config
        self.providers = {
            'copilot': CopilotAdapter(config),
            'chatgpt': ChatGPTAdapter(config),
            'claude': ClaudeAdapter(config)
        }
        self.load_balancer = AILoadBalancer()
        self.cache = AICache(config.redis_url if hasattr(config, 'redis_url') else "redis://localhost:6379")
        self.metrics = AIMetrics()

        # Health monitoring
        self.health_check_interval = 60  # seconds
        self._health_monitor_task = None

    async def start(self):
        """Start the orchestrator"""
        logger.info("Starting AI Orchestrator...")
        self._health_monitor_task = asyncio.create_task(self._health_monitor())

    async def stop(self):
        """Stop the orchestrator"""
        logger.info("Stopping AI Orchestrator...")
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass

    async def process_query(self, ai_query: AIQuery) -> AIResponse:
        """Process an AI query with intelligent routing and failover"""
        logger.info(f"Processing query: {ai_query.query[:50]}...")

        # Check cache first
        cached_response = await self.cache.get(ai_query)
        if cached_response:
            logger.info("Returning cached response")
            return cached_response

        # Select provider
        provider_name = self._select_provider(ai_query)

        # Execute with failover
        response = await self._execute_with_failover(provider_name, ai_query)

        # Cache successful response
        await self.cache.set(ai_query, response)

        # Record metrics
        await self.metrics.record_query(ai_query, response)

        return response

    async def compare_providers(self, ai_query: AIQuery, providers: List[str] = None) -> Dict[str, AIResponse]:
        """Compare responses from multiple providers"""
        if providers is None:
            providers = list(self.providers.keys())

        tasks = []
        for provider_name in providers:
            if provider_name in self.providers:
                task = self._execute_with_failover(provider_name, ai_query)
                tasks.append((provider_name, task))

        results = {}
        for provider_name, task in tasks:
            try:
                response = await task
                results[provider_name] = response
            except Exception as e:
                logger.error(f"Provider {provider_name} failed: {e}")
                results[provider_name] = None

        return results

    def _select_provider(self, ai_query: AIQuery) -> str:
        """Select the best provider for the query"""
        # Check user preferences
        if ai_query.allowed_providers:
            available_providers = ai_query.allowed_providers
        else:
            available_providers = [p for p, adapter in self.providers.items()
                                 if adapter.metrics.status in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED]]

        if not available_providers:
            raise ValueError("No healthy providers available")

        # Cost check
        if ai_query.max_cost is not None:
            affordable_providers = []
            for provider_name in available_providers:
                estimated_cost = self.providers[provider_name].estimate_cost(ai_query.query)
                if estimated_cost <= ai_query.max_cost:
                    affordable_providers.append(provider_name)
            if affordable_providers:
                available_providers = affordable_providers

        return self.load_balancer.select_provider(available_providers, ai_query.query_type)

    async def _execute_with_failover(self, primary_provider: str, ai_query: AIQuery) -> AIResponse:
        """Execute query with automatic failover"""
        tried_providers = set()

        while len(tried_providers) < len(self.providers):
            current_provider = primary_provider if primary_provider not in tried_providers else None

            if current_provider is None:
                # Find next best provider
                available_providers = [p for p in self.providers.keys() if p not in tried_providers]
                if not available_providers:
                    break
                current_provider = self.load_balancer.select_provider(available_providers, ai_query.query_type)

            tried_providers.add(current_provider)

            try:
                provider = self.providers[current_provider]
                if provider.metrics.status in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED]:
                    response = await provider.query(ai_query)
                    # Reset consecutive failures on success
                    provider.metrics.consecutive_failures = 0
                    return response

            except Exception as e:
                logger.warning(f"Provider {current_provider} failed: {e}")
                provider.metrics.consecutive_failures += 1
                continue

        raise Exception("All providers failed")

    async def _health_monitor(self):
        """Background health monitoring"""
        while True:
            try:
                for provider_name, provider in self.providers.items():
                    status = await provider.health_check()
                    provider.metrics.status = status
                    provider.metrics.last_health_check = datetime.utcnow()

                    # Update metrics
                    provider.metrics.response_time = 0.0  # Would be updated from actual queries

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(self.health_check_interval)

    def get_provider_status(self) -> Dict[str, ProviderMetrics]:
        """Get status of all providers"""
        return {name: provider.metrics for name, provider in self.providers.items()}

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics"""
        return self.metrics.get_metrics()


# Global orchestrator instance
_orchestrator_instance = None

def get_ai_orchestrator(config: Config = None) -> AIOrchestrator:
    """Get or create AI orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        if config is None:
            config = Config()
        _orchestrator_instance = AIOrchestrator(config)
    return _orchestrator_instance