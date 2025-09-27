"""
Test suite for AI Orchestrator Module

This module contains comprehensive tests for the AI Orchestrator functionality,
including provider adapters, load balancing, caching, and failover mechanisms.
"""

import asyncio
import unittest
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_orchestrator import (
    AIOrchestrator, AIQuery, AIResponse, QueryType, ProviderType,
    ProviderStatus, CopilotAdapter, ChatGPTAdapter, ClaudeAdapter,
    AILoadBalancer, AICache, AIMetrics, get_ai_orchestrator
)
from config import Config


class TestAIQuery(unittest.TestCase):
    """Test AIQuery data structure"""

    def test_ai_query_creation(self):
        """Test creating an AIQuery instance"""
        query = AIQuery(
            query="What is the current price of AAPL?",
            query_type=QueryType.ANALYSIS,
            context={"user_id": "user123", "portfolio_id": "port456"},
            user_id="user123",
            max_cost=0.10
        )

        self.assertEqual(query.query, "What is the current price of AAPL?")
        self.assertEqual(query.query_type, QueryType.ANALYSIS)
        self.assertEqual(query.context["user_id"], "user123")
        self.assertEqual(query.max_cost, 0.10)


class TestAIResponse(unittest.TestCase):
    """Test AIResponse data structure"""

    def test_ai_response_creation(self):
        """Test creating an AIResponse instance"""
        from datetime import datetime
        response = AIResponse(
            content="AAPL is currently trading at $150.25",
            provider="chatgpt",
            confidence=0.85,
            cost=0.023,
            processing_time=1.2,
            tokens_used=150
        )

        self.assertEqual(response.content, "AAPL is currently trading at $150.25")
        self.assertEqual(response.provider, "chatgpt")
        self.assertEqual(response.confidence, 0.85)
        self.assertEqual(response.cost, 0.023)
        self.assertIsInstance(response.timestamp, datetime)


class TestAILoadBalancer(unittest.TestCase):
    """Test AI Load Balancer functionality"""

    def setUp(self):
        self.load_balancer = AILoadBalancer()

    def test_round_robin_selection(self):
        """Test round-robin provider selection"""
        providers = ["copilot", "chatgpt", "claude"]

        # Test multiple selections
        selections = []
        for _ in range(6):
            selection = self.load_balancer._round_robin_select(providers)
            selections.append(selection)

        # Should cycle through providers
        self.assertIn("copilot", selections)
        self.assertIn("chatgpt", selections)
        self.assertIn("claude", selections)

    def test_task_based_routing(self):
        """Test task-based provider routing"""
        # Code generation should route to Copilot
        provider = self.load_balancer.select_provider(
            ["copilot", "chatgpt", "claude"],
            QueryType.CODE_GENERATION
        )
        self.assertEqual(provider, "copilot")

        # Conversation should route to ChatGPT
        provider = self.load_balancer.select_provider(
            ["copilot", "chatgpt", "claude"],
            QueryType.CONVERSATION
        )
        self.assertEqual(provider, "chatgpt")

        # Risk assessment should route to Claude
        provider = self.load_balancer.select_provider(
            ["copilot", "chatgpt", "claude"],
            QueryType.RISK_ASSESSMENT
        )
        self.assertEqual(provider, "claude")


class TestCopilotAdapter(unittest.TestCase):
    """Test Copilot adapter functionality"""

    def setUp(self):
        self.config = Config()
        # Mock config values for testing
        self.config.github_token = "mock_github_token"
        self.adapter = CopilotAdapter(self.config)

    @patch('ai_orchestrator.Github')
    def test_health_check_success(self, mock_github_class):
        """Test successful health check"""
        # Mock GitHub client
        mock_github = Mock()
        mock_rate_limit = Mock()
        mock_rate_limit.core.remaining = 1000
        mock_github.get_rate_limit.return_value = mock_rate_limit
        mock_github_class.return_value = mock_github

        # Reinitialize adapter with mocked GitHub
        self.adapter = CopilotAdapter(self.config)

        async def run_test():
            status = await self.adapter.health_check()
            self.assertEqual(status, ProviderStatus.HEALTHY)

        asyncio.run(run_test())

    def test_estimate_cost(self):
        """Test cost estimation"""
        query = "def fibonacci(n):"
        cost = self.adapter.estimate_cost(query)
        self.assertEqual(cost, 0.02)  # Fixed cost for Copilot


class TestChatGPTAdapter(unittest.TestCase):
    """Test ChatGPT adapter functionality"""

    def setUp(self):
        self.config = Config()
        self.config.openai_api_key = "mock_openai_key"
        self.adapter = ChatGPTAdapter(self.config)

    def test_estimate_cost(self):
        """Test cost estimation"""
        query = "What is the best trading strategy?"
        cost = self.adapter.estimate_cost(query)
        self.assertGreater(cost, 0)  # Should calculate based on tokens

    @patch('ai_orchestrator.openai.AsyncOpenAI')
    def test_query_success(self, mock_openai_class):
        """Test successful query execution"""
        # Mock OpenAI client and response
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Mock response"
        mock_response.model = "gpt-4-turbo"
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 100
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 50
        mock_response.choices[0].finish_reason = "stop"

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai_class.return_value = mock_client

        # Reinitialize adapter with mocked client
        self.adapter = ChatGPTAdapter(self.config)

        async def run_test():
            query = AIQuery(query="Test query", query_type=QueryType.ANALYSIS)
            response = await self.adapter.query(query)

            self.assertEqual(response.content, "Mock response")
            self.assertEqual(response.provider, "chatgpt")
            self.assertGreater(response.processing_time, 0)

        asyncio.run(run_test())


class TestClaudeAdapter(unittest.TestCase):
    """Test Claude adapter functionality"""

    def setUp(self):
        self.config = Config()
        self.config.anthropic_api_key = "mock_anthropic_key"
        self.adapter = ClaudeAdapter(self.config)

    def test_estimate_cost(self):
        """Test cost estimation"""
        query = "Analyze this portfolio risk"
        cost = self.adapter.estimate_cost(query)
        self.assertGreater(cost, 0)  # Should calculate based on tokens


class TestAICache(unittest.TestCase):
    """Test AI Cache functionality"""

    def setUp(self):
        self.cache = AICache(redis_url="redis://localhost:6379")

    @patch('ai_orchestrator.redis.asyncio.from_url')
    def test_cache_operations(self, mock_redis_from_url):
        """Test cache get/set operations"""
        mock_redis = AsyncMock()
        mock_redis_from_url.return_value = mock_redis

        # Reinitialize cache with mocked Redis
        self.cache = AICache()

        async def run_test():
            query = AIQuery(query="Test query", query_type=QueryType.ANALYSIS)
            response = AIResponse(content="Test response", provider="chatgpt")

            # Test cache miss
            mock_redis.get.return_value = None
            cached = await self.cache.get(query)
            self.assertIsNone(cached)

            # Test cache set
            await self.cache.set(query, response)
            mock_redis.setex.assert_called_once()

        asyncio.run(run_test())


class TestAIMetrics(unittest.TestCase):
    """Test AI Metrics functionality"""

    def setUp(self):
        self.metrics = AIMetrics()

    def test_metrics_recording(self):
        """Test metrics recording"""
        async def run_test():
            query = AIQuery(query="Test", query_type=QueryType.ANALYSIS)
            response = AIResponse(
                content="Response",
                provider="chatgpt",
                cost=0.05,
                processing_time=1.0
            )

            await self.metrics.record_query(query, response)

            metrics = self.metrics.get_metrics()
            self.assertEqual(metrics['total_queries'], 1)
            self.assertEqual(metrics['total_cost'], 0.05)
            self.assertEqual(metrics['provider_usage']['chatgpt'], 1)
            self.assertEqual(metrics['query_types']['analysis'], 1)

        asyncio.run(run_test())


class TestAIOrchestrator(unittest.TestCase):
    """Test AI Orchestrator core functionality"""

    def setUp(self):
        self.config = Config()
        # Set mock API keys for testing
        self.config.openai_api_key = "mock_openai_key"
        self.config.anthropic_api_key = "mock_anthropic_key"
        self.config.github_token = "mock_github_token"

    @patch('ai_orchestrator.redis.asyncio.from_url')
    @patch('ai_orchestrator.openai.AsyncOpenAI')
    @patch('ai_orchestrator.anthropic.AsyncAnthropic')
    @patch('ai_orchestrator.Github')
    def test_orchestrator_creation(self, mock_github, mock_anthropic, mock_openai, mock_redis):
        """Test orchestrator initialization"""
        # Mock all external dependencies
        mock_redis.return_value = AsyncMock()
        mock_openai.return_value = AsyncMock()
        mock_anthropic.return_value = AsyncMock()
        mock_github.return_value = AsyncMock()

        orchestrator = AIOrchestrator(self.config)

        self.assertIsInstance(orchestrator, AIOrchestrator)
        self.assertIn('copilot', orchestrator.providers)
        self.assertIn('chatgpt', orchestrator.providers)
        self.assertIn('claude', orchestrator.providers)

    def test_provider_selection(self):
        """Test provider selection logic"""
        with patch('ai_orchestrator.redis.asyncio.from_url', return_value=AsyncMock()):
            orchestrator = AIOrchestrator(self.config)

            # Test code generation routing
            query = AIQuery(query="Write a Python function", query_type=QueryType.CODE_GENERATION)
            provider = orchestrator._select_provider(query)
            self.assertEqual(provider, "copilot")

            # Test conversation routing
            query = AIQuery(query="Hello", query_type=QueryType.CONVERSATION)
            provider = orchestrator._select_provider(query)
            self.assertEqual(provider, "chatgpt")

            # Test risk assessment routing
            query = AIQuery(query="Analyze risk", query_type=QueryType.RISK_ASSESSMENT)
            provider = orchestrator._select_provider(query)
            self.assertEqual(provider, "claude")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete AI Orchestrator system"""

    def setUp(self):
        self.config = Config()
        self.config.openai_api_key = "mock_openai_key"
        self.config.anthropic_api_key = "mock_anthropic_key"
        self.config.github_token = "mock_github_token"

    @patch('ai_orchestrator.redis.asyncio.from_url')
    @patch('ai_orchestrator.openai.AsyncOpenAI')
    @patch('ai_orchestrator.anthropic.AsyncAnthropic')
    @patch('ai_orchestrator.Github')
    def test_get_ai_orchestrator_singleton(self, mock_github, mock_anthropic, mock_openai, mock_redis):
        """Test singleton orchestrator instance"""
        # Mock all external dependencies
        mock_redis.return_value = AsyncMock()
        mock_openai.return_value = AsyncMock()
        mock_anthropic.return_value = AsyncMock()
        mock_github.return_value = AsyncMock()

        # Test singleton pattern
        orchestrator1 = get_ai_orchestrator(self.config)
        orchestrator2 = get_ai_orchestrator()

        self.assertIs(orchestrator1, orchestrator2)
        self.assertIsInstance(orchestrator1, AIOrchestrator)


if __name__ == '__main__':
    # Set up test environment
    os.environ['OPENAI_API_KEY'] = 'test_key'
    os.environ['ANTHROPIC_API_KEY'] = 'test_key'
    os.environ['GITHUB_TOKEN'] = 'test_key'

    # Run tests
    unittest.main(verbosity=2)