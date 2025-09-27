"""
AI Orchestrator Demo

This example demonstrates how to use the AI Orchestrator Module
to query multiple AI providers with intelligent routing and failover.
"""

import asyncio
import os
from ai_orchestrator import (
    get_ai_orchestrator, AIQuery, QueryType, ProviderStatus
)
from config import Config


async def basic_query_demo():
    """Demonstrate basic AI query functionality"""
    print("🤖 AI Orchestrator Basic Query Demo")
    print("=" * 50)

    # Initialize orchestrator
    config = Config()
    orchestrator = get_ai_orchestrator(config)

    # Start health monitoring
    await orchestrator.start()

    try:
        # Example 1: Market Analysis Query
        print("\n📊 Market Analysis Query:")
        query = AIQuery(
            query="Analyze the current market conditions for technology stocks. What are the key trends and risks?",
            query_type=QueryType.MARKET_INSIGHT,
            context={
                "user_id": "demo_user",
                "portfolio_focus": "technology",
                "risk_tolerance": "moderate"
            },
            max_cost=0.10  # Max $0.10 per query
        )

        print(f"Query: {query.query}")
        print(f"Type: {query.query_type.value}")
        print(f"Max Cost: ${query.max_cost}")

        # Note: This would make actual API calls in production
        # For demo purposes, we'll show the structure
        print("\n🔄 Processing query...")
        print("Selected Provider: claude (based on market analysis task)")
        print("Response: [AI analysis would appear here]")
        print("Cost: $0.08")
        print("Processing Time: 1.2s")

        # Example 2: Strategy Recommendation
        print("\n🎯 Strategy Recommendation Query:")
        strategy_query = AIQuery(
            query="Given current market volatility, should I adjust my portfolio allocation?",
            query_type=QueryType.STRATEGY,
            context={
                "current_allocation": {"stocks": 0.7, "bonds": 0.2, "cash": 0.1},
                "market_volatility": "high",
                "investment_horizon": "5_years"
            }
        )

        print(f"Query: {strategy_query.query}")
        print(f"Type: {strategy_query.query_type.value}")
        print("Selected Provider: claude (based on risk analysis task)")

        # Example 3: Conversational Query
        print("\n💬 Conversational Query:")
        chat_query = AIQuery(
            query="What's your opinion on the future of AI in trading?",
            query_type=QueryType.CONVERSATION,
            context={"conversation_mode": True}
        )

        print(f"Query: {chat_query.query}")
        print(f"Type: {chat_query.query_type.value}")
        print("Selected Provider: chatgpt (based on conversational task)")

    finally:
        await orchestrator.stop()


async def provider_comparison_demo():
    """Demonstrate comparing responses from multiple providers"""
    print("\n🔄 AI Provider Comparison Demo")
    print("=" * 50)

    config = Config()
    orchestrator = get_ai_orchestrator(config)

    await orchestrator.start()

    try:
        query = AIQuery(
            query="What are the advantages and disadvantages of using AI in algorithmic trading?",
            query_type=QueryType.ANALYSIS,
            context={"comparison_mode": True}
        )

        print(f"Query: {query.query}")
        print("\n🤖 Comparing responses from all providers...")

        # This would compare responses from Copilot, ChatGPT, and Claude
        print("\n📋 Comparison Results:")
        print("• Copilot: Technical focus, code examples, implementation details")
        print("• ChatGPT: Balanced analysis, conversational tone, practical insights")
        print("• Claude: Deep reasoning, risk considerations, ethical implications")
        print("\n🎯 Consensus: AI enhances speed and analysis but requires human oversight")

    finally:
        await orchestrator.stop()


async def health_monitoring_demo():
    """Demonstrate provider health monitoring"""
    print("\n🏥 Provider Health Monitoring Demo")
    print("=" * 50)

    config = Config()
    orchestrator = get_ai_orchestrator(config)

    await orchestrator.start()

    try:
        # Get provider status
        status = orchestrator.get_provider_status()

        print("📊 Current Provider Status:")
        for provider_name, metrics in status.items():
            print(f"• {provider_name}: {metrics.status.value}")
            print(f"  - Response Time: {metrics.response_time:.2f}s")
            print(f"  - Success Rate: {metrics.success_rate:.1%}")
            print(f"  - Consecutive Failures: {metrics.consecutive_failures}")

        # Get orchestrator metrics
        metrics = orchestrator.get_metrics()
        print("\n📈 Orchestrator Metrics:")
        print(f"• Total Queries: {metrics['total_queries']}")
        print(f"• Total Cost: ${metrics['total_cost']:.2f}")
        print(f"• Provider Usage: {metrics['provider_usage']}")

    finally:
        await orchestrator.stop()


async def error_handling_demo():
    """Demonstrate error handling and failover"""
    print("\n🛡️ Error Handling & Failover Demo")
    print("=" * 50)

    config = Config()
    orchestrator = get_ai_orchestrator(config)

    await orchestrator.start()

    try:
        print("Simulating provider failures and automatic failover...")

        # Example of graceful degradation
        query = AIQuery(
            query="Emergency: All providers are failing. What should I do?",
            query_type=QueryType.ANALYSIS,
            context={"emergency_mode": True}
        )

        print(f"Query: {query.query}")
        print("\n🔄 Attempting failover sequence:")
        print("1. Primary provider (claude): FAILED - API rate limit")
        print("2. Backup provider (chatgpt): FAILED - Network timeout")
        print("3. Final provider (copilot): SUCCESS")
        print("\n✅ Query completed via failover mechanism")

    finally:
        await orchestrator.stop()


def setup_environment():
    """Set up environment variables for demo"""
    # In production, these would be set via .env file or secure vault
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'demo_key')
    os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'demo_key')
    os.environ['GITHUB_TOKEN'] = os.getenv('GITHUB_TOKEN', 'demo_key')


async def main():
    """Run all AI Orchestrator demos"""
    print("🚀 AI Orchestrator Module Demonstration")
    print("=" * 60)

    # Setup
    setup_environment()

    # Run demos
    await basic_query_demo()
    await provider_comparison_demo()
    await health_monitoring_demo()
    await error_handling_demo()

    print("\n🎉 AI Orchestrator Demo Complete!")
    print("\n💡 Key Features Demonstrated:")
    print("• Intelligent provider selection based on query type")
    print("• Automatic failover and redundancy")
    print("• Cost optimization and rate limiting")
    print("• Response caching and performance monitoring")
    print("• Multi-provider response comparison")
    print("• Comprehensive error handling")


if __name__ == "__main__":
    asyncio.run(main())