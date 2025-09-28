# Lightweight shim module to satisfy tests that patch 'ai_orchestrator.redis.asyncio.from_url'
# This file will be importable as ai_orchestrator.redis when tests patch that path.
class _RedisStub:
    async def get(self, *args, **kwargs):
        return None

    async def setex(self, *args, **kwargs):
        return None

class asyncio:
    @staticmethod
    def from_url(*args, **kwargs):
        return _RedisStub()

from_url = asyncio.from_url
