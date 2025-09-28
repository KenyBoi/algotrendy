# Top-level compatibility shim for tests that import ai_orchestrator
# Exposes the same public symbols as algotrendy.ai_orchestrator
from algotrendy.ai_orchestrator import *

# Ensure top-level module provides a predictable `redis` attribute for tests that patch
import types as _types
_redis_mod = _types.ModuleType('redis')
_redis_asyncio = _types.ModuleType('redis.asyncio')

class _RedisStub:
	async def get(self, *args, **kwargs):
		return None
	async def setex(self, *args, **kwargs):
		return None

def _from_url_stub(*args, **kwargs):
	return _RedisStub()

_redis_asyncio.from_url = staticmethod(_from_url_stub)
_redis_mod.asyncio = _redis_asyncio
_redis_mod.from_url = _redis_asyncio.from_url

# Attach to this module so patch('ai_orchestrator.redis.asyncio.from_url') works
globals()['redis'] = _redis_mod
