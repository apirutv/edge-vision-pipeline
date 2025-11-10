from __future__ import annotations
import json
from typing import Any, Dict
from redis import asyncio as aioredis

class EventBus:
    def __init__(self, redis_url: str):
        self._redis_url = redis_url
        self._redis = None

    async def connect(self):
        if self._redis is None:
            self._redis = aioredis.from_url(
                self._redis_url, encoding="utf-8", decode_responses=True
            )
        return self

    async def close(self):
        if self._redis is not None:
            await self._redis.close()
            self._redis = None

    async def xadd_json(self, stream: str, payload: Dict[str, Any]) -> str:
        assert self._redis is not None, "Call connect() first"
        data = {"json": json.dumps(payload, separators=(",", ":"))}
        return await self._redis.xadd(stream, data, maxlen=10000, approximate=True)
