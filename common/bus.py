# common/bus.py
from __future__ import annotations
import json
from typing import Any, Dict
from redis import asyncio as aioredis
from common.logging import get_logger

log = get_logger("bus")

class EventBus:
    def __init__(self, redis_url: str):
        self._redis_url = redis_url
        self._redis = None

    async def connect(self):
        if self._redis is None:
            log.info(f"Connecting to Redis: {self._redis_url}")
            self._redis = aioredis.from_url(self._redis_url, encoding="utf-8", decode_responses=True)
            # quick ping
            try:
                pong = await self._redis.ping()
                log.info(f"Redis ping: {pong}")
            except Exception as e:
                log.error(f"Redis connection failed: {e}")
                raise
        return self

    async def close(self):
        if self._redis is not None:
            log.info("Closing Redis connection")
            await self._redis.close()
            self._redis = None

    async def xadd_json(self, stream: str, payload: Dict[str, Any]) -> str:
        assert self._redis is not None, "Call connect() first"
        data = {"json": json.dumps(payload, separators=(",", ":"))}
        msg_id = await self._redis.xadd(stream, data, maxlen=10000, approximate=True)
        log.debug(f"XADD stream={stream} id={msg_id}")
        return msg_id
