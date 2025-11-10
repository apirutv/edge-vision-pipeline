# scripts/tail_stream.py
import asyncio, json
from redis import asyncio as aioredis
from common.logging import get_logger

STREAM = "frames.captured"   # match config.runtime.stream_captured
GROUP = "dev"
CONSUMER = "tail01"
REDIS_URL = "redis://127.0.0.1:6379/0"

log = get_logger("tail_stream")

async def ensure_group(r):
    try:
        await r.xgroup_create(STREAM, GROUP, id="$", mkstream=True)
        log.info(f"Created consumer group '{GROUP}' on stream '{STREAM}'")
    except Exception:
        # group probably exists
        pass

async def main():
    log.info(f"Tailing stream={STREAM} as group={GROUP} consumer={CONSUMER}")
    r = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    await ensure_group(r)
    while True:
        resp = await r.xreadgroup(GROUP, CONSUMER, streams={STREAM: ">"}, count=10, block=5000)
        if not resp:
            continue
        for _stream, messages in resp:
            for msg_id, kv in messages:
                payload = json.loads(kv.get("json", "{}"))
                log.info(f"{msg_id} {json.dumps(payload, ensure_ascii=False)}")
                await r.xack(STREAM, GROUP, msg_id)

if __name__ == "__main__":
    asyncio.run(main())
