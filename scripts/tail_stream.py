import asyncio, json
from redis import asyncio as aioredis

STREAM = "frames.captured"   # match config.runtime.stream_captured
GROUP = "dev"
CONSUMER = "tail01"
REDIS_URL = "redis://127.0.0.1:6379/0"

async def ensure_group(r):
    try:
        await r.xgroup_create(STREAM, GROUP, id="$", mkstream=True)
    except Exception:
        pass

async def main():
    r = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    await ensure_group(r)
    print(f"Tail {STREAM} as group={GROUP} consumer={CONSUMER}")
    while True:
        resp = await r.xreadgroup(GROUP, CONSUMER, streams={STREAM: ">"}, count=10, block=5000)
        if not resp:
            continue
        for _stream, messages in resp:
            for msg_id, kv in messages:
                payload = json.loads(kv.get("json", "{}"))
                print(msg_id, json.dumps(payload, ensure_ascii=False))
                await r.xack(STREAM, GROUP, msg_id)

if __name__ == "__main__":
    asyncio.run(main())
