# services/change_detector/main.py
from __future__ import annotations
import asyncio, json
from pathlib import Path
from datetime import datetime, timezone

import yaml
import numpy as np
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity as ssim

from redis import asyncio as aioredis
from common.logging import get_logger

log = get_logger("change_detector")

STREAM_IN = None
STREAM_OUT = None
GROUP = "change-detector"
CONSUMER = "cd-01"

# ---- image helpers ---------------------------------------------------------

def _load_grayscale_resized(path: Path, size=(320, 180)) -> np.ndarray:
    img = Image.open(path).convert("L").resize(size, Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)

def _phash(path: Path) -> imagehash.ImageHash:
    img = Image.open(path).convert("L")
    return imagehash.phash(img)

def _phash_distance(h1: imagehash.ImageHash, h2: imagehash.ImageHash) -> int:
    return (h1 - h2)

def _compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    # expects grayscale arrays same shape
    score = ssim(a, b, data_range=255)
    return float(score)

# ---- redis/state helpers ---------------------------------------------------

async def ensure_group(r, stream: str, group: str):
    try:
        await r.xgroup_create(stream, group, id="$", mkstream=True)
        log.info(f"Created consumer group '{group}' on stream '{stream}'")
    except Exception:
        # group exists
        pass

def _state_key(camera_id: str) -> str:
    return f"state:prev:{camera_id}"

async def get_prev_state(r, camera_id: str) -> dict | None:
    h = await r.hgetall(_state_key(camera_id))
    return h or None

async def set_prev_state(r, camera_id: str, frame_id: str, path: str, phash_hex: str):
    await r.hset(_state_key(camera_id), mapping={
        "frame_id": frame_id,
        "path": path,
        "phash": phash_hex,
        "ts": datetime.now(timezone.utc).isoformat()
    })

# ---- decision helper -------------------------------------------------------

def _changed(decision_cfg: dict, phdist: int | None, ssim_score: float | None) -> bool:
    ph_min = int(decision_cfg.get("phash_hamming_min", 10))
    ssim_max = float(decision_cfg.get("ssim_max", 0.92))
    cond_ph = (phdist is not None) and (phdist >= ph_min)
    cond_ssim = (ssim_score is not None) and (ssim_score <= ssim_max)
    return bool(cond_ph or cond_ssim)

# ---- main worker -----------------------------------------------------------

async def process_message(r, payload: dict, decision_cfg: dict, stream_out: str):
    """
    payload from frames.captured:
      {event,camera_id,frame_id,ts,path,width,height}
    """
    cam_id = payload["camera_id"]
    cur_frame_id = payload["frame_id"]
    cur_path = Path(payload["path"])

    # 1) Load current frame features
    try:
        cur_phash = _phash(cur_path)
        a = _load_grayscale_resized(cur_path)
    except Exception as e:
        log.error(f"Failed to load current frame camera={cam_id} path={cur_path}: {e}")
        return

    # 2) Get previous state
    prev = await get_prev_state(r, cam_id)

    # If no previous, set state and exit (first frame baseline)
    if not prev or ("path" not in prev):
        await set_prev_state(r, cam_id, cur_frame_id, str(cur_path), str(cur_phash))
        log.info(f"[baseline:init] camera={cam_id} frame_id={cur_frame_id} path={cur_path}")
        return

    prev_path = Path(prev["path"])
    prev_frame_id = prev.get("frame_id", "")
    try:
        prev_phash = imagehash.hex_to_hash(prev["phash"])
    except Exception:
        prev_phash = None

    phdist = None
    ssim_score = None

    # 3) Compute deltas
    try:
        if prev_phash is not None:
            phdist = _phash_distance(cur_phash, prev_phash)
    except Exception as e:
        log.warning(f"pHash distance failed camera={cam_id}: {e}")

    try:
        b = _load_grayscale_resized(prev_path)
        ssim_score = _compute_ssim(a, b)
    except Exception as e:
        log.warning(f"SSIM failed camera={cam_id}: {e}")

    # 4) Decide
    is_changed = _changed(decision_cfg, phdist, ssim_score)

    # Cast NumPy scalars → native Python types (avoid JSON errors)
    phdist_py = int(phdist) if phdist is not None else None
    ssim_py  = float(ssim_score) if ssim_score is not None else None

    # 5) Update baseline to current (always)
    await set_prev_state(r, cam_id, cur_frame_id, str(cur_path), str(cur_phash))
    log.info(f"[baseline:update] camera={cam_id} prev={prev_frame_id} -> current={cur_frame_id}")

    # 6) Emit or log result
    if is_changed:
        event = {
            "event": "frame.changed",
            "camera_id": cam_id,
            "frame_id": cur_frame_id,
            "prev_frame_id": prev_frame_id,
            "ts": payload["ts"],
            "path": payload["path"],
            "delta": {
                "phash_hamming": phdist_py,
                "ssim": ssim_py,
            },
        }
        await r.xadd(
            stream_out,
            {"json": json.dumps(event, separators=(",", ":"))},
            maxlen=10000,
            approximate=True,
        )
        log.info(f"[comparison] CHANGED camera={cam_id} frame={cur_frame_id} phash={phdist_py} ssim={ssim_py}")
    else:
        log.info(f"[comparison] NO-CHANGE camera={cam_id} frame={cur_frame_id} phash={phdist_py} ssim={ssim_py}")


async def main(config_path: str = "config/config.yaml"):
    global STREAM_IN, STREAM_OUT
    log.info("change_detector starting…")
    cfg = yaml.safe_load(open(config_path, "r"))

    redis_url = cfg.get("runtime", {}).get("redis_url", "redis://127.0.0.1:6379/0")
    STREAM_IN  = cfg.get("runtime", {}).get("stream_captured", "frames.captured")
    STREAM_OUT = cfg.get("runtime", {}).get("stream_changed",  "frames.changed")
    decision_cfg = cfg.get("change_detection", {}) or {}

    r = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    await ensure_group(r, STREAM_IN, GROUP)

    log.info(f"Consuming stream_in={STREAM_IN} → stream_out={STREAM_OUT} group={GROUP} consumer={CONSUMER}")
    while True:
        resp = await r.xreadgroup(GROUP, CONSUMER, streams={STREAM_IN: ">"}, count=16, block=5000)
        if not resp:
            continue
        for _stream, messages in resp:
            for msg_id, kv in messages:
                try:
                    payload = json.loads(kv.get("json", "{}"))
                    await process_message(r, payload, decision_cfg, STREAM_OUT)
                    await r.xack(STREAM_IN, GROUP, msg_id)
                except Exception as e:
                    log.error(f"Process error msg_id={msg_id}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
