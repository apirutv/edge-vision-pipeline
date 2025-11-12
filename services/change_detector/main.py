# services/change_detector/main.py
from __future__ import annotations
import asyncio, json, os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List

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
CONSUMER = f"cd-01"

# ---------------- image helpers ----------------

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

# ---------------- redis/state helpers ----------------

async def ensure_group(r, stream: str, group: str):
    """
    Create the consumer group at '0-0' so the group can see historical items.
    If it exists, ignore the BUSYGROUP error.
    """
    try:
        # mkstream=True ensures stream exists without producers
        await r.xgroup_create(stream, group, id="0-0", mkstream=True)
        log.info(f"Created consumer group '{group}' at 0-0 on stream '{stream}'")
    except Exception as e:
        msg = str(e)
        if "BUSYGROUP" in msg or "BUSYGROUP" in msg:
            log.info(f"Consumer group '{group}' already exists on '{stream}'")
        else:
            raise

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

# ---------------- decision helper ----------------

def _changed(decision_cfg: dict, phdist: int | None, ssim_score: float | None) -> bool:
    ph_min = int(decision_cfg.get("phash_hamming_min", 10))
    ssim_max = float(decision_cfg.get("ssim_max", 0.92))
    cond_ph = (phdist is not None) and (phdist >= ph_min)
    cond_ssim = (ssim_score is not None) and (ssim_score <= ssim_max)
    return bool(cond_ph or cond_ssim)

# ---------------- core processing ----------------

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

# ---------------- backlog phases ----------------

# vvvvvvvvvv

async def _drain_history(r, decision_cfg: dict, batch_size: int, stream_in: str, stream_out: str):
    """
    Phase 1: deliver entries that have NEVER been delivered to the group (ID '0').
    Non-blocking; exits as soon as none are available.
    """
    log.info("Phase 1: draining never-delivered history...")
    while True:
        resp = await r.xreadgroup(
            GROUP, CONSUMER,
            streams={stream_in: "0"},
            count=batch_size
        )

        if not resp:
            break  # no streams returned → done

        total = 0
        for _stream, messages in resp:
            total += len(messages)
            for msg_id, kv in messages:
                try:
                    legacy_map = decision_cfg.get("legacy_map", {}) if isinstance(decision_cfg, dict) else {}
                    payload = _normalize_payload(kv, legacy_map)
                    if not payload:
                        await r.xadd(
                            _dlq_stream_name(stream_in),
                            {"json": json.dumps({"source": stream_in,
                                                 "id": msg_id,
                                                 "error": {"reason": "schema_mismatch",
                                                           "kv_keys": list(kv.keys())[:20]}})},
                            maxlen=10000, approximate=True
                        )
                        await r.xack(stream_in, GROUP, msg_id)
                        continue

                    await process_message(r, payload, decision_cfg, stream_out)
                    await r.xack(stream_in, GROUP, msg_id)
                except Exception as e:
                    log.error(f"[history] Process error msg_id={msg_id}: {e}")
                    await r.xadd(
                        _dlq_stream_name(stream_in),
                        {"json": json.dumps({"source": stream_in, "id": msg_id, "error": str(e)})},
                        maxlen=10000, approximate=True
                    )
                    await r.xack(stream_in, GROUP, msg_id)

        if total == 0:
            break  # reply container existed but had 0 messages → done

# ^^^^^^^^^^


def _dlq_stream_name(stream_in: str, default: str | None = None) -> str:
    return default or "frames.change_detector.dlq"

async def _recover_pending(r, decision_cfg: dict, batch_size: int, min_idle_ms: int, stream_in: str, stream_out: str):
    """
    Phase 2: claim stale pending entries using XAUTOCLAIM.
    Handle both return shapes from redis-py (version differences).
    """
    log.info("Phase 2: recovering stale pending entries (min_idle_ms=%d)...", min_idle_ms)
    cursor = "0-0"
    while True:
        try:
            # redis-py >=4.2 returns (next_cursor, messages)
            res = await r.xautoclaim(stream_in, GROUP, CONSUMER, min_idle_ms, start=cursor, count=batch_size)
            if isinstance(res, tuple) and len(res) == 2:
                next_cursor, claimed = res
            else:
                # some variants may return list-like messages only
                next_cursor, claimed = cursor, res
        except Exception as e:
            log.warning(f"XAUTOCLAIM not available or failed ({e}); skipping pending recovery.")
            break

        if not claimed:
            # No more claims at this cursor; if cursor didn't move, we're done
            if next_cursor == cursor:
                break
            cursor = next_cursor
            continue

        for msg_id, kv in claimed:
            try:
                payload = json.loads(kv.get("json", "{}"))
                await process_message(r, payload, decision_cfg, stream_out)
                await r.xack(stream_in, GROUP, msg_id)
            except Exception as e:
                log.error(f"[pending] Process error msg_id={msg_id}: {e}")
                await r.xadd(
                    _dlq_stream_name(stream_in),
                    {"json": json.dumps({"source": stream_in, "id": msg_id, "error": str(e)})},
                    maxlen=10000, approximate=True
                )
                await r.xack(stream_in, GROUP, msg_id)

# ---------------- live loop ----------------

async def _live_loop(r, decision_cfg: dict, batch_size: int, block_ms: int, stream_in: str, stream_out: str):
    log.info("Phase 3: live consumption (ID='>')...")
    while True:
        resp = await r.xreadgroup(
            GROUP, CONSUMER,
            streams={stream_in: ">"},
            count=batch_size,
            block=block_ms
        )
        if not resp:
            continue
        for _stream, messages in resp:
            for msg_id, kv in messages:
                try:
                    payload = json.loads(kv.get("json", "{}"))
                    await process_message(r, payload, decision_cfg, stream_out)
                    await r.xack(stream_in, GROUP, msg_id)
                except Exception as e:
                    log.error(f"[live] Process error msg_id={msg_id}: {e}")
                    await r.xadd(
                        _dlq_stream_name(stream_in),
                        {"json": json.dumps({"source": stream_in, "id": msg_id, "error": str(e)})},
                        maxlen=10000, approximate=True
                    )
                    await r.xack(stream_in, GROUP, msg_id)

# ---------------- main ----------------

async def main(config_path: str = "config/config.yaml"):
    global STREAM_IN, STREAM_OUT
    log.info("change_detector starting…")
    cfg = yaml.safe_load(open(config_path, "r"))

    runtime_cfg = (cfg.get("runtime", {}) or {})
    cd_runtime   = (cfg.get("change_detection", {}).get("runtime", {}) 
                    if isinstance(cfg.get("change_detection", {}), dict) else {}) or {}

    redis_url   = runtime_cfg.get("redis_url", "redis://127.0.0.1:6379/0")
    STREAM_IN   = runtime_cfg.get("stream_captured", "frames.captured")
    STREAM_OUT  = runtime_cfg.get("stream_changed",  "frames.changed")

    # thresholds for decision function
    decision_cfg = (cfg.get("change_detection", {}) or {})

    # backlog controls
    min_idle_ms = int(cd_runtime.get("min_idle_ms", os.getenv("CD_MIN_IDLE_MS", 5000)))
    batch_size  = int(cd_runtime.get("batch_size",  os.getenv("CD_BATCH_SIZE", 32)))
    block_ms    = int(cd_runtime.get("block_ms",    os.getenv("CD_BLOCK_MS", 5000)))
    dlq_stream  = cd_runtime.get("dlq_stream", os.getenv("CD_DLQ_STREAM", "frames.change_detector.dlq"))

    # bind DLQ name helper to chosen stream
    global _dlq_stream_name
    _dlq_stream_name = (lambda _in, default=dlq_stream: dlq_stream)

    r = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    await ensure_group(r, STREAM_IN, GROUP)

    log.info(f"Consuming stream_in={STREAM_IN} → stream_out={STREAM_OUT} group={GROUP} consumer={CONSUMER}")

    # --- Phase 1: drain never-delivered history
    await _drain_history(r, decision_cfg, batch_size, STREAM_IN, STREAM_OUT)

    # --- Phase 2: recover stale pending
    await _recover_pending(r, decision_cfg, batch_size, min_idle_ms, STREAM_IN, STREAM_OUT)

    # --- Phase 3: live loop for new items
    await _live_loop(r, decision_cfg, batch_size, block_ms, STREAM_IN, STREAM_OUT)

if __name__ == "__main__":
    asyncio.run(main())
