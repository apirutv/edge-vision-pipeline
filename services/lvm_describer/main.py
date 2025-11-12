# services/lvm_describer/main.py
from __future__ import annotations
import asyncio, json, base64, time, os
from pathlib import Path
from typing import Dict, Any, List

import yaml
import httpx
from redis import asyncio as aioredis
from common.logging import get_logger

log = get_logger("lvm_describer")
REPO_ROOT = Path(__file__).resolve().parents[2]

GROUP = "lvm-describer"
CONSUMER = "lvmd-01"

# ----------------- Redis helpers -----------------
async def ensure_group(r, stream: str, group: str):
    """Create group at 0-0 (see history); ignore BUSYGROUP."""
    try:
        await r.xgroup_create(stream, group, id="0-0", mkstream=True)
        log.info(f"Created consumer group '{group}' at 0-0 on stream '{stream}'")
    except Exception as e:
        if "BUSYGROUP" in str(e):
            log.info(f"Consumer group '{group}' already exists on '{stream}'")
        else:
            raise

def _dlq_name(default: str | None = None) -> str:
    return default or "frames.lvm_describer.dlq"

# ----------------- JSON helpers -----------------
def _json_safe(x: Any) -> Any:
    try:
        json.dumps(x)
        return x
    except Exception:
        try:
            return float(x)
        except Exception:
            return str(x)

# ----------------- Prompt helpers -----------------
def _reference_prompt() -> str:
    return (
        "You are an advanced image scene analysis assistant. "
        "Carefully observe the provided image and output a detailed JSON description ONLY. "
        "Follow this JSON structure strictly:\n\n"
        "{\n"
        "  \"scene\": \"<place or environment>\",\n"
        "  \"objects\": [list of key visible objects],\n"
        "  \"person_present\": true/false,\n"
        "  \"people\": [\n"
        "    {\"description\": \"<appearance, age, gender, clothing, pose, and activity>\"}, ...\n"
        "  ],\n"
        "  \"pet_present\": true/false,\n"
        "  \"pets\": [\n"
        "    {\"description\": \"<species, color, posture, location, and behavior>\"}, ...\n"
        "  ],\n"
        "  \"vehicles_present\": true/false,\n"
        "  \"vehicles\": [\n"
        "    {\n"
        "      \"type\": \"<car, truck, motorbike, bicycle, bus, etc.>\",\n"
        "      \"color\": \"<dominant color>\",\n"
        "      \"brand\": \"<brand name and model name if identifiable, else 'unknown'>\",\n"
        "      \"description\": \"<appearance, brand hints if visible, condition>\",\n"
        "      \"position\": \"<relative position in the image>\"\n"
        "    }, ...\n"
        "  ],\n"
        "  \"activities\": [list of actions or interactions happening in the image]\n"
        "}\n\n"
        "Be precise and factual — describe what is visible, not assumptions."
    )

def _camera_meta_for_id(cfg: Dict[str, Any], camera_id: str) -> Dict[str, Any]:
    for cam in cfg.get("cameras", []):
        if cam.get("id") == camera_id:
            return {
                "location": cam.get("location"),
                "description": cam.get("description"),
                "angle": cam.get("angle"),
                "semantic_tags": cam.get("semantic_tags", []),
            }
    return {}

def _augment_prompt_with_context(base_prompt: str, camera_meta: Dict[str, Any], detections: List[Dict[str, Any]]) -> str:
    lines = [base_prompt, "\nAdditional context (do NOT change JSON schema):"]
    if camera_meta:
        lines.append(
            f"- Camera context: location={camera_meta.get('location')}; "
            f"description={camera_meta.get('description')}; angle={camera_meta.get('angle')}; "
            f"tags={camera_meta.get('semantic_tags')}"
        )
    if detections:
        cnt = {}
        for o in detections:
            lbl = o.get("label", "unknown")
            cnt[lbl] = cnt.get(lbl, 0) + 1
        parts = [f"{k} x{v}" for k, v in cnt.items()]
        lines.append(f"- Detector hints: {', '.join(parts)}")
    return "\n".join(lines)

def _extract_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return {"raw_output": text}
    return {"raw_output": text}

# ----------------- Vision call -----------------
async def call_ollama_vision(host: str, model: str, img_path: Path, prompt: str, timeout_sec: int) -> Dict[str, Any]:
    img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    payload = {"model": model, "prompt": prompt, "images": [img_b64], "stream": False}
    t0 = time.time()
    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        resp = await client.post(f"{host}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
    elapsed = time.time() - t0
    text = (data or {}).get("response", "") or ""
    result = _extract_json(text)
    result["_inference_time_sec"] = round(elapsed, 2)
    result["_inference_time_ms"] = int(elapsed * 1000)
    return result

# ----------------- Payload normalization -----------------
_REQUIRED = ("camera_id", "frame_id", "ts", "path")

def _normalize_payload(kv: dict) -> dict | None:
    if "json" in kv and kv["json"]:
        try:
            payload = json.loads(kv["json"])
            if all(k in payload for k in _REQUIRED):
                return payload
        except Exception:
            pass
    # Rare fallback for flat fields (not expected but harmless)
    try:
        p = {k: kv.get(k) for k in _REQUIRED}
        if all(p.get(k) for k in _REQUIRED):
            return p
    except Exception:
        pass
    return None

# ----------------- Core processing -----------------
async def process_message(
    r, payload: Dict[str, Any], cfg: Dict[str, Any], stream_out: str, outputs_base: Path
):
    """
    payload from frames.detected:
      {event,camera_id,frame_id,ts,path,model,objects:[{label,conf,bbox_xyxy}]}
    """
    cam_id = payload["camera_id"]
    frame_id = payload["frame_id"]
    img_path = Path(payload["path"])
    detections = payload.get("objects", []) or []

    lvm_cfg = cfg.get("lvm", {}) or {}
    model = lvm_cfg.get("model", "qwen3-vl:8b")
    ip = lvm_cfg.get("ip", "127.0.0.1")
    port = int(lvm_cfg.get("port", 11434))
    timeout = int(lvm_cfg.get("timeout_sec", 60))
    retries = int(lvm_cfg.get("retries", 1))
    host = f"http://{ip}:{port}"

    camera_meta = _camera_meta_for_id(cfg, cam_id)
    base_prompt = _reference_prompt()
    prompt = _augment_prompt_with_context(base_prompt, camera_meta, detections)

    out_dir = outputs_base / cam_id / "described"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{frame_id}.json"

    log.info(f"[describing] camera={cam_id} frame={frame_id} ...")

    result = None
    for attempt in range(retries + 1):
        try:
            result = await call_ollama_vision(host, model, img_path, prompt, timeout)
            break
        except Exception as e:
            log.error(f"Ollama call failed (attempt {attempt+1}/{retries+1}) camera={cam_id} frame={frame_id}: {e}")
            await asyncio.sleep(1.0)

    if not result:
        log.error(f"Ollama failed for camera={cam_id} frame={frame_id}")
        return

    final_payload = {
        "camera_id": cam_id,
        "frame_id": frame_id,
        "ts": payload.get("ts"),
        "path": payload.get("path"),
        "model": model,
        # reference schema:
        "scene": result.get("scene"),
        "objects": result.get("objects", []),
        "person_present": result.get("person_present"),
        "people": result.get("people", []),
        "pet_present": result.get("pet_present"),
        "pets": result.get("pets", []),
        "vehicles_present": result.get("vehicles_present"),
        "vehicles": result.get("vehicles", []),
        "activities": result.get("activities", []),
        # timings
        "_inference_time_sec": result.get("_inference_time_sec"),
        "_inference_time_ms": result.get("_inference_time_ms"),
        # raw fallback if needed
        "raw_output": result.get("raw_output") if "raw_output" in result else None,
        # optional hints/context
        "_detector_hints": [o.get("label") for o in detections],
        "_camera_context": camera_meta,
    }
    final_payload = {k: v for k, v in final_payload.items() if v is not None}

    out_json.write_text(json.dumps(_json_safe(final_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"[described] camera={cam_id} frame={frame_id}")

    # compact downstream event
    event = {
        "event": "frame.described",
        "camera_id": cam_id,
        "frame_id": frame_id,
        "ts": payload.get("ts"),
        "path": payload.get("path"),
        "model": model,
        "scene": final_payload.get("scene"),
        "person_present": final_payload.get("person_present"),
        "pet_present": final_payload.get("pet_present"),
        "vehicles_present": final_payload.get("vehicles_present"),
        "activities": final_payload.get("activities", []),
    }
    await r.xadd(stream_out, {"json": json.dumps(_json_safe(event), separators=(",", ":"))}, maxlen=10000, approximate=True)

# ----------------- Backlog phases -----------------
async def _drain_history(r, cfg: dict, batch_size: int, stream_in: str, stream_out: str, outputs_base: Path, dlq: str):
    log.info("Phase 1: draining never-delivered history...")
    while True:
        resp = await r.xreadgroup(GROUP, CONSUMER, streams={stream_in: "0"}, count=batch_size)
        if not resp:
            break
        total = 0
        for _stream, messages in resp:
            total += len(messages)
            for msg_id, kv in messages:
                try:
                    payload = _normalize_payload(kv)
                    if not payload:
                        await r.xadd(dlq, {"json": json.dumps({"source": stream_in, "id": msg_id,
                                                               "error": {"reason": "schema_mismatch"}})},
                                     maxlen=10000, approximate=True)
                        await r.xack(stream_in, GROUP, msg_id)
                        continue
                    await process_message(r, payload, cfg, stream_out, outputs_base)
                    await r.xack(stream_in, GROUP, msg_id)
                except Exception as e:
                    log.error(f"[history] Process error msg_id={msg_id}: {e}")
                    await r.xadd(dlq, {"json": json.dumps({"source": stream_in, "id": msg_id, "error": str(e)})},
                                 maxlen=10000, approximate=True)
                    await r.xack(stream_in, GROUP, msg_id)
        if total == 0:
            break

async def _recover_pending(r, cfg: dict, batch_size: int, min_idle_ms: int,
                           stream_in: str, stream_out: str, outputs_base: Path, dlq: str):
    """
    Phase 2: reclaim stale pending messages.
    Try XAUTOCLAIM (start_id or start), else XPENDING RANGE + XCLAIM.
    """
    log.info("Phase 2: recovering stale pending entries (min_idle_ms=%d)...", min_idle_ms)

    # Preferred: XAUTOCLAIM
    cursor = "0-0"
    while True:
        try:
            next_cursor, claimed = await r.xautoclaim(
                stream_in, GROUP, CONSUMER, min_idle_ms, start_id=cursor, count=batch_size
            )
        except TypeError:
            try:
                next_cursor, claimed = await r.xautoclaim(
                    stream_in, GROUP, CONSUMER, min_idle_ms, start=cursor, count=batch_size
                )
            except Exception as e:
                log.warning(f"XAUTOCLAIM not available or failed ({e}); falling back to XPENDING+XCLAIM.")
                break
        except Exception as e:
            log.warning(f"XAUTOCLAIM not available or failed ({e}); falling back to XPENDING+XCLAIM.")
            break

        if not claimed:
            if next_cursor == cursor:
                return
            cursor = next_cursor
            continue

        for msg_id, kv in claimed:
            try:
                payload = _normalize_payload(kv)
                if not payload:
                    await r.xadd(dlq, {"json": json.dumps({"source": stream_in, "id": msg_id,
                                                           "error": {"reason": "schema_mismatch"}})},
                                 maxlen=10000, approximate=True)
                    await r.xack(stream_in, GROUP, msg_id)
                    continue
                await process_message(r, payload, cfg, stream_out, outputs_base)
                await r.xack(stream_in, GROUP, msg_id)
            except Exception as e:
                log.error(f"[pending/xautoclaim] Process error msg_id={msg_id}: {e}")
                await r.xadd(dlq, {"json": json.dumps({"source": stream_in, "id": msg_id, "error": str(e)})},
                             maxlen=10000, approximate=True)
                await r.xack(stream_in, GROUP, msg_id)

    # Fallback: XPENDING RANGE + XCLAIM
    while True:
        try:
            pend = await r.xpending_range(stream_in, GROUP, min_idle_ms, '-', '+', count=batch_size)
        except Exception as e:
            log.warning(f"XPENDING RANGE failed ({e}); giving up pending recovery.")
            return

        if not pend:
            return

        ids = [p['message_id'] if isinstance(p, dict) else p.message_id for p in pend]
        try:
            claimed = await r.xclaim(stream_in, GROUP, CONSUMER, min_idle_ms, ids)
        except Exception as e:
            log.warning(f"XCLAIM failed ({e}); stopping pending recovery.")
            return

        if not claimed:
            return

        for msg_id, kv in claimed:
            try:
                payload = _normalize_payload(kv)
                if not payload:
                    await r.xadd(dlq, {"json": json.dumps({"source": stream_in, "id": msg_id,
                                                           "error": {"reason": "schema_mismatch"}})},
                                 maxlen=10000, approximate=True)
                    await r.xack(stream_in, GROUP, msg_id)
                    continue
                await process_message(r, payload, cfg, stream_out, outputs_base)
                await r.xack(stream_in, GROUP, msg_id)
            except Exception as e:
                log.error(f"[pending/xclaim] Process error msg_id={msg_id}: {e}")
                await r.xadd(dlq, {"json": json.dumps({"source": stream_in, "id": msg_id, "error": str(e)})},
                             maxlen=10000, approximate=True)
                await r.xack(stream_in, GROUP, msg_id)

# ----------------- Live loop -----------------
async def _live_loop(r, cfg: dict, batch_size: int, block_ms: int,
                     stream_in: str, stream_out: str, outputs_base: Path, dlq: str):
    log.info("Phase 3: live consumption (ID='>')...")
    while True:
        resp = await r.xreadgroup(GROUP, CONSUMER, streams={stream_in: ">"}, count=batch_size, block=block_ms)
        if not resp:
            continue
        for _stream, messages in resp:
            for msg_id, kv in messages:
                try:
                    payload = _normalize_payload(kv)
                    if not payload:
                        await r.xadd(dlq, {"json": json.dumps({"source": stream_in, "id": msg_id,
                                                               "error": {"reason": "schema_mismatch"}})},
                                     maxlen=10000, approximate=True)
                        await r.xack(stream_in, GROUP, msg_id)
                        continue
                    await process_message(r, payload, cfg, stream_out, outputs_base)
                    await r.xack(stream_in, GROUP, msg_id)
                except Exception as e:
                    log.error(f"[live] Process error msg_id={msg_id}: {e}")
                    await r.xadd(dlq, {"json": json.dumps({"source": stream_in, "id": msg_id, "error": str(e)})},
                                 maxlen=10000, approximate=True)
                    await r.xack(stream_in, GROUP, msg_id)

# ----------------- Main -----------------
async def main(config_path: str = "config/config.yaml"):
    log.info("lvm_describer starting…")
    cfg = yaml.safe_load(open(config_path, "r"))
    runtime = cfg.get("runtime", {}) or {}

    redis_url   = runtime.get("redis_url", "redis://127.0.0.1:6379/0")
    stream_in   = runtime.get("stream_detected",  "frames.detected")
    stream_out  = runtime.get("stream_described", "frames.described")
    outputs_base = (REPO_ROOT / runtime.get("outputs_dir", "outputs"))

    # optional runtime tuning under lvm.runtime
    lrt = (cfg.get("lvm", {}).get("runtime", {}) if isinstance(cfg.get("lvm", {}), dict) else {}) or {}
    min_idle_ms = int(lrt.get("min_idle_ms", os.getenv("LVM_MIN_IDLE_MS", 5000)))
    batch_size  = int(lrt.get("batch_size",  os.getenv("LVM_BATCH_SIZE", 8)))
    block_ms    = int(lrt.get("block_ms",    os.getenv("LVM_BLOCK_MS", 5000)))
    dlq_stream  = lrt.get("dlq_stream", os.getenv("LVM_DLQ_STREAM", "frames.lvm_describer.dlq"))
    drain_history_flag = bool(lrt.get("drain_history", True))

    r = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    await ensure_group(r, stream_in, GROUP)

    log.info(f"Consuming stream_in={stream_in} → stream_out={stream_out} model={cfg.get('lvm',{}).get('model')} consumer={CONSUMER}")

    if drain_history_flag:
        await _drain_history(r, cfg, batch_size, stream_in, stream_out, outputs_base, dlq_stream)
    await _recover_pending(r, cfg, batch_size, min_idle_ms, stream_in, stream_out, outputs_base, dlq_stream)
    await _live_loop(r, cfg, batch_size, block_ms, stream_in, stream_out, outputs_base, dlq_stream)

if __name__ == "__main__":
    asyncio.run(main())
