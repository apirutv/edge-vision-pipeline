# services/hailo_detector/main.py
from __future__ import annotations
import asyncio, json, os, shlex
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import yaml
from redis import asyncio as aioredis
from common.logging import get_logger

log = get_logger("hailo_detector")

REPO_ROOT = Path(__file__).resolve().parents[2]  # edge-vision-pipeline/

STREAM_IN = None
STREAM_OUT = None
GROUP = "hailo-detector"
CONSUMER = "hd-01"

# ----------------- Models -----------------
@dataclass
class DetObj:
    label: str
    conf: float
    bbox_xyxy: List[float]  # [x1,y1,x2,y2]

def _filter_and_format(objs: List[DetObj], allow: List[str], conf_min: float) -> List[Dict[str, Any]]:
    out = []
    for o in objs:
        if o.label in allow and o.conf >= conf_min:
            out.append({
                "label": o.label,
                "conf": float(o.conf),
                "bbox_xyxy": [float(x) for x in o.bbox_xyxy],
            })
    return out

# ----------------- Redis helpers -----------------
async def ensure_group(r, stream: str, group: str):
    """
    Create group at 0-0 so backlog is visible on first run; ignore BUSYGROUP.
    """
    try:
        await r.xgroup_create(stream, group, id="0-0", mkstream=True)
        log.info(f"Created consumer group '{group}' at 0-0 on stream '{stream}'")
    except Exception as e:
        if "BUSYGROUP" in str(e):
            log.info(f"Consumer group '{group}' already exists on '{stream}'")
        else:
            raise

def _dlq_name(default: str | None = None) -> str:
    return default or "frames.hailo_detector.dlq"

# ----------------- Mock detector -----------------
def _mock_detect(path: Path) -> List[DetObj]:
    # Always returns one center person
    return [DetObj(label="person", conf=0.99, bbox_xyxy=[0.1, 0.1, 0.9, 0.9])]

# ----------------- Hailo detector hook -----------------
async def _hailo_cli_detect(img_path: Path, cli: str, extra_args: str, out_dir: Path) -> List[DetObj]:
    """
    Invoke your plugin:
      <cli> "<img_path>" <extra_args> --output-dir "<out_dir>"
    Prefer stdout JSON if printed; otherwise read <out_dir>/<stem>.json written by plugin.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Build command safely
    parts = shlex.split(f'{cli}') + [str(img_path)]
    if extra_args:
        parts += shlex.split(extra_args)
    parts += ["--output-dir", str(out_dir)]

    log.debug(f"Hailo CLI: {' '.join(parts)}")
    proc = await asyncio.create_subprocess_exec(
        *parts,
        cwd=str(REPO_ROOT),  # stable CWD
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        log.error(f"Hailo CLI exited {proc.returncode}: {err.decode(errors='ignore')}")
        return []

    data = None
    # Try parse last stdout line as JSON (if you add print at plugin end)
    txt = (out or b"").decode(errors="ignore").strip()
    if txt:
        last_line = txt.splitlines()[-1]
        try:
            data = json.loads(last_line)
        except Exception:
            data = None

    # Fallback to file
    if data is None:
        stem = img_path.stem
        json_path = out_dir / f"{stem}.json"
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as e:
                log.error(f"Failed reading {json_path}: {e}")
                return []

    if not data:
        return []

    # Normalize to flat detections
    seq = data.get("detections") or data.get("frames") or []
    if isinstance(seq, list) and seq and isinstance(seq[0], dict) and "detections" in seq[0]:
        seq = [d for fr in seq for d in fr.get("detections", [])]

    dets: List[DetObj] = []
    for d in seq:
        try:
            bbox = d.get('bbox') or d.get('bbox_xyxy')
            if isinstance(bbox, dict):  # {x,y,w,h} -> [x1,y1,x2,y2]
                x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
                xyxy = [float(x), float(y), float(x + w), float(y + h)]
            elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                xyxy = [float(v) for v in bbox]
            else:
                xyxy = [0.0, 0.0, 0.0, 0.0]
            dets.append(DetObj(
                label=str(d.get("label", "")),
                conf=float(d.get("confidence") or d.get("conf") or 0.0),
                bbox_xyxy=xyxy
            ))
        except Exception:
            continue
    return dets

# ----------------- Schema normalization -----------------
_REQUIRED = ("camera_id", "frame_id", "ts", "path")

def _normalize_payload(kv: dict) -> dict | None:
    # frames.changed is produced by change_detector with a "json" wrapper
    if "json" in kv and kv["json"]:
        try:
            payload = json.loads(kv["json"])
            if all(k in payload for k in _REQUIRED):
                return payload
        except Exception:
            pass
    # very rare fallback if someone sent flat fields
    try:
        p = {k: kv.get(k) for k in _REQUIRED}
        if all(p.get(k) for k in _REQUIRED):
            return p
    except Exception:
        pass
    return None

# ----------------- Message processor -----------------
async def process_message(r, payload: dict, cfg_det: dict, stream_out: str, outputs_base: Path):
    """
    payload from frames.changed:
      {event,camera_id,frame_id,prev_frame_id,ts,path,delta{phash_hamming,ssim}}
    """
    cam_id = payload["camera_id"]
    frame_id = payload["frame_id"]
    img_path = Path(payload["path"])

    mode = str(cfg_det.get("mode", "mock")).lower()
    allow = cfg_det.get("classes_pass", []) or []
    conf_min = float(cfg_det.get("conf_min", 0.5))

    if mode == "hailo":
        cli = cfg_det.get("hailo_cli", "")
        extra = cfg_det.get("hailo_args", "")
        out_dir = outputs_base / cam_id  # outputs/<camera_id>/
        objs = await _hailo_cli_detect(img_path, cli, extra, out_dir)
    else:
        objs = _mock_detect(img_path)

    filtered = _filter_and_format(objs, allow, conf_min)

    if filtered:
        event = {
            "event": "frame.detected",
            "camera_id": cam_id,
            "frame_id": frame_id,
            "ts": payload["ts"],
            "path": payload["path"],
            "model": cfg_det.get("model", "hailo_yolo"),
            "objects": filtered
        }
        await r.xadd(stream_out, {"json": json.dumps(event, separators=(",", ":"))}, maxlen=10000, approximate=True)
        log.info(f"[detected] camera={cam_id} frame={frame_id} n={len(filtered)} classes={[o['label'] for o in filtered]}")
    else:
        log.info(f"[no-detect] camera={cam_id} frame={frame_id}")

# ----------------- Backlog phases -----------------
async def _drain_history(r, cfg_det: dict, batch_size: int, stream_in: str, stream_out: str, outputs_base: Path, dlq: str):
    log.info("Phase 1: draining never-delivered history...")
    while True:
        # non-blocking read at ID '0' (deliver never-delivered items)
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
                        await r.xadd(dlq, {"json": json.dumps({"source": stream_in, "id": msg_id, "error": {"reason": "schema_mismatch"}})}, maxlen=10000, approximate=True)
                        await r.xack(stream_in, GROUP, msg_id)
                        continue
                    await process_message(r, payload, cfg_det, stream_out, outputs_base)
                    await r.xack(stream_in, GROUP, msg_id)
                except Exception as e:
                    log.error(f"[history] Process error msg_id={msg_id}: {e}")
                    await r.xadd(dlq, {"json": json.dumps({"source": stream_in, "id": msg_id, "error": str(e)})}, maxlen=10000, approximate=True)
                    await r.xack(stream_in, GROUP, msg_id)
        if total == 0:
            break

async def _recover_pending(r, cfg_det: dict, batch_size: int, min_idle_ms: int,
                           stream_in: str, stream_out: str, outputs_base: Path, dlq: str):
    """
    Phase 2: reclaim stale pending entries.
    Compatible across redis-py variants and Redis versions:
      - try XAUTOCLAIM with start_id (newer),
      - else try XAUTOCLAIM with start (older clients),
      - else fallback to XPENDING RANGE + XCLAIM.
    """
    log.info("Phase 2: recovering stale pending entries (min_idle_ms=%d)...", min_idle_ms)

    # --- Try XAUTOCLAIM first ---
    cursor = "0-0"
    while True:
        try:
            # most redis-py uses start_id=
            next_cursor, claimed = await r.xautoclaim(
                stream_in, GROUP, CONSUMER, min_idle_ms,
                start_id=cursor, count=batch_size
            )
        except TypeError:
            # older redis-py used start=
            try:
                next_cursor, claimed = await r.xautoclaim(
                    stream_in, GROUP, CONSUMER, min_idle_ms,
                    start=cursor, count=batch_size
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
                    await r.xadd(dlq, {"json": json.dumps({"source": stream_in, "id": msg_id, "error": {"reason": "schema_mismatch"}})}, maxlen=10000, approximate=True)
                    await r.xack(stream_in, GROUP, msg_id)
                    continue
                await process_message(r, payload, cfg_det, stream_out, outputs_base)
                await r.xack(stream_in, GROUP, msg_id)
            except Exception as e:
                log.error(f"[pending/xautoclaim] Process error msg_id={msg_id}: {e}")
                await r.xadd(dlq, {"json": json.dumps({"source": stream_in, "id": msg_id, "error": str(e)})}, maxlen=10000, approximate=True)
                await r.xack(stream_in, GROUP, msg_id)

    # --- Fallback: XPENDING RANGE + XCLAIM ---
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
                    await r.xadd(dlq, {"json": json.dumps({"source": stream_in, "id": msg_id, "error": {"reason": "schema_mismatch"}})}, maxlen=10000, approximate=True)
                    await r.xack(stream_in, GROUP, msg_id)
                    continue
                await process_message(r, payload, cfg_det, stream_out, outputs_base)
                await r.xack(stream_in, GROUP, msg_id)
            except Exception as e:
                log.error(f"[pending/xclaim] Process error msg_id={msg_id}: {e}")
                await r.xadd(dlq, {"json": json.dumps({"source": stream_in, "id": msg_id, "error": str(e)})}, maxlen=10000, approximate=True)
                await r.xack(stream_in, GROUP, msg_id)

# ----------------- Live loop -----------------
async def _live_loop(r, cfg_det: dict, batch_size: int, block_ms: int,
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
                        await r.xadd(dlq, {"json": json.dumps({"source": stream_in, "id": msg_id, "error": {"reason": "schema_mismatch"}})}, maxlen=10000, approximate=True)
                        await r.xack(stream_in, GROUP, msg_id)
                        continue
                    await process_message(r, payload, cfg_det, stream_out, outputs_base)
                    await r.xack(stream_in, GROUP, msg_id)
                except Exception as e:
                    log.error(f"[live] Process error msg_id={msg_id}: {e}")
                    await r.xadd(dlq, {"json": json.dumps({"source": stream_in, "id": msg_id, "error": str(e)})}, maxlen=10000, approximate=True)
                    await r.xack(stream_in, GROUP, msg_id)

# ----------------- Main -----------------
async def main(config_path: str = "config/config.yaml"):
    log.info("hailo_detector starting…")
    cfg = yaml.safe_load(open(config_path, "r"))
    runtime = cfg.get("runtime", {}) or {}
    cfg_det = cfg.get("detection", {}) or {}

    # optional runtime tuning under detection.runtime
    det_rt = (cfg.get("detection", {}).get("runtime", {}) if isinstance(cfg.get("detection", {}), dict) else {}) or {}

    redis_url  = runtime.get("redis_url", "redis://127.0.0.1:6379/0")
    stream_in  = runtime.get("stream_changed",  "frames.changed")
    stream_out = runtime.get("stream_detected", "frames.detected")
    outputs_dir = runtime.get("outputs_dir", "outputs")
    outputs_base = (REPO_ROOT / outputs_dir)

    # backlog knobs (with env fallbacks)
    min_idle_ms = int(det_rt.get("min_idle_ms", os.getenv("HD_MIN_IDLE_MS", 5000)))
    batch_size  = int(det_rt.get("batch_size",  os.getenv("HD_BATCH_SIZE", 16)))
    block_ms    = int(det_rt.get("block_ms",    os.getenv("HD_BLOCK_MS", 5000)))
    dlq_stream  = det_rt.get("dlq_stream", os.getenv("HD_DLQ_STREAM", "frames.hailo_detector.dlq"))
    drain_history_flag = bool(det_rt.get("drain_history", True))

    r = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    await ensure_group(r, stream_in, GROUP)

    log.info(f"Consuming stream_in={stream_in} → stream_out={stream_out} mode={cfg_det.get('mode','mock')} consumer={CONSUMER}")

    if drain_history_flag:
        await _drain_history(r, cfg_det, batch_size, stream_in, stream_out, outputs_base, dlq_stream)
    await _recover_pending(r, cfg_det, batch_size, min_idle_ms, stream_in, stream_out, outputs_base, dlq_stream)
    await _live_loop(r, cfg_det, batch_size, block_ms, stream_in, stream_out, outputs_base, dlq_stream)

if __name__ == "__main__":
    asyncio.run(main())
