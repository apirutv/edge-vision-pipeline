# services/uploader_ingest/main.py
from __future__ import annotations
import asyncio, json, os, hashlib
from pathlib import Path
from typing import Dict, Any
from glob import glob

import aiofiles
import httpx
import yaml
from redis import asyncio as aioredis
from common.logging import get_logger

log = get_logger("uploader_ingest")

GROUP    = "uploader-ingest"
CONSUMER = "up-01"

# ---------- Redis helpers ----------
async def ensure_group(r, stream: str, group: str):
    try:
        # create at 0-0 so history is visible
        await r.xgroup_create(stream, group, id="0-0", mkstream=True)
        log.info(f"Created consumer group '{group}' at 0-0 on '{stream}'")
    except Exception as e:
        if "BUSYGROUP" in str(e):
            log.info(f"Consumer group '{group}' already exists on '{stream}'")
        else:
            raise

# ---------- File/manifest helpers ----------
async def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    async with aiofiles.open(p, "rb") as f:
        while True:
            b = await f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _default_overlay_patterns() -> list[str]:
    return [
        "{stem}_tagged.jpg",
        "{stem}_outframe_*.jpg",
        "{stem}_overlay*.jpg",
        "{stem}_tagged.png",
        "{stem}_outframe_*.png",
        "{stem}_overlay*.png",
    ]

def _find_tagged(outputs_base: Path, cam_id: str, stem: str, patterns: list[str] | None = None) -> Path | None:
    """
    Try multiple overlay patterns inside outputs/<cam_id>/:
      <stem>_tagged.jpg / .png
      <stem>_outframe_*.jpg / .png
      <stem>_overlay*.jpg / .png
    Return the first existing match (prefer exact _tagged.jpg), else None.
    """
    base = outputs_base / cam_id
    # exact preferred first
    exact = base / f"{stem}_tagged.jpg"
    if exact.exists():
        return exact

    pats = patterns or _default_overlay_patterns()[1:]
    for pat in pats:
        pat_fmt = pat.format(stem=stem)
        matches = sorted(glob(str(base / pat_fmt)))
        if matches:
            return Path(matches[0])
    return None

def _paths_from_payload(payload: Dict[str, Any], outputs_base: Path, overlay_patterns: list[str] | None) -> Dict[str, Path]:
    """
    Infer file locations based on repo conventions and flexible overlay discovery.
    - Original frame path is already in payload["path"] (from cam_capture)
    - Hailo detection JSON & tagged.jpg: outputs/<cam>/<stem>.json / flexible overlay patterns
    - LVM description JSON: outputs/<cam>/described/<frame_id>.json
    """
    frame_path = Path(payload["path"])
    cam_id     = str(payload["camera_id"]).strip()  # trim stray spaces
    frame_id   = payload["frame_id"]
    stem       = frame_path.stem  # e.g. 1762885955601_3c90f299f4bf

    det_json   = outputs_base / cam_id / f"{stem}.json"
    tagged     = _find_tagged(outputs_base, cam_id, stem, overlay_patterns) or Path()
    desc_json  = outputs_base / cam_id / "described" / f"{frame_id}.json"

    return {
        "frame": frame_path,
        "tagged": tagged if tagged.exists() else Path(),
        "detections": det_json,
        "description": desc_json,
    }

def _safe_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except Exception:
        return 0

async def _build_manifest(payload: Dict[str, Any], paths: Dict[str, Path], lvm_model: str | None) -> Dict[str, Any]:
    sizes = {k: _safe_size(v) for k, v in paths.items()}
    hashes = {}
    for k, v in paths.items():
        hashes[k + "_sha256"] = (await _sha256(v)) if v.exists() else ""

    manifest = {
        "frame_id": payload["frame_id"],
        "camera_id": payload["camera_id"],
        "ts": payload.get("ts"),
        "path": payload.get("path"),
        "paths": {k: str(v) for k, v in paths.items()},
        "sizes": sizes,
        "hashes": hashes,
        "detector_model": payload.get("model"),     # from frames.described compact event
        "lvm_model": lvm_model,
        "scene_summary": {
            "scene": payload.get("scene"),
            "person_present": payload.get("person_present"),
            "pet_present": payload.get("pet_present"),
            "vehicles_present": payload.get("vehicles_present"),
            "activities": payload.get("activities") or []
        },
        "pi_meta": {"host": os.uname().nodename, "service": "uploader_ingest"}
    }
    return manifest

async def _upload_http(url: str, manifest: Dict[str, Any], files: Dict[str, Path], timeout_sec: int):
    # Multipart form: manifest.json + any present files
    multipart = {
        "manifest": (None, json.dumps(manifest, ensure_ascii=False), "application/json"),
    }
    # only attach existing files
    for field, p in files.items():
        if p and isinstance(p, Path) and p.exists():
            multipart[field] = (p.name, p.open("rb"), "application/octet-stream")
    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        resp = await client.post(url, files=multipart)
        resp.raise_for_status()
        return resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}

# ---------- Core processing ----------
async def process_one(r, kv: Dict[str, Any], cfg: Dict[str, Any], outputs_base: Path,
                      http_url: str, http_timeout: int, overlay_patterns: list[str] | None):
    payload = json.loads(kv.get("json", "{}"))
    if not payload or "frame_id" not in payload or "camera_id" not in payload or "path" not in payload:
        raise RuntimeError("Invalid frames.described payload")

    lvm_model = (cfg.get("lvm", {}) or {}).get("model")

    paths = _paths_from_payload(payload, outputs_base, overlay_patterns)

    # helpful logs
    if paths["tagged"] and paths["tagged"].exists():
        log.info(f"[uploader_ingest] Found tagged overlay: {paths['tagged']}")
    else:
        # include stem & camera in the message for quick debugging
        stem = Path(payload["path"]).stem
        log.warning(f"[uploader_ingest] Missing tagged overlay for frame_id={payload['frame_id']} cam={payload['camera_id']} stem={stem}")

    manifest = await _build_manifest(payload, paths, lvm_model)
    result = await _upload_http(http_url, manifest, paths, http_timeout)
    log.info(f"[uploaded] camera={payload['camera_id']} frame={payload['frame_id']} result={result or 'ok'}")

# ---------- Backlog phases ----------
async def _drain_history(r, cfg, stream_in, batch, outputs_base, http_url, http_timeout, overlay_patterns):
    log.info("Phase 1: draining never-delivered history...")
    while True:
        resp = await r.xreadgroup(GROUP, CONSUMER, streams={stream_in: "0"}, count=batch)
        if not resp:
            break
        total = 0
        for _s, msgs in resp:
            total += len(msgs)
            for msg_id, kv in msgs:
                try:
                    await process_one(r, kv, cfg, outputs_base, http_url, http_timeout, overlay_patterns)
                    await r.xack(stream_in, GROUP, msg_id)
                except Exception as e:
                    log.error(f"[history] msg_id={msg_id} error={e}")
                    await r.xack(stream_in, GROUP, msg_id)
        if total == 0:
            break

async def _recover_pending(r, cfg, stream_in, batch, min_idle_ms, outputs_base, http_url, http_timeout, overlay_patterns):
    log.info("Phase 2: recovering stale pending entries...")
    cursor = "0-0"
    while True:
        try:
            # redis-py newer: start_id
            next_cursor, claimed = await r.xautoclaim(stream_in, GROUP, CONSUMER, min_idle_ms,
                                                      start_id=cursor, count=batch)
        except TypeError:
            try:
                # older redis-py: start
                next_cursor, claimed = await r.xautoclaim(stream_in, GROUP, CONSUMER, min_idle_ms,
                                                          start=cursor, count=batch)
            except Exception:
                break
        except Exception:
            break

        if not claimed:
            if next_cursor == cursor:
                return
            cursor = next_cursor
            continue

        for msg_id, kv in claimed:
            try:
                await process_one(r, kv, cfg, outputs_base, http_url, http_timeout, overlay_patterns)
                await r.xack(stream_in, GROUP, msg_id)
            except Exception as e:
                log.error(f"[pending] msg_id={msg_id} error={e}")
                await r.xack(stream_in, GROUP, msg_id)

async def _live_loop(r, cfg, stream_in, batch, block_ms, outputs_base, http_url, http_timeout, overlay_patterns):
    log.info("Phase 3: live consumption (ID='>')...")
    while True:
        resp = await r.xreadgroup(GROUP, CONSUMER, streams={stream_in: ">"}, count=batch, block=block_ms)
        if not resp:
            continue
        for _s, msgs in resp:
            for msg_id, kv in msgs:
                try:
                    await process_one(r, kv, cfg, outputs_base, http_url, http_timeout, overlay_patterns)
                    await r.xack(stream_in, GROUP, msg_id)
                except Exception as e:
                    log.error(f"[live] msg_id={msg_id} error={e}")
                    await r.xack(stream_in, GROUP, msg_id)

# ---------- Main ----------
async def main(config_path: str = "config/config.yaml"):
    log.info("uploader_ingest starting…")
    cfg = yaml.safe_load(open(config_path, "r"))

    runtime   = cfg.get("runtime", {}) or {}
    uploader  = cfg.get("uploader", {}) or {}
    urt       = uploader.get("runtime", {}) or {}
    files_cfg = uploader.get("files", {}) or {}
    up_cfg    = uploader.get("upload", {}) or {}

    redis_url  = urt.get("redis_url", runtime.get("redis_url", "redis://127.0.0.1:6379/0"))
    stream_in  = runtime.get("stream_described", "frames.described")
    batch      = int(urt.get("batch_size", 8))
    block_ms   = int(urt.get("block_ms", 5000))
    min_idle   = int(urt.get("min_idle_ms", 5000))
    drain_hist = bool(urt.get("drain_history", True))

    outputs_base = Path(files_cfg.get("outputs_base", "outputs")).resolve()

    # Optional configurable overlay patterns
    overlay_patterns = files_cfg.get("overlay_patterns")
    if overlay_patterns and isinstance(overlay_patterns, list):
        overlay_patterns = [str(p) for p in overlay_patterns]
        log.info(f"Using custom overlay_patterns: {overlay_patterns}")
    else:
        overlay_patterns = None  # will fall back to defaults in _find_tagged

    http_url     = up_cfg.get("http", {}).get("url", "http://ubuntu:8000/api/ingest/frame")
    http_timeout = int(up_cfg.get("http", {}).get("timeout_sec", 30))

    r = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    await ensure_group(r, stream_in, GROUP)

    log.info(f"Consuming stream_in={stream_in} → POST {http_url} consumer={CONSUMER}")
    if drain_hist:
        await _drain_history(r, cfg, stream_in, batch, outputs_base, http_url, http_timeout, overlay_patterns)
    await _recover_pending(r, cfg, stream_in, batch, min_idle, outputs_base, http_url, http_timeout, overlay_patterns)
    await _live_loop(r, cfg, stream_in, batch, block_ms, outputs_base, http_url, http_timeout, overlay_patterns)

if __name__ == "__main__":
    asyncio.run(main())
