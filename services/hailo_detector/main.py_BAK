# services/hailo_detector/main.py
from __future__ import annotations
import asyncio, json, shlex
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
    try:
        await r.xgroup_create(stream, group, id="$", mkstream=True)
        log.info(f"Created consumer group '{group}' on stream '{stream}'")
    except Exception:
        pass

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

# ----------------- Main -----------------
async def main(config_path: str = "config/config.yaml"):
    log.info("hailo_detector starting…")
    cfg = yaml.safe_load(open(config_path, "r"))
    runtime = cfg.get("runtime", {}) or {}
    cfg_det = cfg.get("detection", {}) or {}

    redis_url = runtime.get("redis_url", "redis://127.0.0.1:6379/0")
    stream_in  = runtime.get("stream_changed",  "frames.changed")
    stream_out = runtime.get("stream_detected", "frames.detected")
    outputs_dir = runtime.get("outputs_dir", "outputs")
    outputs_base = (REPO_ROOT / outputs_dir)

    r = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    await ensure_group(r, stream_in, GROUP)

    log.info(f"Consuming stream_in={stream_in} → stream_out={stream_out} mode={cfg_det.get('mode','mock')} consumer={CONSUMER}")
    while True:
        resp = await r.xreadgroup(GROUP, CONSUMER, streams={stream_in: ">"}, count=10, block=5000)
        if not resp:
            continue
        for _stream, messages in resp:
            for msg_id, kv in messages:
                try:
                    payload = json.loads(kv.get("json", "{}"))
                    await process_message(r, payload, cfg_det, stream_out, outputs_base)
                    await r.xack(stream_in, GROUP, msg_id)
                except Exception as e:
                    log.error(f"Process error msg_id={msg_id}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
