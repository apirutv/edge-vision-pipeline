# services/redis_dashboard/main.py
from __future__ import annotations
import asyncio, json, os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import uvicorn
import yaml
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from redis import asyncio as aioredis

from common.logging import get_logger

# ----------------------- config & logging -----------------------
ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "config" / "config.yaml"
cfg: Dict[str, Any] = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))

rt = cfg.get("runtime", {}) or {}
streams = {
    "frames.captured":  (rt.get("stream_captured",  "frames.captured"),  "change-detector"),
    "frames.changed":   (rt.get("stream_changed",   "frames.changed"),   "hailo-detector"),
    "frames.detected":  (rt.get("stream_detected",  "frames.detected"),  "lvm-describer"),
    "frames.described": (rt.get("stream_described", "frames.described"),
                         cfg.get("uploader",{}).get("runtime",{}).get("group","uploader-ingest")),
}

REDIS_URL = rt.get("redis_url", "redis://127.0.0.1:6379/0")
LOG_LEVEL = rt.get("log_level", "INFO")
LOG_DIR   = Path(rt.get("log_dir", "logs")).resolve()

DASH_HOST = (cfg.get("redis_dashboard", {}) or {}).get("host", "0.0.0.0")
DASH_PORT = int((cfg.get("redis_dashboard", {}) or {}).get("port", 9090))
REFRESH_MS = int((cfg.get("redis_dashboard", {}) or {}).get("refresh_ms", 2000))

log = get_logger("redis_dashboard", log_dir=str(LOG_DIR), level=LOG_LEVEL)

app = FastAPI(title="Edge Vision Pipeline Redis Dashboard")
redis: aioredis.Redis | None = None

# Map service → logfile (change names if you customize logger filenames)
LOG_FILES = {
    "cam_capture":     LOG_DIR / "cam_capture.log",
    "change_detector": LOG_DIR / "change_detector.log",
    "hailo_detector":  LOG_DIR / "hailo_detector.log",
    "lvm_describer":   LOG_DIR / "lvm_describer.log",
    "uploader_ingest": LOG_DIR / "uploader_ingest.log",
}

# ----------------------- helpers -----------------------
async def get_redis() -> aioredis.Redis:
    global redis
    if redis is None:
        redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    return redis

async def group_stats(r: aioredis.Redis, stream_name: str, group_name: str) -> Dict[str, Any]:
    """Return lag/pending stats for one group."""
    try:
        await r.xgroup_create(stream_name, group_name, id="0-0", mkstream=True)
    except Exception:
        pass
    try:
        groups = await r.xinfo_groups(stream_name)
    except Exception as e:
        return {"stream": stream_name, "group": group_name, "error": str(e)}

    info = next((g for g in groups if g.get("name") == group_name), None)
    if not info:
        return {"stream": stream_name, "group": group_name, "error": "group_not_found"}

    pending = int(info.get("pending", 0))
    lag = int(info.get("lag", 0)) if "lag" in info else None
    last_delivered = info.get("last-delivered-id")
    entries_read = info.get("entries-read")

    if lag is None:
        try:
            sinfo = await r.xinfo_stream(stream_name)
            length = int(sinfo.get("length", 0))
            lag = max(0, length - int(entries_read or 0))
        except Exception:
            lag = -1

    return {
        "stream": stream_name,
        "group": group_name,
        "pending": pending,
        "lag": lag,
        "last_delivered_id": last_delivered,
        "entries_read": entries_read,
    }

async def all_stats() -> List[Dict[str, Any]]:
    r = await get_redis()
    out = []
    for label, (stream_name, group_name) in streams.items():
        if not stream_name or not group_name:
            continue
        st = await group_stats(r, stream_name, group_name)
        st["label"] = label
        out.append(st)
    return out

def tail_last_lines(path: Path, max_lines: int = 200, max_bytes: int = 64_000) -> List[str]:
    """
    Efficiently read up to the last `max_lines` lines of a file, up to `max_bytes`.
    Handles missing files gracefully.
    """
    try:
        if not path.exists() or not path.is_file():
            return [f"[{path.name}] (no file)"]
        size = path.stat().st_size
        with path.open("rb") as f:
            if size <= max_bytes:
                data = f.read()
            else:
                f.seek(-max_bytes, os.SEEK_END)
                data = f.read()
        # decode with best-effort
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        return lines[-max_lines:] if len(lines) > max_lines else lines
    except Exception as e:
        return [f"[{path.name}] error: {e}"]

# ----------------------- METRICS page -----------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    # Dark-mode metrics dashboard
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Edge Vision Pipeline Redis Dashboard</title>
  <style>
    body {{
      background-color: #111; color: #eee; font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; margin: 24px;
    }}
    nav a {{ color: #66ccff; margin-right: 16px; text-decoration: none; }}
    h1 {{ color: #00c6ff; margin-bottom: 4px; }}
    .meta {{ color: #aaa; font-size: 0.9rem; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; background-color: #1a1a1a; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #333; text-align: left; }}
    th {{ background-color: #222; color: #66ccff; font-weight: 600; }}
    tr:hover td {{ background-color: #191919; }}
    .ok {{ color: #4caf50; font-weight: 600; }}
    .warn {{ color: #ffb300; font-weight: 600; }}
    .bad {{ color: #ef5350; font-weight: 700; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .pill {{ padding: 2px 8px; border-radius: 999px; font-size: 0.8rem; background: #333; color: #fff; }}
    .pill.ok {{ background: #2e7d32; }}
    .pill.warn {{ background: #b06d00; }}
    .pill.bad {{ background: #b71c1c; }}
  </style>
</head>
<body>
  <nav>
    <a href="/">Metrics</a>
    <a href="/logs">Live Logs</a>
  </nav>
  <h1>Edge Vision Pipeline Redis Dashboard</h1>
  <div class="meta">Redis URL: <span class="mono">{REDIS_URL}</span> · Auto refresh: {REFRESH_MS} ms</div>
  <table id="tbl">
    <thead>
      <tr><th>Label</th><th>Stream</th><th>Group</th><th>Lag</th><th>Pending</th><th>Last Delivered ID</th><th>Entries Read</th><th>Status</th></tr>
    </thead>
    <tbody id="rows"></tbody>
  </table>

<script>
const rows = document.getElementById('rows');
function cls(val) {{
  if (val === null || val === undefined) return '';
  return (val > 1000) ? 'bad' : (val > 0 ? 'warn' : 'ok');
}}
function statusCell(lag, pending) {{
  if (lag > 0 || pending > 0) return '<span class="pill warn">processing</span>';
  return '<span class="pill ok">idle</span>';
}}
async function refresh() {{
  try {{
    const res = await fetch('/metrics');
    const data = await res.json();
    rows.innerHTML = data.map(item => {{
      const lag = (item.lag ?? -1);
      const pending = (item.pending ?? -1);
      const err = item.error ? `<span class="bad mono">${{item.error}}</span>` : '';
      return `<tr>
        <td class="mono">${{item.label || ''}}</td>
        <td class="mono">${{item.stream}}</td>
        <td class="mono">${{item.group}}</td>
        <td class="${{cls(lag)}}">${{lag}}</td>
        <td class="${{cls(pending)}}">${{pending}}</td>
        <td class="mono">${{item.last_delivered_id || ''}}</td>
        <td>${{item.entries_read ?? ''}}</td>
        <td>${{err || statusCell(lag, pending)}}</td>
      </tr>`;
    }}).join('');
  }} catch (e) {{
    rows.innerHTML = `<tr><td colspan="8" class="bad">Error loading metrics: ${{e}}</td></tr>`;
  }}
}}
setInterval(refresh, {REFRESH_MS});
refresh();
</script>
</body>
</html>"""

@app.get("/metrics", response_class=JSONResponse)
async def metrics():
    try:
        data = await all_stats()
        return JSONResponse(data)
    except Exception as e:
        log.error("metrics error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)

# ----------------------- LOGS page -----------------------
@app.get("/logs", response_class=HTMLResponse)
async def logs_page():
    # Dark-mode live logs UI: five stacked panes, auto-scroll bottom, polling /logz every 1s
    service_cards = "".join([
        f"""
        <section class="card">
          <h2>{name.replace('_',' ').title()}</h2>
          <pre id="log_{name}" class="logbox">(loading...)</pre>
        </section>
        """
        for name in ["cam_capture","change_detector","hailo_detector","lvm_describer","uploader_ingest"]
    ])
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Edge Vision Pipeline · Live Logs</title>
  <style>
    body {{ background:#111; color:#eee; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; margin:24px; }}
    nav a {{ color:#66ccff; margin-right:16px; text-decoration:none; }}
    h1 {{ color:#00c6ff; margin-bottom:8px; }}
    .meta {{ color:#aaa; margin-bottom:16px; }}
    .grid {{ display:flex; flex-direction:column; gap:16px; }}
    .card {{ background:#1a1a1a; border:1px solid #262626; border-radius:10px; padding:12px; }}
    h2 {{ margin:0 0 8px 0; color:#66ccff; font-size:1.0rem; }}
    .logbox {{
      background:#0e0e0e; border:1px solid #222; border-radius:8px; padding:8px;
      height:190px; overflow:auto; white-space:pre-wrap; font-family:ui-monospace,Menlo,Consolas,monospace; font-size:12.5px; line-height:1.35;
    }}
    .dim {{ color:#9aa0a6; }}
    .warn {{ color:#ffb300; }}
    .err  {{ color:#ef5350; }}
  </style>
</head>
<body>
  <nav>
    <a href="/">Metrics</a>
    <a href="/logs">Live Logs</a>
  </nav>
  <h1>Live Logs</h1>
  <div class="meta">Log dir: <span class="dim">{LOG_DIR}</span> · Updating every 1000 ms</div>
  <div class="grid">
    {service_cards}
  </div>
<script>
const panes = {{
  cam_capture:     document.getElementById('log_cam_capture'),
  change_detector: document.getElementById('log_change_detector'),
  hailo_detector:  document.getElementById('log_hailo_detector'),
  lvm_describer:   document.getElementById('log_lvm_describer'),
  uploader_ingest: document.getElementById('log_uploader_ingest'),
}};
async function tick() {{
  try {{
    const res = await fetch('/logz?lines=200');
    const data = await res.json();
    for (const k of Object.keys(panes)) {{
      const lines = (data[k] || []).join('\\n') || '(no data)';
      const el = panes[k];
      const atBottom = (el.scrollTop + el.clientHeight + 10) >= el.scrollHeight;
      el.textContent = lines;
      if (atBottom) el.scrollTop = el.scrollHeight;
    }}
  }} catch (e) {{
    console.error(e);
  }}
}}
setInterval(tick, 1000);
tick();
</script>
</body>
</html>"""

@app.get("/logz", response_class=JSONResponse)
async def log_feed(lines: int = Query(200, ge=10, le=2000)):
    """
    Return last N lines per known service log as JSON arrays.
    This is polled by /logs every ~1s.
    """
    out: Dict[str, List[str]] = {}
    for name, path in LOG_FILES.items():
        out[name] = tail_last_lines(path, max_lines=lines)
    return JSONResponse(out)

# ----------------------- main -----------------------
if __name__ == "__main__":
    log.info("Edge Vision Pipeline Redis Dashboard starting on %s:%d (redis=%s)", DASH_HOST, DASH_PORT, REDIS_URL)
    uvicorn.run("services.redis_dashboard.main:app",
                host=DASH_HOST, port=DASH_PORT,
                reload=False, log_level=LOG_LEVEL.lower())
