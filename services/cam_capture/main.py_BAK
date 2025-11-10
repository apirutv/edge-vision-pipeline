from __future__ import annotations
import asyncio, random, time
from pathlib import Path
from datetime import datetime, timezone
import yaml
import uuid
from common.bus import EventBus
from common.schemas import FrameCaptured

# Simple ffmpeg snapshotter: grabs 1 frame and writes to jpeg
async def grab_snapshot(rtsp_url: str, out_path: Path, timeout_sec: int = 8) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-frames:v", "1",
        "-q:v", "2",
        str(out_path)
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL
    )
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError("ffmpeg snapshot timed out")
    if not out_path.exists():
        raise RuntimeError("snapshot not created")

async def capture_loop(cam: dict, frames_dir: Path, bus: EventBus, stream_name: str):
    interval = int(cam.get("interval_sec", 10))
    cam_id = cam["id"]
    rtsp = cam["rtsp"]
    while True:
        t0 = time.time()
        try:
            ts = datetime.now(timezone.utc)
            date_path = Path(frames_dir) / cam_id / ts.strftime("%Y/%m/%d/%H")
            frame_id = uuid.uuid4().hex[:12]
            jpg_path = date_path / f"{int(ts.timestamp()*1000)}_{frame_id}.jpg"

            await grab_snapshot(rtsp, jpg_path)

            # width/height will be 0 for now; we'll fill later with PIL
            event = FrameCaptured(
                camera_id=cam_id,
                frame_id=frame_id,
                ts=ts.isoformat(),
                path=str(jpg_path),
                width=0,
                height=0,
            ).model_dump()

            await bus.xadd_json(stream_name, event)
        except Exception as e:
            print(f"[cam_capture] error camera={cam_id}: {e}")
        # jitter ±5%
        jitter = interval * 0.05 * (2*random.random() - 1)
        await asyncio.sleep(max(1, interval + jitter - (time.time() - t0)))

async def main(config_path: str = "config/config.yaml"):
    cfg = yaml.safe_load(open(config_path, "r"))
    frames_dir = Path(cfg.get("runtime", {}).get("frames_dir", "data/frames"))
    redis_url = cfg.get("runtime", {}).get("redis_url", "redis://127.0.0.1:6379/0")
    stream_captured = cfg.get("runtime", {}).get("stream_captured", "frames.captured")

    #-----
    
    cameras = cfg.get("cameras", [])
    if not cameras:
        raise SystemExit("No cameras in config")

    bus = await EventBus(redis_url).connect()

    # Only run active cameras with valid RTSP
    selected = [c for c in cameras if c.get("active", True) and c.get("rtsp")]
    if not selected:
        raise SystemExit("No active cameras with RTSP found")

    print("[cam_capture] starting cameras:", ",".join(c["id"] for c in selected))

    tasks = [
        asyncio.create_task(capture_loop(c, frames_dir, bus, stream_captured))
        for c in selected
    ]

    # -----
    try:
        await asyncio.gather(*tasks)
    finally:
        await bus.close()

if __name__ == "__main__":
    asyncio.run(main())
