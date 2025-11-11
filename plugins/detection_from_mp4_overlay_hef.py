#!/usr/bin/env python3
"""
detection_from_mp4_overlay_hef.py

Run Hailo detection on MP4 or JPEG (auto-converted), dump JSON/NDJSON,
and save annotated JPEG overlays with filenames based on the input name.
All outputs default to ./outputs.

This variant explicitly sets the HEF model (default: YOLOv8M) using the
argument style supported by your SDK (uses --hef-path; no postproc flags).
"""

import os
import sys
import json
import time
import tempfile
import subprocess
from pathlib import Path
import argparse

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import numpy as np
import cv2

import hailo
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection_simple.detection_pipeline_simple import GStreamerDetectionApp


# ---------------------------- Defaults --------------------------------------
DEFAULT_MIN_CONF   = 0.30
DEFAULT_MAXFRAMES  = 1
DEFAULT_SQUARE_SZ  = 640
DEFAULT_FPS        = 5
DEFAULT_SECONDS    = 1

# Your model (change if you want a different default)
DEFAULT_HEF_PATH   = "/usr/local/hailo/resources/models/hailo8/yolov8m.hef"


# ---------------------------- Helpers ---------------------------------------
def convert_jpg_to_mp4(jpg_path: Path, square=DEFAULT_SQUARE_SZ, fps=DEFAULT_FPS, seconds=DEFAULT_SECONDS) -> Path:
    """
    Robust JPEG -> MP4 for any orientation:
      - conditionally scale so either width or height == `square`
      - pad to exact square (centered)
      - 1s clip @ fps, yuv420p
    Falls back to GStreamer if ffmpeg fails/missing.
    """
    out_mp4 = Path(tempfile.gettempdir()) / f"{jpg_path.stem}_{seconds}s_{fps}fps.mp4"
    try:
        # If landscape: width=square; if portrait: height=square. Then pad to square.
        vf = (
            f"scale='if(gte(iw,ih),{square},-2)':'if(gte(iw,ih),-2,{square})',"
            f"pad={square}:{square}:(ow-iw)/2:(oh-ih)/2,"
            "setsar=1,format=yuv420p"
        )
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-loop", "1", "-i", str(jpg_path),
            "-t", str(seconds), "-r", str(fps),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(out_mp4)
        ]
        print(f"[convert:ffmpeg] JPEG → MP4 : {out_mp4}")
        subprocess.run(cmd, check=True)
        if out_mp4.exists() and out_mp4.stat().st_size > 0:
            return out_mp4
        else:
            print("[convert:ffmpeg] Output empty; falling back to GStreamer...")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[convert:ffmpeg] Not available/failed; using GStreamer fallback...")

    # ---- GStreamer fallback (square + fixed duration) ----
    frames = fps * seconds
    cmd_gst = [
        "gst-launch-1.0", "-e",
        "filesrc", f"location={str(jpg_path)}", "!",
        "jpegdec", "!", "imagefreeze", "!",
        f"video/x-raw,framerate={fps}/1", "!",
        "videoscale", "!", f"video/x-raw,width={square},height={square},pixel-aspect-ratio=1/1", "!",
        "videoconvert", "!",
        "identity", f"eos-after={frames}", "!",
        "x264enc", "tune=zerolatency", "speed-preset=ultrafast", "key-int-max=10", "!",
        "mp4mux", "!", "filesink", f"location={str(out_mp4)}"
    ]
    print(f"[convert:gst] JPEG → MP4 : {out_mp4}")
    subprocess.run(cmd_gst, check=True)
    return out_mp4


def _safe_bbox(det):
    """Extract bbox values (works across Hailo SDK versions)."""
    try:
        b = det.get_bbox()
        def val(x):
            v = getattr(b, x, None)
            return float(v() if callable(v) else v) if v is not None else None
        x = val("get_xmin") or val("xmin") or val("left") or 0.0
        y = val("get_ymin") or val("ymin") or val("top") or 0.0
        w = val("get_width") or val("width") or val("w") or 0.0
        h = val("get_height") or val("height") or val("h") or 0.0
        return {"x": x, "y": y, "w": w, "h": h}
    except Exception:
        return None


def write_outputs(collector, base_path: Path):
    """Write detections to JSON and NDJSON with input base name (in ./outputs)."""
    json_path   = base_path.with_suffix(".json")
    ndjson_path = base_path.with_suffix(".ndjson")
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frames": collector.frames
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved detections → {json_path}")

    with open(ndjson_path, "w", encoding="utf-8") as f:
        for frame in collector.frames:
            for d in frame["detections"]:
                rec = {"frame": frame["frame"], **d}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved detections (NDJSON) → {ndjson_path}")


# ---------------------------- Overlay helpers -------------------------------
def _caps_size_from_pad(pad):
    caps = pad.get_current_caps()
    if not caps or caps.get_size() == 0:
        return (DEFAULT_SQUARE_SZ, DEFAULT_SQUARE_SZ)
    s = caps.get_structure(0)
    return int(s.get_value("width")), int(s.get_value("height"))

def _maybe_unnormalize_bbox(bb, w, h):
    x, y, bw, bh = bb["x"], bb["y"], bb["w"], bb["h"]
    if max(x, y, bw, bh) <= 2.0:
        x, y, bw, bh = x * w, y * h, bw * w, bh * h
    x = max(0, min(int(round(x)), w - 1))
    y = max(0, min(int(round(y)), h - 1))
    bw = max(1, min(int(round(bw)), w - x))
    bh = max(1, min(int(round(bh)), h - y))
    return x, y, bw, bh

def save_annotated_frame(pad, buffer, detections, collector, frame_id):
    """Draw boxes/labels and save overlay JPEG named after input file (in ./outputs)."""
    w, h = _caps_size_from_pad(pad)
    ok, mapinfo = buffer.map(Gst.MapFlags.READ)
    if not ok:
        return

    try:
        rgb = np.frombuffer(mapinfo.data, dtype=np.uint8)
        if rgb.size != w * h * 3:
            return
        rgb = rgb.reshape((h, w, 3))
        img = rgb.copy()

        for d in detections:
            bb = d.get("bbox")
            if not bb:
                continue
            x, y, bw, bh = _maybe_unnormalize_bbox(bb, w, h)
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 1)  # thin box
            label = f"{d['label']} {d['confidence']:.2f}"
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x, max(0, y - th - base - 4)), (x + tw + 6, y),
                          (0, 255, 0), -1)
            cv2.putText(img, label, (x + 3, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        out_name = f"{collector.input_stem}_outframe_{frame_id:05d}.jpg"
        out_path = collector.output_dir / out_name
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved frame → {out_path}")
    finally:
        buffer.unmap(mapinfo)


# ---------------------------- Collector & Callback ---------------------------
class JsonCollector(app_callback_class):
    def __init__(self, min_conf, max_frames, input_stem, output_dir):
        super().__init__()
        self.frames = []
        self.min_conf = float(min_conf)
        self.max_frames = int(max_frames)
        self.input_stem = input_stem
        self.output_dir = Path(output_dir)

    def add(self, frame, detections):
        self.frames.append({"frame": frame, "detections": detections})


def make_callback(collector: JsonCollector):
    def app_callback(pad, info, _collector: JsonCollector = collector):
        _collector.increment()
        frame_id = _collector.get_count()
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        detections = []
        roi = hailo.get_roi_from_buffer(buffer)
        for det in roi.get_objects_typed(hailo.HAILO_DETECTION):
            conf = float(det.get_confidence())
            if conf < _collector.min_conf:
                continue
            item = {"label": det.get_label(), "confidence": round(conf, 3)}
            bb = _safe_bbox(det)
            if bb:
                item["bbox"] = bb
            detections.append(item)

        if detections:
            print(f"\nFrame {frame_id}")
            for d in detections:
                print(f"  {d['label']} ({d['confidence']})")

        save_annotated_frame(pad, buffer, detections, _collector, frame_id)
        _collector.add(frame_id, detections)

        if _collector.max_frames and frame_id >= _collector.max_frames:
            base_path = _collector.output_dir / _collector.input_stem
            write_outputs(_collector, base_path)
            print(f"\nReached {_collector.max_frames} frames — exiting to avoid loop.")
            os._exit(0)

        return Gst.PadProbeReturn.OK
    return app_callback


# ---------------------------- Main ------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hailo detection with outputs to ./outputs and explicit HEF (uses --hef-path).")
    parser.add_argument("input", help="Path to input .mp4 or .jpg/.jpeg")
    # find the argparse section and add one more argument:
    parser.add_argument("--output-dir", default=None, help="Directory to write JSON/NDJSON and annotated frames (defaults to ./outputs)")
    parser.add_argument("--hef", dest="hef_path", default=DEFAULT_HEF_PATH,
                        help=f"Path to HEF model (default: {DEFAULT_HEF_PATH})")
    parser.add_argument("--min-conf", type=float, default=DEFAULT_MIN_CONF)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAXFRAMES)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    input_stem = input_path.stem

    # All outputs go to ./outputs relative to current working dir
    #output_dir = Path.cwd() / "outputs"
    # Make sure all outputs go to a single shared outputs directory
    #output_dir = Path("/home/apirut/python_projects/edge-vision-pipeline/outputs")
    #camera_id = input_path.parts[-6]  # e.g., gate_left or dining (depends on your path depth)
    #output_dir = Path("/home/apirut/python_projects/edge-vision-pipeline/outputs") / camera_id
    #output_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- force a single canonical outputs directory (or use provided one)
    output_dir = Path(args.output_dir) if args.output_dir else (Path.cwd() / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[PLUGIN] writing to {output_dir} as base '{input_stem}'")

    # Convert JPEG → MP4 if needed
    if input_path.suffix.lower() in [".jpg", ".jpeg"]:
        input_path = convert_jpg_to_mp4(input_path)

    # Feed the Hailo sample via its CLI parser (no .env needed), using --hef-path (your SDK supports this)
    sys.argv = [
        sys.argv[0],
        "--input", str(input_path),
        "--hef-path", args.hef_path,   # << use hef-path, not hef
        # You may add other supported flags from your SDK here, e.g. --use-frame / --show-fps
        # "--use-frame",
        # "--show-fps",
    ]

    collector = JsonCollector(
        min_conf=args.min_conf,
        max_frames=args.max_frames,
        input_stem=input_stem,
        output_dir=output_dir
    )

    cb = make_callback(collector)
    app = GStreamerDetectionApp(cb, collector)
    app.run()

    # If we got here without hard-exit, still write outputs
    base_path = output_dir / input_stem
    write_outputs(collector, base_path)


if __name__ == "__main__":
    main()
