# 📸 Edge Vision Pipeline (Raspberry Pi 5)

> **Purpose**
>
> The *edge-vision-pipeline* runs on a **Raspberry Pi 5 with Hailo AI HAT+**, performing near‑real‑time video analytics and synchronizing structured scene data to the **Server Vision Pipeline** for indexing, retrieval, and reasoning.

It captures frames from IP cameras, detects motion and objects, generates scene descriptions via local LVMs (Ollama/Qwen‑VL), and continuously uploads results to the server’s ingest API for long‑term storage, search, and AI reasoning.

---

## 🧩 1. System Overview

A set of lightweight Python micro‑services connected via **Redis Streams**:

```text
┌──────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌────────────────────┐
│ RTSP Camera  │→→│ cam_capture      │→→│ change_detector  │→→│ hailo_detector   │→→│ lvm_describer (LVM)│
└──────────────┘   └─────────────────┘   └─────────────────┘   └─────────────────┘   └────────────┬─────────┘
                                                                                                   │
                                                                                                   ▼
                                                                               HTTP Upload → server ingest_api
```

All computation occurs on the Pi; results are eventually synced to the server when the network is available.

---

## ⚙️ 2. Key Features

| Stage | Function |
|--------|-----------|
| **Frame Capture** | Snapshots RTSP streams into `data/frames/<camera>/<date>/`. |
| **Change Detection** | Uses perceptual hash + SSIM to forward only changed frames. |
| **Object Detection** | YOLO on Hailo‑8 accelerator → labels + confidence + bounding boxes. |
| **Scene Description** | LVM (Ollama Qwen‑VL) → structured JSON scene description. |
| **Event Streaming** | Publishes results via Redis Streams. |
| **Uploader** | Posts described frames to the server’s `ingest_api`. |
| **Dashboard** | Dark‑mode Redis dashboard showing lag/pending metrics in real time. |
| **Logging** | Unified rotating log system per service. |

---

## 🧱 3. Project Structure

```text
edge-vision-pipeline/
├── common/
│   ├── bus.py                   # Redis helpers
│   └── logging.py               # Rotating file logger
├── config/
│   └── config.yaml              # camera list, thresholds, Redis, server URL
├── services/
│   ├── cam_capture/             # frame grabber
│   ├── change_detector/         # pHash/SSIM change filter
│   ├── hailo_detector/          # YOLO/Hailo integration
│   ├── lvm_describer/           # LVM (Ollama) scene describer
│   ├── uploader_ingest/         # uploads bundles to server ingest API
│   └── redis_dashboard/         # dark-mode Redis streams & log dashboard
├── data/
│   ├── frames/                  # raw snapshots
│   └── outputs/                 # detection & description outputs
├── logs/
└── requirements.txt
```

---

## 🔌 4. Configuration (`config/config.yaml`)

```yaml
runtime:
  redis_url: "redis://127.0.0.1:6379/0"
  frames_dir: "data/frames"
  outputs_dir: "outputs"

  stream_captured:  "frames.captured"
  stream_changed:   "frames.changed"
  stream_detected:  "frames.detected"
  stream_described: "frames.described"

change_detection:
  phash_hamming_min: 10
  ssim_max: 0.92

detection:
  mode: "hailo"
  model: "hailo_yolo"
  conf_min: 0.5
  classes_pass: ["person","dog","cat","car","motorcycle","truck","bicycle"]
  hailo_cli: "python plugins/detection_from_mp4_overlay_hef.py"
  hailo_args: "--hef /usr/local/hailo/resources/models/hailo8/yolov8m.hef --max-frames 1"

lvm:
  host: "http://localhost:11434"
  model: "qwen3-vl:8b"
  timeout_sec: 60

uploader:
  runtime:
    redis_url: "redis://127.0.0.1:6379/0"
    batch_size: 8
    block_ms: 5000
    min_idle_ms: 5000
    drain_history: true
  files:
    outputs_base: "outputs"
    # optional: custom overlay filename patterns
    # overlay_patterns:
    #   - "{stem}_tagged.jpg"
    #   - "{stem}_outframe_*.jpg"
    #   - "{stem}_overlay*.jpg"
  upload:
    http:
      url: "http://<SERVER_IP>:8000/api/ingest/frame"
      timeout_sec: 30

redis_dashboard:
  host: "0.0.0.0"
  port: 9090
  refresh_ms: 2000
```

---

## 🧠 5. Redis Streams on the Pi

| Stream | Producer | Consumer | Purpose |
|---------|-----------|-----------|----------|
| `frames.captured`  | `cam_capture`     | `change_detector` | raw frame metadata |
| `frames.changed`   | `change_detector` | `hailo_detector`  | frames with visual change |
| `frames.detected`  | `hailo_detector`  | `lvm_describer`   | detection results |
| `frames.described` | `lvm_describer`   | `uploader_ingest` | full scene JSON ready to upload |

Messages are stored as:

```json
{"json":"<serialized payload>"}
```

---

## 🧪 6. Running Manually (for development)

```bash
cd ~/python_projects/edge-vision-pipeline
source .venv/bin/activate

export LOG_LEVEL=INFO

python -m services.cam_capture.main
python -m services.change_detector.main
python -m services.hailo_detector.main
python -m services.lvm_describer.main
python -m services.uploader_ingest.main
python -m services.redis_dashboard.main
```

---

## ⚙️ 7. Running as Services with **systemd** (Pi5)

Create template unit: `/etc/systemd/system/evp@.service`

```ini
[Unit]
Description=Edge Vision Pipeline - %i
After=network-online.target redis-server.service
Wants=network-online.target

[Service]
User=apirut
WorkingDirectory=/home/apirut/python_projects/edge-vision-pipeline
Environment=PYTHONUNBUFFERED=1
Environment=LOG_LEVEL=INFO
ExecStart=/home/apirut/python_projects/edge-vision-pipeline/.venv/bin/python -m %i
Restart=always
RestartSec=2
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

Enable each service:

```bash
sudo systemctl daemon-reload

sudo systemctl enable --now evp@services.cam_capture.main
sudo systemctl enable --now evp@services.change_detector.main
sudo systemctl enable --now evp@services.hailo_detector.main
sudo systemctl enable --now evp@services.lvm_describer.main
sudo systemctl enable --now evp@services.uploader_ingest.main
sudo systemctl enable --now evp@services.redis_dashboard.main
```

Monitor logs:

```bash
journalctl -u evp@services.cam_capture.main -f
journalctl -u evp@services.uploader_ingest.main -f
```

---

## 📊 8. Redis Dashboard (Pi)

```text
http://<PI5_LAN_IP>:9090/
```

- Title: **Edge Vision Pipeline Redis Dashboard**  
- Shows lag and pending per stream + live logs for edge services.

---

## 🔍 9. Using `tail_stream.py` on the Pi

`scripts/tail_stream.py` lets you inspect any Redis stream in real time from the CLI.

Examples:

```bash
# Tail frames from edge pipeline
python -m scripts.tail_stream frames.captured
python -m scripts.tail_stream frames.changed
python -m scripts.tail_stream frames.detected
python -m scripts.tail_stream frames.described

# Tail a DLQ or custom stream (if you add one)
python -m scripts.tail_stream frames.change_detector.dlq
```

With a remote Redis (e.g., tail the Pi from the server), use `--url`:

```bash
python -m scripts.tail_stream frames.described --url redis://192.168.0.169:6379/0
```

The script uses the same rotating logger, so logs appear under `logs/` too.

---

## ✅ 10. Summary

- Pi5 runs the capture → detect → describe → upload pipeline as systemd services (`evp@…`).  
- Streams and logs are visible via the web dashboard and `tail_stream.py`.  
- The uploader continuously syncs described frames to the server’s ingest API.
