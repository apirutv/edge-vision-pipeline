# 📸 Edge Vision Pipeline (Raspberry Pi 5)

> **Purpose**  
> The *edge-vision-pipeline* runs on Raspberry Pi 5 with a Hailo AI HAT+ to perform near-real-time video analytics — capturing frames from IP cameras, detecting visual changes, performing object detection (YOLO + Hailo), generating structured scene descriptions via local LVM / LMM models, and publishing those results to a central **server-vision-pipeline** for long-term storage, search, and reasoning.

## 🧩 1. Overview
A fully modular, event-driven pipeline composed of lightweight Python micro-services connected through **Redis Streams**.  
It operates completely offline on the Pi, and synchronizes descriptions with the server once the network is available.

```
┌──────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐
│ RTSP Camera  │→→│ cam_capture      │→→│ change_detector  │→→│ hailo_detector   │→→│ lvm_describer (Ollama)   │
└──────────────┘   └─────────────────┘   └─────────────────┘   └─────────────────┘   └────────────┬──────────────┘
                                                                                                   │
                                                                                                   ▼
                                                                                    Redis Stream → frames.described
```

## ⚙️ 2. Key Features
| Stage | Function |
|-------|-----------|
| **Frame Capture** | Periodically snapshots RTSP streams into `data/frames/<camera>/<date>/`. |
| **Change Detection** | Compares consecutive frames using *pHash* + *SSIM*; forwards only when significant difference. |
| **Object Detection** | Runs YOLO model on Hailo 8 accelerator; outputs labels + confidence + bounding boxes. |
| **Scene Description** | Calls local or remote LVM (e.g., Ollama Qwen3-VL) to produce structured JSON. |
| **Event Streaming** | Publishes all descriptions to Redis Stream `frames.described`. |
| **Local Storage** | Saves overlays, JSONs, and logs for audit/offline re-sync. |
| **Logging** | Unified rotating log system under `logs/`. |

## 🧱 3. Project Structure
```
edge-vision-pipeline/
├── common/
│   ├── bus.py                  # Redis helpers
│   └── logging.py              # Rotating file logger
├── config/
│   └── config.yaml             # cameras, thresholds, redis URL
├── services/
│   ├── cam_capture/            # frame grabber
│   ├── change_detector/        # phash/ssim diff
│   ├── hailo_detector/         # YOLO/Hailo integration
│   └── lvm_describer/          # LVM/Ollama scene describer
├── plugins/                    # Hailo plugin(s)
├── data/
│   ├── frames/                 # snapshots
│   └── outputs/                # JSON + overlay results
├── logs/
│   ├── cam_capture.log
│   ├── change_detector.log
│   ├── hailo_detector.log
│   └── lvm_describer.log
└── requirements.txt
```

## 🔌 4. Configuration (`config/config.yaml`)
```yaml
runtime:
  redis_url: "redis://<server-ip>:6379/0"
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
```

## 🧠 5. Redis Streams
| Stream | Producer | Consumer | Description |
|---------|-----------|-----------|--------------|
| `frames.captured` | `cam_capture` | `change_detector` | raw frame metadata |
| `frames.changed` | `change_detector` | `hailo_detector` | change-flagged frame |
| `frames.detected` | `hailo_detector` | `lvm_describer` | detected objects + bboxes |
| `frames.described` | `lvm_describer` | **server-vision-pipeline** | full scene JSON |

All messages are JSON strings:
```json
{"json":"<serialized payload>"}
```

## 🪄 6. Running the Pipeline
```bash
cd ~/python_projects/edge-vision-pipeline
source .venv/bin/activate
export LOG_LEVEL=INFO

# launch services
python -m services.cam_capture.main
python -m services.change_detector.main
python -m services.hailo_detector.main
python -m services.lvm_describer.main
```

## 🧩 7. Logging
Rotating logs under `logs/`, default size = 5 MB × 5 backups.  
`LOG_LEVEL` environment variable controls verbosity.

## 🔗 8. Integration with Server Vision Pipeline
- Pi publishes each `frame.described` to Ubuntu’s Redis.  
- Server ingestor subscribes, writes JSON copy, and indexes into Chroma.

## 📋 9. Current To-Do List

### ✅ Completed
- Modular micro-services architecture  
- Capture, detection, and description pipeline  
- Hailo integration and LVM describer  
- Unified logging and Redis streaming  
- Edge → Server Redis connection verified  

### 🚧 In Progress
| Area | Task |
|------|------|
| Connectivity | Add connection-health check in `lvm_describer`. |
| Error Handling | Dead-letter stream for failed Redis posts. |
| Configuration | YAML schema validation with Pydantic. |
| Monitoring | `/healthz` endpoint per service. |
| Packaging | Convert to systemd units. |

### 🧭 Planned Enhancements
| Category | Ideas |
|-----------|-------|
| Edge AI | Add on-device tracking and multi-frame activity recognition. |
| Compression | Adaptive JPEG/WebP compression for bandwidth efficiency. |
| Fail-Safe | Local buffer for Redis downtime. |
| Edge Reasoning | Lightweight on-device reasoning agent. |
| Multi-Edge Mesh | LAN-based frame sharing for multi-camera correlation. |
| OTA Updates | Auto-update modules from GitHub. |
