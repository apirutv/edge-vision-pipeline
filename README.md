# Edge Vision Pipeline

Modular edge-first CCTV/AI pipeline for Raspberry Pi 5 + Hailo:
- `cam_capture`: async RTSP frame capture
- `change_detector`: pHash + SSIM motion filter
- `hailo_detector`: object detection via Hailo/YOLO
- `lvm_describer`: LVM (OLLAMA) scene summarizer
- `indexer`: SQLite/FAISS metadata search

Early phase: `cam_capture` + Redis Streams + test tailer.

## Quick start
```bash
sudo apt install -y redis-server ffmpeg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python services/cam_capture/main.py
python scripts/tail_stream.py

Then:
```bash
git add .gitignore README.md
git commit -m "Add README and gitignore"
git push
