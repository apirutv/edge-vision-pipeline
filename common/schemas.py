from __future__ import annotations
from pydantic import BaseModel

class FrameCaptured(BaseModel):
    event: str = "frame.captured"
    camera_id: str
    frame_id: str
    ts: str           # ISO8601 UTC
    path: str         # file path to JPEG
    width: int
    height: int
