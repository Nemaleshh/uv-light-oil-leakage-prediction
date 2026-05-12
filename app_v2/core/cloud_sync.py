"""
cloud_sync.py
Saves each inspection result to MongoDB Atlas.
Gracefully no-ops if MongoDB is not configured / unreachable.
"""

from __future__ import annotations
import threading
from datetime import datetime
from typing import Optional
import numpy as np

_mongo_lock  = threading.Lock()
_collection  = None      # pymongo Collection, set once in init()
_enabled     = False


def init(mongo_uri: str, db_name: str = "uv_inspection") -> bool:
    """
    Connect to MongoDB Atlas.  Returns True on success.
    Safe to call from any thread; if it fails the rest of the app keeps running.
    """
    global _collection, _enabled

    if not mongo_uri or mongo_uri == "YOUR_MONGODB_URI_HERE":
        return False

    try:
        from pymongo import MongoClient
        from pymongo.server_api import ServerApi
        client = MongoClient(mongo_uri, server_api=ServerApi("1"), serverSelectionTimeoutMS=5000)
        client.admin.command("ping")           # Confirm connection
        db = client[db_name]
        with _mongo_lock:
            _collection = db["inspections"]
        _enabled = True
        print("[CloudSync] Connected to MongoDB Atlas ✓")
        return True
    except Exception as exc:
        print(f"[CloudSync] MongoDB unavailable — running offline. ({exc})")
        _enabled = False
        return False


def save_record(
    vin:            str,
    timestamp:      str,
    auto_result:    str,
    confidence:     float,
    manual_confirm: str  = "Pending",
    snap_path:      str  = "",
    source:         str  = "CAMERA",
    raw_frame:      Optional[np.ndarray] = None,
) -> None:
    """
    Fire-and-forget: insert one document.  Any error is silently swallowed
    so it never disrupts the inspection pipeline.
    """
    if not _enabled or _collection is None:
        return

    image_b64 = None
    if raw_frame is not None:
        try:
            import cv2
            import base64
            # Compress the raw frame to a reasonable quality JPEG
            ok, buf = cv2.imencode(".jpg", raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                image_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        except Exception as e:
            print(f"[CloudSync] Image encode failed: {e}")

    doc = {
        "vin":            vin,
        "timestamp":      timestamp,
        "auto_result":    auto_result,
        "confidence":     confidence,
        "manual_confirm": manual_confirm,
        "snap_path":      snap_path,
        "source":         source,
        "image_base64":   image_b64,
        "created_at":     datetime.utcnow(),
    }

    def _insert():
        try:
            with _mongo_lock:
                _collection.insert_one(doc)
        except Exception as exc:
            print(f"[CloudSync] Insert failed: {exc}")

    threading.Thread(target=_insert, daemon=True).start()


def update_manual_confirm(vin: str, timestamp: str, manual_confirm: str) -> None:
    """Update the manual_confirm field for the matching record."""
    if not _enabled or _collection is None:
        return

    def _update():
        try:
            with _mongo_lock:
                _collection.update_one(
                    {"vin": vin, "timestamp": timestamp},
                    {"$set": {"manual_confirm": manual_confirm}},
                )
        except Exception as exc:
            print(f"[CloudSync] Update failed: {exc}")

    threading.Thread(target=_update, daemon=True).start()


def is_enabled() -> bool:
    return _enabled
