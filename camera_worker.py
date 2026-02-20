# camera_worker.py
import threading
import time
import traceback
from typing import Callable, Dict, Optional

import cv2
from ultralytics import YOLO

# Path to your trained model file (adjust if needed)
_MODEL_PATH = "best.pt"
# Confidence threshold for marking detection (tweak as needed)
CONF_THRESH = 0.6
# Minimum seconds between repeated last-seen updates for same camera
LAST_SEEN_COOLDOWN = 5.0

# Internal map: camera_id -> worker info
_workers: Dict[str, Dict] = {}
_workers_lock = threading.Lock()

# Load model once (this may take time)
try:
    _model = YOLO(_MODEL_PATH)
except Exception:
    _model = None
    traceback.print_exc()
    print("Failed to load YOLO model in camera_worker.py; ensure best.pt exists and is compatible.")


def _run_worker(username: str, camera_id: str, url: str, stop_event: threading.Event, callback: Callable[[str, str], None]):
    """
    Worker thread body: open capture, read frames, run model, call callback when detection occurs.
    callback(username, camera_id) is sync (e.g. server.mark_last_seen_sync).
    """
    last_mark = 0.0
    reconnect_delay = 3.0
    while not stop_event.is_set():
        cap = None
        try:
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                # couldn't open, back off
                time.sleep(reconnect_delay)
                continue

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    # stream ended or broken — break to reconnect
                    break

                if _model is None:
                    # no model loaded; avoid busy loop
                    time.sleep(1.0)
                    continue

                # Run inference (synchronous). If you have GPU, this is faster.
                try:
                    results = _model(frame)
                except Exception:
                    traceback.print_exc()
                    # avoid spinning if inference failed
                    time.sleep(1.0)
                    continue

                # results is iterable: check boxes/confidence
                detected = False
                for r in results:
                    # ultralytics result may have r.boxes.conf, r.boxes.cls etc
                    try:
                        confs = getattr(r.boxes, "conf", None)
                        # If class information present, you can also check r.boxes.cls
                        if confs is not None:
                            for i, cval in enumerate(confs):
                                try:
                                    conf = float(cval)
                                except Exception:
                                    conf = 0.0
                                if conf >= CONF_THRESH:
                                    detected = True
                                    break
                        else:
                            # fallback: if any box present treat as detection
                            if len(getattr(r.boxes, "xyxy", [])) > 0:
                                detected = True
                    except Exception:
                        continue
                    if detected:
                        break

                # If detection and cooldown passed, call the DB callback to update last_seen
                if detected:
                    now = time.time()
                    if now - last_mark >= LAST_SEEN_COOLDOWN:
                        last_mark = now
                        try:
                            # callback is synchronous: e.g. server.mark_last_seen_sync
                            callback(username, camera_id)
                        except Exception:
                            traceback.print_exc()

                # small throttle to avoid hogging CPU if frames are very fast
                time.sleep(0.01)

        except Exception:
            traceback.print_exc()
            # on unexpected error, back off a bit
            time.sleep(2.0)
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

        # retry delay before attempting to reopen
        if not stop_event.is_set():
            time.sleep(reconnect_delay)


def start_worker(username: str, camera_id: str, url: str, callback: Callable[[str, str], None]) -> None:
    """
    Start a background worker for the given camera (no-op if already running).
    callback(username,camera_id) is called synchronously when detection occurs.
    """
    with _workers_lock:
        if camera_id in _workers:
            # already running
            return
        stop_event = threading.Event()
        t = threading.Thread(
            target=_run_worker,
            args=(username, camera_id, url, stop_event, callback),
            daemon=True,
            name=f"cam-worker-{camera_id[:8]}",
        )
        _workers[camera_id] = {"thread": t, "stop": stop_event, "username": username, "url": url, "callback": callback}
        t.start()


def stop_worker(camera_id: str) -> None:
    """Stop worker for camera_id if running."""
    with _workers_lock:
        info = _workers.get(camera_id)
        if not info:
            return
        try:
            info["stop"].set()
            info["thread"].join(timeout=2.0)
        except Exception:
            pass
        _workers.pop(camera_id, None)


def restart_worker(camera_id: str, url: Optional[str] = None) -> None:
    """Restart a worker with a (possibly new) url."""
    with _workers_lock:
        info = _workers.get(camera_id)
        username = info["username"] if info else None
    if info:
        stop_worker(camera_id)
    if username and url:
        start_worker(username, camera_id, url, info["callback"] if info else None)


def start_all_from_db(db: Dict[str, Dict], callback: Callable[[str, str], None]) -> None:
    """
    Start workers for all cameras in the provided DB dict.
    db is the dict loaded from users.json (username -> user dict).
    """
    for username, u in db.items():
        for cam in u.get("cameras", []):
            cam_id = cam.get("camera_id")
            url = cam.get("url")
            if cam_id and url:
                try:
                    start_worker(username, cam_id, url, callback)
                except Exception:
                    traceback.print_exc()


def stop_all() -> None:
    with _workers_lock:
        keys = list(_workers.keys())
    for k in keys:
        stop_worker(k)