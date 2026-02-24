# camera_worker.py
"""
Persistent camera worker threads that run YOLO inference continuously
and call a synchronous callback when a detection of interest occurs.

Usage:
    from camera_worker import start_worker, stop_worker, start_all_from_db

The callback must be a synchronous function with signature callback(username, camera_id).
Example: server.mark_last_seen_sync
"""
import threading
import time
import traceback
from typing import Callable, Dict, Optional

import cv2
from ultralytics import YOLO
import os

# ----------------- CONFIG -----------------
# Path to driver model (nano). Put your yolo26nano weights here.
# If not present, code will try to fall back to 'yolov8n.pt' (ultralytics small).
DRIVER_MODEL_PATH = os.getenv("DRIVER_MODEL_PATH", "yolo26n.pt")
# Path to your custom child detection model (as before)
CHILD_MODEL_PATH = os.getenv("CHILD_MODEL_PATH", "best.pt")

# Confidence threshold
CONF_THRESH = 0.6

# Minimum seconds between repeated last-seen updates for same camera
LAST_SEEN_COOLDOWN = 5.0

# Small sleep to avoid very tight loops
LOOP_SLEEP = 0.01

# ------------------------------------------

_workers: Dict[str, Dict] = {}
_workers_lock = threading.Lock()

# Load driver model (nano)
_driver_model = None
_child_model = None

def _safe_load_model(path: str):
    try:
        print(f"[camera_worker] loading model from: {path}")
        m = YOLO(path)
        print(f"[camera_worker] loaded model: {path}, names: {getattr(m, 'names', None)}")
        return m
    except Exception:
        traceback.print_exc()
        return None

# Try to load driver model; fallback to 'yolov8n.pt' if not present
_driver_model = _safe_load_model(DRIVER_MODEL_PATH)
if _driver_model is None:
    try:
        print("[camera_worker] driver model not found; trying fallback 'yolov8n.pt'")
        _driver_model = _safe_load_model("yolov8n.pt")
    except Exception:
        _driver_model = None

# Load child model (custom)
_child_model = _safe_load_model(CHILD_MODEL_PATH)
if _child_model is None:
    print("[camera_worker] WARNING: child model not loaded (best.pt missing). Non-driver cameras will be idle.")


def _model_class_name(model, cls_index) -> Optional[str]:
    """Return class name string for a model and class index if possible."""
    try:
        names = getattr(model, "names", None)
        if names is None:
            return None
        return str(names[int(cls_index)])
    except Exception:
        return None


def _run_worker(username: str, camera_id: str, url: str, stop_event: threading.Event,
                callback: Callable[[str, str], None], is_driver: bool):
    """
    Worker thread body:
     - opens VideoCapture(url)
     - runs model(frame)
     - if detection of interest -> callback(username, camera_id) (sync)
    """
    last_mark = 0.0
    reconnect_delay = 2.5

    # Choose which model to use per this camera
    def choose_model():
        if is_driver:
            return _driver_model
        else:
            return _child_model

    model = choose_model()
    if model is None:
        # nothing to do for this camera
        print(f"[camera_worker] no model available for camera {camera_id} (is_driver={is_driver})")
        # keep trying to reconnect/load model periodically
        while not stop_event.is_set():
            time.sleep(5.0)
        return

    # Allowed child-class names (from your data.yaml)
    CHILD_CLASS_NAMES = {"child", "doll", "teddy bear", "teddy_bear", "teddybear"}

    while not stop_event.is_set():
        cap = None
        try:
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                # couldn't open stream; back off before retry
                time.sleep(reconnect_delay)
                continue

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    # broken stream -> reconnect
                    break

                # Defensive: select the correct model in case user toggled driver flag at runtime
                model = choose_model()
                if model is None:
                    time.sleep(1.0)
                    continue

                # Run inference
                try:
                    results = model(frame)
                except Exception:
                    traceback.print_exc()
                    time.sleep(1.0)
                    continue

                detected = False

                # ultralytics: each 'r' in results has r.boxes (with .conf and .cls or .xyxy)
                for r in results:
                    boxes = getattr(r, "boxes", None)
                    if boxes is None:
                        # no boxes object — fallback: if results contain any mask/other then consider detection
                        continue

                    confs = getattr(boxes, "conf", None)
                    cls_idxs = getattr(boxes, "cls", None)  # class indices (float/int)
                    xyxy = getattr(boxes, "xyxy", None)

                    # iterate detections
                    if cls_idxs is not None and confs is not None:
                        # cls_idxs and confs are arrays/lists with same length
                        for i, cval in enumerate(confs):
                            try:
                                conf = float(cval)
                            except Exception:
                                conf = 0.0
                            if conf < CONF_THRESH:
                                continue
                            try:
                                cls_idx = int(cls_idxs[i])
                            except Exception:
                                # if can't parse class idx, treat as generic detection
                                cls_idx = None

                            # Determine class name (if possible)
                            cls_name = None
                            if cls_idx is not None:
                                cls_name = _model_class_name(model, cls_idx)
                                if cls_name:
                                    cls_name = cls_name.lower().strip()

                            if is_driver:
                                # only accept 'person'
                                if cls_name == "person" or (cls_idx is None and (xyxy is not None and len(xyxy) > 0)):
                                    detected = True
                                    break
                            else:
                                # non-driver: accept child/doll/teddy classes if present in names
                                if cls_name in CHILD_CLASS_NAMES:
                                    detected = True
                                    break
                                # fallback: if model is custom and class names unknown, accept any high-conf detection
                                if cls_name is None:
                                    detected = True
                                    break
                        # break outer if detected
                        if detected:
                            break
                    else:
                        # No class info available; treat any box with sufficient presence as detection
                        if xyxy is not None and len(xyxy) > 0:
                            detected = True
                            break

                if detected:
                    now = time.time()
                    if now - last_mark >= LAST_SEEN_COOLDOWN:
                        last_mark = now
                        try:
                            callback(username, camera_id)
                        except Exception:
                            traceback.print_exc()

                time.sleep(LOOP_SLEEP)

        except Exception:
            traceback.print_exc()
            time.sleep(2.0)
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

        if not stop_event.is_set():
            time.sleep(reconnect_delay)


def start_worker(username: str, camera_id: str, url: str,
                 callback: Callable[[str, str], None], is_driver: bool = False) -> None:
    """
    Start a background worker for the given camera (no-op if already running).
    callback(username, camera_id) is a synchronous function to call when detection occurs.
    is_driver: whether to use the driver (person-only nano) model.
    """
    with _workers_lock:
        if camera_id in _workers:
            # already running
            return
        stop_event = threading.Event()
        t = threading.Thread(
            target=_run_worker,
            args=(username, camera_id, url, stop_event, callback, is_driver),
            daemon=True,
            name=f"cam-worker-{camera_id[:8]}",
        )
        _workers[camera_id] = {"thread": t, "stop": stop_event, "username": username, "url": url, "is_driver": is_driver, "callback": callback}
        t.start()
        print(f"[camera_worker] started worker for camera {camera_id} (driver={is_driver})")


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
        print(f"[camera_worker] stopped worker for camera {camera_id}")


def restart_worker(camera_id: str, url: Optional[str] = None) -> None:
    """Restart a worker with a (possibly new) url."""
    with _workers_lock:
        info = _workers.get(camera_id)
        username = info["username"] if info else None
        is_driver = info["is_driver"] if info else False
    if info:
        stop_worker(camera_id)
    if username and url:
        start_worker(username, camera_id, url, info["callback"], is_driver)


def start_all_from_db(db: Dict[str, Dict], callback: Callable[[str, str], None]) -> None:
    """
    Start workers for all cameras in the provided DB dict.
    db is the dict loaded from users.json (username -> user dict).
    callback is synchronous function that will be called with (username, camera_id) on detection.
    """
    for username, u in db.items():
        for cam in u.get("cameras", []):
            cam_id = cam.get("camera_id")
            url = cam.get("url")
            is_driver = bool(cam.get("is_driver", False))
            if cam_id and url:
                try:
                    start_worker(username, cam_id, url, callback, is_driver=is_driver)
                except Exception:
                    traceback.print_exc()


def stop_all() -> None:
    with _workers_lock:
        keys = list(_workers.keys())
    for k in keys:
        stop_worker(k)