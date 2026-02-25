# utils/camera.py
import cv2
import time
import asyncio
import os
from ultralytics import YOLO

# Load model once (adjust path if needed)
MODEL_PATH_CHILD = "best.pt"
MODEL_PATH_DRIVER = "yolo26n.pt"
child_model = YOLO(MODEL_PATH_CHILD)
driver_model = YOLO(MODEL_PATH_DRIVER)
mnames_child = getattr(child_model, "names", None)
mnames_driver = getattr(driver_model, "names", None)

def get_names(names):
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    elif isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    else:
        return {}

child_names = get_names(mnames_child)
driver_names = get_names(mnames_driver)

# confidence threshold default
DEFAULT_CONFIDENCE = 0.6

child_class_idx = 0
driver_class_idx = 0

def _is_coroutine_callable(func):
    """Return True if calling func returns a coroutine or if func is an async function."""
    if func is None:
        return False
    import inspect
    if inspect.iscoroutinefunction(func):
        return True
    # if it's a normal function that returns coroutine when called, we detect later
    return False

def gen_img(source,
            is_driver,
            username: str | None = None,
            camera_id: str | None = None,
            callback=None,
            conf_thresh: float = DEFAULT_CONFIDENCE,
            resize_width: int | None = None):
    """
    Generator that yields multipart JPEG frames with bounding boxes.

    Args:
      source: video source for cv2.VideoCapture (int for webcam index, rtsp/http url, or file path)
      username: optional username (passed to callback)
      camera_id: optional camera_id (passed to callback)
      callback: optional callable to notify when 'child' detected:
                - if callback returns a coroutine or is async, it will be scheduled with asyncio.create_task(...)
                - callback will be called as callback(username, camera_id)
      conf_thresh: confidence threshold for drawing boxes
      resize_width: optional width to resize frame to (preserve aspect ratio)

    Yields:
      multipart JPEG bytes (b'--frame...').
    """
    def choose_model():
        if is_driver:
            return driver_model
        else:
            return child_model
        
    def choose_class_idx():
        if is_driver:
            return driver_class_idx
        else:
            return child_class_idx

    def choose_names():
        if is_driver:
            return driver_names
        else:
            return child_names
        
    model = choose_model()
    class_idx = choose_class_idx()
    names = choose_names()
    # Open capture (works for webcam index or URL/file)
    try:
        cap = cv2.VideoCapture(source)
    except Exception as e:
        print("camera_utils: cv2.VideoCapture() failed:", e)
        return

    if not cap.isOpened():
        print(f"camera_utils: failed to open source: {source}")
        return

    # small helper to call callback safely
    def _maybe_call_callback(username, camera_id):
        if callback is None:
            return
        try:
            res = callback(username, camera_id)
            # if it's a coroutine object, schedule it
            if asyncio.iscoroutine(res):
                try:
                    asyncio.create_task(res)
                except RuntimeError:
                    # if no running loop, ignore (server should have an event loop)
                    pass
        except Exception:
            # if callback raises synchronously, ignore
            try:
                # if callback is async function, schedule it
                if _is_coroutine_callable(callback):
                    asyncio.create_task(callback(username, camera_id))
            except Exception:
                pass

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                # short sleep then break to avoid tight loop on error
                time.sleep(0.05)
                break

            # optionally resize to speed up inference
            if resize_width:
                h, w = frame.shape[:2]
                if w != resize_width:
                    new_h = int(h * (resize_width / w))
                    frame = cv2.resize(frame, (resize_width, new_h))

            t0 = time.time()
            # run inference (model returns a Results list)
            try:
                results = model(frame)
            except Exception as e:
                # inference error — print and continue streaming original frame
                print("camera_utils: model inference error:", e)
                results = []

            inference_time = (time.time() - t0) * 1000.0

            # draw detections (ultralytics returns a list; iterate each result)
            # Usually we do: res = results[0]
            try:
                if len(results) > 0:
                    res = results[0]
                    boxes = getattr(res, "boxes", None)
                    if boxes is not None:
                        # get attributes if available
                        xyxy_attr = getattr(boxes, "xyxy", None)
                        conf_attr = getattr(boxes, "conf", None)
                        cls_attr = getattr(boxes, "cls", None)

                        # convert to python lists (some objects are tensors)
                        # safest: iterate via index using length of xyxy
                        n = 0
                        try:
                            n = len(xyxy_attr)
                        except Exception:
                            try:
                                # boxes.xyxy might be a tensor-like -> convert to .cpu().numpy() if possible
                                arr = xyxy_attr.cpu().numpy()
                                n = arr.shape[0]
                            except Exception:
                                n = 0

                        for i in range(n):
                            try:
                                # read bbox coords
                                raw = xyxy_attr[i]
                                # raw may be numpy array or tensor, convert to floats
                                x1 = int(float(raw[0]))
                                y1 = int(float(raw[1]))
                                x2 = int(float(raw[2]))
                                y2 = int(float(raw[3]))
                            except Exception:
                                continue

                            # confidence and class
                            try:
                                conf = float(conf_attr[i])
                            except Exception:
                                conf = 0.0
                            try:
                                cls_id = int(cls_attr[i])
                            except Exception:
                                cls_id = -1

                            if conf < conf_thresh:
                                continue

                            label = names.get(cls_id, str(cls_id))

                            # draw rectangle and label
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                            text = f"{label} {conf:.2f}"
                            # put text background
                            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw, y1), (0, 165, 255), -1)
                            cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                            # If this detection is a 'child' (by name or by index), call callback
                            if class_idx is not None and cls_id == class_idx:
                                # schedule callback without blocking
                                _maybe_call_callback(username, camera_id)

                # annotate performance on frame (optional small overlay)
                fps_text = f"Inference: {inference_time:.0f}ms"
                cv2.putText(frame, fps_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except Exception as e:
                # drawing/processing error — print and continue
                print("camera_utils: drawing error:", e)

            # encode to JPEG
            try:
                ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
            except Exception as e:
                print("camera_utils: jpeg encode failed:", e)
                continue

            # yield multipart chunk
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    finally:
        try:
            cap.release()
        except Exception:
            pass
