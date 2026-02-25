import json
import uuid
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import os
import asyncio
import threading
import httpx
from fastapi import FastAPI, HTTPException, Depends, status, Header, Query
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from passlib.context import CryptContext
from jose import jwt, JWTError

# local module
import camera_util
import camera_worker

# Optional google auth to call FCM v1
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleRequest

# ----- CONFIG -----
SECRET_KEY = os.getenv("APP_SECRET_KEY", "replace-with-a-secure-random-secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24))

DB_FILE = Path(os.getenv("DB_FILE", "users.json"))
_db_lock = asyncio.Lock()  # guards writes/reads to the JSON file
_sync_file_lock = threading.Lock()  # used by synchronous helpers (camera util)

# FCM config
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "firebase_config.json")
FCM_SERVER_KEY = os.getenv("FCM_SERVER_KEY", None)  # legacy fallbackw
_FCM_SCOPE = ["https://www.googleapis.com/auth/firebase.messaging"]
_FCM_V1_ENDPOINT_FMT = "https://fcm.googleapis.com/v1/projects/childmobile-ee6f3/messages:send"

# Driver monitor timeout (seconds). Default 30 minutes (1800s). Change to 10 for testing.
DEFAULT_DRIVER_TIMEOUT = 1800

# ----- AUTH helpers -----
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _normalize_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def get_password_hash(password: str) -> str:
    return pwd_context.hash(_normalize_password(password))

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(_normalize_password(plain), hashed)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> dict:
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return payload

# ----- JSON DB helpers (async-safe) -----
async def _read_db() -> Dict[str, dict]:
    def read():
        text = DB_FILE.read_text(encoding="utf-8")
        return json.loads(text) if text.strip() else {}
    return await asyncio.to_thread(read)

async def _write_db(data: Dict[str, dict]) -> None:
    def write():
        tmp = DB_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(DB_FILE)
    await asyncio.to_thread(write)

async def load_user(username: str) -> Optional[dict]:
    db = await _read_db()
    return db.get(username)

async def create_user(username: str, password: str) -> dict:
    async with _db_lock:
        db = await _read_db()
        if username in db:
            raise ValueError("username exists")
        uid = str(uuid.uuid4())
        hashed = get_password_hash(password)
        user = {
            "user_id": uid,
            "username": username,
            "hashed_password": hashed,
            "num_cams": 0,
            "cameras": [],
            "notifications": [],
            "fcm_tokens": [],
            "driver_timeout_seconds": DEFAULT_DRIVER_TIMEOUT,
        }
        db[username] = user
        await _write_db(db)
        return user

async def update_user(username: str, fields: dict) -> dict:
    async with _db_lock:
        db = await _read_db()
        if username not in db:
            raise KeyError("user not found")
        db[username].update(fields)
        await _write_db(db)
        return db[username]

# ----- camera DB helpers (used above) -----
async def list_cameras_for_user(username: str) -> List[dict]:
    user = await load_user(username)
    if not user:
        raise KeyError("user not found")
    return user.get("cameras", [])


async def add_camera_to_user(username: str, name: str, url: str) -> dict:
    async with _db_lock:
        db = await _read_db()
        if username not in db:
            raise KeyError("user not found")
        cam = {
            "camera_id": str(uuid.uuid4()),
            "name": name,
            "url": url,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "is_driver": False,
            "last_human_seen": None,
            "missing_notified": False,
        }
        db[username].setdefault("cameras", []).append(cam)
        db[username]["num_cams"] = len(db[username]["cameras"])
        await _write_db(db)
        try:
            camera_worker.start_worker(username, cam["camera_id"], cam["url"], mark_last_seen_sync,is_driver=cam["is_driver"])
        except Exception:
            pass
        return cam


async def get_camera_by_index(username: str, index: int) -> Optional[dict]:
    user = await load_user(username)
    if not user:
        return None
    cams = user.get("cameras", [])
    if index < 0 or index >= len(cams):
        return None
    return cams[index]


async def delete_camera_by_id(username: str, camera_id: str) -> bool:
    async with _db_lock:
        db = await _read_db()
        if username not in db:
            raise KeyError("user not found")
        cams = db[username].get("cameras", [])
        new_cams = [c for c in cams if c.get("camera_id") != camera_id]
        if len(new_cams) == len(cams):
            return False
        db[username]["cameras"] = new_cams
        db[username]["num_cams"] = len(new_cams)
        await _write_db(db)
        camera_worker.stop_worker(camera_id)
        return True

# Synchronous helper used by camera_util to set last_human_seen safely from threads/processes
def mark_last_seen_sync(username: str, camera_id: str) -> None:
    with _sync_file_lock:
        if not DB_FILE.exists():
            return
        text = DB_FILE.read_text(encoding="utf-8")
        db = json.loads(text) if text.strip() else {}
        user = db.get(username)
        if not user:
            return
        cams = user.setdefault("cameras", [])
        for cam in cams:
            if cam.get("camera_id") == camera_id:
                cam["last_human_seen"] = datetime.utcnow().isoformat() + "Z"
                cam["missing_notified"] = False
                break
        DB_FILE.write_text(json.dumps(db, indent=2, ensure_ascii=False), encoding="utf-8")


# Async wrapper (convenience)
async def _set_camera_last_seen(username: str, camera_id: str):
    await asyncio.to_thread(mark_last_seen_sync, username, camera_id)


# --- device token helpers ---
async def register_device_token(username: str, device_token: str) -> None:
    async with _db_lock:
        db = await _read_db()
        user = db.get(username)
        if not user:
            raise KeyError("user not found")
        tokens = user.setdefault("fcm_tokens", [])
        if device_token not in tokens:
            tokens.append(device_token)
            await _write_db(db)


async def unregister_device_token(username: str, device_token: str) -> None:
    async with _db_lock:
        db = await _read_db()
        user = db.get(username)
        if not user:
            return
        tokens = user.get("fcm_tokens", [])
        if device_token in tokens:
            tokens.remove(device_token)
            await _write_db(db)


# ----- FCM sending (modern v1 if possible) -----
_cached_credentials = None
_cached_project_id = None
_cached_access_token = None
_cached_access_token_expiry = None


def _load_service_account_creds():
    global _cached_credentials, _cached_project_id
    if _cached_credentials is not None:
        return _cached_credentials
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=_FCM_SCOPE
    )
    _cached_credentials = creds
    # read project_id
    try:
        data = json.loads(Path(SERVICE_ACCOUNT_FILE).read_text(encoding="utf-8"))
        _cached_project_id = data.get("project_id")
    except Exception:
        _cached_project_id = None
    return creds


async def _get_fcm_access_token() -> Optional[str]:
    """
    Returns a fresh OAuth2 access token for FCM v1 (Bearer). Caches token until expiry.
    """
    global _cached_access_token, _cached_access_token_expiry
    creds = _load_service_account_creds()
    if not creds:
        return None
    # token refresh is synchronous; run in thread to avoid blocking event loop
    def refresh_and_get():
        nonlocal creds
        creds.refresh(GoogleRequest())
        return creds.token, creds.expiry

    token, expiry = await asyncio.to_thread(refresh_and_get)
    _cached_access_token = token
    _cached_access_token_expiry = expiry
    return token

async def _send_push_v1_single(token: str, title: str, body: str, data: dict | None = None) -> None:
    creds_token = await _get_fcm_access_token()
    if not creds_token:
        # fallback will be handled by caller
        raise RuntimeError("No service account credentials available for FCM v1")
    # get project id
    global _cached_project_id
    if not _cached_project_id:
        try:
            data = json.loads(Path(SERVICE_ACCOUNT_FILE).read_text(encoding="utf-8"))
            _cached_project_id = data.get("project_id")
        except Exception:
            _cached_project_id = None
    if not _cached_project_id:
        raise RuntimeError("No project_id found in service account file")

    url = _FCM_V1_ENDPOINT_FMT.format(project_id=_cached_project_id)
    payload = {
        "message": {
            "token": token,
            "notification": {"title": title, "body": body},
            "data": data or {},
            "android": {"priority": "high"},
            "apns": {"headers": {"apns-priority": "10"}},
        }
    }
    headers = {"Authorization": f"Bearer {creds_token}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(url, json=payload, headers=headers)
        if r.status_code not in (200, 201):
            print("FCM v1 send failed:", r.status_code, r.text)

async def _send_push_to_tokens(tokens: List[str], title: str, body: str, data: dict | None = None):
    """
    Top-level push sender: try FCM v1 via service account per-token; if not available, fallback to legacy batch send.
    """
    if not tokens:
        return
    creds = _load_service_account_creds()
    if creds:
        # send per-token via v1 (one request per token). Do in parallel.
        tasks = [_send_push_v1_single(t, title, body, data) for t in tokens]
        # don't raise if one fails; log instead
        await asyncio.gather(*[ _catch_and_log(t) for t in tasks ])
        return
    # no available sending method
    print("No FCM credentials configured; cannot send push")

async def _catch_and_log(coro):
    try:
        await coro
    except Exception as e:
        print("FCM send error (individual):", e)

async def _send_push_to_user(username: str, title: str, body: str, data: dict | None = None):
    db = await _read_db()
    user = db.get(username)
    if not user:
        return
    tokens = user.get("fcm_tokens", []) or []
    if tokens:
        # fire-and-forget send in background
        asyncio.create_task(_send_push_to_tokens(tokens, title, body, data))

# store notification in DB and send push
async def _add_notification(username: str, message: str) -> None:
    async with _db_lock:
        db = await _read_db()
        user = db.get(username)
        if not user:
            return
        note = {
            "id": str(uuid.uuid4()),
            "message": message,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "read": False,
        }
        user.setdefault("notifications", []).append(note)
        await _write_db(db)

    title = "Driver camera alert"
    data_payload = {"type": "driver_missing", "message": message}
    asyncio.create_task(_send_push_to_user(username, title, message, data_payload))


# background watcher: run periodically and create notifications
async def _driver_monitor_loop():
    """
    Runs periodically; for each user, check driver cameras:
    - if last_human_seen > DRIVER_MISSING_TIMEOUT_SECONDS ago and not notified, create notification.
    - If any other camera has human within timeout, include that in message.
    """
    sleep_seconds = max(1, int(os.getenv("DRIVER_MONITOR_INTERVAL_SECONDS", 60)))
    while True:
        try:
            db = await _read_db()
            now = datetime.utcnow()
            timeout = timedelta(seconds=int(u.get("driver_timeout_seconds")))
            for username, u in db.items():
                for cam in u.get("cameras", []):
                    if not cam.get("is_driver"):
                        continue
                    last_iso = cam.get("last_human_seen")
                    notified = bool(cam.get("missing_notified", False))
                    last_dt = None
                    if last_iso:
                        try:
                            last_dt = datetime.fromisoformat(last_iso.replace("Z", ""))
                        except Exception:
                            last_dt = None
                    if last_dt and (now - last_dt) > timeout and not notified:
                        # check other cameras for recent human presence (within timeout)
                        other_human_found = False
                        for oc in u.get("cameras", []):
                            if oc.get("camera_id") == cam.get("camera_id"):
                                continue
                            lt = oc.get("last_human_seen")
                            if lt:
                                try:
                                    ldt = datetime.fromisoformat(lt.replace("Z", ""))
                                    if (now - ldt) <= timeout:
                                        other_human_found = True
                                        break
                                except Exception:
                                    pass

                        msg = f"No human detected on driver camera '{cam.get('name')}' for {int(timeout.total_seconds())}s"
                        if other_human_found:
                            msg += "; but detected on other cameras — please check."
                        await _add_notification(username, msg)

                        # set camera missing_notified flag atomically
                        async with _db_lock:
                            db2 = await _read_db()
                            uu = db2.get(username)
                            if uu:
                                for c2 in uu.get("cameras", []):
                                    if c2.get("camera_id") == cam.get("camera_id"):
                                        c2["missing_notified"] = True
                                        await _write_db(db2)
                                        break
        except Exception as e:
            print("driver monitor loop error:", e)
        await asyncio.sleep(sleep_seconds)

# --- driver timeout helpers --------------------------------
async def get_driver_timeout_for_user(username: str) -> int:
    db = await _read_db()
    user = db.get(username)
    if not user:
        return DEFAULT_DRIVER_TIMEOUT
    try:
        v = int(user.get("driver_timeout_seconds", DEFAULT_DRIVER_TIMEOUT))
        if v < 1:
            return DEFAULT_DRIVER_TIMEOUT
        return v
    except Exception:
        return DEFAULT_DRIVER_TIMEOUT


async def set_driver_timeout_for_user(username: str, seconds: int) -> int:
    if seconds < 1:
        raise ValueError("timeout must be >= 1 second")
    async with _db_lock:
        db = await _read_db()
        if username not in db:
            raise KeyError("user not found")
        db[username]["driver_timeout_seconds"] = int(seconds)
        await _write_db(db)
        return int(seconds)

# ----- Pydantic Models -----
class LoginRequest(BaseModel):
    username: str
    password: str

class SignupRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    token: str

class CameraCreate(BaseModel):
    name: str = Field(..., min_length=1)
    url: str = Field(..., min_length=4)

class CameraOut(BaseModel):
    camera_id: str
    name: str
    url: str
    created_at: str
    is_driver: bool = False
    last_human_seen: Optional[str] = None
    missing_notified: bool = False

class UserOut(BaseModel):
    user_id: str
    username: str
    num_cams: int
    cameras: List[CameraOut] = []

class CameraUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None

class DriverUpdate(BaseModel):
    is_driver: bool

class DeviceTokenIn(BaseModel):
    token: str
class DriverTimeoutIn(BaseModel):
    driver_timeout_seconds: int
# ----- FastAPI app -----
app = FastAPI(title="Auth + Camera API with Driver Monitor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Auth dependency -----
async def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header format")
    token = parts[1]
    try:
        payload = decode_access_token(token)
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user = await load_user(username)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

# ----- Routes -----
@app.post("/signup")
async def signup(payload: SignupRequest):
    username = payload.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="username required")
    try:
        user = await create_user(username, payload.password)
    except ValueError:
        raise HTTPException(status_code=400, detail="username already exists")
    access_token = create_access_token({"sub": user["username"], "uid": user["user_id"]})
    return {"user_id": user["user_id"], "token": access_token}

@app.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest):
    user = await load_user(payload.username)
    if not user or not verify_password(payload.password, user["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token = create_access_token({"sub": user["username"], "uid": user["user_id"]})
    return {"token": access_token}

@app.get("/current", response_model=UserOut)
async def get_current(user: dict = Depends(get_current_user)):
    return {
        "user_id": user["user_id"],
        "username": user["username"],
        "num_cams": user.get("num_cams", 1),
        "cameras": user.get("cameras", []),
    }

@app.get("/num_cam")
async def get_num_cam(user: dict = Depends(get_current_user)):
    return {"num": len(user.get("cameras", []))}

@app.get("/cameras", response_model=List[CameraOut])
async def list_cameras(user: dict = Depends(get_current_user)):
    return user.get("cameras", [])

@app.post("/cameras", response_model=CameraOut, status_code=201)
async def create_camera(payload: CameraCreate, user: dict = Depends(get_current_user)):
    cam = await add_camera_to_user(user["username"], payload.name, payload.url)
    return cam

@app.delete("/cameras/{camera_id}", status_code=204)
async def remove_camera(camera_id: str, user: dict = Depends(get_current_user)):
    ok = await delete_camera_by_id(user["username"], camera_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Camera not found")
    return Response(status_code=204)

@app.post("/cameras/{camera_id}/driver")
async def set_camera_driver_flag(camera_id: str, payload: DriverUpdate, user: dict = Depends(get_current_user)):
    async with _db_lock:
        db = await _read_db()
        u = db.get(user["username"])
        if not u:
            raise HTTPException(status_code=404, detail="User not found")
        cams = u.setdefault("cameras", [])
        for cam in cams:
            if cam.get("camera_id") == camera_id:
                camera_worker.stop_worker(camera_id)
                cam["is_driver"] = bool(payload.is_driver)
                camera_worker.start_worker(u, camera_id, cam["url"], mark_last_seen_sync,is_driver=cam["is_driver"])
                if payload.is_driver:
                    cam.setdefault("last_human_seen", None)
                    cam["missing_notified"] = False
                await _write_db(db)
                return {"camera_id": camera_id, "is_driver": cam["is_driver"]}
        raise HTTPException(status_code=404, detail="Camera not found")

@app.post("/devices/register")
async def register_device(payload: DeviceTokenIn, user: dict = Depends(get_current_user)):
    try:
        await register_device_token(user["username"], payload.token)
    except KeyError:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "ok"}

@app.post("/devices/unregister")
async def unregister_device(payload: DeviceTokenIn, user: dict = Depends(get_current_user)):
    try:
        await unregister_device_token(user["username"], payload.token)
    except KeyError:
        pass
    return {"status": "ok"}

@app.get("/notifications")
async def get_notifications(user: dict = Depends(get_current_user)):
    # return user's notifications (newest first)
    notes = list(reversed(user.get("notifications", [])))
    return JSONResponse(content=notes)

@app.get("/overlay")
async def overlay(username: str = Query(...), cam_index: int = Query(...)):
    """
    Stream MJPEG frames with detection overlay for the requested user's camera.
    Example: /overlay?username=nai&cam_index=0
    """
    user = await load_user(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    cameras = user.get("cameras", [])
    if cam_index < 0 or cam_index >= len(cameras):
        raise HTTPException(status_code=404, detail="Invalid camera index")

    cam = cameras[cam_index]

    # Pass the synchronous mark_last_seen_sync callback to the generator so it can update DB
    gen = camera_util.gen_img(
        cam["url"],
        username=username,
        camera_id=cam["camera_id"],
        callback=mark_last_seen_sync,  # synchronous callback safe for threads
        is_driver=cam["is_driver"]
    )

    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/settings/driver_timeout")
async def get_driver_timeout(user: dict = Depends(get_current_user)):
    val = await get_driver_timeout_for_user(user["username"])
    return {"driver_timeout_seconds": val}

@app.post("/settings/driver_timeout")
async def post_driver_timeout(payload: DriverTimeoutIn, user: dict = Depends(get_current_user)):
    secs = int(payload.driver_timeout_seconds)
    if secs < 1 or secs > 60 * 60 * 24:
        raise HTTPException(status_code=400, detail = "driver_timeout_seconds out of range")
    try:
        updated = await set_driver_timeout_for_user(user["username"],secs)
    except KeyError:
        raise HTTPException(stats_code=404, detail = "User not found")
    return {"driver_timeout_seconds": updated}

@app.get("/")
async def root():
    return {"message": "FastAPI server running."}


# ----- startup tasks -----
@app.on_event("startup")
async def start_background_tasks():
    # ensure DB exists
    async with _db_lock:
        db = await _read_db()
        if not db:
            uid = str(uuid.uuid4())
            user = {
                "user_id": uid,
                "username": "nai",
                "hashed_password": get_password_hash("nai"),
                "num_cams": 0,
                "cameras": [],
                "notifications": [],
                "fcm_tokens": [],
                "driver_timeout_seconds": DEFAULT_DRIVER_TIMEOUT,
            }
            db["nai"] = user
            await _write_db(db)
    # start driver monitor
    asyncio.create_task(_driver_monitor_loop())

    # start persistent camera detection worker
    try:
        db_snapshot = await _read_db()
        camera_worker.start_all_from_db(db_snapshot,mark_last_seen_sync)
    except Exception:
        import traceback
        traceback.print_exc()
