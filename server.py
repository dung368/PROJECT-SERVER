import json
import uuid
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import asyncio
from fastapi import FastAPI, HTTPException, Depends, status, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
from passlib.context import CryptContext
from jose import jwt, JWTError
from fastapi.responses import StreamingResponse
import cv2
import utils.camera as camutil

# ----- CONFIG -----
SECRET_KEY = "replace-with-a-secure-random-secret"  # replace in prod
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

DB_FILE = Path("users.json")
_db_lock = asyncio.Lock()  # guards writes/reads to the JSON file

# ----- PASSWORD / AUTH HELPERS -----
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _normalize_password(password: str) -> str:
    """Pre-hash password with SHA-256 and return hex digest."""
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
    if not DB_FILE.exists():
        return {}
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


async def create_user(username: str, password: str, full_name: Optional[str], email: Optional[str],
                      num_cams: int) -> dict:
    async with _db_lock:
        db = await _read_db()
        if username in db:
            raise ValueError("username exists")
        uid = str(uuid.uuid4())
        hashed = get_password_hash(password)
        user = {
            "user_id": uid,
            "username": username,
            "full_name": full_name,
            "email": email,
            "hashed_password": hashed,
            "disabled": False,
            "num_cams": int(num_cams),
            "cameras": [],  # list of camera dicts
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


# ----- Camera helpers -----
def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


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
            "created_at": _now_iso(),
        }
        db[username].setdefault("cameras", []).append(cam)
        # keep num_cams in sync
        db[username]["num_cams"] = len(db[username]["cameras"])
        await _write_db(db)
        return cam


async def get_camera_by_index(username: str, index: int) -> Optional[dict]:
    user = await load_user(username)
    if not user:
        return None
    cams = user.get("cameras", [])
    if index < 0 or index >= len(cams):
        return None
    return cams[index]

async def check_stream_url(username: str, url: int) -> bool:
    user = await load_user(username)
    if not user:
        return None
    cams = user.get("cameras", [])
    print(cams)
    for i in cams:
        if (i['url'] == url): return 1
    return 0


# ----- Pydantic MODELS -----
class LoginRequest(BaseModel):
    username: str
    password: str


class SignupRequest(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None
    email: Optional[str] = None
    num_cams: Optional[int] = 1


class TokenResponse(BaseModel):
    token: str


class CameraCreate(BaseModel):
    name: str = Field(..., min_length=1)
    url: str = Field(..., min_length=8)  # simple validation; ensure contains .m3u8 in runtime


class CameraOut(BaseModel):
    camera_id: str
    name: str
    url: str
    created_at: str


class UserOut(BaseModel):
    user_id: str
    username: str
    full_name: Optional[str] = None
    email: Optional[str] = None
    num_cams: int
    cameras: List[CameraOut] = []



# ----- APP + CORS -----
app = FastAPI(title="Simple Auth + Camera API (JSON DB + Cameras)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- AUTH DEPENDENCY -----
async def get_current_user(authorization: Optional[str] = Header(None)):
    """
    Expects header: Authorization: Bearer <token>
    Returns the user dict from JSON DB
    """
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
    if user.get("disabled"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User disabled")
    return user


# ----- ROUTES -----
@app.post("/signup")
async def signup(payload: SignupRequest):
    username = payload.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="username required")
    if payload.num_cams is None or payload.num_cams < 1:
        raise HTTPException(status_code=400, detail="num_cams must be >= 1")
    try:
        user = await create_user(username, payload.password, payload.full_name, payload.email, payload.num_cams)
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
        "full_name": user.get("full_name"),
        "email": user.get("email"),
        "num_cams": user.get("num_cams", 1),
        "cameras": user.get("cameras", []),
    }

@app.get("/num_cam")
async def get_num_cam(user: dict = Depends(get_current_user)):
    """
    Returns the count of cameras for the authenticated user
    """
    return {"num": len(user.get("cameras", []))}


# ----- camera management endpoints -----
@app.get("/cameras", response_model=List[CameraOut])
async def list_cameras(user: dict = Depends(get_current_user)):
    return user.get("cameras", [])


@app.post("/cameras", response_model=CameraOut, status_code=201)
async def create_camera(payload: CameraCreate, user: dict = Depends(get_current_user)):
    # basic url validation: require http/https and .m3u8 in url
    # if not (payload.url.startswith("http://") or payload.url.startswith("https://")):
    #     raise HTTPException(status_code=400, detail="url must start with http:// or https://")
    # if ".m3u8" not in payload.url:
    #     raise HTTPException(status_code=400, detail="url must point to an HLS playlist (contain .m3u8)")
    cam = await add_camera_to_user(user["username"], payload.name, payload.url)
    return cam


@app.get("/cameras/{camera_id}", response_model=CameraOut)
async def get_camera(camera_id: str, user: dict = Depends(get_current_user)):
    cams = user.get("cameras", [])
    for c in cams:
        if c.get("camera_id") == camera_id:
            return c
    raise HTTPException(status_code=404, detail="Camera not found")


@app.get("/cam/{cam_id}")
async def cam_info(cam_id: int, user: dict = Depends(get_current_user)):
    """
    Return camera metadata for the authenticated user by 0-based index (cam_id).
    Example: GET /cam/0 returns the first camera.
    """
    cam = await get_camera_by_index(user["username"], cam_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found for this user")
    # return a simple JSON with the HLS URL (exact format user asked)
    return {
        "user_id": user["user_id"],
        "cam_id": cam_id,
        "camera_id": cam["camera_id"],
        "name": cam["name"],
        "stream_url": cam["url"],  # should be like https://.../nai-cam0/index.m3u8
        "created_at": cam.get("created_at"),
    }


@app.get("/cam/stream/{cam_id}")
async def cam_info(cam_id: int, user: dict = Depends(get_current_user)):
    """
    Return camera metadata for the authenticated user by 0-based index (cam_id).
    Example: GET /cam/0 returns the first camera.
    """
    cam = await get_camera_by_index(user["username"], cam_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found for this user")
    # return a simple JSON with the HLS URL (exact format user asked)
    return {
        "user_id": user["user_id"],
        "cam_id": cam_id,
        "camera_id": cam["camera_id"],
        "name": cam["name"],
        "stream_url": cam["url"],  # should be like https://.../nai-cam0/index.m3u8
        "created_at": cam.get("created_at"),
    }
@app.get("/overlay")
async def cam_info(url: str):
    """
    Return camera metadata for the authenticated user by 0-based index (cam_id).
    Example: GET /cam/0 returns the first camera.
    """

    # cam = await get_camera_by_index(user["username"], cam_id)
    # kt = await check_stream_url(user["username"], url)
    # if not kt:
    #     raise HTTPException(status_code=404, detail="Camera not found for this user")
    # # return a simple JSON with the HLS URL (exact format user asked)
    # url = cam["url"]    
    return StreamingResponse(camutil.gen_img(url), media_type="multipart/x-mixed-replace; boundary=frame")

# Optional: a simple health check
@app.get("/")
async def root():
    return {"message": "FastAPI server running. Use POST /signup or POST /login to obtain token."}



# ----- Startup: ensure DB exists and create a test user if empty -----
@app.on_event("startup")
async def ensure_db():
    async with _db_lock:
        db = await _read_db()
        if not db:
            uid = str(uuid.uuid4())
            user = {
                "user_id": uid,
                "username": "nai",
                "full_name": None,
                "email": None,
                "hashed_password": get_password_hash("nai"),
                "disabled": False,
                "num_cams": 0,
                "cameras": [],
            }
            db["nai"] = user
            await _write_db(db)
