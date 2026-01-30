# main.py
import json
import uuid
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict

import asyncio
from fastapi import FastAPI, HTTPException, Depends, status, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import jwt, JWTError

# ----- CONFIG -----
SECRET_KEY = "replace-with-a-secure-random-secret"  # replace in prod
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

DB_FILE = Path("users.json")
_db_lock = asyncio.Lock()  # guards writes/reads to the JSON file

# ----- PASSWORD / AUTH HELPERS -----
# Use bcrypt via passlib but pre-hash with SHA-256 to avoid bcrypt 72-bytes limit
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
    # offload IO to thread to avoid blocking event loop
    def read():
        text = DB_FILE.read_text(encoding="utf-8")
        return json.loads(text) if text.strip() else {}

    return await asyncio.to_thread(read)


async def _write_db(data: Dict[str, dict]) -> None:
    # write atomically by writing to temp file then renaming (safer)
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


class UserOut(BaseModel):
    user_id: str
    username: str
    full_name: Optional[str] = None
    email: Optional[str] = None
    num_cams: int


class NumCamUpdate(BaseModel):
    number_cam: int


# ----- APP + CORS -----
app = FastAPI(title="Simple Auth + Camera API (JSON DB)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change for production
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
    """
    Create a new user and return user_id + token.
    Body: { username, password, full_name?, email?, num_cams? }
    """
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
    """
    POST /login
    Body: { "username": "...", "password": "..." }
    Response: { "token": "..." }
    """
    user = await load_user(payload.username)
    if not user or not verify_password(payload.password, user["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token = create_access_token({"sub": user["username"], "uid": user["user_id"]})
    return {"token": access_token}


@app.get("/current", response_model=UserOut)
async def get_current(user: dict = Depends(get_current_user)):
    """
    Protected: Returns current user info as JSON.
    """
    return {
        "user_id": user["user_id"],
        "username": user["username"],
        "full_name": user.get("full_name"),
        "email": user.get("email"),
        "num_cams": user.get("num_cams", 1),
    }


@app.post("/num_cam_update")
async def update_num_cam(payload: NumCamUpdate, user: dict = Depends(get_current_user)):
    """Update the authenticated user's number of cameras"""
    number_cam = int(payload.number_cam)
    if number_cam < 1:
        raise HTTPException(status_code=400, detail="number_cam must be >= 1")
    updated = await update_user(user["username"], {"num_cams": number_cam})
    return {"status": "success", "num": updated["num_cams"]}


@app.get("/num_cam")
async def get_num_cam(user: dict = Depends(get_current_user)):
    """
    Protected: Returns {"num": <int>} for the authenticated user
    """
    return {"num": user.get("num_cams", 1)}


@app.get("/cam{cam_id}")
@app.get("/cam/{cam_id}")
async def cam_info(cam_id: int, user: dict = Depends(get_current_user)):
    """
    Protected camera info for the authenticated user. Enforces per-user camera count.
    """
    user_num = user.get("num_cams", 1)
    if cam_id >= user_num:
        raise HTTPException(status_code=404, detail="Camera not found for this user")
    stream_url = f"https://53e8de49be53.ngrok-free.app/nai-cam0/"
    return stream_url;
    # return {
    #     "user_id": user["user_id"],
    #     "cam_id": cam_id,
    #     "name": f"{user.get('username')}-cam{cam_id}",
    #     "stream_url": stream_url,
    #     "description": f"Camera {cam_id} for user {user.get('username')}",
    # }


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
            # Create a default user 'nai' for testing
            uid = str(uuid.uuid4())
            user = {
                "user_id": uid,
                "username": "nai",
                "full_name": None,
                "email": None,
                "hashed_password": get_password_hash("nai"),
                "disabled": False,
                "num_cams": 2,
            }
            db["nai"] = user
            await _write_db(db)
