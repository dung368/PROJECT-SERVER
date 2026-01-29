# main.py
from fastapi import FastAPI, HTTPException, Depends, status, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta

# ----- CONFIG -----
SECRET_KEY = "replace-with-a-secure-random-secret"  # << replace before prod
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

# Example configuration: how many cameras available
NUM_CAMERAS = 4

# ----- PASSWORD / AUTH HELPERS -----
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> dict:
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return payload

# ----- FAKE USER DB -----
# For demo/testing only. Replace with real DB in production.
fake_users_db: Dict[str, dict] = {
    "alice": {
        "username": "alice",
        "full_name": "Alice Example",
        "email": "alice@example.com",
        "hashed_password": get_password_hash("secret123"),
        "disabled": False,
    },
    "bob": {
        "username": "bob",
        "full_name": "Bob Example",
        "email": "bob@example.com",
        "hashed_password": get_password_hash("password"),
        "disabled": False,
    },
}

# ----- Pydantic MODELS -----
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    token: str

class UserOut(BaseModel):
    username: str
    full_name: Optional[str] = None
    email: Optional[str] = None

# ----- APP + CORS -----
app = FastAPI(title="Simple Auth + Camera API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- AUTH DEPENDENCY -----
def get_current_user(authorization: Optional[str] = Header(None)):
    """
    Expects header: Authorization: Bearer <token>
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
    user = fake_users_db.get(username)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    if user.get("disabled"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User disabled")
    return user

# ----- ROUTES -----
@app.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest):
    """
    POST /login
    Body: { "username": "...", "password": "..." }
    Response: { "token": "..." }  (the Dart client expects `res.body['token']`)
    """
    user = fake_users_db.get(payload.username)
    if not user or not verify_password(payload.password, user["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token = create_access_token({"sub": user["username"]})
    return {"token": access_token}

@app.get("/current", response_model=UserOut)
async def get_current(user: dict = Depends(get_current_user)):
    """
    Protected: Returns current user info as JSON.
    """
    return {
        "username": user["username"],
        "full_name": user.get("full_name"),
        "email": user.get("email"),
    }

@app.get("/num_cam")
async def get_num_cam(user: dict = Depends(get_current_user)):
    """
    Protected: Returns {"num": <int>}
    """
    return {"num": NUM_CAMERAS}

@app.get("/cam{cam_id}")
@app.get("/cam/{cam_id}")
async def cam_info(cam_id: int):
    """
    Two routes so both /cam1 and /cam/1 work.
    Return a small JSON describing the camera. In a real app this could redirect
    to a stream endpoint, return MJPEG, or proxy a camera feed.
    """
    if cam_id < 1 or cam_id > NUM_CAMERAS:
        raise HTTPException(status_code=404, detail="Camera not found")
    # Example: provide a streaming URL or any metadata the client needs
    stream_url = f"/stream/cam{cam_id}"  # example local stream endpoint (not implemented here)
    return {
        "cam_id": cam_id,
        "name": f"Camera {cam_id}",
        "stream_url": stream_url,
        "description": f"Placeholder info for camera {cam_id}",
    }

# Optional: a simple health check
@app.get("/")
async def root():
    return {"message": "FastAPI server running. Use POST /login to obtain token."}
