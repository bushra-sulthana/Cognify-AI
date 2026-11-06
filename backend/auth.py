
import hashlib
import hmac
import os
import secrets
import sqlite3
import smtplib
import time

from email.message import EmailMessage
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field

from backend import config

router = APIRouter(prefix="/api/auth", tags=["authentication"])

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "users.db"


def _create_tables(conn):
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS password_resets (
                email TEXT PRIMARY KEY,
                token_hash TEXT NOT NULL,
                expires_at INTEGER NOT NULL
            )
        """)


def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _create_tables(conn)      # idempotent, so it also works before startup has run
    return conn


def init_auth_db():
    get_db().close()


# ---- simple in-memory throttle against password guessing ------------------ #
_FAILED: dict = {}
_MAX_FAILS, _WINDOW = 5, 300      # 5 failed logins per 5 minutes per email


def _throttled(email: str) -> bool:
    now = time.time()
    attempts = [ts for ts in _FAILED.get(email, []) if now - ts < _WINDOW]
    _FAILED[email] = attempts
    return len(attempts) >= _MAX_FAILS


def require_user(request: Request) -> dict:
    """FastAPI dependency: the endpoint needs a logged-in user."""
    if not config.AUTH_REQUIRED:
        return {"email": "local"}
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Please log in.")
    return {"id": user_id, "email": request.session.get("email")}


def hash_password(password: str, salt: bytes | None = None):
    salt = salt or secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt, 600_000
    )
    return salt.hex(), derived.hex()


def password_matches(password: str, salt_hex: str, stored_hash: str):
    _, candidate = hash_password(password, bytes.fromhex(salt_hex))
    return hmac.compare_digest(candidate, stored_hash)


class SignupData(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)


class LoginData(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class ForgotData(BaseModel):
    email: EmailStr


class ResetData(BaseModel):
    token: str
    new_password: str = Field(min_length=10, max_length=128)


def send_reset_email(email: str, token: str):
    host = os.getenv("SMTP_HOST")
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM", username or "")
    port = int(os.getenv("SMTP_PORT", "587"))

    if not all([host, username, password, sender]):
        raise RuntimeError(
            "Password reset email is not configured. Set SMTP_HOST, "
            "SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD and SMTP_FROM."
        )

    reset_url = (
        os.getenv("APP_BASE_URL", "http://127.0.0.1:8000")
        + "/reset-password?token=" + token
    )

    message = EmailMessage()
    message["Subject"] = "Reset your Cognify-AI password"
    message["From"] = sender
    message["To"] = email
    message.set_content(
        "A password reset was requested for your Cognify-AI account.\n\n"
        f"Open this link to reset your password:\n{reset_url}\n\n"
        "This link expires in 15 minutes. If you did not request it, "
        "you can ignore this email."
    )

    with smtplib.SMTP(host, port, timeout=15) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(message)


@router.post("/signup", status_code=201)
def signup(data: SignupData):
    email = str(data.email).strip().lower()
    salt, password_hash = hash_password(data.password)

    try:
        with get_db() as db:
            cursor = db.execute(
                """INSERT INTO users
                   (email, password_hash, salt, created_at)
                   VALUES (?, ?, ?, ?)""",
                (email, password_hash, salt, int(time.time())),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409, detail="An account with this email already exists."
        )

    return {"message": "Account created. Please log in.", "user_id": user_id}


@router.post("/login")
def login(data: LoginData, request: Request):
    email = str(data.email).strip().lower()

    if _throttled(email):
        raise HTTPException(
            status_code=429, detail="Too many failed attempts. Please wait a few minutes."
        )

    with get_db() as db:
        user = db.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()

    if not user or not password_matches(
        data.password, user["salt"], user["password_hash"]
    ):
        _FAILED.setdefault(email, []).append(time.time())
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    _FAILED.pop(email, None)

    request.session.clear()
    request.session["user_id"] = user["id"]
    request.session["email"] = user["email"]

    return {"message": "Logged in successfully.", "email": user["email"]}


@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    return {"message": "Logged out successfully."}


@router.get("/me")
def me(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Please log in.")

    return {"email": request.session.get("email")}


@router.post("/forgot-password")
def forgot_password(data: ForgotData):
    email = str(data.email).strip().lower()

    # Return the same message whether or not the account exists.
    generic_message = (
        "If an account exists for this email, password-reset instructions "
        "will be sent."
    )

    with get_db() as db:
        user = db.execute(
            "SELECT id FROM users WHERE email = ?", (email,)
        ).fetchone()

        if not user:
            return {"message": generic_message}

        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        expires_at = int(time.time()) + 900

        db.execute(
            """INSERT OR REPLACE INTO password_resets
               (email, token_hash, expires_at) VALUES (?, ?, ?)""",
            (email, token_hash, expires_at),
        )

    try:
        send_reset_email(email, token)
    except Exception:
        # Do not expose account existence or email-server details.
        raise HTTPException(
            status_code=503,
            detail="Password-reset email is temporarily unavailable. "
                   "Please try again later.",
        )

    return {"message": generic_message}


@router.post("/reset-password")
def reset_password(data: ResetData):
    token_hash = hashlib.sha256(data.token.encode()).hexdigest()
    now = int(time.time())

    with get_db() as db:
        record = db.execute(
            """SELECT email FROM password_resets
               WHERE token_hash = ? AND expires_at > ?""",
            (token_hash, now),
        ).fetchone()

        if not record:
            raise HTTPException(
                status_code=400, detail="Invalid or expired reset link."
            )

        salt, password_hash = hash_password(data.new_password)

        db.execute(
            "UPDATE users SET password_hash = ?, salt = ? WHERE email = ?",
            (password_hash, salt, record["email"]),
        )
        db.execute(
            "DELETE FROM password_resets WHERE email = ?",
            (record["email"],),
        )

    return {"message": "Password updated. You can now log in."}
