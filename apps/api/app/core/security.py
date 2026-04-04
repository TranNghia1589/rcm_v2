from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("utf-8"))


def _jwt_secret() -> str:
    return os.getenv("APP_JWT_SECRET", "dev-secret-change-me")


def hash_password(password: str) -> str:
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return f"pbkdf2_sha256$120000${salt}${dk.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, iterations, salt, digest = encoded.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), int(iterations))
        return hmac.compare_digest(dk.hex(), digest)
    except Exception:
        return False


def create_access_token(*, user_id: int, email: str, expires_in_sec: int = 60 * 60 * 24 * 7) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "email": email,
        "iat": now,
        "exp": now + int(expires_in_sec),
        "typ": "access",
    }
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    sig = hmac.new(_jwt_secret().encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{_b64url_encode(sig)}"


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
    except ValueError as exc:
        raise ValueError("Invalid token format") from exc

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    expected_sig = hmac.new(_jwt_secret().encode("utf-8"), signing_input, hashlib.sha256).digest()
    got_sig = _b64url_decode(sig_b64)
    if not hmac.compare_digest(expected_sig, got_sig):
        raise ValueError("Invalid token signature")

    payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    exp = int(payload.get("exp", 0))
    if exp <= int(time.time()):
        raise ValueError("Token expired")
    return payload
