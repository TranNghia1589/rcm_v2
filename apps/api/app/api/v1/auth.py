from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.app.api.deps import get_current_user
from apps.api.app.core.security import create_access_token, hash_password, verify_password
from apps.api.app.schemas.auth import AuthTokenResponse, AuthUser, LoginRequest, RegisterRequest
from src.utils.infrastructure.db.postgres_client import PostgresClient, PostgresConfig

router = APIRouter(prefix="/auth", tags=["auth"])

BASE_DIR = Path(__file__).resolve().parents[5]
POSTGRES_CFG = BASE_DIR / "config" / "db" / "postgres.yaml"


@router.post("/register", response_model=AuthTokenResponse)
def register(payload: RegisterRequest) -> AuthTokenResponse:
    pg_cfg = PostgresConfig.from_yaml(POSTGRES_CFG)
    password_hash = hash_password(payload.password)

    with PostgresClient(pg_cfg) as client:
        exists = client.fetch_one("SELECT user_id FROM users WHERE email = %s LIMIT 1", (payload.email,))
        if exists:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

        row = client.fetch_one(
            """
            INSERT INTO users (email, full_name, password_hash)
            VALUES (%s, %s, %s)
            RETURNING user_id, email, COALESCE(full_name, '')
            """,
            (payload.email, payload.full_name, password_hash),
        )

    user = AuthUser(user_id=int(row[0]), email=str(row[1]), full_name=str(row[2]))
    token = create_access_token(user_id=user.user_id, email=user.email)
    return AuthTokenResponse(access_token=token, user=user)


@router.post("/login", response_model=AuthTokenResponse)
def login(payload: LoginRequest) -> AuthTokenResponse:
    pg_cfg = PostgresConfig.from_yaml(POSTGRES_CFG)
    with PostgresClient(pg_cfg) as client:
        row = client.fetch_one(
            """
            SELECT user_id, email, COALESCE(full_name, ''), COALESCE(password_hash, '')
            FROM users
            WHERE email = %s
            LIMIT 1
            """,
            (payload.email,),
        )

    if not row:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    user_id = int(row[0])
    email = str(row[1])
    full_name = str(row[2])
    password_hash = str(row[3] or "")

    if not password_hash or not verify_password(payload.password, password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    user = AuthUser(user_id=user_id, email=email, full_name=full_name)
    token = create_access_token(user_id=user.user_id, email=user.email)
    return AuthTokenResponse(access_token=token, user=user)


@router.get("/me", response_model=AuthUser)
def me(current_user: dict = Depends(get_current_user)) -> AuthUser:
    return AuthUser(
        user_id=int(current_user["user_id"]),
        email=str(current_user["email"]),
        full_name=str(current_user["full_name"]),
    )
