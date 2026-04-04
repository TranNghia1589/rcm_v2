from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from apps.api.app.core.security import decode_access_token
from src.utils.infrastructure.db.postgres_client import PostgresClient, PostgresConfig

BASE_DIR = Path(__file__).resolve().parents[4]
POSTGRES_CFG = BASE_DIR / "config" / "db" / "postgres.yaml"

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict[str, Any]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    token = credentials.credentials
    try:
        payload = decode_access_token(token)
        user_id = int(payload.get("sub"))
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    pg_cfg = PostgresConfig.from_yaml(POSTGRES_CFG)
    with PostgresClient(pg_cfg) as client:
        row = client.fetch_one(
            """
            SELECT user_id, email, COALESCE(full_name, '')
            FROM users
            WHERE user_id = %s
            LIMIT 1
            """,
            (user_id,),
        )

    if not row:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return {
        "user_id": int(row[0]),
        "email": str(row[1] or ""),
        "full_name": str(row[2] or ""),
    }
