"""Password hashing and JWT-style access/refresh tokens for Fashion Brain."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from knowledge_graph.access import CLEARANCE_RANK, Principal
from knowledge_graph.errors import Unauthorized, ValidationFailed

from fashion_brain.store import FashionBrainStore

PBKDF2_ROUNDS = 120_000


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${PBKDF2_ROUNDS}${base64.urlsafe_b64encode(salt).decode()}${base64.urlsafe_b64encode(digest).decode()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, rounds_s, salt_b64, digest_b64 = encoded.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        rounds = int(rounds_s)
        salt = base64.urlsafe_b64decode(salt_b64.encode())
        expected = base64.urlsafe_b64decode(digest_b64.encode())
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_json(payload: dict[str, Any]) -> str:
    return _b64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def _sign(secret: str, message: str) -> str:
    return _b64url(hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).digest())


def encode_token(secret: str, payload: dict[str, Any]) -> str:
    header = _b64url_json({"alg": "HS256", "typ": "JWT"})
    body = _b64url_json(payload)
    signing = f"{header}.{body}"
    return f"{signing}.{_sign(secret, signing)}"


def decode_token(secret: str, token: str) -> dict[str, Any]:
    try:
        header_b64, body_b64, sig = token.split(".")
    except ValueError as exc:
        raise Unauthorized("Invalid access token.") from exc
    signing = f"{header_b64}.{body_b64}"
    expected = _sign(secret, signing)
    if not hmac.compare_digest(expected, sig):
        raise Unauthorized("Invalid access token.")
    pad = "=" * (-len(body_b64) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(body_b64 + pad).decode("utf-8"))
    except Exception as exc:
        raise Unauthorized("Invalid access token.") from exc
    if int(payload.get("exp") or 0) < int(time.time()):
        raise Unauthorized("Access token has expired.")
    return payload


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthService:
    def __init__(self, store: FashionBrainStore, settings):
        self.store = store
        self.settings = settings

    @property
    def secret(self) -> str:
        return self.settings.resolve_jwt_secret()

    def register(
        self,
        email: str,
        password: str,
        *,
        role: str = "user",
        clearance: str = "internal",
        bootstrap_secret: str | None = None,
    ) -> dict[str, Any]:
        email = (email or "").strip().lower()
        if not email or "@" not in email:
            raise ValidationFailed("A valid email is required.")
        if not password or len(password) < 8:
            raise ValidationFailed("Password must be at least 8 characters.")
        if role not in {"user", "admin"}:
            raise ValidationFailed("role must be user or admin.")
        if clearance not in CLEARANCE_RANK:
            raise ValidationFailed("Clearance must be public, internal, or confidential.")
        if role == "admin":
            allowed = False
            if self.store.count_users() == 0:
                allowed = True
            elif bootstrap_secret and self.settings.admin_bootstrap_secret:
                allowed = hmac.compare_digest(bootstrap_secret, self.settings.admin_bootstrap_secret)
            elif self.settings.env in {"development", "test"} and not self.settings.admin_bootstrap_secret:
                allowed = True
            if not allowed:
                raise Unauthorized("Admin registration requires the bootstrap secret.")
        if self.store.get_user_by_email(email):
            raise ValidationFailed("An account with that email already exists.")
        user = self.store.create_user(email, hash_password(password), role, clearance)
        return self.store.public_user(user)

    def login(self, email: str, password: str) -> dict[str, Any]:
        user = self.store.get_user_by_email(email or "")
        if not user or not verify_password(password or "", user["password_hash"]):
            raise Unauthorized("Email or password is incorrect.")
        return self.issue_tokens(user)

    def issue_tokens(self, user: dict[str, Any]) -> dict[str, Any]:
        now = int(time.time())
        access_payload = {
            "sub": user["id"],
            "email": user["email"],
            "role": user["role"],
            "clearance": user["clearance"],
            "typ": "access",
            "iat": now,
            "exp": now + int(self.settings.jwt_access_minutes) * 60,
        }
        access_token = encode_token(self.secret, access_payload)
        refresh_raw = secrets.token_urlsafe(32)
        expires = datetime.now(timezone.utc) + timedelta(days=int(self.settings.jwt_refresh_days))
        expires_at = expires.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        self.store.save_refresh_token(user["id"], hash_refresh_token(refresh_raw), expires_at)
        return {
            "access_token": access_token,
            "refresh_token": refresh_raw,
            "token_type": "bearer",
            "expires_in": int(self.settings.jwt_access_minutes) * 60,
            "user": self.store.public_user(user),
        }

    def refresh(self, refresh_token: str) -> dict[str, Any]:
        record = self.store.get_refresh_by_hash(hash_refresh_token(refresh_token or ""))
        if not record:
            raise Unauthorized("Refresh token is invalid.")
        if record["expires_at"] < datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"):
            self.store.revoke_refresh(record["token_hash"])
            raise Unauthorized("Refresh token has expired.")
        user = self.store.get_user(record["user_id"])
        if not user:
            raise Unauthorized("Refresh token is invalid.")
        self.store.revoke_refresh(record["token_hash"])
        return self.issue_tokens(user)

    def logout(self, refresh_token: str | None = None, *, user_id: str | None = None) -> dict[str, str]:
        if refresh_token:
            self.store.revoke_refresh(hash_refresh_token(refresh_token))
        elif user_id:
            self.store.revoke_user_refresh_tokens(user_id)
        return {"status": "ok"}

    def principal_from_authorization(self, authorization: str | None, clearance_header: str | None) -> Principal | None:
        if not authorization or not authorization.lower().startswith("bearer "):
            return None
        token = authorization.split(" ", 1)[1].strip()
        if not token:
            return None
        # Shared API token still accepted for automation / Postman demos.
        if self.settings.api_token and token == self.settings.api_token:
            level = clearance_header or "public"
            if level not in CLEARANCE_RANK:
                raise ValidationFailed("Clearance must be public, internal, or confidential.")
            return Principal(subject="api", clearance=level, roles=["admin", "user"])
        try:
            payload = decode_token(self.secret, token)
        except Unauthorized:
            return None
        if payload.get("typ") != "access":
            raise Unauthorized("Access token is required.")
        user = self.store.get_user(str(payload.get("sub") or ""))
        if not user:
            raise Unauthorized("User was not found.")
        level = clearance_header or user["clearance"] or "public"
        if level not in CLEARANCE_RANK:
            raise ValidationFailed("Clearance must be public, internal, or confidential.")
        if CLEARANCE_RANK[level] > CLEARANCE_RANK.get(user["clearance"], 0) and user["role"] != "admin":
            # Users cannot raise clearance above their profile; admins may for KG work.
            level = user["clearance"]
        roles = ["user"]
        if user["role"] == "admin":
            roles.append("admin")
        return Principal(subject=user["id"], clearance=level, roles=roles)


def require_admin(principal: Principal) -> None:
    if "admin" not in (principal.roles or []):
        raise Unauthorized("Admin role is required.")
