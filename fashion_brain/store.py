"""SQLite store for Fashion Brain auth, conversations, and memory."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_id() -> str:
    return str(uuid.uuid4())


class FashionBrainStore:
    """Thread-safe SQLite access for users, tokens, chats, and memory."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'admin')),
                    clearance TEXT NOT NULL DEFAULT 'internal',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS refresh_tokens (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    expires_at TEXT NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT 'New chat',
                    clearance TEXT NOT NULL DEFAULT 'internal',
                    archived INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    meta_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id, updated_at);
                CREATE INDEX IF NOT EXISTS idx_refresh_user ON refresh_tokens(user_id);
                """
            )
            self._ensure_column(conn, "conversations", "graph_id", "TEXT")

    def _ensure_column(self, conn, table: str, column: str, decl: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        names = set()
        for row in rows:
            try:
                names.add(row["name"])
            except Exception:
                names.add(row[1])
        if column not in names:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    # --- users ---

    def count_users(self) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()
            return int(row["n"])

    def create_user(self, email: str, password_hash: str, role: str, clearance: str = "internal") -> dict[str, Any]:
        now = utc_now()
        user_id = new_id()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO users (id, email, password_hash, role, clearance, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, email.strip().lower(), password_hash, role, clearance, now, now),
            )
        return self.get_user(user_id)  # type: ignore[return-value]

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return dict(row) if row else None

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
            return dict(row) if row else None

    def update_user(self, user_id: str, *, clearance: str | None = None) -> dict[str, Any] | None:
        user = self.get_user(user_id)
        if not user:
            return None
        now = utc_now()
        next_clearance = clearance if clearance is not None else user["clearance"]
        with self.connect() as conn:
            conn.execute(
                "UPDATE users SET clearance = ?, updated_at = ? WHERE id = ?",
                (next_clearance, now, user_id),
            )
        return self.get_user(user_id)

    def public_user(self, user: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": user["id"],
            "email": user["email"],
            "role": user["role"],
            "clearance": user["clearance"],
            "created_at": user["created_at"],
            "updated_at": user["updated_at"],
        }

    # --- refresh tokens ---

    def save_refresh_token(self, user_id: str, token_hash: str, expires_at: str) -> str:
        token_id = new_id()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO refresh_tokens (id, user_id, token_hash, expires_at, revoked, created_at) VALUES (?, ?, ?, ?, 0, ?)",
                (token_id, user_id, token_hash, expires_at, utc_now()),
            )
        return token_id

    def get_refresh_by_hash(self, token_hash: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM refresh_tokens WHERE token_hash = ? AND revoked = 0",
                (token_hash,),
            ).fetchone()
            return dict(row) if row else None

    def revoke_refresh(self, token_hash: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE refresh_tokens SET revoked = 1 WHERE token_hash = ?", (token_hash,))

    def revoke_user_refresh_tokens(self, user_id: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE refresh_tokens SET revoked = 1 WHERE user_id = ?", (user_id,))

    # --- conversations ---

    def create_conversation(
        self,
        user_id: str,
        title: str = "New chat",
        clearance: str = "internal",
        graph_id: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        conversation_id = new_id()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, user_id, title, clearance, archived, created_at, updated_at, graph_id) "
                "VALUES (?, ?, ?, ?, 0, ?, ?, ?)",
                (conversation_id, user_id, title or "New chat", clearance, now, now, graph_id or None),
            )
        return self.get_conversation(conversation_id, user_id)  # type: ignore[return-value]

    def list_conversations(self, user_id: str, *, archived: bool = False) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE user_id = ? AND archived = ? ORDER BY updated_at DESC",
                (user_id, 1 if archived else 0),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        with self.connect() as conn:
            if user_id:
                row = conn.execute(
                    "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
                    (conversation_id, user_id),
                ).fetchone()
            else:
                row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            return dict(row) if row else None

    def update_conversation(
        self,
        conversation_id: str,
        user_id: str,
        *,
        title: str | None = None,
        archived: bool | None = None,
        graph_id: str | None = None,
        clear_graph_id: bool = False,
    ) -> dict[str, Any] | None:
        current = self.get_conversation(conversation_id, user_id)
        if not current:
            return None
        next_title = title if title is not None else current["title"]
        next_archived = int(archived) if archived is not None else current["archived"]
        if clear_graph_id:
            next_graph = None
        elif graph_id is not None:
            next_graph = graph_id or None
        else:
            next_graph = current.get("graph_id")
        with self.connect() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, archived = ?, graph_id = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (next_title, next_archived, next_graph, utc_now(), conversation_id, user_id),
            )
        return self.get_conversation(conversation_id, user_id)

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
            return cur.rowcount > 0

    def touch_conversation(self, conversation_id: str, title: str | None = None) -> None:
        with self.connect() as conn:
            if title:
                conn.execute(
                    "UPDATE conversations SET updated_at = ?, title = ? WHERE id = ?",
                    (utc_now(), title, conversation_id),
                )
            else:
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (utc_now(), conversation_id),
                )

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        message_id = new_id()
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, created_at, meta_json) VALUES (?, ?, ?, ?, ?, ?)",
                (message_id, conversation_id, role, content, now, json.dumps(meta or {})),
            )
        return {
            "id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "created_at": now,
            "meta": meta or {},
        }

    def list_messages(self, conversation_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["meta"] = json.loads(item.pop("meta_json") or "{}")
            except json.JSONDecodeError:
                item["meta"] = {}
                item.pop("meta_json", None)
            else:
                item.pop("meta_json", None)
            out.append(item)
        return out

    def history_for_ask(self, conversation_id: str, *, limit: int = 16) -> list[dict[str, str]]:
        messages = self.list_messages(conversation_id, limit=limit)
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    # --- memories ---

    def list_memories(self, user_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def add_memory(self, user_id: str, content: str) -> dict[str, Any]:
        memory_id = new_id()
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO memories (id, user_id, content, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (memory_id, user_id, content.strip(), now, now),
            )
        return self.get_memory(memory_id, user_id)  # type: ignore[return-value]

    def get_memory(self, memory_id: str, user_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ? AND user_id = ?",
                (memory_id, user_id),
            ).fetchone()
            return dict(row) if row else None

    def update_memory(self, memory_id: str, user_id: str, content: str) -> dict[str, Any] | None:
        if not self.get_memory(memory_id, user_id):
            return None
        with self.connect() as conn:
            conn.execute(
                "UPDATE memories SET content = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (content.strip(), utc_now(), memory_id, user_id),
            )
        return self.get_memory(memory_id, user_id)

    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                "DELETE FROM memories WHERE id = ? AND user_id = ?",
                (memory_id, user_id),
            )
            return cur.rowcount > 0

    def memory_texts(self, user_id: str, *, limit: int = 20) -> list[str]:
        return [item["content"] for item in self.list_memories(user_id)[:limit]]
