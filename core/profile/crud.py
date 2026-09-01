"""Profile 的 SQLite 读写。

为什么是 SQLite 而不是 JSON 文件（会被问「为什么不用个更专业的数据库」）：
- 单用户、本地优先、零运维、随 Python 内置——**约束推导出选型**。
- 需要事务性和并发安全：以后 Discord bot 和网页可能同时读写同一份数据，
  SQLite 的 WAL 模式能处理，裸 JSON 文件会互相覆盖。
- 上限清楚：真要多用户 SaaS 就换 Postgres，接口层（本文件）不变。
  面试时能说清「什么时候该换」比说「我用了最牛的数据库」更值钱。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from core.config import CONFIG
from core.profile.models import PartnerProfile

_SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    profile_id TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or CONFIG.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")  # 允许并发读写
    conn.executescript(_SCHEMA)
    return conn


def save_profile(profile: PartnerProfile, db_path: Path | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO profiles (profile_id, payload, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(profile_id) DO UPDATE SET payload=excluded.payload, "
            "updated_at=excluded.updated_at",
            (profile.profile_id, profile.model_dump_json(), str(profile.updated_at)),
        )


def load_profile(profile_id: str = "default", db_path: Path | None = None) -> PartnerProfile | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT payload FROM profiles WHERE profile_id=?", (profile_id,)
        ).fetchone()
    return PartnerProfile.model_validate(json.loads(row[0])) if row else None


def delete_profile(profile_id: str = "default", db_path: Path | None = None) -> None:
    """一键清空。数据主体有权随时删除自己的数据——这不是可选功能。"""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM profiles WHERE profile_id=?", (profile_id,))
