"""SQLite audit log for every PDP decision.

Durable record of what was decided, why, and when. Independent of the
best-effort SSE stream; a slow dashboard client cannot cause an audit
entry to be lost.
"""
from pathlib import Path

import aiosqlite

DB_PATH = Path(__file__).resolve().parent.parent / "ztf.db"


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL,
                created_at  REAL    NOT NULL DEFAULT 0,
                src         TEXT    NOT NULL,
                action      TEXT    NOT NULL,
                verdict     TEXT    NOT NULL,
                score       REAL,
                reason      TEXT,
                user        TEXT
            )
            """
        )
        # Migrate older schemas that don't yet have these columns.
        for column_ddl in (
            "ALTER TABLE decisions ADD COLUMN created_at REAL NOT NULL DEFAULT 0",
            "ALTER TABLE decisions ADD COLUMN user TEXT",
        ):
            try:
                await db.execute(column_ddl)
            except Exception:
                pass  # column already exists
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_decisions_id_desc ON decisions(id DESC)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_decisions_created_at ON decisions(created_at)"
        )
        await db.commit()


async def log_decision(event: dict) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO decisions (ts, created_at, src, action, verdict, score, reason, user) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event["ts"],
                float(event.get("created_at", 0.0)),
                event["src"],
                event["action"],
                event["verdict"],
                event.get("score"),
                event["reason"],
                event.get("user"),
            ),
        )
        await db.commit()


async def recent_decisions(limit: int = 50) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]
