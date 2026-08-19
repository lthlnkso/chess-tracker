"""Every game ever played, stored server-side.

The demo was stateless: games lived in the visitor's browser and were posted for
scoring but never kept. That made ordinary questions unanswerable -- "when was
the sixth game played", "was that one an outlier", "did the same person play all
of them" -- and it threw away the only labelled data this project can generate.

Two things make a row valuable later:

  reconstitutable   history + times + human_white is exactly the shape the
                    replay tooling already reads, so any stored game can be
                    re-scored against a future model without conversion.
  attributable      the visitor id is recorded on every game, and claims carry
                    the same id. A visitor almost always plays FIRST and claims
                    AFTERWARDS, so the link that turns games into labelled
                    training data is made retroactively by that id -- which is
                    why it is stored even though claims_at_time is usually empty.

claims_at_time is a separate, deliberately denormalised field: it records what
the visitor had claimed AT THE MOMENT THEY PLAYED. Claims can be withdrawn
later, and for training we want the state as it was, not as it ended up.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    visitor       TEXT,
    created_at    REAL,        -- first move of the game
    completed_at  REAL,        -- when it ended
    recorded_at   REAL NOT NULL,
    n_plies       INTEGER,
    human_white   INTEGER,
    result        TEXT,        -- "1-0" | "0-1" | "1/2-1/2"
    reason        TEXT,        -- checkmate | time forfeit | stalemate | ...
    won           INTEGER,     -- 1 human won, 0 lost, NULL draw/unknown
    bot_elo       INTEGER,
    history       TEXT NOT NULL,
    times         TEXT NOT NULL,
    claims_at_time TEXT        -- JSON list of usernames claimed when played
);
CREATE INDEX IF NOT EXISTS ix_games_visitor ON games(visitor, created_at);
CREATE INDEX IF NOT EXISTS ix_games_created ON games(created_at);
"""


def _won(result, human_white):
    if result not in ("1-0", "0-1"):
        return None
    return 1 if (result == "1-0") == bool(human_white) else 0


class GameDB:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            # CREATE TABLE IF NOT EXISTS does nothing to a table that already
            # exists, so a column added later ships fine and then fails only on
            # the machine with old data. Migrate explicitly -- this bit us once.
            cols = {r[1] for r in c.execute("PRAGMA table_info(games)")}
            for name, decl in (("bot_elo", "INTEGER"), ("won", "INTEGER"),
                               ("claims_at_time", "TEXT"), ("reason", "TEXT")):
                if name not in cols:
                    c.execute(f"ALTER TABLE games ADD COLUMN {name} {decl}")

    @contextlib.contextmanager
    def _conn(self):
        """Open, use, CLOSE -- see the same note in jobq.py.

        `with conn:` commits the transaction; it does not close the handle, so
        a per-call connection that is never closed leaks three descriptors each
        time and eventually exhausts the process NOFILE limit.
        """
        c = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        try:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=30000")
            with c:
                yield c
        finally:
            c.close()

    def record(self, game, visitor=None, claims=None):
        """Store one finished game. Returns its id, or None if unusable."""
        hist = list(game.get("history") or [])
        if len(hist) < 2:
            return None
        times = list(game.get("times") or [])
        hw = bool(game.get("human_white", True))
        result = game.get("result") or None
        now = time.time()
        # created_at is when the game STARTED. The client may send it; if not,
        # derive it from the recorded think times so the timeline is still
        # roughly right rather than silently absent.
        created = game.get("created_at")
        if not created:
            spent = sum(float(t or 0) for t in times) / 1000.0
            created = now - spent
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO games(visitor,created_at,completed_at,recorded_at,"
                "n_plies,human_white,result,reason,won,bot_elo,history,times,"
                "claims_at_time) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (visitor, created, game.get("completed_at") or now, now,
                 len(hist), int(hw), result, game.get("reason") or None,
                 _won(result, hw), game.get("bot_elo"),
                 json.dumps(hist), json.dumps(times),
                 json.dumps(list(claims or []))))
            return cur.lastrowid

    def for_visitor(self, visitor, limit=100):
        with self._conn() as c:
            rows = c.execute(
                "SELECT id,created_at,completed_at,n_plies,human_white,result,"
                "reason,won,history,times,claims_at_time FROM games "
                "WHERE visitor=? ORDER BY created_at LIMIT ?",
                (visitor, limit)).fetchall()
        return [{"id": r[0], "created_at": r[1], "completed_at": r[2],
                 "n_plies": r[3], "human_white": bool(r[4]), "result": r[5],
                 "reason": r[6], "won": r[7],
                 "history": json.loads(r[8]), "times": json.loads(r[9]),
                 "claims_at_time": json.loads(r[10] or "[]")} for r in rows]

    def stats(self):
        with self._conn() as c:
            n = c.execute("SELECT COUNT(*) FROM games").fetchone()[0]
            v = c.execute("SELECT COUNT(DISTINCT visitor) FROM games").fetchone()[0]
            w = c.execute("SELECT SUM(won=1), SUM(won=0) FROM games").fetchone()
        return {"games": n, "visitors": v, "human_won": w[0] or 0,
                "human_lost": w[1] or 0}
