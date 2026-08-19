"""A durable job queue for identify requests, backed by SQLite.

Why SQLite and not Redis: measured on the deploy box this does ~13,000
enqueues and ~17,000 atomic claims per second, against a worker that completes
1.5 identifies per second. The queue is four orders of magnitude faster than the
work it feeds, so the only things that matter are that it survives a restart and
that two workers never claim the same job. SQLite does both with no daemon and
no extra resident memory -- on a 1 GB box shared with other people's services,
not adding a process is the feature.

The claim is a single UPDATE ... RETURNING, which is atomic in SQLite, so
workers can be separate processes or separate machines on a shared volume
without any locking of our own.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    state    TEXT NOT NULL,           -- queued | running | done | failed
    kind     TEXT NOT NULL DEFAULT 'identify',   -- identify | move
    payload  TEXT NOT NULL,
    result   TEXT,
    created  REAL NOT NULL,
    started  REAL,
    finished REAL,
    vid      TEXT
);
CREATE INDEX IF NOT EXISTS ix_state ON jobs(state, id);
CREATE INDEX IF NOT EXISTS ix_created ON jobs(created);
"""


class JobQueue:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            # CREATE TABLE IF NOT EXISTS silently does nothing to a table that
            # already exists, so a schema change ships fine and then fails at
            # runtime on the one machine that has old data -- which is exactly
            # what happened in production on 2026-08-18. Migrate explicitly.
            cols = {r[1] for r in c.execute("PRAGMA table_info(jobs)")}
            if "kind" not in cols:
                c.execute("ALTER TABLE jobs ADD COLUMN "
                          "kind TEXT NOT NULL DEFAULT 'identify'")

    @contextlib.contextmanager
    def _conn(self):
        """Open, use, CLOSE.

        sqlite3.Connection is its own context manager, but `with conn:` only
        commits or rolls back the transaction -- it never closes the handle.
        With a connection opened per call and none of them closed, the server
        leaked three descriptors (db, -wal, -shm) per queue operation and sat
        at 1023 of its 1024 soft NOFILE limit while completely idle. Past that
        every open fails with "unable to open database file", which is not a
        load error and does not look like one: the box reads as 6% busy while
        refusing everything. Measured on 2026-08-19 at 150 concurrent clients.

        check_same_thread=False: the server hands connections to worker threads.
        """
        c = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        try:
            c.execute("PRAGMA journal_mode=WAL")   # readers never block the writer
            c.execute("PRAGMA busy_timeout=30000")
            with c:                                # commit on success, roll back on raise
                yield c
        finally:
            c.close()

    def submit(self, payload, vid=None, kind="identify"):
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO jobs(state,kind,payload,created,vid) "
                "VALUES('queued',?,?,?,?)",
                (kind, json.dumps(payload), time.time(), vid))
            return cur.lastrowid

    def claim(self, kinds=None):
        """Atomically take the oldest queued job, or None.

        `kinds` lets a worker take only what it is built for -- a GPU box can
        claim identify work while a cheap box handles moves, without either
        needing to know the other exists.
        """
        where = "state='queued'"
        args = []
        if kinds:
            where += " AND kind IN (%s)" % ",".join("?" * len(kinds))
            args += list(kinds)
        with self._conn() as c:
            row = c.execute(
                f"UPDATE jobs SET state='running', started=? "
                f"WHERE id=(SELECT id FROM jobs WHERE {where} ORDER BY id LIMIT 1) "
                f"RETURNING id, kind, payload", [time.time()] + args).fetchone()
            return (row[0], row[1], json.loads(row[2])) if row else None

    def claim_batch(self, n, kinds=None):
        """Claim up to n jobs in ONE statement.

        The single-job claim costs a network round trip per job, which on a GPU
        worker is the whole cost: measured 95 ms of compute against two HTTP
        hops, leaving the GPU at 0% utilisation. Claiming 16 at a time amortises
        those hops 16x. UPDATE ... RETURNING is atomic over the whole set, so
        two workers still cannot take the same job.
        """
        where = "state='queued'"
        args = []
        if kinds:
            where += " AND kind IN (%s)" % ",".join("?" * len(kinds))
            args += list(kinds)
        with self._conn() as c:
            rows = c.execute(
                f"UPDATE jobs SET state='running', started=? "
                f"WHERE id IN (SELECT id FROM jobs WHERE {where} ORDER BY id LIMIT ?) "
                f"RETURNING id, kind, payload",
                [time.time()] + args + [int(n)]).fetchall()
        return [(r[0], r[1], json.loads(r[2])) for r in rows]

    def finish_many(self, items):
        """Report a batch of results in one transaction."""
        now = time.time()
        with self._conn() as c:
            c.executemany(
                "UPDATE jobs SET state=?, result=?, finished=? WHERE id=?",
                [("failed" if it.get("failed") else "done",
                  json.dumps(it.get("result")), now, int(it["job"])) for it in items])

    def finish(self, job_id, result, failed=False):
        with self._conn() as c:
            c.execute("UPDATE jobs SET state=?, result=?, finished=? WHERE id=?",
                      ("failed" if failed else "done", json.dumps(result),
                       time.time(), job_id))

    def get(self, job_id):
        with self._conn() as c:
            row = c.execute("SELECT state,result,created,finished FROM jobs "
                            "WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        state, result, created, finished = row
        out = {"state": state, "queued_at": created}
        if state in ("done", "failed"):
            out["result"] = json.loads(result) if result else None
            out["took"] = round((finished or created) - created, 3)
        else:
            out["waiting"] = round(time.time() - created, 1)
        return out

    def depth(self):
        with self._conn() as c:
            q = c.execute("SELECT COUNT(*) FROM jobs WHERE state='queued'").fetchone()[0]
            r = c.execute("SELECT COUNT(*) FROM jobs WHERE state='running'").fetchone()[0]
        return q, r

    def position(self, job_id):
        """How many jobs are ahead of this one -- the number a visitor can be told."""
        with self._conn() as c:
            row = c.execute("SELECT COUNT(*) FROM jobs WHERE state='queued' AND id<?",
                            (job_id,)).fetchone()
        return row[0] if row else 0

    def reap(self, older_than=86400, stuck_after=600):
        """Delete old finished jobs; requeue anything a dead worker abandoned.

        Without the requeue a worker crash strands a visitor on a spinner
        forever, which looks exactly like the model being slow.
        """
        now = time.time()
        with self._conn() as c:
            c.execute("DELETE FROM jobs WHERE state IN ('done','failed') AND finished < ?",
                      (now - older_than,))
            c.execute("UPDATE jobs SET state='queued', started=NULL "
                      "WHERE state='running' AND started < ?", (now - stuck_after,))
