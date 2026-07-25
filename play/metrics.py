"""Optional, fire-and-forget event recording for the browser demo.

Entirely inert unless both ``METRICS_API_URL`` and ``METRICS_API_KEY`` are set,
so cloning this repo and running ``python play/server.py`` records nothing and
talks to nobody. That is the intended default.

When it is configured, events are queued and shipped by a background thread.
Nothing on the request path does network I/O, so a slow or dead metrics service
can never make a move take longer.
"""

from __future__ import annotations

import json
import os
import queue
import secrets
import threading
import time
import urllib.error
import urllib.request

API_URL = os.environ.get("METRICS_API_URL", "").rstrip("/")
API_KEY = os.environ.get("METRICS_API_KEY", "")
ENABLED = bool(API_URL and API_KEY)

FLUSH_SECONDS = float(os.environ.get("METRICS_FLUSH_SECONDS", "2"))
QUEUE_MAX = int(os.environ.get("METRICS_QUEUE_MAX", "500"))

_queue: "queue.Queue[dict]" = queue.Queue(maxsize=QUEUE_MAX)
_started = False


def new_visitor_id() -> str:
    return secrets.token_urlsafe(16)


def record(name: str, props: dict) -> None:
    """Queue an event. Never blocks, never raises."""
    if not ENABLED:
        return
    try:
        _queue.put_nowait(
            {
                "name": name,
                "ts": time.time(),
                "props": {k: v for k, v in props.items() if v is not None},
            }
        )
    except queue.Full:
        pass


def _ship(batch: list[dict]) -> None:
    req = urllib.request.Request(
        f"{API_URL}/events/batch",
        data=json.dumps({"events": batch}).encode(),
        method="POST",
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5):
        pass


def _loop() -> None:
    while True:
        time.sleep(FLUSH_SECONDS)
        batch = []
        while len(batch) < 200:
            try:
                batch.append(_queue.get_nowait())
            except queue.Empty:
                break
        if not batch:
            continue
        try:
            _ship(batch)
        except (urllib.error.URLError, OSError, ValueError):
            # Dropped on purpose. Requeueing a failing batch during an outage
            # would starve fresh events out of a bounded queue.
            pass


def start() -> None:
    global _started
    if not ENABLED or _started:
        return
    threading.Thread(target=_loop, name="metrics", daemon=True).start()
    _started = True
