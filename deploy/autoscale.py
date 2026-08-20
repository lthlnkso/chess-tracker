#!/usr/bin/env python3
"""Scale the fleet on measured queue wait. Runs on main.

The signal is how long a move JOB SAT before a worker picked it up -- `started`
minus `created`, from the queue itself. Not CPU, not connection count, not
moves/s: those all say the box is busy, and busy is fine. Queue wait is the only
one that says a visitor is waiting, which is the thing worth spending money on.

Up is eager and down is patient, deliberately. Being one node short is visible
to every visitor on the site; being one node over costs $0.066/hr. So a single
bad tick scales up, and scaling down needs the queue to stay quiet for several
consecutive ticks in a row.

    systemctl enable --now chess-autoscale

The stats page's "accelerate wind down" button touches ACCEL_FLAG, which divides
every timer by ACCEL_FACTOR for ACCEL_WINDOW seconds. That is for watching a
fan-in happen inside a test rather than waiting out the real cooldown; it only
shortens the clocks, so a fleet that is genuinely busy still will not shrink.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleet as F                                            # noqa: E402
import vultr as V                                            # noqa: E402

QUEUE = "/opt/chess-id/jobs.db"
STATE = "/opt/chess-id/autoscale.json"
ACCEL_FLAG = "/opt/chess-id/accelerate"

TICK = 15.0                 # seconds between decisions
WINDOW = 45                 # seconds of job history each decision looks at
UP_MS = 250.0               # p90 queue wait above this: a visitor is waiting
DOWN_MS = 60.0              # below this: the fleet has room to spare
DOWN_TICKS = 8              # consecutive quiet ticks before removing a node
UP_COOLDOWN = 180.0         # a new node needs ~2 min to boot and warm up
DOWN_COOLDOWN = 120.0
MAX_NODES = 8               # a ceiling in code, not just in judgement
ACCEL_FACTOR = 20.0
ACCEL_WINDOW = 900.0        # the button stays in effect this long


def accel() -> bool:
    try:
        return (time.time() - os.path.getmtime(ACCEL_FLAG)) < ACCEL_WINDOW
    except OSError:
        return False


def queue_wait_p90(window=WINDOW):
    """p90 of (started - created) over move jobs started in the window.

    Returns (p90_ms, n). n matters: a fleet with no traffic has no wait, and
    "no wait" must not read as "quiet, scale down" when it actually means
    "nobody is playing" -- though in this case those want the same answer.
    """
    try:
        c = sqlite3.connect(QUEUE, timeout=5)
        c.execute("PRAGMA busy_timeout=3000")
        rows = [r[0] for r in c.execute(
            "SELECT (started - created) * 1000.0 FROM jobs "
            "WHERE kind='move' AND started IS NOT NULL AND started > ?",
            (time.time() - window,)) if r[0] is not None]
        c.close()
    except sqlite3.Error:
        return None, 0
    if not rows:
        return 0.0, 0
    rows.sort()
    return rows[min(len(rows) - 1, int(0.9 * len(rows)))], len(rows)


def load_state():
    try:
        return json.load(open(STATE))
    except (OSError, ValueError):
        return {"quiet": 0, "last_action": 0.0, "last": "", "history": []}


def save_state(s):
    s["history"] = s.get("history", [])[-40:]
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f)
    os.replace(tmp, STATE)          # never leave a half-written state file


def main():
    print("autoscale up. up>%.0fms, down<%.0fms x%d ticks, max %d nodes"
          % (UP_MS, DOWN_MS, DOWN_TICKS, MAX_NODES), file=sys.stderr, flush=True)
    while True:
        s = load_state()
        fast = accel()
        div = ACCEL_FACTOR if fast else 1.0
        p90, n = queue_wait_p90()
        # The Vultr API can be unreachable -- its keys carry an IP allowlist and
        # a new box is not on it until someone adds it. That must degrade to
        # "watch but do not scale", not take the process down: the queue-wait
        # figure this publishes is useful on its own, and a crash loop would
        # leave the stats page saying "not running" with no reason why.
        api_ok, nodes = True, []
        try:
            _, nodes = F.instances()
        except SystemExit as e:
            api_ok = False
            s["api_error"] = str(e)[:140]
        except Exception as e:                                # noqa: BLE001
            api_ok = False
            s["api_error"] = str(e)[:140]
        now = time.time()
        s["api_ok"] = api_ok
        s["p90_ms"] = p90
        s["samples"] = n
        s["nodes"] = len(nodes)
        s["accelerated"] = fast
        s["checked_at"] = now
        if s.get("last", "").startswith(("fanout", "fanin")) and \
                now - s.get("last_action", 0) > 90:
            s["last"] = ""            # let the live status take over again

        if p90 is None or not api_ok:
            if not api_ok:
                s["last"] = "observing only — " + s.get("api_error", "vultr api unavailable")
            save_state(s)
            time.sleep(TICK / div)
            continue

        cool_up = UP_COOLDOWN / div
        cool_down = DOWN_COOLDOWN / div
        need_quiet = max(1, int(DOWN_TICKS / div))

        if p90 > UP_MS and len(nodes) < MAX_NODES:
            s["quiet"] = 0
            if now - s["last_action"] > cool_up:
                print(f"UP: p90 {p90:.0f} ms over {n} moves, "
                      f"{len(nodes)} -> {len(nodes)+1}", file=sys.stderr, flush=True)
                try:
                    F.fanout(1)
                    s["last_action"], s["last"] = now, f"fanout at p90 {p90:.0f}ms"
                except Exception as e:                       # noqa: BLE001
                    s["last"] = f"fanout failed: {str(e)[:120]}"
                    print(s["last"], file=sys.stderr, flush=True)
        elif p90 < DOWN_MS:
            s["quiet"] += 1
            if (nodes and s["quiet"] >= need_quiet
                    and now - s["last_action"] > cool_down):
                print(f"DOWN: p90 {p90:.0f} ms for {s['quiet']} ticks, "
                      f"{len(nodes)} -> {len(nodes)-1}", file=sys.stderr, flush=True)
                try:
                    F.fanin(1)
                    s["last_action"], s["last"] = now, f"fanin at p90 {p90:.0f}ms"
                    s["quiet"] = 0
                except Exception as e:                       # noqa: BLE001
                    s["last"] = f"fanin failed: {str(e)[:120]}"
                    print(s["last"], file=sys.stderr, flush=True)
        else:
            s["quiet"] = 0

        # Refresh the status line every tick. It used to be written only when
        # something happened, so a transient API failure left its error on the
        # stats page indefinitely -- the page said the autoscaler was blocked
        # for ten minutes after it had recovered.
        if api_ok and not s.get("last", "").startswith(("fanout", "fanin")):
            s["last"] = (f"watching — p90 {p90:.0f} ms over {n} moves, "
                         f"{len(nodes)} node(s)"
                         + (f", quiet {s['quiet']}/{need_quiet}" if s["quiet"] else ""))
        s.pop("api_error", None) if api_ok else None

        s["history"].append({"t": round(now), "p90": round(p90 or 0, 1),
                             "n": len(nodes)})
        save_state(s)
        time.sleep(TICK / div)


if __name__ == "__main__":
    main()
