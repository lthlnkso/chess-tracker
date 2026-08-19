"""A compute worker. Runs anywhere, dials into the queue, needs no inbound port.

The queue lives on a small always-up host; this is where the actual work
happens. That split is the point: the box holding the queue costs a few dollars
a month and never needs to scale, while capacity is rented only while it is
needed -- a GPU pod during a Reddit spike, nothing at 3am.

    WORKER_TOKEN=... python worker.py --api https://chess.lthlnkso.com
    WORKER_TOKEN=... python worker.py --api ... --kinds identify --device cuda

Workers are stateless and interchangeable. Killing one mid-job is safe: the
queue's reap() puts anything a dead worker abandoned back on the queue, so the
visitor waits a little longer rather than forever.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "play"))

UA = "chess-tracker-worker/1"


def call(api, path, token, payload, timeout=60):
    req = urllib.request.Request(
        api.rstrip("/") + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Worker-Token": token,
                 # Cloudflare's bot rules reject unknown agents; without a
                 # browser-ish UA every claim silently 403s.
                 "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True, help="queue host, e.g. https://chess.lthlnkso.com")
    ap.add_argument("--kinds", default="identify,move",
                    help="comma-separated job kinds this worker will take")
    ap.add_argument("--ckpt", default="ckpt/final/ctx5_pre.pt")
    ap.add_argument("--id-ckpt", default="ckpt/final/ctx10_ft.pt")
    ap.add_argument("--gallery", default="play/gallery_ctx10.npz")
    ap.add_argument("--batch", type=int, default=1,
                    help="jobs to claim per round trip; the point of a GPU worker")
    ap.add_argument("--idle-sleep", type=float, default=0.5)
    ap.add_argument("--max-idle-sleep", type=float, default=5.0)
    args = ap.parse_args()

    token = os.environ.get("WORKER_TOKEN", "")
    if not token:
        sys.exit("WORKER_TOKEN not set")

    import server                                        # noqa: E402
    server.load(args.ckpt, args.id_ckpt, args.gallery)
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    print(f"worker up: {args.api} kinds={kinds}", file=sys.stderr, flush=True)

    sleep = args.idle_sleep
    done = 0
    while True:
        try:
            job = call(args.api, "/api/worker/claim", token,
                       {"kinds": kinds, "max": args.batch})
        except urllib.error.HTTPError as e:
            print(f"claim failed: HTTP {e.code}", file=sys.stderr, flush=True)
            time.sleep(5.0)
            continue
        except Exception as e:                            # noqa: BLE001
            print(f"claim error: {e}", file=sys.stderr, flush=True)
            time.sleep(5.0)
            continue

        batch = job.get("jobs") if isinstance(job, dict) and "jobs" in job else (
            [job] if job else [])
        if not batch:
            # Back off when idle so an empty queue is not a busy-loop against
            # the host we are trying to keep cheap.
            time.sleep(sleep)
            sleep = min(sleep * 1.5, args.max_idle_sleep)
            continue
        sleep = args.idle_sleep

        results = []
        for jb in batch:
            jid, kind, payload = jb["job"], jb.get("kind", "identify"), jb["payload"]
            failed, result = False, None
            try:
                if kind == "identify":
                    result = server.identify(payload.get("games") or [],
                                             target=payload.get("target"))
                    result = result or {"top": [], "games_used": 0,
                                        "gallery": server.MODEL.get("gal_n", 0)}
                elif kind == "move":
                    # bot_reply, not think(): the client needs the full board
                    # state back, and building it here from the same function
                    # the synchronous path uses is what keeps the two routes
                    # from drifting into showing different boards.
                    result, _ = server.bot_reply(
                        list(payload.get("history") or []),
                        float(payload.get("temperature") or 0.0),
                        payload.get("times"), payload.get("elo"),
                        float(payload["bot_ms"]) if payload.get("bot_ms") is not None
                        else 1000.0)
                else:
                    failed, result = True, {"error": f"unknown kind {kind}"}
            except Exception as e:                        # noqa: BLE001
                failed, result = True, {"error": str(e)[:200]}
                print(f"job {jid} ({kind}) failed: {e}", file=sys.stderr, flush=True)
            results.append({"job": jid, "result": result, "failed": failed})

        for attempt in range(5):
            try:
                call(args.api, "/api/worker/result", token, {"results": results})
                break
            except Exception as e:                        # noqa: BLE001
                # Never drop finished results on a transient network blip -- the
                # visitors are waiting on them, and reap() would only requeue
                # work we have already paid for.
                print(f"result post retry {attempt+1}: {e}", file=sys.stderr, flush=True)
                time.sleep(2 ** attempt)

        done += len(results)
        if done % 25 == 0:
            print(f"worker: {done} jobs done", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
