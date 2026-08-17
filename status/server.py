"""Read-only status dashboard for the RunPod jobs.

Runs on YOUR machine, not the pod. It shells out to `ssh` and reads log files;
it never writes anything, never restarts anything, and cannot disturb a running
job. The pod was created with only port 22 exposed, so serving a page from the
pod itself would mean recreating it -- which would kill the training. Polling
from outside avoids that entirely.

    python status/server.py            # then open http://localhost:8420
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

STATE = {"ts": 0, "data": {}, "error": None}
LOCK = threading.Lock()
POLL_SECONDS = 30

SSH = ["ssh", "-o", "ConnectTimeout=20", "-o", "BatchMode=yes",
       "-o", "StrictHostKeyChecking=no"]


def sh(host: str, cmd: str, timeout: int = 45) -> str:
    r = subprocess.run(SSH + [host, cmd], capture_output=True, text=True,
                       timeout=timeout)
    return r.stdout


def api_key() -> str | None:
    p = os.path.join(ROOT, ".env")
    if not os.path.exists(p):
        return None
    for line in open(p):
        if line.startswith("RUNPOD_API_KEY="):
            return line.split("=", 1)[1].strip()
    return None


def runpod_state():
    key = api_key()
    if not key:
        return {}
    out = {}
    try:
        req = urllib.request.Request(
            "https://api.runpod.io/graphql", method="POST",
            data=json.dumps({"query": "query { myself { clientBalance currentSpendPerHr } }"}).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                     "User-Agent": "chess-tracker-status/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            m = json.loads(r.read().decode(), strict=False)["data"]["myself"]
        out["balance"] = m["clientBalance"]
        out["spend_hr"] = m["currentSpendPerHr"]
    except Exception as e:
        out["balance_err"] = str(e)[:120]
    try:
        req = urllib.request.Request(
            "https://rest.runpod.io/v1/pods",
            headers={"Authorization": f"Bearer {key}",
                     "User-Agent": "chess-tracker-status/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            pods = json.loads(r.read().decode(), strict=False)
        out["pods"] = [{"name": p.get("name"), "status": p.get("desiredStatus"),
                        "cost": p.get("costPerHr"), "vcpu": p.get("vcpuCount")}
                       for p in pods]
    except Exception as e:
        out["pods_err"] = str(e)[:120]
    return out


STEP_RE = re.compile(
    r"step\s+(\d+).*?move\s+([\d.]+)\s+\(acc\s+([\d.]+)\).*?"
    r"acc\s+([\d.]+),\s+mae\s+([\d.]+)s.*?elo_mae\s+(\d+).*?([\d.]+)\s+it/s")
VAL_RE = re.compile(
    r">> val @ (\d+): move_acc ([\d.]+) \| time_acc ([\d.]+) \(mae ([\d.]+)s\) \| "
    r"elo_mae (\d+)")
FT_RE = re.compile(r"step\s+(\d+)\s+\|\s+loss ([\d.]+).*?gap ([+\-\d.]+).*?([\d.]+) it/s")


def poll(host: str, log: str):
    d = {"host": host}
    raw = sh(host, f"tr '\\r' '\\n' < {log} 2>/dev/null | tail -4000")
    d["markers"] = [m for m in ("INGEST_DONE", "PRETRAIN_DONE", "PRETRAIN_FAILED",
                                "FINETUNE_FAILED", "ALL_DONE", "COLLAPSE", "Traceback")
                    if m in raw]
    steps = STEP_RE.findall(raw)
    if steps:
        s = steps[-1]
        d["pre"] = {"step": int(s[0]), "move_acc": float(s[2]),
                    "time_acc": float(s[3]), "time_mae": float(s[4]),
                    "elo_mae": int(s[5]), "it_s": float(s[6])}
    vals = VAL_RE.findall(raw)
    d["val"] = [{"step": int(v[0]), "move_acc": float(v[1]), "time_acc": float(v[2]),
                 "time_mae": float(v[3]), "elo_mae": int(v[4])} for v in vals[-12:]]
    fts = FT_RE.findall(raw)
    if fts and "PRETRAIN_DONE" in raw:
        f = fts[-1]
        d["ft"] = {"step": int(f[0]), "loss": float(f[1]), "gap": float(f[2]),
                   "it_s": float(f[3])}
    kept = re.findall(r"([\d,]+) games kept", raw)
    if kept:
        d["ingest_kept"] = kept[-1]
    d["worker_done"] = "WORKER_DONE" in raw
    d["months_done"] = re.findall(r"^DONE (\S+)", raw, re.M)
    d["phase"] = ("worker finished" if "WORKER_DONE" in raw else
                  "ingesting" if kept and "step" not in raw[-2000:] else
                  "finished" if "ALL_DONE" in raw else
                  "failed" if any(m in raw for m in
                                  ("PRETRAIN_FAILED", "FINETUNE_FAILED", "Traceback"))
                  else "fine-tuning" if "PRETRAIN_DONE" in raw
                  else "pre-training" if steps else "ingesting")

    gpu = sh(host, "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
                   "--format=csv,noheader 2>/dev/null; uptime | sed 's/.*load average: //'")
    lines = [l for l in gpu.strip().split("\n") if l.strip()]
    if lines:
        d["gpu"] = lines[0].strip()
    if len(lines) > 1:
        d["load"] = lines[-1].strip()

    d["data"] = sh(host, "for m in /workspace/data/mt/*/; do "
                         "n=$(basename $m); "
                         "if [ -f $m/manifest.json ]; then echo \"$n complete\"; "
                         "else echo \"$n partial\"; fi; done 2>/dev/null").strip()
    ev = sh(host, "cat /workspace/ckpt/mt_id/eval.json 2>/dev/null | head -c 4000")
    if ev.strip():
        try:
            d["eval"] = json.loads(ev)["result"]
        except Exception:
            pass
    return d


def poller(hosts, log):
    while True:
        try:
            data = {"jobs": {}, "runpod": runpod_state()}
            for h, hlog in hosts:
                try:
                    data["jobs"][h] = poll(h, hlog)
                except Exception as e:
                    data["jobs"][h] = {"host": h, "error": str(e)[:200]}
            with LOCK:
                STATE.update(ts=time.time(), data=data, error=None)
        except Exception as e:
            with LOCK:
                STATE["error"] = str(e)[:300]
        time.sleep(POLL_SECONDS)


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>chess_tracker status</title>
<meta http-equiv=refresh content=30>
<style>
:root{--bg:#fcfcfb;--ink:#0b0b0b;--mut:#52514e;--line:#e3e2df;--card:#fff;
      --ok:#1baf7a;--warn:#eda100;--bad:#e34948;--acc:#2a78d6}
@media(prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#fff;--mut:#c3c2b7;
      --line:#333331;--card:#232322;--acc:#3987e5}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.w{max-width:1000px;margin:0 auto;padding:26px 20px 60px}
h1{font-size:21px;margin:0 0 2px}
.sub{color:var(--mut);font-size:13px;margin-bottom:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:9px;
 padding:15px 17px;margin-bottom:14px}
.row{display:flex;gap:26px;flex-wrap:wrap}
.kv{font-variant-numeric:tabular-nums}
.kv b{display:block;font-size:22px;font-weight:650}
.kv span{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.4px}
table{width:100%;border-collapse:collapse;font-size:13.5px;
 font-variant-numeric:tabular-nums}
th{text-align:right;color:var(--mut);font-weight:600;padding:3px 0}
th:first-child{text-align:left}
td{text-align:right;padding:3px 0;border-top:1px solid var(--line)}
td:first-child{text-align:left}
.pill{display:inline-block;padding:2px 9px;border-radius:11px;font-size:12px;
 font-weight:650;color:#fff}
.mut{color:var(--mut);font-size:12.5px}
pre{margin:6px 0 0;white-space:pre-wrap;color:var(--mut);font-size:12px}
</style></head><body><div class=w>
<h1>chess_tracker</h1>
<div class=sub>__AGE__ &middot; auto-refresh 30s &middot; read-only, cannot disturb the run</div>
__BODY__
</div></body></html>"""


def render():
    with LOCK:
        ts, data, err = STATE["ts"], STATE["data"], STATE["error"]
    if not ts:
        return PAGE.replace("__BODY__", "<div class=card>starting up, no poll yet…</div>") \
                   .replace("__AGE__", "—")
    age = int(time.time() - ts)
    out = []
    rp = data.get("runpod", {})
    bal, spend = rp.get("balance"), rp.get("spend_hr")
    if bal is not None:
        hrs = (bal / spend) if spend else float("inf")
        out.append("<div class=card><div class=row>"
                   f"<div class=kv><b>${bal:,.2f}</b><span>balance</span></div>"
                   f"<div class=kv><b>${spend:,.2f}/hr</b><span>burn</span></div>"
                   f"<div class=kv><b>{hrs:,.0f}h</b><span>runway</span></div>"
                   "</div>")
        pods = rp.get("pods") or []
        if pods:
            out.append("<div class=mut style='margin-top:10px'>pods: " + ", ".join(
                f"{html.escape(str(p['name']))} ({p['status']}, ${p['cost']}/hr)"
                for p in pods) + "</div>")
        out.append("</div>")

    COL = {"finished": "var(--ok)", "worker finished": "var(--ok)", "failed": "var(--bad)",
           "pre-training": "var(--acc)", "fine-tuning": "var(--acc)",
           "ingesting": "var(--warn)"}
    for host, j in (data.get("jobs") or {}).items():
        out.append("<div class=card>")
        if j.get("error"):
            out.append(f"<b>{html.escape(host)}</b> "
                       f"<span class=pill style='background:var(--bad)'>unreachable</span>"
                       f"<pre>{html.escape(j['error'])}</pre></div>")
            continue
        ph = j.get("phase", "?")
        out.append(f"<b>{html.escape(host)}</b> &nbsp;"
                   f"<span class=pill style='background:{COL.get(ph,'var(--mut)')}'>"
                   f"{html.escape(ph)}</span>")
        if j.get("gpu"):
            out.append(f" <span class=mut>&nbsp;GPU {html.escape(j['gpu'])}"
                       + (f" &middot; load {html.escape(j.get('load',''))}" if j.get("load") else "")
                       + "</span>")
        p = j.get("pre")
        if p:
            out.append("<div class=row style='margin-top:12px'>"
                       f"<div class=kv><b>{p['step']:,}</b><span>step</span></div>"
                       f"<div class=kv><b>{p['it_s']:.1f}</b><span>it/s</span></div>"
                       f"<div class=kv><b>{p['move_acc']:.3f}</b><span>move acc</span></div>"
                       f"<div class=kv><b>{p['time_acc']:.3f}</b><span>time acc</span></div>"
                       f"<div class=kv><b>{p['elo_mae']}</b><span>elo mae</span></div>"
                       "</div>")
        f = j.get("ft")
        if f:
            out.append("<div class=row style='margin-top:12px'>"
                       f"<div class=kv><b>{f['step']:,}</b><span>ft step</span></div>"
                       f"<div class=kv><b>{f['loss']:.3f}</b><span>contrastive loss</span></div>"
                       f"<div class=kv><b>{f['gap']:+.3f}</b><span>pos-neg gap</span></div>"
                       f"<div class=kv><b>{f['it_s']:.1f}</b><span>it/s</span></div>"
                       "</div>")
        v = j.get("val") or []
        if v:
            out.append("<table><tr><th>val step</th><th>move acc</th><th>time acc</th>"
                       "<th>time mae</th><th>elo mae</th></tr>")
            for r in v[-8:]:
                out.append(f"<tr><td>{r['step']:,}</td><td>{r['move_acc']:.3f}</td>"
                           f"<td>{r['time_acc']:.3f}</td><td>{r['time_mae']:.2f}s</td>"
                           f"<td>{r['elo_mae']}</td></tr>")
            out.append("</table>")
        e = j.get("eval")
        if e:
            out.append("<div class=row style='margin-top:12px'>"
                       f"<div class=kv><b>{e.get('recall@1',0):.3f}</b><span>recall@1</span></div>"
                       f"<div class=kv><b>{e.get('recall@10',0):.3f}</b><span>recall@10</span></div>"
                       f"<div class=kv><b>{e.get('recall@100',0):.3f}</b><span>recall@100</span></div>"
                       f"<div class=kv><b>{e.get('gallery',0):,}</b><span>gallery</span></div>"
                       "</div>")
        if j.get("ingest_kept"):
            out.append(f"<div class=row style='margin-top:12px'>"
                       f"<div class=kv><b>{html.escape(j['ingest_kept'])}</b>"
                       f"<span>games kept (current month)</span></div>"
                       f"<div class=kv><b>{len(j.get('months_done') or [])}</b>"
                       f"<span>months finished</span></div></div>")
        if j.get("data"):
            out.append(f"<pre>{html.escape(j['data'])}</pre>")
        if j.get("markers"):
            out.append(f"<div class=mut style='margin-top:8px'>markers: "
                       f"{html.escape(', '.join(j['markers']))}</div>")
        out.append("</div>")
    if err:
        out.append(f"<div class=card><pre>{html.escape(err)}</pre></div>")
    return PAGE.replace("__BODY__", "\n".join(out)).replace("__AGE__", f"updated {age}s ago")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            b = render().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/json":
            with LOCK:
                b = json.dumps(STATE["data"], indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        else:
            self.send_response(404); self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    # host[:logpath] -- every pod mounts the SAME volume, so without a per-host
    # log path the ingest workers would read the trainer's log and report its
    # metrics as their own.
    ap.add_argument("--hosts", default="chesspod:/workspace/mt_run.log",
                    help="comma-separated ssh_alias[:logpath]")
    ap.add_argument("--log", default="/workspace/mt_run.log")
    ap.add_argument("--port", type=int, default=8420)
    args = ap.parse_args()
    hosts = []
    for spec in args.hosts.split(","):
        spec = spec.strip()
        if not spec:
            continue
        h, _, lg = spec.partition(":")
        hosts.append((h, lg or args.log))
    threading.Thread(target=poller, args=(hosts, args.log), daemon=True).start()
    print(f"status page on http://localhost:{args.port}  "
          f"(watching {[h for h,_ in hosts]})")
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
