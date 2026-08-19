"""Exploratory capacity ramp: N concurrent players, N climbing until it breaks.

A virtual player is a *player*, not a request loop. It plays a game the way a
person does -- think, move, wait out the bot's pause, next ply -- then fires one
identify and starts another game. That pacing is the whole point: a hammer loop
answers "how fast can this box serve requests", and the question is "how many
people can play at once", which is a different number by a factor of ~50.

Human think times are drawn from play/think_times.json, the same table of real
lichess 1+0 games the bot's own pauses come from.

The headline metric is move throughput. The one that decides whether it FEELS
broken is `stalled`: the fraction of moves where the answer arrived later than
the bot's think time, because up to that point the wait was already happening
and the visitor cannot tell. Past it, they are watching a frozen board.
"""
import argparse, asyncio, json, os, random, statistics, sys, time
import chess, numpy as np, websockets

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/Users/inteoryx/investigations/chess_tracker"

# ---- human pacing, from the same table the bot uses -------------------------
with open(os.path.join(REPO, "play/think_times.json"), encoding="utf-8") as f:
    _RAW = json.load(f)["buckets"]
_THINK = {}
for b, hist in _RAW.items():
    v = np.array([float(k) for k in hist], dtype=np.float64)
    c = np.array([float(x) for x in hist.values()], dtype=np.float64)
    _THINK[int(b)] = (v, c / c.sum())
_BUCKETS = sorted(_THINK)

def human_think_ms(ply):
    b = min(_BUCKETS, key=lambda x: abs(x - ply))
    v, p = _THINK[b]
    return float(np.random.choice(v, p=p))

with open(os.path.join(REPO, "play/saved/session_12games.json")) as f:
    GAMES = json.load(f)

class Stats:
    def __init__(self):
        self.move_ms, self.id_ms = [], []
        self.moves = self.ids = self.errors = self.timeouts = 0
        self.excess_ms = []       # wait BEYOND the bot's own pause; what is felt
    def reset(self):
        self.__init__()

async def play(url, st, stop, vid):
    """One visitor. Games back to back until told to stop."""
    try:
        async with websockets.connect(url, open_timeout=25, ping_interval=20,
                                      user_agent_header="chess-explore/1") as ws:
            await ws.recv()                                   # hello
            ref = 0
            while not stop.is_set():
                board, hist = chess.Board(), []
                # Real 1+0 games run about 40 plies; end near a sampled target
                # rather than letting random play wander to 200.
                target = max(12, min(80, int(random.gauss(40, 12))))
                while not stop.is_set() and len(hist) < target and not board.is_game_over():
                    await asyncio.sleep(human_think_ms(len(hist)) / 1000.0)
                    mv = random.choice(list(board.legal_moves))
                    ref += 1
                    t0 = time.perf_counter()
                    await ws.send(json.dumps({"type": "submit", "kind": "move",
                                              "ref": str(ref), "vid": vid,
                                              "payload": {"history": hist,
                                                          "uci": mv.uci(),
                                                          "temperature": 0.0}}))
                    bot_ms, res = 0.0, None
                    try:
                        while res is None:
                            m = json.loads(await asyncio.wait_for(ws.recv(), timeout=40))
                            if m.get("type") == "queued":
                                bot_ms = float(m.get("bot_ms") or 0)
                            elif m.get("type") == "error":
                                st.errors += 1
                                res = "err"
                            else:
                                res = m
                    except asyncio.TimeoutError:
                        st.timeouts += 1
                        break
                    dt = (time.perf_counter() - t0) * 1000
                    if res == "err":
                        break
                    if res.get("failed"):
                        st.errors += 1
                        break
                    st.moves += 1
                    st.move_ms.append(dt)
                    # A binary "was it slower than the pause" is misleading: 58%
                    # of real opening plies carry a bot_ms of 0, so any latency
                    # at all trips it while 200 ms is imperceptible. Measure the
                    # excess over the pause the visitor was already expecting.
                    st.excess_ms.append(max(0.0, dt - bot_ms))
                    # The client paints the human's move on the ack and then
                    # waits out the bot's pause, so the remaining wait is what
                    # is LEFT of it -- not the pause on top of the round trip.
                    left = bot_ms - dt
                    if left > 0:
                        await asyncio.sleep(left / 1000.0)
                    r = res.get("result") or {}
                    hist = list(r.get("history") or hist)
                    board = chess.Board()
                    for u in hist:
                        board.push(chess.Move.from_uci(u))
                    if r.get("over"):
                        break

                if stop.is_set():
                    break
                # One identify per finished game, exactly like the real client.
                ref += 1
                t0 = time.perf_counter()
                await ws.send(json.dumps({"type": "submit", "kind": "identify",
                                          "ref": str(ref), "vid": vid,
                                          "payload": {"games": random.sample(GAMES, 3),
                                                      "target": None}}))
                try:
                    got = None
                    while got is None:
                        m = json.loads(await asyncio.wait_for(ws.recv(), timeout=90))
                        if m.get("type") != "queued":
                            got = m
                    if got.get("type") == "error" or got.get("failed"):
                        st.errors += 1
                    else:
                        st.ids += 1
                        st.id_ms.append((time.perf_counter() - t0) * 1000)
                except asyncio.TimeoutError:
                    st.timeouts += 1
    except Exception:                                          # noqa: BLE001
        st.errors += 1

def p(v, q):
    return statistics.quantiles(v, n=100)[q - 1] if len(v) > 2 else (v[0] if v else 0)

async def stage(url, n, secs, warm):
    st, stop = Stats(), asyncio.Event()
    tasks = [asyncio.create_task(play(url, st, stop, f"ex{n}-{i}")) for i in range(n)]
    await asyncio.sleep(warm)
    st.reset()                                   # discard the ramp-in
    t0 = time.perf_counter()
    await asyncio.sleep(secs)
    el = time.perf_counter() - t0
    stop.set()
    await asyncio.gather(*tasks, return_exceptions=True)
    return {"N": n, "moves_s": st.moves / el, "ids_s": st.ids / el,
            "move_p50": p(st.move_ms, 50), "move_p95": p(st.move_ms, 95),
            "id_p50": p(st.id_ms, 50), "id_p95": p(st.id_ms, 95),
            "moves": st.moves, "ids": st.ids,
            "errors": st.errors, "timeouts": st.timeouts,
            "excess_p50": p(st.excess_ms, 50), "excess_p95": p(st.excess_ms, 95)}

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="wss://chess.lthlnkso.com/ws/client")
    ap.add_argument("--steps", default="1,2,4,8,12,16,24,32,48,64,96,128")
    ap.add_argument("--seconds", type=float, default=45)
    ap.add_argument("--warm", type=float, default=10)
    ap.add_argument("--out", default="explore.json")
    args = ap.parse_args()

    rows, best = [], 0.0
    print(f"{'N':>5} {'moves/s':>9} {'id/s':>6} {'mv p50':>8} {'mv p95':>8} "
          f"{'id p95':>8} {'xs p95':>8} {'err':>5}", flush=True)
    for n in [int(x) for x in args.steps.split(",")]:
        r = await stage(args.url, n, args.seconds, args.warm)
        rows.append(r)
        print(f"{r['N']:>5} {r['moves_s']:>9.2f} {r['ids_s']:>6.2f} "
              f"{r['move_p50']:>7.0f}m {r['move_p95']:>7.0f}m {r['id_p95']:>7.0f}m "
              f"{r['excess_p95']:>7.0f}m {r['errors']+r['timeouts']:>5}", flush=True)
        json.dump(rows, open(args.out, "w"), indent=1)
        best = max(best, r["moves_s"])
        bad = (r["errors"] + r["timeouts"]) > max(2, 0.02 * (r["moves"] + r["ids"]))
        # Failure is not only errors: a box that answers every move eventually
        # but takes six seconds has already lost the visitor.
        if bad or r["move_p95"] > 6000 or (r["moves_s"] < 0.6 * best and n > 8):
            print(f"  -> stopping at N={n}: "
                  f"{'errors' if bad else 'p95 blown' if r['move_p95']>6000 else 'throughput collapsed'}",
                  flush=True)
            break
        await asyncio.sleep(4)                   # let the queue drain between steps
    json.dump(rows, open(args.out, "w"), indent=1)

asyncio.run(main())
