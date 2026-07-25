"""Play the pre-trained model in a browser.

The model does not choose moves — it scores candidate *next positions*. So a move
is picked by generating every legal successor, encoding each one, and taking the
model's highest-scoring (or sampling by its probabilities).

Inference reuses the project's own `board_to_planes8` and `SuccessorScorer`
rather than reimplementing the encoding in JavaScript; a subtle mismatch there
would silently produce a weaker opponent and nothing would ever flag it.

    python play/server.py            # then open http://localhost:8000
"""

from __future__ import annotations

import argparse
import http.cookies
import json
import os
import sys
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import numpy as np
import torch
import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics                                     # noqa: E402
from model import SuccessorScorer, Config          # noqa: E402
from bitboards import board_to_planes8, N_PLANES13  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = {}


def load(ckpt: str):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    n_planes = ck["n_planes"]
    m = SuccessorScorer(cfg, n_planes=n_planes)
    m.load_state_dict(ck["model"])
    m.eval()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    MODEL.update(model=m, cfg=cfg, n_planes=n_planes,
                 with_rights=n_planes == N_PLANES13, step=ck.get("step"),
                 val=(ck.get("val") or {}).get("acc"))
    print(f"loaded {ckpt}: step {ck.get('step')}, {n_planes} planes, "
          f"val acc {MODEL['val']:.3f}", file=sys.stderr)


@torch.no_grad()
def think(history_moves: list[str], temperature: float):
    """Score every legal successor of the current position."""
    m, cfg = MODEL["model"], MODEL["cfg"]
    npl, wr = MODEL["n_planes"], MODEL["with_rights"]

    board = chess.Board()
    states = []
    for u in history_moves:
        states.append(board.copy())
        board.push(chess.Move.from_uci(u))
    states.append(board.copy())

    legal = list(board.legal_moves)
    if not legal:
        return None, []

    pov = board.turn                      # encode from the mover's seat
    # Absolute position embeddings only cover cfg.max_len plies; long games get
    # their tail used, which is an approximation the training never saw.
    T = min(len(states), cfg.max_len)
    tail = states[-T:]

    planes = np.zeros((1, T, npl, 8, 8), dtype=np.uint8)
    for i, b in enumerate(tail):
        board_to_planes8(b, pov, planes[0, i], wr)

    cands = np.zeros((1, 1, len(legal), npl, 8, 8), dtype=np.uint8)
    for j, mv in enumerate(legal):
        board.push(mv)
        board_to_planes8(board, pov, cands[0, 0, j], wr)
        board.pop()

    logits = m(torch.from_numpy(planes), torch.from_numpy(cands),
               torch.tensor([[T - 1]]))[0, 0, :len(legal)]
    probs = torch.softmax(logits if temperature <= 0 else logits / temperature, -1)

    order = torch.argsort(probs, descending=True)
    ranked = [{"uci": legal[i].uci(), "san": board.san(legal[i]),
               "p": float(probs[i])} for i in order.tolist()]
    if temperature <= 0:
        choice = ranked[0]["uci"]
    else:
        choice = legal[int(torch.multinomial(probs, 1))].uci()
    return choice, ranked


def state(history_moves: list[str], ranked=None, last=None) -> dict:
    board = chess.Board()
    for u in history_moves:
        board.push(chess.Move.from_uci(u))
    over = board.is_game_over()
    reason = ""
    if over:
        if board.is_checkmate():
            reason = "checkmate"
        elif board.is_stalemate():
            reason = "stalemate"
        elif board.is_insufficient_material():
            reason = "insufficient material"
        else:
            reason = "draw"
    return {
        "fen": board.fen(),
        "history": history_moves,
        "turn": "w" if board.turn else "b",
        "legal": [m.uci() for m in board.legal_moves],
        "check": board.is_check(),
        "over": over,
        "reason": reason,
        "result": board.result() if over else "*",
        "top": (ranked or [])[:6],
        "last": last,
        "ply": len(history_moves),
        "info": {"step": MODEL.get("step"), "val": MODEL.get("val")},
    }


VISITOR_COOKIE = "cp_vid"


def human_is_white(n_plies: int, human_moving: bool) -> bool:
    """Which colour the human has, from whose turn it is.

    Ply parity gives the side to move. If the human is the one moving now they
    own that side; if the model is moving, the human owns the other one — which
    is how "let the model open" games end up recorded as the human playing
    black rather than silently mislabelled.
    """
    white_to_move = n_plies % 2 == 0
    return white_to_move if human_moving else not white_to_move


def outcome_for_human(result: str, human_white: bool) -> str:
    """Map a PGN result to win/loss/draw *from the human's point of view*.

    Recording one side consistently matters: an inverted row is invisible in
    aggregate and would quietly turn a losing model into a winning one.
    """
    if result == "1-0":
        return "win" if human_white else "loss"
    if result == "0-1":
        return "loss" if human_white else "win"
    return "draw"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def visitor_id(self):
        """Read the visitor cookie, or mint one for this response."""
        raw = self.headers.get("Cookie", "")
        if raw:
            jar = http.cookies.SimpleCookie()
            try:
                jar.load(raw)
            except http.cookies.CookieError:
                jar = {}
            if VISITOR_COOKIE in jar:
                return jar[VISITOR_COOKIE].value, False
        return metrics.new_visitor_id(), True

    def _send(self, code, body, ctype="application/json", set_cookie=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if set_cookie:
            self.send_header(
                "Set-Cookie",
                f"{VISITOR_COOKIE}={set_cookie}; Max-Age=63072000; Path=/; "
                "HttpOnly; SameSite=Lax",
            )
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            vid, is_new = self.visitor_id()
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                return self._send(
                    200, f.read(), "text/html; charset=utf-8",
                    set_cookie=vid if is_new else None,
                )
        if self.path.startswith("/pieces/"):
            name = os.path.basename(self.path)
            if not name.endswith(".svg") or "/" in name.replace(os.sep, "/")[:-4]:
                return self._send(404, {"error": "not found"})
            fp = os.path.join(HERE, "pieces", name)
            if not os.path.isfile(fp):
                return self._send(404, {"error": "not found"})
            with open(fp, "rb") as f:
                return self._send(200, f.read(), "image/svg+xml")
        if self.path.startswith("/api/new"):
            return self._send(200, state([]))
        self._send(404, {"error": "not found"})

    def _send_state(self, payload, vid, human_white):
        """Send a game state, recording the outcome if it is a terminal one.

        Every path out of /api/move funnels through here, so a game can't end
        via a branch that forgot to record it.
        """
        if payload.get("over"):
            metrics.record("chess.game_ended", {
                "visitor_id": vid,
                "result": outcome_for_human(payload.get("result", "*"), human_white),
                "reason": payload.get("reason") or "unknown",
                "plies": payload.get("ply"),
                "human_colour": "white" if human_white else "black",
            })
        return self._send(200, payload)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        hist = list(req.get("history", []))
        temp = float(req.get("temperature", 0.0))

        if self.path == "/api/move":
            uci = req.get("uci")
            vid, _ = self.visitor_id()
            human_white = human_is_white(len(hist), human_moving=bool(uci))

            # An empty history means this request opens a new game. The server
            # is stateless, so ply count is the only signal there is.
            if not hist:
                metrics.record("chess.game_started", {
                    "visitor_id": vid,
                    "human_colour": "white" if human_white else "black",
                })

            if uci:
                board = chess.Board()
                for u in hist:
                    board.push(chess.Move.from_uci(u))
                try:
                    mv = chess.Move.from_uci(uci)
                except ValueError:
                    return self._send(400, {"error": f"bad move {uci}"})
                if mv not in board.legal_moves:      # try promotion default
                    mv = chess.Move(mv.from_square, mv.to_square, promotion=chess.QUEEN)
                    if mv not in board.legal_moves:
                        return self._send(400, {"error": f"illegal move {uci}"})
                hist.append(mv.uci())

            board = chess.Board()
            for u in hist:
                board.push(chess.Move.from_uci(u))
            if board.is_game_over():
                return self._send_state(state(hist), vid, human_white)

            choice, ranked = think(hist, temp)
            if choice is None:
                return self._send_state(state(hist), vid, human_white)
            hist.append(choice)
            return self._send_state(
                state(hist, ranked, last=choice), vid, human_white
            )

        if self.path == "/api/hint":
            _, ranked = think(hist, 0.0)
            return self._send(200, state(hist, ranked))

        self._send(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        os.path.dirname(HERE), "ckpt", "pretrain_best.pt"))
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    load(args.ckpt)
    metrics.start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"serving on http://localhost:{args.port}", file=sys.stderr)
    srv.serve_forever()


if __name__ == "__main__":
    main()
