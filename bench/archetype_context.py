"""Does giving the bot Elo-matched games as CONTEXT make it predict better moves?

think() passes `slot 0` for everything and leaves the model's other nine game
slots empty -- the bot sees one game where the identifier packs ten. This fills
them with games from an archetypal player at a target rating, so the model is
conditioned by example rather than by an Elo bin.

Three arms on identical positions:

    none      live game only                       (what ships today)
    matched   + k games from ANOTHER player in the same rating band
    mismatch  + k games from a player in a far band

The mismatch arm is the one that makes the result mean anything. Without it a
gain in `matched` could just be "more context helps", which would say nothing
about rating conditioning.

Scored as top-1 agreement with the move the rated human actually played.
"""
import argparse, io, json, os, random, sys, time
from collections import defaultdict
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) or "."
sys.path.insert(0, "/Users/inteoryx/investigations/chess_tracker")
sys.path.insert(0, "/Users/inteoryx/investigations/chess_tracker/play")

import chess, torch                                              # noqa: E402
import fastboard as fb                                           # noqa: E402
from bitboards import decode_move                                # noqa: E402
from timefeat import time_features, N_TIME_FEATS, N_TIME_BINS  # noqa: E402
from model import Config, MultiTaskModel, N_ELO_BINS  # noqa: E402


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["cfg"]; c = c if not isinstance(c, dict) else Config(**c)
    m = MultiTaskModel(c, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                       d_embed=ck["d_embed"], n_time_bins=N_TIME_BINS,
                       n_elo_bins=N_ELO_BINS,
                       n_game_slots=ck.get("n_game_slots", 1),
                       elo_cond=bool(ck.get("elo_cond"))).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck


class Shard:
    def __init__(self, path):
        self.meta = np.load(f"{path}/meta.npy", mmap_mode="r")
        self.mv = np.memmap(f"{path}/moves.u16", dtype=np.uint16, mode="r")
        self.ck = np.memmap(f"{path}/clocks.u16", dtype=np.uint16, mode="r")

    def game(self, gi, max_plies):
        row = self.meta[gi]
        o, n = int(row["offset"]), int(row["nply"])
        b, hist = chess.Board(), []
        for c in self.mv[o:o + min(n, max_plies)]:
            try:
                m = decode_move(int(c))
            except Exception:                                      # noqa: BLE001
                break
            if m not in b.legal_moves:
                break
            hist.append(m); b.push(m)
        clk = np.asarray(self.ck[o:o + n], dtype=np.float64)
        fe, _, _ = time_features(clk, int(row["tc_base"]), int(row["tc_inc"]))
        return hist, fe, row


def encode_block(hist, fe, pov, seat, npl, wr, n_extra, upto=None):
    """(planes, extra, my_turn) for one game, from `pov`'s seat."""
    T = len(hist) if upto is None else min(upto, len(hist))
    b, snaps, rights = chess.Board(), [], []
    for t in range(T):
        snaps.append(fb.snapshot(b, pov))
        if wr:
            rights.append(fb.rights(b, pov))
        b.push(hist[t])
    if not snaps:
        return None
    pl = fb.encode_batch(np.asarray(snaps, dtype=np.uint64), n_planes=npl,
                         rights_arr=np.asarray(rights, np.int16) if wr else None)
    ex = fe[:T] if len(fe) >= T else np.vstack(
        [fe, np.zeros((T - len(fe), N_TIME_FEATS), np.float32)])
    # timefeat now emits 3 features; a trunk trained with 2 must be fed 2, or
    # in_proj fails on a shape it cannot explain. Slice rather than assume.
    ex = ex[:, :n_extra]
    mt = np.zeros(T, bool); mt[seat::2] = True
    return pl, ex.astype(np.float32), mt


@torch.no_grad()
def predict(model, device, blocks, live_board, npl, wr, n_slots):
    """Top-1 legal move at the end of the LAST block, given all blocks as context."""
    legal = list(live_board.legal_moves)
    if not legal:
        return None
    pov = live_board.turn
    T = sum(b[0].shape[0] for b in blocks)
    planes = np.concatenate([b[0] for b in blocks], 0)[None]
    extra = np.concatenate([b[1] for b in blocks], 0)[None]
    mine = np.concatenate([b[2] for b in blocks], 0)[None]
    slot = np.concatenate([np.full(b[0].shape[0], min(i, n_slots - 1), np.int64)
                           for i, b in enumerate(blocks)])[None]
    ppos = np.concatenate([np.arange(b[0].shape[0]) for b in blocks])[None]

    cs = np.empty((len(legal), fb.N_BB), np.uint64)
    cr = np.empty((len(legal), 5), np.int16)
    for j, mv in enumerate(legal):
        _, rr = fb.successor(live_board, mv, pov, cs[j]); cr[j] = rr
    cands = fb.encode_batch(cs, n_planes=npl,
                            rights_arr=cr if wr else None)[None, None]

    t = lambda a, d=None: torch.as_tensor(a, dtype=d, device=device)
    out = model(t(planes), t(extra, torch.float32), t(cands),
                t([[T - 1]], torch.long), torch.zeros((1, T), dtype=torch.bool, device=device),
                t(mine), t(slot, torch.long), t(ppos, torch.long))
    return legal[int(out[0][0, 0, :len(legal)].argmax())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/final/ctx10_pre.pt")
    ap.add_argument("--shard", default="data/2026-06-big")
    ap.add_argument("--bands", default="1200:1400,2000:2200")
    ap.add_argument("--far", type=int, default=1000, help="rating gap for the mismatch arm")
    ap.add_argument("--games", type=int, default=20, help="target games per band")
    ap.add_argument("--ctx", type=int, default=9, help="context games")
    ap.add_argument("--ctx-plies", type=int, default=50)
    ap.add_argument("--first-ply", type=int, default=8)
    ap.add_argument("--last-ply", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="archetype.json")
    a = ap.parse_args()

    dev = torch.device(a.device)
    model, ck = load_model(a.ckpt, dev)
    npl, wr = ck["n_planes"], ck["n_planes"] == 13
    n_extra = int(ck["n_extra"])
    n_slots = ck.get("n_game_slots", 1)
    print(f"{os.path.basename(a.ckpt)}: {n_slots} slots, {n_extra} clock feats, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params, dev={dev}",
          flush=True)

    sh = Shard(a.shard)
    meta = sh.meta
    G = len(meta)
    pid = np.concatenate([np.asarray(meta["white_pid"]), np.asarray(meta["black_pid"])])
    gi = np.concatenate([np.arange(G)] * 2)
    seat = np.concatenate([np.zeros(G, np.int8), np.ones(G, np.int8)])
    elo = np.concatenate([np.asarray(meta["white_elo"]), np.asarray(meta["black_elo"])]).astype(int)
    npl_g = np.concatenate([np.asarray(meta["nply"])] * 2)
    # Targets need to be long enough to predict over; donors only need to be
    # long enough to serve as context. One filter for both cut the donor pool
    # about fivefold and made the archetype pool look scarce when it is not.
    keep_t = (elo > 0) & (npl_g >= a.last_ply + 4)
    keep_d = (elo > 0) & (npl_g >= 24)
    keep = keep_t | keep_d
    is_t = keep_t[keep]
    pid, gi, seat, elo = pid[keep], gi[keep], seat[keep], elo[keep]

    by_player = defaultdict(list)
    for i in range(len(pid)):
        by_player[int(pid[i])].append(i)
    prate = {p: float(np.median(elo[idx])) for p, idx in by_player.items()}
    long_by_player = defaultdict(list)
    for p, idx in by_player.items():
        long_by_player[p] = [i for i in idx if is_t[i]]
    def pool(lo, hi, need, want_long=False):
        src = long_by_player if want_long else by_player
        return [p for p, r in prate.items() if lo <= r < hi and len(src[p]) >= need]

    rng = random.Random(17)
    rows = []
    for spec in a.bands.split(","):
        lo, hi = (int(x) for x in spec.split(":"))
        targets = pool(lo, hi, 1, want_long=True)
        donors = pool(lo, hi, a.ctx)
        far_lo = max(600, lo - a.far) if lo - a.far >= 600 else lo + a.far
        far = pool(far_lo, far_lo + 200, a.ctx)
        if not targets or not donors or not far:
            print(f"  band {lo}-{hi}: not enough players "
                  f"(t{len(targets)} d{len(donors)} f{len(far)})"); continue
        rng.shuffle(targets); rng.shuffle(donors); rng.shuffle(far)
        print(f"  band {lo}-{hi}: {len(targets):,} targets, {len(donors):,} donors, "
              f"{len(far):,} far donors (~{far_lo})", flush=True)

        def ctx_blocks(donor_pool, exclude):
            d = next(p for p in donor_pool if p != exclude)
            idx = by_player[d][:a.ctx]
            bl = []
            for i in idx:
                hist, fe, _ = sh.game(int(gi[i]), a.ctx_plies)
                pv = chess.WHITE if seat[i] == 0 else chess.BLACK
                blk = encode_block(hist, fe, pv, int(seat[i]), npl, wr, n_extra)
                if blk: bl.append(blk)
            return bl

        hit = {"none": 0, "matched": 0, "mismatch": 0}; n = 0
        changed = {"matched": 0, "mismatch": 0}
        t0 = time.perf_counter()
        for tgt in targets[: a.games]:
            i = long_by_player[tgt][0]
            hist, fe, _ = sh.game(int(gi[i]), a.last_ply + 2)
            pv = chess.WHITE if seat[i] == 0 else chess.BLACK
            s = int(seat[i])
            m_ctx = ctx_blocks(donors, tgt)
            f_ctx = ctx_blocks(far, tgt)
            board = chess.Board()
            for t in range(min(len(hist) - 1, a.last_ply)):
                if t >= a.first_ply and (t % 2) == s:
                    live = encode_block(hist, fe, pv, s, npl, wr, n_extra, upto=t + 1)
                    if live:
                        truth = hist[t]
                        picks = {}
                        for arm, extra_ctx in (("none", []), ("matched", m_ctx),
                                               ("mismatch", f_ctx)):
                            p = predict(model, dev, extra_ctx + [live], board,
                                        npl, wr, n_slots)
                            picks[arm] = p
                            hit[arm] += (p == truth)
                        for arm in ("matched", "mismatch"):
                            changed[arm] += (picks[arm] != picks["none"])
                        n += 1
                board.push(hist[t])
        el = time.perf_counter() - t0
        row = {"band": f"{lo}-{hi}", "n": n,
               **{k: hit[k] / max(n, 1) for k in hit},
               **{f"changed_{k}": changed[k] / max(n, 1) for k in changed}}
        rows.append(row)
        print(f"    {n} positions in {el:.0f}s | "
              + "  ".join(f"{k} {row[k]*100:.1f}%" for k in ("none","matched","mismatch"))
              + f" | move changed by matched ctx {row['changed_matched']*100:.1f}%"
              + f", by mismatch {row['changed_mismatch']*100:.1f}%",
              flush=True)
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
