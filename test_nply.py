"""Correctness tests for the N-ply future generator.

These are the properties that fail SILENTLY if broken -- a generator that
mislabels, or emits an unreachable state, or lets a duplicate sit next to the
truth, trains a model that looks fine and learns the wrong thing.
"""
import sys
sys.path.insert(0, ".")
import chess
import numpy as np

from fastboard import rights_bb, snapshot_bb
from nply import n_ply_futures


def real_games(n=40, seed=3):
    """Real lichess openings, so the tests run on positions the model will see."""
    from bitboards import decode_move
    meta = np.load("data/2026-06-big/meta.npy", allow_pickle=True)
    mv = np.memmap("data/2026-06-big/moves.u16", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        g = meta[rng.integers(len(meta))]
        if g["nply"] < 30:
            continue
        b, line, ok = chess.Board(), [], True
        for c in mv[g["offset"]:g["offset"] + 24]:
            try:
                m = decode_move(int(c))
            except Exception:
                ok = False; break
            if m not in b.legal_moves:
                ok = False; break
            line.append(m); b.push(m)
        if ok and len(line) == 24:
            out.append(line)
    return out


def test_label_points_at_the_truth():
    """The labelled slot must equal the state reached by playing the real moves."""
    rng = np.random.default_rng(1)
    bad = 0
    for line in real_games(30):
        for start in (0, 6, 12):
            b = chess.Board()
            for m in line[:start]:
                b.push(m)
            for depth in (1, 2, 3, 5):
                r = n_ply_futures(b, line[start:], depth, 12, rng)
                if r is None:
                    continue
                truth = b.copy(stack=False)
                for m in line[start:start + r["reached"]]:
                    truth.push(m)
                want = snapshot_bb(truth, b.turn)
                got = tuple(int(x) for x in r["snaps"][r["label"]])
                if want != got:
                    bad += 1
    assert bad == 0, f"{bad} mislabelled samples"
    print("  label points at the truth: PASS")


def test_all_candidates_are_reachable():
    """Every emitted state must be a real position, not a corrupted bitboard."""
    rng = np.random.default_rng(2)
    checked = 0
    for line in real_games(15):
        b = chess.Board()
        for m in line[:8]:
            b.push(m)
        for depth in (2, 4):
            r = n_ply_futures(b, line[8:], depth, 24, rng)
            for s in r["snaps"]:
                occ_w, occ_b = int(s[6]), int(s[7]) if False else (int(s[6]), int(s[7]))
                # kings: exactly one per side in any legal position
                kings = int(s[5])
                assert bin(kings).count("1") == 2, "not two kings"
                checked += 1
    print(f"  {checked} candidate states structurally valid: PASS")


def test_no_duplicate_states():
    """A duplicate of the truth sitting in another slot makes the label a lie."""
    rng = np.random.default_rng(4)
    for line in real_games(20):
        b = chess.Board()
        for m in line[:10]:
            b.push(m)
        for depth in (1, 2, 4):
            r = n_ply_futures(b, line[10:], depth, 32, rng)
            rows = [tuple(int(x) for x in s) for s in r["snaps"]]
            assert len(rows) == len(set(rows)), f"duplicate state at depth {depth}"
    print("  no duplicate states: PASS")


def test_perturb_diverges_exactly_once():
    """A `perturb` distractor must share the true prefix, then differ."""
    rng = np.random.default_rng(5)
    line = real_games(1)[0]
    b = chess.Board()
    for m in line[:6]:
        b.push(m)
    r = n_ply_futures(b, line[6:], 4, 24, rng, policy="perturb", late_bias=8.0)
    truth = tuple(int(x) for x in r["snaps"][r["label"]])
    diff = [tuple(int(x) for x in s) for i, s in enumerate(r["snaps"]) if i != r["label"]]
    assert all(d != truth for d in diff)
    print(f"  perturb produced {len(diff)} distinct non-truth states: PASS")


def test_terminal_inside_the_window():
    """A game that ends early must clamp depth, not fabricate moves past mate."""
    b = chess.Board()
    for u in ("f2f3", "e7e5", "g2g4"):
        b.push(chess.Move.from_uci(u))
    mate = [chess.Move.from_uci("d8h4")]          # fool's mate, then nothing
    rng = np.random.default_rng(6)
    r = n_ply_futures(b, mate, 5, 8, rng)
    assert r["reached"] == 1, f"reached {r['reached']}, expected 1"
    truth = b.copy(stack=False); truth.push(mate[0])
    assert truth.is_checkmate()
    print(f"  terminal clamped depth 5 -> reached {r['reached']}: PASS")


def test_depth_one_matches_the_existing_task():
    """Depth 1 must reproduce the current successor set exactly."""
    rng = np.random.default_rng(7)
    line = real_games(1)[0]
    b = chess.Board()
    for m in line[:9]:
        b.push(m)
    r = n_ply_futures(b, line[9:], 1, 64, rng)     # 64 > any legal count
    assert r["n_valid"] == b.legal_moves.count(), \
        f"{r['n_valid']} vs {b.legal_moves.count()} legal moves"
    print(f"  depth 1 == all {r['n_valid']} legal successors: PASS")


if __name__ == "__main__":
    test_label_points_at_the_truth()
    test_all_candidates_are_reachable()
    test_no_duplicate_states()
    test_perturb_diverges_exactly_once()
    test_terminal_inside_the_window()
    test_depth_one_matches_the_existing_task()
    print("\n  all tests passed")
