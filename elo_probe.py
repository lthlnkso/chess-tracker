"""Does rating conditioning actually change how the model plays?

A zero-initialised embedding can train to nothing and the loss curve will look
fine, because move accuracy is dominated by the trunk. The only honest check is
behavioural: hold the position fixed, vary the requested rating, and see whether
the chosen move changes.

Reports, over many real positions:
  disagree   how often the top move at 1000 differs from the top move at 2200
  KL         divergence between the two move distributions
  agree_acc  top-move agreement with the actual continuation, per rating band,
             which is the thing conditioning is supposed to improve when the
             requested rating matches the player who really moved

    python elo_probe.py --ckpt ckpt/final/ctx5_pre_elo.pt --shard data/mt/2026-01
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import chess

from bitboards import decode_move, board_to_planes8, N_PLANES13
from timefeat import time_features, N_TIME_FEATS, N_TIME_BINS
from model import MultiTaskModel, Config, N_ELO_BINS, elo_to_bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--plies-per-game", type=int, default=6)
    ap.add_argument("--bands", default="1000,1400,1800,2200")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    wr = ck["n_planes"] == N_PLANES13
    model = MultiTaskModel(cfg, n_planes=ck["n_planes"], n_extra=ck["n_extra"],
                           d_embed=ck["d_embed"], n_time_bins=N_TIME_BINS,
                           n_elo_bins=N_ELO_BINS,
                           n_game_slots=ck.get("n_game_slots", 1),
                           elo_cond=bool(ck.get("elo_cond"))).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    if not ck.get("elo_cond"):
        raise SystemExit("this checkpoint has no rating conditioning")
    print(f"ckpt step {ck.get('step')} | elo_cond=True", flush=True)

    bands = [int(x) for x in args.bands.split(",")]
    meta = np.load(f"{args.shard}/meta.npy", mmap_mode="r")
    moves = np.memmap(f"{args.shard}/moves.u16", dtype=np.uint16, mode="r")
    clocks = np.memmap(f"{args.shard}/clocks.u16", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(args.seed)

    npl, mlpg = ck["n_planes"], ck.get("max_len_per_game", cfg.max_len)
    n_pos = 0
    disagree = 0
    kl_sum = 0.0
    hits = {b: 0 for b in bands}
    true_band_hits, true_band_n = 0, 0

    for gi in rng.choice(len(meta), args.games * 4, replace=False):
        if n_pos >= args.games * args.plies_per_game:
            break
        row = meta[gi]
        o, n = int(row["offset"]), int(row["nply"])
        if n < 20 or int(clocks[o]) == 0xFFFF:
            continue
        codes = np.asarray(moves[o:o + n])
        clk = np.asarray(clocks[o:o + n])
        fe, _, _ = time_features(clk, int(row["tc_base"]), int(row["tc_inc"]))

        board = chess.Board()
        states = []
        for c in codes:
            states.append(board.copy())
            board.push(decode_move(int(c)))

        picks = rng.choice(np.arange(4, min(n, mlpg) - 1),
                           size=min(args.plies_per_game, max(1, min(n, mlpg) - 5)),
                           replace=False)
        for t in picks:
            b = states[t]
            legal = list(b.legal_moves)
            if len(legal) < 2:
                continue
            pov = b.turn
            seat = 0 if pov == chess.WHITE else 1
            true_elo = int(row["white_elo"] if seat == 0 else row["black_elo"])
            tail = states[max(0, t - mlpg + 1):t + 1]
            T = len(tail)
            planes = np.zeros((1, T, npl, 8, 8), np.uint8)
            for i, bb in enumerate(tail):
                board_to_planes8(bb, pov, planes[0, i], wr)
            cands = np.zeros((1, 1, len(legal), npl, 8, 8), np.uint8)
            for j, mv in enumerate(legal):
                b.push(mv); board_to_planes8(b, pov, cands[0, 0, j], wr); b.pop()
            extra = np.zeros((1, T, N_TIME_FEATS), np.float32)
            extra[0] = fe[max(0, t - mlpg + 1):t + 1][:T]
            my_turn = np.zeros((1, T), bool); my_turn[0, -1] = True
            ply_idx = np.array([[T - 1]], dtype=np.int64)

            P = [torch.from_numpy(planes).to(device), torch.from_numpy(extra).to(device),
                 torch.from_numpy(cands).to(device), torch.from_numpy(ply_idx).to(device),
                 torch.zeros((1, T), dtype=torch.bool, device=device),
                 torch.from_numpy(my_turn).to(device)]
            probs = {}
            with torch.no_grad():
                for band in bands:
                    eb = elo_to_bin(torch.tensor([band])).to(device)
                    ml, *_ = model(*P, None, None, elo_bin=eb)
                    probs[band] = torch.softmax(ml[0, 0, :len(legal)].float(), -1).cpu().numpy()
            lo, hi = probs[bands[0]], probs[bands[-1]]
            if int(lo.argmax()) != int(hi.argmax()):
                disagree += 1
            kl_sum += float((lo * (np.log(lo + 1e-9) - np.log(hi + 1e-9))).sum())
            played = int(codes[t])
            j_true = next((j for j, mv in enumerate(legal)
                           if decode_move(played) == mv), None)
            if j_true is not None:
                for band in bands:
                    hits[band] += int(probs[band].argmax() == j_true)
                if true_elo > 0:
                    eb = elo_to_bin(torch.tensor([true_elo])).to(device)
                    with torch.no_grad():
                        ml, *_ = model(*P, None, None, elo_bin=eb)
                    p = torch.softmax(ml[0, 0, :len(legal)].float(), -1).cpu().numpy()
                    true_band_hits += int(p.argmax() == j_true)
                    true_band_n += 1
            n_pos += 1

    print(f"\n{n_pos:,} positions")
    print(f"  top move differs {bands[0]} vs {bands[-1]}: {disagree/max(n_pos,1)*100:.1f}%")
    print(f"  mean KL({bands[0]} || {bands[-1]}): {kl_sum/max(n_pos,1):.4f}")
    print("\n  agreement with the move actually played:")
    for b in bands:
        print(f"    requested {b}: {hits[b]/max(n_pos,1)*100:.1f}%")
    if true_band_n:
        print(f"    requested = the mover's REAL rating: "
              f"{true_band_hits/true_band_n*100:.1f}%   <- should be highest")
    print("\nELO_PROBE_DONE")


if __name__ == "__main__":
    main()
