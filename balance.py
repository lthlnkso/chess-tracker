"""Elo-balanced sampling.

Lichess 60+0 is not uniform in strength -- the 2026 sample is mean 1816, sd 398,
so a random batch is mostly 1500-2100 players and the model sees almost nothing
at the tails. Weighting each row by the inverse frequency of its Elo bin makes
every strength band contribute roughly equally.

Two things this deliberately does NOT do:

- It does not touch validation. Metrics stay on the natural distribution, or the
  reported Elo MAE would be measured against a population that does not exist.
- It does not fully flatten the extremes. Bins with almost no players would get
  enormous weights and the same handful of games would be redrawn constantly, so
  weights are clipped.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler

DEFAULT_LO, DEFAULT_HI, DEFAULT_BIN = 800.0, 2600.0, 100.0


def elo_bins(elos: np.ndarray, lo=DEFAULT_LO, hi=DEFAULT_HI, width=DEFAULT_BIN):
    """Clamped bin index per row; ends absorb everything beyond the range."""
    e = np.clip(np.asarray(elos, dtype=np.float64), lo - width, hi + width)
    return np.clip(((e - lo) // width).astype(np.int64) + 1,
                   0, int((hi - lo) // width) + 1)


def balanced_weights(elos: np.ndarray, max_ratio: float = 20.0,
                     min_count: int = 50) -> np.ndarray:
    """Per-row sampling weight ~ 1 / (rows in that Elo bin).

    `max_ratio` caps how much a rare bin can be up-weighted relative to the
    median bin. Without it a bin holding four 2800-rated games would be redrawn
    endlessly and the model would memorise those four games rather than learn
    what strong play looks like.
    """
    b = elo_bins(elos)
    counts = np.bincount(b, minlength=b.max() + 1).astype(np.float64)
    counts[counts < min_count] = np.inf          # too thin to represent: drop
    w = 1.0 / counts[b]
    if not np.isfinite(w).any() or w.max() == 0:
        return np.ones(len(elos))
    med = np.median(w[w > 0])
    w = np.clip(w, 0.0, med * max_ratio)
    s = w.sum()
    return (w / s * len(w)) if s > 0 else np.ones(len(elos))


def describe(elos: np.ndarray, weights: np.ndarray, n_show: int = 6) -> str:
    """Effective Elo distribution before vs after weighting, for the log."""
    b = elo_bins(elos)
    raw = np.bincount(b, minlength=b.max() + 1).astype(float)
    wt = np.bincount(b, weights=weights, minlength=b.max() + 1)
    raw, wt = raw / raw.sum(), wt / max(wt.sum(), 1e-9)
    keep = np.argsort(-raw)[:n_show]
    parts = [f"bin{int(i)} {100*raw[i]:.0f}%->{100*wt[i]:.0f}%" for i in sorted(keep)]
    eff = (weights.sum() ** 2) / max((weights ** 2).sum(), 1e-9)
    return "  ".join(parts) + f"   | effective N {eff/len(weights)*100:.0f}% of rows"


class EloBalancedSampler(Sampler):
    """Draw rows so Elo bands are equally represented, at any scale.

    Two-stage: pick an Elo bin, then a row uniformly inside it. This replaces
    `WeightedRandomSampler`, which routes through `torch.multinomial` and hard-
    fails above 2^24 categories -- one month of 60+0 is 48M game-sides, so the
    weighted approach cannot work here at all. Sampling a bin then a member is
    also exact rather than approximate, and O(1) per draw.

    Bins holding fewer than `min_count` rows are dropped: up-weighting a handful
    of games to band parity just teaches the model those games.
    """

    def __init__(self, elos, num_samples=None, min_count=200, seed=0):
        b = elo_bins(np.asarray(elos))
        order = np.argsort(b, kind="stable")
        sb = b[order]
        bounds = np.flatnonzero(np.r_[True, sb[1:] != sb[:-1], True])
        self.groups, self.bin_ids = [], []
        for i in range(len(bounds) - 1):
            g = order[bounds[i]:bounds[i + 1]]
            if len(g) >= min_count:
                self.groups.append(g)
                self.bin_ids.append(int(sb[bounds[i]]))
        if not self.groups:
            raise ValueError("no Elo bin has enough rows to balance")
        self.n = int(num_samples or len(elos))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __iter__(self):
        nb = len(self.groups)
        # draw in chunks so a long epoch does not build a giant Python list
        left = self.n
        while left > 0:
            take = min(left, 1 << 16)
            bins = self.rng.integers(0, nb, size=take)
            for bi in bins:
                g = self.groups[bi]
                yield int(g[self.rng.integers(0, len(g))])
            left -= take

    def describe(self):
        sizes = [len(g) for g in self.groups]
        return (f"{len(self.groups)} Elo bins kept (bins {min(self.bin_ids)}-"
                f"{max(self.bin_ids)}), rows/bin min {min(sizes):,} max {max(sizes):,}"
                f" -> each bin drawn {100/len(self.groups):.1f}% of the time")


# Gap buckets in rating points between the two players. Capped at 500 because
# beyond that the pairing stops being a normal game -- lichess matchmaking makes
# a 700-point gap rare enough that the ones that exist are mostly rematches,
# handicap play or a smurf, and training on them teaches the exception.
GAP_EDGES = np.array([50, 150, 250, 350, 450, 550], dtype=np.int64)


def gap_buckets(gaps):
    """|rating difference| -> bucket index. Last bucket is 'over 550, ignore'."""
    return np.digitize(np.abs(np.asarray(gaps)), GAP_EDGES)


class EloGapBalancedSampler(Sampler):
    """Equal representation across (Elo band x opponent-gap bucket).

    Same two-stage draw as EloBalancedSampler -- pick a cell, then a row inside
    it -- for the same reason: one month of 60+0 is 48M game-sides and
    WeightedRandomSampler routes through torch.multinomial, which hard-fails
    above 2^24 categories.

    Balancing the gap as well as the band is what puts mismatched games in front
    of the model at a useful rate. In the raw data they are rare: lichess pairs
    people close, so a 300+ gap is a small fraction of games, and a model trained
    on the natural distribution sees almost none of the situation the demo
    actually creates -- a visitor far stronger than the bot.

    Cells past `max_bucket` are dropped entirely rather than balanced.
    """

    def __init__(self, elos, gaps, num_samples=None, min_count=200, seed=0,
                 max_bucket=len(GAP_EDGES) - 1):
        eb = elo_bins(np.asarray(elos))
        gb = gap_buckets(gaps)
        keep = gb <= max_bucket
        # one integer per (band, bucket) cell
        cell = eb.astype(np.int64) * (max_bucket + 1) + gb
        cell = np.where(keep, cell, -1)
        order = np.argsort(cell, kind="stable")
        sc = cell[order]
        bounds = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1], True])
        self.groups, self.cell_ids = [], []
        for i in range(len(bounds) - 1):
            if sc[bounds[i]] < 0:
                continue                       # dropped: gap too wide
            g = order[bounds[i]:bounds[i + 1]]
            if len(g) >= min_count:
                self.groups.append(g)
                self.cell_ids.append((int(sc[bounds[i]]) // (max_bucket + 1),
                                      int(sc[bounds[i]]) % (max_bucket + 1)))
        if not self.groups:
            raise ValueError("no (Elo, gap) cell has enough rows to balance")
        self.n = int(num_samples or len(elos))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __iter__(self):
        nb = len(self.groups)
        left = self.n
        while left > 0:
            take = min(left, 1 << 16)
            for bi in self.rng.integers(0, nb, size=take):
                g = self.groups[bi]
                yield int(g[self.rng.integers(0, len(g))])
            left -= take

    def describe(self):
        sizes = [len(g) for g in self.groups]
        bands = sorted({c[0] for c in self.cell_ids})
        buckets = sorted({c[1] for c in self.cell_ids})
        return (f"{len(self.groups)} (Elo x gap) cells kept over {len(bands)} bands "
                f"and {len(buckets)} gap buckets, rows/cell min {min(sizes):,} "
                f"max {max(sizes):,} -> each cell drawn "
                f"{100/len(self.groups):.2f}% of the time")
