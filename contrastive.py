"""Metric-learning objectives for the identification fine-tune.

The trunk and the 128-d embedding never change. What changes is the rule that
decides how hard to pull a player's own games together and push other players'
games apart. There are four families that are genuinely different in what they
optimise -- not four spellings of the same gradient:

    pair-softmax   supcon        every positive against every negative in the
                                 batch, one softmax, no mining, no margin
    mined-pair     ms            keep only pairs that violate, then weight each
                                 survivor by how badly (soft, per-pair)
    unified-pair   circle        per-pair *adaptive* margin -- a pair already
                                 satisfied gets a near-zero weight, so the
                                 decision boundary is a circle, not a line
    proxy          proxyanchor   one learned vector per player; every proxy is
                                 touched every step, so identity information
                                 lives in the proxy bank rather than only in
                                 whatever happened to share a batch

Two more are here because they are the obvious things to ask about, not because
they are expected to win:

    proxy-margin   arcface       angular-margin softmax over player classes. The
                                 face-recognition default, but it only updates
                                 the proxies that appear in the batch, and with
                                 ~250k players each proxy is seen a handful of
                                 times in a whole run. Included so that claim is
                                 measured rather than asserted.
    triplet        triplet       hardest positive vs hardest negative. Kept for
                                 the record -- it collapses here within 40 steps
                                 (see the docstring on `supcon_loss` in model.py).

Everything takes L2-normalised embeddings and returns `(loss, stats)`. The stats
keys are identical across losses so the collapse guard in finetune_mt.py does not
have to know which one is running.

Proxy-based losses own parameters, so they are nn.Modules and the caller must
put `loss_fn.parameters()` into the optimiser -- see `needs_proxies`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import supcon_loss, batch_hard_triplet

NEG_INF = float("-inf")


def _pair_stats(emb: torch.Tensor, labels: torch.Tensor) -> dict:
    """The collapse diagnostics, computed the same way for every loss.

    `gap` is the number to watch: it is pos_cos - neg_cos, and a run that has
    mapped every game to one point has gap 0 regardless of what its loss says.
    """
    with torch.no_grad():
        cos = emb @ emb.T
        eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
        same = (labels[:, None] == labels[None, :]) & ~eye
        neg = ~same & ~eye
        pos_cos = cos[same].mean().item() if same.any() else float("nan")
        neg_cos = cos[neg].mean().item() if neg.any() else float("nan")
        return {"pos_cos": pos_cos, "neg_cos": neg_cos,
                "gap": pos_cos - neg_cos, "emb_std": emb.std(0).mean().item()}


def _masked_lse(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Row-wise logsumexp over `keep`; rows with nothing kept return -inf.

    Used instead of log(1 + sum(exp(...))) because at the margins these losses
    use (beta = 50, gamma = 80) the naive form overflows float32 long before the
    gradient stops being informative.
    """
    return x.masked_fill(~keep, NEG_INF).logsumexp(dim=1)


def _softplus_lse(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """log(1 + sum_{keep} exp(x)), stable, and exactly 0 when nothing is kept."""
    lse = _masked_lse(x, keep)
    return torch.where(torch.isfinite(lse), F.softplus(lse), torch.zeros_like(lse))


def _masks(labels: torch.Tensor):
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    same = (labels[:, None] == labels[None, :]) & ~eye
    return same, (~same & ~eye)


# --- pair-based -------------------------------------------------------------

class SupCon(nn.Module):
    """Supervised contrastive loss. The incumbent, and the control arm."""

    name = "supcon"

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb, labels):
        return supcon_loss(emb, labels, temperature=self.temperature)


class MultiSimilarity(nn.Module):
    """Multi-Similarity loss (Wang et al., CVPR 2019).

    Two ideas SupCon does not have. First, *mining*: a negative is only worth a
    gradient if it is closer than the anchor's hardest positive (within eps), and
    symmetrically for positives. Second, *soft weighting*: surviving pairs are
    weighted by how far they sit past the base similarity, with a much sharper
    slope on negatives (beta=50) than positives (alpha=2), because a single
    intruding negative hurts retrieval far more than a slightly loose positive.
    """

    name = "ms"

    def __init__(self, alpha: float = 2.0, beta: float = 50.0,
                 base: float = 0.5, eps: float = 0.1):
        super().__init__()
        self.alpha, self.beta, self.base, self.eps = alpha, beta, base, eps

    def forward(self, emb, labels):
        sim = (emb @ emb.T).float()
        same, diff = _masks(labels)

        # Mining. An anchor with no positive (or no negative) yields +/-inf here;
        # the comparison then keeps nothing for that anchor, which is correct.
        min_pos = sim.masked_fill(~same, float("inf")).min(1).values
        max_neg = sim.masked_fill(~diff, NEG_INF).max(1).values
        pos_keep = same & (sim < (max_neg + self.eps)[:, None])
        neg_keep = diff & (sim > (min_pos - self.eps)[:, None])

        pos = _softplus_lse(-self.alpha * (sim - self.base), pos_keep) / self.alpha
        neg = _softplus_lse(self.beta * (sim - self.base), neg_keep) / self.beta

        valid = same.any(1)
        if not valid.any():
            raise ValueError("no anchor has a positive; check the PK sampler")
        loss = (pos + neg)[valid].mean()

        st = _pair_stats(emb, labels)
        with torch.no_grad():
            st["mined_pos"] = pos_keep.sum(1).float().mean().item()
            st["mined_neg"] = neg_keep.sum(1).float().mean().item()
        return loss, st


class CircleLoss(nn.Module):
    """Circle loss, pair formulation (Sun et al., CVPR 2020).

    The weight on a pair is proportional to how far it still is from its own
    optimum, so pairs that are already fine stop pulling. That is the difference
    from a fixed margin: the gradient budget flows to the pairs that are wrong,
    and the decision boundary in (s_p, s_n) space becomes an arc rather than the
    line that triplet and softmax losses impose.
    """

    name = "circle"

    def __init__(self, m: float = 0.25, gamma: float = 80.0):
        super().__init__()
        self.m, self.gamma = m, gamma

    def forward(self, emb, labels):
        sim = (emb @ emb.T).float()
        same, diff = _masks(labels)

        # Weights are treated as constants -- they set the step size, they are
        # not themselves something to descend.
        ap = torch.clamp_min(1 + self.m - sim, 0.0).detach()
        an = torch.clamp_min(sim + self.m, 0.0).detach()
        logit_p = -self.gamma * ap * (sim - (1 - self.m))
        logit_n = self.gamma * an * (sim - self.m)

        lse_p = _masked_lse(logit_p, same)
        lse_n = _masked_lse(logit_n, diff)
        valid = torch.isfinite(lse_p) & torch.isfinite(lse_n)
        if not valid.any():
            raise ValueError("no anchor has both a positive and a negative")
        loss = F.softplus(lse_p[valid] + lse_n[valid]).mean()
        return loss, _pair_stats(emb, labels)


class Triplet(nn.Module):
    """Batch-hard triplet. Present so the collapse is reproducible, not hidden."""

    name = "triplet"

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb, labels):
        loss, st = batch_hard_triplet(emb, labels, margin=self.margin)
        return loss, {**_pair_stats(emb, labels), **st}


# --- proxy-based ------------------------------------------------------------

class _ProxyBank(nn.Module):
    """One unit vector per player.

    Sized by the number of *train* players, so the caller has to remap raw
    lichess player ids onto a contiguous range first. At ~250k players this is
    32M parameters -- four times the trunk -- which is fine on a 3090 but is the
    reason these arms get their own learning-rate group.
    """

    def __init__(self, n_classes: int, d: int):
        super().__init__()
        self.n_classes = n_classes
        self.weight = nn.Parameter(torch.empty(n_classes, d))
        nn.init.normal_(self.weight, std=0.02)

    def unit(self):
        return F.normalize(self.weight, dim=-1)


class ProxyAnchor(_ProxyBank):
    """Proxy-Anchor loss (Kim et al., CVPR 2020).

    The reason this is the proxy method worth trying here rather than ArcFace:
    the negative term ranges over many proxies at once, so they receive gradient
    far faster than under ArcFace, which only touches the proxies appearing in
    the batch -- with a PK sampler that is 64-128 out of 226k, roughly five
    updates per proxy across an entire run.

    Two departures from the paper, both forced by scale. It was published on
    datasets with ~100-11k classes; there are 226k players here.

      * `neg_sample` -- the negative term is taken over a random subset of
        proxies rather than all of them. At 226k the full term buys nothing (the
        mean repulsion from a near-isotropic random bank is close to zero) while
        contributing all the gradient noise, and the first attempt collapsed to
        a single point inside 200 steps because of it.
      * alpha 16, not 32, for the same reason: less amplification of a bank that
        is mostly untrained at any given moment.

    The other half of the fix is not here -- see --proxy-warmup in finetune_mt.py,
    which holds the trunk still until the bank means something.
    """

    name = "proxyanchor"

    def __init__(self, n_classes: int, d: int = 128, alpha: float = 16.0,
                 delta: float = 0.1, neg_sample: int = 16384):
        super().__init__(n_classes, d)
        self.alpha, self.delta, self.neg_sample = alpha, delta, neg_sample

    def forward(self, emb, labels):
        W = self.unit()
        if self.neg_sample and self.n_classes > self.neg_sample:
            idx = torch.randint(self.n_classes, (self.neg_sample,), device=emb.device)
            idx = torch.unique(torch.cat([labels, idx]))     # positives always in
            sim = (emb @ W[idx].T).float()                   # (B, S)
            pos = idx[None, :] == labels[:, None]
        else:
            sim = (emb @ W.T).float()
            pos = F.one_hot(labels, self.n_classes).bool()

        # Transposed relative to the pair losses: the anchor is the proxy, so
        # reductions run down the batch dimension and out over the proxies.
        p_term = _softplus_lse((-self.alpha * (sim - self.delta)).T, pos.T)
        n_term = _softplus_lse((self.alpha * (sim + self.delta)).T, ~pos.T)

        with_pos = pos.any(0)
        if not with_pos.any():
            raise ValueError("batch has no labelled proxy")
        loss = p_term[with_pos].mean() + n_term.mean()
        return loss, _pair_stats(emb, labels)


class ArcFace(_ProxyBank):
    """Additive angular margin softmax (Deng et al., CVPR 2019).

    s is deliberately modest (32, not the 64 face pipelines use): with proxies
    this undertrained a large scale mostly amplifies proxy noise.

    `neg_sample` is partial-FC: score against the batch's true classes plus a
    random sample of the rest, rather than all 226k. This is the standard way
    ArcFace is trained past ~100k identities, and here it is also the difference
    between fitting on the GPU and not -- the full 128x226k logit matrix and its
    arccos intermediates OOM'd a 24 GB card that was already hosting three other
    arms.
    """

    name = "arcface"

    def __init__(self, n_classes: int, d: int = 128, s: float = 32.0,
                 m: float = 0.3, neg_sample: int = 16384):
        super().__init__(n_classes, d)
        self.s, self.m, self.neg_sample = s, m, neg_sample

    def forward(self, emb, labels):
        W = self.unit()
        if self.neg_sample and self.n_classes > self.neg_sample:
            idx = torch.randint(self.n_classes, (self.neg_sample,), device=emb.device)
            # unique() returns sorted, so searchsorted recovers each label's
            # column in the sampled set.
            idx = torch.unique(torch.cat([labels, idx]))
            labels = torch.searchsorted(idx, labels)
            W = W[idx]
        cos = (emb @ W.T).float().clamp(-1 + 1e-7, 1 - 1e-7)
        theta = cos.arccos()
        tgt = torch.cos(theta.gather(1, labels[:, None]) + self.m)
        # Past pi - m the shifted cosine turns back up and the margin would
        # *reward* a worse angle; fall back to a plain additive penalty there.
        raw = cos.gather(1, labels[:, None])
        tgt = torch.where(theta.gather(1, labels[:, None]) + self.m < math.pi,
                          tgt, raw - self.m * math.sin(self.m))
        logits = self.s * cos.scatter(1, labels[:, None], tgt)
        loss = F.cross_entropy(logits, labels)

        st = _pair_stats(emb, labels)
        with torch.no_grad():
            st["proxy_acc"] = (logits.argmax(1) == labels).float().mean().item()
        return loss, st


# --- registry ---------------------------------------------------------------

PAIR_LOSSES = ("supcon", "ms", "circle", "triplet")
PROXY_LOSSES = ("proxyanchor", "arcface")
ALL_LOSSES = PAIR_LOSSES + PROXY_LOSSES

# The four the sweep actually funds: one per family, chosen so that if they
# ensemble at all, it is because they disagree for structural reasons.
SWEEP_LOSSES = ("supcon", "ms", "circle", "proxyanchor")


def needs_proxies(name: str) -> bool:
    return name in PROXY_LOSSES


def make_loss(name: str, n_classes: int = 0, d_embed: int = 128, **kw):
    """Build a loss by name. `n_classes` is required for the proxy families."""
    name = name.lower()
    if name == "supcon":
        return SupCon(temperature=kw.get("temperature", 0.07))
    if name == "ms":
        return MultiSimilarity()
    if name == "circle":
        return CircleLoss()
    if name == "triplet":
        return Triplet(margin=kw.get("margin", 0.0))
    if name == "proxyanchor":
        if not n_classes:
            raise ValueError("proxyanchor needs n_classes")
        return ProxyAnchor(n_classes, d_embed)
    if name == "arcface":
        if not n_classes:
            raise ValueError("arcface needs n_classes")
        return ArcFace(n_classes, d_embed)
    raise ValueError(f"unknown loss {name!r}; have {ALL_LOSSES}")


def default_pk(name: str, p: int, k: int) -> tuple[int, int]:
    """Batch shape a loss actually wants, at a fixed batch size P*K.

    Pair losses need K >= 2 or an anchor has no positive at all. Proxy losses do
    not compare samples to each other, so every slot spent on a second game of
    the same player is a slot not spent covering another proxy -- K=2 keeps a
    little in-batch structure while doubling proxy coverage.
    """
    if needs_proxies(name) and k > 2:
        total = p * k
        return total // 2, 2
    return p, k
