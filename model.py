"""Causal transformer over POV bitboard sequences.

One token per position. The pretraining task is next-move prediction: from the
positions up to and including ply t, predict the move played at ply t. That is
causal and leak-free, because position t+1 is exactly what the move at ply t
produces -- the model never sees it when predicting.

The same encoder carries the 128-dim game embedding the identification mission
needs (`embed_game`), so step 3 is a head swap rather than a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitboards import N_PLANES

SQ = 64
N_FROM_TO = SQ * SQ      # 4096 from-square x to-square
N_PROMO = 5              # none, N, B, R, Q
PAD_MOVE = 0xFFFF


@dataclass
class Config:
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1024
    max_len: int = 160
    dropout: float = 0.1
    d_embed: int = 128       # game-vector width


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.d_head = c.d_model // c.n_heads
        self.ln1 = nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.ff = nn.Sequential(
            nn.Linear(c.d_model, c.d_ff), nn.GELU(), nn.Linear(c.d_ff, c.d_model)
        )
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(d, dim=2)
        q, k, v = (z.view(b, t, self.n_heads, self.d_head).transpose(1, 2) for z in (q, k, v))
        # Padding is right-side only and always masked out of the loss, so a plain
        # causal mask is enough -- real positions never attend to padding.
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.drop(self.proj(a))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class ChessTransformer(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.cfg = c
        self.in_proj = nn.Linear(N_PLANES * SQ, c.d_model)
        self.pos = nn.Embedding(c.max_len, c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head_move = nn.Linear(c.d_model, N_FROM_TO)
        self.head_promo = nn.Linear(c.d_model, N_PROMO)
        self.head_embed = nn.Linear(c.d_model, c.d_embed)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def encode(self, planes: torch.Tensor) -> torch.Tensor:
        """planes: (B, T, 18, 8, 8) uint8 -> hidden states (B, T, d_model)."""
        b, t = planes.shape[:2]
        x = planes.reshape(b, t, -1).to(self.in_proj.weight.dtype)
        x = self.in_proj(x) + self.pos(torch.arange(t, device=planes.device))[None]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    def forward(self, planes: torch.Tensor):
        h = self.encode(planes)
        return self.head_move(h), self.head_promo(h)

    @torch.no_grad()
    def embed_game(self, planes: torch.Tensor, pad_mask: torch.Tensor,
                   my_turn: torch.Tensor | None = None) -> torch.Tensor:
        """L2-normalised 128-dim vector per game, for the player-centroid step.

        Pools over the attributed player's own turns when `my_turn` is given --
        their choices are the identity signal, not the opponent's.
        """
        h = self.encode(planes)
        keep = ~pad_mask
        if my_turn is not None:
            keep = keep & my_turn
        w = keep.unsqueeze(-1).to(h.dtype)
        pooled = (h * w).sum(1) / w.sum(1).clamp(min=1)
        return F.normalize(self.head_embed(pooled), dim=-1)


class SuccessorScorer(nn.Module):
    """Scores candidate successor states against the game so far.

    History (B, T, 8, 8, 8) goes through a causal transformer; each candidate
    (B, P, C, 8, 8, 8) goes through a small shared board encoder. The logit is a
    scaled dot product, so the C candidates at a ply cost one matmul rather than C
    transformer passes.
    """

    def __init__(self, c: Config = Config(), n_planes: int = 8):
        super().__init__()
        self.cfg = c
        self.n_planes = n_planes
        d_in = n_planes * SQ

        self.in_proj = nn.Linear(d_in, c.d_model)
        self.pos = nn.Embedding(c.max_len, c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.ln_f = nn.LayerNorm(c.d_model)

        self.cand_enc = nn.Sequential(
            nn.Linear(d_in, c.d_ff), nn.GELU(),
            nn.Linear(c.d_ff, c.d_model), nn.GELU(),
            nn.Linear(c.d_model, c.d_model),
        )
        self.ln_c = nn.LayerNorm(c.d_model)
        self.head_embed = nn.Linear(c.d_model, c.d_embed)
        self.scale = c.d_model ** -0.5
        self.apply(ChessTransformer._init)

    def encode(self, planes: torch.Tensor) -> torch.Tensor:
        b, t = planes.shape[:2]
        x = planes.reshape(b, t, -1).to(self.in_proj.weight.dtype)
        x = self.in_proj(x) + self.pos(torch.arange(t, device=planes.device))[None]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    def forward(self, planes, cands, ply_idx):
        """-> logits (B, P, C). Context at ply t includes position t, not t+1."""
        h = self.encode(planes)                                   # (B, T, D)
        ctx = h.gather(1, ply_idx[..., None].expand(-1, -1, h.size(-1)))  # (B, P, D)

        B, P, C = cands.shape[:3]
        e = self.cand_enc(cands.reshape(B, P, C, -1).to(ctx.dtype))
        e = self.ln_c(e)                                          # (B, P, C, D)
        return torch.einsum("bpd,bpcd->bpc", ctx, e) * self.scale

    @torch.no_grad()
    def embed_game(self, planes, pad_mask, my_turn=None) -> torch.Tensor:
        h = self.encode(planes)
        keep = ~pad_mask
        if my_turn is not None:
            keep = keep & my_turn
        w = keep.unsqueeze(-1).to(h.dtype)
        pooled = (h * w).sum(1) / w.sum(1).clamp(min=1)
        return F.normalize(self.head_embed(pooled), dim=-1)


class PlayerEncoder(nn.Module):
    """Pre-trained trunk + a fresh projection head, for metric learning.

    The trunk modules are named exactly as in `SuccessorScorer`, so the
    pre-trained checkpoint loads with `strict=False`: the transformer transfers,
    the candidate scorer is dropped, and `proj` starts fresh. Pooling is over the
    player's own turns -- their choices carry the identity, the opponent's do not.
    """

    def __init__(self, c: Config = Config(), n_planes: int = 13, d_embed: int = 128):
        super().__init__()
        self.cfg = c
        self.n_planes = n_planes
        self.d_embed = d_embed
        self.in_proj = nn.Linear(n_planes * SQ, c.d_model)
        self.pos = nn.Embedding(c.max_len, c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.ln_f = nn.LayerNorm(c.d_model)
        self.proj = nn.Sequential(
            nn.Linear(c.d_model, c.d_model), nn.GELU(), nn.Linear(c.d_model, d_embed)
        )
        self.apply(ChessTransformer._init)

    def load_pretrained(self, state_dict: dict) -> tuple[int, int]:
        own = self.state_dict()
        take = {k: v for k, v in state_dict.items()
                if k in own and own[k].shape == v.shape and not k.startswith("proj")}
        self.load_state_dict(take, strict=False)
        return len(take), len(own)

    def forward(self, planes, pad_mask, my_turn=None) -> torch.Tensor:
        b, t = planes.shape[:2]
        x = planes.reshape(b, t, -1).to(self.in_proj.weight.dtype)
        x = self.in_proj(x) + self.pos(torch.arange(t, device=planes.device))[None]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        h = self.ln_f(x)

        keep = ~pad_mask
        if my_turn is not None:
            keep = keep & my_turn
        w = keep.unsqueeze(-1).to(h.dtype)
        pooled = (h * w).sum(1) / w.sum(1).clamp(min=1)
        return F.normalize(self.proj(pooled), dim=-1)


def supcon_loss(emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07):
    """Supervised contrastive loss (Khosla et al.) on L2-normalised embeddings.

    Chosen over batch-hard triplet because triplet has a degenerate minimum at
    "map everything to one point": there d_pos == d_neg, softplus(0) = ln 2, and
    the run parks there. Observed here within 40 steps.

    SupCon cannot do that. Collapse makes every similarity identical, the softmax
    uniform, and the loss its *maximum* ln(B-1) -- so the gradient points away
    from collapse rather than into it.
    """
    device = emb.device
    b = len(labels)
    sim = (emb @ emb.T) / temperature
    eye = torch.eye(b, dtype=torch.bool, device=device)
    same = (labels[:, None] == labels[None, :]) & ~eye

    sim = sim - sim.max(dim=1, keepdim=True).values.detach()      # stability
    exp_sim = torch.exp(sim).masked_fill(eye, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)

    n_pos = same.sum(1)
    valid = n_pos > 0
    if valid.sum() == 0:
        raise ValueError("no anchor has a positive; check the PK sampler")
    loss = -((log_prob * same).sum(1)[valid] / n_pos[valid]).mean()

    with torch.no_grad():
        cos = emb @ emb.T
        pos_cos = cos[same].mean().item() if same.any() else float("nan")
        neg_mask = ~same & ~eye
        neg_cos = cos[neg_mask].mean().item()
        stats = {
            "pos_cos": pos_cos,
            "neg_cos": neg_cos,
            "gap": pos_cos - neg_cos,          # collapse => gap -> 0
            "emb_std": emb.std(0).mean().item(),
        }
    return loss, stats


def batch_hard_triplet(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.0):
    """Batch-hard triplet loss (Hermans et al.) on L2-normalised embeddings.

    For every anchor, the hardest positive and hardest negative *in the batch*.
    Random triplets go slack almost immediately once the easy ones are solved,
    which is why the PK sampler exists.
    """
    d = torch.cdist(emb.float(), emb.float(), p=2)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)

    pos = d.masked_fill(~same | eye, float("-inf"))
    hardest_pos = pos.max(1).values

    neg = d.masked_fill(same, float("inf"))
    hardest_neg = neg.min(1).values

    valid = torch.isfinite(hardest_pos) & torch.isfinite(hardest_neg)
    if valid.sum() == 0:
        raise ValueError("batch has no valid triplet; check the PK sampler")
    hp, hn = hardest_pos[valid], hardest_neg[valid]

    if margin > 0:
        loss = F.relu(hp - hn + margin).mean()
    else:
        loss = F.softplus(hp - hn).mean()      # soft margin: nothing to tune

    with torch.no_grad():
        stats = {
            "d_pos": hp.mean().item(),
            "d_neg": hn.mean().item(),
            "frac_violating": (hp >= hn).float().mean().item(),
        }
    return loss, stats


def successor_loss(logits, label, cand_mask, ply_mask):
    """Cross-entropy over each ply's candidate set, ignoring padded plies."""
    logits = logits.masked_fill(~cand_mask, float("-inf"))
    sel = ply_mask & cand_mask.any(-1)
    lg = logits[sel]
    tg = label[sel]
    loss = F.cross_entropy(lg, tg)
    with torch.no_grad():
        n_cand = cand_mask[sel].sum(-1).float()
        acc = (lg.argmax(-1) == tg).float().mean()
        # Guessing uniformly among that ply's legal moves -- the honest baseline.
        chance = (1.0 / n_cand).mean()
    return loss, {"acc": acc.item(), "chance": chance.item(),
                  "cands": n_cand.mean().item(), "n": int(sel.sum())}


def move_targets(moves: torch.Tensor):
    """Unpack uint16 move codes into (from_to index, promo class, valid mask)."""
    valid = moves != PAD_MOVE
    m = moves.clamp(min=0) * valid          # keep padding out of the bit math
    frm = m & 0x3F
    to = (m >> 6) & 0x3F
    promo = (m >> 12) & 0x7
    # python-chess piece types 2..5 (N,B,R,Q) -> classes 1..4; 0 stays "none".
    return frm * SQ + to, (promo - 1).clamp(min=0), valid


def loss_and_stats(logits_move, logits_promo, moves, pad_mask):
    tgt_ft, tgt_promo, valid = move_targets(moves)
    valid = valid & ~pad_mask
    if valid.sum() == 0:
        raise ValueError("batch contains no supervised positions")

    lm = logits_move[valid]
    lp = logits_promo[valid]
    tm = tgt_ft[valid]
    tp = tgt_promo[valid]

    loss = F.cross_entropy(lm, tm) + 0.2 * F.cross_entropy(lp, tp)
    with torch.no_grad():
        pred = lm.argmax(-1)
        acc = (pred == tm).float().mean()
        top5 = (lm.topk(5, dim=-1).indices == tm[:, None]).any(-1).float().mean()
    return loss, {"acc": acc.item(), "top5": top5.item(), "n": int(valid.sum())}
