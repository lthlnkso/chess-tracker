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


# --- multi-task pre-training ------------------------------------------------

class MultiTaskModel(nn.Module):
    """One trunk, four heads.

        trunk  -> per-ply hidden states
          |- move head   (per-ply)  score candidate successor positions
          |- time head   (per-ply)  predict log1p(ms) of the *upcoming* move
          |- elo head    (pooled)   predict the attributed player's rating
          `- embed head  (pooled)   128-d player vector, fed the Elo estimate

    Per-ply input is the board planes plus `n_extra` clock features. The Elo
    head is deliberately kept at deployment: its output feeds the embedding, and
    a rating estimate also prunes the retrieval gallery.
    """

    def __init__(self, c: Config = Config(), n_planes: int = 8,
                 n_extra: int = 2, d_embed: int = 128,
                 n_time_bins: int = 9, n_elo_bins: int = 20, n_game_slots: int = 1,
                 elo_cond: bool = False, elo_steer: bool = False):
        super().__init__()
        self.cfg = c
        self.n_planes = n_planes
        self.n_extra = n_extra
        self.d_embed = d_embed
        self.n_time_bins = n_time_bins
        self.n_elo_bins = n_elo_bins
        self.n_game_slots = n_game_slots
        d_in = n_planes * SQ + n_extra

        self.in_proj = nn.Linear(d_in, c.d_model)
        self.pos = nn.Embedding(c.max_len, c.d_model)
        # Marks which game a position belongs to. Ply index stays per-game so
        # "ply 3" means the same thing everywhere; without this the model could
        # not tell a game boundary from an ordinary move.
        self.game_emb = nn.Embedding(max(n_game_slots, 1), c.d_model) \
            if n_game_slots > 1 else None
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.ln_f = nn.LayerNorm(c.d_model)

        # move head: candidates share this encoder, scored by dot product
        self.cand_enc = nn.Sequential(
            nn.Linear(n_planes * SQ, c.d_ff), nn.GELU(),
            nn.Linear(c.d_ff, c.d_model), nn.GELU(),
            nn.Linear(c.d_model, c.d_model),
        )
        self.ln_c = nn.LayerNorm(c.d_model)
        self.scale = c.d_model ** -0.5

        # Both auxiliaries are classifiers, not regressors. Think time is a
        # small integer count (85% of bullet plies are 0/1/2 s) and rating is
        # ordinal -- a softmax fits both far better than squared error, and
        # yields a distribution rather than a point estimate.
        self.time_head = nn.Sequential(
            nn.Linear(c.d_model, c.d_model // 2), nn.GELU(),
            nn.Linear(c.d_model // 2, n_time_bins),
        )
        self.elo_head = nn.Sequential(
            nn.Linear(c.d_model, c.d_model // 2), nn.GELU(),
            nn.Linear(c.d_model // 2, n_elo_bins),
        )
        # pooled hidden state + predicted Elo + pooled time summary
        self.embed_head = nn.Sequential(
            nn.Linear(c.d_model + n_elo_bins + n_extra, c.d_model), nn.GELU(),
            nn.Linear(c.d_model, d_embed),
        )
        # Optional rating conditioning. score_candidates() is a dot product of
        # the trunk state with candidate encodings, so nothing about the model's
        # MOVE choice can depend on rating unless rating enters the trunk -- the
        # elo_head only ever fed embed_head. Adding it here makes the played
        # style settable at inference, which is the point: the demo's opponent
        # currently plays a single blend averaged over every rating in the data,
        # matching no human in the gallery.
        #
        # One extra row is the "unknown rating" slot, used for games where the
        # dump carries no rating and as the inference default. Zero-init means a
        # checkpoint that has never trained this is bit-identical to before.
        self.elo_cond = (nn.Embedding(n_elo_bins + 1, c.d_model)
                         if elo_cond else None)
        # Self-conditioning: the model's own running rating estimate steers move
        # choice, so no caller has to supply a rating.
        #
        # The estimate must be CAUSAL. elo_head normally reads a pool over the
        # whole game, and feeding that back into the move at ply t would let the
        # move be chosen using plies after t -- the same leak the time features
        # are carefully shifted to avoid. causal_elo_p() pools the PREFIX only.
        #
        # Zero-init, matching elo_cond: a checkpoint that never trained this is
        # bit-identical to one built without it.
        # Gated, and the gate is what starts at zero -- not the projection.
        #
        # ctx comes out of ln_f, so the candidate dot product assumes a
        # layer-normalised scale (||ctx|| ~ 20 at d_model 384). Adding an
        # unbounded Linear to it lets every move logit grow with the projection's
        # weights until the cross-entropy overflows: measured NaN at step 11k,
        # then again at 15k after the gradient fix. tanh bounds the direction and
        # the scalar gate bounds its size.
        #
        # The PROJECTION keeps its normal init and only the GATE is zeroed. Zero
        # the projection instead and the contribution is still zero, but so is
        # the gradient reaching it, and it never learns anything.
        self.elo_steer = nn.Linear(n_elo_bins, c.d_model) if elo_steer else None
        self.steer_gate = nn.Parameter(torch.zeros(1)) if elo_steer else None

        self.apply(ChessTransformer._init)
        if self.elo_cond is not None:
            nn.init.zeros_(self.elo_cond.weight)


    # -- trunk ------------------------------------------------------------
    def encode(self, planes, extra, game_slot=None, ply_pos=None, elo_bin=None):
        b, t = planes.shape[:2]
        x = planes.reshape(b, t, -1).to(self.in_proj.weight.dtype)
        if self.n_extra:
            x = torch.cat([x, extra.to(x.dtype)], dim=-1)
        if ply_pos is None:
            ply_pos = torch.arange(t, device=planes.device)[None].expand(b, t)
        x = self.in_proj(x) + self.pos(ply_pos.clamp(max=self.pos.num_embeddings - 1))
        if self.game_emb is not None and game_slot is not None:
            x = x + self.game_emb(game_slot.clamp(max=self.game_emb.num_embeddings - 1))
        if self.elo_cond is not None:
            # One rating per sample, broadcast over its plies.
            if elo_bin is None:
                elo_bin = torch.full((b,), self.elo_cond.num_embeddings - 1,
                                     dtype=torch.long, device=planes.device)
            x = x + self.elo_cond(elo_bin.clamp(
                max=self.elo_cond.num_embeddings - 1))[:, None, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    def _pool(self, h, pad_mask, my_turn=None):
        keep = ~pad_mask
        if my_turn is not None:
            keep = keep & my_turn
        w = keep.unsqueeze(-1).to(h.dtype)
        return (h * w).sum(1) / w.sum(1).clamp(min=1)

    def causal_elo_p(self, h, pad_mask, my_turn=None):
        """Rating estimate from the PREFIX at every ply -- (B, T, n_elo_bins).

        A running mean over plies <= t, so the estimate available when choosing
        move t never contains move t or anything after it. Uses the same
        elo_head as the whole-game estimate, so the two share a calibration.
        """
        keep = ~pad_mask
        if my_turn is not None:
            keep = keep & my_turn
        # float32, and NO GRADIENT. Both matter.
        #
        # cumsum's backward is a reverse-cumsum, so ply 0 receives the summed
        # gradient of every later ply -- measured 12,728x the gradient reaching
        # the last ply over a 1600-ply context. clip_grad_norm_ clips the GLOBAL
        # norm, so it rescales that imbalance rather than removing it, and the
        # first run went NaN at ~11k steps. elo_head is already trained by its
        # own supervised loss on the whole-game pool, so this is a READ-OUT of
        # it: the move loss has no business flowing back through 1,600 prefix
        # terms. elo_steer still trains normally, on the detached estimate.
        with torch.no_grad():
            w = keep.unsqueeze(-1).float()
            running = (h.float() * w).cumsum(1) / w.cumsum(1).clamp(min=1)
            return F.softmax(self.elo_head(running.to(h.dtype)).float(), dim=-1)

    # -- heads ------------------------------------------------------------
    def score_candidates(self, h, cands, ply_idx, elo_p=None):
        ctx = h.gather(1, ply_idx[..., None].expand(-1, -1, h.size(-1)))
        if self.elo_steer is not None and elo_p is not None:
            # the prefix estimate AT each scored ply, not a game-level one
            ep = elo_p.gather(1, ply_idx[..., None].expand(-1, -1, elo_p.size(-1)))
            ctx = ctx + self.steer_gate * torch.tanh(self.elo_steer(ep.to(ctx.dtype)))
        B, P, C = cands.shape[:3]
        e = self.ln_c(self.cand_enc(cands.reshape(B, P, C, -1).to(ctx.dtype)))
        return torch.einsum("bpd,bpcd->bpc", ctx, e) * self.scale

    def forward(self, planes, extra, cands, ply_idx, pad_mask, my_turn=None,
                game_slot=None, ply_pos=None, elo_bin=None):
        h = self.encode(planes, extra, game_slot, ply_pos, elo_bin)
        ep = (self.causal_elo_p(h, pad_mask, my_turn)
              if self.elo_steer is not None else None)
        move_logits = self.score_candidates(h, cands, ply_idx, ep)
        time_logits = self.time_head(h)                       # (B, T, n_time_bins)
        pooled = self._pool(h, pad_mask, my_turn)
        elo_logits = self.elo_head(pooled)                    # (B, n_elo_bins)
        return move_logits, time_logits, elo_logits, pooled, h

    def embed(self, planes, extra, pad_mask, my_turn=None, game_slot=None,
              ply_pos=None, elo_bin=None):
        """128-d player vector. Elo estimate and pooled clock stats feed in."""
        h = self.encode(planes, extra, game_slot, ply_pos, elo_bin)
        pooled = self._pool(h, pad_mask, my_turn)
        elo_logits = self.elo_head(pooled)
        elo_p = F.softmax(elo_logits.float(), dim=-1)          # (B, n_elo_bins)
        keep = (~pad_mask if my_turn is None else (~pad_mask & my_turn))
        w = keep.unsqueeze(-1).to(extra.dtype)
        time_summary = (extra * w).sum(1) / w.sum(1).clamp(min=1)   # (B, n_extra)
        z = torch.cat([pooled, elo_p.to(pooled.dtype),
                       time_summary.to(pooled.dtype)], dim=-1)
        return F.normalize(self.embed_head(z), dim=-1), elo_logits


# Defaults only. The real population mean for 2026 60+0 is ~1816, not 1500, so a
# hardcoded centre makes the head fight weight decay to learn a large bias for
# nothing. Callers should pass statistics measured on the shard.
# --- Elo binning -----------------------------------------------------------
# Fixed 100-point bins with saturating ends. Targets are smoothed across
# neighbouring bins because rating is *ordinal*: predicting 1900 for a 1850
# player should cost far less than predicting 1200, and plain one-hot
# cross-entropy treats those two mistakes identically.

ELO_LO, ELO_HI, ELO_BIN = 800.0, 2600.0, 100.0
N_ELO_BINS = int((ELO_HI - ELO_LO) / ELO_BIN) + 2      # + below / above
ELO_CENTRES = torch.tensor(
    [ELO_LO - ELO_BIN / 2] +
    [ELO_LO + ELO_BIN * (i + 0.5) for i in range(int((ELO_HI - ELO_LO) / ELO_BIN))] +
    [ELO_HI + ELO_BIN / 2])


def elo_to_bin(elo):
    """Rating -> index into ELO_CENTRES; N_ELO_BINS means "unknown".

    Bin 0 is everything below ELO_LO and the last real bin everything above
    ELO_HI, matching how ELO_CENTRES is laid out. Ratings of 0 (absent from the
    dump) map to the unknown slot rather than being silently treated as 800.
    """
    e = torch.as_tensor(elo)
    idx = torch.floor((e.float() - ELO_LO) / ELO_BIN).long() + 1
    idx = idx.clamp(0, N_ELO_BINS - 1)
    return torch.where(e > 0, idx, torch.full_like(idx, N_ELO_BINS))


def elo_soft_targets(elo, sigma_bins: float = 1.0):
    """Gaussian mass over adjacent bins, centred on the true rating."""
    c = ELO_CENTRES.to(elo.device)
    d = (elo.float()[:, None] - c[None, :]) / (sigma_bins * ELO_BIN)
    t = torch.softmax(-0.5 * d.pow(2), dim=-1)
    return t


def elo_expectation(logits):
    """Point estimate for reporting: expected value under the predicted bins."""
    return (torch.softmax(logits.float(), -1) * ELO_CENTRES.to(logits.device)).sum(-1)


WP_K = 0.00368208            # centipawn -> win probability (lichess constant)


def cpl_loss(move_logits, batch):
    """Expected squared win-probability error against the move the human played.

        loss = sum_i p_i * (WP_i - WP_human)^2

    Cross-entropy treats all 31 non-played candidates as equally wrong. This
    grades them: a candidate of the SAME quality as the human's move costs
    nothing, and a blunder costs a lot -- so the trunk learns the player's error
    profile rather than their exact moves. Symmetric on purpose. Predicting a
    move much BETTER than the human played is penalised as hard as a blunder,
    because a model of this player should not be a stronger player.

    Win probability, not centipawns: candidates span ~550cp and a third are
    >=300cp blunders, so a raw-cp target is dominated by moves the model already
    avoids. The logistic bounds the term in [0,1] and puts its gradient where
    the position is still in the balance.

    Plies with no corpus entry are dropped, not zero-filled -- an unlabelled ply
    carries no information about quality and must not vote for uniformity.
    """
    ev = batch["cand_eval"]                              # (B, S, C) cp, NaN = unknown
    known = batch["cand_mask"] & ~torch.isnan(ev)
    rows = batch["ply_mask"] & batch["cpl_ok"] & known.any(-1)
    if not bool(rows.any()):
        return None, 0

    wp = torch.sigmoid(WP_K * torch.nan_to_num(ev.float(), nan=0.0))
    lab = batch["label"].unsqueeze(-1)
    wp_j = wp.gather(-1, lab)                            # the human's move
    known_j = known.gather(-1, lab).squeeze(-1)
    rows = rows & known_j                                # need the human's own eval
    if not bool(rows.any()):
        return None, 0

    logits = move_logits.masked_fill(~known, float("-inf"))
    p = torch.softmax(logits.float(), -1)
    p = torch.where(known, p, torch.zeros_like(p))
    d2 = (wp - wp_j) ** 2
    per_ply = (p * d2).sum(-1)
    return per_ply[rows].mean(), int(rows.sum().item())


def multitask_loss(move_logits, time_logits, elo_logits, batch, w_move=1.0,
                   w_time=0.3, w_elo=0.3, time_centres=None, w_cpl=0.0):
    """Weighted sum of the three objectives, each reported in its own units."""
    out = {}
    loss, _ = successor_loss(move_logits, batch["label"], batch["cand_mask"],
                             batch["ply_mask"])
    with torch.no_grad():
        sel = batch["ply_mask"] & batch["cand_mask"].any(-1)
        lg = move_logits.masked_fill(~batch["cand_mask"], float("-inf"))[sel]
        out["move_acc"] = (lg.argmax(-1) == batch["label"][sel]).float().mean().item()
    out["move"] = loss.item()
    total = w_move * loss

    if w_cpl > 0.0 and "cand_eval" in batch:
        cl, n = cpl_loss(move_logits, batch)
        if cl is not None:
            out["cpl"] = cl.item()
            out["cpl_plies"] = n
            total = total + w_cpl * cl
        else:
            out["cpl"] = float("nan")
            out["cpl_plies"] = 0

    tvalid = batch["time_valid"] & ~batch["pad_mask"]
    if tvalid.any():
        tl = F.cross_entropy(time_logits[tvalid].float(), batch["time_target"][tvalid])
        out["time"] = tl.item()
        with torch.no_grad():
            pred = time_logits[tvalid].float().argmax(-1)
            out["time_acc"] = (pred == batch["time_target"][tvalid]).float().mean().item()
            if time_centres is not None:
                cen = time_centres.to(time_logits.device)
                exp_s = (torch.softmax(time_logits[tvalid].float(), -1) * cen).sum(-1)
                out["time_mae_s"] = (exp_s - cen[batch["time_target"][tvalid]]).abs().mean().item()
        total = total + w_time * tl
    else:
        out["time"] = float("nan")

    evalid = batch["elo"] > 0
    if evalid.any():
        tgt = elo_soft_targets(batch["elo"][evalid])
        lp = F.log_softmax(elo_logits[evalid].float(), dim=-1)
        el = -(tgt * lp).sum(-1).mean()
        out["elo"] = el.item()
        with torch.no_grad():
            est = elo_expectation(elo_logits[evalid])
            out["elo_mae"] = (est - batch["elo"][evalid].float()).abs().mean().item()
        total = total + w_elo * el
    else:
        out["elo"] = float("nan")

    out["total"] = total.item()
    return total, out
