"""How deep must the shortlist go before the right player is nearly always in it?

A verification model can only re-order what the first stage hands it. So the
ceiling on any re-ranking scheme is recall@depth of plain cosine -- and the cost
of a cross-encoder scales linearly with that depth.
"""
import argparse, numpy as np, torch, chess
from rerank import load_model, embed_query
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ckpt/final/ctx5_ft2.pt")
ap.add_argument("--shard", default="data/2026-06-big")
ap.add_argument("--gallery", default="play/gallery_2026.npz")
ap.add_argument("--queries", type=int, default=300)
ap.add_argument("--ks", default="5,10")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
g = np.load(a.gallery, allow_pickle=True)
C = torch.from_numpy(g["centroids"].astype(np.float32))
names = [str(n).lower() for n in g["names"]]
idx = {n:i for i,n in enumerate(names)}
model, ck = load_model(a.ckpt, dev)
meta = np.load(f"{a.shard}/meta.npy", mmap_mode="r")
moves = np.memmap(f"{a.shard}/moves.u16", dtype=np.uint16, mode="r")
clocks = np.memmap(f"{a.shard}/clocks.u16", dtype=np.uint16, mode="r")
sn = open(f"{a.shard}/players.txt", encoding="utf-8").read().split("\n")
pid = np.concatenate([np.asarray(meta["white_pid"]), np.asarray(meta["black_pid"])])
gid = np.concatenate([np.arange(len(meta))]*2)
seat = np.concatenate([np.zeros(len(meta),np.int8), np.ones(len(meta),np.int8)])
ok = np.concatenate([np.asarray(clocks[np.asarray(meta["offset"],np.int64)])!=0xFFFF]*2)
o=np.argsort(pid,kind="stable"); pid,gid,seat,ok = pid[o],gid[o],seat[o],ok[o]
bnd = np.flatnonzero(np.r_[True, pid[1:]!=pid[:-1], True])
rng = np.random.default_rng(0)
order = list(range(len(bnd)-1)); rng.shuffle(order)
KS = [int(x) for x in a.ks.split(",")]
DEPTHS = [1,10,100,1000,10000,100000]
res = {k: [] for k in KS}
for k in KS:
    n=0
    for i in order:
        if n >= a.queries: break
        sl=slice(bnd[i],bnd[i+1]); m=ok[sl]
        g_,s_ = gid[sl][m], seat[sl][m]
        if len(g_) < k: continue
        p=int(pid[sl][0])
        if p>=len(sn): continue
        j = idx.get(sn[p].lower())
        if j is None: continue
        sel = rng.permutation(len(g_))[:k]
        try:
            q,_,_ = embed_query(model, ck, (meta,moves,clocks),
                                [(int(g_[x]),int(s_[x])) for x in sel], dev)
        except Exception: continue
        sims = q @ C.T
        rank = int((sims > sims[j]).sum()) + 1
        res[k].append(rank); n+=1
print(f"\ngallery {len(names):,} players\n")
print(f"{'games':>6}" + "".join(f"{'r@'+str(d):>10}" for d in DEPTHS))
for k in KS:
    r = np.array(res[k])
    print(f"{k:>6}" + "".join(f"{(r<=d).mean()*100:>9.1f}%" for d in DEPTHS)
          + f"   median rank {np.median(r):.0f}  n={len(r)}")
