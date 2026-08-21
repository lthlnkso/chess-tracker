# Horizontal worker scaling, measured 2026-08-19

Origin: `vhp-4c-8gb-amd` in Newark (main server + 2 move workers + 1 identify
worker). Worker boxes: the same plan, 3 move + 1 identify each, no main server.
Load generated from a Mac mini outside the datacentre, which is where real load
comes from — it peaked at 48% of its 8 cores, so it was never the constraint.

## The premise this rests on

A remote MOVE worker on the Mac measured *worse than none*: move p50 167 -> 195
ms, because a Cloudflare round trip each way costs more than the 30 ms of
compute it offloads. Inside the datacentre that inverts completely:

| path | RTT |
|---|---|
| worker box -> origin, same datacentre | **0.66 ms** |
| Mac -> production via Cloudflare | 21 ms |

0.66 ms against 30 ms of compute is 2% overhead. Remote move workers only make
sense on the near side of that number.

## Results

Comfortable ceiling = the largest N where move p50 stays under ~250 ms.

| fleet | ceiling | peak moves/s | move p50 at ceiling |
|---|---|---|---|
| origin alone | ~96 players | 47.7 | 197 ms |
| + 1 worker box | **~160** | 82.0 | 202 ms |
| + 3 worker boxes | ~192 | 95.5 | 310 ms |

The first worker box is worth roughly a doubling. The second and third are not:
three boxes buy 20% more than one.

## Why it flattens

Sampled through a 192-player run, **both ends sit at 100% CPU** — origin median
80% / peak 100% across 4 cores, worker box median 74% / peak 100%. The origin
was carrying **199 threads**, one per connected WebSocket plus the workers.

Workers only take the model forward pass. The origin still pays, per move, for
the connection thread, the legality and game-over check in `move_submit`, the
SQLite write and the result push — and that work is CPU-bound and cannot be
handed off without moving the connection itself.

So worker boxes lift the *compute* ceiling and the origin's *connection* ceiling
becomes binding around 200 players. Going past that wants a bigger origin (or
less per-move origin work), not more workers.

## What to actually do in a spike

Start **one** worker box. $0.0658/hr, $1.58/day, and it takes the site from ~96
to ~160 concurrent players. A second is worth little until the origin grows.

## Addendum: load balancing, not worker fan-out

Worker fan-out stalled at ~192 players because workers only take the forward
pass -- the origin still paid, per move, for the connection thread, the legality
check, the queue write and the result push. Balancing across whole nodes
distributes that too. Measured through nginx across main plus two nodes:

| players | moves/s | move p50 | move p95 |
|---|---|---|---|
| 192 | 77.8 | 125 ms | 202 ms |
| 320 | 120.4 | 249 ms | 980 ms |
| 448 | **137.9** | 809 ms | 2031 ms |

Against 95.5 moves/s peak for three worker boxes behind a single origin. Same
hardware count, 44% more throughput, and 125 ms at 192 players where the
worker-fan-out arrangement was already past its knee.

That is the argument for the balancer over more workers: workers relieve the
compute ceiling, nodes relieve the connection ceiling, and the connection
ceiling is the one that binds first.
