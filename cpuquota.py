"""How many cores this container may actually use.

Extracted from cpl_label.py so every entry point shares one implementation.
The failure it prevents: on an eval pod, nproc reported 48 while the cgroup
allowed ~8, and 25 dataloader workers starved the GPU to 0% -- the k=10 gallery
build ran at 28 bundles/s against k=5's 503/s, an 18x slowdown that looked like
a model problem and was a scheduling problem.

    python cpuquota.py        # prints the usable core count
"""

from __future__ import annotations

import os


def cpu_quota():
    """Cores this CONTAINER may use, which is not what nproc reports.

    RunPod containers expose the host's core count through nproc,
    os.cpu_count() AND sched_getaffinity, while cgroup caps the real quota.
    Measured on an A5000 pod: all three said 96, cpu.max said 765000/100000 =
    7.65. Spawning 94 engines onto 7.65 cores does not merely fail to help --
    context switching made throughput WORSE than running 7, and the symptom
    (engines idle in S state) looks exactly like task starvation.
    """
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            return max(1, int(int(q) / int(p)))
    except Exception:                                   # noqa: BLE001
        pass
    try:
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0:
            return max(1, q // p)
    except Exception:                                   # noqa: BLE001
        pass
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
        else (os.cpu_count() or 1)


if __name__ == "__main__":
    print(cpu_quota())
