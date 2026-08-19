#!/usr/bin/env python3
"""Vultr control for chess-tracker, with the budget cap enforced in code.

The $300 promotional credit expires at the end of the month and there is a hard
instruction not to exceed it without explicit authorization. A cap that lives
only in someone's head is not a cap, so every command that can create a charge
checks projected spend first and refuses rather than warns.

    ./deploy/vultr.py plans              # what is available, sorted by price
    ./deploy/vultr.py spend              # charged so far, and the run rate
    ./deploy/vultr.py bench <plan>       # rent one for an hour, measure, destroy
    ./deploy/vultr.py up <plan> <label>  # create (asks first)
    ./deploy/vultr.py down <label>       # destroy

The API key is read from .env and never printed. Vultr keys carry an IP
allowlist, so a 401 here usually means this machine's address is not on it --
the error says which address to add.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://api.vultr.com/v2"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The hard ceiling. Everything below refuses to cross it; raising it is a
# deliberate edit, not a flag someone can pass in a hurry.
BUDGET_USD = 300.0
# Leave room so a forgotten instance cannot walk the total up to the line while
# nobody is looking.
RESERVE_USD = 25.0


# Enabling API access on the account reissued the token, so the original VULTR
# value is dead and VULTR_USER is live. Try the working one first and fall back,
# rather than making the next person rediscover that.
KEY_NAMES = ("VULTR_USER", "VULTR")


def key() -> str:
    env = {}
    for line in open(os.path.join(HERE, ".env"), encoding="utf-8"):
        if "=" in line and not line.lstrip().startswith("#"):
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    for name in KEY_NAMES:
        if env.get(name):
            return env[name]
    sys.exit(f"none of {KEY_NAMES} found in .env")


def api(path, method="GET", body=None):
    req = urllib.request.Request(
        API + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key(),
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as f:
            raw = f.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:300]
        # The two 401s mean completely different things and the message is the
        # only way to tell them apart, so say which one it is rather than
        # letting the caller assume the key is bad.
        if e.code == 401 and "Unauthorized IP" in detail:
            sys.exit(f"Vultr rejected this machine's IP.\n  {detail}\n"
                     "  Add it under Account -> API -> Access Control in the "
                     "Vultr portal, then retry.")
        if e.code == 401 and "not enabled" in detail:
            # Two different toggles, and which one applies depends on whose key
            # this is. Neither is on the user detail page that shows
            # "API Access: Disabled" -- that page only reports the state.
            sys.exit(
                "Vultr API access is switched off. This is NOT the IP "
                "allowlist and NOT a bad key.\n  " + detail + "\n"
                "  Account key  : Account -> API (under OTHER) -> 'Enable API'\n"
                "                 under Personal Access Token.\n"
                "  Org user key : Account -> Users (under OTHER) -> the Edit "
                "User (pencil)\n                 icon on that row -> 'Enable "
                "API' under User API Key.\n"
                "  Needs a root account or the 'Manage Users' permission.")
        sys.exit(f"HTTP {e.code} on {method} {path}\n  {detail}")


# --------------------------------------------------------------------------
def spend():
    """What has been charged, and what the current fleet costs per day."""
    acct = api("/account")["account"]
    charged = float(acct["pending_charges"])
    # Vultr carries credit as a negative balance; -300 means $300 available.
    credit = -float(acct["balance"])
    ins = api("/instances").get("instances", [])
    plans = {p["id"]: p for p in api("/plans?per_page=500").get("plans", [])}
    per_month = sum(float(plans.get(i["plan"], {}).get("monthly_cost", 0)) for i in ins)
    per_day = per_month / 30.0

    now = datetime.datetime.now(datetime.timezone.utc)
    nxt = (now.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
    days_left = (nxt - now).days + 1

    print(f"charged so far   ${charged:8.2f}")
    print(f"credit remaining ${credit:8.2f}")
    print(f"running          {len(ins)} instance(s), ${per_month:.2f}/mo "
          f"= ${per_day:.2f}/day")
    print(f"days left in month {days_left}")
    projected = charged + per_day * days_left
    print(f"projected by month end ${projected:.2f} of ${BUDGET_USD:.0f} "
          f"({100 * projected / BUDGET_USD:.0f}%)")
    for i in ins:
        p = plans.get(i["plan"], {})
        print(f"  {i.get('label') or i['id'][:8]:24s} {i['plan']:26s} "
              f"${float(p.get('monthly_cost', 0)):6.2f}/mo  {i.get('main_ip')}")
    return charged, per_day, days_left


def guard(extra_monthly: float):
    """Refuse rather than warn. Returns only if the new charge stays inside."""
    charged, per_day, days_left = spend()
    extra_per_day = extra_monthly / 30.0
    projected = charged + (per_day + extra_per_day) * days_left
    room = BUDGET_USD - RESERVE_USD
    print(f"\nadding ${extra_monthly:.2f}/mo -> projected ${projected:.2f} "
          f"against a ${room:.0f} working ceiling "
          f"(${BUDGET_USD:.0f} cap less ${RESERVE_USD:.0f} reserve)")
    if projected > room:
        sys.exit(f"REFUSED: that would project ${projected:.2f}, over the "
                 f"${room:.0f} ceiling. Raising BUDGET_USD is a deliberate edit.")
    print("within budget.\n")



# --------------------------------------------------------------------------
def guard_hours(hourly_total: float, hours: float):
    """Budget check for something rented by the hour, not kept.

    guard() projects a monthly rate across the rest of the month, which is the
    right question for the always-on host and the wrong one for a 20-minute
    benchmark -- three $48/mo boxes look like $144/mo and would be refused for a
    seven cent experiment.
    """
    charged, per_day, days_left = spend()
    cost = hourly_total * hours
    projected = charged + per_day * days_left + cost
    room = BUDGET_USD - RESERVE_USD
    print(f"\nephemeral: ${hourly_total:.4f}/hr x {hours:.2f} h = ${cost:.2f}"
          f"  -> projected ${projected:.2f} of ${room:.0f}")
    if projected > room:
        sys.exit(f"REFUSED: projects ${projected:.2f}, over ${room:.0f}.")
    print("within budget.\n")


def ssh_key_id(pubkey_path):
    """Upload this machine's public key once; reuse it after that."""
    pub = open(os.path.expanduser(pubkey_path)).read().strip()
    for k in api("/ssh-keys").get("ssh_keys", []):
        if k["ssh_key"].strip().split()[:2] == pub.split()[:2]:
            return k["id"]
    return api("/ssh-keys", "POST",
               {"name": "chess-tracker", "ssh_key": pub})["ssh_key"]["id"]


def create(plan, label, region="ewr", os_id=2284, sshkey=None):
    body = {"region": region, "plan": plan, "os_id": os_id, "label": label,
            "hostname": label, "backups": "disabled"}
    if sshkey:
        body["sshkey_id"] = [sshkey]
    return api("/instances", "POST", body)["instance"]


def destroy(iid):
    api(f"/instances/{iid}", "DELETE")


def wait_active(iid, timeout=420):
    """Poll until Vultr reports the instance running with an IP."""
    for _ in range(timeout // 5):
        i = api(f"/instances/{iid}")["instance"]
        if i["status"] == "active" and i["main_ip"] not in ("", "0.0.0.0"):
            return i
        time.sleep(5)
    raise RuntimeError(f"instance {iid} never became active")

def plans_cmd(maxmo=None):
    """Every plan that could host this, cheapest first.

    Filtered to what the service actually needs: the main process is 740 MB
    resident with the gallery loaded and each move worker is 253 MB, so 2 GB is
    the floor and 4 GB is the first size with real headroom.
    """
    ps = api("/plans?per_page=500").get("plans", [])
    ok = [p for p in ps
          if p.get("ram", 0) >= 4096
          and p.get("vcpu_count", 0) >= 2
          and (maxmo is None or float(p.get("monthly_cost", 1e9)) <= maxmo)]
    ok.sort(key=lambda p: float(p["monthly_cost"]))
    print(f"{'plan':30s} {'vCPU':>4} {'RAM':>7} {'disk':>6} {'$/mo':>7} {'$/hr':>7}  type")
    for p in ok:
        mo = float(p["monthly_cost"])
        print(f"{p['id']:30s} {p['vcpu_count']:>4} {p['ram']/1024:>6.0f}G "
              f"{p.get('disk',0):>5}G {mo:>7.2f} {mo/730:>7.4f}  {p.get('type','')}")
    print(f"\n{len(ok)} plans >=2 vCPU and >=4 GB"
          + (f" under ${maxmo}/mo" if maxmo else ""))


def regions_for(plan):
    for p in api("/plans?per_page=500").get("plans", []):
        if p["id"] == plan:
            return p.get("locations", [])
    sys.exit(f"unknown plan {plan}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "spend":
        spend()
    elif cmd == "plans":
        plans_cmd(float(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "regions":
        print(", ".join(regions_for(sys.argv[2])))
    elif cmd == "guard":
        guard(float(sys.argv[2]))
    elif cmd == "sshkey":
        print(ssh_key_id(sys.argv[2] if len(sys.argv) > 2
                         else "~/.ssh/id_ed25519.pub"))
    elif cmd == "down":
        for i in api("/instances").get("instances", []):
            if i.get("label") == sys.argv[2]:
                print("destroying", i["label"], i["id"])
                destroy(i["id"])
    elif cmd == "down-all":
        for i in api("/instances").get("instances", []):
            print("destroying", i.get("label"), i["id"])
            destroy(i["id"])
    else:
        sys.exit(f"unknown command {cmd}\n{__doc__}")


if __name__ == "__main__":
    main()
