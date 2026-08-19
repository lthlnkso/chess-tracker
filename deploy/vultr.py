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


def key() -> str:
    for line in open(os.path.join(HERE, ".env"), encoding="utf-8"):
        if line.startswith("VULTR="):
            return line.split("=", 1)[1].strip()
    sys.exit("VULTR not found in .env")


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
    balance = float(acct["balance"])
    ins = api("/instances").get("instances", [])
    plans = {p["id"]: p for p in api("/plans?per_page=500").get("plans", [])}
    per_month = sum(float(plans.get(i["plan"], {}).get("monthly_cost", 0)) for i in ins)
    per_day = per_month / 30.0

    now = datetime.datetime.now(datetime.timezone.utc)
    nxt = (now.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
    days_left = (nxt - now).days + 1

    print(f"charged so far   ${charged:8.2f}")
    print(f"credit balance   ${balance:8.2f}")
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
    else:
        sys.exit(f"unknown command {cmd}\n{__doc__}")


if __name__ == "__main__":
    main()
