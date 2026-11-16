#!/usr/bin/env python3
"""Check a manifest against its job script before anyone asks for approval.

Reads the request from the #PBS lines and the runs from the manifest, then prints the
core-hours requested, the credits under an assumed charge, the time every run could use and
the memory each case is expected to need. Exit 1 if the runs cannot all start within the
walltime, a case would pass its node's memory request, or a limit is under twice the
estimate. Nothing here is a measurement: estimates only size the request, and the constants
below must be replaced from pilot results before a manifest is frozen.

  plan_budget.py MANIFEST PBS_SCRIPT [--rate R] [--startup-s S] [--runs]

--rate is the assumed credits per core-hour requested. The site's conversion is unconfirmed.
--startup-s is the time expected for gates before the first run.
"""
import argparse
import json
import math
import re
import sys

from run_manifest import GRACE_S, expand

# From pilots 1523275 (16 cores of a shared Gold 6438N node) and 1522998 (a whole 64-core node),
# one repetition each, and bench16 1523703 (three repetitions, shared node). Nothing here has been
# measured across nodes.
# Serial traverse, core-microseconds per particle per force evaluation at N = 1e6: 19.3 on the
# whole node, 31 to 44 on the shared one. 31 is kept; limits are at least twice the estimate.
KERNEL_US = 31.0
# Growth with N: 17, 23, 31 and 39 us at 1e5, 3e5, 1e6 and 2e6. Beyond 2e6 is extrapolation.
KERNEL_N_EXPONENT = 0.33
# Extra kernel core-time against serial on the whole node: 2.08 at 64 x 1, 1.71 at 8 x 8, and 1.59
# at 1 x 64, so threads cost too at 64. These factors give 1.90, 1.89 and 1.60. Causes untested;
# more than 64 ranks is extrapolated.
RANK_KERNEL_PER_DOUBLING = 0.15
THREAD_KERNEL_PER_DOUBLING = 0.10
# Smaller theta opens more cells: 3.7 times the traverse time from 0.5 to 0.3, theta^-2.6. Other
# leaf sizes took 1.17 to 1.37 times leaf size 8; 1.5 covers them.
THETA_REF, THETA_EXPONENT, OTHER_LEAF_FACTOR = 0.5, 2.6, 1.5
# Tree build per rank, microseconds per particle per evaluation at N = 1e6: slowest of the pilot (16 x 1).
BUILD_US = 0.28
# Allgatherv: 24 bytes per particle into every rank of a node, at 5 GB/s, the lowest seen on one node.
# A measured exchange includes waiting for the slowest rank: 0.076 s per step at 8 x 8 and N = 1e6,
# above the byte estimate, so each exchange takes at least EXCHANGE_FLOOR_S per million particles.
# Between nodes is not measured.
NODE_BYTES_PER_S = 5e9
EXCHANGE_FLOOR_S = 0.08
# Setup, microseconds per particle: serial setup less its two force evaluations.
SETUP_US = 2.2
# A run does the initial evaluation, one per warmup step and one per timed step.
EXTRA_EVALUATIONS = 1
# Peak RSS per rank in MiB: 156.2 at 16 x 1, N = 1e6; 135 per million between serial 1e6 and 2e6. The
# 30% stays because per-rank MPI memory may grow with ranks, which 16 ranks cannot show.
RSS_BASE_MIB, RSS_PER_MILLION_MIB, RSS_MARGIN = 20.0, 136.4, 1.3
# Gate time after the runner's deadline: see jobs/vanda/gates.sh and measure.sh.
END_RESERVE_S, GATE_GRACE_S, SNAPSHOT_NODE_S, CLEANUP_NODE_S, ANALYZE_LIMIT_S = 60, 20, 12, 40, 120


def request(path):
    text = open(path).read()
    m = re.search(r"^#PBS -l select=(\d+):ncpus=(\d+):mpiprocs=\d+:mem=(\d+)gb", text, re.M)
    w = re.search(r"^#PBS -l walltime=(\d+):(\d+):(\d+)", text, re.M)
    if not m or not w:
        sys.exit(f"{path}: no select or walltime line")
    h, mi, s = map(int, w.groups())
    return int(m[1]), int(m[2]), int(m[3]), h * 3600 + mi * 60 + s


def estimate_s(p):
    n = p["n"]
    if p["program"] == "validate":
        return 60.0
    kernel = KERNEL_US * (n / 1e6) ** KERNEL_N_EXPONENT * 1e-6 * n
    kernel *= max(1.0, (THETA_REF / p["theta"]) ** THETA_EXPONENT) * (1.0 if p["leaf_size"] == 8 else OTHER_LEAF_FACTOR)
    build = BUILD_US * (n / 1e6) ** 0.3 * 1e-6 * n
    evaluations = p["steps"] + p["warmup"] + EXTRA_EVALUATIONS
    if p["program"] == "serial":
        return SETUP_US * 1e-6 * n + evaluations * (kernel + build)
    per_node = p["ranks"] / p.get("nodes", 1)
    exchange = max(24 * n * per_node / NODE_BYTES_PER_S, EXCHANGE_FLOOR_S * n / 1e6) if p["ranks"] > 1 else 0.0
    kernel *= 1 + RANK_KERNEL_PER_DOUBLING * math.log2(p["ranks"])
    kernel *= 1 + THREAD_KERNEL_PER_DOUBLING * math.log2(p["threads"])
    return SETUP_US * 1e-6 * n + evaluations * (kernel / (p["ranks"] * p["threads"]) + build + exchange)


def node_mem_gb(p):
    per_node = p["ranks"] / p.get("nodes", 1) if p["program"] == "hybrid" else 1
    return per_node * RSS_MARGIN * (RSS_BASE_MIB + RSS_PER_MILLION_MIB * p["n"] / 1e6) / 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("pbs")
    ap.add_argument("--rate", type=float, default=0.0)
    ap.add_argument("--startup-s", type=int, default=300)
    ap.add_argument("--runs", action="store_true", help="print every case")
    a = ap.parse_args()
    nodes, cores, mem_gb, walltime = request(a.pbs)
    points = expand(json.load(open(a.manifest)))
    overhead = GATE_GRACE_S + nodes * (SNAPSHOT_NODE_S + 2) + 2 + nodes * (CLEANUP_NODE_S + 5)
    budget = walltime - END_RESERVE_S - 2 * overhead - ANALYZE_LIMIT_S - a.startup_s
    problems, worst, expected, runs = [], 0, 0.0, 0
    for p in points:
        est, mem, reps = estimate_s(p), node_mem_gb(p), p.get("reps", 1)
        worst += reps * (p["timeout_s"] + GRACE_S)
        expected += reps * est
        runs += reps
        if a.runs:
            print(f"  {p['name']}: est {est:.0f} s, limit {p['timeout_s']} s, node memory {mem:.0f} GB, reps {reps}")
        if mem > mem_gb:
            problems.append(f"{p['name']}: {mem:.0f} GB per node over the {mem_gb} GB requested")
        if p["timeout_s"] < 2 * est:
            problems.append(f"{p['name']}: limit {p['timeout_s']} s under twice the {est:.0f} s estimate")
    core_h = nodes * cores * walltime / 3600
    print(f"request: {nodes} x {cores} cores, {mem_gb} GB each, {walltime} s = {core_h:g} core-hours requested")
    print(f"credits at assumed rate {a.rate:g} per core-hour: {a.rate * core_h:g} if charged on the request")
    print(f"runs: {runs}, estimated {expected:.0f} s, worst case {worst} s, budget {budget} s after "
          f"{a.startup_s} s of gates and {walltime - budget - a.startup_s} s reserved at the end")
    print(f"expected job time: {(expected + a.startup_s) / 60:.0f} min = "
          f"{nodes * cores * (expected + a.startup_s) / 3600:.0f} core-hours used")
    print(f"largest node memory: {max(node_mem_gb(p) for p in points):.0f} GB of {mem_gb} GB")
    if worst > budget:
        problems.append(f"worst case {worst} s over the {budget} s budget, so later runs could be skipped")
    for x in problems:
        print("PROBLEM", x)
    print("PLAN " + ("REFUSED" if problems else "OK"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
