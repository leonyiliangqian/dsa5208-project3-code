#!/usr/bin/env python3
"""Recheck a finished run directory from its raw records, independently of analyze.py.

Every run in plan.json must have one record that matches its case: parameters, effective team
and schedule, manifest, commit, binary and host. Phase times must be consistent with the wall time
(see phase_problems), stdout must agree with the record to its printed precision, stderr must be empty, and summary.csv must agree with the raw values. Nothing is
compared against an expected speed. Prints one line per case. Exit 1 if any check fails.

  verify_run.py RUN_DIR
"""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

from run_manifest import GRACE_S, expand

PHASES = ("integrate_s", "exchange_s", "build_s", "traverse_s")


def phase_problems(r, ranks):
    """Phase times against wall time. Each phase and the wall time are maxima over ranks, reduced
    independently (MPI_MAX per value in bh_hybrid.cpp), so with several ranks they may come from
    different ranks: the sum of the phase maxima may exceed the wall time (a rank waiting in the
    exchange for a slower one), but no phase can exceed it, and the slowest rank's phases, each at
    most its maximum, account for its wall time, so the sum cannot fall far below it. One rank
    (or serial): all values come from the same process, so they must add up."""
    wall = r["wall_s"]
    total = sum(r[k] for k in PHASES)
    out = [f"{k} {r[k]:.3f} negative" for k in PHASES if r[k] < 0]
    out += [f"{k} {r[k]:.3f} over wall {wall:.3f}" for k in PHASES if r[k] > wall * 1.001 + 0.01]
    if ranks <= 1:
        if not 0.9 * wall <= total <= 1.1 * wall + 0.1:
            out.append(f"phases {total:.3f} against wall {wall:.3f}")
    elif total < 0.9 * wall:
        out.append(f"phases {total:.3f} under wall {wall:.3f}")
    for lo, hi in (("traverse_min_s", "traverse_mean_s"), ("traverse_mean_s", "traverse_s"), ("wall_min_s", "wall_s")):
        if lo in r and hi in r and r[lo] > r[hi] * 1.000001 + 1e-6:
            out.append(f"{lo} {r[lo]:.6f} over {hi} {r[hi]:.6f}")
    return out


def stdout_wall_ok(wall_s, text):
    """The program prints wall_s=%.4f from the same double it writes with %.6f into the record,
    so the two may differ by half a unit in the 4th decimal (double rounding), never more."""
    m = re.search(r"\bwall_s=([0-9.]+)", text)
    return bool(m) and abs(float(m.group(1)) - wall_s) <= 0.5e-4 + 1e-6


def main():
    run = Path(sys.argv[1])
    plan = json.load(open(run / "plan.json"))
    manifest_bytes = (run / "manifest.json").read_bytes()
    points = {p["name"]: p for p in expand(json.loads(manifest_bytes))}
    bad = []

    def check(ok, what):
        if not ok:
            bad.append(what)

    check(hashlib.sha256(manifest_bytes).hexdigest() == plan["manifest_sha256"], "manifest hash against plan")
    seen = {"binary": {}, "source": set(), "rev": set(), "model": set(), "host": set(), "job": set()}
    raw = {}
    counts = {"completed": 0, "failed": 0, "timed_out": 0, "skipped": 0}
    for planned in plan["runs"]:
        case, rep, p = planned["case"], planned["rep"], points.get(planned["case"])
        check(p is not None, f"{case}: not in the manifest")
        folder = "validate" if planned["program"] == "validate" else "raw"
        path = run / folder / f"{case}_rep{rep}.json"
        if not path.exists():
            counts["skipped"] += 1
            bad.append(f"{case} rep={rep}: no record")
            continue
        d = json.load(open(path))
        if d.get("skipped"):
            counts["skipped"] += 1
            bad.append(f"{case} rep={rep}: skipped")
            continue
        check(d["tags"]["manifest_sha256"] == plan["manifest_sha256"], f"{case}: manifest hash")
        if d["timed_out"]:
            counts["timed_out"] += 1
        counts["completed" if d["ok"] and d["returncode"] == 0 and not d["timed_out"] else "failed"] += 1
        check(d["ok"] and d["returncode"] == 0, f"{case} rep={rep}: failed or timed out")
        check(d["elapsed_s"] <= p["timeout_s"] + GRACE_S, f"{case}: elapsed over the limit")
        if planned["program"] == "validate":
            print(f"{case} rep={rep} ok={d['ok']} elapsed={d['elapsed_s']:.1f}/{p['timeout_s']}")
            continue
        r = d["result"]
        raw[(case, rep)] = d
        check(r["n"] == max(1, p["n"] // plan.get("scale", 1)), f"{case}: n {r['n']} against {p['n']}")
        for key in ("theta", "steps", "warmup", "leaf_size", "seed"):
            check(r[key] == p[key], f"{case}: {key} {r[key]} against {p[key]}")
        check(not d.get("problems"), f"{case}: problems {d.get('problems')}")
        check(d["allocated_cores"] == (p["ranks"] * p["threads"] if p["program"] == "hybrid" else 1), f"{case}: cores")
        if p["program"] == "hybrid":
            check(r["ranks"] == p["ranks"] and r["threads_effective"] == p["threads"], f"{case}: team")
            check(r["schedule_effective"] == p["schedule"], f"{case}: schedule")
            if p["chunk"] > 0:
                check(r["chunk_effective"] == p["chunk"], f"{case}: chunk")
            check(d["env"]["OMP_NUM_THREADS"] == str(p["threads"]), f"{case}: OMP_NUM_THREADS")
            mapping = f"ppr:{p['ranks'] // p.get('nodes', 1)}:node:PE={p['threads']}"
            check(mapping in d["command"] and "core" in d["command"], f"{case}: mapping")
        else:
            check(d["command"][0] == "taskset", f"{case}: not pinned")
        check(len(r["hosts"]) == p.get("nodes", 1), f"{case}: hosts {r['hosts']}")
        phases = sum(r[k] for k in PHASES)
        for problem in phase_problems(r, p["ranks"] if p["program"] == "hybrid" else 1):
            check(False, f"{case}: {problem}")
        check(d["elapsed_s"] >= r["setup_s"] + r["wall_s"] - 0.01, f"{case}: elapsed under setup + wall")
        check(not (run / folder / f"{case}_rep{rep}.stderr").read_text().strip(), f"{case}: stderr not empty")
        check(stdout_wall_ok(r["wall_s"], (run / folder / f"{case}_rep{rep}.stdout").read_text()), f"{case}: stdout wall")
        seen["binary"].setdefault(d["binary"], set()).add(d["binary_sha256"])
        seen["source"].add(d["source_tree_sha256"])
        seen["rev"].add((d["git_rev"], d["git_dirty"]))
        seen["model"].add(d["cpu_model"])
        seen["host"].add(d["launch_host"])
        seen["job"].add(d["scheduler_job_id"])
        print(f"{case} rep={rep} ok={d['ok']} elapsed={d['elapsed_s']:.1f}/{p['timeout_s']} wall={r['wall_s']:.3f} "
              f"phases={phases:.3f}")
    for key, values in seen.items():
        if key != "binary":
            check(len(values) == 1, f"{key} differs between records: {values}")
    check(all(len(v) == 1 for v in seen["binary"].values()), "a binary changed during the job")
    check(not any(rev[1] for rev in seen["rev"]), "dirty source")

    rows = list(csv.DictReader(open(run / "summary.csv")))
    check(len(rows) == len({c for c, _ in raw}), f"{len(rows)} summary rows for {len(raw)} raw records")
    for row in rows:
        walls = [d["result"]["wall_s"] for (c, _), d in raw.items() if c == row["cases"]]
        check(walls and abs(sorted(walls)[len(walls) // 2] - float(row["wall_median_s"])) < 1e-5, f"{row['cases']}: summary wall")

    print("counts", " ".join(f"{k}={v}" for k, v in counts.items()), f"planned={len(plan['runs'])}")
    print("provenance", seen["rev"], seen["model"], seen["host"], seen["job"], seen["source"])
    print("binaries", {k: sorted(v) for k, v in seen["binary"].items()})
    for what in bad:
        print("FAIL", what)
    print("VERIFY " + ("FAIL" if bad else "PASS"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
