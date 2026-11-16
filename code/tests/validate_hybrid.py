#!/usr/bin/env python3
"""Hybrid gate: serial vs MPI+OpenMP Barnes-Hut on identical problems.

This measures implementation error, not approximation error: both programs use
the same theta, leaf size, softening, timestep and initial state, and the final
states are compared particle by particle by stable ID.

Tolerances are fixed here, before results are seen, and are not loosened to
make a run pass:
  accelerations after 1 step      |d| <= 1e-12 |ref| + 1e-14 max|ref|
  positions, velocities, accel.
  after 10 steps                  |d| <= 1e-10 |ref| + 1e-13 max|ref|
The hybrid computes each particle's force with the same code on the same tree,
so differences are expected to be zero or rounding-level; any larger value is a
bug (wrong ownership, stale exchange, race), not a tolerance question.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile

from execution_context import execution_context

ACC_TOL = (1e-12, 1e-14)
TRAJ_TOL = (1e-10, 1e-13)

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--serial", required=True)
parser.add_argument("--hybrid", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--mpirun", default="mpirun")
parser.add_argument("--mpirun-flags", default=os.environ.get("MPIRUN_FLAGS", "--bind-to none"),
                    help="extra launcher flags, e.g. binding policy")
parser.add_argument("--max-cores", type=int, default=os.cpu_count() or 4,
                    help="skip configurations with ranks*threads above this")
parser.add_argument("--suite", choices=("single", "multinode", "bound"), default="single",
                    help="multinode adds 16-core cases meant to span two nodes; bound replaces the "
                         "cases with ones using all --max-cores, placed as measurements are")
parser.add_argument("--expect-hosts", type=int, default=0,
                    help="require multi-rank runs to report exactly min(ranks, this) hosts")
opts = parser.parse_args()
serial, hybrid = str(Path(opts.serial).resolve()), str(Path(opts.hybrid).resolve())
work = Path(tempfile.mkdtemp(prefix="validate_hybrid_"))
failures, records = [], []


def run(cmd, env=None, timeout=300):
    """Run with a timeout so a hung collective fails the gate instead of stalling it."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", f"TIMEOUT after {timeout}s")


def hybrid_cmd(ranks, extra, threads=1):
    flags = opts.mpirun_flags.split()
    if opts.suite == "bound":
        # The placement of scripts/run_manifest.py, so the comparison covers what is timed.
        per_node = math.ceil(ranks / max(opts.expect_hosts, 1))
        flags += ["--map-by", f"ppr:{per_node}:node:PE={threads}", "--bind-to", "core",
                  "-x", "OMP_NUM_THREADS", "-x", "OMP_PROC_BIND", "-x", "OMP_PLACES"]
    return [opts.mpirun, "-np", str(ranks), *flags, hybrid, *extra]


def hybrid_env(threads):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    if opts.suite == "bound":
        env.update(OMP_PROC_BIND="close", OMP_PLACES="cores")
    # Open MPI forwards the launching environment to local ranks.
    return env


def load_dump(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            v = line.split()
            rows.append((int(v[0]), [float(x) for x in v[1:]]))
    return rows


def compare(ref_rows, test_rows, fields, tol):
    """Max violation ratio and max abs diff over the named field groups."""
    names = {"pos": slice(0, 3), "vel": slice(3, 6), "acc": slice(6, 9)}
    ids_ref = [i for i, _ in ref_rows]
    ids_test = [i for i, _ in test_rows]
    if ids_test != ids_ref or len(set(ids_test)) != len(ids_test):
        return {"ok": False, "why": "particle IDs differ or repeat"}
    mass_ref = math.fsum(r[9] for _, r in ref_rows)
    mass_test = math.fsum(r[9] for _, r in test_rows)
    if mass_ref != mass_test:
        return {"ok": False, "why": f"total mass {mass_test!r} != {mass_ref!r}"}
    out = {"ok": True}
    for field in fields:
        sl = names[field]
        norms = [math.sqrt(sum(x * x for x in r[sl])) for _, r in ref_rows]
        scale = max(norms, default=0.0)
        worst_ratio, worst_abs = 0.0, 0.0
        for (_, r), (_, t), norm in zip(ref_rows, test_rows, norms):
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(r[sl], t[sl])))
            if not math.isfinite(d):
                worst_ratio = math.inf
                break
            bound = tol[0] * norm + tol[1] * scale
            worst_abs = max(worst_abs, d)
            worst_ratio = max(worst_ratio, d / bound if bound > 0 else (0.0 if d == 0 else math.inf))
        out[f"{field}_max_abs_diff"] = worst_abs
        out[f"{field}_max_tol_ratio"] = worst_ratio
        if worst_ratio > 1.0:
            out["ok"] = False
            out["why"] = f"{field} exceeds tolerance (ratio {worst_ratio:.3g})"
    return out


def serial_reference(n, steps, theta, leaf, seed=42):
    path = work / f"serial_n{n}_s{steps}_t{theta}_l{leaf}.txt"
    if not path.exists():
        r = run([serial, "--n", str(n), "--steps", str(steps), "--theta", str(theta),
                 "--leaf-size", str(leaf), "--seed", str(seed), "--dump", str(path)])
        if r.returncode != 0:
            raise SystemExit(f"FAIL: serial reference run failed: {r.stderr}")
    return load_dump(path)


def check(name, n, steps, theta, leaf, ranks, threads, schedule="static", chunk=0):
    if ranks * threads > opts.max_cores:
        print(f"skip {name}: {ranks}x{threads} exceeds --max-cores {opts.max_cores}")
        return
    ref = serial_reference(n, steps, theta, leaf)
    dump = work / f"hybrid_{name}.txt"
    extra = ["--n", str(n), "--steps", str(steps), "--theta", str(theta), "--leaf-size",
             str(leaf), "--seed", "42", "--schedule", schedule, "--chunk", str(chunk),
             "--dump", str(dump)]
    r = run(hybrid_cmd(ranks, extra, threads), env=hybrid_env(threads))
    rec = {"case": name, "n": n, "steps": steps, "theta": theta, "leaf_size": leaf,
           "ranks": ranks, "threads": threads, "schedule": schedule, "chunk": chunk,
           "returncode": r.returncode}
    if r.returncode != 0:
        rec.update(ok=False, why=f"exit {r.returncode}: {r.stderr.strip()[-300:]}")
    else:
        result = next((l for l in r.stdout.splitlines() if l.startswith("result ")), None)
        meta = json.loads(result[len("result "):]) if result else {}
        hosts = meta.get("hosts", [])
        want_hosts = min(ranks, opts.expect_hosts) if opts.expect_hosts else len(hosts)
        rec["hosts"] = hosts
        if meta.get("replicated") is not True or meta.get("threads") != threads:
            rec.update(ok=False, why=f"bad metadata: replicated={meta.get('replicated')} "
                                     f"threads={meta.get('threads')}")
        elif len(hosts) != want_hosts:
            rec.update(ok=False, why=f"ran on {len(hosts)} host(s) {hosts}, expected {want_hosts}")
        else:
            tol = ACC_TOL if steps == 1 else TRAJ_TOL
            fields = ["acc"] if steps == 1 else ["pos", "vel", "acc"]
            rec.update(compare(ref, load_dump(dump), fields, tol))
            rec["schedule_effective"] = meta.get("schedule_effective")
    status = "PASS" if rec["ok"] else "FAIL " + rec.get("why", "")
    diffs = " ".join(f"{k}={v:.2e}" for k, v in rec.items() if k.endswith("max_abs_diff"))
    print(f"hybrid case={name} n={n} steps={steps} ranks={ranks} threads={threads} "
          f"schedule={schedule}/{chunk} hosts={len(rec.get('hosts', []))} {diffs} {status}")
    records.append(rec)
    if not rec["ok"]:
        failures.append(name)


# One-step accelerations, then 10-step trajectories, across rank/thread splits,
# schedules, uneven division and ranks that own no particles.
if opts.suite == "bound":
    # Every core of the job, bound as in measurements. Only this suite compares bound runs
    # with serial; jobs 1503779 and 1514458 compared fewer cores or unbound runs.
    P = opts.max_cores
    if P % 8 or not opts.expect_hosts or P % opts.expect_hosts:
        raise SystemExit(f"FAIL: --suite bound needs --max-cores divisible by 8 and by --expect-hosts, got {P}")
    check(f"bound_{P}x1_traj", 20011, 10, 0.5, 8, P, 1, "dynamic", 64)
    check(f"bound_{P // 8}x8_traj", 20011, 10, 0.5, 8, P // 8, 8, "dynamic", 64)
    check(f"bound_{P // 4}x4_guided", 20003, 10, 0.5, 8, P // 4, 4, "guided", 0)
    check(f"bound_{P // 8}x8_accel", 20011, 1, 0.5, 8, P // 8, 8)
    check(f"bound_{P // 8}x8_theta0", 4001, 1, 0.0, 8, P // 8, 8)
    check(f"bound_{P}x1_zero_owned", 10, 10, 0.5, 8, P, 1)
else:
    check("1x1_static", 2000, 1, 0.5, 8, 1, 1)
    check("1x4_static", 2000, 1, 0.5, 8, 1, 4)
    check("1x4_dynamic1", 2000, 1, 0.5, 8, 1, 4, "dynamic", 1)
    check("2x1_static", 2000, 1, 0.5, 8, 2, 1)
    check("3x2_dynamic16", 2000, 1, 0.5, 8, 3, 2, "dynamic", 16)
    check("4x2_guided", 2000, 1, 0.5, 8, 4, 2, "guided", 0)
    check("theta0_3x2", 1500, 1, 0.0, 8, 3, 2)
    check("leaf1_2x2", 1500, 1, 0.7, 1, 2, 2)
    check("traj_1x1", 2000, 10, 0.5, 8, 1, 1)
    check("traj_1x4_dynamic", 2000, 10, 0.5, 8, 1, 4, "dynamic", 8)
    check("traj_3x2_static", 2001, 10, 0.5, 8, 3, 2)
    check("traj_4x2_guided", 2003, 10, 0.5, 8, 4, 2, "guided", 0)
    check("uneven_n7_4ranks", 7, 10, 0.5, 2, 4, 1)
    check("zero_owned_n3_5ranks", 3, 10, 0.5, 8, 5, 1)
    check("singleton_3ranks", 1, 5, 0.5, 8, 3, 1)

if opts.suite == "multinode":
    # Up to 16 cores over two nodes. With round-robin mapping (--map-by node),
    # each node carries ranks*threads/2 <= 8 threads. N is never divisible by
    # the rank count; the 16-rank N=10 case leaves six ranks owning nothing.
    check("mn_2x8_static", 20011, 1, 0.5, 8, 2, 8)
    check("mn_4x4_dynamic64", 20011, 10, 0.5, 8, 4, 4, "dynamic", 64)
    check("mn_8x2_guided", 20003, 10, 0.5, 8, 8, 2, "guided", 0)
    check("mn_16x1_static", 2003, 10, 0.5, 8, 16, 1)
    check("mn_6x2_dynamic1", 1001, 10, 0.7, 4, 6, 2, "dynamic", 1)
    check("mn_zero_owned_16x1", 10, 10, 0.5, 8, 16, 1)
    check("mn_theta0_4x4", 4001, 1, 0.0, 8, 4, 4)

# Race symptoms: repeated threaded runs must give bitwise-identical states.
repeat_dumps = []
for k in range(3):
    dump = work / f"repeat_{k}.txt"
    r = run(hybrid_cmd(2, ["--n", "3000", "--steps", "5", "--schedule", "dynamic", "--chunk", "1",
                           "--dump", str(dump)], 4), env=hybrid_env(4))
    repeat_dumps.append(dump.read_bytes() if r.returncode == 0 and dump.exists() else b"")
repeat_ok = all(repeat_dumps) and len(set(repeat_dumps)) == 1
print(f"hybrid repeat 3 runs 2x4 dynamic/1 bitwise_identical={repeat_ok} "
      f"{'PASS' if repeat_ok else 'FAIL'}")
if not repeat_ok:
    failures.append("repeatability")

# Invalid arguments: every rank must exit together with status 2, no hang.
bad = [["--n", "12abc"], ["--theta", "-1"], ["--schedule", "fifo"], ["--steps", "0"]]
for args in bad:
    r = run(hybrid_cmd(3, args), env=hybrid_env(1), timeout=60)
    ok = r.returncode == 2
    print(f"hybrid reject {' '.join(args)} exit={r.returncode} {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("reject " + " ".join(args))

# Negative control: the comparator must flag a 1e-9 relative perturbation.
ref = serial_reference(2000, 10, 0.5, 8)
tampered = [(i, list(r)) for i, r in ref]
tampered[123][1][0] *= 1.0 + 1e-9
control = compare(ref, tampered, ["pos"], TRAJ_TOL)
control_ok = not control["ok"]
print(f"hybrid negative_control position*(1+1e-9) detected={control_ok} "
      f"{'PASS' if control_ok else 'FAIL'}")
if not control_ok:
    failures.append("negative control")

sources = sorted(root.glob("common/*.hpp")) + [root / "serial/barnes_hut.cpp",
                                                root / "hybrid/bh_hybrid.cpp"]
evidence = {
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    **execution_context(),
    "platform": platform.platform(),
    "mpirun_flags": opts.mpirun_flags, "suite": opts.suite, "expect_hosts": opts.expect_hosts,
    "source_sha256": {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
                      for f in sources},
    "binary_sha256": {Path(b).name: hashlib.sha256(Path(b).read_bytes()).hexdigest()
                      for b in (serial, hybrid)},
    "tolerances": {"acceleration_1_step": ACC_TOL, "trajectory_10_steps": TRAJ_TOL},
    "cases": records, "repeat_bitwise_identical": repeat_ok,
    "negative_control_detected": control_ok, "failures": failures,
}
Path(opts.out).parent.mkdir(parents=True, exist_ok=True)
Path(opts.out).write_text(json.dumps(evidence, indent=2) + "\n")
if failures:
    raise SystemExit(f"FAIL: {len(failures)} hybrid check(s): {', '.join(failures)}")
print("PASS: hybrid gate (serial vs MPI+OpenMP, repeatability, argument rejection)")
