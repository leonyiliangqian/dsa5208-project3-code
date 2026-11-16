#!/usr/bin/env python3
"""Run every case of an experiment manifest through run_case.py.

A manifest (manifests/*.json) lists cases; any field given as a list is
swept (Cartesian product). Programs:
  serial    bh_serial, pinned with taskset when --first-cpu is given
  hybrid    bh_hybrid on `nodes` nodes, `ranks` in total, `threads` per rank,
            placed with --map-by ppr:<ranks/nodes>:node:PE=<threads> --bind-to core
  validate  bh_serial --validate (accuracy against direct summation)
  gpu       bh_gpu, `threads` host threads, one `backend`: device (mandatory offload,
            --require-device) or cpu. Each workload must be planned on both backends.
Repetitions are interleaved: rep 1 of every case, then rep 2, ...

Each run has its own timeout (case `timeout_s`; run_case.py stops the whole
process group). With --deadline, a run is started only if its timeout plus
grace and a margin still fit before the deadline; otherwise it is recorded as
skipped, never silently dropped. --plan-only prints the expanded runs and
writes nothing but plan.json.

  run_manifest.py MANIFEST --out RUN_DIR --bin BIN [--mpirun 'mpirun ...']
      [--cores-per-node C] [--nodes K] [--first-cpu CPU] [--deadline EPOCH]
      [--scale S] [--plan-only]

--scale divides every N (local pipeline tests only; recorded as a tag).
Exit status: 0 if every run was ok, 1 otherwise.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
GRACE_S = 15
MARGIN_S = 30
SWEEPABLE = ("n", "theta", "steps", "warmup", "leaf_size", "seed", "ranks", "threads",
             "nodes", "schedule", "chunk", "backend")


def expand(manifest):
    defaults = manifest.get("defaults", {})
    out = []
    for case in manifest["cases"]:
        c = {**defaults, **case}
        keys = [k for k in SWEEPABLE if isinstance(c.get(k), list)]
        for values in itertools.product(*(c[k] for k in keys)):
            point = dict(c, **dict(zip(keys, values)))
            point["name"] = case_name(point)
            out.append(point)
    names = [p["name"] for p in out]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        sys.exit(f"manifest expands to duplicate case names: {sorted(dup)}")
    return out


def case_name(p):
    prog = p["program"]
    parts = [p["study"], prog]
    if prog == "hybrid":
        parts.append(f"{p['ranks']}x{p['threads']}")
        if p.get("nodes", 1) > 1:
            parts.append(f"{p['nodes']}nodes")
        parts.append(f"{p['schedule']}{p['chunk']}")
    if prog == "gpu":
        parts += [p["backend"], f"t{p['threads']}"]
    parts += [f"n{p['n']}", f"th{p['theta']}", f"L{p['leaf_size']}"]
    return "_".join(str(x) for x in parts).replace(".", "p")


def program_args(p, scale):
    n = max(1, p["n"] // scale)
    args = ["--n", str(n), "--theta", str(p["theta"]), "--leaf-size", str(p["leaf_size"])]
    if p["program"] == "validate":
        return ["--validate"] + args
    args += ["--steps", str(p["steps"]), "--warmup", str(p["warmup"]), "--seed", str(p["seed"])]
    if p["program"] == "hybrid":
        args += ["--schedule", p["schedule"], "--chunk", str(p["chunk"])]
    if p["program"] == "gpu":
        args += ["--backend", p["backend"]] + (["--require-device"] if p["backend"] == "device" else [])
    return args


def check_gpu_pairs(points):
    """A device timing means nothing without the same workload on the CPU backend."""
    pairs = {}
    for p in points:
        if p["program"] != "gpu":
            continue
        if p.get("backend") not in ("device", "cpu"):
            sys.exit(f"{p['name']}: backend must be device or cpu")
        key = tuple(p[k] for k in ("n", "theta", "steps", "warmup", "leaf_size", "seed", "threads", "reps"))
        pairs.setdefault(key, set()).add(p["backend"])
    unpaired = [k for k, b in pairs.items() if b != {"device", "cpu"}]
    if unpaired:
        sys.exit(f"gpu workloads not planned on both backends: {unpaired}")


def plan_runs(points, a):
    check_gpu_pairs(points)
    runs = []
    reps = max(p.get("reps", 1) for p in points) if points else 0
    for rep in range(1, reps + 1):
        for p in points:
            if rep > p.get("reps", 1):
                continue
            prog = p["program"]
            nodes = p.get("nodes", 1)
            if prog == "hybrid":
                r, t = p["ranks"], p["threads"]
                if r % nodes:
                    sys.exit(f"{p['name']}: {r} ranks do not divide over {nodes} nodes")
                if nodes > a.nodes:
                    sys.exit(f"{p['name']}: needs {nodes} nodes, job has {a.nodes}")
                if (r // nodes) * t > a.cores_per_node:
                    sys.exit(f"{p['name']}: {r // nodes} x {t} per node exceeds {a.cores_per_node} cores")
                launcher = (f"{a.mpirun} -np {r} --map-by ppr:{r // nodes}:node:PE={t} --bind-to core "
                            "-x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES")
                env = {"OMP_NUM_THREADS": str(t), "OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}
                cores, binary = r * t, "bh_hybrid"
            elif prog == "gpu":
                t = p["threads"]
                if t > a.cores_per_node:
                    sys.exit(f"{p['name']}: {t} threads exceed {a.cores_per_node} cores")
                launcher = ""
                env = {"OMP_NUM_THREADS": str(t), "OMP_PROC_BIND": "close", "OMP_PLACES": "cores",
                       "OMP_SCHEDULE": "dynamic,64"}
                if p["backend"] == "device":
                    env["OMP_TARGET_OFFLOAD"] = "MANDATORY"
                cores, binary = t, "bh_gpu"
            else:
                launcher = f"taskset -c {a.first_cpu}" if a.first_cpu is not None else ""
                env, cores, binary = {"OMP_NUM_THREADS": "1"}, 1, "bh_serial"
            runs.append({"case": p["name"], "rep": rep, "study": p["study"], "program": prog,
                         "launcher": launcher, "env": env, "cores": cores,
                         "argv": [str(Path(a.bin) / binary)] + program_args(p, a.scale),
                         "timeout_s": int(p["timeout_s"])})
    return runs


def record_skip(out, run, reason, tags):
    raw = Path(out) / ("validate" if run["program"] == "validate" else "raw")
    raw.mkdir(parents=True, exist_ok=True)
    rec = {"case": run["case"], "rep": run["rep"], "ok": False, "skipped": True,
           "problems": [f"skipped: {reason}"], "command": shlex.split(run["launcher"]) + run["argv"],
           "allocated_cores": run["cores"], "timeout_s": run["timeout_s"], "tags": tags, "result": None}
    (raw / f"{run['case']}_rep{run['rep']}.json").write_text(json.dumps(rec, indent=2) + "\n")


CURRENT = {"pid": None}


def stop(signum, _frame):
    """Stopped by the job: stop the run in progress (its own process group), then exit."""
    pid = CURRENT["pid"]
    if pid:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    sys.exit(128 + signum)


def run_validate(out, run, tags):
    """Accuracy check: output kept verbatim with exit status and timing."""
    d = Path(out) / "validate"
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{run['case']}_rep{run['rep']}"
    argv = shlex.split(run["launcher"]) + run["argv"]
    t0 = time.time()
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    CURRENT["pid"] = p.pid
    timed_out = False
    try:
        text, _ = p.communicate(timeout=run["timeout_s"])
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(p.pid, 15)
        try:
            text, _ = p.communicate(timeout=GRACE_S)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 9)
            text, _ = p.communicate()
    (d / f"{stem}.txt").write_text(text or "")
    ok = p.returncode == 0 and not timed_out
    (d / f"{stem}.json").write_text(json.dumps(
        {"case": run["case"], "rep": run["rep"], "ok": ok, "returncode": p.returncode,
         "timed_out": timed_out, "elapsed_s": round(time.time() - t0, 3), "command": argv,
         "timeout_s": run["timeout_s"], "tags": tags}, indent=2) + "\n")
    print(f"{stem}: validate exit={p.returncode} ok={ok}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--mpirun", default="mpirun")
    ap.add_argument("--cores-per-node", type=int, default=os.cpu_count())
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--first-cpu", type=int)
    ap.add_argument("--deadline", type=float, help="epoch seconds; no run starts that could end after it")
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    mpath = Path(a.manifest)
    mbytes = mpath.read_bytes()
    manifest = json.loads(mbytes)
    msha = hashlib.sha256(mbytes).hexdigest()
    runs = plan_runs(expand(manifest), a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(
        {"manifest": str(mpath), "manifest_id": manifest["id"], "manifest_status": manifest.get("status"),
         "manifest_sha256": msha, "scale": a.scale, "nodes": a.nodes, "cores_per_node": a.cores_per_node,
         "deadline": a.deadline, "runs": runs}, indent=2) + "\n")
    total = sum(r["timeout_s"] + GRACE_S for r in runs)
    print(f"manifest {manifest['id']} ({manifest.get('status')}) sha256={msha[:12]} runs={len(runs)} "
          f"worst_case_s={total} scale={a.scale}")
    if a.plan_only:
        for r in runs:
            env = " ".join(f"{k}={v}" for k, v in r["env"].items())
            print(f"  {r['case']} rep{r['rep']} P={r['cores']} timeout={r['timeout_s']}s: "
                  f"{env} {r['launcher']} {' '.join(r['argv'])}".replace("  ", " "))
        return 0
    (out / "manifest.json").write_bytes(mbytes)

    all_ok = True
    for r in runs:
        tags = {"study": r["study"], "manifest_id": manifest["id"], "manifest_sha256": msha,
                "manifest_status": manifest.get("status", ""), "scale": str(a.scale)}
        if a.deadline is not None and time.time() + r["timeout_s"] + GRACE_S + MARGIN_S > a.deadline:
            record_skip(out, r, "not enough time before the job deadline", tags)
            print(f"{r['case']}_rep{r['rep']}: SKIP (deadline)")
            all_ok = False
            continue
        if r["program"] == "validate":
            all_ok &= run_validate(out, r, tags)
            continue
        cmd = [sys.executable, str(HERE / "run_case.py"), "--out", str(out), "--case", r["case"],
               "--rep", str(r["rep"]), "--allocated-cores", str(r["cores"]),
               "--timeout", str(r["timeout_s"]), "--grace", str(GRACE_S), "--launcher", r["launcher"]]
        cmd += [x for k, v in tags.items() for x in ("--tag", f"{k}={v}")]
        cmd += ["--", *r["argv"]]
        child = subprocess.Popen(cmd, env={**os.environ, **r["env"]}, start_new_session=True)
        CURRENT["pid"] = child.pid
        rc = child.wait()
        CURRENT["pid"] = None
        all_ok &= rc == 0
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
