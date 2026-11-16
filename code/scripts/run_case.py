#!/usr/bin/env python3
"""Run one measured configuration and keep everything needed to trust it later.

Each call writes <out>/raw/<case>_rep<k>.json (never overwriting) plus the raw
stdout/stderr, recording the exact command, scheduler job ID, host, CPU model,
source revision and dirty-tree hash, binary hash, compiler/MPI versions, OpenMP
environment and the program's own result line. A failed run is recorded with
its exit code rather than skipped, so failures stay visible in the evidence.

Scheduler-agnostic: a Slurm or PBS job script calls this once per repetition.

  run_case.py --out artifacts/cluster/runs/<RUN_ID> --case strong_8x2 --rep 1 \\
      --launcher "mpirun -np 8 --map-by ppr:4:socket:PE=2 --bind-to core" \\
      -- bin/linux/bh_hybrid --n 1000000 --steps 5 --schedule dynamic --chunk 64
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import signal
import socket
import subprocess
import sys

from gpu_record import gpu_problems

root = Path(__file__).resolve().parents[1]


def sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return (r.stdout or r.stderr).strip()
    except Exception as e:  # provenance is best-effort; never fail the run on it
        return f"unavailable: {e}"


def source_state():
    # Compute nodes may lack git; submit.sh passes the login node's HEAD instead.
    rev = (sh(f"git -C {shlex.quote(str(root))} rev-parse HEAD 2>/dev/null")
           or os.environ.get("SOURCE_GIT_REV") or "no-git")
    # Hash the source files themselves so an uncommitted edit is still traceable.
    files = sorted(list(root.glob("common/*.hpp")) + list(root.glob("serial/*.cpp")) +
                   list(root.glob("hybrid/*.cpp")) + list(root.glob("gpu/*.cpp")) + [root / "Makefile"])
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    # Tracked files only: run outputs are untracked by design and must not mark
    # an unmodified source tree as dirty.
    dirty = sh(f"git -C {shlex.quote(str(root))} status --porcelain --untracked-files=no -- . 2>/dev/null")
    return {"git_rev": rev, "source_tree_sha256": h.hexdigest(), "git_dirty": bool(dirty)}


def run_bounded(argv, timeout, grace):
    """Run argv in its own process group. On timeout the whole group (launcher,
    ranks, helpers) gets SIGTERM, then SIGKILL after grace seconds, so mpirun
    can still shut down its remote daemons. Returns (code, stdout, stderr,
    timed_out, signal_used)."""
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         start_new_session=True)

    def forward(signum, _frame):
        # The job (e.g. a gate's time limit) is stopping this recorder: pass
        # the signal to the run's own process group so nothing is orphaned.
        try:
            os.killpg(p.pid, signum)
        except ProcessLookupError:
            pass
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err, False, None
    except subprocess.TimeoutExpired:
        pass
    used = "SIGTERM"
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        out, err = p.communicate(timeout=grace)
    except subprocess.TimeoutExpired:
        used = "SIGKILL"
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = p.communicate()
    return 124, out or "", (err or "") + f"\nTIMEOUT after {timeout}s ({used})\n", True, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run directory, e.g. artifacts/cluster/runs/<RUN_ID>")
    ap.add_argument("--case", required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--launcher", default="", help="MPI launcher prefix; empty for serial")
    ap.add_argument("--allocated-cores", type=int, required=True,
                    help="physical cores allocated to this run; the P in efficiency S/P")
    ap.add_argument("--timeout", type=int, default=7200,
                    help="seconds before the run's process group gets TERM (then KILL after --grace)")
    ap.add_argument("--grace", type=int, default=15)
    ap.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE",
                    help="metadata recorded with the run (e.g. study=strong, manifest=<file>)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        sys.exit("no program given after --")

    raw = Path(a.out) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    stem = f"{a.case}_rep{a.rep}"
    record_path = raw / f"{stem}.json"
    if record_path.exists():
        sys.exit(f"refusing to overwrite {record_path}")

    tags = dict(t.split("=", 1) for t in a.tag if "=" in t)
    full = shlex.split(a.launcher) + cmd
    started = datetime.datetime.now(datetime.timezone.utc)
    code, out, err, timed_out, kill_signal = run_bounded(full, a.timeout, a.grace)
    (raw / f"{stem}.stdout").write_text(out)
    (raw / f"{stem}.stderr").write_text(err)

    result = None
    for line in (out if isinstance(out, str) else "").splitlines():
        if line.startswith("result "):
            result = json.loads(line[len("result "):])
    binary = Path(cmd[0])
    # A run is usable only if it exited cleanly, reported a result, kept its
    # replicated state consistent and actually ran the requested thread count.
    problems = []
    if timed_out:
        problems.append(f"timed out after {a.timeout}s ({kill_signal})")
    if code != 0:
        problems.append(f"exit {code}")
    if result is None:
        problems.append("no result line")
    else:
        if result.get("replicated", True) is not True:
            problems.append("replication check failed")
        if result.get("threads_effective") != result.get("threads"):
            problems.append("effective threads differ from requested")
        if result["ranks"] * result["threads"] > a.allocated_cores:
            problems.append("ranks x threads exceeds allocated cores (oversubscribed)")
        if result.get("program") == "bh_gpu":
            problems += gpu_problems(result, full, os.environ)
    record = {
        "case": a.case, "rep": a.rep, "returncode": code,
        "ok": not problems, "problems": problems,
        "started_utc": started.isoformat(),
        "local_time": datetime.datetime.now().astimezone().isoformat(),
        "command": full, "allocated_cores": a.allocated_cores,
        "timeout_s": a.timeout, "timed_out": timed_out, "kill_signal": kill_signal,
        "elapsed_s": round((datetime.datetime.now(datetime.timezone.utc) - started).total_seconds(), 3),
        "tags": tags, "run_marker": os.environ.get("P3_RUN_MARKER"),
        "scheduler_job_id": os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID"),
        "scheduler_nodelist": os.environ.get("SLURM_JOB_NODELIST") or sh("cat $PBS_NODEFILE 2>/dev/null | sort -u | paste -sd,"),
        "launch_host": socket.gethostname(), "platform": platform.platform(),
        "cpu_model": sh("lscpu 2>/dev/null | grep -m1 'Model name' | cut -d: -f2 || sysctl -n machdep.cpu.brand_string"),
        "binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest() if binary.exists() else None,
        "compiler": sh("${CXX:-c++} --version | head -1"),
        "mpicxx": sh("mpicxx --version 2>/dev/null | head -1"),
        "mpi": sh("mpirun --version 2>&1 | head -1"),
        "modules": os.environ.get("LOADEDMODULES", ""),
        "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES",
                                               "OMP_SCHEDULE", "OMP_TARGET_OFFLOAD", "CUDA_VISIBLE_DEVICES",
                                               "SLURM_CPUS_PER_TASK", "SLURM_NTASKS", "PBS_NCPUS", "CXXFLAGS")},
        "gpu_hardware": sh("nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader")
        if binary.name == "bh_gpu" else None,
        **source_state(),
        "result": result,
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{stem}: exit={code} ok={record['ok']} wall_s={result.get('wall_s') if result else None}"
          + (f" problems={problems}" if problems else ""))
    sys.exit(0 if record["ok"] else 1)


if __name__ == "__main__":
    main()
