#!/usr/bin/env python3
"""Summarise one run directory from its raw records only.

A configuration is everything that could change a timing: program, problem and
numerical settings, ranks x threads, allocated cores, schedule and chunk, host
count, placement (launcher flags, OMP_PROC_BIND, OMP_PLACES), CPU model, source
tree and compile flags. Repetitions of one
configuration are summarised by median and min/max of the timed region and of
each phase (per-phase values are rank maxima, so they need not sum to wall).

Speedup uses serial Barnes-Hut with the same problem, CPU model, source tree
and compile flags in the same run directory: S = T_serial / T_parallel and
E = S / P with P the recorded allocated physical cores. Weak-scaling
efficiency needs a stated baseline and is computed by the plotting step.
Failed repetitions are listed, never silently dropped.

  analyze.py artifacts/cluster/runs/<RUN_ID>
writes <run>/summary.csv and <run>/summary.json.
  analyze.py RUN_DIR [RUN_DIR ...] --out DIR
combines several run directories (e.g. one study split over jobs) into
DIR/summary.{csv,json}; records keep their run directory in `runs`.
Skipped and timed-out runs appear under "failed" with their reasons.

GPU timing: a bh_gpu record that fails gpu_record.gpu_problems counts as failed even
if it says ok. A device row is compared only with the cpu row of the same problem,
threads, placement, hosts, binary, CPU model and source tree
(gpu_speedup_vs_cpu = T_cpu / T_device; repetition counts kept beside it).
It never gets a serial speedup: the CPU backend is the reference for this test.
"""
import csv
import json
from pathlib import Path
import statistics
import sys

from gpu_record import gpu_problems

PROBLEM = ("n", "theta", "dt", "steps", "warmup", "leaf_size", "seed", "softening")
PHASES = ("integrate_s", "exchange_s", "build_s", "traverse_s")
GPU_PHASES = ("flatten_s", "map_nodes_s", "to_device_s", "from_device_s")


def build_of(rec):
    """Where and how the binary was built; serial and hybrid must agree to be compared."""
    r = rec["result"]
    return (rec.get("cpu_model"), rec.get("source_tree_sha256"), r.get("cxxflags"))


def placement_of(rec):
    """Launcher words before the program (binding/mapping flags) plus OpenMP placement."""
    cmd = rec["command"]
    prog = next(i for i, w in enumerate(cmd) if Path(w).name in ("bh_serial", "bh_hybrid", "bh_gpu"))
    env = rec.get("env", {})
    return (" ".join(cmd[:prog]), env.get("OMP_PROC_BIND") or "", env.get("OMP_PLACES") or "")


def config_of(rec):
    r = rec["result"]
    return (r["program"], *(r[k] for k in PROBLEM), r["ranks"], r["threads"], rec["allocated_cores"],
            r.get("schedule_requested", "-"), r.get("chunk_requested", 0), len(r.get("hosts", [])),
            *placement_of(rec), *build_of(rec), r.get("backend", "-"), rec.get("binary_sha256"),
            tuple(r.get("hosts", [])) if r["program"] == "bh_gpu" else ())


def main(run_dirs, out_dir=None):
    runs = [Path(d) for d in run_dirs]
    run = Path(out_dir) if out_dir else runs[0]
    run.mkdir(parents=True, exist_ok=True)
    records = []
    for d in runs:
        for p in sorted((d / "raw").glob("*.json")):
            rec = json.loads(p.read_text())
            rec["_run"] = d.name
            records.append(rec)
    if not records:
        sys.exit(f"no raw records in {[str(d / 'raw') for d in runs]}")
    for rec in records:
        res = rec.get("result") or {}
        if rec.get("ok") and res.get("program") == "bh_gpu":
            refused = gpu_problems(res, rec.get("command", []), rec.get("env"))
            if refused:
                rec["ok"], rec["problems"] = False, ["refused by analyze: " + p for p in refused]
    failed = [r for r in records if not r.get("ok")]
    groups = {}
    for rec in records:
        if rec.get("ok"):
            groups.setdefault(config_of(rec), []).append(rec)

    rows = []
    for recs in groups.values():
        res = [r["result"] for r in recs]
        first, walls = res[0], [x["wall_s"] for x in res]
        cpu, tree, flags = build_of(recs[0])
        row = {"program": first["program"], **{k: first[k] for k in PROBLEM},
               "ranks": first["ranks"], "threads": first["threads"],
               "cores": recs[0]["allocated_cores"], "hosts": len(first.get("hosts", [])),
               "schedule": first.get("schedule_requested", "-"), "chunk": first.get("chunk_requested", 0),
               "reps": len(recs), "wall_median_s": statistics.median(walls),
               "wall_min_s": min(walls), "wall_max_s": max(walls)}
        for ph in PHASES:
            row[f"{ph[:-2]}_median_s"] = statistics.median(x[ph] for x in res)
        if all(x.get("traverse_mean_s") for x in res):
            row["traverse_imbalance"] = statistics.median(x["traverse_s"] / x["traverse_mean_s"] for x in res)
        if first["program"] == "bh_gpu":
            for ph in GPU_PHASES:
                row[f"{ph[:-2]}_median_s"] = statistics.median(x[ph] for x in res)
            row.update(backend=first["backend"], binary_sha256=recs[0].get("binary_sha256"),
                       gpu_hardware=recs[0].get("gpu_hardware"), host_names=first.get("hosts"),
                       distinct_cpus=first.get("distinct_cpus"), proc_bind=first.get("proc_bind"))
        launcher, bind, places = placement_of(recs[0])
        row.update(launcher=launcher, omp_proc_bind=bind, omp_places=places,
                   cpu_model=cpu, source_tree_sha256=tree, cxxflags=flags,
                   cases=",".join(sorted({r["case"] for r in recs})),
                   studies=",".join(sorted({r.get("tags", {}).get("study", "") for r in recs} - {""})),
                   runs=",".join(sorted({r["_run"] for r in recs})))
        if all(x.get("peak_rss_mib_rank_max") is not None for x in res):
            row["peak_rss_mib_rank_max"] = max(x["peak_rss_mib_rank_max"] for x in res)
            row["peak_rss_mib_rank_sum"] = max(x.get("peak_rss_mib_rank_sum", 0) for x in res)
        rows.append(row)

    # One serial baseline per (problem, build); two would make speedup ambiguous.
    serial = {}
    for r in rows:
        if r["program"] == "bh_serial":
            k = tuple(r[x] for x in PROBLEM) + (r["cpu_model"], r["source_tree_sha256"], r["cxxflags"])
            if k in serial:
                sys.exit(f"ambiguous serial baseline for {k}; split the run directory")
            serial[k] = r
    for r in rows:
        base = serial.get(tuple(r[x] for x in PROBLEM) + (r["cpu_model"], r["source_tree_sha256"], r["cxxflags"]))
        if base and r["program"] not in ("bh_serial", "bh_gpu"):
            r["speedup_vs_serial"] = base["wall_median_s"] / r["wall_median_s"]
            r["efficiency"] = r["speedup_vs_serial"] / r["cores"]

    pair_key = lambda r: tuple(r[x] for x in PROBLEM) + (r["threads"], r["cpu_model"], r["source_tree_sha256"],
                                                         r["cxxflags"], r["binary_sha256"], r["omp_proc_bind"], r["omp_places"],
                                                         tuple(r["host_names"]))
    cpu_rows = {pair_key(r): r for r in rows if r.get("backend") == "cpu"}
    for r in rows:
        if r.get("backend") == "device" and pair_key(r) in cpu_rows:
            r["gpu_speedup_vs_cpu"] = cpu_rows[pair_key(r)]["wall_median_s"] / r["wall_median_s"]
            r["cpu_reps"] = cpu_rows[pair_key(r)]["reps"]

    rows.sort(key=lambda r: (r["n"], r["program"], r.get("backend", ""), r["cores"], r["ranks"], r["schedule"], r["chunk"]))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(run / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (run / "summary.json").write_text(json.dumps(
        {"run": run.name, "inputs": [str(d) for d in runs], "records": len(records),
         "failed": [{"case": r["case"], "rep": r["rep"], "run": r["_run"], "problems": r.get("problems"),
                     "skipped": bool(r.get("skipped")), "timed_out": bool(r.get("timed_out"))}
                    for r in failed],
         "rows": rows}, indent=2) + "\n")
    print(f"{run.name}: {len(records)} records, {len(failed)} failed, {len(rows)} configurations")
    for r in rows:
        sp = f" speedup={r['speedup_vs_serial']:.2f} eff={r['efficiency']:.2f}" if "speedup_vs_serial" in r else ""
        if "gpu_speedup_vs_cpu" in r:
            sp = f" cpu/device={r['gpu_speedup_vs_cpu']:.2f}"
        print(f"  {r['program']:9s} {r.get('backend', '')} n={r['n']} {r['ranks']}x{r['threads']} P={r['cores']} "
              f"{r['schedule']}/{r['chunk']} reps={r['reps']} wall_med={r['wall_median_s']:.4f}s "
              f"[{r['wall_min_s']:.4f},{r['wall_max_s']:.4f}]{sp}")


if __name__ == "__main__":
    args = sys.argv[1:]
    out = None
    if "--out" in args:
        i = args.index("--out")
        out = args[i + 1]
        del args[i:i + 2]
    if not args or (len(args) > 1 and out is None):
        sys.exit(__doc__)
    main(args, out)
