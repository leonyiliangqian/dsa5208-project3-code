#!/bin/bash
# G2 pipeline on tiny inputs: manifest -> run_manifest.py -> analyze.py -> report_figures.py,
# then the whole job script. No GPU here, so a stub bh_gpu stands in for the device paths, and
# the real program must refuse them. Everything is a pipeline test; no timing is a measurement,
# and nothing here shows that the job works on a real GPU.
#   docker/run.sh bash tests/gpu_timing_e2e.sh
set -u
cd "$(dirname "$0")/.."
code=$PWD
OUT=$(mktemp -d)
trap 'rm -rf "$OUT" bin/g2e2e bin/vanda-g2stub bin/vanda-g2real bin/vanda-g2cold' EXIT
pass=0 fail=0
check() {
    local what=$1; shift
    if "$@" >/dev/null 2>&1; then echo "PASS $what"; pass=$((pass + 1))
    else echo "FAIL $what"; fail=$((fail + 1)); fi
}
expect_fail() { ! "$@"; }
M=manifests/local_gpu_timing.json
J() { python3 -c "import json,glob,sys; $1" ; }

# A stand-in bh_gpu. Timings are fixed so the ratio can be checked exactly.
mkdir -p "$OUT/stub"
cat > "$OUT/stub/bh_gpu" <<'STUB'
#!/usr/bin/env python3
import json, os, sys, time
mode = os.environ.get("STUB_MODE", "good")
a = sys.argv[1:]
get = lambda k, d: a[a.index(k) + 1] if k in a else d
backend = get("--backend", "device")
real = backend == "device" and mode != "fallback"
print(f"device_check num_devices=1 kernel_on_device={int(real)} separate_memory={int(real)}")
if "--device-check" in a: sys.exit(0)
if int(get("--n", 0)) >= 1000000: time.sleep(3)  # long enough for gpu_busy.sh to sample
report = "device" if mode == "cpu_claims_device" else backend
t = int(os.environ.get("OMP_NUM_THREADS", "1"))
print("result " + json.dumps({"program": "bh_gpu", "backend": report, "kernel_on_device": real or mode == "cpu_claims_device",
    "separate_memory": real or mode == "cpu_claims_device", "num_devices": 1, "n": int(get("--n", 1000)), "theta": float(get("--theta", .5)),
    "dt": 0.001, "steps": int(get("--steps", 5)), "warmup": int(get("--warmup", 0)), "leaf_size": int(get("--leaf-size", 8)),
    "seed": int(get("--seed", 42)), "softening": 0.025, "ranks": 1, "threads": t, "threads_effective": 1 if mode == "small_team" else t, "distinct_cpus": 1 if mode == "stacked" else t,
    "proc_bind": "false" if mode == "unbound" else "close", "schedule_effective": "static" if mode == "other_schedule" else "dynamic",
    "chunk_effective": 64, "hosts": ["stub"], "nodes": 10,
    "setup_s": 0.1, "wall_s": 2.0 if backend == "device" else 6.0, "integrate_s": 0.1, "exchange_s": 0, "build_s": 0.2,
    "traverse_s": 1.0 if backend == "device" else 5.0, "flatten_s": 0.25, "map_nodes_s": 0.0625 if backend == "device" else 0, "to_device_s": 0.125 if backend == "device" else 0,
    "from_device_s": 0.125 if backend == "device" else 0, "peak_rss_mib_rank_max": 1.0, "peak_rss_mib_rank_sum": 1.0,
    "compiler": "stub", "cxxflags": "-O3"}))
STUB
chmod +x "$OUT/stub/bh_gpu"
runm() { python3 scripts/run_manifest.py "$M" --bin "$OUT/stub" --cores-per-node 4 --out "$1"; }

# 1. Planning: pairs enforced, device runs forced to require a device.
python3 scripts/run_manifest.py $M --plan-only --bin b --cores-per-node 4 --out "$OUT/plan" > "$OUT/plan.txt"
check "plan: 12 runs, 3 repetitions of 2 sizes on 2 backends" J "
d=json.load(open('$OUT/plan/plan.json'))['runs']; sys.exit(not (len(d)==12 and sorted({r['rep'] for r in d})==[1,2,3]))"
check "plan: every device run requires a device and mandatory offload; no cpu run does" J "
d=json.load(open('$OUT/plan/plan.json'))['runs']
dev=[r for r in d if '--backend' in r['argv'] and r['argv'][r['argv'].index('--backend')+1]=='device']
cpu=[r for r in d if r not in dev]
sys.exit(not (len(dev)==len(cpu)==6 and all('--require-device' in r['argv'] and r['env']['OMP_TARGET_OFFLOAD']=='MANDATORY' for r in dev)
  and not any('--require-device' in r['argv'] or 'OMP_TARGET_OFFLOAD' in r['env'] for r in cpu)))"
check "plan: both backends get the same workload arguments and timeout" J "
d=json.load(open('$OUT/plan/plan.json'))['runs']
key=lambda r:[x for x in r['argv'][1:] if x not in ('device','cpu','--require-device')]
sys.exit(not all(key(d[i])==key(d[i+1]) and d[i]['timeout_s']==d[i+1]['timeout_s'] and d[i]['env']['OMP_NUM_THREADS']==d[i+1]['env']['OMP_NUM_THREADS'] for i in range(0,12,2)))"
python3 - "$OUT" <<'PY'
import json, sys
m = json.load(open("manifests/local_gpu_timing.json")); m["defaults"]["backend"] = ["device"]
json.dump(m, open(sys.argv[1] + "/nopair.json", "w"))
m["defaults"]["backend"] = ["device", "fpga"]
json.dump(m, open(sys.argv[1] + "/bad.json", "w"))
PY
check "plan: refuses device without its cpu partner" expect_fail python3 scripts/run_manifest.py "$OUT/nopair.json" --plan-only --bin b --cores-per-node 4 --out "$OUT/x"
check "plan: refuses an unknown backend" expect_fail python3 scripts/run_manifest.py "$OUT/bad.json" --plan-only --bin b --cores-per-node 4 --out "$OUT/x"
python3 scripts/run_manifest.py manifests/gpu_timing.json --plan-only --bin b --cores-per-node 36 --out "$OUT/plan2" > "$OUT/plan2.txt"
check "frozen manifest: 24 runs, worst case under the 3258 s deadline budget" J "
d=json.load(open('$OUT/plan2/plan.json')); s=sum(r['timeout_s']+15 for r in d['runs'])
sys.exit(not (len(d['runs'])==24 and s<=3258-300 and d['manifest_status'].startswith('frozen')))"

# 2. Stub on a working device: pairing, ratio, phases, provenance, figures.
runm "$OUT/good" > "$OUT/good.log" 2>&1
check "good: runner exit 0" test $? -eq 0
python3 scripts/analyze.py "$OUT/good" > "$OUT/good_an.log" 2>&1
check "good: cpu/device ratio is exactly 3 for both sizes, 3 repetitions each" J "
r=json.load(open('$OUT/good/summary.json'))['rows']; dv=[x for x in r if x['backend']=='device']
sys.exit(not (len(r)==4 and len(dv)==2 and all(x['gpu_speedup_vs_cpu']==3 and x['reps']==3 for x in dv)))"
check "good: both backends run the same team, placement and recorded schedule" J "
r=[json.load(open(f)) for f in glob.glob('$OUT/good/raw/*.json')]
sys.exit(not all(x['env']['OMP_NUM_THREADS']=='2' and x['env']['OMP_PROC_BIND']=='close' and x['env']['OMP_PLACES']=='cores' and x['env']['OMP_SCHEDULE']=='dynamic,64' for x in r))"
check "good: phases kept, no serial speedup" J "
r=json.load(open('$OUT/good/summary.json'))['rows']; x=[y for y in r if y['backend']=='device'][0]
sys.exit(not (x['flatten_median_s']==.25 and x['to_device_median_s']==.125 and x['map_nodes_median_s']==.0625 and x['cpu_reps']==3 and x['traverse_median_s']==1 and not any('speedup_vs_serial' in y for y in r)))"
check "good: records carry revision, binary hash, offload setting and GPU field" J "
r=[json.load(open(f)) for f in glob.glob('$OUT/good/raw/*.json')]
dev=[x for x in r if x['result']['backend']=='device']
sys.exit(not (len(r)==12 and all(x['git_rev'] and x['binary_sha256'] and 'gpu_hardware' in x and x['tags']['manifest_id']=='local-gpu-timing-v1' for x in r)
  and all(x['env']['OMP_TARGET_OFFLOAD']=='MANDATORY' for x in dev)))"
python3 scripts/report_figures.py "$OUT/good/summary.json" --allow-local --out "$OUT/fig" > "$OUT/fig.log" 2>&1
check "figures: runtime figure and table written" test -s "$OUT/fig/fig_gpu_runtime.tex" -a -s "$OUT/fig/tab_gpu.tex"
check "figures: log axes on both" grep -q 'xmode=log.*ymode=log' "$OUT/fig/fig_gpu_runtime.tex"
check "figures: LOCAL banner in caption" grep -q 'LOCAL PIPELINE TEST' "$OUT/fig/tab_gpu.tex"
check "figures: refuse local input for the report" expect_fail python3 scripts/report_figures.py "$OUT/good/summary.json"

# 3. Fallback and false claims: recorded as failures, never timed.
STUB_MODE=fallback runm "$OUT/fb" > "$OUT/fb.log" 2>&1
check "fallback: runner reports failure" test $? -eq 1
check "fallback: 6 device records refused with the reason" J "
r=[json.load(open(f)) for f in glob.glob('$OUT/fb/raw/*.json')]; bad=[x for x in r if not x['ok']]
sys.exit(not (len(bad)==6 and all(any('device fallback' in p for p in x['problems']) for x in bad)))"
python3 scripts/analyze.py "$OUT/fb" > /dev/null 2>&1
check "fallback: no device row and no ratio" J "
d=json.load(open('$OUT/fb/summary.json')); sys.exit(not (len(d['failed'])==6 and all(x['backend']=='cpu' and 'gpu_speedup_vs_cpu' not in x for x in d['rows'])))"
python3 scripts/report_figures.py "$OUT/fb/summary.json" --allow-local --out "$OUT/figfb" > /dev/null 2>&1
check "fallback: no GPU figure" test ! -e "$OUT/figfb/fig_gpu_runtime.tex"
STUB_MODE=cpu_claims_device runm "$OUT/claim" > /dev/null 2>&1
check "cpu run reporting a device is refused" J "
r=[json.load(open(f)) for f in glob.glob('$OUT/claim/raw/*cpu*.json')]
sys.exit(not (len(r)==6 and not any(x['ok'] for x in r)))"

for mode in small_team stacked unbound; do
    STUB_MODE=$mode runm "$OUT/$mode" > /dev/null 2>&1
    check "team fault $mode: refused on both backends" J "
r=[json.load(open(f)) for f in glob.glob('$OUT/$mode/raw/*.json')]; sys.exit(not (len(r)==12 and not any(x['ok'] for x in r)))"
done
STUB_MODE=other_schedule runm "$OUT/other" > /dev/null 2>&1
check "cpu kernel on another schedule: refused" J "
r=[json.load(open(f)) for f in glob.glob('$OUT/other/raw/*.json')]
sys.exit(not (all(not x['ok'] for x in r if '_cpu_' in x['case']) and all(x['ok'] for x in r if '_device_' in x['case'])))"

# 4. Analysis trusts no stored ok, and pairs only the same workload and binary.
cp -r "$OUT/good" "$OUT/tamper"
python3 - "$OUT/tamper" <<'PY'
import glob, json, sys
for f in glob.glob(sys.argv[1] + "/raw/*device*.json"):
    r = json.load(open(f)); r["result"]["kernel_on_device"] = False; json.dump(r, open(f, "w"))
PY
python3 scripts/analyze.py "$OUT/tamper" > /dev/null 2>&1
check "tampered ok record with a fallback is refused by analyze" J "
d=json.load(open('$OUT/tamper/summary.json'))
sys.exit(not (len(d['failed'])==6 and all('refused by analyze' in x['problems'][0] for x in d['failed']) and all(x['backend']=='cpu' for x in d['rows'])))"
cp -r "$OUT/good" "$OUT/diffn"; cp -r "$OUT/good" "$OUT/diffbin"; cp -r "$OUT/good" "$OUT/diffhost"; cp -r "$OUT/good" "$OUT/onefail"
python3 - "$OUT" <<'PY'
import glob, json, sys
for f in glob.glob(sys.argv[1] + "/diffn/raw/*_cpu_*.json"):
    r = json.load(open(f)); r["result"]["steps"] += 1; json.dump(r, open(f, "w"))
for f in glob.glob(sys.argv[1] + "/diffbin/raw/*_cpu_*.json"):
    r = json.load(open(f)); r["binary_sha256"] = "other"; json.dump(r, open(f, "w"))
for f in glob.glob(sys.argv[1] + "/diffhost/raw/*_cpu_*.json"):
    r = json.load(open(f)); r["result"]["hosts"] = ["other"]; json.dump(r, open(f, "w"))
for f in glob.glob(sys.argv[1] + "/onefail/raw/*_cpu_*rep3.json"):
    r = json.load(open(f)); r["ok"] = False; json.dump(r, open(f, "w"))
PY
for d in diffn diffbin diffhost onefail; do python3 scripts/analyze.py "$OUT/$d" > /dev/null 2>&1; done
check "different host: no ratio" J "
sys.exit(any('gpu_speedup_vs_cpu' in x for x in json.load(open('$OUT/diffhost/summary.json'))['rows']))"
check "unequal repetitions are shown beside the ratio" J "
r=[x for x in json.load(open('$OUT/onefail/summary.json'))['rows'] if x['backend']=='device']
sys.exit(not (r and all(x['reps']==3 and x['cpu_reps']==2 for x in r)))"
check "different workload: no ratio" J "
sys.exit(any('gpu_speedup_vs_cpu' in x for x in json.load(open('$OUT/diffn/summary.json'))['rows']))"
check "different binary: no ratio" J "
sys.exit(any('gpu_speedup_vs_cpu' in x for x in json.load(open('$OUT/diffbin/summary.json'))['rows']))"

# 5. The real program without a device: --require-device exits 7, the cpu backend still runs.
make -s BIN=bin/g2e2e gpu > "$OUT/build.log" 2>&1
check "real build" test -x bin/g2e2e/bh_gpu
python3 scripts/run_manifest.py $M --bin bin/g2e2e --cores-per-node 4 --out "$OUT/real" > "$OUT/real.log" 2>&1
check "real, no device: runner reports failure" test $? -eq 1
check "real, no device: all 6 device runs exit 7; all 6 cpu runs ok" J "
r=[json.load(open(f)) for f in glob.glob('$OUT/real/raw/*.json')]
dv=[x for x in r if '_device_' in x['case']]; cp=[x for x in r if '_cpu_' in x['case']]
sys.exit(not (len(dv)==len(cp)==6 and all(x['returncode']==7 and not x['ok'] for x in dv) and all(x['ok'] for x in cp)))"
check "real cpu backend: team, distinct CPUs, binding and schedule as requested" J "
r=[json.load(open(f))['result'] for f in glob.glob('$OUT/real/raw/*_cpu_*.json')]
sys.exit(not (len(r)==6 and all(x['threads_effective']==2 and x['distinct_cpus']==2 and x['proc_bind']=='close' and x['schedule_effective']=='dynamic' and x['chunk_effective']==64 for x in r)))"
check "real cpu backend: phases account for the timed region" J "
r=[json.load(open(f))['result'] for f in glob.glob('$OUT/real/raw/*_cpu_*.json')]
parts=lambda x: sum(x[k] for k in ('integrate_s','build_s','flatten_s','map_nodes_s','to_device_s','traverse_s','from_device_s'))
sys.exit(not all(0.8*x['wall_s'] <= parts(x) <= 1.001*x['wall_s'] for x in r))"
OMP_NUM_THREADS=2 OMP_PROC_BIND=false OMP_SCHEDULE=static bin/g2e2e/bh_gpu --backend cpu --n 1000 --steps 1 > "$OUT/raw_unbound.txt" 2>&1
check "real program reports the settings it ran with, not those requested elsewhere" J "
x=json.loads(open('$OUT/raw_unbound.txt').read().split('result ')[1]); sys.exit(not (x['proc_bind']=='false' and x['schedule_effective']=='static'))"
python3 scripts/analyze.py "$OUT/real" > /dev/null 2>&1
check "real, no device: analysis has cpu rows only" J "
sys.exit(not all(x['backend']=='cpu' for x in json.load(open('$OUT/real/summary.json'))['rows']))"

# 6. The job script, started elsewhere as PBS does.
hostname > "$OUT/nf"
mkdir -p "$OUT/home" "$OUT/fake"
cat > "$OUT/fake/fakecxx" <<STUB
#!/bin/bash
# Stands in for nvc++: the "binary" it builds is the stub.
while [ \$# -gt 0 ]; do [ "\$1" = -o ] && out=\$2; shift; done
cp "$OUT/stub/bh_gpu" "\$out"
STUB
cat > "$OUT/fake/nvidia-smi" <<'STUB'
#!/bin/bash
case "$*" in *query-gpu=timestamp*) echo "2026/01/01 00:00:00.000, 0, 100, 500" ;;
             *query-compute-apps*) echo "1, stub, 500 MiB" ;;
             *query-gpu=name*) echo "Stub GPU, GPU-stub, 0" ;; *) echo "stub nvidia-smi" ;; esac
STUB
chmod +x "$OUT/fake/fakecxx" "$OUT/fake/nvidia-smi"
job() { ( cd "$OUT/home" && env PBS_O_WORKDIR="$code" PBS_NODEFILE="$OUT/nf" ALLOW_NO_MODULES=1 "$@" bash "$code/jobs/vanda/gpu_timing.pbs" ); }
job PBS_JOBID=g2cold.local RUNS_ROOT="$OUT/cold" RUN_ID=cold MANIFEST=$M JOB_CXX=g++ JOB_OMPFLAGS=-fopenmp > "$OUT/cold.log" 2>&1
check "job refuses an unfrozen manifest, exit 3" test $? -eq 3
job PBS_JOBID=g2real.local RUNS_ROOT="$OUT/jr" RUN_ID=r JOB_CXX=g++ JOB_OMPFLAGS=-fopenmp > "$OUT/jr.log" 2>&1
check "job on a real no-device build stops at the device gate, exit 5" test $? -eq 5
check "no-device job ran no timing" test ! -e "$OUT/jr/r/raw"
check "device gate recorded FAIL rc=7" grep -q '^device FAIL rc=7' "$OUT/jr/r/gates.txt"
# Without a module command the job sources /etc/profile, which would drop the fake nvidia-smi.
module() { :; }; export -f module
job PATH="$OUT/fake:$PATH" PBS_JOBID=g2stub.local RUNS_ROOT="$OUT/js" RUN_ID=s MANIFEST=$M ALLOW_UNFROZEN=1 JOB_CXX="$OUT/fake/fakecxx" > "$OUT/js.log" 2>&1
rc=$?
unset -f module
[ "$rc" = 0 ] || tail -25 "$OUT/js.log"
check "job with the stub passes all gates, exit 0" test "$rc" -eq 0
check "stub job gates in order" test "$(cut -d' ' -f1,2 "$OUT/js/s/gates.txt" | tr '\n' ' ')" = "build PASS device PASS gpu_busy PASS runs PASS analyze PASS "
check "stub job summary has the ratio" J "
sys.exit(not any(x.get('gpu_speedup_vs_cpu')==3 for x in json.load(open('$OUT/js/s/summary.json'))['rows']))"

echo "$([ "$fail" = 0 ] && echo PASS || echo FAIL): G2 pipeline, $pass passed, $fail failed"
[ "$fail" = 0 ]
