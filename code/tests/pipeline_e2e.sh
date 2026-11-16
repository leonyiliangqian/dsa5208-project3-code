#!/bin/bash
# End-to-end test of the measurement pipeline on tiny inputs in Docker:
# manifest -> run_manifest.py (run_case.py per run) -> analyze.py ->
# report_figures.py, plus the timeout, deadline-skip and refusal paths.
# Local pipeline evidence only; its timings are not measurements.
#   docker/run.sh bash tests/pipeline_e2e.sh
# Writes artifacts/local/runs/pipeline-e2e-<UTC>/ (records, summary, figures,
# results.txt, provenance.txt).
set -u
cd "$(dirname "$0")/.."
OUT=${E2E_OUT:-artifacts/local/runs/pipeline-e2e-$(date -u +%Y%m%dT%H%M%SZ)}
BIN=bin/e2e
mkdir -p "$OUT"
RESULTS=$OUT/results.txt
: > "$RESULTS"
pass=0 fail=0
check() {
    local what=$1; shift
    if "$@"; then echo "PASS $what" | tee -a "$RESULTS"; pass=$((pass + 1))
    else echo "FAIL $what" | tee -a "$RESULTS"; fail=$((fail + 1)); fi
}
{
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "source_git_rev=${SOURCE_GIT_REV:-unknown} source_dirty=${SOURCE_DIRTY:-unknown}"
    echo "environment=local Docker container, $(nproc) CPUs; pipeline test, not a measurement"
    mpirun --version 2>&1 | head -1; c++ --version | head -1
    sha256sum manifests/local_e2e.json scripts/run_manifest.py scripts/run_case.py scripts/analyze.py \
        scripts/report_figures.py tests/pipeline_e2e.sh
} > "$OUT/provenance.txt"

make -s BIN=$BIN all hybrid > "$OUT/build.log" 2>&1
check "build" test -x $BIN/bh_serial -a -x $BIN/bh_hybrid

# 1. The whole local manifest (33 runs including a deliberate 3 s timeout).
python3 scripts/run_manifest.py manifests/local_e2e.json --out "$OUT/run" --bin $BIN \
    --cores-per-node "$(nproc)" --first-cpu 0 > "$OUT/run_manifest.log" 2>&1
rc=$?
check "runner reports failure because of the timeout probe (exit 1)" test "$rc" -eq 1
check "plan and manifest copy written" test -s "$OUT/run/plan.json" -a -s "$OUT/run/manifest.json"
n_raw=$(ls "$OUT"/run/raw/*.json | wc -l); n_val=$(ls "$OUT"/run/validate/*.json | wc -l)
check "35 timed records + 2 validate records (got $n_raw + $n_val)" test "$n_raw" -eq 35 -a "$n_val" -eq 2
probe=$OUT/run/raw/timeout_probe_serial_n2000000_th0p5_L8_rep1.json
check "timeout probe recorded as timed out with SIGTERM" python3 -c "
import json,sys; d=json.load(open('$probe')); sys.exit(not (d['timed_out'] and d['kill_signal']=='SIGTERM' and not d['ok'] and d['elapsed_s'] < 20))"
# /proc scan rather than pgrep (not in the dev image); it must find this
# shell itself, so an empty /proc cannot make the check pass vacuously; the
# split string keeps the check from matching its own command line.
check "no bh_serial left running after the timeout" python3 -c "
import os,sys
cmds=[open(f'/proc/{p}/cmdline','rb').read().replace(bytes([0]),b' ').decode() for p in os.listdir('/proc') if p.isdigit() and os.path.exists(f'/proc/{p}/cmdline')]
sys.exit(not (any('pipeline_e2e.sh' in c for c in cmds) and not any('bin/e2e/' + 'bh_serial' in c for c in cmds)))"
check "records carry manifest id, sha256 and study tags" python3 -c "
import json,glob,sys; r=[json.load(open(f)) for f in glob.glob('$OUT/run/raw/*.json')]
sys.exit(not all(x['tags'].get('manifest_id')=='local-e2e-v1' and len(x['tags'].get('manifest_sha256',''))==64 and x['tags'].get('study') for x in r))"

# 2. Deadline: with a deadline already near, every run is skipped and recorded.
python3 scripts/run_manifest.py manifests/local_e2e.json --out "$OUT/deadline" --bin $BIN \
    --cores-per-node "$(nproc)" --deadline "$(( $(date +%s) + 30 ))" > "$OUT/deadline.log" 2>&1
check "deadline: every run recorded as skipped" python3 -c "
import json,glob,sys; r=[json.load(open(f)) for f in glob.glob('$OUT/deadline/*/*.json')]
sys.exit(not (len(r)==37 and all(x.get('skipped') for x in r)))"

# 2b. The job stops the runner mid-run (as a gate time limit does): TERM must
#     reach the run in progress, a hybrid MPI run in its own process group.
cat > "$OUT/term_manifest.json" <<'JSON'
{"id": "local-e2e-term", "status": "local pipeline test only",
 "defaults": {"theta": 0.5, "leaf_size": 8, "steps": 200, "warmup": 1, "seed": 42, "schedule": "dynamic",
              "chunk": 64, "nodes": 1, "reps": 1, "timeout_s": 600, "n": 400000},
 "cases": [{"study": "term", "program": "hybrid", "ranks": 2, "threads": 1}]}
JSON
python3 scripts/run_manifest.py "$OUT/term_manifest.json" --out "$OUT/term" --bin $BIN \
    --cores-per-node "$(nproc)" > "$OUT/term.log" 2>&1 &
runner=$!
sleep 6
kill -TERM $runner; wait $runner; rc=$?
sleep 3
check "TERM to runner: runner exits 143 (got $rc)" test "$rc" -eq 143
check "TERM to runner: no bh_hybrid or mpirun left" python3 -c "
import os,sys
cmds=[open(f'/proc/{p}/cmdline','rb').read().replace(bytes([0]),b' ').decode() for p in os.listdir('/proc') if p.isdigit() and os.path.exists(f'/proc/{p}/cmdline')]
sys.exit(not (any('pipeline_e2e.sh' in c for c in cmds) and not any(('bin/e2e/' + 'bh_hybrid') in c or c.startswith('mpirun') for c in cmds)))"
check "TERM to runner: interrupted run recorded, not ok" python3 -c "
import json,glob,sys; r=[json.load(open(f)) for f in glob.glob('$OUT/term/raw/*.json')]
sys.exit(not (len(r)==1 and not r[0]['ok']))"

# 3. Analysis and figures.
python3 scripts/analyze.py "$OUT/run" > "$OUT/analyze.log" 2>&1
check "analyze: summary written" test -s "$OUT/run/summary.json" -a -s "$OUT/run/summary.csv"
check "analyze: 1 failed run (the probe), speedups present" python3 -c "
import json,sys; d=json.load(open('$OUT/run/summary.json'))
sys.exit(not (len(d['failed'])==1 and d['failed'][0]['timed_out'] and any('speedup_vs_serial' in r for r in d['rows'])))"
python3 scripts/analyze.py "$OUT/run" "$OUT/deadline" --out "$OUT/combined" > "$OUT/analyze_combined.log" 2>&1
check "analyze: combining run directories" python3 -c "
import json,sys; d=json.load(open('$OUT/combined/summary.json')); sys.exit(not (len(d['inputs'])==2 and len(d['failed'])==36))"

python3 scripts/report_figures.py "$OUT/run/summary.json" > "$OUT/refuse1.log" 2>&1
check "figures: local input refused without --allow-local" test $? -ne 0
python3 scripts/report_figures.py "$OUT/run/summary.json" --allow-local --out ../report/generated > "$OUT/refuse2.log" 2>&1
check "figures: local input refused into report/generated" test $? -ne 0
python3 scripts/report_figures.py "$OUT/run/summary.json" --allow-local --out "$OUT/figures" > "$OUT/figures.log" 2>&1
for f in fig_serial_n fig_strong_runtime fig_strong_speedup fig_strong_efficiency tab_ranks_threads tab_schedule \
         fig_weak_runtime fig_weak_efficiency fig_phases fig_comm_fraction tab_accuracy_cost tab_failed_runs; do
    check "figures: $f.tex written with LOCAL banner" grep -q 'LOCAL PIPELINE TEST' "$OUT/figures/$f.tex"
done
for f in fig_serial_n fig_strong_runtime fig_weak_runtime fig_phases; do
    check "figures: $f.tex has log time axis" grep -q 'ymode=log' "$OUT/figures/$f.tex"
done
for f in fig_serial_n fig_strong_runtime fig_strong_speedup fig_strong_efficiency fig_weak_runtime \
         fig_weak_efficiency fig_phases fig_comm_fraction; do
    check "figures: $f.tex has log x axis" grep -q 'xmode=log' "$OUT/figures/$f.tex"
done
check "figures: failed-run table names the probe" grep -q 'timeout\\_probe' "$OUT/figures/tab_failed_runs.tex"
check "figures: braces balanced in every fragment" python3 -c "
import glob,sys; sys.exit(any(open(f).read().count('{')!=open(f).read().count('}') for f in glob.glob('$OUT/figures/*.tex')))"

rm -rf $BIN
echo "pipeline_e2e passed=$pass failed=$fail $([ "$fail" -eq 0 ] && echo PASS || echo FAIL)" | tee -a "$RESULTS"
[ "$fail" -eq 0 ]
