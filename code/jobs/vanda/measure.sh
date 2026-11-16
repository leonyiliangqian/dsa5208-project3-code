# Measurement job body shared by pilot.pbs, bench1.pbs and bench2.pbs (source
# it from the job script, in the code/ directory). The job script sets:
#   RUN_PREFIX       run directory prefix (pilot1, bench1, bench2)
#   MANIFEST         manifests/<file>.json (overridable with qsub -v MANIFEST=...)
#   EXPECT_NODES     distinct hosts the job must have
#   CPUS_PER_NODE    exact cpuset size per node (the chunk's ncpus)
#   PLACE_R PLACE_T  placement probed before measuring (ranks per node x threads)
#   REQUIRE_FROZEN   1 for formal benchmarks: refuse a manifest whose status
#                    does not start with "frozen" (the pilot runs a proposed one)
#
#   SCALE_CHECKS     1 adds, after placement, critical gates that use every core:
#                    MPI-only placement, data exchange, and bound runs against serial.
#                    Every launch states its placement and binding; the exchange gate
#                    uses place_mpi's (make internode, after job 1526501)
#
# Gates (jobs/vanda/gates.sh): allocation (critical: hosts, exact cpusets and
# ONE CPU model, since timings are compared only within a model), build
# (critical, -march=native, which one model makes safe), placement (critical),
# runs (required: scripts/run_manifest.py with per-run timeouts and a deadline
# that leaves room for cleanup), analyze (required). Every job launches through
# pbs_tmrsh (multinode_setup.sh); REQUIRE_PBS_TM=0 is for local simulation only.
# PILOT_SCALE divides every N and PILOT_PLACEMENT="R T" overrides the probe,
# for local simulation only, as does SIM_CPUS_PER_NODE (the job scripts'
# CPUS_PER_NODE default); all are recorded.

set -uo pipefail
cd "${PBS_O_WORKDIR:?run under PBS, from the code/ directory}"
JOB=${PBS_JOBID%%.*}
JOB_START_S=$(date +%s)
RUN_ID=${RUN_ID:-$RUN_PREFIX-$(date -u +%Y%m%dT%H%M%SZ)-$JOB}
OUT=${RUNS_ROOT:-artifacts/cluster/runs}/$RUN_ID
mkdir -p "$OUT/env"
exec > >(tee "$OUT/job.log") 2>&1
echo "run_id=$RUN_ID job=${PBS_JOBID} host=$(hostname) start=$(date -Is) source_rev=${SOURCE_GIT_REV:-unset}"
echo "manifest=$MANIFEST scale=${PILOT_SCALE:-1} placement_override=${PILOT_PLACEMENT:-none} sim_cpus_per_node=${SIM_CPUS_PER_NODE:-none}"

# tm launches with binding hang on Vanda (job 1514029); pbs_tmrsh passed twonode (job 1514458).
# One node is no exception: with only a hostfile, job 1519637 still selected tm and hung.
[ "${REQUIRE_PBS_TM:-1}" = 1 ] && export MPI_LAUNCHER=${MPI_LAUNCHER:-pbs_tmrsh}
source jobs/vanda/multinode_setup.sh
{
    echo "== numactl"; numactl --hardware 2>&1 || true
    echo "== memory"; free -g 2>&1 || true
} >> "$OUT/env/environment.txt" 2>&1

status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$MANIFEST") \
    || { echo "FATAL: cannot read manifest $MANIFEST"; exit 3; }
echo "manifest_status=$status sha256=$(sha256sum "$MANIFEST" | cut -c1-64)"
if [ "${REQUIRE_FROZEN:-1}" = 1 ] && [ "${status#frozen}" = "$status" ]; then
    echo "FATAL: $MANIFEST is not frozen (status: $status); formal runs need a frozen manifest"
    exit 3
fi

BIN=bin/vanda-$JOB
export CXXFLAGS="-O3 -std=c++17 -Wall -Wextra -march=native"
export OUT BIN MPI
WALLTIME_S=$(sed -n 's/^#PBS -l walltime=\([0-9]*\):\([0-9]*\):\([0-9]*\).*/\1 \2 \3/p' "$0" \
             | awk '{print $1*3600 + $2*60 + $3}')
WALLTIME_S=${WALLTIME_S:-3600}
LEFTOVER_PATTERN="vanda-$JOB/"
export LEFTOVER_JOBID=${PBS_JOBID:-}
source jobs/vanda/gates.sh
echo "walltime_s=$WALLTIME_S end_reserve_s=$END_RESERVE_S gate_overhead_s=$(gate_overhead_s) remote_exec=$REMOTE_EXEC"

FIRST_CPU=$(grep Cpus_allowed_list /proc/self/status | cut -f2 | cut -d, -f1 | cut -d- -f1)
read -r PR PT <<< "${PILOT_PLACEMENT:-$PLACE_R $PLACE_T}"

gate allocation 90 critical env ALLOC_EXPECT_NODES="$EXPECT_NODES" ALLOC_CPUS_PER_NODE="$CPUS_PER_NODE" \
    ALLOC_SAME_MODEL=1 bash jobs/vanda/allocation_check.sh "$OUT" "$NODES" "$LAUNCH"
gate build     300 critical make BIN="$BIN" all hybrid
gate placement  90 critical make BIN="$BIN" MPIRUN="$MPI" placement PLACE_NODES="$EXPECT_NODES" \
    PLACE_R="$PR" PLACE_T="$PT"
if [ "${SCALE_CHECKS:-0}" = 1 ]; then
    ALL_CORES=$(( EXPECT_NODES * CPUS_PER_NODE ))
    gate place_mpi  180 critical make BIN="$BIN" MPIRUN="$MPI" placement PLACE_NODES="$EXPECT_NODES" \
        PLACE_R="$CPUS_PER_NODE" PLACE_T=1
    gate internode  180 critical make BIN="$BIN" MPIRUN="$MPI" internode INTERNODE_NP="$ALL_CORES" \
        INTERNODE_HOSTS="$EXPECT_NODES"
    gate serial     240 critical make BIN="$BIN" EVIDENCE="$OUT" check
    gate bound_vs_serial 480 critical make BIN="$BIN" EVIDENCE="$OUT" check-hybrid MPIRUN=mpirun \
        MPIRUN_FLAGS="${MPI#mpirun }" HYBRID_GATE_ARGS="--max-cores $ALL_CORES --suite bound --expect-hosts $EXPECT_NODES"
fi
# The runner gets a deadline that leaves this gate's worst-case overhead and
# the analysis gate; the gate's own limit is then cut to the same budget.
DEADLINE=$(( JOB_START_S + WALLTIME_S - END_RESERVE_S - $(gate_overhead_s) - 120 - $(gate_overhead_s) ))
gate runs   99999 required python3 scripts/run_manifest.py "$MANIFEST" --out "$OUT" --bin "$BIN" \
    --mpirun "$MPI" --cores-per-node "$CPUS_PER_NODE" --nodes "$EXPECT_NODES" --first-cpu "$FIRST_CPU" \
    --deadline "$DEADLINE" --scale "${PILOT_SCALE:-1}"
gate analyze   120 required python3 scripts/analyze.py "$OUT"
[ -f "$OUT/summary.txt" ] || cp "$OUT/gate_analyze.log" "$OUT/summary.txt" 2>/dev/null

verdict >/dev/null && finish 0
finish 1
