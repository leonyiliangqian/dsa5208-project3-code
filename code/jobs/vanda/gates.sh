# Gate runner shared by the Vanda job scripts (source it; not executable).
#
#   gate NAME LIMIT_S KIND CMD...      KIND: critical | required | optional
#   need GATE NAME KIND                true if GATE passed, else records NAME SKIP
#   skip NAME KIND REASON              records NAME SKIP (e.g. a tool is missing)
#   verdict                            prints the summary; 0 only if all
#                                      critical and required gates passed
#
# critical: anything but PASS stops the job (exit 5); later gates would not
#           mean what they claim.
# required: recorded; the job continues; anything but PASS (FAIL, TIMEOUT,
#           SKIP) prevents an overall PASS.
# optional: recorded and listed separately; does not affect the verdict.
# gates.txt lines: NAME STATUS rc=.. secs=.. kind=KIND [note]
#
# Each gate runs under `timeout` in its own process group (TERM at the limit,
# KILL GATE_GRACE_S later). A watcher in its own process group snapshots
# process state on every node shortly before the limit (gate_NAME.snapshot).
# After every gate, leftover processes owned by this job are terminated on
# every node and checked (tests/toolchain/leftover_check.sh; only processes
# carrying this job's PBS_JOBID or P3_RUN_MARKER, or this job's binary path,
# are touched). If any node is not CLEAN (owned process left, unattributed
# MPI process present, or no report), the job stops (exit 6).
#
# Time accounting, per gate on N nodes, worst case:
#   limit + GATE_GRACE_S                     the gate itself
#   + N * (SNAPSHOT_NODE_S + 2) + 2          watcher finishing after a timeout
#   + N * (CLEANUP_NODE_S + 5)               sequential per-node cleanup
# Before each gate the limit is cut so that this worst case still ends
# END_RESERVE_S before JOB_START_S + WALLTIME_S; if fewer than 20 s would be
# left for the gate it is skipped. This is a bounded best effort, not a
# guarantee that PBS never kills the job: JOB_START_S is taken when the script
# starts (after PBS has started the walltime clock), steps outside gates
# (module loading, the environment record, the final summary) are not
# bounded, and a process stuck in uninterruptible sleep survives KILL.
#
# Expects: OUT, JOB_START_S (epoch), WALLTIME_S, LEFTOVER_PATTERN, RUN_ID;
# optional REMOTE_EXEC (pbs_tmrsh by default, ssh in local simulation).

END_RESERVE_S=${END_RESERVE_S:-60}
GATE_GRACE_S=${GATE_GRACE_S:-20}
SNAPSHOT_NODE_S=${SNAPSHOT_NODE_S:-12}
CLEANUP_NODE_S=${CLEANUP_NODE_S:-40}
if [ -z "${REMOTE_EXEC:-}" ]; then
    if command -v pbs_tmrsh >/dev/null 2>&1; then REMOTE_EXEC=pbs_tmrsh
    else REMOTE_EXEC="ssh -o BatchMode=yes -o ConnectTimeout=10"; fi
fi
# Unique to this run; inherited by everything the gates start on this node.
export P3_RUN_MARKER=${P3_RUN_MARKER:-${RUN_ID:-run}.$(hostname -s).$$.$JOB_START_S}
export OUT LEFTOVER_PATTERN REMOTE_EXEC SNAPSHOT_NODE_S
: > "$OUT/gates.txt"
: > "$OUT/cleanup.log"

stamp() { date +%H:%M:%S; }

remaining_s() { echo $(( JOB_START_S + WALLTIME_S - END_RESERVE_S - $(date +%s) )); }

# Short host name, lower case, for comparing node file entries with hostname.
short() { echo "${1%%.*}" | tr '[:upper:]' '[:lower:]'; }

node_count() { sort -u "${PBS_NODEFILE:-/dev/null}" | wc -l | tr -d ' '; }

source jobs/vanda/remote_capture.sh

# Worst-case seconds a gate needs beyond its own limit (see header).
gate_overhead_s() {
    local n; n=$(node_count)
    echo $(( GATE_GRACE_S + n * (SNAPSHOT_NODE_S + 2) + 2 + n * (CLEANUP_NODE_S + 5) ))
}

record() { echo "$*" | tee -a "$OUT/gates.txt"; }

# Summary of gates.txt. Exit 0 only if every critical and required gate (and
# every cleanup) passed; optional gates are listed but do not count.
verdict() {
    local bad opt req
    req=$(grep -vc ' kind=optional' "$OUT/gates.txt")
    bad=$(grep -v ' kind=optional' "$OUT/gates.txt" | awk '$2 != "PASS"')
    opt=$(grep ' kind=optional' "$OUT/gates.txt" | awk '{print $1 "=" $2}' | paste -sd' ' -)
    echo "verdict required_or_critical=$req not_passed=$(printf '%s' "$bad" | grep -c .) optional: ${opt:-none}"
    [ -n "$bad" ] && printf '%s\n' "$bad" | sed 's/^/  not passed: /'
    if [ -z "$bad" ]; then echo "verdict PASS"; return 0; fi
    echo "verdict FAIL"; return 1
}

finish() {
    echo; echo "===== summary $(stamp)"; cat "$OUT/gates.txt"
    verdict
    echo "end=$(date -Is) elapsed_s=$(( $(date +%s) - JOB_START_S )) walltime_s=$WALLTIME_S"
    exit "$1"
}

cleanup_all() {
    local label=$1 h rc=0 me nodes clean
    me=$(short "$(hostname)")
    echo "== after $label $(stamp)" >> "$OUT/cleanup.log"
    for h in $(sort -u "${PBS_NODEFILE:-/dev/null}"); do
        if [ "$(short "$h")" = "$me" ]; then
            timeout -k 5 "$CLEANUP_NODE_S" bash tests/toolchain/leftover_check.sh "$LEFTOVER_PATTERN" \
                >> "$OUT/cleanup.log" 2>&1 || rc=1
        else
            remote_capture 5 "$CLEANUP_NODE_S" "$h" env LEFTOVER_JOBID="${LEFTOVER_JOBID:-}" \
                P3_RUN_MARKER="$P3_RUN_MARKER" bash "$PWD/tests/toolchain/leftover_check.sh" \
                "$LEFTOVER_PATTERN" >> "$OUT/cleanup.log" 2>&1 || rc=1
        fi
    done
    nodes=$(node_count)
    # Every node must report CLEAN; a missing report counts as failure.
    clean=$(sed -n "/^== after $label /,\$p" "$OUT/cleanup.log" | grep -c ' CLEAN$')
    [ "$rc" -eq 0 ] && [ "$clean" -eq "$nodes" ]
}

skip() {
    local name=$1 kind=$2; shift 2
    record "$name SKIP rc=- secs=0 kind=$kind $*"
    if [ "$kind" = critical ]; then echo "STOP: critical gate $name skipped"; finish 5; fi
}

need() {
    grep -q "^$1 PASS " "$OUT/gates.txt" && return 0
    skip "$2" "$3" "needs $1 to pass"
    return 1
}

gate() {
    local name=$1 limit=$2 kind=$3; shift 3
    local left rc status t0 note="" overhead watcher snap_at waited
    overhead=$(gate_overhead_s)
    left=$(( $(remaining_s) - overhead ))
    if [ "$left" -lt "$limit" ]; then
        note="limit cut from ${limit}s to ${left}s (worst-case overhead ${overhead}s) to fit the walltime"
        limit=$left
    fi
    # Skip on lack of time only; a short limit of the gate's own is honoured.
    if [ "$left" -lt 20 ]; then
        skip "$name" "$kind" "no time left before the walltime (worst-case overhead ${overhead}s)"
        return
    fi
    echo; echo "===== gate $name ($kind, limit ${limit}s) $(stamp): $*"
    t0=$(date +%s)
    snap_at=$(( limit > 15 ? limit - 8 : limit / 2 ))
    setsid bash jobs/vanda/snapshot_watch.sh "$snap_at" "$name" < /dev/null &
    watcher=$!
    timeout -k "$GATE_GRACE_S" "$limit" "$@" > "$OUT/gate_$name.log" 2>&1
    rc=$?
    case $rc in
        0) status=PASS ;;
        124|137) status=TIMEOUT; note="${note:+$note; }no completion within ${limit}s" ;;
        *) status=FAIL ;;
    esac
    # After a timeout let the snapshot finish (it is bounded); otherwise the
    # watcher is still asleep or mid-snapshot and is discarded. Either way its
    # whole process group is gone before cleanup starts.
    if [ "$status" = TIMEOUT ]; then
        waited=0
        while kill -0 "$watcher" 2>/dev/null \
              && [ "$waited" -lt $(( $(node_count) * (SNAPSHOT_NODE_S + 2) + 2 )) ]; do
            sleep 1; waited=$((waited + 1))
        done
    fi
    kill -TERM -- "-$watcher" 2>/dev/null; sleep 0.2; kill -KILL -- "-$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    tail -n 30 "$OUT/gate_$name.log"
    record "$name $status rc=$rc secs=$(( $(date +%s) - t0 )) kind=$kind $note"
    if ! cleanup_all "$name"; then
        sed -n "/^== after $name /,\$p" "$OUT/cleanup.log"
        echo "STOP: could not verify that gate $name left no processes of this job on every node"
        record "cleanup FAIL rc=- secs=0 kind=critical after $name"
        finish 6
    fi
    if [ "$status" != PASS ] && [ "$kind" = critical ]; then
        echo "STOP: critical gate $name $status; later gates would not mean what they claim"
        finish 5
    fi
}
