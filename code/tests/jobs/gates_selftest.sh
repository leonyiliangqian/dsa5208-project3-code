#!/bin/bash
# Fault-injection test for jobs/vanda/gates.sh in the two-container
# simulation (docker/twonode/run.sh: node1 and node2, ssh between them, ssh
# forwarding PBS_JOBID as PBS tm is assumed to do):
#   docker/twonode/run.sh 'SOURCE_GIT_REV=<rev> SOURCE_DIRTY=<0|1> bash tests/jobs/gates_selftest.sh'
# Each scenario runs as a separate simulated job (the library exits on STOP)
# with PBS_JOBID=selftest.local; its directory under SELFTEST_OUT keeps
# gates.txt, cleanup.log, gate logs, snapshots and stdout. results.txt and
# provenance.txt summarise the run.
set -u
cd "$(dirname "$0")/../.."
OUTROOT=${SELFTEST_OUT:-artifacts/local/runs/gates-selftest-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUTROOT"
RESULTS=$OUTROOT/results.txt
: > "$RESULTS"
pass=0 fail=0
PAT="vanda-selftest/"
ME=$(id -un)
FAKE=$PWD/bin/selftest-fake     # shared by both containers through /work
mkdir -p "$FAKE"
cp /bin/sleep "$FAKE/orted"; cp /bin/sleep "$FAKE/mpirun"

{
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "source_git_rev=${SOURCE_GIT_REV:-unknown} source_dirty=${SOURCE_DIRTY:-unknown}"
    echo "environment=local Docker two-container simulation (docker/twonode), ssh launch; not PBS, not Vanda"
    echo "host=$(hostname) kernel=$(uname -r)"
    mpirun --version 2>&1 | head -1
    sha256sum jobs/vanda/gates.sh jobs/vanda/snapshot_watch.sh jobs/vanda/remote_capture.sh \
        tests/toolchain/run_to_file.sh tests/toolchain/leftover_check.sh \
        tests/toolchain/proc_snapshot.sh tests/jobs/gates_selftest.sh docker/twonode/Dockerfile docker/twonode/run.sh
} > "$OUTROOT/provenance.txt"

# scenario NAME BODY (env overrides as prefix assignments). Sets SC_OUT, SC_RC, SC_SECS.
scenario() {
    local name=$1 body=$2 out=$OUTROOT/$1 t0
    mkdir -p "$out"
    t0=$(date +%s)
    PBS_JOBID=selftest.local LEFTOVER_JOBID=selftest.local RUN_ID=selftest-$name \
        OUT=$out JOB_START_S=$t0 WALLTIME_S=${WALLTIME_S:-900} LEFTOVER_PATTERN=$PAT \
        REMOTE_EXEC=${REMOTE_EXEC:-"ssh -o BatchMode=yes"} END_RESERVE_S=${END_RESERVE_S:-60} \
        GATE_GRACE_S=${GATE_GRACE_S:-5} CLEANUP_NODE_S=${CLEANUP_NODE_S:-40} \
        bash -c "source jobs/vanda/gates.sh; $body
                 verdict >/dev/null && finish 0; finish 1" > "$out/stdout" 2>&1
    SC_RC=$?
    SC_OUT=$out
    SC_SECS=$(( $(date +%s) - t0 ))
    echo "== scenario $name exit=$SC_RC elapsed_s=$SC_SECS" | tee -a "$RESULTS"
}

check() {  # check DESCRIPTION CONDITION...
    local what=$1; shift
    if "$@"; then echo "PASS $what" | tee -a "$RESULTS"; pass=$((pass + 1))
    else echo "FAIL $what" | tee -a "$RESULTS"; fail=$((fail + 1)); fi
}
has() { grep -Eq -- "$1" "$2"; }
lacks() { ! grep -Eq -- "$1" "$2"; }
# Is a process matching a pattern alive on a node? ([x]yz keeps pgrep from
# matching the shell that runs it.)
alive() {  # alive NODE REGEX
    if [ "$1" = node1 ]; then pgrep -u "$ME" -f "$2" >/dev/null
    else ssh -o BatchMode=yes node2 "pgrep -u $ME -f '$2'" >/dev/null; fi
}
gone_everywhere() { ! alive node1 "$1" && ! alive node2 "$1"; }
no_watchers() { gone_everywhere '[s]napshot_watch.sh' && gone_everywhere '[p]roc_snapshot.sh'; }
# start_fake NODE NAME ENVSPEC: an MPI-like long-running process (comm NAME)
# that ignores TERM, started outside any gate with the given environment.
start_fake() {
    local cmd="setsid env $3 bash -c 'trap \"\" TERM; exec $FAKE/$2 600' </dev/null >/dev/null 2>&1 &"
    if [ "$1" = node1 ]; then bash -c "$cmd"; else ssh -o BatchMode=yes node2 "$cmd"; fi
    sleep 1
}
stop_fake() { pkill -KILL -u "$ME" -f "[s]elftest-fake/$1" 2>/dev/null
              ssh -o BatchMode=yes node2 "pkill -KILL -u $ME -f '[s]elftest-fake/$1'" 2>/dev/null; true; }

# 1. Two-node MPI job that never finishes (required): TIMEOUT, snapshot of
#    both nodes, both nodes CLEAN, later gate runs, verdict FAIL.
scenario hang "
gate hang 12 required mpirun --hostfile \$PBS_NODEFILE -np 2 --map-by node bash -c 'exec -a ${PAT}hang sleep 600'
gate after 30 required true"
check "hang: TIMEOUT recorded"                 has '^hang TIMEOUT rc=124 .*kind=required' "$SC_OUT/gates.txt"
check "hang: snapshot taken on both nodes"     test "$(grep -c '^== snapshot host=' "$SC_OUT/gate_hang.snapshot")" -eq 2
check "hang: snapshot saw the hung rank"       has "${PAT}hang" "$SC_OUT/gate_hang.snapshot"
check "hang: both nodes CLEAN after it"        test "$(sed -n '/^== after hang /,/^== after after /p' "$SC_OUT/cleanup.log" | grep -c ' CLEAN$')" -eq 2
check "hang: later gate ran"                   has '^after PASS ' "$SC_OUT/gates.txt"
check "hang: required TIMEOUT -> verdict FAIL" test "$SC_RC" -eq 1
check "hang: no rank left on either node"      gone_everywhere "[v]anda-selftest/hang"
check "hang: no watcher/snapshot process left" no_watchers
check "after: no snapshot for a quick gate"    test ! -e "$SC_OUT/gate_after.snapshot"

# 2. Owned strays on node2 that ignore TERM: one owned through the job's
#    binary path, one (an "orted") owned only through PBS_JOBID. Both killed.
scenario stray "
gate stray 30 required ssh -o BatchMode=yes node2 \"setsid bash -c 'trap \\\"\\\" TERM; exec -a ${PAT}stray sleep 600' </dev/null >/dev/null 2>&1 &\"
gate stray_jobid 30 required ssh -o BatchMode=yes node2 \"setsid bash -c 'trap \\\"\\\" TERM; exec $FAKE/orted 600' </dev/null >/dev/null 2>&1 &\""
check "stray: path-owned stray killed (node2)" has 'leftover_check host=node2 owned_found=1 owned_remaining=0 foreign=0 unattributed=0 CLEAN' "$SC_OUT/cleanup.log"
check "stray: jobid-owned orted found"         has "leftover host=node2 owned: .*selftest-fake/orted" "$SC_OUT/cleanup.log"
check "stray: verdict PASS (exit 0)"           test "$SC_RC" -eq 0
check "stray: nothing left"                    gone_everywhere "[v]anda-selftest/stray"
check "stray: fake orted gone"                 gone_everywhere "[s]elftest-fake/orted"

# 3a. Negative control: an MPI daemon of the same user belonging to ANOTHER
#     job (different PBS_JOBID) on node2 is ignored and survives.
start_fake node2 orted "PBS_JOBID=other.job"
scenario foreign "
gate quick 30 required true"
check "foreign: reported as foreign, CLEAN"    has 'leftover_check host=node2 owned_found=0 owned_remaining=0 foreign=1 unattributed=0 CLEAN' "$SC_OUT/cleanup.log"
check "foreign: job verdict PASS"              test "$SC_RC" -eq 0
check "foreign: unrelated orted still alive"   alive node2 "[s]elftest-fake/orted"
stop_fake orted

# 3b. Negative control: a same-user "mpirun" on node1 with no job marker at
#     all (ownership unknown) is NOT killed, and the job stops (UNCERTAIN).
start_fake node1 mpirun "-u PBS_JOBID -u P3_RUN_MARKER -u LEFTOVER_JOBID"
scenario unattributed "
gate quick 30 required true
gate later 30 required true"
check "unattributed: reported, not killed"     has 'leftover host=node1 unattributed \(not killed\): .*selftest-fake/mpirun' "$SC_OUT/cleanup.log"
check "unattributed: node1 UNCERTAIN"          has 'host=node1 .* unattributed=1 UNCERTAIN' "$SC_OUT/cleanup.log"
check "unattributed: job stopped, exit 6"      test "$SC_RC" -eq 6
check "unattributed: later gate not run"       lacks '^later ' "$SC_OUT/gates.txt"
check "unattributed: process still alive"      alive node1 "[s]elftest-fake/mpirun"
stop_fake mpirun

# 4. Kinds: optional FAIL and optional SKIP stay visible but do not decide
#    the verdict; a required SKIP (dependency) does.
scenario optional "
gate req 30 required true
gate opt 30 optional false
skip opt_tool optional 'tool not found'"
check "optional: optional FAIL/SKIP listed"    has 'optional: opt=FAIL opt_tool=SKIP' "$SC_OUT/stdout"
check "optional: verdict PASS (exit 0)"        test "$SC_RC" -eq 0
scenario deps "
gate base 30 required false
need base dependant required && gate dependant 30 required true
gate other 30 required true"
check "deps: dependant SKIP recorded"          has '^dependant SKIP rc=- secs=0 kind=required needs base to pass' "$SC_OUT/gates.txt"
check "deps: unrelated gate ran"               has '^other PASS ' "$SC_OUT/gates.txt"
check "deps: required FAIL/SKIP -> exit 1"     test "$SC_RC" -eq 1

# 5. End of the budget: worst-case overhead with 2 nodes and grace 5 s is
#    5 + 2*14 + 2 + 2*45 = 125 s; with WALLTIME_S=230 and END_RESERVE_S=60 the
#    first gate's 600 s limit is cut to about 45 s, it times out, the next
#    gate is skipped, and the job finishes inside WALLTIME_S - END_RESERVE_S.
WALLTIME_S=230 END_RESERVE_S=60 scenario budget "
gate long 600 required bash -c 'exec -a ${PAT}long sleep 600'
gate next 600 required true"
cut=$(sed -n 's/^long TIMEOUT .*limit cut from 600s to \([0-9]*\)s.*/\1/p' "$SC_OUT/gates.txt")
check "budget: limit cut to fit (got ${cut:-none}s)" test -n "$cut" -a "${cut:-0}" -ge 40 -a "${cut:-99}" -le 45
check "budget: next gate skipped, no time"     has '^next SKIP .*no time left' "$SC_OUT/gates.txt"
check "budget: finished within 230-60 s (${SC_SECS}s)" test "$SC_SECS" -le 170
check "budget: verdict FAIL (exit 1)"          test "$SC_RC" -eq 1

# 6. Remote cleanup hangs (a remote command that never returns): bounded by
#    CLEANUP_NODE_S + 5; the job stops with exit 6 within that bound.
printf '#!/bin/bash\nexec sleep 1000\n' > "$FAKE/hang_exec"; chmod +x "$FAKE/hang_exec"
REMOTE_EXEC=$FAKE/hang_exec CLEANUP_NODE_S=10 scenario cleanup_hangs "
gate first 20 required true
gate second 20 required true"
check "cleanup_hangs: exit 6"                  test "$SC_RC" -eq 6
check "cleanup_hangs: STOP message"            has 'STOP: could not verify' "$SC_OUT/stdout"
check "cleanup_hangs: second not run"          lacks '^second ' "$SC_OUT/gates.txt"
# first gate ~0 s + node1 cleanup (seconds) + node2 bound 10+5 s
check "cleanup_hangs: bounded (${SC_SECS}s <= 30)" test "$SC_SECS" -le 30

# 6c. Remote output goes to the job's output file, not to the caller's stdout,
#     as pbs_tmrsh does (job 1513781). Reports must still reach cleanup.log and
#     the snapshot, and the gates after the first must run.
printf '#!/bin/bash\nexec ssh -o BatchMode=yes "$@" >> %s/tm_job.OU 2>&1\n' "$FAKE" > "$FAKE/tm_exec"; chmod +x "$FAKE/tm_exec"
REMOTE_EXEC=$FAKE/tm_exec scenario tm_output "
gate hang 12 required mpirun --hostfile \$PBS_NODEFILE -np 2 --map-by node bash -c 'exec -a ${PAT}tmhang sleep 600'
gate after 30 required true"
check "tm_output: not stopped by cleanup (exit 1)" test "$SC_RC" -eq 1
check "tm_output: later gate ran"              has '^after PASS ' "$SC_OUT/gates.txt"
check "tm_output: 2 CLEAN after hang"          test "$(sed -n '/^== after hang /,/^== after after /p' "$SC_OUT/cleanup.log" | grep -c ' CLEAN$')" -eq 2
check "tm_output: 2 CLEAN after later gate"    test "$(sed -n '/^== after after /,$p' "$SC_OUT/cleanup.log" | grep -c ' CLEAN$')" -eq 2
check "tm_output: snapshot of both nodes"      test "$(grep -c '^== snapshot host=' "$SC_OUT/gate_hang.snapshot")" -eq 2
check "tm_output: no report file left"         test -z "$(find "$SC_OUT" -name '.remote.*')"

# 6a. Worst case at the end of the budget: the gate times out at its cut
#     limit while the remote snapshot and the remote cleanup both hang until
#     their bounds (grace 5, snapshot 2*(12+2)+2, cleanup node1 quick + node2
#     40+5). The job must still end within WALLTIME_S - END_RESERVE_S.
WALLTIME_S=230 END_RESERVE_S=60 REMOTE_EXEC=$FAKE/hang_exec scenario budget_worst "
gate long 600 required bash -c 'exec -a ${PAT}long sleep 600'
gate next 600 required true"
check "budget_worst: gate TIMEOUT"             has '^long TIMEOUT ' "$SC_OUT/gates.txt"
check "budget_worst: stopped on cleanup, exit 6" test "$SC_RC" -eq 6
check "budget_worst: within 230-60 s (${SC_SECS}s)" test "$SC_SECS" -le 170

# 6b. A gate whose own limit is short (12 s) runs with that limit when time
#     remains; it is not mistaken for "no time left" (regression check).
scenario short_limit "
gate short 12 required true"
check "short_limit: ran and passed"            has '^short PASS ' "$SC_OUT/gates.txt"

# 7. Critical gate fails: STOP, exit 5, nothing after it runs.
scenario critical "
gate crit 20 critical false
gate later 20 required true"
check "critical: exit 5"                       test "$SC_RC" -eq 5
check "critical: later not run"                lacks '^later ' "$SC_OUT/gates.txt"

check "end: no watcher/snapshot process left"  no_watchers
rm -rf "$FAKE"
echo "gates_selftest passed=$pass failed=$fail $([ "$fail" -eq 0 ] && echo PASS || echo FAIL)" | tee -a "$RESULTS"
[ "$fail" -eq 0 ]
