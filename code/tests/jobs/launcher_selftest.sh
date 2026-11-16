#!/bin/bash
# Fault-injection test for MPI_LAUNCHER=pbs_tmrsh on two simulated nodes:
#   docker/twonode/run.sh 'SOURCE_GIT_REV=<rev> SOURCE_DIRTY=<0|1> bash tests/jobs/launcher_selftest.sh'
# A local stand-in named pbs_tmrsh calls ssh, because PBS is not available here. So
# this tests the launch options, repeated mappings, failure handling and refusals. It
# does not test PBS, and it cannot show that the route is reliable on Vanda.
set -u
cd "$(dirname "$0")/../.."
OUTROOT=${SELFTEST_OUT:-artifacts/local/runs/launcher-selftest-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUTROOT"
RESULTS=$OUTROOT/results.txt
: > "$RESULTS"
pass=0 fail=0
ME=$(id -un)
FAKE=$PWD/bin/launcher-fake
BIN=bin/launcher-test
rm -rf "$FAKE"
mkdir -p "$FAKE/path" "$FAKE/agent" "$FAKE/broken" "$FAKE/hang"
SSH_REAL=$(command -v ssh)
# Any use of plain ssh by the launcher is recorded here and refused.
printf '#!/bin/bash\necho "ssh $*" >> %s/ssh.log\nexit 99\n' "$FAKE" > "$FAKE/path/ssh"
printf '#!/bin/bash\necho "$*" >> %s/agent.log\nexec %s -o BatchMode=yes "$@"\n' "$FAKE" "$SSH_REAL" > "$FAKE/agent/pbs_tmrsh"
printf '#!/bin/bash\necho "$*" >> %s/agent.log\nexit 1\n' "$FAKE" > "$FAKE/broken/pbs_tmrsh"
printf '#!/bin/bash\nexec sleep 1000\n' > "$FAKE/hang/pbs_tmrsh"
chmod +x "$FAKE"/path/* "$FAKE"/agent/* "$FAKE"/broken/* "$FAKE"/hang/*
# On the container's own disk: the Mac-backed /work mount reports every file as executable.
: > /tmp/not_executable; chmod -x /tmp/not_executable
AGENT=$FAKE/agent/pbs_tmrsh

{
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "source_git_rev=${SOURCE_GIT_REV:-unknown} source_dirty=${SOURCE_DIRTY:-unknown}"
    echo "environment=local Docker two-container simulation; pbs_tmrsh is a stand-in over ssh; not PBS, not Vanda, no measurement"
    mpirun --version 2>&1 | head -1
    sha256sum jobs/vanda/multinode_setup.sh jobs/vanda/allocation_check.sh jobs/vanda/gates.sh tests/jobs/launcher_selftest.sh
} > "$OUTROOT/provenance.txt"

check() {
    local what=$1; shift
    if "$@"; then echo "PASS $what" | tee -a "$RESULTS"; pass=$((pass + 1))
    else echo "FAIL $what" | tee -a "$RESULTS"; fail=$((fail + 1)); fi
}
has() { grep -Eq -- "$1" "$2"; }
no_plain_ssh() { [ ! -s "$FAKE/ssh.log" ]; }
two_clean_per_gate() {  # every cleanup section holds both nodes' CLEAN reports
    awk '/^== after /{if (s && c != 2) bad=1; s=1; c=0} / CLEAN$/{c++} END{if (!s || c != 2 || bad) exit 1}' "$1/cleanup.log"
}

# scenario NAME AGENT BODY: one simulated job. Sets SC_OUT, SC_RC, SC_SECS.
scenario() {
    local name=$1 agent=$2 body=$3 out=$OUTROOT/$1 t0
    mkdir -p "$out/env"
    : > "$FAKE/ssh.log"; : > "$FAKE/agent.log"
    t0=$(date +%s)
    PATH=$FAKE/path:$PATH PBS_JOBID=launcher.sim LEFTOVER_JOBID=launcher.sim RUN_ID=launcher-$name OUT=$out JOB_START_S=$t0 \
        WALLTIME_S=${WALLTIME_S:-900} LEFTOVER_PATTERN=launcher-test/ ALLOW_NO_MODULES=1 REQUIRE_PBS_TM=1 \
        MPI_LAUNCHER=pbs_tmrsh PBS_TMRSH_BIN=$agent REMOTE_EXEC=${REMOTE_EXEC:-$AGENT} GATE_GRACE_S=5 BIN=$BIN \
        bash -c "source jobs/vanda/multinode_setup.sh; source jobs/vanda/gates.sh; $body
                 verdict >/dev/null && finish 0; finish 1" > "$out/stdout" 2>&1
    SC_RC=$?
    SC_OUT=$out
    SC_SECS=$(( $(date +%s) - t0 ))
    echo "== scenario $name exit=$SC_RC elapsed_s=$SC_SECS" | tee -a "$RESULTS"
}

# 1. Repeated mappings across both nodes, three times each, and where each rank ran.
body='make BIN=$BIN MPIRUN="$MPI" $BIN/placement_check >/dev/null
for r in 1 2 3; do
    gate pe4_2x4_$r 60 required make BIN=$BIN MPIRUN="$MPI" placement PLACE_NODES=2 PLACE_R=2 PLACE_T=4
    gate pe2_4x2_$r 60 required make BIN=$BIN MPIRUN="$MPI" placement PLACE_NODES=2 PLACE_R=4 PLACE_T=2
    gate pe1_8x1_$r 60 required make BIN=$BIN MPIRUN="$MPI" placement PLACE_NODES=2 PLACE_R=8 PLACE_T=1
    gate pe8_1x8_$r 60 required make BIN=$BIN MPIRUN="$MPI" placement PLACE_NODES=2 PLACE_R=1 PLACE_T=8
done'
scenario repeated "$AGENT" "$body"
check "repeated: 12 of 12 mappings pass" test "$(grep -c ' PASS rc=0' "$SC_OUT/gates.txt")" -eq 12
check "repeated: verdict PASS"           test "$SC_RC" -eq 0
check "repeated: launched through the agent" test -s "$FAKE/agent.log"
check "repeated: plain ssh never called" no_plain_ssh
check "repeated: both nodes clean after every gate" two_clean_per_gate "$SC_OUT"
for g in pe4_2x4_1 pe2_4x2_2 pe1_8x1_3 pe8_1x8_3; do
    check "repeated: $g ran on both hosts, bound" bash -c \
        "grep -q 'host=node1' '$SC_OUT/gate_$g.log' && grep -q 'host=node2' '$SC_OUT/gate_$g.log' && grep -Eq 'PASS' '$SC_OUT/gate_$g.log'"
done

# 2. The launch itself: agent in the command line, no daemonize, daemon stays attached.
body='gate launch 60 required bash -c "$MPI --mca plm_base_verbose 10 -np 2 --map-by ppr:1:node --bind-to none hostname 2> $OUT/verbose.txt"
gate attached 60 required bash -c "$MPI -np 2 --map-by ppr:1:node --bind-to none sleep 6 & sleep 3; /usr/bin/ssh -o BatchMode=yes node2 \"ps -C orted -o ppid= \" > $OUT/orted_ppid.txt; wait"'
scenario launch "$AGENT" "$body"
check "launch: verdict PASS"              test "$SC_RC" -eq 0
check "launch: rsh launcher selected"     has 'plm\) Selected component \[rsh\]' "$SC_OUT/verbose.txt"
check "launch: agent runs the daemons"    has 'pbs_tmrsh <template>' "$SC_OUT/verbose.txt"
check "launch: no --daemonize"            bash -c "! grep -q -- '--daemonize' '$SC_OUT/verbose.txt'"
check "launch: remote daemon still has its starter as parent" bash -c \
    "test -s '$SC_OUT/orted_ppid.txt' && ! grep -Eq '^ *1 *$' '$SC_OUT/orted_ppid.txt'"

# 3. Refusals before anything starts: no agent, or one that cannot run.
refuse() {  # refuse NAME BIN
    local out=$OUTROOT/refuse_$1 rc
    mkdir -p "$out/env"
    : > "$FAKE/ssh.log"
    PATH=$FAKE/path:$PATH OUT=$out ALLOW_NO_MODULES=1 MPI_LAUNCHER=pbs_tmrsh PBS_TMRSH_BIN=$2 \
        bash -c 'source jobs/vanda/multinode_setup.sh; echo reached_after_setup' > "$out/stdout" 2>&1
    rc=$?
    check "refuse $1: exit 3"            test "$rc" -eq 3
    check "refuse $1: says no ssh fallback" has 'not falling back to ssh' "$out/stdout"
    check "refuse $1: stops at setup"    bash -c "! grep -q reached_after_setup '$out/stdout'"
    check "refuse $1: plain ssh never called" no_plain_ssh
}
refuse missing "$FAKE/does_not_exist"
refuse not_executable /tmp/not_executable
refuse empty ""
out=$OUTROOT/refuse_unknown; mkdir -p "$out/env"
OUT=$out ALLOW_NO_MODULES=1 MPI_LAUNCHER=ssh bash -c 'source jobs/vanda/multinode_setup.sh' > "$out/stdout" 2>&1
check "refuse MPI_LAUNCHER=ssh: exit 3" test "$?" -eq 3

# 4. The agent fails on launch: reported as a failure, no ssh, later gates still run.
scenario agent_fails "$FAKE/broken/pbs_tmrsh" 'gate launch 30 required $MPI -np 2 --map-by ppr:1:node --bind-to none hostname
gate after 20 required true'
check "agent_fails: gate FAIL"            has '^launch FAIL ' "$SC_OUT/gates.txt"
check "agent_fails: later gate ran"       has '^after PASS ' "$SC_OUT/gates.txt"
check "agent_fails: verdict FAIL"         test "$SC_RC" -eq 1
check "agent_fails: plain ssh never called" no_plain_ssh
check "agent_fails: agent was tried"      test -s "$FAKE/agent.log"

# 5. The agent hangs on launch: bounded, snapshot taken, no fallback, job continues.
scenario agent_hangs "$FAKE/hang/pbs_tmrsh" 'gate launch 15 required $MPI -np 2 --map-by ppr:1:node --bind-to none hostname
gate after 20 required true'
check "agent_hangs: gate did not pass"    bash -c "grep -Eq '^launch (TIMEOUT|FAIL) ' '$SC_OUT/gates.txt'"
check "agent_hangs: later gate ran"       has '^after PASS ' "$SC_OUT/gates.txt"
check "agent_hangs: bounded (${SC_SECS}s <= 90)" test "$SC_SECS" -le 90
check "agent_hangs: plain ssh never called" no_plain_ssh
check "agent_hangs: both nodes clean"     two_clean_per_gate "$SC_OUT"

# 6. Cleanup cannot reach the other node: the job stops, and does not use ssh instead.
REMOTE_EXEC=$FAKE/broken/pbs_tmrsh scenario cleanup_broken "$AGENT" 'gate first 20 required true
gate second 20 required true'
check "cleanup_broken: exit 6"            test "$SC_RC" -eq 6
check "cleanup_broken: second not run"    bash -c "! grep -q '^second ' '$SC_OUT/gates.txt'"
check "cleanup_broken: plain ssh never called" no_plain_ssh

# 7. A rank exits nonzero, and a remote daemon dies mid-run: both bounded and reported.
scenario rank_fails "$AGENT" 'gate rank 30 required $MPI -np 2 --map-by ppr:1:node --bind-to none bash -c "exit 3"'
check "rank_fails: gate FAIL"             has '^rank FAIL ' "$SC_OUT/gates.txt"
check "rank_fails: both nodes clean"      two_clean_per_gate "$SC_OUT"
scenario daemon_dies "$AGENT" 'gate dies 40 required bash -c "$MPI -np 2 --map-by ppr:1:node --bind-to none bash -c \"exec -a launcher-test/rank sleep 300\" & sleep 3; /usr/bin/ssh -o BatchMode=yes node2 pkill -KILL -u $(id -un) orted; wait \$!"
gate after 20 required true'
check "daemon_dies: gate did not pass"    bash -c "grep -Eq '^dies (TIMEOUT|FAIL) ' '$SC_OUT/gates.txt'"
check "daemon_dies: later gate ran"       has '^after PASS ' "$SC_OUT/gates.txt"
check "daemon_dies: both nodes clean"     two_clean_per_gate "$SC_OUT"
# A rank carrying the job's path is owned by the job, as a real binary is.
check "daemon_dies: no rank left on node2" bash -c "! /usr/bin/ssh -o BatchMode=yes node2 pgrep -u $ME -f '[l]auncher-test/rank' >/dev/null"
check "daemon_dies: no rank left on node1" bash -c "! pgrep -u $ME -f '[l]auncher-test/rank' >/dev/null"

# 8. The allocation check, on canned launch records. Values follow job 1513781.
HEAD=$(hostname -s)
ALLOWED=0,2,4,6,8,10,12,14
good_nodes="node $HEAD allowed=$ALLOWED arch=x86_64 model=m ancestry=bash<orted<pbs_shepherd<pbs_mom pbs_jobid_env=yes isa=a
node other allowed=$ALLOWED arch=x86_64 model=m ancestry=bash<orted<pbs_shepherd<pbs_mom pbs_jobid_env=yes isa=a"
good_log='[h:1] mca:base:select:(  plm) Selected component [rsh]
[h:1] [[1,0],0] plm:rsh: final template argv:
	/opt/pbs/bin/pbs_tmrsh <template> orted -mca ess env'
alloc() {  # alloc NAME WANT_RC LAUNCH NODES_TEXT LOG_TEXT
    local out=$OUTROOT/alloc_$1 rc
    mkdir -p "$out/env"
    printf '%s\n' "$4" > "$out/in_nodes"; printf '%s\n' "$5" > "$out/in_log"
    printf '#!/bin/bash\ncat %s/in_nodes\ncat %s/in_log >&2\n' "$out" "$out" > "$out/stub"
    chmod +x "$out/stub"
    MPI=$out/stub ALLOC_EXPECT_NODES=${ALLOC_N:-2} ALLOC_CPUS_PER_NODE=8 bash jobs/vanda/allocation_check.sh "$out" "${ALLOC_N:-2}" "$3" > "$out/stdout" 2>&1
    rc=$?
    check "allocation $1: exit $2 (got $rc)" test "$rc" -eq "$2"
}
alloc good 0 pbs_tmrsh "$good_nodes" "$good_log"
sshd_nodes="node $HEAD allowed=$ALLOWED arch=x86_64 model=m ancestry=bash<orted pbs_jobid_env=yes isa=a
node other allowed=$ALLOWED arch=x86_64 model=m ancestry=bash<orted<sshd<systemd pbs_jobid_env=yes isa=a"
alloc via_sshd 1 pbs_tmrsh "$sshd_nodes" "$good_log"
alloc ssh_agent 1 pbs_tmrsh "$good_nodes" "${good_log/\/opt\/pbs\/bin\/pbs_tmrsh/\/usr\/bin\/ssh}"
alloc daemonized 1 pbs_tmrsh "$good_nodes" "${good_log/orted -mca/orted --daemonize -mca}"
alloc tm_selected 1 pbs_tmrsh "$good_nodes" "${good_log/\[rsh\]/[tm]}"
alloc no_log 1 pbs_tmrsh "$good_nodes" "nothing"
alloc template_wording_absent 0 pbs_tmrsh "$good_nodes" "[h:1] mca:base:select:(  plm) Selected component [rsh]"
alloc daemons_also_log 0 pbs_tmrsh "$good_nodes" "$good_log
[o:2] mca:base:select:(  plm) Selected component [rsh]"
alloc daemonized_leaves_pbs 1 pbs_tmrsh "${good_nodes//<pbs_shepherd<pbs_mom/<systemd}" "[h:1] mca:base:select:(  plm) Selected component [rsh]"
alloc outside_pbs 1 pbs_tmrsh "${good_nodes//<pbs_shepherd<pbs_mom/<systemd}" "$good_log"
alloc no_jobid 1 pbs_tmrsh "${good_nodes//pbs_jobid_env=yes/pbs_jobid_env=no}" "$good_log"
alloc wrong_cpuset 1 pbs_tmrsh "${good_nodes//$ALLOWED/0,2,4,6}" "$good_log"
alloc one_host 1 pbs_tmrsh "node $HEAD allowed=$ALLOWED arch=x86_64 model=m ancestry=bash pbs_jobid_env=yes isa=a" "$good_log"
tm_log='[h:1] mca:base:select:(  plm) Selected component [tm]'
alloc tm_mode_unchanged 0 tm "$good_nodes" "$tm_log"

# 9. One node. Values follow job 1519637, whose only process ran under a daemon it started.
one_node="node $HEAD allowed=$ALLOWED arch=x86_64 model=m ancestry=bash<orted<pbs_shepherd<pbs_mom pbs_jobid_env=yes isa=a"
ALLOC_N=1 alloc one_good 0 pbs_tmrsh "$one_node" "$good_log"
ALLOC_N=1 alloc one_no_daemon 0 pbs_tmrsh "${one_node/orted/mpirun}" "$good_log"
ALLOC_N=1 alloc one_tm_selected 1 pbs_tmrsh "$one_node" "$tm_log"
ALLOC_N=1 alloc one_via_sshd 1 pbs_tmrsh "${one_node/pbs_shepherd<pbs_mom/sshd<systemd}" "$good_log"
ALLOC_N=1 alloc one_outside_pbs 1 pbs_tmrsh "${one_node/pbs_shepherd<pbs_mom/systemd}" "$good_log"
ALLOC_N=1 alloc one_no_jobid 1 pbs_tmrsh "${one_node/pbs_jobid_env=yes/pbs_jobid_env=no}" "$good_log"
ALLOC_N=1 alloc hostfile_rsh 0 hostfile "$one_node" "${good_log}"
# The launch job 1519637 passed here and then hung on: hostfile, with tm selected.
ALLOC_N=1 alloc hostfile_tm_1519637 1 hostfile "$one_node" "$tm_log"

# 10. One host, the shape of job 1519637: mpirun is not on its only node, so all ranks run
# under one daemon started through the agent. Docker cannot show tm itself hanging. The
# stand-in agent goes through sshd, so the allocation check must refuse it, and only for that.
one_file=$OUTROOT/nodefile_one
for i in 1 2 3 4 5 6 7 8; do echo node2; done > "$one_file"
# The measurement runner on this launch line, as measure.sh calls it: the two-node test
# cases, on one node.
one_manifest=$OUTROOT/one_host_manifest.json
python3 -c 'import json, sys
d = json.load(open("manifests/local_sim_twonode.json"))
d["id"], d["defaults"]["nodes"] = "local-one-host", 1
json.dump(d, open(sys.argv[1], "w"))' "$one_manifest"
body='export MPI; make BIN=$BIN MPIRUN="$MPI" $BIN/placement_check >/dev/null
gate allocation 60 optional env ALLOC_EXPECT_NODES=1 ALLOC_CPUS_PER_NODE=8 bash jobs/vanda/allocation_check.sh "$OUT" "$NODES" "$LAUNCH"
gate pe4_2x4 60 required make BIN=$BIN MPIRUN="$MPI" placement PLACE_NODES=1 PLACE_R=2 PLACE_T=4
gate pe1_8x1 60 required make BIN=$BIN MPIRUN="$MPI" placement PLACE_NODES=1 PLACE_R=8 PLACE_T=1
gate pe8_1x8 60 required make BIN=$BIN MPIRUN="$MPI" placement PLACE_NODES=1 PLACE_R=1 PLACE_T=8
make BIN=$BIN all hybrid >/dev/null
gate runs 240 required python3 scripts/run_manifest.py '"$one_manifest"' --out "$OUT" --bin "$BIN" --mpirun "$MPI" \
    --cores-per-node 8 --nodes 1 --first-cpu 0 --deadline $(( $(date +%s) + 200 )) --scale 1'
PBS_NODEFILE=$one_file scenario one_host "$AGENT" "$body"
check "one_host: verdict PASS"              test "$SC_RC" -eq 0
check "one_host: daemon started by the agent" test -s "$FAKE/agent.log"
check "one_host: plain ssh never called"    no_plain_ssh
check "one_host: rsh selected, one host"    has "1_hosts=ok cpus_8_each=ok arch_x86_64=ok plm_rsh_selected=ok" "$SC_OUT/gate_allocation.log"
check "one_host: stand-in refused for sshd" has "remote_orted_not_sshd=FAIL remote_under_pbs_mom=FAIL" "$SC_OUT/gate_allocation.log"
check "one_host: every measurement run passed" has '^runs PASS ' "$SC_OUT/gates.txt"
check "one_host: ranks ran on the one host" bash -c "grep -q 'host=node2' '$SC_OUT/gate_pe4_2x4.log' && ! grep -q 'host=node1' '$SC_OUT/gate_pe4_2x4.log'"

# 11. A one-node job script needs pbs_tmrsh unless simulation is asked for by name.
wrap_out=$OUTROOT/wrapper
for w in pilot bench1; do
    mkdir -p "$wrap_out"
    PATH=$FAKE/path:$PATH PBS_O_WORKDIR=$PWD PBS_JOBID=wrap$w.sim ALLOW_NO_MODULES=1 RUNS_ROOT=$wrap_out \
        PBS_NODEFILE=$one_file bash "jobs/vanda/$w.pbs" > "$wrap_out/$w.stdout" 2>&1
    rc=$?
    check "wrapper $w: no pbs_tmrsh, exit 3 (got $rc)" test "$rc" -eq 3
    check "wrapper $w: says no ssh fallback" has 'not falling back to ssh' "$wrap_out/$w.stdout"
    check "wrapper $w: stops before any gate" bash -c "! grep -q '===== gate' '$wrap_out/$w.stdout'"
done

rm -rf "$FAKE" "$BIN"
echo "launcher_selftest passed=$pass failed=$fail $([ "$fail" -eq 0 ] && echo PASS || echo FAIL)" | tee -a "$RESULTS"
[ "$fail" -eq 0 ]
