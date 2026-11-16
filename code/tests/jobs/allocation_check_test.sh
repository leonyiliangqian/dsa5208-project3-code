#!/bin/bash
# Regression test for jobs/vanda/allocation_check.sh without MPI or PBS: a
# stand-in for mpirun replays node reports and a launcher log, so host-name
# case, Open MPI's head-node alias, mixed CPU models and the CPU count are
# checked against known answers. Where the Vanda evidence of jobs 1526470
# (pilot2, mixed models) and 1526501 (bench1, one node) is present, their
# own node reports are replayed too. Local checks only: they show what the
# script decides on given input, not how Vanda behaves.
#   bash tests/jobs/allocation_check_test.sh
set -u
cd "$(dirname "$0")/../.."
work=$(mktemp -d "${TMPDIR:-/tmp}/alloc_test.XXXXXX")
pass=0 fail=0
check() {
    local what=$1; shift
    if "$@"; then echo "PASS $what"; pass=$((pass + 1)); else echo "FAIL $what"; fail=$((fail + 1)); fi
}
# The stand-in prints $FAKE_NODES on stdout and $FAKE_LOG on stderr, as mpirun
# would with node_report.sh on every node and plm/ras verbose logging.
cat > "$work/fake_mpirun" <<'EOF'
#!/bin/sh
cat "$FAKE_NODES"; cat "$FAKE_LOG" >&2
EOF
chmod +x "$work/fake_mpirun"

SR='a.b,avx512f'   # identical ISA lists unless a case says otherwise
GOLD='Intel(R)_Xeon(R)_Gold_6438N' PLAT='Intel(R)_Xeon(R)_Platinum_8452Y'
UNDER_MOM='bash<orted<pbs_shepherd<pbs_mom'
node() {  # node NAME CPUS MODEL ANCESTRY
    echo "node $1 allowed=$2 arch=x86_64 model=$3 ancestry=$4 pbs_jobid_env=yes isa=$SR"
}
log() {  # log NODE_LIST: plm rsh selected, pbs_tmrsh template with the given node list
    cat <<EOF
[x:1] mca:base:select:(  plm) Selected component [rsh]
[x:1] mca:base:select:(  ras) Selected component [tm]
[x:1] [[1,0],0] plm:rsh: final template argv:
	/opt/pbs/bin/pbs_tmrsh <template> orted -mca orte_node_regex "$1" -mca orte_hnp_uri "x"
EOF
}
# run NAME NODES HEAD CPUS SAME_MODEL: writes $work/NAME/out, sets RC.
run() {
    local name=$1 nodes=$2 head=$3 cpus=$4 same=$5 d=$work/$1
    mkdir -p "$d/env"
    FAKE_NODES=$d/nodes.in FAKE_LOG=$d/log.in MPI="$work/fake_mpirun" ALLOC_HEAD=$head \
        ALLOC_EXPECT_NODES=$nodes ALLOC_CPUS_PER_NODE=$cpus ALLOC_SAME_MODEL=$same \
        bash jobs/vanda/allocation_check.sh "$d" "$nodes" pbs_tmrsh > "$d/out" 2>&1
    RC=$?
}
setup() { mkdir -p "$work/$1"; cat > "$work/$1/nodes.in"; }

# 1. One node as in job 1526501: hostname CN-158, PBS name cn-158, and Open MPI
#    listing both. Passes; the alias is recorded, not failed.
node CN-158 0-56,58,60,62,64,66,68,70 "$PLAT" "$UNDER_MOM" | setup one_alias
log "CN-[3:158],cn-[3:158]@0(2)" > "$work/one_alias/log.in"
run one_alias 1 CN-158 64 1
check "one node, case alias: PASS"                 test "$RC" -eq 0
check "one node, case alias: alias recorded"       grep -q "case_aliases=\['cn-158'\]" "$work/one_alias/out"

# 2. Two nodes with the head given in PBS form (lower case, domain): the head is
#    still recognised, so only the other node must be under orted and pbs_mom.
{ node IN-004 0-63 "$GOLD" 'bash<mpirun<bash<pbs_mom'; node IN-005 0-63 "$GOLD" "$UNDER_MOM"; } | setup two_case_head
log "IN-[3:4],in-[3:4],in-[3:5]@0(3)" > "$work/two_case_head/log.in"
run two_case_head 2 in-004.stdct.loc 64 1
check "two nodes, head named in PBS case: PASS"    test "$RC" -eq 0

# 3. One host reported under two spellings is one host, not two.
{ node CN-158 0-63 "$PLAT" "$UNDER_MOM"; node cn-158 0-63 "$PLAT" "$UNDER_MOM"; } | setup case_duplicate
log "CN-[3:158],cn-[3:158]@0(2)" > "$work/case_duplicate/log.in"
run case_duplicate 2 CN-158 64 1
check "one host under two spellings: FAIL"         test "$RC" -ne 0
check "one host under two spellings: hosts check"  grep -q "2_hosts=FAIL" "$work/case_duplicate/out"

# 4. Mixed models as in job 1526470: a timing job (same model required) fails;
#    a correctness-only check would pass and record the mix.
{ node CN-158 0-56,58,60,62,64,66,68,70 "$PLAT" "$UNDER_MOM"; node IN-004 0-63 "$GOLD" 'bash<orted<pbs_shepherd<pbs_mom'; } | setup mixed
log "IN-[3:4],in-[3:4],cn-[3:158]@0(3)" > "$work/mixed/log.in"
run mixed 2 IN-004 64 1
check "mixed models, same model required: FAIL"    test "$RC" -ne 0
check "mixed models: one_cpu_model=FAIL"           grep -q "one_cpu_model=FAIL" "$work/mixed/out"
mkdir -p "$work/mixed_ok"; cp "$work/mixed/nodes.in" "$work/mixed/log.in" "$work/mixed_ok/"
run mixed_ok 2 IN-004 64 0
check "mixed models, not required: PASS"           test "$RC" -eq 0
check "mixed models, not required: recorded"       grep -q "mixed_models=True" "$work/mixed_ok/out"

# 5. Same model on both nodes passes; a 72-CPU cpuset where 64 are expected fails.
{ node IN-004 0-63 "$GOLD" "$UNDER_MOM"; node IN-005 0-63 "$GOLD" "$UNDER_MOM"; } | setup same
log "IN-[3:4],in-[3:4-5]@0(3)" > "$work/same/log.in"
run same 2 IN-004 64 1
check "same model on both nodes: PASS"             test "$RC" -eq 0
check "compressed node list expanded: alias in-004" grep -q "case_aliases=\['in-004'\]" "$work/same/out"
node CN-159 0-71 "$PLAT" "$UNDER_MOM" | setup wrong_cpus
log "CN-[3:159],cn-[3:159]@0(2)" > "$work/wrong_cpus/log.in"
run wrong_cpus 1 CN-159 64 1
check "72 CPUs where 64 expected: FAIL"            grep -q "cpus_64_each=FAIL" "$work/wrong_cpus/out"

# 6. Replay of the retrieved Vanda evidence, when present.
for spec in "pilot2-20261106T233758Z-1526470 2 IN-004 1" "bench1-20261106T130250Z-1526501 1 CN-158 0"; do
    set -- $spec
    ev=artifacts/cluster/runs/$1/env
    [ -s "$ev/nodes.txt" ] || { echo "SKIP replay $1 (evidence not present)"; continue; }
    mkdir -p "$work/$1"; cp "$ev/nodes.txt" "$work/$1/nodes.in"; cp "$ev/launch_verbose.txt" "$work/$1/log.in"
    run "$1" "$2" "$3" 64 1
    check "replay $1: exit $4 as on Vanda" test "$RC" -eq "$4"
done
check "replay 1526501: alias recorded" sh -c "[ ! -d '$work/bench1-20261106T130250Z-1526501' ] || grep -q \"case_aliases=\['cn-158'\]\" '$work/bench1-20261106T130250Z-1526501/out'"

echo "allocation_check_test: $pass passed, $fail failed (work: $work)"
[ "$fail" -eq 0 ]
