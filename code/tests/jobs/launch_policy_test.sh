#!/bin/bash
# Launch-policy regression test for the inter-node gate (job 1526501 failed
# there) and the other cluster launch paths. Run in the dev image:
#   docker/run.sh bash tests/jobs/launch_policy_test.sh
# 1. Static: every cluster launch states its binding; make internode expands to
#    the place_mpi placement for 1, 2 and 4 hosts and refuses a rank count that
#    does not divide over the hosts, before launching anything.
# 2. Local MPI: the inter-node probe exchanges data with 64 ranks on one node
#    (oversubscribed, unbound: Docker has fewer cores), runs bound with the
#    default placement on the container's cores, and fails when it finds fewer
#    hosts than expected. Local checks only; not evidence of Vanda behaviour.
set -u
cd "$(dirname "$0")/../.."
BIN=${BIN:-bin/launch-policy-test}
pass=0 fail=0
check() {
    local what=$1; shift
    if "$@"; then echo "PASS $what"; pass=$((pass + 1)); else echo "FAIL $what"; fail=$((fail + 1)); fi
}
has() { grep -q -e "$1" <<< "$2"; }

# 1. Static checks.
for spec in "64 1 64" "128 2 64" "256 4 64" "16 1 16"; do
    set -- $spec
    cmd=$(make -s -n BIN=x MPIRUN=mpirun internode INTERNODE_NP="$1" INTERNODE_HOSTS="$2" -o x/internode_check)
    check "internode $1 ranks on $2 hosts: ppr:$3:node:PE=1, bound" \
        has "-np $1 --map-by ppr:$3:node:PE=1 --bind-to core --report-bindings" "$cmd"
    check "internode $1 ranks on $2 hosts: no bare --map-by node" sh -c "! printf '%s' '$cmd' | grep -q -e '--map-by node '"
done
out=$(make -s BIN=x MPIRUN=false internode INTERNODE_NP=10 INTERNODE_HOSTS=4 -o x/internode_check 2>&1); rc=$?
check "internode refuses 10 ranks on 4 hosts before launching" test "$rc" -ne 0
check "internode says why" has "not a multiple of INTERNODE_HOSTS=4" "$out"
check "placement launch is bound" has "--bind-to core" "$(make -s -n BIN=x MPIRUN=mpirun placement -o x/placement_check)"
check "allocation probe states --bind-to none" grep -q -e "--bind-to none bash tests/toolchain/node_report.sh" jobs/vanda/allocation_check.sh
check "timed runs are bound (run_manifest.py)" grep -q -e ':node:PE={t} --bind-to core' scripts/run_manifest.py
check "bound correctness suite is bound (validate_hybrid.py)" grep -q -e '"--bind-to", "core"' tests/validate_hybrid.py
check "measure.sh runs internode through the make target" grep -q "internode INTERNODE_NP=" jobs/vanda/measure.sh

# 2. Local MPI checks (skipped without mpirun).
if command -v mpirun >/dev/null; then
    make -s BIN="$BIN" "$BIN/internode_check" > /dev/null
    BASE=mpirun; [ "$(id -u)" = 0 ] && BASE="mpirun --allow-run-as-root"
    MPI="$BASE --oversubscribe"
    out=$(timeout 300 make -s BIN="$BIN" MPIRUN="$MPI" internode INTERNODE_NP=64 INTERNODE_HOSTS=1 \
          INTERNODE_PLACE="--map-by ppr:64:node --bind-to none" 2>&1); rc=$?
    check "64 ranks on one node: exit 0" test "$rc" -eq 0
    check "64 ranks on one node: data verified" has "ranks=64 hosts=1 .*data_ok=1 .*expect_hosts=1 PASS" "$out"
    n=$(nproc)
    out=$(timeout 300 make -s BIN="$BIN" MPIRUN="$BASE" internode \
          INTERNODE_NP="$n" INTERNODE_HOSTS=1 2>&1); rc=$?
    check "$n ranks, default bound placement: exit 0" test "$rc" -eq 0
    check "$n ranks, default bound placement: bindings reported" test "$(grep -c 'MCW rank' <<< "$out")" -eq "$n"
    out=$(timeout 120 make -s BIN="$BIN" MPIRUN="$MPI" internode INTERNODE_NP=4 INTERNODE_HOSTS=2 \
          INTERNODE_PLACE="--map-by slot --bind-to none" 2>&1); rc=$?
    check "probe on one host expecting two: fails" test "$rc" -ne 0
    check "probe on one host expecting two: says FAIL" has "hosts=1 .*expect_hosts=2 FAIL" "$out"
else
    echo "SKIP local MPI checks (no mpirun)"
fi
echo "launch_policy_test: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
