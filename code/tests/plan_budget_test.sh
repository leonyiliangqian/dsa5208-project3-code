#!/bin/bash
# plan_budget.py must pass the plans to be proposed and refuse ones that would not fit.
#   bash tests/plan_budget_test.sh     (run from code/)
set -uo pipefail
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
failures=0
expect() { # NAME WANT_RC CMD...
    local name=$1 want=$2; shift 2
    "$@" > "$work/out" 2>&1; local rc=$?
    if [ "$rc" = "$want" ]; then echo "budget $name PASS"; else echo "budget $name FAIL rc=$rc"; cat "$work/out"; failures=$((failures + 1)); fi
}
plan() { python3 scripts/plan_budget.py "$@"; }
expect pilot4 0 plan manifests/pilot_four_node.json jobs/vanda/pilot4.pbs --rate 1
expect pilot2 0 plan manifests/pilot_two_node.json jobs/vanda/pilot2.pbs
expect bench4 0 plan manifests/studies_four_node.json jobs/vanda/bench4.pbs --rate 1
expect bench1 0 plan manifests/studies_single_node.json jobs/vanda/bench1.pbs
sed 's/mem=240gb/mem=100gb/' jobs/vanda/pilot4.pbs > "$work/lowmem.pbs"
expect control_memory 1 plan manifests/pilot_four_node.json "$work/lowmem.pbs"
expect control_memory_reason 0 grep -q "over the 100 GB requested" <(plan manifests/pilot_four_node.json "$work/lowmem.pbs")
sed 's/walltime=00:45:00/walltime=00:20:00/' jobs/vanda/pilot4.pbs > "$work/short.pbs"
expect control_walltime 1 plan manifests/pilot_four_node.json "$work/short.pbs"
sed -E 's/"timeout_s": [0-9]+/"timeout_s": 5/' manifests/pilot_four_node.json > "$work/tight.json"
expect control_limit 1 plan "$work/tight.json" jobs/vanda/pilot4.pbs
expect charge_stated 0 grep -q '192 if charged' <(plan manifests/pilot_four_node.json jobs/vanda/pilot4.pbs --rate 1)
echo "$([ "$failures" = 0 ] && echo PASS || echo FAIL): plan budget, $failures failed"
[ "$failures" = 0 ]
