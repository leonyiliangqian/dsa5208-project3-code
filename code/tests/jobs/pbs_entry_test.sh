#!/bin/bash
# A PBS job starts in $HOME, not where qsub ran. Job 1514774 died there in one second,
# because pilot.pbs sourced a relative path before changing directory; simulations
# started in code/ could not see it. Runs each wrapper from another directory against
# a stub measure.sh, checks every job script's order, and shows a broken copy is caught.
#   bash tests/jobs/pbs_entry_test.sh     (run from code/)
set -uo pipefail
code=$PWD
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
failures=0
result() { # NAME OK(0|1) [detail]
    if [ "$2" = 0 ]; then echo "entry $1 PASS ${3:-}"; else echo "entry $1 FAIL ${3:-}"; failures=$((failures + 1)); fi
}
# Exit 0 if the first relative source in the script comes after a cd to PBS_O_WORKDIR.
order_ok() {
    awk '/^cd "\$\{PBS_O_WORKDIR/ {c=NR; exit} /^(source|\.) +jobs\// {exit 1} END {exit c ? 0 : 1}' "$1"
}
# Runs a wrapper from an unrelated directory; the stub prints the directory it ran in.
entry_run() { # TREE SCRIPT
    ( cd "$work/home" && PBS_O_WORKDIR=$1 PBS_JOBID=entry.test bash "$1/jobs/vanda/$2" 2>&1 )
}
mkdir -p "$work/home" "$work/tree/jobs/vanda" "$work/broken/jobs/vanda"
printf 'echo "measure_ran_in=$PWD"\n' > "$work/tree/jobs/vanda/measure.sh"
cp "$work/tree/jobs/vanda/measure.sh" "$work/broken/jobs/vanda/measure.sh"

scripts=0
for f in jobs/vanda/*.pbs; do
    scripts=$((scripts + 1))
    order_ok "$f"; result "order_$(basename "$f" .pbs)" $?
done
[ "$scripts" -ge 8 ]; result script_count $? "scripts=$scripts"

wrappers=0
for f in jobs/vanda/*.pbs; do
    grep -q '^source jobs/vanda/measure.sh' "$f" || continue
    wrappers=$((wrappers + 1))
    name=$(basename "$f")
    cp "$f" "$work/tree/jobs/vanda/$name"
    out=$(entry_run "$work/tree" "$name")
    [ "$out" = "measure_ran_in=$work/tree" ] || [ "$out" = "measure_ran_in=$(cd "$work/tree" && pwd -P)" ]
    result "from_elsewhere_${name%.pbs}" $? "$out"
    out=$(cd "$work/home" && env -u PBS_O_WORKDIR bash "$work/tree/jobs/vanda/$name" 2>&1); rc=$?
    [ "$rc" != 0 ] && [[ "$out" != *measure_ran_in* ]]; result "no_workdir_${name%.pbs}" $? "rc=$rc"
done
[ "$wrappers" = 8 ]; result wrapper_count $? "wrappers=$wrappers"

# Negative control: the 1514774 failure, restored in a copy, must be caught both ways.
grep -v '^cd "\${PBS_O_WORKDIR' jobs/vanda/pilot.pbs > "$work/broken/jobs/vanda/pilot.pbs"
! order_ok "$work/broken/jobs/vanda/pilot.pbs"; result control_order_caught $?
out=$(entry_run "$work/broken" pilot.pbs)
[[ "$out" == *"No such file or directory"* ]]; result control_run_reproduces $? "$out"

echo "$([ "$failures" = 0 ] && echo PASS || echo FAIL): PBS entry directory, $failures failed"
[ "$failures" = 0 ]
