#!/bin/bash
# Read-only checks on the Vanda login node before submitting any job. Prints the
# CPU queue limits, candidate modules and allocation information needed to pick
# -q and -P. Submits nothing and compiles nothing.
#   cd ~/dsa5208-p3/code && bash jobs/vanda/preflight.sh
set -u
out=artifacts/cluster/discovery/preflight_$(hostname -s)_$(date +%Y%m%dT%H%M%S).txt
mkdir -p "$(dirname "$out")"
type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1
{
    echo "== date/host"; date -Is; hostname -f
    echo "== source"; git rev-parse HEAD 2>&1; git status --porcelain -- . | head
    for q in auto_free cpu_serial cpu_parallel; do
        echo "== qstat -Qf $q"
        qstat -Qf "$q" 2>&1 | grep -iE 'queue_type|enabled|started|resources_(max|min|default|available)|max_run|max_queued|acl|route_destinations|charge|from_route_only'
    done
    echo "== candidate modules"
    module -t avail OpenMPI 2>&1 | head -20
    module -t avail Python 2>&1 | grep -iE 'GCCcore-13|^Python' | head -20
    echo "== module show OpenMPI/4.1.6-GCC-13.2.0"
    module show OpenMPI/4.1.6-GCC-13.2.0 2>&1 | grep -iE 'load|prepend_path\("PATH"|whatis' | head -15
    echo "== allocations (needs a prior 'amgr login'; output only, nothing changes)"
    hpc project 2>&1 | head -40
    echo "== my jobs"; qstat -u "$USER" 2>&1 | head
} > "$out" 2>&1
cat "$out"
echo; echo "saved to $out"
