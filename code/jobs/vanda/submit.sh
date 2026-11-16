#!/bin/bash
# Submit a Project 3 job on Vanda with an explicit queue and allocation.
#
#   bash jobs/vanda/submit.sh correctness --queue auto_free --project <projectid>        # dry run
#   bash jobs/vanda/submit.sh correctness --queue auto_free --project <projectid> --yes  # submit
#   --var KEY=VALUE (repeatable) passes job variables, e.g. --var MPI_MODULE=OpenMPI/5.0.3-GCC-13.3.0
#
# The site charges personal credits when -P is omitted, so a project is required
# unless --no-project is given deliberately. Without --yes nothing is submitted;
# the exact qsub command is only printed. Run from the code/ directory.
set -euo pipefail
job=${1:?usage: submit.sh <correctness> --queue Q (--project P | --no-project) [--yes]}
shift
queue="" project="" no_project=0 yes=0 vars=""
while [ $# -gt 0 ]; do
    case $1 in
        --queue) queue=$2; shift 2 ;;
        --project) project=$2; shift 2 ;;
        --no-project) no_project=1; shift ;;
        --yes) yes=1; shift ;;
        --var) vars=${vars:+$vars,}$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
script=jobs/vanda/$job.pbs
[ -f "$script" ] || { echo "no such job script: $script" >&2; exit 2; }
[ -n "$queue" ] || { echo "--queue is required (check preflight output)" >&2; exit 2; }
if [ -z "$project" ] && [ "$no_project" != 1 ]; then
    echo "--project is required; omitting -P charges personal credits." >&2
    echo "Pass --no-project only if you have confirmed that is acceptable." >&2
    exit 2
fi
[ -z "$(git status --porcelain --untracked-files=no -- . 2>/dev/null)" ] || echo "WARNING: modified tracked files in code/; runs will record git_dirty=true"

mkdir -p artifacts/cluster/logs
cmd=(qsub -q "$queue")
[ -n "$project" ] && cmd+=(-P "$project")
# Compute nodes have no git, so record the commit here, on the login node.
rev=$(git rev-parse HEAD 2>/dev/null || echo unknown)
vars=${vars:+$vars,}SOURCE_GIT_REV=$rev
cmd+=(-v "$vars")
cmd+=(-o "$PWD/artifacts/cluster/logs/" "$script")
echo "command: ${cmd[*]}"
if [ "$yes" = 1 ]; then
    "${cmd[@]}"
else
    echo "dry run: add --yes to submit"
fi
