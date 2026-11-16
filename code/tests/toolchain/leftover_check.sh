#!/bin/bash
# After a gate, on one node: find this user's processes the gate may have left
# behind, terminate only those positively owned by this job, and report.
#   leftover_check.sh PATTERN        (PATTERN: job-unique, e.g. vanda-1507215/)
# Environment: LEFTOVER_JOBID (this job's PBS_JOBID, may be empty) and
# P3_RUN_MARKER (unique per job run, exported by jobs/vanda/gates.sh).
#
# A candidate is one of this user's processes that is an MPI launcher/daemon
# (mpirun, orted, ...) or whose command line contains PATTERN. Each is
# classified by its own environment and command line:
#   foreign       PBS_JOBID set and different from ours: another job; ignored
#   owned         PBS_JOBID or P3_RUN_MARKER equals ours, or the command line
#                 contains PATTERN (a path only this job creates)
#   unattributed  anything else, e.g. an MPI daemon with neither marker or an
#                 unreadable environment: never killed
# Owned processes get TERM, then KILL, and must be gone. Unattributed ones are
# re-checked after a short wait (exiting daemons vanish); if any remain the
# result is UNCERTAIN, because this job cannot show they are not its own.
# Exit 0 only for CLEAN (no owned process remaining and none unattributed).
# The checker's own ancestry is excluded.
pattern=${1:?pattern}
host=$(hostname -s)
jobid=${LEFTOVER_JOBID:-}
marker=${P3_RUN_MARKER:-}

ancestry=" "
p=$$
while [ "${p:-0}" -gt 1 ] && [ -r /proc/$p/stat ]; do
    ancestry="$ancestry$p "
    p=$(sed 's/.*) //' /proc/$p/stat | cut -d' ' -f2)
done

env_of() {  # env_of DIR NAME: value of NAME in that process's environment
    tr '\0' '\n' < "$1/environ" 2>/dev/null | sed -n "s/^$2=//p" | head -1
}

# scan CLASS: print "pid comm cmd" for candidates of that class
scan() {
    local want=$1 d pid comm cmd penv_job penv_mark readable class
    for d in /proc/[0-9]*; do
        [ -O "$d" ] || continue
        pid=${d#/proc/}
        case "$ancestry" in *" $pid "*) continue ;; esac
        comm=$(cat "$d/comm" 2>/dev/null) || continue
        cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
        case "$cmd" in *leftover_check.sh*) continue ;; esac
        local mpi_like=0 has_pattern=0
        case "$comm" in mpirun|mpiexec|orterun|orted|prterun|prted) mpi_like=1 ;; esac
        case "$cmd" in *"$pattern"*) has_pattern=1 ;; esac
        [ $mpi_like = 1 ] || [ $has_pattern = 1 ] || continue
        readable=0
        [ -r "$d/environ" ] && tr '\0' '\n' < "$d/environ" >/dev/null 2>&1 && readable=1
        penv_job=$(env_of "$d" PBS_JOBID)
        penv_mark=$(env_of "$d" P3_RUN_MARKER)
        if [ -n "$penv_job" ] && [ "$penv_job" != "$jobid" ]; then
            class=foreign
        elif { [ -n "$jobid" ] && [ "$penv_job" = "$jobid" ]; } \
             || { [ -n "$marker" ] && [ "$penv_mark" = "$marker" ]; } \
             || [ $has_pattern = 1 ]; then
            class=owned
        else
            class=unattributed
        fi
        [ $readable = 1 ] || [ $class = owned ] || class=unattributed
        [ "$class" = "$want" ] && echo "$pid $comm ${cmd:0:160}"
    done
}

count() { printf '%s' "$1" | grep -c .; }
pids() { printf '%s\n' "$1" | cut -d' ' -f1; }

owned=$(scan owned)
n_owned=$(count "$owned")
if [ "$n_owned" -gt 0 ]; then
    printf '%s\n' "$owned" | sed "s/^/leftover host=$host owned: /"
    kill -TERM $(pids "$owned") 2>/dev/null
    sleep 3
    rest=$(scan owned)
    [ -n "$rest" ] && kill -KILL $(pids "$rest") 2>/dev/null && sleep 2
fi
remaining=$(scan owned)
n_rem=$(count "$remaining")
[ "$n_rem" -gt 0 ] && printf '%s\n' "$remaining" | sed "s/^/leftover host=$host owned remaining: /"

foreign=$(scan foreign)
[ -n "$foreign" ] && printf '%s\n' "$foreign" | sed "s/^/leftover host=$host foreign (ignored): /"
unattr=$(scan unattributed)
if [ -n "$unattr" ]; then
    sleep 3
    unattr=$(scan unattributed)
fi
[ -n "$unattr" ] && printf '%s\n' "$unattr" | sed "s/^/leftover host=$host unattributed (not killed): /"
n_unattr=$(count "$unattr")

if [ "$n_rem" -gt 0 ]; then verdict=DIRTY
elif [ "$n_unattr" -gt 0 ]; then verdict=UNCERTAIN
else verdict=CLEAN; fi
echo "leftover_check host=$host owned_found=$n_owned owned_remaining=$n_rem foreign=$(count "$foreign") unattributed=$n_unattr $verdict"
[ "$verdict" = CLEAN ]
