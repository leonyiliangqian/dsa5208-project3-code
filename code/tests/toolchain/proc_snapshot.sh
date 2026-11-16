#!/bin/bash
# Snapshot of this user's MPI-related processes on one node, taken just before
# a gate's time limit expires: process tree with state and kernel wait
# channel, per-thread wait channels, and, if gdb is installed, every thread's
# stack. Read-only apart from briefly attaching gdb.
#   proc_snapshot.sh PATTERN
pattern=${1:?pattern}
echo "== snapshot host=$(hostname -s) t=$(date +%s.%N | cut -c1-14)"
ps -u "$(id -u)" -o pid,ppid,pgid,stat,wchan:32,etime,args --forest | cut -c1-220
for d in /proc/[0-9]*; do
    [ -O "$d" ] || continue
    comm=$(cat "$d/comm" 2>/dev/null) || continue
    cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)
    case "$cmd" in *proc_snapshot.sh*) continue ;; esac
    case "$comm" in
        mpirun|orterun|orted) ;;
        *) case "$cmd" in *"$pattern"*) ;; *) continue ;; esac ;;
    esac
    pid=${d#/proc/}
    echo "-- pid=$pid comm=$comm $(grep -E '^(State|Threads|Cpus_allowed_list):' "$d/status" | tr '\n\t' '  ')"
    for t in "$d"/task/*; do
        printf '   tid=%s state=%s wchan=%s\n' "${t##*/}" \
            "$(sed 's/.*) //' "$t/stat" 2>/dev/null | cut -d' ' -f1)" "$(cat "$t/wchan" 2>/dev/null)"
    done
    if command -v gdb >/dev/null 2>&1; then
        timeout -k 1 4 gdb -batch -nx -p "$pid" -ex 'thread apply all bt 12' 2>&1 \
            | grep -E '^(Thread|#)' | head -60 | sed 's/^/   /'
    fi
done
