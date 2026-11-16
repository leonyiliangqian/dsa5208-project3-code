#!/bin/bash
# Started by gate() in its own process group (setsid): sleep DELAY seconds,
# then append a process snapshot of every node of the job to
# OUT/gate_LABEL.snapshot (tests/toolchain/proc_snapshot.sh). Each node is
# bounded by SNAPSHOT_NODE_S (+2 s kill grace), so the whole watcher ends at
# most DELAY + nodes * (SNAPSHOT_NODE_S + 2) seconds after it starts. gate()
# kills the group when the gate finishes first.
#   snapshot_watch.sh DELAY LABEL
# Environment: OUT, PBS_NODEFILE, LEFTOVER_PATTERN, REMOTE_EXEC, SNAPSHOT_NODE_S.
delay=${1:?delay} label=${2:?label}
sleep "$delay"
short() { echo "${1%%.*}" | tr '[:upper:]' '[:lower:]'; }
source jobs/vanda/remote_capture.sh
me=$(short "$(hostname)")
for h in $(sort -u "${PBS_NODEFILE:-/dev/null}"); do
    if [ "$(short "$h")" = "$me" ]; then
        timeout -k 2 "$SNAPSHOT_NODE_S" bash tests/toolchain/proc_snapshot.sh "$LEFTOVER_PATTERN"
    else
        remote_capture 2 "$SNAPSHOT_NODE_S" "$h" \
            bash "$PWD/tests/toolchain/proc_snapshot.sh" "$LEFTOVER_PATTERN"
    fi
done >> "$OUT/gate_$label.snapshot" 2>&1
