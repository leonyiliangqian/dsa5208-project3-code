#!/bin/bash
# Run a command with its output appended to a file, keeping its exit status.
#   run_to_file.sh FILE CMD...
# For remote execution: pbs_tmrsh sends remote output to the PBS job's output
# file instead of its own stdout, so the caller reads FILE on the shared
# filesystem. See remote_capture in jobs/vanda/remote_capture.sh.
out=${1:?file}
shift
"$@" >> "$out" 2>&1
