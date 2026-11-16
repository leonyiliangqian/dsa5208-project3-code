# Run a command on another node of the job and print its output here (source it;
# not executable).
#   remote_capture KILL_GRACE_S LIMIT_S HOST CMD...
# Returns the exit status of REMOTE_EXEC. pbs_tmrsh writes remote output to the
# PBS job's output file, not to its own stdout (job 1513781 lost a node's report
# this way), so the remote side writes to a file under OUT and it is read back
# here. OUT and this directory must be on a filesystem shared by the nodes. A
# report that never arrives prints nothing, which callers treat as a failure.
# Expects OUT, REMOTE_EXEC and short().
remote_capture() {
    local grace=$1 limit=$2 host=$3 dir tmp rc
    shift 3
    case $OUT in /*) dir=$OUT ;; *) dir=$PWD/$OUT ;; esac
    tmp=$dir/.remote.$(short "$host").$$.$RANDOM
    timeout -k "$grace" "$limit" $REMOTE_EXEC "$host" \
        bash "$PWD/tests/toolchain/run_to_file.sh" "$tmp" "$@"
    rc=$?
    [ -f "$tmp" ] && cat "$tmp"
    rm -f "$tmp"
    return "$rc"
}
