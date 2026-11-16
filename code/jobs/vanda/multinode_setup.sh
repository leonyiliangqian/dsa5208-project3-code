# Job setup shared by twonode.pbs, diag.pbs and measure.sh (source it from the
# code/ directory with OUT set): load modules, choose and force the PBS tm
# launcher (sets MPI, LAUNCH, NODES, PREFIX), and record the environment in
# OUT/env/environment.txt.

# ---------------------------------------------------------------- modules
MPI_MODULE=${MPI_MODULE:-OpenMPI/4.1.6-GCC-13.2.0}
PY_MODULE=${PY_MODULE:-Python/3.11.5-GCCcore-13.2.0}
type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1
if type module >/dev/null 2>&1; then
    module purge
    module load "$MPI_MODULE" || { echo "FATAL: cannot load $MPI_MODULE"; exit 3; }
elif [ "${ALLOW_NO_MODULES:-0}" != 1 ]; then
    echo "FATAL: no module command on this node"; exit 3
fi
if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 7))' 2>/dev/null; then
    type module >/dev/null 2>&1 && module load "$PY_MODULE"
fi
python3 -c 'import sys; sys.exit(sys.version_info < (3, 7))' \
    || { echo "FATAL: python3 >= 3.7 not available (tried $PY_MODULE)"; exit 3; }

# ---------------------------------------------------------------- launcher
# PBS-integrated launch is required, not just available. Job 1506687 showed
# that ompi_info lists the tm components even when they cannot be loaded
# (libpbs.so.0 lives in /opt/pbs/lib, off the default library path), and
# that run silently fell back to pbs_tmrsh with a hostfile. Here tm is forced
# for both node discovery (ras) and launch (plm), so mpirun fails rather than
# falling back; the allocation check then confirms from mpirun's own
# component-selection log and the remote process ancestry that tm was used.
# MPI_LAUNCHER chooses the launch route; a route that cannot be used stops the job.
#   tm         mpirun's PBS tm launcher. Job 1514029: every tm launch with a PE mapping
#              and core binding hung on Vanda, even hostname; cause unknown.
#   pbs_tmrsh  mpirun's rsh launcher with pbs_tmrsh as its only agent, so daemons
#              start as PBS tasks. One run passed in job 1514029; reliability unproven.
#              Daemons stay attached so PBS keeps owning them, and only mpirun starts
#              them, the launch path job 1514029 did not exercise beyond two nodes.
#   hostfile   node file only: local simulation (no PBS). It does not avoid tm under PBS:
#              with libpbs on the path mpirun still selects tm, even on one node (job 1519637).
# REQUIRE_PBS_TM=0 selects hostfile unless MPI_LAUNCHER says otherwise.
[ -d /opt/pbs/lib ] && export LD_LIBRARY_PATH=/opt/pbs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
REQUIRE_PBS_TM=${REQUIRE_PBS_TM:-1}
if [ "$REQUIRE_PBS_TM" = 1 ]; then MPI_LAUNCHER=${MPI_LAUNCHER:-tm}; else MPI_LAUNCHER=${MPI_LAUNCHER:-hostfile}; fi
NODES=$(sort -u "${PBS_NODEFILE:-/dev/null}" | wc -l)
PREFIX=${EBROOTOPENMPI:-$(dirname "$(dirname "$(command -v mpirun)")")}
case $MPI_LAUNCHER in
    tm)
        LAUNCH=tm; LAUNCH_FLAGS="--mca plm tm --mca ras tm" ;;
    pbs_tmrsh)
        PBS_TMRSH_BIN=${PBS_TMRSH_BIN:-$(command -v pbs_tmrsh || true)}
        if [ -z "$PBS_TMRSH_BIN" ] || [ ! -x "$PBS_TMRSH_BIN" ]; then
            echo "FATAL: MPI_LAUNCHER=pbs_tmrsh but no executable pbs_tmrsh ('$PBS_TMRSH_BIN'); not falling back to ssh"
            exit 3
        fi
        # Cleanup and snapshots use the same route.
        export REMOTE_EXEC=${REMOTE_EXEC:-$PBS_TMRSH_BIN}
        LAUNCH=pbs_tmrsh
        LAUNCH_FLAGS="--hostfile $PBS_NODEFILE --mca plm rsh --mca plm_rsh_agent $PBS_TMRSH_BIN"
        LAUNCH_FLAGS="$LAUNCH_FLAGS --mca plm_rsh_no_tree_spawn 1 --mca orte_leave_session_attached 1" ;;
    hostfile)
        LAUNCH=hostfile; LAUNCH_FLAGS="--hostfile $PBS_NODEFILE" ;;
    *)
        echo "FATAL: unknown MPI_LAUNCHER '$MPI_LAUNCHER'"; exit 3 ;;
esac
MPI="mpirun --prefix $PREFIX $LAUNCH_FLAGS -x PATH -x LD_LIBRARY_PATH"
echo "nodes=$NODES launcher=$LAUNCH mpirun_base=$MPI"

# ---------------------------------------------------------------- environment record
{
    echo "== date";      date -Is
    echo "== head";      hostname -f
    echo "== launcher";  echo "launch=$LAUNCH nodes=$NODES"; echo "$MPI"
    echo "== pbs";       env | grep -E '^(PBS_|NCPUS|OMP_|SOURCE_GIT_REV)' | sort
    echo "== nodefile";  sort "${PBS_NODEFILE:-/dev/null}" | uniq -c
    echo "== head cpus_allowed"; grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status
    echo "== head lscpu"; lscpu
    echo "== os";        head -4 /etc/os-release; uname -r
    echo "== modules";   module list 2>&1 || true
    echo "== compilers"; mpicxx --version | head -1
    echo "== mpi";       mpirun --version 2>&1 | head -2
    echo "== ompi_info plm/ras (listing only; the selection log is env/launch_verbose.txt)"
    ompi_info 2>/dev/null | grep -E 'MCA (plm|ras):' || true
    echo "== Open MPI's own build flags (the library, not this project's code)"
    ompi_info 2>/dev/null | grep -o "CXXFLAGS=[^']*" | head -1
    echo "== python";    python3 --version
} > "$OUT/env/environment.txt" 2>&1
