#!/bin/bash
# Read-only inventory of an HPC login node, for writing correct job scripts.
# Run on Atlas or Vanda after logging in yourself:
#     bash cluster_discover.sh > discover_$(hostname -s)_$(date +%Y%m%d).txt 2>&1
# It submits no jobs, changes no files except its own output, reads no
# credentials, and makes no network requests beyond local scheduler queries.

section() { printf '\n===== %s\n' "$*"; }
try() { printf '$ %s\n' "$*"; timeout 30 bash -c "$*" 2>&1 | head -n "${LINES_MAX:-80}"; }
# `module` is a shell function, invisible to `bash -c`; run those in this shell.
try_here() { printf '$ %s\n' "$*"; eval "$*" 2>&1 | head -n "${LINES_MAX:-80}"; }
type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1

section identity
try 'date -Is; hostname -f; whoami; id -gn'
try 'cat /etc/os-release | head -4; uname -r'

section cpu-and-memory
try 'lscpu'
try 'nproc; free -g'
try 'numactl --hardware'

section modules
LINES_MAX=400 try_here 'module -t avail'
try_here 'module list'
try_here 'module spider openmpi 2>&1 | head -40; module spider gcc 2>&1 | head -40; module spider intel 2>&1 | head -20'

section compilers-and-mpi-on-path
try 'which gcc g++ icpx icpc mpicxx mpicc mpirun mpiexec srun'
try 'g++ --version | head -1; mpicxx --version | head -1'
try 'mpicxx --showme 2>/dev/null || mpicxx -show 2>/dev/null'
try 'mpirun --version 2>&1 | head -3'
try 'ompi_info 2>/dev/null | grep -iE "Open MPI:|thread support|C\+\+ compiler:" | head -5'

section scheduler-commands
try 'which sbatch srun squeue sinfo sacctmgr qsub qstat pbsnodes bsub bqueues'

section slurm-if-present
try 'sinfo -s'
try 'sinfo -o "%P %a %l %D %c %m %G" | head -30'
try 'sacctmgr -nP show assoc user=$USER format=account,partition,qos 2>/dev/null | head'
try 'scontrol show config 2>/dev/null | grep -iE "SelectType|TaskPlugin|MpiDefault|DefMemPerCPU|MaxTime" '

section pbs-if-present
try 'qstat -Q'
try 'qstat -Qf 2>/dev/null | grep -iE "^Queue|resources_max|resources_default|enabled|started" | head -60'
try 'pbsnodes -a 2>/dev/null | grep -iE "^[a-z]|resources_available.(ncpus|mem|host)" | head -40'

section storage
try 'echo $HOME; df -h $HOME; quota -s 2>/dev/null | head; ls -d /scratch* /hpctmp* /home/svu* 2>/dev/null'

section site-docs-hints
try 'ls /app /apps /opt/apps 2>/dev/null | head -40'
try 'cat /etc/motd 2>/dev/null | head -40'
echo
echo "discovery complete"
