#!/bin/bash
# One line per node for the multi-node allocation check: host, CPUs this
# process may use (the job's cpuset on that node), machine architecture, CPU
# model, selected instruction-set flags (recorded only; not a compatibility
# test), and this process's ancestry, which shows how it was launched (under
# PBS tm the remote daemon descends from pbs_mom, under ssh from sshd), and
# whether PBS_JOBID reached this process (leftover_check.sh relies on it to
# attribute remote MPI daemons to the job).
isa=$(lscpu | sed -n 's/^Flags: *//p' | tr ' ' '\n' \
      | grep -E '^(avx|amx|sse|ssse|fma|bmi|f16c|vaes|vpclmul|gfni|sha|aes|movbe|popcnt|adx|clflushopt|clwb|serialize|waitpkg|movdir|cldemote|lzcnt|abm|rdrand|rdseed|xsave)' \
      | sort | paste -sd, -)
chain=; p=$$
while [ "${p:-0}" -gt 1 ] && [ -r /proc/$p/comm ]; do
    chain="$chain${chain:+<}$(tr -d ' ' < /proc/$p/comm)"
    p=$(sed 's/.*) //' /proc/$p/stat | cut -d' ' -f2)
done
printf 'node %s allowed=%s arch=%s model=%s ancestry=%s pbs_jobid_env=%s isa=%s\n' "$(hostname -s)" \
    "$(grep Cpus_allowed_list /proc/self/status | cut -f2)" "$(uname -m)" \
    "$(lscpu | sed -n 's/^Model name: *//p' | tr ' ' '_')" "$chain" \
    "$([ -n "${PBS_JOBID:-}" ] && echo yes || echo no)" "$isa"
