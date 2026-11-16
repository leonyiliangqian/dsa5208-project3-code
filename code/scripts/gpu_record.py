"""What a GPU timing record must show before anyone may use it.

The program's result line, the command and the environment have to agree. A device run
that fell back to the host, or a run that does not say where it ran, is not a timing.
Used by run_case.py when recording and again by analyze.py, which trusts no stored "ok".
"""


def requested_backend(command):
    return command[command.index("--backend") + 1] if "--backend" in command else "device"


def gpu_problems(result, command, env):
    requested = requested_backend(command)
    problems = []
    if result.get("backend") != requested:
        problems.append(f"backend {result.get('backend')} reported, {requested} requested")
    if requested == "device":
        if result.get("kernel_on_device") is not True or result.get("separate_memory") is not True:
            problems.append("device fallback: kernel did not run on a device with separate memory")
        if "--require-device" not in command:
            problems.append("device run without --require-device")
        if (env or {}).get("OMP_TARGET_OFFLOAD") != "MANDATORY":
            problems.append("device run without OMP_TARGET_OFFLOAD=MANDATORY")
    elif result.get("kernel_on_device") is not False:
        problems.append("cpu backend does not report a host kernel")
    # The same team on both backends: real threads, each on its own CPU.
    threads = result.get("threads")
    if result.get("threads_effective") != threads:
        problems.append(f"team of {result.get('threads_effective')}, {threads} requested")
    if result.get("distinct_cpus") != threads:
        problems.append(f"{threads} threads on {result.get('distinct_cpus')} distinct CPUs")
    if result.get("proc_bind") != "close":
        problems.append(f"proc_bind {result.get('proc_bind')}, close expected")
    if requested == "cpu" and (result.get("schedule_effective"), result.get("chunk_effective")) != ("dynamic", 64):
        problems.append("cpu kernel not on schedule dynamic,64")
    return problems
