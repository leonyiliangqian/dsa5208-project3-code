"""Where a validator ran, for its evidence file.

PBS sets PBS_ENVIRONMENT=PBS_BATCH in a batch job. Simulated jobs set PBS_JOBID but
not that, so they stay local. This is a label from the environment, not proof.
"""
import os
import socket

KINDS = {
    "pbs_batch": "cluster correctness validation under PBS batch; not performance evidence",
    "local": "local development validation; not cluster evidence and not performance evidence",
}


def execution_context(env=None):
    env = os.environ if env is None else env
    in_batch = env.get("PBS_ENVIRONMENT") == "PBS_BATCH" and bool(env.get("PBS_JOBID"))
    context = "pbs_batch" if in_batch else "local"
    return {
        "kind": KINDS[context],
        "execution_context": context,
        "pbs_jobid": env.get("PBS_JOBID") if in_batch else None,
        "pbs_queue": env.get("PBS_QUEUE") if in_batch else None,
        "hostname": socket.gethostname(),
        "source_git_rev": env.get("SOURCE_GIT_REV", "unknown"),
    }
